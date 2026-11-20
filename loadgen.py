"""Sends realistic, drifting traffic to the prediction API, for demos and dashboards.

    python loadgen.py --url http://localhost:8000          # uses $API_KEY
    docker compose --profile demo up --build                # runs it next to the API

Clients come from the stream pool (real clients the model never trained on).
Drift ramps up over --ramp-minutes, following the same scenario as the
monitoring batches (batch 0 conditions at the start, batch 9 at the end),
then stays there. Watch the Grafana dashboard's drift panels climb.

Only the standard library is used to talk to the API.
"""
import argparse
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request

import numpy as np

from make_data import TARGET, make_batch
from simulate import N_BATCHES, scenario

_stop = False


def _handle_stop(*_):
    global _stop
    _stop = True


def drift_level(elapsed, ramp_seconds):
    """Scenario position (0 to N_BATCHES - 1) after `elapsed` seconds."""
    if ramp_seconds <= 0:
        return N_BATCHES - 1
    return min(N_BATCHES - 1, (N_BATCHES - 1) * elapsed / ramp_seconds)


def make_request(n, level, rng):
    """A /predict body of n clients drawn at scenario position `level`."""
    lo, hi = int(np.floor(level)), int(np.ceil(level))
    w = level - lo
    a, b = scenario(lo), scenario(hi)
    limit_tilt, age_tilt, concept = (x + w * (y - x) for x, y in zip(a, b, strict=True))
    batch = make_batch(n, limit_tilt=limit_tilt, age_tilt=age_tilt, concept=concept, rng=rng)
    return {"clients": batch.drop(columns=[TARGET]).to_dict(orient="records")}


def post(url, body, key, timeout=30):
    req = urllib.request.Request(
        url.rstrip("/") + "/predict",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", **({"X-API-Key": key} if key else {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - URL comes from the operator
        return json.load(r)


def wait_until_ready(url, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not _stop:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/ready", timeout=5):  # noqa: S310
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=os.environ.get("API_URL", "http://localhost:8000"))
    ap.add_argument("--rate", type=float, default=20, help="clients per second (default 20)")
    ap.add_argument("--batch", type=int, default=20, help="clients per request (default 20)")
    ap.add_argument("--ramp-minutes", type=float, default=10, help="time to reach full drift")
    ap.add_argument("--duration-minutes", type=float, default=0,
                    help="stop after this long (0 = run until stopped)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    key = os.environ.get("API_KEY", "").split(",")[0].strip()
    if not wait_until_ready(args.url):
        print(f"API at {args.url} is not ready", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    interval = args.batch / args.rate
    start, sent, errors, last_report = time.monotonic(), 0, 0, 0.0
    while not _stop:
        elapsed = time.monotonic() - start
        if args.duration_minutes and elapsed >= args.duration_minutes * 60:
            break
        level = drift_level(elapsed, args.ramp_minutes * 60)
        try:
            post(args.url, make_request(args.batch, level, rng), key)
            sent += args.batch
        except (urllib.error.URLError, OSError) as e:
            errors += 1
            print(f"request failed: {e}", file=sys.stderr)
        if elapsed - last_report >= 30:
            print(f"{elapsed / 60:5.1f} min: {sent} clients sent, drift level {level:.1f}/{N_BATCHES - 1}, "
                  f"{errors} errors", flush=True)
            last_report = elapsed
        time.sleep(max(0.0, interval - (time.monotonic() - start - elapsed)))
    print(f"stopped: {sent} clients sent, {errors} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
