"""Measures the API's capacity: throughput and latency at a given concurrency.

    python loadtest.py --url http://localhost:8000 --concurrency 8 --seconds 30 --batch 1
    (uses $API_KEY)

Results for the compose stack are in docs/RUNBOOK.md (Capacity). Standard library
only, so it runs anywhere the API is reachable.
"""
import argparse
import json
import os
import statistics
import threading
import time
import urllib.request

import numpy as np

from loadgen import make_request


def run(url, key, concurrency, seconds, batch):
    bodies = [json.dumps(make_request(batch, 0, np.random.default_rng(i))).encode() for i in range(50)]
    latencies, errors, lock = [], [0], threading.Lock()
    deadline = time.monotonic() + seconds

    def worker(n):
        i = n
        while time.monotonic() < deadline:
            req = urllib.request.Request(url.rstrip("/") + "/predict", data=bodies[i % len(bodies)],
                                         headers={"content-type": "application/json", "X-API-Key": key})
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
                    r.read()
                ok = True
            except OSError:
                ok = False
            elapsed = time.perf_counter() - start
            with lock:
                (latencies.append(elapsed) if ok else errors.__setitem__(0, errors[0] + 1))
            i += concurrency

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    q = statistics.quantiles(latencies, n=100) if len(latencies) > 1 else [0] * 99
    return {
        "concurrency": concurrency, "batch": batch, "requests": len(latencies), "errors": errors[0],
        "requests_per_s": round(len(latencies) / seconds, 1),
        "clients_per_s": round(len(latencies) * batch / seconds, 1),
        "p50_ms": round(q[49] * 1000, 1), "p95_ms": round(q[94] * 1000, 1), "p99_ms": round(q[98] * 1000, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=os.environ.get("API_URL", "http://localhost:8000"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--batch", type=int, default=1, help="clients per request")
    args = ap.parse_args(argv)
    key = os.environ.get("API_KEY", "").split(",")[0].strip()
    print(json.dumps(run(args.url, key, args.concurrency, args.seconds, args.batch)))


if __name__ == "__main__":
    main()
