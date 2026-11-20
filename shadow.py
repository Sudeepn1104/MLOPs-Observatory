"""Shadow deployment: score live traffic with a challenger before promoting it.

The champion answers every request. When a challenger is set (the MLflow
alias `challenger`), the API also scores each request with it, after the
response has been sent, and logs both predictions side by side. Nothing the
challenger says ever reaches a client.

That answers the question an offline holdout can't: how would the new model
behave on the traffic we actually get? Labels arrive months later, so the
comparison is about behaviour: how often the two models disagree, and whether
the new one would flag far more or fewer clients.

    python retrain.py --shadow          # train a challenger and put it in shadow
    python shadow.py start v2           # or put an existing version in shadow
    python shadow.py status             # compare it with the champion on live traffic
    python shadow.py promote            # make the challenger the champion
    python shadow.py promote v1         # or make any version the champion (a rollback)
    python shadow.py stop               # reject it
    python shadow.py demo               # for the compose demo: shadow a fresh retrain once

The same is available over the API: GET /shadow, and with the admin key
POST /shadow/{version}, POST /shadow/promote and DELETE /shadow.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

import registry

MIN_ROWS = 500  # live clients scored by both models before a verdict
MAX_FLAG_RATE_CHANGE = 0.10  # a bigger change in the share flagged needs a person's review


def summarize(pairs, challenger, champion_version):
    """Champion vs challenger on shadow traffic.

    pairs: DataFrame with champion_probability, challenger_probability,
           champion_default, challenger_default (one row per scored client)
    challenger: the challenger's registry entry
    """
    rows = len(pairs)
    out = {
        "champion": champion_version,
        "challenger": challenger["version"],
        "rows": rows,
        "min_rows": MIN_ROWS,
        "offline": {
            "evaluated_on": challenger.get("evaluated_on"),
            "challenger_auc": (challenger.get("metrics") or {}).get("auc"),
            "champion_auc": (challenger.get("champion_metrics_on_holdout") or {}).get("auc"),
            "beats_champion": challenger.get("beats_champion_offline"),
        },
    }
    if rows:
        cp, sp = pairs["champion_probability"], pairs["challenger_probability"]
        cd, sd = pairs["champion_default"].astype(bool), pairs["challenger_default"].astype(bool)
        out["live"] = {
            "agreement": round(float((cd == sd).mean()), 4),
            "mean_abs_probability_difference": round(float((cp - sp).abs().mean()), 4),
            "probability_correlation": round(float(cp.corr(sp)), 4) if rows > 2 else None,
            "champion_flag_rate": round(float(cd.mean()), 4),
            "challenger_flag_rate": round(float(sd.mean()), 4),
            "champion_mean_probability": round(float(cp.mean()), 4),
            "challenger_mean_probability": round(float(sp.mean()), 4),
        }
    out["verdict"], out["reason"] = verdict(out)
    return out


def verdict(s):
    """(verdict, reason): promote, review, reject or keep_watching."""
    if s["offline"]["beats_champion"] is False:
        return "reject", "it did not beat the champion on the recent labelled holdout"
    if s["rows"] < MIN_ROWS:
        return "keep_watching", f"{s['rows']} of {MIN_ROWS} live clients scored by both models so far"
    change = s["live"]["challenger_flag_rate"] - s["live"]["champion_flag_rate"]
    if abs(change) > MAX_FLAG_RATE_CHANGE:
        return "review", (
            f"it would flag {change:+.0%} of clients compared with the champion: "
            "a change in behaviour that large needs a person's sign-off"
        )
    if s["offline"]["beats_champion"] is None:
        return "review", "no offline comparison with the champion is recorded for this version"
    return "promote", "better offline, and its behaviour on live traffic is close to the champion's"


# ---------- the log (written by serve.py) ----------
SCHEMA = (
    "CREATE TABLE IF NOT EXISTS shadow_predictions ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT, ts REAL, "
    "champion_version TEXT, challenger_version TEXT, "
    "champion_probability REAL, challenger_probability REAL, "
    "champion_default INTEGER, challenger_default INTEGER)"
)


def read_pairs(db, challenger_version, limit=50_000):
    """The most recent shadow comparisons for one challenger version."""
    if not Path(db).exists():
        return pd.DataFrame(columns=["champion_version", "champion_probability", "challenger_probability",
                                     "champion_default", "challenger_default"])
    con = sqlite3.connect(db, timeout=10)
    try:
        con.execute(SCHEMA)
        return pd.read_sql_query(
            "SELECT champion_version, champion_probability, challenger_probability, "
            "champion_default, challenger_default FROM shadow_predictions "
            "WHERE challenger_version = ? ORDER BY id DESC LIMIT ?",
            con, params=(challenger_version, limit),
        )
    finally:
        con.close()


def status(root=".", db="logs/predictions.db"):
    challenger = registry.get_challenger(root)
    if challenger is None:
        return None
    champion = registry.get_active(registry.load_registry(root))["version"]
    pairs = read_pairs(db, challenger["version"])
    pairs = pairs[pairs["champion_version"] == champion]  # only against the current champion
    return summarize(pairs, challenger, champion)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Shadow deployment of a challenger model.")
    ap.add_argument("action", choices=["status", "start", "promote", "stop", "demo"])
    ap.add_argument("version", nargs="?",
                    help="start: the version to shadow; promote: a version other than the challenger")
    ap.add_argument("--root", default=".")
    ap.add_argument("--log", default="logs/predictions.db", help="the API's prediction log")
    args = ap.parse_args(argv)

    if args.action == "demo":
        # idempotent: only when the registry holds the champion alone
        alone = len(registry.load_registry(args.root)["models"]) == 1
        if alone and registry.get_challenger(args.root) is None:
            from retrain import retrain

            e = retrain(trigger="demo (shadow)", root=args.root, shadow=True)
            print(f"{e['version']} trained on recent batches and put in shadow")
        else:
            print("registry already has more than one version; leaving it as it is")
    elif args.action == "start":
        if not args.version:
            ap.error("start needs a version, e.g. python shadow.py start v2")
        e = registry.start_shadow(args.version, args.root)
        print(f"{e['version']} is now in shadow: the API scores it on live traffic but never serves it")
    elif args.action == "stop":
        e = registry.stop_shadow(args.root)
        print(f"{e['version']} rejected and removed from shadow" if e else "no challenger in shadow")
    elif args.action == "promote":
        target = args.version or (registry.get_challenger(args.root) or {}).get("version")
        if target is None:
            print("no challenger in shadow; name a version to promote, e.g. v1", file=sys.stderr)
            return 1
        e = registry.promote(target, args.root)
        print(f"{e['version']} is now the champion: the API serves it from the next request")
    else:
        s = status(args.root, args.log)
        if s is None:
            print("no challenger in shadow")
            return 0
        print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
