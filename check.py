"""Scheduled health check: applies the retrain rule and reports the result.

    python check.py                               # check the project workspace
    python check.py --api-url http://api:8000     # also check live API traffic
    python check.py --json report.json --markdown report.md

Exit codes: 0 healthy (or waiting for data), 1 alert, 2 the check itself failed.
Designed for cron / CI: .github/workflows/monitor.yml runs it weekly and turns
an alert into a GitHub issue.
"""
import argparse
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from monitor import AUC_DROP, PSI_RED, RULE_WINDOW, evaluate, load_logs
from registry import get_active, load_registry

ICON = {"alert": "🔴", "healthy": "🟢", "waiting": "⏳"}


def fetch_live_drift(api_url, timeout=10):
    url = api_url.rstrip("/") + "/monitoring/drift"
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - URL comes from the operator
        return json.load(r)


def build_report(root=".", api_url=None):
    reg = load_registry(root)
    active = get_active(reg)
    batch_log, _ = load_logs(root)
    reasons, own = evaluate(batch_log, active)

    live = None
    if api_url:
        live = fetch_live_drift(api_url)
        if live.get("status") == "red":
            reasons.append(
                f"live traffic: max PSI {live['max_psi']:.2f} on {live['worst_feature']} "
                f"over the last {live['rows']} requests"
            )

    if reasons:
        status = "alert"
    elif own.empty:
        status = "waiting"
    else:
        status = "healthy"

    latest = None
    if not own.empty:
        row = own.iloc[-1]
        latest = {
            "batch": int(row["batch"]),
            "auc": float(row["auc"]),
            "f1": float(row["f1"]),
            "max_psi": float(row["max_psi"]),
            "red_features": [f for f in str(row["red_features"]).split(",") if f and f != "nan"],
        }

    return {
        "status": status,
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": active["version"],
        "baseline_auc": active["metrics"]["auc"],
        "batches_scored_by_model": len(own),
        "reasons": reasons,
        "latest_batch": latest,
        "live_drift": live,
        "rule": {"psi_red": PSI_RED, "auc_drop": AUC_DROP, "window": RULE_WINDOW},
    }


def to_markdown(report):
    r = report
    lines = [
        f"## {ICON[r['status']]} Model health: {r['status'].upper()}",
        "",
        f"- **Active model:** {r['model']} (baseline AUC {r['baseline_auc']:.3f})",
        f"- **Batches scored by {r['model']}:** {r['batches_scored_by_model']}",
        f"- **Checked at:** {r['checked_at']}",
    ]
    if r["latest_batch"]:
        b = r["latest_batch"]
        red = ", ".join(b["red_features"]) or "none"
        lines.append(
            f"- **Latest batch ({b['batch']}):** AUC {b['auc']:.3f}, F1 {b['f1']:.3f}, "
            f"max PSI {b['max_psi']:.2f}, red features: {red}"
        )
    if r["live_drift"]:
        d = r["live_drift"]
        if d.get("status") == "insufficient_data":
            lines.append(f"- **Live traffic:** not enough requests yet ({d['rows']})")
        else:
            lines.append(
                f"- **Live traffic:** {d['status']}, max PSI {d['max_psi']:.2f} "
                f"({d['worst_feature']}) over {d['rows']} requests"
            )
    lines.append("")
    if r["reasons"]:
        lines.append("### Why the retrain rule fired")
        lines += [f"- {x}" for x in r["reasons"]]
        lines += ["", "**Action:** retrain and compare (Model registry tab, or `python retrain.py`)."]
    elif r["status"] == "waiting":
        lines.append(f"{r['model']} has no scored batches yet, so there is nothing to judge.")
    else:
        lines.append("Inputs are stable and AUC is within the model's baseline. No action needed.")
    lines += [
        "",
        f"<sub>Rule: max PSI > {r['rule']['psi_red']} or AUC more than {r['rule']['auc_drop']} "
        f"below baseline (mean of the last {r['rule']['window']} batches).</sub>",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=".", help="workspace to check (default: project)")
    p.add_argument("--api-url", help="also check live traffic via the API's /monitoring/drift")
    p.add_argument("--json", type=Path, help="write the report as JSON here")
    p.add_argument("--markdown", type=Path, help="write the report as Markdown here")
    args = p.parse_args(argv)

    try:
        report = build_report(args.root, args.api_url)
    except Exception as e:  # noqa: BLE001 - a broken check must be visible, not "healthy"
        print(f"check failed: {e}", file=sys.stderr)
        return 2

    md = to_markdown(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    if args.markdown:
        args.markdown.write_text(md)
    print(md)
    return 1 if report["status"] == "alert" else 0


if __name__ == "__main__":
    sys.exit(main())
