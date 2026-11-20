"""Generates monitoring/grafana/dashboards/observatory.json (run after editing).

    python monitoring/build_dashboard.py

Written as code so every panel's query is easy to review, and so
tests/test_monitoring.py can check each query uses a metric the API exports.
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "grafana" / "dashboards" / "observatory.json"
DS = {"type": "prometheus", "uid": "prometheus"}
V = '{model_version=~".+"}'  # every model version

_id = iter(range(1, 100))


def target(expr, legend="", ref="A"):
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref, "range": True}


def thresholds(*steps):
    """steps: (value, color) pairs; the first value is None (the base)."""
    return {"mode": "absolute", "steps": [{"value": v, "color": c} for v, c in steps]}


def row(title, y):
    return {"id": next(_id), "type": "row", "title": title, "collapsed": False, "panels": [],
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}}


def stat(title, expr, x, w, unit="none", decimals=None, steps=None, legend="", desc="", text=False, y=0):
    defaults = {"unit": unit, "thresholds": steps or thresholds((None, "blue"))}
    if decimals is not None:
        defaults["decimals"] = decimals
    opts = {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "background" if steps else "value", "graphMode": "none",
            "textMode": "name" if text else "value", "justifyMode": "center"}
    return {"id": next(_id), "type": "stat", "title": title, "description": desc,
            "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": 4},
            "targets": [target(expr, legend)],
            "fieldConfig": {"defaults": defaults, "overrides": []}, "options": opts}


def series(title, targets, x, y, w, h, unit="none", steps=None, desc="", max_=None, line_steps=False):
    style = {"mode": "dashed" if line_steps else "off"}
    defaults = {"unit": unit, "custom": {"lineWidth": 2, "fillOpacity": 5, "showPoints": "never",
                                         "thresholdsStyle": style}}
    if steps:
        defaults["thresholds"] = steps
    if max_ is not None:
        defaults["max"] = max_
        defaults["min"] = 0
    return {"id": next(_id), "type": "timeseries", "title": title, "description": desc,
            "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": {"legend": {"displayMode": "table", "placement": "right",
                                   "calcs": ["lastNotNull"]},
                        "tooltip": {"mode": "multi", "sort": "desc"}}}


def build():
    green_red = thresholds((None, "green"), (0.1, "red"))
    psi_steps = thresholds((None, "green"), (0.1, "yellow"), (0.2, "red"))
    panels = [
        stat("Model in production", "observatory_active_model_info", 0, 4, legend="{{version}}",
             text=True,
             desc="The MLflow champion the API serves. Changes as soon as a retrain promotes a new version."),
        stat("Requests / s", "sum(rate(observatory_http_requests_total[1m]))", 4, 4,
             unit="reqps", decimals=1),
        stat("Clients scored / s", "sum(rate(observatory_predictions_total[1m]))", 8, 4, decimals=1),
        stat("Share flagged as default",
             "sum(rate(observatory_predicted_defaults_total[5m]))"
             " / sum(rate(observatory_predictions_total[5m]))",
             12, 4, unit="percentunit", decimals=1,
             desc="Share of scored clients at or above the model's decision threshold."),
        stat("Features drifted", f"max(observatory_live_drift_features_drifted{V})", 16, 4,
             steps=thresholds((None, "green"), (1, "orange"), (3, "red")),
             desc="Features whose per-type drift score (Evidently's tests) is 0.1 or more, over the "
                  "last 500 scored clients."),
        stat("Server errors (5xx)",
             'sum(rate(observatory_http_requests_total{status=~"5.."}[5m])) or vector(0)', 20, 4,
             unit="reqps", decimals=2, steps=thresholds((None, "green"), (0.001, "red"))),
        series("Live drift per feature (per-type tests, Evidently's method)",
               [target(f"observatory_live_drift_score{V}", "{{feature}}")], 0, 4, 12, 9,
               steps=green_red, line_steps=True,
               desc="Wasserstein distance (normed) for amounts, Jensen-Shannon for codes, on the last "
                    "500 scored clients vs the model's training data. Red at 0.1 or above. "
                    "Needs no labels."),
        series("Live PSI per feature (drives the retrain rule)",
               [target(f"observatory_live_drift_psi{V}", "{{feature}}")], 12, 4, 12, 9,
               steps=psi_steps, line_steps=True,
               desc="Population Stability Index on the same window. Yellow 0.1 to 0.2, red above 0.2."),
        series("Predicted risk of scored clients",
               [target(f"histogram_quantile({q}, sum by (le) "
                       "(rate(observatory_predicted_probability_bucket[5m])))",
                       f"p{int(q * 100)}", r) for q, r in ((0.5, "A"), (0.9, "B"))]
               + [target("sum(rate(observatory_predicted_probability_sum[5m])) / "
                         "sum(rate(observatory_predicted_probability_count[5m]))", "mean", "C")],
               0, 13, 8, 8, unit="percentunit", max_=1,
               desc="The model's calibrated probability of default. A shift here is prediction drift: "
                    "visible before any label arrives."),
        series("Request rate by route and status",
               [target("sum by (route, status) (rate(observatory_http_requests_total[1m]))",
                       "{{route}} {{status}}")], 8, 13, 8, 8, unit="reqps"),
        series("Latency",
               [target(f"histogram_quantile({q}, sum by (le) "
                       f'(rate(observatory_http_request_duration_seconds_bucket{{route="/predict"}}[5m])))',
                       f"/predict p{int(q * 100)}", r) for q, r in ((0.5, "A"), (0.95, "B"), (0.99, "C"))],
               16, 13, 8, 8, unit="s"),
        row("Shadow deployment: the challenger scores the same traffic, clients only see the champion", 21),
        stat("Challenger in shadow", "observatory_challenger_model_info", 0, 4, legend="{{version}}",
             text=True, y=22,
             desc="The version the MLflow alias 'challenger' points at. Empty when nothing is in "
                  "shadow. Start one with: python retrain.py --shadow"),
        stat("Decision agreement",
             "1 - sum(rate(observatory_shadow_disagreements_total[5m]))"
             " / sum(rate(observatory_shadow_predictions_total[5m]))",
             4, 5, unit="percentunit", decimals=1, y=22,
             steps=thresholds((None, "red"), (0.8, "orange"), (0.9, "green")),
             desc="Share of clients where the challenger makes the same decision as the champion."),
        stat("Mean |probability difference|",
             "sum(rate(observatory_shadow_abs_probability_difference_total[5m]))"
             " / sum(rate(observatory_shadow_predictions_total[5m]))",
             9, 5, unit="percentunit", decimals=1, y=22),
        stat("Shadow errors", "sum(rate(observatory_shadow_errors_total[5m])) or vector(0)", 14, 5,
             unit="reqps", decimals=2, y=22, steps=thresholds((None, "green"), (0.001, "red")),
             desc="Failures scoring the challenger. They never reach a client."),
        stat("Shadow scoring p95",
             "histogram_quantile(0.95, sum by (le) (rate(observatory_shadow_duration_seconds_bucket[5m])))",
             19, 5, unit="s", decimals=1, y=22,
             desc="Runs after the response is sent, so it adds nothing to client latency."),
        series("Share flagged as default: champion vs challenger",
               [target("sum(rate(observatory_predicted_defaults_total[5m]))"
                       " / sum(rate(observatory_predictions_total[5m]))", "champion", "A"),
                target("sum by (challenger_version) (rate(observatory_shadow_predicted_defaults_total[5m]))"
                       " / sum by (challenger_version) (rate(observatory_shadow_predictions_total[5m]))",
                       "challenger {{challenger_version}}", "B")],
               0, 26, 12, 8, unit="percentunit", max_=1,
               desc="A large gap means promoting would change many decisions: shadow.py asks for a "
                    "person's review when it exceeds 10 points."),
        series("Disagreement rate",
               [target("sum by (challenger_version) (rate(observatory_shadow_disagreements_total[5m]))"
                       " / sum by (challenger_version) (rate(observatory_shadow_predictions_total[5m]))",
                       "{{challenger_version}} vs champion")],
               12, 26, 12, 8, unit="percentunit", max_=1),
    ]
    return {
        "uid": "mlops-observatory",
        "title": "MLOps Observatory: model in production",
        "description": "Traffic, predictions and live drift of the credit-default model served by serve.py.",
        "tags": ["mlops", "drift", "observatory"],
        "timezone": "browser",
        "editable": False,
        "graphTooltip": 1,
        "refresh": "10s",
        "time": {"from": "now-30m", "to": "now"},
        "schemaVersion": 39,
        "version": 1,
        "panels": panels,
        "templating": {"list": []},
        "annotations": {"list": []},
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {OUT}")
