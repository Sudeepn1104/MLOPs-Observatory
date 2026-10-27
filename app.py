"""Tiny MLOps Observatory: Streamlit dashboard (Day 4).

Run:  streamlit run app.py
Needs: python make_data.py, python train.py, python simulate.py first.
"""
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from drift import check_drift
from make_data import make_loans
from registry import get_active, load_registry, reset_registry
from retrain import AUC_DROP, PSI_RED, RECENT_BATCHES, retrain, retrain_policy

TARGET = "default"
PSI_YELLOW = 0.1
STATUS_ICON = {"green": "🟢", "yellow": "🟡", "red": "🔴"}

st.set_page_config(page_title="Tiny MLOps Observatory", layout="wide")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_reference(path):
    return pd.read_csv(path)


@st.cache_data
def load_logs():
    return (
        pd.read_csv("logs/batch_log.csv"),
        pd.read_csv("logs/feature_drift.csv"),
    )


def overall_status(max_psi):
    if max_psi < PSI_YELLOW:
        return "green"
    if max_psi < PSI_RED:
        return "yellow"
    return "red"


# ---------- guard: required files ----------
needed = [
    "models/model_v1.joblib",
    "models/reference_data.csv",
    "models/model_v1_metrics.json",
    "logs/batch_log.csv",
    "logs/feature_drift.csv",
    "logs/batches.csv",
]
missing = [p for p in needed if not Path(p).exists()]
if missing:
    st.error(
        f"Missing files: {', '.join(missing)}. "
        "Run: python make_data.py, python train.py, python simulate.py"
    )
    st.stop()

if "flash" in st.session_state:
    st.toast(st.session_state.pop("flash"))

reg = load_registry()
active = get_active(reg)
model = load_model(active["file"])
ref = load_reference(active["reference"])
batch_log, feature_log = load_logs()
baseline_auc = reg["models"][0]["metrics"]["auc"]  # v1 test AUC = healthy baseline

# ---------- retrain rule + optional auto-retrain ----------
reasons = retrain_policy(batch_log, baseline_auc)

st.sidebar.header("Retrain policy")
st.sidebar.write(
    f"Retrain when **max PSI > {PSI_RED}** or when **AUC drops more than {AUC_DROP}** "
    "below the baseline (mean of the last 2 batches)."
)
auto = st.sidebar.toggle("Auto-retrain when the rule fires", value=False)
st.sidebar.caption(
    "Auto mode retrains once per drift episode (only while v1 is the only version). "
    "Use Reset in the Model registry tab to demo again."
)
if auto and reasons and len(reg["models"]) == 1:
    entry = retrain(trigger="auto: " + "; ".join(reasons))
    verdict = "promoted" if entry["promoted"] else "rejected"
    st.session_state["flash"] = f"Auto-retrain ran: {entry['version']} {verdict}"
    st.rerun()

st.title("Tiny MLOps Observatory")
st.caption("Watches a loan-default model for data drift and performance drop.")

tab_monitor, tab_inject, tab_registry = st.tabs(
    ["Monitoring", "Drift injector (live demo)", "Model registry"]
)

# =====================================================
# TAB 1: monitoring over the simulated batches
# =====================================================
with tab_monitor:
    latest = batch_log.iloc[-1]
    first = batch_log.iloc[0]
    status = overall_status(latest["max_psi"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active model", active["version"])
    c2.metric(
        "Drift status",
        f"{STATUS_ICON[status]} {status.upper()}",
        f"max PSI {latest['max_psi']:.2f}",
        delta_color="off",
    )
    c3.metric(
        "AUC (latest batch)",
        f"{latest['auc']:.3f}",
        f"{latest['auc'] - first['auc']:+.3f} vs batch 0",
    )
    c4.metric(
        "Accuracy (latest batch)",
        f"{latest['accuracy']:.3f}",
        f"{latest['accuracy'] - first['accuracy']:+.3f} vs batch 0",
    )

    if reasons:
        st.error("RETRAIN RULE FIRED: " + "; ".join(reasons) + ".")
        if len(reg["models"]) == 1:
            st.info("Go to the **Model registry** tab to retrain, or turn on auto-retrain in the sidebar.")
        else:
            st.success(
                f"Already retrained. Active model is {active['version']}. "
                "See the Model registry tab."
            )
    elif status == "yellow":
        st.warning("Watch: some features are drifting (PSI 0.1 to 0.2). No retrain needed yet.")
    else:
        st.success("All features stable. No retrain needed.")
    st.caption("These monitoring logs were produced by model v1 on the simulated batches.")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Drift heatmap (PSI per feature per batch)")
        pivot = feature_log.pivot(index="feature", columns="batch", values="psi")
        colorscale = [
            [0.0, "#2e9e5b"],
            [PSI_YELLOW / 0.3, "#2e9e5b"],
            [PSI_YELLOW / 0.3, "#e8b931"],
            [PSI_RED / 0.3, "#e8b931"],
            [PSI_RED / 0.3, "#d64545"],
            [1.0, "#d64545"],
        ]
        fig = px.imshow(
            pivot,
            color_continuous_scale=colorscale,
            zmin=0,
            zmax=0.3,
            aspect="auto",
            text_auto=".2f",
            labels=dict(x="Batch", y="Feature", color="PSI"),
        )
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Green < 0.1, yellow 0.1 to 0.2, red > 0.2 (values above 0.3 shown as red).")

    with right:
        st.subheader("Model performance per batch")
        metric_choice = st.multiselect(
            "Metrics",
            ["accuracy", "auc", "f1", "recall"],
            default=["accuracy", "auc", "f1"],
        )
        fig2 = go.Figure()
        for m in metric_choice:
            fig2.add_trace(
                go.Scatter(x=batch_log["batch"], y=batch_log[m], mode="lines+markers", name=m)
            )
        fig2.add_hline(
            y=baseline_auc, line_dash="dot", annotation_text="v1 test AUC", opacity=0.6
        )
        fig2.update_layout(xaxis_title="Batch", yaxis_title="Score", yaxis_range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Accuracy alone can hide problems (most loans do not default), "
            "so AUC and F1 are tracked too."
        )

    st.subheader("Max PSI per batch")
    fig3 = px.line(batch_log, x="batch", y="max_psi", markers=True)
    fig3.add_hline(y=PSI_RED, line_dash="dash", line_color="#d64545", annotation_text="0.2 retrain")
    fig3.add_hline(y=PSI_YELLOW, line_dash="dash", line_color="#e8b931", annotation_text="0.1 watch")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Inspect a batch")
    b = st.slider("Batch", int(batch_log["batch"].min()), int(batch_log["batch"].max()),
                  int(batch_log["batch"].max()))
    detail = feature_log[feature_log["batch"] == b].drop(columns="batch").copy()
    detail.insert(0, "", detail["status"].map(STATUS_ICON))
    st.dataframe(detail, hide_index=True, use_container_width=True)

    # downloadable drift report (full log, one row per batch per feature)
    report = feature_log.merge(
        batch_log[["batch", "accuracy", "auc", "f1"]], on="batch", how="left"
    )
    st.download_button(
        "Download drift report (CSV)",
        report.to_csv(index=False).encode(),
        file_name="drift_report.csv",
        mime="text/csv",
    )

# =====================================================
# TAB 2: drift injector, generate a batch live
# =====================================================
with tab_inject:
    st.write(
        "Inject drift into a fresh batch and watch the **active model "
        f"({active['version']})** react. Use **input drift** to move the data, "
        "**concept drift** to change the default rule."
    )
    s1, s2, s3, s4 = st.columns(4)
    income_shift = s1.slider("Income shift (log scale)", -1.0, 0.5, 0.0, 0.05)
    age_shift = s2.slider("Age shift (years)", -10.0, 30.0, 0.0, 1.0)
    concept = s3.slider("Concept drift (0 to 1)", 0.0, 1.0, 0.0, 0.1)
    size = s4.select_slider("Batch size", [200, 500, 1000, 2000], value=500)

    if st.button("Generate batch and check", type="primary"):
        batch = make_loans(size, income_shift=income_shift, age_shift=age_shift, concept=concept)
        X, y = batch.drop(columns=[TARGET]), batch[TARGET]
        st.session_state["inject"] = (check_drift(ref, X), y, model.predict(X),
                                       model.predict_proba(X)[:, 1])

    if "inject" in st.session_state:
        rep, y, pred, proba = st.session_state["inject"]
        max_psi = float(rep["psi"].max())
        st_now = overall_status(max_psi)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Drift status", f"{STATUS_ICON[st_now]} {st_now.upper()}", f"max PSI {max_psi:.2f}",
                  delta_color="off")
        k2.metric("Accuracy", f"{accuracy_score(y, pred):.3f}")
        k3.metric("AUC", f"{roc_auc_score(y, proba):.3f}")
        k4.metric("F1", f"{f1_score(y, pred, zero_division=0):.3f}")

        rep = rep.copy()
        rep.insert(0, "", rep["status"].map(STATUS_ICON))
        st.dataframe(rep, hide_index=True, use_container_width=True)
        st.caption(
            "Tip: input drift alone turns PSI red but barely hurts AUC. "
            "Add concept drift to see performance fall. Drift is measured against "
            f"the training data of the active model ({active['version']})."
        )

# =====================================================
# TAB 3: model registry, retrain, v1 vs v2
# =====================================================
with tab_registry:
    r1, r2, r3 = st.columns([1, 1, 2])
    r1.metric("Active model", active["version"])
    r2.metric("Versions", len(reg["models"]))
    r3.write(
        f"Retraining learns from the last **{RECENT_BATCHES} batches** of labeled data. "
        "A new model is promoted only if its AUC beats the active model on the same holdout."
    )

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Retrain now", type="primary"):
        entry = retrain(trigger="manual (button)")
        verdict = "promoted to active" if entry["promoted"] else "rejected (not better)"
        st.session_state["flash"] = f"{entry['version']} {verdict}"
        st.rerun()
    if b2.button("Reset to v1"):
        reset_registry()
        st.session_state["flash"] = "Registry reset to v1"
        st.rerun()

    st.subheader("All versions")
    rows = []
    for m in reg["models"]:
        rows.append(
            {
                "version": m["version"],
                "status": m["status"],
                "trained_at": m["trained_at"],
                "trigger": m["trigger"],
                "train_rows": m["train_rows"],
                "accuracy": m["metrics"]["accuracy"],
                "f1": m["metrics"]["f1"],
                "auc": m["metrics"]["auc"],
                "evaluated_on": m["evaluated_on"],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "Metrics in different rows use different test sets, so compare models "
        "only in the chart below (same holdout)."
    )

    retrained = [m for m in reg["models"] if "champion_metrics_on_holdout" in m]
    if retrained:
        last = retrained[-1]
        old_v, new_v = last["compared_with"], last["version"]
        st.subheader(f"{old_v} vs {new_v} on the same recent holdout")
        cmp = pd.DataFrame(
            [
                {"model": old_v, "metric": k, "score": v}
                for k, v in last["champion_metrics_on_holdout"].items()
            ]
            + [{"model": new_v, "metric": k, "score": v} for k, v in last["metrics"].items()]
        )
        fig4 = px.bar(cmp, x="metric", y="score", color="model", barmode="group", text_auto=".2f")
        fig4.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig4, use_container_width=True)
        if last["promoted"]:
            st.success(f"{new_v} beat {old_v} on AUC and is now the active model.")
        else:
            st.warning(f"{new_v} did not beat {old_v}, so {old_v} stays active.")
        st.caption(f"Trigger: {last['trigger']}")
    else:
        st.info("No retraining yet. Click **Retrain now** or enable auto-retrain in the sidebar.")
