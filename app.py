"""Tiny MLOps Observatory: Streamlit dashboard.

Run:  streamlit run app.py
Needs: python make_data.py, python train.py, python simulate.py first.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import model as mdl
import registry
from bootstrap import build_all, needs_build
from drift import check_drift
from make_data import FEATURE_NAMES, TRAIN, make_batch
from monitor import (
    AUC_DROP,
    PSI_RED,
    RULE_WINDOW,
    can_auto_retrain,
    evaluate,
    load_batches,
    load_logs,
    reset_stream,
    stream_next,
)
from registry import get_active, load_registry, reset_registry
from retrain import RECENT_BATCHES, retrain
from workspace import sandbox_enabled, session_root

TARGET = "default"
PSI_YELLOW = 0.1
STATUS_ICON = {"green": "🟢", "yellow": "🟡", "red": "🔴"}

st.set_page_config(page_title="Tiny MLOps Observatory", layout="wide")


@st.cache_resource(max_entries=8)
def load_model(model_uri):
    # an MLflow artifact URI contains the run ID, so it is unique per trained
    # model: a v2 re-created after Reset can never be served from this cache
    return registry.load_model({"model_uri": model_uri})


@st.cache_data(max_entries=8)
def load_reference(reference_uri):
    return registry.load_reference({"reference_uri": reference_uri})


def overall_status(max_psi):
    if max_psi < PSI_YELLOW:
        return "green"
    if max_psi < PSI_RED:
        return "yellow"
    return "red"


# ---------- guard: required files ----------
# builds on first start, and rebuilds when the host kept a model from older code
if needs_build():
    try:
        with st.spinner("Building data, model v1 and logs (about 30 seconds)..."):
            build_all()
    except Exception as e:
        st.error(f"Could not build the files automatically ({e}). "
                 "Run: python make_data.py, python train.py, python simulate.py")
        st.stop()
    registry.reconnect()  # the build replaced the database file this process may have open
    st.cache_data.clear()
    st.cache_resource.clear()
    if needs_build():  # never loop: if the build didn't fix it, say so
        st.error(f"The build ran but the app is still not ready: {needs_build()}.")
        st.stop()
    st.rerun()

if "flash" in st.session_state:
    st.toast(st.session_state.pop("flash"))

root = session_root(st.session_state)  # project dir locally, private dir on the live demo
reg = load_registry(root)
active = get_active(reg)
model = load_model(active["model_uri"])
ref = load_reference(active["reference_uri"])
batch_log, feature_log = load_logs(root)
baseline_auc = active["metrics"]["auc"]  # each model is judged against its own baseline

# ---------- retrain rule + optional auto-retrain ----------
reasons, own_log = evaluate(batch_log, active)

st.sidebar.header("Retrain policy")
st.sidebar.write(
    f"Retrain when **max PSI > {PSI_RED}** or when **AUC drops more than {AUC_DROP}** "
    f"below the active model's own baseline (mean of its last {RULE_WINDOW} batches)."
)
auto = st.sidebar.toggle("Auto-retrain when the rule fires", value=False)
st.sidebar.caption(
    "Auto mode retrains at most once per new batch, so a rejected challenger "
    "doesn't trigger a retrain on every refresh."
)
if sandbox_enabled():
    st.sidebar.info(
        "You have a private sandbox: your retrains, resets and streamed batches "
        "don't affect other visitors. It resets when you reload the page."
    )
if auto and reasons and can_auto_retrain(reg, batch_log):
    entry = retrain(trigger="auto: " + "; ".join(reasons), root=root)
    verdict = "promoted" if entry["promoted"] else "rejected"
    st.session_state["flash"] = f"Auto-retrain ran: {entry['version']} {verdict}"
    st.rerun()

st.title("Tiny MLOps Observatory")
st.caption(
    "Watches a credit-default model, trained on real client data, for data drift and performance drop."
)

tab_monitor, tab_inject, tab_explain, tab_registry, tab_card = st.tabs(
    ["Monitoring", "Drift injector (live demo)", "Explain", "Model registry", "Model card"]
)

# =====================================================
# TAB 1: monitoring over the simulated batches
# =====================================================
with tab_monitor:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active model", active["version"])
    if own_log.empty:
        status = None
        c2.metric("Drift status", "⏳ WAITING", "no batches yet", delta_color="off")
        c3.metric("AUC (latest batch)", "n/a")
        c4.metric(f"{active['version']} baseline AUC", f"{baseline_auc:.3f}")
    else:
        latest = own_log.iloc[-1]
        status = overall_status(latest["max_psi"])
        c2.metric(
            "Drift status",
            f"{STATUS_ICON[status]} {status.upper()}",
            f"max PSI {latest['max_psi']:.2f}",
            delta_color="off",
        )
        c3.metric(
            f"AUC (batch {int(latest['batch'])})",
            f"{latest['auc']:.3f}",
            f"{latest['auc'] - baseline_auc:+.3f} vs {active['version']} baseline",
        )
        c4.metric(
            f"Batches scored by {active['version']}",
            len(own_log),
            f"{latest['accuracy']:.3f} accuracy (latest)",
            delta_color="off",
        )

    if status is None:
        st.info(
            f"{active['version']} has just been promoted and hasn't scored any batches yet. "
            "Stream a batch below to start monitoring it."
        )
    elif reasons:
        st.error("RETRAIN RULE FIRED: " + "; ".join(reasons) + ".")
        st.info("Go to the **Model registry** tab to retrain, or turn on auto-retrain in the sidebar.")
        rejected_since = [
            m for m in reg["models"]
            if m["status"] == "rejected" and m.get("compared_with") == active["version"]
        ]
        if rejected_since:
            st.warning(
                f"A challenger ({rejected_since[-1]['version']}) was already tried and didn't "
                f"beat {active['version']}. Stream more batches before retraining again."
            )
    elif status == "yellow":
        st.warning("Watch: some features are drifting (PSI 0.1 to 0.2). No retrain needed yet.")
    else:
        st.success(f"{active['version']} is healthy: inputs stable and AUC within its baseline.")

    s1, s2, _ = st.columns([1, 1, 3])
    n_stream = 0
    if s1.button("Stream next batch", type="primary"):
        n_stream = 1
    if s2.button("Stream 3 batches"):
        n_stream = 3
    if n_stream:
        for _ in range(n_stream):
            row = stream_next(reg, root)
        st.session_state["flash"] = (
            f"Streamed {n_stream} batch(es); batch {row['batch']} scored by {row['model']}: "
            f"AUC {row['auc']:.3f}, max PSI {row['max_psi']:.2f}"
        )
        st.rerun()
    st.caption(
        "Batches 0 to 9 are the drift scenario, scored by v1. Streamed batches keep the "
        "drifted conditions of batch 9 and are scored by whichever model is active, "
        "so you can watch a retrained model take over."
    )

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
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Green < 0.1, yellow 0.1 to 0.2, red > 0.2 (values above 0.3 shown as red). "
            "Each batch is compared with the training data of the model that scored it, "
            "so after a retrain the drifted world becomes the new normal."
        )

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
            y=baseline_auc, line_dash="dot",
            annotation_text=f"{active['version']} baseline AUC",
            annotation_position="bottom right", opacity=0.6,
        )
        switches = batch_log[batch_log["model"] != batch_log["model"].shift()].iloc[1:]
        for _, sw in switches.iterrows():
            fig2.add_vline(
                x=sw["batch"] - 0.5, line_dash="dash", line_color="#888",
                annotation_text=f"{sw['model']} takes over", annotation_position="top left",
            )
        fig2.update_layout(xaxis_title="Batch", yaxis_title="Score", yaxis_range=[0, 1])
        st.plotly_chart(fig2, width="stretch")
        st.caption(
            "Accuracy alone can hide problems (78% of clients do not default), "
            "so AUC and F1 are tracked too."
        )

    st.subheader("Max PSI per batch")
    fig3 = px.line(batch_log, x="batch", y="max_psi", markers=True)
    fig3.add_hline(y=PSI_RED, line_dash="dash", line_color="#d64545", annotation_text="0.2 retrain")
    fig3.add_hline(y=PSI_YELLOW, line_dash="dash", line_color="#e8b931", annotation_text="0.1 watch")
    st.plotly_chart(fig3, width="stretch")

    st.subheader("Inspect a batch")
    b = st.slider("Batch", int(batch_log["batch"].min()), int(batch_log["batch"].max()),
                  int(batch_log["batch"].max()))
    detail = feature_log[feature_log["batch"] == b].drop(columns="batch").copy()
    detail.insert(0, "", detail["status"].map(STATUS_ICON))
    st.dataframe(detail, hide_index=True, width="stretch")

    # downloadable drift report (full log, one row per batch per feature)
    report = feature_log.merge(
        batch_log[["batch", "model", "accuracy", "auc", "f1"]], on="batch", how="left"
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
    limit_tilt = s1.slider(
        "Credit limit tilt", -1.0, 1.0, 0.0, 0.1,
        help="Resamples real clients: negative favours lower credit limits, positive higher ones.",
    )
    age_tilt = s2.slider(
        "Age tilt", -1.0, 1.0, 0.0, 0.1,
        help="Resamples real clients: positive favours older clients, negative younger ones.",
    )
    concept = s3.slider("Concept drift (0 to 1)", 0.0, 1.0, 0.0, 0.1)
    size = s4.select_slider("Batch size", [200, 500, 1000, 2000], value=500)

    if st.button("Generate batch and check", type="primary"):
        batch = make_batch(size, limit_tilt=limit_tilt, age_tilt=age_tilt, concept=concept)
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
        st.dataframe(rep, hide_index=True, width="stretch")
        st.caption(
            "Tip: input drift alone turns PSI red but barely hurts AUC. "
            "Add concept drift to see performance fall. Drift is measured against "
            f"the training data of the active model ({active['version']})."
        )

# =====================================================
# TAB: explain predictions with SHAP
# =====================================================
@st.cache_resource(max_entries=8)
def get_explainer(model_uri):
    return mdl.explainer(load_model(model_uri))


@st.cache_data(max_entries=8)
def global_importance(model_uri, reference_uri):
    sample = load_reference(reference_uri).sample(1000, random_state=0)
    imp = mdl.global_importance(load_model(model_uri), sample, get_explainer(model_uri))
    return imp.rename("mean |SHAP|").reset_index().rename(columns={"index": "feature"})


with tab_explain:
    st.write(
        f"Why does **{active['version']}** flag a client? SHAP splits each prediction into one "
        "contribution per feature, in log-odds: positive pushes towards default, negative "
        "away from it. They add up exactly to the model's output."
    )
    left, right = st.columns([2, 3])
    with left:
        st.subheader("What the model relies on")
        imp = global_importance(active["model_uri"], active["reference_uri"])
        fig_imp = px.bar(imp.iloc[::-1], x="mean |SHAP|", y="feature", orientation="h")
        fig_imp.update_layout(height=380, margin={"t": 10, "b": 10}, yaxis_title=None)
        st.plotly_chart(fig_imp, width="stretch")
        st.caption("Mean absolute contribution over 1,000 training clients.")

    with right:
        st.subheader("Explain one client")
        batches = load_batches(root)
        last_batch = int(batches["batch"].max())
        e1, e2 = st.columns(2)
        b_pick = e1.number_input("Batch", 0, last_batch, last_batch)
        rows = batches[batches["batch"] == b_pick].reset_index(drop=True)
        i_pick = e2.number_input("Client in batch", 0, len(rows) - 1, 0)
        client = rows.drop(columns=[TARGET, "batch"]).iloc[[int(i_pick)]]
        proba = float(model.predict_proba(client)[0, 1])
        flagged = bool(model.predict(client)[0])
        contrib, base = mdl.contributions(model, client, get_explainer(active["model_uri"]))
        k1, k2, k3 = st.columns(3)
        k1.metric("Probability of default", f"{proba:.1%}")
        k2.metric(f"Decision (threshold {mdl.threshold(model):.0%})", "🔴 flag" if flagged else "🟢 approve")
        k3.metric("Actually defaulted", "yes" if int(rows.loc[int(i_pick), TARGET]) else "no")
        c = contrib.iloc[0].sort_values(key=abs)
        values = client.iloc[0]
        fig_c = go.Figure(go.Bar(
            x=c.values, y=[f"{f} = {values[f]:g}" for f in c.index], orientation="h",
            marker_color=["#d64545" if v > 0 else "#2e9e5b" for v in c.values],
            hovertemplate="%{x:+.3f} log-odds<extra></extra>",
        ))
        fig_c.update_layout(height=380, margin={"t": 10, "b": 10},
                            xaxis_title="Contribution (log-odds; red raises risk, green lowers it)")
        st.plotly_chart(fig_c, width="stretch")
        st.caption(
            f"Average client: log-odds {base:+.2f}. Plus these contributions: "
            f"{base + c.sum():+.2f}, which is a probability of "
            f"{1 / (1 + np.exp(-(base + c.sum()))):.1%} before the threshold step."
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
        entry = retrain(trigger="manual (button)", root=root)
        verdict = "promoted to active" if entry["promoted"] else "rejected (not better)"
        st.session_state["flash"] = f"{entry['version']} {verdict}"
        st.rerun()
    if b2.button("Reset to v1"):
        reset_registry(root)
        reset_stream(root)
        st.session_state["flash"] = "Registry reset to v1 and streamed batches cleared"
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
                "threshold": m["metrics"].get("threshold"),
                "evaluated_on": m["evaluated_on"],
                "mlflow_run": (m.get("run_id") or "")[:8],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"The registry is MLflow: each row is a version of the registered model "
        f"`{registry.MODEL_NAME}`, and the active one is where the `{registry.ALIAS}` alias "
        "points. `mlflow_run` is the run that holds its parameters, metrics and model file."
    )
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
        st.plotly_chart(fig4, width="stretch")
        if last["promoted"]:
            st.success(f"{new_v} beat {old_v} on AUC and is now the active model.")
        else:
            st.warning(f"{new_v} did not beat {old_v}, so {old_v} stays active.")
        st.caption(f"Trigger: {last['trigger']}")
    else:
        st.info("No retraining yet. Click **Retrain now** or enable auto-retrain in the sidebar.")

# =====================================================
# TAB 4: model card (reads the registry, so it stays current)
# =====================================================
with tab_card:
    st.header(f"Model card: credit default classifier ({active['version']})")

    train_df = pd.read_csv(TRAIN) if Path(TRAIN).exists() else None

    st.subheader("Overview")
    st.write(
        "A gradient-boosting classifier (scikit-learn's HistGradientBoostingClassifier, "
        "hyperparameters tuned by 5-fold cross-validation) that predicts whether a credit "
        "card client will default on next month's payment. It outputs a calibrated "
        "probability, and flags a client when that probability reaches a decision threshold "
        "tuned for F1. Built to demonstrate monitoring, drift detection and retraining, not "
        "to make real lending decisions."
    )
    am = active["metrics"]
    o1, o2, o3, o4, o5 = st.columns(5)
    o1.metric("Version", active["version"])
    o2.metric("Trained on", active["trained_at"][:10])
    o3.metric("AUC", f"{am['auc']:.3f}")
    o4.metric("F1", f"{am['f1']:.3f}")
    o5.metric("Decision threshold", f"{am['threshold']:.2f}" if "threshold" in am else "0.50")
    st.caption(f"Metrics measured on: {active['evaluated_on']}. Trigger: {active['trigger']}.")

    st.subheader("Data")
    if train_df is not None:
        d1, d2, d3 = st.columns(3)
        d1.metric("Training pool rows", f"{len(train_df):,}")
        d2.metric("Default rate", f"{train_df[TARGET].mean():.1%}")
        d3.metric("Features", train_df.shape[1] - 1)
    st.write(
        "**Real data:** the UCI *Default of Credit Card Clients* dataset (Yeh and Lien, "
        "2009): 30,000 clients in Taiwan, April to September 2005, with real default "
        "labels. 20,000 rows train v1. The other 10,000 are never trained on and feed the "
        "monitoring batches. Source and license: `data/SOURCE.md`.\n\n"
        f"**Features ({len(FEATURE_NAMES)}):** {', '.join(FEATURE_NAMES)}. `SEX` is in the "
        "dataset but deliberately left out of the model.\n\n"
        "**What is simulated:** input drift resamples real clients (for example more older "
        "clients), so every row is real. Concept drift replaces a share of labels with a "
        "different default rule, because the dataset has no real concept drift to replay."
    )

    st.subheader("Intended use")
    st.write(
        "Portfolio demo of an MLOps loop: monitor, detect drift, retrain, compare, "
        "promote. Not for real credit decisions."
    )

    st.subheader("Monitoring and retraining")
    st.markdown(
        f"- **Drift test:** PSI per feature (green < 0.1, yellow 0.1 to 0.2, red > 0.2) "
        "plus a KS test, both written from scratch with numpy and scipy.\n"
        f"- **Retrain rule:** max PSI > {PSI_RED}, or AUC more than {AUC_DROP} below the "
        f"active model's own baseline (mean of its last {RULE_WINDOW} batches). Each model "
        "is judged only on batches it scored itself.\n"
        f"- **Retraining data:** the last {RECENT_BATCHES} labeled batches.\n"
        "- **Promotion:** a new version replaces the active one only if its AUC is at "
        "least as good on the same holdout.\n"
        "- **Tracking and registry:** every training run is logged to MLflow (parameters, "
        "metrics, model, reference data). The model in production is the registered "
        f"version the `{registry.ALIAS}` alias points at. Model files use the skops "
        "format, which cannot run code when loaded (unlike pickle)."
    )

    st.subheader("Calibration: can the probabilities be trusted?")
    calib = registry.load_artifact(active, "calibration.csv")
    if "ece" in am:
        c1, c2 = st.columns([1, 2])
        c1.metric("Calibration error (ECE)", f"{am['ece']:.3f}")
        c1.metric("Brier score", f"{am['brier']:.3f}")
        c1.caption(
            "ECE is the average gap between the predicted default rate and the real one. "
            "The earlier RandomForest with balanced class weights scored 0.21: its '45%' "
            "meant about 22%."
        )
        if calib is not None:
            figc = go.Figure()
            figc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                                      line={"dash": "dot", "color": "#888"}))
            figc.add_trace(go.Scatter(
                x=calib["predicted"], y=calib["observed"], mode="lines+markers",
                name=active["version"], text=calib["n"],
                hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<br>%{text} clients",
            ))
            figc.update_layout(xaxis_title="Predicted probability of default",
                               yaxis_title="Observed default rate", height=320,
                               margin={"t": 10, "b": 10}, xaxis_range=[0, 1], yaxis_range=[0, 1])
            c2.plotly_chart(figc, width="stretch")

    st.subheader("Fairness across age groups")
    fair = registry.load_artifact(active, "fairness.csv")
    fsum = registry.load_artifact(active, "fairness_summary.json")
    if fair is not None and fsum is not None:
        f1_, f2_, f3_ = st.columns(3)
        mark = lambda ok: "✅" if ok else "❌"  # noqa: E731
        f1_.metric(f"{mark(fsum['passes_four_fifths'])} Selection-rate ratio (≥ 0.8)",
                   f"{fsum['selection_rate_ratio']:.2f}")
        f2_.metric(f"{mark(fsum['passes_equal_opportunity'])} Recall gap (≤ 0.1)", f"{fsum['tpr_gap']:.3f}")
        if "calibration_gap" in fsum:
            f3_.metric(f"{mark(fsum['calibrated_within_groups'])} Calibration gap (≤ 0.05)",
                       f"{fsum['calibration_gap']:.3f}")
        st.dataframe(fair, hide_index=True, width="stretch")
        st.caption(
            "Measured on the same data as the metrics above. Selection rate is the share of a "
            "group flagged as likely to default; recall is the share of real defaulters "
            "caught. When real default rates differ between groups, a model that is "
            "calibrated within each group flags each group at its own rate, so the "
            "four-fifths check can fail while recall and calibration are even. These "
            "criteria can't all hold at once; which matters most is a policy and legal "
            "decision, not a modelling one."
        )
    else:
        st.info("This version has no fairness report.")

    st.subheader("Known limits and failure modes")
    st.markdown(
        "- **Moderate ceiling.** Test AUC is about 0.77. RandomForest, logistic regression "
        "and gradient boosting all land within 0.01 AUC of each other, so the limit is the "
        "information in these 10 features, not the algorithm.\n"
        "- **The threshold is a business choice.** It's tuned for F1, which treats a missed "
        "defaulter and a false alarm as equally bad. A lender would set it from the real "
        "cost of each.\n"
        "- **Old data from one market.** Taiwan, 2005. Credit behaviour today, or "
        "elsewhere, can differ a lot.\n"
        "- **Silent failure under concept drift.** If the default rule changes but the "
        "inputs look the same, PSI stays green while AUC falls (the injector tab shows "
        "this). Input monitoring alone cannot catch it.\n"
        "- **Needs labels.** Performance metrics require true outcomes. Real credit "
        "takes months to show a default, so in production the AUC signal arrives late and PSI "
        "is the early warning.\n"
        "- **Small retraining set.** v2 learns from about 1,400 rows, so it can be "
        "unstable. The promotion check guards against a worse model but not against a "
        "lucky holdout.\n"
        "- **PSI limits.** PSI is sensitive to bin choice and to batch size (500 rows "
        "here), and it checks one feature at a time, so it misses joint shifts.\n"
        "- **Age is a feature.** The fairness section shows how outcomes differ by age. "
        "Whether age may be used at all depends on the jurisdiction; a real credit model "
        "would need a legal review."
    )

    st.subheader("Version history")
    for m in reversed(reg["models"]):
        st.write(
            f"**{m['version']}** ({m['status']}): trained {m['trained_at'].replace('T', ' ')}, "
            f"trigger: {m['trigger']}, AUC {m['metrics']['auc']:.3f}"
        )
