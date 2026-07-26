import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from feedback_log import log_decision, load_feedback_summary
from rationale_engine import explain_row
from recommendation_engine import recommend
from trend_analysis import summarize_transitions, detect_anomalies
from report_generator import build_report_pdf

st.set_page_config(page_title="Paper Grade Intelligence", layout="wide")

FEATURES = [
    "Stock_Flow",
    "Filler_Flow",
    "Steam_Pressure",
    "Machine_Speed",
    "Moisture",
    "Ash",
    "Caliper",
    "Wire_Vacuum",
    "Refiner_Load",
    "Draw_Tension",
    "Target_BasisWeight",
    "Time",
]


@st.cache_data
def load_data():
    return pd.read_csv("data/paper_grade_data.csv")


@st.cache_resource
def load_models():
    clf = joblib.load("models/offspec_model.pkl")
    reg_path = Path("models/stabilization_model.pkl")
    reg = joblib.load(reg_path) if reg_path.exists() else None
    return clf, reg


@st.cache_data
def load_correlation_summary():
    path = Path("models/correlation_summary.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_trend_summary(_df):
    summary = summarize_transitions(_df)
    return detect_anomalies(summary)


df = load_data()
clf, reg = load_models()
corr_summary = load_correlation_summary()
trend_summary = load_trend_summary(df)

st.title("📄 Paper Grade Intelligence Dashboard")
st.caption(
    "Predicts Off-Spec risk during grade changes, explains why, recommends "
    "corrective setpoints tagged with their source, and tracks operator "
    "feedback on suggestion quality."
)

tab_monitor, tab_why, tab_corr, tab_trends, tab_feedback = st.tabs(
    ["🔴 Live Monitor", "🧭 Why?", "🔗 Correlations",
     "📈 Trends & Anomalies", "✅ Suggestion Quality"]
)

# ---------------------------------------------------------------------
# Sidebar: pick a transition / time step to inspect
# ---------------------------------------------------------------------
st.sidebar.header("Select Grade Transition")
transition_ids = sorted(df["Transition_ID"].unique())
transition_id = st.sidebar.selectbox("Transition ID", transition_ids, index=0)

t_df = df[df["Transition_ID"] == transition_id].sort_values("Time").reset_index(drop=True)
max_time = int(t_df["Time"].max())
time_step = st.sidebar.slider("Time step into transition", 0, max_time, min(5, max_time))

row = t_df[t_df["Time"] == time_step].iloc[0]
grade_from, grade_to = int(row["Grade_From"]), int(row["Grade_To"])
st.sidebar.markdown(f"**Grade change:** {grade_from} → {grade_to} gsm")

X_row = pd.DataFrame([row[FEATURES]])
risk = float(clf.predict_proba(X_row)[0][1])
eta = float(reg.predict(X_row)[0]) if reg is not None else None

# ---------------------------------------------------------------------
# Tab 1: Live Monitor
# ---------------------------------------------------------------------
with tab_monitor:
    c1, c2, c3 = st.columns(3)
    c1.metric("Off-Spec Risk", f"{risk*100:.1f}%")
    c2.metric("Actual Basis Weight", f"{row['Actual_BasisWeight']:.2f}",
              delta=f"{row['Actual_BasisWeight'] - row['Target_BasisWeight']:.2f} vs target")
    if eta is not None:
        c3.metric("Est. Time to Stable", f"{eta:.1f} steps")

    if risk > 0.6:
        st.error("⚠️ High risk of Off-Spec production. Review recommendations below.")
    elif risk > 0.3:
        st.warning("Elevated risk - monitor closely.")
    else:
        st.success("System tracking within safe limits.")

    st.subheader("Basis Weight Trajectory")
    traj = t_df[["Time", "Actual_BasisWeight", "Target_BasisWeight"]].set_index("Time")
    st.line_chart(traj)
    st.caption("Vertical read: dashed target vs. actual basis weight across the transition. "
               "Use the sidebar slider to move the current-time marker.")

    with st.expander("Current Process Parameters"):
        st.write(row[FEATURES])

    st.divider()
    st.subheader("Recommendations")

    recs = recommend(row, df)

    for idx, r in enumerate(recs):
        with st.container(border=True):
            st.markdown(f"**{r.action}**")
            st.write(r.rationale)
            cols = st.columns([2, 2, 1, 1])
            cols[0].write(f"Source: `{r.source}`")
            cols[1].write(f"Confidence: **{r.confidence}%**")

            key_base = f"{transition_id}_{time_step}_{idx}"
            if cols[2].button("✅ Accept", key=f"accept_{key_base}"):
                log_decision(transition_id, time_step, r.parameter, r.action,
                             r.source, r.confidence, "accepted")
                st.success("Recorded: accepted")
            if cols[3].button("❌ Reject", key=f"reject_{key_base}"):
                log_decision(transition_id, time_step, r.parameter, r.action,
                             r.source, r.confidence, "rejected")
                st.error("Recorded: rejected")

    st.divider()
    st.subheader("Session Report")
    st.caption("Snapshot this transition/time-step - risk, trajectory chart, "
               "rationale, and recommendations - as a PDF you can share or file.")

    if st.button("📄 Generate PDF Report"):
        rationale_for_report = explain_row(clf, df, row)
        pdf_buf = build_report_pdf(
            transition_id=transition_id,
            time_step=time_step,
            row=row,
            t_df=t_df,
            risk=risk,
            eta=eta,
            rationale_result=rationale_for_report,
            recommendations=recs,
        )
        st.download_button(
            "⬇️ Download Report PDF",
            data=pdf_buf,
            file_name=f"grade_report_t{transition_id}_step{time_step}.pdf",
            mime="application/pdf",
        )

# ---------------------------------------------------------------------
# Tab 2: Why? (rationale)
# ---------------------------------------------------------------------
with tab_why:
    st.subheader("Why is the model predicting this risk?")
    result = explain_row(clf, df, row)
    st.metric("Risk probability", f"{result['risk_probability']*100:.1f}%")
    st.caption(result["source"])

    for d in result["top_drivers"]:
        st.write(f"**{d['feature'].replace('_', ' ')}**")
        cols = st.columns(3)
        cols[0].write(f"Current: {d['current_value']}")
        cols[1].write(f"Healthy reference: {d['healthy_reference']}")
        cols[2].write(f"Model importance: {d['model_importance']}")
        st.caption(d["text"])
        st.divider()

# ---------------------------------------------------------------------
# Tab 3: Correlations
# ---------------------------------------------------------------------
with tab_corr:
    st.subheader("Correlation Heatmap")
    heatmap_path = Path("models/correlation_heatmap.png")
    if heatmap_path.exists():
        st.image(str(heatmap_path), use_container_width=True)
    else:
        st.info("Run correlation_analysis.py to generate the heatmap.")

    if corr_summary:
        st.subheader("Known Control-Loop Correlations vs. Off-Spec Risk")
        known = corr_summary["targets"]["Off_Spec"]["known_loop_correlations"]
        st.bar_chart(pd.Series(known))

        st.subheader("🆕 Newly Discovered Correlations (outside standard control loops)")
        new_corr = corr_summary["targets"]["Off_Spec"]["new_correlations"]
        st.bar_chart(pd.Series(new_corr))
        st.caption(
            "These sensors (e.g. Wire Vacuum, Refiner Load, Draw Tension) are "
            "historian-logged but not part of the MD controller's standard "
            "loop set. A meaningful correlation here indicates an "
            "opportunity to expand the control/monitoring scope."
        )
    else:
        st.info("Run correlation_analysis.py to generate correlation_summary.json.")

# ---------------------------------------------------------------------
# Tab 4: Trends & Anomalies (across all historical transitions)
# ---------------------------------------------------------------------
with tab_trends:
    st.subheader("Stabilization Time vs. Peak Deviation, All Transitions")
    st.caption(
        "Each point is one historical grade transition. Points flagged as "
        "anomalies deviated more than 2 standard deviations from other "
        "transitions to the same target grade."
    )

    chart_df = trend_summary.copy()
    chart_df["Status"] = chart_df["Is_Anomaly"].map({True: "Anomaly", False: "Normal"})
    st.scatter_chart(
        chart_df,
        x="Stabilization_Time",
        y="Peak_Deviation_Pct",
        color="Status",
    )

    n_anom = int(trend_summary["Is_Anomaly"].sum())
    st.metric("Anomalous transitions flagged", f"{n_anom} / {len(trend_summary)}")

    st.subheader("Flagged Transitions")
    anomalies = trend_summary[trend_summary["Is_Anomaly"]].sort_values(
        "Peak_Deviation_Pct", ascending=False
    )
    if anomalies.empty:
        st.info("No anomalous transitions detected in the current dataset.")
    else:
        st.dataframe(
            anomalies[[
                "Transition_ID", "Grade_From", "Grade_To",
                "Peak_Deviation_Pct", "Stabilization_Time",
                "Off_Spec_Rate", "Anomaly_Reasons",
            ]].reset_index(drop=True),
            use_container_width=True,
        )
        st.caption(
            "Tip: pick a Transition ID from this table in the sidebar to "
            "inspect it in the Live Monitor tab."
        )

# ---------------------------------------------------------------------
# Tab 5: Suggestion Quality (operator feedback)
# ---------------------------------------------------------------------
with tab_feedback:
    st.subheader("Recommendation Acceptance Rate by Source")
    summary = load_feedback_summary()
    if summary is None:
        st.info("No operator feedback recorded yet. Accept/Reject a "
                 "recommendation in the Live Monitor tab to populate this.")
    else:
        st.dataframe(summary.style.format("{:.0%}"))
        st.bar_chart(summary)