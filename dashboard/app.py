"""Streamlit security dashboard - a miniature Network Security Operations Centre.

Run with::

    streamlit run dashboard/app.py

The dashboard talks to the model **directly** through :class:`ThreatPredictor`
rather than through the REST API. That is deliberate: the two are independent
consumers of the same inference layer, so the dashboard keeps working when the
API is down, and neither can drift from the other because both call the same
code. The API's reachability is surfaced in the sidebar for transparency.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import ATTACK_CLASSES, BENIGN_LABEL, PATHS, SERVICE
from src.data_loader import load_sample_data
from src.evaluate import load_metrics
from src.predict import ThreatPredictor, example_flow

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="NIDS | Threat Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_COLOURS = {
    "LOW": "#2E933C",
    "MEDIUM": "#E5B25D",
    "HIGH": "#E06C3B",
    "CRITICAL": "#C1292E",
}

ATTACK_COLOURS = {
    "BENIGN": "#2E933C",
    "DDoS": "#C1292E",
    "DoS": "#E06C3B",
    "PortScan": "#E5B25D",
    "BruteForce": "#8E6C88",
    "WebAttack": "#4059AD",
    "Botnet": "#6B2737",
    "Infiltration": "#3D348B",
    "Other": "#7A7A7A",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
      .banner {
          padding: 0.6rem 1rem; border-radius: 6px; margin-bottom: 1rem;
          background: #FFF4E5; border-left: 5px solid #E5B25D; font-size: 0.9rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Loading model artifacts…")
def load_predictor() -> ThreatPredictor | None:
    """Load the model once per Streamlit session."""
    try:
        return ThreatPredictor()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame | None:
    """Load the bundled synthetic sample, if present."""
    try:
        return load_sample_data()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def load_report() -> dict[str, Any] | None:
    """Load the training metrics report written by ``train_model.py``."""
    try:
        return load_metrics()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Scoring flows…")
def score_frame(frame: pd.DataFrame, _predictor: ThreatPredictor) -> pd.DataFrame:
    """Score a frame, cached on the frame's content so reruns are instant."""
    return _predictor.predict_csv(frame, explain=False)


def mask_ip(value: Any) -> str:
    """Anonymise the host portion of an IPv4 address before display.

    The dashboard is a demo surface that may be screenshotted or shared, so
    endpoint identifiers are truncated. Retaining the first two octets keeps
    the subnet visible, which is what an analyst actually triages on.
    """
    text = str(value)
    parts = text.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.x.x"
    return text


def risk_badge(level: str) -> str:
    colour = RISK_COLOURS.get(level, "#7A7A7A")
    return (
        f"<span style='background:{colour};color:white;padding:2px 10px;"
        f"border-radius:10px;font-weight:600;font-size:0.85rem'>{level}</span>"
    )


def style_risk(frame: pd.DataFrame) -> Any:
    """Colour the risk_level column of a results table."""
    def colourise(value: Any) -> str:
        return f"color: {RISK_COLOURS.get(str(value), '#333')}; font-weight: 600"

    if "risk_level" in frame.columns:
        return frame.style.map(colourise, subset=["risk_level"])
    return frame


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

predictor = load_predictor()
sample_df = load_sample()
report = load_report()

with st.sidebar:
    st.title("🛡️ NIDS Control")

    if predictor is None:
        st.error("No trained model found.")
        st.code("python scripts/prepare_data.py\npython scripts/train_model.py", language="bash")
    else:
        st.success(f"Model: **{predictor.bundle.model_name}**")
        st.caption(
            f"v{predictor.bundle.model_version} · trained {predictor.bundle.trained_at[:10]}\n\n"
            f"{len(predictor.class_names)} classes · {predictor.bundle.feature_count} features"
        )
        st.caption(
            "Anomaly detector: "
            + ("loaded ✅" if predictor.anomaly is not None else "not available ⚠️")
        )

    st.divider()
    st.caption(f"REST API target: `{SERVICE.api_url}`")
    st.caption("The dashboard scores locally; the API is a separate consumer of the same model.")

    st.divider()
    st.warning(
        "**Portfolio project.** Risk scores are engineering heuristics, not "
        "calibrated probabilities or a production security standard.",
        icon="⚠️",
    )

if predictor is None:
    st.title("Network Intrusion Detection System")
    st.error(
        "No trained model is available. Run the two pipeline scripts below, then reload this page."
    )
    st.code(
        "python scripts/generate_sample_data.py   # optional: synthetic demo data\n"
        "python scripts/prepare_data.py --use-sample\n"
        "python scripts/train_model.py",
        language="bash",
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("Network Intrusion Detection & Threat Monitor")

tab_overview, tab_analyzer, tab_csv, tab_live, tab_perf = st.tabs(
    ["📊 Overview", "🔍 Flow Analyzer", "📁 CSV Analyzer", "📡 Live Monitor", "📈 Model Performance"]
)


# --------------------------------------------------------------------------- #
# Tab 1 - Overview
# --------------------------------------------------------------------------- #

with tab_overview:
    if sample_df is None:
        st.info(
            "No sample dataset found. Generate one with "
            "`python scripts/generate_sample_data.py` to populate this view."
        )
    else:
        st.markdown(
            "<div class='banner'><b>Synthetic demonstration traffic.</b> "
            "These flows were generated programmatically to exercise the pipeline. "
            "They are not real network captures and not CIC-IDS2017.</div>",
            unsafe_allow_html=True,
        )

        max_rows = st.slider(
            "Flows to analyse", 500, min(len(sample_df), 20_000), min(4_000, len(sample_df)), step=500
        )
        subset = sample_df.head(max_rows)
        scored = score_frame(subset, predictor)
        summary = predictor.summarize(scored)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Flows analysed", f"{summary['total_records']:,}")
        c2.metric("Benign", f"{summary['benign_records']:,}")
        c3.metric("Malicious", f"{summary['malicious_records']:,}")
        c4.metric("Attack rate", f"{summary['attack_rate'] * 100:.1f}%")
        c5.metric("Anomalies", f"{summary['anomalies_detected']:,}")
        c6.metric("Critical threats", f"{summary['critical_threats']:,}")

        st.divider()
        left, right = st.columns([3, 2])

        with left:
            st.subheader("Attack distribution")
            counts = (
                scored.loc[scored["is_malicious"], "prediction"].value_counts().reset_index()
            )
            counts.columns = ["Attack type", "Flows"]
            if counts.empty:
                st.info("No malicious flows detected in this subset.")
            else:
                fig = px.bar(
                    counts, x="Attack type", y="Flows", color="Attack type",
                    color_discrete_map=ATTACK_COLOURS, text="Flows",
                )
                fig.update_layout(showlegend=False, height=360, margin=dict(t=20, b=20))
                st.plotly_chart(fig, width='stretch')

        with right:
            st.subheader("Risk breakdown")
            risk_counts = scored["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk level", "Flows"]
            fig = px.pie(
                risk_counts, names="Risk level", values="Flows", hole=0.45,
                color="Risk level", color_discrete_map=RISK_COLOURS,
            )
            fig.update_layout(height=360, margin=dict(t=20, b=20))
            st.plotly_chart(fig, width='stretch')

        # ---- Timeline ---------------------------------------------------- #
        timestamp_col = next(
            (c for c in subset.columns if c.strip().lower() == "timestamp"), None
        )
        if timestamp_col is not None:
            st.subheader("Traffic timeline")
            timeline = scored.copy()
            timeline["ts"] = pd.to_datetime(subset[timestamp_col].values, errors="coerce")
            timeline = timeline.dropna(subset=["ts"])
            if not timeline.empty:
                grouped = (
                    timeline.set_index("ts")
                    .resample("30min")
                    .agg(total=("prediction", "size"), malicious=("is_malicious", "sum"))
                    .reset_index()
                )
                grouped["benign"] = grouped["total"] - grouped["malicious"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=grouped["ts"], y=grouped["benign"], name="Benign",
                    stackgroup="one", line=dict(color=RISK_COLOURS["LOW"], width=0),
                ))
                fig.add_trace(go.Scatter(
                    x=grouped["ts"], y=grouped["malicious"], name="Malicious",
                    stackgroup="one", line=dict(color=RISK_COLOURS["CRITICAL"], width=0),
                ))
                fig.update_layout(
                    height=320, margin=dict(t=20, b=20),
                    xaxis_title="Time (30-minute buckets)", yaxis_title="Flows",
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(fig, width='stretch')

        # ---- Threat table ------------------------------------------------ #
        st.subheader("Highest-risk flows")
        threats = scored[scored["is_malicious"]].sort_values("risk_score", ascending=False).head(25)
        if threats.empty:
            st.info("No malicious flows to display.")
        else:
            display = pd.DataFrame({
                "Timestamp": (
                    subset[timestamp_col].iloc[threats.index].values
                    if timestamp_col else ["—"] * len(threats)
                ),
                "Source": [
                    mask_ip(v) for v in (
                        subset["Source IP"].iloc[threats.index].values
                        if "Source IP" in subset.columns else ["—"] * len(threats)
                    )
                ],
                "Destination": [
                    mask_ip(v) for v in (
                        subset["Destination IP"].iloc[threats.index].values
                        if "Destination IP" in subset.columns else ["—"] * len(threats)
                    )
                ],
                "Port": (
                    subset["Destination Port"].iloc[threats.index].values
                    if "Destination Port" in subset.columns else ["—"] * len(threats)
                ),
                "Attack": threats["prediction"].values,
                "Risk": threats["risk_score"].values,
                "Level": threats["risk_level"].values,
                "Confidence": [f"{c:.1%}" for c in threats["confidence"].values],
                "Anomaly": ["yes" if a else "no" for a in threats["is_anomaly"].values],
            })
            st.dataframe(display, width='stretch', hide_index=True, height=420)
            st.caption("Source and destination addresses are masked to their /16 prefix.")


# --------------------------------------------------------------------------- #
# Tab 2 - Single flow analyzer
# --------------------------------------------------------------------------- #

with tab_analyzer:
    st.subheader("Analyse a single network flow")
    st.caption(
        "Enter flow statistics manually, or start from a template. Any field left "
        "at its default is imputed using the medians learned during training."
    )

    template_choice = st.radio(
        "Start from a template",
        ["benign", "scan", "flood"],
        horizontal=True,
        format_func=lambda k: {
            "benign": "Typical HTTPS session",
            "scan": "Short probe to an unusual port",
            "flood": "High-rate one-directional burst",
        }[k],
    )
    template = example_flow(template_choice)
    st.caption("Templates are illustrative inputs, not recorded traffic.")

    with st.form("flow_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Identity & duration**")
            destination_port = st.number_input(
                "Destination port", 0, 65535, int(template["destination_port"])
            )
            protocol = st.selectbox(
                "Protocol", [6, 17, 1], index=0,
                format_func=lambda p: {6: "TCP (6)", 17: "UDP (17)", 1: "ICMP (1)"}[p],
            )
            flow_duration = st.number_input(
                "Flow duration (µs)", 0.0, 1e12, float(template["flow_duration"]), step=1000.0
            )
            st.markdown("**Volume**")
            total_fwd_packets = st.number_input(
                "Total forward packets", 0.0, 1e9, float(template["total_fwd_packets"])
            )
            total_backward_packets = st.number_input(
                "Total backward packets", 0.0, 1e9, float(template["total_backward_packets"])
            )
        with col2:
            st.markdown("**Bytes**")
            total_length_of_fwd_packets = st.number_input(
                "Total forward bytes", 0.0, 1e12, float(template["total_length_of_fwd_packets"])
            )
            total_length_of_bwd_packets = st.number_input(
                "Total backward bytes", 0.0, 1e12, float(template["total_length_of_bwd_packets"])
            )
            flow_bytes_s = st.number_input(
                "Flow bytes/s", 0.0, 1e12, float(template["flow_bytes_s"])
            )
            flow_packets_s = st.number_input(
                "Flow packets/s", 0.0, 1e12, float(template["flow_packets_s"])
            )
            st.markdown("**Packet shape**")
            packet_length_mean = st.number_input(
                "Packet length mean", 0.0, 1e6, float(template["packet_length_mean"])
            )
            packet_length_std = st.number_input(
                "Packet length std", 0.0, 1e6, float(template["packet_length_std"])
            )
        with col3:
            st.markdown("**TCP flags**")
            syn_flag_count = st.number_input("SYN count", 0.0, 1e9, float(template["syn_flag_count"]))
            ack_flag_count = st.number_input("ACK count", 0.0, 1e9, float(template["ack_flag_count"]))
            fin_flag_count = st.number_input("FIN count", 0.0, 1e9, float(template["fin_flag_count"]))
            psh_flag_count = st.number_input("PSH count", 0.0, 1e9, float(template["psh_flag_count"]))
            rst_flag_count = st.number_input("RST count", 0.0, 1e9, float(template["rst_flag_count"]))
            st.markdown("**Directional means**")
            fwd_packet_length_mean = st.number_input(
                "Fwd packet length mean", 0.0, 1e6, float(template["fwd_packet_length_mean"])
            )
            bwd_packet_length_mean = st.number_input(
                "Bwd packet length mean", 0.0, 1e6, float(template["bwd_packet_length_mean"])
            )

        submitted = st.form_submit_button("Analyse flow", type="primary", width='stretch')

    if submitted:
        flow = {
            "destination_port": destination_port,
            "protocol": protocol,
            "flow_duration": flow_duration,
            "total_fwd_packets": total_fwd_packets,
            "total_backward_packets": total_backward_packets,
            "total_length_of_fwd_packets": total_length_of_fwd_packets,
            "total_length_of_bwd_packets": total_length_of_bwd_packets,
            "flow_bytes_s": flow_bytes_s,
            "flow_packets_s": flow_packets_s,
            "packet_length_mean": packet_length_mean,
            "packet_length_std": packet_length_std,
            "average_packet_size": packet_length_mean,
            "fwd_packet_length_mean": fwd_packet_length_mean,
            "bwd_packet_length_mean": bwd_packet_length_mean,
            "syn_flag_count": syn_flag_count,
            "ack_flag_count": ack_flag_count,
            "fin_flag_count": fin_flag_count,
            "psh_flag_count": psh_flag_count,
            "rst_flag_count": rst_flag_count,
        }
        result = predictor.predict_one(flow, explain=True, top_k=6)

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prediction", result.prediction)
        m2.metric("Confidence", f"{result.confidence:.1%}")
        m3.metric("Risk score", f"{result.risk_score}/100")
        m4.metric("Anomaly", "YES" if result.is_anomaly else "no")
        st.markdown("Risk level: " + risk_badge(result.risk_level), unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.markdown("**Why this classification**")
            if result.top_features:
                contrib = pd.DataFrame(result.top_features)
                contrib["abs"] = contrib["contribution"].abs()
                contrib = contrib.sort_values("abs")
                fig = px.bar(
                    contrib, x="contribution", y="display_name", orientation="h",
                    color="direction",
                    color_discrete_map={"increases": "#C1292E", "decreases": "#2E933C"},
                )
                fig.update_layout(
                    height=320, margin=dict(t=10, b=10), yaxis_title="", xaxis_title="SHAP contribution",
                    legend=dict(orientation="h", y=1.15),
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("Explanations are unavailable for this model.")

        with right:
            st.markdown("**Risk score composition**")
            comp = pd.DataFrame(
                [{"Signal": k.replace("_", " ").title(), "Contribution": v}
                 for k, v in result.risk_components.items()]
            )
            fig = px.bar(comp, x="Contribution", y="Signal", orientation="h")
            fig.update_layout(height=250, margin=dict(t=10, b=10), yaxis_title="")
            st.plotly_chart(fig, width='stretch')

            if result.indicators:
                st.markdown("**Traffic indicators triggered**")
                for indicator in result.indicators:
                    st.markdown(f"- {indicator}")

        with st.expander("Full class probability distribution"):
            proba = pd.DataFrame(
                [{"Class": k, "Probability": v} for k, v in result.class_probabilities.items()]
            ).sort_values("Probability", ascending=False)
            st.dataframe(proba, width='stretch', hide_index=True)


# --------------------------------------------------------------------------- #
# Tab 3 - CSV analyzer
# --------------------------------------------------------------------------- #

with tab_csv:
    st.subheader("Batch-analyse a CSV of flows")
    st.caption(
        f"Upload a CSV of network flow records. Maximum {SERVICE.max_upload_mb:.0f} MB / "
        f"{SERVICE.max_upload_rows:,} rows. Column names are normalised automatically; "
        "missing features are imputed."
    )

    uploaded = st.file_uploader("Flow records (CSV)", type=["csv"])
    if uploaded is not None:
        size_mb = uploaded.size / (1024 * 1024)
        if size_mb > SERVICE.max_upload_mb:
            st.error(f"File is {size_mb:.1f} MB, above the {SERVICE.max_upload_mb:.0f} MB limit.")
        else:
            try:
                frame = pd.read_csv(uploaded)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not parse the CSV: {exc}")
                frame = None

            if frame is not None and not frame.empty:
                if len(frame) > SERVICE.max_upload_rows:
                    st.warning(
                        f"File has {len(frame):,} rows; scoring the first "
                        f"{SERVICE.max_upload_rows:,}."
                    )
                    frame = frame.head(SERVICE.max_upload_rows)

                scored = predictor.predict_csv(frame, explain=False)
                summary = predictor.summarize(scored)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Records", f"{summary['total_records']:,}")
                c2.metric("Malicious", f"{summary['malicious_records']:,}")
                c3.metric("Attack rate", f"{summary['attack_rate'] * 100:.1f}%")
                c4.metric("Critical", f"{summary['critical_threats']:,}")

                dist = pd.DataFrame(
                    [{"Class": k, "Flows": v} for k, v in summary["attack_distribution"].items()]
                )
                fig = px.bar(
                    dist, x="Class", y="Flows", color="Class",
                    color_discrete_map=ATTACK_COLOURS, text="Flows",
                )
                fig.update_layout(showlegend=False, height=340, margin=dict(t=20, b=20))
                st.plotly_chart(fig, width='stretch')

                st.markdown("**High-risk flows**")
                high_risk = scored[scored["risk_level"].isin(["HIGH", "CRITICAL"])]
                if high_risk.empty:
                    st.success("No high-risk flows found in this file.")
                else:
                    preview_cols = [
                        c for c in ("prediction", "confidence", "risk_score", "risk_level",
                                    "anomaly_score", "is_anomaly", "indicators")
                        if c in high_risk.columns
                    ]
                    st.dataframe(
                        high_risk[preview_cols].sort_values("risk_score", ascending=False).head(200),
                        width='stretch', height=380,
                    )

                st.download_button(
                    "⬇️ Download full predictions (CSV)",
                    data=scored.to_csv(index=False).encode("utf-8"),
                    file_name=f"nids_predictions_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv",
                    width='stretch',
                )
            elif frame is not None:
                st.error("The uploaded CSV contains no rows.")
    else:
        if sample_df is not None:
            st.info("No file uploaded. You can export the bundled sample and re-upload it to try this tab.")
            st.download_button(
                "⬇️ Download the synthetic sample CSV",
                data=sample_df.head(2000).to_csv(index=False).encode("utf-8"),
                file_name="sample_flows.csv",
                mime="text/csv",
            )


# --------------------------------------------------------------------------- #
# Tab 4 - Live monitor (simulated)
# --------------------------------------------------------------------------- #

with tab_live:
    st.subheader("Live network monitor")
    st.markdown(
        "<div class='banner'><b>Simulated traffic.</b> This replays rows from a stored "
        "CSV through the model one at a time. It does not capture from a network "
        "interface and shows no live production traffic.</div>",
        unsafe_allow_html=True,
    )

    if sample_df is None:
        st.info("Generate the sample dataset first: `python scripts/generate_sample_data.py`.")
    else:
        if "sim_index" not in st.session_state:
            st.session_state.sim_index = 0
            st.session_state.sim_history = []

        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        streaming = c1.toggle("▶ Stream", value=False, help="Auto-advance through the capture.")
        batch_size = c2.selectbox("Flows per tick", [1, 5, 10, 25], index=1)
        delay = c3.selectbox("Tick delay (s)", [0.5, 1.0, 2.0], index=1)
        if c4.button("↺ Reset monitor"):
            st.session_state.sim_index = 0
            st.session_state.sim_history = []
            st.rerun()

        start = st.session_state.sim_index
        end = min(start + batch_size, len(sample_df))
        if start >= len(sample_df):
            st.success("Reached the end of the simulated capture. Reset to replay.")
        else:
            window = sample_df.iloc[start:end]
            results = predictor.predict_frame(window, explain=False)

            for offset, result in enumerate(results):
                row = window.iloc[offset]
                st.session_state.sim_history.append({
                    "Timestamp": row.get("Timestamp", "—"),
                    "Source": mask_ip(row.get("Source IP", "—")),
                    "Destination": mask_ip(row.get("Destination IP", "—")),
                    "Port": row.get("Destination Port", "—"),
                    "Attack": result.prediction,
                    "Risk": result.risk_score,
                    "Level": result.risk_level,
                    "Confidence": result.confidence,
                    "Anomaly": result.is_anomaly,
                })
            st.session_state.sim_index = end

            latest = results[-1]
            history = pd.DataFrame(st.session_state.sim_history)
            malicious_total = int((history["Attack"] != BENIGN_LABEL).sum())

            st.divider()
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Packets analysed", f"{len(history):,}")
            m2.metric("Threats detected", f"{malicious_total:,}")
            m3.metric("Current status", "THREAT" if latest.is_malicious else "CLEAR")
            m4.metric("Confidence", f"{latest.confidence:.0%}")
            m5.metric("Anomaly", "YES" if latest.is_anomaly else "NO")

            if latest.is_malicious:
                st.error(
                    f"**THREAT DETECTED** — {latest.prediction} · "
                    f"risk {latest.risk_score}/100 · {latest.risk_level}",
                    icon="🚨",
                )
            else:
                st.success(
                    f"Traffic nominal — BENIGN · risk {latest.risk_score}/100", icon="✅"
                )

            st.markdown("**Recent flows**")
            recent = history.tail(15).iloc[::-1]
            st.dataframe(recent, width='stretch', hide_index=True, height=380)

            progress = st.session_state.sim_index / len(sample_df)
            st.progress(progress, text=f"Replayed {st.session_state.sim_index:,} of {len(sample_df):,} flows")

            if streaming:
                time.sleep(float(delay))
                st.rerun()


# --------------------------------------------------------------------------- #
# Tab 5 - Model performance
# --------------------------------------------------------------------------- #

with tab_perf:
    st.subheader("Model performance")

    if report is None:
        st.info("No metrics report found. Run `python scripts/train_model.py` to generate one.")
    else:
        dataset = report.get("dataset", {})
        st.markdown(
            f"Selected **{report.get('selected_model')}** on "
            f"**{report.get('selection_split')}** {report.get('selection_metric')}. "
            f"Trained on {dataset.get('train_rows', 0):,} rows; "
            f"tested on {dataset.get('test_rows', 0):,} held-out rows."
        )

        test = report.get("test_result", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{test.get('accuracy', 0):.3f}")
        c2.metric("Macro F1", f"{test.get('macro_f1', 0):.3f}")
        c3.metric("Macro recall", f"{test.get('macro_recall', 0):.3f}")
        c4.metric("Detection rate", f"{test.get('attack_detection_rate', 0):.3f}")
        c5.metric("False alarm rate", f"{test.get('false_alarm_rate', 0):.3f}")

        st.caption(
            f"False negatives (missed attacks): **{test.get('false_negatives', 0):,}** · "
            f"False positives (analyst noise): **{test.get('false_positives', 0):,}**. "
            "In intrusion detection a false negative is the expensive error, which is "
            "why recall and macro-F1 drive model selection rather than accuracy."
        )

        st.divider()
        left, right = st.columns([3, 2])

        with left:
            st.markdown("**Model comparison (validation split)**")
            table = pd.DataFrame(report.get("comparison_table", []))
            if not table.empty:
                st.dataframe(table, width='stretch', hide_index=True)

        with right:
            st.markdown("**Per-class F1 (test)**")
            per_class = test.get("per_class", {})
            if per_class:
                rows = [
                    {"Class": name, "F1": stats.get("f1", 0),
                     "Recall": stats.get("recall", 0), "Support": stats.get("support", 0)}
                    for name, stats in per_class.items()
                    if name in ATTACK_CLASSES or name in dataset.get("classes", [])
                ]
                if rows:
                    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        st.markdown("**Confusion matrix (test split)**")
        matrix = test.get("confusion_matrix")
        labels = test.get("labels")
        if matrix and labels:
            array = np.array(matrix)
            fig = px.imshow(
                array, x=labels, y=labels, text_auto=True, aspect="auto",
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Flows"),
            )
            fig.update_layout(height=520, margin=dict(t=20, b=20))
            st.plotly_chart(fig, width='stretch')

        source_is_synthetic = "synthetic" in str(report.get("dataset", {}).get("source", "")).lower()
        st.warning(
            "If these metrics were produced from the bundled **synthetic** sample, they "
            "demonstrate that the pipeline runs correctly and say nothing about real-world "
            "detection performance. Retrain on CIC-IDS2017 for meaningful numbers.",
            icon="⚠️",
        )

        with st.expander("Raw metrics report (JSON)"):
            st.json(report)
