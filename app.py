"""
app.py
Streamlit dashboard for the Security Operations Pipeline.
Shows real-time pipeline execution, classification results, threat findings,
AI decisions, and firewall state — all updating live.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from executor import BLOCKED_IPS_LOG, FIREWALL_LOG, PIPELINE_SUMMARY_LOG
from pipeline import run_pipeline

# FIXED: Updated file paths
DATA_PATH = Path(__file__).parent / "test_data_clean.csv"
LOGS_DIR = Path(__file__).parent / "logs"

SEVERITY_COLOUR = {
    "CRITICAL": "#ef4444",
    "HIGH": "#f97316",
    "MEDIUM": "#eab308",
    "LOW": "#22c55e",
}
ACTION_COLOUR = {
    "BLOCK_SIGNATURE": "#ef4444",
    "ESCALATE": "#f97316",
    "SUPPRESS": "#eab308",
    "MONITOR": "#3b82f6",
}

st.set_page_config(
    page_title="SecOps Pipeline",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { background-color: #0f1117; }
    .stMetric { background: #1e2130; border-radius: 8px; padding: 12px; }
    .finding-card {
        background: #1e2130; border-radius: 8px; padding: 14px;
        margin-bottom: 10px; border-left: 4px solid #3b82f6;
    }
    .finding-card.CRITICAL { border-left-color: #ef4444; }
    .finding-card.HIGH     { border-left-color: #f97316; }
    h1, h2, h3 { color: #e2e8f0; }
    .stAlert { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/security-shield-green.png",
        width=72,
    )
    st.title("SecOps Pipeline")
    st.caption("Doctoral Research Dashboard")
    st.divider()

    csv_path = st.text_input("Dataset CSV", value=str(DATA_PATH))
    use_agent = st.toggle("Enable AI Agent (OpenAI)", value=True)
    batch_size = st.slider("Classifier batch size", 50, 500, 200, step=50)

    st.divider()
    run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)
    st.divider()

    st.subheader("Log Files")
    for log in [FIREWALL_LOG, BLOCKED_IPS_LOG, PIPELINE_SUMMARY_LOG]:
        exists = log.exists()
        st.markdown(
            f"{'🟢' if exists else '🔴'} `{log.name}`",
            unsafe_allow_html=False,
        )
        if exists:
            try:
                rows = sum(1 for _ in open(log)) - 1
                st.caption(f"  {rows} records")
            except Exception:
                pass


# ── Main Area ──────────────────────────────────────────────────────────────────
st.title("🔐 Security Operations Pipeline")
st.caption("Real-time threat detection · ML classification · AI-assisted response")
st.divider()


def _check_prerequisites() -> tuple[bool, str]:
    if not Path(csv_path).exists():
        return False, f"Dataset not found at `{csv_path}`. Please check the file path."
    return True, ""


def _render_metric_row(result) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records Processed", f"{result.records_processed:,}")
    c2.metric(
        "Attacks Detected",
        f"{result.attacks_detected:,}",
        delta=f"{result.attacks_detected / max(result.records_processed, 1) * 100:.1f}%",
    )
    c3.metric("Threat Findings", len(result.findings))
    c4.metric("AI Decisions", len(result.decisions))
    c5.metric(
        "Actions Blocked",
        result.execution_counts.get("blocked", 0),
        delta_color="inverse",
    )


def _render_classification_tab(df: pd.DataFrame) -> None:
    st.subheader("Classification Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Attack Type Distribution**")
        counts = df["attack_type"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#1e2130")
        ax.set_facecolor("#1e2130")
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": "#e2e8f0", "fontsize": 9},
        )
        for at in autotexts:
            at.set_color("#0f1117")
            at.set_fontsize(8)
        ax.set_title("Attack Label Distribution", color="#e2e8f0", pad=10)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("**Confidence Score Distribution**")
        fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor="#1e2130")
        ax2.set_facecolor("#1e2130")
        ax2.tick_params(colors="#94a3b8")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#334155")
        attacks_df = df[df["is_attack"] == True]
        normal_df = df[df["is_attack"] == False]
        if not normal_df.empty:
            ax2.hist(
                normal_df["confidence"],
                bins=20,
                alpha=0.7,
                color="#22c55e",
                label="Benign",
            )
        if not attacks_df.empty:
            ax2.hist(
                attacks_df["confidence"],
                bins=20,
                alpha=0.7,
                color="#ef4444",
                label="Attack",
            )
        ax2.set_xlabel("Confidence Score", color="#94a3b8")
        ax2.set_ylabel("Count", color="#94a3b8")
        ax2.set_title("Model Confidence", color="#e2e8f0")
        ax2.legend(facecolor="#1e2130", labelcolor="#e2e8f0")
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("**Sample Records (attacks only)**")
    # Use only columns that exist in the dataset
    available_cols = ["attack_type", "confidence"]
    for col in ["Flow Packets/s", "Flow Bytes/s", "Flow Duration"]:
        if col in df.columns:
            available_cols.append(col)

    sample = df[df["is_attack"] == True][available_cols].head(20).reset_index(drop=True)
    if not sample.empty:
        st.dataframe(
            sample.style.background_gradient(subset=["confidence"], cmap="Reds"),
            use_container_width=True,
        )


def _render_findings_tab(findings) -> None:
    st.subheader("Threat Hunting Findings")
    if not findings:
        st.info("No threat hunting findings in this run.")
        return

    for f in findings:
        sev = f.severity
        colour = SEVERITY_COLOUR.get(sev, "#6b7280")
        with st.container():
            st.markdown(
                f"""
                <div class="finding-card {sev}">
                    <span style="color:{colour};font-weight:700">[{sev}]</span>
                    <span style="color:#94a3b8;font-size:12px"> {f.rule_id} · {f.rule_name}</span><br/>
                    <span style="color:#e2e8f0">{f.description}</span><br/>
                    <span style="color:#64748b;font-size:11px">
                    {f.affected_flows} flows · {f.timestamp[:19]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Evidence"):
                st.json(f.evidence)


def _render_decisions_tab(decisions: list[dict]) -> None:
    st.subheader("AI Agent Response Decisions")
    if not decisions:
        st.info("No AI decisions in this run (agent disabled or no findings).")
        return

    for d in decisions:
        action = d.get("action", "UNKNOWN")
        colour = ACTION_COLOUR.get(action, "#6b7280")
        priority = d.get("priority", "—")
        escalate = "⚠ Escalate to human" if d.get("escalate_to_human") else ""
        with st.container():
            st.markdown(
                f"""
                <div class="finding-card">
                    <span style="color:{colour};font-weight:700">{action}</span>
                    <span style="color:#94a3b8;font-size:12px"> · {d.get("rule_id", "—")} · Priority {priority} {escalate}</span><br/>
                    <span style="color:#94a3b8;font-size:13px">{d.get("reasoning", "—")}</span><br/>
                    <code style="color:#6ee7b7;font-size:11px">{d.get("signature_rule") or ""}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

    actions = [d.get("action", "UNKNOWN") for d in decisions]
    action_counts = pd.Series(actions).value_counts()
    fig, ax = plt.subplots(figsize=(6, 2.5), facecolor="#1e2130")
    ax.set_facecolor("#1e2130")
    bar_colours = [ACTION_COLOUR.get(a, "#6b7280") for a in action_counts.index]
    ax.barh(action_counts.index, action_counts.values, color=bar_colours)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.set_xlabel("Count", color="#94a3b8")
    ax.set_title("Decision Actions", color="#e2e8f0")
    st.pyplot(fig)
    plt.close(fig)


def _render_firewall_tab(result) -> None:
    st.subheader("Firewall & Log State")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Blocked", result.execution_counts.get("blocked", 0))
    c2.metric("Escalated", result.execution_counts.get("escalated", 0))
    c3.metric("Suppressed", result.execution_counts.get("suppressed", 0))
    c4.metric("Monitored", result.execution_counts.get("monitored", 0))

    for label, path in [
        ("🔥 Firewall Rules Log", FIREWALL_LOG),
        ("🚫 Blocked IPs Log", BLOCKED_IPS_LOG),
        ("📋 Pipeline Summary Log", PIPELINE_SUMMARY_LOG),
    ]:
        st.markdown(f"**{label}**")
        if path.exists():
            try:
                log_df = pd.read_csv(path)
                if not log_df.empty:
                    st.dataframe(log_df.tail(20), use_container_width=True)
                else:
                    st.caption("No records yet.")
            except Exception as e:
                st.caption(f"Error reading log: {e}")
        else:
            st.caption("Log file not yet created.")
        st.divider()


# ── Pipeline Execution ─────────────────────────────────────────────────────────
if run_btn:
    ok, msg = _check_prerequisites()
    if not ok:
        st.error(msg)
        st.stop()

    progress_bar = st.progress(0, text="Initialising…")
    status_box = st.empty()

    STAGE_STEPS = {
        "Loading dataset": 10,
        "Dataset loaded": 20,
        "Classifying flows": 40,
        "Classification complete": 60,
        "Threat hunting": 70,
        "Hunting complete": 80,
        "AI agent reasoning": 85,
        "Agent decisions ready": 92,
        "Executing decisions": 96,
        "Pipeline complete": 100,
    }

    def on_progress(stage, current, total):
        pct = STAGE_STEPS.get(stage, 50)
        progress_bar.progress(pct, text=stage)
        status_box.info(f"⏳ {stage}…")

    with st.spinner("Running pipeline…"):
        t0 = time.perf_counter()
        result = run_pipeline(
            csv_path=csv_path,
            batch_size=batch_size,
            on_progress=on_progress,
            use_agent=use_agent,
        )

    progress_bar.empty()
    status_box.empty()

    if result.error:
        st.error(f"Pipeline failed: {result.error}")
        st.stop()

    st.success(
        f"Pipeline complete in {result.duration_seconds:.2f}s — "
        f"{result.records_processed:,} records · {result.attacks_detected:,} attacks · "
        f"{len(result.findings)} findings · {len(result.decisions)} AI decisions"
    )

    _render_metric_row(result)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Classification",
            "🎯 Threat Findings",
            "🤖 AI Decisions",
            "🔥 Firewall & Logs",
        ]
    )
    with tab1:
        _render_classification_tab(result.classified_df)
    with tab2:
        _render_findings_tab(result.findings)
    with tab3:
        _render_decisions_tab(result.decisions)
    with tab4:
        _render_firewall_tab(result)

    st.session_state["last_result"] = result

else:
    # ── Idle state ─────────────────────────────────────────────────────────────
    ok, msg = _check_prerequisites()
    if not ok:
        st.warning(msg)
    else:
        st.info("Dataset ready. Click **▶ Run Pipeline** in the sidebar.")

    # Show historical logs if they exist
    if PIPELINE_SUMMARY_LOG.exists():
        try:
            hist = pd.read_csv(PIPELINE_SUMMARY_LOG)
            if not hist.empty:
                st.subheader("Previous Pipeline Runs")
                st.dataframe(hist, use_container_width=True)
        except Exception:
            pass
