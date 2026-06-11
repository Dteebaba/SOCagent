"""
app.py - WITH AI CHAT FEATURE
Streamlit dashboard for the Security Operations Pipeline.
Now includes interactive AI chat for explaining results!
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
from chat_assistant import ResultsChatAssistant

# File paths
DATA_PATH = Path(__file__).parent / "sample_data_5k.csv"
LOGS_DIR = Path(__file__).parent / "logs"

# Color schemes
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

# Custom CSS
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
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize chat assistant in session state
if 'chat_assistant' not in st.session_state:
    try:
        st.session_state.chat_assistant = ResultsChatAssistant()
    except ValueError:
        st.session_state.chat_assistant = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
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

# Main Area
st.title("🔐 Security Operations Pipeline")
st.caption("Real-time threat detection · ML classification · AI-assisted response")
st.divider()


def _check_prerequisites() -> tuple[bool, str]:
    if not Path(csv_path).exists():
        return False, f"Dataset not found at `{csv_path}`. Please check the file path."
    return True, ""


def _render_metric_row(result) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Rows Analyzed", f"{result.records_processed:,}")
    c2.metric(
        "Threats Discovered",
        f"{result.attacks_detected:,}",
        delta=f"{result.attacks_detected / max(result.records_processed, 1) * 100:.1f}% of dataset",
    )
    c3.metric("Benign Found", f"{result.benign_count:,}")

    with st.expander("ℹ️ What counts as a Threat?"):
        st.markdown(
            "**Threats Discovered** = network flows the Random Forest model classified as "
            "any non-Benign label from the CIC-DDoS2019 dataset. This includes:\n\n"
            "- **DrDoS amplification variants**: DrDoS_DNS, DrDoS_LDAP, DrDoS_MSSQL, "
            "DrDoS_NTP, DrDoS_NetBIOS, DrDoS_SNMP, DrDoS_UDP\n"
            "- **Direct flood attacks**: LDAP, MSSQL, NetBIOS, Portmap, Syn, TFTP, "
            "UDP, UDP-lag, UDPLag, WebDDoS\n\n"
            "**Benign Found** = flows the model identified as normal, non-malicious traffic.\n\n"
            "*Note: Because Benign samples are part of the training distribution, the model "
            "classifies them correctly — Benign flows are not counted as threats.*"
        )

    st.markdown("")
    c4, c5, c6 = st.columns(3)
    c4.metric("Threat Findings (Rule-Based)", len(result.findings))
    c5.metric("AI Decisions", len(result.decisions))
    c6.metric(
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


def _render_chat_tab(result) -> None:
    """Render the AI chat interface for asking questions about results."""
    st.subheader("💬 Ask AI About Results")

    # Check if chat assistant is available
    if st.session_state.chat_assistant is None:
        st.error("⚠️ Chat feature requires OPENAI_API_KEY to be set in Streamlit Secrets.")
        st.info("Go to Streamlit Cloud → App Settings → Secrets → Add: OPENAI_API_KEY")
        return

    if result is None:
        st.info("👆 Run the pipeline first, then come back here to ask questions about the results!")
        return

    st.markdown("**Select what you want to analyze:**")

    # Result type selector
    result_type = st.radio(
        "Choose result type:",
        ["Classified Flow", "Threat Finding", "AI Decision", "General Question"],
        horizontal=True
    )

    # Context selector based on type
    selected_context = None
    context_data = None

    if result_type == "Classified Flow" and result.classified_df is not None:
        flow_options = [
            f"Flow #{i}: {row['attack_type']} ({row['confidence']:.2%})"
            for i, row in result.classified_df.head(20).iterrows()
        ]
        if flow_options:
            selected = st.selectbox("Select a flow:", flow_options)
            flow_idx = int(selected.split("#")[1].split(":")[0])
            # Convert to dict and handle numpy types
            row_data = result.classified_df.loc[flow_idx]

            # Convert all values to Python native types
            context_data = {}
            for k, v in row_data.items():
                if hasattr(v, 'item'):  # numpy scalar
                    context_data[k] = v.item()
                elif isinstance(v, (list, dict)):
                    context_data[k] = v
                else:
                    context_data[k] = str(v) if v is not None else None

            selected_context = "flow"

    elif result_type == "Threat Finding" and result.findings:
        finding_options = [
            f"{f.rule_id}: {f.description[:50]}... ({f.severity})"
            for f in result.findings
        ]
        if finding_options:
            selected = st.selectbox("Select a finding:", finding_options)
            finding_idx = finding_options.index(selected)
            f = result.findings[finding_idx]
            context_data = {
                'rule_id': f.rule_id,
                'rule_name': f.rule_name,
                'attack_type': getattr(f, 'attack_type', 'Unknown'),
                'severity': f.severity,
                'description': f.description,
                'affected_flows': f.affected_flows,
                'evidence': f.evidence
            }
            selected_context = "finding"

    elif result_type == "AI Decision" and result.decisions:
        decision_options = [
            f"{d.get('action', 'UNKNOWN')}: {d.get('attack_type', 'Unknown')} ({d.get('severity', 'MEDIUM')})"
            for d in result.decisions
        ]
        if decision_options:
            selected = st.selectbox("Select a decision:", decision_options)
            decision_idx = decision_options.index(selected)
            context_data = result.decisions[decision_idx]
            selected_context = "decision"

    elif result_type == "General Question":
        context_data = {
            'records_processed': result.records_processed,
            'attacks_detected': result.attacks_detected,
            'detection_rate': result.attacks_detected / max(result.records_processed, 1),
            'findings_count': len(result.findings),
            'decisions_count': len(result.decisions)
        }
        selected_context = "general"

    # Show suggested questions
    if selected_context:
        st.markdown("**💡 Suggested questions:**")
        suggestions = st.session_state.chat_assistant.get_suggested_questions(selected_context)

        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions[:4]):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.pending_question = suggestion

    # Chat interface
    st.divider()

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Question input
    question = st.text_input(
        "Ask your question:",
        value=st.session_state.get('pending_question', ''),
        key="question_input",
        placeholder="Type your question here..."
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_btn = st.button("Ask AI", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_assistant.reset_conversation()
            if 'pending_question' in st.session_state:
                del st.session_state['pending_question']
            st.rerun()

    # Process question
    if ask_btn and question and selected_context and context_data:
        with st.spinner("🤔 AI is thinking..."):
            try:
                # Get AI response based on context type
                if selected_context == "flow":
                    answer = st.session_state.chat_assistant.analyze_flow(context_data, question)
                elif selected_context == "finding":
                    answer = st.session_state.chat_assistant.analyze_finding(context_data, question)
                elif selected_context == "decision":
                    answer = st.session_state.chat_assistant.analyze_decision(context_data, question)
                else:  # general
                    answer = st.session_state.chat_assistant.general_question(context_data, question)

                # Add to chat history
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

                # Clear pending question
                if 'pending_question' in st.session_state:
                    del st.session_state['pending_question']

                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")


# Pipeline Execution
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
        f"{result.records_processed:,} rows analyzed · {result.attacks_detected:,} threats · "
        f"{result.benign_count:,} benign · "
        f"{len(result.findings)} findings · {len(result.decisions)} AI decisions"
    )

    _render_metric_row(result)
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Classification",
            "🎯 Threat Findings",
            "🤖 AI Decisions",
            "🔥 Firewall & Logs",
            "💬 Ask AI"
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
    with tab5:
        _render_chat_tab(result)

    st.session_state["last_result"] = result

else:
    # Idle state
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

    # Show chat tab with last result if available
    if 'last_result' in st.session_state:
        st.divider()
        tab_chat = st.tabs(["💬 Ask AI About Last Run"])
        with tab_chat[0]:
            _render_chat_tab(st.session_state['last_result'])