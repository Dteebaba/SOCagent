"""
executor.py
Maintains a simulated firewall rule table and writes three structured log files.
Adapted for CIC-DDoS2019 - no real IPs, signature-based blocking.

Log files (written to logs/ directory):
  1. detection_log.csv      — Every flow classified by the RF model
  2. agent_reasoning_log.csv — Every agent decision with full reasoning
  3. response_log.csv        — Every response action executed
  4. pipeline_summary.csv    — Summary of each pipeline run
  5. firewall_rules.csv      — Active firewall rules
  6. blocked_ips.csv         — Blocked IP log (simulated for this dataset)
"""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DETECTION_LOG = LOGS_DIR / "detection_log.csv"
AGENT_LOG = LOGS_DIR / "agent_reasoning_log.csv"
RESPONSE_LOG = LOGS_DIR / "response_log.csv"
PIPELINE_SUMMARY_LOG = LOGS_DIR / "pipeline_summary.csv"
FIREWALL_LOG = LOGS_DIR / "firewall_rules.csv"
BLOCKED_IPS_LOG = LOGS_DIR / "blocked_ips.csv"

DETECTION_HEADERS = [
    "timestamp",
    "flow_index",
    "attack_type",
    "confidence",
    "is_attack",
    "hunting_flagged",
]
AGENT_HEADERS = [
    "timestamp",
    "rule_id",
    "attack_type",
    "action",
    "severity",
    "priority",
    "reasoning",
    "signature_rule",
    "escalate_to_human",
    "confidence_score",
]
RESPONSE_HEADERS = [
    "timestamp",
    "rule_id",
    "attack_type",
    "action",
    "signature_written",
    "flows_mitigated",
    "escalated",
]

VALID_ACTIONS = {"BLOCK_SIGNATURE", "ESCALATE", "SUPPRESS", "MONITOR"}


class FirewallExecutor:
    """
    Simulated response executor.
    Maintains an in-memory signature rule table and persists all
    actions to three CSV log files. No real IPs — blocking is done
    by flow feature signatures.
    """

    def __init__(self):
        # Simulated rule table — key is signature, value is rule details
        self._rule_table: dict = {}
        # Escalation queue — incidents waiting for human review
        self._escalation_queue: list = []
        # Suppressed alerts
        self._suppressed: list = []
        self._ensure_log_headers()

    def _ensure_log_headers(self) -> None:
        """Create log files with headers if they don't exist."""
        for path, headers in [
            (DETECTION_LOG, DETECTION_HEADERS),
            (AGENT_LOG, AGENT_HEADERS),
            (RESPONSE_LOG, RESPONSE_HEADERS),
        ]:
            if not path.exists():
                with open(path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=headers).writerow(
                        {h: h for h in headers}
                    )

        # Initialize pipeline summary log
        if not PIPELINE_SUMMARY_LOG.exists():
            with open(PIPELINE_SUMMARY_LOG, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    'run_id', 'timestamp', 'records', 'attacks', 
                    'findings', 'decisions', 'blocked', 'escalated', 
                    'suppressed', 'monitored', 'duration_sec'
                ])

        # Initialize firewall log
        if not FIREWALL_LOG.exists():
            with open(FIREWALL_LOG, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'rule_id', 'attack_type', 'signature', 'status'
                ])

        # Initialize blocked IPs log (simulated)
        if not BLOCKED_IPS_LOG.exists():
            with open(BLOCKED_IPS_LOG, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'attack_type', 'rule_id', 'flow_indices', 'count'
                ])

    def log_detections(self, df: pd.DataFrame, hunting_flags: list = None) -> None:
        """
        Write classification results to detection_log.csv.

        Parameters
        ----------
        df : pd.DataFrame
            Classified flow DataFrame with attack_type, confidence, is_attack
        hunting_flags : list of int indices that were hunting-flagged
        """
        flagged_indices = set(hunting_flags or [])
        rows = []
        for idx, row in df.iterrows():
            rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "flow_index": idx,
                    "attack_type": row.get("attack_type", "Unknown"),
                    "confidence": round(float(row.get("confidence", 0)), 4),
                    "is_attack": row.get("is_attack", False),
                    "hunting_flagged": idx in flagged_indices,
                }
            )

        with open(DETECTION_LOG, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DETECTION_HEADERS)
            writer.writerows(rows)

        logger.info("Logged %d detection records", len(rows))

    def log_agent_decisions(self, decisions: list) -> None:
        """
        Write agent reasoning and decisions to agent_reasoning_log.csv.

        Parameters
        ----------
        decisions : list of dict from SecurityAgent.analyse()
        """
        rows = []
        for d in decisions:
            rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "rule_id": d.get("rule_id", ""),
                    "attack_type": d.get("attack_type", ""),
                    "action": d.get("action", ""),
                    "severity": d.get("severity", ""),
                    "priority": d.get("priority", 3),
                    "reasoning": d.get("reasoning", "")[:500],
                    "signature_rule": d.get("signature_rule", ""),
                    "escalate_to_human": d.get("escalate_to_human", False),
                    "confidence_score": round(float(d.get("confidence_score", 0)), 4),
                }
            )

        with open(AGENT_LOG, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=AGENT_HEADERS)
            writer.writerows(rows)

        logger.info("Logged %d agent decisions", len(rows))

    def execute_decisions(self, decisions: list, classified_df: pd.DataFrame = None) -> dict:
        """
        Execute response decisions from the AI agent.
        Updates rule table, escalation queue, and response log.

        Parameters
        ----------
        decisions : list of dict from SecurityAgent.analyse()
        classified_df : pd.DataFrame of classified flows (optional)

        Returns
        -------
        dict with counts of each action taken
        """
        counts = {"blocked": 0, "escalated": 0, "suppressed": 0, "monitored": 0}

        response_rows = []

        for decision in decisions:
            action = str(decision.get("action", "MONITOR")).upper()
            if action not in VALID_ACTIONS:
                action = "MONITOR"

            rule_id = decision.get("rule_id", "unknown")
            attack_type = decision.get("attack_type", "Unknown")
            signature = decision.get("signature_rule", None)
            escalate = bool(decision.get("escalate_to_human", False))
            flows_mitigated = 0

            if action == "BLOCK_SIGNATURE":
                # Write signature to rule table
                sig_key = (
                    f"{attack_type}_{rule_id}_{datetime.utcnow().strftime('%H%M%S')}"
                )
                self._rule_table[sig_key] = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "attack_type": attack_type,
                    "rule_id": rule_id,
                    "signature": signature or f"Block all {attack_type} flows",
                    "status": "ACTIVE",
                }

                # Write to firewall log
                with open(FIREWALL_LOG, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.utcnow().isoformat(),
                        rule_id,
                        attack_type,
                        signature or f"Block all {attack_type} flows",
                        "ACTIVE"
                    ])

                # Count how many flows this rule would mitigate
                if classified_df is not None and "attack_type" in classified_df.columns:
                    flows_mitigated = len(
                        classified_df[classified_df["attack_type"] == attack_type]
                    )
                counts["blocked"] += 1
                logger.info(
                    "BLOCK_SIGNATURE written for %s — %d flows mitigated",
                    attack_type,
                    flows_mitigated,
                )

            elif action == "ESCALATE":
                self._escalation_queue.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "rule_id": rule_id,
                        "attack_type": attack_type,
                        "reasoning": decision.get("reasoning", ""),
                        "severity": decision.get("severity", "MEDIUM"),
                        "priority": decision.get("priority", 3),
                    }
                )
                counts["escalated"] += 1
                logger.info("ESCALATED incident for %s", attack_type)

            elif action == "SUPPRESS":
                self._suppressed.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "rule_id": rule_id,
                        "attack_type": attack_type,
                        "reasoning": decision.get("reasoning", ""),
                    }
                )
                counts["suppressed"] += 1
                logger.info("SUPPRESSED false positive for %s", attack_type)

            elif action == "MONITOR":
                counts["monitored"] += 1
                logger.info("MONITOR set for %s", attack_type)

            response_rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "rule_id": rule_id,
                    "attack_type": attack_type,
                    "action": action,
                    "signature_written": signature or "",
                    "flows_mitigated": flows_mitigated,
                    "escalated": escalate,
                }
            )

        # Write response log
        with open(RESPONSE_LOG, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESPONSE_HEADERS)
            writer.writerows(response_rows)

        logger.info("Executed %d decisions: %s", len(decisions), counts)
        return counts

    def write_pipeline_summary(
        self, 
        run_id: str, 
        records_processed: int,
        attacks_detected: int,
        findings_generated: int,
        decisions_made: int,
        execution_counts: dict,
        duration_seconds: float
    ) -> None:
        """
        Write pipeline run summary to pipeline_summary.csv

        Parameters
        ----------
        run_id : str
            Unique identifier for this pipeline run
        records_processed : int
            Total number of flows processed
        attacks_detected : int
            Number of attack flows detected
        findings_generated : int
            Number of threat hunting findings
        decisions_made : int
            Number of AI agent decisions
        execution_counts : dict
            Counts of each action type (blocked, escalated, etc.)
        duration_seconds : float
            Total pipeline execution time
        """
        with open(PIPELINE_SUMMARY_LOG, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                run_id,
                datetime.utcnow().isoformat(),
                records_processed,
                attacks_detected,
                findings_generated,
                decisions_made,
                execution_counts.get('blocked', 0),
                execution_counts.get('escalated', 0),
                execution_counts.get('suppressed', 0),
                execution_counts.get('monitored', 0),
                round(duration_seconds, 2)
            ])

        logger.info("Pipeline summary written: run_id=%s", run_id)

    def get_rule_table(self) -> pd.DataFrame:
        """Return current simulated firewall rule table as DataFrame."""
        if not self._rule_table:
            return pd.DataFrame(
                columns=["timestamp", "attack_type", "rule_id", "signature", "status"]
            )
        return pd.DataFrame(self._rule_table.values())

    def get_escalation_queue(self) -> pd.DataFrame:
        """Return current escalation queue as DataFrame."""
        if not self._escalation_queue:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "rule_id",
                    "attack_type",
                    "reasoning",
                    "severity",
                    "priority",
                ]
            )
        return pd.DataFrame(self._escalation_queue)

    def get_suppressed(self) -> pd.DataFrame:
        """Return suppressed alerts as DataFrame."""
        if not self._suppressed:
            return pd.DataFrame(
                columns=["timestamp", "rule_id", "attack_type", "reasoning"]
            )
        return pd.DataFrame(self._suppressed)

    def get_summary(self) -> dict:
        """Return summary statistics of all actions taken."""
        return {
            "active_rules": len(self._rule_table),
            "escalated_incidents": len(self._escalation_queue),
            "suppressed_alerts": len(self._suppressed),
            "detection_log": str(DETECTION_LOG),
            "agent_log": str(AGENT_LOG),
            "response_log": str(RESPONSE_LOG),
        }