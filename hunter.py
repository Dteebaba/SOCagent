"""
hunter.py
Five threat hunting rules built around CIC-DDoS2019 features.
No IPs in this dataset - hunting is done in feature space.

Rule 1 - Temporal Cluster    : 5+ same attack type flows in rolling window
Rule 2 - Low Confidence Flag : Classifier confidence between 0.5 and 0.75
Rule 3 - Benign Anomaly      : Benign flow with suspiciously high packet rate
Rule 4 - High Volume Burst   : Single attack type dominates >20% of all flows
Rule 5 - Multi-Vector Attack : 3+ distinct attack types in same batch
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Rule thresholds
CLUSTER_WINDOW_SIZE = 5  # Number of flows to check in rolling window
CLUSTER_THRESHOLD = 5  # Min same-type flows to trigger
LOW_CONF_MIN = 0.50  # Low confidence lower bound
LOW_CONF_MAX = 0.75  # Low confidence upper bound
BENIGN_PACKET_RATE_MULTIPLIER = 3.0  # How many times above mean to flag
HIGH_VOLUME_THRESHOLD = 0.20  # Single attack type share to trigger R004
MULTI_VECTOR_MIN_TYPES = 3  # Distinct attack types required for R005


@dataclass
class HuntingFinding:
    rule_id: str
    rule_name: str
    severity: str
    description: str
    evidence: dict
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    affected_flows: int = 0
    flow_indices: list = field(default_factory=list)


class ThreatHunter:
    """
    Executes three threat hunting rules against classified flow records.
    Operates in feature space since CIC-DDoS2019 has no real IPs.
    """

    def __init__(self):
        self.benign_packet_rate_mean = None
        self.benign_packet_rate_std = None

    def fit_baseline(self, df: pd.DataFrame):
        """
        Calculate baseline statistics from Benign flows.
        Call this once before hunting so Rule 3 has a reference point.
        """
        benign = df[df["attack_type"] == "Benign"]
        if not benign.empty and "Flow Packets/s" in benign.columns:
            self.benign_packet_rate_mean = benign["Flow Packets/s"].mean()
            self.benign_packet_rate_std = benign["Flow Packets/s"].std()
            logger.info(
                "Baseline fitted: mean Flow Packets/s = %.2f, std = %.2f",
                self.benign_packet_rate_mean,
                self.benign_packet_rate_std,
            )

    def hunt(self, df: pd.DataFrame) -> list:
        """
        Run all three hunting rules against classified flow DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain attack_type, confidence, is_attack columns
            plus the original CIC-DDoS2019 feature columns.

        Returns
        -------
        list of HuntingFinding
        """
        findings = []
        findings.extend(self._rule1_temporal_cluster(df))
        findings.extend(self._rule2_low_confidence(df))
        findings.extend(self._rule3_benign_anomaly(df))
        findings.extend(self._rule4_high_volume_burst(df))
        findings.extend(self._rule5_multi_vector_attack(df))

        logger.info(
            "Hunting complete: %d findings from %d flows", len(findings), len(df)
        )
        return findings

    def _rule1_temporal_cluster(self, df: pd.DataFrame) -> list:
        """
        Rule 1 - Temporal Cluster
        Detects when 5 or more flows of the same attack type appear
        consecutively in the data stream — indicating a coordinated attack
        rather than isolated events.
        """
        findings = []
        attack_flows = df[df["is_attack"] == True].copy()

        if attack_flows.empty:
            return findings

        # Group consecutive same-type attacks using rolling window
        attack_type_series = attack_flows["attack_type"].reset_index(drop=True)

        i = 0
        while i < len(attack_type_series):
            current_type = attack_type_series[i]
            # Count consecutive same type
            j = i
            while j < len(attack_type_series) and attack_type_series[j] == current_type:
                j += 1
            cluster_size = j - i

            if cluster_size >= CLUSTER_THRESHOLD:
                cluster_rows = attack_flows.iloc[i:j]
                avg_confidence = cluster_rows["confidence"].mean()
                avg_packet_rate = (
                    cluster_rows["Flow Packets/s"].mean()
                    if "Flow Packets/s" in cluster_rows.columns
                    else None
                )

                findings.append(
                    HuntingFinding(
                        rule_id="R001",
                        rule_name="Temporal Cluster Detection",
                        severity="HIGH" if cluster_size >= 10 else "MEDIUM",
                        description=(
                            f"Coordinated {current_type} attack detected: "
                            f"{cluster_size} consecutive flows, "
                            f"avg confidence {avg_confidence:.2f}"
                        ),
                        evidence={
                            "attack_type": current_type,
                            "cluster_size": cluster_size,
                            "avg_confidence": round(avg_confidence, 4),
                            "avg_flow_packets_per_sec": round(avg_packet_rate, 2)
                            if avg_packet_rate
                            else "N/A",
                            "threshold": CLUSTER_THRESHOLD,
                        },
                        affected_flows=cluster_size,
                        flow_indices=list(attack_flows.iloc[i:j].index),
                    )
                )
                logger.warning(
                    "Rule 1 triggered: %d %s flows clustered",
                    cluster_size,
                    current_type,
                )
            i = j if j > i else i + 1

        return findings

    def _rule2_low_confidence(self, df: pd.DataFrame) -> list:
        """
        Rule 2 - Low Confidence Flag
        Flags flows where the classifier confidence falls between 0.50
        and 0.75. These are ambiguous classifications that warrant
        deeper investigation by the agent.
        """
        findings = []

        if "confidence" not in df.columns:
            return findings

        low_conf = df[
            (df["confidence"] >= LOW_CONF_MIN)
            & (df["confidence"] <= LOW_CONF_MAX)
            & (df["is_attack"] == True)
        ]

        if low_conf.empty:
            return findings

        # Group by attack type for reporting
        for attack_type, group in low_conf.groupby("attack_type"):
            avg_conf = group["confidence"].mean()
            findings.append(
                HuntingFinding(
                    rule_id="R002",
                    rule_name="Low Confidence Classifier Flag",
                    severity="MEDIUM",
                    description=(
                        f"Uncertain classification: {len(group)} {attack_type} flows "
                        f"with avg confidence {avg_conf:.2f} — requires agent review"
                    ),
                    evidence={
                        "attack_type": attack_type,
                        "flow_count": len(group),
                        "avg_confidence": round(avg_conf, 4),
                        "min_confidence": round(group["confidence"].min(), 4),
                        "max_confidence": round(group["confidence"].max(), 4),
                        "confidence_range": [LOW_CONF_MIN, LOW_CONF_MAX],
                    },
                    affected_flows=len(group),
                    flow_indices=list(group.index),
                )
            )
            logger.warning(
                "Rule 2 triggered: %d uncertain %s flows", len(group), attack_type
            )

        return findings

    def _rule3_benign_anomaly(self, df: pd.DataFrame) -> list:
        """
        Rule 3 - Benign Anomaly Detection
        Flags flows classified as Benign but with Flow Packets/s
        significantly above the baseline mean. Catches stealthy
        attacks that evade the classifier.
        """
        findings = []

        if "Flow Packets/s" not in df.columns:
            return findings

        benign_flows = df[df["attack_type"] == "Benign"]
        if benign_flows.empty:
            return findings

        # Use fitted baseline or calculate from current batch
        if self.benign_packet_rate_mean is None:
            mean_rate = benign_flows["Flow Packets/s"].mean()
            std_rate = benign_flows["Flow Packets/s"].std()
        else:
            mean_rate = self.benign_packet_rate_mean
            std_rate = self.benign_packet_rate_std

        threshold = mean_rate + (BENIGN_PACKET_RATE_MULTIPLIER * std_rate)

        anomalous = benign_flows[benign_flows["Flow Packets/s"] > threshold]

        if anomalous.empty:
            return findings

        findings.append(
            HuntingFinding(
                rule_id="R003",
                rule_name="Benign Anomaly Detection",
                severity="MEDIUM",
                description=(
                    f"Suspicious benign flows detected: {len(anomalous)} flows "
                    f"classified as Benign but with Flow Packets/s above "
                    f"{threshold:.2f} (threshold = mean + {BENIGN_PACKET_RATE_MULTIPLIER}x std)"
                ),
                evidence={
                    "anomalous_flow_count": len(anomalous),
                    "baseline_mean_packets_per_sec": round(mean_rate, 4),
                    "baseline_std": round(std_rate, 4),
                    "threshold": round(threshold, 4),
                    "max_observed": round(anomalous["Flow Packets/s"].max(), 4),
                    "avg_observed": round(anomalous["Flow Packets/s"].mean(), 4),
                },
                affected_flows=len(anomalous),
                flow_indices=list(anomalous.index),
            )
        )
        logger.warning(
            "Rule 3 triggered: %d anomalous benign flows above threshold %.2f",
            len(anomalous),
            threshold,
        )

        return findings

    def _rule4_high_volume_burst(self, df: pd.DataFrame) -> list:
        """
        Rule 4 - High Volume Burst
        Flags any single attack type that accounts for more than 20% of all
        flows in the batch. Indicates a dominant, large-scale attack campaign
        rather than scattered probes.
        """
        findings = []

        if df.empty or "attack_type" not in df.columns:
            return findings

        attack_flows = df[df["is_attack"] == True]
        if attack_flows.empty:
            return findings

        total_flows = len(df)
        type_counts = attack_flows["attack_type"].value_counts()

        for attack_type, count in type_counts.items():
            share = count / total_flows
            if share >= HIGH_VOLUME_THRESHOLD:
                avg_confidence = attack_flows[
                    attack_flows["attack_type"] == attack_type
                ]["confidence"].mean()

                findings.append(
                    HuntingFinding(
                        rule_id="R004",
                        rule_name="High Volume Attack Burst",
                        severity="HIGH" if share >= 0.40 else "MEDIUM",
                        description=(
                            f"Dominant attack campaign: {attack_type} comprises "
                            f"{share*100:.1f}% of all observed flows "
                            f"({count}/{total_flows})"
                        ),
                        evidence={
                            "attack_type": attack_type,
                            "flow_count": int(count),
                            "total_flows": total_flows,
                            "share_percent": round(share * 100, 2),
                            "avg_confidence": round(avg_confidence, 4),
                            "threshold_percent": HIGH_VOLUME_THRESHOLD * 100,
                        },
                        affected_flows=int(count),
                        flow_indices=list(
                            attack_flows[attack_flows["attack_type"] == attack_type].index
                        ),
                    )
                )
                logger.warning(
                    "Rule 4 triggered: %s dominates at %.1f%% of flows",
                    attack_type,
                    share * 100,
                )

        return findings

    def _rule5_multi_vector_attack(self, df: pd.DataFrame) -> list:
        """
        Rule 5 - Multi-Vector Attack Detection
        Fires when 3 or more distinct attack types appear in the same batch.
        Multi-vector DDoS campaigns are significantly harder to mitigate and
        indicate sophisticated, coordinated threat actors.
        """
        findings = []

        if df.empty or "attack_type" not in df.columns:
            return findings

        attack_flows = df[df["is_attack"] == True]
        if attack_flows.empty:
            return findings

        distinct_types = attack_flows["attack_type"].unique().tolist()
        n_types = len(distinct_types)

        if n_types < MULTI_VECTOR_MIN_TYPES:
            return findings

        type_breakdown = {
            t: int((attack_flows["attack_type"] == t).sum())
            for t in distinct_types
        }
        dominant = max(type_breakdown, key=lambda t: type_breakdown[t])

        findings.append(
            HuntingFinding(
                rule_id="R005",
                rule_name="Multi-Vector DDoS Detection",
                severity="CRITICAL" if n_types >= 5 else "HIGH",
                description=(
                    f"Multi-vector attack campaign: {n_types} distinct attack types "
                    f"observed simultaneously — coordinated threat actor likely"
                ),
                evidence={
                    "distinct_attack_types": n_types,
                    "attack_types": distinct_types,
                    "type_breakdown": type_breakdown,
                    "dominant_type": dominant,
                    "total_attack_flows": int(len(attack_flows)),
                    "threshold_types": MULTI_VECTOR_MIN_TYPES,
                },
                affected_flows=int(len(attack_flows)),
                flow_indices=list(attack_flows.index),
            )
        )
        logger.warning(
            "Rule 5 triggered: %d distinct attack types detected (multi-vector)",
            n_types,
        )

        return findings
