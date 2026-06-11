"""
pipeline.py
Orchestrates the full security operations pipeline:
  classifier → hunter → agent → executor

Can be run standalone on a CSV dataset or called programmatically
(e.g., from app.py) with an optional row-level callback for streaming.
"""

import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from agent import SecurityAgent
from classifier import NetworkFlowClassifier
from executor import FirewallExecutor
from hunter import HuntingFinding, ThreatHunter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

DEFAULT_CSV = Path(__file__).parent / "sample_data_5k.csv"


class PipelineResult:
    """Holds all artifacts produced by a single pipeline run."""

    def __init__(self):
        self.run_id: str = str(uuid.uuid4())[:8]
        self.classified_df: pd.DataFrame = pd.DataFrame()
        self.findings: list[HuntingFinding] = []
        self.decisions: list[dict] = []
        self.execution_counts: dict = {}
        self.duration_seconds: float = 0.0
        self.error: str | None = None

    @property
    def has_ground_truth(self) -> bool:
        return (
            not self.classified_df.empty
            and "true_label" in self.classified_df.columns
        )

    @property
    def attacks_detected(self) -> int:
        """Model-predicted attack count."""
        if self.classified_df.empty or "is_attack" not in self.classified_df.columns:
            return 0
        return int(self.classified_df["is_attack"].sum())

    @property
    def benign_count(self) -> int:
        """Ground-truth benign count when available; falls back to model prediction."""
        if self.has_ground_truth:
            return int((self.classified_df["true_label"] == "Benign").sum())
        if self.classified_df.empty or "is_attack" not in self.classified_df.columns:
            return 0
        return int((~self.classified_df["is_attack"]).sum())

    @property
    def true_attacks_count(self) -> int:
        """Ground-truth attack count (None when no label column present)."""
        if not self.has_ground_truth:
            return 0
        return int((self.classified_df["true_label"] != "Benign").sum())

    @property
    def records_processed(self) -> int:
        return len(self.classified_df)


ProgressCallback = Callable[[str, int, int], None]


def run_pipeline(
    csv_path: Path | str = DEFAULT_CSV,
    model_path: Path | str | None = None,
    batch_size: int = 200,
    on_progress: ProgressCallback | None = None,
    use_agent: bool = True,
    benign_threshold: float = 0.0,
) -> PipelineResult:
    """
    Execute the full security operations pipeline on a CSV dataset.

    Parameters
    ----------
    csv_path    : Path to the network flow CSV file.
    model_path  : Optional path to the Random Forest pickle (deprecated - model auto-downloads).
    batch_size  : Records per classifier batch.
    on_progress : Optional callback(stage, current, total) for UI streaming.
    use_agent   : Whether to call the AI agent (can be disabled for offline testing).

    Returns
    -------
    PipelineResult
    """
    result = PipelineResult()
    start_time = time.perf_counter()

    def _progress(stage: str, current: int = 0, total: int = 0) -> None:
        logger.info("[%s] %s (%d/%d)", result.run_id, stage, current, total)
        if on_progress:
            on_progress(stage, current, total)

    try:
        # ── Stage 1: Load dataset ──────────────────────────────────────────────
        _progress("Loading dataset", 0, 1)
        df = pd.read_csv(csv_path)
        # Preserve ground truth label column if present (CIC-DDoS2019 datasets
        # include a 'label' column with the true class — rename so it survives
        # classification without colliding with predicted 'attack_type').
        if "label" in df.columns:
            df = df.rename(columns={"label": "true_label"})
            logger.info("Ground truth 'label' column preserved as 'true_label'")
        _progress("Dataset loaded", len(df), len(df))
        logger.info("Loaded %d records from %s", len(df), csv_path)

        # ── Stage 2: Classification ────────────────────────────────────────────
        _progress("Classifying flows", 0, len(df))
        # Classifier auto-downloads from Google Drive
        clf = NetworkFlowClassifier()
        result.classified_df = clf.classify_batch(df, batch_size=batch_size, benign_threshold=benign_threshold)
        _progress("Classification complete", len(df), len(df))
        logger.info(
            "Classification done — %d attacks in %d records",
            result.attacks_detected,
            result.records_processed,
        )

        # ── Stage 3: Threat hunting ────────────────────────────────────────────
        _progress("Threat hunting", 0, 1)
        hunter = ThreatHunter()
        # Fit baseline from current data if possible
        if not result.classified_df.empty:
            hunter.fit_baseline(result.classified_df)
        result.findings = hunter.hunt(result.classified_df)
        _progress("Hunting complete", len(result.findings), len(result.findings))
        logger.info("Hunting produced %d findings", len(result.findings))

        # ── Stage 4: AI reasoning ──────────────────────────────────────────────
        if use_agent and result.findings:
            _progress("AI agent reasoning", 0, len(result.findings))
            agent = SecurityAgent()
            result.decisions = agent.analyse_batch(result.findings)
            _progress("Agent decisions ready", len(result.decisions), len(result.findings))
        else:
            if not use_agent:
                logger.info("Agent disabled — skipping AI reasoning")
            else:
                logger.info("No findings — skipping agent call")
            result.decisions = []

        # ── Stage 5: Execution ─────────────────────────────────────────────────
        _progress("Executing decisions", 0, len(result.decisions))
        executor = FirewallExecutor()

        # Log detections
        if not result.classified_df.empty:
            hunting_flagged_indices = []
            for finding in result.findings:
                hunting_flagged_indices.extend(finding.flow_indices)
            executor.log_detections(result.classified_df, hunting_flagged_indices)

        # Log agent decisions
        if result.decisions:
            executor.log_agent_decisions(result.decisions)

        # Execute decisions
        result.execution_counts = executor.execute_decisions(
            result.decisions, 
            result.classified_df
        )

        result.duration_seconds = time.perf_counter() - start_time

        # Write pipeline summary
        executor.write_pipeline_summary(
            run_id=result.run_id,
            records_processed=result.records_processed,
            attacks_detected=result.attacks_detected,
            findings_generated=len(result.findings),
            decisions_made=len(result.decisions),
            execution_counts=result.execution_counts,
            duration_seconds=result.duration_seconds,
        )
        _progress("Pipeline complete", 1, 1)

        logger.info(
            "Pipeline run %s finished in %.2fs | %d records | %d attacks | "
            "%d findings | %d decisions | blocked=%d escalated=%d suppressed=%d monitored=%d",
            result.run_id,
            result.duration_seconds,
            result.records_processed,
            result.attacks_detected,
            len(result.findings),
            len(result.decisions),
            result.execution_counts.get("blocked", 0),
            result.execution_counts.get("escalated", 0),
            result.execution_counts.get("suppressed", 0),
            result.execution_counts.get("monitored", 0),
        )

    except Exception as exc:
        result.error = str(exc)
        result.duration_seconds = time.perf_counter() - start_time
        logger.exception("Pipeline run %s failed: %s", result.run_id, exc)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SecOps Pipeline CLI")
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Path to network flow CSV (default: sample_data_5k.csv)",
    )
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Skip the OpenAI agent step",
    )
    args = parser.parse_args()

    result = run_pipeline(
        csv_path=args.csv,
        use_agent=not args.no_agent,
    )

    if result.error:
        print(f"\nPipeline FAILED: {result.error}")
    else:
        print(f"\n{'='*60}")
        print(f"Run ID          : {result.run_id}")
        print(f"Records         : {result.records_processed}")
        print(f"Attacks found   : {result.attacks_detected}")
        print(f"Findings        : {len(result.findings)}")
        print(f"Decisions       : {len(result.decisions)}")
        print(f"Blocked         : {result.execution_counts.get('blocked', 0)}")
        print(f"Escalated       : {result.execution_counts.get('escalated', 0)}")
        print(f"Suppressed      : {result.execution_counts.get('suppressed', 0)}")
        print(f"Monitored       : {result.execution_counts.get('monitored', 0)}")
        print(f"Duration        : {result.duration_seconds:.2f}s")
        print(f"{'='*60}")
        print(f"Logs written to : {Path(__file__).parent / 'logs'}/")
