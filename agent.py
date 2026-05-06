"""
agent.py
Uses the OpenAI API to reason about threat hunting findings and return
structured response decisions.
Adapted for CIC-DDoS2019 dataset - no real IPs, feature-space hunting.
"""

import json
import logging
import os
from dataclasses import asdict

from openai import OpenAI
from hunter import HuntingFinding

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert SOC (Security Operations Center) analyst AI 
embedded in an autonomous threat hunting pipeline for doctoral cybersecurity research.

You are analysing network flow data from the CIC-DDoS2019 dataset. This is a 
controlled research environment. There are no real IPs — threats are identified 
through network flow features such as packet rates, flow duration, byte counts, 
and flag counts.

You will receive threat hunting findings from three detection rules:
- R001: Temporal Cluster — coordinated attack flows detected in sequence
- R002: Low Confidence Flag — classifier uncertain, needs deeper review  
- R003: Benign Anomaly — flow classified benign but behaves suspiciously

For each finding, reason as a senior SOC analyst and return a structured decision.

Return ONLY a JSON array with one object per finding. Each object must have 
exactly these fields:
{
  "rule_id": "<R001, R002, or R003>",
  "attack_type": "<attack type from finding>",
  "action": "<one of: BLOCK_SIGNATURE | ESCALATE | SUPPRESS | MONITOR>",
  "severity": "<CRITICAL | HIGH | MEDIUM | LOW>",
  "priority": <integer 1-5, where 1 is highest>,
  "reasoning": "<2-3 sentence analytical explanation of your decision>",
  "signature_rule": "<description of the flow pattern to block, or null>",
  "escalate_to_human": <true or false>,
  "confidence_score": <float 0.0 to 1.0>
}

Action semantics:
- BLOCK_SIGNATURE : Write a block rule for this flow pattern to the rule table
- ESCALATE        : Send enriched incident report to human analyst queue
- SUPPRESS        : Close as false positive, no action needed
- MONITOR         : Flag for increased observation, no blocking yet

Return ONLY the JSON array. No markdown. No explanation. No code blocks."""


def _build_user_prompt(findings: list) -> str:
    serialised = json.dumps(
        [asdict(f) for f in findings], indent=2, default=str
    )
    return (
        f"Analyse the following {len(findings)} threat hunting findings "
        f"from the CIC-DDoS2019 network flow dataset and return your "
        f"structured decisions:\n\n{serialised}"
    )


class SecurityAgent:
    """
    AI reasoning agent that analyses threat hunting findings via OpenAI API
    and returns structured autonomous response decisions.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        # Support Replit AI proxy or direct OpenAI key
        base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY", "")

        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables!")
            raise ValueError(
                "OPENAI_API_KEY must be set in Replit Secrets or environment variables. "
                "Go to Replit Secrets (lock icon) and add: OPENAI_API_KEY = sk-..."
            )

        if base_url:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            logger.info("Agent using Replit AI proxy")
        else:
            self.client = OpenAI(api_key=api_key)
            logger.info("Agent using direct OpenAI API key")

        self.model = model
        logger.info("SecurityAgent initialised with model: %s", self.model)

    def analyse(self, findings: list) -> list:
        """
        Reason about hunting findings and return structured decisions.

        Parameters
        ----------
        findings : list of HuntingFinding
            Produced by ThreatHunter.hunt()

        Returns
        -------
        list of dict
            One structured decision per finding.
        """
        if not findings:
            logger.info("No findings to analyse — skipping agent call")
            return []

        user_prompt = _build_user_prompt(findings)
        logger.info(
            "Sending %d findings to agent (%s)", len(findings), self.model
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=2048,
            )

            raw = response.choices[0].message.content or "[]"
            logger.debug("Raw agent response: %s", raw)
            decisions = self._parse_response(raw)
            logger.info("Agent produced %d decisions", len(decisions))
            return decisions

        except Exception as e:
            logger.error("Agent API call failed: %s", str(e))
            return self._fallback_decisions(findings)

    def _parse_response(self, raw: str) -> list:
        """Parse JSON array from model response robustly."""
        try:
            data = json.loads(raw.strip())
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            # Try to extract JSON array from response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(raw[start:end])
                except json.JSONDecodeError:
                    pass
        logger.error("Failed to parse agent response — using fallback")
        return []

    def _fallback_decisions(self, findings: list) -> list:
        """
        Fallback decisions when API call fails.
        Ensures pipeline continues even without agent response.
        """
        fallback = []
        for finding in findings:
            severity = getattr(finding, 'severity', 'MEDIUM')
            # Extract attack type from description
            desc_words = finding.description.split()
            attack_type = desc_words[1] if len(desc_words) > 1 else "Unknown"

            fallback.append({
                "rule_id": finding.rule_id,
                "attack_type": attack_type,
                "action": "ESCALATE" if severity in ["HIGH", "CRITICAL"] else "MONITOR",
                "severity": severity,
                "priority": 1 if severity == "CRITICAL" else 2 if severity == "HIGH" else 3,
                "reasoning": (
                    "Fallback decision applied due to agent API unavailability. "
                    "Manual review recommended."
                ),
                "signature_rule": None,
                "escalate_to_human": True,
                "confidence_score": 0.5
            })
        logger.warning("Using %d fallback decisions", len(fallback))
        return fallback

    def analyse_batch(self, findings: list, chunk_size: int = 10) -> list:
        """Analyse findings in chunks to stay within token limits."""
        all_decisions = []
        for i in range(0, len(findings), chunk_size):
            chunk = findings[i: i + chunk_size]
            all_decisions.extend(self.analyse(chunk))
        return all_decisions
