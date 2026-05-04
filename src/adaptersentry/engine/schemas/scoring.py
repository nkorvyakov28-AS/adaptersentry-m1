"""EnsembleSignal and RiskVerdict — separated scoring contracts.

Key design decision: these two concepts are INTENTIONALLY separate.

EnsembleSignal
    Pure statistical output of the weighted detector ensemble.
    Deterministic given the same feature values and analyzer config.
    No policy logic. No metadata influence.
    This is what you tune when adjusting detector sensitivity.

RiskVerdict
    Policy-level judgment for a CI gate or enforcement decision.
    Incorporates EnsembleSignal AND non-statistical signals:
      - missing/suspicious adapter metadata (provenance signal)
      - degraded parsing (evasion signal)
      - training_status (INIT_ONLY suppresses false positives)
    recommended_action is what automated gates should read.
    m2_recommended signals whether M2 behavioral sandbox is warranted.

The current AdapterReport collapses these two into overlapping fields
(overall_risk / ensemble_score) with no documented distinction. This module
replaces that with an explicit, schema-stable separation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from adaptersentry.schemas.adapter_report import TrainingStatus
from adaptersentry.schemas.finding import Severity
from adaptersentry.engine.schemas.signals import FeatureSignal
from adaptersentry.schemas.score_breakdown import ScoreBreakdown


class EnsembleSignal(BaseModel):
    """Pure statistical M1 signal — no policy logic, no metadata influence.

    score            — weighted sigmoid ensemble in [0, 100]
    risk_level       — threshold-mapped severity label
    top_contributors — up to 5 signals ranked by weighted contribution
    detector_weights — normalised weights used; included for reproducibility audit
    score_breakdown  — per-family sub-score decomposition (M1-SCORE-01)
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    score: float = Field(ge=0.0, le=100.0, description="Weighted sigmoid ensemble score.")
    risk_level: Severity
    top_contributors: list[FeatureSignal] = Field(
        default_factory=list,
        description="Top signals by weighted contribution to ensemble score (max 5).",
    )
    detector_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Normalised weights used. Included for reproducibility audit.",
    )
    score_breakdown: ScoreBreakdown | None = Field(
        default=None,
        description="Per-family sub-score decomposition. None for legacy scan paths.",
    )


class RiskVerdict(BaseModel):
    """Policy-level verdict — what a CI gate or enforcement policy should read.

    Combines:
      - EnsembleSignal  (statistical)
      - Metadata signals (missing provenance, metadata depth anomaly)
      - ParseStatus     (degraded parsing is itself a security signal)
      - TrainingStatus  (INIT_ONLY suppresses false positives)

    recommended_action
        'allow'  — LOW risk, no anomalies, full provenance present
        'review' — MEDIUM risk or ambiguous signals; human review recommended
        'block'  — HIGH/CRITICAL; fail-closed default

    m2_recommended
        True when M2 behavioral sandbox is recommended based on risk signals.
        Threshold: overall_score >= 14 (HIGH boundary) OR missing metadata.

    policy_signals
        Non-statistical signals that influenced verdict:
        MISSING_METADATA, METADATA_DEPTH_EXCEEDED, DEGRADED_PARSE, etc.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    overall_score: int = Field(ge=0, le=100, description="Additive rule-based score (0–100).")
    overall_level: Severity
    recommended_action: Literal["allow", "review", "block"]
    m2_recommended: bool = Field(
        description="True if M2 behavioral sandbox is recommended for this adapter."
    )
    false_positive_suppressed: int = Field(default=0)
    training_status: TrainingStatus
    policy_signals: list[FeatureSignal] = Field(
        default_factory=list,
        description=(
            "Non-statistical signals that influenced verdict: "
            "metadata, parse degradation, provenance."
        ),
    )
