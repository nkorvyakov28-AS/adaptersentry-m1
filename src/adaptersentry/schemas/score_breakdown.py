"""ScoreBreakdown — per-family risk sub-scores for M1 scan results.

ScoreBreakdown decomposes the ensemble risk score into seven interpretable
sub-scores, one per feature family. Each sub-score carries its raw signal
value, a normalized [0, 1] score, the weight used in the ensemble, and
human-readable reasons explaining the signal.

Families
--------
parse          — file / tensor parsing health
metadata       — adapter provenance and metadata completeness
norm           — delta magnitude features (NormFeatures)
distribution   — ΔW statistical shape (kurtosis, skewness, percentiles)
entropy        — entropy and compression signals (both classic and M1-ANAL-02)
similarity     — spectral rank + inter-layer weight similarity (M1-ANAL-03)
training_pattern — cross-layer consistency, Wasserstein, training status

This schema is separate from ConfidenceScore / AnalysisQualityScore
(M1-SCORE-03) to preserve the circular-logic guard: sub-scores decompose
the risk signal and must not feed back into confidence or quality axes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SubScore(BaseModel):
    """Risk sub-score for one feature family."""

    model_config = ConfigDict(frozen=True)

    family: str = Field(
        description=(
            "Feature family name: parse, metadata, norm, distribution, "
            "entropy, similarity, or training_pattern."
        )
    )
    raw_score: float = Field(
        description="Unnormalized signal score (may exceed [0, 1]).",
    )
    normalized_score: float = Field(
        ge=0.0, le=1.0,
        description="Score clamped to [0, 1] after family-specific normalization.",
    )
    weight: float = Field(
        ge=0.0,
        description="Weight of this family in the overall score (sums to 1.0 across families).",
    )
    weighted_contribution: float = Field(
        description="normalized_score × weight — contribution to the total weighted score.",
    )
    top_reasons: list[str] = Field(
        default_factory=list,
        description="Up to 3 human-readable signals that most contributed to this sub-score.",
    )
    cap_applied: bool = Field(
        default=False,
        description="True when the policy cap reduced normalized_score from its computed value.",
    )
    floor_applied: bool = Field(
        default=False,
        description="True when the policy floor raised normalized_score from its computed value.",
    )


class ScoreBreakdown(BaseModel):
    """Full per-family decomposition of the M1 risk score."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(
        default="1.0.0",
        description="ScoreBreakdown schema version — independent from ScanResult schema version.",
    )
    sub_scores: list[SubScore] = Field(
        description="One SubScore per feature family, in canonical order.",
    )
    total_weighted: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Weighted sum of normalized_scores across all families. "
            "Maps to ensemble score via sigmoid: score ≈ sigmoid(total_weighted) × 100."
        ),
    )
    dominant_family: str = Field(
        description="Family with the highest weighted_contribution.",
    )
    # Policy application fields (populated when a ScoringPolicy was applied)
    applied_policy_version: str | None = Field(
        default=None,
        description="Version of the ScoringPolicy used. None when no policy was applied.",
    )
    escalation_rules_fired: list[str] = Field(
        default_factory=list,
        description="Names of escalation rules that fired, with their reason strings.",
    )
    score_adjustment: float = Field(
        default=0.0,
        description="Total score_bump added by fired escalation rules.",
    )
    adjusted_total_weighted: float = Field(
        default=0.0,
        description=(
            "total_weighted + score_adjustment, clamped to [0, 1]. "
            "This is the policy-adjusted score used for final risk mapping."
        ),
    )
