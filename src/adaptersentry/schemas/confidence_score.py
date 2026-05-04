"""ConfidenceScore and AnalysisQualityScore — M1-SCORE-03.

Two orthogonal axes that MUST NOT share signals with the risk score:

AnalysisQualityScore
    How complete and trustworthy the analysis data is.
    Signals: parse coverage, metadata completeness, feature completeness,
             proxy NaN/degenerate ratio.
    Does NOT use: kurtosis, entropy, outlier rates, energy_concentration,
                  wasserstein, cross_layer_consistency.

ConfidenceScore
    How certain the risk verdict is, given the available data.
    Signals: sample size (n_layers), feature family success rate,
             analysis quality, scan mode (full vs. fast).
    Does NOT use: any anomaly features from the ensemble risk path.

Circular-logic guard:
    confidence_score and quality_score are derived only from data-quality
    and coverage signals. They must never feed back into the risk score axis.

SaaS mapping:
    Free tier  → confidence_score visible; value is "medium" for most adapters
                 on fast mode → "upgrade for full analysis" upsell.
    Paid tier  → full scan produces confidence="high" for clean adapters,
                 providing the certainty users pay for.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AnalysisQualityScore(BaseModel):
    """Quality of the analysis itself — how complete and trustworthy the data is.

    All signals are about DATA QUALITY, not about adapter anomalies.
    """

    model_config = ConfigDict(frozen=True)

    # Raw counts
    n_layers_total: int = Field(description="Total LoRA layer pairs found in the adapter.")
    n_layers_parsed_ok: int = Field(description="Layers with no parse_error.")

    # Coverage axes [0, 1]
    parse_coverage: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of layers parsed without error. 1.0 = perfect.",
    )
    metadata_completeness: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Fraction of expected metadata fields present "
            "(base_model, peft_type, target_modules, rank)."
        ),
    )
    feature_completeness: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Fraction of layers where ALL feature schemas are non-None "
            "(norm, distribution, entropy_compression)."
        ),
    )
    degenerate_ratio: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Proxy for NaN/degenerate layers: fraction of successfully parsed layers "
            "where all per-matrix stats are exactly zero (all-zero tensor or failed computation)."
        ),
    )

    # Composite [0, 1]
    overall_quality: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Weighted composite quality score. "
            "1.0 = full coverage, complete metadata, all features computed, no degenerate layers."
        ),
    )


class ConfidenceScore(BaseModel):
    """Confidence in the risk verdict — how certain is the M1 assessment.

    All signals are about ANALYSIS BREADTH and COVERAGE, not about anomalies.
    Verdict certainty is the primary SaaS-visible field.
    """

    model_config = ConfigDict(frozen=True)

    # Raw signals
    n_layers: int = Field(description="Number of LoRA layers analyzed.")
    n_families_successful: int = Field(
        description="Feature families that succeeded for at least one layer.",
    )

    # Component factors [0, 1]
    sample_size_factor: float = Field(
        ge=0.0, le=1.0,
        description=(
            "min(1.0, n_layers / 16). "
            "More layers → more statistical power. Saturates at 16 layers."
        ),
    )
    analysis_quality: float = Field(
        ge=0.0, le=1.0,
        description="overall_quality from AnalysisQualityScore.",
    )
    inter_family_agreement: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Fraction of layers where all feature schemas were computed. "
            "High agreement means multiple independent families validated the data."
        ),
    )
    scan_mode_factor: float = Field(
        ge=0.0, le=1.0,
        description=(
            "1.0 for full scan (all detectors ran). "
            "0.65 for degraded/fast mode (approximated features)."
        ),
    )

    # Composite [0, 1]
    overall_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Weighted composite confidence score.",
    )

    # Human-facing verdict certainty
    verdict_certainty: Literal["high", "medium", "low"] = Field(
        description=(
            "high   → overall_confidence ≥ 0.75 (full scan, many layers, all features) "
            "medium → overall_confidence ≥ 0.45 (partial coverage or fast mode) "
            "low    → overall_confidence < 0.45 (few layers, parse errors, fast mode)"
        ),
    )

    # Reason tracing
    limiting_factors: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons that limited the confidence score.",
    )
