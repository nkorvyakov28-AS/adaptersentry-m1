"""FeatureSignal and FeatureFamilyResult — structured anomaly signals.

Replaces the current flat ``flags: list[str]`` with machine-parseable signals.
Each signal carries name, value, threshold, direction, layer, severity, and
an optional human_message for CLI rendering.

The human_message field is NOT part of the stable contract — it may change
between minor versions. All machine consumers should read name/value/threshold.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from adaptersentry.schemas.finding import Severity


class FeatureSignal(BaseModel):
    """A single machine-parseable anomaly signal from one detector.

    name        — stable identifier, e.g. 'HIGH_KURTOSIS_DELTA', 'LOW_DELTA_ENTROPY'
    family      — feature family: 'distribution', 'norm', 'entropy', 'outlier',
                  'spectral', 'inter_layer'
    value       — observed metric value
    threshold   — threshold above/below which the signal fires
    direction   — 'above' means signal fires when value > threshold (e.g. high kurtosis)
                  'below' means signal fires when value < threshold (e.g. low entropy)
    layer       — layer name for per-layer signals; None for adapter-level signals
    matrix      — which matrix the signal applies to: 'A', 'B', 'delta', or 'adapter'
    severity    — LOW / MEDIUM / HIGH / CRITICAL
    confidence  — [0, 1]; 1.0 for deterministic rule-based signals
    human_message — optional; CLI renders this; not part of the stable contract
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    family: str
    value: float
    threshold: float
    direction: Literal["above", "below"] = "above"
    layer: str | None = None
    matrix: Literal["A", "B", "delta", "adapter"] | None = None
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    human_message: str | None = None


class FeatureFamilyResult(BaseModel):
    """Result of one feature family for one layer (or the adapter as a whole).

    family_schema_version is independent from the top-level schema_version.
    New raw_features keys added to a family bump its MINOR version.
    Renaming or removing a key bumps its MAJOR version.

    raw_features is included in debug-json output; omitted from summary-json.
    Keys are stable within a given family_schema_version.

    Defined families and their current family_schema_version:
        norm        — 1.0.0  (fro_norm_delta, max_abs_delta, mean_abs_delta, delta_norm_ratio)
        distribution — 1.0.0 (kurtosis_delta, skewness_delta, mean_delta, std_delta)
        entropy     — 1.0.0  (entropy_delta, entropy_A, entropy_B)
        outlier     — 1.0.0  (zscore_outlier_rate_A, zscore_outlier_rate_B, isolation_score_A)
        spectral    — 1.0.0  (effective_rank, energy_concentration, rank_ratio)
        inter_layer — 1.0.0  (cross_layer_consistency, wasserstein_mean, delta_cosine_sim_mean)
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    family: str
    family_schema_version: str
    layer: str | None = Field(
        default=None,
        description="None for adapter-level families (inter_layer).",
    )
    status: Literal["ok", "degraded", "skipped", "failed"]
    signals: list[FeatureSignal] = Field(default_factory=list)
    raw_features: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Raw computed scalars. Included in debug-json; omitted from summary-json. "
            "Keys are stable within family_schema_version."
        ),
    )
    error: str | None = Field(
        default=None,
        description="Error message when status is 'failed' or 'degraded'.",
    )
