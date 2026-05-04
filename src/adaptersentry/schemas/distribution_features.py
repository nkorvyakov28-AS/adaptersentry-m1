"""DistributionFeatures — distribution shape statistics for ΔW = B @ A.

All statistics are derived from the *combined* delta matrix, never from
lora_A or lora_B alone. This is consistent with the M1 contract that treats
ΔW = B @ A as the effective weight patch.

delta_kurtosis
    Excess kurtosis of ΔW elements. Heavy-tailed distributions (kurtosis >> 3)
    are associated with backdoor or outlier injection patterns.

delta_skewness
    Third standardized moment of ΔW. Large |skewness| combined with high
    kurtosis is a stronger signal than kurtosis alone.

delta_mean
    Mean of ΔW elements. A nonzero mean implies a net bias shift; large
    values are anomalous.

delta_std
    Standard deviation of ΔW elements. Abnormally large std relative to
    the model's expected weight scale is an anomaly signal.

delta_median
    Median of ΔW elements. Divergence between mean and median signals asymmetry.

delta_p01 / delta_p99
    1st and 99th percentiles of ΔW. Wide spread (large |p01| or |p99|)
    combined with high kurtosis is a strong backdoor signal.

delta_iqr
    Interquartile range (p75 − p25) of ΔW. Robust spread measure unaffected
    by outlier injection.

delta_zero_ratio
    Fraction of ΔW elements within 1e-8 of zero. Unexpectedly high ratios
    indicate sparse backdoor patterns or uninitialized submatrices.

delta_entropy
    Normalized Shannon entropy of ΔW value distribution (0–1, 256 bins).
    Near-zero → constant/sparse; near-one → uniform noise injection.

When the delta matrix exceeds _MAX_DELTA_NUMEL elements (same guard as
NormFeatures), delta_kurtosis and delta_skewness are computed on a random
sample of the A matrix alone (cheaper path, clearly marked in field docs).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DistributionFeatures(BaseModel):
    """Distribution shape statistics for the LoRA delta ΔW = B @ A."""

    model_config = ConfigDict(frozen=True)

    delta_kurtosis: float = Field(
        description=(
            "Excess kurtosis of ΔW elements. "
            "Benign adapters typically fall in [-2, 6]. "
            "Values above 10 are a MEDIUM anomaly signal."
        )
    )
    delta_skewness: float = Field(
        description=(
            "Third standardized moment of ΔW elements. "
            "|skewness| > 2 combined with kurtosis > 6 is a stronger signal."
        )
    )
    delta_mean: float = Field(description="Mean of ΔW elements.")
    delta_std: float = Field(description="Standard deviation of ΔW elements.")
    delta_median: float = Field(
        default=0.0,
        description="Median of ΔW elements. Divergence from mean signals asymmetry.",
    )
    delta_p01: float = Field(
        default=0.0,
        description="1st percentile of ΔW. Large |p01| with high kurtosis is a backdoor signal.",
    )
    delta_p99: float = Field(
        default=0.0,
        description="99th percentile of ΔW. Large |p99| with high kurtosis is a backdoor signal.",
    )
    delta_iqr: float = Field(
        default=0.0,
        description="Interquartile range (p75 − p25) of ΔW. Robust spread measure.",
    )
    delta_zero_ratio: float = Field(
        default=0.0,
        description=(
            "Fraction of ΔW elements within 1e-8 of zero. "
            "High values indicate sparse or uninitialized submatrices."
        ),
    )
    delta_entropy: float = Field(
        default=0.0,
        description=(
            "Normalized Shannon entropy of ΔW value distribution (0–1, 256 bins). "
            "Near-zero → constant/sparse; near-one → uniform noise injection."
        ),
    )

    # Provenance flag: True when stats were computed on a sample, not the full delta
    computed_on_sample: bool = Field(
        default=False,
        description=(
            "True when the delta matrix exceeded _MAX_DELTA_NUMEL and stats were "
            "computed on a random row sample of lora_A instead of the full ΔW."
        ),
    )
