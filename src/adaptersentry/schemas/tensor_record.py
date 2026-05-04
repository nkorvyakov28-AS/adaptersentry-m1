"""TensorRecord schema — per-layer analysis results.

TensorRecord is the stable per-layer contract used by reporters and future
M2/M3/M4 modules.  It maps directly to the per-layer dict produced by
``analyzer.analyze()["layers"]``.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from adaptersentry.schemas.errors import ErrorCategory
from adaptersentry.schemas.norm_features import NormFeatures
from adaptersentry.schemas.distribution_features import DistributionFeatures
from adaptersentry.schemas.entropy_compression_features import EntropyCompressionFeatures


class TensorRecord(BaseModel):
    """Per-layer tensor analysis record."""

    model_config = ConfigDict(frozen=True)

    layer_name: str = Field(description="Canonical LoRA layer path")
    shape_a: list[int] = Field(description="lora_A weight matrix shape")
    shape_b: list[int] = Field(description="lora_B weight matrix shape")
    dtype: str = Field(default="float32")
    numel: int = Field(
        default=0,
        description="Total parameters across lora_A and lora_B matrices",
    )
    rank: int = Field(description="SVD effective rank (99% energy threshold)")
    energy_concentration: float = Field(
        description="Fraction of weight-space energy in the dominant singular value"
    )
    kurtosis_a: float = Field(description="Excess kurtosis of lora_A weights")
    kurtosis_b: float = Field(description="Excess kurtosis of lora_B weights")
    mean_a: float
    std_a: float
    mean_b: float
    std_b: float
    skewness_a: float
    entropy_a: float = Field(description="Normalized Shannon entropy of lora_A (0–1)")
    entropy_b: float = Field(description="Normalized Shannon entropy of lora_B (0–1)")
    zscore_outlier_rate_a: float = Field(
        description="Fraction of lora_A weights beyond 3σ"
    )
    zscore_outlier_rate_b: float = Field(
        description="Fraction of lora_B weights beyond 3σ"
    )
    isolation_score_a: float | None = Field(
        default=None,
        description="IsolationForest mean decision score for lora_A (negative = anomalous)",
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Per-layer anomaly flag strings",
    )
    parse_error: ErrorCategory | None = Field(
        default=None,
        description="Parse-phase error class; None means the tensor loaded cleanly",
    )
    norm_features: NormFeatures | None = Field(
        default=None,
        description=(
            "Magnitude features from ΔW = B @ A. "
            "None when the pair is incomplete or has a shape mismatch."
        ),
    )
    distribution_features: DistributionFeatures | None = Field(
        default=None,
        description=(
            "Distribution shape statistics from ΔW = B @ A "
            "(kurtosis, skewness, mean, std). "
            "None when the pair is incomplete or analysis failed."
        ),
    )
    entropy_compression_features: EntropyCompressionFeatures | None = Field(
        default=None,
        description=(
            "Entropy and compression statistics for lora_A and lora_B. "
            "None when analysis failed."
        ),
    )

    @classmethod
    def from_layer_dict(cls, layer_name: str, d: dict) -> "TensorRecord":
        """Construct from a per-layer dict produced by analyzer.analyze()."""
        shape_a = d.get("shape_A", [])
        shape_b = d.get("shape_B", [])
        numel = math.prod(shape_a) + math.prod(shape_b) if shape_a and shape_b else 0
        parse_error_raw = d.get("parse_error")
        parse_error = ErrorCategory(parse_error_raw) if parse_error_raw else None
        nf_dict = d.get("norm_features")
        norm_features = NormFeatures(**nf_dict) if nf_dict else None
        df_dict = d.get("distribution_features")
        distribution_features = DistributionFeatures(**df_dict) if df_dict else None
        ec_dict = d.get("entropy_compression_features")
        entropy_compression_features = EntropyCompressionFeatures(**ec_dict) if ec_dict else None
        return cls(
            layer_name=layer_name,
            shape_a=shape_a,
            shape_b=shape_b,
            numel=numel,
            rank=d.get("rank", 0),
            energy_concentration=d.get("energy_concentration", 0.0),
            kurtosis_a=d.get("kurtosis_A", 0.0),
            kurtosis_b=d.get("kurtosis_B", 0.0),
            mean_a=d.get("mean_A", 0.0),
            std_a=d.get("std_A", 0.0),
            mean_b=d.get("mean_B", 0.0),
            std_b=d.get("std_B", 0.0),
            skewness_a=d.get("skewness_A", 0.0),
            entropy_a=d.get("entropy_A", 0.0),
            entropy_b=d.get("entropy_B", 0.0),
            zscore_outlier_rate_a=d.get("zscore_outlier_rate_A", 0.0),
            zscore_outlier_rate_b=d.get("zscore_outlier_rate_B", 0.0),
            isolation_score_a=d.get("isolation_score_A"),
            flags=d.get("flags", []),
            parse_error=parse_error,
            norm_features=norm_features,
            distribution_features=distribution_features,
            entropy_compression_features=entropy_compression_features,
        )
