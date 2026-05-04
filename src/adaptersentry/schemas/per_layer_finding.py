"""PerLayerFinding — per-layer ranked suspicious finding (M1-RPT-01).

PerLayerFinding summarises the anomaly signals for one LoRA layer in a
human-readable, machine-stable format. The top-10 are stored in ScanResult;
the full list is available in DebugReport.

Ranking is by severity_score (severity_weight × flag_count), resolving
ties by raw flag count. Only layers with at least one anomaly signal are
included. Layers with parse_error are always ranked above clean layers.

triggered_families lists only the families that raised signals for this
layer (norm, distribution, entropy, outlier, spectral, entropy_compression,
training_pattern, parse). Families that ran but found nothing are omitted.

signals contains up to 5 human-readable, stable-wording descriptions mapped
from the raw flag prefixes via RULE_CATALOG. Wording is designed to remain
stable across minor versions so users can write scripts against it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from adaptersentry.schemas.finding import Severity


class PerLayerFinding(BaseModel):
    """Ranked anomaly summary for one LoRA layer."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(
        ge=1,
        description="Position in the ranked list (1 = most suspicious).",
    )
    layer_name: str = Field(
        description="Canonical LoRA layer path (e.g. 'model.layers.0.self_attn.q_proj').",
    )
    severity: Severity = Field(
        description="Severity level derived from the highest-weight triggered signals.",
    )
    severity_score: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Numeric ranking score in [0, 1]. "
            "Derived from flag severity weights; used to produce the rank ordering. "
            "Does NOT equal the adapter-level ensemble score."
        ),
    )
    triggered_families: list[str] = Field(
        description=(
            "Feature families that raised at least one signal for this layer. "
            "Possible values: norm, distribution, entropy, outlier, spectral, "
            "entropy_compression, training_pattern, parse."
        ),
    )
    flag_count: int = Field(
        ge=0,
        description="Total number of raw anomaly flags for this layer.",
    )
    signals: list[str] = Field(
        description=(
            "Up to 5 human-readable, stable signal descriptions. "
            "Derived from RULE_CATALOG titles for the triggered flags."
        ),
    )
    remediation_hint: str | None = Field(
        default=None,
        description=(
            "Highest-priority remediation advice for this layer. "
            "Drawn from RULE_CATALOG for the most severe triggered rule."
        ),
    )
