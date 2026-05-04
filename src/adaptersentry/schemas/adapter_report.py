"""AdapterReport — the top-level stable M1 report contract.

This is the versioned, machine-readable output of a completed M1 scan.
It is the future interface to M2 (behavioral analysis), M3 (signature
lookup), and M4 (runtime enforcement).

Schema version: 1.0.0
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from adaptersentry.schemas.adapter_metadata import AdapterMetadata
from adaptersentry.schemas.errors import ScanError
from adaptersentry.schemas.finding import Finding, Severity
from adaptersentry.schemas.inter_layer_similarity_features import InterLayerSimilarityFeatures
from adaptersentry.schemas.tensor_record import TensorRecord


class ParseStatus(str, Enum):
    """Parse-phase outcome — scoped to file/tensor loading only.

    ok       — all tensors loaded and converted without error
    degraded — at least one tensor had a parse error; analysis continued on the rest
    failed   — unrecoverable file-level failure; no tensors could be analysed
    """

    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"


class AnalysisMode(str, Enum):
    """Overall quality of the completed analysis."""

    FULL = "full"         # all detectors ran successfully
    DEGRADED = "degraded"  # some detectors failed or were skipped
    FAILED = "failed"     # analysis could not be completed


class TrainingStatus(str, Enum):
    """Adapter training state as classified by init_detector."""

    TRAINED = "TRAINED"
    INIT_ONLY = "INIT_ONLY"
    PARTIALLY_TRAINED = "PARTIALLY_TRAINED"
    UNKNOWN = "UNKNOWN"


class ToolInfo(BaseModel):
    """Scanner tool identity and version."""

    model_config = ConfigDict(frozen=True)

    name: str = "adaptersentry"
    version: str
    module: str = "M1-StaticAnalyzer"
    informationUri: str = "https://github.com/nkorvyakov28-AS/adaptersentry-m1"


class ScanTarget(BaseModel):
    """The artifact being scanned.

    ``path`` is the resolved absolute filesystem path at scan time.  It reflects
    the caller's local directory structure and should be redacted before sharing
    scan results externally (e.g. when uploading to a remote API or SARIF store).
    """

    model_config = ConfigDict(frozen=True)

    path: str
    file_size_bytes: int | None = None


class RiskSummary(BaseModel):
    """Aggregated risk signals for the adapter."""

    model_config = ConfigDict(frozen=True)

    overall_risk: int = Field(ge=0, le=100, description="Rule-based additive score (0–100)")
    risk_level: Severity
    ensemble_score: float = Field(ge=0.0, le=100.0, description="Ensemble score (0–100)")
    ensemble_risk_level: Severity
    training_status: TrainingStatus
    false_positive_suppressed: int = Field(
        default=0, description="Init-artifact flags suppressed"
    )
    n_layers: int
    n_findings: int
    cross_layer_consistency: float = Field(
        ge=0.0, le=1.0, description="Cross-layer flag distribution (1=uniform)"
    )
    wasserstein_mean: float | None = None


class AdapterReport(BaseModel):
    """Top-level M1 scan report — the versioned stable contract.

    schema_version tracks breaking changes to this structure.
    Consumers should check schema_version before deserializing.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    schema_version: str = Field(default="1.0.0")
    tool: ToolInfo
    scan_target: ScanTarget
    adapter_metadata: AdapterMetadata
    tensor_records: list[TensorRecord]
    findings: list[Finding]
    errors: list[ScanError]
    risk_summary: RiskSummary
    analysis_mode: AnalysisMode
    parse_status: ParseStatus = Field(
        default=ParseStatus.OK,
        description="Outcome of the file/tensor loading phase",
    )
    started_at: str = Field(description="ISO 8601 timestamp when analysis started")
    completed_at: str = Field(description="ISO 8601 timestamp when analysis completed")

    inter_layer_similarity_features: InterLayerSimilarityFeatures | None = Field(
        default=None,
        description=(
            "Adapter-level inter-layer ΔW similarity statistics (M1-ANAL-03). "
            "None when fewer than 2 layers or computation failed."
        ),
    )

    # Preserved for backward compat / benchmark integration
    _legacy_flags: list[str] = []

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert to the legacy dict format produced by analyzer.analyze().

        Used by the benchmark pipeline and callers that have not yet migrated
        to the new AdapterReport API.
        """
        rs = self.risk_summary
        return {
            "adapter_path": self.scan_target.path,
            "timestamp": self.completed_at,
            "overall_risk": rs.overall_risk,
            "risk_level": rs.risk_level.value,
            "flags": [f for finding in self.findings for f in finding.evidence.get("flags", [])],
            "layers": {
                tr.layer_name: {
                    "shape_A": tr.shape_a,
                    "shape_B": tr.shape_b,
                    "rank": tr.rank,
                    "energy_concentration": tr.energy_concentration,
                    "kurtosis_A": tr.kurtosis_a,
                    "kurtosis_B": tr.kurtosis_b,
                    "mean_A": tr.mean_a,
                    "std_A": tr.std_a,
                    "mean_B": tr.mean_b,
                    "std_B": tr.std_b,
                    "skewness_A": tr.skewness_a,
                    "entropy_A": tr.entropy_a,
                    "entropy_B": tr.entropy_b,
                    "zscore_outlier_rate_A": tr.zscore_outlier_rate_a,
                    "zscore_outlier_rate_B": tr.zscore_outlier_rate_b,
                    "isolation_score_A": tr.isolation_score_a,
                    "flags": tr.flags,
                }
                for tr in self.tensor_records
            },
            "metadata": self.adapter_metadata.raw,
            "ensemble_score": rs.ensemble_score,
            "ensemble_risk_level": rs.ensemble_risk_level.value,
            "ensemble_explanation": [],
            "cross_layer_consistency": rs.cross_layer_consistency,
            "wasserstein_distances": (
                {"_mean": rs.wasserstein_mean} if rs.wasserstein_mean is not None else {}
            ),
            "training_status": rs.training_status.value,
            "false_positive_suppressed": rs.false_positive_suppressed,
            "summary": (
                f"Analyzed {rs.n_layers} LoRA layer(s). "
                f"Training status: {rs.training_status.value}. "
                f"Risk: {rs.risk_level.value} ({rs.overall_risk}/100) | "
                f"Ensemble: {rs.ensemble_risk_level.value} ({rs.ensemble_score:.1f}/100). "
                f"{rs.n_findings} finding(s) ({rs.false_positive_suppressed} init artifact(s) suppressed)."
            ),
        }
