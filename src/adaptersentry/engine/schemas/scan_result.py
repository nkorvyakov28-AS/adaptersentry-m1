"""ScanResult — the stable public contract for a completed M1 scan.

Two output modes:
  summary-json   ScanResult as-is.  Stable public contract for CI gates.
  debug-json     DebugReport extends ScanResult with tensor_records and raw stats.
                 NOT part of the stable contract; schema_version for debug envelope
                 is always 'debug-1.0.0'.

Design invariants:
  - schema_version MUST be checked before deserializing.
  - extra="ignore" means new fields added in a future writer are safely dropped
    by an older reader without raising a ValidationError.
  - The stable summary-json shape NEVER includes raw per-layer statistics.
    Those belong exclusively in DebugReport.
  - Both modes embed ScanIdentity and AdapterArtifactIdentity so results can
    be correlated with batch runs and cache entries via scan_id and content_hash.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from adaptersentry.schemas.adapter_metadata import AdapterMetadata
from adaptersentry.schemas.adapter_report import AnalysisMode, ParseStatus
from adaptersentry.schemas.confidence_score import AnalysisQualityScore, ConfidenceScore
from adaptersentry.schemas.errors import ScanError
from adaptersentry.schemas.finding import Finding
from adaptersentry.schemas.per_layer_finding import PerLayerFinding
from adaptersentry.schemas.tensor_record import TensorRecord
from adaptersentry.engine.schemas.identity import AdapterArtifactIdentity, ScanIdentity
from adaptersentry.engine.schemas.signals import FeatureFamilyResult
from adaptersentry.engine.schemas.scoring import EnsembleSignal, RiskVerdict


class ScanStatus(str, Enum):
    """Top-level outcome of a scan attempt.

    ok       — all phases completed; all enabled detectors ran
    degraded — some phases or detectors failed; result is partial but meaningful
    failed   — file-level failure; no meaningful result produced
    cached   — result served from cache; no recomputation performed
    """

    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"
    CACHED = "cached"


class ScanResult(BaseModel):
    """Public stable contract for a completed M1 scan.

    schema_version = "1.0.0" — consumers MUST check this before deserializing.
    extra="ignore" means unknown fields from newer writers are dropped safely.

    CLI summary-json mode emits this object as-is.
    CLI debug-json mode emits DebugReport (a subclass that adds tensor_records).
    """

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    schema_version: str = "1.0.0"

    # Who scanned what
    identity: ScanIdentity
    artifact: AdapterArtifactIdentity

    # What we found about the adapter itself
    adapter_metadata: AdapterMetadata

    # Policy-level verdict (what CI gates should read)
    verdict: RiskVerdict

    # Statistical signal (pure ensemble — for tuning, not for enforcement)
    ensemble: EnsembleSignal

    # Structured findings — severity-tagged, layer-attributed
    findings: list[Finding] = Field(default_factory=list)

    # Errors — parse, analysis, or scoring failures
    errors: list[ScanError] = Field(default_factory=list)

    # Scan coverage and quality
    status: ScanStatus
    parse_status: ParseStatus
    analysis_mode: AnalysisMode
    n_layers: int = Field(default=0)
    n_layers_analyzed: int = Field(
        default=0,
        description="Layers that completed full analysis without a parse_error.",
    )

    # M1-RPT-01: top-10 suspicious layers ranked by severity
    top_layer_findings: list[PerLayerFinding] = Field(
        default_factory=list,
        description=(
            "Top-10 suspicious layers ranked by severity score (M1-RPT-01). "
            "Empty for clean adapters. Full list available in DebugReport."
        ),
    )

    # M1-SCORE-03: analysis quality and verdict confidence (separate from risk score)
    quality_score: AnalysisQualityScore | None = Field(
        default=None,
        description=(
            "Analysis quality score — parse coverage, metadata completeness, "
            "feature completeness. Does NOT derive from anomaly signals."
        ),
    )
    confidence_score: ConfidenceScore | None = Field(
        default=None,
        description=(
            "Verdict confidence score — certainty of the risk assessment "
            "given sample size and analysis coverage. Does NOT derive from anomaly signals. "
            "SaaS: free tier sees verdict_certainty; paid tier gets full confidence breakdown."
        ),
    )


class DebugReport(ScanResult):
    """Internal debug extension — NOT part of the stable public contract.

    debug_schema_version = 'debug-1.0.0'. Do not parse this in CI gates or
    external tooling. Keys in this extension may change between minor versions.

    tensor_records          — typed per-layer records (NormFeatures, parse_error, etc.)
    feature_family_results  — raw FeatureFamilyResult list from all detectors
    raw_layer_stats         — legacy flat dict from analyzer._run_analysis()["layers"]
    """

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    debug_schema_version: str = "debug-1.0.0"
    tensor_records: list[TensorRecord] = Field(default_factory=list)
    feature_family_results: list[FeatureFamilyResult] = Field(default_factory=list)
    raw_layer_stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Legacy flat layer dict from _run_analysis(). Retained during transition.",
    )
    raw_flags: list[str] = Field(
        default_factory=list,
        description="Legacy flat flag strings. Deprecated; will be removed in schema 2.0.",
    )
    wasserstein_distances: dict[str, float] = Field(default_factory=dict)
    cross_layer_consistency: float = Field(default=1.0)
