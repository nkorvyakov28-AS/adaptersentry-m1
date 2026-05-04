"""worker_main — the per-adapter scan pipeline run in a worker process.

This function is the unit of work dispatched by the orchestrator. It is a
module-level function (not a method or closure) so it is picklable by
multiprocessing.Pool when using the 'spawn' start method.

Pipeline phases:
  ① ArtifactIdentityResolver — content_hash, header_hash, logical_id
  ② CacheResolver — cache hit check
  ③ M1 analysis via analyzer.scan()
  ④ Assemble ScanResult from AdapterReport
  ⑤ Assemble DebugReport (per-layer stats)

Never raises. All failures are captured in ScanResult.errors with the
appropriate ScanStatus. The orchestrator's result iterator always receives
a result object, never an exception.

Trust boundary: adapter_path arrives as a resolved absolute string (resolved
by the orchestrator before building AdapterScanRequest). Workers treat the
path as validated — no further resolution is needed.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from adaptersentry.engine.schemas.requests import AdapterScanRequest
from adaptersentry.engine.schemas.scan_result import ScanResult, ScanStatus, DebugReport

if TYPE_CHECKING:
    from adaptersentry.engine.schemas.identity import AdapterArtifactIdentity

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(start_ns: int) -> int:
    return int((time.monotonic_ns() - start_ns) / 1_000_000)


def _sha256(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode()).hexdigest()


def worker_main(
    req: AdapterScanRequest,
    analyzer_config_hash: str,
    cache_root: Path | None = None,
) -> tuple[ScanResult, DebugReport]:
    """Run the full scan pipeline for one adapter.

    Returns (ScanResult, DebugReport). ScanResult is the public contract;
    DebugReport extends it with raw per-layer data.

    Never raises — all failure modes produce a ScanResult with status=failed
    or status=degraded and populated errors list.

    Args:
        req:                   Immutable scan request.
        analyzer_config_hash:  Precomputed hash of the active AnalyzerConfig.
        cache_root:            Path to cache store root (None = cache disabled).
    """
    from adaptersentry.engine.identity import ArtifactIdentityResolver
    from adaptersentry.engine.cache import CacheStore
    from adaptersentry.engine.schemas.identity import ScanIdentity
    from adaptersentry.engine.schemas.scoring import EnsembleSignal, RiskVerdict
    from adaptersentry.schemas.adapter_report import (
        AnalysisMode, ParseStatus, TrainingStatus,
    )
    from adaptersentry.schemas.finding import Severity
    from adaptersentry.schemas.errors import ScanError, ErrorCategory, ScanPhase
    from adaptersentry.version import __version__
    from adaptersentry.analyzer import scan as m1_scan
    from adaptersentry.engine.config import AnalyzerConfig, ScanMode

    # Reconstruct scan_mode from config hash — derive fast flag for this run.
    # Workers receive only the hash, not the full config object. We use the
    # request's enabled_families as a proxy; if the batch was started with
    # --mode fast, the config hash will differ from full-mode hash naturally.
    # For simplicity: check if the request carries a scan_mode hint; otherwise
    # default to full (safe for correctness, conservative).
    _fast = getattr(req, "scan_mode", "full") == "fast"

    start_ns = time.monotonic_ns()
    started_at = _utcnow()
    adapter_path = Path(req.adapter_path)

    if not adapter_path.is_absolute():
        return _make_failed_result(
            req, started_at, 0, analyzer_config_hash, __version__,
            error_code="INVALID_PATH",
            error_msg=f"adapter_path must be absolute; got {req.adapter_path!r}",
        )

    # ── Phase ①: Artifact Identity ──────────────────────────────────────────
    try:
        identity = ArtifactIdentityResolver.resolve(adapter_path, req.source)
    except Exception as exc:
        logger.error("Identity resolution failed for %s: %s", req.adapter_path, exc)
        return _make_failed_result(
            req, started_at, _elapsed_ms(start_ns), analyzer_config_hash, __version__,
            error_code="IDENTITY_FAILED", error_msg=str(exc),
        )

    # ── Phase ②: Cache Check ─────────────────────────────────────────────────
    if cache_root is not None and not req.force_rescan:
        try:
            cache = CacheStore.open(cache_root)
            entry = cache.lookup(identity.content_hash, analyzer_config_hash)
            if entry is not None:
                raw_bytes = cache.validate_and_read(entry, __version__)
                if raw_bytes is not None:
                    cache.record_hit(entry)
                    cache.close()
                    result = ScanResult.model_validate_json(raw_bytes)
                    debug = DebugReport.model_validate(result.model_dump())
                    logger.debug("Cache hit for %s (scan_id=%s)", req.adapter_path, entry.scan_id)
                    return result, debug
            cache.close()
        except Exception as exc:
            logger.warning("Cache check failed for %s: %s — treating as miss", req.adapter_path, exc)

    # ── Phase ③: M1 Analysis ─────────────────────────────────────────────────
    try:
        adapter_report = m1_scan(adapter_path, claimed_rank=req.claimed_rank, fast=_fast)
    except Exception as exc:
        logger.error("M1 scan raised unexpectedly for %s: %s", req.adapter_path, exc)
        return _make_failed_result(
            req, started_at, _elapsed_ms(start_ns), analyzer_config_hash, __version__,
            error_code="ANALYSIS_CRASHED", error_msg=str(exc),
        )

    completed_at = _utcnow()
    wall_ms = _elapsed_ms(start_ns)

    # ── Phase ④: Assemble ScanResult from AdapterReport ──────────────────────
    rs = adapter_report.risk_summary

    # Determine scan_id deterministically
    scan_id = _sha256(
        identity.content_hash + ":" + analyzer_config_hash + ":1.0.0"
    )

    scan_identity = ScanIdentity(
        scan_id=scan_id,
        run_id=req.run_id,
        analyzer_version=__version__,
        analyzer_config_hash=analyzer_config_hash,
        schema_version="1.0.0",
        started_at=started_at,
        completed_at=completed_at,
        wall_time_ms=wall_ms,
    )

    # Map AdapterReport.parse_status + analysis_mode → ScanStatus
    if adapter_report.parse_status == ParseStatus.FAILED:
        status = ScanStatus.FAILED
    elif adapter_report.analysis_mode == AnalysisMode.DEGRADED:
        status = ScanStatus.DEGRADED
    else:
        status = ScanStatus.OK

    # Build EnsembleSignal — fill detector_weights + compute ScoreBreakdown
    from adaptersentry.scoring.ensemble import EnsembleDetector
    from adaptersentry.scoring.score_breakdown import compute_score_breakdown, get_default_policy
    from adaptersentry.scoring.confidence import compute_quality_score, compute_confidence_score

    _detector = EnsembleDetector()
    try:
        score_breakdown = compute_score_breakdown(adapter_report, policy=get_default_policy())
    except Exception as _sb_exc:
        logger.warning("ScoreBreakdown computation failed: %s", _sb_exc)
        score_breakdown = None

    try:
        quality_score = compute_quality_score(adapter_report)
        confidence_score = compute_confidence_score(adapter_report, quality_score)
    except Exception as _conf_exc:
        logger.warning("ConfidenceScore computation failed: %s", _conf_exc)
        quality_score = None
        confidence_score = None

    from adaptersentry.reporting.per_layer import compute_per_layer_findings
    try:
        top_layer_findings = compute_per_layer_findings(
            adapter_report.tensor_records,
            inter_layer_features=adapter_report.inter_layer_similarity_features,
            top_k=10,
        )
    except Exception as _plf_exc:
        logger.warning("PerLayerFinding computation failed: %s", _plf_exc)
        top_layer_findings = []

    ensemble = EnsembleSignal(
        score=rs.ensemble_score,
        risk_level=rs.ensemble_risk_level,
        detector_weights=_detector.weights,
        score_breakdown=score_breakdown,
    )

    # Determine recommended_action from risk level
    level_val = rs.ensemble_risk_level.value
    if level_val in ("HIGH", "CRITICAL"):
        action = "block"
    elif level_val == "MEDIUM":
        action = "review"
    else:
        action = "allow"

    # M2 recommended if HIGH/CRITICAL or missing metadata
    m2_rec = level_val in ("HIGH", "CRITICAL") or not adapter_report.adapter_metadata.metadata_present

    verdict = RiskVerdict(
        overall_score=rs.overall_risk,
        overall_level=rs.risk_level,
        recommended_action=action,
        m2_recommended=m2_rec,
        false_positive_suppressed=rs.false_positive_suppressed,
        training_status=rs.training_status,
        policy_signals=[],
    )

    n_layers_analyzed = sum(
        1 for tr in adapter_report.tensor_records if tr.parse_error is None
    )

    result = ScanResult(
        identity=scan_identity,
        artifact=identity,
        adapter_metadata=adapter_report.adapter_metadata,
        verdict=verdict,
        ensemble=ensemble,
        findings=list(adapter_report.findings),
        errors=list(adapter_report.errors),
        status=status,
        parse_status=adapter_report.parse_status,
        analysis_mode=adapter_report.analysis_mode,
        n_layers=rs.n_layers,
        n_layers_analyzed=n_layers_analyzed,
        quality_score=quality_score,
        confidence_score=confidence_score,
        top_layer_findings=top_layer_findings,
    )

    # ── Phase ⑤: Assemble DebugReport ────────────────────────────────────────
    # Reconstruct typed FeatureFamilyResult objects from each TensorRecord
    # via the migration bridge (CARD-08). This populates feature_family_results
    # without re-running analysis, using the data already in the TensorRecords.
    from adaptersentry.engine.feature_extractor import FeatureExtractor
    from adaptersentry.engine.schemas.signals import FeatureFamilyResult
    feature_family_results = [
        ffr
        for tr in adapter_report.tensor_records
        for ffr in FeatureExtractor.families_from_record(tr)
    ]
    # Adapter-level inter_layer family (M1-ANAL-03)
    il = adapter_report.inter_layer_similarity_features
    if il is not None:
        feature_family_results.append(FeatureFamilyResult(
            family="inter_layer",
            family_schema_version="1.0.0",
            layer=None,
            status="ok",
            raw_features={
                "cosine_sim_mean": float(il.cosine_sim_mean),
                "cosine_sim_std": float(il.cosine_sim_std),
                "pearson_mean": float(il.pearson_mean),
                "n_pairs_computed": float(il.n_pairs_computed),
                "n_suspicious_pairs": float(il.n_suspicious_pairs),
            },
        ))

    debug = DebugReport(
        **result.model_dump(),
        debug_schema_version="debug-1.0.0",
        tensor_records=list(adapter_report.tensor_records),
        feature_family_results=feature_family_results,
        raw_flags=[
            flag
            for finding in adapter_report.findings
            for flag in finding.evidence.get("flags", [])
        ],
        cross_layer_consistency=rs.cross_layer_consistency,
    )

    logger.debug(
        "Scan complete: %s → %s (%dms, %d layers)",
        adapter_path.name, status.value, wall_ms, rs.n_layers,
    )

    return result, debug


def _make_failed_result(
    req: AdapterScanRequest,
    started_at: str,
    wall_ms: int,
    analyzer_config_hash: str,
    tool_version: str,
    *,
    error_code: str,
    error_msg: str,
) -> tuple[ScanResult, DebugReport]:
    """Construct a minimal failed ScanResult when a phase crashes."""
    from adaptersentry.engine.schemas.identity import AdapterArtifactIdentity, ScanIdentity
    from adaptersentry.engine.schemas.requests import ArtifactSource
    from adaptersentry.engine.schemas.scoring import EnsembleSignal, RiskVerdict
    from adaptersentry.schemas.adapter_metadata import AdapterMetadata
    from adaptersentry.schemas.adapter_report import AnalysisMode, ParseStatus, TrainingStatus
    from adaptersentry.schemas.errors import ScanError, ErrorCategory, ScanPhase
    from adaptersentry.schemas.finding import Severity

    completed_at = _utcnow()
    scan_id = _sha256(req.request_id + ":" + analyzer_config_hash + ":failed")

    dummy_identity = AdapterArtifactIdentity(
        logical_id=_sha256(req.adapter_path),
        content_hash="sha256:" + "0" * 64,
        header_hash="sha256:" + "0" * 64,
        file_size_bytes=0,
        source=req.source,
        resolved_at=completed_at,
    )

    result = ScanResult(
        identity=ScanIdentity(
            scan_id=scan_id,
            run_id=req.run_id,
            analyzer_version=tool_version,
            analyzer_config_hash=analyzer_config_hash,
            schema_version="1.0.0",
            started_at=started_at,
            completed_at=completed_at,
            wall_time_ms=wall_ms,
        ),
        artifact=dummy_identity,
        adapter_metadata=AdapterMetadata.from_parsed({}),
        verdict=RiskVerdict(
            overall_score=0,
            overall_level=Severity.LOW,
            recommended_action="review",
            m2_recommended=False,
            training_status=TrainingStatus.UNKNOWN,
        ),
        ensemble=EnsembleSignal(
            score=0.0,
            risk_level=Severity.LOW,
        ),
        errors=[ScanError.malformed(code=error_code, message=error_msg, phase=ScanPhase.PARSE)],
        status=ScanStatus.FAILED,
        parse_status=ParseStatus.FAILED,
        analysis_mode=AnalysisMode.FAILED,
    )
    debug = DebugReport(**result.model_dump(), debug_schema_version="debug-1.0.0")
    return result, debug
