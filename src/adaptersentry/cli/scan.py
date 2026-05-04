"""scan subcommand — the primary CLI action for AdapterSentry.

All business logic stays in adaptersentry.analyzer; this module is a thin
argument-parsing and output-routing layer.

Output formats
--------------
text          Human-readable table with ΔW norm, flags, and findings.
summary-json  ScanResult — stable public contract; CI gates should use this.
debug-json    DebugReport — ScanResult extended with tensor_records and raw
              layer stats. NOT part of the stable contract; for local debugging.
json          Legacy alias for summary-json (AdapterReport schema; deprecated).
sarif         SARIF 2.1.0 format for integration with GitHub Code Scanning.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_FORMATS = ("text", "summary-json", "debug-json", "json", "sarif")
_MODES = ("full", "fast")
_SEVERITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
_SEVERITY_ORDER = {s: i for i, s in enumerate(_SEVERITIES)}


def build_parser(subparsers: Any) -> None:
    """Register the ``scan`` subcommand with the top-level parser."""
    p = subparsers.add_parser(
        "scan",
        help="Scan a LoRA adapter .safetensors file",
        description=(
            "Run M1 static analysis on a LoRA adapter file.\n\n"
            "Exit codes:\n"
            "  0  — analysis completed, no findings at or above --fail-on threshold\n"
            "  1  — operational failure (file not found, parse error, etc.)\n"
            "  2  — findings at or above --fail-on threshold detected\n"
        ),
    )
    p.add_argument(
        "adapter",
        type=Path,
        metavar="ADAPTER",
        help="Path to .safetensors adapter file",
    )
    p.add_argument(
        "--format",
        choices=_FORMATS,
        default="text",
        metavar="FORMAT",
        dest="fmt",
        help="Output format: text (default), summary-json, debug-json, json (legacy), sarif",
    )
    p.add_argument(
        "--mode",
        choices=_MODES,
        default="full",
        dest="scan_mode",
        help=(
            "Scan depth: full (default) — all detectors at full depth; "
            "fast — truncated SVD, sampling, no IsolationForest on large tensors."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write output to FILE instead of stdout",
    )
    p.add_argument(
        "--rank",
        type=int,
        default=None,
        metavar="R",
        help="Declared LoRA rank r (overrides adapter_config metadata)",
    )
    p.add_argument(
        "--fail-on",
        choices=_SEVERITIES,
        default=None,
        metavar="SEVERITY",
        dest="fail_on",
        help=(
            "Exit with code 2 if any finding meets or exceeds SEVERITY. "
            "Severity order: LOW < MEDIUM < HIGH < CRITICAL."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        dest="report_verbose",
        help=(
            "Verbose text output: full score breakdown, per-layer findings, "
            "analysis quality block (text format only)."
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        dest="no_color",
        help="Disable ANSI colour output (text format only)",
    )


def _build_scan_result(adapter_path: Path, report: Any, claimed_rank: int | None) -> Any:
    """Construct a ScanResult from an AdapterReport for the single-scan CLI path."""
    import hashlib
    from datetime import datetime, timezone

    from adaptersentry.engine.config import AnalyzerConfig
    from adaptersentry.engine.identity import ArtifactIdentityResolver
    from adaptersentry.engine.schemas.identity import ScanIdentity
    from adaptersentry.engine.schemas.requests import ArtifactSource
    from adaptersentry.engine.schemas.scan_result import ScanResult, ScanStatus
    from adaptersentry.engine.schemas.scoring import EnsembleSignal, RiskVerdict
    from adaptersentry.schemas.adapter_report import AnalysisMode, ParseStatus, TrainingStatus
    from adaptersentry.schemas.finding import Severity
    from adaptersentry.version import __version__

    source = ArtifactSource(kind="local_path", local_path=str(adapter_path.resolve()))
    try:
        artifact = ArtifactIdentityResolver.resolve(adapter_path, source)
    except Exception:
        path_hash = hashlib.sha256(str(adapter_path.resolve()).encode()).hexdigest()
        from adaptersentry.engine.schemas.identity import AdapterArtifactIdentity
        file_size = adapter_path.stat().st_size if adapter_path.exists() else 0
        artifact = AdapterArtifactIdentity(
            logical_id="sha256:" + path_hash,
            content_hash="sha256:" + path_hash,
            header_hash="sha256:" + path_hash,
            file_size_bytes=file_size,
            source=source,
            resolved_at=datetime.now(timezone.utc).isoformat(),
        )

    config = AnalyzerConfig()
    config_hash = config.config_hash()
    scan_id = "sha256:" + hashlib.sha256(
        (artifact.content_hash + ":" + config_hash + ":1.0.0").encode()
    ).hexdigest()
    now = datetime.now(timezone.utc).isoformat()

    identity = ScanIdentity(
        scan_id=scan_id,
        run_id="cli-scan",
        analyzer_version=__version__,
        analyzer_config_hash=config_hash,
        schema_version="1.0.0",
        started_at=now,
        completed_at=now,
        wall_time_ms=0,
    )

    rs = report.risk_summary
    level_val = rs.ensemble_risk_level.value
    action = "block" if level_val in ("HIGH", "CRITICAL") else "review" if level_val == "MEDIUM" else "allow"
    m2_rec = level_val in ("HIGH", "CRITICAL") or not report.adapter_metadata.metadata_present

    verdict = RiskVerdict(
        overall_score=rs.overall_risk,
        overall_level=rs.risk_level,
        recommended_action=action,
        m2_recommended=m2_rec,
        false_positive_suppressed=rs.false_positive_suppressed,
        training_status=rs.training_status,
        policy_signals=[],
    )
    ensemble = EnsembleSignal(score=rs.ensemble_score, risk_level=rs.ensemble_risk_level)

    if report.parse_status == ParseStatus.FAILED:
        status = ScanStatus.FAILED
    elif report.analysis_mode == AnalysisMode.DEGRADED:
        status = ScanStatus.DEGRADED
    else:
        status = ScanStatus.OK

    n_layers_analyzed = sum(1 for tr in report.tensor_records if tr.parse_error is None)

    return ScanResult(
        identity=identity,
        artifact=artifact,
        adapter_metadata=report.adapter_metadata,
        verdict=verdict,
        ensemble=ensemble,
        findings=list(report.findings),
        errors=list(report.errors),
        status=status,
        parse_status=report.parse_status,
        analysis_mode=report.analysis_mode,
        n_layers=rs.n_layers,
        n_layers_analyzed=n_layers_analyzed,
    )


def _build_debug_report(result: Any, report: Any) -> Any:
    """Extend a ScanResult into a DebugReport for --format debug-json."""
    from adaptersentry.engine.feature_extractor import FeatureExtractor
    from adaptersentry.engine.schemas.scan_result import DebugReport

    feature_family_results = [
        ffr
        for tr in report.tensor_records
        for ffr in FeatureExtractor.families_from_record(tr)
    ]
    rs = report.risk_summary
    return DebugReport(
        **result.model_dump(),
        debug_schema_version="debug-1.0.0",
        tensor_records=list(report.tensor_records),
        feature_family_results=feature_family_results,
        raw_flags=[
            flag
            for finding in report.findings
            for flag in finding.evidence.get("flags", [])
        ],
        cross_layer_consistency=rs.cross_layer_consistency,
    )


def run(args: Any) -> int:
    """Execute the scan subcommand."""
    import logging

    from adaptersentry.analyzer import scan
    from adaptersentry.reporters import json as json_reporter
    from adaptersentry.reporters import sarif as sarif_reporter
    from adaptersentry.reporters import text as text_reporter
    from adaptersentry.schemas.adapter_report import ParseStatus

    logger = logging.getLogger("adaptersentry.cli.scan")

    try:
        fast = getattr(args, "scan_mode", "full") == "fast"
        report = scan(args.adapter, claimed_rank=args.rank, fast=fast)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unexpected analysis error", exc_info=True)
        print(f"error: analysis failed: {exc}", file=sys.stderr)
        return 1

    fmt = args.fmt
    no_color = getattr(args, "no_color", False)

    if fmt == "json":
        logger.warning("--format json is deprecated; use --format summary-json")

    if fmt == "text":
        from adaptersentry.reporting.human_summary import render_human_summary
        report_verbose = getattr(args, "report_verbose", False)
        output = render_human_summary(report, verbose=report_verbose, no_color=no_color)
    elif fmt in ("summary-json", "json"):
        try:
            result = _build_scan_result(args.adapter, report, args.rank)
            output = result.model_dump_json(indent=2)
        except Exception as exc:
            logger.debug("ScanResult construction failed, falling back", exc_info=True)
            output = json_reporter.render(report)
    elif fmt == "debug-json":
        try:
            result = _build_scan_result(args.adapter, report, args.rank)
            debug = _build_debug_report(result, report)
            output = debug.model_dump_json(indent=2)
        except Exception as exc:
            logger.debug("DebugReport construction failed, falling back", exc_info=True)
            output = json_reporter.render(report)
    else:  # sarif
        output = sarif_reporter.render_json(report)

    output_path: Path | None = getattr(args, "output", None)
    if output_path:
        try:
            output_path.write_text(output, encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to write output to %s: %s", output_path, exc)
            return 1
        if not getattr(args, "quiet", False):
            logger.info("Output written to %s", output_path)
    else:
        sys.stdout.write(output)
        if not output.endswith("\n"):
            sys.stdout.write("\n")

    if report.parse_status == ParseStatus.FAILED:
        return 1

    fail_on: str | None = getattr(args, "fail_on", None)
    if fail_on:
        threshold_order = _SEVERITY_ORDER[fail_on]
        triggered = any(
            _SEVERITY_ORDER[finding.severity.value] >= threshold_order
            for finding in report.findings
        )
        if triggered:
            return 2

    return 0
