"""M1 Static Analyzer — weight tensor inspection for LoRA adapter detection.

Treats each LoRA adapter like a delta firmware patch: we inspect the patch
itself (static analysis) without running the model, looking for anomalous
weight distributions that indicate backdoors or alignment bypass.

Public API
----------
analyze(path, claimed_rank)  → dict          Legacy dict format (stable)
scan(path, claimed_rank)     → AdapterReport New typed API (stable from v1.0.0)
load_adapter(path)           → (tensors, metadata)

All private symbols re-exported here are for backward compatibility with
tests that were written before the src-layout refactor.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adaptersentry.detectors.cross_layer import detect_cross_layer_anomalies
from adaptersentry.detectors.entropy import compute_entropy, detect_entropy_anomalies
from adaptersentry.detectors.init_detector import (
    get_adapter_training_status,
    suppress_init_flags,
)
from adaptersentry.detectors.outlier import detect_outlier_anomalies
from adaptersentry.detectors.wasserstein import compute_wasserstein_distance
from adaptersentry.features.delta_norm import compute_norm_features
from adaptersentry.features.distribution import compute_distribution_features
from adaptersentry.features.layer_stats import detect_layer_anomalies
from adaptersentry.features.tensor_stats import compute_svd_stats, compute_tensor_stats
from adaptersentry.parsers.metadata import _metadata_depth, parse_adapter_metadata
from adaptersentry.parsers.safetensors import _group_lora_layers, load_adapter
from adaptersentry.scoring.risk_scorer import RiskScorer

logger = logging.getLogger(__name__)

# Module-level scorer for the backward-compat wrappers
_default_scorer = RiskScorer()


# ---------------------------------------------------------------------------
# Backward-compat thin wrappers (used by tests via `from adaptersentry.analyzer import …`)
# ---------------------------------------------------------------------------


def _score_from_flags(flags: list[str]) -> int:
    """Derive a 0–100 risk score from accumulated anomaly flags."""
    return _default_scorer.score_flags(flags)


def _risk_level(score: int) -> str:
    """Map a numeric risk score to a severity label."""
    return _default_scorer.risk_level(score)


# ---------------------------------------------------------------------------
# Internal analysis pipeline
# ---------------------------------------------------------------------------

_MAX_SAFE_METADATA_DEPTH = 5


def _run_analysis(
    adapter_path: Path,
    claimed_rank: int | None = None,
    *,
    fast: bool = False,
) -> dict[str, Any]:
    """Core M1 analysis pipeline — returns the full dict report."""
    tensors, metadata = load_adapter(adapter_path)
    layers = _group_lora_layers(tensors)

    all_flags: list[str] = []
    layer_reports: dict[str, Any] = {}
    _il_pairs: list[tuple[str, int, Any, Any]] = []  # (name, idx, tensor_A, tensor_B)

    depth = _metadata_depth(metadata)
    if depth > _MAX_SAFE_METADATA_DEPTH:
        all_flags.append(
            f"METADATA_DEPTH: nesting={depth} > {_MAX_SAFE_METADATA_DEPTH}"
            " (possible metadata evasion attempt)"
        )

    if not metadata:
        all_flags.append(
            "MISSING_ADAPTER_METADATA: safetensors header contains no metadata"
            " — adapter provenance unknown (legitimate distributed adapters usually"
            " carry rank, base model, and peft_type)"
        )

    if claimed_rank is None:
        for key in ("r", "rank", "lora_r"):
            raw = metadata.get(key)
            if raw is not None:
                try:
                    claimed_rank = int(raw)
                    logger.debug("Inferred claimed_rank=%d from metadata[%r]", claimed_rank, key)
                    break
                except (ValueError, TypeError):
                    pass

    if not layers:
        logger.warning("No lora_A/lora_B tensor pairs found in %s", adapter_path)

    for _il_idx, (layer_name, pair) in enumerate(layers.items()):
        tensor_A = pair.get("A")
        tensor_B = pair.get("B")

        if tensor_A is None or tensor_B is None:
            logger.debug("Skipping incomplete pair for layer: %s", layer_name)
            continue

        _il_pairs.append((layer_name, _il_idx, tensor_A, tensor_B))

        try:
            stats_A = compute_tensor_stats(tensor_A, fast=fast)
            stats_B = compute_tensor_stats(tensor_B, fast=fast)
            svd = compute_svd_stats(tensor_A, fast=fast)
            layer_flags = detect_layer_anomalies(layer_name, stats_A, stats_B, svd, claimed_rank)

            entropy_A = compute_entropy(tensor_A)
            entropy_B = compute_entropy(tensor_B)
            layer_flags += detect_entropy_anomalies(entropy_A, layer_name, "A")
            layer_flags += detect_entropy_anomalies(entropy_B, layer_name, "B")

            zs_A, iso_A, outlier_flags_A = detect_outlier_anomalies(
                tensor_A, layer_name, "A", fast=fast
            )
            zs_B, _iso_B, outlier_flags_B = detect_outlier_anomalies(
                tensor_B, layer_name, "B", run_isolation_forest=False, fast=fast
            )
            layer_flags += outlier_flags_A + outlier_flags_B

            norm_feats = compute_norm_features(tensor_A, tensor_B, fast=fast)
            dist_feats = compute_distribution_features(tensor_A, tensor_B, fast=fast)

            all_flags.extend(layer_flags)
            layer_reports[layer_name] = {
                "shape_A": list(tensor_A.shape),
                "shape_B": list(tensor_B.shape),
                "rank": svd["effective_rank"],
                "energy_concentration": svd["energy_concentration"],
                "kurtosis_A": stats_A["kurtosis"],
                "kurtosis_B": stats_B["kurtosis"],
                "mean_A": stats_A["mean"],
                "std_A": stats_A["std"],
                "mean_B": stats_B["mean"],
                "std_B": stats_B["std"],
                "skewness_A": stats_A["skewness"],
                "entropy_A": entropy_A,
                "entropy_B": entropy_B,
                "zscore_outlier_rate_A": zs_A["outlier_rate"],
                "zscore_outlier_rate_B": zs_B["outlier_rate"],
                "isolation_score_A": iso_A.get("mean_score"),
                "flags": layer_flags,
                "norm_features": norm_feats.model_dump() if norm_feats else None,
                "distribution_features": dist_feats.model_dump() if dist_feats else None,
            }
        except Exception as exc:
            # Preserve a minimal degraded record so the layer is visible in the
            # report and downstream cross-layer detectors account for it.
            logger.warning("Layer %r: analysis failed — marking degraded: %s", layer_name, exc)
            degraded_flag = f"DEGRADED_LAYER: analysis failed — {exc}"
            all_flags.append(degraded_flag)
            layer_reports[layer_name] = {
                "shape_A": list(tensor_A.shape),
                "shape_B": list(tensor_B.shape),
                "rank": 0,
                "energy_concentration": 0.0,
                "kurtosis_A": 0.0,
                "kurtosis_B": 0.0,
                "mean_A": 0.0,
                "std_A": 0.0,
                "mean_B": 0.0,
                "std_B": 0.0,
                "skewness_A": 0.0,
                "entropy_A": 0.0,
                "entropy_B": 0.0,
                "zscore_outlier_rate_A": 0.0,
                "zscore_outlier_rate_B": 0.0,
                "isolation_score_A": None,
                "flags": [degraded_flag],
                "parse_error": "degraded",
            }

    # Init-only detection and flag suppression
    training_status = get_adapter_training_status(layer_reports)
    false_positive_suppressed = 0

    if training_status == "INIT_ONLY":
        all_flags, false_positive_suppressed = suppress_init_flags(all_flags)
        for lname in layer_reports:
            clean_layer_flags, _ = suppress_init_flags(layer_reports[lname].get("flags", []))
            layer_reports[lname]["flags"] = clean_layer_flags
        all_flags.append(
            "INIT_ONLY_ADAPTER: all layers are at LoRA zero-init state"
            " (B=0, A=uniform-random); init artifact flags suppressed"
        )
        logger.info(
            "%s: INIT_ONLY — suppressed %d init-artifact flag(s)",
            adapter_path.name, false_positive_suppressed,
        )
    elif training_status == "PARTIALLY_TRAINED":
        all_flags.append(
            "SUSPICIOUS_PARTIAL_TRAINING: some layers trained, others still at"
            " LoRA init state — possible targeted-layer injection"
        )
        logger.warning(
            "%s: PARTIALLY_TRAINED — possible targeted-layer injection", adapter_path.name
        )

    cross_consistency, cross_flags = detect_cross_layer_anomalies(layer_reports)
    all_flags.extend(cross_flags)

    wasserstein_distances: dict[str, float] = {}
    for layer_name, pair in layers.items():
        tensor_A = pair.get("A")
        tensor_B = pair.get("B")
        if tensor_A is not None and tensor_B is not None:
            try:
                w2 = compute_wasserstein_distance(tensor_A, tensor_B)
                wasserstein_distances[layer_name] = w2
            except Exception:  # noqa: BLE001
                pass

    if wasserstein_distances:
        wasserstein_distances["_mean"] = float(
            sum(wasserstein_distances.values()) / len(wasserstein_distances)
        )

    # Inter-layer similarity (M1-ANAL-03)
    _il_feats = None
    if len(_il_pairs) >= 2:
        try:
            from adaptersentry.features.inter_layer_similarity import (
                compute_inter_layer_similarity,
            )
            _il_feats = compute_inter_layer_similarity(_il_pairs, fast=fast)
        except Exception as _il_exc:
            logger.warning("Inter-layer similarity failed: %s", _il_exc)

    partial_report = {
        "flags": all_flags,
        "layers": layer_reports,
        "cross_layer_consistency": cross_consistency,
        "wasserstein_distances": wasserstein_distances,
    }
    overall_risk, risk_level = RiskScorer().score_report(partial_report)

    try:
        from adaptersentry.scoring.ensemble import EnsembleDetector
        detector = EnsembleDetector()
        w2_mean = wasserstein_distances.get("_mean", 0.0)
        ensemble_score = float(detector.score(
            layer_reports,
            wasserstein_score=w2_mean,
            cross_layer_consistency=cross_consistency,
        ))
        ensemble_risk_level = detector.risk_level(int(ensemble_score))
        ensemble_explanation = detector.explain(
            layer_reports,
            wasserstein_score=w2_mean,
            cross_layer_consistency=cross_consistency,
        )
    except Exception:  # noqa: BLE001
        ensemble_score = float(overall_risk)
        ensemble_risk_level = risk_level
        ensemble_explanation = []

    return {
        "adapter_path": str(adapter_path.resolve()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_risk": overall_risk,
        "risk_level": risk_level,
        "flags": all_flags,
        "layers": layer_reports,
        "metadata": metadata,
        "ensemble_score": ensemble_score,
        "ensemble_risk_level": ensemble_risk_level,
        "ensemble_explanation": ensemble_explanation,
        "cross_layer_consistency": cross_consistency,
        "wasserstein_distances": wasserstein_distances,
        "training_status": training_status,
        "false_positive_suppressed": false_positive_suppressed,
        "inter_layer_similarity_features": _il_feats.model_dump() if _il_feats else None,
        "summary": (
            f"Analyzed {len(layer_reports)} LoRA layer(s). "
            f"Training status: {training_status}. "
            f"Risk: {risk_level} ({overall_risk}/100) | "
            f"Ensemble: {ensemble_risk_level} ({ensemble_score:.1f}/100). "
            f"{len(all_flags)} flag(s) ({false_positive_suppressed} init artifact(s) suppressed)."
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze(adapter_path: Path, claimed_rank: int | None = None) -> dict[str, Any]:
    """Run M1 static analysis on a LoRA adapter file.

    Loads the adapter without executing the model, computes per-layer weight
    statistics, applies anomaly detection rules, and returns a structured JSON
    risk report conforming to the M1 output schema defined in README.md.

    Args:
        adapter_path: Path to the .safetensors file.
        claimed_rank: Expected LoRA rank r from the adapter config, if known.
                      Inferred from metadata keys "r", "rank", or "lora_r"
                      when not provided.

    Returns:
        Risk report dict conforming to the M1 JSON schema.

    Raises:
        FileNotFoundError: If the adapter file does not exist.
        ValueError: If the file cannot be parsed as safetensors.
        Exception: Other unexpected failures from downstream detectors may
            propagate.  Use scan() for a fully exception-safe public API.
    """
    return _run_analysis(adapter_path, claimed_rank)


def scan(
    adapter_path: Path,
    claimed_rank: int | None = None,
    *,
    fast: bool = False,
) -> "Any":  # AdapterReport — imported lazily to keep analyze() fast
    """Run M1 static analysis and return a typed AdapterReport.

    This is the preferred API for new callers.  The legacy ``analyze()``
    function is preserved for backward compatibility with existing tools
    and benchmarks.

    Unlike analyze(), scan() never raises — all failure modes are captured in
    AdapterReport.parse_status / .errors so the caller always receives a
    schema-stable result.  Prefer scan() over analyze() for new integrations.

    Args:
        adapter_path: Path to the .safetensors file.
        claimed_rank: Expected LoRA rank r, inferred from metadata if omitted.

    Returns:
        AdapterReport — versioned schema-stable result.  Check parse_status
        and analysis_mode before consuming findings or tensor_records.
    """
    from adaptersentry.schemas.adapter_report import (
        AdapterReport, AnalysisMode, ParseStatus, RiskSummary, ScanTarget, ToolInfo, TrainingStatus,
    )
    from adaptersentry.schemas.adapter_metadata import AdapterMetadata
    from adaptersentry.schemas.errors import ErrorCategory, ScanError
    from adaptersentry.schemas.finding import flags_to_findings, Severity
    from adaptersentry.schemas.inter_layer_similarity_features import InterLayerSimilarityFeatures
    from adaptersentry.schemas.tensor_record import TensorRecord
    from adaptersentry.version import __version__

    started_at = datetime.now(timezone.utc).isoformat()

    # ── File-level failure path ───────────────────────────────────────────────
    try:
        raw = _run_analysis(adapter_path, claimed_rank, fast=fast)
    except Exception as exc:
        completed_at = datetime.now(timezone.utc).isoformat()
        try:
            file_size: int | None = adapter_path.stat().st_size
        except OSError:
            file_size = None
        return AdapterReport(
            tool=ToolInfo(version=__version__),
            scan_target=ScanTarget(path=str(adapter_path.resolve()), file_size_bytes=file_size),
            adapter_metadata=AdapterMetadata.from_parsed({}),
            tensor_records=[],
            findings=[],
            errors=[ScanError.malformed(
                code="INVALID_SAFETENSORS",
                message=str(exc),
                detail=type(exc).__name__,
            )],
            risk_summary=RiskSummary(
                overall_risk=0,
                risk_level=Severity.LOW,
                ensemble_score=0.0,
                ensemble_risk_level=Severity.LOW,
                training_status=TrainingStatus.UNKNOWN,
                n_layers=0,
                n_findings=0,
                cross_layer_consistency=1.0,
            ),
            analysis_mode=AnalysisMode.FAILED,
            parse_status=ParseStatus.FAILED,
            started_at=started_at,
            completed_at=completed_at,
        )

    completed_at = datetime.now(timezone.utc).isoformat()

    # Parse metadata
    parsed_meta = parse_adapter_metadata(raw.get("metadata", {}))
    adapter_meta = AdapterMetadata.from_parsed(parsed_meta)

    # Tensor records
    tensor_records = [
        TensorRecord.from_layer_dict(lname, ldata)
        for lname, ldata in raw.get("layers", {}).items()
    ]

    # Findings
    findings = flags_to_findings(raw.get("flags", []), raw.get("layers", {}))

    # Errors — promote degraded layer flags to ScanError entries
    errors: list[ScanError] = []
    for tr in tensor_records:
        if tr.parse_error is not None:
            errors.append(ScanError.degraded(
                code="PARTIAL_LAYER_ANALYSIS",
                message=f"Layer {tr.layer_name!r} parse error: {tr.parse_error.value}",
                detail=next(
                    (f for f in tr.flags if f.startswith("DEGRADED_LAYER")), None
                ),
            ))

    # Derive parse_status from tensor records
    if any(tr.parse_error == ErrorCategory.MALFORMED for tr in tensor_records):
        parse_status = ParseStatus.DEGRADED
    elif any(tr.parse_error is not None for tr in tensor_records):
        parse_status = ParseStatus.DEGRADED
    else:
        parse_status = ParseStatus.OK

    # Risk summary
    ts_raw = raw.get("training_status", "TRAINED")
    try:
        training_status = TrainingStatus(ts_raw)
    except ValueError:
        training_status = TrainingStatus.UNKNOWN

    w2_distances = raw.get("wasserstein_distances", {})
    w2_mean = w2_distances.get("_mean")

    risk_summary = RiskSummary(
        overall_risk=raw["overall_risk"],
        risk_level=Severity(raw["risk_level"]),
        ensemble_score=raw["ensemble_score"],
        ensemble_risk_level=Severity(raw["ensemble_risk_level"]),
        training_status=training_status,
        false_positive_suppressed=raw.get("false_positive_suppressed", 0),
        n_layers=len(raw.get("layers", {})),
        n_findings=len(findings),
        cross_layer_consistency=raw.get("cross_layer_consistency", 1.0),
        wasserstein_mean=w2_mean if isinstance(w2_mean, float) else None,
    )

    il_dict = raw.get("inter_layer_similarity_features")
    try:
        il_feats = InterLayerSimilarityFeatures(**il_dict) if il_dict else None
    except Exception:
        il_feats = None

    try:
        file_size = adapter_path.stat().st_size
    except OSError:
        file_size = None

    return AdapterReport(
        tool=ToolInfo(version=__version__),
        scan_target=ScanTarget(
            path=str(adapter_path.resolve()),
            file_size_bytes=file_size,
        ),
        adapter_metadata=adapter_meta,
        tensor_records=tensor_records,
        findings=findings,
        errors=errors,
        risk_summary=risk_summary,
        analysis_mode=AnalysisMode.FULL if not errors else AnalysisMode.DEGRADED,
        parse_status=parse_status,
        started_at=started_at,
        completed_at=completed_at,
        inter_layer_similarity_features=il_feats,
    )


# ---------------------------------------------------------------------------
# Legacy CLI (kept for backward compatibility — new CLI is `adaptersentry scan`)
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptersentry-m1",
        description="AdapterSentry M1 — Static LoRA adapter analyzer",
    )
    parser.add_argument(
        "--adapter", required=True, type=Path, metavar="FILE",
        help="Path to .safetensors adapter file",
    )
    parser.add_argument(
        "--output", type=Path, default=None, metavar="FILE",
        help="Write JSON report to this file (default: stdout)",
    )
    parser.add_argument(
        "--rank", type=int, default=None, metavar="R",
        help="Expected LoRA rank r from adapter config (overrides metadata)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    return parser


def main() -> None:
    """CLI entry point for the legacy adaptersentry-m1 command."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        report = analyze(args.adapter, claimed_rank=args.rank)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    json_out = json.dumps(report, indent=2)

    if args.output:
        args.output.write_text(json_out)
        logger.info("Report written to %s", args.output)
    else:
        try:
            from rich.console import Console
            from rich.syntax import Syntax
            Console().print(Syntax(json_out, "json", theme="monokai"))
        except ImportError:
            sys.stdout.write(json_out + "\n")


if __name__ == "__main__":
    main()
