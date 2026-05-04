"""FeatureExtractor — typed per-layer feature extraction for the scan engine.

Provides two entry points:

extract_layer()
    The new typed pipeline path. Runs all feature families for one LoRA layer
    pair and returns (TensorRecord, list[FeatureFamilyResult], list[ScanError]).
    Used by worker_main() when calling the typed feature pipeline directly.

families_from_record()
    Migration bridge. Reconstructs typed FeatureFamilyResult objects from a
    TensorRecord that was already produced by the legacy analyzer._run_analysis()
    flat-dict path. Used during the transition period while the worker still
    calls analyzer.scan() internally.

Family schema versions (family_schema_version field on FeatureFamilyResult):
    norm        — 1.0.0
    distribution — 1.0.0
    entropy     — 1.0.0
    outlier     — 1.0.0
    spectral    — 1.0.0

Security Notes:
    - Pure computation; no I/O, no eval/exec/pickle.
    - Tensor inputs are untrusted; all extraction is guarded by try/except.
    - Per-family errors become ScanError(DEGRADED, FEATURE), not exceptions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from adaptersentry.engine.schemas.signals import FeatureFamilyResult
from adaptersentry.schemas.errors import ErrorCode, ScanError, ScanPhase
from adaptersentry.schemas.tensor_record import TensorRecord

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Compute typed FeatureFamilyResult objects for one LoRA layer pair.

    Usage — typed path (from raw tensors):
        extractor = FeatureExtractor()
        record, families, errors = extractor.extract_layer(
            "model.layers.0.self_attn.q_proj", tensor_A, tensor_B, claimed_rank=8
        )

    Usage — migration bridge (from existing TensorRecord):
        families = FeatureExtractor.families_from_record(record)
    """

    # -------------------------------------------------------------------------
    # Primary typed path
    # -------------------------------------------------------------------------

    def extract_layer(
        self,
        layer_name: str,
        tensor_A: np.ndarray,
        tensor_B: np.ndarray,
        claimed_rank: int | None = None,
        *,
        fast: bool = False,
    ) -> tuple[TensorRecord, list[FeatureFamilyResult], list[ScanError]]:
        """Extract all feature families for one LoRA pair.

        Never raises — all per-family errors become ScanError(DEGRADED, FEATURE)
        and a FeatureFamilyResult with status='failed'.

        Args:
            layer_name:   Canonical layer path (e.g. "model.layers.0.q_proj").
            tensor_A:     lora_A weight matrix.
            fast:         If True, use fast-mode optimisations (truncated SVD,
                          sampling, IsolationForest size guard).
            tensor_B:     lora_B weight matrix.
            claimed_rank: Expected LoRA rank from adapter config, if known.

        Returns:
            (TensorRecord, list[FeatureFamilyResult], list[ScanError])
        """
        from adaptersentry.detectors.entropy import compute_entropy
        from adaptersentry.detectors.outlier import detect_outlier_anomalies
        from adaptersentry.features.delta_norm import compute_norm_features
        from adaptersentry.features.distribution import compute_distribution_features
        from adaptersentry.features.tensor_stats import compute_svd_stats, compute_tensor_stats

        families: list[FeatureFamilyResult] = []
        errors: list[ScanError] = []

        # Accumulate into a layer dict compatible with TensorRecord.from_layer_dict()
        layer_dict: dict = {
            "shape_A": list(tensor_A.shape),
            "shape_B": list(tensor_B.shape),
            "flags": [],
        }

        # ── spectral ────────────────────────────────────────────────────────
        try:
            svd = compute_svd_stats(tensor_A, fast=fast)
            layer_dict["rank"] = svd["effective_rank"]
            layer_dict["energy_concentration"] = svd["energy_concentration"]
            families.append(FeatureFamilyResult(
                family="spectral",
                family_schema_version="1.0.0",
                layer=layer_name,
                status="ok",
                raw_features={
                    "effective_rank": float(svd["effective_rank"]),
                    "energy_concentration": float(svd["energy_concentration"]),
                    "rank_ratio": float(svd.get("rank_ratio", 0.0)),
                },
            ))
        except Exception as exc:
            logger.debug("Layer %r: SVD failed — %s", layer_name, exc)
            errors.append(ScanError.degraded(
                ErrorCode.SVD_FAILED,
                f"Layer {layer_name!r}: SVD failed",
                detail=str(exc),
                phase=ScanPhase.FEATURE,
            ))
            families.append(FeatureFamilyResult(
                family="spectral",
                family_schema_version="1.0.0",
                layer=layer_name,
                status="failed",
                error=str(exc),
            ))
            layer_dict.update({"rank": 0, "energy_concentration": 0.0})

        # ── distribution ─────────────────────────────────────────────────────
        try:
            stats_A = compute_tensor_stats(tensor_A, fast=fast)
            stats_B = compute_tensor_stats(tensor_B, fast=fast)
            layer_dict.update({
                "kurtosis_A": stats_A["kurtosis"],
                "kurtosis_B": stats_B["kurtosis"],
                "mean_A": stats_A["mean"],
                "std_A": stats_A["std"],
                "mean_B": stats_B["mean"],
                "std_B": stats_B["std"],
                "skewness_A": stats_A["skewness"],
            })
            dist_feats = compute_distribution_features(tensor_A, tensor_B, fast=fast)
            layer_dict["distribution_features"] = dist_feats.model_dump() if dist_feats else None

            raw: dict[str, float] = {
                # Per-matrix stats (A)
                "kurtosis_A": float(stats_A["kurtosis"]),
                "skewness_A": float(stats_A["skewness"]),
                "median_A": float(stats_A["median"]),
                "p01_A": float(stats_A["p01"]),
                "p99_A": float(stats_A["p99"]),
                "iqr_A": float(stats_A["iqr"]),
                "zero_ratio_A": float(stats_A["zero_ratio"]),
                # Per-matrix stats (B)
                "kurtosis_B": float(stats_B["kurtosis"]),
                "skewness_B": float(stats_B["skewness"]),
                "median_B": float(stats_B["median"]),
                "p01_B": float(stats_B["p01"]),
                "p99_B": float(stats_B["p99"]),
                "iqr_B": float(stats_B["iqr"]),
                "zero_ratio_B": float(stats_B["zero_ratio"]),
                # Entropy backfilled below by the entropy family
                "entropy_A": 0.0,
                "entropy_B": 0.0,
            }
            if dist_feats:
                raw.update({
                    "kurtosis_delta": float(dist_feats.delta_kurtosis),
                    "skewness_delta": float(dist_feats.delta_skewness),
                    "mean_delta": float(dist_feats.delta_mean),
                    "std_delta": float(dist_feats.delta_std),
                    "median_delta": float(dist_feats.delta_median),
                    "p01_delta": float(dist_feats.delta_p01),
                    "p99_delta": float(dist_feats.delta_p99),
                    "iqr_delta": float(dist_feats.delta_iqr),
                    "zero_ratio_delta": float(dist_feats.delta_zero_ratio),
                    "entropy_delta": float(dist_feats.delta_entropy),
                })
            families.append(FeatureFamilyResult(
                family="distribution",
                family_schema_version="1.0.0",
                layer=layer_name,
                status="ok",
                raw_features=raw,
            ))
        except Exception as exc:
            logger.debug("Layer %r: distribution analysis failed — %s", layer_name, exc)
            errors.append(ScanError.degraded(
                ErrorCode.PARTIAL_LAYER_ANALYSIS,
                f"Layer {layer_name!r}: distribution analysis failed",
                detail=str(exc),
                phase=ScanPhase.FEATURE,
            ))
            families.append(FeatureFamilyResult(
                family="distribution",
                family_schema_version="1.0.0",
                layer=layer_name,
                status="failed",
                error=str(exc),
            ))
            layer_dict.update({
                "kurtosis_A": 0.0, "kurtosis_B": 0.0,
                "mean_A": 0.0, "std_A": 0.0,
                "mean_B": 0.0, "std_B": 0.0,
                "skewness_A": 0.0,
                "distribution_features": None,
            })

        # ── entropy ──────────────────────────────────────────────────────────
        try:
            entropy_A = compute_entropy(tensor_A)
            entropy_B = compute_entropy(tensor_B)
            layer_dict.update({"entropy_A": entropy_A, "entropy_B": entropy_B})
            # Backfill entropy into the distribution family's raw_features
            for ffr in families:
                if ffr.family == "distribution" and ffr.status == "ok":
                    updated_raw = dict(ffr.raw_features)
                    updated_raw["entropy_A"] = float(entropy_A)
                    updated_raw["entropy_B"] = float(entropy_B)
                    # Replace the frozen FeatureFamilyResult with updated one
                    families[families.index(ffr)] = FeatureFamilyResult(
                        family=ffr.family,
                        family_schema_version=ffr.family_schema_version,
                        layer=ffr.layer,
                        status=ffr.status,
                        signals=list(ffr.signals),
                        raw_features=updated_raw,
                        error=ffr.error,
                    )
                    break
            families.append(FeatureFamilyResult(
                family="entropy",
                family_schema_version="1.0.0",
                layer=layer_name,
                status="ok",
                raw_features={
                    "entropy_A": float(entropy_A),
                    "entropy_B": float(entropy_B),
                },
            ))
        except Exception as exc:
            logger.debug("Layer %r: entropy analysis failed — %s", layer_name, exc)
            errors.append(ScanError.degraded(
                ErrorCode.PARTIAL_LAYER_ANALYSIS,
                f"Layer {layer_name!r}: entropy analysis failed",
                detail=str(exc),
                phase=ScanPhase.FEATURE,
            ))
            families.append(FeatureFamilyResult(
                family="entropy",
                family_schema_version="1.0.0",
                layer=layer_name,
                status="failed",
                error=str(exc),
            ))
            layer_dict.update({"entropy_A": 0.6, "entropy_B": 0.6})

        # ── entropy_compression ───────────────────────────────────────────────
        # O(n) in both fast and full — zlib (64KB cap), histogram byte_entropy.
        # Previously gated to full mode only; now runs in both because:
        # 1. M1 spec requires it in both modes (O(n), no sampling needed)
        # 2. OPT-04 Rust makes byte_entropy <0.1ms/layer (was 7ms)
        # 3. feature_completeness in compute_quality_score requires it non-None
        try:
            from adaptersentry.features.entropy_compression import (
                compute_entropy_compression_features,
            )
            ec_feats = compute_entropy_compression_features(tensor_A, tensor_B)
            layer_dict["entropy_compression_features"] = (
                ec_feats.model_dump() if ec_feats else None
            )
            if ec_feats:
                families.append(FeatureFamilyResult(
                    family="entropy_compression",
                    family_schema_version="1.0.0",
                    layer=layer_name,
                    status="ok",
                    raw_features={
                        "value_repeat_ratio_A": float(ec_feats.value_repeat_ratio_a),
                        "unique_value_ratio_A": float(ec_feats.unique_value_ratio_a),
                        "approx_compression_ratio_A": float(ec_feats.approx_compression_ratio_a),
                        "byte_entropy_A": float(ec_feats.byte_entropy_a),
                        "sign_entropy_A": float(ec_feats.sign_entropy_a),
                        "sign_balance_A": float(ec_feats.sign_balance_a),
                        "quantization_suspect_score_A": float(ec_feats.quantization_suspect_score_a),
                        "value_repeat_ratio_B": float(ec_feats.value_repeat_ratio_b),
                        "unique_value_ratio_B": float(ec_feats.unique_value_ratio_b),
                        "approx_compression_ratio_B": float(ec_feats.approx_compression_ratio_b),
                        "byte_entropy_B": float(ec_feats.byte_entropy_b),
                        "sign_entropy_B": float(ec_feats.sign_entropy_b),
                        "sign_balance_B": float(ec_feats.sign_balance_b),
                        "quantization_suspect_score_B": float(ec_feats.quantization_suspect_score_b),
                    },
                ))
            else:
                families.append(FeatureFamilyResult(
                    family="entropy_compression",
                    family_schema_version="1.0.0",
                    layer=layer_name,
                    status="skipped",
                ))
        except Exception as exc:
            logger.debug("Layer %r: entropy_compression analysis failed — %s", layer_name, exc)
            errors.append(ScanError.degraded(
                ErrorCode.PARTIAL_LAYER_ANALYSIS,
                f"Layer {layer_name!r}: entropy_compression analysis failed",
                detail=str(exc),
                phase=ScanPhase.FEATURE,
            ))
            families.append(FeatureFamilyResult(
                family="entropy_compression",
                family_schema_version="1.0.0",
                layer=layer_name,
                status="failed",
                error=str(exc),
            ))
            layer_dict["entropy_compression_features"] = None

        # ── outlier ──────────────────────────────────────────────────────────
        try:
            zs_A, iso_A, _ = detect_outlier_anomalies(tensor_A, layer_name, "A", fast=fast)
            zs_B, _, _ = detect_outlier_anomalies(
                tensor_B, layer_name, "B", run_isolation_forest=False, fast=fast
            )
            iso_score: float | None = iso_A.get("mean_score")
            layer_dict.update({
                "zscore_outlier_rate_A": float(zs_A["outlier_rate"]),
                "zscore_outlier_rate_B": float(zs_B["outlier_rate"]),
                "isolation_score_A": iso_score,
            })
            raw_outlier: dict[str, float] = {
                "zscore_outlier_rate_A": float(zs_A["outlier_rate"]),
                "zscore_outlier_rate_B": float(zs_B["outlier_rate"]),
            }
            if iso_score is not None:
                raw_outlier["isolation_score_A"] = float(iso_score)
            families.append(FeatureFamilyResult(
                family="outlier",
                family_schema_version="1.0.0",
                layer=layer_name,
                status="ok",
                raw_features=raw_outlier,
            ))
        except Exception as exc:
            logger.debug("Layer %r: outlier analysis failed — %s", layer_name, exc)
            errors.append(ScanError.degraded(
                ErrorCode.ISOLATION_FOREST_SKIPPED,
                f"Layer {layer_name!r}: outlier analysis failed",
                detail=str(exc),
                phase=ScanPhase.FEATURE,
            ))
            families.append(FeatureFamilyResult(
                family="outlier",
                family_schema_version="1.0.0",
                layer=layer_name,
                status="failed",
                error=str(exc),
            ))
            layer_dict.update({
                "zscore_outlier_rate_A": 0.0,
                "zscore_outlier_rate_B": 0.0,
                "isolation_score_A": None,
            })

        # ── norm ─────────────────────────────────────────────────────────────
        try:
            norm_feats = compute_norm_features(tensor_A, tensor_B, fast=fast)
            layer_dict["norm_features"] = norm_feats.model_dump() if norm_feats else None
            if norm_feats:
                families.append(FeatureFamilyResult(
                    family="norm",
                    family_schema_version="1.0.0",
                    layer=layer_name,
                    status="ok",
                    raw_features={
                        "fro_norm_delta": float(norm_feats.fro_norm_delta),
                        "max_abs_delta": float(norm_feats.max_abs_delta),
                        "mean_abs_delta": float(norm_feats.mean_abs_delta),
                        "delta_norm_ratio": float(norm_feats.delta_norm_ratio),
                    },
                ))
            else:
                families.append(FeatureFamilyResult(
                    family="norm",
                    family_schema_version="1.0.0",
                    layer=layer_name,
                    status="skipped",
                ))
        except Exception as exc:
            logger.debug("Layer %r: norm analysis failed — %s", layer_name, exc)
            errors.append(ScanError.degraded(
                ErrorCode.PARTIAL_LAYER_ANALYSIS,
                f"Layer {layer_name!r}: norm analysis failed",
                detail=str(exc),
                phase=ScanPhase.FEATURE,
            ))
            families.append(FeatureFamilyResult(
                family="norm",
                family_schema_version="1.0.0",
                layer=layer_name,
                status="failed",
                error=str(exc),
            ))
            layer_dict["norm_features"] = None

        record = TensorRecord.from_layer_dict(layer_name, layer_dict)
        return record, families, errors

    # -------------------------------------------------------------------------
    # Migration bridge — reconstructs families from an existing TensorRecord
    # -------------------------------------------------------------------------

    @staticmethod
    def families_from_record(record: TensorRecord) -> list[FeatureFamilyResult]:
        """Reconstruct typed FeatureFamilyResult objects from a TensorRecord.

        Used during the M1 transition when the worker obtains a TensorRecord
        from the legacy analyzer.scan() flat-dict path. The returned families
        are equivalent to what extract_layer() would produce for the same data.

        A degraded or failed record produces families with status='degraded'.
        """
        is_failed = record.parse_error is not None
        family_status = "degraded" if is_failed else "ok"

        families: list[FeatureFamilyResult] = []

        # spectral
        families.append(FeatureFamilyResult(
            family="spectral",
            family_schema_version="1.0.0",
            layer=record.layer_name,
            status=family_status,
            raw_features={
                "effective_rank": float(record.rank),
                "energy_concentration": float(record.energy_concentration),
                "rank_ratio": 0.0,
            },
        ))

        # distribution
        dist_raw: dict[str, float] = {
            "kurtosis_A": float(record.kurtosis_a),
            "kurtosis_B": float(record.kurtosis_b),
            "entropy_A": float(record.entropy_a),
            "entropy_B": float(record.entropy_b),
        }
        if record.distribution_features is not None:
            df = record.distribution_features
            dist_raw.update({
                "kurtosis_delta": float(df.delta_kurtosis),
                "skewness_delta": float(df.delta_skewness),
                "mean_delta": float(df.delta_mean),
                "std_delta": float(df.delta_std),
                "median_delta": float(df.delta_median),
                "p01_delta": float(df.delta_p01),
                "p99_delta": float(df.delta_p99),
                "iqr_delta": float(df.delta_iqr),
                "zero_ratio_delta": float(df.delta_zero_ratio),
                "entropy_delta": float(df.delta_entropy),
            })
        families.append(FeatureFamilyResult(
            family="distribution",
            family_schema_version="1.0.0",
            layer=record.layer_name,
            status=family_status,
            raw_features=dist_raw,
        ))

        # entropy
        families.append(FeatureFamilyResult(
            family="entropy",
            family_schema_version="1.0.0",
            layer=record.layer_name,
            status=family_status,
            raw_features={
                "entropy_A": float(record.entropy_a),
                "entropy_B": float(record.entropy_b),
            },
        ))

        # outlier
        outlier_raw: dict[str, float] = {
            "zscore_outlier_rate_A": float(record.zscore_outlier_rate_a),
            "zscore_outlier_rate_B": float(record.zscore_outlier_rate_b),
        }
        if record.isolation_score_a is not None:
            outlier_raw["isolation_score_A"] = float(record.isolation_score_a)
        families.append(FeatureFamilyResult(
            family="outlier",
            family_schema_version="1.0.0",
            layer=record.layer_name,
            status=family_status,
            raw_features=outlier_raw,
        ))

        # entropy_compression
        if record.entropy_compression_features is not None:
            ec = record.entropy_compression_features
            families.append(FeatureFamilyResult(
                family="entropy_compression",
                family_schema_version="1.0.0",
                layer=record.layer_name,
                status=family_status,
                raw_features={
                    "value_repeat_ratio_A": float(ec.value_repeat_ratio_a),
                    "unique_value_ratio_A": float(ec.unique_value_ratio_a),
                    "approx_compression_ratio_A": float(ec.approx_compression_ratio_a),
                    "byte_entropy_A": float(ec.byte_entropy_a),
                    "sign_entropy_A": float(ec.sign_entropy_a),
                    "sign_balance_A": float(ec.sign_balance_a),
                    "quantization_suspect_score_A": float(ec.quantization_suspect_score_a),
                    "value_repeat_ratio_B": float(ec.value_repeat_ratio_b),
                    "unique_value_ratio_B": float(ec.unique_value_ratio_b),
                    "approx_compression_ratio_B": float(ec.approx_compression_ratio_b),
                    "byte_entropy_B": float(ec.byte_entropy_b),
                    "sign_entropy_B": float(ec.sign_entropy_b),
                    "sign_balance_B": float(ec.sign_balance_b),
                    "quantization_suspect_score_B": float(ec.quantization_suspect_score_b),
                },
            ))
        else:
            families.append(FeatureFamilyResult(
                family="entropy_compression",
                family_schema_version="1.0.0",
                layer=record.layer_name,
                status="skipped",
            ))

        # norm
        if record.norm_features is not None:
            nf = record.norm_features
            families.append(FeatureFamilyResult(
                family="norm",
                family_schema_version="1.0.0",
                layer=record.layer_name,
                status=family_status,
                raw_features={
                    "fro_norm_delta": float(nf.fro_norm_delta),
                    "max_abs_delta": float(nf.max_abs_delta),
                    "mean_abs_delta": float(nf.mean_abs_delta),
                    "delta_norm_ratio": float(nf.delta_norm_ratio),
                },
            ))
        else:
            families.append(FeatureFamilyResult(
                family="norm",
                family_schema_version="1.0.0",
                layer=record.layer_name,
                status="skipped",
            ))

        return families
