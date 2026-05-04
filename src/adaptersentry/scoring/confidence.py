"""compute_quality_score / compute_confidence_score — M1-SCORE-03.

Circular-logic guard: these functions use ONLY data-quality and coverage
signals. No kurtosis, no entropy, no outlier rates, no energy_concentration,
no wasserstein, no cross_layer_consistency. Those belong to the risk path.

Security Notes:
    Pure computation on already-validated Pydantic models; no I/O.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from adaptersentry.schemas.adapter_report import AdapterReport

logger = logging.getLogger(__name__)

# Confidence formula weights — must sum to 1.0
_CONF_WEIGHTS = {
    "sample_size":   0.30,
    "quality":       0.25,
    "inter_family":  0.25,
    "scan_mode":     0.20,
}

# Quality formula weights — must sum to 1.0
_QUAL_WEIGHTS = {
    "parse_coverage":        0.40,
    "metadata_completeness": 0.25,
    "feature_completeness":  0.25,
    "degenerate":            0.10,
}

_SAMPLE_SIZE_SATURATION = 16    # n_layers ≥ this → sample_size_factor = 1.0
_CONFIDENCE_HIGH_THRESHOLD   = 0.75
_CONFIDENCE_MEDIUM_THRESHOLD = 0.45

# Expected metadata keys for metadata_completeness
_EXPECTED_METADATA_FIELDS = 4   # base_model, peft_type, target_modules, rank


def compute_quality_score(report: "AdapterReport") -> "AnalysisQualityScore":
    """Compute AnalysisQualityScore from a completed AdapterReport.

    Signals used (all data-quality, NOT anomaly features):
    - Parse error counts from TensorRecord.parse_error
    - Metadata field presence from AdapterMetadata
    - Feature schema presence (norm_features, distribution_features,
      entropy_compression_features are non-None)
    - Degenerate-tensor proxy (all per-matrix stats exactly 0.0 on valid layer)

    Never raises.
    """
    from adaptersentry.schemas.confidence_score import AnalysisQualityScore

    records = report.tensor_records
    n_total = len(records)

    if n_total == 0:
        return AnalysisQualityScore(
            n_layers_total=0,
            n_layers_parsed_ok=0,
            parse_coverage=0.0,
            metadata_completeness=_metadata_completeness(report),
            feature_completeness=0.0,
            degenerate_ratio=0.0,
            overall_quality=0.0,
        )

    ok_records = [tr for tr in records if tr.parse_error is None]
    n_ok = len(ok_records)
    parse_coverage = float(n_ok) / float(n_total)

    meta_completeness = _metadata_completeness(report)

    # feature_completeness: all three typed feature schemas non-None
    full = sum(
        1 for tr in ok_records
        if (tr.norm_features is not None
            and tr.distribution_features is not None
            and tr.entropy_compression_features is not None)
    )
    feature_completeness = float(full) / float(n_ok) if n_ok > 0 else 0.0

    # degenerate_ratio: proxy for all-zero / failed tensors that parsed OK but
    # produced degenerate per-matrix stats. Uses mean and std only — these are
    # data-quality signals (not anomaly features). Kurtosis is excluded here
    # (it lives on the risk-score axis) — circular-logic guard.
    degen = sum(
        1 for tr in ok_records
        if (tr.mean_a == 0.0 and tr.std_a == 0.0
            and tr.mean_b == 0.0 and tr.std_b == 0.0)
    )
    degenerate_ratio = float(degen) / float(n_ok) if n_ok > 0 else 0.0

    overall_quality = float(np.clip(
        _QUAL_WEIGHTS["parse_coverage"]        * parse_coverage
        + _QUAL_WEIGHTS["metadata_completeness"] * meta_completeness
        + _QUAL_WEIGHTS["feature_completeness"]  * feature_completeness
        + _QUAL_WEIGHTS["degenerate"]            * (1.0 - degenerate_ratio),
        0.0, 1.0,
    ))

    return AnalysisQualityScore(
        n_layers_total=n_total,
        n_layers_parsed_ok=n_ok,
        parse_coverage=parse_coverage,
        metadata_completeness=meta_completeness,
        feature_completeness=feature_completeness,
        degenerate_ratio=degenerate_ratio,
        overall_quality=overall_quality,
    )


def _metadata_completeness(report: "AdapterReport") -> float:
    meta = report.adapter_metadata
    if not meta.metadata_present:
        return 0.0
    score = 0.0
    if meta.base_model:
        score += 1.0
    if meta.peft_type:
        score += 1.0
    if meta.target_modules:
        score += 1.0
    # rank — check raw metadata
    raw = meta.raw or {}
    if any(raw.get(k) for k in ("r", "rank", "lora_r")):
        score += 1.0
    return float(score / _EXPECTED_METADATA_FIELDS)


def compute_confidence_score(
    report: "AdapterReport",
    quality: "AnalysisQualityScore",
) -> "ConfidenceScore":
    """Compute ConfidenceScore from AdapterReport and AnalysisQualityScore.

    Signals used (data-quality and coverage only):
    - n_layers (sample size)
    - n_families_successful (feature schema presence)
    - analysis_quality from compute_quality_score()
    - inter_family_agreement (feature completeness proxy)
    - analysis_mode (full vs. degraded — proxy for scan mode)

    Never raises.
    """
    from adaptersentry.schemas.adapter_report import AnalysisMode
    from adaptersentry.schemas.confidence_score import ConfidenceScore

    n_layers = quality.n_layers_parsed_ok

    # Sample size — saturates at _SAMPLE_SIZE_SATURATION layers
    sample_size_factor = float(np.clip(n_layers / _SAMPLE_SIZE_SATURATION, 0.0, 1.0))

    # Inter-family agreement — same as feature_completeness from quality
    inter_family_agreement = quality.feature_completeness

    # Scan mode factor from analysis_mode
    if report.analysis_mode == AnalysisMode.FULL:
        scan_mode_factor = 1.0
    elif report.analysis_mode == AnalysisMode.DEGRADED:
        scan_mode_factor = 0.65
    else:  # FAILED
        scan_mode_factor = 0.0

    # n_families_successful: count schemas that were computed for ≥1 layer
    records = report.tensor_records
    has_norm  = any(tr.norm_features is not None for tr in records)
    has_dist  = any(tr.distribution_features is not None for tr in records)
    has_ec    = any(tr.entropy_compression_features is not None for tr in records)
    has_il    = report.inter_layer_similarity_features is not None
    n_families_successful = sum([has_norm, has_dist, has_ec, has_il])

    # Overall confidence
    overall_confidence = float(np.clip(
        _CONF_WEIGHTS["sample_size"]  * sample_size_factor
        + _CONF_WEIGHTS["quality"]    * quality.overall_quality
        + _CONF_WEIGHTS["inter_family"] * inter_family_agreement
        + _CONF_WEIGHTS["scan_mode"]  * scan_mode_factor,
        0.0, 1.0,
    ))

    # Verdict certainty
    if overall_confidence >= _CONFIDENCE_HIGH_THRESHOLD:
        verdict_certainty = "high"
    elif overall_confidence >= _CONFIDENCE_MEDIUM_THRESHOLD:
        verdict_certainty = "medium"
    else:
        verdict_certainty = "low"

    # Limiting factors — human-readable reasons
    limiting_factors: list[str] = []
    if sample_size_factor < 0.5:
        limiting_factors.append(
            f"small adapter ({n_layers} layers analyzed — full confidence requires ≥8)"
        )
    if quality.parse_coverage < 0.8:
        limiting_factors.append(
            f"parse errors on {quality.n_layers_total - quality.n_layers_parsed_ok} "
            f"of {quality.n_layers_total} layers"
        )
    if quality.metadata_completeness < 0.5:
        limiting_factors.append(
            "incomplete adapter metadata — provenance unverified"
        )
    if inter_family_agreement < 0.7:
        limiting_factors.append(
            "some feature families did not compute on all layers"
        )
    if scan_mode_factor < 1.0:
        limiting_factors.append(
            "analysis mode degraded — some detectors failed or were skipped"
        )

    return ConfidenceScore(
        n_layers=n_layers,
        n_families_successful=n_families_successful,
        sample_size_factor=sample_size_factor,
        analysis_quality=quality.overall_quality,
        inter_family_agreement=inter_family_agreement,
        scan_mode_factor=scan_mode_factor,
        overall_confidence=overall_confidence,
        verdict_certainty=verdict_certainty,
        limiting_factors=limiting_factors,
    )
