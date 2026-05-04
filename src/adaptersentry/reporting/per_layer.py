"""compute_per_layer_findings — rank layers by anomaly severity (M1-RPT-01).

Takes TensorRecord list (from AdapterReport) and returns ranked PerLayerFinding
objects. No re-analysis — uses already-computed flags and feature schemas.

Security Notes:
    Pure computation; no I/O, no eval/exec.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adaptersentry.schemas.tensor_record import TensorRecord
    from adaptersentry.schemas.inter_layer_similarity_features import InterLayerSimilarityFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flag-prefix → feature family mapping
# ---------------------------------------------------------------------------

_FLAG_TO_FAMILY: dict[str, str] = {
    "HIGH_KURTOSIS":              "distribution",
    "LOW_ENTROPY":                "entropy",
    "HIGH_ENTROPY":               "entropy",
    "HIGH_ZSCORE_OUTLIER_RATE":   "outlier",
    "HIGH_ISOLATION_ANOMALY":     "outlier",
    "RANK_INFLATION":             "norm",
    "NEAR_ZERO_B_MATRIX":         "norm",
    "HIGH_ENERGY_CONCENTRATION":  "spectral",
    "HIGH_RISK_TARGET_MODULE":    "spectral",
    "DEGRADED_LAYER":             "parse",
    "SUSPICIOUS_LAYER_CLUSTER":   "training_pattern",
    "CROSS_LAYER_CONCENTRATION":  "training_pattern",
    "SUSPICIOUS_PARTIAL_TRAINING":"training_pattern",
    "METADATA_DEPTH":             "parse",
}

# Severity → numeric weight for scoring
_SEV_WEIGHTS = {"CRITICAL": 4.0, "HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}

# Normalisation denominator: 5 HIGH flags → score 1.0 (anything above caps at 1.0)
_SCORE_NORM = 15.0

# Severity thresholds from normalized score
_SEV_THRESHOLDS = (
    (0.55, "CRITICAL"),
    (0.30, "HIGH"),
    (0.12, "MEDIUM"),
    (0.0,  "LOW"),
)


def _flag_family(flag: str) -> str | None:
    for prefix, family in _FLAG_TO_FAMILY.items():
        if flag.startswith(prefix):
            return family
    return None


def _flag_severity_weight(flag: str) -> float:
    from adaptersentry.schemas.finding import _severity_for_rule, _extract_rule_id
    rule_id = _extract_rule_id(flag)
    sev = _severity_for_rule(rule_id)
    return _SEV_WEIGHTS.get(sev.value, 1.0)


def _flag_signal_text(flag: str) -> str:
    """Return a stable human-readable title for a flag string."""
    from adaptersentry.schemas.finding import _title_for_rule, _extract_rule_id
    rule_id = _extract_rule_id(flag)
    return _title_for_rule(rule_id)


def _flag_remediation(flag: str) -> str | None:
    from adaptersentry.schemas.finding import _remediation_for_rule, _extract_rule_id
    return _remediation_for_rule(_extract_rule_id(flag))


def _severity_from_score(score: float) -> "Severity":
    from adaptersentry.schemas.finding import Severity
    for threshold, label in _SEV_THRESHOLDS:
        if score >= threshold:
            return Severity(label)
    return Severity.LOW


def _score_record(record: "TensorRecord") -> float:
    """Compute a normalized [0, 1] severity score for one TensorRecord."""
    raw = 0.0
    if record.parse_error is not None:
        raw += 2.0
    for flag in record.flags:
        raw += _flag_severity_weight(flag)
    return float(min(raw / _SCORE_NORM, 1.0))


def compute_per_layer_findings(
    tensor_records: list["TensorRecord"],
    inter_layer_features: "InterLayerSimilarityFeatures | None" = None,
    top_k: int = 10,
) -> list["PerLayerFinding"]:
    """Compute ranked PerLayerFinding list from TensorRecord data.

    Only layers with at least one anomaly signal (flag or parse_error) are
    included. Results are sorted by severity_score descending, then by
    flag_count descending for ties.

    Args:
        tensor_records:      Per-layer records from AdapterReport.tensor_records.
        inter_layer_features: Adapter-level inter-layer similarity (M1-ANAL-03).
                              Used to flag layers that appear in suspicious pairs.
        top_k:               Maximum results to return (default 10).

    Returns:
        List of PerLayerFinding, ranked 1..N (N ≤ top_k).
    """
    from adaptersentry.schemas.per_layer_finding import PerLayerFinding

    # Collect names of layers appearing in suspicious inter-layer pairs
    suspicious_il_layers: set[str] = set()
    if inter_layer_features is not None:
        for pair in inter_layer_features.top_suspicious_pairs:
            suspicious_il_layers.add(pair.layer_a)
            suspicious_il_layers.add(pair.layer_b)

    candidates: list[tuple[float, int, "TensorRecord"]] = []

    for record in tensor_records:
        score = _score_record(record)

        # Bonus for inter-layer suspicion (no overlap with existing flag signals)
        if record.layer_name in suspicious_il_layers:
            score = float(min(score + 0.15, 1.0))

        flag_count = len(record.flags) + (1 if record.parse_error is not None else 0)

        if score > 0.0 or flag_count > 0:
            candidates.append((score, flag_count, record))

    # Sort: score desc, then flag_count desc, then layer_name asc for stability
    candidates.sort(key=lambda t: (-t[0], -t[1], t[2].layer_name))
    top = candidates[:top_k]

    results: list[PerLayerFinding] = []
    for rank_idx, (score, flag_count, record) in enumerate(top, start=1):
        # Triggered families
        families: set[str] = set()
        if record.parse_error is not None:
            families.add("parse")
        if record.layer_name in suspicious_il_layers:
            families.add("inter_layer")
        for flag in record.flags:
            fam = _flag_family(flag)
            if fam:
                families.add(fam)

        # Stable signal texts (deduplicated, max 5)
        seen_titles: set[str] = set()
        signals: list[str] = []
        if record.parse_error is not None:
            signals.append(f"Parse error: {record.parse_error.value}")
            seen_titles.add("parse_error")

        # Sort flags by severity weight descending so most severe appear first
        sorted_flags = sorted(record.flags, key=_flag_severity_weight, reverse=True)
        for flag in sorted_flags:
            title = _flag_signal_text(flag)
            if title not in seen_titles and len(signals) < 5:
                signals.append(title)
                seen_titles.add(title)

        if record.layer_name in suspicious_il_layers and len(signals) < 5:
            signals.append("Similar ΔW to non-adjacent layer (inter-layer similarity)")

        # Remediation — use the most severe flag's remediation
        remediation: str | None = None
        for flag in sorted_flags:
            rem = _flag_remediation(flag)
            if rem:
                remediation = rem
                break

        results.append(PerLayerFinding(
            rank=rank_idx,
            layer_name=record.layer_name,
            severity=_severity_from_score(score),
            severity_score=score,
            triggered_families=sorted(families),
            flag_count=flag_count,
            signals=signals,
            remediation_hint=remediation,
        ))

    return results
