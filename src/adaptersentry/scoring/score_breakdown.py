"""compute_score_breakdown — decompose ensemble risk into per-family sub-scores.

Takes a completed AdapterReport and returns a ScoreBreakdown with one SubScore
per feature family. Sub-scores are computed from already-extracted features
(TensorRecord fields) — no re-analysis, no I/O.

Family weights sum to 1.0. Each family produces a normalized_score in [0, 1]
and a list of top_reasons explaining the dominant signals.

Security Notes:
    Pure computation on already-validated Pydantic models; no I/O.
    Does not feed sub-scores back into risk_score or confidence axes
    (circular-logic guard per M1 contract).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from adaptersentry.schemas.adapter_report import AdapterReport
    from adaptersentry.schemas.score_breakdown import SubScore
    from adaptersentry.schemas.scoring_policy import EscalationCondition, ScoringPolicy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Family weights — must sum to 1.0
# ---------------------------------------------------------------------------

_FAMILY_WEIGHTS: dict[str, float] = {
    "parse":            0.10,
    "metadata":         0.10,
    "norm":             0.10,
    "distribution":     0.30,
    "entropy":          0.10,
    "similarity":       0.20,
    "training_pattern": 0.10,
}

assert abs(sum(_FAMILY_WEIGHTS.values()) - 1.0) < 1e-9, "family weights must sum to 1.0"

# ---------------------------------------------------------------------------
# Per-family scoring helpers
# ---------------------------------------------------------------------------


def _score_parse(report: "AdapterReport") -> tuple[float, list[str]]:
    from adaptersentry.schemas.adapter_report import ParseStatus
    from adaptersentry.schemas.errors import ErrorCategory

    reasons: list[str] = []
    raw = 0.0

    if report.parse_status == ParseStatus.FAILED:
        raw += 0.8
        reasons.append("parse_status=FAILED — unrecoverable file-level failure")

    degraded = [tr for tr in report.tensor_records if tr.parse_error is not None]
    malformed = [tr for tr in degraded if tr.parse_error == ErrorCategory.MALFORMED]

    if degraded:
        frac = len(degraded) / max(len(report.tensor_records), 1)
        raw += float(np.clip(frac * 0.5, 0.0, 0.5))
        reasons.append(
            f"{len(degraded)}/{len(report.tensor_records)} layers with parse errors"
        )
    if malformed:
        raw += 0.2
        reasons.append(f"{len(malformed)} MALFORMED tensor(s)")

    return raw, reasons[:3]


def _score_metadata(report: "AdapterReport") -> tuple[float, list[str]]:
    meta = report.adapter_metadata
    reasons: list[str] = []
    raw = 0.0

    if not meta.metadata_present:
        raw += 0.5
        reasons.append("no adapter metadata — provenance unknown")

    if meta.metadata_present:
        raw_meta = meta.raw or {}
        depth = _dict_depth(raw_meta)
        if depth > 5:
            raw += 0.3
            reasons.append(f"metadata nesting depth={depth} > 5 (possible evasion)")

    if meta.metadata_present and not meta.base_model:
        raw += 0.15
        reasons.append("base_model field missing from metadata")

    if meta.metadata_present and not meta.peft_type:
        raw += 0.05
        reasons.append("peft_type field missing from metadata")

    return raw, reasons[:3]


def _dict_depth(d: object, current: int = 0) -> int:
    if not isinstance(d, dict) or not d:
        return current
    return max(_dict_depth(v, current + 1) for v in d.values())


def _score_norm(report: "AdapterReport") -> tuple[float, list[str]]:
    reasons: list[str] = []
    ratios: list[float] = []
    max_abs_vals: list[float] = []
    worst_layer = ""
    worst_ratio = 0.0

    for tr in report.tensor_records:
        if tr.norm_features is None:
            continue
        nf = tr.norm_features
        ratio = float(nf.delta_norm_ratio)
        ratios.append(ratio)
        max_abs_vals.append(float(nf.max_abs_delta))
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_layer = tr.layer_name

    if not ratios:
        return 0.0, []

    mean_ratio = float(np.mean(ratios))
    # Normal range: delta_norm_ratio around 1.0.  Values > 5 are suspicious.
    raw = float(np.clip((mean_ratio - 1.0) / 10.0, 0.0, 1.0))

    if worst_ratio > 3.0:
        short = worst_layer.split(".")[-1] if worst_layer else "?"
        reasons.append(
            f"delta_norm_ratio={worst_ratio:.2f} in …{short} (normal ≈1.0)"
        )

    mean_max_abs = float(np.mean(max_abs_vals)) if max_abs_vals else 0.0
    if mean_max_abs > 1.0:
        raw = float(np.clip(raw + 0.1, 0.0, 1.0))
        reasons.append(f"mean max_abs_delta={mean_max_abs:.3f} across layers")

    return raw, reasons[:3]


def _score_distribution(report: "AdapterReport") -> tuple[float, list[str]]:
    reasons: list[str] = []
    kurt_vals: list[float] = []
    skew_vals: list[float] = []
    zero_vals: list[float] = []
    worst_kurt = 0.0
    worst_kurt_layer = ""

    for tr in report.tensor_records:
        # Per-matrix kurtosis from TensorRecord flat fields
        k = max(float(tr.kurtosis_a), float(tr.kurtosis_b), 0.0)
        kurt_vals.append(float(np.tanh(k / 20.0)))
        if k > worst_kurt:
            worst_kurt = k
            worst_kurt_layer = tr.layer_name

        df = tr.distribution_features
        if df:
            skew_vals.append(float(abs(df.delta_skewness)))
            zero_vals.append(float(df.delta_zero_ratio))

    if not kurt_vals:
        return 0.0, []

    kurt_score = float(np.mean(kurt_vals))
    skew_score = float(np.clip(np.mean(skew_vals) / 5.0, 0.0, 1.0)) if skew_vals else 0.0
    zero_score = float(np.clip(np.mean(zero_vals) * 2.0, 0.0, 1.0)) if zero_vals else 0.0

    raw = float(np.clip(0.6 * kurt_score + 0.25 * skew_score + 0.15 * zero_score, 0.0, 1.0))

    if worst_kurt > 10.0:
        short = worst_kurt_layer.split(".")[-1] if worst_kurt_layer else "?"
        reasons.append(f"kurtosis={worst_kurt:.1f} in …{short} (threshold 10.0)")

    mean_skew = float(np.mean(skew_vals)) if skew_vals else 0.0
    if mean_skew > 2.0:
        reasons.append(f"mean |delta_skewness|={mean_skew:.2f} (benign < 2.0)")

    mean_zero = float(np.mean(zero_vals)) if zero_vals else 0.0
    if mean_zero > 0.3:
        reasons.append(f"mean zero_ratio={mean_zero:.2%} in ΔW (possible sparse pattern)")

    return raw, reasons[:3]


def _score_entropy(report: "AdapterReport") -> tuple[float, list[str]]:
    reasons: list[str] = []
    deviations: list[float] = []
    compression_scores: list[float] = []
    quant_scores: list[float] = []
    sign_entropy_scores: list[float] = []

    for tr in report.tensor_records:
        # Classic Shannon entropy deviation from benign band [0.10, 0.99]
        for e in (tr.entropy_a, tr.entropy_b):
            dev = max(0.0, 0.1 - e, e - 0.99)
            deviations.append(dev / 0.99)

        ec = tr.entropy_compression_features
        if ec:
            # Low byte_entropy → suspicious repetition in raw bytes
            for be in (ec.byte_entropy_a, ec.byte_entropy_b):
                if be < 0.7:
                    compression_scores.append(float(np.clip((0.7 - be) / 0.7, 0.0, 1.0)))

            # High compression ratio (> 1.0) or very low (< 0.7)
            for cr in (ec.approx_compression_ratio_a, ec.approx_compression_ratio_b):
                if cr < 0.7:
                    compression_scores.append(float(np.clip((0.7 - cr) / 0.7, 0.0, 1.0)))

            # High quantization suspect score
            for qs in (ec.quantization_suspect_score_a, ec.quantization_suspect_score_b):
                if qs > 0.7:
                    quant_scores.append(float(np.clip((qs - 0.7) / 0.3, 0.0, 1.0)))

            # Low sign entropy → all positive or all negative
            for se in (ec.sign_entropy_a, ec.sign_entropy_b):
                if se < 0.5:
                    sign_entropy_scores.append(float(np.clip((0.5 - se) / 0.5, 0.0, 1.0)))

    def _mean(lst: list[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    dev_score = _mean(deviations)
    comp_score = _mean(compression_scores)
    quant_score = _mean(quant_scores)
    sign_score = _mean(sign_entropy_scores)

    raw = float(np.clip(
        0.4 * dev_score + 0.3 * comp_score + 0.2 * quant_score + 0.1 * sign_score,
        0.0, 1.0,
    ))

    if dev_score > 0.05:
        reasons.append(f"entropy deviation from benign band (score={dev_score:.3f})")
    if comp_score > 0.1:
        reasons.append(f"low byte_entropy or compression anomaly (score={comp_score:.3f})")
    if quant_score > 0.1:
        reasons.append(f"quantization pattern detected (score={quant_score:.3f})")

    return raw, reasons[:3]


def _score_similarity(report: "AdapterReport") -> tuple[float, list[str]]:
    reasons: list[str] = []
    energy_vals: list[float] = []

    for tr in report.tensor_records:
        energy_vals.append(float(tr.energy_concentration))

    energy_score = float(np.mean(energy_vals)) if energy_vals else 0.0

    il_score = 0.0
    il = report.inter_layer_similarity_features
    if il is not None:
        # High mean cosine → non-random inter-layer similarity
        cos_anomaly = float(np.clip((il.cosine_sim_mean - 0.3) / 0.7, 0.0, 1.0))
        # Suspicious pairs add direct signal
        suspicious_signal = float(np.clip(il.n_suspicious_pairs / 5.0, 0.0, 1.0))
        il_score = float(0.6 * cos_anomaly + 0.4 * suspicious_signal)

        if il.n_suspicious_pairs > 0:
            reasons.append(
                f"{il.n_suspicious_pairs} non-adjacent layer pair(s) with cosine > 0.85"
            )
        if il.cosine_sim_mean > 0.3:
            reasons.append(
                f"inter-layer cosine_sim_mean={il.cosine_sim_mean:.3f} (benign < 0.30)"
            )

    if energy_score > 0.7:
        reasons.append(
            f"mean energy_concentration={energy_score:.3f} (low effective rank)"
        )

    raw = float(np.clip(0.5 * energy_score + 0.5 * il_score, 0.0, 1.0))
    return raw, reasons[:3]


def _score_training_pattern(report: "AdapterReport") -> tuple[float, list[str]]:
    from adaptersentry.schemas.adapter_report import TrainingStatus

    reasons: list[str] = []
    rs = report.risk_summary
    raw = 0.0

    # Low consistency = concentrated anomalies
    consistency_anomaly = float(np.clip(1.0 - rs.cross_layer_consistency, 0.0, 1.0))
    raw += 0.5 * consistency_anomaly
    if consistency_anomaly > 0.3:
        reasons.append(
            f"cross_layer_consistency={rs.cross_layer_consistency:.3f}"
            " (flag concentration detected)"
        )

    # High Wasserstein = distributional mismatch between A and B
    if rs.wasserstein_mean is not None and rs.wasserstein_mean > 0.1:
        w2_score = float(np.clip(rs.wasserstein_mean / 0.5, 0.0, 1.0))
        raw += 0.3 * w2_score
        reasons.append(
            f"wasserstein_mean={rs.wasserstein_mean:.4f} (A↔B distributional mismatch)"
        )

    if rs.training_status == TrainingStatus.PARTIALLY_TRAINED:
        raw += 0.2
        reasons.append("PARTIALLY_TRAINED: targeted-layer injection pattern")

    return float(np.clip(raw, 0.0, 1.0)), reasons[:3]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_FAMILY_ORDER = [
    "parse", "metadata", "norm", "distribution",
    "entropy", "similarity", "training_pattern",
]

_SCORERS = {
    "parse":            _score_parse,
    "metadata":         _score_metadata,
    "norm":             _score_norm,
    "distribution":     _score_distribution,
    "entropy":          _score_entropy,
    "similarity":       _score_similarity,
    "training_pattern": _score_training_pattern,
}

# ---------------------------------------------------------------------------
# Default scoring policy
# ---------------------------------------------------------------------------


def _build_default_policy() -> "ScoringPolicy":
    from adaptersentry.schemas.scoring_policy import (
        EscalationCondition, EscalationRule, ScoringPolicy,
    )
    return ScoringPolicy(
        version="1.0.0",
        family_weights=dict(_FAMILY_WEIGHTS),
        family_caps={f: 1.0 for f in _FAMILY_ORDER},
        family_floors={f: 0.0 for f in _FAMILY_ORDER},
        escalation_rules=[
            EscalationRule(
                name="PARTIALLY_TRAINED_HIGH_DISTRIBUTION",
                description=(
                    "Partial training combined with high distribution anomaly — "
                    "targeted backdoor injection pattern."
                ),
                conditions=[
                    EscalationCondition(family="training_pattern", operator="gt", threshold=0.25),
                    EscalationCondition(family="distribution", operator="gt", threshold=0.40),
                ],
                score_bump=0.12,
                reason="PARTIALLY_TRAINED + high kurtosis/skewness → targeted layer injection",
            ),
            EscalationRule(
                name="NO_METADATA_HIGH_RISK",
                description=(
                    "Missing adapter provenance combined with distribution anomaly — "
                    "unverifiable high-risk adapter."
                ),
                conditions=[
                    EscalationCondition(family="metadata", operator="gte", threshold=0.40),
                    EscalationCondition(family="distribution", operator="gt", threshold=0.30),
                ],
                score_bump=0.08,
                reason="Missing metadata + distribution anomaly → unverifiable high-risk adapter",
            ),
            EscalationRule(
                name="SUSPICIOUS_INTER_LAYER_SIMILARITY",
                description=(
                    "High inter-layer similarity indicates copied or replicated backdoor "
                    "weights across transformer layers."
                ),
                conditions=[
                    EscalationCondition(family="similarity", operator="gt", threshold=0.50),
                ],
                score_bump=0.10,
                reason="Inter-layer cosine similarity > 0.50 → replicated backdoor pattern",
            ),
            EscalationRule(
                name="HIGH_ENTROPY_ANOMALY_NO_METADATA",
                description="Entropy anomaly with no metadata — possible obfuscated payload.",
                conditions=[
                    EscalationCondition(family="entropy", operator="gt", threshold=0.40),
                    EscalationCondition(family="metadata", operator="gte", threshold=0.40),
                ],
                score_bump=0.06,
                reason="Entropy anomaly + missing metadata → possible obfuscated payload",
            ),
        ],
    )


# Lazy singleton — built on first use to avoid import-time cost
_DEFAULT_POLICY: "ScoringPolicy | None" = None


def get_default_policy() -> "ScoringPolicy":
    """Return the module-level default ScoringPolicy (built once, cached)."""
    global _DEFAULT_POLICY
    if _DEFAULT_POLICY is None:
        _DEFAULT_POLICY = _build_default_policy()
    return _DEFAULT_POLICY


# ---------------------------------------------------------------------------
# Policy application
# ---------------------------------------------------------------------------


def _eval_condition(cond: "EscalationCondition", ss_by_family: dict) -> bool:
    """Evaluate one EscalationCondition against a {family: SubScore} dict."""
    ss = ss_by_family.get(cond.family)
    if ss is None:
        return False
    val = ss.normalized_score
    return {
        "gt":  val >  cond.threshold,
        "gte": val >= cond.threshold,
        "lt":  val <  cond.threshold,
        "lte": val <= cond.threshold,
    }[cond.operator]


def apply_policy(
    sub_scores: list["SubScore"],
    policy: "ScoringPolicy",
) -> tuple[list["SubScore"], list[str], float]:
    """Apply a ScoringPolicy to a list of SubScores.

    Steps:
      1. Apply per-family caps and floors to normalized_score.
      2. Recompute weighted_contribution with (possibly overridden) weight.
      3. Evaluate escalation rules on the adjusted sub-scores.
      4. Sum score_bumps from fired rules.

    Args:
        sub_scores: Raw SubScores from compute_score_breakdown (before policy).
        policy:     ScoringPolicy to apply.

    Returns:
        (adjusted_sub_scores, fired_rule_names_with_reasons, total_bump)
    """
    from adaptersentry.schemas.score_breakdown import SubScore

    adjusted: list[SubScore] = []
    for s in sub_scores:
        weight = policy.family_weights.get(s.family, s.weight)
        cap   = policy.family_caps.get(s.family, 1.0)
        floor_ = policy.family_floors.get(s.family, 0.0)

        raw_norm = s.normalized_score
        capped   = min(cap, raw_norm)
        floored  = max(floor_, capped)
        final    = floored

        cap_applied   = (capped < raw_norm)
        floor_applied = (floored > capped)

        adjusted.append(SubScore(
            family=s.family,
            raw_score=s.raw_score,
            normalized_score=final,
            weight=weight,
            weighted_contribution=float(final * weight),
            top_reasons=s.top_reasons,
            cap_applied=cap_applied,
            floor_applied=floor_applied,
        ))

    ss_by_family = {s.family: s for s in adjusted}
    fired: list[str] = []
    total_bump = 0.0

    for rule in policy.escalation_rules:
        if all(_eval_condition(c, ss_by_family) for c in rule.conditions):
            fired.append(f"{rule.name}: {rule.reason}")
            total_bump += rule.score_bump

    return adjusted, fired, float(total_bump)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_score_breakdown(
    report: "AdapterReport",
    policy: "ScoringPolicy | None" = None,
) -> "ScoreBreakdown":
    """Compute ScoreBreakdown from a completed AdapterReport.

    Never raises — individual family failures produce 0.0 sub-scores.

    Args:
        report: Completed AdapterReport from analyzer.scan().
        policy: ScoringPolicy to apply. Defaults to DEFAULT_POLICY when None.

    Returns:
        ScoreBreakdown with one SubScore per feature family, with policy
        caps/floors and escalation rules applied.
    """
    from adaptersentry.schemas.score_breakdown import ScoreBreakdown, SubScore

    if policy is None:
        policy = get_default_policy()

    # Step 1: raw sub-scores (before policy)
    raw_sub_scores: list[SubScore] = []
    for family in _FAMILY_ORDER:
        weight = policy.family_weights.get(family, _FAMILY_WEIGHTS[family])
        try:
            raw, reasons = _SCORERS[family](report)
        except Exception as exc:
            logger.warning("ScoreBreakdown: %r scorer failed — %s", family, exc)
            raw, reasons = 0.0, [f"scorer error: {exc}"]

        normalized = float(np.clip(raw, 0.0, 1.0))
        raw_sub_scores.append(SubScore(
            family=family,
            raw_score=float(raw),
            normalized_score=normalized,
            weight=weight,
            weighted_contribution=float(normalized * weight),
            top_reasons=reasons,
        ))

    # Step 2: apply policy (caps, floors, escalation)
    adjusted_sub_scores, fired_rules, score_adj = apply_policy(raw_sub_scores, policy)

    total_weighted = float(np.clip(
        sum(s.weighted_contribution for s in adjusted_sub_scores), 0.0, 1.0
    ))
    adjusted_total = float(np.clip(total_weighted + score_adj, 0.0, 1.0))
    dominant = max(adjusted_sub_scores, key=lambda s: s.weighted_contribution)

    return ScoreBreakdown(
        sub_scores=adjusted_sub_scores,
        total_weighted=total_weighted,
        dominant_family=dominant.family,
        applied_policy_version=policy.version,
        escalation_rules_fired=fired_rules,
        score_adjustment=float(score_adj),
        adjusted_total_weighted=adjusted_total,
    )
