"""Verdict of a ScanResult 2.0.0 — one place, agreed rules.

    1. Reference profile present (stage 5):
         p ≤ alpha_block  → block / HIGH
         p ≤ alpha_review → review / MEDIUM
    2. No reference profile: intra-adapter comparison only; at most "review"
       under the default policy.
    3. Structural red flags → at least review + behavioural check;
       under policy "strict" → block with reason STRICT_POLICY_STRUCTURAL.
    4. Degraded or failed parsing → at least review (fail-closed).

The ScanResult model re-checks the fail-closed invariants on construction, so a
bug here cannot produce an "allow" for a degraded scan.
"""

from __future__ import annotations

from adaptersentry.detectors.intra_adapter import REVIEW_Z
from adaptersentry.schemas.severity import Severity
from adaptersentry.schemas.result import (
    Confidence,
    IntraAnomaly,
    Policy,
    Reason,
    ReferenceComparison,
    Verdict,
)

_RANK = {"allow": 0, "review": 1, "block": 2}
_LEVEL_FOR = {"allow": Severity.LOW, "review": Severity.MEDIUM, "block": Severity.HIGH}


def derive_verdict(
    *,
    status: str,
    policy: Policy,
    intra: IntraAnomaly | None,
    reference: ReferenceComparison | None,
    red_flags: list[str],
    metadata_present: bool,
    confidence_limits: list[str],
    n_not_analyzed: int,
) -> Verdict:
    """Derive action, level, reasons, behavioural recommendation and confidence."""
    action = "allow"
    reasons: list[Reason] = []

    def raise_to(target: str) -> None:
        nonlocal action
        if _RANK[target] > _RANK[action]:
            action = target

    if status == "failed":
        reasons.append(Reason(code="PARSE_FAILED", severity=Severity.MEDIUM,
                              message="Adapter could not be parsed; absence of findings is not evidence of safety"))
        raise_to("review")
    elif status == "degraded":
        reasons.append(Reason(code="DEGRADED_PARSE", severity=Severity.MEDIUM,
                              message=f"{n_not_analyzed} tensor(s) not analysed; the payload may live there"))
        raise_to("review")

    if reference is not None:
        if reference.p_value <= reference.alpha_block:
            reasons.append(Reason(code="REFERENCE_P_BELOW_ALPHA", severity=Severity.HIGH,
                                  message=f"p = {reference.p_value:.4g} ≤ {reference.alpha_block} "
                                          f"against {reference.n_reference} benign adapters"))
            raise_to("block")
        elif reference.p_value <= reference.alpha_review:
            reasons.append(Reason(code="REFERENCE_P_BELOW_ALPHA", severity=Severity.MEDIUM,
                                  message=f"p = {reference.p_value:.4g} ≤ {reference.alpha_review} "
                                          f"against {reference.n_reference} benign adapters"))
            raise_to("review")
    elif status != "failed":
        reasons.append(Reason(code="NO_REFERENCE_PROFILE", severity=Severity.LOW,
                              message="No calibrated reference for this model family; "
                                      "verdict relies on intra-adapter comparison"))

    if intra is not None and intra.max_robust_z >= REVIEW_Z:
        where = ""
        if intra.depth_spikes:
            top = intra.depth_spikes[0]
            where = f": {top.module_type} layer {top.layer}, robust z = {top.z:.1f}"
        reasons.append(Reason(code="INTRA_OUTLIER", severity=Severity.MEDIUM,
                              message=f"Module departs from its family{where} (threshold {REVIEW_Z}, uncalibrated)"))
        raise_to("review")

    for flag in red_flags:
        reasons.append(Reason(code=flag, severity=Severity.MEDIUM, message="Structural red flag"))
        raise_to("review")
    if red_flags and policy == "strict":
        reasons.append(Reason(code="STRICT_POLICY_STRUCTURAL", severity=Severity.HIGH,
                              message="Strict policy blocks on structural red flags: " + ", ".join(red_flags)))
        raise_to("block")

    if not metadata_present:
        reasons.append(Reason(code="MISSING_METADATA", severity=Severity.LOW,
                              message="safetensors header carries no metadata; provenance unknown"))

    level = _LEVEL_FOR[action]
    if action == "block" and reference is not None and reference.p_value <= reference.alpha_block / 10:
        level = Severity.CRITICAL

    limits = list(confidence_limits)
    if reference is None:
        limits.append("no reference profile for this model family")
    if intra is not None:
        limits.append("intra-adapter thresholds are not yet calibrated")
    elif status != "failed":
        limits.append("too few modules per family for intra-adapter comparison")
    if n_not_analyzed:
        limits.append(f"{n_not_analyzed} tensor(s) not analysed")

    if status != "ok":
        conf = "low"
    elif reference is None or confidence_limits:
        conf = "medium" if intra is not None else "low"
    else:
        conf = "high"

    return Verdict(
        action=action,  # type: ignore[arg-type]
        level=level,
        reasons=reasons,
        m2_recommended=action != "allow" or not metadata_present,
        confidence=Confidence(level=conf, limiting_factors=limits),  # type: ignore[arg-type]
    )
