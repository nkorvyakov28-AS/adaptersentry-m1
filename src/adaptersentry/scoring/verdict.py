"""Policy verdict derivation shared by the single-file CLI and the batch worker.

A scanner that answers "allow" for a file it could not fully parse is worse than
no scanner: an attacker only has to make our parser fail while the real loader
still applies the weights. Degraded or failed parsing is therefore treated as a
security signal and never yields ``allow`` (fail-closed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from adaptersentry.engine.schemas.signals import FeatureSignal
from adaptersentry.schemas.adapter_report import AnalysisMode, ParseStatus
from adaptersentry.schemas.finding import Severity

Action = Literal["allow", "review", "block"]

_ACTION_RANK: dict[str, int] = {"allow": 0, "review": 1, "block": 2}


@dataclass(frozen=True)
class VerdictDecision:
    """Outcome of verdict derivation for one adapter."""

    action: Action
    m2_recommended: bool
    policy_signals: list[FeatureSignal] = field(default_factory=list)


def action_for_level(level: Severity) -> Action:
    """Map an ensemble risk level to the baseline recommended action."""
    if level in (Severity.HIGH, Severity.CRITICAL):
        return "block"
    if level == Severity.MEDIUM:
        return "review"
    return "allow"


def _stricter(a: Action, b: Action) -> Action:
    return a if _ACTION_RANK[a] >= _ACTION_RANK[b] else b


def _policy_signal(name: str, severity: Severity, message: str) -> FeatureSignal:
    return FeatureSignal(
        name=name,
        family="policy",
        value=1.0,
        threshold=0.5,
        matrix="adapter",
        severity=severity,
        human_message=message,
    )


def derive_verdict(
    *,
    ensemble_level: Severity,
    parse_status: ParseStatus,
    analysis_mode: AnalysisMode,
    metadata_present: bool,
) -> VerdictDecision:
    """Derive the policy verdict for a scanned adapter.

    Args:
        ensemble_level: Risk level from the statistical ensemble.
        parse_status: File-level parse outcome.
        analysis_mode: Whether all analysis paths completed.
        metadata_present: Whether the safetensors header carried metadata.

    Returns:
        VerdictDecision. Failed or degraded parsing always escalates the action
        to at least ``review`` and recommends behavioral follow-up.
    """
    action = action_for_level(ensemble_level)
    signals: list[FeatureSignal] = []
    m2 = ensemble_level in (Severity.HIGH, Severity.CRITICAL)

    if parse_status == ParseStatus.FAILED or analysis_mode == AnalysisMode.FAILED:
        signals.append(_policy_signal(
            "PARSE_FAILED", Severity.MEDIUM,
            "Adapter could not be parsed; absence of findings is not evidence of safety",
        ))
        action = _stricter(action, "review")
        m2 = True
    elif parse_status == ParseStatus.DEGRADED or analysis_mode == AnalysisMode.DEGRADED:
        signals.append(_policy_signal(
            "DEGRADED_PARSE", Severity.MEDIUM,
            "Part of the adapter could not be analysed; unanalysed tensors may carry the payload",
        ))
        action = _stricter(action, "review")
        m2 = True

    if not metadata_present:
        signals.append(_policy_signal(
            "MISSING_METADATA", Severity.LOW,
            "safetensors header carries no metadata; adapter provenance unknown",
        ))
        m2 = True

    return VerdictDecision(action=action, m2_recommended=m2, policy_signals=signals)
