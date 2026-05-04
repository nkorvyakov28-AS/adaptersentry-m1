"""CombinedReport — M1 + M2 unified verdict (M2 fields are placeholder stubs).

M1-only scans populate the m1 field only. The m2 and behavioral_result fields
remain at their defaults (status='not_run'). Consumers MUST check
policy_gate.m2_triggered before reading behavioral_result.

This schema is intentionally forward-compatible: when M2 is implemented,
BehavioralResult.schema_version will be bumped from '0.1.0-placeholder' to
a real version, and the fields will be filled in by the M2 runner.

CombinedReport.final_verdict is the authoritative signal for enforcement:
  'allow'  — M1 low-risk AND (M2 not triggered OR M2 cleared)
  'review' — any MEDIUM signal or M2 inconclusive
  'block'  — M1 HIGH/CRITICAL OR M2 flagged
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from adaptersentry.engine.schemas.identity import AdapterArtifactIdentity
from adaptersentry.engine.schemas.scan_result import ScanResult


class BehavioralResult(BaseModel):
    """M2 behavioral sandbox result — placeholder until M2 is implemented."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: str = "0.1.0-placeholder"
    status: Literal["not_run", "completed", "failed"] = "not_run"
    sandbox_verdict: str | None = None
    probe_results: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class PolicyGateResult(BaseModel):
    """Result of the M1→M2 policy gate decision."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    m2_triggered: bool = False
    trigger_reason: str | None = None
    gate_policy: str = "default_v1"


class CombinedReport(BaseModel):
    """Unified report merging M1 static + M2 behavioral results.

    schema_version = "1.0.0".
    M1-only scans: m2 remains at BehavioralResult(status='not_run').
    final_verdict is derived from both M1 verdict and M2 result.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: str = "1.0.0"
    report_id: str = Field(description="sha256(m1.identity.scan_id + ':combined').")
    artifact: AdapterArtifactIdentity
    m1: ScanResult
    policy_gate: PolicyGateResult = Field(default_factory=PolicyGateResult)
    m2: BehavioralResult = Field(default_factory=BehavioralResult)
    final_verdict: Literal["allow", "review", "block"]
    generated_at: str = Field(description="ISO 8601 UTC.")

    @classmethod
    def from_m1_only(cls, m1: ScanResult, generated_at: str) -> "CombinedReport":
        """Construct a CombinedReport from an M1-only ScanResult."""
        import hashlib

        report_id = "sha256:" + hashlib.sha256(
            (m1.identity.scan_id + ":combined").encode()
        ).hexdigest()

        gate = PolicyGateResult(
            m2_triggered=m1.verdict.m2_recommended,
            trigger_reason=(
                f"verdict.m2_recommended=True (score={m1.verdict.overall_score})"
                if m1.verdict.m2_recommended else None
            ),
        )

        return cls(
            report_id=report_id,
            artifact=m1.artifact,
            m1=m1,
            policy_gate=gate,
            m2=BehavioralResult(),
            final_verdict=m1.verdict.recommended_action,
            generated_at=generated_at,
        )
