"""CombinedReport — M1 + M2 unified verdict.

schema_version = "1.0.0" — public stable contract.

M1-only scans populate the m1 field only; the m2 field defaults to
BehavioralResult(status='not_run'). Consumers MUST check
policy_gate.m2_triggered before reading behavioral signals from m2.

The BehavioralResult / ProbeResult shapes here are part of the public
contract. Computation lives in downstream M2 implementations; this schema
is a stable wire format that those implementations populate.

CombinedReport.final_verdict is the authoritative signal for enforcement:
  'allow'  — M1 low-risk AND (M2 not triggered OR M2 cleared)
  'review' — any MEDIUM signal or M2 inconclusive
  'block'  — M1 HIGH/CRITICAL OR M2 confirmed
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from adaptersentry.engine.schemas.identity import AdapterArtifactIdentity
from adaptersentry.engine.schemas.scan_result import ScanResult


ProbeVerdictLiteral = Literal["confirmed", "cleared", "inconclusive", "skipped", "error"]
BehavioralStatusLiteral = Literal["not_run", "completed", "failed", "skipped"]
BehavioralVerdictLiteral = Literal["confirmed", "cleared", "inconclusive", "skipped"]
SkipReasonLiteral = Literal[
    "BASE_MODEL_MISMATCH",
    "BASE_MODEL_UNKNOWN",
    "PROBE_SUITE_TIMEOUT",
    "ADAPTER_LOAD_FAILED",
]


class ProbeResult(BaseModel):
    """Per-probe outcome from the M2 behavioral sandbox.

    Schema is wire-stable; downstream implementations populate the metric
    fields. The `verdict` enum is the authoritative per-probe signal.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    probe_id: str
    probe_set_version: str = "v0.1"
    trigger_type: str
    verdict: ProbeVerdictLiteral

    trigger_confirmed: bool = False
    semantic_drift: float = 0.0
    kl_drift: float = 0.0
    string_match: bool = False
    refusal_bypass: bool = False
    severity_weight: float = 0.0

    base_output_hash: str | None = None
    patched_output_hash: str | None = None
    elapsed_ms: int = 0
    error: str | None = None


class BehavioralResult(BaseModel):
    """M2 behavioral sandbox result — wire-stable v1.0.0.

    Defaults to status='not_run' so M1-only scans can construct an empty
    instance. M2 implementations fill in the metric / probe fields.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: str = "1.0.0"
    status: BehavioralStatusLiteral = "not_run"

    behavioral_verdict: BehavioralVerdictLiteral | None = None
    trigger_confirmed: bool = False
    behavioral_score: float = 0.0
    semantic_drift_score: float = 0.0

    base_model_used: str | None = None
    base_model_sha: str | None = None
    probe_set_version: str | None = None

    n_probes_run: int = 0
    n_probes_confirmed: int = 0
    n_probes_skipped: int = 0

    skip_reason: SkipReasonLiteral | None = None
    probe_results: list[ProbeResult] = Field(default_factory=list)
    targeted_layers: list[str] = Field(default_factory=list)

    sandbox_verdict: str | None = None
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
