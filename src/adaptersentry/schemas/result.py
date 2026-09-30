"""ScanResult 2.0.0 — the public result contract of AdapterSentry M1.

One document per scanned adapter. Consumers (CI gates, SOC tooling, the
behavioural stage) read ``verdict``; everything else explains it.

Layout
------
scan        who scanned, when, with which version, mode and policy
artifact    which file (hashes, size) and where it came from (provenance)
adapter     what the adapter is (format, rank, scaling, modules, base family)
status      ok | degraded | failed
coverage    what was analysed and what was not, and why
verdict     action, level, reasons, m2_recommended, confidence
anomaly     what the verdict rests on: intra-adapter comparison, reference profile
findings    human-readable findings with exact locations
modules     per-module features (opt-in, null by default)
errors      technical parse/analysis errors

Invariants enforced on construction (fail-closed contract):
    - status != ok               => verdict.action != "allow"
    - policy == default and no reference profile => verdict.action != "block"
    - block without a reference profile => reason STRICT_POLICY_STRUCTURAL present
    - coverage counts are consistent

Security Notes:
    - Pure data model; no I/O. ``provenance.path`` holds only the file name
      unless the caller explicitly opts in to full paths.
    - Documents of another major schema version are rejected by
      ``load_scan_result`` rather than silently misread.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adaptersentry.schemas.errors import ScanError
from adaptersentry.schemas.severity import Severity

SCHEMA_VERSION = "2.0.0"

# Upper bound on coverage.not_analyzed entries; the rest are counted only.
MAX_LISTED_NOT_ANALYZED = 50

Action = Literal["allow", "review", "block"]
Policy = Literal["default", "strict"]


class _Model(BaseModel):
    # Frozen: results are facts, not mutable state. Unknown fields are ignored so a
    # 2.x reader tolerates fields added by later minor versions.
    model_config = ConfigDict(frozen=True, extra="ignore")


# ── scan ───────────────────────────────────────────────────────────────────────


class ScanInfo(_Model):
    """Who scanned, when, and how."""

    scan_id: str = Field(description="sha256(content_hash:config_hash:schema_version); deterministic.")
    run_id: str | None = Field(default=None, description="Batch run identifier, if any.")
    analyzer_version: str
    config_hash: str = Field(description="Hash of analysis settings, including mode and policy.")
    mode: Literal["fast", "full"]
    policy: Policy = Field(description="Verdict policy; 'strict' may block on structural red flags.")
    started_at: str = Field(description="ISO 8601 UTC.")
    completed_at: str = Field(description="ISO 8601 UTC.")
    wall_time_ms: int = Field(ge=0)


# ── artifact ───────────────────────────────────────────────────────────────────


class SiblingFile(_Model):
    """A file found next to the adapter that deserves attention."""

    name: str
    kind: Literal["code", "pickle", "archive", "other"]


class Provenance(_Model):
    """Where the file came from."""

    kind: Literal["local_path", "hf_hub"]
    path: str | None = Field(
        default=None,
        description="File name by default; the absolute path only when full paths are requested.",
    )
    hf_repo_id: str | None = None
    hf_revision: str | None = None
    config_file: str | None = Field(default=None, description="adapter_config.json found next to the file.")
    sibling_files: list[SiblingFile] = Field(default_factory=list)


class Artifact(_Model):
    """Identity of the scanned file."""

    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    header_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    logical_id: str
    file_size_bytes: int = Field(ge=0)
    provenance: Provenance


# ── adapter ────────────────────────────────────────────────────────────────────


class RankInfo(_Model):
    """LoRA rank measured from tensor shapes."""

    min: int = Field(ge=0)
    max: int = Field(ge=0)
    distinct: list[int] = Field(default_factory=list)


class MoEInfo(_Model):
    """Mixture-of-experts structure of the adapted model."""

    n_experts: int = Field(ge=0)
    router_adapted: bool = Field(description="A LoRA is applied to the expert router.")


class AdapterInfo(_Model):
    """What the adapter is, as declared and as measured."""

    format: Literal["peft_lora", "kohya_lora", "unknown"]
    peft_type: str | None = None
    base_model_declared: str | None = None
    base_family: str = Field(default="unknown", description="llama, qwen, gemma, mistral, … or unknown.")
    base_family_source: Literal["config", "shapes", "unknown"] = "unknown"
    n_model_layers: int | None = Field(default=None, ge=0)
    rank_declared: int | None = Field(default=None, ge=0)
    rank_actual: RankInfo | None = None
    lora_alpha: float | None = None
    scaling: Literal["alpha_over_r", "rslora", "pattern", "unknown"] = "unknown"
    init_method: str | None = None
    use_dora: bool = False
    target_modules_declared: list[str] = Field(default_factory=list)
    target_modules_actual: list[str] = Field(default_factory=list)
    modules_to_save: list[str] = Field(default_factory=list)
    trainable_token_indices: bool = False
    moe: MoEInfo | None = None
    training_state: Literal["trained", "init_only", "partial", "unknown"] = "unknown"
    metadata_present: bool = False
    config_present: bool = False


# ── status / coverage ──────────────────────────────────────────────────────────


class NotAnalyzed(_Model):
    """A tensor present in the file that was not analysed."""

    tensor_key: str
    reason: str
    category: Literal["malformed", "unsupported", "degraded"]


class Coverage(_Model):
    """What was analysed. Anything not analysed may carry the payload."""

    n_tensors_total: int = Field(ge=0)
    n_tensors_analyzed: int = Field(ge=0)
    n_modules: int = Field(ge=0)
    n_modules_analyzed: int = Field(ge=0)
    n_not_analyzed: int = Field(ge=0, description="Total count; the list below is capped.")
    not_analyzed: list[NotAnalyzed] = Field(default_factory=list)
    non_finite_modules: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "Coverage":
        if self.n_tensors_analyzed > self.n_tensors_total:
            raise ValueError("n_tensors_analyzed exceeds n_tensors_total")
        if self.n_modules_analyzed > self.n_modules:
            raise ValueError("n_modules_analyzed exceeds n_modules")
        if len(self.not_analyzed) > self.n_not_analyzed:
            raise ValueError("not_analyzed lists more entries than n_not_analyzed")
        if len(self.not_analyzed) > MAX_LISTED_NOT_ANALYZED:
            raise ValueError(f"not_analyzed is capped at {MAX_LISTED_NOT_ANALYZED} entries")
        return self


# ── verdict ────────────────────────────────────────────────────────────────────


class Reason(_Model):
    """Why the verdict is what it is."""

    code: str = Field(description="PARSE_FAILED, DEGRADED_PARSE, NO_REFERENCE_PROFILE, INTRA_OUTLIER, …")
    severity: Severity
    message: str


class Confidence(_Model):
    """How much to trust the verdict, and what limits it."""

    level: Literal["high", "medium", "low"]
    limiting_factors: list[str] = Field(default_factory=list)


class Verdict(_Model):
    """What automated enforcement should do. Read this field."""

    action: Action
    level: Severity
    reasons: list[Reason] = Field(default_factory=list)
    m2_recommended: bool = Field(description="Behavioural verification is recommended.")
    confidence: Confidence


# ── anomaly ────────────────────────────────────────────────────────────────────


class DepthSpike(_Model):
    """A layer whose value departs from the robust depth trend of its module family."""

    module_type: str
    layer: int = Field(ge=0)
    z: float


class IntraAnomaly(_Model):
    """Comparison of modules within this adapter (needs no reference profile)."""

    max_robust_z: float = Field(ge=0.0)
    n_outlier_modules: int = Field(ge=0)
    depth_spikes: list[DepthSpike] = Field(default_factory=list)
    energy_gini: float | None = Field(default=None, ge=0.0, le=1.0)
    logrms_dispersion: dict[str, float] = Field(default_factory=dict)
    attn_mlp_ratio: float | None = None


class ReferenceComparison(_Model):
    """Comparison against a calibrated profile of benign adapters."""

    profile_id: str
    profile_version: str
    stratum: str
    n_reference: int = Field(ge=1)
    distance: float = Field(ge=0.0)
    p_value: float = Field(gt=0.0, le=1.0)
    alpha_review: float = Field(gt=0.0, lt=1.0)
    alpha_block: float = Field(gt=0.0, lt=1.0)


class TokenConcentration(_Model):
    """How concentrated an update of lm_head / embed_tokens is on a few tokens."""

    module: str
    top_token_ids: list[int] = Field(default_factory=list)
    top_shares: list[float] = Field(default_factory=list)
    hoyer: float = Field(ge=0.0, le=1.0)
    gini: float = Field(ge=0.0, le=1.0)


class Contributor(_Model):
    """One of the strongest contributions to the anomaly assessment."""

    feature: str
    module_type: str
    layer: int | None = None
    head: int | None = None
    expert: int | None = None
    value: float
    z: float


class Anomaly(_Model):
    """What the verdict rests on."""

    intra: IntraAnomaly | None = None
    reference: ReferenceComparison | None = None
    token_concentration: list[TokenConcentration] = Field(default_factory=list)
    top_contributors: list[Contributor] = Field(default_factory=list, max_length=10)


# ── findings ───────────────────────────────────────────────────────────────────


class Location(_Model):
    """Where in the adapter a finding applies."""

    layer: int | None = None
    module: str | None = None
    head: int | None = None
    expert: int | None = None
    tensor_key: str | None = None


class Finding(_Model):
    """A human-readable finding."""

    rule_id: str
    severity: Severity
    title: str
    message: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    locations: list[Location] = Field(default_factory=list)
    remediation: str | None = None


# ── modules (opt-in detail) ────────────────────────────────────────────────────


class ModuleFeatures(_Model):
    """Exact features of the effective update ΔW = s·B·A of one module."""

    log_rms: float | None = None
    log_sigma1: float | None = None
    top1_e: float | None = None
    srank_r: float | None = None
    pr_r: float | None = None
    spec_h: float | None = None
    excess_conc: float | None = None
    kurt_dw: float | None = None
    row_hoyer: float | None = None
    row_gini: float | None = None
    row_top8: float | None = None
    col_hoyer: float | None = None


class HeadRecord(_Model):
    """Features of one attention head (opt-in)."""

    head: int = Field(ge=0)
    features: ModuleFeatures


class ModuleRecord(_Model):
    """One LoRA A/B pair."""

    layer: int | None = None
    module: str
    kind: Literal["attention", "mlp", "embedding", "lm_head", "router", "other"]
    expert: int | None = None
    shape_a: list[int]
    shape_b: list[int]
    rank: int = Field(ge=0)
    scale: float | None = None
    status: Literal["ok", "not_analyzed", "non_finite"]
    features: ModuleFeatures | None = None
    z: dict[str, float] = Field(default_factory=dict)
    heads: list[HeadRecord] | None = None


# ── ScanResult ─────────────────────────────────────────────────────────────────


class ScanResult(_Model):
    """AdapterSentry M1 result, schema 2.0.0."""

    schema_version: str = Field(default=SCHEMA_VERSION, pattern=r"^2\.\d+\.\d+$")
    scan: ScanInfo
    artifact: Artifact | None = Field(
        default=None, description="Null only when the file could not be read at all.",
    )
    adapter: AdapterInfo
    status: Literal["ok", "degraded", "failed"]
    coverage: Coverage
    verdict: Verdict
    anomaly: Anomaly = Field(default_factory=Anomaly)
    findings: list[Finding] = Field(default_factory=list)
    modules: list[ModuleRecord] | None = Field(
        default=None, description="Per-module detail; present only when requested.",
    )
    errors: list[ScanError] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fail_closed(self) -> "ScanResult":
        action = self.verdict.action
        if self.status != "ok" and action == "allow":
            raise ValueError(f"status={self.status!r} must not yield action 'allow' (fail-closed)")
        if action == "block" and self.anomaly.reference is None:
            if self.scan.policy == "default":
                raise ValueError("policy 'default' cannot block without a reference profile")
            if not any(r.code == "STRICT_POLICY_STRUCTURAL" for r in self.verdict.reasons):
                raise ValueError("block without a reference profile requires reason STRICT_POLICY_STRUCTURAL")
        return self


class UnsupportedSchemaVersion(ValueError):
    """Raised when a document of another major schema version is loaded."""


def load_scan_result(data: str | bytes | dict[str, Any]) -> ScanResult:
    """Parse a ScanResult document, rejecting other major schema versions.

    Args:
        data: JSON text/bytes or an already-decoded dict.

    Returns:
        The validated ScanResult.

    Raises:
        UnsupportedSchemaVersion: If ``schema_version`` is missing or not 2.x.
        pydantic.ValidationError: If the document violates the 2.0 contract.
    """
    doc = json.loads(data) if isinstance(data, (str, bytes)) else data
    if not isinstance(doc, dict):
        raise UnsupportedSchemaVersion("ScanResult document must be a JSON object")
    version = doc.get("schema_version")
    if not isinstance(version, str) or not version.startswith("2."):
        raise UnsupportedSchemaVersion(
            f"Unsupported ScanResult schema_version {version!r}; this reader supports 2.x. "
            "Re-scan the adapter with AdapterSentry >= 2.0 to get a 2.x result."
        )
    return ScanResult.model_validate(doc)
