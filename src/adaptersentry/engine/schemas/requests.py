"""AdapterScanRequest — the unit of work fed into the worker pool.

Immutable once created. All paths must be resolved through pathlib.Path.resolve()
before entering this schema — adapter-controlled strings are untrusted input.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ArtifactSource(BaseModel):
    """Where the adapter came from — local filesystem, HF Hub, or URL."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    kind: Literal["local_path", "hf_hub", "url"]
    local_path: str | None = Field(
        default=None,
        description="Resolved absolute path. Populated for kind='local_path'.",
    )
    hf_repo_id: str | None = Field(
        default=None,
        description="'username/repo-name'. Populated for kind='hf_hub'.",
    )
    hf_revision: str | None = Field(
        default=None,
        description="Git commit SHA or tag. Populated for kind='hf_hub'.",
    )
    url: str | None = Field(
        default=None,
        description="Source URL. Populated for kind='url'.",
    )


class AdapterScanRequest(BaseModel):
    """Unit of work — one adapter, one scan attempt.

    request_id is deterministic: SHA256(canonical_adapter_path + ':' + run_id).
    This means retrying the same job in the same run produces the same request_id,
    enabling idempotent manifest updates.

    schema_version is a contract field consumed by consumers reading persisted requests.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: str = "1.0.0"
    request_id: str = Field(
        description=(
            "Deterministic ID: sha256(canonical_path + ':' + (run_id or '')). "
            "Stable across retries of the same job in the same run."
        )
    )
    run_id: str | None = Field(
        default=None,
        description="Batch run identifier. None for single-adapter scan.",
    )
    adapter_path: str = Field(
        description=(
            "Absolute, resolved path. "
            "Must be produced via pathlib.Path.resolve() — never accept raw user strings."
        )
    )
    source: ArtifactSource
    claimed_rank: int | None = None
    scan_mode: str = Field(
        default="full",
        description="Scan depth: 'full' (all detectors) or 'fast' (optimised for throughput).",
    )
    force_rescan: bool = False
    enabled_families: list[str] = Field(
        default_factory=lambda: ["norm", "distribution", "entropy", "outlier", "spectral"],
        description=(
            "Feature families to activate. 'inter_layer' runs automatically when ≥2 layers present."
        ),
    )
    submitted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp.",
    )
    retry_count: int = Field(default=0, ge=0, le=3)
