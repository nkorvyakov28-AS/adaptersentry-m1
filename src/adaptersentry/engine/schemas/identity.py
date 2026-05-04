"""Artifact identity and scan identity schemas.

AdapterArtifactIdentity separates *logical identity* (what adapter this is,
stable across file moves) from *physical path* (where the file is right now).

ScanIdentity captures *which analyzer run* produced a result, enabling cache
correlation and reproducibility audits.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from adaptersentry.engine.schemas.requests import ArtifactSource


class AdapterArtifactIdentity(BaseModel):
    """Stable, content-addressed identity for a LoRA adapter artifact.

    logical_id
        For HF Hub sources: sha256(hf_repo_id + ':' + hf_revision + ':' + filename).
        For local files:    sha256(canonical_absolute_path).
        Does NOT change when the file is renamed if hf_repo_id is known.
        Intentionally NOT content-addressed — two files from the same HF repo
        revision are distinct logical adapters even if their bytes differ.

    content_hash
        sha256 of the entire file. Primary cache lookup key.
        Prefixed: 'sha256:<hex>'.

    header_hash
        sha256 of the safetensors header bytes only (tensor index + metadata dict,
        before tensor data). Changes when layout or metadata changes but tensor
        values are the same — useful for metadata-only invalidation diagnostics.
        Prefixed: 'sha256:<hex>'.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    logical_id: str = Field(description="Stable adapter identity — see module docstring.")
    content_hash: str = Field(description="sha256 of full file. Format: 'sha256:<hex>'.")
    header_hash: str = Field(
        description="sha256 of safetensors header bytes. Format: 'sha256:<hex>'."
    )
    file_size_bytes: int
    source: ArtifactSource
    resolved_at: str = Field(description="ISO 8601 UTC when hashes were computed.")


class ScanIdentity(BaseModel):
    """Identity of a specific scan execution.

    scan_id is deterministic: sha256(content_hash + ':' + analyzer_config_hash + ':' + schema_version).
    Given the same file and analyzer config, scan_id is identical across independent runs.
    Changing any detector weight, threshold, or tool version changes analyzer_config_hash
    and therefore scan_id — this is the desired behaviour.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    scan_id: str = Field(
        description=(
            "Deterministic scan ID: "
            "sha256(content_hash + ':' + analyzer_config_hash + ':' + schema_version)."
        )
    )
    run_id: str | None = Field(
        default=None,
        description="Batch run ID. Allows correlating this scan with a benchmark run.",
    )
    analyzer_version: str = Field(description="adaptersentry.__version__, e.g. '0.2.0'.")
    analyzer_config_hash: str = Field(
        description=(
            "sha256 of canonical JSON serialization of the active AnalyzerConfig: "
            "enabled families, detector weights, thresholds, schema_version. "
            "Any parameter change changes this hash, invalidating cached results."
        )
    )
    schema_version: str = "1.0.0"
    started_at: str = Field(description="ISO 8601 UTC.")
    completed_at: str = Field(description="ISO 8601 UTC.")
    wall_time_ms: int = Field(default=0, description="Wall-clock duration in milliseconds.")
