"""Artifact identity: content/header hashes and logical identity of an adapter file.

AdapterArtifactIdentity separates *logical identity* (what adapter this is,
stable across file moves) from *physical path* (where the file is right now).
The per-scan identity lives in ScanResult.scan.
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
