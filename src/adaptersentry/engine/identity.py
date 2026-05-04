"""ArtifactIdentityResolver — content-addressed identity for adapter files.

Computes two hashes for each adapter:
  content_hash — SHA-256 of the full file bytes (primary cache key)
  header_hash  — SHA-256 of the safetensors header only (diagnostic; not the cache key)

Both are prefixed: 'sha256:<hex>'.

The logical_id is derived from the source:
  HF Hub: sha256(hf_repo_id + ':' + hf_revision + ':' + filename)
  local:  sha256(canonical_absolute_path)

Trust boundary: this module reads raw file bytes. All paths must be resolved
via pathlib.Path.resolve() before being passed here. Never pass user-supplied
strings directly.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from adaptersentry.engine.schemas.identity import AdapterArtifactIdentity
from adaptersentry.engine.schemas.requests import ArtifactSource

logger = logging.getLogger(__name__)

# Stream-hash in 4 MB chunks to avoid loading large adapters into memory at once
_CHUNK_SIZE = 4 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of an entire file using streaming reads."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _sha256_str(s: str) -> str:
    return f"sha256:{hashlib.sha256(s.encode()).hexdigest()}"


def _read_safetensors_header(path: Path) -> bytes:
    """Read only the safetensors header bytes (first 8 + N bytes).

    safetensors format: [8-byte LE uint64 header_length][header_length bytes JSON]
    We read only this prefix, never the tensor data.
    Returns the header bytes, or b'' on any parse error (header_hash degrades gracefully).
    """
    try:
        with path.open("rb") as f:
            length_bytes = f.read(8)
            if len(length_bytes) < 8:
                return b""
            header_length = int.from_bytes(length_bytes, "little")
            # Sanity guard: reject absurdly large headers (>256 MB)
            if header_length > 256 * 1024 * 1024:
                logger.warning(
                    "Safetensors header length %d exceeds 256 MB guard — "
                    "skipping header hash for %s",
                    header_length, path.name,
                )
                return b""
            header_bytes = f.read(header_length)
            return length_bytes + header_bytes
    except (OSError, ValueError) as exc:
        logger.debug("Could not read safetensors header for %s: %s", path.name, exc)
        return b""


def _derive_logical_id(path: Path, source: ArtifactSource) -> str:
    """Compute logical_id from source metadata.

    For HF Hub: stable across file copies; changes on repo/revision change.
    For local: stable across OS-level renames if the path doesn't change.
    """
    if source.kind == "hf_hub" and source.hf_repo_id:
        rev = source.hf_revision or "HEAD"
        filename = path.name
        key = f"{source.hf_repo_id}:{rev}:{filename}"
        return _sha256_str(key)
    return _sha256_str(str(path))


class ArtifactIdentityResolver:
    """Resolves the full identity of a single adapter file.

    Usage:
        identity = ArtifactIdentityResolver.resolve(path, source)
    """

    @staticmethod
    def resolve(path: Path, source: ArtifactSource) -> AdapterArtifactIdentity:
        """Compute all identity fields for the given adapter file.

        Reads the full file twice:
          1. Streaming pass for content_hash (4 MB chunks — avoids full RAM load)
          2. Header-only read for header_hash

        Args:
            path: Resolved absolute path to the adapter file.
            source: ArtifactSource describing where the file came from.

        Returns:
            AdapterArtifactIdentity with all fields populated.

        Raises:
            FileNotFoundError: If path does not exist.
            PermissionError: If path is not readable.
            OSError: For other I/O errors.
        """
        if not path.exists():
            raise FileNotFoundError(f"Adapter file not found: {path}")

        logical_id = _derive_logical_id(path, source)
        content_hash = _sha256_file(path)

        header_bytes = _read_safetensors_header(path)
        header_hash = _sha256_bytes(header_bytes) if header_bytes else _sha256_bytes(b"")

        file_size = path.stat().st_size
        resolved_at = datetime.now(timezone.utc).isoformat()

        logger.debug(
            "Resolved identity for %s: content=%s header=%s size=%d",
            path.name, content_hash[:20], header_hash[:20], file_size,
        )

        return AdapterArtifactIdentity(
            logical_id=logical_id,
            content_hash=content_hash,
            header_hash=header_hash,
            file_size_bytes=file_size,
            source=source,
            resolved_at=resolved_at,
        )
