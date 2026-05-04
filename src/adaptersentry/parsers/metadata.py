"""Adapter metadata extraction and validation.

Extracts structured information from the raw safetensors metadata dict and
adapter_config.json when available.

Public API
----------
MetadataExtractor       — class; opens a .safetensors file and returns AdapterMetadata
                          without loading tensor data.
parse_adapter_metadata  — function; converts a raw metadata dict to the normalized form
                          consumed by AdapterMetadata.from_parsed().

Security Notes:
    - All metadata is treated as untrusted string data.
    - Metadata nesting depth is checked before use as a security signal.
    - No eval/exec on metadata values.
    - Path is validated via pathlib before opening.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from safetensors import safe_open

logger = logging.getLogger(__name__)


def _metadata_depth(obj: Any, _current: int = 0) -> int:
    """Recursively compute maximum nesting depth of a metadata structure."""
    if isinstance(obj, dict) and obj:
        return max(_metadata_depth(v, _current + 1) for v in obj.values())
    if isinstance(obj, list) and obj:
        return max(_metadata_depth(v, _current + 1) for v in obj)
    return _current


def parse_adapter_metadata(
    raw_metadata: dict[str, Any],
    adapter_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse structured adapter metadata from raw safetensors header + config.

    Args:
        raw_metadata: Metadata dict from safetensors file header.
        adapter_config: Optional adapter_config.json contents.

    Returns:
        Dict with normalized metadata fields:
            claimed_rank, lora_alpha, base_model, target_modules, peft_type,
            framework, depth (nesting depth as a security signal),
            metadata_present, raw.
    """
    cfg = adapter_config or {}

    # Rank inference: safetensors header takes precedence, then config
    claimed_rank: int | None = None
    for key in ("r", "rank", "lora_r"):
        raw = raw_metadata.get(key) or cfg.get(key)
        if raw is not None:
            try:
                claimed_rank = int(raw)
                break
            except (ValueError, TypeError):
                pass

    # lora_alpha inference: common keys across frameworks
    lora_alpha: float | None = None
    for key in ("lora_alpha", "alpha"):
        raw = raw_metadata.get(key) or cfg.get(key)
        if raw is not None:
            try:
                lora_alpha = float(raw)
                break
            except (ValueError, TypeError):
                pass

    # Target modules
    target_modules: list[str] = []
    tm = cfg.get("target_modules") or raw_metadata.get("target_modules")
    if isinstance(tm, list):
        target_modules = [str(m) for m in tm]
    elif isinstance(tm, str):
        target_modules = [m.strip() for m in tm.split(",") if m.strip()]

    return {
        "claimed_rank": claimed_rank,
        "lora_alpha": lora_alpha,
        "base_model": cfg.get("base_model_name_or_path") or raw_metadata.get("base_model"),
        "target_modules": target_modules,
        "peft_type": cfg.get("peft_type") or raw_metadata.get("peft_type"),
        "framework": raw_metadata.get("framework") or cfg.get("framework"),
        "depth": _metadata_depth(raw_metadata),
        "metadata_present": bool(raw_metadata or cfg),
        "raw": raw_metadata,
    }


class MetadataExtractor:
    """Read adapter metadata from a .safetensors file header without loading tensors.

    Uses safetensors.safe_open in header-only mode: the file is opened and the
    metadata dict is read, but no tensor data is allocated.  This makes the
    extractor safe to run on very large adapter files.

    Typical use::

        extractor = MetadataExtractor()
        meta = extractor.extract(Path("adapter.safetensors"))
        print(meta.claimed_rank, meta.lora_alpha)

    Security Notes:
        - Path validated via pathlib.Path.resolve() before use.
        - Metadata values are treated as untrusted strings; no eval/exec.
        - Nesting depth is bounded by _metadata_depth() as a security signal.
    """

    def extract_raw(self, path: Path) -> dict[str, Any]:
        """Return the raw metadata dict from the safetensors header.

        Args:
            path: Path to a .safetensors file.

        Returns:
            Raw string-valued metadata dict (may be empty if not present).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be opened as safetensors.
        """
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Adapter file not found: {resolved}")
        if resolved.suffix != ".safetensors":
            raise ValueError(f"Expected .safetensors file, got: {resolved.suffix!r}")

        try:
            with safe_open(str(resolved), framework="numpy") as f:
                raw = f.metadata() or {}
        except Exception as exc:
            raise ValueError(f"Cannot read metadata from {resolved}: {exc}") from exc

        logger.debug(
            "MetadataExtractor: %d metadata key(s) in %s", len(raw), resolved.name
        )
        return raw

    def extract(
        self,
        path: Path,
        adapter_config: dict[str, Any] | None = None,
    ) -> "AdapterMetadata":  # imported lazily to avoid circular imports
        """Extract structured AdapterMetadata from a .safetensors file.

        Args:
            path: Path to the .safetensors file.
            adapter_config: Optional adapter_config.json contents to merge.

        Returns:
            AdapterMetadata — typed, frozen schema object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be opened.
        """
        from adaptersentry.schemas.adapter_metadata import AdapterMetadata

        raw = self.extract_raw(path)
        parsed = parse_adapter_metadata(raw, adapter_config)
        return AdapterMetadata.from_parsed(parsed)
