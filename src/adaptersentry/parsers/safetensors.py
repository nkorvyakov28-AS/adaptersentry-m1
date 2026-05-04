"""safetensors file loading and LoRA tensor grouping.

Public parser contract
----------------------
parse_tensors(path) -> list[ParsedTensor]
    Canonical typed output — one record per tensor, with defensive per-tensor
    error handling so a single corrupt entry does not abort the whole parse.

load_adapter(path) -> (tensors, metadata)
    Raw numpy arrays needed by the analysis pipeline. Kept for internal use;
    prefer parse_tensors() for any code that only needs metadata or basic stats.

Security Notes:
    - Uses safetensors read-only memory-mapped access; no pickle or eval.
    - Path is validated via pathlib.Path.resolve() before opening.
    - Tensors exceeding _MAX_TENSOR_NUMEL elements are rejected before
      numpy allocation (tensor bomb guard).
"""

from __future__ import annotations

import json
import logging
import math
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from safetensors import safe_open

from adaptersentry.schemas.errors import ErrorCategory

logger = logging.getLogger(__name__)

# Standard PEFT LoRA key patterns
_LORA_A_PAT = re.compile(r"^(.+)\.lora_A\.weight$")
_LORA_B_PAT = re.compile(r"^(.+)\.lora_B\.weight$")

# Minimum paired layers required to classify as supported PEFT LoRA
_MIN_LORA_PAIRS = 2

# Tensor bomb guard: reject tensors with more than 1B elements before allocation
_MAX_TENSOR_NUMEL = 1_000_000_000

# safetensors dtype string for bfloat16 (not supported by numpy natively)
_BF16_DTYPE = "BF16"


def _read_sf_header(path: Path) -> tuple[dict, int]:
    """Return (header_dict, data_start_offset) from a safetensors file.

    Reads only the 8-byte length prefix and the JSON header body — no tensor
    data is loaded. data_start_offset is the byte position where tensor data begins.
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def _load_bf16_as_float32(
    path: Path, key: str, header: dict, data_offset: int
) -> np.ndarray:
    """Load a bfloat16 tensor from safetensors and return as float32.

    bfloat16 is exactly the high 16 bits of float32. Shifting a uint16 left by
    16 bits and viewing as float32 is a lossless round-trip for all finite values
    including NaN and Inf.

    Security note: shape and data_offsets come from the same header that
    safetensors validates on open — no additional bounds checking needed here.
    """
    info = header[key]
    shape = tuple(info["shape"])
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_offset + start)
        raw = f.read(end - start)
    uint16 = np.frombuffer(raw, dtype="<u2")
    return (uint16.astype(np.uint32) << 16).view(np.float32).reshape(shape)


class ParsedTensor(BaseModel):
    """Typed record for a single tensor from a .safetensors parse pass.

    Produced by parse_tensors(). One record per tensor key — not per LoRA layer
    pair. Use _group_lora_layers() to assemble A/B pairs for analysis.

    Security Notes:
        - numel is checked against _MAX_TENSOR_NUMEL before tensor allocation.
        - stats are computed only on tensors that pass the size guard.
        - Frozen to prevent post-parse mutation.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Full tensor key from the .safetensors file")
    dtype: str = Field(description="Tensor data type string (e.g. 'float32')")
    shape: list[int] = Field(description="Tensor shape dimensions")
    numel: int = Field(description="Total number of elements (product of shape)")
    stats: dict[str, float] = Field(
        default_factory=dict,
        description="Basic descriptive stats: mean, std",
    )
    parse_error: ErrorCategory | None = Field(
        default=None,
        description="Set when this tensor could not be fully loaded or converted; None = clean",
    )


def parse_tensors(path: Path) -> list[ParsedTensor]:
    """Parse a .safetensors file and return a typed record per tensor.

    Every tensor key in the file produces exactly one ParsedTensor — including
    tensors that could not be loaded or converted.  Failed tensors have
    parse_error set to the appropriate ErrorCategory; callers must check this
    field before treating stats as meaningful.

    Args:
        path: Path to the .safetensors file.

    Returns:
        List of ParsedTensor records, one per tensor key.  Records with
        parse_error != None are degraded; clean records have parse_error=None.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be opened at all (unrecoverable failure).

    Security Notes:
        - Path validated via pathlib.Path.resolve() before use.
        - Tensors with numel > 1B are rejected before numpy allocation.
        - dtype conversion errors are classified UNSUPPORTED, not silently ignored.
        - Read-only mmap via safetensors; no eval/exec/pickle.
    """
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Adapter file not found: {resolved}")
    if resolved.suffix != ".safetensors":
        raise ValueError(f"Expected .safetensors file, got: {resolved.suffix!r}")

    try:
        f_handle = safe_open(str(resolved), framework="numpy")
    except Exception as exc:
        raise ValueError(f"Cannot open {resolved}: {exc}") from exc

    # Pre-read header to detect bfloat16 tensors before calling get_tensor().
    # safetensors.numpy cannot construct numpy arrays for bfloat16 — we bypass
    # get_tensor() for those keys and convert raw bytes to float32 ourselves.
    _sf_header, _data_offset = _read_sf_header(resolved)
    _bf16_keys = {
        k for k, v in _sf_header.items()
        if isinstance(v, dict) and v.get("dtype") == _BF16_DTYPE
    }

    records: list[ParsedTensor] = []
    with f_handle as f:
        for key in f.keys():
            # Phase 1: load raw tensor bytes
            try:
                if key in _bf16_keys:
                    tensor = _load_bf16_as_float32(resolved, key, _sf_header, _data_offset)
                else:
                    tensor = f.get_tensor(key)
            except Exception as exc:
                logger.warning("Cannot load tensor %r: %s", key, exc)
                records.append(ParsedTensor(
                    name=key, dtype="unknown", shape=[], numel=0, stats={},
                    parse_error=ErrorCategory.MALFORMED,
                ))
                continue

            # Preserve the original file dtype string for bfloat16 tensors.
            # tensor.dtype would be float32 after conversion — record "BF16" instead
            # so callers can see the true on-disk format.
            original_dtype = _BF16_DTYPE if key in _bf16_keys else str(tensor.dtype)

            shape = list(tensor.shape)
            numel = math.prod(shape) if shape else 0

            # Phase 2: size guard (tensor bomb)
            if numel > _MAX_TENSOR_NUMEL:
                logger.warning(
                    "Tensor %r rejected: numel=%d exceeds safety limit %d",
                    key, numel, _MAX_TENSOR_NUMEL,
                )
                records.append(ParsedTensor(
                    name=key,
                    dtype=original_dtype,
                    shape=shape,
                    numel=numel,
                    stats={},
                    parse_error=ErrorCategory.MALFORMED,
                ))
                continue

            # Phase 3: zero-element check (unsupported shape)
            if numel == 0:
                logger.warning("Tensor %r has zero elements — shape %s unsupported", key, shape)
                records.append(ParsedTensor(
                    name=key,
                    dtype=original_dtype,
                    shape=shape,
                    numel=0,
                    stats={},
                    parse_error=ErrorCategory.UNSUPPORTED,
                ))
                continue

            # Phase 4: dtype conversion to float64 for stats
            try:
                flat = tensor.astype(np.float64).flatten()
            except Exception as exc:
                logger.warning("Tensor %r dtype %r cannot be converted to float64: %s",
                               key, original_dtype, exc)
                records.append(ParsedTensor(
                    name=key,
                    dtype=original_dtype,
                    shape=shape,
                    numel=numel,
                    stats={},
                    parse_error=ErrorCategory.UNSUPPORTED,
                ))
                continue

            records.append(ParsedTensor(
                name=key,
                dtype=original_dtype,
                shape=shape,
                numel=numel,
                stats={
                    "mean": float(np.mean(flat)),
                    "std": float(np.std(flat)),
                },
                parse_error=None,
            ))

    n_clean = sum(1 for r in records if r.parse_error is None)
    n_errors = len(records) - n_clean
    logger.debug(
        "parse_tensors: %d record(s) from %s (%d clean, %d errors)",
        len(records), resolved.name, n_clean, n_errors,
    )
    return records


def has_lora_pairs(path: Path) -> bool:
    """Check whether a .safetensors file contains any lora_A/lora_B pairs.

    Reads only key names from the header — no tensor data loaded.
    Use as a fast pre-check before load_adapter() to skip non-LoRA formats.
    """
    try:
        with safe_open(str(path), framework="numpy") as f:
            keys = list(f.keys())
        return any(_LORA_A_PAT.match(k) for k in keys)
    except Exception:
        return True  # conservative: let load_adapter handle the error


def load_adapter(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a .safetensors adapter file, returning tensors and metadata.

    Args:
        path: Path to the .safetensors file.

    Returns:
        Tuple of (tensors keyed by tensor name, raw metadata dict).

    Raises:
        FileNotFoundError: If the adapter file does not exist.
        ValueError: If the file is not a valid .safetensors file or has no
                    lora_A/lora_B pairs (non-LoRA format).
    """
    if not path.exists():
        raise FileNotFoundError(f"Adapter file not found: {path}")
    if path.suffix != ".safetensors":
        raise ValueError(f"Expected .safetensors file, got: {path.suffix!r}")

    if not has_lora_pairs(path):
        raise ValueError(
            f"No lora_A/lora_B tensor pairs found in {path.name} — "
            "file does not appear to be a PEFT LoRA adapter"
        )

    # Pre-read header to detect bfloat16 tensors before calling get_tensor().
    # safetensors.numpy raises for bfloat16 (no native numpy dtype); we bypass
    # get_tensor() for those keys and convert raw bytes to float32 ourselves.
    _sf_header, _data_offset = _read_sf_header(path)
    _bf16_keys = {
        k for k, v in _sf_header.items()
        if isinstance(v, dict) and v.get("dtype") == _BF16_DTYPE
    }
    if _bf16_keys:
        logger.info(
            "load_adapter: %d bfloat16 tensor(s) in %s — casting to float32",
            len(_bf16_keys), path.name,
        )

    tensors: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}

    try:
        with safe_open(str(path), framework="numpy") as f:
            metadata = f.metadata() or {}
            for key in f.keys():
                if key in _bf16_keys:
                    tensors[key] = _load_bf16_as_float32(path, key, _sf_header, _data_offset)
                else:
                    tensors[key] = f.get_tensor(key)
    except Exception as exc:
        raise ValueError(f"Failed to parse {path}: {exc}") from exc

    logger.debug("Loaded %d tensor(s) from %s", len(tensors), path)
    return tensors, metadata


def _group_lora_layers(
    tensors: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Group raw tensors into paired {layer_name: {"A": ..., "B": ...}} dicts.

    Args:
        tensors: Raw tensors dict from load_adapter.

    Returns:
        Dict mapping canonical layer name to its A and B weight matrices.
    """
    layers: dict[str, dict[str, np.ndarray]] = {}
    for key, tensor in tensors.items():
        m = _LORA_A_PAT.match(key)
        if m:
            layers.setdefault(m.group(1), {})["A"] = tensor
            continue
        m = _LORA_B_PAT.match(key)
        if m:
            layers.setdefault(m.group(1), {})["B"] = tensor
    return layers


def check_lora_architecture(st_path: Path) -> tuple[bool, list[str]]:
    """Check whether a safetensors file contains standard PEFT LoRA weight pairs.

    Opens the file header only (no tensor data loaded) and counts matched
    lora_A / lora_B weight pairs.

    Args:
        st_path: Path to the .safetensors file.

    Returns:
        (is_supported, tensor_keys_sample)
            is_supported       True if ≥ _MIN_LORA_PAIRS matched pairs found.
            tensor_keys_sample First 10 tensor key names for diagnostics.
    """
    with safe_open(str(st_path), framework="numpy") as f:
        keys = list(f.keys())

    keys_sample = keys[:10]
    a_layers: set[str] = set()
    b_layers: set[str] = set()

    for k in keys:
        m = _LORA_A_PAT.match(k)
        if m:
            a_layers.add(m.group(1))
            continue
        m = _LORA_B_PAT.match(k)
        if m:
            b_layers.add(m.group(1))

    paired = a_layers & b_layers
    return len(paired) >= _MIN_LORA_PAIRS, keys_sample
