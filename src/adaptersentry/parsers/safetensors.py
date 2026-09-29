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
    - Path is resolved, and must be a regular file within _MAX_FILE_BYTES,
      before anything is opened (no FIFOs, devices or symlinks to them).
    - The JSON header length is bounded and checked against the file size
      before it is read.
    - dtype, element count and LoRA rank are validated from the header
      before any tensor is allocated (tensor bomb guard).
    - A tensor that fails validation or loading is skipped and reported, so
      the caller can mark the scan DEGRADED instead of silently ignoring it.
"""

from __future__ import annotations

import json
import logging
import math
import re
import stat
import struct
from dataclasses import dataclass, field
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

# Upper bound on elements loaded from one file (all tensors together).
# load_adapter() materialises every LoRA tensor as float32, so this caps
# peak memory at ~12 GB. Streaming per-layer loading replaces this in 2.0.
_MAX_TOTAL_NUMEL = 3_000_000_000

# Same bound the safetensors library itself applies to the JSON header.
_MAX_HEADER_BYTES = 100_000_000

# Largest adapter file we are willing to open.
_MAX_FILE_BYTES = 64 * 1024**3

# LoRA rank above which a pair is not analysed (SVD cost grows with rank²).
_MAX_LORA_RANK = 1024

# safetensors dtype string for bfloat16 (not supported by numpy natively)
_BF16_DTYPE = "BF16"

# dtypes accepted for LoRA weights, with their byte width.
_ALLOWED_DTYPES: dict[str, int] = {"F32": 4, "F16": 2, "BF16": 2}


@dataclass(frozen=True)
class SkippedTensor:
    """A tensor present in the file that was not loaded, and why."""

    key: str
    reason: str
    category: ErrorCategory


@dataclass
class LoadedAdapter:
    """Result of load_adapter_checked(): loaded tensors plus what was skipped."""

    tensors: dict[str, np.ndarray]
    metadata: dict[str, Any]
    skipped: list[SkippedTensor] = field(default_factory=list)


def _check_regular_file(path: Path) -> Path:
    """Resolve *path* and require a regular .safetensors file of sane size.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If it is not a regular file, has the wrong suffix after
            symlink resolution, or exceeds _MAX_FILE_BYTES.
    """
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Adapter file not found: {resolved}")
    st = resolved.stat()
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"Not a regular file: {resolved}")
    if resolved.suffix != ".safetensors":
        raise ValueError(f"Expected .safetensors file, got: {resolved.suffix!r}")
    if st.st_size > _MAX_FILE_BYTES:
        raise ValueError(
            f"File too large: {st.st_size} bytes exceeds limit {_MAX_FILE_BYTES}"
        )
    return resolved


def _read_sf_header(path: Path) -> tuple[dict, int]:
    """Return (header_dict, data_start_offset) from a safetensors file.

    Reads only the 8-byte length prefix and the JSON header body — no tensor
    data is loaded. data_start_offset is the byte position where tensor data begins.

    Raises:
        ValueError: If the file is shorter than 8 bytes, the declared header
            length exceeds _MAX_HEADER_BYTES or the file itself, or the header
            is not a JSON object. The length is checked before any allocation.
    """
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) < 8:
            raise ValueError(f"File too short to be safetensors: {file_size} bytes")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len > _MAX_HEADER_BYTES:
            raise ValueError(
                f"safetensors header length {header_len} exceeds limit {_MAX_HEADER_BYTES}"
            )
        if 8 + header_len > file_size:
            raise ValueError(
                f"safetensors header length {header_len} exceeds file size {file_size}"
            )
        raw = f.read(header_len)
    try:
        header = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"safetensors header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    return header, 8 + header_len


def _header_entry_problem(info: Any, data_len: int) -> str | None:
    """Validate one tensor entry of a safetensors header.

    Returns:
        None if the entry is acceptable for loading, otherwise a reason string.
    """
    if not isinstance(info, dict):
        return "header entry is not an object"
    dtype = info.get("dtype")
    if dtype not in _ALLOWED_DTYPES:
        return f"unsupported dtype {dtype!r}"
    shape = info.get("shape")
    if not isinstance(shape, list) or not all(
        isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
    ):
        return f"invalid shape {shape!r}"
    numel = math.prod(shape) if shape else 1
    if numel > _MAX_TENSOR_NUMEL:
        return f"numel {numel} exceeds limit {_MAX_TENSOR_NUMEL}"
    offsets = info.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(o, int) and not isinstance(o, bool) for o in offsets)
    ):
        return f"invalid data_offsets {offsets!r}"
    start, end = offsets
    if not 0 <= start <= end <= data_len:
        return f"data_offsets {offsets!r} outside tensor data ({data_len} bytes)"
    if end - start != numel * _ALLOWED_DTYPES[dtype]:
        return f"byte length {end - start} does not match shape {shape} and dtype {dtype}"
    return None


def _rank_problem(key: str, shape: list[int]) -> str | None:
    """Return a reason if a LoRA tensor's rank dimension exceeds _MAX_LORA_RANK."""
    if len(shape) != 2:
        return None
    if _LORA_A_PAT.match(key) and shape[0] > _MAX_LORA_RANK:
        return f"LoRA rank {shape[0]} exceeds limit {_MAX_LORA_RANK}"
    if _LORA_B_PAT.match(key) and shape[1] > _MAX_LORA_RANK:
        return f"LoRA rank {shape[1]} exceeds limit {_MAX_LORA_RANK}"
    return None


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
    if len(raw) != end - start or len(raw) != 2 * math.prod(shape):
        raise ValueError(f"Truncated bfloat16 tensor {key!r}")
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
    resolved = _check_regular_file(path)

    # Validate the header (bounded length) before handing the file to safetensors.
    _sf_header, _data_offset = _read_sf_header(resolved)

    try:
        f_handle = safe_open(str(resolved), framework="numpy")
    except Exception as exc:
        raise ValueError(f"Cannot open {resolved}: {exc}") from exc

    # bfloat16 tensors bypass get_tensor(): safetensors.numpy cannot build them,
    # so raw bytes are converted to float32 ourselves.
    _bf16_keys = {
        k for k, v in _sf_header.items()
        if isinstance(v, dict) and v.get("dtype") == _BF16_DTYPE
    }

    records: list[ParsedTensor] = []
    with f_handle as f:
        for key in f.keys():
            # Phase 0: tensor bomb guard on the header-declared shape, before allocation
            info = _sf_header.get(key)
            declared = info.get("shape") if isinstance(info, dict) else None
            if isinstance(declared, list) and all(isinstance(d, int) for d in declared):
                declared_numel = math.prod(declared) if declared else 1
                if declared_numel > _MAX_TENSOR_NUMEL:
                    logger.warning(
                        "Tensor %r rejected before allocation: numel=%d exceeds limit %d",
                        key, declared_numel, _MAX_TENSOR_NUMEL,
                    )
                    records.append(ParsedTensor(
                        name=key,
                        dtype=str(info.get("dtype", "unknown")),
                        shape=declared,
                        numel=declared_numel,
                        stats={},
                        parse_error=ErrorCategory.MALFORMED,
                    ))
                    continue

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
        # Fail closed: an unreadable file is not treated as a LoRA adapter.
        return False


def load_adapter_checked(path: Path) -> LoadedAdapter:
    """Load the LoRA tensors of a .safetensors adapter, reporting what was skipped.

    Every tensor is validated from the header before allocation (dtype
    allowlist, element count, byte length, LoRA rank) and loaded in its own
    error scope. Tensors that fail are not loaded; they are returned in
    ``skipped`` so the caller can mark the scan DEGRADED. Tensors that are not
    part of a lora_A/lora_B pair are listed as skipped too: they are not
    analysed, and an unanalysed tensor may carry the payload.

    Args:
        path: Path to the .safetensors file.

    Returns:
        LoadedAdapter with loaded LoRA tensors, header metadata and skipped tensors.

    Raises:
        FileNotFoundError: If the adapter file does not exist.
        ValueError: If the file is not a regular, bounded safetensors file, its
            header is invalid, it contains no lora_A/lora_B keys, the total
            element count exceeds _MAX_TOTAL_NUMEL, or no complete LoRA pair
            could be loaded.
    """
    resolved = _check_regular_file(path)
    header, data_offset = _read_sf_header(resolved)
    data_len = resolved.stat().st_size - data_offset

    raw_meta = header.get("__metadata__")
    metadata: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    keys = [k for k in header if k != "__metadata__"]

    if not any(_LORA_A_PAT.match(k) for k in keys):
        raise ValueError(
            f"No lora_A/lora_B tensor pairs found in {resolved.name} — "
            "file does not appear to be a PEFT LoRA adapter"
        )

    skipped: list[SkippedTensor] = []
    to_load: list[str] = []
    total_numel = 0
    for key in keys:
        info = header[key]
        if not (_LORA_A_PAT.match(key) or _LORA_B_PAT.match(key)):
            skipped.append(SkippedTensor(
                key, "not a lora_A/lora_B tensor; not analysed", ErrorCategory.UNSUPPORTED,
            ))
            continue
        problem = _header_entry_problem(info, data_len) or _rank_problem(key, info["shape"])
        if problem is not None:
            skipped.append(SkippedTensor(key, problem, ErrorCategory.MALFORMED))
            continue
        total_numel += math.prod(info["shape"]) if info["shape"] else 1
        to_load.append(key)

    if total_numel > _MAX_TOTAL_NUMEL:
        raise ValueError(
            f"Adapter declares {total_numel} LoRA elements, exceeding limit {_MAX_TOTAL_NUMEL}"
        )

    tensors: dict[str, np.ndarray] = {}
    try:
        handle = safe_open(str(resolved), framework="numpy")
    except Exception as exc:
        raise ValueError(f"Failed to parse {resolved}: {exc}") from exc
    with handle as f:
        for key in to_load:
            try:
                if header[key]["dtype"] == _BF16_DTYPE:
                    tensors[key] = _load_bf16_as_float32(resolved, key, header, data_offset)
                else:
                    tensors[key] = f.get_tensor(key)
            except Exception as exc:
                logger.warning("Tensor %r could not be loaded: %s", key, exc)
                skipped.append(SkippedTensor(key, f"load failed: {exc}", ErrorCategory.MALFORMED))

    if not any(
        {"A", "B"} <= pair.keys() for pair in _group_lora_layers(tensors).values()
    ):
        raise ValueError(f"No complete lora_A/lora_B pair could be loaded from {resolved.name}")

    if skipped:
        logger.warning(
            "load_adapter: %d tensor(s) in %s were not analysed", len(skipped), resolved.name,
        )
    logger.debug("Loaded %d tensor(s) from %s", len(tensors), resolved)
    return LoadedAdapter(tensors=tensors, metadata=metadata, skipped=skipped)


def load_adapter(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a .safetensors adapter file, returning tensors and metadata.

    Thin wrapper over load_adapter_checked() kept for API compatibility.
    Skipped tensors are dropped here; use load_adapter_checked() to see them.

    Args:
        path: Path to the .safetensors file.

    Returns:
        Tuple of (LoRA tensors keyed by tensor name, raw metadata dict).

    Raises:
        FileNotFoundError: If the adapter file does not exist.
        ValueError: If the file is not a valid, bounded .safetensors LoRA adapter.
    """
    loaded = load_adapter_checked(path)
    return loaded.tensors, loaded.metadata


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
