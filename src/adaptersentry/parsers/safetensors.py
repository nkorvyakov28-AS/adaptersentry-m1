"""safetensors header validation — the guards every read goes through.

A .safetensors file is ``[8-byte little-endian header length][JSON header][tensor bytes]``.
Everything here runs before any tensor is allocated:

    check_regular_file     resolved path must be a regular .safetensors file ≤ MAX_FILE_BYTES
                           (no FIFOs or devices, including through symlinks)
    read_header            header length ≤ MAX_HEADER_BYTES and within the file; JSON object
    header_entry_problem   dtype allowlist, shape, byte length, offsets, element count
    load_bf16_as_float32   bfloat16 is not a numpy dtype: convert raw bytes, length-checked

Security Notes:
    - No pickle, eval or exec. JSON is parsed only after the length checks.
    - Tensor bomb guard: at most MAX_TENSOR_NUMEL elements per tensor, checked
      on the header-declared shape before allocation.
"""

from __future__ import annotations

import json
import logging
import math
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adaptersentry.schemas.errors import ErrorCategory

logger = logging.getLogger(__name__)

# Tensor bomb guard: reject tensors with more than 1B elements before allocation.
MAX_TENSOR_NUMEL = 1_000_000_000
# Same bound the safetensors library itself applies to the JSON header.
MAX_HEADER_BYTES = 100_000_000
# Largest adapter file we are willing to open.
MAX_FILE_BYTES = 64 * 1024**3
# LoRA rank above which a pair is not analysed.
MAX_LORA_RANK = 1024

BF16_DTYPE = "BF16"
# dtypes accepted for LoRA weights, with their byte width.
ALLOWED_DTYPES: dict[str, int] = {"F32": 4, "F16": 2, "BF16": 2}


@dataclass(frozen=True)
class SkippedTensor:
    """A tensor present in the file that was not analysed, and why."""

    key: str
    reason: str
    category: ErrorCategory


def check_regular_file(path: Path) -> Path:
    """Resolve *path* and require a regular .safetensors file of sane size.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If it is not a regular file, has the wrong suffix after
            symlink resolution, or exceeds MAX_FILE_BYTES.
    """
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Adapter file not found: {resolved}")
    st = resolved.stat()
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"Not a regular file: {resolved}")
    if resolved.suffix != ".safetensors":
        raise ValueError(f"Expected .safetensors file, got: {resolved.suffix!r}")
    if st.st_size > MAX_FILE_BYTES:
        raise ValueError(
            f"File too large: {st.st_size} bytes exceeds limit {MAX_FILE_BYTES}"
        )
    return resolved


def read_header(path: Path) -> tuple[dict, int]:
    """Return (header_dict, data_start_offset) from a safetensors file.

    Reads only the 8-byte length prefix and the JSON header body — no tensor
    data is loaded. data_start_offset is the byte position where tensor data begins.

    Raises:
        ValueError: If the file is shorter than 8 bytes, the declared header
            length exceeds MAX_HEADER_BYTES or the file itself, or the header
            is not a JSON object. The length is checked before any allocation.
    """
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) < 8:
            raise ValueError(f"File too short to be safetensors: {file_size} bytes")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len > MAX_HEADER_BYTES:
            raise ValueError(
                f"safetensors header length {header_len} exceeds limit {MAX_HEADER_BYTES}"
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


def header_entry_problem(info: Any, data_len: int) -> str | None:
    """Validate one tensor entry of a safetensors header.

    Returns:
        None if the entry is acceptable for loading, otherwise a reason string.
    """
    if not isinstance(info, dict):
        return "header entry is not an object"
    dtype = info.get("dtype")
    if dtype not in ALLOWED_DTYPES:
        return f"unsupported dtype {dtype!r}"
    shape = info.get("shape")
    if not isinstance(shape, list) or not all(
        isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
    ):
        return f"invalid shape {shape!r}"
    numel = math.prod(shape) if shape else 1
    if numel > MAX_TENSOR_NUMEL:
        return f"numel {numel} exceeds limit {MAX_TENSOR_NUMEL}"
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
    if end - start != numel * ALLOWED_DTYPES[dtype]:
        return f"byte length {end - start} does not match shape {shape} and dtype {dtype}"
    return None


def load_bf16_as_float32(
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
