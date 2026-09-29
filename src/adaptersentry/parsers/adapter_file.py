"""Streaming access to a LoRA adapter file: inventory first, one pair at a time.

``open_adapter`` reads and validates only the safetensors header and returns an
inventory of every tensor: LoRA A/B pairs (with layer, expert, module and kind),
full weights saved via ``modules_to_save``, DoRA magnitude vectors, trainable
token deltas, and everything that cannot be analysed (with the reason).

``iter_pairs`` then loads one A/B pair at a time as float32 and releases it
before the next, so peak memory is bounded by the largest pair (~20 MB for a
70B model at r=64) instead of the whole adapter (~3.3 GB).

Security Notes:
    - Reuses the v1.0.3 guards: regular file, bounded size, bounded header,
      per-entry dtype/shape/byte-length/offset/rank validation before any
      allocation (see parsers.safetensors).
    - Nothing is skipped silently: every tensor ends up in exactly one bucket
      of the inventory, and load failures during iteration are returned, not
      swallowed.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from adaptersentry.parsers.names import ModuleLocation, locate, parse_key
from adaptersentry.parsers.safetensors import (
    _BF16_DTYPE,
    SkippedTensor,
    _check_regular_file,
    _header_entry_problem,
    _load_bf16_as_float32,
    _MAX_LORA_RANK,
    _read_sf_header,
)
from adaptersentry.schemas.errors import ErrorCategory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorEntry:
    """A header entry: key, dtype, shape."""

    key: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1


@dataclass(frozen=True)
class LoraPair:
    """One LoRA A/B pair, located in the model.

    For standard LoRA, A is (r, in) and B is (out, r), and ΔW = s·B·A is (out, in).
    For embedding LoRA (``lora_embedding_A/B``), A is (r, vocab) and B is
    (dim, r); PEFT applies (B·A)ᵀ, so token rows of ΔW are columns of B·A
    (``transposed=True``).
    """

    base: str
    a: TensorEntry
    b: TensorEntry
    location: ModuleLocation
    transposed: bool = False

    @property
    def rank(self) -> int:
        return self.a.shape[0]


@dataclass(frozen=True)
class FullWeight:
    """A complete weight matrix stored in the adapter (e.g. modules_to_save)."""

    entry: TensorEntry
    location: ModuleLocation


@dataclass
class AdapterInventory:
    """Everything in the file, classified without loading tensor data."""

    path: Path
    header: dict[str, Any]
    data_offset: int
    metadata: dict[str, str]
    n_tensors_total: int
    pairs: list[LoraPair] = field(default_factory=list)
    full_weights: list[FullWeight] = field(default_factory=list)
    dora_magnitudes: list[str] = field(default_factory=list)
    trainable_tokens: list[str] = field(default_factory=list)
    not_analyzed: list[SkippedTensor] = field(default_factory=list)

    @property
    def max_pair_bytes_f32(self) -> int:
        """Largest float32 footprint of a single pair (the streaming memory bound)."""
        return max((4 * (p.a.numel + p.b.numel) for p in self.pairs), default=0)


@dataclass(frozen=True)
class PairData:
    """One loaded pair; ``error`` is set and arrays are None if loading failed."""

    pair: LoraPair
    a: np.ndarray | None
    b: np.ndarray | None
    error: str | None = None


def _entry(key: str, info: dict[str, Any]) -> TensorEntry:
    return TensorEntry(key=key, dtype=str(info["dtype"]), shape=tuple(int(d) for d in info["shape"]))


def _pair_problem(a: TensorEntry, b: TensorEntry) -> str | None:
    if len(a.shape) != 2 or len(b.shape) != 2:
        return f"LoRA tensors must be 2-D, got A{list(a.shape)} B{list(b.shape)}"
    if a.shape[0] != b.shape[1]:
        return f"rank mismatch: A{list(a.shape)} vs B{list(b.shape)}"
    if a.shape[0] < 1:
        return "rank 0"
    if a.shape[0] > _MAX_LORA_RANK:
        return f"LoRA rank {a.shape[0]} exceeds limit {_MAX_LORA_RANK}"
    return None


def open_adapter(path: Path) -> AdapterInventory:
    """Validate the file and classify every tensor, without loading tensor data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a regular, bounded safetensors file or
            its header is invalid.
    """
    resolved = _check_regular_file(path)
    header, data_offset = _read_sf_header(resolved)
    data_len = resolved.stat().st_size - data_offset
    # safetensors rejects the whole file if any header entry is inconsistent; the
    # real loader would refuse it too, so this is a failed parse, found up front.
    try:
        with safe_open(str(resolved), framework="numpy"):
            pass
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to parse {resolved.name}: {exc}") from exc

    raw_meta = header.get("__metadata__")
    metadata = {k: v for k, v in raw_meta.items() if isinstance(k, str) and isinstance(v, str)} \
        if isinstance(raw_meta, dict) else {}
    keys = [k for k in header if k != "__metadata__"]

    inv = AdapterInventory(
        path=resolved, header=header, data_offset=data_offset,
        metadata=metadata, n_tensors_total=len(keys),
    )

    halves: dict[tuple[str, bool], dict[str, TensorEntry]] = {}
    for key in keys:
        info = header[key]
        problem = _header_entry_problem(info, data_len)
        if problem is not None:
            inv.not_analyzed.append(SkippedTensor(key, problem, ErrorCategory.MALFORMED))
            continue
        entry = _entry(key, info)
        parsed = parse_key(key)
        if parsed.role in ("lora_A", "lora_B", "embedding_A", "embedding_B"):
            embedding = parsed.role.startswith("embedding")
            side = "A" if parsed.role.endswith("A") else "B"
            halves.setdefault((parsed.base or "", embedding), {})[side] = entry
        elif parsed.role == "dora_magnitude":
            inv.dora_magnitudes.append(key)
        elif parsed.role == "trainable_tokens":
            inv.trainable_tokens.append(key)
            inv.not_analyzed.append(SkippedTensor(
                key, "trainable token delta; not analysed yet", ErrorCategory.UNSUPPORTED,
            ))
        elif len(entry.shape) == 2:
            base = key[: -len(".weight")] if key.endswith(".weight") else key
            inv.full_weights.append(FullWeight(entry=entry, location=locate(base)))
        else:
            inv.not_analyzed.append(SkippedTensor(
                key, f"{len(entry.shape)}-D tensor outside a LoRA pair; not analysed",
                ErrorCategory.UNSUPPORTED,
            ))

    for (base, embedding), sides in halves.items():
        if set(sides) != {"A", "B"}:
            for entry in sides.values():
                inv.not_analyzed.append(SkippedTensor(
                    entry.key, "LoRA tensor without its A/B counterpart", ErrorCategory.MALFORMED,
                ))
            continue
        problem = _pair_problem(sides["A"], sides["B"])
        if problem is not None:
            for entry in sides.values():
                inv.not_analyzed.append(SkippedTensor(entry.key, problem, ErrorCategory.MALFORMED))
            continue
        inv.pairs.append(LoraPair(
            base=base, a=sides["A"], b=sides["B"], location=locate(base), transposed=embedding,
        ))

    inv.pairs.sort(key=lambda p: (
        p.location.layer if p.location.layer is not None else -1,
        p.location.expert if p.location.expert is not None else -1,
        p.base,
    ))
    return inv


def _load(handle: Any, inv: AdapterInventory, entry: TensorEntry) -> np.ndarray:
    if entry.dtype == _BF16_DTYPE:
        return _load_bf16_as_float32(inv.path, entry.key, inv.header, inv.data_offset)
    return np.asarray(handle.get_tensor(entry.key), dtype=np.float32)


def iter_pairs(inv: AdapterInventory) -> Iterator[PairData]:
    """Load LoRA pairs one at a time as float32.

    The caller should drop each PairData before requesting the next to keep
    peak memory at a single pair. Load failures are yielded with ``error`` set.
    """
    with safe_open(str(inv.path), framework="numpy") as handle:
        for pair in inv.pairs:
            try:
                a = _load(handle, inv, pair.a)
                b = _load(handle, inv, pair.b)
            except Exception as exc:  # noqa: BLE001 — reported to the caller, never swallowed
                logger.warning("LoRA pair %r could not be loaded: %s", pair.base, exc)
                yield PairData(pair=pair, a=None, b=None, error=f"load failed: {exc}")
                continue
            yield PairData(pair=pair, a=a, b=b)
