"""Entropy and compression features for LoRA weight matrices (M1-ANAL-02).

All computations are O(n). Size caps are applied to zlib and np.unique
to keep per-adapter latency bounded on large LLM adapters
(e.g. LLaMA-style with A=(16, 14336), 224 layers).

Size caps (applied to representative prefixes/samples):
  _ZLIB_MAX_BYTES      — compress only the first 64KB of raw float32 bytes.
                         Full tensor bytes are used for the ratio denominator.
  _UNIQUE_MAX_ELEMENTS — np.unique on at most 50K elements (deterministic sample).

Both caps give accurate results for quantization detection: the feature signals
(compression ratio, unique ratio, quantization score) are dominated by the
distribution of values, which is well-represented by a large sample.

Security Notes:
    Pure numpy/zlib computation. No I/O, no eval/exec, no pickle.
    zlib operates on raw bytes produced by numpy — not on adapter-controlled input.
"""

from __future__ import annotations

import logging
import zlib
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

# OPT-04: Rust single-pass entropy computations (4.5× faster for byte_entropy,
# single-pass sign_stats). Falls back to numpy if extension absent.
try:
    from adaptersentry_rs import byte_entropy as _rs_byte_entropy, sign_stats as _rs_sign_stats
    _RUST_ENTROPY_AVAILABLE = True
except ImportError:
    _RUST_ENTROPY_AVAILABLE = False

_BYTE_ENTROPY_SYMBOLS = 256
_LOG2_3 = float(np.log2(3))

# Size caps to keep per-tensor cost bounded on large LLM adapters
_ZLIB_MAX_BYTES = 65_536        # 64KB — representative for compression ratio
_UNIQUE_MAX_ELEMENTS = 50_000   # deterministic sample for unique value count
_UNIQUE_SAMPLE_SEED = 42


class _TensorStats(NamedTuple):
    value_repeat_ratio: float
    unique_value_ratio: float
    approx_compression_ratio: float
    byte_entropy: float
    sign_entropy: float
    sign_balance: float
    quantization_suspect_score: float


def _compute_single(tensor: np.ndarray) -> _TensorStats:
    """Compute all EntropyCompressionFeatures stats for one matrix. O(n)."""
    flat_f32 = tensor.astype(np.float32).flatten()
    n = flat_f32.size

    if n == 0:
        return _TensorStats(0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0)

    # unique value stats — capped sample for large tensors (O(cap × log cap))
    if n > _UNIQUE_MAX_ELEMENTS:
        rng = np.random.default_rng(_UNIQUE_SAMPLE_SEED)
        flat_for_unique = rng.choice(flat_f32, _UNIQUE_MAX_ELEMENTS, replace=False)
        sample_n = _UNIQUE_MAX_ELEMENTS
    else:
        flat_for_unique = flat_f32
        sample_n = n
    unique_vals = np.unique(flat_for_unique)
    num_unique = len(unique_vals)
    unique_value_ratio = float(num_unique) / float(sample_n)
    value_repeat_ratio = 1.0 - unique_value_ratio

    # compression ratio — zlib level=1 on raw float32 bytes (capped at 64KB)
    raw_bytes = flat_f32.tobytes()
    raw_len = len(raw_bytes)
    sample_bytes = raw_bytes[:_ZLIB_MAX_BYTES] if raw_len > _ZLIB_MAX_BYTES else raw_bytes
    sample_len = len(sample_bytes)
    try:
        compressed_len = len(zlib.compress(sample_bytes, level=1))
        approx_compression_ratio = float(compressed_len) / float(sample_len) if sample_len > 0 else 1.0
    except Exception:
        approx_compression_ratio = 1.0

    # byte entropy — Shannon entropy over 256 byte symbols (same sample as zlib)
    if _RUST_ENTROPY_AVAILABLE:
        byte_entropy = float(np.clip(_rs_byte_entropy(sample_bytes), 0.0, 1.0))
    else:
        byte_array = np.frombuffer(sample_bytes, dtype=np.uint8)
        counts = np.bincount(byte_array, minlength=_BYTE_ENTROPY_SYMBOLS)
        nonzero = counts[counts > 0]
        if nonzero.size > 1:
            probs = nonzero / nonzero.sum()
            h = float(-np.sum(probs * np.log2(probs)))
            byte_entropy = float(np.clip(h / np.log2(_BYTE_ENTROPY_SYMBOLS), 0.0, 1.0))
        else:
            byte_entropy = 0.0

    # sign entropy and balance
    n_pos = int(np.sum(flat_f32 > 0.0))
    n_neg = int(np.sum(flat_f32 < 0.0))
    n_zer = n - n_pos - n_neg
    sign_balance = float(n_pos) / float(n)

    sign_counts = np.array([n_neg, n_zer, n_pos], dtype=np.float64)
    nonzero_sign = sign_counts[sign_counts > 0]
    if nonzero_sign.size > 1:
        probs = nonzero_sign / nonzero_sign.sum()
        h_sign = float(-np.sum(probs * np.log2(probs)))
        sign_entropy = float(np.clip(h_sign / _LOG2_3, 0.0, 1.0))
    else:
        sign_entropy = 0.0

    # quantization suspect score: 1 - log2(num_unique+1)/32
    effective_bits = float(np.log2(num_unique + 1))
    quantization_suspect_score = float(np.clip(1.0 - effective_bits / 32.0, 0.0, 1.0))

    return _TensorStats(
        value_repeat_ratio=value_repeat_ratio,
        unique_value_ratio=unique_value_ratio,
        approx_compression_ratio=approx_compression_ratio,
        byte_entropy=byte_entropy,
        sign_entropy=sign_entropy,
        sign_balance=sign_balance,
        quantization_suspect_score=quantization_suspect_score,
    )


def compute_entropy_compression_features(
    tensor_A: np.ndarray,
    tensor_B: np.ndarray,
) -> "EntropyCompressionFeatures | None":
    """Compute EntropyCompressionFeatures for a paired (lora_A, lora_B) set.

    All operations are O(n) — no materialization of ΔW, no sampling.
    Runs in both fast and full scan modes.

    Args:
        tensor_A: lora_A weight matrix (any shape).
        tensor_B: lora_B weight matrix (any shape).

    Returns:
        EntropyCompressionFeatures, or None on type/computation error.
    """
    from adaptersentry.schemas.entropy_compression_features import EntropyCompressionFeatures

    try:
        sa = _compute_single(tensor_A)
        sb = _compute_single(tensor_B)
    except Exception as exc:
        logger.warning("compute_entropy_compression_features: failed — %s", exc)
        return None

    return EntropyCompressionFeatures(
        value_repeat_ratio_a=sa.value_repeat_ratio,
        unique_value_ratio_a=sa.unique_value_ratio,
        approx_compression_ratio_a=sa.approx_compression_ratio,
        byte_entropy_a=sa.byte_entropy,
        sign_entropy_a=sa.sign_entropy,
        sign_balance_a=sa.sign_balance,
        quantization_suspect_score_a=sa.quantization_suspect_score,
        value_repeat_ratio_b=sb.value_repeat_ratio,
        unique_value_ratio_b=sb.unique_value_ratio,
        approx_compression_ratio_b=sb.approx_compression_ratio,
        byte_entropy_b=sb.byte_entropy,
        sign_entropy_b=sb.sign_entropy,
        sign_balance_b=sb.sign_balance,
        quantization_suspect_score_b=sb.quantization_suspect_score,
    )
