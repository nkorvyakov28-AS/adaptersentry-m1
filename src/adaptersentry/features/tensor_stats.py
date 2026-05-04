"""Per-tensor statistical feature extraction.

Both functions support an optional ``fast`` flag that reduces computation
cost on large tensors while preserving backdoor detection signals.

Security Notes:
    - Pure numpy/scipy computation; no I/O, no external calls.
    - Kurtosis / skewness precision loss on near-constant tensors is caught
      and returns 0.0 rather than nan.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# OPT-04: single-pass Rust stats (2.4× faster than numpy; sorts once for all percentiles).
try:
    from adaptersentry_rs import tensor_stats_f32 as _rs_tensor_stats
    _RUST_STATS_AVAILABLE = True
except ImportError:
    _RUST_STATS_AVAILABLE = False

_SAMPLE_THRESHOLD = 100_000
_SAMPLE_SIZE = 50_000
_SAMPLE_SEED = 42
_SVD_TRUNCATE_THRESHOLD = 512
_SVD_N_COMPONENTS = 50


def compute_tensor_stats(tensor: np.ndarray, *, fast: bool = False) -> dict[str, float]:
    """Compute descriptive statistics for a weight tensor.

    Uses the Rust extension (OPT-04) when available: single-pass kurtosis/
    skewness + one sort for all percentiles. Falls back to numpy if the
    extension is not installed.

    Args:
        tensor: Weight matrix (any shape — flattened internally).
        fast:   If True and tensor > 100K elements, compute on a 50K-element
                deterministic sample.

    Returns:
        Dict with keys: mean, std, median, p01, p99, iqr, zero_ratio,
        kurtosis, skewness.
    """
    flat_f32 = tensor.astype(np.float32).flatten()

    if fast and flat_f32.size > _SAMPLE_THRESHOLD:
        rng = np.random.default_rng(_SAMPLE_SEED)
        flat_f32 = flat_f32[rng.choice(flat_f32.size, _SAMPLE_SIZE, replace=False)]

    if _RUST_STATS_AVAILABLE and flat_f32.size >= 4:
        kurt, skew, mean, std, median, p01, p99, iqr, zero_ratio = _rs_tensor_stats(flat_f32)
        return {
            "mean": mean, "std": std,
            "median": median, "p01": p01, "p99": p99,
            "iqr": iqr, "zero_ratio": zero_ratio,
            "kurtosis": kurt, "skewness": skew,
        }

    # Fallback: numpy (float64 for precision)
    flat = flat_f32.astype(np.float64)
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    median = float(np.median(flat))
    p01, p25, p75, p99 = (float(v) for v in np.percentile(flat, [1, 25, 75, 99]))
    iqr = p75 - p25
    zero_ratio = float(np.mean(np.abs(flat) < 1e-8))

    if flat.size >= 4:
        d = flat - flat.mean()
        d2 = d * d
        m2 = float(d2.mean())
        if m2 > 1e-30:
            kurt = float((d2 * d2).mean() / (m2 * m2)) - 3.0
            skewness = float((d2 * d).mean() / (m2 ** 1.5))
        else:
            kurt, skewness = 0.0, 0.0
    else:
        kurt, skewness = 0.0, 0.0

    return {
        "mean": mean, "std": std,
        "median": median, "p01": p01, "p99": p99,
        "iqr": iqr, "zero_ratio": zero_ratio,
        "kurtosis": kurt, "skewness": skewness,
    }


def compute_svd_stats(tensor: np.ndarray, *, fast: bool = False) -> dict[str, Any]:
    """Compute SVD-based statistics: effective rank and energy concentration.

    Args:
        tensor: Weight matrix (2-D preferred; higher dims are reshaped).
        fast:   If True, use randomised truncated SVD for matrices >= 512×512.
    """
    mat = tensor.astype(np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    elif mat.ndim > 2:
        mat = mat.reshape(mat.shape[0], -1)

    rows, cols = mat.shape
    use_truncated = fast and min(rows, cols) >= _SVD_TRUNCATE_THRESHOLD

    if use_truncated:
        try:
            from sklearn.utils.extmath import randomized_svd
            k = min(_SVD_N_COMPONENTS, min(rows, cols) - 1)
            _, singular_values, _ = randomized_svd(mat, n_components=k, random_state=_SAMPLE_SEED)
            singular_values = np.sort(singular_values)[::-1]
            truncated = True
        except Exception:
            singular_values = np.linalg.svd(mat, compute_uv=False)
            truncated = False
    else:
        singular_values = np.linalg.svd(mat, compute_uv=False)
        truncated = False

    total_energy = float(np.sum(singular_values**2))
    if total_energy == 0.0:
        return {"effective_rank": 0, "energy_concentration": 0.0, "singular_values_count": 0}

    energy_concentration = float(singular_values[0] ** 2 / total_energy)
    cumulative = np.cumsum(singular_values**2) / total_energy

    if not truncated:
        effective_rank = int(np.searchsorted(cumulative, 0.99) + 1)
    else:
        idx = np.searchsorted(cumulative, 0.99)
        effective_rank = int(idx + 1) if idx < len(cumulative) else len(singular_values)

    return {
        "effective_rank": effective_rank,
        "energy_concentration": energy_concentration,
        "singular_values_count": len(singular_values),
    }
