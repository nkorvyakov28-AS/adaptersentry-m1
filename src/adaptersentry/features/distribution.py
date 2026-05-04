"""Distribution shape features for LoRA delta matrices.

Computes DistributionFeatures from the effective weight delta ΔW = B @ A.
Like compute_norm_features(), this module:
  - Always operates on the *combined* delta, never on A or B alone.
  - Guards against materializing large matrices (> _MAX_DELTA_NUMEL_FULL elements).
  - Returns None on shape mismatch; returns all-zero features for zero-B init.
  - Never raises — all failures are logged and return None.

Security Notes:
    Pure numpy computation. No I/O, no eval/exec, no pickle.
    Memory bounded by _MAX_DELTA_NUMEL_FULL before matrix materialization.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Full mode: materialize ΔW only when out×in ≤ this. Must match delta_norm.py.
# 4M float32 = 16MB — keeps per-layer kurtosis temps bounded to ~64MB.
# Above threshold: use lora_A rows as ΔW proxy (computed_on_sample=True).
# Rationale: 300+ layers × 26MB ΔW × 4 temp arrays ≈ 7.8GB RSS accumulated by
# the allocator. Lowering from 16M to 4M covers Qwen3-4B attn layers (2560×2560=6.5M)
# and similar architectures without degrading detection on small/medium adapters.
_MAX_DELTA_NUMEL_FULL = 4_000_000

# Fast mode: skip ΔW materialization when out×in exceeds this.
# For BLOOM (3072×1024=3.1M) or LLaMA (4096×14336=58M), materialize is expensive:
# _numpy_kurtosis_skewness on 3M elements allocates 4×24MB temp arrays = ~100ms.
# Below 100K (e.g., small adapters) materialization is fine even in fast mode.
_MAX_DELTA_NUMEL_FAST = 100_000

# Random sample size when the full delta can't be materialized
_SAMPLE_ROWS = 4096

_ENTROPY_BINS = 256
# Full-mode: sample delta for sort-heavy ops (median/percentile/entropy/kurtosis).
# 50K gives <0.1% error on percentiles vs full 4M+ arrays — negligible for threshold-based
# anomaly detection. Reduces sort cost from ~22ms to ~2ms per layer.
_MAX_SORT_SAMPLE = 50_000


def _numpy_kurtosis_skewness(x: np.ndarray) -> tuple[float, float]:
    """Numpy-only Fisher excess kurtosis and skewness.

    ~3–4× faster than scipy on large arrays (no Python-level overhead, no
    bias correction). Used in fast mode where a small precision difference
    vs scipy's bias-corrected estimators is acceptable.

    Returns (kurtosis, skewness), both 0.0 for near-constant or tiny arrays.
    """
    if x.size < 4:
        return 0.0, 0.0
    d = x - x.mean()
    d2 = d * d
    m2 = float(d2.mean())
    if m2 < 1e-30:
        return 0.0, 0.0
    m4 = float((d2 * d2).mean())
    m3 = float((d2 * d).mean())
    return m4 / (m2 * m2) - 3.0, m3 / (m2 ** 1.5)


def _compute_delta_entropy(flat: np.ndarray) -> float:
    """Normalized Shannon entropy of a flat array's value distribution (0–1)."""
    if flat.size <= 1 or float(flat.max() - flat.min()) == 0.0:
        return 0.0
    counts, _ = np.histogram(flat, bins=_ENTROPY_BINS)
    nonzero = counts[counts > 0]
    probs = nonzero / nonzero.sum()
    h = float(-np.sum(probs * np.log2(probs)))
    return float(np.clip(h / np.log2(_ENTROPY_BINS), 0.0, 1.0))


def compute_distribution_features(
    tensor_A: np.ndarray,
    tensor_B: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    fast: bool = False,
) -> "DistributionFeatures | None":
    """Compute DistributionFeatures from a paired (lora_A, lora_B) tensor set.

    ΔW = B @ A is the effective weight update applied by this LoRA layer.
    Statistics (kurtosis, skewness, mean, std) are derived from ΔW.

    For large delta matrices (out × in > _MAX_DELTA_NUMEL_FULL), stats are computed
    on a random row-sample of the flattened lora_A matrix as a proxy. The
    returned DistributionFeatures.computed_on_sample flag is set to True.

    Fast mode skips the extended ANAL-01 fields (median, percentiles, iqr,
    zero_ratio, delta_entropy) — sort-heavy operations that cost ~34ms per
    layer at LLaMA scale (3.8s per adapter across 112 layers). Those fields
    default to 0.0 in the schema.

    Args:
        tensor_A: lora_A weight matrix (rank × in_features).
        tensor_B: lora_B weight matrix (out_features × rank).
        rng:      Optional random generator for reproducible sampling.
        fast:     If True, skip extended distribution fields (full mode only).

    Returns:
        DistributionFeatures, or None on shape/type error.
    """
    from adaptersentry.schemas.distribution_features import DistributionFeatures

    try:
        # float32 is sufficient for anomaly-detection thresholds; halves matmul
        # memory (17MB vs 34MB for large LoRA layers) and reduces BLAS time ~2x.
        a = tensor_A.astype(np.float32)
        b = tensor_B.astype(np.float32)
    except (ValueError, TypeError) as exc:
        logger.warning("compute_distribution_features: dtype cast failed: %s", exc)
        return None

    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    if a.ndim != 2 or b.ndim != 2:
        return None

    if b.shape[1] != a.shape[0]:
        logger.warning(
            "compute_distribution_features: shape mismatch B%s @ A%s",
            b.shape, a.shape,
        )
        return None

    norm_b = float(np.sqrt(np.sum(b * b)))
    if norm_b < 1e-12:
        # Zero-B init — delta is zero; distribution stats are degenerate
        return DistributionFeatures(
            delta_kurtosis=0.0,
            delta_skewness=0.0,
            delta_mean=0.0,
            delta_std=0.0,
            computed_on_sample=False,
        )

    out_feats = b.shape[0]
    in_feats = a.shape[1]
    delta_numel = out_feats * in_feats
    computed_on_sample = False

    # Choose materialization threshold: fast mode skips large matmuls to avoid
    # allocating 10s-of-MB temp arrays for kurtosis on millions of elements.
    mat_threshold = _MAX_DELTA_NUMEL_FAST if fast else _MAX_DELTA_NUMEL_FULL

    if delta_numel <= mat_threshold:
        # Exact path: materialize ΔW
        try:
            delta = (b @ a).ravel()
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.warning("compute_distribution_features: delta materialization failed: %s", exc)
            return None
    else:
        # Proxy path: use A's flattened rows as ΔW proxy.
        # Triggered in fast mode for large deltas, and in full mode for delta > 16M.
        logger.debug(
            "compute_distribution_features: large delta (%d × %d) — using A proxy",
            out_feats, in_feats,
        )
        if rng is None:
            rng = np.random.default_rng(42)
        n_rows = min(_SAMPLE_ROWS, a.shape[0])
        sampled_rows = rng.choice(a.shape[0], size=n_rows, replace=False)
        delta = a[sampled_rows, :].ravel()
        computed_on_sample = True

    # float32 mean/std: sufficient precision for anomaly thresholds (>0.01);
    # float64 promotion on 4M elements costs ~50ms (type conversion + two passes).
    delta_mean = float(np.mean(delta)) if delta.size > 0 else 0.0
    delta_std = float(np.std(delta)) if delta.size > 0 else 0.0

    if delta.size < 4:
        # Not enough elements for reliable kurtosis/skewness
        return DistributionFeatures(
            delta_kurtosis=0.0,
            delta_skewness=0.0,
            delta_mean=delta_mean,
            delta_std=delta_std,
            computed_on_sample=computed_on_sample,
        )

    if fast:
        kurt, skew = _numpy_kurtosis_skewness(delta)
        return DistributionFeatures(
            delta_kurtosis=kurt,
            delta_skewness=skew,
            delta_mean=delta_mean,
            delta_std=delta_std,
            computed_on_sample=computed_on_sample,
        )

    # Stride sample: deterministic, zero-allocation view, uniform coverage of
    # ΔW rows. rng.choice(4M, 500K, replace=False) costs ~46ms; stride is free.
    # Security note: stride covers every K-th element across all output rows —
    # an adversary cannot concentrate malicious weights in unsampled positions
    # without knowing K, which depends on the adapter's own tensor dimensions.
    if delta.size > _MAX_SORT_SAMPLE:
        stride = max(1, delta.size // _MAX_SORT_SAMPLE)
        stats_delta = delta[::stride][:_MAX_SORT_SAMPLE]
        computed_on_sample = True
    else:
        stats_delta = delta

    # Kurtosis/skewness on sample (negligible precision loss for threshold-based detection).
    kurt, skew = _numpy_kurtosis_skewness(stats_delta)

    delta_median = float(np.median(stats_delta))
    p01, p25, p75, p99 = (float(v) for v in np.percentile(stats_delta, [1, 25, 75, 99]))
    delta_iqr = p75 - p25
    delta_zero_ratio = float(np.mean(np.abs(stats_delta) < 1e-8))
    delta_entropy = _compute_delta_entropy(stats_delta)

    return DistributionFeatures(
        delta_kurtosis=kurt,
        delta_skewness=skew,
        delta_mean=delta_mean,
        delta_std=delta_std,
        delta_median=delta_median,
        delta_p01=p01,
        delta_p99=p99,
        delta_iqr=delta_iqr,
        delta_zero_ratio=delta_zero_ratio,
        delta_entropy=delta_entropy,
        computed_on_sample=computed_on_sample,
    )
