"""Delta norm utilities for LoRA weight matrices.

LoRA decomposes a weight update into W = W_0 + BA.  This module provides
utilities for computing norms of the effective delta (B @ A), which can
serve as an additional signal for rank and magnitude anomalies.

Public API
----------
compute_norm_features(A, B) -> NormFeatures | None
    Primary entry point.  Returns all four magnitude features from ΔW = B @ A,
    or None on shape mismatch.  Zero-B init is handled as a valid special case.

compute_delta_frobenius_norm(A, B) -> float
    Legacy helper — used internally by compute_norm_features.

compute_effective_rank_ratio(A, claimed_rank) -> float | None
    Supplementary ratio signal (A only, not the delta).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Full mode: materialise ΔW only when out×in ≤ this; above → Cholesky path.
# Must match _MAX_DELTA_NUMEL_FULL in features/distribution.py.
# 4M float32 = 16MB — keeps peak allocation per layer bounded to ~32MB.
# Rationale: same as distribution.py — 300+ layers × large ΔW allocs accumulate
# multi-GB RSS in glibc's allocator even with sequential processing.
_MAX_DELTA_NUMEL_FULL = 4_000_000

# Fast mode: skip ΔW materialisation when out×in exceeds this.
# BLOOM (3072×1024=3.1M) or LLaMA (4096×14336=58M) exceed this threshold.
# The Cholesky path computes fro_norm exactly; max_abs/mean_abs are 0.0.
_MAX_DELTA_NUMEL_FAST = 100_000


def compute_delta_frobenius_norm(
    tensor_A: np.ndarray,
    tensor_B: np.ndarray,
) -> float:
    """Compute the Frobenius norm of the composed LoRA delta B @ A.

    For large tensors the composition is approximated via the identity:
        ||BA||_F ≤ ||B||_F × ||A||_F
    The exact product is computed only when the inner dimension is ≤ 1024
    to bound memory usage.

    Args:
        tensor_A: lora_A weight matrix (rank × in_features).
        tensor_B: lora_B weight matrix (out_features × rank).

    Returns:
        Frobenius norm of B @ A, or the product of individual norms as an
        upper-bound estimate when the exact product is skipped.
    """
    # float32 matmul halves allocation (17MB vs 34MB for LLaMA-scale layers);
    # Frobenius norm precision is more than sufficient at float32.
    a = tensor_A.astype(np.float32)
    b = tensor_B.astype(np.float32)

    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    rank = a.shape[0]
    if rank <= 1024:
        try:
            delta = b @ a
            return float(np.sqrt(np.sum(delta * delta, dtype=np.float64)))
        except (ValueError, np.linalg.LinAlgError):
            pass

    # Fallback: upper bound via submultiplicativity
    return float(np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b)))


def compute_norm_features(
    tensor_A: np.ndarray,
    tensor_B: np.ndarray,
    *,
    fast: bool = False,
) -> "NormFeatures | None":  # NormFeatures imported lazily to avoid circular import
    """Compute NormFeatures from a paired (lora_A, lora_B) tensor set.

    ΔW = B @ A is the effective weight update applied by this LoRA layer.
    All four magnitude features are derived from ΔW, never from A or B alone.

    Special cases
    -------------
    Zero-B init  (||B||_F < 1e-12):
        Returns NormFeatures with all fields 0.0.  This is the standard LoRA
        initialisation state (B = 0, A = random) and is NOT an anomaly.

    Shape mismatch (B.shape[1] != A.shape[0]):
        Returns None.  The caller should emit a DEGRADED_LAYER flag.

    Large delta matrix (out × in > _MAX_DELTA_NUMEL_FULL):
        fro_norm_delta is computed exactly via the Cholesky path.
        max_abs_delta and mean_abs_delta are 0.0 (not materialised).

    Future normalisation hook
    -------------------------
    delta_norm_ratio is unnormalized.  When claimed_rank and lora_alpha are
    available from AdapterMetadata, the scorer can compute:
        normalized_fro = fro_norm_delta * (claimed_rank / lora_alpha)

    Args:
        tensor_A: lora_A weight matrix (rank × in_features).
        tensor_B: lora_B weight matrix (out_features × rank).

    Returns:
        NormFeatures, or None on unrecoverable shape/type error.

    Security Notes:
        - Pure numpy computation; no I/O, no eval/exec.
        - Memory bounded by _MAX_DELTA_NUMEL_FULL before matrix materialisation.
        - Shape mismatch is caught and returned as None, not raised.
    """
    from adaptersentry.schemas.norm_features import NormFeatures

    # float32 matmul: halves peak memory for large LoRA layers (17MB vs 34MB),
    # reduces BLAS time ~2x. Norm ratios are compared at coarse thresholds only.
    a = tensor_A.astype(np.float32)
    b = tensor_B.astype(np.float32)

    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    if a.ndim != 2 or b.ndim != 2:
        logger.warning("compute_norm_features: unexpected tensor rank (A=%dD, B=%dD)", a.ndim, b.ndim)
        return None

    # Shape compatibility: B is (out, r), A is (r, in) — inner dim must match
    if b.shape[1] != a.shape[0]:
        logger.warning(
            "compute_norm_features: shape mismatch B%s @ A%s — inner dims %d != %d",
            b.shape, a.shape, b.shape[1], a.shape[0],
        )
        return None

    norm_b = float(np.sqrt(np.sum(b * b)))
    norm_a = float(np.sqrt(np.sum(a * a)))

    # Zero-B initialization — delta is exactly zero, not anomalous
    if norm_b < 1e-12:
        return NormFeatures(
            fro_norm_delta=0.0,
            max_abs_delta=0.0,
            mean_abs_delta=0.0,
            delta_norm_ratio=0.0,
        )

    out_feats = b.shape[0]
    in_feats = a.shape[1]
    rank = a.shape[0]
    delta_numel = out_feats * in_feats

    mat_threshold = _MAX_DELTA_NUMEL_FAST if fast else _MAX_DELTA_NUMEL_FULL

    if delta_numel <= mat_threshold:
        # Exact path: materialise ΔW and compute all features.
        # Compute abs_delta once — avoids two 24MB allocations for max + mean.
        try:
            delta = b @ a  # (out_features, in_features) — float32
            # In-place abs: eliminates a 17MB allocation vs np.abs(delta).
            # Compute fro first (needs signed values for dot-product sum).
            fro = float(np.sqrt(np.einsum('ij,ij->', delta, delta, dtype=np.float64)))
            np.abs(delta, out=delta)
            max_abs = float(np.max(delta))
            mean_abs = float(np.mean(delta, dtype=np.float64))
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.warning("compute_norm_features: exact delta failed: %s", exc)
            return None
    else:
        # Cholesky path: compute fro without materialising (out × in) ΔW.
        # max_abs and mean_abs require full materialisation — skipped here.
        logger.debug(
            "compute_norm_features: large delta (%d × %d = %d elements) — using Cholesky path",
            out_feats, in_feats, delta_numel,
        )
        try:
            aat = a @ a.T  # (rank, rank) — cheap
            L = np.linalg.cholesky(aat + 1e-12 * np.eye(rank))
            bl = b @ L  # (out_features, rank) — feasible
            fro = float(np.sqrt(np.sum(bl * bl)))
        except (np.linalg.LinAlgError, ValueError):
            # Last resort: submultiplicativity bound
            fro = norm_a * norm_b
        max_abs = 0.0
        mean_abs = 0.0

    denom = norm_a * norm_b
    ratio = fro / denom if denom > 1e-12 else 0.0

    return NormFeatures(
        fro_norm_delta=fro,
        max_abs_delta=max_abs,
        mean_abs_delta=mean_abs,
        delta_norm_ratio=ratio,
    )


def compute_effective_rank_ratio(
    tensor_A: np.ndarray,
    claimed_rank: int | None,
) -> float | None:
    """Return effective_rank / claimed_rank, or None if claimed_rank is unknown.

    A ratio >> 1 suggests the adapter encodes more directions than declared,
    which is the rank-inflation anomaly pattern.

    Args:
        tensor_A: lora_A weight matrix used for SVD.
        claimed_rank: Rank r from adapter metadata, or None.

    Returns:
        Float ratio, or None when claimed_rank is None or zero.
    """
    if not claimed_rank:
        return None

    from adaptersentry.features.tensor_stats import compute_svd_stats
    svd = compute_svd_stats(tensor_A)
    eff = svd.get("effective_rank", 0)
    return float(eff) / claimed_rank
