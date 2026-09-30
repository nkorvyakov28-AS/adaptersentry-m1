"""Exact statistics of a LoRA update ΔW = s·B·A without materialising ΔW.

For a 70B-class MLP layer ΔW is 8192×28672 (235 M values); B·A is never formed.
Everything below goes through the r×r Gram matrices

    G_A = A·Aᵀ   (r×r)        G_B = Bᵀ·B   (r×r)

which cost O((d+k)·r²) and give exact results:

    singular values   σ²(ΔW) = s² · eig(Λ^½ Qᵀ G_B Q Λ^½),  G_A = Q Λ Qᵀ
    Frobenius norm    ‖ΔW‖²  = s² · Σ (G_A ∘ G_B)
    row norms         ‖row_i‖² = s² · b_iᵀ G_A b_i
    column norms      ‖col_j‖² = s² · a_jᵀ G_B a_j
    inner product     ⟨ΔW₁, ΔW₂⟩ = s₁s₂ · tr((B₁ᵀB₂)(A₂A₁ᵀ))
    entry mean        μ = s · (1ᵀB)(A·1) / (d·k)
    kurtosis          exact mean and variance; fourth moment by row importance sampling

The symmetric eigendecomposition of G_A is used instead of a Cholesky factor
so rank-deficient A (zero rows, duplicated directions) needs no fallback.
Accuracy is ~1e-7 relative on the leading singular values, far below the
~1e-3 rounding of weights stored in bfloat16 (``NOISE_FLOOR``).

Security Notes:
    - Pure numpy; no I/O. Inputs are finite float arrays already validated by
      the parser (non-finite layers never reach this module).
    - Memory is O((d+k)·r); nothing of size d×k is allocated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Singular values below this fraction of σ₁ are indistinguishable from bf16 rounding.
NOISE_FLOOR = 1e-3


@dataclass(frozen=True)
class CoreStats:
    """Exact statistics of ΔW = s·B·A.

    ``sigma`` holds all r singular values in descending order (zeros included).
    ``row_norms`` / ``col_norms`` are Euclidean norms of the rows (length d) and
    columns (length k) of ΔW.
    """

    sigma: np.ndarray
    fro: float
    row_norms: np.ndarray
    col_norms: np.ndarray
    shape: tuple[int, int]
    rank: int
    scale: float

    @property
    def is_zero(self) -> bool:
        return self.fro == 0.0

    def significant_sigma(self) -> np.ndarray:
        """Singular values above the bf16 noise floor (relative to σ₁)."""
        if self.sigma.size == 0 or self.sigma[0] == 0.0:
            return self.sigma[:0]
        return self.sigma[self.sigma >= NOISE_FLOOR * self.sigma[0]]


def _gram(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # float32 BLAS for the large products, float64 for everything r×r afterwards.
    g_a = (a @ a.T).astype(np.float64)
    g_b = (b.T @ b).astype(np.float64)
    return g_a, g_b


def compute_core(a: np.ndarray, b: np.ndarray, scale: float = 1.0) -> CoreStats:
    """Exact spectrum, norm, and row/column norms of ΔW = scale·B·A.

    Args:
        a: LoRA A, shape (r, k).
        b: LoRA B, shape (d, r).
        scale: effective LoRA scale s (alpha/r or alpha/sqrt(r)).

    Returns:
        CoreStats.

    Raises:
        ValueError: If shapes are not (r, k) and (d, r) with matching r.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError(f"expected A (r,k) and B (d,r); got A{a.shape} B{b.shape}")
    a32 = np.ascontiguousarray(a, dtype=np.float32)
    b32 = np.ascontiguousarray(b, dtype=np.float32)
    r = a32.shape[0]
    s2 = float(scale) ** 2

    g_a, g_b = _gram(a32, b32)
    lam, q = np.linalg.eigh(g_a)
    lam = np.clip(lam, 0.0, None)
    half = q * np.sqrt(lam)                      # Q Λ^½
    core = half.T @ g_b @ half                   # Λ^½ Qᵀ G_B Q Λ^½  (r×r, symmetric PSD)
    ev = np.clip(np.linalg.eigvalsh((core + core.T) * 0.5), 0.0, None)[::-1]
    sigma = np.sqrt(s2 * ev)

    fro2 = s2 * float(np.sum(g_a * g_b))
    row2 = s2 * np.einsum("ir,ir->i", b32 @ g_a.astype(np.float32), b32, dtype=np.float64)
    col2 = s2 * np.einsum("rj,rj->j", g_b.astype(np.float32) @ a32, a32, dtype=np.float64)

    return CoreStats(
        sigma=sigma,
        fro=float(np.sqrt(max(fro2, 0.0))),
        row_norms=np.sqrt(np.clip(row2, 0.0, None)),
        col_norms=np.sqrt(np.clip(col2, 0.0, None)),
        shape=(b32.shape[0], a32.shape[1]),
        rank=r,
        scale=float(scale),
    )


def delta_inner(
    a1: np.ndarray, b1: np.ndarray, s1: float,
    a2: np.ndarray, b2: np.ndarray, s2: float,
) -> float:
    """Exact Frobenius inner product ⟨s₁B₁A₁, s₂B₂A₂⟩ without forming either matrix.

    Both updates must have the same shape (d, k); ranks may differ.
    """
    if b1.shape[0] != b2.shape[0] or a1.shape[1] != a2.shape[1]:
        raise ValueError(f"shape mismatch: {b1.shape[0]}x{a1.shape[1]} vs {b2.shape[0]}x{a2.shape[1]}")
    cross_b = (b1.T.astype(np.float32) @ b2.astype(np.float32)).astype(np.float64)   # r1×r2
    cross_a = (a2.astype(np.float32) @ a1.T.astype(np.float32)).astype(np.float64)   # r2×r1
    return float(s1) * float(s2) * float(np.trace(cross_b @ cross_a))


def delta_cosine(
    a1: np.ndarray, b1: np.ndarray, s1: float,
    a2: np.ndarray, b2: np.ndarray, s2: float,
) -> float | None:
    """Cosine similarity of two updates, or None if either is zero."""
    n1 = compute_core(a1, b1, s1).fro
    n2 = compute_core(a2, b2, s2).fro
    if n1 == 0.0 or n2 == 0.0:
        return None
    return delta_inner(a1, b1, s1, a2, b2, s2) / (n1 * n2)


def kurtosis_dw(
    a: np.ndarray,
    b: np.ndarray,
    scale: float = 1.0,
    *,
    core: CoreStats | None = None,
    n_rows: int = 256,
    chunk: int = 32,
    seed: int = 0,
) -> float | None:
    """Pearson kurtosis (normal = 3) of the entries of ΔW, by importance sampling.

    The mean and variance of ΔW's entries are exact (from the r×r core). Only
    the fourth moment is estimated: rows are drawn with probability
    proportional to their exact energy ‖row_i‖², each drawn row is computed in
    full (k values, O(k·r)), and its contribution is re-weighted by 1/p_i.
    The estimator is unbiased, and rows that carry a concentrated update — the
    case the feature exists for — are almost always drawn. Uniform sampling of
    rows or entries would miss e.g. 3 spiky rows out of 4096 most of the time.

    Args:
        a, b: LoRA factors, A (r, k) and B (d, r).
        scale: effective LoRA scale.
        core: precomputed compute_core(a, b, scale), to avoid recomputing norms.
        n_rows: rows drawn (with replacement). Cost O(n_rows·k·r).
        chunk: rows materialised at once; bounds memory to chunk·k floats.
        seed: fixed for reproducibility on the same file.

    Returns:
        Kurtosis, or None for a zero or constant update.
    """
    core = core if core is not None else compute_core(a, b, scale)
    d, k = core.shape
    n = float(d) * float(k)
    s = float(scale)

    b64_colsum = b.astype(np.float64).sum(axis=0)                  # 1ᵀB   (r,)
    a64_rowsum = a.astype(np.float64).sum(axis=1)                  # A·1   (r,)
    mu = s * float(b64_colsum @ a64_rowsum) / n                    # exact mean
    m2 = core.fro ** 2 / n - mu * mu                               # exact central 2nd moment
    if m2 <= 0.0:
        return None

    energy = core.row_norms.astype(np.float64) ** 2
    total = float(energy.sum())
    if total <= 0.0:
        return None
    p = energy / total
    rng = np.random.default_rng(seed)
    rows = rng.choice(d, size=n_rows, replace=True, p=p)

    a32 = np.ascontiguousarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    # Each distinct row is computed once; duplicates only change its weight.
    uniq, counts = np.unique(rows, return_counts=True)
    acc = 0.0
    for start in range(0, uniq.size, chunk):
        idx = uniq[start:start + chunk]
        block = b32[idx] @ a32                                      # (chunk, k), float32
        block *= np.float32(s)
        block -= np.float32(mu)
        np.square(block, out=block)
        per_row_m4 = np.einsum("ij,ij->i", block, block, dtype=np.float64)  # Σ_j (x_ij − μ)⁴
        acc += float(np.sum(counts[start:start + chunk] * per_row_m4 / p[idx]))
    m4 = acc / n_rows / n                                          # E over entries
    # Rows with zero energy (never drawn) still contribute μ⁴ each: add them exactly.
    zero_rows = int(np.count_nonzero(energy == 0.0))
    m4 += zero_rows * k * mu ** 4 / n
    return float(m4 / (m2 * m2))
