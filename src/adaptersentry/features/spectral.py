"""ESR per-module features of the effective update ΔW = s·B·A.

All features are computed from the exact low-rank core (features.lowrank_core);
ΔW is never materialised. Shape features are invariant to the LoRA scale, the
learning rate and the number of training steps; only ``log_rms`` and
``log_sigma1`` carry magnitude, on a log scale so that a change of scale is a
shift rather than a blow-up.

    log_rms      log(‖ΔW‖_F / √(d·k))           update strength per entry
    log_sigma1   log σ₁                          strength of the leading direction
    top1_e       σ₁² / Σσ²                       share of the leading direction
    srank_r      (Σσ² / σ₁²) / r                 stable rank, normalised by rank
    pr_r         ((Σσ²)² / Σσ⁴) / r              participation ratio, normalised
    spec_h       −Σ e·log e / log r              spectral entropy of energy shares
                                                 (σ below the bf16 noise floor dropped)
    excess_conc  log(srank_null / srank)         concentration beyond a random
                                                 d×r null model (Marchenko–Pastur)
    kurt_dw      Pearson kurtosis of ΔW entries  rare, very large changes
    row_hoyer    Hoyer sparsity of row norms     update concentrated on few outputs
    row_gini     Gini of row norms               (tokens for lm_head / embeddings)
    row_top8     energy share of the 8 strongest rows
    col_hoyer    Hoyer sparsity of column norms  update reads from few inputs

For embedding LoRA PEFT applies (B·A)ᵀ, so the rows of ΔW (tokens) are the
columns of B·A; ``transposed=True`` swaps them.

Security Notes:
    - Pure numpy on validated, finite inputs; no I/O.
    - Memory O((d+k)·r) plus the kurtosis row chunks (~15 MB at 70B scale).
"""

from __future__ import annotations

import math

import numpy as np

from adaptersentry.features.lowrank_core import CoreStats, compute_core, kurtosis_dw
from adaptersentry.schemas.result import HeadRecord, ModuleFeatures

# Rows drawn for the kurtosis estimate (full / fast mode).
_KURTOSIS_ROWS_FULL = 256
_KURTOSIS_ROWS_FAST = 64

# Output-split projections (heads are row blocks of B) and input-split ones
# (heads are column blocks of A).
_HEAD_SPLIT_OUTPUT = frozenset({"q_proj", "k_proj", "v_proj", "query", "key", "value", "wq", "wk", "wv"})
_HEAD_SPLIT_INPUT = frozenset({"o_proj", "out_proj", "wo"})


def hoyer(x: np.ndarray) -> float | None:
    """Hoyer sparsity of a non-negative vector: 0 = uniform, 1 = one non-zero entry."""
    n = x.size
    l2 = float(np.linalg.norm(x))
    if n < 2 or l2 == 0.0:
        return None
    return float((math.sqrt(n) - float(np.sum(np.abs(x))) / l2) / (math.sqrt(n) - 1.0))


def gini(x: np.ndarray) -> float | None:
    """Gini coefficient of a non-negative vector: 0 = equal, → 1 = concentrated."""
    n = x.size
    total = float(np.sum(x))
    if n < 2 or total == 0.0:
        return None
    s = np.sort(np.abs(x).astype(np.float64))
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(np.sum((2.0 * i - n - 1.0) * s) / (n * total))


def top_k_share(x: np.ndarray, k: int = 8) -> float | None:
    """Share of total energy (Σx²) held by the k largest entries."""
    energy = x.astype(np.float64) ** 2
    total = float(np.sum(energy))
    if total == 0.0:
        return None
    kk = min(k, energy.size)
    return float(np.sum(np.partition(energy, energy.size - kk)[-kk:]) / total)


def _log(x: float) -> float | None:
    return math.log(x) if x > 0.0 else None


def features_from_core(
    core: CoreStats,
    *,
    kurtosis: float | None,
    transposed: bool = False,
) -> ModuleFeatures:
    """Build ModuleFeatures from an exact core (and an optional kurtosis estimate)."""
    if core.is_zero:
        return ModuleFeatures()
    d, k = core.shape
    r = core.rank
    sigma = core.sigma
    energy = sigma.astype(np.float64) ** 2
    total = float(energy.sum())
    shares = energy / total
    top1 = float(shares[0])
    srank = 1.0 / top1
    pr = 1.0 / float(np.sum(shares ** 2))

    sig = core.significant_sigma().astype(np.float64)
    if r > 1 and sig.size > 0:
        e_sig = sig ** 2 / float(np.sum(sig ** 2))
        e_sig = e_sig[e_sig > 0]
        spec_h = float(-np.sum(e_sig * np.log(e_sig)) / math.log(r))
    else:
        spec_h = 0.0

    srank_null = r / (1.0 + math.sqrt(r / max(min(d, k), 1))) ** 2

    rows, cols = (core.col_norms, core.row_norms) if transposed else (core.row_norms, core.col_norms)
    return ModuleFeatures(
        log_rms=_log(core.fro / math.sqrt(d * k)),
        log_sigma1=_log(float(sigma[0])),
        top1_e=top1,
        srank_r=srank / r,
        pr_r=pr / r,
        spec_h=spec_h,
        excess_conc=math.log(srank_null / srank),
        kurt_dw=kurtosis,
        row_hoyer=hoyer(rows),
        row_gini=gini(rows),
        row_top8=top_k_share(rows, 8),
        col_hoyer=hoyer(cols),
    )


def module_features(
    a: np.ndarray,
    b: np.ndarray,
    scale: float,
    *,
    transposed: bool = False,
    fast: bool = False,
) -> tuple[ModuleFeatures, CoreStats]:
    """ESR features of one LoRA module.

    Args:
        a, b: LoRA factors, A (r, k) and B (d, r), float32.
        scale: effective LoRA scale from resolve_scale().
        transposed: True for embedding LoRA (token rows are columns of B·A).
        fast: fewer kurtosis rows (64 instead of 256).

    Returns:
        (features, core). A zero update (untrained B) yields empty features.
    """
    core = compute_core(a, b, scale)
    kurt = None
    if not core.is_zero:
        n_rows = _KURTOSIS_ROWS_FAST if fast else _KURTOSIS_ROWS_FULL
        if transposed:
            # Token rows of ΔW are columns of B·A: estimate on the transpose.
            kurt = kurtosis_dw(b.T, a.T, scale, n_rows=n_rows)
        else:
            kurt = kurtosis_dw(a, b, scale, core=core, n_rows=n_rows)
    return features_from_core(core, kurtosis=kurt, transposed=transposed), core


def infer_head_dim(width: int) -> int | None:
    """Head size for splitting a projection of the given width, or None.

    Adapter files carry no head count. 128 is the head size of Llama-3, Qwen2.5,
    Mistral and Gemma-3; 64 and 256 cover older and Gemma-2 models. The result
    is a heuristic and heads are reported only on request.
    """
    for hd in (128, 64, 256):
        if width % hd == 0 and width // hd >= 2:
            return hd
    return None


def head_features(
    a: np.ndarray,
    b: np.ndarray,
    scale: float,
    module: str,
    *,
    head_dim: int | None = None,
) -> list[HeadRecord] | None:
    """Per-head features of an attention projection, or None if not applicable.

    q/k/v heads are row blocks of B (output split); o_proj heads are column
    blocks of A (input split). Kurtosis is omitted per head (cost); every other
    feature is exact.
    """
    if module in _HEAD_SPLIT_OUTPUT:
        width = b.shape[0]
    elif module in _HEAD_SPLIT_INPUT:
        width = a.shape[1]
    else:
        return None
    hd = head_dim or infer_head_dim(width)
    if hd is None or width % hd != 0:
        return None

    records: list[HeadRecord] = []
    for h in range(width // hd):
        sl = slice(h * hd, (h + 1) * hd)
        if module in _HEAD_SPLIT_OUTPUT:
            core = compute_core(a, b[sl], scale)
        else:
            core = compute_core(a[:, sl], b, scale)
        records.append(HeadRecord(head=h, features=features_from_core(core, kurtosis=None)))
    return records
