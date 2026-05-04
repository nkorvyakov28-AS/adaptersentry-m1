"""Inter-layer ΔW similarity analysis for LoRA adapters (M1-ANAL-03).

Computes pairwise cosine similarity and Pearson correlation between ΔW matrices
across all adapter layers. Groups by module type for reporting; restricts
pairwise comparisons to same-(A_shape, B_shape) layers for vector compatibility.

fast mode: lora_A rows used as ΔW proxy — O(n) per layer, no matmul.
full mode: ΔW = B @ A materialized when out × in ≤ _MAX_DELTA_NUMEL.

Security Notes:
    Pure numpy computation; no I/O, no eval/exec, no pickle.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np

logger = logging.getLogger(__name__)

_MAX_DELTA_NUMEL = 16_000_000
_SUSPICIOUS_COSINE_THRESHOLD = 0.85
_MAX_LAYERS_PER_GROUP = 100  # guard: cap pairwise work per shape group
_TOP_K_PAIRS = 5
# Full mode: stride-sample ΔW to this many elements before pairwise comparison.
# Rationale: dot-product on 4.3M float64 vectors = ~34ms/pair × 4950 pairs = 168s.
# At 10K elements: ~0.02ms/pair × 4950 = 0.9s, negligible precision loss for
# cosine > 0.85 detection. Memory: 80KB/layer vs 34MB/layer.
_MAX_VEC_ELEMENTS = 10_000


def _module_type(layer_name: str) -> str:
    """Extract the leaf module type (e.g. 'q_proj') from a LoRA layer path.

    Returns the last non-suffix, non-digit path component. For a name like
    'model.layers.0.self_attn.q_proj', this returns 'q_proj'.
    """
    skip = {"lora_A", "lora_B", "weight", "default", "base_model", "model"}
    parts = layer_name.split(".")
    for p in reversed(parts):
        if p not in skip and not p.isdigit():
            return p
    return "unknown"


def _make_vector(
    tensor_A: np.ndarray,
    tensor_B: np.ndarray,
    *,
    fast: bool,
) -> tuple[np.ndarray, bool]:
    """Return (flat_vector, used_proxy) for one layer pair."""
    try:
        a = tensor_A.astype(np.float64)
        b = tensor_B.astype(np.float64)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(-1, 1)
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
            return a.flatten(), True
        out_feats, in_feats = b.shape[0], a.shape[1]
        if not fast and out_feats * in_feats <= _MAX_DELTA_NUMEL:
            delta = (b @ a).flatten()
            if delta.size > _MAX_VEC_ELEMENTS:
                stride = max(1, delta.size // _MAX_VEC_ELEMENTS)
                # .copy() breaks the reference to the 83MB flatten buffer so it can be
                # freed immediately. Without it, the stride VIEW keeps the full ΔW alive
                # for the entire batch: 72 layers × 83MB = 6GB accumulation.
                return delta[::stride][:_MAX_VEC_ELEMENTS].copy(), False
            return delta, False
        # Fast mode proxy: lora_A rows. Cap to _MAX_VEC_ELEMENTS so that
        # pairwise dot products stay O(1ms) not O(108ms) for large adapters
        # (77K-element vectors → 108ms/pair × 4950 pairs = 535s).
        proxy = a.flatten()
        if proxy.size > _MAX_VEC_ELEMENTS:
            stride = max(1, proxy.size // _MAX_VEC_ELEMENTS)
            return proxy[::stride][:_MAX_VEC_ELEMENTS].copy(), True
        return proxy, True
    except Exception:
        return tensor_A.astype(np.float64).flatten(), True


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.sqrt(np.dot(u, u)), np.sqrt(np.dot(v, v))
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))


def _pearson(u: np.ndarray, v: np.ndarray) -> float | None:
    uc, vc = u - u.mean(), v - v.mean()
    nu, nv = np.sqrt(np.dot(uc, uc)), np.sqrt(np.dot(vc, vc))
    if nu < 1e-12 or nv < 1e-12:
        return None
    return float(np.clip(np.dot(uc, vc) / (nu * nv), -1.0, 1.0))


def compute_inter_layer_similarity(
    layer_pairs: list[tuple[str, int, np.ndarray, np.ndarray]],
    *,
    fast: bool = False,
) -> "InterLayerSimilarityFeatures | None":
    """Compute InterLayerSimilarityFeatures across all adapter layers.

    Args:
        layer_pairs: List of (layer_name, global_index, tensor_A, tensor_B).
                     global_index is the layer's position in the adapter's
                     original layer order — used to determine adjacency.
        fast:        If True, use lora_A rows as ΔW proxy (no matmul).

    Returns:
        InterLayerSimilarityFeatures, or None if fewer than 2 layers.
    """
    from adaptersentry.schemas.inter_layer_similarity_features import (
        InterLayerSimilarityFeatures,
        SimilarPair,
    )

    if len(layer_pairs) < 2:
        return None

    # Compute per-layer vectors and group by (A_shape, B_shape)
    LayerEntry = tuple  # (name, global_idx, module_type, vector, used_proxy)
    shape_groups: dict[tuple, list[LayerEntry]] = {}
    any_proxy = False

    for name, gidx, ta, tb in layer_pairs:
        vec, used_proxy = _make_vector(ta, tb, fast=fast)
        if used_proxy:
            any_proxy = True
        a_shape = tuple(ta.shape)
        b_shape = tuple(tb.shape)
        key = (a_shape, b_shape)
        mod = _module_type(name)
        shape_groups.setdefault(key, []).append((name, gidx, mod, vec, used_proxy))

    # Pairwise cosine + pearson within each shape group
    all_cosines: list[float] = []
    all_pearsons: list[float] = []
    all_suspicious: list[SimilarPair] = []
    module_cosines: dict[str, list[float]] = {}

    for entries in shape_groups.values():
        # Cap per-group size to bound O(n^2) work
        if len(entries) > _MAX_LAYERS_PER_GROUP:
            entries = entries[:_MAX_LAYERS_PER_GROUP]

        for (na, ia, ma, va, _), (nb, ib, mb, vb, _) in combinations(entries, 2):
            try:
                cos = _cosine(va, vb)
                pea = _pearson(va, vb)
            except Exception:
                continue

            all_cosines.append(cos)
            if pea is not None:
                all_pearsons.append(pea)

            # Track per-module-type similarity (use the pair's shared group if types match)
            for mod in (ma, mb):
                module_cosines.setdefault(mod, []).append(cos)

            # Non-adjacent suspicious pairs
            if abs(ia - ib) > 1 and cos >= _SUSPICIOUS_COSINE_THRESHOLD:
                all_suspicious.append(SimilarPair(
                    layer_a=na, layer_b=nb,
                    index_a=ia, index_b=ib,
                    cosine_sim=cos,
                    pearson=pea,
                ))

    if not all_cosines:
        return InterLayerSimilarityFeatures(computed_on_proxy=any_proxy)

    cosines_arr = np.array(all_cosines)
    top_suspicious = sorted(all_suspicious, key=lambda p: p.cosine_sim, reverse=True)[:_TOP_K_PAIRS]
    module_group_similarities = {
        mod: float(np.mean(vals))
        for mod, vals in module_cosines.items()
        if vals
    }

    return InterLayerSimilarityFeatures(
        cosine_sim_mean=float(np.mean(cosines_arr)),
        cosine_sim_std=float(np.std(cosines_arr)),
        pearson_mean=float(np.mean(all_pearsons)) if all_pearsons else 0.0,
        n_pairs_computed=len(all_cosines),
        n_suspicious_pairs=len(all_suspicious),
        top_suspicious_pairs=top_suspicious,
        module_group_similarities=module_group_similarities,
        computed_on_proxy=any_proxy,
    )
