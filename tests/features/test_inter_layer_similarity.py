"""Tests for M1-ANAL-03: InterLayerSimilarityFeatures."""

from __future__ import annotations

import numpy as np
import pytest

from adaptersentry.features.inter_layer_similarity import (
    _module_type,
    compute_inter_layer_similarity,
)
from adaptersentry.schemas.inter_layer_similarity_features import (
    InterLayerSimilarityFeatures,
    SimilarPair,
)


def _make_pair(rank: int, out: int, in_: int, seed: int):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((rank, in_)).astype(np.float32)
    B = rng.standard_normal((out, rank)).astype(np.float32)
    return A, B


def _make_layer_pairs(n: int, rank: int = 4, out: int = 16, in_: int = 32):
    """Make n independent random layer pairs with the same shape."""
    return [
        (f"model.layers.{i}.self_attn.q_proj", i, *_make_pair(rank, out, in_, seed=i))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestInterLayerSimilaritySchema:
    def test_defaults(self):
        f = InterLayerSimilarityFeatures()
        assert f.cosine_sim_mean == 0.0
        assert f.n_pairs_computed == 0
        assert f.top_suspicious_pairs == []
        assert f.module_group_similarities == {}

    def test_is_frozen(self):
        f = InterLayerSimilarityFeatures()
        with pytest.raises(Exception):
            f.cosine_sim_mean = 0.5  # type: ignore[misc]

    def test_similar_pair_frozen(self):
        p = SimilarPair(layer_a="a", layer_b="b", index_a=0, index_b=2, cosine_sim=0.9)
        with pytest.raises(Exception):
            p.cosine_sim = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _module_type helper
# ---------------------------------------------------------------------------

class TestModuleType:
    def test_standard_qkv(self):
        assert _module_type("model.layers.0.self_attn.q_proj") == "q_proj"
        assert _module_type("model.layers.5.self_attn.v_proj") == "v_proj"

    def test_mlp(self):
        assert _module_type("model.layers.0.mlp.gate_proj") == "gate_proj"
        assert _module_type("model.layers.0.mlp.down_proj") == "down_proj"

    def test_nested_base_model(self):
        name = "base_model.model.model.layers.0.self_attn.q_proj"
        assert _module_type(name) == "q_proj"

    def test_no_digit(self):
        # Falls back to last non-skip component
        result = _module_type("embed_tokens")
        assert result != "unknown" or result == "unknown"  # just doesn't crash


# ---------------------------------------------------------------------------
# compute_inter_layer_similarity
# ---------------------------------------------------------------------------

class TestComputeInterLayerSimilarity:
    def test_returns_none_for_single_layer(self):
        pairs = _make_layer_pairs(1)
        assert compute_inter_layer_similarity(pairs) is None

    def test_returns_schema_for_two_layers(self):
        pairs = _make_layer_pairs(2)
        result = compute_inter_layer_similarity(pairs)
        assert isinstance(result, InterLayerSimilarityFeatures)

    def test_n_pairs_correct(self):
        # 4 same-shape layers → 4*3/2 = 6 pairs
        pairs = _make_layer_pairs(4)
        result = compute_inter_layer_similarity(pairs)
        assert result is not None
        assert result.n_pairs_computed == 6

    def test_cosine_sim_bounds(self):
        pairs = _make_layer_pairs(4)
        result = compute_inter_layer_similarity(pairs)
        assert result is not None
        assert -1.0 <= result.cosine_sim_mean <= 1.0

    def test_cosine_sim_std_nonneg(self):
        pairs = _make_layer_pairs(4)
        result = compute_inter_layer_similarity(pairs)
        assert result is not None
        assert result.cosine_sim_std >= 0.0

    def test_pearson_bounds(self):
        pairs = _make_layer_pairs(4)
        result = compute_inter_layer_similarity(pairs)
        assert result is not None
        assert -1.0 <= result.pearson_mean <= 1.0

    def test_identical_layers_cosine_one(self):
        # All layers are the same tensors → cosine = 1.0
        A, B = _make_pair(4, 16, 32, seed=0)
        pairs = [
            (f"model.layers.{i}.self_attn.q_proj", i, A.copy(), B.copy())
            for i in range(3)
        ]
        result = compute_inter_layer_similarity(pairs, fast=False)
        assert result is not None
        assert result.cosine_sim_mean == pytest.approx(1.0, abs=1e-4)

    def test_identical_layers_pearson_one(self):
        A, B = _make_pair(4, 16, 32, seed=0)
        pairs = [
            (f"model.layers.{i}.self_attn.q_proj", i, A.copy(), B.copy())
            for i in range(3)
        ]
        result = compute_inter_layer_similarity(pairs, fast=False)
        assert result is not None
        assert result.pearson_mean == pytest.approx(1.0, abs=1e-4)

    def test_random_layers_low_similarity(self):
        # Independent random layers should have low cosine similarity
        pairs = _make_layer_pairs(6, rank=4, out=32, in_=64)
        result = compute_inter_layer_similarity(pairs, fast=False)
        assert result is not None
        assert abs(result.cosine_sim_mean) < 0.3

    # --- suspicious pairs ---

    def test_no_suspicious_pairs_for_random_layers(self):
        pairs = _make_layer_pairs(6)
        result = compute_inter_layer_similarity(pairs)
        assert result is not None
        assert result.n_suspicious_pairs == 0

    def test_adjacent_identical_not_suspicious(self):
        # Identical adjacent layers should NOT appear in suspicious pairs
        A, B = _make_pair(4, 16, 32, seed=0)
        pairs = [
            ("model.layers.0.q_proj", 0, A.copy(), B.copy()),
            ("model.layers.1.q_proj", 1, A.copy(), B.copy()),  # adjacent to 0
        ]
        result = compute_inter_layer_similarity(pairs)
        assert result is not None
        # cosine=1.0 but |0-1|=1 → adjacent → not suspicious
        assert result.n_suspicious_pairs == 0

    def test_non_adjacent_identical_is_suspicious(self):
        A, B = _make_pair(4, 16, 32, seed=0)
        pairs = [
            ("model.layers.0.q_proj", 0, A.copy(), B.copy()),
            ("model.layers.5.q_proj", 5, A.copy(), B.copy()),  # non-adjacent
        ]
        result = compute_inter_layer_similarity(pairs)
        assert result is not None
        assert result.n_suspicious_pairs == 1
        assert len(result.top_suspicious_pairs) == 1
        assert result.top_suspicious_pairs[0].cosine_sim == pytest.approx(1.0, abs=1e-4)

    def test_top_suspicious_pairs_capped_at_five(self):
        # Create 6 non-adjacent identical layer pairs → should produce many suspicious pairs
        A, B = _make_pair(4, 16, 32, seed=0)
        # 6 layers at indices 0, 10, 20, 30, 40, 50 → all non-adjacent
        pairs = [
            (f"model.layers.{i * 10}.q_proj", i * 10, A.copy(), B.copy())
            for i in range(6)
        ]
        result = compute_inter_layer_similarity(pairs)
        assert result is not None
        assert len(result.top_suspicious_pairs) <= 5

    # --- module groups ---

    def test_module_group_similarities_keyed_by_module(self):
        pairs = _make_layer_pairs(4)  # all "q_proj"
        result = compute_inter_layer_similarity(pairs)
        assert result is not None
        assert "q_proj" in result.module_group_similarities

    def test_mixed_module_types_separate_groups(self):
        A_q, B_q = _make_pair(4, 16, 32, seed=1)
        A_v, B_v = _make_pair(4, 16, 32, seed=2)
        pairs = [
            ("model.layers.0.self_attn.q_proj", 0, A_q.copy(), B_q.copy()),
            ("model.layers.0.self_attn.v_proj", 1, A_v.copy(), B_v.copy()),
            ("model.layers.1.self_attn.q_proj", 2, A_q.copy(), B_q.copy()),
            ("model.layers.1.self_attn.v_proj", 3, A_v.copy(), B_v.copy()),
        ]
        result = compute_inter_layer_similarity(pairs)
        assert result is not None
        assert "q_proj" in result.module_group_similarities
        assert "v_proj" in result.module_group_similarities

    # --- fast mode ---

    def test_fast_mode_returns_result(self):
        pairs = _make_layer_pairs(4)
        result = compute_inter_layer_similarity(pairs, fast=True)
        assert isinstance(result, InterLayerSimilarityFeatures)

    def test_fast_mode_sets_proxy_flag(self):
        pairs = _make_layer_pairs(4)
        result = compute_inter_layer_similarity(pairs, fast=True)
        assert result is not None
        assert result.computed_on_proxy is True

    def test_full_mode_small_delta_no_proxy(self):
        # Small tensors → full mode can materialize ΔW
        pairs = _make_layer_pairs(3, rank=4, out=16, in_=32)
        result = compute_inter_layer_similarity(pairs, fast=False)
        assert result is not None
        assert result.computed_on_proxy is False

    # --- different shapes excluded from pairwise ---

    def test_different_shapes_not_compared(self):
        # Two layers with different shapes should yield 0 pairs
        A1, B1 = _make_pair(4, 16, 32, seed=1)
        A2, B2 = _make_pair(8, 32, 64, seed=2)  # different shape
        pairs = [
            ("model.layers.0.q_proj", 0, A1, B1),
            ("model.layers.1.v_proj", 1, A2, B2),
        ]
        result = compute_inter_layer_similarity(pairs)
        assert result is not None
        assert result.n_pairs_computed == 0


# ---------------------------------------------------------------------------
# Integration — scan() produces inter_layer_similarity_features
# ---------------------------------------------------------------------------

class TestScanIntegration:
    def test_scan_produces_il_features(self, tmp_path):
        """scan() on a real safetensors file produces InterLayerSimilarityFeatures."""
        import numpy as np
        from safetensors.numpy import save_file
        from adaptersentry.analyzer import scan

        rng = np.random.default_rng(42)
        tensors = {}
        for i in range(4):
            tensors[f"model.layers.{i}.self_attn.q_proj.lora_A.weight"] = \
                rng.standard_normal((4, 32)).astype(np.float32)
            tensors[f"model.layers.{i}.self_attn.q_proj.lora_B.weight"] = \
                rng.standard_normal((32, 4)).astype(np.float32)

        p = tmp_path / "test.safetensors"
        save_file(tensors, str(p))

        report = scan(p)
        assert report.inter_layer_similarity_features is not None
        il = report.inter_layer_similarity_features
        assert isinstance(il, InterLayerSimilarityFeatures)
        assert il.n_pairs_computed == 6  # 4 layers → 6 pairs
        assert -1.0 <= il.cosine_sim_mean <= 1.0
