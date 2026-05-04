"""Tests for M1 Static Analyzer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.analyzer import (
    _group_lora_layers,
    _metadata_depth,
    _risk_level,
    _score_from_flags,
    analyze,
    compute_svd_stats,
    compute_tensor_stats,
    detect_layer_anomalies,
    load_adapter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_adapter(
    tmp_path: Path,
    layers: dict[str, tuple[np.ndarray, np.ndarray]],
    metadata: dict[str, str] | None = None,
    filename: str = "adapter.safetensors",
) -> Path:
    """Write a synthetic .safetensors adapter to tmp_path and return its path.

    Args:
        tmp_path: Temporary directory provided by pytest.
        layers: Mapping of layer_name -> (lora_A_tensor, lora_B_tensor).
        metadata: Optional string-valued metadata dict.
        filename: Output filename.

    Returns:
        Path to the created .safetensors file.
    """
    tensors: dict[str, np.ndarray] = {}
    for layer_name, (a, b) in layers.items():
        tensors[f"{layer_name}.lora_A.weight"] = a.astype(np.float32)
        tensors[f"{layer_name}.lora_B.weight"] = b.astype(np.float32)

    path = tmp_path / filename
    save_file(tensors, str(path), metadata=metadata or {})
    return path


@pytest.fixture()
def clean_adapter(tmp_path: Path) -> Path:
    """A well-behaved adapter with two normal-distribution LoRA layers."""
    rng = np.random.default_rng(42)
    layers = {
        "model.layers.0.self_attn.q_proj": (
            rng.standard_normal((8, 64)),
            rng.standard_normal((64, 8)),
        ),
        "model.layers.0.self_attn.v_proj": (
            rng.standard_normal((8, 64)),
            rng.standard_normal((64, 8)),
        ),
    }
    return _make_adapter(tmp_path, layers, metadata={"r": "8"})


@pytest.fixture()
def high_kurtosis_adapter(tmp_path: Path) -> Path:
    """An adapter with heavy-tailed (high-kurtosis) A weights in one layer."""
    rng = np.random.default_rng(0)
    normal_a = rng.standard_normal((8, 64))
    normal_b = rng.standard_normal((64, 8))

    # Laplace distribution has kurtosis=3 (Fisher), high enough to trigger flag
    # Use a sparse tensor instead: mostly zeros, a few large values
    sparse = np.zeros((8, 64), dtype=np.float32)
    sparse[0, 0] = 100.0
    sparse[3, 5] = -100.0

    layers = {
        "model.layers.0.self_attn.q_proj": (sparse, normal_b),
        "model.layers.0.self_attn.v_proj": (normal_a, normal_b),
    }
    return _make_adapter(tmp_path, layers)


@pytest.fixture()
def near_zero_b_adapter(tmp_path: Path) -> Path:
    """An adapter where all lora_B weights are essentially zero."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal((8, 64))
    b = np.zeros((64, 8), dtype=np.float32)  # typical init, suspicious post-train
    layers = {"model.layers.0.self_attn.q_proj": (a, b)}
    return _make_adapter(tmp_path, layers)


@pytest.fixture()
def high_risk_module_adapter(tmp_path: Path) -> Path:
    """An adapter targeting embed_tokens — a high-risk module."""
    rng = np.random.default_rng(3)
    layers = {
        "model.embed_tokens": (
            rng.standard_normal((8, 64)),
            rng.standard_normal((64, 8)),
        )
    }
    return _make_adapter(tmp_path, layers)


# ---------------------------------------------------------------------------
# load_adapter
# ---------------------------------------------------------------------------


class TestLoadAdapter:
    def test_loads_valid_file(self, clean_adapter: Path) -> None:
        tensors, metadata = load_adapter(clean_adapter)
        assert len(tensors) == 4  # 2 layers × 2 matrices
        assert isinstance(metadata, dict)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_adapter(tmp_path / "nonexistent.safetensors")

    def test_wrong_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "adapter.pt"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match=".safetensors"):
            load_adapter(p)

    def test_metadata_preserved(self, clean_adapter: Path) -> None:
        _, metadata = load_adapter(clean_adapter)
        assert metadata.get("r") == "8"


# ---------------------------------------------------------------------------
# _group_lora_layers
# ---------------------------------------------------------------------------


class TestGroupLoraLayers:
    def test_pairs_a_and_b(self) -> None:
        rng = np.random.default_rng(0)
        tensors = {
            "model.layers.0.q_proj.lora_A.weight": rng.standard_normal((4, 32)),
            "model.layers.0.q_proj.lora_B.weight": rng.standard_normal((32, 4)),
        }
        groups = _group_lora_layers(tensors)
        assert "model.layers.0.q_proj" in groups
        assert "A" in groups["model.layers.0.q_proj"]
        assert "B" in groups["model.layers.0.q_proj"]

    def test_ignores_non_lora_keys(self) -> None:
        tensors = {"model.embed_tokens.weight": np.zeros((10, 10))}
        assert _group_lora_layers(tensors) == {}

    def test_multiple_layers(self) -> None:
        rng = np.random.default_rng(1)
        tensors = {
            "base.q_proj.lora_A.weight": rng.standard_normal((4, 8)),
            "base.q_proj.lora_B.weight": rng.standard_normal((8, 4)),
            "base.v_proj.lora_A.weight": rng.standard_normal((4, 8)),
            "base.v_proj.lora_B.weight": rng.standard_normal((8, 4)),
        }
        groups = _group_lora_layers(tensors)
        assert len(groups) == 2


# ---------------------------------------------------------------------------
# compute_tensor_stats
# ---------------------------------------------------------------------------


class TestComputeTensorStats:
    def test_returns_required_keys(self) -> None:
        t = np.random.default_rng(0).standard_normal((8, 64))
        stats = compute_tensor_stats(t)
        assert set(stats) >= {"mean", "std", "kurtosis", "skewness"}

    def test_normal_distribution_kurtosis_near_zero(self) -> None:
        rng = np.random.default_rng(42)
        t = rng.standard_normal(10_000)
        stats = compute_tensor_stats(t)
        assert abs(stats["kurtosis"]) < 0.5  # excess kurtosis ≈ 0 for Gaussian

    def test_zero_tensor_has_zero_std(self) -> None:
        stats = compute_tensor_stats(np.zeros((4, 4)))
        assert stats["std"] == pytest.approx(0.0)

    def test_accepts_any_shape(self) -> None:
        for shape in [(10,), (4, 8), (2, 4, 8)]:
            compute_tensor_stats(np.ones(shape))  # should not raise


# ---------------------------------------------------------------------------
# compute_svd_stats
# ---------------------------------------------------------------------------


class TestComputeSvdStats:
    def test_returns_required_keys(self) -> None:
        svd = compute_svd_stats(np.random.default_rng(0).standard_normal((8, 64)))
        assert set(svd) >= {"effective_rank", "energy_concentration", "singular_values_count"}

    def test_rank_one_matrix_high_energy_concentration(self) -> None:
        # A rank-1 matrix: all energy in the first singular value
        a = np.outer(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0, 7.0]))
        svd = compute_svd_stats(a)
        assert svd["energy_concentration"] == pytest.approx(1.0, abs=1e-6)
        assert svd["effective_rank"] == 1

    def test_identity_matrix_energy_spread(self) -> None:
        # Identity: energy spread evenly, concentration should be low
        svd = compute_svd_stats(np.eye(16))
        assert svd["energy_concentration"] == pytest.approx(1.0 / 16, abs=1e-6)

    def test_zero_matrix_returns_zeros(self) -> None:
        svd = compute_svd_stats(np.zeros((4, 4)))
        assert svd["effective_rank"] == 0
        assert svd["energy_concentration"] == 0.0

    def test_1d_input_handled(self) -> None:
        svd = compute_svd_stats(np.array([1.0, 2.0, 3.0]))
        assert svd["effective_rank"] >= 1


# ---------------------------------------------------------------------------
# detect_layer_anomalies
# ---------------------------------------------------------------------------


class TestDetectLayerAnomalies:
    def _normal_stats(self) -> dict[str, float]:
        return {"mean": 0.0, "std": 0.01, "kurtosis": 0.1, "skewness": 0.0}

    def _normal_svd(self) -> dict[str, Any]:
        return {"effective_rank": 8, "energy_concentration": 0.1, "singular_values_count": 8}

    def test_clean_layer_no_flags(self) -> None:
        flags = detect_layer_anomalies(
            "model.layers.0.q_proj",
            self._normal_stats(),
            self._normal_stats(),
            self._normal_svd(),
            claimed_rank=8,
        )
        assert flags == []

    def test_high_kurtosis_a_flagged(self) -> None:
        stats_A = {**self._normal_stats(), "kurtosis": 15.0}
        flags = detect_layer_anomalies(
            "model.layers.0.q_proj", stats_A, self._normal_stats(), self._normal_svd(), None
        )
        assert any("HIGH_KURTOSIS_A" in f for f in flags)

    def test_high_energy_concentration_flagged(self) -> None:
        svd = {**self._normal_svd(), "energy_concentration": 0.98}
        flags = detect_layer_anomalies(
            "model.layers.0.q_proj",
            self._normal_stats(),
            self._normal_stats(),
            svd,
            claimed_rank=8,
        )
        assert any("HIGH_ENERGY_CONCENTRATION" in f for f in flags)

    def test_rank_inflation_flagged(self) -> None:
        svd = {**self._normal_svd(), "effective_rank": 256}
        flags = detect_layer_anomalies(
            "model.layers.0.q_proj",
            self._normal_stats(),
            self._normal_stats(),
            svd,
            claimed_rank=8,
        )
        assert any("RANK_INFLATION" in f for f in flags)

    def test_near_zero_b_flagged(self) -> None:
        stats_B = {**self._normal_stats(), "std": 0.0}
        flags = detect_layer_anomalies(
            "model.layers.0.q_proj",
            self._normal_stats(),
            stats_B,
            self._normal_svd(),
            None,
        )
        assert any("NEAR_ZERO_B_MATRIX" in f for f in flags)

    def test_high_risk_module_embed_tokens_flagged(self) -> None:
        flags = detect_layer_anomalies(
            "model.embed_tokens",
            self._normal_stats(),
            self._normal_stats(),
            self._normal_svd(),
            None,
        )
        assert any("HIGH_RISK_TARGET_MODULE" in f for f in flags)

    def test_high_risk_module_lm_head_flagged(self) -> None:
        flags = detect_layer_anomalies(
            "model.lm_head",
            self._normal_stats(),
            self._normal_stats(),
            self._normal_svd(),
            None,
        )
        assert any("HIGH_RISK_TARGET_MODULE" in f for f in flags)

    def test_no_rank_inflation_without_claimed_rank(self) -> None:
        svd = {**self._normal_svd(), "effective_rank": 512}
        flags = detect_layer_anomalies(
            "model.layers.0.q_proj",
            self._normal_stats(),
            self._normal_stats(),
            svd,
            claimed_rank=None,
        )
        assert not any("RANK_INFLATION" in f for f in flags)


# ---------------------------------------------------------------------------
# _metadata_depth
# ---------------------------------------------------------------------------


class TestMetadataDepth:
    def test_flat_dict(self) -> None:
        assert _metadata_depth({"a": "1", "b": "2"}) == 1

    def test_nested_dict(self) -> None:
        assert _metadata_depth({"a": {"b": {"c": "deep"}}}) == 3

    def test_empty(self) -> None:
        assert _metadata_depth({}) == 0

    def test_scalar(self) -> None:
        assert _metadata_depth("hello") == 0


# ---------------------------------------------------------------------------
# _score_from_flags / _risk_level
# ---------------------------------------------------------------------------


class TestScoring:
    def test_no_flags_scores_zero(self) -> None:
        assert _score_from_flags([]) == 0

    def test_score_capped_at_100(self) -> None:
        flags = ["RANK_INFLATION: x"] * 10
        assert _score_from_flags(flags) == 100

    def test_risk_level_thresholds(self) -> None:
        assert _risk_level(0) == "LOW"
        assert _risk_level(24) == "LOW"
        assert _risk_level(25) == "MEDIUM"
        assert _risk_level(49) == "MEDIUM"
        assert _risk_level(50) == "HIGH"
        assert _risk_level(74) == "HIGH"
        assert _risk_level(75) == "CRITICAL"
        assert _risk_level(100) == "CRITICAL"


# ---------------------------------------------------------------------------
# analyze (end-to-end)
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_schema_keys_present(self, clean_adapter: Path) -> None:
        report = analyze(clean_adapter)
        required = {
            "adapter_path", "timestamp", "overall_risk", "risk_level",
            "flags", "layers", "metadata", "summary",
        }
        assert required <= set(report)

    def test_clean_adapter_low_risk(self, clean_adapter: Path) -> None:
        report = analyze(clean_adapter)
        assert report["risk_level"] == "LOW"
        assert report["overall_risk"] == 0

    def test_layer_schema_keys(self, clean_adapter: Path) -> None:
        report = analyze(clean_adapter)
        for layer in report["layers"].values():
            assert "shape_A" in layer
            assert "shape_B" in layer
            assert "rank" in layer
            assert "energy_concentration" in layer
            assert "kurtosis_A" in layer
            assert "kurtosis_B" in layer
            assert "flags" in layer

    def test_near_zero_b_raises_flag(self, near_zero_b_adapter: Path) -> None:
        report = analyze(near_zero_b_adapter)
        assert any("NEAR_ZERO_B_MATRIX" in f for f in report["flags"])

    def test_high_risk_module_raises_flag(self, high_risk_module_adapter: Path) -> None:
        report = analyze(high_risk_module_adapter)
        assert any("HIGH_RISK_TARGET_MODULE" in f for f in report["flags"])
        assert report["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL")

    def test_claimed_rank_from_cli_overrides_metadata(self, clean_adapter: Path) -> None:
        # claimed_rank=1 vs effective_rank from 8-wide matrices should trigger inflation
        report = analyze(clean_adapter, claimed_rank=1)
        # Effective rank of an 8×64 normal matrix will be >> 2 (=1*2)
        assert any("RANK_INFLATION" in f for f in report["flags"])

    def test_report_is_json_serialisable(self, clean_adapter: Path) -> None:
        report = analyze(clean_adapter)
        serialised = json.dumps(report)
        assert json.loads(serialised) == report

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            analyze(tmp_path / "ghost.safetensors")

    def test_two_layers_reported(self, clean_adapter: Path) -> None:
        report = analyze(clean_adapter)
        assert len(report["layers"]) == 2

    # ------------------------------------------------------------------
    # Integration: new detector fields in layer reports
    # ------------------------------------------------------------------

    def test_layer_contains_entropy_fields(self, clean_adapter: Path) -> None:
        report = analyze(clean_adapter)
        for layer in report["layers"].values():
            assert "entropy_A" in layer
            assert "entropy_B" in layer
            assert 0.0 <= layer["entropy_A"] <= 1.0
            assert 0.0 <= layer["entropy_B"] <= 1.0

    def test_layer_contains_zscore_fields(self, clean_adapter: Path) -> None:
        report = analyze(clean_adapter)
        for layer in report["layers"].values():
            assert "zscore_outlier_rate_A" in layer
            assert "zscore_outlier_rate_B" in layer
            assert 0.0 <= layer["zscore_outlier_rate_A"] <= 1.0
            assert 0.0 <= layer["zscore_outlier_rate_B"] <= 1.0

    def test_layer_contains_isolation_score_field(self, clean_adapter: Path) -> None:
        report = analyze(clean_adapter)
        for layer in report["layers"].values():
            assert "isolation_score_A" in layer
            # Clean adapter: score should be present (not None) and reasonable
            score = layer["isolation_score_A"]
            assert score is not None
            assert isinstance(score, float)

    def test_near_zero_b_low_entropy_flagged(self, near_zero_b_adapter: Path) -> None:
        report = analyze(near_zero_b_adapter)
        all_flags = report["flags"]
        assert any("LOW_ENTROPY_B" in f for f in all_flags)

    def test_report_json_serialisable_with_new_fields(self, clean_adapter: Path) -> None:
        report = analyze(clean_adapter)
        import json as _json
        roundtrip = _json.loads(_json.dumps(report))
        assert roundtrip["layers"].keys() == report["layers"].keys()
