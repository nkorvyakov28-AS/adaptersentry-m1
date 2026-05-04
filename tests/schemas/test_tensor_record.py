"""Tests for TensorRecord schema."""

from __future__ import annotations

import json

import pytest

from adaptersentry.schemas.tensor_record import TensorRecord


def _layer_dict() -> dict:
    return {
        "shape_A": [8, 64],
        "shape_B": [64, 8],
        "rank": 8,
        "energy_concentration": 0.15,
        "kurtosis_A": 0.2,
        "kurtosis_B": 0.1,
        "mean_A": 0.0,
        "std_A": 0.01,
        "mean_B": 0.0,
        "std_B": 0.02,
        "skewness_A": -0.05,
        "entropy_A": 0.82,
        "entropy_B": 0.78,
        "zscore_outlier_rate_A": 0.002,
        "zscore_outlier_rate_B": 0.003,
        "isolation_score_A": 0.05,
        "flags": [],
    }


class TestTensorRecord:
    def test_from_layer_dict(self) -> None:
        rec = TensorRecord.from_layer_dict("model.layers.0.q_proj", _layer_dict())
        assert rec.layer_name == "model.layers.0.q_proj"
        assert rec.shape_a == [8, 64]
        assert rec.shape_b == [64, 8]
        assert rec.rank == 8

    def test_json_serializable(self) -> None:
        rec = TensorRecord.from_layer_dict("layer", _layer_dict())
        data = json.loads(rec.model_dump_json())
        assert data["layer_name"] == "layer"
        assert data["rank"] == 8
        assert isinstance(data["flags"], list)

    def test_isolation_score_optional(self) -> None:
        d = _layer_dict()
        del d["isolation_score_A"]
        rec = TensorRecord.from_layer_dict("layer", d)
        assert rec.isolation_score_a is None

    def test_flags_default_empty(self) -> None:
        d = _layer_dict()
        del d["flags"]
        rec = TensorRecord.from_layer_dict("layer", d)
        assert rec.flags == []

    def test_immutable(self) -> None:
        rec = TensorRecord.from_layer_dict("layer", _layer_dict())
        with pytest.raises(Exception):
            rec.rank = 99  # type: ignore[misc]

    def test_energy_concentration_range(self) -> None:
        rec = TensorRecord.from_layer_dict("layer", _layer_dict())
        assert 0.0 <= rec.energy_concentration <= 1.0

    def test_numel_is_sum_of_both_matrix_sizes(self) -> None:
        rec = TensorRecord.from_layer_dict("layer", _layer_dict())
        # shape_A=[8,64] → 512, shape_B=[64,8] → 512, total=1024
        assert rec.numel == 8 * 64 + 64 * 8

    def test_numel_zero_when_shapes_missing(self) -> None:
        d = _layer_dict()
        d["shape_A"] = []
        d["shape_B"] = []
        rec = TensorRecord.from_layer_dict("layer", d)
        assert rec.numel == 0
