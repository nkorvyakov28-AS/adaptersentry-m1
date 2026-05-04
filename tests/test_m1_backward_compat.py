"""Backward compatibility tests.

These tests guarantee that the dict API produced by ``analyze()`` remains
stable across refactoring.  Any change to the output format of ``analyze()``
must be justified and these tests must pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.analyzer import analyze


def _make_adapter(tmp_path: Path) -> Path:
    rng = np.random.default_rng(99)
    path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "model.layers.0.q_proj.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
            "model.layers.0.q_proj.lora_B.weight": rng.standard_normal((64, 8)).astype(np.float32),
        },
        str(path),
        metadata={"r": "8"},
    )
    return path


REQUIRED_REPORT_KEYS = {
    "adapter_path",
    "timestamp",
    "overall_risk",
    "risk_level",
    "flags",
    "layers",
    "metadata",
    "summary",
    "ensemble_score",
    "ensemble_risk_level",
    "training_status",
    "false_positive_suppressed",
    "cross_layer_consistency",
    "wasserstein_distances",
}

REQUIRED_LAYER_KEYS = {
    "shape_A",
    "shape_B",
    "rank",
    "energy_concentration",
    "kurtosis_A",
    "kurtosis_B",
    "mean_A",
    "std_A",
    "mean_B",
    "std_B",
    "skewness_A",
    "entropy_A",
    "entropy_B",
    "zscore_outlier_rate_A",
    "zscore_outlier_rate_B",
    "isolation_score_A",
    "flags",
}


class TestAnalyzeDictContract:
    def test_all_required_top_level_keys_present(self, tmp_path: Path) -> None:
        report = analyze(_make_adapter(tmp_path))
        missing = REQUIRED_REPORT_KEYS - set(report)
        assert not missing, f"Missing keys: {missing}"

    def test_all_required_layer_keys_present(self, tmp_path: Path) -> None:
        report = analyze(_make_adapter(tmp_path))
        for lname, layer in report["layers"].items():
            missing = REQUIRED_LAYER_KEYS - set(layer)
            assert not missing, f"Layer {lname!r} missing keys: {missing}"

    def test_overall_risk_is_int_in_range(self, tmp_path: Path) -> None:
        report = analyze(_make_adapter(tmp_path))
        assert isinstance(report["overall_risk"], int)
        assert 0 <= report["overall_risk"] <= 100

    def test_ensemble_score_is_float_in_range(self, tmp_path: Path) -> None:
        report = analyze(_make_adapter(tmp_path))
        assert isinstance(report["ensemble_score"], float)
        assert 0.0 <= report["ensemble_score"] <= 100.0

    def test_risk_level_is_valid_string(self, tmp_path: Path) -> None:
        report = analyze(_make_adapter(tmp_path))
        assert report["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_training_status_is_valid_string(self, tmp_path: Path) -> None:
        report = analyze(_make_adapter(tmp_path))
        assert report["training_status"] in ("TRAINED", "INIT_ONLY", "PARTIALLY_TRAINED")

    def test_report_is_json_serializable(self, tmp_path: Path) -> None:
        report = analyze(_make_adapter(tmp_path))
        serialized = json.dumps(report)
        assert json.loads(serialized) == report

    def test_flags_is_list_of_strings(self, tmp_path: Path) -> None:
        report = analyze(_make_adapter(tmp_path))
        assert isinstance(report["flags"], list)
        for flag in report["flags"]:
            assert isinstance(flag, str)

    def test_layers_is_dict(self, tmp_path: Path) -> None:
        report = analyze(_make_adapter(tmp_path))
        assert isinstance(report["layers"], dict)

    def test_clean_adapter_low_risk(self, tmp_path: Path) -> None:
        report = analyze(_make_adapter(tmp_path))
        assert report["risk_level"] == "LOW"
        assert report["overall_risk"] == 0

    def test_summary_string_is_present(self, tmp_path: Path) -> None:
        report = analyze(_make_adapter(tmp_path))
        assert isinstance(report["summary"], str)
        assert "Risk:" in report["summary"]
