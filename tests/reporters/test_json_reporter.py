"""Tests for the JSON reporter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.analyzer import scan
from adaptersentry.reporters import json as json_reporter


def _make_adapter(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
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


class TestJsonReporter:
    def test_render_returns_valid_json(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        output = json_reporter.render(report)
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_render_contains_schema_version(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        data = json.loads(json_reporter.render(report))
        assert data["schema_version"] == "1.0.0"

    def test_render_contains_required_keys(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        data = json.loads(json_reporter.render(report))
        for key in ("tool", "scan_target", "findings", "risk_summary", "tensor_records"):
            assert key in data, f"Missing key: {key}"

    def test_write_creates_file(self, tmp_path: Path) -> None:
        adapter = _make_adapter(tmp_path)
        report = scan(adapter)
        out = tmp_path / "report.json"
        json_reporter.write(report, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["schema_version"] == "1.0.0"

    def test_to_dict_returns_dict(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        d = json_reporter.to_dict(report)
        assert isinstance(d, dict)
        assert "risk_summary" in d

    def test_render_indentation(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        compact = json_reporter.render(report, indent=0)
        indented = json_reporter.render(report, indent=4)
        assert len(indented) > len(compact)
        # Both parse to equivalent dicts
        assert json.loads(compact) == json.loads(indented)
