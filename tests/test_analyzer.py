"""Tests for the new adaptersentry.analyzer module and scan() API."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.analyzer import analyze, scan
from adaptersentry.schemas.adapter_report import AdapterReport, AnalysisMode, TrainingStatus
from adaptersentry.schemas.finding import Severity


def _make_adapter(tmp_path: Path, **kw) -> Path:
    rng = np.random.default_rng(kw.get("seed", 42))
    path = tmp_path / kw.get("filename", "adapter.safetensors")
    layers = kw.get("layers", {
        "model.layers.0.q_proj": (
            rng.standard_normal((8, 64)).astype(np.float32),
            rng.standard_normal((64, 8)).astype(np.float32),
        ),
    })
    tensors = {}
    for lname, (a, b) in layers.items():
        tensors[f"{lname}.lora_A.weight"] = a
        tensors[f"{lname}.lora_B.weight"] = b
    save_file(tensors, str(path), metadata=kw.get("metadata", {"r": "8"}))
    return path


class TestScanAPI:
    def test_returns_adapter_report(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        report = scan(path)
        assert isinstance(report, AdapterReport)

    def test_schema_version(self, tmp_path: Path) -> None:
        assert scan(_make_adapter(tmp_path)).schema_version == "1.0.0"

    def test_tool_name(self, tmp_path: Path) -> None:
        assert scan(_make_adapter(tmp_path)).tool.name == "adaptersentry"

    def test_scan_target_path(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        report = scan(path)
        assert report.scan_target.path == str(path.resolve())

    def test_tensor_records_count(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        report = scan(path)
        assert len(report.tensor_records) == 1

    def test_clean_adapter_low_risk(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        assert report.risk_summary.risk_level == Severity.LOW

    def test_clean_adapter_trained_status(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        assert report.risk_summary.training_status == TrainingStatus.TRAINED

    def test_report_is_json_serializable(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        data = json.loads(report.model_dump_json())
        assert isinstance(data, dict)

    def test_missing_file_returns_failed_report(self, tmp_path: Path) -> None:
        from adaptersentry.schemas.adapter_report import AnalysisMode, ParseStatus
        report = scan(tmp_path / "ghost.safetensors")
        assert report.parse_status == ParseStatus.FAILED
        assert report.analysis_mode == AnalysisMode.FAILED
        assert report.tensor_records == []

    def test_claimed_rank_override(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        # Low claimed_rank forces RANK_INFLATION detection
        report = scan(path, claimed_rank=1)
        rule_ids = {f.rule_id for f in report.findings}
        assert any("RANK_INFLATION" in rid for rid in rule_ids)

    def test_findings_are_finding_objects(self, tmp_path: Path) -> None:
        from adaptersentry.schemas.finding import Finding
        report = scan(_make_adapter(tmp_path))
        for f in report.findings:
            assert isinstance(f, Finding)

    def test_analysis_mode_full_on_clean(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        assert report.analysis_mode == AnalysisMode.FULL

    def test_legacy_dict_from_scan(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        legacy = report.to_legacy_dict()
        assert "overall_risk" in legacy
        assert "layers" in legacy
        assert "summary" in legacy
