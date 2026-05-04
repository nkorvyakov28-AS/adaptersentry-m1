"""Tests for AdapterReport schema construction and serialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.analyzer import scan
from adaptersentry.schemas.adapter_report import (
    AdapterReport,
    AnalysisMode,
    RiskSummary,
    ScanTarget,
    ToolInfo,
    TrainingStatus,
)
from adaptersentry.schemas.adapter_metadata import AdapterMetadata
from adaptersentry.schemas.finding import Finding, Severity
from adaptersentry.schemas.errors import ScanError
from adaptersentry.schemas.tensor_record import TensorRecord


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_clean_adapter(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
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


def _minimal_report(version: str = "0.2.0") -> AdapterReport:
    """Build a minimal AdapterReport directly (no file I/O)."""
    from adaptersentry.version import __version__

    return AdapterReport(
        tool=ToolInfo(version=__version__),
        scan_target=ScanTarget(path="/tmp/adapter.safetensors"),
        adapter_metadata=AdapterMetadata(),
        tensor_records=[],
        findings=[],
        errors=[],
        risk_summary=RiskSummary(
            overall_risk=0,
            risk_level=Severity.LOW,
            ensemble_score=3.5,
            ensemble_risk_level=Severity.LOW,
            training_status=TrainingStatus.TRAINED,
            n_layers=0,
            n_findings=0,
            cross_layer_consistency=1.0,
        ),
        analysis_mode=AnalysisMode.FULL,
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
    )


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------


class TestAdapterReportConstruction:
    def test_schema_version_default(self) -> None:
        report = _minimal_report()
        assert report.schema_version == "1.0.0"

    def test_tool_info_populated(self) -> None:
        report = _minimal_report()
        assert report.tool.name == "adaptersentry"
        assert report.tool.module == "M1-StaticAnalyzer"

    def test_analysis_mode_full(self) -> None:
        assert _minimal_report().analysis_mode == AnalysisMode.FULL

    def test_risk_summary_fields(self) -> None:
        rs = _minimal_report().risk_summary
        assert rs.overall_risk == 0
        assert rs.risk_level == Severity.LOW
        assert 0.0 <= rs.ensemble_score <= 100.0


class TestAdapterReportSerialization:
    def test_json_roundtrip(self) -> None:
        report = _minimal_report()
        data = json.loads(report.model_dump_json())
        assert data["schema_version"] == "1.0.0"
        assert data["analysis_mode"] == "full"
        assert "risk_summary" in data

    def test_all_required_fields_present(self) -> None:
        data = json.loads(_minimal_report().model_dump_json())
        for field in (
            "schema_version", "tool", "scan_target", "adapter_metadata",
            "tensor_records", "findings", "errors", "risk_summary",
            "analysis_mode", "started_at", "completed_at",
        ):
            assert field in data, f"Missing field: {field}"

    def test_no_custom_json_hacks_needed(self) -> None:
        # Pydantic model_dump_json should work without any custom serializer
        report = _minimal_report()
        serialized = report.model_dump_json()
        assert isinstance(serialized, str)
        assert json.loads(serialized) is not None


# ---------------------------------------------------------------------------
# Integration via scan()
# ---------------------------------------------------------------------------


class TestScanReturnsAdapterReport:
    def test_scan_returns_adapter_report(self, tmp_path: Path) -> None:
        path = _make_clean_adapter(tmp_path)
        report = scan(path)
        assert isinstance(report, AdapterReport)

    def test_scan_report_json_serializable(self, tmp_path: Path) -> None:
        path = _make_clean_adapter(tmp_path)
        report = scan(path)
        data = json.loads(report.model_dump_json())
        assert data["schema_version"] == "1.0.0"

    def test_scan_report_has_tensor_records(self, tmp_path: Path) -> None:
        path = _make_clean_adapter(tmp_path)
        report = scan(path)
        assert len(report.tensor_records) == 1  # one layer

    def test_scan_report_risk_summary_in_range(self, tmp_path: Path) -> None:
        path = _make_clean_adapter(tmp_path)
        report = scan(path)
        rs = report.risk_summary
        assert 0 <= rs.overall_risk <= 100
        assert 0.0 <= rs.ensemble_score <= 100.0

    def test_scan_report_training_status(self, tmp_path: Path) -> None:
        path = _make_clean_adapter(tmp_path)
        report = scan(path)
        assert report.risk_summary.training_status == TrainingStatus.TRAINED

    def test_to_legacy_dict_has_required_keys(self, tmp_path: Path) -> None:
        path = _make_clean_adapter(tmp_path)
        report = scan(path)
        legacy = report.to_legacy_dict()
        required = {"adapter_path", "timestamp", "overall_risk", "risk_level",
                    "flags", "layers", "metadata", "summary"}
        assert required <= set(legacy), f"Missing: {required - set(legacy)}"

    def test_missing_file_returns_failed_report(self, tmp_path: Path) -> None:
        from adaptersentry.analyzer import scan
        from adaptersentry.schemas.adapter_report import AnalysisMode, ParseStatus
        report = scan(tmp_path / "ghost.safetensors")
        assert report.parse_status == ParseStatus.FAILED
        assert report.analysis_mode == AnalysisMode.FAILED
        assert len(report.errors) == 1
        assert report.errors[0].code == "INVALID_SAFETENSORS"
        assert report.tensor_records == []
