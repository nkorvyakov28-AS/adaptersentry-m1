"""Tests for the SARIF 2.1.0 reporter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.analyzer import scan
from adaptersentry.reporters import sarif as sarif_reporter
from adaptersentry.schemas.adapter_report import (
    AdapterReport, AnalysisMode, RiskSummary, ScanTarget, ToolInfo, TrainingStatus,
)
from adaptersentry.schemas.adapter_metadata import AdapterMetadata
from adaptersentry.schemas.finding import Severity


def _make_adapter(tmp_path: Path, kind: str = "clean") -> Path:
    rng = np.random.default_rng(99)
    path = tmp_path / "adapter.safetensors"
    if kind == "clean":
        a = rng.standard_normal((8, 64)).astype(np.float32)
        b = rng.standard_normal((64, 8)).astype(np.float32)
    else:
        # high-kurtosis adapter
        a = np.zeros((8, 64), dtype=np.float32)
        a[0, 0] = 1000.0
        b = rng.standard_normal((64, 8)).astype(np.float32)
    save_file(
        {
            "model.layers.0.q_proj.lora_A.weight": a,
            "model.layers.0.q_proj.lora_B.weight": b,
        },
        str(path),
        metadata={"r": "8"},
    )
    return path


class TestSarifStructure:
    def test_top_level_keys(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        sarif = sarif_reporter.render(report)
        assert sarif["version"] == "2.1.0"
        assert "$schema" in sarif
        assert "runs" in sarif
        assert len(sarif["runs"]) == 1

    def test_tool_driver_populated(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        driver = sarif_reporter.render(report)["runs"][0]["tool"]["driver"]
        assert driver["name"] == "adaptersentry"
        assert "version" in driver
        assert "informationUri" in driver

    def test_artifacts_present(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        artifacts = sarif_reporter.render(report)["runs"][0]["artifacts"]
        assert len(artifacts) == 1
        assert "location" in artifacts[0]

    def test_results_list_exists(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        sarif = sarif_reporter.render(report)
        assert "results" in sarif["runs"][0]
        assert isinstance(sarif["runs"][0]["results"], list)

    def test_finding_has_rule_id(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path, kind="malicious"))
        sarif = sarif_reporter.render(report)
        results = sarif["runs"][0]["results"]
        if results:
            r = results[0]
            assert "ruleId" in r
            assert "level" in r
            assert "message" in r
            assert "locations" in r

    def test_level_values_valid(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path, kind="malicious"))
        sarif = sarif_reporter.render(report)
        valid_levels = {"error", "warning", "note", "none"}
        for result in sarif["runs"][0]["results"]:
            assert result["level"] in valid_levels

    def test_security_severity_in_results(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path, kind="malicious"))
        sarif = sarif_reporter.render(report)
        for result in sarif["runs"][0]["results"]:
            props = result.get("properties", {})
            # security-severity must be a string convertible to float in [0, 10]
            if "security-severity" in props:
                val = float(props["security-severity"])
                assert 0.0 <= val <= 10.0

    def test_rules_match_result_rule_ids(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path, kind="malicious"))
        sarif = sarif_reporter.render(report)
        run = sarif["runs"][0]
        rule_ids = {r["id"] for r in run["tool"]["driver"]["rules"]}
        result_ids = {r["ruleId"] for r in run["results"]}
        assert result_ids <= rule_ids, f"Results reference unknown rules: {result_ids - rule_ids}"

    def test_render_json_is_valid_json(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        output = sarif_reporter.render_json(report)
        assert isinstance(json.loads(output), dict)

    def test_write_creates_file(self, tmp_path: Path) -> None:
        report = scan(_make_adapter(tmp_path))
        out = tmp_path / "results.sarif"
        sarif_reporter.write(report, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["version"] == "2.1.0"
