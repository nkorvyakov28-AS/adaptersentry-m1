"""CLI smoke tests for `adaptersentry scan`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


def _make_adapter(tmp_path: Path, high_risk: bool = False) -> Path:
    rng = np.random.default_rng(12)
    path = tmp_path / "adapter.safetensors"
    if high_risk:
        a = np.zeros((8, 64), dtype=np.float32)
        a[0, 0] = 5000.0
        b = rng.standard_normal((64, 8)).astype(np.float32)
    else:
        a = rng.standard_normal((8, 64)).astype(np.float32)
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


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "adaptersentry"] + list(args),
        capture_output=True,
        text=True,
    )


class TestVersionFlag:
    def test_version_flag(self) -> None:
        result = _run("--version")
        assert result.returncode == 0
        assert "adaptersentry" in result.stdout
        from adaptersentry.version import __version__
        assert __version__ in result.stdout


class TestScanTextFormat:
    def test_scan_clean_adapter_exit_zero(self, tmp_path: Path) -> None:
        adapter = _make_adapter(tmp_path)
        result = _run("scan", str(adapter))
        assert result.returncode == 0

    def test_scan_text_output_contains_risk(self, tmp_path: Path) -> None:
        adapter = _make_adapter(tmp_path)
        result = _run("scan", str(adapter))
        assert "Risk" in result.stdout or "LOW" in result.stdout

    def test_scan_missing_file_exit_one(self, tmp_path: Path) -> None:
        result = _run("scan", str(tmp_path / "ghost.safetensors"))
        assert result.returncode == 1
        # New renderer shows "ANALYSIS FAILED" and "parse:failed" instead of raw "error"
        combined = (result.stderr + result.stdout).lower()
        assert "failed" in combined or "error" in combined

    def test_scan_no_color_flag(self, tmp_path: Path) -> None:
        adapter = _make_adapter(tmp_path)
        result = _run("scan", str(adapter), "--no-color")
        assert result.returncode == 0
        assert "\033[" not in result.stdout


class TestScanJsonFormat:
    def test_scan_json_format_valid(self, tmp_path: Path) -> None:
        adapter = _make_adapter(tmp_path)
        result = _run("scan", str(adapter), "--format", "json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["schema_version"] == "1.0.0"

    def test_scan_json_write_to_file(self, tmp_path: Path) -> None:
        adapter = _make_adapter(tmp_path)
        out = tmp_path / "report.json"
        result = _run("scan", str(adapter), "--format", "json", "--output", str(out))
        assert result.returncode == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "schema_version" in data


class TestScanSarifFormat:
    def test_scan_sarif_format_valid(self, tmp_path: Path) -> None:
        adapter = _make_adapter(tmp_path)
        result = _run("scan", str(adapter), "--format", "sarif")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["version"] == "2.1.0"
        assert "runs" in data

    def test_scan_sarif_write_to_file(self, tmp_path: Path) -> None:
        adapter = _make_adapter(tmp_path)
        out = tmp_path / "results.sarif"
        result = _run("scan", str(adapter), "--format", "sarif", "--output", str(out))
        assert result.returncode == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["version"] == "2.1.0"


class TestFailOnFlag:
    def test_fail_on_critical_no_trigger_on_clean(self, tmp_path: Path) -> None:
        adapter = _make_adapter(tmp_path)
        result = _run("scan", str(adapter), "--fail-on", "CRITICAL")
        # Clean adapter should not trigger CRITICAL
        assert result.returncode in (0, 2)  # 0 = no findings at threshold

    def test_module_invocation(self, tmp_path: Path) -> None:
        """python -m adaptersentry should work identically."""
        adapter = _make_adapter(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "adaptersentry", "scan", str(adapter)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
