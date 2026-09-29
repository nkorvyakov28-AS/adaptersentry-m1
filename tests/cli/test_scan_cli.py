"""CLI tests for `adaptersentry scan` (ScanResult 2.0.0 output, exit codes, flags)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from adaptersentry.schemas.result import load_scan_result
from tests.adapter_factory import write_adapter


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "adaptersentry", *args], capture_output=True, text=True)


class TestVersion:
    def test_version_flag(self) -> None:
        from adaptersentry.version import __version__
        out = _run("--version")
        assert out.returncode == 0 and __version__ in out.stdout


class TestFormats:
    def test_text_default(self, tmp_path: Path) -> None:
        out = _run("scan", str(write_adapter(tmp_path / "a")))
        assert out.returncode == 0
        assert "Verdict:" in out.stdout and "ALLOW" in out.stdout
        assert "\x1b[" not in out.stdout  # not a tty → no colours

    def test_json_is_scan_result_2(self, tmp_path: Path) -> None:
        out = _run("scan", str(write_adapter(tmp_path / "a")), "--format", "json")
        result = load_scan_result(out.stdout)
        assert result.schema_version == "2.0.0" and result.modules is None
        assert result.artifact.provenance.path == "adapter_model.safetensors"

    def test_full_json_includes_modules(self, tmp_path: Path) -> None:
        out = _run("scan", str(write_adapter(tmp_path / "a")), "--format", "full-json")
        assert load_scan_result(out.stdout).modules

    def test_full_paths_flag(self, tmp_path: Path) -> None:
        path = write_adapter(tmp_path / "a")
        out = _run("scan", str(path), "--format", "json", "--full-paths")
        assert load_scan_result(out.stdout).artifact.provenance.path == str(path.resolve())

    def test_sarif(self, tmp_path: Path) -> None:
        out = _run("scan", str(write_adapter(tmp_path / "a", inject_layer=5)), "--format", "sarif")
        doc = json.loads(out.stdout)
        assert doc["version"] == "2.1.0"
        run = doc["runs"][0]
        assert run["properties"]["action"] == "review"
        assert any(r["ruleId"] == "ADAPTER_VERDICT" for r in run["results"])

    def test_output_file(self, tmp_path: Path) -> None:
        report = tmp_path / "report.json"
        out = _run("scan", str(write_adapter(tmp_path / "a")), "--format", "json", "--output", str(report))
        assert out.returncode == 0 and out.stdout == ""
        assert load_scan_result(report.read_text()).status == "ok"

    def test_unknown_format_rejected(self, tmp_path: Path) -> None:
        assert _run("scan", "x.safetensors", "--format", "debug-json").returncode != 0


class TestExitCodes:
    def test_failed_scan_exits_one_but_reports_review(self, tmp_path: Path) -> None:
        out = _run("scan", str(tmp_path / "missing.safetensors"), "--format", "json")
        assert out.returncode == 1
        assert load_scan_result(out.stdout).verdict.action == "review"

    def test_fail_on_review_triggers(self, tmp_path: Path) -> None:
        path = write_adapter(tmp_path / "a", inject_layer=5)
        assert _run("scan", str(path), "--fail-on", "review").returncode == 2
        assert _run("scan", str(path), "--fail-on", "block").returncode == 0

    def test_fail_on_clean_adapter(self, tmp_path: Path) -> None:
        assert _run("scan", str(write_adapter(tmp_path / "a")), "--fail-on", "review").returncode == 0

    def test_strict_policy_blocks_structural_red_flag(self, tmp_path: Path) -> None:
        path = write_adapter(tmp_path / "a", extra={"base_model.model.lm_head.weight": np.zeros((8, 256), np.float32)})
        assert _run("scan", str(path), "--fail-on", "block").returncode == 0
        assert _run("scan", str(path), "--fail-on", "block", "--policy", "strict").returncode == 2


class TestBatch:
    def test_batch_counts_verdicts_and_fail_on(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        write_adapter(corpus / "clean", seed=1, n_layers=6)
        write_adapter(corpus / "inj", seed=2, inject_layer=4)
        out = subprocess.run(
            [sys.executable, "-m", "adaptersentry", "batch", "--input-dir", str(corpus),
             "--output-dir", str(tmp_path / "results"), "--no-cache", "--workers", "1",
             "--run-id", "t1", "--fail-on", "review"],
            capture_output=True, text=True, env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        )
        assert out.returncode == 2, out.stderr
        summary = json.loads((tmp_path / "results" / "t1" / "run_summary.json").read_text())
        assert summary["verdicts"] == {"allow": 1, "review": 1, "block": 0}
