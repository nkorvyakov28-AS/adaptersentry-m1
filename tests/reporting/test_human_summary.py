"""Tests for M1-RPT-02: render_human_summary CLI output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.reporting.human_summary import render_human_summary


def _make_adapter(tmp_path: Path, n_layers: int = 4, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        tensors[f"model.layers.{i}.q_proj.lora_A.weight"] = \
            rng.standard_normal((4, 32)).astype(np.float32)
        tensors[f"model.layers.{i}.q_proj.lora_B.weight"] = \
            rng.standard_normal((32, 4)).astype(np.float32)
    p = tmp_path / "adapter.safetensors"
    save_file(tensors, str(p))
    return p


def _make_report(tmp_path: Path, n_layers: int = 4):
    from adaptersentry.analyzer import scan
    return scan(_make_adapter(tmp_path, n_layers=n_layers))


# ---------------------------------------------------------------------------
# Basic rendering
# ---------------------------------------------------------------------------

class TestRenderHumanSummary:
    def test_returns_string(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report)
        assert isinstance(result, str)

    def test_nonempty(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report)
        assert len(result) > 100

    def test_ends_with_newline(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report)
        assert result.endswith("\n")

    def test_no_color_no_ansi(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True)
        assert "\033[" not in result

    def test_color_by_default_has_ansi(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=False)
        assert "\033[" in result


# ---------------------------------------------------------------------------
# Fixed block content
# ---------------------------------------------------------------------------

class TestCompactBlocks:
    def test_header_present(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True)
        assert "AdapterSentry M1" in result

    def test_verdict_block(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True)
        assert "VERDICT" in result

    def test_risk_level_present(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True)
        rs = report.risk_summary
        assert rs.ensemble_risk_level.value in result

    def test_confidence_present(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True)
        assert "Confidence:" in result

    def test_action_present(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True)
        assert "Action:" in result
        assert any(a in result for a in ("allow", "review", "block"))

    def test_top_signals_block(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True)
        assert "TOP RISK SIGNALS" in result

    def test_verbose_hint_in_compact(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True, verbose=False)
        assert "--verbose" in result

    def test_no_verbose_hint_in_verbose(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True, verbose=True)
        assert "--verbose" not in result

    def test_target_path_in_output(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True)
        assert "Target:" in result

    def test_layer_count_in_output(self, tmp_path):
        report = _make_report(tmp_path, n_layers=6)
        result = render_human_summary(report, no_color=True)
        assert "Layers:" in result


# ---------------------------------------------------------------------------
# Verbose sections
# ---------------------------------------------------------------------------

class TestVerboseBlocks:
    def test_score_breakdown_in_verbose(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True, verbose=True)
        assert "SCORE BREAKDOWN" in result

    def test_score_breakdown_has_all_families(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True, verbose=True)
        for family in ("parse", "metadata", "norm", "distribution", "entropy", "similarity"):
            assert family in result

    def test_score_breakdown_not_in_compact(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True, verbose=False)
        assert "SCORE BREAKDOWN" not in result

    def test_analysis_quality_in_verbose(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True, verbose=True)
        assert "ANALYSIS QUALITY" in result

    def test_analysis_quality_not_in_compact(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True, verbose=False)
        assert "ANALYSIS QUALITY" not in result

    def test_parse_coverage_shown(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True, verbose=True)
        assert "Parse coverage" in result

    def test_total_weighted_shown(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True, verbose=True)
        assert "Total weighted:" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_findings_shows_clean(self, tmp_path):
        report = _make_report(tmp_path)
        result = render_human_summary(report, no_color=True)
        if not report.findings:
            assert "No findings." in result

    def test_degraded_analysis_shown(self, tmp_path):
        from adaptersentry.schemas.adapter_report import AnalysisMode
        report = _make_report(tmp_path)
        if report.analysis_mode == AnalysisMode.DEGRADED:
            result = render_human_summary(report, no_color=True)
            assert "DEGRADED" in result

    def test_does_not_crash_on_empty_adapter(self, tmp_path):
        """Non-LoRA safetensors file — parse will fail gracefully."""
        rng = np.random.default_rng(0)
        p = tmp_path / "nonlora.safetensors"
        save_file({"dense.weight": rng.standard_normal((4, 8)).astype(np.float32)}, str(p))
        from adaptersentry.analyzer import scan
        report = scan(p)
        result = render_human_summary(report, no_color=True)
        assert "AdapterSentry M1" in result

    def test_verbose_does_not_raise(self, tmp_path):
        report = _make_report(tmp_path, n_layers=8)
        result = render_human_summary(report, no_color=True, verbose=True)
        assert len(result) > 200


# ---------------------------------------------------------------------------
# CLI --verbose flag integration
# ---------------------------------------------------------------------------

class TestCLIVerboseFlag:
    def _run_scan_text(self, adapter_path: Path, extra_args: list[str] = ()) -> str:
        import subprocess, sys
        cmd = [sys.executable, "-m", "adaptersentry", "scan", str(adapter_path),
               "--format", "text", "--no-color"] + list(extra_args)
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout + result.stderr

    def test_compact_output_no_verbose(self, tmp_path):
        p = _make_adapter(tmp_path)
        out = self._run_scan_text(p)
        assert "VERDICT" in out
        assert "SCORE BREAKDOWN" not in out

    def test_verbose_output_has_breakdown(self, tmp_path):
        p = _make_adapter(tmp_path)
        out = self._run_scan_text(p, ["--verbose"])
        assert "SCORE BREAKDOWN" in out
        assert "ANALYSIS QUALITY" in out
