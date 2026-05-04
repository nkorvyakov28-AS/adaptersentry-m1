"""Tests for CARD-09 CLI output modes: summary-json, debug-json, text."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Use a real adapter fixture or a generated one
_ADAPTER = None  # resolved lazily


def _find_test_adapter() -> Path | None:
    """Return a .safetensors file in the test fixtures directory, if any."""
    for pattern in ("tests/fixtures/*.safetensors", "tests/**/*.safetensors"):
        hits = list(Path(".").glob(pattern))
        if hits:
            return hits[0]
    return None


def _run_scan(fmt: str, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Run adaptersentry scan --format FMT on the test adapter."""
    adapter = _find_test_adapter()
    if adapter is None:
        pytest.skip("No .safetensors test adapter available for CLI format tests")

    cmd = [
        sys.executable, "-m", "adaptersentry",
        "scan", str(adapter),
        "--format", fmt,
    ] + (extra_args or [])

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )


class TestFormatChoices:
    """Verify that format choices are registered correctly in the parser."""

    def test_unknown_format_exits_nonzero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "adaptersentry", "scan", "x.safetensors", "--format", "invalid"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode != 0

    def test_summary_json_is_a_valid_choice(self) -> None:
        from adaptersentry.cli.scan import _FORMATS
        assert "summary-json" in _FORMATS

    def test_debug_json_is_a_valid_choice(self) -> None:
        from adaptersentry.cli.scan import _FORMATS
        assert "debug-json" in _FORMATS

    def test_text_is_a_valid_choice(self) -> None:
        from adaptersentry.cli.scan import _FORMATS
        assert "text" in _FORMATS

    def test_json_legacy_alias_is_a_valid_choice(self) -> None:
        from adaptersentry.cli.scan import _FORMATS
        assert "json" in _FORMATS

    def test_sarif_is_a_valid_choice(self) -> None:
        from adaptersentry.cli.scan import _FORMATS
        assert "sarif" in _FORMATS


class TestBuildScanResult:
    """Unit-tests for the _build_scan_result helper."""

    @pytest.fixture
    def mock_adapter_report(self, tmp_path) -> object:
        """Build a minimal AdapterReport for testing."""
        import numpy as np
        from safetensors.numpy import save_file

        rng = np.random.default_rng(42)
        adapter_path = tmp_path / "test_adapter.safetensors"
        tensors = {
            "model.layers.0.self_attn.q_proj.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
            "model.layers.0.self_attn.q_proj.lora_B.weight": rng.standard_normal((64, 8)).astype(np.float32),
        }
        save_file(tensors, str(adapter_path))
        return adapter_path

    def test_summary_json_is_valid_json(self, mock_adapter_report) -> None:
        from adaptersentry.analyzer import scan
        from adaptersentry.cli.scan import _build_scan_result

        adapter_path = mock_adapter_report
        report = scan(adapter_path)
        result = _build_scan_result(adapter_path, report, None)
        data = json.loads(result.model_dump_json())
        assert isinstance(data, dict)
        assert "schema_version" in data

    def test_summary_json_has_scan_identity(self, mock_adapter_report) -> None:
        from adaptersentry.analyzer import scan
        from adaptersentry.cli.scan import _build_scan_result

        adapter_path = mock_adapter_report
        report = scan(adapter_path)
        result = _build_scan_result(adapter_path, report, None)
        data = json.loads(result.model_dump_json())
        assert "identity" in data
        assert "scan_id" in data["identity"]
        assert data["identity"]["scan_id"].startswith("sha256:")

    def test_summary_json_has_artifact_identity(self, mock_adapter_report) -> None:
        from adaptersentry.analyzer import scan
        from adaptersentry.cli.scan import _build_scan_result

        adapter_path = mock_adapter_report
        report = scan(adapter_path)
        result = _build_scan_result(adapter_path, report, None)
        data = json.loads(result.model_dump_json())
        assert "artifact" in data
        assert "content_hash" in data["artifact"]

    def test_summary_json_has_verdict(self, mock_adapter_report) -> None:
        from adaptersentry.analyzer import scan
        from adaptersentry.cli.scan import _build_scan_result

        adapter_path = mock_adapter_report
        report = scan(adapter_path)
        result = _build_scan_result(adapter_path, report, None)
        data = json.loads(result.model_dump_json())
        assert "verdict" in data
        assert "recommended_action" in data["verdict"]
        assert data["verdict"]["recommended_action"] in ("allow", "review", "block")

    def test_summary_json_does_not_include_tensor_records(self, mock_adapter_report) -> None:
        from adaptersentry.analyzer import scan
        from adaptersentry.cli.scan import _build_scan_result

        adapter_path = mock_adapter_report
        report = scan(adapter_path)
        result = _build_scan_result(adapter_path, report, None)
        data = json.loads(result.model_dump_json())
        # ScanResult (summary) must not contain tensor_records (those are in DebugReport)
        assert "tensor_records" not in data

    def test_scan_id_is_deterministic_for_same_file(self, mock_adapter_report) -> None:
        from adaptersentry.analyzer import scan
        from adaptersentry.cli.scan import _build_scan_result

        adapter_path = mock_adapter_report
        report = scan(adapter_path)
        r1 = _build_scan_result(adapter_path, report, None)
        r2 = _build_scan_result(adapter_path, report, None)
        assert r1.identity.scan_id == r2.identity.scan_id


class TestBuildDebugReport:
    def test_debug_json_contains_tensor_records(self, tmp_path) -> None:
        import numpy as np
        from safetensors.numpy import save_file
        from adaptersentry.analyzer import scan
        from adaptersentry.cli.scan import _build_scan_result, _build_debug_report

        rng = np.random.default_rng(42)
        adapter_path = tmp_path / "test.safetensors"
        tensors = {
            "model.layers.0.q_proj.lora_A.weight": rng.standard_normal((8, 32)).astype(np.float32),
            "model.layers.0.q_proj.lora_B.weight": rng.standard_normal((32, 8)).astype(np.float32),
        }
        save_file(tensors, str(adapter_path))

        report = scan(adapter_path)
        result = _build_scan_result(adapter_path, report, None)
        debug = _build_debug_report(result, report)

        data = json.loads(debug.model_dump_json())
        assert "tensor_records" in data
        assert len(data["tensor_records"]) > 0

    def test_debug_json_contains_feature_family_results(self, tmp_path) -> None:
        import numpy as np
        from safetensors.numpy import save_file
        from adaptersentry.analyzer import scan
        from adaptersentry.cli.scan import _build_scan_result, _build_debug_report

        rng = np.random.default_rng(42)
        adapter_path = tmp_path / "test.safetensors"
        tensors = {
            "model.layers.0.q_proj.lora_A.weight": rng.standard_normal((8, 32)).astype(np.float32),
            "model.layers.0.q_proj.lora_B.weight": rng.standard_normal((32, 8)).astype(np.float32),
        }
        save_file(tensors, str(adapter_path))

        report = scan(adapter_path)
        result = _build_scan_result(adapter_path, report, None)
        debug = _build_debug_report(result, report)

        assert len(debug.feature_family_results) > 0

    def test_debug_schema_version_present(self, tmp_path) -> None:
        import numpy as np
        from safetensors.numpy import save_file
        from adaptersentry.analyzer import scan
        from adaptersentry.cli.scan import _build_scan_result, _build_debug_report

        rng = np.random.default_rng(42)
        adapter_path = tmp_path / "test.safetensors"
        tensors = {
            "model.layers.0.q_proj.lora_A.weight": rng.standard_normal((8, 32)).astype(np.float32),
            "model.layers.0.q_proj.lora_B.weight": rng.standard_normal((32, 8)).astype(np.float32),
        }
        save_file(tensors, str(adapter_path))

        report = scan(adapter_path)
        result = _build_scan_result(adapter_path, report, None)
        debug = _build_debug_report(result, report)

        data = json.loads(debug.model_dump_json())
        assert "debug_schema_version" in data
        assert data["debug_schema_version"] == "debug-1.0.0"
