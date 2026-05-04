"""Tests for worker_main — per-adapter scan pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.engine.schemas.requests import AdapterScanRequest, ArtifactSource
from adaptersentry.engine.schemas.scan_result import ScanResult, ScanStatus, DebugReport
from adaptersentry.engine.worker import worker_main


_CONFIG_HASH = "sha256:" + "c" * 64


def _make_req(adapter_path: Path, run_id: str = "run_test") -> AdapterScanRequest:
    return AdapterScanRequest(
        request_id="sha256:" + "r" * 64,
        run_id=run_id,
        adapter_path=str(adapter_path),
        source=ArtifactSource(kind="local_path", local_path=str(adapter_path)),
    )


def _make_adapter(tmp_path: Path, name: str = "adapter.safetensors") -> Path:
    rng = np.random.default_rng(42)
    tensors = {
        "model.layers.0.q_proj.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
        "model.layers.0.q_proj.lora_B.weight": rng.standard_normal((64, 8)).astype(np.float32),
        "model.layers.1.v_proj.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
        "model.layers.1.v_proj.lora_B.weight": rng.standard_normal((64, 8)).astype(np.float32),
    }
    path = tmp_path / name
    save_file(tensors, str(path), metadata={"r": "8"})
    return path


class TestWorkerMainReturnTypes:
    def test_returns_tuple_of_result_and_debug(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        req = _make_req(path)
        result, debug = worker_main(req, _CONFIG_HASH)
        assert isinstance(result, ScanResult)
        assert isinstance(debug, DebugReport)

    def test_result_has_required_fields(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        req = _make_req(path)
        result, _ = worker_main(req, _CONFIG_HASH)
        assert result.identity.scan_id
        assert result.artifact.content_hash.startswith("sha256:")
        assert result.status in ScanStatus

    def test_scan_id_is_deterministic(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        req = _make_req(path)
        r1, _ = worker_main(req, _CONFIG_HASH)
        r2, _ = worker_main(req, _CONFIG_HASH)
        assert r1.identity.scan_id == r2.identity.scan_id

    def test_debug_contains_tensor_records(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        req = _make_req(path)
        _, debug = worker_main(req, _CONFIG_HASH)
        assert len(debug.tensor_records) > 0


class TestWorkerMainFailurePaths:
    def test_missing_file_returns_failed_result(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.safetensors"
        req = _make_req(path)
        result, _ = worker_main(req, _CONFIG_HASH)
        assert result.status == ScanStatus.FAILED
        assert len(result.errors) > 0

    def test_missing_file_never_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.safetensors"
        req = _make_req(path)
        # Should return a result, not raise
        result, debug = worker_main(req, _CONFIG_HASH)
        assert result is not None
        assert debug is not None

    def test_empty_file_returns_failed_or_degraded(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.safetensors"
        path.write_bytes(b"")
        req = _make_req(path)
        result, _ = worker_main(req, _CONFIG_HASH)
        assert result.status in (ScanStatus.FAILED, ScanStatus.DEGRADED)


class TestWorkerVerdictFields:
    def test_ok_adapter_allows_or_review(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        req = _make_req(path)
        result, _ = worker_main(req, _CONFIG_HASH)
        assert result.verdict.recommended_action in ("allow", "review", "block")

    def test_missing_metadata_sets_m2_recommended(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(1)
        path = tmp_path / "no_meta.safetensors"
        save_file(
            {
                "layer.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
                "layer.lora_B.weight": rng.standard_normal((64, 8)).astype(np.float32),
            },
            str(path),
        )
        req = _make_req(path)
        result, _ = worker_main(req, _CONFIG_HASH)
        # No metadata → m2_recommended should be True
        assert result.verdict.m2_recommended is True

    def test_run_id_propagated(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        req = _make_req(path, run_id="my_benchmark_run")
        result, _ = worker_main(req, _CONFIG_HASH)
        assert result.identity.run_id == "my_benchmark_run"


class TestWorkerDistributionFeatures:
    def test_distribution_features_populated(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        req = _make_req(path)
        _, debug = worker_main(req, _CONFIG_HASH)
        for tr in debug.tensor_records:
            if tr.parse_error is None:
                assert tr.distribution_features is not None

    def test_distribution_features_finite(self, tmp_path: Path) -> None:
        import math
        path = _make_adapter(tmp_path)
        req = _make_req(path)
        _, debug = worker_main(req, _CONFIG_HASH)
        for tr in debug.tensor_records:
            df = tr.distribution_features
            if df is not None:
                assert math.isfinite(df.delta_kurtosis)
                assert math.isfinite(df.delta_skewness)
                assert math.isfinite(df.delta_mean)
                assert math.isfinite(df.delta_std)
