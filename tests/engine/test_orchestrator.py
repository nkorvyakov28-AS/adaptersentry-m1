"""Tests for orchestrator persistent workers — OPT-01.

Covers _pool_initializer (global state + pre-imports) and _worker_entry
(reads globals instead of per-task pickle args).  No subprocess spawned;
both functions are called directly in-process.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.engine import orchestrator
from adaptersentry.engine.orchestrator import _pool_initializer, _worker_entry
from adaptersentry.engine.schemas.requests import AdapterScanRequest, ArtifactSource
from adaptersentry.engine.schemas.scan_result import DebugReport, ScanResult, ScanStatus


_CONFIG_HASH = "sha256:" + "d" * 64


def _make_adapter(tmp_path: Path, name: str = "adapter.safetensors") -> Path:
    rng = np.random.default_rng(42)
    tensors = {
        "model.layers.0.q_proj.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
        "model.layers.0.q_proj.lora_B.weight": rng.standard_normal((64, 8)).astype(np.float32),
    }
    path = tmp_path / name
    save_file(tensors, str(path), metadata={"r": "8"})
    return path


def _make_req(adapter_path: Path) -> AdapterScanRequest:
    return AdapterScanRequest(
        request_id="sha256:" + "e" * 64,
        run_id="run_opt01_test",
        adapter_path=str(adapter_path),
        source=ArtifactSource(kind="local_path", local_path=str(adapter_path)),
    )


class TestPoolInitializer:
    def test_sets_config_hash_global(self) -> None:
        _pool_initializer(_CONFIG_HASH, "")
        assert orchestrator._WORKER_CONFIG_HASH == _CONFIG_HASH

    def test_sets_cache_root_none_when_empty(self) -> None:
        _pool_initializer(_CONFIG_HASH, "")
        assert orchestrator._WORKER_CACHE_ROOT is None

    def test_sets_cache_root_from_path(self, tmp_path: Path) -> None:
        _pool_initializer(_CONFIG_HASH, str(tmp_path))
        assert orchestrator._WORKER_CACHE_ROOT == tmp_path

    def test_overwrites_previous_config_hash(self) -> None:
        _pool_initializer(_CONFIG_HASH, "")
        alt = "sha256:" + "f" * 64
        _pool_initializer(alt, "")
        assert orchestrator._WORKER_CONFIG_HASH == alt

    def test_preloads_analyzer(self) -> None:
        _pool_initializer(_CONFIG_HASH, "")
        assert "adaptersentry.analyzer" in sys.modules

    def test_preloads_identity(self) -> None:
        _pool_initializer(_CONFIG_HASH, "")
        assert "adaptersentry.engine.identity" in sys.modules

    def test_preloads_cache(self) -> None:
        _pool_initializer(_CONFIG_HASH, "")
        assert "adaptersentry.engine.cache" in sys.modules

    def test_preloads_feature_extractor(self) -> None:
        _pool_initializer(_CONFIG_HASH, "")
        assert "adaptersentry.engine.feature_extractor" in sys.modules


class TestWorkerEntryWithGlobals:
    def test_returns_typed_tuple(self, tmp_path: Path) -> None:
        _pool_initializer(_CONFIG_HASH, "")
        path = _make_adapter(tmp_path)
        result, debug, request_id = _worker_entry(_make_req(path))
        assert isinstance(result, ScanResult)
        assert isinstance(debug, DebugReport)
        assert isinstance(request_id, str)

    def test_returns_request_id(self, tmp_path: Path) -> None:
        _pool_initializer(_CONFIG_HASH, "")
        req = _make_req(_make_adapter(tmp_path))
        _, _, request_id = _worker_entry(req)
        assert request_id == req.request_id

    def test_config_hash_from_global(self, tmp_path: Path) -> None:
        _pool_initializer(_CONFIG_HASH, "")
        result, _, _ = _worker_entry(_make_req(_make_adapter(tmp_path)))
        assert result.identity.analyzer_config_hash == _CONFIG_HASH

    def test_missing_file_returns_failed(self, tmp_path: Path) -> None:
        _pool_initializer(_CONFIG_HASH, "")
        req = _make_req(tmp_path / "missing.safetensors")
        result, _, _ = _worker_entry(req)
        assert result.status == ScanStatus.FAILED

    def test_takes_single_req_not_tuple(self, tmp_path: Path) -> None:
        """Regression: old API passed a (req, hash, cache) tuple — new API is just req."""
        _pool_initializer(_CONFIG_HASH, "")
        req = _make_req(_make_adapter(tmp_path))
        # Calling with a plain AdapterScanRequest must not raise
        result, _, _ = _worker_entry(req)
        assert result is not None
