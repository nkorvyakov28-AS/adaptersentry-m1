"""Tests for Ray-based orchestrator (OPT-03).

Tests run without a real Ray cluster — Ray is initialised in local mode.
The ScanWorkerActor is tested directly (not via the full batch pipeline)
to avoid heavy multiprocessing setup in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

ray = pytest.importorskip("ray", reason="ray not installed — skip OPT-03 tests")


def _make_adapter(tmp_path: Path, name: str = "adapter.safetensors") -> Path:
    rng = np.random.default_rng(7)
    tensors = {
        "model.layers.0.q_proj.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
        "model.layers.0.q_proj.lora_B.weight": rng.standard_normal((64, 8)).astype(np.float32),
        "model.layers.1.v_proj.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
        "model.layers.1.v_proj.lora_B.weight": rng.standard_normal((64, 8)).astype(np.float32),
    }
    path = tmp_path / name
    save_file(tensors, str(path), metadata={"r": "8"})
    return path


def _make_req(adapter_path: Path):
    from adaptersentry.engine.schemas.requests import AdapterScanRequest, ArtifactSource
    import hashlib
    req_id = "sha256:" + hashlib.sha256(str(adapter_path).encode()).hexdigest()
    return AdapterScanRequest(
        request_id=req_id,
        run_id="run_ray_test",
        adapter_path=str(adapter_path),
        source=ArtifactSource(kind="local_path", local_path=str(adapter_path)),
        scan_mode="fast",
        force_rescan=False,
        submitted_at="2026-05-03T00:00:00+00:00",
    )


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    """Start a local Ray cluster for all tests in this module."""
    ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()


class TestScanWorkerActorClass:
    def test_actor_class_buildable(self):
        from adaptersentry.engine.orchestrator_ray import _make_worker_actor_class
        assert _make_worker_actor_class() is not None

    def test_actor_scan_returns_v2_result(self, tmp_path):
        from adaptersentry.engine.orchestrator_ray import _make_worker_actor_class
        from adaptersentry.scanner import config_hash
        from adaptersentry.schemas.result import ScanResult

        actor_cls = _make_worker_actor_class()
        actor = actor_cls.remote(config_hash("fast", "default"), "")
        req = _make_req(_make_adapter(tmp_path))
        result, from_cache, returned_id = ray.get(actor.scan.remote(req))
        assert isinstance(result, ScanResult) and from_cache is False
        assert returned_id == req.request_id
        assert result.status in ("ok", "degraded")


class TestActorFailed:
    def test_actor_crash_is_fail_closed(self):
        from adaptersentry.engine.orchestrator_ray import _actor_failed

        result = _actor_failed(_make_req(Path("/p.safetensors")), "OOM")
        assert result.status == "failed" and result.verdict.action == "review"
        assert "crashed" in result.errors[0].message


class TestRunBatchRayImport:
    def test_import_without_ray_raises(self, monkeypatch):
        """run_batch_ray raises RuntimeError when ray is not importable."""
        import sys
        import importlib
        from adaptersentry.engine import orchestrator_ray

        # Temporarily hide ray
        original = sys.modules.get("ray")
        sys.modules["ray"] = None  # type: ignore
        try:
            with pytest.raises((RuntimeError, ImportError)):
                orchestrator_ray.run_batch_ray(
                    requests=[],
                    manifest_db=None, cache_store=None,
                    results_dir=Path("/tmp"), run_jsonl_path=Path("/tmp/r.jsonl"),
                    analyzer_config_hash="sha256:" + "a" * 64,
                )
        finally:
            if original is not None:
                sys.modules["ray"] = original
            else:
                del sys.modules["ray"]

    def test_empty_requests_returns_empty_stats(self, tmp_path):
        """run_batch_ray with no requests returns empty dict."""
        from adaptersentry.engine.orchestrator_ray import run_batch_ray

        stats = run_batch_ray(
            requests=[],
            manifest_db=None, cache_store=None,
            results_dir=tmp_path, run_jsonl_path=tmp_path / "r.jsonl",
            analyzer_config_hash="sha256:" + "a" * 64,
        )
        assert stats == {}
