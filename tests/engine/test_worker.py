"""Tests for the batch worker (worker_main) and the mp orchestrator entry points."""

from __future__ import annotations

from pathlib import Path

import pytest

import adaptersentry.engine.orchestrator as orch
from adaptersentry.engine.cache import CacheStore
from adaptersentry.engine.schemas.requests import AdapterScanRequest, ArtifactSource
from adaptersentry.engine.worker import worker_main
from adaptersentry.scanner import config_hash
from adaptersentry.schemas.result import ScanResult
from tests.adapter_factory import write_adapter

_HASH = config_hash("full", "default")


def _req(path: Path, *, run_id: str = "run_test", policy: str = "default", force: bool = False) -> AdapterScanRequest:
    return AdapterScanRequest(
        request_id="sha256:" + "r" * 64, run_id=run_id, adapter_path=str(path),
        source=ArtifactSource(kind="local_path", local_path=str(path)),
        policy=policy, force_rescan=force,
    )


class TestWorkerMain:
    def test_returns_result_and_cache_flag(self, tmp_path: Path) -> None:
        result, from_cache = worker_main(_req(write_adapter(tmp_path / "a")), _HASH)
        assert isinstance(result, ScanResult) and from_cache is False
        assert result.status == "ok" and result.scan.run_id == "run_test"

    def test_policy_is_forwarded(self, tmp_path: Path) -> None:
        import numpy as np
        path = write_adapter(tmp_path / "p", extra={"base_model.model.lm_head.weight": np.zeros((8, 256), np.float32)})
        result, _ = worker_main(_req(path, policy="strict"), config_hash("full", "strict"))
        assert result.scan.policy == "strict" and result.verdict.action == "block"

    def test_missing_file_fails_closed(self, tmp_path: Path) -> None:
        result, _ = worker_main(_req(tmp_path / "missing.safetensors"), _HASH)
        assert result.status == "failed" and result.verdict.action == "review"

    def test_relative_path_rejected(self) -> None:
        result, _ = worker_main(_req(Path("relative.safetensors")), _HASH)
        assert result.status == "failed" and "absolute" in result.errors[0].message

    def test_truncated_file_fails_closed(self, tmp_path: Path) -> None:
        good = write_adapter(tmp_path / "t")
        bad = tmp_path / "t" / "bad.safetensors"
        bad.write_bytes(good.read_bytes()[:-32])
        result, _ = worker_main(_req(bad), _HASH)
        assert result.verdict.action == "review" and result.verdict.m2_recommended

    def test_cache_hit_returns_stored_result(self, tmp_path: Path) -> None:
        path = write_adapter(tmp_path / "c")
        cache_root = tmp_path / "cache"
        first, hit1 = worker_main(_req(path), _HASH, cache_root)
        assert hit1 is False
        store = CacheStore.open(cache_root)
        from adaptersentry.version import __version__
        store.write(result_bytes=first.model_dump_json().encode(), content_hash=first.artifact.content_hash,
                    analyzer_config_hash=_HASH, scan_id=first.scan.scan_id,
                    schema_version=first.schema_version, writer_version=__version__)
        store.close()
        second, hit2 = worker_main(_req(path), _HASH, cache_root)
        assert hit2 is True and second == first

    def test_force_rescan_bypasses_cache(self, tmp_path: Path) -> None:
        path = write_adapter(tmp_path / "f")
        _, hit = worker_main(_req(path, force=True), _HASH, tmp_path / "cache")
        assert hit is False


class TestPoolEntryPoints:
    def test_initializer_sets_globals_and_blas(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        orch._pool_initializer("sha256:" + "x" * 64, str(tmp_path))
        assert orch._WORKER_CONFIG_HASH == "sha256:" + "x" * 64
        assert orch._WORKER_CACHE_ROOT == tmp_path
        import os
        assert os.environ["OMP_NUM_THREADS"] == "1"

    def test_initializer_empty_cache_root(self) -> None:
        orch._pool_initializer(_HASH, "")
        assert orch._WORKER_CACHE_ROOT is None

    def test_worker_entry_returns_triple(self, tmp_path: Path) -> None:
        orch._pool_initializer(_HASH, "")
        result, from_cache, request_id = orch._worker_entry(_req(write_adapter(tmp_path / "w")))
        assert isinstance(result, ScanResult) and from_cache is False and request_id == "sha256:" + "r" * 64


class TestRunBatch:
    def test_end_to_end_batch_with_cache(self, tmp_path: Path) -> None:
        from adaptersentry.engine.manifest import ManifestDB
        from adaptersentry.engine.orchestrator import build_manifest, run_batch

        paths = [write_adapter(tmp_path / f"a{i}", seed=i, n_layers=6) for i in range(3)]
        broken = tmp_path / "a0" / "broken.safetensors"
        broken.write_bytes(paths[0].read_bytes()[:-50])
        paths.append(broken)
        cache_root = tmp_path / "cache"
        results = tmp_path / "results"

        with ManifestDB.open(tmp_path / "m.sqlite") as db:
            reqs = build_manifest(paths, "run1", db)
            store = CacheStore.open(cache_root)
            stats = run_batch(reqs, db, store, results, results / "run.jsonl", _HASH,
                              cache_root=cache_root, n_workers=2)
            store.close()
        assert stats.get("ok", 0) == 3 and stats.get("failed", 0) == 1
        lines = (results / "run.jsonl").read_text().splitlines()
        assert len(lines) == 4

        with ManifestDB.open(tmp_path / "m.sqlite") as db:
            reqs = build_manifest(paths[:3], "run2", db)
            store = CacheStore.open(cache_root)
            stats2 = run_batch(reqs, db, store, results, results / "run2.jsonl", _HASH,
                               cache_root=cache_root, n_workers=2)
            store.close()
        assert stats2.get("cached", 0) == 3
