"""Tests for CacheStore — hit/miss/integrity/version guard."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from adaptersentry.engine.cache import CacheStore
from adaptersentry.engine.schemas.cache import CacheEntry

_FAKE_VERSION = "0.2.0"
_FAKE_CONFIG_HASH = "sha256:" + "a" * 64
_FAKE_CONTENT_HASH = "sha256:" + "b" * 64
_FAKE_SCAN_ID = "scan_test_001"


def _write_result(store: CacheStore, result_json: str = '{"schema_version":"1.0.0"}') -> CacheEntry:
    return store.write(
        result_bytes=result_json.encode(),
        content_hash=_FAKE_CONTENT_HASH,
        analyzer_config_hash=_FAKE_CONFIG_HASH,
        scan_id=_FAKE_SCAN_ID,
        schema_version="1.0.0",
        writer_version=_FAKE_VERSION,
    )


class TestCacheHitMiss:
    def test_miss_on_empty_cache(self, tmp_path: Path) -> None:
        store = CacheStore.open(tmp_path / "cache")
        entry = store.lookup(_FAKE_CONTENT_HASH, _FAKE_CONFIG_HASH)
        assert entry is None
        store.close()

    def test_write_then_lookup_hit(self, tmp_path: Path) -> None:
        store = CacheStore.open(tmp_path / "cache")
        _write_result(store)
        entry = store.lookup(_FAKE_CONTENT_HASH, _FAKE_CONFIG_HASH)
        assert entry is not None
        assert entry.scan_id == _FAKE_SCAN_ID
        store.close()

    def test_different_config_hash_is_miss(self, tmp_path: Path) -> None:
        store = CacheStore.open(tmp_path / "cache")
        _write_result(store)
        other_hash = "sha256:" + "c" * 64
        entry = store.lookup(_FAKE_CONTENT_HASH, other_hash)
        assert entry is None
        store.close()


class TestIntegrityGuard:
    def test_valid_entry_reads_correctly(self, tmp_path: Path) -> None:
        store = CacheStore.open(tmp_path / "cache")
        result_json = '{"schema_version":"1.0.0","status":"ok"}'
        _write_result(store, result_json)
        entry = store.lookup(_FAKE_CONTENT_HASH, _FAKE_CONFIG_HASH)
        assert entry is not None
        raw = store.validate_and_read(entry, _FAKE_VERSION)
        assert raw is not None
        assert json.loads(raw)["status"] == "ok"
        store.close()

    def test_tampered_object_returns_none_and_deletes_entry(self, tmp_path: Path) -> None:
        store = CacheStore.open(tmp_path / "cache")
        entry = _write_result(store)
        # Corrupt the object file
        obj_path = (tmp_path / "cache" / "objects" / entry.result_path)
        obj_path.write_bytes(gzip.compress(b"tampered content"))
        # Lookup still finds the index entry
        found = store.lookup(_FAKE_CONTENT_HASH, _FAKE_CONFIG_HASH)
        assert found is not None
        # validate_and_read detects the tamper and returns None
        raw = store.validate_and_read(found, _FAKE_VERSION)
        assert raw is None
        # Entry should be deleted from index
        assert store.lookup(_FAKE_CONTENT_HASH, _FAKE_CONFIG_HASH) is None
        store.close()

    def test_missing_object_file_returns_none(self, tmp_path: Path) -> None:
        store = CacheStore.open(tmp_path / "cache")
        entry = _write_result(store)
        # Delete the object file
        obj_path = (tmp_path / "cache" / "objects" / entry.result_path)
        obj_path.unlink()
        found = store.lookup(_FAKE_CONTENT_HASH, _FAKE_CONFIG_HASH)
        assert found is not None
        raw = store.validate_and_read(found, _FAKE_VERSION)
        assert raw is None
        store.close()


class TestVersionGuard:
    def test_future_writer_version_rejected(self, tmp_path: Path) -> None:
        store = CacheStore.open(tmp_path / "cache")
        _write_result(store)
        entry = store.lookup(_FAKE_CONTENT_HASH, _FAKE_CONFIG_HASH)
        assert entry is not None
        # Reader uses a different (older) version
        raw = store.validate_and_read(entry, "0.1.0")
        assert raw is None
        store.close()

    def test_same_version_accepted(self, tmp_path: Path) -> None:
        store = CacheStore.open(tmp_path / "cache")
        _write_result(store)
        entry = store.lookup(_FAKE_CONTENT_HASH, _FAKE_CONFIG_HASH)
        raw = store.validate_and_read(entry, _FAKE_VERSION)
        assert raw is not None
        store.close()


class TestHitCounting:
    def test_record_hit_increments_count(self, tmp_path: Path) -> None:
        store = CacheStore.open(tmp_path / "cache")
        _write_result(store)
        entry = store.lookup(_FAKE_CONTENT_HASH, _FAKE_CONFIG_HASH)
        assert entry is not None
        store.record_hit(entry)
        store.record_hit(entry)
        stats = store.stats()
        assert stats["total_hits"] == 2
        store.close()


class TestCacheStats:
    def test_stats_on_empty_cache(self, tmp_path: Path) -> None:
        store = CacheStore.open(tmp_path / "cache")
        stats = store.stats()
        assert stats["entries"] == 0
        assert stats["total_hits"] == 0
        store.close()

    def test_stats_after_writes(self, tmp_path: Path) -> None:
        store = CacheStore.open(tmp_path / "cache")
        _write_result(store)
        # Second entry with different content_hash
        store.write(
            result_bytes=b'{"schema_version":"1.0.0"}',
            content_hash="sha256:" + "d" * 64,
            analyzer_config_hash=_FAKE_CONFIG_HASH,
            scan_id="scan_002",
            schema_version="1.0.0",
            writer_version=_FAKE_VERSION,
        )
        stats = store.stats()
        assert stats["entries"] == 2
        store.close()


class TestContextManager:
    def test_context_manager(self, tmp_path: Path) -> None:
        with CacheStore.open(tmp_path / "cache") as store:
            _write_result(store)
            assert store.lookup(_FAKE_CONTENT_HASH, _FAKE_CONFIG_HASH) is not None
