"""Tests for ManifestDB state machine and resumability."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptersentry.engine.manifest import ManifestDB, TERMINAL_STATES, NON_TERMINAL_STATES


def _req_json(request_id: str, path: str = "/fake/adapter.safetensors") -> str:
    import json
    return json.dumps({"request_id": request_id, "adapter_path": path})


class TestManifestDBBasic:
    def test_open_creates_db(self, tmp_path: Path) -> None:
        db = ManifestDB.open(tmp_path / "manifest.sqlite")
        assert (tmp_path / "manifest.sqlite").exists()
        db.close()

    def test_upsert_and_get(self, tmp_path: Path) -> None:
        db = ManifestDB.open(tmp_path / "m.sqlite")
        db.upsert("req1", "run1", "/path/a.safetensors", _req_json("req1"))
        row = db.get("req1")
        assert row is not None
        assert row.state == "queued"
        assert row.run_id == "run1"
        db.close()

    def test_get_nonexistent_returns_none(self, tmp_path: Path) -> None:
        db = ManifestDB.open(tmp_path / "m.sqlite")
        assert db.get("does_not_exist") is None
        db.close()

    def test_upsert_idempotent(self, tmp_path: Path) -> None:
        db = ManifestDB.open(tmp_path / "m.sqlite")
        db.upsert("req1", "run1", "/a.safetensors", _req_json("req1"))
        db.upsert("req1", "run1", "/a.safetensors", _req_json("req1"), state="queued")
        rows = db.get_by_run("run1")
        assert len(rows) == 1
        db.close()


class TestStateTransitions:
    def test_update_state(self, tmp_path: Path) -> None:
        db = ManifestDB.open(tmp_path / "m.sqlite")
        db.upsert("req1", "run1", "/a.safetensors", _req_json("req1"))
        db.update_state("req1", "leased")
        row = db.get("req1")
        assert row.state == "leased"
        db.close()

    def test_update_all_fields(self, tmp_path: Path) -> None:
        db = ManifestDB.open(tmp_path / "m.sqlite")
        db.upsert("req1", "run1", "/a.safetensors", _req_json("req1"))
        db.update_state(
            "req1", "persisted",
            content_hash="sha256:abc",
            completed_at="2026-01-01T00:00:00Z",
        )
        row = db.get("req1")
        assert row.state == "persisted"
        assert row.content_hash == "sha256:abc"
        db.close()


class TestQueueAndStats:
    def test_get_queued_returns_queued_only(self, tmp_path: Path) -> None:
        db = ManifestDB.open(tmp_path / "m.sqlite")
        db.upsert("r1", "run1", "/a.safetensors", _req_json("r1"))
        db.upsert("r2", "run1", "/b.safetensors", _req_json("r2"))
        db.update_state("r2", "persisted")
        queued = db.get_queued("run1")
        assert len(queued) == 1
        assert queued[0].request_id == "r1"
        db.close()

    def test_stats_counts_by_state(self, tmp_path: Path) -> None:
        db = ManifestDB.open(tmp_path / "m.sqlite")
        for i in range(3):
            db.upsert(f"r{i}", "run1", f"/a{i}.safetensors", _req_json(f"r{i}"))
        db.update_state("r0", "persisted")
        db.update_state("r1", "failed")
        stats = db.stats("run1")
        assert stats["queued"] == 1
        assert stats["persisted"] == 1
        assert stats["failed"] == 1
        db.close()


class TestDuplicateSuppression:
    def test_has_content_hash_in_run_returns_id(self, tmp_path: Path) -> None:
        db = ManifestDB.open(tmp_path / "m.sqlite")
        db.upsert("r1", "run1", "/a.safetensors", _req_json("r1"))
        db.update_state("r1", "persisted", content_hash="sha256:abc123")
        result = db.has_content_hash_in_run("run1", "sha256:abc123")
        assert result == "r1"
        db.close()

    def test_has_content_hash_in_run_miss(self, tmp_path: Path) -> None:
        db = ManifestDB.open(tmp_path / "m.sqlite")
        db.upsert("r1", "run1", "/a.safetensors", _req_json("r1"))
        db.update_state("r1", "queued", content_hash="sha256:abc123")
        # Not in terminal state — should not match
        result = db.has_content_hash_in_run("run1", "sha256:abc123")
        assert result is None
        db.close()


class TestResume:
    def test_reset_non_terminal_resets_leased(self, tmp_path: Path) -> None:
        db = ManifestDB.open(tmp_path / "m.sqlite")
        db.upsert("r1", "run1", "/a.safetensors", _req_json("r1"))
        db.update_state("r1", "leased")
        n = db.reset_non_terminal("run1")
        assert n == 1
        assert db.get("r1").state == "queued"
        db.close()

    def test_reset_non_terminal_skips_terminal(self, tmp_path: Path) -> None:
        db = ManifestDB.open(tmp_path / "m.sqlite")
        db.upsert("r1", "run1", "/a.safetensors", _req_json("r1"))
        db.update_state("r1", "persisted")
        n = db.reset_non_terminal("run1")
        assert n == 0
        assert db.get("r1").state == "persisted"
        db.close()

    def test_reset_exceeds_retry_limit_marks_failed(self, tmp_path: Path) -> None:
        db = ManifestDB.open(tmp_path / "m.sqlite")
        db.upsert("r1", "run1", "/a.safetensors", _req_json("r1"))
        # Simulate already retried twice
        db.update_state("r1", "leased", retry_count=2)
        db.reset_non_terminal("run1")
        row = db.get("r1")
        assert row.state == "failed"
        db.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        with ManifestDB.open(tmp_path / "m.sqlite") as db:
            db.upsert("r1", "run1", "/a.safetensors", _req_json("r1"))
            assert db.get("r1") is not None
