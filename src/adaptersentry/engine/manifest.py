"""ManifestDB — SQLite-backed batch state machine for scan jobs.

Each row in the manifest represents one AdapterScanRequest and its current
lifecycle state. The manifest enables:
  - resumable batch scans (restart after crash without re-scanning completed jobs)
  - duplicate suppression within a run (same content_hash in the same run)
  - lease-based worker coordination (expired leases are re-queued by orchestrator)

State machine:
    queued → leased → running → parsed → analyzed → scored → persisted (terminal)
    queued → leased → running → ... → cached_hit     (terminal, served from cache)
    queued → leased → running → ... → skipped_duplicate (terminal, dedup)
    any non-terminal → retriable_failed → queued     (retry, up to 3 times)
    any non-terminal → failed                        (terminal, error)

SQLite WAL mode is enabled to allow concurrent readers (workers checking in)
while the orchestrator writes. All manifest writes go through the orchestrator
process — workers return results via the pool result iterator, not by writing
to the manifest directly.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

TERMINAL_STATES = frozenset({
    "persisted", "cached_hit", "skipped_duplicate", "failed",
})

NON_TERMINAL_STATES = frozenset({
    "queued", "leased", "running", "parsed", "analyzed", "scored", "retriable_failed",
})

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS manifest (
    request_id      TEXT    PRIMARY KEY,
    run_id          TEXT,
    adapter_path    TEXT    NOT NULL,
    state           TEXT    NOT NULL DEFAULT 'queued',
    retry_count     INTEGER NOT NULL DEFAULT 0,
    content_hash    TEXT,
    lease_expires_at TEXT,
    submitted_at    TEXT    NOT NULL,
    started_at      TEXT,
    completed_at    TEXT,
    error_message   TEXT,
    request_json    TEXT    NOT NULL,
    UNIQUE(run_id, adapter_path)
);

CREATE INDEX IF NOT EXISTS idx_manifest_run_state
    ON manifest (run_id, state);

CREATE INDEX IF NOT EXISTS idx_manifest_content_hash
    ON manifest (content_hash);

CREATE INDEX IF NOT EXISTS idx_manifest_lease
    ON manifest (state, lease_expires_at)
    WHERE state = 'leased';
"""


class ManifestRow:
    """Lightweight wrapper around a manifest row dict."""

    __slots__ = (
        "request_id", "run_id", "adapter_path", "state", "retry_count",
        "content_hash", "lease_expires_at", "submitted_at", "started_at",
        "completed_at", "error_message", "request_json",
    )

    def __init__(self, row: dict[str, Any]) -> None:
        for slot in self.__slots__:
            setattr(self, slot, row.get(slot))


class ManifestDB:
    """SQLite-backed manifest store for one or more batch runs.

    Thread safety: NOT thread-safe. Designed for single-process, single-writer
    use. Workers do not write to the manifest; only the orchestrator does.
    """

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._conn: sqlite3.Connection | None = None
        self._connect()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(str(self._path), timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        logger.debug("Manifest opened: %s", self._path)

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        assert self._conn is not None
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Write operations ────────────────────────────────────────────────────

    def upsert(
        self,
        request_id: str,
        run_id: str | None,
        adapter_path: str,
        request_json: str,
        state: str = "queued",
        submitted_at: str | None = None,
    ) -> None:
        """Insert or update a manifest row (idempotent)."""
        now = submitted_at or datetime.now(timezone.utc).isoformat()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO manifest
                    (request_id, run_id, adapter_path, state, retry_count,
                     submitted_at, request_json)
                VALUES (?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    state        = excluded.state,
                    request_json = excluded.request_json
                """,
                (request_id, run_id, adapter_path, state, now, request_json),
            )

    def update_state(
        self,
        request_id: str,
        state: str,
        *,
        retry_count: int | None = None,
        lease_expires_at: str | None = None,
        content_hash: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update the state (and optional fields) of a manifest row."""
        fields: list[str] = ["state = ?"]
        values: list[Any] = [state]

        if retry_count is not None:
            fields.append("retry_count = ?")
            values.append(retry_count)
        if lease_expires_at is not None:
            fields.append("lease_expires_at = ?")
            values.append(lease_expires_at)
        if content_hash is not None:
            fields.append("content_hash = ?")
            values.append(content_hash)
        if started_at is not None:
            fields.append("started_at = ?")
            values.append(started_at)
        if completed_at is not None:
            fields.append("completed_at = ?")
            values.append(completed_at)
        if error_message is not None:
            fields.append("error_message = ?")
            values.append(error_message)

        values.append(request_id)
        with self._transaction() as conn:
            conn.execute(
                f"UPDATE manifest SET {', '.join(fields)} WHERE request_id = ?",
                values,
            )

    # ── Read operations ─────────────────────────────────────────────────────

    def get(self, request_id: str) -> ManifestRow | None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM manifest WHERE request_id = ?", (request_id,)
        ).fetchone()
        return ManifestRow(dict(row)) if row else None

    def get_by_run(self, run_id: str) -> list[ManifestRow]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM manifest WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [ManifestRow(dict(r)) for r in rows]

    def get_queued(self, run_id: str, limit: int = 256) -> list[ManifestRow]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM manifest WHERE run_id = ? AND state = 'queued' "
            "ORDER BY adapter_path LIMIT ?",
            (run_id, limit),
        ).fetchall()
        return [ManifestRow(dict(r)) for r in rows]

    def get_expired_leases(self, run_id: str) -> list[ManifestRow]:
        """Return leased rows whose lease has expired."""
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat()
        rows = self._conn.execute(
            "SELECT * FROM manifest WHERE run_id = ? AND state = 'leased' "
            "AND lease_expires_at < ?",
            (run_id, now),
        ).fetchall()
        return [ManifestRow(dict(r)) for r in rows]

    def has_content_hash_in_run(self, run_id: str, content_hash: str) -> str | None:
        """Return the request_id of a completed job with this content_hash in the run, or None."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT request_id FROM manifest "
            "WHERE run_id = ? AND content_hash = ? AND state IN ('persisted', 'cached_hit') "
            "LIMIT 1",
            (run_id, content_hash),
        ).fetchone()
        return row["request_id"] if row else None

    def stats(self, run_id: str) -> dict[str, int]:
        """Return a count per state for the given run."""
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT state, COUNT(*) as n FROM manifest WHERE run_id = ? GROUP BY state",
            (run_id,),
        ).fetchall()
        return {r["state"]: r["n"] for r in rows}

    # ── Resume support ───────────────────────────────────────────────────────

    def reset_non_terminal(self, run_id: str) -> int:
        """Reset all non-terminal rows to 'queued' (for resume after crash).

        Increments retry_count on each reset. Rows that have already been
        retried 3 times are moved directly to 'failed'.

        Returns the number of rows reset to queued.
        """
        rows = self.get_by_run(run_id)
        n_reset = 0
        for row in rows:
            if row.state in TERMINAL_STATES:
                continue
            if row.state in NON_TERMINAL_STATES:
                new_retry = (row.retry_count or 0) + 1
                if new_retry >= 3:
                    self.update_state(
                        row.request_id,
                        "failed",
                        error_message="Exceeded retry limit during crash recovery",
                    )
                    logger.warning(
                        "Job %s exceeded retry limit during resume — marking failed",
                        row.request_id,
                    )
                else:
                    self.update_state(
                        row.request_id,
                        "queued",
                        retry_count=new_retry,
                        lease_expires_at=None,
                    )
                    n_reset += 1
        return n_reset

    def __enter__(self) -> "ManifestDB":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @classmethod
    def open(cls, db_path: Path) -> "ManifestDB":
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(db_path)
