"""CacheStore — content-addressed local result cache.

Layout:
    <cache_root>/
        index.sqlite              — CacheEntry rows (fast lookup)
        objects/
            {hash[:2]}/
                {hash[2:]}.gz     — gzip-compressed ScanResult JSON

Cache key: (content_hash, analyzer_config_hash).
Cache hit requires both components to match AND the result file to pass
integrity validation (stored result_hash vs. recomputed hash).

Poisoning guard:
    On every cache read, the result file is re-hashed and compared to
    CacheEntry.result_hash. A mismatch causes the entry to be deleted and
    a full rescan to be triggered. We NEVER serve a result whose integrity
    cannot be verified — fail-closed.

Writer version guard:
    Entries written by a future adaptersentry version are rejected by older
    readers. This prevents silent behavior mismatches. The comparison is a
    simple string equality on writer_version — not semver — because any
    version bump changes analyzer_config_hash anyway.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from adaptersentry.engine.schemas.cache import CacheEntry

logger = logging.getLogger(__name__)

_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_index (
    content_hash          TEXT NOT NULL,
    analyzer_config_hash  TEXT NOT NULL,
    schema_version        TEXT NOT NULL,
    scan_id               TEXT NOT NULL,
    result_path           TEXT NOT NULL,
    result_hash           TEXT NOT NULL,
    cached_at             TEXT NOT NULL,
    hit_count             INTEGER NOT NULL DEFAULT 0,
    last_hit_at           TEXT,
    ttl_days              INTEGER,
    writer_version        TEXT NOT NULL,
    PRIMARY KEY (content_hash, analyzer_config_hash)
);

CREATE INDEX IF NOT EXISTS idx_cache_content
    ON cache_index (content_hash);
"""


class CacheIntegrityError(Exception):
    """Raised when a cache entry fails integrity validation."""


class CacheStore:
    """Local filesystem cache for ScanResult objects.

    Thread safety: NOT thread-safe. Designed for single-process orchestrator use.
    Workers read from the cache (via CacheStore.lookup) but never write.
    In the current architecture, all cache writes go through the orchestrator.
    """

    def __init__(self, cache_root: Path) -> None:
        self._root = cache_root
        self._objects = cache_root / "objects"
        self._objects.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(cache_root / "index.sqlite"), timeout=10.0)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.executescript(_INDEX_SCHEMA)
        self._db.commit()

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            yield self._db
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

    def close(self) -> None:
        self._db.close()

    # ── Read operations ──────────────────────────────────────────────────────

    def lookup(
        self,
        content_hash: str,
        analyzer_config_hash: str,
    ) -> CacheEntry | None:
        """Check the index for a matching entry. Does NOT validate the result file.

        Returns a CacheEntry on hit, None on miss.
        Callers must call validate_and_read() before trusting the result.
        """
        row = self._db.execute(
            "SELECT * FROM cache_index WHERE content_hash = ? AND analyzer_config_hash = ?",
            (content_hash, analyzer_config_hash),
        ).fetchone()
        if row is None:
            return None

        entry = CacheEntry(**dict(row))

        # TTL check
        if entry.ttl_days is not None:
            cached_dt = datetime.fromisoformat(entry.cached_at.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - cached_dt).days
            if age_days > entry.ttl_days:
                logger.info("Cache entry expired (age=%d days, ttl=%d): %s", age_days, entry.ttl_days, entry.scan_id)
                self._delete_entry_by_key(content_hash, analyzer_config_hash)
                return None

        return entry

    def validate_and_read(
        self,
        entry: CacheEntry,
        current_writer_version: str,
    ) -> bytes | None:
        """Read and integrity-validate a cached result.

        Returns the raw (decompressed) ScanResult JSON bytes on success.
        Returns None and deletes the entry if validation fails.

        Validation steps:
        1. Writer version compatibility check — older reader rejects future entries
        2. File existence check
        3. SHA-256 re-hash vs stored result_hash (poisoning guard)
        """
        # Writer version guard — reject entries from future versions
        if entry.writer_version != current_writer_version:
            logger.info(
                "Cache entry writer_version mismatch: stored=%s current=%s — rejecting",
                entry.writer_version, current_writer_version,
            )
            return None

        obj_path = self._objects / entry.result_path
        if not obj_path.exists():
            logger.warning("Cache object missing: %s — deleting index entry", obj_path)
            self._delete_entry(entry)
            return None

        compressed = obj_path.read_bytes()
        actual_hash = "sha256:" + hashlib.sha256(compressed).hexdigest()

        if actual_hash != entry.result_hash:
            logger.error(
                "CACHE INTEGRITY VIOLATION: scan_id=%s expected=%s got=%s — "
                "deleting entry and triggering rescan (fail-closed)",
                entry.scan_id, entry.result_hash, actual_hash,
            )
            self._delete_entry(entry)
            return None

        return gzip.decompress(compressed)

    # ── Write operations ─────────────────────────────────────────────────────

    def write(
        self,
        result_bytes: bytes,
        content_hash: str,
        analyzer_config_hash: str,
        scan_id: str,
        schema_version: str,
        writer_version: str,
        ttl_days: int | None = None,
    ) -> CacheEntry:
        """Compress and store result_bytes, then write the index entry.

        Returns the CacheEntry for the newly written result.
        Idempotent: if the entry already exists with the same result_hash, it is a no-op.
        """
        compressed = gzip.compress(result_bytes, compresslevel=6)
        file_hash = "sha256:" + hashlib.sha256(compressed).hexdigest()
        # Strip prefix for path derivation
        hex_part = file_hash.removeprefix("sha256:")
        rel_path = f"{hex_part[:2]}/{hex_part[2:]}.gz"
        obj_path = self._objects / rel_path
        obj_path.parent.mkdir(parents=True, exist_ok=True)

        if not obj_path.exists():
            obj_path.write_bytes(compressed)

        now = datetime.now(timezone.utc).isoformat()
        entry = CacheEntry(
            content_hash=content_hash,
            analyzer_config_hash=analyzer_config_hash,
            schema_version=schema_version,
            scan_id=scan_id,
            result_path=rel_path,
            result_hash=file_hash,
            cached_at=now,
            hit_count=0,
            writer_version=writer_version,
            ttl_days=ttl_days,
        )

        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_index
                    (content_hash, analyzer_config_hash, schema_version, scan_id,
                     result_path, result_hash, cached_at, hit_count, writer_version, ttl_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    content_hash, analyzer_config_hash, schema_version, scan_id,
                    rel_path, file_hash, now, writer_version, ttl_days,
                ),
            )

        logger.debug("Cache write: scan_id=%s path=%s", scan_id, rel_path)
        return entry

    def record_hit(self, entry: CacheEntry) -> None:
        """Increment hit_count and update last_hit_at for the given entry."""
        now = datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                "UPDATE cache_index SET hit_count = hit_count + 1, last_hit_at = ? "
                "WHERE content_hash = ? AND analyzer_config_hash = ?",
                (now, entry.content_hash, entry.analyzer_config_hash),
            )

    def _delete_entry(self, entry: CacheEntry) -> None:
        self._delete_entry_by_key(entry.content_hash, entry.analyzer_config_hash)

    def _delete_entry_by_key(self, content_hash: str, analyzer_config_hash: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "DELETE FROM cache_index WHERE content_hash = ? AND analyzer_config_hash = ?",
                (content_hash, analyzer_config_hash),
            )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        row = self._db.execute(
            "SELECT COUNT(*) as entries, SUM(hit_count) as total_hits FROM cache_index"
        ).fetchone()
        return {"entries": row["entries"] or 0, "total_hits": row["total_hits"] or 0}

    @classmethod
    def open(cls, cache_root: Path) -> "CacheStore":
        cache_root.mkdir(parents=True, exist_ok=True)
        return cls(cache_root)

    def __enter__(self) -> "CacheStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
