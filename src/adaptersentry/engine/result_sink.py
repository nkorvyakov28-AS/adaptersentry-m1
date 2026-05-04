"""ResultSink — atomic write + JSONL append for scan results.

Every write is atomic:
  1. Write to a temporary file in the same directory
  2. fsync the temp file
  3. Rename (atomic on POSIX) to the final path

This guarantees the final file is either complete or absent — never partial.
A crash during step 2 leaves the temp file, which is cleaned up on the next
run. A crash after step 3 has no effect — the final file is intact.

The JSONL file is an append-only audit log for the batch run. Each line is
a complete ScanResult JSON object. Multiple writers must NOT share a JSONL
file — the current implementation is single-writer only.

Idempotency:
    If a result file for scan_id already exists, write() recomputes its hash
    and compares to the new result. Identical → no-op. Different → overwrites
    with a warning log (indicates a non-determinism bug).
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adaptersentry.engine.schemas.scan_result import ScanResult, DebugReport
    from adaptersentry.engine.cache import CacheStore
    from adaptersentry.engine.manifest import ManifestDB

logger = logging.getLogger(__name__)


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _stable_content_hash(result: "ScanResult") -> str:
    """Hash the analysis-relevant fields, excluding timing metadata.

    started_at, completed_at, wall_time_ms change on every scan of the
    same adapter (wall-clock time varies). Including them causes false
    RESULT CONFLICT warnings when the same adapter is re-scanned with
    identical analysis output (e.g. --resume, --force-rescan).

    Excluded from hash: identity.started_at, identity.completed_at,
    identity.wall_time_ms — all timing-volatile, not part of the analysis.
    """
    import json as _json
    d = result.model_dump(
        exclude={"identity": {"started_at", "completed_at", "wall_time_ms"}}
    )
    return _sha256_bytes(_json.dumps(d, sort_keys=True, default=str).encode())


def _atomic_write(path: Path, data: bytes) -> None:
    """Write data to path atomically (tmp → fsync → rename)."""
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_bytes(data)
        # fsync the file data and metadata to survive a power loss
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        tmp.rename(path)
    except Exception:
        # Clean up the temp file if something went wrong
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class ResultSink:
    """Writes scan results to disk with atomicity and idempotency guarantees.

    Usage:
        with ResultSink(results_dir, run_jsonl) as sink:
            sink.write(result, debug, manifest_db, cache_store)
    """

    def __init__(self, results_dir: Path, run_jsonl_path: Path) -> None:
        self._results_dir = results_dir
        self._jsonl_path = run_jsonl_path
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        result: "ScanResult",
        debug: "DebugReport | None",
        manifest_db: "ManifestDB",
        cache_store: "CacheStore | None",
        request_id: str,
        *,
        write_debug: bool = False,
    ) -> None:
        """Persist result; update cache and manifest on success.

        Args:
            result:      Public ScanResult (summary-json contract).
            debug:       Optional DebugReport (debug-json; None = not written).
            manifest_db: Manifest to update on completion.
            cache_store: Cache store to write the entry (None = cache disabled).
            request_id:  Manifest row to update.
            write_debug: If True, write debug JSON alongside the summary JSON.
        """
        scan_id = result.identity.scan_id
        result_bytes = result.model_dump_json(indent=2).encode()
        # Use stable hash (excludes timing fields) for idempotency comparison.
        # started_at / completed_at / wall_time_ms vary between identical scans;
        # comparing full bytes causes false RESULT CONFLICT on --resume / --force-rescan.
        new_stable_hash = _stable_content_hash(result)

        result_path = self._results_dir / f"{scan_id}.json"

        # Idempotency check
        if result_path.exists():
            import json as _json
            try:
                existing = _json.loads(result_path.read_bytes())
                from adaptersentry.engine.schemas.scan_result import ScanResult as _SR
                existing_result = _SR.model_validate(existing)
                existing_stable_hash = _stable_content_hash(existing_result)
            except Exception:
                existing_stable_hash = None

            if existing_stable_hash == new_stable_hash:
                logger.debug("Idempotent write: scan_id=%s already exists unchanged", scan_id)
            else:
                logger.error(
                    "RESULT CONFLICT: scan_id=%s exists with different analysis output — overwriting "
                    "(timing fields excluded; this indicates a true non-determinism bug)",
                    scan_id,
                )
            # Still fall through to update manifest/cache in case previous run crashed
            # before completing those steps

        # Atomic write of summary JSON
        _atomic_write(result_path, result_bytes)

        # Append to run JSONL audit log (single-writer; no locking needed)
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")

        # Write debug JSON if requested
        if write_debug and debug is not None:
            debug_path = self._results_dir / f"{scan_id}.debug.json"
            _atomic_write(debug_path, debug.model_dump_json(indent=2).encode())

        # Write cache entry
        if cache_store is not None:
            try:
                from adaptersentry.version import __version__
                from adaptersentry.engine.config import get_default_config
                cache_store.write(
                    result_bytes=result_bytes,
                    content_hash=result.artifact.content_hash,
                    analyzer_config_hash=result.identity.analyzer_config_hash,
                    scan_id=scan_id,
                    schema_version=result.schema_version,
                    writer_version=__version__,
                )
            except Exception as exc:
                logger.warning("Cache write failed for scan_id=%s: %s", scan_id, exc)

        # Update manifest — only after all writes complete
        from datetime import datetime, timezone
        manifest_db.update_state(
            request_id,
            state="persisted",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.debug("Persisted scan_id=%s", scan_id)

    def write_cached_hit(
        self,
        result: "ScanResult",
        manifest_db: "ManifestDB",
        request_id: str,
    ) -> None:
        """Record a cache hit in the manifest and JSONL without re-writing the result file."""
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")

        from datetime import datetime, timezone
        manifest_db.update_state(
            request_id,
            state="cached_hit",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    def write_failed(
        self,
        result: "ScanResult",
        manifest_db: "ManifestDB",
        request_id: str,
        error_msg: str,
    ) -> None:
        """Record a failed scan in JSONL and manifest."""
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")

        from datetime import datetime, timezone
        manifest_db.update_state(
            request_id,
            state="failed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            error_message=error_msg,
        )

    def __enter__(self) -> "ResultSink":
        return self

    def __exit__(self, *_: object) -> None:
        pass  # no resources to release
