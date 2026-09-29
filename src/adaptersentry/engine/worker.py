"""Batch worker: one adapter → one ScanResult 2.0.0.

``worker_main`` runs in a worker process (multiprocessing or Ray). It checks the
content-addressed cache first and otherwise calls ``adaptersentry.scanner.scan``.
It never raises: every failure becomes a fail-closed ScanResult.
"""

from __future__ import annotations

import logging
from pathlib import Path

from adaptersentry.engine.schemas.requests import AdapterScanRequest
from adaptersentry.schemas.result import ScanResult

logger = logging.getLogger(__name__)


def _cached(req: AdapterScanRequest, config_hash: str, cache_root: Path) -> ScanResult | None:
    from adaptersentry.engine.cache import CacheStore
    from adaptersentry.engine.identity import ArtifactIdentityResolver
    from adaptersentry.schemas.result import load_scan_result
    from adaptersentry.version import __version__

    identity = ArtifactIdentityResolver.resolve(Path(req.adapter_path), req.source)
    cache = CacheStore.open(cache_root)
    try:
        entry = cache.lookup(identity.content_hash, config_hash)
        if entry is None:
            return None
        raw = cache.validate_and_read(entry, __version__)
        if raw is None:
            return None
        cache.record_hit(entry)
        return load_scan_result(raw)
    finally:
        cache.close()


def worker_main(
    req: AdapterScanRequest,
    analyzer_config_hash: str,
    cache_root: Path | None = None,
) -> tuple[ScanResult, bool]:
    """Scan one adapter for a batch run.

    Args:
        req: the scan request (absolute, resolved adapter path).
        analyzer_config_hash: scanner.config_hash(mode, policy) of the batch;
            the cache key component.
        cache_root: cache directory, or None to disable caching.

    Returns:
        (result, from_cache). Never raises.
    """
    from adaptersentry.scanner import failed_result, scan

    mode = "fast" if req.scan_mode == "fast" else "full"
    path = Path(req.adapter_path)
    if not path.is_absolute():
        return failed_result(path, f"adapter_path must be absolute; got {req.adapter_path!r}",
                             mode=mode, policy=req.policy, run_id=req.run_id), False

    if cache_root is not None and not req.force_rescan:
        try:
            hit = _cached(req, analyzer_config_hash, cache_root)
            if hit is not None:
                return hit, True
        except Exception as exc:  # noqa: BLE001 — a cache problem must never stop the scan
            logger.warning("Cache lookup failed for %r: %s — scanning", path.name, exc)

    try:
        return scan(path, mode=mode, policy=req.policy, run_id=req.run_id,
                    hf_repo_id=req.source.hf_repo_id, hf_revision=req.source.hf_revision), False
    except Exception as exc:  # noqa: BLE001 — scan() does not raise; defence in depth
        logger.error("Scan crashed for %r: %s", path.name, exc)
        return failed_result(path, f"scan crashed: {exc}", mode=mode, policy=req.policy,
                             run_id=req.run_id), False
