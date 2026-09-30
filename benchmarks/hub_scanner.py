"""
benchmarks/hub_scanner.py — v2
================================
Large-scale observational benchmark pipeline for AdapterSentry M1.

v2 additions over v1:
  - Better failure classification (unsupported_architecture vs analysis_failed)
  - Parallel scanning (--workers N, ThreadPoolExecutor)
  - Local-only rescan mode (--local-only / --candidates-from)

Records are built from ``adaptersentry.scanner.scan`` (ScanResult 2.0.0):
verdict action/level, intra-adapter anomaly summary, reason codes and findings.

IMPORTANT FRAMING
-----------------
This benchmark measures M1 static scan behaviour at scale. It does NOT
assess malware detection accuracy — no labeled ground truth exists for the
public HuggingFace Hub adapter population. A ``review`` verdict or a high
intra-adapter robust z flags an adapter as an investigation candidate; it does
not confirm malicious content. The intra-adapter thresholds are uncalibrated.

Usage examples
--------------
    adaptersentry-bench --limit 500 --output-dir output/hf_benchmark_500
    adaptersentry-bench --limit 1000 --resume --workers 4
    adaptersentry-bench --local-only \\
        --candidates-from output/hf_benchmark_500/candidates.json \\
        --output-dir output/hf_benchmark_500_v2 --workers 4

Security Notes:
    - Only adapter_model.safetensors is downloaded; base model weights are never fetched.
    - All local paths use pathlib.Path — no path traversal.
    - No eval/exec/pickle on downloaded content.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from adaptersentry.schemas.result import ScanResult as M1ScanResult

logger = logging.getLogger(__name__)

_ADAPTER_FILE = "adapter_model.safetensors"
_CONFIG_FILE = "adapter_config.json"

# Substring of the scanner error raised when a file holds no analysable LoRA pair
_NO_LORA_PAIR_MARKER = "no lora_a/lora_b pair"

# Record limits for human-readable findings
_MAX_TOP_FINDINGS = 5
_MAX_FINDING_CHARS = 120


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CandidateRepo:
    """A HuggingFace repo identified as a candidate for scanning."""

    repo_id: str
    hf_downloads: int = 0
    hf_tags: list[str] = field(default_factory=list)
    adapter_size_bytes: int | None = None
    has_adapter_config: bool = False
    revision: str | None = None  # commit sha at discovery time, when the Hub reported it

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CandidateRepo":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ScanResult:
    """Benchmark record for one adapter repo (flattened from a ScanResult 2.0.0).

    Not to be confused with ``adaptersentry.schemas.result.ScanResult``: this is
    the one-line JSONL record of the Hub survey. Records written by the 1.x
    benchmark are not field-compatible; unknown fields are ignored on load.

    status values
    -------------
    success                  — M1 scan completed with status ok or degraded
                               (see ``scan_status``)
    download_failed          — could not fetch adapter_model.safetensors
    size_exceeded            — file size exceeds --max-download-mb
    unsupported_architecture — the scanner found no analysable LoRA A/B pair
    analysis_failed          — M1 scan failed for another reason, or raised
    not_cached               — local-only mode: file not found in cache
    skipped                  — (reserved)
    """

    repo_id: str
    scan_timestamp: str
    status: str
    error_message: str | None = None      # kept for backward compat; prefer error_detail
    skip_reason: str | None = None
    # Repo metadata
    hf_downloads: int = 0
    hf_tags: list[str] = field(default_factory=list)
    adapter_size_bytes: int | None = None
    # M1 scan outputs (None for non-success statuses)
    scan_status: str | None = None        # ScanResult.status: ok | degraded | failed
    action: str | None = None             # verdict.action: allow | review | block
    level: str | None = None              # verdict.level: LOW | MEDIUM | HIGH | CRITICAL
    training_state: str | None = None     # adapter.training_state
    max_robust_z: float | None = None     # anomaly.intra.max_robust_z (None when not computed)
    n_outlier_modules: int | None = None  # anomaly.intra.n_outlier_modules
    reason_codes: list[str] = field(default_factory=list)   # verdict.reasons[].code
    n_findings: int | None = None
    top_findings: list[str] = field(default_factory=list)   # "RULE_ID: title", max 5 × 120 chars
    rank_declared: int | None = None      # adapter.rank_declared (from adapter_config.json)
    n_modules: int | None = None          # coverage.n_modules
    # Failure detail fields
    error_type: str | None = None         # exception class name or "no_lora_pairs_found"
    error_detail: str | None = None       # str(exception)[:200]
    tensor_keys_sample: list[str] = field(default_factory=list)  # unsupported_architecture

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScanResult":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


# ---------------------------------------------------------------------------
# Thread-safe JSONL writer
# ---------------------------------------------------------------------------

_default_write_lock = threading.Lock()


def append_result(
    result: ScanResult,
    results_path: Path,
    lock: threading.Lock | None = None,
) -> None:
    """Append one scan result as a JSON line — thread-safe.

    The lock protects only the file-open / write / close sequence; M1 analysis
    runs outside the lock so threads do not serialise on I/O.
    """
    ctx = lock if lock is not None else _default_write_lock
    line = json.dumps(result.to_dict()) + "\n"
    with ctx:
        with results_path.open("a") as f:
            f.write(line)


# ---------------------------------------------------------------------------
# Atomic per-repo scanner (M1 scan → benchmark record)
# ---------------------------------------------------------------------------


def read_tensor_keys_sample(st_path: Path, n: int = 10) -> list[str]:
    """Return the first ``n`` tensor names from the safetensors header, or [] on error.

    Header only; no tensor data is loaded. Used for diagnostics of adapters the
    scanner could not analyse.
    """
    from safetensors import safe_open

    try:
        with safe_open(str(st_path), framework="numpy") as f:
            return list(f.keys())[:n]
    except Exception:  # noqa: BLE001 — diagnostics only
        return []


def scan_one_adapter(
    candidate: CandidateRepo,
    st_path: Path,
    mode: str = "full",
) -> ScanResult:
    """Run the M1 scan and flatten it into a benchmark record. Never raises.

    A failed scan whose error reports no LoRA A/B pair is recorded as
    ``unsupported_architecture``; any other failed scan, or an exception, as
    ``analysis_failed``.
    """
    ts = datetime.now(timezone.utc).isoformat()
    size: int | None = st_path.stat().st_size if st_path.exists() else None

    try:
        m1_result = run_m1(st_path, mode=mode, repo_id=candidate.repo_id,
                           revision=candidate.revision)
    except Exception as exc:  # noqa: BLE001 — scanner never raises; guard anyway
        return ScanResult(
            repo_id=candidate.repo_id,
            scan_timestamp=ts,
            status="analysis_failed",
            error_type=type(exc).__name__,
            error_detail=str(exc)[:200],
            error_message=str(exc)[:400],
            hf_downloads=candidate.hf_downloads,
            hf_tags=candidate.hf_tags,
            adapter_size_bytes=size,
        )

    if m1_result.status == "failed":
        message = m1_result.errors[0].message if m1_result.errors else "scan failed"
        if _NO_LORA_PAIR_MARKER in message.lower():
            return ScanResult(
                repo_id=candidate.repo_id,
                scan_timestamp=ts,
                status="unsupported_architecture",
                scan_status="failed",
                error_type="no_lora_pairs_found",
                error_detail=message[:200],
                tensor_keys_sample=read_tensor_keys_sample(st_path),
                hf_downloads=candidate.hf_downloads,
                hf_tags=candidate.hf_tags,
                adapter_size_bytes=size,
            )
        detail = m1_result.errors[0].detail if m1_result.errors else None
        return ScanResult(
            repo_id=candidate.repo_id,
            scan_timestamp=ts,
            status="analysis_failed",
            scan_status="failed",
            error_type=detail or (m1_result.errors[0].code if m1_result.errors else "ScanFailed"),
            error_detail=message[:200],
            error_message=message[:400],
            hf_downloads=candidate.hf_downloads,
            hf_tags=candidate.hf_tags,
            adapter_size_bytes=size,
        )

    return _extract_result(candidate, st_path, m1_result)


# ---------------------------------------------------------------------------
# Per-repo worker (download + scan, or local scan only)
# ---------------------------------------------------------------------------


def _process_repo(
    candidate: CandidateRepo,
    adapters_dir: Path,
    max_size_mb: float,
    local_only: bool,
    mode: str = "full",
) -> ScanResult:
    """Full pipeline for one repo — safe to call from a thread pool.

    Returns a ScanResult for every possible outcome; never raises.
    """
    ts = datetime.now(timezone.utc).isoformat()
    safe_name = candidate.repo_id.replace("/", "__")

    if local_only:
        st_path = adapters_dir / safe_name / _ADAPTER_FILE
        if not st_path.exists():
            return ScanResult(
                repo_id=candidate.repo_id,
                scan_timestamp=ts,
                status="not_cached",
                skip_reason="adapter_model.safetensors not found in local cache",
                hf_downloads=candidate.hf_downloads,
                hf_tags=candidate.hf_tags,
            )
        return scan_one_adapter(candidate, st_path, mode=mode)

    # Network mode: download then scan
    try:
        # adapter_config.json is saved next to the adapter; the scanner reads it there.
        st_path, _config = download_adapter(
            repo_id=candidate.repo_id,
            adapters_dir=adapters_dir,
            max_size_mb=max_size_mb,
            fetch_config=True,
            revision=candidate.revision,
        )
    except ValueError as exc:
        return ScanResult(
            repo_id=candidate.repo_id,
            scan_timestamp=ts,
            status="size_exceeded",
            skip_reason=str(exc)[:300],
            hf_downloads=candidate.hf_downloads,
            hf_tags=candidate.hf_tags,
            adapter_size_bytes=candidate.adapter_size_bytes,
        )
    except RuntimeError as exc:
        return ScanResult(
            repo_id=candidate.repo_id,
            scan_timestamp=ts,
            status="download_failed",
            error_type="DownloadError",
            error_detail=str(exc)[:200],
            error_message=str(exc)[:400],
            hf_downloads=candidate.hf_downloads,
            hf_tags=candidate.hf_tags,
            adapter_size_bytes=candidate.adapter_size_bytes,
        )

    return scan_one_adapter(candidate, st_path, mode=mode)


def _load_local_config(config_path: Path) -> dict[str, Any]:
    """Load adapter_config.json from a local path, returning {} on any error."""
    if not config_path.exists():
        return {}
    try:
        with config_path.open() as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------


class _ProgressTracker:
    """Thread-safe progress logger with ETA estimation."""

    def __init__(self, total: int) -> None:
        self.total = total
        self._completed = 0
        self._start = time.monotonic()
        self._lock = threading.Lock()

    def record(self, result: ScanResult) -> None:
        with self._lock:
            self._completed += 1
            n = self._completed

        elapsed = time.monotonic() - self._start
        eta_fmt = "?"
        if n > 0 and elapsed > 0:
            rate = n / elapsed
            remaining = self.total - n
            eta_s = remaining / rate if rate > 0 else 0
            eta_fmt = f"~{int(eta_s // 60)}m{int(eta_s % 60):02d}s"

        z_str = f"{result.max_robust_z:.1f}" if result.max_robust_z is not None else "—"
        verdict = f"{result.action}/{result.level}" if result.action else result.status
        logger.info(
            "[%d/%d] %s  max_z=%s [%s]  ETA %s",
            n, self.total, result.repo_id, z_str, verdict, eta_fmt,
        )


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------


def load_or_discover_candidates(
    target_n: int,
    max_size_mb: float,
    min_downloads: int,
    sleep_seconds: float,
    sample_seed: int,
    candidates_path: Path,
) -> list[CandidateRepo]:
    """Load cached candidates from disk, or discover fresh ones from HF Hub."""
    if candidates_path.exists():
        logger.info("Loading cached candidates from %s", candidates_path)
        with candidates_path.open() as f:
            data = json.load(f)
        return [CandidateRepo.from_dict(c) for c in data["candidates"]]

    logger.info(
        "Discovering candidates: target=%d  max_size=%.0f MB  min_downloads=%d",
        target_n, max_size_mb, min_downloads,
    )

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required: pip install huggingface_hub") from exc

    api = HfApi()
    candidates = _discover(api, target_n, max_size_mb, min_downloads, sleep_seconds)

    logger.info("Discovered %d candidate repositories", len(candidates))

    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    with candidates_path.open("w") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "target_n": target_n,
                "max_size_mb": max_size_mb,
                "min_downloads": min_downloads,
                "sample_seed": sample_seed,
                "total": len(candidates),
                "candidates": [c.to_dict() for c in candidates],
            },
            f,
            indent=2,
        )
    logger.info("Candidates saved to %s", candidates_path)
    return candidates


def _discover(
    api: Any,
    target_n: int,
    max_size_mb: float,
    min_downloads: int,
    sleep_seconds: float,
) -> list[CandidateRepo]:
    """Query HF Hub using siblings expansion for efficient file-name filtering."""
    oversample = min(max(target_n * 6, 3000), 10_000)
    max_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb else 0
    candidates: list[CandidateRepo] = []
    name_filtered: list[dict[str, Any]] = []

    logger.info("Querying HF Hub (filter=peft, sorted by downloads, limit=%d)...", oversample)
    try:
        models = list(
            api.list_models(
                filter="peft",
                sort="downloads",
                limit=oversample,
                # With expand, only the listed properties are returned.
                expand=["siblings", "sha", "downloads", "tags"],
            )
        )
    except Exception as exc:
        logger.error("list_models failed: %s", exc)
        return []

    logger.info("API returned %d models; filtering by file presence...", len(models))

    for model in models:
        if len(name_filtered) >= target_n * 3:
            break
        downloads = getattr(model, "downloads", 0) or 0
        if downloads < min_downloads:
            continue
        siblings = getattr(model, "siblings", None) or []
        file_names = {getattr(s, "rfilename", "") for s in siblings}
        if _ADAPTER_FILE not in file_names:
            continue
        name_filtered.append({
            "model": model,
            "has_config": _CONFIG_FILE in file_names,
            "downloads": downloads,
            "tags": list(getattr(model, "tags", None) or []),
            "sha": getattr(model, "sha", None),
        })

    logger.info(
        "%d repos have %s; fetching sizes (sleep=%.1fs)...",
        len(name_filtered), _ADAPTER_FILE, sleep_seconds,
    )

    for i, entry in enumerate(name_filtered):
        if len(candidates) >= target_n:
            break
        model = entry["model"]
        repo_id: str = model.id
        if i > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)

        adapter_size: int | None = None
        try:
            for item in api.list_repo_tree(repo_id):
                if getattr(item, "path", None) == _ADAPTER_FILE:
                    adapter_size = getattr(item, "size", None)
                    break
        except Exception as exc:
            logger.debug("list_repo_tree failed for %s: %s", repo_id, exc)

        if max_bytes and adapter_size and adapter_size > max_bytes:
            logger.debug("Skipping %s: %.0f MB > limit", repo_id, (adapter_size or 0) / 1024 / 1024)
            continue

        candidates.append(CandidateRepo(
            repo_id=repo_id,
            hf_downloads=entry["downloads"],
            hf_tags=entry["tags"],
            adapter_size_bytes=adapter_size,
            has_adapter_config=entry["has_config"],
            revision=entry["sha"],
        ))

        if (i + 1) % 100 == 0 or len(candidates) % 100 == 0:
            logger.info("Discovery: %d/%d checked → %d candidates", i + 1, len(name_filtered), len(candidates))

    return candidates


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_adapter(
    repo_id: str,
    adapters_dir: Path,
    max_size_mb: float,
    fetch_config: bool = True,
    revision: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Download adapter_model.safetensors with transparent caching.

    ``revision`` pins the download to the commit recorded at discovery time
    (None = the default branch head).

    Raises ValueError if size exceeds limit; RuntimeError on download failure.
    """
    from huggingface_hub import hf_hub_download, HfApi

    safe_name = repo_id.replace("/", "__")
    local_dir = (adapters_dir / safe_name).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    cached_st = local_dir / _ADAPTER_FILE
    if not cached_st.exists() and max_size_mb:
        try:
            api = HfApi()
            for item in api.list_repo_tree(repo_id):
                if getattr(item, "path", None) == _ADAPTER_FILE:
                    size = getattr(item, "size", None)
                    if size and size > int(max_size_mb * 1024 * 1024):
                        raise ValueError(
                            f"{repo_id}: {_ADAPTER_FILE} is "
                            f"{size / 1024 / 1024:.0f} MB, exceeds limit {max_size_mb:.0f} MB"
                        )
                    break
        except ValueError:
            raise
        except Exception:
            pass

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=_ADAPTER_FILE,
            local_dir=str(local_dir),
            revision=revision,
        )
        st_path = Path(downloaded).resolve()
    except Exception as exc:
        raise RuntimeError(f"Download failed for {repo_id}: {exc}") from exc

    config: dict[str, Any] = {}
    if fetch_config:
        config_cached = local_dir / _CONFIG_FILE
        if not config_cached.exists():
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=_CONFIG_FILE,
                    local_dir=str(local_dir),
                    revision=revision,
                )
            except Exception:
                pass
        config = _load_local_config(config_cached)

    return st_path, config


# ---------------------------------------------------------------------------
# M1 runner
# ---------------------------------------------------------------------------


def run_m1(
    st_path: Path,
    mode: str = "full",
    repo_id: str | None = None,
    revision: str | None = None,
) -> "M1ScanResult":
    """Run the AdapterSentry M1 scan (never raises by contract; failures are in ``status``).

    ``adapter_config.json`` is read by the scanner from the adapter's directory.
    """
    from adaptersentry.scanner import scan

    return scan(st_path, mode=mode, hf_repo_id=repo_id, hf_revision=revision)


def _format_finding(rule_id: str, title: str) -> str:
    return f"{rule_id}: {title}"[:_MAX_FINDING_CHARS]


def _extract_result(
    candidate: CandidateRepo,
    st_path: Path,
    m1_result: "M1ScanResult",
) -> ScanResult:
    """Flatten a non-failed ScanResult 2.0.0 + candidate metadata into a benchmark record."""
    intra = m1_result.anomaly.intra
    return ScanResult(
        repo_id=candidate.repo_id,
        scan_timestamp=datetime.now(timezone.utc).isoformat(),
        status="success",
        hf_downloads=candidate.hf_downloads,
        hf_tags=candidate.hf_tags,
        adapter_size_bytes=st_path.stat().st_size,
        scan_status=m1_result.status,
        action=m1_result.verdict.action,
        level=m1_result.verdict.level.value,
        training_state=m1_result.adapter.training_state,
        max_robust_z=intra.max_robust_z if intra is not None else None,
        n_outlier_modules=intra.n_outlier_modules if intra is not None else None,
        reason_codes=[r.code for r in m1_result.verdict.reasons],
        n_findings=len(m1_result.findings),
        top_findings=[
            _format_finding(f.rule_id, f.title) for f in m1_result.findings[:_MAX_TOP_FINDINGS]
        ],
        rank_declared=m1_result.adapter.rank_declared,
        n_modules=m1_result.coverage.n_modules,
    )


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------


def load_completed_repo_ids(results_path: Path) -> set[str]:
    """Return repo IDs already present in results.jsonl (all statuses)."""
    completed: set[str] = set()
    if not results_path.exists():
        return completed
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                repo_id = obj.get("repo_id")
                if repo_id:
                    completed.add(repo_id)
            except json.JSONDecodeError:
                pass
    return completed


def load_all_results(results_path: Path) -> list[ScanResult]:
    """Load all lines from results.jsonl into ScanResult objects."""
    results: list[ScanResult] = []
    if not results_path.exists():
        return results
    known_fields = set(ScanResult.__dataclass_fields__)
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                results.append(ScanResult.from_dict({k: v for k, v in obj.items() if k in known_fields}))
            except Exception:
                pass
    return results


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(
    output_dir: Path,
    limit: int,
    resume: bool,
    max_download_mb: float,
    sleep_seconds: float,
    min_downloads: int,
    sample_seed: int,
    top_n: int,
    workers: int = 1,
    local_only: bool = False,
    candidates_from: Path | None = None,
    mode: str = "full",
) -> None:
    """Main benchmark pipeline — discover/load candidates, scan, report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    write_lock = threading.Lock()

    # ── Candidate source ─────────────────────────────────────────────────────
    if local_only:
        # No HF Hub calls — load from an existing candidates.json
        src = candidates_from or (output_dir / "candidates.json")
        if not src.exists():
            logger.error("--local-only requires candidates.json at %s (use --candidates-from)", src)
            raise SystemExit(1)
        logger.info("Local-only mode: loading candidates from %s", src)
        with src.open() as f:
            data = json.load(f)
        candidates = [CandidateRepo.from_dict(c) for c in data["candidates"]]
        # Adapter files live alongside the source candidates.json
        adapters_dir = src.parent / "adapters"
    else:
        candidates_path = output_dir / "candidates.json"
        candidates = load_or_discover_candidates(
            target_n=limit,
            max_size_mb=max_download_mb,
            min_downloads=min_downloads,
            sleep_seconds=sleep_seconds,
            sample_seed=sample_seed,
            candidates_path=candidates_path,
        )
        adapters_dir = output_dir / "adapters"

    # ── Resume ───────────────────────────────────────────────────────────────
    completed_ids: set[str] = set()
    if resume or results_path.exists():
        completed_ids = load_completed_repo_ids(results_path)
        if completed_ids:
            logger.info("Resuming: %d repos already processed", len(completed_ids))

    to_process = [c for c in candidates if c.repo_id not in completed_ids]
    logger.info(
        "Candidates: %d total  %d already done  %d to process  workers=%d",
        len(candidates), len(completed_ids), len(to_process), workers,
    )

    if not to_process:
        logger.info("Nothing to process — generating reports from existing results.")
    else:
        progress = _ProgressTracker(total=len(to_process))

        if workers > 1:
            _run_parallel(
                to_process, adapters_dir, max_download_mb, local_only,
                sleep_seconds, results_path, write_lock, progress,
                workers=workers, mode=mode,
            )
        else:
            _run_sequential(
                to_process, adapters_dir, max_download_mb, local_only,
                sleep_seconds, results_path, write_lock, progress,
                mode=mode,
            )

    # ── Final reports ────────────────────────────────────────────────────────
    logger.info("Generating final reports...")
    from benchmarks.report import write_csv, write_aggregate, write_markdown_report

    all_results = load_all_results(results_path)
    write_csv(all_results, output_dir / "results.csv")
    agg = write_aggregate(all_results, output_dir / "aggregate.json", candidates, limit, top_n)
    write_markdown_report(agg, all_results, output_dir / "report.md", limit)

    logger.info(
        "Done. output_dir=%s  succeeded=%d  unsupported=%d  failed=%d",
        output_dir,
        agg["totals"]["succeeded"],
        agg["totals"]["unsupported_architecture"],
        agg["totals"]["failed"],
    )


def _run_sequential(
    to_process: list[CandidateRepo],
    adapters_dir: Path,
    max_download_mb: float,
    local_only: bool,
    sleep_seconds: float,
    results_path: Path,
    write_lock: threading.Lock,
    progress: _ProgressTracker,
    mode: str = "full",
) -> None:
    for i, candidate in enumerate(to_process):
        if i > 0 and sleep_seconds > 0 and not local_only:
            time.sleep(sleep_seconds)
        result = _process_repo(candidate, adapters_dir, max_download_mb, local_only, mode)
        append_result(result, results_path, write_lock)
        progress.record(result)


def _run_parallel(
    to_process: list[CandidateRepo],
    adapters_dir: Path,
    max_download_mb: float,
    local_only: bool,
    sleep_seconds: float,
    results_path: Path,
    write_lock: threading.Lock,
    progress: _ProgressTracker,
    workers: int,
    mode: str = "full",
) -> None:
    """Submit all repos to a thread pool; collect and persist results as they complete."""
    if workers > 1 and sleep_seconds > 0 and not local_only:
        logger.warning(
            "Running %d workers with sleep_seconds=%.1f — effective sleep per repo "
            "is reduced in parallel mode. Consider --sleep-seconds 0 with --workers.",
            workers, sleep_seconds,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Stagger submission slightly to avoid thundering-herd on the HF API
        futures = {}
        for i, candidate in enumerate(to_process):
            if i > 0 and sleep_seconds > 0 and not local_only:
                time.sleep(sleep_seconds / workers)
            fut = pool.submit(_process_repo, candidate, adapters_dir, max_download_mb, local_only, mode)
            futures[fut] = candidate

        for fut in as_completed(futures):
            result: ScanResult = fut.result()  # never raises — _process_repo catches all
            append_result(result, results_path, write_lock)
            progress.record(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="adaptersentry-bench",
        description=(
            "AdapterSentry M1 — large-scale HuggingFace Hub observational benchmark.\n\n"
            "NOTE: This is not a malware classifier. A 'review' verdict or a high\n"
            "intra-adapter z flags adapters for manual review; it does not confirm\n"
            "malicious content."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--limit", type=int, default=500, metavar="N",
        help="Target number of adapter repos to scan (default: 500)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None, metavar="DIR",
        help="Output directory (default: output/hf_benchmark_<limit>)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume: skip repos already in results.jsonl",
    )
    p.add_argument(
        "--sleep-seconds", type=float, default=0.5, metavar="S",
        help="Sleep between HF API calls and downloads (default: 0.5). "
             "With --workers > 1, sleep is divided across workers.",
    )
    p.add_argument(
        "--max-download-mb", type=float, default=500.0, metavar="MB",
        help="Maximum adapter file size in MB (default: 500)",
    )
    p.add_argument(
        "--min-downloads", type=int, default=0, metavar="N",
        help="Minimum HF download count to include a repo (default: 0)",
    )
    p.add_argument(
        "--top-n", type=int, default=20, metavar="N",
        help="Top-N entries in aggregate suspicious lists (default: 20)",
    )
    p.add_argument(
        "--sample-seed", type=int, default=42, metavar="SEED",
        help="Reproducibility seed recorded in candidates.json (default: 42)",
    )
    p.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help="Parallel scan workers (default: 1). "
             "With --local-only, 4 workers typically process 500 adapters in 2–5 minutes.",
    )
    p.add_argument(
        "--local-only", action="store_true",
        help="Rescan from local cache only — no HF Hub API calls or downloads. "
             "Requires adapters previously downloaded by a network run.",
    )
    p.add_argument(
        "--candidates-from", type=Path, default=None, metavar="FILE",
        help="Path to candidates.json from a previous run. "
             "Used with --local-only to rescan a different output directory. "
             "Adapter files are resolved relative to this file's parent directory.",
    )
    p.add_argument(
        "--mode", choices=("full", "fast"), default="full",
        help="M1 scan mode (default: full)",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    return p


def main() -> None:
    """CLI entry point for adaptersentry-bench."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    output_dir = args.output_dir or Path(f"output/hf_benchmark_{args.limit}")

    run_pipeline(
        output_dir=output_dir,
        limit=args.limit,
        resume=args.resume,
        max_download_mb=args.max_download_mb,
        sleep_seconds=args.sleep_seconds,
        min_downloads=args.min_downloads,
        sample_seed=args.sample_seed,
        top_n=args.top_n,
        workers=args.workers,
        local_only=args.local_only,
        candidates_from=args.candidates_from,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
