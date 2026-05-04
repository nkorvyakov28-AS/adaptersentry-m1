"""Benchmark harness for the AdapterSentry scan engine.

Measures throughput, per-adapter latency (p50/p95/p99), peak memory,
and error rate under two scenarios:

  cold   Fresh cache directory — every adapter scanned from scratch.
  warm   Same corpus scanned twice — second run should be all cache hits.

Usage:
    python -m benchmarks.harness --n 100 --workers 4 --output results/run.json
    python -m benchmarks.harness --n 1000 --workers 4 --scenario warm --output results/run.json

Exit code:
    0  metrics collected and written
    1  benchmark failed to complete
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkMetrics:
    scenario: str                  # "cold" or "warm"
    n_adapters: int                # adapters in corpus
    n_workers: int
    total_wall_s: float            # total elapsed seconds for the batch
    throughput_per_min: float      # adapters / minute
    latency_p50_ms: float          # per-adapter wall time from ScanResult
    latency_p95_ms: float
    latency_p99_ms: float
    peak_total_rss_mb: float       # peak RSS of orchestrator + all children
    peak_per_worker_mb: float      # peak_total / n_workers (approximation)
    error_rate: float              # fraction of scans with status=failed
    cache_hit_rate: float          # fraction with status=cached
    stats: dict = field(default_factory=dict)  # raw ok/degraded/failed/cached counts
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def passes_dod(self, scenario: str) -> tuple[bool, list[str]]:
        """Check Definition-of-Done thresholds for the given scenario.

        DoD (CARD-11):
          warm/1000  total < 120s
          cold/100   throughput >= 100 adapters/min
          any        peak_per_worker_mb < 512
        """
        failures: list[str] = []

        if scenario == "warm" and self.n_adapters >= 1000:
            if self.total_wall_s > 120:
                failures.append(
                    f"warm-1000 wall time {self.total_wall_s:.1f}s > 120s DoD limit"
                )

        if scenario == "cold" and self.n_adapters >= 100:
            if self.throughput_per_min < 100:
                failures.append(
                    f"cold-100 throughput {self.throughput_per_min:.1f}/min < 100/min DoD limit"
                )

        if self.peak_per_worker_mb > 512:
            failures.append(
                f"peak_per_worker {self.peak_per_worker_mb:.1f}MB > 512MB DoD limit"
            )

        return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# RSS memory tracker (background thread)
# ---------------------------------------------------------------------------

class _RssTracker:
    """Samples RSS of the current process + all children every 500ms."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._peak_bytes: list[float] = [0.0]
        self._proc = psutil.Process(os.getpid())
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=2.0)
        return self._peak_bytes[0] / (1024 ** 2)  # bytes → MB

    def _poll(self) -> None:
        while not self._stop.wait(timeout=0.5):
            try:
                rss = self._proc.memory_info().rss
                for child in self._proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                self._peak_bytes[0] = max(self._peak_bytes[0], float(rss))
            except psutil.NoSuchProcess:
                break


# ---------------------------------------------------------------------------
# Latency extraction from result files
# ---------------------------------------------------------------------------

def _collect_latencies(results_dir: Path) -> list[float]:
    """Read wall_time_ms from every ScanResult JSON in results_dir."""
    latencies: list[float] = []
    for path in results_dir.glob("*.json"):
        if path.name.endswith(".debug.json") or path.name == "run_summary.json":
            continue
        try:
            data = json.loads(path.read_text())
            ms = data.get("identity", {}).get("wall_time_ms")
            if ms is not None:
                latencies.append(float(ms))
        except Exception:
            pass
    return latencies


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------

_MILESTONES = {1, 5, 10, 50, 100, 200, 300, 500}


def _milestone_tracker(
    results_dir: Path,
    t0: float,
    n_total: int,
    stop_event: threading.Event,
) -> None:
    """Background thread: prints timing when completed scan count hits a milestone."""
    reported: set[int] = set()
    targets = sorted(m for m in _MILESTONES if m <= n_total)

    while True:
        completed = len(list(results_dir.glob("*.json")))
        for m in targets:
            if completed >= m and m not in reported:
                elapsed = time.monotonic() - t0
                rate = completed / elapsed * 60
                print(
                    f"  ✓ {m:>4} scans — {elapsed:6.1f}s elapsed  "
                    f"({elapsed/m*1000:.0f}ms/scan  {rate:.1f}/min)",
                    flush=True,
                )
                reported.add(m)
        if len(reported) >= len(targets):
            break
        if stop_event.is_set():
            # One final check after batch completes
            completed = len(list(results_dir.glob("*.json")))
            for m in targets:
                if completed >= m and m not in reported:
                    elapsed = time.monotonic() - t0
                    rate = completed / elapsed * 60
                    print(
                        f"  ✓ {m:>4} scans — {elapsed:6.1f}s elapsed  "
                        f"({elapsed/m*1000:.0f}ms/scan  {rate:.1f}/min)",
                        flush=True,
                    )
                    reported.add(m)
            break
        time.sleep(0.15)


def _run_scenario(
    corpus_paths: list[Path],
    scenario: str,
    n_workers: int,
    work_dir: Path,
    cache_dir: Path | None,
    scan_mode: str = "full",
    backend: str = "mp",
) -> BenchmarkMetrics:
    """Run one benchmark scenario and return metrics.

    Args:
        corpus_paths: Paths to .safetensors files.
        scenario:     "cold" or "warm".
        n_workers:    Worker pool size.
        work_dir:     Temp dir for manifest + results.
        cache_dir:    Cache root. None = no cache. For warm, same dir used twice.
        scan_mode:    "full" (default) or "fast".
    """
    from adaptersentry.engine.config import AnalyzerConfig
    from adaptersentry.engine.manifest import ManifestDB
    from adaptersentry.engine.orchestrator import build_manifest, run_batch

    run_id = f"bench-{scenario}-{uuid.uuid4().hex[:8]}"
    manifest_path = work_dir / f"manifest_{run_id}.db"
    results_dir = work_dir / f"results_{run_id}"
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = work_dir / f"audit_{run_id}.jsonl"

    config = AnalyzerConfig()
    config_hash = config.config_hash()

    # Warm scenario: first pass primes the cache, second pass measures hit rate.
    # We re-use the same manifest file to avoid re-queuing completed jobs,
    # so force_rescan=True on the second pass.
    passes = 2 if scenario == "warm" else 1

    stats: dict[str, int] = {}
    total_wall_s = 0.0
    peak_rss_mb = 0.0

    for pass_idx in range(passes):
        force = pass_idx > 0  # second pass forces rescan to bypass terminal state

        # Clear results_dir between passes to keep latency from pass 2 only
        if pass_idx > 0:
            for f in results_dir.glob("*.json"):
                f.unlink(missing_ok=True)

        manifest_db = ManifestDB.open(manifest_path)
        requests = build_manifest(
            corpus_paths,
            run_id,
            manifest_db,
            force_rescan=force,
            scan_mode=scan_mode,
        )

        if not requests:
            manifest_db.close()
            continue

        tracker = _RssTracker()
        tracker.start()
        t0 = time.monotonic()

        stop_evt = threading.Event()
        mile_thread = threading.Thread(
            target=_milestone_tracker,
            args=(results_dir, t0, len(corpus_paths), stop_evt),
            daemon=True,
        )
        mile_thread.start()

        if backend == "ray":
            from adaptersentry.engine.orchestrator_ray import run_batch_ray
            stats = run_batch_ray(
                requests=requests,
                manifest_db=manifest_db,
                cache_store=None,
                results_dir=results_dir,
                run_jsonl_path=jsonl_path,
                analyzer_config_hash=config_hash,
                cache_root=cache_dir,
                n_workers=n_workers,
                write_debug=False,
            )
        else:
            stats = run_batch(
                requests=requests,
                manifest_db=manifest_db,
                cache_store=None,
                results_dir=results_dir,
                run_jsonl_path=jsonl_path,
                analyzer_config_hash=config_hash,
                cache_root=cache_dir,
                n_workers=n_workers,
                write_debug=False,
            )

        elapsed = time.monotonic() - t0
        stop_evt.set()
        mile_thread.join(timeout=2.0)
        rss = tracker.stop()
        manifest_db.close()

        # Only record timing from the last pass
        total_wall_s = elapsed
        peak_rss_mb = rss

    latencies = _collect_latencies(results_dir)
    n = len(corpus_paths)
    throughput = (n / total_wall_s) * 60 if total_wall_s > 0 else 0.0
    n_cached = stats.get("cached", 0)
    n_failed = stats.get("failed", 0)

    return BenchmarkMetrics(
        scenario=scenario,
        n_adapters=n,
        n_workers=n_workers,
        total_wall_s=round(total_wall_s, 3),
        throughput_per_min=round(throughput, 1),
        latency_p50_ms=round(_percentile(latencies, 50), 1),
        latency_p95_ms=round(_percentile(latencies, 95), 1),
        latency_p99_ms=round(_percentile(latencies, 99), 1),
        peak_total_rss_mb=round(peak_rss_mb, 1),
        peak_per_worker_mb=round(peak_rss_mb / max(n_workers, 1), 1),
        error_rate=round(n_failed / max(n, 1), 4),
        cache_hit_rate=round(n_cached / max(n, 1), 4),
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run_benchmark(
    n_adapters: int = 100,
    n_workers: int = 4,
    scenarios: list[str] | None = None,
    rank: int = 8,
    n_layers: int = 4,
    hidden_dim: int = 64,
    output_path: Path | None = None,
    keep_corpus: bool = False,
    scan_mode: str = "full",
) -> list[BenchmarkMetrics]:
    """Run the full benchmark suite and return metrics for each scenario.

    Args:
        n_adapters:  Number of synthetic adapters to generate.
        n_workers:   Worker pool size.
        scenarios:   List of scenarios to run: "cold" and/or "warm".
                     Default: ["cold", "warm"].
        rank:        LoRA rank for the synthetic corpus.
        n_layers:    Transformer layers per adapter.
        hidden_dim:  Hidden dimension for tensor shapes.
        output_path: Write metrics JSON to this path (optional).
        keep_corpus: If True, do not delete the generated corpus after the run.

    Returns:
        List of BenchmarkMetrics, one per scenario.
    """
    from benchmarks.corpus import generate_corpus

    if scenarios is None:
        scenarios = ["cold", "warm"]

    results: list[BenchmarkMetrics] = []

    with tempfile.TemporaryDirectory(prefix="adaptersentry_bench_") as tmp:
        tmp_path = Path(tmp)
        corpus_dir = tmp_path / "corpus"

        logger.info("Generating %d synthetic adapters (rank=%d, layers=%d)...",
                    n_adapters, rank, n_layers)
        corpus_paths = generate_corpus(
            n_adapters, corpus_dir,
            rank=rank, n_layers=n_layers, hidden_dim=hidden_dim,
        )

        for scenario in scenarios:
            logger.info("Running scenario: %s (%d adapters, %d workers)",
                        scenario, n_adapters, n_workers)

            # Cold: fresh cache each run. Warm: persistent cache across passes.
            if scenario == "cold":
                cache_dir = None  # no cache — every scan runs from scratch
            else:
                cache_dir = tmp_path / "cache"
                cache_dir.mkdir(exist_ok=True)

            work_dir = tmp_path / f"work_{scenario}"
            work_dir.mkdir(exist_ok=True)

            metrics = _run_scenario(
                corpus_paths=corpus_paths,
                scenario=scenario,
                n_workers=n_workers,
                work_dir=work_dir,
                cache_dir=cache_dir,
                scan_mode=scan_mode,
            )
            results.append(metrics)

            passed, failures = metrics.passes_dod(scenario)
            status = "PASS" if passed else "FAIL"
            logger.info(
                "[%s] %s | %.1f/min | p95=%.0fms | rss/worker=%.0fMB | err=%.1f%% | cache=%.1f%%",
                status, scenario,
                metrics.throughput_per_min,
                metrics.latency_p95_ms,
                metrics.peak_per_worker_mb,
                metrics.error_rate * 100,
                metrics.cache_hit_rate * 100,
            )
            for f in failures:
                logger.warning("  DoD FAIL: %s", f)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([asdict(m) for m in results], indent=2),
            encoding="utf-8",
        )
        logger.info("Metrics written to %s", output_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.harness",
        description="AdapterSentry scan engine benchmark harness",
    )
    p.add_argument("--n", type=int, default=100, metavar="N",
                   help="Number of synthetic adapters (default: 100)")
    p.add_argument("--workers", type=int, default=4, metavar="W",
                   help="Worker pool size (default: 4)")
    p.add_argument("--scenario", choices=["cold", "warm", "both"], default="both",
                   help="Scenario to run (default: both)")
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (default: 8)")
    p.add_argument("--layers", type=int, default=4,
                   help="Transformer layers per adapter (default: 4)")
    p.add_argument("--hidden-dim", type=int, default=64,
                   help="Hidden dimension for tensor shapes (default: 64)")
    p.add_argument("--output", type=Path, default=None, metavar="FILE",
                   help="Write metrics JSON to FILE")
    p.add_argument("--mode", choices=("full", "fast"), default="full",
                   help="Scan mode: full (default) or fast")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return p


def main() -> int:
    p = _build_parser()
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    scenarios = ["cold", "warm"] if args.scenario == "both" else [args.scenario]

    results = run_benchmark(
        n_adapters=args.n,
        n_workers=args.workers,
        scenarios=scenarios,
        rank=args.rank,
        n_layers=args.layers,
        hidden_dim=args.hidden_dim,
        output_path=args.output,
        scan_mode=args.mode,
    )

    # Print summary to stdout
    print(json.dumps([asdict(m) for m in results], indent=2))

    # Exit 1 if any DoD check failed
    all_passed = True
    for m in results:
        passed, failures = m.passes_dod(m.scenario)
        if not passed:
            all_passed = False
            for f in failures:
                print(f"DoD FAIL: {f}", file=sys.stderr)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
