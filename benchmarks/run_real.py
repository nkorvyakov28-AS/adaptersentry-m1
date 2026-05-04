"""Run benchmark harness against a directory of real adapters.

Usage:
    python benchmarks/run_real.py --input-dir PATH --workers N --output FILE
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--mode", choices=("full", "fast"), default="full",
                   help="Scan mode: full (default) or fast")
    p.add_argument("--backend", choices=("mp", "ray"), default="mp",
                   help="Worker backend: mp (multiprocessing, default) or ray")
    args = p.parse_args()

    # Collect all .safetensors files recursively
    paths = sorted(args.input_dir.rglob("*.safetensors"))
    print(f"Found {len(paths)} adapters in {args.input_dir}", flush=True)

    if not paths:
        print("ERROR: no .safetensors files found", file=sys.stderr)
        return 1

    from benchmarks.harness import _run_scenario, BenchmarkMetrics

    with tempfile.TemporaryDirectory(prefix="adaptersentry_real_bench_") as tmp:
        work_dir = Path(tmp) / "work"
        work_dir.mkdir()

        print(f"Scenario: cold  workers={args.workers}  adapters={len(paths)}  mode={args.mode}  backend={args.backend}", flush=True)
        print("Running — this may take several minutes...", flush=True)

        t0 = time.monotonic()
        metrics = _run_scenario(
            corpus_paths=paths,
            scenario="cold",
            n_workers=args.workers,
            work_dir=work_dir,
            cache_dir=None,
            scan_mode=args.mode,
            backend=args.backend,
        )
        elapsed = time.monotonic() - t0

    # Reference from hf_benchmark_500_v3: 500 adapters in 3h15m sequential
    ref_adapters = 500
    ref_minutes = 195.0
    ref_tput = ref_adapters / ref_minutes

    speedup = metrics.throughput_per_min / ref_tput if ref_tput > 0 else 0
    estimated_500_min = 500 / metrics.throughput_per_min if metrics.throughput_per_min > 0 else 0

    print("\n" + "=" * 60)
    print(f"RESULTS — {len(paths)} real adapters, {args.workers} workers, mode={args.mode}")
    print("=" * 60)
    print(f"  Total wall time       : {metrics.total_wall_s / 60:.1f} min ({metrics.total_wall_s:.0f}s)")
    print(f"  Throughput            : {metrics.throughput_per_min:.1f} adapters/min")
    print(f"  Latency p50           : {metrics.latency_p50_ms:.0f} ms")
    print(f"  Latency p95           : {metrics.latency_p95_ms:.0f} ms")
    print(f"  Latency p99           : {metrics.latency_p99_ms:.0f} ms")
    print(f"  Peak RSS / worker     : {metrics.peak_per_worker_mb:.0f} MB")
    print(f"  Error rate            : {metrics.error_rate:.1%}")
    print(f"  Cache hits            : {metrics.cache_hit_rate:.1%}")
    print()
    print(f"COMPARISON vs hf_benchmark_500_v3 (sequential, 3h 15m)")
    print(f"  Reference throughput  : {ref_tput:.2f} adapters/min")
    print(f"  Speedup               : {speedup:.1f}×")
    print(f"  Est. time for 500     : {estimated_500_min:.1f} min  (vs 195 min before)")
    print(f"  Time saved on 500     : {195 - estimated_500_min:.1f} min")
    print("=" * 60)

    data = {
        "metrics": asdict(metrics),
        "comparison": {
            "reference_label": "hf_benchmark_500_v3",
            "reference_adapters": ref_adapters,
            "reference_minutes": ref_minutes,
            "reference_throughput_per_min": ref_tput,
            "speedup_x": round(speedup, 2),
            "estimated_500_minutes": round(estimated_500_min, 1),
            "time_saved_minutes": round(195 - estimated_500_min, 1),
        },
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2))
        print(f"Results written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
