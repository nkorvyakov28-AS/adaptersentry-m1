# Testing Guide

This guide covers how to run the AdapterSentry test suite, what the tests cover,
and the conventions every contributor must follow.

---

## Prerequisites

Install in development mode before running tests:

```bash
git clone https://github.com/nkorvyakov28-AS/adaptersentry-m1
cd adaptersentry-m1
pip install -e ".[dev]"
```

---

## Running the full suite

```bash
pytest tests/ -q
```

All 773+ tests must pass. This is a hard gate — no commit lands with a failing test.

---

## Targeted runs

Run only the subset relevant to your change:

```bash
# Schema stability and migration
pytest tests/schemas -q

# Batch scan engine (worker, cache, manifest, orchestrator, identity)
pytest tests/engine -q

# Feature families (distribution, norm, entropy, inter-layer similarity)
pytest tests/features -q

# Detectors and ensemble scorer
pytest tests/test_detectors.py tests/test_ensemble.py -q

# Scoring (ScoreBreakdown, ScoringPolicy, ConfidenceScore)
pytest tests/scoring -q

# Reporting (PerLayerFinding, human summary renderer)
pytest tests/reporting -q

# CLI (scan command, output formats, exit codes)
pytest tests/cli -q

# Output reporters (SARIF, JSON)
pytest tests/reporters -q

# Parser (safetensors, metadata)
pytest tests/parsers -q

# Analyzer public API and backward compatibility
pytest tests/test_analyzer.py tests/test_m1_backward_compat.py -q

```

Add `-v` to any command for per-test names, or `-x` to stop at the first failure.

---

## Test directory layout

```
tests/
├── fixtures/               # Frozen schema snapshots for migration tests
│   └── scan_result_v1.0.0.json
├── schemas/                # ScanResult, AdapterReport, TensorRecord, error taxonomy
├── parsers/                # safetensors parser, metadata extraction
├── features/               # DistributionFeatures, DeltaNorm, EntropyCompression,
│                           # InterLayerSimilarity
├── engine/                 # FeatureExtractor, Worker, Orchestrator (mp + Ray),
│                           # BLAKE3 cache, SQLite manifest, scan identity
├── scoring/                # ScoreBreakdown, ScoringPolicy, ConfidenceScore
├── reporting/              # PerLayerFinding ranker, render_human_summary
├── reporters/              # SARIF reporter, JSON reporter
├── cli/                    # CLI surface: flags, output formats, exit codes
├── test_analyzer.py        # scan() / analyze() public API
├── test_detectors.py       # All M1 detectors
├── test_ensemble.py        # EnsembleDetector (typed + legacy paths)
├── test_risk_scorer.py     # Risk scoring weights and escalation
├── test_init_detector.py   # INIT_ONLY / PARTIALLY_TRAINED detection
├── test_m1.py              # End-to-end M1 pipeline
├── test_m1_backward_compat.py  # API stability across minor versions
├── test_bench.py           # Benchmark harness unit tests
└── test_harness.py         # Corpus generator and regression gate
```

---

## What the tests cover

| Area | Files | Key coverage |
|------|-------|-------------|
| Schema stability | `schemas/test_migration.py` | Frozen v1.0.0 fixture loads without data loss; `extra="ignore"` forward-compat |
| Parser | `parsers/test_safetensors.py`, `parsers/test_metadata.py` | LoRA pair detection, bfloat16 conversion, non-LoRA rejection, per-tensor error isolation |
| Feature families | `features/` | DistributionFeatures extended fields, EntropyCompression O(n) signals, InterLayerSimilarity cosine/Pearson pairs |
| Detectors | `test_detectors.py` | Kurtosis, energy concentration, Wasserstein, cross-layer, entropy, outlier, IsolationForest |
| Ensemble | `test_ensemble.py` | `score_families()` typed path, `score()` legacy path, risk level boundaries |
| Scoring | `scoring/` | ScoreBreakdown 7-family decomposition, ScoringPolicy cap/floor/escalation, ConfidenceScore circular-logic guard |
| Reporting | `reporting/` | PerLayerFinding top-10 ranking, RULE_CATALOG wording stability, render_human_summary compact + verbose blocks |
| Engine | `engine/` | Worker crash isolation, BLAKE3 content cache, SQLite manifest idempotency, deterministic scan_id, Ray orchestrator |
| CLI | `cli/test_scan_cli.py`, `engine/test_scan_cli_formats.py` | All output formats (text, summary-json, debug-json, sarif), `--fail-on`, `--verbose`, `--no-color` |
| Backward compat | `test_m1_backward_compat.py` | Public API surface unchanged across minor versions |

---

## Test conventions

### No torch in tests or benchmarks

Use `safetensors.numpy` exclusively — `torch` is not a project dependency:

```python
# Correct
from safetensors.numpy import save_file
import numpy as np
tensors = {"layer.lora_A.weight": arr_a, "layer.lora_B.weight": arr_b}
save_file(tensors, str(path))

# Wrong — torch is not available in the test environment
from safetensors.torch import save_file
```

### Synthetic adapters only

Tests must never embed real adapter content. Construct minimal synthetic tensors
from numpy random generators:

```python
def _make_adapter(tmp_path: Path, rank: int = 8, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    path = tmp_path / "adapter.safetensors"
    tensors = {
        "model.layers.0.q_proj.lora_A.weight": rng.standard_normal((rank, 64)).astype(np.float32),
        "model.layers.0.q_proj.lora_B.weight": rng.standard_normal((64, rank)).astype(np.float32),
    }
    save_file(tensors, str(path), metadata={"r": str(rank)})
    return path
```

### No hardcoded paths

Use `tmp_path` (pytest fixture) for file I/O. Never reference absolute paths or
local corpus directories inside test code.

```python
# Correct — pytest provides tmp_path
def test_scan(tmp_path: Path) -> None:
    path = _make_adapter(tmp_path)
    ...

# Wrong — hardcoded system path
path = Path("/home/user/adapters/corpus/some.safetensors")
```

### No mocking the core pipeline

Do not mock `analyze()`, `scan()`, `FeatureExtractor.extract_layer()`, or
`EnsembleDetector.score_families()`. Tests must exercise the real pipeline on
synthetic inputs.

### Score A and B only as supplementary signals

Tests that assert on feature values must reflect the ΔW = B @ A contract:
primary scoring uses the composed delta. Per-tensor A/B stats (`tensor_stats_A`,
`tensor_stats_B`) are supplementary and must not be used as the sole basis for
risk assertions.

### Schema version fixture rule

Every stable schema version requires a frozen JSON fixture under `tests/fixtures/`.
When incrementing `schema_version`, add a new fixture and a corresponding
migration test. Do not delete old fixtures — backward-compat tests read them.

---

## Rust extension tests

The optional Rust hot-path extension (`adaptersentry-rs`) has its own test suite
managed by Cargo:

```bash
cd adaptersentry-rs
cargo test --release
```

Run this whenever you modify code in `adaptersentry-rs/src/`. The Rust suite
is separate from pytest and does not contribute to the 773-test count.

To build the extension for local use before testing the Python integration:

```bash
pip install maturin
VIRTUAL_ENV=$(python -c "import sys; print(sys.prefix)") maturin develop --release
```

---

## Benchmark harness

The benchmark harness validates throughput performance, not correctness. Run it
separately from the main test suite.

### Synthetic corpus (CI default)

```bash
./benchmarks/run.sh                        # cold + warm, 100 adapters, 4 workers
./benchmarks/run.sh --n 100 --workers 4 --scenario cold
./benchmarks/run.sh --ci                   # run + check regression against baseline
```

Output is written to `benchmarks/results/current.json`. The regression gate in
`--ci` mode compares against `benchmarks/results/baseline.json` (committed).

### Real adapter corpus

The real corpus (`output/hf_benchmark_500/adapters/`, 498 adapters) is available
on the development machine but is not committed and must not be referenced from
test code. Benchmark runs against the real corpus are manual only:

```bash
# Fast mode — safe at 8 workers
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python benchmarks/run_real.py \
    --input-dir output/hf_benchmark_500/adapters \
    --workers 8 \
    --mode fast \
    --output benchmarks/results/real_500_fast_8w.json

# Full mode — limit to 4 workers (memory budget)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python benchmarks/run_real.py \
    --input-dir output/hf_benchmark_500/adapters \
    --workers 4 \
    --mode full \
    --output benchmarks/results/real_500_full_4w.json
```

Always set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for
real-corpus runs. Without these limits, BLAS thread oversubscription causes a
6× throughput regression on multi-worker configurations.

---

## Commit gate

The full test suite must pass before every commit:

```bash
pytest tests/ -q
```

A commit with any failing test will be rejected during code review. There are no
exceptions for "known failures" or "flaky tests" — investigate and fix before landing.

For Rust changes, also run `cargo test --release` in `adaptersentry-rs/`.
