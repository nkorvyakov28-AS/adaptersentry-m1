# Testing

**Analysis Date:** 2026-05-06

## Framework

- **Runner:** pytest 7.4+
- **Config:** `pyproject.toml` `[tool.pytest.ini_options]` — `testpaths = ["tests"]`, `python_files = ["test_*.py"]`, `python_classes = ["Test*"]`, `python_functions = ["test_*"]`
- **Invocation:** `python3 -m pytest tests/ -q --tb=short`
- **Count:** 773 tests (as of 2026-05-06)
- **No coverage plugin configured** — coverage not enforced in CI currently

## Structure

`tests/` mirrors `src/adaptersentry/` subdirectory layout:

```
tests/
├── test_m1.py              # Core M1 analysis pipeline integration tests
├── test_analyzer.py        # Analyzer module tests
├── test_ensemble.py        # EnsembleDetector tests
├── test_detectors.py       # Detector-level tests
├── test_risk_scorer.py     # RiskScorer tests
├── test_init_detector.py   # InitDetector tests
├── test_harness.py         # Benchmark harness tests
├── test_bench.py           # Benchmark runner (hub_scanner) tests
├── test_m1_backward_compat.py # Schema and API backward compat regression
├── cli/test_scan_cli.py    # CLI scan command tests (includes version check)
├── engine/
│   ├── test_cache.py
│   ├── test_feature_extractor.py
│   ├── test_identity.py
│   ├── test_manifest.py
│   ├── test_orchestrator.py
│   ├── test_orchestrator_ray.py
│   ├── test_scan_cli_formats.py
│   └── test_worker.py
├── features/               # Per-layer feature tests
├── parsers/                # Parser tests
├── reporters/              # Output format tests
├── reporting/              # Human summary + per-layer tests
├── schemas/
│   ├── test_adapter_report.py
│   ├── test_errors.py
│   ├── test_migration.py   # Schema snapshot tests (frozen fixture round-trips)
│   └── test_tensor_record.py
├── scoring/                # Confidence, score breakdown, scoring policy tests
└── fixtures/
    └── scan_result_v1.0.0.json  # Frozen schema snapshot for migration tests
```

## Test Patterns

### Synthetic Adapters (Primary Fixture Strategy)

Tests construct minimal `.safetensors` files in-process using `safetensors.numpy.save_file` — no torch, no external corpus. Helper `_make_adapter()` is defined in most test modules:

```python
def _make_adapter(
    tmp_path: Path,
    layers: dict[str, tuple[np.ndarray, np.ndarray]],
    metadata: dict[str, str] | None = None,
    filename: str = "adapter.safetensors",
) -> Path:
    tensors: dict[str, np.ndarray] = {}
    for layer_name, (a, b) in layers.items():
        tensors[f"{layer_name}.lora_A.weight"] = a.astype(np.float32)
        tensors[f"{layer_name}.lora_B.weight"] = b.astype(np.float32)
    path = tmp_path / filename
    save_file(tensors, str(path), metadata=metadata or {})
    return path
```

### pytest Fixtures

`pytest.fixture()` used for reusable test adapters (e.g., `clean_adapter`, `suspicious_adapter`) and for module-level Ray skip guards (`scope="module", autouse=True`).

### Mocking

Minimal mocking — tests prefer real implementations. `monkeypatch` used sparingly for:
- Simulating Ray import failures (`test_orchestrator_ray.py`)
- Injecting synthetic exceptions in benchmark scan calls (`test_bench.py`)
- Replacing `download_adapter` in hub scanner tests

No `unittest.mock.MagicMock` or `patch` observed in engine/analysis tests — worker tests call `_pool_initializer` and `_worker_entry` directly in-process.

### Class-Based Test Organization

Tests organized into classes (`Test*`) grouping related cases:

```python
class TestPoolInitializer:
    def test_sets_config_hash_global(self) -> None: ...
    def test_sets_cache_root_none_when_empty(self) -> None: ...

class TestWorkerEntry:
    def test_returns_typed_tuple(self, tmp_path: Path) -> None: ...
    def test_missing_file_returns_failed(self, tmp_path: Path) -> None: ...
```

### Schema Migration Tests

Frozen JSON snapshots in `tests/fixtures/` test backward compatibility:

```python
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

class TestScanResultV1Migration:
    def test_loads_v1_fixture(self): ...
    def test_extra_fields_ignored(self): ...
```

Adding a new `schema_version` requires a new fixture file. Removing or renaming a field in an existing fixture is a breaking-change caught by these tests.

## Security Branch Coverage

Tests explicitly cover:
- Tensor bomb guard (oversized tensor rejection)
- Missing metadata handling (DEGRADED status, not ignored)
- Malformed header handling
- Path traversal prevention (resolved paths)
- Worker non-raising guarantee (all exceptions → `ScanResult.errors`)
- Schema version mismatch detection
- Cache miss / hit behavior under corrupted cache entries

## Known Coverage Gaps

- No regression test for RCA-08 OOM (memory view guard bug)
- No tests for `src/adaptersentry/integrations/github_action.py`
- Rust extension (`adaptersentry-rs`) hot-paths never exercised in CI (not built in CI)
- `test_orchestrator_ray.py` skipped when Ray not installed (expected in standard CI)

## Benchmark Tests

`tests/test_bench.py` and `tests/test_harness.py` test the benchmark infrastructure (corpus loading, report generation, hub scanner) — not analysis correctness. Uses `monkeypatch` to avoid real HF Hub calls.

Real corpus benchmarks run separately:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 benchmarks/run_real.py \
    --input-dir <corpus> --workers 8 --mode fast --backend ray \
    --output benchmarks/results/real_500_fast_8w_ray.json
```

## Pre-Commit Gate

Full suite must pass before any commit:
```bash
python3 -m pytest tests/ -q --tb=no
```

Expected output: `773 passed` (or more as new tests are added).

---

*Testing analysis: 2026-05-06*
