# Testing Guide

## Run

```bash
uv sync --frozen --extra dev
uv run pytest tests/ -q               # full suite; must pass before every commit
```

Targeted runs:

```bash
pytest tests/parsers -q      # header guards, adapter_config.json, streaming, hostile inputs
pytest tests/features -q     # exact low-rank statistics, per-module / per-head features
pytest tests/detectors -q    # intra-adapter comparison
pytest tests/scoring -q      # verdict rules
pytest tests/schemas -q      # ScanResult 2.0.0 contract, invariants, published JSON Schema
pytest tests/engine -q       # cache, identity, manifest, worker, Ray orchestrator
pytest tests/cli tests/reporters tests/reporting -q
pytest tests/test_scanner.py -q   # end-to-end scan() on synthetic adapters
```

Tests that need optional pieces skip themselves when they are missing: the Ray tests without
`ray`, and `tests/test_rust_extension.py` without the legacy Rust extension.

## Layout

| Path | Covers |
|---|---|
| `tests/adapter_factory.py` | `write_adapter()` — writes a synthetic PEFT adapter (and `adapter_config.json`) with configurable layers, rank, config, extra tensors and an optional injected outlier module |
| `tests/parsers/` | `parsers/` |
| `tests/features/` | `features/` (exactness against a materialised ΔW on small shapes) |
| `tests/detectors/` | `detectors/intra_adapter.py` |
| `tests/scoring/` | `scoring/verdict.py` |
| `tests/schemas/` | `schemas/` and `tests/fixtures/scan_result_v2.0.0.json` |
| `tests/engine/` | `engine/` |
| `tests/cli/`, `tests/reporters/`, `tests/reporting/` | CLI, output formats, text sanitising |
| `tests/test_scanner.py` | the full pipeline |
| `tests/test_bench.py`, `tests/test_harness.py` | benchmark utilities (no network) |

## Conventions

- **No torch.** Write tensors with `safetensors.numpy`.
- **Synthetic adapters only.** Build them with `tests/adapter_factory.py` in `tmp_path`;
  never commit weight files or depend on files outside the repository.
- **No hard-coded paths.** Use `tmp_path` and paths relative to the test file.
- **Test security branches.** Every guard in `parsers/` and every fail-closed path should have
  a test with a hostile or malformed input.
- **Test exactness where it is claimed.** Features computed through the low-rank core are
  compared with the same quantity on an explicitly built ΔW for small shapes.
- **Schema contract.** If `ScanResult` changes, regenerate
  `docs/output-schema/scan-result-2.0.0.schema.json` from `ScanResult.model_json_schema()`;
  `tests/schemas/test_result.py` fails when it is stale.

## Benchmarks

The harness in `benchmarks/` is separate from the test suite; see
[benchmarks/README.md](../../benchmarks/README.md) and
[../benchmarks/methodology.md](../benchmarks/methodology.md). Benchmark numbers are
measurements of speed and memory, not of detection accuracy.
