# Contributing to AdapterSentry

## Development setup

```bash
git clone https://github.com/nkorvyakov28-AS/adaptersentry
cd adaptersentry
pip install -e ".[dev]"

# Optional: Rust hot-path extensions (requires Rust toolchain)
pip install maturin
cd adaptersentry-rs
VIRTUAL_ENV=$(python -c "import sys; print(sys.prefix)") maturin develop --release
cd ..
```

## Running tests

```bash
pytest tests/ -q                      # full suite (773 tests)
pytest tests/test_analyzer.py -v      # single file
```

All 773 tests must pass before submitting a PR.

## Code conventions

- Python 3.11+; all public functions have type hints and docstrings.
- No `print()` — use `logging` for diagnostics, reporters for output.
- No hardcoded paths — use `pathlib.Path`.
- Security review required for any code that touches file I/O or external data:
  see `SECURITY.md` for the checklist.

## Adding a detector

1. Create `src/adaptersentry/detectors/my_detector.py`.
2. Export from `src/adaptersentry/detectors/__init__.py`.
3. Wire through `engine/feature_extractor.py` (`FeatureExtractor.extract_layer()`) and
   `EnsembleDetector.score_families()` — do **not** add logic to the legacy
   `analyzer._run_analysis()` flat-dict path.
4. Add weight to `scoring/risk_scorer.py` `DEFAULT_WEIGHTS`.
5. Add a test in `tests/test_detectors.py`.

## Batch backend

The batch engine supports two backends selectable via `--backend`:

- `mp` — multiprocessing pool (default, single-machine)
- `ray` — Ray actor pool (`--backend ray`); enables crash isolation and horizontal scaling

Run benchmarks with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` to
avoid BLAS thread over-subscription on multi-worker runs.

## Commit style

Conventional commits: `feat:`, `fix:`, `perf:`, `refactor:`, `test:`, `docs:`, `chore:`, `bench:`.
Scoped variants are fine where helpful (e.g. `fix(cli):`, `test(schemas):`).
One logical change per commit; messages describe **why**, not what (the diff shows what).

## Security

See [SECURITY.md](SECURITY.md) for the responsible disclosure process.
Do not open public issues for vulnerabilities in AdapterSentry itself.
