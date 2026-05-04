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
/root/lora_env/bin/python3 -m pytest tests/ -q   # full suite (773 tests)
/root/lora_env/bin/python3 -m pytest tests/test_analyzer.py -v  # single file
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
3. Call from `analyzer._run_analysis()`.
4. Add weight to `scoring/risk_scorer.py` `DEFAULT_WEIGHTS`.
5. Add a test in `tests/test_detectors.py`.

## Commit style

Conventional commits: `feat(m1):`, `fix(cli):`, `test(schemas):`, `docs:`, `refactor:`.

## Security

See [SECURITY.md](SECURITY.md) for the responsible disclosure process.
Do not open public issues for vulnerabilities in AdapterSentry itself.
