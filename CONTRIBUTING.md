# Contributing to AdapterSentry

## Setup

```bash
git clone https://github.com/nkorvyakov28-AS/adaptersentry-m1
cd adaptersentry-m1
uv sync --frozen --extra dev          # or: pip install -e ".[dev]"
pytest tests/ -q
```

The full suite must pass before a PR is merged. No Rust toolchain is needed.

## Where to start

- [docs/architecture/overview.md](docs/architecture/overview.md) — how a scan works
- [docs/guides/development.md](docs/guides/development.md) — where changes go, security
  invariants, conventions
- [docs/guides/testing.md](docs/guides/testing.md) — test layout and rules

## Rules

- Python 3.11+; type hints and docstrings on public functions; `pathlib.Path`; `logging`
  instead of `print()`.
- Treat everything in an adapter file as hostile: validate before allocating, never
  deserialise pickle or evaluate content from the file.
- Failures must be fail-closed: a scan that could not complete is never `allow`.
- Tests use synthetic adapters written with `safetensors.numpy` (`tests/adapter_factory.py`);
  no torch, no committed weight files.
- Do not claim detection accuracy that has not been measured; label uncalibrated thresholds
  as such.
- Changes to `ScanResult` follow the schema rules in the development guide.

## Commits

Conventional commits: `feat:`, `fix:`, `perf:`, `refactor:`, `test:`, `docs:`, `chore:`,
`bench:` (scopes such as `fix(cli):` are fine). One logical change per commit; the message
explains **why** — the diff shows what.

## Security

Report vulnerabilities in AdapterSentry privately, as described in [SECURITY.md](SECURITY.md);
do not open public issues for them. A malicious adapter found in the wild can be reported as a
GitHub issue labelled `malicious-adapter`.
