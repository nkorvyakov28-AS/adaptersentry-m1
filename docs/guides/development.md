# Development Guide

## Setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or pip).

```bash
git clone https://github.com/nkorvyakov28-AS/adaptersentry-m1
cd adaptersentry-m1
uv sync --frozen --extra dev          # locked dependencies from uv.lock
# or: pip install -e ".[dev]"

uv run pytest tests/ -q               # or: pytest tests/ -q
```

Optional: `uv sync --frozen --extra dev --extra ray` for the Ray batch backend.

The `adaptersentry-rs/` Rust crate is a legacy extension from 1.x and is not used by the 2.x
package; you do not need a Rust toolchain.

## How the code is organised

Start with the docstring of `src/adaptersentry/scanner.py` — it lists the pipeline steps —
and [../architecture/overview.md](../architecture/overview.md) for what each module does.

## Where changes go

**A new per-module feature**

1. Compute it in `features/spectral.py` (`module_features`, and `head_features` if it makes
   sense per head) from the exact low-rank core in `features/lowrank_core.py`. Never build
   ΔW = B·A: memory must stay O((d + k)·r).
2. Prefer shape features that do not depend on the LoRA scale or training length; if a
   feature carries magnitude, use a log scale.
3. To compare it within module families, add it to `_FEATURES` in
   `detectors/intra_adapter.py`.
4. Document it in `docs/output-schema/scan-result.md` and add tests in `tests/features/`.

**A new structural finding** — `detectors/structural.py`. Decide whether it is a red flag
(changes the verdict) or informational, and add the rule to the table in
`docs/output-schema/scan-result.md`.

**A verdict rule** — only in `scoring/verdict.py`. The `ScanResult` model re-checks the
fail-closed invariants; keep them passing.

**A threshold** — if it changes results, include it in `scanner.config_hash()` so cached
results are invalidated. Label uncalibrated thresholds as such in code and docs.

**The result schema** — `schemas/result.py`. Additive changes only within 2.x; then
regenerate `docs/output-schema/scan-result-2.0.0.schema.json` from
`ScanResult.model_json_schema()` (a test fails if it is stale) and update
`tests/fixtures/scan_result_v2.0.0.json` if needed. A change in meaning needs a new major
schema version.

**Parsing** — `parsers/`. Every value from the file is untrusted: validate from the header
before allocating, and put every tensor in exactly one inventory bucket, with a reason if it
is not analysed.

## Security invariants

- No `pickle`, `eval`, `exec`, or compiling regular expressions taken from the file.
- Validate paths with `Path.resolve()`; accept regular files only.
- Reject tensors above 1 B elements (and other header limits) before allocation.
- Missing metadata and degraded parsing are signals: report them, never swallow them.
- Do not load base models or run inference in static analysis.
- Sanitise file-derived strings in human-readable output (`reporting/sanitize.py`).
- A failure anywhere must yield a result with action at least `review`, never `allow`.

## Code conventions

- `pathlib.Path`, never string paths.
- `logging`, never `print()` (except CLI output).
- Type hints and docstrings on public functions.
- No new dependency without updating `pyproject.toml`, `requirements.txt` and `uv.lock`.
- Keep changes focused; no broad refactors in a feature PR.

## Before opening a PR

- `pytest tests/ -q` passes.
- Security-relevant branches have tests.
- Public-facing changes have a `CHANGELOG.md` entry; version bumps touch
  `src/adaptersentry/version.py` and `pyproject.toml`.
- Docs and the JSON Schema match the code.

See [../../CONTRIBUTING.md](../../CONTRIBUTING.md) for commit style.
