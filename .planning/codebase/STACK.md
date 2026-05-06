# Technology Stack

**Analysis Date:** 2026-05-06

## Languages

**Primary:**
- Python 3.11+ - All application logic, CLI, analysis pipeline, schemas, tests
- Rust (edition 2021) - Hot-path numeric extensions (`adaptersentry-rs/`)

**Secondary:**
- None (no JavaScript, TypeScript, or shell scripts in the core product)

## Runtime

**Environment:**
- Python 3.13 (dev VPS; CI matrix covers 3.11, 3.12, 3.13)
- Interpreter path: project virtualenv (`python3`)

**Package Manager:**
- pip (setuptools backend)
- Lockfile: Not present — dependencies pinned by lower bounds only in `pyproject.toml`

## Frameworks

**Core:**
- pydantic 2.5+ — All schema and data model validation; `BaseModel` used throughout `src/adaptersentry/schemas/` and `src/adaptersentry/engine/schemas/`
- safetensors 0.4+ — Read-only memory-mapped parsing of `.safetensors` adapter files via `safetensors.numpy` (`src/adaptersentry/parsers/safetensors.py`)

**Numeric/ML:**
- numpy 1.24+ — Array operations, tensor math throughout `src/adaptersentry/features/` and `src/adaptersentry/detectors/`
- scipy 1.11+ — `scipy.stats.wasserstein_distance` used in `src/adaptersentry/detectors/wasserstein.py`
- scikit-learn 1.3+ — `sklearn.ensemble.IsolationForest` in `src/adaptersentry/detectors/outlier.py`; `sklearn.utils.extmath.randomized_svd` in `src/adaptersentry/features/tensor_stats.py`

**CLI/Output:**
- rich 13.0+ — Terminal pretty-print in `src/adaptersentry/analyzer.py` (Console, Syntax) for `--format text` output
- argparse (stdlib) — CLI argument parsing in `src/adaptersentry/cli/main.py`, `scan.py`, `batch.py`

**Testing:**
- pytest 7.4+ — Test runner; config in `pyproject.toml` `[tool.pytest.ini_options]`

**Build/Dev:**
- setuptools 68+ / wheel — Python package build
- maturin — Rust extension build (optional; for `adaptersentry-rs/`)

## Key Dependencies

**Critical:**
- `safetensors>=0.4.0` — Core adapter file I/O; all tensor parsing goes through this library; `safetensors.numpy` used (NOT `safetensors.torch`) to keep torch out of the dependency tree
- `pydantic>=2.5.0` — Schema contracts; `ScanResult.schema_version = "1.0.0"` is the stable public contract enforced via pydantic models
- `numpy>=1.24.0` — All feature computation; used in every detector and feature module
- `scipy>=1.11.0` — Wasserstein distance detector (`src/adaptersentry/detectors/wasserstein.py`)
- `scikit-learn>=1.3.0` — IsolationForest outlier detector and randomized SVD (`src/adaptersentry/detectors/outlier.py`, `src/adaptersentry/features/tensor_stats.py`)

**Infrastructure:**
- `huggingface_hub>=0.20.0` — Used in benchmark pipeline (`benchmarks/hub_scanner.py`) for discovering and downloading public LoRA adapter repos from HF Hub; also listed as a core dependency for the `adaptersentry-bench` CLI entry point
- `psutil>=5.9.0` — Process memory monitoring in benchmarks (`benchmarks/harness.py`)
- `rich>=13.0.0` — Terminal output formatting; used in `src/adaptersentry/analyzer.py` for the legacy text format path

**Optional / Extras:**
- `ray[default]>=2.9.0` — Ray actor pool backend for batch scanning; installed with `pip install adaptersentry[ray]`; used in `src/adaptersentry/engine/orchestrator_ray.py`; falls back to `multiprocessing.Pool` if not installed

**Rust Extension (Optional, OPT-04):**
- `pyo3 0.22` — Python/Rust FFI via PyO3 extension module
- `numpy 0.22` (Rust crate) — Buffer protocol access to numpy arrays from Rust
- Built with `maturin develop --release`; extension is `adaptersentry_rs`
- Python call sites fall back to numpy implementations if the Rust extension is not compiled

## Configuration

**Environment:**
- No `.env` files used; no runtime secrets required for core analysis
- Worker BLAS threads controlled via `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` (set to `1` in worker initializer to prevent CPU over-subscription)
- Cache location: `~/.adaptersentry/cache/` (default; overridden via CLI `--cache-dir`)
- Manifest location: `~/.adaptersentry/manifest.sqlite` (default batch state DB)
- GitHub Actions integration reads `GITHUB_OUTPUT` and `GITHUB_STEP_SUMMARY` env vars

**Build:**
- `pyproject.toml` — single source of truth for package metadata, dependencies, entry points, and pytest config
- `requirements.txt` — mirrors `[project.dependencies]`; provided for `pip install -r` workflows
- `adaptersentry-rs/Cargo.toml` — Rust crate config (edition 2021, opt-level 3, LTO thin)

## Platform Requirements

**Development:**
- Python 3.11 or newer (3.11/3.12/3.13 all supported per CI matrix)
- Rust stable toolchain + maturin (optional; only needed to rebuild the Rust extension)
- No Node.js, Docker, or other runtimes required

**Production:**
- Deployed as a pip-installable Python package published to PyPI
- CI: GitHub Actions (ubuntu-latest), triggers on push to `main` and all PRs
- Release: `pypa/gh-action-pypi-publish` + `softprops/action-gh-release` on version tags

---

*Stack analysis: 2026-05-06*
