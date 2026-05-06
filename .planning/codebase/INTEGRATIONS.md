# External Integrations

**Analysis Date:** 2026-05-06

## APIs & External Services

**HuggingFace Hub:**
- Used for benchmarking only — the `adaptersentry-bench` CLI entry point fetches public LoRA adapter repos for observational benchmarking
- SDK/Client: `huggingface_hub` (`HfApi`, `hf_hub_download`) — lazily imported in `benchmarks/hub_scanner.py`
- Auth: No token required for public repos; HF tokens may be provided via standard `HF_TOKEN` env var (handled transparently by `huggingface_hub` SDK)
- Downloaded files: only `adapter_model.safetensors` and `adapter_config.json` — base model weights are never fetched
- NOT used in the core M1 scan pipeline (`src/adaptersentry/`) — only in `benchmarks/`

**GitHub Actions (outgoing integration):**
- M1 scan results are emitted as GitHub Actions step outputs and step summary via `src/adaptersentry/integrations/github_action.py`
- Outputs set: `risk_level`, `ensemble_score`, `n_findings` (via `GITHUB_OUTPUT` file)
- Step summary: Markdown table of scan results written to `GITHUB_STEP_SUMMARY`
- Auth: None — uses GitHub-provided env vars `GITHUB_OUTPUT`, `GITHUB_STEP_SUMMARY`, `GITHUB_ACTIONS`
- Detection: `is_github_actions()` checks `GITHUB_ACTIONS == "true"` before any writes

**GitHub Code Scanning (SARIF):**
- M1 produces SARIF 2.1.0 output (`src/adaptersentry/reporters/sarif.py`) consumable by `github/codeql-action/upload-sarif`
- SARIF schema URI: `https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json`
- Severity mapping: CRITICAL/HIGH → `error`, MEDIUM → `warning`, LOW → `note`
- Security-severity scores (CVSS-like): CRITICAL=9.0, HIGH=7.5, MEDIUM=5.0, LOW=2.5

## Data Storage

**Databases:**
- SQLite — two internal SQLite databases, both local filesystem only:
  - `~/.adaptersentry/manifest.sqlite`: batch scan state machine (`src/adaptersentry/engine/manifest.py`); WAL mode, per-run job lifecycle tracking
  - `<cache_root>/index.sqlite`: content-addressed result cache index (`src/adaptersentry/engine/cache.py`); WAL mode, keyed by (content_hash, analyzer_config_hash)
- Both use `sqlite3` from Python stdlib — no ORM, raw SQL with parameterized queries
- No network-attached database; no PostgreSQL, MySQL, or cloud DB

**File Storage:**
- Local filesystem only
- Cache objects: `<cache_root>/objects/{hash[:2]}/{hash[2:]}.gz` — gzip-compressed ScanResult JSON
- Batch results: `results/<run_id>/` — per-adapter JSON files + `run.jsonl` audit log + `run_summary.json`
- No S3, GCS, or other object storage integration

**Caching:**
- Content-addressed local cache in `src/adaptersentry/engine/cache.py`
- Cache key: `(content_hash, analyzer_config_hash)` — SHA-256 of file bytes + SHA-256 of active detector config
- Poisoning guard: every cache read re-hashes the compressed result file and compares to stored `result_hash`; mismatch deletes the entry (fail-closed)
- Writer version guard: entries from future analyzer versions are rejected by older readers

## Authentication & Identity

**Auth Provider:**
- None — M1 is a local static analysis tool with no user authentication
- No login, no API keys, no sessions

**Content Identity:**
- Adapter files are identified by SHA-256 content hash + SHA-256 safetensors header hash, computed in `src/adaptersentry/engine/identity.py`
- HF Hub adapters get a stable `logical_id` derived from `sha256(repo_id:revision:filename)` — stable across file copies

## Monitoring & Observability

**Error Tracking:**
- None — no Sentry, Datadog, or external error tracking service
- Errors are structured as `ScanError` objects (`src/adaptersentry/schemas/errors.py`) embedded in `ScanResult`

**Logs:**
- Python `logging` module throughout; never `print()`
- Log level controlled by `--verbose` CLI flag (DEBUG) or defaults to WARNING
- No external log shipping; logs go to stderr via `logging.basicConfig`

**Benchmarks / Profiling:**
- `benchmarks/harness.py` uses `psutil.Process` for per-process memory monitoring during benchmark runs
- Ray dashboard available at `http://localhost:8265` when using the Ray backend (`--backend ray`)

## CI/CD & Deployment

**Hosting:**
- Package published to PyPI via `pypa/gh-action-pypi-publish` (trusted publisher; configured on pypi.org)
- GitHub Releases created by `softprops/action-gh-release` with auto-generated release notes

**CI Pipeline:**
- GitHub Actions; workflow files: `.github/workflows/ci.yml`, `.github/workflows/release.yml`
- CI: pytest on Python 3.11/3.12/3.13 matrix + import/CLI smoke check on every push to `main` and PRs
- Release: triggered on `v*.*.*` tags; runs tests, builds sdist+wheel, publishes to PyPI, creates GitHub Release

**Build:**
- `python -m build` produces sdist and wheel
- No Docker images, no Kubernetes manifests

## Webhooks & Callbacks

**Incoming:**
- None — no webhook endpoints; M1 is a CLI/library tool, not a service

**Outgoing:**
- None — no outgoing HTTP webhooks
- GitHub Actions integration writes to local files (`GITHUB_OUTPUT`, `GITHUB_STEP_SUMMARY`) provided by the Actions runner, not outgoing network calls

## Environment Configuration

**Required env vars:**
- None for core scan functionality — M1 works offline with no env vars set

**Optional env vars:**
- `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` — BLAS thread cap; set to `1` in worker initializer (`src/adaptersentry/engine/orchestrator.py`, `src/adaptersentry/engine/orchestrator_ray.py`)
- `HF_TOKEN` — HuggingFace authentication for gated models (benchmarks only; handled by `huggingface_hub` SDK)
- `GITHUB_ACTIONS`, `GITHUB_OUTPUT`, `GITHUB_STEP_SUMMARY` — GitHub Actions runner env vars; consumed by `src/adaptersentry/integrations/github_action.py`

**Secrets location:**
- No application secrets; no credential files present in the repository

---

*Integration audit: 2026-05-06*
