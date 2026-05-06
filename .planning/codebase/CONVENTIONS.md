# Code Conventions

**Analysis Date:** 2026-05-06

## Code Style

- **Language:** Python 3.11+ with full type hints on all public functions and methods
- **Docstrings:** Module-level and class-level docstrings on all public APIs; inline comments only for non-obvious invariants or security rationale
- **Imports:** `from __future__ import annotations` at top of every module; stdlib then third-party then internal, separated by blank lines
- **Formatting:** No explicit formatter config found; consistent 4-space indentation throughout
- **Line length:** Not enforced by a linter in CI; practical max ~100 chars observed
- **No `print()`:** `logging` module used exclusively for diagnostics; `rich.Console` used only in the CLI text reporter

## Naming Conventions

| Construct | Convention | Example |
|---|---|---|
| Modules | `snake_case` | `feature_extractor.py`, `risk_scorer.py` |
| Public classes | `PascalCase` | `EnsembleDetector`, `ScanResult`, `FeatureExtractor` |
| Private functions | leading `_` | `_run_analysis()`, `_pool_initializer()`, `_sha256()` |
| Private attributes | leading `_` | `_default_scorer`, `_WEIGHT_SUM` |
| Constants | `UPPER_SNAKE_CASE` | `DETECTOR_WEIGHTS`, `LEASE_DURATION_SECS`, `_SIGMOID_SCALE` |
| Pydantic models | `PascalCase` | `AdapterScanRequest`, `TensorRecord`, `FeatureFamilyResult` |
| Test helpers | `_make_` prefix | `_make_adapter()`, `_make_req()` |
| Test classes | `Test` prefix | `TestPoolInitializer`, `TestWorkerEntry`, `TestScanResultV1Migration` |

## File Organization Patterns

- **Schemas before logic:** All typed data contracts (`schemas/`, `engine/schemas/`) defined separately from the code that uses them
- **src-layout:** Package lives under `src/adaptersentry/`; not importable without install (editable install used in dev)
- **Mirror structure:** `tests/` mirrors `src/adaptersentry/` subdirectory layout exactly
- **Entry points in `__init__.py`:** Public API explicitly re-exported from `src/adaptersentry/__init__.py`
- **Version in one place:** `src/adaptersentry/version.py` is the single source of truth; referenced from `pyproject.toml`, `tests/cli/test_scan_cli.py`

## Error Handling

**Workers never raise.** `worker_main()` catches all exceptions and returns a `ScanResult` with `status=FAILED` and errors in `ScanResult.errors`. The orchestrator's result iterator always receives an object.

**Structured errors over unstructured exceptions.** Failures use `ScanError(code=ErrorCode.X, phase=ScanPhase.Y, message=...)` — typed, queryable, and never swallowed silently.

**Degraded is a valid state.** A scan that completes with partial data returns `status=DEGRADED`, not `FAILED`. Callers can distinguish "no result" from "partial result."

**Parser errors propagate up.** `parsers/safetensors.py` raises `FileNotFoundError`/`ValueError` for file-level failures; these are caught by `worker_main` and turned into structured errors.

**Never swallow unknown exceptions silently.** Broad `except Exception as exc:` blocks always `logger.error()` or `logger.warning()` with the exception before continuing.

**Security signals over silent failures:**
- Missing metadata → `DEGRADED` status, not ignored
- Degraded parsing → structured `ScanError`, not skipped
- Path traversal → `pathlib.Path.resolve()` before any use

## Key Patterns

**Pydantic for all schemas:**
```python
class ScanResult(BaseModel):
    model_config = ConfigDict(extra="ignore")  # safe forward-compat
    schema_version: str = Field(default="1.0.0")
    status: ScanStatus
    errors: list[ScanError] = Field(default_factory=list)
```

**`pathlib.Path` everywhere, never string paths:**
```python
adapter_path = Path(req.adapter_path).resolve()
cache_root = Path(cache_root_str) if cache_root_str else None
```

**Logging with `%s` format strings (not f-strings):**
```python
logger.error("Identity resolution failed for %s: %s", req.adapter_path, exc)
logger.warning("Cache check failed for %s: %s — treating as miss", req.adapter_path, exc)
```

**BLAS thread capping in workers (set before numpy import):**
```python
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
```

**Typed return tuples over dicts:**
```python
def worker_main(req: AdapterScanRequest, ...) -> tuple[ScanResult, DebugReport]:
def extract_layer(self, name: str, a: np.ndarray, b: np.ndarray, ...) -> tuple[TensorRecord, list[FeatureFamilyResult], list[ScanError]]:
```

## Security Anti-Patterns (Explicitly Banned)

- No `eval()`, `exec()`, `pickle` on adapter-controlled input
- No `safetensors.torch` (brings torch into the dependency tree); always use `safetensors.numpy`
- No full base model loading in M1
- No inference code paths in static analysis code
- No string paths when `pathlib.Path` is available
- No broad exception swallowing without logging
- No new feature logic in `_run_analysis()` (legacy path); use `FeatureExtractor` instead

## Memory Safety (RCA-08)

Never return `arr[::stride][:N]` as a view from a function whose result is stored in a collection. Views keep the full `arr` buffer alive, causing OOM across batches. Always `.copy()` when the slice will outlive the function scope.

```python
# BAD — view keeps full buffer alive
return arr[::stride][:N]

# GOOD — copy releases the large buffer
return arr[::stride][:N].copy()
```

See `docs/internal/decisions/2026-05-03-oom-fix.md`.

---

*Conventions analysis: 2026-05-06*
