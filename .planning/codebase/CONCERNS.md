# Concerns

**Analysis Date:** 2026-05-06

## Technical Debt

### Dual Pipeline Paths (High Priority)
The codebase has two co-existing analysis pipelines in active tension:

- **Legacy:** `_run_analysis()` in `src/adaptersentry/analyzer.py` returns a flat dict. `worker_main()` still calls `analyzer.scan()` which internally uses this path.
- **Typed (new):** `FeatureExtractor.extract_layer()` + `EnsembleDetector.score_families()` returns typed `FeatureFamilyResult` objects — the intended future path.

The typed path exists but is never called in production; `FeatureExtractor.families_from_record()` serves only as a migration bridge from the legacy flat-dict output. Until `worker_main()` is migrated to call `FeatureExtractor.extract_layer()` directly, the typed pipeline provides no production value.

**Risk:** New feature logic added to the typed path doesn't affect production scans. New logic added to the legacy path violates coding rules.

### Module-Level Scorer Singleton
`src/adaptersentry/analyzer.py` has a module-level `_default_scorer = RiskScorer()` for backward-compat wrappers, while `worker_main()` instantiates `EnsembleDetector()` and `RiskScorer()` per scan. Two instantiation patterns for the same classes increase maintenance surface.

### Deprecated Fields Without Removal Timeline
`DebugReport` carries deprecated `raw_flags` / `raw_layer_stats` fields with no documented removal plan or deprecation warning.

### `psutil` and `huggingface_hub` as Required Core Dependencies
Both are listed in `[project.dependencies]` but used only in benchmarks (`benchmarks/harness.py`, `benchmarks/hub_scanner.py`). All users install them even when not running benchmarks. Should be moved to `[project.optional-dependencies]` extras.

---

## Known Bugs

### Missing Header-Size Guard in `_read_sf_header()` (OOM/DoS Vector)
`src/adaptersentry/parsers/safetensors.py:_read_sf_header()` lacks the 256 MB header-size guard that `src/adaptersentry/engine/identity.py` already implements. A malformed adapter with a crafted header length field can force an OOM allocation before the tensor-bomb guard triggers.

**File:** `src/adaptersentry/parsers/safetensors.py`
**Severity:** HIGH — OOM/DoS on malicious input at the parse boundary.

### `load_adapter()` Skips `path.resolve()` (Symlink Bypass)
The legacy `load_adapter()` function in `src/adaptersentry/parsers/safetensors.py` does not call `path.resolve()` before opening the file. A symlink pointing outside the intended trust boundary can be followed silently. `worker_main()` resolves paths before building `AdapterScanRequest`, but direct callers of `load_adapter()` via the public API are not protected.

**File:** `src/adaptersentry/parsers/safetensors.py:load_adapter()`
**Severity:** MEDIUM — symlink bypass for callers that use the public `load_adapter()` API directly.

### `.bin` Format: Accepted by Manifest, Rejected by Parsers
The manifest accepts `.bin` files as adapter artifacts, but the parser rejects them with a `ValueError` ("Expected .safetensors file"). This creates a confusing user experience where a batch run accepts the file path, queues it, then immediately fails it.

**Files:** `src/adaptersentry/engine/manifest.py`, `src/adaptersentry/parsers/safetensors.py`

---

## Security Concerns

### No `gitleaks` in CI
The OPSEC audit script (`tools/opsec_audit.sh`) has not been ported to this repository. CI has no automated secret scanning. Developers must run the manual pre-push grep manually.

**Risk:** Accidental credential or internal identifier commit to the public repository.
**Mitigation needed:** Port `gitleaks` check to `.github/workflows/ci.yml`.

### `assert self._conn` Guards Disabled Under `python -O`
`src/adaptersentry/engine/manifest.py` uses `assert self._conn` as a runtime guard against uninitialized DB connections. Python's `-O` optimization flag strips `assert` statements — these guards evaporate in optimized builds.

**File:** `src/adaptersentry/engine/manifest.py`
**Severity:** LOW under current deployment model; could become HIGH if `-O` is ever used.

### Missing Header-Size Guard (repeated from Bugs — DoS angle)
See above. In security terms: a crafted adapter header can cause the scanner to allocate unbounded memory before the tensor-bomb guard in `src/adaptersentry/engine/identity.py` runs. The guard exists in identity resolution but not in the parser path that runs first.

---

## Performance Concerns

### `EnsembleDetector()` and `RiskScorer()` Instantiated Per Scan
In `worker_main()`, these objects are constructed on every adapter scan. They carry no mutable state and could be pre-instantiated in `_pool_initializer()` alongside the other pre-imported modules.

**File:** `src/adaptersentry/engine/worker.py`

### `DebugReport` Construction is Mandatory Overhead
`DebugReport` is always assembled in `worker_main()` even when the caller uses `--format summary-json` and never accesses per-layer debug data. This adds unnecessary allocation cost for every scan in summary mode.

**File:** `src/adaptersentry/engine/worker.py`

### `CacheStore` and `ManifestDB` Not Thread-Safe
Documented in code but a refactor trap: any future move toward multi-threaded orchestration will require a significant concurrency overhaul.

**Files:** `src/adaptersentry/engine/cache.py`, `src/adaptersentry/engine/manifest.py`

---

## Fragile Areas

### Bridge Between Flat-Dict and Typed Schema Silently Zero-Fills
`FeatureExtractor.families_from_record()` (the migration bridge) silently zero-fills missing feature fields from the legacy flat-dict output when constructing typed `FeatureFamilyResult` objects. Silent zero-fill masks real absent data.

**File:** `src/adaptersentry/engine/feature_extractor.py`

### Human Summary Silently Omits Score Breakdown on Exception
`src/adaptersentry/reporting/human_summary.py` catches exceptions during score breakdown rendering and silently omits the section rather than surfacing a warning. Callers see an incomplete summary with no indication of the failure.

**File:** `src/adaptersentry/reporting/human_summary.py`

### Rust Extension Not Built in CI
The Rust extension (`adaptersentry-rs/`) provides hot-path numeric acceleration but is never compiled in the CI matrix. Its correctness relative to the Python fallback paths is not tested. A regression in the Rust path would only surface at runtime in environments that explicitly build the extension.

---

## Test Coverage Gaps

- **RCA-08 OOM regression:** No test verifies that slices returned from per-layer extraction functions are `.copy()`-ed (the fix for the memory view OOM bug documented in `docs/internal/decisions/2026-05-03-oom-fix.md`)
- **`integrations/github_action.py`:** Zero tests; GitHub Actions output behavior is entirely untested
- **Rust fast-paths:** Never exercised in CI (extension not built)
- **`assert` guards under `-O`:** No test verifies behavior when Python optimization strips assertions from `ManifestDB`

---

## Dependencies at Risk

| Dependency | Concern |
|---|---|
| `psutil>=5.9.0` | Used only in benchmarks; should be optional extra |
| `huggingface_hub>=0.20.0` | Used only in benchmarks; should be optional extra |
| `ray[default]>=2.9.0` | Correctly optional; `[ray]` extra — no issue |
| `scikit-learn>=1.3.0` | Lower-bound pinning only; no lockfile; version drift risk |
| `scipy>=1.11.0` | Lower-bound pinning only; no lockfile; version drift risk |

---

*Concerns analysis: 2026-05-06*
