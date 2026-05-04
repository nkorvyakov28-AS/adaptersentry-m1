# Development Guide

This guide covers everything needed to contribute to AdapterSentry M1 — from
setting up the environment to adding new detection logic correctly.

---

## Prerequisites

- Python 3.11 or later (3.11, 3.12, and 3.13 are tested in CI)
- Git
- Optional: Rust toolchain (`rustup`) for building the `adaptersentry-rs` extension
- Optional: Ray for the distributed batch backend

---

## Environment setup

Clone the repository and install in editable mode with dev dependencies:

```bash
git clone https://github.com/nkorvyakov28-AS/adaptersentry-m1
cd adaptersentry-m1
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify the install:

```bash
adaptersentry scan --help
python3 -c "import adaptersentry; print(adaptersentry.__version__)"
```

### Optional: Ray backend

The Ray backend (`--backend ray`) provides better crash isolation and horizontal
scalability. Install it as an extra:

```bash
pip install "adaptersentry[ray]"
```

### Optional: Rust extension

`adaptersentry-rs` is a compiled extension that accelerates the hot path —
`tensor_stats`, `byte_entropy`, and `inter_frame_similarity` inner loops.
With Rust, full-mode throughput is approximately 5-6x faster per adapter.

Requires a Rust toolchain and `maturin`:

```bash
pip install maturin
cd adaptersentry-rs
VIRTUAL_ENV=$(python3 -c "import sys; print(sys.prefix)") maturin develop --release
cd ..
```

The Python package detects the extension automatically at import time and falls
back to the pure-Python implementation if the extension is not present.

---

## Running tests

The test gate is mandatory: all tests must pass before any commit.

```bash
# Full suite
python3 -m pytest tests/ -q

# Targeted runs
python3 -m pytest tests/test_analyzer.py -v
python3 -m pytest tests/schemas/ -q
python3 -m pytest tests/engine/ -q
python3 -m pytest tests/features/ -q
```

Currently 773 tests pass. A failing test blocks the commit — do not use
`--ignore` or `-k` to skip tests in order to proceed.

For Rust extension changes, also run the Rust test suite:

```bash
cd adaptersentry-rs && cargo test --release
```

---

## Architecture

The M1 pipeline is strictly layered. Data flows in one direction only:

```
parsers/
    └── has_lora_pairs()       key-only pre-check, no tensor load
    └── load_adapter()         loads safetensors, groups lora_A/lora_B pairs
            │
engine/feature_extractor.py
    └── FeatureExtractor.extract_layer()
            │   returns (TensorRecord, list[FeatureFamilyResult], list[ScanError])
            │
features/
    ├── layer_stats.py         tensor_stats, svd_stats
    ├── distribution.py        DistributionFeatures — kurtosis, skewness, percentiles
    ├── delta_norm.py          NormFeatures — fro_norm, delta_norm_ratio
    ├── entropy_compression.py EntropyCompressionFeatures — byte_entropy, compression_ratio
    ├── inter_layer_similarity.py InterLayerSimilarityFeatures — pairwise cosine + Pearson
    └── tensor_stats.py        low-level stats helpers
            │
detectors/
    ├── wasserstein.py         W1 distance between lora_A and lora_B distributions
    ├── cross_layer.py         cross-layer consistency score
    ├── entropy.py             Shannon entropy detector
    ├── outlier.py             IsolationForest + z-score outlier rate
    └── init_detector.py       INIT_ONLY / PARTIALLY_TRAINED detection
            │
scoring/
    ├── ensemble.py            EnsembleDetector.score_families() — main typed path
    ├── score_breakdown.py     compute_score_breakdown() → ScoreBreakdown (7 families)
    ├── confidence.py          compute_quality_score() + compute_confidence_score()
    └── risk_scorer.py         RiskVerdict thresholds
            │
reporting/
    ├── per_layer.py           rank_layer_findings() → list[PerLayerFinding] top-10
    └── human_summary.py       render_human_summary() → fixed-block CLI text
            │
schemas/
    └── ScanResult v1.0.0      stable public contract (summary-json output)
```

---

## Where to add new features

### New feature family (detection signal)

New detection logic belongs in the `features/` layer and must go through
`FeatureExtractor.extract_layer()` and `EnsembleDetector.score_families()`.

Do NOT add new signals to `_run_analysis()` — that is the legacy flat-dict path
kept for backward compatibility only.

Steps:

1. Create `src/adaptersentry/features/my_feature.py`. Define a `FeatureFamilyResult`
   subclass (via pydantic) and a function that takes `(lora_a, lora_b, delta_w, *, fast)`
   and returns the typed result.
2. Register the family in `engine/feature_extractor.py` — add a call inside
   `extract_layer()` and include the result in the returned list.
3. Handle the family in `scoring/ensemble.py` — add a branch in `score_families()`
   that reads the typed fields and contributes a weighted sub-score.
4. Add the weight to `scoring/risk_scorer.py` `DEFAULT_WEIGHTS`.
5. Export the schema from `schemas/` if it is part of the public contract.
6. Add tests in `tests/features/` and `tests/scoring/`.

### New detector (standalone anomaly scorer)

If the new signal is self-contained and produces a single anomaly score without
needing a full `FeatureFamilyResult`, add it under `detectors/`:

1. Create `src/adaptersentry/detectors/my_detector.py`.
2. Export from `src/adaptersentry/detectors/__init__.py`.
3. Call from `scoring/ensemble.py` inside `score_families()`, not from `_run_analysis()`.
4. Add weight to `scoring/risk_scorer.py` `DEFAULT_WEIGHTS`.
5. Add tests in `tests/test_detectors.py`.

### New output format

Output formats live in `src/adaptersentry/reporters/`. Add a new reporter module,
register it in `cli/scan.py` under the `--format` dispatcher.

---

## Code conventions

### Paths

Always use `pathlib.Path`, never bare strings. Validate untrusted paths before use:

```python
from pathlib import Path

def load_something(path: Path) -> ...:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Not a file: {resolved}")
    ...
```

### Logging

Use `logging`, not `print()`. Diagnostic messages go to the logger; formatted
output goes through the reporter layer.

```python
import logging
log = logging.getLogger(__name__)
log.debug("Loaded %d tensor pairs", n_pairs)
```

### Type hints and docstrings

All public functions require type hints and a one-line docstring minimum.
Internal helpers should have hints where the types are non-obvious.

```python
def compute_delta(lora_a: np.ndarray, lora_b: np.ndarray) -> np.ndarray:
    """Return the effective weight update ΔW = B @ A."""
    return lora_b @ lora_a
```

### Schemas

Use pydantic v2 models for all data that crosses layer boundaries. Set
`model_config = ConfigDict(extra="ignore")` on engine schemas for
forward-compatibility with newer writer versions.

### Sampling and memory

When a function returns a slice of an array that will be stored in a collection
(e.g. accumulated across layers in a dict or list), always call `.copy()` on
the slice. A stride view keeps the full source buffer alive:

```python
# Wrong — view keeps full delta buffer alive across all layers
return delta.flatten()[::stride][:N]

# Correct — copy releases the source buffer on function return
return delta.flatten()[::stride][:N].copy()
```

This is the root cause of the full-mode OOM bug fixed in v0.4.0 (RCA-08).

### Fast / full mode

Every feature function must accept `*, fast: bool = False`. Fast mode is opt-in
for callers; the default is always `fast=False`.

| Signal | Fast mode behaviour |
|--------|---------------------|
| `compute_tensor_stats` | 50K-element sample when `numel > 100K` |
| `compute_svd_stats` | randomized SVD k=50 when `min(shape) >= 512` |
| `detect_outlier_anomalies` | skip IsolationForest when `numel > 5M` |
| `inter_layer_similarity` | use lora_A rows as ΔW proxy instead of materializing ΔW |
| `EntropyCompressionFeatures` | O(n), runs identically in both modes |

---

## Security invariants

These invariants are non-negotiable. Any PR that violates them will not be merged.

- Never call `eval()`, `exec()`, or `pickle.loads()` on adapter-controlled input.
- Validate all file paths with `Path.resolve()` before opening them.
- Reject tensors with more than 1 billion elements before allocation (tensor bomb guard).
- Do not load base model weights in M1. The analyzer is read-only on the adapter file only.
- Do not add inference or execution behaviour to static-analysis code paths.
- Missing or malformed metadata is a security signal, not a cosmetic issue — surface it
  as a structured `ScanError` with the appropriate error code, do not silently skip.
- Parser failures must produce a structured `ParseStatus.FAILED` result, not an
  uncaught exception.

---

## Commit hygiene

### Conventional commit format

```
<type>(<scope>): <short description>

<optional body — explain why, not what>
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`, `bench`

Examples:

```
feat(features): add quantization_suspect_score to EntropyCompressionFeatures
fix(engine): copy stride slice in _make_vector to prevent buffer retention
perf(rust): accelerate byte_entropy inner loop with SIMD
test(scoring): add regression test for score_breakdown on INIT_ONLY adapters
docs: document fast/full mode sampling thresholds
```

### One logical change per commit

Split unrelated changes into separate commits. Atomicity matters more than commit
count. If a commit message requires "and" to describe what it does, split it.

### Test gate

Run the full suite before committing:

```bash
python3 -m pytest tests/ -q
```

All tests must pass. Do not commit with known failures.

---

## Checklist before submitting a PR

1. All 773 tests pass (`pytest tests/ -q`).
2. New behaviour has corresponding tests.
3. Public API changes have updated docstrings.
4. No new `print()` calls — use `logging`.
5. No hardcoded paths — use `pathlib.Path`.
6. No new dependencies without updating `pyproject.toml`.
7. Schema changes: increment `schema_version`, add a migration test, update the fixture
   at `tests/fixtures/scan_result_v1.0.0.json`.
8. `ΔW = B @ A` — never score lora_A or lora_B in isolation as a primary signal.
9. `confidence_score` and `quality_score` must not be derived from the same anomaly
   features as the risk score (circular logic guard — see `scoring/confidence.py`).
10. Security checklist in `SECURITY.md` reviewed for any code touching file I/O or
    external input.
