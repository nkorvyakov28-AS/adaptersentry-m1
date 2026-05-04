# Scan Modes — `full` vs `fast`

> v0.4.0. Both `adaptersentry scan` and `adaptersentry batch` accept `--mode full|fast`.

## Summary

| | `--mode full` | `--mode fast` |
|--|--------------|--------------|
| **Use for** | Security audits, final verification | Corpus screening, CI pre-filter |
| **SVD** | Full spectrum on lora_A (r×d_in) | Same — SVD runs on A only, already fast |
| **Tensor stats** | Entire tensor | 50K-element deterministic sample for tensors > 100K |
| **IsolationForest** | Always runs (20 trees, 2K samples) | Skipped |
| **ΔW norm** | float32 B@A materialized | Cholesky path, no ΔW materialization |
| **ΔW distribution** | float32 B@A + stride-sample 50K | Proxy (lora_A rows) |
| **Entropy/compression** | O(n), runs in both | O(n), runs in both |
| **Inter-layer similarity** | ΔW stride-sampled to 10K | lora_A rows stride-sampled to 10K |
| **Wasserstein** | 10K stride samples | 10K stride samples |
| **Typical single adapter** | ~40s (168 layers) | ~4.5s (168 layers) |
| **Cache key** | Separate from fast | Separate from full — different `config_hash` |
| **Detection coverage** | Complete | Equivalent for known backdoor patterns |

## When to use each mode

### `full` — security audit

Use when you need the highest confidence result:

- Final check before loading an adapter into production
- Investigating a suspicious adapter flagged by `fast`
- Auditing adapters from an untrusted or unknown source
- Generating a report for security review

```bash
adaptersentry scan suspicious.safetensors --mode full
adaptersentry batch --input-dir ./flagged --mode full --workers 4
```

### `fast` — corpus screening

Use when scanning a large number of adapters for initial triage:

- Scanning a HuggingFace Hub corpus before selective full audit
- CI gate on a model registry with many adapters
- Daily automated scans of a growing adapter library
- Getting a quick risk overview before deep investigation

```bash
# Screen 500 adapters, then full-scan anything HIGH or CRITICAL
adaptersentry batch --input-dir ./adapters --mode fast --workers 8 --fail-on HIGH
```

## Recommended workflow

```
Large corpus (N adapters)
        │
        ▼
adaptersentry batch --mode fast
        │
        ├─── LOW / MEDIUM → allow or manual review
        │
        └─── HIGH / CRITICAL → adaptersentry scan --mode full
                                        │
                                        ├─── confirmed → block
                                        └─── false positive → allow with annotation
```

## What fast mode changes

### SVD computation

SVD runs on `tensor_A` (shape: r×d_in, e.g. 16×4864) — not on the materialized ΔW.
For r=16 and d_in=4864, SVD is fast in both modes (~10ms). The truncated SVD path
(`randomized_svd k=50`) only activates for matrices where `min(rows, cols) ≥ 512`,
which tensor_A never satisfies at typical LoRA ranks (r=4–64).

**Why SVD on A, not ΔW:** ΔW = B@A has rank ≤ r. The singular values of ΔW can be
derived from A's singular values combined with B's. Computing SVD on the full
(d_out × d_in) ΔW matrix would be catastrophically slow for large adapters
(e.g. 896×4864 = 4.3M elements).

### ΔW materialization

Full mode materializes ΔW = B @ A in float32 for norm and distribution families
when `out × in ≤ 4M` (v0.4.0: lowered from 16M to prevent peak-RSS accumulation on
high-rank adapters with 200+ layers). At rank 16 with d_out=896 and d_in=4864:
4.3M elements — just above the threshold, uses proxy/Cholesky path.

Fast mode uses proxy paths:
- **norm**: Cholesky — computes `||B@A||_F` via `||B L||_F` where `A@A^T = L L^T`
- **distribution**: lora_A flattened rows as ΔW proxy (distribution shape of A ≈ ΔW)

### Statistical sampling

Full mode computes kurtosis, skewness, and percentiles on a 50K-element stride sample
of the materialized ΔW. The stride samples across output rows, providing uniform
coverage. Error on percentiles < 0.1% vs the full 4M+ array.

Fast mode returns kurtosis/skewness only (no percentiles) using the A proxy.

### IsolationForest

IsolationForest with 20 trees and 2000 samples runs per layer in full mode.
Fast mode skips it. Z-score outlier detection (O(n)) always runs in both modes
and catches the same class of sparse injection patterns.

### Inter-layer similarity

Both modes cap comparison vectors to 10K elements (stride-sampled).
Full mode uses the actual ΔW (materialized then sampled); fast mode uses
lora_A rows (same size cap). Pairwise comparison runs on at most 100 layers
per shape group → 4950 pairs maximum.

## Cache independence

Fast and full scans produce different `scan_id` values for the same file because
`analyzer_config_hash` includes `scan_mode`. A fast scan result never serves as
a cache hit for a full scan request.

## Detection equivalence

The following signals are **unaffected** by scan mode:

| Signal | Why unaffected |
|--------|---------------|
| `energy_concentration` | SVD on lora_A — same in both modes |
| `RANK_INFLATION` | Effective rank from lora_A SVD — mode-independent |
| `HIGH_KURTOSIS` | 50K sample-stable for heavy-tailed distributions |
| `LOW_ENTROPY` | Entropy computed on full lora_A/B in both modes |
| `delta_norm_ratio` | Norm ratio uses Cholesky path in fast — exact fro_norm |
| `MISSING_ADAPTER_METADATA` | Metadata check is mode-independent |
| `INIT_ONLY` / `PARTIALLY_TRAINED` | Init detector is mode-independent |
| `value_repeat_ratio`, `byte_entropy` | O(n) entropy/compression — both modes |

The following signals may be **slightly reduced** in sensitivity in fast mode:

| Signal | Reduction | Impact |
|--------|-----------|--------|
| `HIGH_ISOLATION_ANOMALY` | Skipped entirely | Low — Z-score covers same class |
| `max_abs_delta`, `mean_abs_delta` | Not computed (proxy path) | Low — fro_norm ratio still available |
| Per-layer percentiles (p01, p99, iqr) | Not computed in fast | Low — kurtosis/skewness still available |
| `inter_layer cosine` | Proxy vs exact ΔW | Very low — structural similarity captured |
