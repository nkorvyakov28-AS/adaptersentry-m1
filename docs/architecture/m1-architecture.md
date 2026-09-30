# M1 static analyzer — architecture (v2.0)

M1 answers one question about a LoRA adapter file, without loading a base model or running
inference: **does anything in these weights look like a planted behaviour, and how sure can
we be?** The answer is a `ScanResult 2.0.0` (see [scan-result.md](../output-schema/scan-result.md)).

## Pipeline

```
adaptersentry.scan(path)                                   src/adaptersentry/scanner.py
│
├─ 1. identity        content/header hashes, provenance            engine/identity.py
├─ 2. config          adapter_config.json next to the file         parsers/adapter_config.py
├─ 3. inventory       validate header, classify every tensor       parsers/adapter_file.py
│                     (no tensor data loaded)                       parsers/safetensors.py
│                                                                   parsers/names.py
├─ 4. per module      stream one LoRA pair at a time:
│                       non-finite check
│                       scale s (α/r, α/√r, patterns)              parsers/adapter_config.py
│                       exact ΔW = s·B·A statistics                 features/lowrank_core.py
│                       ESR features (+ per head)                   features/spectral.py
├─ 5. intra-adapter   robust z within each module family            detectors/intra_adapter.py
├─ 6. structural      red flags and completeness notes              detectors/structural.py
└─ 7. verdict         one rule set, fail-closed                     scoring/verdict.py
```

`scan()` never raises. Any failure becomes a result with `status: failed` and an action of at
least `review`.

## 1–3. Reading an untrusted file

Everything is validated **before** any tensor is allocated (`parsers/safetensors.py`):
regular file only (no FIFOs or devices, also through symlinks), `.safetensors` suffix after
resolution, ≤ 64 GiB, header length ≤ 100 MB and within the file, header is a JSON object;
per entry: dtype ∈ {F32, F16, BF16}, consistent shape/byte length/offsets, ≤ 1 B elements.
The file is then opened once with the safetensors library, which refuses structurally
inconsistent files (as the real loader would).

`open_adapter()` classifies every key into exactly one bucket: LoRA A/B pairs (standard and
embedding LoRA, with layer, expert, module name and kind parsed by fixed linear-time
patterns), full weights (`modules_to_save`), DoRA magnitude vectors, trainable-token deltas,
or not analysable (with the reason). Nothing is dropped silently; everything that is not
analysed ends up in `coverage.not_analyzed` and makes the scan `degraded`.

`adapter_config.json` is untrusted too: bounded regular file, malformed fields ignored and
reported. PEFT treats `rank_pattern` / `alpha_pattern` keys as regular expressions; they are
**not** compiled (a crafted pattern could backtrack indefinitely). Literal patterns are
honoured; others are reported as unresolved.

## 4. Exact statistics of ΔW without building it

A 70B-class MLP update ΔW is 8192 × 28672 (235 M values). All statistics go through the
r × r Gram matrices G_A = A·Aᵀ and G_B = Bᵀ·B:

| Quantity | Formula |
|---|---|
| singular values | σ²(ΔW) = s²·eig(Λ^½ Qᵀ G_B Q Λ^½), with G_A = Q Λ Qᵀ |
| Frobenius norm | ‖ΔW‖² = s²·Σ(G_A ∘ G_B) |
| row / column norms | ‖rowᵢ‖² = s²·bᵢᵀ G_A bᵢ, ‖colⱼ‖² = s²·aⱼᵀ G_B aⱼ |
| inner product of two updates | ⟨B₁A₁, B₂A₂⟩ = tr((B₁ᵀB₂)(A₂A₁ᵀ)) |
| entry kurtosis | mean and variance exact; fourth moment by drawing rows with probability ∝ ‖rowᵢ‖² (importance sampling, unbiased) |

The eigendecomposition of G_A (rather than a Cholesky factor) handles rank-deficient A without
a fallback. Singular values below 10⁻³·σ₁ are treated as bf16 rounding noise for shape
features. Uniform sampling of entries or of a row/column grid is deliberately **not** used for
kurtosis: it misses rare concentrated rows, which is the pattern of an inserted payload.

**ESR features** (`features/spectral.py`) are scale-invariant except `log_rms` and
`log_sigma1`: stable rank and participation ratio (/r), spectral entropy, concentration of
the leading direction and its excess over a random d × r baseline, entry kurtosis, and the
Hoyer / Gini / top-8 concentration of row and column norms (rows are tokens for `lm_head`
and embeddings; embedding LoRA is transposed as PEFT applies it). Attention projections can be
split into heads (q/k/v by output rows, o_proj by input columns).

Memory is bounded by one pair plus ~15 MB for the kurtosis row chunks.

## 5. Intra-adapter comparison

Modules of the same kind should look alike across layers, apart from a smooth trend with
depth. For every family with **≥ 16 modules** and every feature:

1. remove the depth trend with a Theil–Sen line (median of pairwise slopes — robust to the
   outliers being looked for);
2. robust z of the residuals = (x − median) / (1.4826 · MAD) (mean-absolute-deviation
   fallback when MAD = 0; zero-spread families are skipped).

`|z| > 3.5` marks an outlier for the report; `|z| ≥ 8` raises the verdict to `review`. A
large adapter produces thousands of z-values, so the verdict threshold sits far above the
outlier threshold. Both are **uncalibrated** (set on synthetic adapters: clean families ≤ 6.6
from 16 modules up, single injected layers ≥ 48; families under 16 modules were unreliable,
hence the minimum).

## 6. Structural findings

Red flags (verdict ≥ `review`; `block` under `--policy strict`): full `lm_head` / embedding /
router matrices shipped in the adapter; LoRA on a MoE router; ranks different from the
declared `r` without a `rank_pattern` (typical of merged adapters); code, pickle or archive
files next to the adapter. Completeness notes (confidence only): DoRA, unresolved scale
patterns, ignored config fields, target-module mismatch, trainable-token deltas.

## 7. Verdict

`scoring/verdict.py`, in order: failed/degraded parsing → at least `review`; reference
profile p ≤ α_block → `block`, ≤ α_review → `review` (reference profiles are the next
release); intra-adapter |z| ≥ 8 → `review`; red flags → `review` (strict: `block`). The
default policy never blocks without a reference profile. The `ScanResult` model re-checks the
fail-closed invariants on construction.

Confidence is `low` for degraded/failed scans, at most `medium` without a reference profile,
and lists its limiting factors.

## Invariants

- No inference, no base-model loading, no pickle, no `eval`/`exec`, no regex compiled from
  file content.
- Degraded or missing analysis is a signal: it never yields `allow`.
- Peak memory is bounded by one LoRA pair, independent of model size.
- Reports carry the file name, not the absolute path, unless `--full-paths` is given.
