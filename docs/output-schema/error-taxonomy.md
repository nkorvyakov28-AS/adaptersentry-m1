# What was not analysed, and why

A scan that could not look at part of an adapter is a security signal, not a detail: a payload
can live exactly where the scanner did not look. AdapterSentry reports this in two places.

## `coverage.not_analyzed` — per tensor

`status` is `degraded` when at least one tensor is listed here; the verdict is then at least
`review` (reason `DEGRADED_PARSE`).

| `category` | Typical `reason` |
|---|---|
| `malformed` | invalid shape or byte range in the header; unsupported dtype (only F32/F16/BF16 are read); LoRA rank above 1024; A without B (or the reverse); A/B rank mismatch; load failure |
| `unsupported` | full weight matrix (`modules_to_save`) — needs the base model; trainable-token delta; a tensor outside a LoRA pair (bias, norms, …) |
| `malformed` (in `non_finite_modules`) | NaN or Inf weights — every comparison with NaN is false, so such a layer would otherwise look clean |

The list is capped at 50 entries; `coverage.n_not_analyzed` holds the total.

## `errors` — the scan itself

`status` is `failed` when nothing could be analysed; the verdict is `review` (reason
`PARSE_FAILED`) and the CLI exits with code 1.

```
ScanError(category, code, message, detail, phase, severity)
```

| `code` | When |
|---|---|
| `INVALID_SAFETENSORS` | not a regular file; wrong suffix after resolving symlinks; larger than 64 GiB; file shorter than 8 bytes; header length above 100 MB or beyond the file; header not a JSON object; structurally inconsistent header (the safetensors loader would refuse the file too); no LoRA A/B pair at all |

## What is never done

Pickle-based files (`.bin`, `.pt`) are never deserialised, and nothing from the file is
executed, evaluated, or compiled as a regular expression (PEFT `rank_pattern` /
`alpha_pattern` regexes are reported as `UNRESOLVED_SCALE_PATTERN` instead).
