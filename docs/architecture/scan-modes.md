# Scan Modes — `full` vs `fast`

`adaptersentry scan`, `adaptersentry batch` and `adaptersentry.scan()` accept
`mode="full"` (default) or `mode="fast"`.

The two modes differ in **one** thing: the number of rows sampled to estimate the kurtosis
of ΔW entries (`kurt_dw`).

| | `full` | `fast` |
|---|---|---|
| Rows sampled for the kurtosis estimate (per module) | 256 | 64 |
| Singular values, norms, row/column norms, all other features | exact | exact |
| Intra-adapter comparison, structural findings, verdict rules | same | same |

Everything except `kurt_dw` is computed exactly from the r×r Gram matrices in both modes
(`features/lowrank_core.py`). The kurtosis estimate uses importance sampling of rows, so rare
rows with large changes are still likely to be picked in `fast` mode, but its variance is
higher and a borderline kurtosis z-score can differ between the modes.

## Which to use

- `full` — the default; use it for any verdict you act on.
- `fast` — screening very large corpora where scan time matters more than the precision of
  one feature. The exact spectral features are computed in both modes, so `fast` saves only
  the extra kurtosis sampling; measure on your own adapters before relying on it.

## Caching

The mode is part of `config_hash` (`scanner.config_hash(mode, policy)`), and so of `scan_id`
and the batch cache key. A `fast` result is never served for a `full` request, or the reverse.
