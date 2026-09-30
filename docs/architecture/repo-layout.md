# Repository Layout

```
adaptersentry-m1/
├── src/adaptersentry/          installable package
│   ├── __init__.py             public API: scan, load_scan_result, ScanResult
│   ├── __main__.py             python -m adaptersentry
│   ├── scanner.py              scan() — the pipeline
│   ├── version.py
│   ├── parsers/                safetensors guards, adapter inventory + streaming,
│   │                           adapter_config.json, tensor-name parsing
│   ├── features/               exact low-rank ΔW statistics, per-module / per-head features
│   ├── detectors/              intra-adapter comparison, structural findings
│   ├── scoring/                verdict rules
│   ├── schemas/                ScanResult 2.0.0, errors, severity
│   ├── reporters/              text, JSON, SARIF
│   ├── reporting/              sanitising of untrusted strings for terminal output
│   ├── engine/                 batch engine: manifest, cache, identity, workers, orchestrators
│   └── cli/                    adaptersentry scan / batch
├── tests/                      pytest suite (mirrors src/); adapter_factory.py writes synthetic adapters
├── benchmarks/                 benchmark harness and Hub scanner (optional [bench] extra)
├── adaptersentry-rs/           legacy Rust extension; not used by the 2.x package
├── docs/                       architecture, guides, CLI, output schema (plain Markdown)
├── tools/                      release helper
└── .github/workflows/          CI and release
```

## Rules

1. Product code lives only in `src/adaptersentry/`.
2. `benchmarks/` uses `adaptersentry` as a library and is not part of the core package.
3. Tests use synthetic adapters written with `safetensors.numpy`; no weight files are
   committed (`*.safetensors`, `*.bin`, `*.pt`, `*.ckpt` are ignored).
4. Releases are git tags and GitHub Releases.
