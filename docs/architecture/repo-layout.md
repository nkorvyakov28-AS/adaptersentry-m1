# Repository Layout

```
adaptersentry/
├── src/adaptersentry/       ← installable OSS package
│   ├── analyzer.py          ← main entry point: analyze() / scan()
│   ├── version.py
│   ├── __init__.py          ← public API surface
│   ├── __main__.py          ← python -m adaptersentry
│   ├── cli/                 ← argument parsing, output routing
│   ├── parsers/             ← file loading, metadata extraction
│   ├── features/            ← tensor/layer statistics
│   ├── detectors/           ← per-signal anomaly detectors
│   ├── scoring/             ← risk scoring and ensemble
│   ├── schemas/             ← Pydantic report contracts (v1.0.0)
│   ├── engine/              ← batch scan engine (worker pool, cache, manifest)
│   ├── reporters/           ← text / JSON / SARIF output
│   └── integrations/        ← GitHub Actions helper
├── benchmarks/              ← benchmark harness and corpus generator
├── tests/                   ← pytest suite (mirrors src layout)
├── docs/                    ← architecture, schema, CLI docs (public)
├── docs/internal/           ← internal RFCs and roadmap (gitignored)
├── scripts/                 ← release utilities (gitignored)
└── .github/workflows/       ← CI and release automation
```

## Rules

1. `src/adaptersentry/` is the only location for installable product code.
2. `benchmarks/` consumes `adaptersentry` as a library; it is not part of the core package.
3. `docs/internal/` is gitignored — internal RFCs, roadmap details, and competitive analysis live here.
4. `docs/` (public) is flat Markdown — no build step required.
5. Releases are managed via git tags and GitHub Releases — no `releases/` directory in the repo.
6. Tests mirror the package structure where there is enough material to warrant a subdirectory.
