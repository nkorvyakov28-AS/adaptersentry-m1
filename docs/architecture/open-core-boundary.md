# Open-Core Boundary

AdapterSentry M1 — the static analyzer in this repository — is open source under
Apache 2.0.

| Layer | Status | Scope |
|---|---|---|
| M1 Static Analyzer | Available (this repository) | Static analysis of adapter weights and files |
| M2 Behavioral Sandbox | Planned | Advanced behavioral analysis |
| M3, M4 | Planned | — |

Nothing beyond M1 is part of this package.

## What M1 contains

Everything under `src/adaptersentry/`: parsers, features, detectors, verdict rules, the
`ScanResult 2.0.0` schema, reporters, the CLI and the batch engine. See
[repo-layout.md](repo-layout.md).

## Public contract

Integrations should depend only on:

1. `adaptersentry.scan(path, *, mode, policy, include_modules, include_heads, full_paths,
   run_id, hf_repo_id, hf_revision) -> ScanResult`
2. `adaptersentry.load_scan_result(data) -> ScanResult` — validates a stored document and
   rejects other major schema versions
3. `adaptersentry.ScanResult` and its JSON form, `ScanResult 2.0.0`
   ([field reference](../output-schema/scan-result.md),
   [JSON Schema](../output-schema/scan-result-2.0.0.schema.json))
4. The CLI: `adaptersentry scan` (`--format text|json|full-json|sarif`) and
   `adaptersentry batch`, with their exit codes

Decisions should read `verdict.action`. Within 2.x, fields may be added but the meaning of
existing fields does not change.

All other modules (`parsers`, `features`, `detectors`, `scoring`, `engine`, …) are internal
and may change in any release.
