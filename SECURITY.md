# Security Policy

## Security Model — M1 Static Analyzer

AdapterSentry M1 operates in **read-only / parse-only mode** on untrusted `.safetensors` files.
The following properties hold for the scan path (`adaptersentry scan`, `adaptersentry batch`,
`scan()`, `scan_to_result()`):

- M1 does not run inference, load a base model, or execute any model code.
  `eval()`, `exec()` and `pickle` are never used on adapter content.
- Only `.safetensors` is parsed. Pickle-based formats (`.bin`, `.pt`) are never deserialised;
  such files fail the scan.
- **Files.** The path is resolved with `pathlib.Path.resolve()` and must be a regular file
  (no FIFOs or devices, including through symlinks) with a `.safetensors` suffix after
  resolution and a size of at most 64 GiB. Batch discovery skips non-regular files.
- **Header.** The declared JSON header length is bounded (100 MB) and checked against the
  file size before it is read.
- **Tensors.** Before any tensor is allocated, its header entry is validated: dtype must be
  F32, F16 or BF16; shape, byte length and offsets must be consistent; a single tensor may
  not exceed 1 billion elements, all LoRA tensors together may not exceed 3 billion, and the
  LoRA rank may not exceed 1024. Each tensor is loaded in its own error scope.
- **Fail-closed verdict.** A tensor that fails validation or loading, a tensor that is not
  part of a `lora_A`/`lora_B` pair (and is therefore not analysed), or a layer with NaN/Inf
  weights makes the scan `DEGRADED`. A file that cannot be parsed makes it `FAILED`. Neither
  ever yields `recommended_action: "allow"`: the action is at least `review`, with
  `m2_recommended: true` and a `PARSE_FAILED` or `DEGRADED_PARSE` policy signal.
  A low score on a degraded scan is not evidence that the adapter is safe.
- **Output.** Tensor names, paths and messages are escaped before they are printed as text,
  so control or bidi characters in an adapter cannot forge terminal output. JSON and SARIF
  are produced with standard JSON escaping.
- The Rust extension (optional) orders floats with a total order and cannot panic on NaN.
- Workers validate that adapter paths are absolute before processing, enforcing the trust
  boundary between the orchestrator and the worker pool.

Operational notes:

- `--ray-address` connects to an existing Ray cluster, which has no authentication by
  default. Use it only on localhost or a trusted private network.
- JSON and SARIF reports contain the absolute path of the scanned file. Review them before
  attaching them to a public issue.

The behavioral sandbox (M2), signature engine (M3), and runtime monitor (M4) are not yet
implemented. Their security models will be documented when those components ship.

---

## Reporting a Malicious Adapter Found in the Wild

If you identify a publicly distributed LoRA adapter that you believe is malicious, please
report it so it can be investigated and disclosed responsibly.

**How to report:**

1. Open a GitHub issue in this repository with the label `malicious-adapter`, **or**
   email `security@adaptersentry.io`.
2. Include:
   - The HuggingFace repository ID (e.g., `author/model-name`)
   - The M1 scan report (it contains the local file path; redact it if needed):
     ```
     adaptersentry scan ./adapter.safetensors --format summary-json --output report.json
     ```
   - A brief description of why you consider the adapter suspicious
3. **Do not attach the `.safetensors` file itself** to public GitHub issues.
   If sharing the file is necessary for investigation, coordinate via email.

Public GitHub issues are appropriate for this category of report because the subject is the
third-party adapter, not AdapterSentry itself.

---

## Reporting a Vulnerability in AdapterSentry

Use responsible disclosure. Do not open public GitHub issues for security vulnerabilities
in AdapterSentry code, dependencies, or infrastructure.

**How to report:**

Email `security@adaptersentry.io` with:

- A description of the vulnerability and its potential impact
- Reproduction steps (minimal reproducing example preferred)
- Affected version or commit hash
- Optional: CVSS v3.1 severity estimate

Please do not exploit any vulnerability beyond what is necessary to confirm it exists.

---

## Response Timeline

| Stage | Target |
|---|---|
| Acknowledgement | Within 48 hours of receipt |
| Triage and initial assessment | Within 5 business days |
| Patch or mitigation for High / Critical | Within 30 days |
| Patch or mitigation for Medium / Low | Within 90 days |
| Public disclosure | Coordinated with reporter; default after patch is available |

These are targets, not guarantees. Complex issues may require more time.
We will communicate status updates if a deadline cannot be met.

---

## Supported Versions

| Version | Supported |
|---|---|
| v1.0.3 (current) | ✅ Yes |
| v1.0.2 and earlier | ❌ No — can report `allow` for a file it failed to parse; upgrade |

Security fixes are released for the most recent version.
