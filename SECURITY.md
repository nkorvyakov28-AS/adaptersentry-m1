# Security Policy

## Security Model — M1 Static Analyzer

AdapterSentry M1 operates in **read-only / parse-only mode** on untrusted `.safetensors` files.
The following properties hold for all M1 code paths:

- M1 does not run inference, load a base model, or execute any model code.
- No tensor operation depends on untrusted file content as executable code.
- All file paths provided by the caller are resolved through `pathlib.Path.resolve()`
  before use, preventing path traversal.
- Tensors exceeding 1 billion elements are rejected before allocation (tensor bomb guard).
  The check runs on the header-declared shape before any memory is allocated.
- Adapter metadata is read as plain strings. Metadata nesting depth is capped at 5 levels;
  payloads exceeding the limit are flagged as a security signal and not processed further.
- `eval()`, `exec()`, and `pickle.load()` are never called on untrusted content.
- The `safetensors` library provides safe, header-validated tensor parsing;
  raw pickle and PyTorch checkpoint formats (`.pt`, `.bin`) are explicitly rejected.
- Workers validate that adapter paths are absolute before processing, enforcing the
  trust boundary between the orchestrator and worker pool.

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
   - The M1 scan report:
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
| v1.0.1 (current) | ✅ Yes |
| v1.0.0 | ✅ Yes |
| v0.x.x | ❌ No |

Only the two most recent releases receive security patches.
