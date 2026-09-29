"""Structural findings: what the adapter contains, independent of its numbers.

Some properties are suspicious by construction, even though each also has
legitimate uses. They are reported as findings; the ones marked ``red_flag``
raise the verdict to at least "review" (and to "block" under the strict
policy):

    FULL_WEIGHT_REPLACEMENT   a complete lm_head / embedding / router matrix is
                              shipped (modules_to_save): direct control over
                              which tokens the model emits. Legitimate when
                              new tokens are added. Cannot be analysed without
                              the base model.
    ROUTER_ADAPTED            LoRA on a MoE router: can steer a trigger to a
                              chosen expert.
    RANK_MISMATCH             ranks differ from the declared r without a
                              rank_pattern: typical of merged adapters
                              (e.g. LoRATK "cat" merges, rank r₁+r₂).
    SIBLING_EXECUTABLE        code, pickle or archive files next to the adapter
                              (trust_remote_code, pickle payloads).

Informational findings (no verdict change): DoRA, unresolved scale patterns,
config problems, target-module mismatch, trainable-token deltas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adaptersentry.parsers.adapter_config import AdapterConfig
from adaptersentry.parsers.adapter_file import AdapterInventory
from adaptersentry.schemas.severity import Severity
from adaptersentry.schemas.result import Finding, Location, SiblingFile

_CODE_EXT = frozenset({".py", ".pyc", ".sh", ".js", ".so", ".dll", ".dylib", ".exe"})
_PICKLE_EXT = frozenset({".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".joblib", ".npy", ".npz"})
_ARCHIVE_EXT = frozenset({".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".xz", ".bz2"})
_MAX_SIBLINGS_SCANNED = 1000


@dataclass(frozen=True)
class StructuralResult:
    """Findings plus which of them are red flags."""

    findings: list[Finding]
    red_flags: list[str]           # rule ids that raise the verdict
    confidence_limits: list[str]   # human-readable limits on analysis completeness


def find_siblings(adapter_path: Path) -> list[SiblingFile]:
    """Code, pickle and archive files in the adapter's directory (non-recursive, bounded)."""
    out: list[SiblingFile] = []
    try:
        directory = adapter_path.resolve().parent
        for i, entry in enumerate(directory.iterdir()):
            if i >= _MAX_SIBLINGS_SCANNED:
                break
            if entry.name == adapter_path.name or not entry.is_file():
                continue
            suffix = entry.suffix.lower()
            if suffix in _CODE_EXT:
                out.append(SiblingFile(name=entry.name, kind="code"))
            elif suffix in _PICKLE_EXT:
                out.append(SiblingFile(name=entry.name, kind="pickle"))
            elif suffix in _ARCHIVE_EXT:
                out.append(SiblingFile(name=entry.name, kind="archive"))
    except OSError:
        return out
    return sorted(out, key=lambda s: s.name)


def structural_findings(
    inv: AdapterInventory,
    config: AdapterConfig | None,
    config_problems: list[str],
    siblings: list[SiblingFile],
    *,
    unresolved_scale_pattern: bool,
) -> StructuralResult:
    """Derive structural findings from the inventory, config and provenance."""
    findings: list[Finding] = []
    red: list[str] = []
    limits: list[str] = []

    # Full weight matrices shipped in the adapter.
    replaced = [w for w in inv.full_weights if w.location.kind in ("lm_head", "embedding", "router")]
    if replaced:
        red.append("FULL_WEIGHT_REPLACEMENT")
        findings.append(Finding(
            rule_id="FULL_WEIGHT_REPLACEMENT", severity=Severity.MEDIUM,
            title="Adapter replaces a complete output, embedding or router matrix",
            message=(
                "The adapter ships full weight matrices (modules_to_save) for "
                + ", ".join(sorted({w.location.module for w in replaced}))
                + ". They directly control which tokens the model reads or emits and cannot be "
                "analysed without the base model. Legitimate when new tokens are added."
            ),
            evidence={"tensors": [w.entry.key for w in replaced][:20],
                      "shapes": [list(w.entry.shape) for w in replaced][:20]},
            locations=[Location(layer=w.location.layer, module=w.location.module, tensor_key=w.entry.key)
                       for w in replaced][:20],
            remediation="Verify why the full matrix is shipped; run a behavioural check.",
        ))
    other_full = [w for w in inv.full_weights if w not in replaced]
    if other_full:
        findings.append(Finding(
            rule_id="FULL_WEIGHT_NOT_ANALYSED", severity=Severity.LOW,
            title="Full weight matrices shipped and not analysed",
            message=f"{len(other_full)} complete weight matrices (e.g. a classification head) are not analysed.",
            evidence={"tensors": [w.entry.key for w in other_full][:20]},
        ))

    # LoRA on a MoE router.
    routers = [p for p in inv.pairs if p.location.kind == "router"]
    if routers:
        red.append("ROUTER_ADAPTED")
        findings.append(Finding(
            rule_id="ROUTER_ADAPTED", severity=Severity.MEDIUM,
            title="LoRA applied to the mixture-of-experts router",
            message="Changing the router can send inputs containing a trigger to a chosen expert.",
            locations=[Location(layer=p.location.layer, module=p.location.module, tensor_key=p.a.key)
                       for p in routers][:20],
            remediation="Confirm router adaptation is intended; run a behavioural check.",
        ))

    # Ranks versus the declared r.
    ranks = sorted({p.rank for p in inv.pairs})
    declared = config.r if config is not None else None
    patterned = bool(config and config.rank_pattern)
    if declared is not None and not patterned and ranks and ranks != [declared]:
        red.append("RANK_MISMATCH")
        findings.append(Finding(
            rule_id="RANK_MISMATCH", severity=Severity.MEDIUM,
            title="Tensor ranks differ from the declared rank",
            message=(
                f"adapter_config.json declares r={declared} without a rank_pattern, but the tensors "
                f"have ranks {ranks}. Merged adapters (e.g. concatenation, rank r₁+r₂) look like this."
            ),
            evidence={"declared_r": declared, "actual_ranks": ranks},
        ))

    # Executable or pickle files next to the adapter.
    risky = [s for s in siblings if s.kind in ("code", "pickle", "archive")]
    if risky:
        red.append("SIBLING_EXECUTABLE")
        findings.append(Finding(
            rule_id="SIBLING_EXECUTABLE", severity=Severity.MEDIUM,
            title="Code, pickle or archive files next to the adapter",
            message=(
                "Files that can execute code when loaded (custom code, pickle checkpoints, archives) "
                "are distributed with the adapter: " + ", ".join(s.name for s in risky[:10])
            ),
            evidence={"files": [s.model_dump() for s in risky[:50]]},
            remediation="Do not load them; never enable trust_remote_code for this repository.",
        ))

    # Informational.
    if inv.dora_magnitudes or (config is not None and config.use_dora):
        limits.append("DoRA adapter: its effective update depends on base weights; analysis is partial")
        findings.append(Finding(
            rule_id="DORA_PARTIAL_ANALYSIS", severity=Severity.LOW,
            title="DoRA adapter analysed partially",
            message="DoRA rescales the base weight; without the base model only the low-rank part is analysed.",
        ))
    if unresolved_scale_pattern:
        limits.append("rank/alpha pattern with regular expressions not resolved; default scale used")
        findings.append(Finding(
            rule_id="UNRESOLVED_SCALE_PATTERN", severity=Severity.LOW,
            title="Scale pattern not resolved",
            message="adapter_config.json uses regular-expression rank/alpha patterns, which are not executed "
                    "for safety; the default scale was used for matching modules.",
        ))
    if config is None:
        limits.append("no adapter_config.json: scale assumed alpha = r")
    if config_problems:
        findings.append(Finding(
            rule_id="CONFIG_FIELDS_IGNORED", severity=Severity.LOW,
            title="Invalid fields in adapter_config.json were ignored",
            message="; ".join(config_problems[:10]),
        ))
    if config is not None and config.target_modules:
        actual = {p.location.module for p in inv.pairs}
        declared_set = set(config.target_modules)
        unexpected = sorted(m for m in actual if m not in declared_set)
        if unexpected and "all-linear" not in declared_set:
            findings.append(Finding(
                rule_id="TARGET_MODULES_MISMATCH", severity=Severity.LOW,
                title="Adapted modules differ from target_modules",
                message=f"Tensors adapt {unexpected}, which are not in the declared target_modules.",
                evidence={"declared": sorted(declared_set), "unexpected": unexpected},
            ))
    if inv.trainable_tokens:
        findings.append(Finding(
            rule_id="TRAINABLE_TOKENS", severity=Severity.LOW,
            title="Adapter changes individual token embeddings",
            message="trainable_token_indices deltas are present and not analysed yet.",
            evidence={"tensors": inv.trainable_tokens[:20]},
        ))

    return StructuralResult(findings=findings, red_flags=red, confidence_limits=limits)
