"""AdapterSentry M1 scan pipeline — produces a ScanResult 2.0.0.

    scan(path) → ScanResult

Pipeline:
    1. identity      content/header hashes, provenance (file name by default)
    2. config        adapter_config.json next to the file (bounded, untrusted)
    3. inventory     classify every tensor from the header, nothing loaded
    4. per module    stream one LoRA pair at a time: non-finite check, effective
                     scale, exact ESR features (optional per-head)
    5. intra         compare each module with its family (robust z over depth)
    6. structural    full-weight replacement, router LoRA, rank mismatch,
                     executable siblings, DoRA, …
    7. verdict       one rule set (scoring.verdict_v2), fail-closed

``scan`` never raises: every failure becomes a structured result whose action
is at least "review".

Security Notes:
    - All file access goes through parsers.adapter_file / adapter_config,
      which validate before allocating; no pickle, eval or exec.
    - Peak memory is bounded by one LoRA pair (~20 MB at 70B scale).
    - Reports carry the file name only unless ``full_paths=True``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from adaptersentry.detectors.intra_adapter import OUTLIER_Z, REVIEW_Z, ModuleEntry, compare_within_adapter
from adaptersentry.detectors.structural import find_siblings, structural_findings
from adaptersentry.features.spectral_v2 import head_features, module_features
from adaptersentry.parsers.adapter_config import read_adapter_config, resolve_scale
from adaptersentry.parsers.adapter_file import AdapterInventory, iter_pairs, open_adapter
from adaptersentry.schemas.errors import ScanError, ScanPhase
from adaptersentry.schemas.finding import Severity
from adaptersentry.schemas.result import (
    MAX_LISTED_NOT_ANALYZED,
    SCHEMA_VERSION,
    AdapterInfo,
    Anomaly,
    Artifact,
    Coverage,
    Finding,
    Location,
    ModuleRecord,
    MoEInfo,
    NotAnalyzed,
    Policy,
    Provenance,
    RankInfo,
    ScanInfo,
    ScanResult,
    TokenConcentration,
)
from adaptersentry.scoring.verdict_v2 import derive_verdict
from adaptersentry.version import __version__

logger = logging.getLogger(__name__)

Mode = Literal["fast", "full"]

_FAMILY_HINTS = ("llama", "qwen", "gemma", "mistral", "mixtral", "phi", "deepseek", "falcon",
                 "gpt-neox", "gpt2", "bloom", "olmo", "internlm", "yi", "baichuan", "chatglm", "t5")
_TOP_TOKENS = 10


def config_hash(mode: Mode, policy: Policy) -> str:
    """Hash of every setting that changes the result (cache key component)."""
    settings = {
        "schema": SCHEMA_VERSION, "mode": mode, "policy": policy,
        "outlier_z": OUTLIER_Z, "review_z": REVIEW_Z,
    }
    blob = json.dumps(settings, sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_family(declared: str | None) -> str:
    if not declared:
        return "unknown"
    low = declared.lower()
    for hint in _FAMILY_HINTS:
        if hint in low:
            return "llama" if hint == "llama" else ("mistral" if hint == "mixtral" else hint)
    return "unknown"


def _artifact(path: Path, *, full_paths: bool, source_kind: str, hf_repo_id: str | None,
              hf_revision: str | None, config_file: str | None, siblings: list) -> Artifact | None:
    from adaptersentry.engine.identity import ArtifactIdentityResolver
    from adaptersentry.engine.schemas.requests import ArtifactSource

    source = ArtifactSource(kind="local_path", local_path=str(path.resolve())) if source_kind == "local_path" \
        else ArtifactSource(kind="hf_hub", hf_repo_id=hf_repo_id, hf_revision=hf_revision)
    try:
        ident = ArtifactIdentityResolver.resolve(path.resolve(), source)
    except (OSError, ValueError) as exc:
        logger.warning("Identity of %r could not be resolved: %s", path.name, exc)
        return None
    return Artifact(
        content_hash=ident.content_hash,
        header_hash=ident.header_hash,
        logical_id=ident.logical_id,
        file_size_bytes=ident.file_size_bytes,
        provenance=Provenance(
            kind=source_kind,  # type: ignore[arg-type]
            path=str(path.resolve()) if full_paths else path.name,
            hf_repo_id=hf_repo_id, hf_revision=hf_revision,
            config_file=config_file, sibling_files=siblings,
        ),
    )


def _failed_result(
    path: Path, exc: Exception, *, scan_info: ScanInfo, artifact: Artifact | None,
    config_present: bool,
) -> ScanResult:
    verdict = derive_verdict(
        status="failed", policy=scan_info.policy, intra=None, reference=None, red_flags=[],
        metadata_present=False, confidence_limits=[], n_not_analyzed=0,
    )
    return ScanResult(
        scan=scan_info, artifact=artifact,
        adapter=AdapterInfo(format="unknown", config_present=config_present),
        status="failed",
        coverage=Coverage(n_tensors_total=0, n_tensors_analyzed=0, n_modules=0,
                          n_modules_analyzed=0, n_not_analyzed=0),
        verdict=verdict,
        errors=[ScanError.malformed(code="INVALID_SAFETENSORS", message=str(exc),
                                    detail=type(exc).__name__, phase=ScanPhase.PARSE)],
    )


def scan(
    path: Path,
    *,
    mode: Mode = "full",
    policy: Policy = "default",
    include_modules: bool = False,
    include_heads: bool = False,
    full_paths: bool = False,
    run_id: str | None = None,
    hf_repo_id: str | None = None,
    hf_revision: str | None = None,
) -> ScanResult:
    """Scan one LoRA adapter file and return a ScanResult 2.0.0. Never raises.

    Args:
        path: .safetensors adapter file.
        mode: "full" or "fast" (fewer kurtosis rows).
        policy: "default" (no block without a reference profile) or "strict"
            (structural red flags block).
        include_modules: include per-module features in the result.
        include_heads: include per-head features (implies include_modules).
        full_paths: report the absolute path instead of the file name.
        run_id: batch identifier to record.
        hf_repo_id, hf_revision: provenance for files downloaded from the Hub.
    """
    started_at = _now()
    t0 = time.perf_counter()
    path = Path(path)
    include_modules = include_modules or include_heads
    cfg_hash = config_hash(mode, policy)

    cfg_read = read_adapter_config(path)
    config = cfg_read.config
    siblings = find_siblings(path)
    source_kind = "hf_hub" if hf_repo_id else "local_path"
    artifact = _artifact(path, full_paths=full_paths, source_kind=source_kind, hf_repo_id=hf_repo_id,
                         hf_revision=hf_revision, config_file=cfg_read.path_name, siblings=siblings)
    scan_id = "sha256:" + hashlib.sha256(
        f"{artifact.content_hash if artifact else path.name}:{cfg_hash}:{SCHEMA_VERSION}".encode()
    ).hexdigest()

    def scan_info() -> ScanInfo:
        return ScanInfo(
            scan_id=scan_id, run_id=run_id, analyzer_version=__version__, config_hash=cfg_hash,
            mode=mode, policy=policy, started_at=started_at, completed_at=_now(),
            wall_time_ms=int((time.perf_counter() - t0) * 1000),
        )

    try:
        inv = open_adapter(path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        return _failed_result(path, exc, scan_info=scan_info(), artifact=artifact,
                              config_present=config is not None)
    if not inv.pairs:
        return _failed_result(
            path, ValueError("no lora_A/lora_B pair could be analysed — not a PEFT LoRA adapter"),
            scan_info=scan_info(), artifact=artifact, config_present=config is not None,
        )

    try:
        return _scan_inventory(
            path, inv, cfg_read, siblings, artifact, scan_info,
            mode=mode, policy=policy, include_modules=include_modules, include_heads=include_heads,
        )
    except Exception as exc:  # noqa: BLE001 — fail closed on any unexpected analysis error
        logger.error("Analysis of %r failed: %s", path.name, exc)
        return _failed_result(path, exc, scan_info=scan_info(), artifact=artifact,
                              config_present=config is not None)


def _scan_inventory(path, inv: AdapterInventory, cfg_read, siblings, artifact, scan_info, *,
                    mode: Mode, policy: Policy, include_modules: bool, include_heads: bool) -> ScanResult:
    config = cfg_read.config
    not_analyzed: list[NotAnalyzed] = [
        NotAnalyzed(tensor_key=s.key, reason=s.reason, category=s.category.value) for s in inv.not_analyzed
    ]
    for w in inv.full_weights:
        not_analyzed.append(NotAnalyzed(
            tensor_key=w.entry.key, reason="full weight matrix; needs the base model to analyse",
            category="unsupported",
        ))

    records: list[ModuleRecord] = []
    entries: list[ModuleEntry] = []
    non_finite: list[str] = []
    token_conc: list[TokenConcentration] = []
    unresolved_pattern = False
    zero_modules = 0

    for pd in iter_pairs(inv):
        pair = pd.pair
        loc = pair.location
        base_record = dict(layer=loc.layer, module=loc.module, kind=loc.kind, expert=loc.expert,
                           shape_a=list(pair.a.shape), shape_b=list(pair.b.shape), rank=pair.rank)
        if pd.error is not None:
            for key in (pair.a.key, pair.b.key):
                not_analyzed.append(NotAnalyzed(tensor_key=key, reason=pd.error, category="malformed"))
            records.append(ModuleRecord(**base_record, status="not_analyzed"))
            continue
        if not (np.isfinite(pd.a).all() and np.isfinite(pd.b).all()):
            non_finite.append(pair.base)
            records.append(ModuleRecord(**base_record, status="non_finite"))
            continue

        res = resolve_scale(pair.base, pair.rank, config)
        unresolved_pattern |= res.unresolved_pattern
        feats, core = module_features(pd.a, pd.b, res.scale, transposed=pair.transposed, fast=mode == "fast")
        heads = None
        if include_heads and loc.kind == "attention" and not pair.transposed:
            heads = head_features(pd.a, pd.b, res.scale, loc.module)
        if core.is_zero:
            zero_modules += 1

        if loc.kind in ("lm_head", "embedding") and not core.is_zero:
            rows = core.col_norms if pair.transposed else core.row_norms
            energy = rows.astype(np.float64) ** 2
            top = np.argsort(energy)[::-1][:_TOP_TOKENS]
            token_conc.append(TokenConcentration(
                module=loc.module,
                top_token_ids=[int(i) for i in top],
                top_shares=[float(energy[i] / energy.sum()) for i in top],
                hoyer=feats.row_hoyer or 0.0,
                gini=feats.row_gini or 0.0,
            ))

        idx = len(records)
        records.append(ModuleRecord(**base_record, scale=res.scale, status="ok", features=feats, heads=heads))
        entries.append(ModuleEntry(index=idx, module=loc.module, kind=loc.kind, layer=loc.layer,
                                   expert=loc.expert, features=feats, energy=core.fro ** 2))
        del pd

    intra = compare_within_adapter(entries)
    if intra is not None:
        records = [
            r.model_copy(update={"z": intra.z_by_module.get(i, {})}) if i in intra.z_by_module else r
            for i, r in enumerate(records)
        ]

    struct = structural_findings(inv, config, cfg_read.problems, siblings,
                                 unresolved_scale_pattern=unresolved_pattern)
    findings = list(struct.findings)
    if intra is not None:
        pair_by_index = {i: inv.pairs[i] for i in range(len(inv.pairs))}
        ranked = sorted(intra.z_by_module.items(), key=lambda kv: -max(abs(v) for v in kv[1].values()))
        for idx, zs in ranked[:10]:
            feat, zval = max(zs.items(), key=lambda kv: abs(kv[1]))
            if abs(zval) <= OUTLIER_Z:
                break
            rec, pair = records[idx], pair_by_index[idx]
            strong = abs(zval) >= REVIEW_Z
            findings.append(Finding(
                rule_id="INTRA_DEPTH_SPIKE" if strong else "INTRA_OUTLIER",
                severity=Severity.MEDIUM if strong else Severity.LOW,
                title="Module departs from its family" if strong else "Mild outlier within its family",
                message=(f"{rec.module} in layer {rec.layer}: {feat} robust z = {zval:.1f} "
                         f"({'≥' if strong else '<'} review threshold {REVIEW_Z}, uncalibrated)"),
                evidence={"feature": feat, "z": round(zval, 3), "value": getattr(rec.features, feat)},
                locations=[Location(layer=rec.layer, module=rec.module, expert=rec.expert,
                                    tensor_key=pair.b.key)],
                remediation="Behavioural check of this adapter" if strong else None,
            ))
    for tc in token_conc:
        findings.append(Finding(
            rule_id="TOKEN_CONCENTRATION", severity=Severity.LOW,
            title=f"Update of {tc.module} by token",
            message=f"Top-{_TOP_TOKENS} tokens hold {sum(tc.top_shares):.1%} of the update energy "
                    f"(Hoyer {tc.hoyer:.2f}).",
            evidence={"top_token_ids": tc.top_token_ids, "top_shares": [round(s, 4) for s in tc.top_shares]},
        ))

    n_not = len(not_analyzed) + 2 * len(non_finite)
    listed = not_analyzed + [
        NotAnalyzed(tensor_key=f"{base}.lora_A/lora_B", reason="NaN or Inf weights", category="malformed")
        for base in non_finite
    ]
    n_ok = len(entries)
    coverage = Coverage(
        n_tensors_total=inv.n_tensors_total,
        n_tensors_analyzed=2 * n_ok,
        n_modules=len(inv.pairs),
        n_modules_analyzed=n_ok,
        n_not_analyzed=n_not,
        not_analyzed=listed[:MAX_LISTED_NOT_ANALYZED],
        non_finite_modules=non_finite[:MAX_LISTED_NOT_ANALYZED],
    )
    status = "ok" if n_not == 0 else ("failed" if n_ok == 0 else "degraded")

    meta = inv.metadata
    declared_base = (config.base_model_name_or_path if config else None) or meta.get("base_model_name_or_path")
    layers = [p.location.layer for p in inv.pairs if p.location.layer is not None]
    experts = [p.location.expert for p in inv.pairs if p.location.expert is not None]
    ranks = sorted({p.rank for p in inv.pairs})
    rank_declared = config.r if config and config.r else None
    if rank_declared is None:
        for key in ("r", "rank", "lora_r"):
            if key in meta and meta[key].isdigit():
                rank_declared = int(meta[key])
                break
    if config is None:
        scaling = "unknown"
    elif config.rank_pattern or config.alpha_pattern:
        scaling = "pattern"
    elif config.use_rslora:
        scaling = "rslora"
    else:
        scaling = "alpha_over_r"
    if n_ok and zero_modules == n_ok:
        training_state = "init_only"
    elif zero_modules:
        training_state = "partial"
    elif n_ok:
        training_state = "trained"
    else:
        training_state = "unknown"
    adapter = AdapterInfo(
        format="peft_lora",
        peft_type=(config.peft_type if config else None) or meta.get("peft_type"),
        base_model_declared=declared_base,
        base_family=_base_family(declared_base),
        base_family_source="config" if declared_base else "unknown",
        n_model_layers=(max(layers) + 1) if layers else None,
        rank_declared=rank_declared,
        rank_actual=RankInfo(min=ranks[0], max=ranks[-1], distinct=ranks),
        lora_alpha=config.lora_alpha if config else None,
        scaling=scaling,  # type: ignore[arg-type]
        init_method=config.init_lora_weights if config else None,
        use_dora=bool(inv.dora_magnitudes) or bool(config and config.use_dora),
        target_modules_declared=list(config.target_modules) if config else [],
        target_modules_actual=sorted({p.location.module for p in inv.pairs}),
        modules_to_save=sorted({w.location.module for w in inv.full_weights}
                               | set(config.modules_to_save if config else [])),
        trainable_token_indices=bool(inv.trainable_tokens) or bool(config and config.trainable_token_indices),
        moe=MoEInfo(n_experts=max(experts) + 1,
                    router_adapted=any(p.location.kind == "router" for p in inv.pairs)) if experts else None,
        training_state=training_state,  # type: ignore[arg-type]
        metadata_present=bool(meta),
        config_present=config is not None,
    )

    verdict = derive_verdict(
        status=status, policy=policy, intra=intra.summary if intra else None, reference=None,
        red_flags=struct.red_flags, metadata_present=bool(meta) or config is not None,
        confidence_limits=struct.confidence_limits, n_not_analyzed=n_not,
    )
    anomaly = Anomaly(
        intra=intra.summary if intra else None,
        reference=None,
        token_concentration=token_conc,
        top_contributors=intra.top_contributors if intra else [],
    )
    if not include_modules:
        out_modules = None
    elif include_heads:
        out_modules = records
    else:
        out_modules = [r.model_copy(update={"heads": None}) for r in records]

    return ScanResult(
        scan=scan_info(), artifact=artifact, adapter=adapter, status=status, coverage=coverage,
        verdict=verdict, anomaly=anomaly, findings=findings, modules=out_modules, errors=[],
    )
