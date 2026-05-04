"""Cross-layer consistency detector for LoRA adapters.

Legitimate adapters distribute their learned changes across many layers.
Backdoored adapters concentrate anomalies in a small number of targeted layers
(e.g., embed_tokens, lm_head). This detector measures that concentration.

Analogy: like detecting lateral movement by finding that 90% of anomalous
network events originate from a single host segment.

Security Notes:
    - Pure computation on already-loaded report dicts — no I/O, no deserialization.
    - No eval/exec/pickle.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_CONSISTENCY_THRESHOLD = 0.3       # below this → anomalous concentration
_CLUSTER_FLAG_THRESHOLD = 0.30     # >30% of flags in <10% of layers


def compute_cross_layer_consistency(layer_reports: dict) -> float:
    """Compute consistency score measuring how evenly anomaly flags are distributed.

    C = 1 - std(per_layer_flag_counts) / (mean(per_layer_flag_counts) + 1e-8)

    High C (near 1.0): flags spread evenly — each layer similarly anomalous.
    Low C (near 0.0): flags concentrated in a few layers — backdoor signature.

    Args:
        layer_reports: Dict mapping layer name to its per-layer report dict,
                       as produced by analyzer.analyze(). Each report must have
                       a "flags" key containing a list of flag strings.

    Returns:
        Float in [0.0, 1.0]. Returns 1.0 if there are 0 or 1 layers.
    """
    if len(layer_reports) <= 1:
        return 1.0

    counts = np.array(
        [len(v.get("flags", [])) for v in layer_reports.values()],
        dtype=np.float64,
    )
    mean = float(counts.mean())
    std = float(counts.std())
    consistency = 1.0 - std / (mean + 1e-8)
    return float(np.clip(consistency, 0.0, 1.0))


def detect_cross_layer_anomalies(layer_reports: dict) -> tuple[float, list[str]]:
    """Detect backdoor-style flag concentration across adapter layers.

    Checks two conditions:

    1. CROSS_LAYER_CONCENTRATION: overall consistency score C < 0.3.
       Low C means a small number of layers hold most of the anomaly signal —
       consistent with targeted backdoor injection.

    2. SUSPICIOUS_LAYER_CLUSTER: more than 30% of all flags are concentrated
       in fewer than 10% of layers.  Absolute cluster signal, independent of C.

    Args:
        layer_reports: Dict mapping layer name to per-layer report dict.

    Returns:
        Tuple of (consistency_score: float, flags: list[str]).
    """
    consistency = compute_cross_layer_consistency(layer_reports)
    flags: list[str] = []
    n_layers = len(layer_reports)

    if n_layers < 2:
        return consistency, flags

    if consistency < _CONSISTENCY_THRESHOLD:
        flags.append(
            f"CROSS_LAYER_CONCENTRATION: consistency={consistency:.4f}"
            f" < {_CONSISTENCY_THRESHOLD}"
            " (anomaly flags concentrated in specific layers — backdoor pattern)"
        )

    layer_flag_counts = {
        name: len(report.get("flags", []))
        for name, report in layer_reports.items()
    }
    total_flags = sum(layer_flag_counts.values())

    if total_flags > 0:
        top_n = max(1, int(np.ceil(n_layers * 0.10)))
        sorted_counts = sorted(layer_flag_counts.values(), reverse=True)
        top_flags = sum(sorted_counts[:top_n])
        cluster_fraction = top_flags / total_flags

        if cluster_fraction > _CLUSTER_FLAG_THRESHOLD:
            flagged_layers = [
                name for name, cnt in layer_flag_counts.items()
                if cnt == sorted_counts[0] and sorted_counts[0] > 0
            ]
            flags.append(
                f"SUSPICIOUS_LAYER_CLUSTER: {cluster_fraction:.1%} of flags"
                f" in top {top_n}/{n_layers} layer(s)"
                f" ({', '.join(flagged_layers[:3])})"
                " (possible targeted layer injection)"
            )

    return consistency, flags
