"""Shannon entropy analysis for LoRA weight tensors.

Entropy is computed over the empirical weight-value distribution via histogram
binning, then normalized to [0, 1]. Think of it like measuring the information
density of a firmware patch:

- Near-zero entropy  → sparse/constant weights (possible untrained matrix or
                        backdoor trigger hiding in a near-zero distribution)
- Near-unit entropy  → weights spread like uniform noise (possible noise
                        injection or deliberate evasion via random padding)

Normal trained LoRA adapters follow a roughly Gaussian distribution, which
sits in the moderate entropy band (~0.50–0.95 for 256 histogram bins).

Security Notes:
    - No file I/O; pure numpy computation — no attack surface.
    - Edge cases (empty tensors, constant tensors) are handled explicitly.
"""

from __future__ import annotations

import numpy as np

# Anomaly thresholds tuned for 256-bin histograms on LoRA-scale weight matrices
_LOW_ENTROPY_THRESHOLD: float = 0.1   # near-constant → suspicious sparsity
_HIGH_ENTROPY_THRESHOLD: float = 0.99  # near-uniform → suspicious noise injection


def compute_entropy(tensor: np.ndarray, n_bins: int = 256) -> float:
    """Compute normalized Shannon entropy of a weight tensor's value distribution.

    Builds an empirical histogram over weight values and computes:
        H_norm = H / log2(n_bins)
    where H is the Shannon entropy over non-empty bins.

    Args:
        tensor: Weight matrix of any shape; flattened internally.
        n_bins: Number of histogram bins (default 256 covers fp32 precision).

    Returns:
        Normalized entropy in [0.0, 1.0]. Returns 0.0 for constant tensors.

    Raises:
        ValueError: If tensor is empty (zero elements).
    """
    flat = tensor.astype(np.float64).flatten()
    if flat.size == 0:
        raise ValueError("Cannot compute entropy of an empty tensor.")

    if flat.size == 1 or float(flat.max() - flat.min()) == 0.0:
        return 0.0

    counts, _ = np.histogram(flat, bins=n_bins)
    nonzero = counts[counts > 0]
    probabilities = nonzero / nonzero.sum()
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    max_entropy = np.log2(n_bins)
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def detect_entropy_anomalies(
    entropy: float,
    layer_name: str,
    matrix_label: str,
    low_threshold: float = _LOW_ENTROPY_THRESHOLD,
    high_threshold: float = _HIGH_ENTROPY_THRESHOLD,
) -> list[str]:
    """Flag entropy values outside the expected range for trained LoRA weights.

    Args:
        entropy: Normalized entropy in [0, 1] from compute_entropy.
        layer_name: Layer identifier — included in flag context.
        matrix_label: "A" or "B" — which LoRA matrix was measured.
        low_threshold: Values below this trigger LOW_ENTROPY flag (default 0.1).
        high_threshold: Values above this trigger HIGH_ENTROPY flag (default 0.99).

    Returns:
        List of anomaly flag strings (empty list if entropy is in normal range).
    """
    flags: list[str] = []
    if entropy < low_threshold:
        flags.append(
            f"LOW_ENTROPY_{matrix_label}: entropy={entropy:.4f} < {low_threshold}"
            f" in {layer_name} (sparse/constant weights — possible untrained matrix"
            " or backdoor trigger)"
        )
    elif entropy > high_threshold:
        flags.append(
            f"HIGH_ENTROPY_{matrix_label}: entropy={entropy:.4f} > {high_threshold}"
            f" in {layer_name} (near-uniform distribution — possible noise injection"
            " or evasion padding)"
        )
    return flags
