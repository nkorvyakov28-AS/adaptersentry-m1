"""Score an adapter against a calibrated family reference (ESR stage 5.3).

At scan time this turns the offline profile (``profiles/<family>.npz``, built by
``benchmarks/build_profile.py`` + ``build_calibration.py``) into a
``ReferenceComparison``: a squared Mahalanobis distance of the adapter's residual
vector to the benign centre, and a split-conformal p-value against the held-out
reference scores,

    p = (1 + #{D2_ref >= D2}) / (n_ref + 1),

which bounds the benign flag rate at alpha by construction. The profile ships
precomputed (centre, precision, reference scores); scoring needs only numpy, no
scikit-learn and no refit.

Matching is by the family substring the profile was built on (the longest
substring that occurs in the declared base model wins, so ``qwen3`` is preferred
over ``qwen`` would be). A missing profile, missing rank, or anything else
unresolved returns ``None`` — the verdict then falls back to intra-adapter only,
exactly as before any profile existed (fail-open on *absence*, never on doubt).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from adaptersentry.schemas.result import ReferenceComparison
from adaptersentry.scoring.profile_vector import (
    adapter_vector,
    mahalanobis_sq,
    transformed_values,
)

logger = logging.getLogger(__name__)

_PROFILE_DIR = Path(__file__).resolve().parent.parent / "profiles"

# Format tag of the shipped profiles; bump when the vector layout or the
# calibration procedure changes so a stale profile cannot be scored silently.
PROFILE_FORMAT = "esr-c-1"

# The reference comparison may escalate the verdict to "review" but not to
# "block" yet. The benign false-positive rate is calibrated (~alpha by
# construction), but the detection rate against *poisoned* adapters has not been
# measured — that needs a poisoned corpus (GPU). Until it is, blocking a real
# user's adapter on the profile alone is not justified. Flip to True only once
# the poisoned-corpus TPR is in. The scanner passes this to derive_verdict.
REFERENCE_CAN_BLOCK = False


@lru_cache(maxsize=1)
def _available() -> tuple[tuple[str, Path], ...]:
    """Discovered (family_key, path) profiles, longest family key first."""
    if not _PROFILE_DIR.is_dir():
        return ()
    found: list[tuple[str, Path]] = []
    for path in sorted(_PROFILE_DIR.glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as z:
                if "cal_mu" not in z:  # uncalibrated profile: cannot score
                    continue
                family = str(z["family"])
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Skipping unreadable reference profile %s: %s", path.name, exc)
            continue
        found.append((family, path))
    found.sort(key=lambda fp: len(fp[0]), reverse=True)
    return tuple(found)


@lru_cache(maxsize=16)
def _load(path_str: str) -> dict[str, Any]:
    """Load one calibrated profile npz into memory (cached)."""
    with np.load(Path(path_str), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _select(base_model_declared: str | None) -> dict[str, Any] | None:
    """The calibrated profile whose family key occurs in the declared base model."""
    if not base_model_declared:
        return None
    low = base_model_declared.lower()
    for family, path in _available():
        if family.lower() in low:
            return _load(str(path))
    return None


def compare_to_reference(
    *,
    base_model_declared: str | None,
    rank: int | None,
    n_model_layers: int | None,
    entries: list,
) -> ReferenceComparison | None:
    """Conformal comparison of an adapter to its family reference, or ``None``.

    ``entries`` are the scanner's per-module records (each with ``module``,
    ``layer`` and a ``features`` object). ``None`` means no calibrated profile
    applies (unknown family, missing rank), so the caller keeps the
    intra-adapter-only verdict.
    """
    profile = _select(base_model_declared)
    if profile is None or not rank or rank <= 0 or not entries:
        return None

    log_rank = float(np.log(float(rank)))
    denom = (n_model_layers - 1) if (n_model_layers and n_model_layers > 1) else 0
    modules = []
    for e in entries:
        if e.features is None:
            continue
        vals = transformed_values(e.features)
        if not vals:
            continue
        rel = (e.layer / denom) if (e.layer is not None and denom > 0) else 0.0
        modules.append((e.module, rel, vals))
    if not modules:
        return None

    vec = adapter_vector(modules, log_rank, profile)
    mu, prec = profile["cal_mu"], profile["cal_precision"]
    ref_scores = np.asarray(profile["cal_ref_scores"], dtype=np.float64)
    n_ref = int(ref_scores.size)
    if n_ref < 1 or mu.shape[0] != vec.shape[0]:
        return None

    d2 = float(mahalanobis_sq(vec[None, :], mu, prec)[0])
    d2 = max(d2, 0.0)  # guard tiny negative from precision round-off
    p = (1 + int(np.count_nonzero(ref_scores >= d2))) / (n_ref + 1)

    family = str(profile["family"])
    n_adapters = int(profile["n_adapters"]) if "n_adapters" in profile else n_ref
    return ReferenceComparison(
        profile_id=family,
        profile_version=f"{PROFILE_FORMAT}+n{n_adapters}",
        stratum="family-pooled; detrended by (relative depth, log-rank)",
        n_reference=n_ref,
        distance=d2,
        p_value=p,
        alpha_review=float(profile["cal_alpha_review"]),
        alpha_block=float(profile["cal_alpha_block"]),
    )
