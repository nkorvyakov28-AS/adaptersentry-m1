"""M1 detector subpackage — per-layer anomaly detection modules.

Each detector is a stateless module operating on numpy arrays.
"""

from .entropy import compute_entropy, detect_entropy_anomalies
from .outlier import (
    detect_outlier_anomalies,
    isolation_forest_score,
    zscore_outlier_rate,
)
from .init_detector import (
    is_init_only_adapter,
    get_adapter_training_status,
    suppress_init_flags,
)

__all__ = [
    "compute_entropy",
    "detect_entropy_anomalies",
    "zscore_outlier_rate",
    "isolation_forest_score",
    "detect_outlier_anomalies",
    "is_init_only_adapter",
    "get_adapter_training_status",
    "suppress_init_flags",
]
