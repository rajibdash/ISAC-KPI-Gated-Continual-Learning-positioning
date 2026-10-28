"""Illustrative research-policy defaults; calibrate for each testbed."""

from dataclasses import dataclass, fields
from math import isfinite


@dataclass(frozen=True)
class Config:
    """Controller thresholds, buffer bounds, and optimizer settings."""

    window: int = 16
    persistence: int = 3
    cooldown_steps: int = 100
    rsrp_low_dbm: float = -95.0
    bler_high: float = 0.10
    sensing_snr_min_db: float = 10.0
    label_confidence_min: float = 0.95
    error_high_m: float = 2.0
    drift_high: float = 0.8
    min_new_samples: int = 32
    pending_capacity: int = 256
    replay_capacity: int = 2048
    learning_rate: float = 1e-3
    train_steps: int = 80
    batch_per_source: int = 32
    min_recent_improvement_m: float = 0.01
    max_old_regression_m: float = 0.10
    max_grad_norm: float = 5.0
    normalization_floor: float = 1e-4

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isfinite(value):
                raise ValueError(f"{field.name} must be finite and numeric")
        for name in (
            "window", "persistence", "cooldown_steps", "min_new_samples",
            "pending_capacity", "replay_capacity", "train_steps", "batch_per_source",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "cooldown_steps" else 1
            if not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.pending_capacity < self.min_new_samples:
            raise ValueError("pending_capacity must be >= min_new_samples")
        for name in ("bler_high", "label_confidence_min"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "error_high_m", "drift_high", "min_recent_improvement_m",
            "max_old_regression_m",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("learning_rate", "max_grad_norm", "normalization_floor"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
