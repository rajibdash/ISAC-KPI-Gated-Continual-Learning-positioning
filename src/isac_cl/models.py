"""Position model, observation records, and array validation helpers."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import nn

Array = NDArray[np.float32]
ValidationSet = tuple[ArrayLike, ArrayLike]


def finite_array(value: ArrayLike, name: str) -> Array:
    """Convert real numeric data to float32 without silently dropping imaginary parts."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        result = raw.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite float32 values")
    return result


def finite_vector(value: ArrayLike, size: int) -> Array:
    """Validate a finite, fixed-length real vector."""
    result = finite_array(value, "vector")
    if result.shape != (size,):
        raise ValueError(f"Expected a finite vector of shape ({size},)")
    return result


def labeled_arrays(x: ArrayLike, y: ArrayLike, dim: int) -> tuple[Array, Array]:
    """Validate nonempty aligned feature and three-coordinate label matrices."""
    features, labels = finite_array(x, "features"), finite_array(y, "labels")
    if features.ndim != 2 or features.shape[1] != dim or len(features) == 0:
        raise ValueError(f"features must have nonempty shape [N, {dim}]")
    if labels.shape != (len(features), 3):
        raise ValueError("labels must have shape [N, 3] matching features")
    return features, labels


@dataclass(frozen=True)
class Observation:
    """Extracted real features and measured KPIs; coordinates are in meters.

    Label confidence is supplied by an independent label source, not the network.
    """

    features: ArrayLike
    rsrp_dbm: float
    sensing_snr_db: float
    label_confidence: float = 0.0
    xyz_m: ArrayLike | None = None
    bler: float | None = None


@dataclass(frozen=True)
class Evaluation:
    """Rolling eligible-window metrics and current observation eligibility."""

    ready: bool
    label_ok: bool
    comm_bad: bool
    drift_score: float
    position_error_m: float | None


class PositionNet(nn.Module):
    """Small MLP mapping normalized fingerprints to XYZ coordinates."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
