"""Fixed-reference, rolling-window KPI and observation evaluation."""

from collections import deque

import numpy as np
from numpy.typing import ArrayLike

from .config import Config
from .models import Array, Evaluation, Observation, finite_array, finite_vector


class EvaluationEngine:
    """Detect diagonal mean shifts, not covariance or general concept drift.

    Reference statistics are fitted only on initial training/calibration data.
    Unreliable observations clear all accumulated adaptation evidence.
    """

    def __init__(self, reference_x: ArrayLike, config: Config) -> None:
        self.c = config
        ref = finite_array(reference_x, "reference_x").astype(np.float64)
        if ref.ndim != 2 or len(ref) < 2 or ref.shape[1] < 1:
            raise ValueError("reference_x must have shape [N>=2, D>=1]")
        self.dim = ref.shape[1]
        self.mean = ref.mean(axis=0)
        self.scale = np.maximum(ref.std(axis=0), config.normalization_floor)
        self.mean.flags.writeable = self.scale.flags.writeable = False
        self.history: deque[tuple[Array, float, float, float | None]] = deque(
            maxlen=config.window
        )

    def normalize(self, x: ArrayLike) -> Array:
        """Normalize one vector or a matrix with the frozen reference."""
        a = finite_array(x, "features")
        if a.ndim not in (1, 2) or a.shape[-1] != self.dim:
            raise ValueError(f"features must have shape [D] or [N,D], D={self.dim}")
        return finite_array((a - self.mean) / self.scale, "normalized features")

    def label_is_reliable(self, obs: Observation) -> bool:
        """Validate observation inputs and evaluate independent label quality."""
        finite_vector(obs.features, self.dim)
        if not all(np.isfinite(v) for v in (
            obs.rsrp_dbm, obs.sensing_snr_db, obs.label_confidence,
        )) or not 0 <= obs.label_confidence <= 1:
            raise ValueError("KPIs must be finite; label_confidence must be in [0, 1]")
        if obs.bler is not None and (
            not np.isfinite(obs.bler) or not 0 <= obs.bler <= 1
        ):
            raise ValueError("BLER must be finite and in [0, 1]")
        if obs.xyz_m is not None:
            finite_vector(obs.xyz_m, 3)
        return (
            obs.xyz_m is not None
            and obs.sensing_snr_db >= self.c.sensing_snr_min_db
            and obs.label_confidence >= self.c.label_confidence_min
        )

    def evaluate(self, obs: Observation, prediction: ArrayLike) -> Evaluation:
        """Evaluate one observation; partial windows report metrics but are not ready."""
        prediction = finite_vector(prediction, 3)
        if not self.label_is_reliable(obs):
            self.history.clear()
            return Evaluation(False, False, False, 0.0, None)

        error = float(np.linalg.norm(
            prediction.astype(np.float64) - finite_vector(obs.xyz_m, 3)
        ))
        self.history.append((
            self.normalize(obs.features), error, obs.rsrp_dbm, obs.bler,
        ))
        rows = list(self.history)
        center = np.stack([r[0] for r in rows]).mean(axis=0, dtype=np.float64)
        drift = float(np.sqrt(np.mean(center ** 2)))
        # Log-sum-exp form avoids overflow when averaging power in linear units.
        rsrp = np.array([r[2] for r in rows], dtype=np.float64)
        peak = rsrp.max()
        mean_rsrp = peak + 10 * np.log10(np.mean(10 ** ((rsrp - peak) / 10)))
        blers = [r[3] for r in rows if r[3] is not None]
        comm_bad = mean_rsrp < self.c.rsrp_low_dbm or (
            bool(blers) and float(np.mean(blers)) > self.c.bler_high
        )
        return Evaluation(
            len(rows) == self.c.window, True, bool(comm_bad), drift,
            float(np.mean([r[1] for r in rows])),
        )
