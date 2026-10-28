"""Persistence, sample-count, and adaptation-attempt cooldown policy."""

import math

from .config import Config
from .models import Evaluation


def nonnegative_integer(value: int, name: str) -> None:
    """Reject invalid controller counters before mutating state."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


class KPIGate:
    """Count consecutive eligible windows; communication alone never triggers."""

    def __init__(self, config: Config) -> None:
        self.c = config
        self.streak = 0
        self.last_attempt = -config.cooldown_steps
        self._last_step = -1

    def validate_step(self, step: int) -> None:
        """Check observation ordering without mutating controller state."""
        nonnegative_integer(step, "step")
        if step <= self._last_step:
            raise ValueError("step must be strictly increasing")

    def decide(self, step: int, evaluation: Evaluation, pending_count: int) -> tuple[bool, str]:
        """Return a trigger and reason once per strictly increasing observation step."""
        self.validate_step(step)
        nonnegative_integer(pending_count, "pending_count")
        if not math.isfinite(evaluation.drift_score) or evaluation.drift_score < 0:
            raise ValueError("drift_score must be finite and nonnegative")
        error = evaluation.position_error_m
        if error is not None and (not math.isfinite(error) or error < 0):
            raise ValueError("position_error_m must be finite and nonnegative")
        self._last_step = step
        error_bad = error is not None and error > self.c.error_high_m
        evidence = evaluation.ready and evaluation.label_ok and (
            error_bad or evaluation.drift_score > self.c.drift_high
        )
        self.streak = min(self.streak + 1, self.c.persistence) if evidence else 0
        if not evaluation.label_ok:
            return False, "unreliable_observation_or_label"
        if not evaluation.ready:
            return False, "warming_up"
        if not evidence:
            return False, "communication_only_degradation" if evaluation.comm_bad else "stable"
        if self.streak < self.c.persistence:
            return False, "waiting_for_persistence"
        if step - self.last_attempt < self.c.cooldown_steps:
            return False, "cooldown"
        if pending_count < self.c.min_new_samples:
            return False, "collecting_labeled_samples"
        return True, "position_error" if error_bad else "feature_drift"

    def record_attempt(self, step: int) -> None:
        """Start cooldown for an actual attempt, including rejected candidates."""
        nonnegative_integer(step, "step")
        if step != self._last_step or step <= self.last_attempt:
            raise ValueError("attempt must match the latest decision and not be repeated")
        self.last_attempt = step
        self.streak = 0
