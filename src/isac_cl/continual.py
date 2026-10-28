"""Isolated replay-based adaptation with independent validation safeguards."""

from collections import deque
from copy import deepcopy
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike
import torch
from torch import nn

from .config import Config
from .evaluation import EvaluationEngine
from .models import (
    Array, Evaluation, Observation, ValidationSet, finite_vector, labeled_arrays,
)

Sample = tuple[Array, Array]


class AdaptationResult(TypedDict, total=False):
    """Attempt outcome; validation metrics are absent for insufficient data."""

    accepted: bool
    reason: str
    old_rmse_before_m: float
    old_rmse_after_m: float
    recent_rmse_before_m: float
    recent_rmse_after_m: float


class Reservoir:
    """Bounded uniform reservoir over every admitted sample, stored by value."""

    def __init__(self, capacity: int, seed: int = 42) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.items: list[Sample] = []
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, x: Array, y: Array) -> None:
        """Insert one validated feature/label pair using reservoir sampling."""
        item = (x.copy(), y.copy())
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            index = int(self.rng.integers(self.seen))
            if index < self.capacity:
                self.items[index] = item


class ContinualLearningEngine:
    """Train a deep-copied candidate, deploying only after both validation guards.

    Callers own trajectory/time-based train/validation separation. Pending data
    is consumed into replay after every completed attempt, even on rejection.
    Instances are synchronous and not safe for concurrent calls.
    """

    def __init__(
        self, model: nn.Module, evaluator: EvaluationEngine, config: Config,
        initial_train_x: ArrayLike, initial_train_y: ArrayLike, seed: int = 42,
    ) -> None:
        if evaluator.c != config:
            raise ValueError("evaluator and learner must use the same config")
        parameters = list(model.parameters())
        if not parameters or not any(p.requires_grad for p in parameters):
            raise ValueError("model must have trainable parameters")
        self.model = model
        self.evaluator = evaluator
        self.c = config
        self.device = parameters[0].device
        self.rng = np.random.default_rng(seed)
        self.memory = Reservoir(config.replay_capacity, seed)
        self.pending: deque[Sample] = deque(maxlen=config.pending_capacity)
        x, y = labeled_arrays(initial_train_x, initial_train_y, evaluator.dim)
        for a, b in zip(x, y):
            self.memory.add(a, b)

    def tensors(self, x: ArrayLike, y: ArrayLike) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and normalize a labeled batch on the deployed model's device."""
        features, labels = labeled_arrays(x, y, self.evaluator.dim)
        return (
            torch.as_tensor(self.evaluator.normalize(features), device=self.device),
            torch.as_tensor(labels, device=self.device),
        )

    @torch.no_grad()
    def predict(self, features: ArrayLike) -> Array:
        """Infer finite XYZ coordinates using only the deployed model."""
        x = self.evaluator.normalize(finite_vector(features, self.evaluator.dim))
        self.model.eval()
        prediction = self.model(torch.as_tensor(x, device=self.device)[None])
        if prediction.shape != (1, 3):
            raise ValueError("model output must have shape [N, 3]")
        return finite_vector(prediction[0].cpu().numpy(), 3)

    def admit(self, obs: Observation, evaluation: Evaluation) -> None:
        """Copy eligible observations into bounded FIFO pending storage."""
        reliable = self.evaluator.label_is_reliable(obs)
        if evaluation.label_ok and reliable:
            self.pending.append((
                finite_vector(obs.features, self.evaluator.dim).copy(),
                finite_vector(obs.xyz_m, 3).copy(),
            ))

    def batch(self, items: list[Sample], count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample with replacement using the engine-local seeded generator."""
        indices = self.rng.integers(len(items), size=count)
        x, y = zip(*(items[int(i)] for i in indices))
        return self.tensors(np.stack(x), np.stack(y))

    @torch.no_grad()
    def rmse(self, model: nn.Module, validation: ValidationSet) -> float:
        """Compute 3D position RMSE (not per-coordinate RMSE) in meters."""
        x, y = self.tensors(*validation)
        was_training = model.training
        model.eval()
        try:
            prediction = model(x)
            if prediction.shape != y.shape:
                raise ValueError("model output must have shape [N, 3]")
            # Float64 avoids overflow while squaring otherwise finite positions.
            distances = (prediction.double() - y.double()).square().sum(dim=1)
            return float(distances.mean().sqrt().cpu())
        finally:
            model.train(was_training)

    def _train_candidate(self, candidate: nn.Module, fresh: list[Sample]) -> bool:
        """Return False on nonfinite loss, gradient, or updated model state."""
        optimizer = torch.optim.Adam(candidate.parameters(), lr=self.c.learning_rate)
        candidate.train()
        for _ in range(self.c.train_steps):
            x, y = self.batch(fresh, self.c.batch_per_source)
            if self.memory.items:
                old_x, old_y = self.batch(self.memory.items, self.c.batch_per_source)
                x, y = torch.cat((x, old_x)), torch.cat((y, old_y))
            optimizer.zero_grad(set_to_none=True)
            prediction = candidate(x)
            if prediction.shape != y.shape:
                raise ValueError("model output must have shape [N, 3]")
            loss = nn.functional.mse_loss(prediction, y)
            if not torch.isfinite(loss):
                return False
            loss.backward()
            norm = nn.utils.clip_grad_norm_(candidate.parameters(), self.c.max_grad_norm)
            if not torch.isfinite(norm):
                return False
            optimizer.step()
            if not all(torch.isfinite(v).all() for v in candidate.state_dict().values()):
                return False
        return True

    def adapt(
        self, old_validation: ValidationSet, recent_validation: ValidationSet,
    ) -> AdaptationResult:
        """Attempt supervised adaptation without modifying deployed state on rejection.

        Invalid validation arrays raise before any training or buffer consumption.
        Numerical failures reject the candidate and still consume fresh data.
        """
        if len(self.pending) < self.c.min_new_samples:
            return {"accepted": False, "reason": "insufficient_data"}
        before_old = self.rmse(self.model, old_validation)
        before_recent = self.rmse(self.model, recent_validation)
        candidate = deepcopy(self.model)
        fresh = list(self.pending)
        numerical_ok = bool(np.isfinite([before_old, before_recent]).all())
        if numerical_ok:
            numerical_ok = self._train_candidate(candidate, fresh)
        after_old = self.rmse(candidate, old_validation) if numerical_ok else float("inf")
        after_recent = self.rmse(candidate, recent_validation) if numerical_ok else float("inf")
        numerical_ok = numerical_ok and bool(np.isfinite([after_old, after_recent]).all())
        accepted = (
            numerical_ok
            and after_recent < before_recent - self.c.min_recent_improvement_m
            and after_old <= before_old + self.c.max_old_regression_m
        )
        if accepted:
            self.model.load_state_dict(candidate.state_dict())
        for x, y in fresh:
            self.memory.add(x, y)
        self.pending.clear()
        return {
            "accepted": bool(accepted),
            "reason": "accepted" if accepted else (
                "validation_rejected" if numerical_ok else "numerical_failure"
            ),
            "old_rmse_before_m": before_old,
            "old_rmse_after_m": after_old,
            "recent_rmse_before_m": before_recent,
            "recent_rmse_after_m": after_recent,
        }
