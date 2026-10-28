from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from isac_cl import (
    Config, ContinualLearningEngine, Evaluation, EvaluationEngine, Observation,
    PositionNet, Reservoir,
)


ELIGIBLE = Evaluation(True, True, False, 1, 1)


def make_learner(regression=2.0, model=None):
    config = Config(
        window=1, persistence=1, cooldown_steps=3, min_new_samples=2,
        pending_capacity=3, replay_capacity=8, train_steps=60,
        learning_rate=0.05, batch_per_source=2,
        max_old_regression_m=regression,
    )
    engine = EvaluationEngine([[-1], [1]], config)
    model = nn.Linear(1, 3) if model is None else model
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return ContinualLearningEngine(
        model, engine, config, [[-1], [1]], np.zeros((2, 3)), seed=42,
    )


def fill_pending(learner):
    for x in (2, 3):
        learner.admit(Observation([x], -80, 20, 1, [1, 0, 0]), ELIGIBLE)


def validations():
    # Distinct fingerprints from both replay and pending training samples.
    return (([[-0.5], [0.5]], np.zeros((2, 3))),
            ([[2.25], [2.75]], [[1, 0, 0], [1, 0, 0]]))


def test_insufficient_data_preserves_buffers():
    learner = make_learner()
    learner.admit(Observation([2], -80, 20, 1, [1, 0, 0]), ELIGIBLE)
    assert learner.adapt(*validations()) == {
        "accepted": False, "reason": "insufficient_data",
    }
    assert len(learner.pending) == 1 and learner.memory.seen == 2


def test_candidate_acceptance_and_replay_insertion():
    learner = make_learner()
    deployed = learner.model
    fill_pending(learner)
    result = learner.adapt(*validations())
    assert result["accepted"] and result["reason"] == "accepted"
    assert result["recent_rmse_after_m"] < result["recent_rmse_before_m"] - 0.1
    assert learner.model is deployed
    assert len(learner.pending) == 0
    assert learner.memory.seen == 4
    assert not np.allclose(learner.predict([2.5]), 0)


def test_old_regression_rejects_without_touching_deployed_weights():
    learner = make_learner(regression=0)
    fill_pending(learner)
    before = {k: v.clone() for k, v in learner.model.state_dict().items()}
    result = learner.adapt(*validations())
    assert not result["accepted"] and result["reason"] == "validation_rejected"
    assert result["recent_rmse_after_m"] < result["recent_rmse_before_m"]
    assert result["old_rmse_after_m"] > result["old_rmse_before_m"]
    for key, value in learner.model.state_dict().items():
        assert torch.equal(value, before[key])
    assert learner.memory.seen == 4 and len(learner.pending) == 0


def test_pending_fifo_copies_and_rechecks_label_quality():
    learner = make_learner()
    x = np.array([2.0])
    obs = Observation(x, -80, 20, 1, [1, 0, 0])
    learner.admit(obs, ELIGIBLE)
    x[0] = 100
    assert learner.pending[0][0][0] == 2
    learner.admit(replace(obs, label_confidence=0), ELIGIBLE)
    learner.admit(obs, replace(ELIGIBLE, label_ok=False))
    assert len(learner.pending) == 1
    for value in (3, 4, 5):
        learner.admit(replace(obs, features=[value]), ELIGIBLE)
    assert [item[0][0] for item in learner.pending] == [3, 4, 5]
    assert learner.memory.seen == 2


def test_invalid_validation_does_not_consume_pending():
    learner = make_learner()
    fill_pending(learner)
    for invalid in (([], []), ([[0]], [[0, 0]]), ([[0]], [[np.nan, 0, 0]])):
        with pytest.raises(ValueError):
            learner.adapt(validations()[0], invalid)
        assert len(learner.pending) == 2 and learner.memory.seen == 2


class NonfiniteTrainingModel(nn.Linear):
    def forward(self, x):
        output = super().forward(x)
        return output * float("nan") if self.training else output


class BadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return torch.full_like(grad, float("inf"))


class NonfiniteGradientModel(nn.Linear):
    def forward(self, x):
        return BadGradient.apply(super().forward(x))


@pytest.mark.parametrize("model_type", [NonfiniteTrainingModel, NonfiniteGradientModel])
def test_numerical_failure_rejects_and_consumes_pending(model_type):
    learner = make_learner(model=model_type(1, 3))
    fill_pending(learner)
    result = learner.adapt(*validations())
    assert result["reason"] == "numerical_failure" and not result["accepted"]
    assert all(torch.count_nonzero(p) == 0 for p in learner.model.parameters())
    assert learner.memory.seen == 4 and not learner.pending


def test_reservoir_is_bounded_deterministic_and_copies():
    a, b = Reservoir(3, seed=7), Reservoir(3, seed=7)
    x, y = np.array([1.0]), np.zeros(3)
    a.add(x, y)
    b.add(x, y)
    x[0] = 99
    assert a.items[0][0][0] == 1
    for i in range(1, 100):
        a.add(np.array([i]), y)
        b.add(np.array([i]), y)
    assert a.seen == 100 and len(a.items) == 3
    for left, right in zip(a.items, b.items):
        np.testing.assert_array_equal(left[0], right[0])


def test_mlp_output_and_position_rmse():
    model = PositionNet(2)
    assert model(torch.zeros(5, 2)).shape == (5, 3)
    learner = make_learner()
    assert learner.rmse(learner.model, ([[0]], [[3, 4, 0]])) == 5
