import numpy as np
import pytest

from isac_cl import KPIGate, Observation, process_observation
from test_continual import make_learner, validations


def test_pipeline_defers_without_validation_then_adapts_and_cools_down():
    learner = make_learner(regression=0)
    gate = KPIGate(learner.c)
    obs = Observation([2], -80, 20, 1, [1, 0, 0])
    first = process_observation(0, obs, learner.evaluator, gate, learner)
    assert first["reason"] == "collecting_labeled_samples"
    waiting = process_observation(1, obs, learner.evaluator, gate, learner)
    assert waiting["reason"] == "waiting_for_independent_validation"
    assert gate.last_attempt == -learner.c.cooldown_steps
    attempted = process_observation(
        2, obs, learner.evaluator, gate, learner, *validations(),
    )
    assert attempted["adaptation"]["reason"] == "validation_rejected"
    np.testing.assert_array_equal(attempted["prediction_xyz_m"], [0, 0, 0])
    assert gate.last_attempt == 2
    cooldown = process_observation(3, obs, learner.evaluator, gate, learner, *validations())
    assert cooldown["reason"] == "cooldown"
    assert len(learner.pending) == 1


@pytest.mark.parametrize("invalid_step", [0, -1, 0.5, True])
def test_invalid_pipeline_steps_do_not_mutate_evidence_or_pending(invalid_step):
    learner = make_learner()
    gate = KPIGate(learner.c)
    obs = Observation([2], -80, 20, 1, [1, 0, 0])
    process_observation(0, obs, learner.evaluator, gate, learner)
    history = list(learner.evaluator.history)
    with pytest.raises(ValueError):
        process_observation(invalid_step, obs, learner.evaluator, gate, learner)
    assert len(learner.pending) == 1
    assert len(learner.evaluator.history) == len(history)
    np.testing.assert_array_equal(learner.evaluator.history[0][0], history[0][0])
    assert gate.streak == 1
