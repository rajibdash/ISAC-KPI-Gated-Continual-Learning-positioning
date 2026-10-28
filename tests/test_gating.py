from dataclasses import replace

import pytest

from isac_cl import Config, Evaluation, KPIGate


BAD = Evaluation(True, True, False, 2.0, 0.0)


def test_persistence_pending_and_attempt_cooldown():
    gate = KPIGate(Config(persistence=2, cooldown_steps=4, min_new_samples=2))
    assert gate.decide(0, BAD, 2) == (False, "waiting_for_persistence")
    assert gate.decide(1, BAD, 1) == (False, "collecting_labeled_samples")
    assert gate.decide(2, BAD, 2) == (True, "feature_drift")
    gate.record_attempt(2)
    assert gate.decide(3, BAD, 2)[1] == "waiting_for_persistence"
    assert gate.decide(4, BAD, 2) == (False, "cooldown")
    assert gate.decide(6, BAD, 2) == (True, "feature_drift")


@pytest.mark.parametrize("evaluation,reason", [
    (replace(BAD, label_ok=False), "unreliable_observation_or_label"),
    (replace(BAD, ready=False), "warming_up"),
    (replace(BAD, drift_score=0), "stable"),
    (replace(BAD, drift_score=0, comm_bad=True), "communication_only_degradation"),
])
def test_reason_codes_and_streak_reset(evaluation, reason):
    gate = KPIGate(Config(persistence=2))
    gate.decide(0, BAD, 100)
    assert gate.decide(1, evaluation, 100) == (False, reason)
    assert gate.streak == 0
    assert gate.decide(2, BAD, 100)[1] == "waiting_for_persistence"


def test_error_can_trigger_without_comm_degradation():
    gate = KPIGate(Config(persistence=1))
    assert gate.decide(0, replace(BAD, drift_score=0, position_error_m=3), 100) == (
        True, "position_error"
    )


def test_duplicate_steps_cannot_manufacture_persistence():
    gate = KPIGate(Config())
    gate.decide(0, BAD, 100)
    with pytest.raises(ValueError, match="increasing"):
        gate.decide(0, BAD, 100)
    assert gate.streak == 1
    with pytest.raises(ValueError):
        gate.decide(1, BAD, -1)
