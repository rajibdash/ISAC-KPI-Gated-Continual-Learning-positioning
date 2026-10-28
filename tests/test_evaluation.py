from dataclasses import replace

import numpy as np
import pytest

from isac_cl import Config, EvaluationEngine, Observation


def observation(**changes):
    return replace(Observation([1.0, 1.0], -80, 20, 1.0, [0, 0, 0]), **changes)


def evaluator(**changes):
    return EvaluationEngine([[-1, -1], [1, 1]], Config(window=2, **changes))


def test_window_and_fixed_reference_drift():
    engine = evaluator()
    first = engine.evaluate(observation(), [3, 4, 0])
    assert not first.ready
    assert first.position_error_m == 5
    assert first.drift_score == 1
    assert engine.evaluate(observation(), [0, 0, 0]).ready
    np.testing.assert_array_equal(engine.mean, [0, 0])
    np.testing.assert_array_equal(engine.scale, [1, 1])


@pytest.mark.parametrize("changes", [
    {"xyz_m": None}, {"sensing_snr_db": 0}, {"label_confidence": 0.1},
])
def test_unreliable_labels_clear_window(changes):
    engine = evaluator()
    engine.evaluate(observation(), [0, 0, 0])
    result = engine.evaluate(observation(**changes), [0, 0, 0])
    assert not result.ready and not result.label_ok
    assert result.position_error_m is None and result.drift_score == 0
    assert len(engine.history) == 0
    assert not engine.evaluate(observation(), [0, 0, 0]).ready


@pytest.mark.parametrize("rsrp,bler,expected", [
    (-100, None, True), (-80, 0.5, True), (-80, None, False),
    (-95, 0.1, False), (-80, 0.0, False),
])
def test_communication_paths(rsrp, bler, expected):
    result = evaluator().evaluate(observation(rsrp_dbm=rsrp, bler=bler), [0, 0, 0])
    assert result.comm_bad is expected


def test_rsrp_is_averaged_as_power_and_missing_bler_is_ignored():
    engine = evaluator()
    engine.evaluate(observation(rsrp_dbm=-110, bler=0.2), [0, 0, 0])
    assert engine.evaluate(observation(rsrp_dbm=-80), [0, 0, 0]).comm_bad
    engine = evaluator()
    engine.evaluate(observation(rsrp_dbm=-110), [0, 0, 0])
    assert not engine.evaluate(observation(rsrp_dbm=-80), [0, 0, 0]).comm_bad


@pytest.mark.parametrize("changes", [
    {"features": [np.nan, 0]}, {"features": [1]}, {"features": [1j, 0]},
    {"xyz_m": [0, 0]}, {"xyz_m": [0, np.inf, 0]},
    {"bler": -0.1}, {"bler": np.nan}, {"label_confidence": 2},
    {"rsrp_dbm": np.inf}, {"sensing_snr_db": np.nan},
])
def test_invalid_observations_raise(changes):
    with pytest.raises(ValueError):
        evaluator().evaluate(observation(**changes), [0, 0, 0])


@pytest.mark.parametrize("reference", [[], [[0, 1]], [[np.inf], [0]], [[], []]])
def test_invalid_reference(reference):
    with pytest.raises(ValueError):
        EvaluationEngine(reference, Config())


def test_constant_reference_and_invalid_normalization():
    engine = EvaluationEngine([[0, 0], [0, 0]], Config())
    assert np.isfinite(engine.normalize([1, 1])).all()
    for bad in (1, [1], [[[0, 0]]], [np.nan, 0]):
        with pytest.raises(ValueError):
            engine.normalize(bad)


@pytest.mark.parametrize("changes", [
    {"window": 0}, {"persistence": 1.5}, {"cooldown_steps": -1},
    {"learning_rate": np.nan}, {"pending_capacity": 1},
    {"label_confidence_min": 2}, {"max_old_regression_m": -1},
    {"max_grad_norm": 0}, {"window": True},
])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        Config(**changes)
