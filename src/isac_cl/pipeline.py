"""Synchronous orchestration of inference, evaluation, gating, and adaptation."""

from typing import TypedDict

from .continual import AdaptationResult, ContinualLearningEngine
from .evaluation import EvaluationEngine
from .gating import KPIGate
from .models import Array, Evaluation, Observation, ValidationSet


class ProcessResult(TypedDict):
    """Prediction is from the deployed model before any adaptation at this step."""

    prediction_xyz_m: Array
    evaluation: Evaluation
    reason: str
    adaptation: AdaptationResult | None


def process_observation(
    step: int, obs: Observation, evaluator: EvaluationEngine, gate: KPIGate,
    learner: ContinualLearningEngine, old_validation: ValidationSet | None = None,
    recent_validation: ValidationSet | None = None,
) -> ProcessResult:
    """Process one observation; unavailable validation defers the attempt/cooldown."""
    if learner.evaluator is not evaluator or gate.c != evaluator.c:
        raise ValueError("pipeline engines must share the evaluator and configuration")
    gate.validate_step(step)
    prediction = learner.predict(obs.features)
    evaluation = evaluator.evaluate(obs, prediction)
    learner.admit(obs, evaluation)
    trigger, reason = gate.decide(step, evaluation, len(learner.pending))
    result = None
    if trigger:
        if old_validation is None or recent_validation is None:
            reason = "waiting_for_independent_validation"
        else:
            # Validate before starting cooldown, so invalid caller data is not an attempt.
            learner.tensors(*old_validation)
            learner.tensors(*recent_validation)
            gate.record_attempt(step)
            result = learner.adapt(old_validation, recent_validation)
    return {
        "prediction_xyz_m": prediction,
        "evaluation": evaluation,
        "reason": reason,
        "adaptation": result,
    }
