from collections.abc import Callable
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from schemas import ModelTypeChurn, TrainingConfigChurn

MODEL_REGISTRY: dict[ModelTypeChurn, Callable[..., Any]] = {
    ModelTypeChurn.LOGREG: LogisticRegression,
    ModelTypeChurn.RANDOM_FOREST: RandomForestClassifier,
}


def create_model(config: TrainingConfigChurn) -> Any:
    try:
        factory = MODEL_REGISTRY[config.model_type]
    except KeyError as error:
        raise ValueError(f"Model type {config.model_type} is not registered") from error
    return factory(**config.hyperparameters)
