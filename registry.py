from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from schemas import ModelTypeChurn, TrainingConfigChurn

MODEL_REGISTRY = {
    ModelTypeChurn.LOGREG: lambda params: LogisticRegression(**params),
    ModelTypeChurn.RANDOM_FOREST: lambda params: RandomForestClassifier(**params),
}


def get_registry_entry(config: TrainingConfigChurn):
    if config.model_type not in MODEL_REGISTRY:
        raise KeyError(f"{config.model_type=} is not supported")
    return MODEL_REGISTRY[config.model_type](config.hyperparameters)
