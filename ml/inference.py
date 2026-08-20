import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import TypeAdapter

from ml.preprocessing import FEATURE_COLUMNS
from ml.storage import ModelBundle
from schemas import FeatureVectorChurn, PredictionResponseChurn


def _make_prediction_response(
    classes: NDArray[np.int_], probas: NDArray[np.float64]
) -> PredictionResponseChurn:
    class_names = [str(value) for value in classes]
    return PredictionResponseChurn(
        predicted_churn=int(classes[probas.argmax()]),
        probabilities=dict(zip(class_names, probas)),
    )


def _features_to_frame(features: list[FeatureVectorChurn]) -> pd.DataFrame:
    adapter = TypeAdapter(list[FeatureVectorChurn])
    frame = pd.DataFrame(adapter.dump_python(features))
    return frame.reindex(columns=FEATURE_COLUMNS)


def predict_single(
    bundle: ModelBundle, features: FeatureVectorChurn
) -> PredictionResponseChurn:
    if bundle.model is None:
        raise ValueError("Model is not trained")
    probabilities = bundle.model.predict_proba(_features_to_frame([features]))[0]
    return _make_prediction_response(bundle.model.classes_, probabilities)


def predict_batch(
    bundle: ModelBundle, features: list[FeatureVectorChurn]
) -> list[PredictionResponseChurn]:
    if bundle.model is None:
        raise ValueError("Model is not trained")
    probabilities = bundle.model.predict_proba(_features_to_frame(features))
    return [
        _make_prediction_response(bundle.model.classes_, proba)
        for proba in probabilities
    ]
