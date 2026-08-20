import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import TypeAdapter

from model_storage import ModelBundle
from preprocessing import FEATURE_COLUMNS
from schemas import FeatureVectorChurn, PredictionResponseChurn


def _make_prediction_response(
    classes: NDArray[np.int_], probas: NDArray[np.float64]
) -> PredictionResponseChurn:
    classes_str = [str(x) for x in classes]
    return PredictionResponseChurn(
        predicted_churn=classes[probas.argmax()],
        probabilities=dict(zip(classes_str, probas)),
    )


def predict_single(
    bundle: ModelBundle, features: FeatureVectorChurn
) -> PredictionResponseChurn:
    type_adapter = TypeAdapter(list[FeatureVectorChurn])
    X = pd.DataFrame(type_adapter.dump_python([features]))
    X = X.reindex(columns=FEATURE_COLUMNS)

    assert bundle.model is not None
    proba = bundle.model.predict_proba(X)[0]
    classes = bundle.model.classes_

    return _make_prediction_response(classes, proba)


def predict_batch(
    bundle: ModelBundle, features: list[FeatureVectorChurn]
) -> list[PredictionResponseChurn]:
    type_adapter = TypeAdapter(list[FeatureVectorChurn])
    X = pd.DataFrame(type_adapter.dump_python(features))
    X = X.reindex(columns=FEATURE_COLUMNS)

    assert bundle.model is not None
    probas = bundle.model.predict_proba(X)
    classes = bundle.model.classes_

    return [_make_prediction_response(classes, proba) for proba in probas]
