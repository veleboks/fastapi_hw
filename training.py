from datetime import UTC, datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_storage import ModelBundle, ModelMetadata, dump_model_bundle
from preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def _build_training_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def _train_churn_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    if X_train.empty or y_train.empty:
        raise ValueError("training dataset is empty")

    model = _build_training_pipeline()
    model.fit(X_train, y_train)
    return model


def evaluate_churn_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    predictions = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
    }


def build_model_bundle(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> ModelBundle:
    model = _train_churn_model(X_train, y_train)
    metrics = evaluate_churn_model(model, X_test, y_test)
    trained_at = datetime.now(UTC)
    bundle = ModelBundle(
        model=model,
        metadata=ModelMetadata(trained=True, trained_at=trained_at, metrics=metrics),
    )
    dump_model_bundle(bundle)  # cache
    return bundle
