from datetime import UTC, datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, PreparedDataset
from ml.registry import create_model
from ml.storage import ModelBundle, ModelMetadata
from schemas import TrainingConfigChurn


def build_training_pipeline(config: TrainingConfigChurn) -> Pipeline:
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("classifier", create_model(config))]
    )


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    predictions = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
    }


class ModelTrainer:
    def train(
        self, config: TrainingConfigChurn, dataset: PreparedDataset
    ) -> ModelBundle:
        if dataset.X_train.empty or dataset.y_train.empty:
            raise ValueError("training dataset is empty")
        model = build_training_pipeline(config)
        model.fit(dataset.X_train, dataset.y_train)
        metrics = evaluate_model(model, dataset.X_test, dataset.y_test)
        return ModelBundle(
            model=model,
            metadata=ModelMetadata(
                config=config,
                trained=True,
                trained_at=datetime.now(UTC),
                metrics=metrics,
            ),
        )
