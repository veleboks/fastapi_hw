import logging
from datetime import UTC, datetime

import pytest

from ml.dataset import load_dataset
from ml.history import JsonTrainingHistoryRepository
from ml.preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    prepare_dataset,
)
from ml.registry import create_model
from ml.service import ChurnModelService
from ml.storage import JoblibModelRepository, ModelBundle, ModelMetadata
from ml.training import ModelTrainer
from schemas import (
    DatasetRowChurn,
    FeatureVectorChurn,
    ModelTypeChurn,
    TrainingConfigChurn,
    TrainingHistoryRecord,
)


def make_dataset(size: int = 40) -> list[DatasetRowChurn]:
    regions = ["europe", "asia", "america", "africa"]
    devices = ["mobile", "desktop", "tablet"]
    payments = ["card", "paypal", "crypto"]
    return [
        DatasetRowChurn(
            monthly_fee=30.0 + index,
            usage_hours=80.0 + index * 2,
            support_requests=index % 5,
            account_age_months=6 + index,
            failed_payments=index % 3,
            region=regions[index % len(regions)],
            device_type=devices[index % len(devices)],
            payment_method=payments[index % len(payments)],
            autopay_enabled=index % 2,
            churn=index % 2,
        )
        for index in range(size)
    ]


def make_service(tmp_path, dataset: list[DatasetRowChurn]) -> ChurnModelService:
    split = prepare_dataset(dataset) if dataset else None
    return ChurnModelService(
        dataset=dataset,
        split=split,
        model_repository=JoblibModelRepository(tmp_path / "model.joblib"),
        history_repository=JsonTrainingHistoryRepository(tmp_path / "history.json"),
        trainer=ModelTrainer(),
        model_bundle=ModelBundle(),
    )


def test_prepare_dataset_splits_and_classifies_features():
    split = prepare_dataset(make_dataset())

    assert len(split.X_train) == 32
    assert len(split.X_test) == 8
    assert split.numeric_columns == NUMERIC_COLUMNS
    assert split.categorical_columns == CATEGORICAL_COLUMNS
    assert set(split.y_train) == {0, 1}


def test_model_registry_creates_supported_models():
    logreg = create_model(TrainingConfigChurn(model_type=ModelTypeChurn.LOGREG))
    forest = create_model(
        TrainingConfigChurn(
            model_type=ModelTypeChurn.RANDOM_FOREST,
            hyperparameters={"n_estimators": 10, "random_state": 42},
        )
    )

    assert logreg.__class__.__name__ == "LogisticRegression"
    assert forest.__class__.__name__ == "RandomForestClassifier"
    assert forest.n_estimators == 10


def test_model_trainer_returns_metrics():
    split = prepare_dataset(make_dataset())
    bundle = ModelTrainer().train(
        TrainingConfigChurn(hyperparameters={"max_iter": 100}), split
    )

    assert bundle.metadata.trained is True
    assert set(bundle.metadata.metrics or {}) == {"accuracy", "f1"}
    assert bundle.model is not None


def test_training_handles_missing_values_without_target_leakage():
    dataset = make_dataset()
    dataset[0] = dataset[0].model_copy(update={"monthly_fee": None, "region": None})
    split = prepare_dataset(dataset)

    bundle = ModelTrainer().train(TrainingConfigChurn(), split)
    preprocessor = bundle.model.named_steps["preprocessor"]
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    encoder = categorical_pipeline.named_steps["encoder"]

    assert numeric_pipeline.named_steps["imputer"].strategy == "mean"
    assert encoder.cv.n_splits == 5
    assert encoder.cv.shuffle is True
    assert encoder.cv.random_state == 42


def test_load_dataset_converts_empty_features_to_missing_values(tmp_path):
    dataset_path = tmp_path / "churn.csv"
    dataset_path.write_text(
        "monthly_fee,usage_hours,support_requests,account_age_months,"
        "failed_payments,region,device_type,payment_method,autopay_enabled,churn\n"
        ",120,2,18,0,,mobile,card,1,0\n",
        encoding="utf-8",
    )

    row = load_dataset(dataset_path)[0]

    assert row.monthly_fee is None
    assert row.region is None


def test_load_missing_dataset_returns_empty_list(tmp_path, caplog):
    missing_path = tmp_path / "missing.csv"

    with caplog.at_level(logging.WARNING, logger="ml.dataset"):
        dataset = load_dataset(missing_path)

    assert dataset == []
    assert "Dataset file not found" in caplog.text


def test_model_repository_saves_and_loads_bundle(tmp_path):
    repository = JoblibModelRepository(tmp_path / "nested" / "model.joblib")
    bundle = ModelBundle(
        model="trained-model",
        metadata=ModelMetadata(
            config=TrainingConfigChurn(),
            trained=True,
            metrics={"accuracy": 0.8, "f1": 0.7},
        ),
    )

    repository.save(bundle)
    loaded = repository.load()

    assert loaded.model == "trained-model"
    assert loaded.metadata.trained is True
    assert loaded.metadata.metrics == {"accuracy": 0.8, "f1": 0.7}


def test_training_history_appends_and_filters_records(tmp_path):
    repository = JsonTrainingHistoryRepository(tmp_path / "history.json")
    timestamp = datetime.now(UTC)
    repository.append(
        TrainingHistoryRecord(
            timestamp=timestamp,
            model_type=ModelTypeChurn.LOGREG,
            hyperparameters={},
            metrics={"accuracy": 0.8, "f1": 0.7},
        )
    )
    repository.append(
        TrainingHistoryRecord(
            timestamp=timestamp,
            model_type=ModelTypeChurn.RANDOM_FOREST,
            hyperparameters={"n_estimators": 10},
            metrics={"accuracy": 0.9, "f1": 0.8},
        )
    )

    records = repository.recent(ModelTypeChurn.LOGREG, limit=10)

    assert len(records) == 1
    assert records[0].model_type == ModelTypeChurn.LOGREG


def test_service_predicts_single_and_multiple_clients(tmp_path):
    dataset = make_dataset()
    service = make_service(tmp_path, dataset)
    service.train(TrainingConfigChurn(hyperparameters={"max_iter": 100}))
    feature = FeatureVectorChurn.model_validate(
        dataset[0].model_dump(exclude={"churn"})
    )

    single = service.predict(feature)
    batch = service.predict([feature, feature])

    assert single.predicted_churn in {0, 1}
    assert set(single.probabilities) == {"0", "1"}
    assert len(batch) == 2


def test_service_rejects_prediction_without_trained_model(tmp_path):
    service = make_service(tmp_path, make_dataset())
    feature = FeatureVectorChurn.model_validate(
        make_dataset(1)[0].model_dump(exclude={"churn"})
    )

    with pytest.raises(ValueError, match="Model is not trained"):
        service.predict(feature)


def test_service_health_reports_ready_and_degraded_states(tmp_path):
    ready_service = make_service(tmp_path / "ready", make_dataset())
    ready_service.model_bundle = ModelBundle(
        model=object(), metadata=ModelMetadata(trained=True)
    )
    degraded_service = make_service(tmp_path / "degraded", make_dataset())

    assert ready_service.health().status.value == "ok"
    assert ready_service.health().dataset_loaded is True
    assert ready_service.health().model_loaded is True
    assert degraded_service.health().status.value == "degraded"
    assert degraded_service.health().model_loaded is False


def test_service_health_writes_state_to_log(tmp_path, caplog):
    service = make_service(tmp_path, make_dataset())

    with caplog.at_level(logging.INFO, logger="ml.service"):
        service.health()

    assert "Health check:" in caplog.text
    assert "dataset_loaded=True" in caplog.text
    assert "model_loaded=False" in caplog.text
