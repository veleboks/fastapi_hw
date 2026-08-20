import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from dataset import info_dataset, load_dataset
from exception_handlers import register_exception_handlers
from model_history import append_training_record, load_training_history
from model_inference import predict_batch, predict_single
from model_storage import ModelMetadata, load_cached_model_bundle
from preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    PreparedDataset,
    prepare_dataset,
)
from schemas import (
    DatasetRowChurn,
    ErrorResponse,
    FeatureVectorChurn,
    MetricsHistoryResponse,
    ModelTypeChurn,
    PredictionResponseChurn,
    TrainingConfigChurn,
    TrainingHistoryRecord,
)
from training import build_model_bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.dataset = load_dataset(Path("data/churn_dataset.csv"))
    app.state.split = prepare_dataset(app.state.dataset) if app.state.dataset else None
    app.state.model_bundle = load_cached_model_bundle()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    yield


app = FastAPI(lifespan=lifespan)
register_exception_handlers(app)


@app.get("/")
def root() -> dict:
    return {"message": "ml churn service is running"}


@app.post(
    "/predict",
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Invalid feature values or extra features",
            "content": {
                "application/json": {
                    "example": {
                        "code": "VALIDATION_ERROR",
                        "message": "Request validation failed",
                        "details": [
                            {
                                "location": ["body", "monthly_fee"],
                                "message": "Input should be a valid number",
                                "type": "float_parsing",
                            }
                        ],
                    }
                }
            },
        },
        503: {
            "model": ErrorResponse,
            "description": "Model is not trained",
            "content": {
                "application/json": {
                    "example": {
                        "code": "HTTP_503",
                        "message": "Model is not trained",
                        "details": None,
                    }
                }
            },
        },
    },
)
def predict(
    features: FeatureVectorChurn | list[FeatureVectorChurn],
) -> PredictionResponseChurn | list[PredictionResponseChurn]:
    bundle = app.state.model_bundle
    if not bundle.metadata.trained:
        raise HTTPException(status_code=503, detail="Model is not trained")

    if isinstance(features, list):
        return predict_batch(bundle, features)
    return predict_single(bundle, features)


@app.get("/dataset/preview")
def preview(n: int = 5) -> list[DatasetRowChurn]:
    return app.state.dataset[:n]


@app.get("/dataset/info")
def info() -> dict[str, Any]:
    return info_dataset(app.state.dataset)


@app.get("/dataset/split-info")
def split_info() -> dict[str, Any]:
    if app.state.split is None:
        raise HTTPException(status_code=503, detail="Dataset is empty")
    split: PreparedDataset = app.state.split
    return split.split_info()


@app.get("/model/schema")
def model_schema() -> dict[str, list[dict[str, str]]]:
    def describe(columns: list[str]) -> list[dict[str, str]]:
        return [
            {
                "name": name,
                "type": FeatureVectorChurn.model_fields[name].annotation.__name__,
            }
            for name in columns
        ]

    return {
        "numeric": describe(NUMERIC_COLUMNS),
        "categorical": describe(CATEGORICAL_COLUMNS),
    }


@app.post(
    "/model/train",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Training or hyperparameter error",
            "content": {
                "application/json": {
                    "example": {
                        "code": "PROCESSING_ERROR",
                        "message": "Unable to process the request",
                        "details": "Unknown training parameter",
                    }
                }
            },
        },
        422: {
            "model": ErrorResponse,
            "description": "Invalid training configuration",
        },
        503: {
            "model": ErrorResponse,
            "description": "Dataset is empty or unavailable",
        },
    },
)
def train_model(config: TrainingConfigChurn | None = None) -> dict[str, float]:
    dataset = getattr(app.state, "dataset", None)
    if not dataset:
        raise HTTPException(status_code=503, detail="dataset is not loaded or empty")

    if config is None:
        config = TrainingConfigChurn()

    split: PreparedDataset | None = app.state.split
    if split is None:
        raise HTTPException(status_code=503, detail="Dataset is empty")
    try:
        bundle = build_model_bundle(
            config, split.X_train, split.y_train, split.X_test, split.y_test
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except TypeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    app.state.model_bundle = bundle

    if bundle.metadata.config is None or bundle.metadata.trained_at is None:
        raise HTTPException(status_code=500, detail="Model metadata is incomplete")

    append_training_record(
        TrainingHistoryRecord(
            timestamp=bundle.metadata.trained_at,
            model_type=bundle.metadata.config.model_type,
            hyperparameters=bundle.metadata.config.hyperparameters,
            metrics=bundle.metadata.metrics or {},
        )
    )

    assert bundle.metadata.metrics is not None

    return bundle.metadata.metrics


@app.get("/model/metrics", response_model=MetricsHistoryResponse)
def model_metrics(
    model_type: ModelTypeChurn | None = None,
    limit: int = Query(default=10, ge=1, le=100),
) -> MetricsHistoryResponse:
    history = load_training_history()
    if model_type is not None:
        history = [record for record in history if record.model_type == model_type]

    recent_history = list(reversed(history))[:limit]
    return MetricsHistoryResponse(
        latest=recent_history[0] if recent_history else None,
        history=recent_history,
    )


@app.get("/model/status")
def status() -> ModelMetadata:
    bundle = app.state.model_bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model metadata is absent")
    return app.state.model_bundle.metadata
