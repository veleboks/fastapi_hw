import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from dataset import info_dataset, load_dataset
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
    FeatureVectorChurn,
    PredictionResponseChurn,
    TrainingConfigChurn,
)
from training import build_model_bundle

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.dataset = load_dataset(Path("data/churn_dataset.csv"))
    app.state.split = prepare_dataset(app.state.dataset)
    app.state.model_bundle = load_cached_model_bundle()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root() -> dict:
    return {"message": "ml churn service is running"}


@app.post("/predict")
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


@app.post("/model/train")
def train_model(config: TrainingConfigChurn | None = None) -> dict[str, float]:
    dataset = getattr(app.state, "dataset", None)
    if not dataset:
        raise HTTPException(status_code=503, detail="dataset is not loaded or empty")

    if config is None:
        config = TrainingConfigChurn()

    split: PreparedDataset = app.state.split
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

    assert bundle.metadata.metrics is not None

    return bundle.metadata.metrics


@app.get("/model/status")
def status() -> ModelMetadata:
    bundle = app.state.model_bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model metadata is absent")
    return app.state.model_bundle.metadata
