from fastapi import APIRouter, HTTPException, Query

from core.dependencies import ModelService
from ml.dataset import info_dataset
from ml.preprocessing import PreparedDataset
from schemas import (
    ErrorResponse,
    FeatureVectorChurn,
    MetricsHistoryResponse,
    ModelTypeChurn,
    PredictionResponseChurn,
    TrainingConfigChurn,
)

router = APIRouter()


@router.get("/")
def root() -> dict[str, str]:
    return {"message": "ml churn service is running"}


@router.post(
    "/predict",
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict(
    features: FeatureVectorChurn | list[FeatureVectorChurn],
    service: ModelService,
) -> PredictionResponseChurn | list[PredictionResponseChurn]:
    if not service.model_bundle.metadata.trained:
        raise HTTPException(status_code=503, detail="Model is not trained")
    return service.predict(features)


@router.get("/dataset/preview")
def dataset_preview(
    service: ModelService,
    n: int = 5,
):
    return service.dataset[:n]


@router.get("/dataset/info")
def dataset_info(service: ModelService):
    return info_dataset(service.dataset)


@router.get("/dataset/split-info")
def split_info(service: ModelService):
    split: PreparedDataset | None = service.split
    if split is None:
        raise HTTPException(status_code=503, detail="Dataset is empty")
    return split.split_info()


@router.get("/model/schema")
def model_schema(service: ModelService):
    return service.schema()


@router.post(
    "/model/train",
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def train_model(
    service: ModelService,
    config: TrainingConfigChurn | None = None,
):
    if not service.dataset or service.split is None:
        raise HTTPException(status_code=503, detail="Dataset is empty or unavailable")

    try:
        return service.train(config or TrainingConfigChurn())
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/model/metrics", response_model=MetricsHistoryResponse)
def model_metrics(
    service: ModelService,
    model_type: ModelTypeChurn | None = None,
    limit: int = Query(default=10, ge=1, le=100),
):
    return service.metrics(model_type, limit)


@router.get("/model/status")
def model_status(service: ModelService):
    return service.model_bundle.metadata
