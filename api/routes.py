from fastapi import APIRouter, HTTPException, Query, Request

from ml.dataset import info_dataset
from ml.preprocessing import PreparedDataset
from ml.service import ChurnModelService
from schemas import (
    ErrorResponse,
    FeatureVectorChurn,
    MetricsHistoryResponse,
    ModelTypeChurn,
    PredictionResponseChurn,
    TrainingConfigChurn,
)

router = APIRouter()


def get_service(request: Request) -> ChurnModelService:
    return request.app.state.service


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
    request: Request,
    features: FeatureVectorChurn | list[FeatureVectorChurn],
) -> PredictionResponseChurn | list[PredictionResponseChurn]:
    service = get_service(request)
    if not service.model_bundle.metadata.trained:
        raise HTTPException(status_code=503, detail="Model is not trained")
    return service.predict(features)


@router.get("/dataset/preview")
def dataset_preview(request: Request, n: int = 5):
    return get_service(request).dataset[:n]


@router.get("/dataset/info")
def dataset_info(request: Request):
    return info_dataset(get_service(request).dataset)


@router.get("/dataset/split-info")
def split_info(request: Request):
    split: PreparedDataset | None = get_service(request).split
    if split is None:
        raise HTTPException(status_code=503, detail="Dataset is empty")
    return split.split_info()


@router.get("/model/schema")
def model_schema(request: Request):
    return get_service(request).schema()


@router.post(
    "/model/train",
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def train_model(request: Request, config: TrainingConfigChurn | None = None):
    service = get_service(request)
    if not service.dataset or service.split is None:
        raise HTTPException(status_code=503, detail="Dataset is empty or unavailable")

    try:
        return service.train(config or TrainingConfigChurn())
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/model/metrics", response_model=MetricsHistoryResponse)
def model_metrics(
    request: Request,
    model_type: ModelTypeChurn | None = None,
    limit: int = Query(default=10, ge=1, le=100),
):
    return get_service(request).metrics(model_type, limit)


@router.get("/model/status")
def model_status(request: Request):
    return get_service(request).model_bundle.metadata
