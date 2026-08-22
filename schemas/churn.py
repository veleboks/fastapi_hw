from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "monthly_fee": 49.9,
                    "usage_hours": 120.0,
                    "support_requests": 2,
                    "account_age_months": 18,
                    "failed_payments": 0,
                    "region": "north",
                    "device_type": "mobile",
                    "payment_method": "card",
                    "autopay_enabled": 1,
                }
            ]
        },
    )

    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


class PredictionResponseChurn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "predicted_churn": 0,
                    "probabilities": {"0": 0.84, "1": 0.16},
                }
            ]
        }
    )

    predicted_churn: int
    probabilities: dict[str, float]


class ModelTypeChurn(str, Enum):
    LOGREG = "logreg"
    RANDOM_FOREST = "random_forest"


class TrainingConfigChurn(BaseModel):
    model_type: ModelTypeChurn = ModelTypeChurn.LOGREG
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class TrainingHistoryRecord(BaseModel):
    timestamp: datetime
    model_type: ModelTypeChurn
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]


class MetricsHistoryResponse(BaseModel):
    latest: TrainingHistoryRecord | None
    history: list[TrainingHistoryRecord]


class ChurnHealthStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"


class ChurnHealth(BaseModel):
    status: ChurnHealthStatus
    dataset_loaded: bool
    model_loaded: bool
