from datetime import datetime
from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RegionChurn(str, Enum):
    EUROPE = "europe"
    ASIA = "asia"
    AMERICA = "america"
    AFRICA = "africa"


class DeviceTypeChurn(str, Enum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"


class PaymentMethodChurn(str, Enum):
    CARD = "card"
    PAYPAL = "paypal"
    CRYPTO = "crypto"


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
                    "region": "europe",
                    "device_type": "mobile",
                    "payment_method": "card",
                    "autopay_enabled": 1,
                }
            ]
        },
    )

    monthly_fee: float | None
    usage_hours: float | None
    support_requests: int | None
    account_age_months: int | None
    failed_payments: int | None
    region: RegionChurn | None
    device_type: DeviceTypeChurn | None
    payment_method: PaymentMethodChurn | None
    autopay_enabled: int | None = Field(ge=0, le=1)

    @model_validator(mode="after")
    def require_at_least_one_feature(self) -> Self:
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("At least one feature must be provided")
        return self


class DatasetRowChurn(BaseModel):
    monthly_fee: float | None = None
    usage_hours: float | None = None
    support_requests: int | None = None
    account_age_months: int | None = None
    failed_payments: int | None = None
    region: RegionChurn | None = None
    device_type: DeviceTypeChurn | None = None
    payment_method: PaymentMethodChurn | None = None
    autopay_enabled: int | None = Field(default=None, ge=0, le=1)
    churn: int = Field(ge=0, le=1)


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
