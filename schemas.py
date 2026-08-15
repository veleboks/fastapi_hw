from pydantic import BaseModel, ConfigDict


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(
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
        }
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
                    "probabilities": {
                        "0": 0.84,
                        "1": 0.16,
                    },
                }
            ]
        }
    )

    predicted_churn: int
    probabilities: dict[str, float]
