import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_ml_service import make_dataset, make_service

from api.routes import router
from core.dependencies import get_model_service
from core.exceptions import register_exception_handlers

VALID_FEATURES = {
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


@pytest.fixture
def client(tmp_path):
    service = make_service(tmp_path, make_dataset())
    test_app = FastAPI()
    register_exception_handlers(test_app)
    test_app.include_router(router)
    test_app.dependency_overrides[get_model_service] = lambda: service

    with TestClient(test_app) as test_client:
        yield test_client


def test_training_status_and_prediction_flow(client):
    training_response = client.post(
        "/model/train",
        json={
            "model_type": "logreg",
            "hyperparameters": {"max_iter": 100},
        },
    )
    assert training_response.status_code == 200
    assert set(training_response.json()) == {"accuracy", "f1"}

    status_response = client.get("/model/status")
    assert status_response.status_code == 200
    assert status_response.json()["trained"] is True

    prediction_response = client.post(
        "/predict",
        json=VALID_FEATURES,
    )
    body = prediction_response.json()
    assert prediction_response.status_code == 200
    assert body["predicted_churn"] in {0, 1}
    assert set(body["probabilities"]) == {"0", "1"}


def test_prediction_without_training_returns_structured_error(client):
    response = client.post(
        "/predict",
        json=VALID_FEATURES,
    )

    assert response.status_code == 503
    assert response.json() == {
        "code": "HTTP_503",
        "message": "Model is not trained",
        "details": None,
    }


def test_predict_validation_error_has_common_format(client):
    response = client.post("/predict", json={"monthly_fee": "not-a-number"})

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"
    assert response.json()["message"] == "Request validation failed"
    assert response.json()["details"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("region", "moon"),
        ("device_type", "fridge"),
        ("payment_method", "cash"),
        ("autopay_enabled", 2),
    ],
)
def test_predict_rejects_invalid_domain_values(client, field, value):
    payload = {**VALID_FEATURES, field: value}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_model_schema_is_available(client):
    response = client.get("/model/schema")
    body = response.json()

    assert response.status_code == 200
    assert len(body["numeric"]) == 6
    assert len(body["categorical"]) == 3
