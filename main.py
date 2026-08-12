from fastapi import FastAPI
from schemas import FeatureVectorChurn, DatasetRowChurn
from contextlib import asynccontextmanager
from dataset import load_dataset, info_dataset
from pathlib import Path
from typing import Any


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.dataset = load_dataset(Path("data/churn_dataset.csv"))
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root() -> dict:
    return {"message": "ml churn service is running"}


@app.post("/predict")
def predict(features: FeatureVectorChurn) -> FeatureVectorChurn:
    return features


@app.get("/dataset/preview")
def preview(n: int = 5) -> list[DatasetRowChurn]:
    return app.state.dataset[:n]


@app.get("/dataset/info")
def info() -> dict[str, Any]:
    return info_dataset(app.state.dataset)
