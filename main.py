from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from core.config import DATASET_FILEPATH
from core.exceptions import register_exception_handlers
from core.logging import configure_logging
from ml.dataset import load_dataset
from ml.preprocessing import prepare_dataset
from ml.service import ChurnModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    dataset = load_dataset(DATASET_FILEPATH)
    split = prepare_dataset(dataset) if dataset else None
    app.state.service = ChurnModelService.create(dataset, split)
    yield


app = FastAPI(lifespan=lifespan)
register_exception_handlers(app)
app.include_router(router)
