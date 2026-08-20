from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from core.bootstrap import create_model_service
from core.exceptions import register_exception_handlers
from core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.service = create_model_service()
    yield


app = FastAPI(lifespan=lifespan)
register_exception_handlers(app)
app.include_router(router)
