from typing import Annotated

from fastapi import Depends, Request

from ml.service import ChurnModelService


def get_model_service(request: Request) -> ChurnModelService:
    return request.app.state.service


ModelService = Annotated[ChurnModelService, Depends(get_model_service)]
