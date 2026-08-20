import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas import ErrorResponse

logger = logging.getLogger(__name__)


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
) -> JSONResponse:
    payload = ErrorResponse(code=code, message=message, details=details)
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _error_response(
        422,
        "VALIDATION_ERROR",
        "Request validation failed",
        [
            {
                "location": error["loc"],
                "message": error["msg"],
                "type": error["type"],
            }
            for error in exc.errors()
        ],
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, str):
        message, details = exc.detail, None
    else:
        message, details = "Request failed", exc.detail
    return _error_response(exc.status_code, f"HTTP_{exc.status_code}", message, details)


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.exception("Request processing failed")
    return _error_response(
        400, "PROCESSING_ERROR", "Unable to process the request", str(exc)
    )


async def unexpected_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    logger.exception("Unexpected server error")
    return _error_response(500, "INTERNAL_ERROR", "Internal server error")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(Exception, unexpected_exception_handler)
