"""
Custom exceptions and FastAPI exception handlers for the Score API.

All handlers return ErrorResponse JSON — no raw FastAPI/Pydantic formats
are ever exposed to callers.

register_exception_handlers(app) is called once in create_app().
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .schemas import ErrorResponse

logger = logging.getLogger(__name__)


# ── Custom exceptions ─────────────────────────────────────────────────────────

class RepositoryConnectionError(Exception):
    """Raised by the repository when a DB connection fails or times out."""


class UserNotFoundError(Exception):
    """Raised by the score router when no prediction exists for a user."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        super().__init__(f"No prediction found for user_id='{user_id}'")


# ── Exception handlers ────────────────────────────────────────────────────────

def register_exception_handlers(app: FastAPI) -> None:
    """Register all custom exception handlers. Called once in create_app()."""

    @app.exception_handler(UserNotFoundError)
    async def user_not_found_handler(request: Request, exc: UserNotFoundError) -> JSONResponse:
        logger.info("user_not_found user_id=%s path=%s", exc.user_id, request.url.path)
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="user_not_found",
                message=str(exc),
                field="user_id",
            ).model_dump(),
        )

    @app.exception_handler(RepositoryConnectionError)
    async def db_error_handler(request: Request, exc: RepositoryConnectionError) -> JSONResponse:
        logger.error("database_unavailable path=%s", request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="database_unavailable",
                message="Could not connect to the prediction store. Retry shortly.",
                retry_after_seconds=30,
            ).model_dump(),
            headers={"Retry-After": "30"},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first_error = exc.errors()[0]
        field = ".".join(str(x) for x in first_error["loc"] if x != "body")
        logger.warning("validation_error field=%s path=%s", field, request.url.path)
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error",
                message=first_error["msg"],
                field=field,
            ).model_dump(),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        logger.warning("invalid_request_value path=%s msg=%s", request.url.path, exc)
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="invalid_date",
                message=str(exc),
                field="date",
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception path=%s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error",
                message="An unexpected error occurred. Contact the ML engineering team.",
            ).model_dump(),
        )
