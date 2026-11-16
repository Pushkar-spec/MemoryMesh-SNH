"""Custom exception handlers enforcing uniform API error format.

Governed by docs/SPECIFICATION.md Section 7.1.
All errors follow: {"error": {"code": str, "message": str}}
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from memorymesh.errors import (
    AliasCollisionError,
    BaselineError,
    ConflictResolutionError,
    InvalidLifecycleTransitionError,
    LLMError,
    MemoryNotFoundError,
    NormalizationError,
    UnknownSourceError,
)

logger = logging.getLogger(__name__)


def make_error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """Construct a uniform JSON error response."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register uniform exception handlers on the FastAPI application."""

    @app.exception_handler(MemoryNotFoundError)
    async def handle_not_found(request: Request, exc: MemoryNotFoundError) -> JSONResponse:
        return make_error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="not_found",
            message=str(exc),
        )

    @app.exception_handler(UnknownSourceError)
    async def handle_unknown_source(request: Request, exc: UnknownSourceError) -> JSONResponse:
        return make_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="unknown_source",
            message=str(exc),
        )

    @app.exception_handler(NormalizationError)
    async def handle_normalization_error(request: Request, exc: NormalizationError) -> JSONResponse:
        return make_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="validation_error",
            message=str(exc),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first_err = exc.errors()[0] if exc.errors() else {}
        loc = " -> ".join(str(item) for item in first_err.get("loc", []))
        msg = first_err.get("msg", "Invalid request parameter or payload.")
        full_msg = f"{loc}: {msg}" if loc else msg
        return make_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="validation_error",
            message=full_msg,
        )

    @app.exception_handler(ValidationError)
    async def handle_pydantic_validation_error(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        first_err = exc.errors()[0] if exc.errors() else {}
        loc = " -> ".join(str(item) for item in first_err.get("loc", []))
        msg = first_err.get("msg", "Validation error.")
        full_msg = f"{loc}: {msg}" if loc else msg
        return make_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="validation_error",
            message=full_msg,
        )

    @app.exception_handler(ConflictResolutionError)
    async def handle_conflict(request: Request, exc: ConflictResolutionError) -> JSONResponse:
        return make_error_response(
            status_code=status.HTTP_409_CONFLICT,
            code="conflict",
            message=str(exc),
        )

    @app.exception_handler(InvalidLifecycleTransitionError)
    async def handle_lifecycle_conflict(
        request: Request, exc: InvalidLifecycleTransitionError
    ) -> JSONResponse:
        return make_error_response(
            status_code=status.HTTP_409_CONFLICT,
            code="conflict",
            message=str(exc),
        )

    @app.exception_handler(AliasCollisionError)
    async def handle_alias_collision(request: Request, exc: AliasCollisionError) -> JSONResponse:
        return make_error_response(
            status_code=status.HTTP_409_CONFLICT,
            code="conflict",
            message=str(exc),
        )

    @app.exception_handler(LLMError)
    async def handle_llm_error(request: Request, exc: LLMError) -> JSONResponse:
        logger.error("LLM provider failure: %s", exc)
        return make_error_response(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code="llm_unavailable",
            message=f"LLM provider error: {exc}",
        )

    @app.exception_handler(BaselineError)
    async def handle_baseline_error(request: Request, exc: BaselineError) -> JSONResponse:
        logger.error("Baseline provider failure: %s", exc)
        return make_error_response(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code="llm_unavailable",
            message=f"Baseline error: {exc}",
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code_map = {
            status.HTTP_404_NOT_FOUND: "not_found",
            status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
            status.HTTP_422_UNPROCESSABLE_ENTITY: "validation_error",
            status.HTTP_409_CONFLICT: "conflict",
            status.HTTP_502_BAD_GATEWAY: "llm_unavailable",
        }
        code = code_map.get(exc.status_code, "internal" if exc.status_code >= 500 else "error")
        msg = (
            "Internal server error."
            if exc.status_code >= 500
            else (str(exc.detail) if exc.detail else "HTTP error.")
        )
        return make_error_response(
            status_code=exc.status_code,
            code=code,
            message=msg,
        )

    @app.exception_handler(Exception)
    async def handle_internal_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server error: %s", exc)
        return make_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="internal",
            message="Internal server error.",
        )
