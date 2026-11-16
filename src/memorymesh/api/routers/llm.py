"""LLM call auditing and latency tracking API routes.

Governed by docs/SPECIFICATION.md Section 7.2.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from memorymesh.api.deps import get_service
from memorymesh.models import LLMCallLog
from memorymesh.service import MemoryService

router = APIRouter(prefix="/llm", tags=["llm"])


@router.get(
    "/calls",
    response_model=list[LLMCallLog],
    status_code=status.HTTP_200_OK,
    summary="List logged LLM provider invocations",
)
def get_llm_calls(
    service: Annotated[MemoryService, Depends(get_service)],
    prompt_id: str | None = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[LLMCallLog]:
    """Retrieve logged LLM calls with latency, prompt summary, and parsed output."""
    return service.get_llm_call_logs(
        prompt_id=prompt_id,
        status=status_filter,
        limit=limit,
        offset=offset,
    )
