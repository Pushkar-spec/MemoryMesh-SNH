"""Question answering and baseline comparison API routes.

Governed by docs/SPECIFICATION.md Section 7.2.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from memorymesh.api.deps import get_service
from memorymesh.models import AskRequest, AskResponse
from memorymesh.service import MemoryService

router = APIRouter(tags=["ask"])


@router.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer natural language question with optional baseline comparison",
)
def ask_question(
    payload: AskRequest,
    service: Annotated[MemoryService, Depends(get_service)],
) -> AskResponse:
    """Answer a question grounded in temporal memories with optional naive RAG comparison."""
    return service.ask_with_compare(
        question=payload.question,
        now=payload.now,
        compare=payload.compare,
    )
