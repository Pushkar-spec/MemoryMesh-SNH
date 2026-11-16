"""Write-path API routes for MemoryMesh.

Governed by docs/SPECIFICATION.md Section 7.2.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from memorymesh.api.deps import get_service
from memorymesh.errors import MemoryNotFoundError
from memorymesh.models import (
    DecisionLogEntry,
    IngestMessage,
    IngestReport,
    IngestRequest,
    Memory,
    ResetResponse,
    ResolveDisputeRequest,
)
from memorymesh.service import MemoryService

router = APIRouter(tags=["write"])


@router.post(
    "/ingest",
    response_model=IngestReport,
    status_code=status.HTTP_200_OK,
    summary="Ingest message and resolve claims",
)
def ingest_message(
    payload: IngestRequest,
    service: Annotated[MemoryService, Depends(get_service)],
) -> IngestReport:
    """Ingest a message and extract/resolve memory claims."""
    ingest_msg = IngestMessage(
        source_id=payload.source_id,
        event_time=payload.event_time,
        text=payload.text,
    )
    if payload.mode == "structured" and payload.claims is not None:
        return service.ingest_claims(
            message=ingest_msg,
            claims=payload.claims,
            recorded_at=payload.recorded_at,
        )
    return service.ingest(
        message=ingest_msg,
        recorded_at=payload.recorded_at,
    )


@router.post(
    "/memories/{id}/acknowledge",
    response_model=Memory,
    status_code=status.HTTP_200_OK,
    summary="Acknowledge flagged memory review",
)
def acknowledge_memory_review(
    id: int,
    service: Annotated[MemoryService, Depends(get_service)],
) -> Memory:
    """Clear review flag on a memory and log acknowledgement."""
    with service._ingest_lock:
        service.acknowledge_review(memory_id=id)
        mem = service.repo.get_memory(id)
        if mem is None:
            raise MemoryNotFoundError(f"Memory with id={id} not found.")
        return mem


@router.post(
    "/disputes/resolve",
    response_model=DecisionLogEntry,
    status_code=status.HTTP_200_OK,
    summary="Resolve a memory dispute",
)
def resolve_dispute(
    payload: ResolveDisputeRequest,
    service: Annotated[MemoryService, Depends(get_service)],
) -> DecisionLogEntry:
    """Resolve an active dispute group by selecting a winner memory."""
    return service.resolve_dispute(winner_memory_id=payload.winner_memory_id)


@router.post(
    "/demo/reset",
    response_model=ResetResponse,
    status_code=status.HTTP_200_OK,
    summary="Reset database for demo",
)
def reset_demo(
    service: Annotated[MemoryService, Depends(get_service)],
) -> ResetResponse:
    """Recreate clean schema and reset in-memory state."""
    service.reset_database()
    return ResetResponse(ok=True)
