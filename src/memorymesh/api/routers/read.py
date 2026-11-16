"""Read-path API routes for MemoryMesh.

Governed by docs/SPECIFICATION.md Section 7.2.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from memorymesh.api.deps import get_service
from memorymesh.models import (
    CatalogResponse,
    DecisionLogEntry,
    DisputeView,
    GraphResponse,
    HealthResponse,
    ImpactItem,
    MemoryStatus,
    MemoryView,
    QueryResult,
    SlotTimeline,
)
from memorymesh.service import MemoryService

router = APIRouter(tags=["read"])


def validate_tz_aware(dt: datetime | None, param_name: str = "now") -> datetime | None:
    """Validate that a datetime parameter is timezone-aware."""
    if dt is not None and (dt.tzinfo is None or dt.utcoffset() is None):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Timestamp parameter '{param_name}' must be timezone-aware "
                "(e.g. ISO 8601 with 'Z' or offset)."
            ),
        )
    return dt


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check and configuration status",
)
def get_health(
    service: Annotated[MemoryService, Depends(get_service)],
) -> HealthResponse:
    """Report service health, active LLM provider, and database connection status."""
    provider_str = (
        service.cfg.engine.llm.provider.value
        if hasattr(service.cfg.engine.llm.provider, "value")
        else str(service.cfg.engine.llm.provider)
    )
    db_target = ":memory:"
    try:
        with service._ingest_lock:
            db_rows = service.repo.conn.execute("PRAGMA database_list").fetchall()
            if db_rows and db_rows[0]["file"]:
                db_target = str(db_rows[0]["file"])
    except Exception:
        db_target = "connected"

    return HealthResponse(
        status="ok",
        llm_provider=provider_str,
        db=db_target,
    )


@router.get(
    "/catalog",
    response_model=CatalogResponse,
    status_code=status.HTTP_200_OK,
    summary="Configuration catalog for settings and metadata",
)
def get_catalog(
    service: Annotated[MemoryService, Depends(get_service)],
) -> CatalogResponse:
    """Retrieve system configuration catalog for sources, attributes, and entities."""
    return service.get_catalog()


@router.get(
    "/memories",
    response_model=list[MemoryView],
    status_code=status.HTTP_200_OK,
    summary="List memories matching filters",
)
def list_memories(
    service: Annotated[MemoryService, Depends(get_service)],
    entity: str | None = None,
    attribute: str | None = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    needs_review: bool | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    now: datetime | None = None,
) -> list[MemoryView]:
    """List memories formatted as presentation views."""
    validate_tz_aware(now, "now")
    normalized_status: str | None = None
    if status_filter is not None:
        normalized_status = status_filter.strip().lower()
        valid_statuses = {s.value for s in MemoryStatus}
        if normalized_status not in valid_statuses:
            status_list = sorted(valid_statuses)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid status '{status_filter}'. Must be one of: {status_list}",
            )

    return service.list_memory_views(
        entity=entity,
        attribute=attribute,
        status=normalized_status,
        needs_review=needs_review,
        limit=limit,
        offset=offset,
        now=now,
    )


@router.get(
    "/memories/{id}",
    response_model=MemoryView,
    status_code=status.HTTP_200_OK,
    summary="Get single memory view",
)
def get_memory(
    id: int,
    service: Annotated[MemoryService, Depends(get_service)],
    now: datetime | None = None,
) -> MemoryView:
    """Retrieve single memory view by ID or return 404."""
    validate_tz_aware(now, "now")
    return service.get_memory_view(memory_id=id, now=now)


@router.get(
    "/memories/{id}/explain",
    response_model=QueryResult,
    status_code=status.HTTP_200_OK,
    summary="Explain memory origin, provenance, and decisions",
)
def explain_memory(
    id: int,
    service: Annotated[MemoryService, Depends(get_service)],
) -> QueryResult:
    """Inspect persisted state, evidence, decisions, and graph edges for a memory."""
    return service.explain(memory_id=id)


@router.get(
    "/memories/{id}/impact",
    response_model=list[ImpactItem],
    status_code=status.HTTP_200_OK,
    summary="Calculate downstream impact of a memory change",
)
def get_memory_impact(
    id: int,
    service: Annotated[MemoryService, Depends(get_service)],
) -> list[ImpactItem]:
    """Traverse dependency graph and compute impacted downstream memories."""
    return service.impact(memory_id=id)


@router.get(
    "/timeline",
    response_model=list[SlotTimeline],
    status_code=status.HTTP_200_OK,
    summary="Get timeline grouped by entity and attribute slot",
)
def get_timeline(
    service: Annotated[MemoryService, Depends(get_service)],
    entity: str | None = None,
    attribute: str | None = None,
    now: datetime | None = None,
) -> list[SlotTimeline]:
    """Retrieve memories and rejected claims grouped into temporal slot lanes."""
    validate_tz_aware(now, "now")
    return service.get_timeline(entity=entity, attribute=attribute, now=now)


@router.get(
    "/graph",
    response_model=GraphResponse,
    status_code=status.HTTP_200_OK,
    summary="Get entity and memory dependency graph",
)
def get_graph(
    service: Annotated[MemoryService, Depends(get_service)],
    entity: str | None = None,
    now: datetime | None = None,
) -> GraphResponse:
    """Construct dependency and relationship graph for entities and memories."""
    validate_tz_aware(now, "now")
    return service.get_graph(entity=entity, now=now)


@router.get(
    "/disputes",
    response_model=list[DisputeView],
    status_code=status.HTTP_200_OK,
    summary="List active memory disputes",
)
def get_disputes(
    service: Annotated[MemoryService, Depends(get_service)],
    entity: str | None = None,
    now: datetime | None = None,
) -> list[DisputeView]:
    """Retrieve pairs of active memories in dispute."""
    validate_tz_aware(now, "now")
    return service.disputes(entity=entity, now=now)


@router.get(
    "/decisions",
    response_model=list[DecisionLogEntry],
    status_code=status.HTTP_200_OK,
    summary="List decision log history",
)
def get_decisions(
    service: Annotated[MemoryService, Depends(get_service)],
    rule_id: str | None = None,
    message_id: int | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DecisionLogEntry]:
    """Retrieve auditable decision log history chronologically."""
    return service.get_decision_logs(
        rule_id=rule_id,
        message_id=message_id,
        limit=limit,
        offset=offset,
    )
