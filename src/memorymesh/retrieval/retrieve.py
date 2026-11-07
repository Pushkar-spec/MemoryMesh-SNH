"""Temporal retrieval module for MemoryMesh.

Governed by docs/SPECIFICATION.md Section 4.1, docs/ARCHITECTURE.md, and docs/AGENTS.md.
Implements the half-open validity interval: valid_from <= requested_time < valid_to.
"""

from __future__ import annotations

from datetime import datetime

from memorymesh.models import Memory
from memorymesh.repository import Repository


def build_context_key(context: dict[str, str] | None) -> str:
    """Build canonical context_key 'k=v;k=v' sorted, or '' for empty/global context."""
    if not context:
        return ""
    return ";".join(f"{k}={context[k]}" for k in sorted(context.keys()))


def retrieve_valid_memories(
    repo: Repository,
    entity_id: int,
    attribute: str,
    *,
    valid_at: datetime,
    known_at: datetime | None = None,
    context: dict[str, str] | None = None,
    include_rejected: bool = False,
) -> list[Memory]:
    """Retrieve all memories valid at valid_at using the half-open interval.

    Rule: valid_from <= valid_at < valid_to (or valid_to IS NULL).
    Returns memories ordered by valid_from DESC, id DESC.
    """
    ctx_key = build_context_key(context)
    return repo.find_memories(
        entity_id=entity_id,
        attribute=attribute,
        valid_at=valid_at,
        known_at=known_at,
        context_key=ctx_key,
        include_rejected=include_rejected,
    )


def retrieve_temporal_memory(
    repo: Repository,
    entity_id: int,
    attribute: str,
    *,
    valid_at: datetime,
    known_at: datetime | None = None,
    context: dict[str, str] | None = None,
) -> Memory | None:
    """Retrieve the single most recent memory valid at valid_at, or None if empty."""
    memories = retrieve_valid_memories(
        repo=repo,
        entity_id=entity_id,
        attribute=attribute,
        valid_at=valid_at,
        known_at=known_at,
        context=context,
        include_rejected=False,
    )
    return memories[0] if memories else None


def retrieve_current(
    repo: Repository,
    entity_id: int,
    attribute: str,
    *,
    now: datetime,
    context: dict[str, str] | None = None,
) -> Memory | None:
    """Retrieve the current active memory for (entity_id, attribute, context) at timestamp now."""
    return retrieve_temporal_memory(
        repo=repo,
        entity_id=entity_id,
        attribute=attribute,
        valid_at=now,
        known_at=now,
        context=context,
    )


def retrieve_as_of(
    repo: Repository,
    entity_id: int,
    attribute: str,
    *,
    valid_at: datetime,
    context: dict[str, str] | None = None,
    known_at: datetime | None = None,
) -> Memory | None:
    """Retrieve memory valid as of historical timestamp valid_at (defaults known_at to valid_at)."""
    effective_known_at = valid_at if known_at is None else known_at
    return retrieve_temporal_memory(
        repo=repo,
        entity_id=entity_id,
        attribute=attribute,
        valid_at=valid_at,
        known_at=effective_known_at,
        context=context,
    )
