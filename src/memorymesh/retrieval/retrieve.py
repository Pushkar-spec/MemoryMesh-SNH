"""Temporal retrieval module for MemoryMesh.

Governed by docs/SPECIFICATION.md Section 4.1, docs/ARCHITECTURE.md, and docs/AGENTS.md.
Implements the half-open validity interval: valid_from <= requested_time < valid_to.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from memorymesh.models import Memory
from memorymesh.repository import Repository


def normalize_context(context: dict[str, str] | None) -> dict[str, str]:
    """Normalize context dictionary: strip whitespace, reject blank keys/values and duplicate keys.

    Context keys and values are strictly case-sensitive. Keys and values preserve their casing,
    while leading and trailing whitespace is stripped. Two keys that are identical after
    whitespace stripping (e.g. 'env' and ' env ') collide and raise ValueError. Keys differing
    in casing (e.g. 'env' and 'Env') are distinct keys and do not collide.

    Raises:
        ValueError: If a key or value is empty/whitespace, or if normalized keys collide.
    """
    if not context:
        return {}
    cleaned: dict[str, str] = {}
    seen_raw: dict[str, str] = {}
    for raw_k, raw_v in context.items():
        k_str = str(raw_k).strip()
        v_str = str(raw_v).strip()
        if not k_str:
            raise ValueError("Context key cannot be empty or whitespace.")
        if not v_str:
            raise ValueError(f"Context value for key '{k_str}' cannot be empty or whitespace.")
        if k_str in cleaned:
            raise ValueError(
                f"Duplicate context key after whitespace normalization: '{k_str}' "
                f"(from '{raw_k}' and '{seen_raw[k_str]}')"
            )
        cleaned[k_str] = v_str
        seen_raw[k_str] = str(raw_k)
    return cleaned


def build_context_key(context: dict[str, str] | None) -> str:
    """Build canonical collision-safe context_key via deterministic JSON serialization.

    - Normalizes whitespace on keys and values.
    - Rejects empty keys, empty values, and duplicate keys.
    - Serializes via compact sorted canonical JSON (no delimiter ambiguity).
    - Returns '' for empty/global context to preserve default global scope behavior.
    """
    cleaned = normalize_context(context)
    if not cleaned:
        return ""
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def match_context_scope(
    memory_context: dict[str, str] | None,
    query_context: dict[str, str] | None,
) -> bool:
    """Determine whether memory context C_m matches query context C_q (C_m ⊆ C_q).

    Per docs/RESOLUTION_ENGINE.md Section 6.1:
    A memory with context C_m matches a query with context C_q if C_m ⊆ C_q
    (every key/value of the memory appears in the query). Global memories ({})
    match every query.
    """
    m_ctx = memory_context or {}
    q_ctx = query_context or {}
    for k, v in m_ctx.items():
        if q_ctx.get(k) != v:
            return False
    return True


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
    Supports context scope matching and specificity fallback:
    - Matches memories where memory.context ⊆ query_context.
    - Selects memories from the highest specificity group (len(memory.context)).
    - Returns memories ordered deterministically: context_key ASC, valid_from DESC, id ASC.
    """
    if known_at is not None:
        if known_at.tzinfo is None or known_at.utcoffset() is None:
            raise ValueError("known_at must be a timezone-aware datetime.")
        known_at = known_at.astimezone(UTC)

    all_candidates = repo.find_memories(
        entity_id=entity_id,
        attribute=attribute,
        valid_at=valid_at,
        known_at=known_at,
        context_key=None,
        include_rejected=include_rejected,
    )

    q_ctx = normalize_context(context) if context else {}
    matching = [m for m in all_candidates if match_context_scope(m.context, q_ctx)]
    if not matching:
        return []

    max_spec = max(len(m.context or {}) for m in matching)
    selected = [m for m in matching if len(m.context or {}) == max_spec]
    return sorted(selected, key=lambda m: (m.context_key, -m.valid_from.timestamp(), m.id or 0))


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
    if known_at is not None:
        if known_at.tzinfo is None or known_at.utcoffset() is None:
            raise ValueError("known_at must be a timezone-aware datetime.")
        effective_known_at = known_at.astimezone(UTC)
    else:
        effective_known_at = valid_at
    return retrieve_temporal_memory(
        repo=repo,
        entity_id=entity_id,
        attribute=attribute,
        valid_at=valid_at,
        known_at=effective_known_at,
        context=context,
    )
