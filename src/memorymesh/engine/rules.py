"""Deterministic resolution engine rules (R-INSERT, R-REPLACE).

Governed by docs/RESOLUTION_ENGINE.md, docs/SPECIFICATION.md Section 1.1, and docs/AGENTS.md.
All functions in this module are pure functions: inputs in, Decision out.
No database access, no network calls, and no datetime.now() invocations.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from memorymesh.config import EngineConfig
from memorymesh.errors import (
    ConflictResolutionError,
    InvalidValidityWindowError,
    UnsupportedRelationError,
)
from memorymesh.models import (
    Action,
    AddEdge,
    AttributeDef,
    Cardinality,
    Decision,
    EdgeType,
    InsertMemory,
    Memory,
    MemoryStatus,
    Mutation,
    NewEvidence,
    NewMemory,
    Relation,
    ResolvedClaim,
    UpdateMemory,
    format_datetime_utc,
)


def covers(m: Memory, t: datetime) -> bool:
    """Determine whether a memory covers a point in time t.

    Per docs/RESOLUTION_ENGINE.md Section 1:
    m.valid_from <= t and (m.valid_to is null or t < m.valid_to)
    """
    if m.valid_from > t:
        return False
    if m.valid_to is not None and t >= m.valid_to:
        return False
    return True


def resolve(
    rc: ResolvedClaim,
    chain: list[Memory],
    schema: AttributeDef,
    cfg: EngineConfig,
    classify: Callable[[ResolvedClaim, Memory], Relation],
    dep_targets: list[int],
    recorded_at: datetime,
) -> Decision:
    """Pure resolution function evaluating deterministic rules on candidate memories.

    In P1.1, implements:
    - R-INSERT: First claim in slot, or open-ended after closed windows.
    - R-REPLACE: Newer conflicting claim replaces covering live memory on single-valued attribute.

    Args:
        rc: The resolved claim to evaluate.
        chain: Non-rejected memories for (entity, attribute, context_key), ordered by valid_from.
        schema: Attribute catalog definition.
        cfg: Engine configuration parameters.
        classify: Callable returning the relation between claim and memory.
        dep_targets: List of memory IDs this claim depends on.
        recorded_at: Ingestion timestamp used for recorded_at and closed_at.

    Returns:
        A deterministic Decision object containing mutations and affected memory IDs.
    """
    # 1. Validate validity-window inputs
    if rc.valid_to is not None and rc.valid_to <= rc.valid_from:
        raise InvalidValidityWindowError(
            f"valid_to must be greater than valid_from: valid_to ({rc.valid_to}) <= "
            f"valid_from ({rc.valid_from})."
        )

    # 2. Filter candidate memories covering rc.valid_from
    cover = [m for m in chain if covers(m, rc.valid_from)]
    live = [m for m in cover if m.status in (MemoryStatus.ACTIVE, MemoryStatus.DISPUTED)]

    # 3. R-INSERT: Empty chain or no covering live memories with no later memories
    if not live:
        later = next((m for m in chain if m.valid_from > rc.valid_from), None)
        if later is None:
            return _produce_r_insert(
                rc=rc,
                dep_targets=dep_targets,
                recorded_at=recorded_at,
            )
        raise ConflictResolutionError(
            "Cannot replace: claim is not newer than existing memory (prior to later fact). "
            "Backfilling into an earlier gap is deferred to Phase 6 (R-BACKFILL)."
        )

    # 4. Enforce attribute cardinality
    if schema.cardinality != Cardinality.SINGLE:
        raise ConflictResolutionError(
            f"Multi-valued attribute cardinality ('{schema.cardinality.value}') cannot "
            "trigger R-REPLACE; multi-valued resolution is deferred to Phase 2 (R-COEXIST)."
        )

    # 5. Single-valued conflict check
    for m in live:
        rel = classify(rc, m)
        if rel == Relation.CONFLICTS:
            # Verify claim is strictly newer than all live memories being replaced
            newer = all(rc.valid_from > mem.valid_from for mem in live)
            if not newer:
                raise ConflictResolutionError(
                    "Cannot replace: claim is not newer than existing memory (equal or earlier "
                    "valid_from). Non-newer conflicts require R-DISPUTE/R-REJECT (deferred to P3)."
                )

            return _produce_r_replace(
                rc=rc,
                live=live,
                dep_targets=dep_targets,
                recorded_at=recorded_at,
            )

        if rel == Relation.IDENTICAL:
            raise UnsupportedRelationError(
                f"Claim matches existing memory value '{m.value_norm}' (Relation.IDENTICAL). "
                "Corroborating existing memories is deferred to Phase 3 (R-CORROBORATE)."
            )

        rel_str = rel.value if hasattr(rel, "value") else str(rel)
        raise UnsupportedRelationError(
            f"Relation '{rel_str}' is not supported in Phase 1 (deferred to later phases)."
        )

    raise ConflictResolutionError("No applicable rule matched candidate memories.")


def _produce_r_insert(
    rc: ResolvedClaim,
    dep_targets: list[int],
    recorded_at: datetime,
) -> Decision:
    """Build an R-INSERT Decision."""
    new_mem = NewMemory(
        entity_id=rc.entity_id,
        attribute=rc.attribute,
        value_norm=rc.value_norm,
        value_display=rc.value_display,
        context=rc.claim.context,
        context_key=rc.context_key,
        valid_from=rc.valid_from,
        valid_to=rc.valid_to,
        closed_at=recorded_at if rc.valid_to is not None else None,
        status=MemoryStatus.ACTIVE,
        confidence=1.0,  # Hard-coded trust 1.0 for P1.1
        raw_text=rc.claim.span or "",
        message_id=rc.message_id,
        source_id=rc.source.id,
    )
    new_evidence = NewEvidence(
        source_id=rc.source.id,
        message_id=rc.message_id,
        strength=1.0,
        observed_at=recorded_at,
        span=rc.claim.span,
    )

    mutations: list[Mutation] = [InsertMemory(memory=new_mem, evidence=new_evidence)]

    for target_id in dep_targets:
        mutations.append(
            AddEdge(
                src="$new",
                dst=target_id,
                type=EdgeType.DEPENDS_ON,
                reason="claim dependency",
            )
        )

    return Decision(
        action=Action.INSERT,
        rule_id="R-INSERT",
        explanation=(
            f"Initial claim for {rc.claim.subject}.{rc.attribute} (no existing candidate memory)."
        ),
        scores={"new_score": 1.0},
        mutations=mutations,
        affected_memory_ids=[],
    )


def _produce_r_replace(
    rc: ResolvedClaim,
    live: list[Memory],
    dep_targets: list[int],
    recorded_at: datetime,
) -> Decision:
    """Build an R-REPLACE Decision closing existing memory windows at rc.valid_from."""
    new_mem = NewMemory(
        entity_id=rc.entity_id,
        attribute=rc.attribute,
        value_norm=rc.value_norm,
        value_display=rc.value_display,
        context=rc.claim.context,
        context_key=rc.context_key,
        valid_from=rc.valid_from,
        valid_to=rc.valid_to,
        closed_at=recorded_at if rc.valid_to is not None else None,
        status=MemoryStatus.ACTIVE,
        confidence=1.0,
        raw_text=rc.claim.span or "",
        message_id=rc.message_id,
        source_id=rc.source.id,
    )
    new_evidence = NewEvidence(
        source_id=rc.source.id,
        message_id=rc.message_id,
        strength=1.0,
        observed_at=recorded_at,
        span=rc.claim.span,
    )

    mutations: list[Mutation] = [InsertMemory(memory=new_mem, evidence=new_evidence)]
    affected_ids: list[int] = []

    for old_m in live:
        affected_ids.append(old_m.id)
        mutations.append(
            UpdateMemory(
                memory_id=old_m.id,
                fields={
                    "status": MemoryStatus.HISTORICAL,
                    "valid_to": rc.valid_from,
                    "closed_at": recorded_at,
                },
            )
        )
        mutations.append(
            AddEdge(
                src="$new",
                dst=old_m.id,
                type=EdgeType.SUPERSEDES,
                reason=f"superseded by newer claim at {format_datetime_utc(rc.valid_from)}",
            )
        )

    for target_id in dep_targets:
        mutations.append(
            AddEdge(
                src="$new",
                dst=target_id,
                type=EdgeType.DEPENDS_ON,
                reason="claim dependency",
            )
        )

    live_ids_str = ", ".join(f"M{m_id}" for m_id in affected_ids)
    explanation = (
        f"Newer conflicting claim replaces existing memory {live_ids_str} "
        f"on single-valued attribute '{rc.attribute}'."
    )

    return Decision(
        action=Action.REPLACE,
        rule_id="R-REPLACE",
        explanation=explanation,
        scores={"new_score": 1.0, "old_score": 1.0},
        mutations=mutations,
        affected_memory_ids=affected_ids,
    )
