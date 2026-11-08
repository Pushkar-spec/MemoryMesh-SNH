"""Deterministic resolution engine rules (R-INSERT, R-REPLACE, R-COEXIST, R-REFINE).

Governed by docs/SPECIFICATION.md Section 1.1, docs/ARCHITECTURE.md, and docs/AGENTS.md.
All functions in this module are pure functions: inputs in, Decision out.
No database access, no network calls, and no datetime.now() invocations.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from memorymesh.config import EngineConfig
from memorymesh.engine.confidence import calculate_base_confidence, calculate_strength
from memorymesh.errors import (
    ConflictResolutionError,
    InvalidValidityWindowError,
    UnsupportedRelationError,
)
from memorymesh.models import (
    Action,
    AddEdge,
    AddEvidence,
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

    # Calculate claim strength
    s_new = (
        calculate_strength(rc.source.reliability, rc.claim.extract_confidence)
        if cfg.ablation.use_trust
        else 1.0
    )
    s_new = min(cfg.thresholds.max_confidence, s_new)

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
                s_new=s_new,
            )
        raise ConflictResolutionError(
            "Cannot replace: claim is not newer than existing memory (prior to later fact). "
            "Backfilling into an earlier gap is deferred to Phase 6 (R-BACKFILL)."
        )

    # 4. Multi-valued attribute resolution (R-COEXIST, R-CORROBORATE)
    if schema.cardinality == Cardinality.MULTI:
        for m in live:
            rel = classify(rc, m)
            if rel == Relation.IDENTICAL:
                return _produce_r_corroborate_or_dup(
                    rc=rc,
                    target=m,
                    s_new=s_new,
                    cfg=cfg,
                    recorded_at=recorded_at,
                )

        # Distinct values on multi-valued attributes coexist independently
        return _produce_r_coexist(
            rc=rc,
            live=live,
            dep_targets=dep_targets,
            recorded_at=recorded_at,
            s_new=s_new,
        )

    # 5. Single-valued resolution (R-CORROBORATE, R-REPLACE, R-REFINE)
    for m in live:
        rel = classify(rc, m)
        if rel == Relation.IDENTICAL:
            return _produce_r_corroborate_or_dup(
                rc=rc,
                target=m,
                s_new=s_new,
                cfg=cfg,
                recorded_at=recorded_at,
            )

        if rel == Relation.REFINES:
            # Verify claim is strictly newer than all live memories being refined
            newer = all(rc.valid_from > mem.valid_from for mem in live)
            if not newer:
                raise ConflictResolutionError(
                    "Cannot refine: claim is not newer than existing memory (equal or earlier "
                    "valid_from)."
                )

            return _produce_r_refine(
                rc=rc,
                live=live,
                dep_targets=dep_targets,
                recorded_at=recorded_at,
                s_new=s_new,
            )

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
                s_new=s_new,
            )

        rel_str = rel.value if hasattr(rel, "value") else str(rel)
        raise UnsupportedRelationError(
            f"Relation '{rel_str}' is not supported (deferred to later phases)."
        )

    raise ConflictResolutionError("No applicable rule matched candidate memories.")


def _produce_r_corroborate_or_dup(
    rc: ResolvedClaim,
    target: Memory,
    s_new: float,
    cfg: EngineConfig,
    recorded_at: datetime,
) -> Decision:
    """Build an R-CORROBORATE or R-DUP-SAME-SOURCE decision for an identical claim."""
    new_ev = NewEvidence(
        source_id=rc.source.id,
        message_id=rc.message_id,
        strength=s_new,
        observed_at=recorded_at,
        span=rc.claim.span or "",
    )

    # Collect distinct sources that have already provided evidence for this memory
    existing_sources = {e.source_id for e in target.evidence}
    if target.source_id:
        existing_sources.add(target.source_id)

    old_conf = float(target.confidence)

    if rc.source.id in existing_sources:
        # Same source: a source cannot corroborate itself.
        # Evidence is recorded, but confidence remains unchanged.
        mutations: list[Mutation] = [
            AddEvidence(memory_id=target.id, evidence=new_ev),
        ]
        explanation = (
            f"Claim value '{rc.value_norm}' from same source '{rc.source.id}' "
            f"corroborates existing memory M{target.id} without confidence boost."
        )
        return Decision(
            action=Action.CORROBORATE,
            rule_id="R-DUP-SAME-SOURCE",
            explanation=explanation,
            scores={"new_score": s_new, "old_score": old_conf, "confidence": old_conf},
            mutations=mutations,
            affected_memory_ids=[target.id],
        )

    # Different source: corroboration raises confidence via noisy-OR aggregation
    existing_ev: list[Any] = list(target.evidence)
    if not existing_ev and target.source_id:
        existing_ev.append(
            NewEvidence(
                source_id=target.source_id,
                message_id=target.message_id,
                strength=old_conf,
                observed_at=target.recorded_at,
                span=target.raw_text,
            )
        )

    all_evidence = existing_ev + [new_ev]
    new_conf = calculate_base_confidence(
        all_evidence,
        max_confidence=cfg.thresholds.max_confidence,
    )

    mutations = [
        AddEvidence(memory_id=target.id, evidence=new_ev),
        UpdateMemory(memory_id=target.id, fields={"confidence": new_conf}),
    ]
    explanation = (
        f"Claim value '{rc.value_norm}' from source '{rc.source.id}' "
        f"corroborates existing memory M{target.id} (confidence {old_conf:.3f} -> {new_conf:.3f})."
    )
    return Decision(
        action=Action.CORROBORATE,
        rule_id="R-CORROBORATE",
        explanation=explanation,
        scores={"new_score": s_new, "old_score": old_conf, "confidence": new_conf},
        mutations=mutations,
        affected_memory_ids=[target.id],
    )


def _produce_r_insert(
    rc: ResolvedClaim,
    dep_targets: list[int],
    recorded_at: datetime,
    s_new: float = 1.0,
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
        confidence=s_new,
        raw_text=rc.claim.span or "",
        message_id=rc.message_id,
        source_id=rc.source.id,
    )
    new_evidence = NewEvidence(
        source_id=rc.source.id,
        message_id=rc.message_id,
        strength=s_new,
        observed_at=recorded_at,
        span=rc.claim.span or "",
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
        scores={"new_score": s_new},
        mutations=mutations,
        affected_memory_ids=[],
    )


def _produce_r_replace(
    rc: ResolvedClaim,
    live: list[Memory],
    dep_targets: list[int],
    recorded_at: datetime,
    s_new: float = 1.0,
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
        confidence=s_new,
        raw_text=rc.claim.span or "",
        message_id=rc.message_id,
        source_id=rc.source.id,
    )
    new_evidence = NewEvidence(
        source_id=rc.source.id,
        message_id=rc.message_id,
        strength=s_new,
        observed_at=recorded_at,
        span=rc.claim.span or "",
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
        scores={"new_score": s_new, "old_score": 1.0},
        mutations=mutations,
        affected_memory_ids=affected_ids,
    )


def _produce_r_coexist(
    rc: ResolvedClaim,
    live: list[Memory],
    dep_targets: list[int],
    recorded_at: datetime,
    s_new: float = 1.0,
) -> Decision:
    """Build an R-COEXIST Decision inserting a new memory on a multi-valued attribute."""
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
        confidence=s_new,
        raw_text=rc.claim.span or "",
        message_id=rc.message_id,
        source_id=rc.source.id,
    )
    new_evidence = NewEvidence(
        source_id=rc.source.id,
        message_id=rc.message_id,
        strength=s_new,
        observed_at=recorded_at,
        span=rc.claim.span or "",
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

    coexisting_ids = [m.id for m in live]
    explanation = (
        f"Claim value '{rc.value_norm}' coexists with existing live memories "
        f"on multi-valued attribute '{rc.attribute}'."
    )

    return Decision(
        action=Action.COEXIST,
        rule_id="R-COEXIST",
        explanation=explanation,
        scores={"new_score": s_new},
        mutations=mutations,
        affected_memory_ids=coexisting_ids,
    )


def _produce_r_refine(
    rc: ResolvedClaim,
    live: list[Memory],
    dep_targets: list[int],
    recorded_at: datetime,
    s_new: float = 1.0,
) -> Decision:
    """Build an R-REFINE Decision superseding coarser live memories with a more specific value."""
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
        confidence=s_new,
        raw_text=rc.claim.span or "",
        message_id=rc.message_id,
        source_id=rc.source.id,
    )
    new_evidence = NewEvidence(
        source_id=rc.source.id,
        message_id=rc.message_id,
        strength=s_new,
        observed_at=recorded_at,
        span=rc.claim.span or "",
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
                type=EdgeType.REFINES,
                reason=(
                    f"refines coarser memory M{old_m.id} ('{old_m.value_norm}') "
                    f"with '{rc.value_norm}'"
                ),
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
        f"Newer more-specific claim refines existing memory {live_ids_str} "
        f"on attribute '{rc.attribute}'."
    )

    return Decision(
        action=Action.REFINE,
        rule_id="R-REFINE",
        explanation=explanation,
        scores={"new_score": s_new, "old_score": 1.0},
        mutations=mutations,
        affected_memory_ids=affected_ids,
    )
