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
from memorymesh.engine.confidence import (
    calculate_base_confidence,
    calculate_decay,
    calculate_effective_confidence,
    calculate_strength,
)
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
    Polarity,
    Relation,
    ResolvedClaim,
    UpdateMemory,
    format_datetime_utc,
)


def _compute_memory_score(
    m: Memory,
    valid_at: datetime,
    schema: AttributeDef,
    cfg: EngineConfig,
) -> float:
    """Calculate comparable memory score at a given reference time."""
    if not cfg.ablation.use_trust:
        return 1.0
    decay = (
        calculate_decay(m.valid_from, valid_at, schema.half_life_days)
        if cfg.ablation.use_temporal
        else 1.0
    )
    return calculate_effective_confidence(
        m.confidence,
        decay_factor=decay,
        needs_review=m.needs_review,
        review_penalty=cfg.thresholds.review_penalty,
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
    superseded_ancestor_values: set[str] | None = None,
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
        superseded_ancestor_values: Historical values superseded by live memories in this slot.

    Returns:
        A deterministic Decision object containing mutations and affected memory IDs.
    """
    # 1. Validate validity-window inputs
    if rc.valid_to is not None and rc.valid_to <= rc.valid_from:
        raise InvalidValidityWindowError(
            f"valid_to must be greater than valid_from: valid_to ({rc.valid_to}) <= "
            f"valid_from ({rc.valid_from})."
        )

    # Low-confidence extraction skip (R-SKIP-LOWCONF)
    if rc.claim.extract_confidence < cfg.thresholds.min_extract_confidence:
        return Decision(
            action=Action.SKIP,
            rule_id="R-SKIP-LOWCONF",
            explanation=(
                f"Skipped claim with extract_confidence {rc.claim.extract_confidence:.2f} "
                f"< min_extract_confidence {cfg.thresholds.min_extract_confidence:.2f}."
            ),
            scores={"extract_confidence": rc.claim.extract_confidence},
            mutations=[],
            affected_memory_ids=[],
        )

    # Calculate claim strength
    s_new = (
        calculate_strength(rc.source.reliability, rc.claim.extract_confidence)
        if cfg.ablation.use_trust
        else 1.0
    )
    s_new = min(cfg.thresholds.max_confidence, s_new)

    # Retract claim polarity (R-RETRACT, R-SKIP-NOTARGET, R-REJECT-RETRACT)
    if rc.claim.polarity == Polarity.RETRACT:
        return _produce_r_retract(
            rc=rc,
            chain=chain,
            schema=schema,
            cfg=cfg,
            classify=classify,
            recorded_at=recorded_at,
            s_new=s_new,
        )

    # 2. Filter candidate memories covering rc.valid_from
    cover = [m for m in chain if covers(m, rc.valid_from)]
    live = [m for m in cover if m.status in (MemoryStatus.ACTIVE, MemoryStatus.DISPUTED)]
    hist = [m for m in cover if m.status == MemoryStatus.HISTORICAL]

    # 3. No covering live memories
    if not live:
        if hist:
            # Conflict with history only (new claim reaches back into a closed historical window)
            rel = classify(rc, hist[0])
            if rel == Relation.IDENTICAL:
                return _produce_r_corroborate_or_dup(
                    rc=rc,
                    target=hist[0],
                    s_new=s_new,
                    cfg=cfg,
                    recorded_at=recorded_at,
                )
            if rel == Relation.REFINES:
                return _produce_r_historical_refine(
                    rc=rc,
                    target=hist[0],
                    dep_targets=dep_targets,
                    recorded_at=recorded_at,
                    s_new=s_new,
                )
            if rel == Relation.CONFLICTS:
                hist_score = _compute_memory_score(hist[0], rc.valid_from, schema, cfg)
                if s_new < hist_score * cfg.thresholds.reject_ratio:
                    return _produce_r_reject(
                        rc=rc,
                        live=[hist[0]],
                        recorded_at=recorded_at,
                        s_new=s_new,
                        old_score=hist_score,
                        reject_ratio=cfg.thresholds.reject_ratio,
                    )
                return _produce_r_dispute(
                    rc=rc,
                    live=[hist[0]],
                    dep_targets=dep_targets,
                    recorded_at=recorded_at,
                    s_new=s_new,
                    old_score=hist_score,
                    margin=cfg.thresholds.replace_margin,
                    reject_ratio=cfg.thresholds.reject_ratio,
                    is_revert=False,
                )

        later = next((m for m in chain if m.valid_from > rc.valid_from), None)
        if later is None:
            return _produce_r_insert(
                rc=rc,
                dep_targets=dep_targets,
                recorded_at=recorded_at,
                s_new=s_new,
            )

        # R-BACKFILL gate: older fact filling an earlier gap prior to later memory.
        # Evaluate later memory at its inception time (later.valid_from) to obtain its
        # established baseline score without backwards-in-time decay anomalies.
        later_score = _compute_memory_score(later, later.valid_from, schema, cfg)
        if s_new < later_score * cfg.thresholds.reject_ratio:
            return _produce_r_reject(
                rc=rc,
                live=[later],
                recorded_at=recorded_at,
                s_new=s_new,
                old_score=later_score,
                reject_ratio=cfg.thresholds.reject_ratio,
            )

        return _produce_r_backfill(
            rc=rc,
            later=later,
            dep_targets=dep_targets,
            recorded_at=recorded_at,
            s_new=s_new,
            later_score=later_score,
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
            old_score = max(
                _compute_memory_score(m_live, rc.valid_from, schema, cfg) for m_live in live
            )

            # Check revert guard against superseded historical ancestors
            if superseded_ancestor_values is not None:
                superseded_values = superseded_ancestor_values
            else:
                # Deterministic fallback for pure unit tests without repository access:
                # only consider historical memories directly closed by live memories
                superseded_values = {
                    m_hist.value_norm
                    for m_hist in chain
                    if m_hist.status == MemoryStatus.HISTORICAL
                    and any(m_hist.valid_to == m_live.valid_from for m_live in live)
                }
            is_revert = rc.value_norm in superseded_values
            margin = cfg.thresholds.revert_margin if is_revert else cfg.thresholds.replace_margin

            newer = all(rc.valid_from > mem.valid_from for mem in live)

            if newer and s_new >= old_score * margin:
                return _produce_r_replace(
                    rc=rc,
                    live=live,
                    dep_targets=dep_targets,
                    recorded_at=recorded_at,
                    s_new=s_new,
                    old_score=old_score,
                    margin=margin,
                    is_revert=is_revert,
                )
            elif s_new < old_score * cfg.thresholds.reject_ratio:
                return _produce_r_reject(
                    rc=rc,
                    live=live,
                    recorded_at=recorded_at,
                    s_new=s_new,
                    old_score=old_score,
                    reject_ratio=cfg.thresholds.reject_ratio,
                )
            else:
                return _produce_r_dispute(
                    rc=rc,
                    live=live,
                    dep_targets=dep_targets,
                    recorded_at=recorded_at,
                    s_new=s_new,
                    old_score=old_score,
                    margin=margin,
                    reject_ratio=cfg.thresholds.reject_ratio,
                    is_revert=is_revert,
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


def _produce_r_backfill(
    rc: ResolvedClaim,
    later: Memory,
    dep_targets: list[int],
    recorded_at: datetime,
    s_new: float = 1.0,
    later_score: float = 1.0,
) -> Decision:
    """Build an R-BACKFILL Decision for an older fact filling an earlier gap."""
    valid_to = later.valid_from
    if rc.valid_to is not None:
        valid_to = min(rc.valid_to, later.valid_from)

    new_mem = NewMemory(
        entity_id=rc.entity_id,
        attribute=rc.attribute,
        value_norm=rc.value_norm,
        value_display=rc.value_display,
        context=rc.claim.context,
        context_key=rc.context_key,
        valid_from=rc.valid_from,
        valid_to=valid_to,
        closed_at=recorded_at,
        status=MemoryStatus.HISTORICAL,
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
        action=Action.BACKFILL,
        rule_id="R-BACKFILL",
        explanation=(
            f"Backfilled historical claim for {rc.claim.subject}.{rc.attribute} filling gap "
            f"prior to memory M{later.id} (valid_to={format_datetime_utc(valid_to)})."
        ),
        scores={"new_score": s_new, "later_score": later_score},
        mutations=mutations,
        affected_memory_ids=[later.id],
    )


def _produce_r_replace(
    rc: ResolvedClaim,
    live: list[Memory],
    dep_targets: list[int],
    recorded_at: datetime,
    s_new: float = 1.0,
    old_score: float = 1.0,
    margin: float = 0.80,
    is_revert: bool = False,
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
    if is_revert:
        explanation = (
            f"Newer claim reverts attribute '{rc.attribute}' to previously superseded "
            f"value '{rc.value_norm}' with sufficient trust (score {s_new:.3f} >= "
            f"{old_score * margin:.3f}, revert_margin={margin})."
        )
    else:
        explanation = (
            f"Newer conflicting claim replaces existing memory {live_ids_str} "
            f"on single-valued attribute '{rc.attribute}' (score {s_new:.3f} >= "
            f"{old_score * margin:.3f}, replace_margin={margin})."
        )

    return Decision(
        action=Action.REPLACE,
        rule_id="R-REPLACE",
        explanation=explanation,
        scores={"new_score": s_new, "old_score": old_score, "margin": margin},
        mutations=mutations,
        affected_memory_ids=affected_ids,
    )


def _produce_r_reject(
    rc: ResolvedClaim,
    live: list[Memory],
    recorded_at: datetime,
    s_new: float,
    old_score: float,
    reject_ratio: float,
) -> Decision:
    """Build an R-REJECT Decision storing the rejected claim for audit with contradicts edges."""
    new_mem = NewMemory(
        entity_id=rc.entity_id,
        attribute=rc.attribute,
        value_norm=rc.value_norm,
        value_display=rc.value_display,
        context=rc.claim.context,
        context_key=rc.context_key,
        valid_from=rc.valid_from,
        valid_to=rc.valid_to,
        closed_at=recorded_at,
        status=MemoryStatus.REJECTED,
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
    affected_ids = [m.id for m in live]
    for old_m in live:
        mutations.append(
            AddEdge(
                src="$new",
                dst=old_m.id,
                type=EdgeType.CONTRADICTS,
                reason="rejected conflicting claim with insufficient trust",
            )
        )

    explanation = (
        f"Conflicting claim '{rc.value_norm}' rejected on attribute '{rc.attribute}': "
        f"score {s_new:.3f} is below reject threshold {old_score * reject_ratio:.3f} "
        f"(old_score={old_score:.3f}, reject_ratio={reject_ratio})."
    )
    return Decision(
        action=Action.REJECT,
        rule_id="R-REJECT",
        explanation=explanation,
        scores={
            "new_score": s_new,
            "old_score": old_score,
            "reject_ratio": reject_ratio,
        },
        mutations=mutations,
        affected_memory_ids=affected_ids,
    )


def _produce_r_dispute(
    rc: ResolvedClaim,
    live: list[Memory],
    dep_targets: list[int],
    recorded_at: datetime,
    s_new: float,
    old_score: float,
    margin: float,
    reject_ratio: float,
    is_revert: bool = False,
) -> Decision:
    """Build an R-DISPUTE Decision placing both new and existing memories in dispute."""
    dispute_valid_to = rc.valid_to

    # Historical interval clamping applies strictly when disputed target memories are HISTORICAL
    is_historical_dispute = bool(live and all(m.status == MemoryStatus.HISTORICAL for m in live))
    if is_historical_dispute:
        hist_bound = min((m.valid_to for m in live if m.valid_to is not None), default=None)
        if hist_bound is not None:
            if dispute_valid_to is None or dispute_valid_to > hist_bound:
                dispute_valid_to = hist_bound

    if dispute_valid_to is not None and dispute_valid_to <= rc.valid_from:
        raise InvalidValidityWindowError(
            f"Disputed memory valid_to ({dispute_valid_to}) must be greater than "
            f"valid_from ({rc.valid_from})."
        )

    # Invariant INV-CLOSED-AT-REQUIRED: valid_to is not None implies closed_at is not None.
    # Distinguish valid/event time from recorded/known time:
    # - For finite validity intervals (e.g. historical disputes), the memory is known to be
    #   closed at recorded_at (closed_at = recorded_at).
    # - For open-ended active disputes (valid_to is None), closed_at remains None.
    closed_at = recorded_at if dispute_valid_to is not None else None

    new_mem = NewMemory(
        entity_id=rc.entity_id,
        attribute=rc.attribute,
        value_norm=rc.value_norm,
        value_display=rc.value_display,
        context=rc.claim.context,
        context_key=rc.context_key,
        valid_from=rc.valid_from,
        valid_to=dispute_valid_to,
        closed_at=closed_at,
        status=MemoryStatus.DISPUTED,
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
                fields={"status": MemoryStatus.DISPUTED},
            )
        )
        mutations.append(
            AddEdge(
                src="$new",
                dst=old_m.id,
                type=EdgeType.CONTRADICTS,
                reason="conflicting claims placed in dispute",
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
    if is_revert:
        explanation = (
            f"Potential reversion of attribute '{rc.attribute}' to '{rc.value_norm}' "
            f"blocked by revert guard: score {s_new:.3f} < {old_score * margin:.3f} "
            f"(revert_margin={margin}); placed in dispute."
        )
    else:
        explanation = (
            f"Conflicting claim '{rc.value_norm}' (score={s_new:.3f}) and existing memory "
            f"{live_ids_str} (score={old_score:.3f}) placed in dispute."
        )

    return Decision(
        action=Action.DISPUTE,
        rule_id="R-DISPUTE",
        explanation=explanation,
        scores={
            "new_score": s_new,
            "old_score": old_score,
            "margin": margin,
            "reject_ratio": reject_ratio,
        },
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
        scores={
            "new_score": s_new,
            "old_score": 1.0,
            "is_historical_refine": 0.0,
        },
        mutations=mutations,
        affected_memory_ids=affected_ids,
    )


def _produce_r_historical_refine(
    rc: ResolvedClaim,
    target: Memory,
    dep_targets: list[int],
    recorded_at: datetime,
    s_new: float = 1.0,
) -> Decision:
    """Build an R-REFINE Decision for a claim refining a closed historical memory.

    Adjusts the coarse historical memory's validity interval to [target.valid_from, rc.valid_from)
    and inserts the refined claim as a HISTORICAL memory over [rc.valid_from, refined_valid_to).
    If rc.valid_to is earlier than target.valid_to, inserts an after/continuation segment over
    [rc.valid_to, target.valid_to) representing the coarse value for the remainder of the original
    historical window. This prevents gaps and overlapping validity intervals on single-valued
    attributes while preserving full provenance and temporal invariants.
    """
    if target.valid_to is None:
        raise ConflictResolutionError("Target historical memory has no valid_to.")

    if rc.valid_from <= target.valid_from:
        raise ConflictResolutionError(
            f"Historical refinement valid_from ({rc.valid_from}) must be strictly greater than "
            f"target memory valid_from ({target.valid_from})."
        )

    refined_valid_to = (
        min(rc.valid_to, target.valid_to) if rc.valid_to is not None else target.valid_to
    )
    if refined_valid_to <= rc.valid_from:
        raise InvalidValidityWindowError(
            f"Refined memory valid_to ({refined_valid_to}) must be greater than "
            f"valid_from ({rc.valid_from})."
        )

    # 1. Refined memory segment: [rc.valid_from, refined_valid_to)
    refined_mem = NewMemory(
        entity_id=rc.entity_id,
        attribute=rc.attribute,
        value_norm=rc.value_norm,
        value_display=rc.value_display,
        context=rc.claim.context,
        context_key=rc.context_key,
        valid_from=rc.valid_from,
        valid_to=refined_valid_to,
        closed_at=recorded_at,
        status=MemoryStatus.HISTORICAL,
        confidence=s_new,
        raw_text=rc.claim.span or "",
        message_id=rc.message_id,
        source_id=rc.source.id,
    )
    refined_evidence = NewEvidence(
        source_id=rc.source.id,
        message_id=rc.message_id,
        strength=s_new,
        observed_at=recorded_at,
        span=rc.claim.span or "",
    )

    mutations: list[Mutation] = [
        InsertMemory(memory=refined_mem, evidence=refined_evidence),
        AddEdge(
            src="$new",
            dst=target.id,
            type=EdgeType.REFINES,
            reason=(
                f"refines historical memory M{target.id} ('{target.value_norm}') "
                f"with '{rc.value_norm}' from {format_datetime_utc(rc.valid_from)}"
            ),
        ),
    ]
    for target_id in dep_targets:
        mutations.append(
            AddEdge(
                src="$new",
                dst=target_id,
                type=EdgeType.DEPENDS_ON,
                reason="claim dependency",
            )
        )

    # 2. Before segment: update target to [target.valid_from, rc.valid_from)
    mutations.append(
        UpdateMemory(
            memory_id=target.id,
            fields={
                "valid_to": rc.valid_from,
                "closed_at": recorded_at,
            },
        )
    )

    # 3. After (continuation) segment: if refinement ends before target.valid_to,
    # preserve coarse value and all original evidence records for [rc.valid_to, target.valid_to)
    if rc.valid_to is not None and rc.valid_to < target.valid_to:
        continuation_mem = NewMemory(
            entity_id=target.entity_id,
            attribute=target.attribute,
            value_norm=target.value_norm,
            value_display=target.value_display,
            context=target.context,
            context_key=target.context_key,
            valid_from=rc.valid_to,
            valid_to=target.valid_to,
            closed_at=recorded_at,
            status=MemoryStatus.HISTORICAL,
            confidence=target.confidence,
            raw_text=target.raw_text,
            message_id=target.message_id,
            source_id=target.source_id,
        )
        if not target.evidence:
            raise ConflictResolutionError(
                f"Cannot partially refine historical memory M{target.id}: target has no evidence "
                f"records to substantiate continuation interval "
                f"[{format_datetime_utc(rc.valid_to)}, {format_datetime_utc(target.valid_to)})."
            )

        continuation_evidences = [
            NewEvidence(
                source_id=ev.source_id,
                message_id=ev.message_id,
                strength=ev.strength,
                observed_at=ev.observed_at,
                span=ev.span,
            )
            for ev in target.evidence
        ]
        mutations.append(InsertMemory(memory=continuation_mem, evidence=continuation_evidences))
        mutations.append(
            AddEdge(
                src="$continuation",
                dst=target.id,
                type=EdgeType.SUPERSEDES,
                reason=(
                    f"continuation of historical M{target.id} ('{target.value_norm}') "
                    f"following refinement window [{format_datetime_utc(rc.valid_from)}, "
                    f"{format_datetime_utc(rc.valid_to)})"
                ),
            )
        )

    if rc.valid_to is not None and rc.valid_to < target.valid_to:
        explanation = (
            f"Historical claim '{rc.value_norm}' refines historical memory M{target.id} "
            f"('{target.value_norm}') on attribute '{rc.attribute}' for sub-interval "
            f"[{format_datetime_utc(rc.valid_from)}, {format_datetime_utc(rc.valid_to)}); "
            f"splits original window into before [{format_datetime_utc(target.valid_from)}, "
            f"{format_datetime_utc(rc.valid_from)}), refined "
            f"[{format_datetime_utc(rc.valid_from)}, {format_datetime_utc(rc.valid_to)}), "
            f"and continuation "
            f"[{format_datetime_utc(rc.valid_to)}, {format_datetime_utc(target.valid_to)})."
        )
    else:
        explanation = (
            f"Historical claim '{rc.value_norm}' refines existing historical memory M{target.id} "
            f"('{target.value_norm}') on attribute '{rc.attribute}'; closes coarser observation at "
            f"{format_datetime_utc(rc.valid_from)}."
        )

    return Decision(
        action=Action.REFINE,
        rule_id="R-REFINE",
        explanation=explanation,
        scores={
            "new_score": s_new,
            "old_score": target.confidence,
            "is_historical_refine": 1.0,
            "target_original_valid_to_ts": target.valid_to.timestamp(),
            "target_original_closed_at_ts": (
                target.closed_at.timestamp() if target.closed_at is not None else 0.0
            ),
        },
        mutations=mutations,
        affected_memory_ids=[target.id],
    )


def _produce_r_retract(
    rc: ResolvedClaim,
    chain: list[Memory],
    schema: AttributeDef,
    cfg: EngineConfig,
    classify: Callable[[ResolvedClaim, Memory], Relation],
    recorded_at: datetime,
    s_new: float,
) -> Decision:
    """Build an R-RETRACT, R-SKIP-NOTARGET, or R-REJECT-RETRACT decision.

    Specification (Section 4.4):
    target = live memory with IDENTICAL value that covers rc.valid_from
    if none:
        SKIP (rule R-SKIP-NOTARGET, explanation "nothing to retract")
    else:
        if s_new >= effective_conf(target) * reject_ratio:
            target -> HISTORICAL, valid_to = rc.valid_from, closed_at = recorded_at
            affected = [target]
        else:
            R-REJECT-RETRACT (retraction too weak; logged, no change)
    """
    # 1. Find live memories that cover rc.valid_from
    live = [
        m
        for m in chain
        if m.status in (MemoryStatus.ACTIVE, MemoryStatus.DISPUTED) and covers(m, rc.valid_from)
    ]

    target: Memory | None = None
    for m in live:
        rel = classify(rc, m)
        if rel == Relation.IDENTICAL or m.value_norm == rc.value_norm:
            target = m
            break

    # If single-valued and claim has no explicit value, fallback to any live covering memory
    if target is None and schema.cardinality == Cardinality.SINGLE and not rc.value_norm and live:
        target = live[0]

    if target is None:
        return Decision(
            action=Action.SKIP,
            rule_id="R-SKIP-NOTARGET",
            explanation=(
                f"Nothing to retract: no active memory matching '{rc.value_norm}' "
                f"covers valid_from {rc.valid_from.isoformat()}."
            ),
            scores={"s_new": s_new},
            mutations=[],
            affected_memory_ids=[],
        )

    # Validate temporal window: valid_to must be strictly greater than valid_from
    if rc.valid_from <= target.valid_from:
        raise InvalidValidityWindowError(
            f"Cannot retract memory M{target.id}: retraction valid_from ({rc.valid_from}) "
            f"must be strictly greater than target valid_from ({target.valid_from})."
        )

    target_score = _compute_memory_score(target, rc.valid_from, schema, cfg)
    threshold = target_score * cfg.thresholds.reject_ratio

    if s_new < threshold:
        return Decision(
            action=Action.REJECT,
            rule_id="R-REJECT-RETRACT",
            explanation=(
                f"Retraction claim strength {s_new:.2f} is below threshold "
                f"({threshold:.2f}) to retract memory M{target.id}."
            ),
            scores={
                "s_new": s_new,
                "target_score": target_score,
                "reject_ratio": cfg.thresholds.reject_ratio,
            },
            mutations=[],
            affected_memory_ids=[target.id],
        )

    return Decision(
        action=Action.RETRACT,
        rule_id="R-RETRACT",
        explanation=(
            f"Memory M{target.id} retracted as of {rc.valid_from.isoformat()} "
            f"(closed_at={recorded_at.isoformat()})."
        ),
        scores={
            "s_new": s_new,
            "target_score": target_score,
            "reject_ratio": cfg.thresholds.reject_ratio,
        },
        mutations=[
            UpdateMemory(
                memory_id=target.id,
                fields={
                    "status": MemoryStatus.HISTORICAL,
                    "valid_to": rc.valid_from,
                    "closed_at": recorded_at,
                },
            )
        ],
        affected_memory_ids=[target.id],
    )
