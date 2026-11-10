"""MemoryService: Unified service facade for memory ingestion and retrieval.

Governed by docs/SPECIFICATION.md Section 2, docs/ARCHITECTURE.md, and docs/AGENTS.md.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from memorymesh.config import AppConfig, load_app_config
from memorymesh.db import DEFAULT_DB_PATH, init_db
from memorymesh.engine.classifier import classify_claim_memory
from memorymesh.engine.confidence import (
    calculate_base_confidence,
    calculate_decay,
    calculate_effective_confidence,
    is_stale,
)
from memorymesh.engine.entity import (
    EntityCandidate,
    EntityResolutionAction,
    normalize_entity_name,
    resolve_entity,
)
from memorymesh.engine.invariants import (
    INV_CLOSED_AT_REQUIRED,
    INV_VALID_INTERVAL,
    check_intervals_overlap,
    validate_confidence_recomputation,
    validate_temporal_window,
)
from memorymesh.engine.normalize import normalize_value
from memorymesh.engine.propagate import (
    compute_impact_paths,
    compute_propagation,
    resolve_direct_dependencies,
    resolve_direct_dependents,
    resolve_transitive_dependents,
)
from memorymesh.engine.rules import resolve
from memorymesh.errors import (
    ForeignKeyConstraintError,
    InvalidLifecycleTransitionError,
    InvalidValidityWindowError,
    MemoryNotFoundError,
    NormalizationError,
    UnknownSourceError,
)
from memorymesh.models import (
    Action,
    AddEdge,
    AddEvidence,
    AttributeDef,
    Cardinality,
    Claim,
    DecisionLogEntry,
    DisputeView,
    EdgeType,
    Evidence,
    ImpactItem,
    IngestMessage,
    IngestReport,
    InsertMemory,
    InvariantReport,
    InvariantViolation,
    Memory,
    MemoryStatus,
    MemoryView,
    NewEvidence,
    QueryPlan,
    QueryResult,
    RecomputeConfidence,
    Relation,
    ResolvedClaim,
    Source,
    UpdateMemory,
    ValueType,
    format_datetime_utc,
)
from memorymesh.repository import Repository
from memorymesh.retrieval.retrieve import (
    build_context_key,
    normalize_context,
    retrieve_current,
    retrieve_valid_memories,
)


def _make_classifier(schema: AttributeDef) -> Callable[[ResolvedClaim, Memory], Relation]:
    def classify(new_claim: ResolvedClaim, existing_mem: Memory) -> Relation:
        return classify_claim_memory(new_claim, existing_mem, schema)

    return classify


def _resolve_memory_placeholder(
    ref: int | str,
    inserted_ids: list[int],
    current_id: int | None,
) -> int:
    """Resolve an integer memory ID or symbolic placeholder to a persisted memory ID."""
    if isinstance(ref, int):
        return ref
    if ref == "$new":
        if current_id is not None:
            return current_id
        if inserted_ids:
            return inserted_ids[-1]
        raise ValueError("Cannot resolve '$new': no memory inserted yet.")
    if ref in ("$new:0", "$refined"):
        if len(inserted_ids) >= 1:
            return inserted_ids[0]
        raise ValueError(f"Cannot resolve '{ref}': first memory not inserted yet.")
    if ref in ("$new:1", "$continuation"):
        if len(inserted_ids) >= 2:
            return inserted_ids[1]
        raise ValueError(f"Cannot resolve '{ref}': second memory not inserted yet.")
    if ref.startswith("$new:"):
        idx = int(ref.split(":", 1)[1])
        if 0 <= idx < len(inserted_ids):
            return inserted_ids[idx]
        raise ValueError(
            f"Cannot resolve '{ref}': index {idx} out of range ({len(inserted_ids)} inserted)."
        )
    raise ValueError(f"Unknown memory placeholder: {ref}")


class MemoryService:
    """Core domain service orchestrating memory ingestion, resolution, and retrieval."""

    def __init__(
        self,
        db_target: str | Path | sqlite3.Connection | Repository = DEFAULT_DB_PATH,
        cfg: AppConfig | None = None,
        llm: Any | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize MemoryService with a repository, configuration, and optional clock."""
        if isinstance(db_target, Repository):
            self.repo = db_target
        elif isinstance(db_target, sqlite3.Connection):
            self.repo = Repository(db_target)
        else:
            conn = init_db(db_target)
            self.repo = Repository(conn)

        self.cfg = cfg or load_app_config()
        self.llm = llm
        self.clock = clock or (lambda: datetime.now(UTC))

    def _build_entity_candidates(self) -> list[EntityCandidate]:
        """Build candidate entities from both configuration and database."""
        candidates_map: dict[str, EntityCandidate] = {}

        # 1. Configured entities
        for cfg_ent in self.cfg.entities:
            candidates_map[cfg_ent.name] = EntityCandidate(
                canonical_name=cfg_ent.name,
                entity_type=cfg_ent.type,
                aliases=list(cfg_ent.aliases),
                db_id=None,
            )

        # 2. Database entities and aliases
        db_entities = self.repo.list_entities()
        db_aliases = self.repo.list_entity_aliases()

        aliases_by_id: dict[int, list[str]] = {}
        for row in db_aliases:
            ent_id = int(row["entity_id"])
            aliases_by_id.setdefault(ent_id, []).append(row["alias_norm"])

        for row in db_entities:
            c_name = row["canonical_name"]
            e_id = int(row["id"])
            extra_aliases = aliases_by_id.get(e_id, [])

            if c_name in candidates_map:
                existing = candidates_map[c_name]
                merged = list(dict.fromkeys(existing.aliases + extra_aliases))
                candidates_map[c_name] = existing._replace(db_id=e_id, aliases=merged)
            else:
                candidates_map[c_name] = EntityCandidate(
                    canonical_name=c_name,
                    entity_type=row.get("type", "thing"),
                    aliases=extra_aliases,
                    db_id=e_id,
                )

        return list(candidates_map.values())

    def _resolve_and_persist_entity(self, subject: str, recorded_at: datetime) -> int:
        """Resolve subject using pure entity engine and persist resolution outcomes."""
        candidates = self._build_entity_candidates()
        auto_thresh = float(self.cfg.engine.fuzzy.entity_auto_match)
        tie_thresh = float(self.cfg.engine.fuzzy.entity_tiebreak_min)

        res = resolve_entity(
            subject=subject,
            candidates=candidates,
            auto_match_threshold=auto_thresh,
            tiebreak_min_threshold=tie_thresh,
        )

        norm_subject = normalize_entity_name(subject)

        # High-confidence exact or fuzzy match with known database ID
        if res.matched_entity_id is not None:
            self.repo.insert_entity_alias(norm_subject, res.matched_entity_id)
            return res.matched_entity_id

        # Matched configured entity that was not yet created in DB
        if res.action in (EntityResolutionAction.EXACT_MATCH, EntityResolutionAction.FUZZY_MATCH):
            stored = self.repo.get_entity_by_name(res.canonical_name)
            if stored is not None:
                entity_id = int(stored["id"])
            else:
                entity_id = self.repo.insert_entity(
                    canonical_name=res.canonical_name,
                    entity_type=res.entity_type,
                    needs_review=res.needs_review,
                    created_at=recorded_at,
                )
            self.repo.insert_entity_alias(norm_subject, entity_id)
            # Register all configured aliases for this entity
            for cfg_ent in self.cfg.entities:
                if cfg_ent.name == res.canonical_name:
                    for alias in [cfg_ent.name, *cfg_ent.aliases]:
                        self.repo.insert_entity_alias(normalize_entity_name(alias), entity_id)
            return entity_id

        # Ambiguous, review-level, or unknown: create separate entity with review flag
        entity_id = self.repo.insert_entity(
            canonical_name=res.canonical_name,
            entity_type=res.entity_type,
            needs_review=res.needs_review,
            created_at=recorded_at,
        )
        # Do not register a colliding alias against the new entity or transfer it
        if res.action != EntityResolutionAction.AMBIGUOUS_MATCH:
            self.repo.insert_entity_alias(norm_subject, entity_id)
        return entity_id

    # =========================================================================
    # Write: Structured Ingestion
    # =========================================================================

    def ingest_claims(
        self,
        message: IngestMessage,
        claims: list[Claim],
        *,
        recorded_at: datetime | None = None,
    ) -> IngestReport:
        """Ingest structured claims directly, bypassing LLM extraction, with atomic transaction."""
        eff_recorded_at = recorded_at or message.event_time

        # Atomic transaction: entire message and all claims/mutations either succeed or roll back
        with self.repo.transaction():
            # Validate source exists or register from configuration catalog
            source = self.repo.get_source(message.source_id)
            if source is None:
                source_cfg = self.cfg.get_source(message.source_id)
                if source_cfg is not None:
                    source = Source(
                        id=source_cfg.id,
                        name=source_cfg.name,
                        type=source_cfg.type.value
                        if hasattr(source_cfg.type, "value")
                        else str(source_cfg.type),
                        reliability=source_cfg.reliability,
                        notes=source_cfg.notes,
                    )
                    self.repo.insert_source(source)
                else:
                    raise UnknownSourceError(
                        f"Unknown source '{message.source_id}'. "
                        "Please configure it in sources.yaml."
                    )

            text_hash = hashlib.sha256(message.text.encode("utf-8")).hexdigest()

            # Check duplicate message
            existing_msg_id = self.repo.get_message_by_hash(
                message.source_id, text_hash, message.event_time
            )
            if existing_msg_id is not None:
                return IngestReport(
                    message_id=existing_msg_id,
                    duplicate=True,
                    claims_extracted=len(claims),
                    decisions=[],
                )

            message_id = self.repo.insert_message(
                source_id=message.source_id,
                text=message.text,
                event_time=message.event_time,
                ingested_at=eff_recorded_at,
                text_hash=text_hash,
            )

            decisions_log: list[DecisionLogEntry] = []

            for claim in claims:
                # Resolve or insert entity using two-tier deterministic resolution
                entity_id = self._resolve_and_persist_entity(
                    claim.subject,
                    eff_recorded_at,
                )

                # Resolve or obtain attribute definition from catalog
                attr_def = self.repo.get_attribute_schema(claim.attribute)
                if attr_def is None:
                    attr_cfg = self.cfg.get_attribute(claim.attribute)
                    if attr_cfg is not None:
                        attr_def = AttributeDef(
                            attribute=attr_cfg.attribute,
                            description=attr_cfg.description,
                            cardinality=Cardinality(
                                attr_cfg.cardinality.value
                                if hasattr(attr_cfg.cardinality, "value")
                                else str(attr_cfg.cardinality)
                            ),
                            value_type=ValueType(
                                attr_cfg.value_type.value
                                if hasattr(attr_cfg.value_type, "value")
                                else str(attr_cfg.value_type)
                            ),
                            half_life_days=attr_cfg.half_life_days,
                            allowed_values=list(attr_cfg.allowed_values),
                            aliases=list(attr_cfg.aliases),
                        )
                        self.repo.insert_attribute_schema(attr_def)
                    else:
                        raise ValueError(
                            f"Unknown attribute '{claim.attribute}'. "
                            "Not found in database or catalog."
                        )

                if not self.cfg.engine.ablation.use_context:
                    clean_context: dict[str, str] = {}
                    context_key = ""
                else:
                    clean_context = normalize_context(claim.context) if claim.context else {}
                    context_key = build_context_key(clean_context)
                claim.context = clean_context
                valid_from = claim.valid_from or message.event_time
                if valid_from.tzinfo is None or valid_from.utcoffset() is None:
                    raise ValueError(
                        "Claim valid_from or message event_time must be timezone-aware."
                    )
                valid_from = valid_from.astimezone(UTC)

                if claim.valid_to is not None:
                    if claim.valid_to.tzinfo is None or claim.valid_to.utcoffset() is None:
                        raise ValueError("Claim valid_to must be timezone-aware.")
                    valid_to = claim.valid_to.astimezone(UTC)
                else:
                    valid_to = None

                try:
                    value_norm = normalize_value(
                        claim.value,
                        attr_def.value_type,
                        allowed_values=attr_def.allowed_values,
                    )
                except NormalizationError as exc:
                    # Documented SKIP behavior: skip invalid claim, do not create memory, log SKIP
                    log_entry = DecisionLogEntry(
                        id=0,
                        message_id=message_id,
                        claim_json=claim.model_dump(),
                        action=Action.SKIP,
                        rule_id="R-SKIP-VALUE",
                        memory_id=None,
                        affected_memory_ids=[],
                        scores={},
                        explanation=f"Skipped invalid value: {exc}",
                        created_at=eff_recorded_at,
                    )
                    saved_log = self.repo.insert_decision_log(log_entry)
                    decisions_log.append(saved_log)
                    continue

                rc = ResolvedClaim(
                    claim=claim,
                    entity_id=entity_id,
                    attribute=claim.attribute,
                    value_norm=value_norm,
                    value_display=claim.value,
                    context_key=context_key,
                    valid_from=valid_from,
                    valid_to=valid_to,
                    source=source,
                    message_id=message_id,
                )

                chain = self.repo.get_memory_chain(
                    entity_id=entity_id,
                    attribute=claim.attribute,
                    context_key=context_key,
                    include_rejected=False,
                )

                classify = _make_classifier(attr_def)

                # Resolve dependency targets from claim.depends_on
                dep_targets: list[int] = []
                for dep in claim.depends_on:
                    target_ent = self.repo.get_entity_by_name(dep.entity)
                    if target_ent is None:
                        target_ent = self.repo.get_entity_by_alias(
                            normalize_entity_name(dep.entity)
                        )
                    if target_ent:
                        target_mem = retrieve_current(
                            self.repo,
                            int(target_ent["id"]),
                            dep.attribute,
                            now=valid_from,
                            context=claim.context if self.cfg.engine.ablation.use_context else None,
                        )
                        if target_mem:
                            dep_targets.append(target_mem.id)

                # Retrieve superseded ancestor values from the supersession graph
                live_ids = [
                    m.id for m in chain if m.status in (MemoryStatus.ACTIVE, MemoryStatus.DISPUTED)
                ]
                superseded_ancestor_values = self.repo.get_superseded_ancestor_values(live_ids)

                decision = resolve(
                    rc=rc,
                    chain=chain,
                    schema=attr_def,
                    cfg=self.cfg.engine,
                    classify=classify,
                    dep_targets=dep_targets,
                    recorded_at=eff_recorded_at,
                    superseded_ancestor_values=superseded_ancestor_values,
                )

                # Apply mutations atomically in order
                inserted_mem_ids: list[int] = []
                current_new_id: int | None = None
                for mutation in decision.mutations:
                    if isinstance(mutation, InsertMemory):
                        current_new_id = self.repo.insert_memory(
                            mutation.memory, recorded_at=eff_recorded_at
                        )
                        inserted_mem_ids.append(current_new_id)
                        ev_list = (
                            mutation.evidence
                            if isinstance(mutation.evidence, list)
                            else [mutation.evidence]
                        )
                        for ev in ev_list:
                            self.repo.insert_evidence(current_new_id, ev)
                    elif isinstance(mutation, UpdateMemory):
                        self.repo.update_memory(mutation.memory_id, mutation.fields)
                    elif isinstance(mutation, AddEvidence):
                        target_id = _resolve_memory_placeholder(
                            mutation.memory_id, inserted_mem_ids, current_new_id
                        )
                        self.repo.insert_evidence(target_id, mutation.evidence)
                    elif isinstance(mutation, RecomputeConfidence):
                        target_id = _resolve_memory_placeholder(
                            mutation.memory_id, inserted_mem_ids, current_new_id
                        )
                        self.recompute_confidence(target_id)
                    elif isinstance(mutation, AddEdge):
                        src_id = _resolve_memory_placeholder(
                            mutation.src, inserted_mem_ids, current_new_id
                        )
                        dst_id = _resolve_memory_placeholder(
                            mutation.dst, inserted_mem_ids, current_new_id
                        )
                        self.repo.insert_edge(
                            src_memory_id=src_id,
                            dst_memory_id=dst_id,
                            edge_type=mutation.type,
                            reason=mutation.reason,
                            created_at=eff_recorded_at,
                        )

                primary_mem_id = (
                    inserted_mem_ids[0]
                    if inserted_mem_ids
                    else (decision.affected_memory_ids[0] if decision.affected_memory_ids else None)
                )

                log_entry = DecisionLogEntry(
                    id=0,
                    message_id=message_id,
                    claim_json=claim.model_dump(),
                    action=decision.action,
                    rule_id=decision.rule_id,
                    memory_id=primary_mem_id,
                    affected_memory_ids=decision.affected_memory_ids,
                    scores=decision.scores,
                    explanation=decision.explanation,
                    created_at=eff_recorded_at,
                )
                saved_log = self.repo.insert_decision_log(log_entry)
                decisions_log.append(saved_log)

                # P5.1 / P6.2: Propagate changes downstream if an existing memory was affected
                if decision.rule_id in ("R-REPLACE", "R-DISPUTE", "R-RETRACT"):
                    if decision.rule_id == "R-RETRACT":
                        changed_ids = list(decision.affected_memory_ids)
                        for m_id in changed_ids:
                            dispute_nbrs = self.repo.get_dispute_group(m_id)
                            for nbr in dispute_nbrs:
                                if nbr.id != m_id and nbr.status == MemoryStatus.DISPUTED:
                                    self.repo.update_memory(
                                        nbr.id,
                                        {
                                            "needs_review": True,
                                            "review_reason": (
                                                f"Conflicting memory M{m_id} was retracted; "
                                                "pending dispute resolution"
                                            ),
                                        },
                                    )
                    else:
                        changed_ids = [
                            m_id for m_id in decision.affected_memory_ids if m_id != primary_mem_id
                        ]
                    if changed_ids:
                        if decision.rule_id == "R-REPLACE":
                            cause = (
                                f"superseded by M{primary_mem_id}"
                                if primary_mem_id
                                else "superseded"
                            )
                        elif decision.rule_id == "R-DISPUTE":
                            cause = "placed in dispute"
                        else:
                            cause = "retracted"
                        prop_logs = self.propagate_change(
                            changed_ids=changed_ids,
                            cause_text=cause,
                            recorded_at=eff_recorded_at,
                            message_id=message_id,
                            trigger_mode="automatic",
                        )
                        decisions_log.extend(prop_logs)

            return IngestReport(
                message_id=message_id,
                duplicate=False,
                claims_extracted=len(claims),
                decisions=decisions_log,
            )

    def resolve_dispute(
        self,
        winner_memory_id: int,
        *,
        recorded_at: datetime | None = None,
        explanation: str | None = None,
    ) -> DecisionLogEntry:
        """Explicitly resolve an active dispute in favor of winner_memory_id.

        Governed by docs/RESOLUTION_ENGINE.md Section 4.5 (R-DISPUTE-RESOLVE):
        - Winner memory status becomes ACTIVE.
        - For each other member in the dispute group (connected by contradicts edges):
          - If loser.valid_from < winner.valid_from (older):
            loser status becomes HISTORICAL, valid_to = winner.valid_from,
            closed_at = eff_recorded_at, and SUPERSEDES edge is created.
          - Otherwise (loser.valid_from >= winner.valid_from):
            loser status becomes REJECTED, closed_at = eff_recorded_at.
        - Atomically records a DecisionLogEntry with rule_id="R-DISPUTE-RESOLVE".
        """
        raw_recorded_at = recorded_at if recorded_at is not None else self.clock()
        if raw_recorded_at.tzinfo is None or raw_recorded_at.utcoffset() is None:
            raise ValueError("Recorded timestamp must be a timezone-aware datetime.")
        eff_recorded_at = raw_recorded_at.astimezone(UTC)

        with self.repo.transaction():
            winner = self.repo.get_memory(winner_memory_id)
            if winner is None:
                raise MemoryNotFoundError(f"Memory with ID {winner_memory_id} does not exist.")
            if winner.status != MemoryStatus.DISPUTED:
                raise ValueError(
                    f"Memory {winner_memory_id} has status '{winner.status}', expected 'disputed'."
                )

            group = self.repo.get_dispute_group(winner_memory_id)
            if len(group) < 2:
                raise ValueError(
                    f"Memory {winner_memory_id} does not belong to an active "
                    "multi-memory dispute group."
                )

            losers = [m for m in group if m.id != winner_memory_id]

            self.repo.update_memory(
                winner.id,
                {"status": MemoryStatus.ACTIVE, "needs_review": False, "review_reason": ""},
            )

            loser_summaries: list[str] = []
            for loser in losers:
                if loser.status == MemoryStatus.HISTORICAL:
                    # Already historical (e.g. retracted counterpart)
                    self.repo.insert_edge(
                        src_memory_id=winner.id,
                        dst_memory_id=loser.id,
                        edge_type=EdgeType.SUPERSEDES,
                        reason="dispute resolution: winner supersedes retracted claim",
                        created_at=eff_recorded_at,
                    )
                    loser_summaries.append(
                        f"M{loser.id} (already HISTORICAL) superseded by winner M{winner.id}"
                    )
                elif loser.valid_from < winner.valid_from:
                    self.repo.update_memory(
                        loser.id,
                        {
                            "status": MemoryStatus.HISTORICAL,
                            "valid_to": winner.valid_from,
                            "closed_at": eff_recorded_at,
                        },
                    )
                    self.repo.insert_edge(
                        src_memory_id=winner.id,
                        dst_memory_id=loser.id,
                        edge_type=EdgeType.SUPERSEDES,
                        reason="dispute resolution: winner supersedes older claim",
                        created_at=eff_recorded_at,
                    )
                    loser_summaries.append(
                        f"M{loser.id} (valid_from={loser.valid_from.isoformat()}) "
                        "transitioned to HISTORICAL"
                    )
                else:
                    self.repo.update_memory(
                        loser.id,
                        {
                            "status": MemoryStatus.REJECTED,
                            "closed_at": eff_recorded_at,
                        },
                    )
                    loser_summaries.append(
                        f"M{loser.id} (valid_from={loser.valid_from.isoformat()}) "
                        "transitioned to REJECTED"
                    )

            expl = explanation or (
                f"Dispute resolved in favor of memory M{winner.id} "
                f"('{winner.value_display}', confidence={winner.confidence:.2f}). "
                + "; ".join(loser_summaries)
            )
            closed_loser_ids = [
                loser.id
                for loser in losers
                if loser.status == MemoryStatus.DISPUTED and loser.valid_from < winner.valid_from
            ]
            claim_summary = {
                "entity_id": winner.entity_id,
                "attribute": winner.attribute,
                "context_key": winner.context_key,
                "winner_memory_id": winner.id,
                "winner_value": winner.value_display,
                "loser_memory_ids": [loser.id for loser in losers],
                "closed_memory_ids": closed_loser_ids,
            }
            log_entry = DecisionLogEntry(
                id=0,
                message_id=winner.message_id,
                claim_json=claim_summary,
                action=Action.DISPUTE,
                rule_id="R-DISPUTE-RESOLVE",
                memory_id=winner.id,
                affected_memory_ids=[winner.id] + [loser.id for loser in losers],
                scores={"winner_confidence": winner.confidence},
                explanation=expl,
                created_at=eff_recorded_at,
            )
            saved_log = self.repo.insert_decision_log(log_entry)

            # P5.1: Propagate changes downstream for closed dispute losers
            loser_ids = [loser.id for loser in losers]
            if loser_ids:
                self.propagate_change(
                    changed_ids=loser_ids,
                    cause_text=f"superseded by dispute winner M{winner.id}",
                    recorded_at=eff_recorded_at,
                    message_id=winner.message_id,
                    trigger_mode="automatic",
                )

            return saved_log

    def retract(
        self,
        memory_id: int,
        reason: str = "",
        *,
        valid_to: datetime | None = None,
        recorded_at: datetime | None = None,
    ) -> DecisionLogEntry:
        """Retract an active or disputed memory.

        - Marks the memory as HISTORICAL with valid_to and closed_at.
        - Idempotent: repeated retractions on an already-retracted memory return existing log.
        - Validates lifecycle: attempting to retract a REJECTED memory or a non-retracted
          HISTORICAL memory raises InvalidLifecycleTransitionError.
        - Validates temporal window: valid_to must be strictly greater than target valid_from.
        - Propagates change downstream to dependents (needs_review=True, R-PROPAGATE).
        - Atomic: executed within a transaction.
        """
        raw_recorded_at = recorded_at if recorded_at is not None else self.clock()
        if raw_recorded_at.tzinfo is None or raw_recorded_at.utcoffset() is None:
            raise ValueError("Recorded timestamp must be a timezone-aware datetime.")
        eff_recorded_at = raw_recorded_at.astimezone(UTC)

        if valid_to is not None:
            if valid_to.tzinfo is None or valid_to.utcoffset() is None:
                raise ValueError("valid_to timestamp must be a timezone-aware datetime.")
            eff_valid_to = valid_to.astimezone(UTC)
        else:
            eff_valid_to = eff_recorded_at

        with self.repo.transaction():
            target = self.repo.get_memory(memory_id)
            if target is None:
                raise MemoryNotFoundError(f"Memory M{memory_id} not found.")

            # Idempotency check: if target is already HISTORICAL, check if it was retracted
            if target.status == MemoryStatus.HISTORICAL:
                retract_log = self.repo.get_retraction_log(memory_id)
                if retract_log is not None:
                    return retract_log
                raise InvalidLifecycleTransitionError(
                    f"Cannot retract memory M{memory_id} with status 'historical' "
                    "(already superseded or closed)."
                )

            if target.status == MemoryStatus.REJECTED:
                raise InvalidLifecycleTransitionError(
                    f"Cannot retract memory M{memory_id} with status 'rejected'."
                )

            # Temporal validity check: valid_to must be strictly greater than valid_from
            if eff_valid_to <= target.valid_from:
                raise InvalidValidityWindowError(
                    f"Cannot retract memory M{memory_id}: valid_to ({eff_valid_to}) "
                    f"must be strictly greater than valid_from ({target.valid_from})."
                )

            # Identify any conflicting memories in dispute before updating status
            was_disputed = target.status == MemoryStatus.DISPUTED
            dispute_counterparts = (
                [m for m in self.repo.get_dispute_group(memory_id) if m.id != memory_id]
                if was_disputed
                else []
            )

            # Transition memory to HISTORICAL
            self.repo.update_memory(
                memory_id,
                {
                    "status": MemoryStatus.HISTORICAL,
                    "valid_to": eff_valid_to,
                    "closed_at": eff_recorded_at,
                },
            )

            if was_disputed:
                for nbr in dispute_counterparts:
                    if nbr.status == MemoryStatus.DISPUTED:
                        self.repo.update_memory(
                            nbr.id,
                            {
                                "needs_review": True,
                                "review_reason": (
                                    f"Conflicting memory M{memory_id} was retracted; "
                                    "pending dispute resolution"
                                ),
                            },
                        )

            msg_id = target.message_id
            if msg_id is None:
                msg_id = self.repo.insert_message(
                    source_id=target.source_id or "system",
                    text=f"Retraction for memory M{memory_id}: {reason or 'Manual retraction'}",
                    event_time=eff_recorded_at,
                    ingested_at=eff_recorded_at,
                )

            explanation_text = reason or (
                f"Memory M{memory_id} retracted as of {eff_valid_to.isoformat()} "
                f"(closed_at={eff_recorded_at.isoformat()})."
            )
            log_entry = DecisionLogEntry(
                id=0,
                message_id=msg_id,
                claim_json={
                    "memory_id": memory_id,
                    "polarity": "retract",
                    "reason": reason,
                    "valid_to": format_datetime_utc(eff_valid_to),
                },
                action=Action.RETRACT,
                rule_id="R-RETRACT",
                memory_id=memory_id,
                affected_memory_ids=[memory_id],
                scores={},
                explanation=explanation_text,
                created_at=eff_recorded_at,
            )
            saved_log = self.repo.insert_decision_log(log_entry)

            # Downstream propagation
            self.propagate_change(
                changed_ids=[memory_id],
                cause_text="retracted",
                recorded_at=eff_recorded_at,
                message_id=msg_id,
                trigger_mode="automatic",
            )

            return saved_log

    retract_memory = retract

    # =========================================================================
    # Read: Retrieval
    # =========================================================================

    def _build_memory_view(
        self,
        m: Memory,
        ref_time: datetime,
        scope_match: Literal["exact", "inherited_partial", "inherited_global"] = "exact",
    ) -> MemoryView:
        attr_schema = self.repo.get_attribute_schema(m.attribute)
        half_life = attr_schema.half_life_days if attr_schema else None
        decay = calculate_decay(m.valid_from, ref_time, half_life)
        eff_conf = calculate_effective_confidence(
            m.confidence,
            decay_factor=decay,
            needs_review=m.needs_review,
            review_penalty=self.cfg.engine.thresholds.review_penalty,
        )
        stale = is_stale(eff_conf, stale_floor=self.cfg.engine.thresholds.stale_floor)
        source_names = list(
            dict.fromkeys(
                [e.source_id for e in m.evidence] + ([m.source_id] if m.source_id else [])
            )
        )
        return MemoryView(
            memory=m,
            effective_confidence=eff_conf,
            is_stale=stale,
            source_names=source_names,
            scope_match=scope_match,
        )

    def current(
        self,
        entity: str,
        attribute: str,
        context: dict[str, str] | None = None,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        *,
        now: datetime,
    ) -> QueryResult:
        """Query memory valid at valid_at (defaults to now) as known at known_at."""
        req_valid_at = valid_at or now
        if known_at is not None:
            if known_at.tzinfo is None or known_at.utcoffset() is None:
                raise ValueError("known_at must be a timezone-aware datetime.")
            req_known_at = known_at.astimezone(UTC)
        else:
            req_known_at = now

        plan = QueryPlan(
            intent="current" if valid_at is None else "as_of",
            entity=entity,
            attribute=attribute,
            valid_at=req_valid_at,
            known_at=req_known_at,
            context=context or {},
        )

        ent = self.repo.get_entity_by_name(entity)
        if ent is None:
            ent = self.repo.get_entity_by_alias(normalize_entity_name(entity))
        if ent is None:
            return QueryResult(plan=plan, items=[], notes=[f"Entity '{entity}' not found."])

        entity_id = int(ent["id"])
        if not self.cfg.engine.ablation.use_context:
            effective_context = None
        else:
            effective_context = normalize_context(context) if context else None

        memories = retrieve_valid_memories(
            repo=self.repo,
            entity_id=entity_id,
            attribute=attribute,
            valid_at=req_valid_at,
            known_at=req_known_at,
            context=effective_context,
            include_rejected=False,
        )

        q_ctx = effective_context or {}
        items: list[MemoryView] = []
        for m in memories:
            m_ctx = m.context or {}
            if m_ctx == q_ctx:
                scope_label: Literal["exact", "inherited_partial", "inherited_global"] = "exact"
            elif not m_ctx:
                scope_label = "inherited_global"
            else:
                scope_label = "inherited_partial"
            items.append(self._build_memory_view(m, req_valid_at, scope_match=scope_label))

        ctx_key = build_context_key(effective_context)
        disputed_pairs = self.repo.find_disputed_pairs(
            entity_id=entity_id,
            attribute=attribute,
            context_key=ctx_key,
            known_at=req_known_at,
        )
        if not disputed_pairs and any(
            item.scope_match in ("inherited_global", "inherited_partial") for item in items
        ):
            disputed_pairs = self.repo.find_disputed_pairs(
                entity_id=entity_id,
                attribute=attribute,
                context_key="",
                known_at=req_known_at,
            )
        dispute_views = [
            DisputeView(
                a=self._build_memory_view(pair[0], req_valid_at),
                b=self._build_memory_view(pair[1], req_valid_at),
                edge_reason=pair[2],
            )
            for pair in disputed_pairs
        ]

        notes = [] if items else ["No memory valid at the requested time."]

        # Check for ambiguity among equally specific incomparable scopes
        is_ambiguous = False
        ambiguity_reason = None
        attr_schema = self.repo.get_attribute_schema(attribute)
        is_single = attr_schema is None or attr_schema.cardinality == Cardinality.SINGLE

        if is_single and len(items) > 1:
            distinct_contexts = {build_context_key(m.context) for m in memories}
            if len(distinct_contexts) > 1:
                is_ambiguous = True
                sorted_scopes = sorted(distinct_contexts)
                ambiguity_reason = (
                    f"Ambiguous context match: found {len(items)} equally specific incomparable "
                    f"context scopes ({', '.join(sorted_scopes)}) for single-valued "
                    f"attribute '{attribute}'."
                )
                notes.append(ambiguity_reason)

        return QueryResult(
            plan=plan,
            items=items,
            disputes=dispute_views,
            notes=notes,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=ambiguity_reason,
        )

    def disputes(
        self,
        entity: str | None = None,
        *,
        known_at: datetime | None = None,
    ) -> list[DisputeView]:
        """Retrieve active disputes, optionally filtered by entity name and belief time."""
        entity_id: int | None = None
        if entity is not None:
            ent = self.repo.get_entity_by_name(entity)
            if ent is None:
                ent = self.repo.get_entity_by_alias(normalize_entity_name(entity))
            if ent is None:
                return []
            entity_id = int(ent["id"])

        if known_at is not None:
            if known_at.tzinfo is None or known_at.utcoffset() is None:
                raise ValueError("known_at must be a timezone-aware datetime.")
            known_utc = known_at.astimezone(UTC)
        else:
            known_utc = None

        ref_time = known_utc if known_utc is not None else self.clock()
        if ref_time.tzinfo is None or ref_time.utcoffset() is None:
            raise ValueError("Reference time must be a timezone-aware datetime.")
        ref_time_utc = ref_time.astimezone(UTC)

        pairs = self.repo.find_disputed_pairs(entity_id=entity_id, known_at=known_utc)
        return [
            DisputeView(
                a=self._build_memory_view(pair[0], ref_time_utc),
                b=self._build_memory_view(pair[1], ref_time_utc),
                edge_reason=pair[2],
            )
            for pair in pairs
        ]

    # =========================================================================
    # Evidence & Confidence Operations
    # =========================================================================

    def add_evidence(
        self,
        memory_id: int,
        evidence: NewEvidence,
    ) -> Evidence:
        """Add corroborating evidence to a memory and recompute its confidence atomically."""
        with self.repo.transaction():
            mem = self.repo.get_memory(memory_id)
            if mem is None:
                raise MemoryNotFoundError(f"Memory with ID {memory_id} does not exist.")

            source = self.repo.get_source(evidence.source_id)
            if source is None:
                source_cfg = self.cfg.get_source(evidence.source_id)
                if source_cfg is not None:
                    source = Source(
                        id=source_cfg.id,
                        name=source_cfg.name,
                        type=source_cfg.type.value
                        if hasattr(source_cfg.type, "value")
                        else str(source_cfg.type),
                        reliability=source_cfg.reliability,
                        notes=source_cfg.notes,
                    )
                    self.repo.insert_source(source)
                else:
                    raise UnknownSourceError(
                        f"Unknown source '{evidence.source_id}'. "
                        "Please configure it in sources.yaml."
                    )

            ev_id = self.repo.insert_evidence(memory_id, evidence)
            self.recompute_confidence(memory_id)
            persisted = self.repo.get_evidence(ev_id)
            if persisted is None:
                raise RuntimeError(f"Failed to retrieve persisted evidence {ev_id}.")
            return persisted

    def recompute_confidence(self, memory_id: int) -> float:
        """Recompute a memory's base confidence from all its persisted evidence records."""
        with self.repo.transaction():
            mem = self.repo.get_memory(memory_id)
            if mem is None:
                raise MemoryNotFoundError(f"Memory with ID {memory_id} does not exist.")

            evidence_list = self.repo.get_evidence_for_memory(memory_id)
            new_conf = calculate_base_confidence(
                evidence_list,
                max_confidence=self.cfg.engine.thresholds.max_confidence,
            )
            self.repo.update_memory_confidence(memory_id, new_conf)
            return new_conf

    def get_effective_confidence(
        self,
        memory_id: int,
        at: datetime | None = None,
    ) -> float:
        """Calculate effective confidence for a memory accounting for temporal decay and review.

        Args:
            memory_id: Target memory ID.
            at: Query reference timestamp (timezone-aware). If omitted, defaults to the
                current UTC time from the clock.

        Returns:
            Effective confidence score in [0.0, 1.0].
        """
        mem = self.repo.get_memory(memory_id)
        if mem is None:
            raise MemoryNotFoundError(f"Memory with ID {memory_id} does not exist.")

        attr_def = self.repo.get_attribute_schema(mem.attribute)
        half_life = attr_def.half_life_days if attr_def else None
        if at is not None:
            ref_time = at
        else:
            ref_time = self.clock()
            if ref_time.tzinfo is None or ref_time.utcoffset() is None:
                raise ValueError("Clock must return a timezone-aware datetime.")

        decay = calculate_decay(mem.valid_from, ref_time, half_life)
        return calculate_effective_confidence(
            mem.confidence,
            decay_factor=decay,
            needs_review=mem.needs_review,
            review_penalty=self.cfg.engine.thresholds.review_penalty,
        )

    def get_evidence(self, evidence_id: int) -> Evidence | None:
        """Retrieve a single evidence row by ID."""
        return self.repo.get_evidence(evidence_id)

    def get_memory_evidence(self, memory_id: int) -> list[Evidence]:
        """Retrieve all evidence records supporting a memory."""
        return self.repo.get_evidence_for_memory(memory_id)

    def get_current(
        self,
        entity: str,
        attribute: str,
        context: dict[str, str] | None = None,
        *,
        now: datetime,
        known_at: datetime | None = None,
    ) -> Memory | None:
        """Convenience method returning the single unambiguous ACTIVE Memory at now, or None.

        If the query result contains disputed memories, multiple conflicting candidates,
        or no active result, returns None to avoid treating disputed or ambiguous facts
        as authoritative. Detailed results remain accessible via self.current().
        """
        res = self.current(
            entity=entity,
            attribute=attribute,
            context=context,
            now=now,
            known_at=known_at,
        )
        # Any disputed memory in items means the attribute is under dispute
        if any(item.memory.status == MemoryStatus.DISPUTED for item in res.items):
            return None

        active_items = [item for item in res.items if item.memory.status == MemoryStatus.ACTIVE]
        if len(active_items) == 1:
            return active_items[0].memory

        return None

    def get_as_of(
        self,
        entity: str,
        attribute: str,
        *,
        valid_at: datetime,
        context: dict[str, str] | None = None,
        known_at: datetime | None = None,
    ) -> Memory | None:
        """Convenience method returning the single Memory valid as of valid_at, or None."""
        eff_now = self.clock()
        res = self.current(
            entity=entity,
            attribute=attribute,
            context=context,
            valid_at=valid_at,
            known_at=known_at,
            now=eff_now,
        )
        if res.is_ambiguous:
            return None
        return res.items[0].memory if res.items else None

    # =========================================================================
    # P5.1: Dependency Resolution and Propagation
    # =========================================================================

    def get_dependencies(self, memory_id: int) -> list[Memory]:
        """Retrieve direct upstream memories that memory_id depends on.

        Edge: memory_id --DEPENDS_ON--> upstream.
        Raises MemoryNotFoundError if memory_id does not exist.
        Returns memories in deterministic id ASC order.
        """
        mem = self.repo.get_memory(memory_id)
        if mem is None:
            raise MemoryNotFoundError(f"Memory with id={memory_id} not found.")

        edges = self.repo.get_dependency_edges(memory_id, direction="dependencies")
        upstream_ids = resolve_direct_dependencies(edges, memory_id)
        return self.repo.get_memories_by_ids(upstream_ids)

    def get_dependents(self, memory_id: int) -> list[Memory]:
        """Retrieve direct downstream memories that depend on memory_id.

        Edge: dependent --DEPENDS_ON--> memory_id.
        Raises MemoryNotFoundError if memory_id does not exist.
        Returns memories in deterministic id ASC order.
        """
        mem = self.repo.get_memory(memory_id)
        if mem is None:
            raise MemoryNotFoundError(f"Memory with id={memory_id} not found.")

        edges = self.repo.get_dependency_edges(memory_id, direction="dependents")
        downstream_ids = resolve_direct_dependents(edges, memory_id)
        return self.repo.get_memories_by_ids(downstream_ids)

    def get_transitive_dependents(self, memory_id: int) -> list[tuple[Memory, list[int]]]:
        """Retrieve all reachable downstream dependent memories using cycle-safe traversal.

        Raises MemoryNotFoundError if memory_id does not exist.
        Returns list of (dependent_memory, dependency_path) in deterministic discovery order.
        """
        mem = self.repo.get_memory(memory_id)
        if mem is None:
            raise MemoryNotFoundError(f"Memory with id={memory_id} not found.")

        edges = self.repo.get_transitive_dependency_edges([memory_id], direction="downstream")
        pairs = resolve_transitive_dependents(edges, memory_id)
        if not pairs:
            return []

        dep_ids = [p[0] for p in pairs]
        memories = self.repo.get_memories_by_ids(dep_ids)
        mem_map = {m.id: m for m in memories}

        return [(mem_map[dep_id], path) for dep_id, path in pairs if dep_id in mem_map]

    def propagate_change(
        self,
        changed_ids: Sequence[int],
        cause_text: str,
        *,
        recorded_at: datetime | None = None,
        message_id: int | None = None,
        trigger_mode: Literal["automatic", "manual"] = "manual",
        event_id: str | None = None,
    ) -> list[DecisionLogEntry]:
        """Propagate an upstream change through the dependency graph.

        For every affected downstream memory (active or disputed):
        - Marks memory as needs_review=True with an explainable review_reason.
        - Preserves factual content, evidence, and confidence.
        - Records an explainable R-PROPAGATE decision log row.
        - Idempotent: does not duplicate review flags or decision logs if already flagged.
        - Cycle-safe: terminates cleanly if cycles exist.
        - Atomic: entire operation executes within a transaction (or savepoint).
        """
        if not self.cfg.engine.ablation.use_propagation:
            return []

        if not changed_ids:
            return []

        raw_recorded_at = recorded_at if recorded_at is not None else self.clock()
        if raw_recorded_at.tzinfo is None or raw_recorded_at.utcoffset() is None:
            raise ValueError("Recorded timestamp must be a timezone-aware datetime.")
        eff_recorded_at = raw_recorded_at.astimezone(UTC)

        if trigger_mode not in ("automatic", "manual"):
            raise ValueError(
                f"Unsupported trigger_mode: '{trigger_mode}'. Expected 'automatic' or 'manual'."
            )

        with self.repo.transaction():
            # Validate explicit message_id inside transaction before graph reads or mutations
            if message_id is not None:
                msg_record = self.repo.get_message(message_id)
                if msg_record is None:
                    raise ForeignKeyConstraintError(
                        f"Explicit message_id {message_id} does not exist in messages table."
                    )

            edges = self.repo.get_transitive_dependency_edges(changed_ids, direction="downstream")
            if not edges:
                return []

            # Find all candidate dependent IDs in the subgraph
            candidate_ids = {e.src_memory_id for e in edges if e.type == EdgeType.DEPENDS_ON}
            candidate_memories = self.repo.get_memories_by_ids(sorted(candidate_ids))
            candidate_map = {m.id: m for m in candidate_memories}
            memory_statuses = {m.id: m.status for m in candidate_memories}

            propagation_items = compute_propagation(
                edges=edges,
                memory_statuses=memory_statuses,
                changed_ids=changed_ids,
                cause_text=cause_text,
            )

            logs: list[DecisionLogEntry] = []
            for item in propagation_items:
                # Determine attribution message_id:
                # - If message_id is explicitly passed and validated, use it
                # - If omitted, attribute to the changed root memory's message_id
                if message_id is not None:
                    eff_msg_id = message_id
                else:
                    upstream_root = self.repo.get_memory(item.upstream_id)
                    if upstream_root is None or upstream_root.message_id is None:
                        raise ForeignKeyConstraintError(
                            f"Changed root memory M{item.upstream_id} "
                            "has no valid message attribution."
                        )
                    if self.repo.get_message(upstream_root.message_id) is None:
                        raise ForeignKeyConstraintError(
                            f"Changed root memory M{item.upstream_id} references non-existent "
                            f"message ID {upstream_root.message_id}."
                        )
                    eff_msg_id = upstream_root.message_id

                # Event-level idempotency: skip if this exact event is already recorded
                if self.repo.has_propagation_decision_log(
                    memory_id=item.memory_id,
                    upstream_id=item.upstream_id,
                    message_id=eff_msg_id,
                    cause=cause_text,
                    event_id=event_id,
                    trigger_mode=trigger_mode,
                ):
                    continue

                # Update memory review status while preserving multiple causes
                target_mem = candidate_map.get(item.memory_id)
                current_reason = target_mem.review_reason if target_mem else None
                if current_reason:
                    if item.review_reason not in current_reason:
                        combined_reason = f"{current_reason}; {item.review_reason}"
                    else:
                        combined_reason = current_reason
                else:
                    combined_reason = item.review_reason

                self.repo.mark_memory_for_review(item.memory_id, combined_reason)
                if target_mem is not None:
                    target_mem.review_reason = combined_reason
                    target_mem.needs_review = True

                # Record R-PROPAGATE decision log
                path_str = " -> ".join(f"M{x}" for x in item.path)
                expl_prefix = "Manual propagation: " if trigger_mode == "manual" else ""
                explanation = (
                    f"{expl_prefix}Memory M{item.memory_id} flagged for review: depends on "
                    f"changed upstream M{item.upstream_id} via path {path_str}, "
                    f"which was {cause_text}."
                )

                claim_json = {
                    "trigger": "propagation",
                    "trigger_mode": trigger_mode,
                    "upstream_memory_id": item.upstream_id,
                    "path": item.path,
                    "cause": cause_text,
                }
                if event_id is not None:
                    claim_json["event_id"] = event_id

                log_entry = DecisionLogEntry(
                    id=0,
                    message_id=eff_msg_id,
                    claim_json=claim_json,
                    action=Action.PROPAGATE,
                    rule_id="R-PROPAGATE",
                    memory_id=item.memory_id,
                    affected_memory_ids=[item.memory_id],
                    scores={"upstream_id": item.upstream_id, "path_length": len(item.path)},
                    explanation=explanation,
                    created_at=eff_recorded_at,
                )
                saved_log = self.repo.insert_decision_log(log_entry)
                logs.append(saved_log)

            return logs

    def impact(self, memory_id: int) -> list[ImpactItem]:
        """Inspect the downstream blast radius and impact of a memory.

        Traverses DEPENDS_ON edges downstream from memory_id without mutating database state.
        Identifies direct and transitive dependents, dependency paths, memory status,
        review flags, and relevant propagation decision logs.

        Args:
            memory_id: Root memory ID to analyze.

        Returns:
            List of ImpactItem objects for all reachable downstream memories.

        Raises:
            MemoryNotFoundError: If memory_id does not exist.
        """
        root_mem = self.repo.get_memory(memory_id)
        if root_mem is None:
            raise MemoryNotFoundError(f"Memory with id={memory_id} not found.")

        edges = self.repo.get_transitive_dependency_edges([memory_id], direction="downstream")
        if not edges:
            return []

        dep_tuples = compute_impact_paths(edges, memory_id)
        if not dep_tuples:
            return []

        dep_ids = [t[0] for t in dep_tuples]
        dep_memories = self.repo.get_memories_by_ids(dep_ids, include_evidence=True)
        dep_map = {m.id: m for m in dep_memories}

        ref_time = self.clock()
        if ref_time.tzinfo is None or ref_time.utcoffset() is None:
            raise ValueError("Clock must return a timezone-aware datetime.")
        ref_time = ref_time.astimezone(UTC)

        items: list[ImpactItem] = []
        for dep_id, primary_path, all_paths, is_truncated in dep_tuples:
            mem = dep_map.get(dep_id)
            if mem is None:
                continue

            view = self._build_memory_view(mem, ref_time=ref_time)

            is_direct = len(primary_path) == 2
            kind = "Direct dependent" if is_direct else "Transitive dependent"
            path_str = " -> ".join(f"M{x}" for x in primary_path)

            alt_paths = [p for p in all_paths if p != primary_path]
            if alt_paths:
                alt_str = (
                    " (also reachable via: "
                    + ", ".join(" -> ".join(f"M{x}" for x in p) for p in alt_paths)
                    + ")"
                )
            else:
                alt_str = ""

            status_str = f"status={mem.status.value}, needs_review={mem.needs_review}"
            reason_parts = [f"{kind} of M{memory_id} via {path_str}{alt_str}", status_str]

            if is_truncated:
                reason_parts.append("alternate paths truncated by safety limit")

            # Check decision log for existing R-PROPAGATE decisions related to this root
            prop_logs = self.repo.get_decision_logs_for_memory(dep_id)
            relevant_props = [
                d
                for d in prop_logs
                if d.rule_id == "R-PROPAGATE"
                and d.claim_json.get("upstream_memory_id") == memory_id
            ]
            if relevant_props:
                latest_prop = relevant_props[-1]
                cause = latest_prop.claim_json.get("cause", "")
                reason_parts.append(f"propagated via {latest_prop.rule_id} (cause: '{cause}')")
            elif mem.needs_review and mem.review_reason:
                reason_parts.append(f"review_reason: '{mem.review_reason}'")

            reason = "; ".join(reason_parts)
            items.append(
                ImpactItem(
                    memory=view,
                    path=primary_path,
                    reason=reason,
                    all_paths=all_paths,
                    paths_truncated=is_truncated,
                )
            )

        return items

    def acknowledge_review(
        self,
        memory_id: int,
        *,
        recorded_at: datetime | None = None,
        reviewed_by: str | None = None,
        explanation: str | None = None,
    ) -> DecisionLogEntry:
        """Acknowledge that a memory flagged for review has been reviewed by a human.

        Governed by P5.2:
        - Clears needs_review (set to False) and clears review_reason on the memory.
        - Preserves factual value, confidence, evidence, validity interval, and status.
        - Records an auditable R-ACKNOWLEDGE decision log entry with Action.ACKNOWLEDGE.
        - Idempotent: repeating acknowledgement on an already acknowledged memory returns
          the existing decision log entry without duplicate writes.
        - Atomic: entire operation executes within a transaction.
        - Raises ValueError if the memory was never marked for review.
        - Raises MemoryNotFoundError if the memory does not exist.
        """
        raw_recorded_at = recorded_at if recorded_at is not None else self.clock()
        if raw_recorded_at.tzinfo is None or raw_recorded_at.utcoffset() is None:
            raise ValueError("Recorded timestamp must be a timezone-aware datetime.")
        eff_recorded_at = raw_recorded_at.astimezone(UTC)

        with self.repo.transaction():
            mem = self.repo.get_memory(memory_id)
            if mem is None:
                raise MemoryNotFoundError(f"Memory with ID {memory_id} does not exist.")

            if not mem.needs_review:
                existing_ack = self.repo.get_current_acknowledgement_log(memory_id)
                if existing_ack is not None:
                    return existing_ack
                raise ValueError(f"Memory M{memory_id} is not marked for review.")

            old_reason = mem.review_reason
            self.repo.acknowledge_memory_review(memory_id)

            expl = explanation or (
                f"Review acknowledged for memory M{memory_id}"
                + (f" by {reviewed_by}" if reviewed_by else "")
                + (f" (cleared reason: '{old_reason}')" if old_reason else ".")
            )
            claim_json: dict[str, Any] = {
                "trigger": "review_acknowledgement",
                "memory_id": mem.id,
                "entity_id": mem.entity_id,
                "attribute": mem.attribute,
                "context_key": mem.context_key,
                "cleared_reason": old_reason,
            }
            if reviewed_by is not None:
                claim_json["reviewed_by"] = reviewed_by

            log_entry = DecisionLogEntry(
                id=0,
                message_id=mem.message_id,
                claim_json=claim_json,
                action=Action.ACKNOWLEDGE,
                rule_id="R-ACKNOWLEDGE",
                memory_id=mem.id,
                affected_memory_ids=[mem.id],
                scores={"confidence": mem.confidence},
                explanation=expl,
                created_at=eff_recorded_at,
            )
            return self.repo.insert_decision_log(log_entry)

    # =========================================================================
    # P4.1: Decision Log, Explain, and Invariant Checks
    # =========================================================================

    def get_decision_logs(
        self,
        memory_id: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[DecisionLogEntry]:
        """Retrieve decision log history chronologically.

        If memory_id is provided, returns all decisions where the memory was
        either the primary subject or an affected memory, up to limit.
        Raises MemoryNotFoundError if memory_id does not exist.
        If memory_id is None, returns all decisions up to limit.
        """
        if memory_id is not None:
            mem = self.repo.get_memory(memory_id)
            if mem is None:
                raise MemoryNotFoundError(f"Memory with id={memory_id} not found.")
            return self.repo.get_decision_logs_for_memory(memory_id, limit=limit, offset=offset)

        return self.repo.get_all_decision_logs(limit=limit, offset=offset)

    def explain(self, memory_id: int) -> QueryResult:
        """Answer 'Why does this memory exist in its current state?' deterministically.

        Inspects persisted state: memory view, evidence, decision history,
        graph edges, and dispute status. Strictly read-only without creating
        new decision logs or modifying state.
        """
        mem = self.repo.get_memory(memory_id)
        if mem is None:
            raise MemoryNotFoundError(f"Memory with id={memory_id} not found.")

        ref_time = self.clock()
        if ref_time.tzinfo is None or ref_time.utcoffset() is None:
            raise ValueError("Clock must return a timezone-aware datetime.")
        ref_time = ref_time.astimezone(UTC)

        view = self._build_memory_view(mem, ref_time=ref_time)
        decisions = self.repo.get_decision_logs_for_memory(memory_id)
        edges = self.repo.get_edges_for_memory(memory_id, direction="both")

        # Check for active disputes involving this memory
        disputes: list[DisputeView] = []
        if mem.status == MemoryStatus.DISPUTED:
            disputed_pairs = self.repo.find_disputed_pairs(
                entity_id=mem.entity_id,
                attribute=mem.attribute,
                context_key=mem.context_key,
            )
            disputes = [
                DisputeView(
                    a=self._build_memory_view(pair[0], ref_time),
                    b=self._build_memory_view(pair[1], ref_time),
                    edge_reason=pair[2],
                )
                for pair in disputed_pairs
                if pair[0].id == memory_id or pair[1].id == memory_id
            ]

        # Compose deterministic explanation notes
        notes: list[str] = [
            f"Memory M{mem.id} is currently {mem.status.value.upper()}.",
            (
                f"Base confidence is {mem.confidence:.4f}; "
                f"effective confidence at {ref_time.isoformat()} is "
                f"{view.effective_confidence:.4f} (stale={view.is_stale})."
            ),
        ]
        if mem.context_key:
            notes.append(f"Context scope: '{mem.context_key}'.")
        if mem.needs_review:
            notes.append(f"Review required: {mem.review_reason or 'Flagged for human review'}.")
        if mem.evidence:
            notes.append(
                f"Supported by {len(mem.evidence)} evidence record(s) from source(s): "
                f"{', '.join(view.source_names) or 'unknown'}."
            )
        else:
            notes.append("No supporting evidence records found.")

        if decisions:
            notes.append(
                f"Associated with {len(decisions)} decision log entry/entries. "
                f"Most recent decision: {decisions[-1].action.value} via "
                f"{decisions[-1].rule_id}."
            )
        else:
            notes.append("No associated decision log entries found.")

        if edges:
            edge_descs: list[str] = []
            for e in edges:
                other_id = e.dst_memory_id if e.src_memory_id == mem.id else e.src_memory_id
                edge_descs.append(f"{e.type.value} with M{other_id}")
            notes.append(f"Connected by {len(edges)} edge(s): {', '.join(edge_descs)}.")

        if disputes:
            notes.append(f"Memory is actively involved in {len(disputes)} dispute(s).")

        plan = QueryPlan(
            intent="why",
            entity=mem.entity_name,
            attribute=mem.attribute,
            context=mem.context or {},
            valid_at=mem.valid_from,
            memory_id=mem.id,
        )

        return QueryResult(
            plan=plan,
            items=[view],
            rejected_claims=[view] if mem.status == MemoryStatus.REJECTED else [],
            disputes=disputes,
            decisions=decisions,
            edges=edges,
            notes=notes,
        )

    def check_invariants(self) -> InvariantReport:
        """Run read-only integrity checks across all persisted invariants.

        Verifies 8 key structural and business invariants without modifying state:
        - INV-CLOSED-AT-REQUIRED: Memories with valid_to must have closed_at set.
        - INV-VALID-INTERVAL: valid_to must be strictly greater than valid_from.
        - INV-NO-ACTIVE-OVERLAP: Single-valued attributes have no overlapping active windows.
        - INV-REJECTED-EXCLUDED: Rejected memories are never returned by normal retrieval.
        - INV-CONFIDENCE-CONSISTENT: Stored base confidence matches noisy-OR recomputation.
        - INV-IDEMPOTENT-INGEST: Verified via deduplication state / non-mutative structure.
        - INV-FOREIGN-KEY-INTEGRITY: Edges, logs, and evidence reference valid memories.
        - INV-SOURCE-ATTRIBUTE-INTEGRITY: References to sources, entities, and attributes exist.
        """
        violations: list[InvariantViolation] = []

        # Total checked is the sum of persisted records audited across all invariants:
        # memories (temporal windows, overlap, confidence, rejected exclusion)
        # + messages (deduplication key uniqueness)
        # + edges (foreign-key targets)
        # + decision_log (memory_id / affected_ids references)
        # + evidence (memory_id / source_id references)
        record_counts = self.repo.get_record_counts()
        total_checked = sum(record_counts.values())

        # Load all memories
        all_memories = self.repo.get_all_memories(include_evidence=True)

        # 1. INV-CLOSED-AT-REQUIRED & 2. INV-VALID-INTERVAL
        for mem in all_memories:
            for violation in validate_temporal_window(mem.valid_from, mem.valid_to, mem.closed_at):
                if violation.code == INV_CLOSED_AT_REQUIRED:
                    violations.append(
                        InvariantViolation(
                            invariant_id=violation.code,
                            description=(
                                f"Memory M{mem.id} has valid_to set "
                                f"({mem.valid_to.isoformat() if mem.valid_to else None}) "
                                f"but closed_at is NULL."
                            ),
                            affected_ids=[mem.id],
                            details={
                                "memory_id": mem.id,
                                "valid_to": mem.valid_to.isoformat() if mem.valid_to else None,
                                "closed_at": None,
                            },
                        )
                    )
                elif violation.code == INV_VALID_INTERVAL:
                    violations.append(
                        InvariantViolation(
                            invariant_id=violation.code,
                            description=(
                                f"Memory M{mem.id} valid_to "
                                f"({mem.valid_to.isoformat() if mem.valid_to else None}) is not "
                                f"strictly greater than valid_from ({mem.valid_from.isoformat()})."
                            ),
                            affected_ids=[mem.id],
                            details={
                                "memory_id": mem.id,
                                "valid_from": mem.valid_from.isoformat(),
                                "valid_to": mem.valid_to.isoformat() if mem.valid_to else None,
                            },
                        )
                    )

        # 3. INV-NO-ACTIVE-OVERLAP
        # Single-valued attributes: no two ACTIVE memories for the same
        # entity_id, attribute, and context_key may have overlapping validity intervals
        active_single_valued: dict[tuple[int, str, str], list[Memory]] = {}
        for mem in all_memories:
            if mem.status == MemoryStatus.ACTIVE:
                attr_def = self.repo.get_attribute_schema(mem.attribute)
                is_single = attr_def.cardinality == Cardinality.SINGLE if attr_def else True
                if is_single:
                    key = (mem.entity_id, mem.attribute, mem.context_key)
                    active_single_valued.setdefault(key, []).append(mem)

        for key, mem_list in active_single_valued.items():
            if len(mem_list) > 1:
                for i in range(len(mem_list)):
                    for j in range(i + 1, len(mem_list)):
                        m1 = mem_list[i]
                        m2 = mem_list[j]
                        if check_intervals_overlap(
                            m1.valid_from, m1.valid_to, m2.valid_from, m2.valid_to
                        ):
                            violations.append(
                                InvariantViolation(
                                    invariant_id="INV-NO-ACTIVE-OVERLAP",
                                    description=(
                                        f"Active single-valued memories M{m1.id} and M{m2.id} "
                                        f"overlap for entity_id={key[0]}, attribute='{key[1]}', "
                                        f"context='{key[2]}'."
                                    ),
                                    affected_ids=[m1.id, m2.id],
                                    details={
                                        "entity_id": key[0],
                                        "attribute": key[1],
                                        "context_key": key[2],
                                        "m1_window": [
                                            m1.valid_from.isoformat(),
                                            m1.valid_to.isoformat() if m1.valid_to else None,
                                        ],
                                        "m2_window": [
                                            m2.valid_from.isoformat(),
                                            m2.valid_to.isoformat() if m2.valid_to else None,
                                        ],
                                    },
                                )
                            )

        # 4. INV-REJECTED-EXCLUDED
        # Normal retrieval must never return memories with status == REJECTED
        rejected_memories = [m for m in all_memories if m.status == MemoryStatus.REJECTED]
        for rm in rejected_memories:
            # Ensure known_at is on or after rejection time (closed_at or recorded_at)
            # so the memory is evaluated within its rejected state.
            if rm.recorded_at.tzinfo is None or rm.recorded_at.utcoffset() is None:
                raise ValueError(
                    f"Memory M{rm.id} has invalid timezone-naive recorded_at: {rm.recorded_at}. "
                    "All persisted timestamps must be timezone-aware."
                )
            known_at = (rm.closed_at or rm.recorded_at).astimezone(UTC)
            retrieved = retrieve_valid_memories(
                repo=self.repo,
                entity_id=rm.entity_id,
                attribute=rm.attribute,
                valid_at=rm.valid_from,
                known_at=known_at,
                context=rm.context,
                include_rejected=False,
            )
            retrieved_ids = [m.id for m in retrieved]
            if rm.id in retrieved_ids:
                violations.append(
                    InvariantViolation(
                        invariant_id="INV-REJECTED-EXCLUDED",
                        description=(
                            f"Rejected memory M{rm.id} was returned by retrieve_valid_memories "
                            f"with include_rejected=False."
                        ),
                        affected_ids=[rm.id],
                        details={"memory_id": rm.id, "retrieved_ids": retrieved_ids},
                    )
                )

        # 5. INV-CONFIDENCE-CONSISTENT
        # Stored base confidence matches noisy-OR recomputation within tolerance (1e-4)
        max_conf = self.cfg.engine.thresholds.max_confidence
        for mem in all_memories:
            mismatch = validate_confidence_recomputation(
                mem.confidence, mem.evidence, max_confidence=max_conf, tolerance=1e-4
            )
            if mismatch is not None:
                violations.append(
                    InvariantViolation(
                        invariant_id="INV-CONFIDENCE-CONSISTENT",
                        description=(
                            f"Memory M{mem.id} stored confidence ({mem.confidence:.4f}) does not "
                            f"match recomputed evidence confidence ({mismatch:.4f})."
                        ),
                        affected_ids=[mem.id],
                        details={
                            "memory_id": mem.id,
                            "stored_confidence": mem.confidence,
                            "expected_confidence": mismatch,
                            "evidence_count": len(mem.evidence),
                        },
                    )
                )

        # 6. INV-IDEMPOTENT-INGEST
        # Check message uniqueness if messages exist
        msg_rows = self.repo.get_message_integrity_records()
        seen_keys: dict[tuple[str, str, str], int] = {}
        for mr in msg_rows:
            key = (str(mr["source_id"]), str(mr["text_hash"]), str(mr["event_time"]))
            mid = int(mr["id"])
            if key in seen_keys:
                violations.append(
                    InvariantViolation(
                        invariant_id="INV-IDEMPOTENT-INGEST",
                        description=(
                            f"Duplicate message key {key} found in messages "
                            f"(id={mid} and id={seen_keys[key]})."
                        ),
                        affected_ids=[mid, seen_keys[key]],
                        details={"key": list(key), "message_ids": [seen_keys[key], mid]},
                    )
                )
            else:
                seen_keys[key] = mid

        # 7. INV-FOREIGN-KEY-INTEGRITY
        dangling_refs = self.repo.find_dangling_references()
        for dr in dangling_refs:
            ref_type = dr["type"]
            affected: list[int] = []
            if "src_memory_id" in dr:
                affected.append(int(dr["src_memory_id"]))
            if "dst_memory_id" in dr:
                affected.append(int(dr["dst_memory_id"]))
            if "memory_id" in dr:
                affected.append(int(dr["memory_id"]))
            if "affected_id" in dr:
                affected.append(int(dr["affected_id"]))

            violations.append(
                InvariantViolation(
                    invariant_id="INV-FOREIGN-KEY-INTEGRITY",
                    description=(
                        f"Dangling reference detected ({ref_type}) in record {dr.get('id')}."
                    ),
                    affected_ids=affected,
                    details=dr,
                )
            )

        # 8. INV-SOURCE-ATTRIBUTE-INTEGRITY
        schema_violations = self.repo.find_schema_integrity_violations()
        for sv in schema_violations:
            sv_type = sv["type"]
            mem_id = int(sv["id"]) if "id" in sv else 0
            violations.append(
                InvariantViolation(
                    invariant_id="INV-SOURCE-ATTRIBUTE-INTEGRITY",
                    description=f"Schema integrity violation ({sv_type}) in record {mem_id}.",
                    affected_ids=[mem_id] if mem_id else [],
                    details=sv,
                )
            )

        return InvariantReport(
            is_valid=(len(violations) == 0),
            total_checked=total_checked,
            violations=violations,
        )
