"""MemoryService: Unified service facade for memory ingestion and retrieval.

Governed by docs/SPECIFICATION.md Section 2, docs/ARCHITECTURE.md, and docs/AGENTS.md.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from memorymesh.config import AppConfig, load_app_config
from memorymesh.db import DEFAULT_DB_PATH, init_db
from memorymesh.engine.rules import resolve
from memorymesh.errors import UnknownSourceError
from memorymesh.models import (
    Action,
    AddEdge,
    AttributeDef,
    Cardinality,
    Claim,
    DecisionLogEntry,
    IngestMessage,
    IngestReport,
    InsertMemory,
    Memory,
    MemoryView,
    QueryPlan,
    QueryResult,
    Relation,
    ResolvedClaim,
    Source,
    UpdateMemory,
    ValueType,
)
from memorymesh.repository import Repository
from memorymesh.retrieval.retrieve import (
    build_context_key,
    retrieve_current,
    retrieve_valid_memories,
)


class MemoryService:
    """Core domain service orchestrating memory ingestion, resolution, and retrieval."""

    def __init__(
        self,
        db_target: str | Path | sqlite3.Connection | Repository = DEFAULT_DB_PATH,
        cfg: AppConfig | None = None,
        llm: Any | None = None,
    ) -> None:
        """Initialize MemoryService with a repository and configuration."""
        if isinstance(db_target, Repository):
            self.repo = db_target
        elif isinstance(db_target, sqlite3.Connection):
            self.repo = Repository(db_target)
        else:
            conn = init_db(db_target)
            self.repo = Repository(conn)

        self.cfg = cfg or load_app_config()
        self.llm = llm

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
                # Resolve or insert entity
                ent = self.repo.get_entity_by_name(claim.subject)
                if ent is None:
                    entity_id = self.repo.insert_entity(
                        canonical_name=claim.subject,
                        created_at=eff_recorded_at,
                    )
                else:
                    entity_id = int(ent["id"])

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

                context_key = build_context_key(claim.context)
                valid_from = claim.valid_from or message.event_time

                rc = ResolvedClaim(
                    claim=claim,
                    entity_id=entity_id,
                    attribute=claim.attribute,
                    value_norm=claim.value.strip(),
                    value_display=claim.value,
                    context_key=context_key,
                    valid_from=valid_from,
                    valid_to=claim.valid_to,
                    source=source,
                    message_id=message_id,
                )

                chain = self.repo.get_memory_chain(
                    entity_id=entity_id,
                    attribute=claim.attribute,
                    context_key=context_key,
                    include_rejected=False,
                )

                def classify(new_claim: ResolvedClaim, existing_mem: Memory) -> Relation:
                    if new_claim.value_norm == existing_mem.value_norm:
                        return Relation.IDENTICAL
                    return Relation.CONFLICTS

                # Resolve dependency targets from claim.depends_on
                dep_targets: list[int] = []
                for dep in claim.depends_on:
                    target_ent = self.repo.get_entity_by_name(dep.entity)
                    if target_ent:
                        target_mem = retrieve_current(
                            self.repo,
                            int(target_ent["id"]),
                            dep.attribute,
                            now=valid_from,
                        )
                        if target_mem:
                            dep_targets.append(target_mem.id)

                decision = resolve(
                    rc=rc,
                    chain=chain,
                    schema=attr_def,
                    cfg=self.cfg.engine,
                    classify=classify,
                    dep_targets=dep_targets,
                    recorded_at=eff_recorded_at,
                )

                # Apply mutations atomically in order
                new_mem_id: int | None = None
                for mutation in decision.mutations:
                    if isinstance(mutation, InsertMemory):
                        new_mem_id = self.repo.insert_memory(
                            mutation.memory, recorded_at=eff_recorded_at
                        )
                        self.repo.insert_evidence(new_mem_id, mutation.evidence)
                    elif isinstance(mutation, UpdateMemory):
                        self.repo.update_memory(mutation.memory_id, mutation.fields)
                    elif isinstance(mutation, AddEdge):
                        src_id = new_mem_id if mutation.src == "$new" else mutation.src
                        dst_id = new_mem_id if mutation.dst == "$new" else mutation.dst
                        self.repo.insert_edge(
                            src_memory_id=src_id,
                            dst_memory_id=dst_id,
                            edge_type=mutation.type,
                            reason=mutation.reason,
                            created_at=eff_recorded_at,
                        )

                primary_mem_id = (
                    new_mem_id
                    if decision.action in (Action.INSERT, Action.REPLACE)
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

            return IngestReport(
                message_id=message_id,
                duplicate=False,
                claims_extracted=len(claims),
                decisions=decisions_log,
            )

    # =========================================================================
    # Read: Retrieval
    # =========================================================================

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
        req_known_at = known_at or now

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
            return QueryResult(plan=plan, items=[], notes=[f"Entity '{entity}' not found."])

        entity_id = int(ent["id"])
        memories = retrieve_valid_memories(
            repo=self.repo,
            entity_id=entity_id,
            attribute=attribute,
            valid_at=req_valid_at,
            known_at=req_known_at,
            context=context,
            include_rejected=False,
        )

        items = [
            MemoryView(
                memory=m,
                effective_confidence=m.confidence,
                is_stale=False,
                source_names=[m.source_id],
                scope_match="exact",
            )
            for m in memories
        ]
        notes = [] if items else ["No memory valid at the requested time."]
        return QueryResult(plan=plan, items=items, notes=notes)

    def get_current(
        self,
        entity: str,
        attribute: str,
        context: dict[str, str] | None = None,
        *,
        now: datetime,
    ) -> Memory | None:
        """Convenience method returning the single active Memory at now, or None."""
        res = self.current(entity=entity, attribute=attribute, context=context, now=now)
        return res.items[0].memory if res.items else None

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
        res = self.current(
            entity=entity,
            attribute=attribute,
            context=context,
            valid_at=valid_at,
            known_at=known_at,
            now=valid_at,
        )
        return res.items[0].memory if res.items else None
