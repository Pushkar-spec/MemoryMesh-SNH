"""Tests for refinement resolution (R-REFINE).

Governed by docs/RESOLUTION_ENGINE.md, docs/SPECIFICATION.md, and docs/ARCHITECTURE.md.
P2.2 Implementation: Part B.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memorymesh.config import load_engine_config
from memorymesh.db import init_db
from memorymesh.engine.rules import resolve
from memorymesh.errors import ConflictResolutionError, UnsupportedRelationError
from memorymesh.models import (
    Action,
    AddEdge,
    AttributeDef,
    Cardinality,
    Claim,
    EdgeType,
    IngestMessage,
    InsertMemory,
    Memory,
    MemoryStatus,
    Relation,
    ResolvedClaim,
    Source,
    UpdateMemory,
    ValueType,
)
from memorymesh.repository import Repository
from memorymesh.service import MemoryService


@pytest.fixture
def sample_source() -> Source:
    return Source(
        id="sysadmin_priya",
        name="Priya Sharma",
        type="admin",
        reliability=0.95,
        notes="Senior Admin",
    )


@pytest.fixture
def version_schema() -> AttributeDef:
    return AttributeDef(
        attribute="python_version",
        description="Python runtime version",
        cardinality=Cardinality.SINGLE,
        value_type=ValueType.VERSION,
    )


@pytest.fixture
def service() -> MemoryService:
    conn = init_db(":memory:")
    repo = Repository(conn)
    repo.insert_source(
        Source(
            id="sysadmin_priya",
            name="Priya Sharma",
            type="admin",
            reliability=0.95,
            notes="Senior Admin",
        )
    )
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="python_version",
            description="Python runtime version",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
        )
    )
    return MemoryService(repo)


class TestRefineEngineRules:
    """Direct unit tests for resolve() with Relation.REFINES."""

    def test_newer_refinement_triggers_r_refine(
        self, sample_source: Source, version_schema: AttributeDef
    ) -> None:
        cfg = load_engine_config()
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 2, 1, 10, 0, tzinfo=UTC)

        existing_mem = Memory(
            id=10,
            entity_id=1,
            attribute="python_version",
            value_norm="3.12",
            value_display="3.12",
            context_key="",
            valid_from=t1,
            recorded_at=t1,
            status=MemoryStatus.ACTIVE,
            confidence=1.0,
            message_id=1,
            source_id="sysadmin_priya",
        )

        rc = ResolvedClaim(
            claim=Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12.4",
                extract_confidence=0.95,
            ),
            entity_id=1,
            attribute="python_version",
            value_norm="3.12.4",
            value_display="3.12.4",
            context_key="",
            valid_from=t2,
            source=sample_source,
            message_id=2,
        )

        decision = resolve(
            rc=rc,
            chain=[existing_mem],
            schema=version_schema,
            cfg=cfg,
            classify=lambda _rc, _m: Relation.REFINES,
            dep_targets=[],
            recorded_at=t2,
        )

        assert decision.action == Action.REFINE
        assert decision.rule_id == "R-REFINE"
        assert decision.affected_memory_ids == [10]

        # Verify mutations: UpdateMemory closes old memory, InsertMemory creates new,
        # AddEdge adds refines edge
        updates = [m for m in decision.mutations if isinstance(m, UpdateMemory)]
        inserts = [m for m in decision.mutations if isinstance(m, InsertMemory)]
        edges = [m for m in decision.mutations if isinstance(m, AddEdge)]

        assert len(updates) == 1
        assert updates[0].memory_id == 10
        assert updates[0].fields["status"] == MemoryStatus.HISTORICAL
        assert updates[0].fields["valid_to"] == t2
        assert updates[0].fields["closed_at"] == t2

        assert len(inserts) == 1
        assert inserts[0].memory.value_norm == "3.12.4"
        assert inserts[0].memory.status == MemoryStatus.ACTIVE
        assert inserts[0].memory.valid_from == t2
        assert inserts[0].memory.valid_to is None

        assert len(edges) == 1
        assert edges[0].src == "$new"
        assert edges[0].dst == 10
        assert edges[0].type == EdgeType.REFINES

    def test_non_newer_refinement_raises_conflict_resolution_error(
        self, sample_source: Source, version_schema: AttributeDef
    ) -> None:
        cfg = load_engine_config()
        t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
        t_older = datetime(2026, 2, 1, 10, 0, tzinfo=UTC)

        existing_mem = Memory(
            id=10,
            entity_id=1,
            attribute="python_version",
            value_norm="3.12",
            value_display="3.12",
            context_key="",
            valid_from=t1,
            recorded_at=t1,
            status=MemoryStatus.ACTIVE,
            confidence=1.0,
            message_id=1,
            source_id="sysadmin_priya",
        )

        rc = ResolvedClaim(
            claim=Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12.4",
                extract_confidence=0.95,
            ),
            entity_id=1,
            attribute="python_version",
            value_norm="3.12.4",
            value_display="3.12.4",
            context_key="",
            valid_from=t_older,
            source=sample_source,
            message_id=2,
        )

        with pytest.raises(ConflictResolutionError, match="Cannot replace: claim is not newer"):
            resolve(
                rc=rc,
                chain=[existing_mem],
                schema=version_schema,
                cfg=cfg,
                classify=lambda _rc, _m: Relation.REFINES,
                dep_targets=[],
                recorded_at=t1,
            )


class TestRefineServiceEndToEnd:
    """End-to-end integration tests using MemoryService for refinement."""

    def test_refine_flow_and_temporal_queries(self, service: MemoryService) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 2, 15, 10, 0, tzinfo=UTC)

        # Ingest general version 3.12
        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs Python 3.12",
            event_time=t1,
        )
        claim1 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
        assert report1.decisions[0].action == Action.INSERT

        # Ingest more specific version 3.12.4
        msg2 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A exact Python is 3.12.4",
            event_time=t2,
        )
        claim2 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12.4",
            extract_confidence=0.95,
        )
        report2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
        assert len(report2.decisions) == 1
        assert report2.decisions[0].action == Action.REFINE
        assert report2.decisions[0].rule_id == "R-REFINE"

        # Verify old memory is historical and new memory is active
        ent = service.repo.get_entity_by_name("Server A")
        assert ent is not None
        entity_id = int(ent["id"])

        all_mems = service.repo.get_memory_chain(entity_id=entity_id, attribute="python_version")
        assert len(all_mems) == 2

        active_mems = [m for m in all_mems if m.status == MemoryStatus.ACTIVE]
        hist_mems = [m for m in all_mems if m.status == MemoryStatus.HISTORICAL]
        assert len(active_mems) == 1
        assert active_mems[0].value_norm == "3.12.4"
        assert active_mems[0].valid_from == t2
        assert active_mems[0].valid_to is None

        assert len(hist_mems) == 1
        assert hist_mems[0].value_norm == "3.12"
        assert hist_mems[0].valid_to == t2
        assert hist_mems[0].closed_at == t2

        # Verify refines edge exists
        edges = service.repo.get_edges_for_memory(active_mems[0].id)
        refines_edges = [e for e in edges if e.type == EdgeType.REFINES]
        assert len(refines_edges) == 1
        assert refines_edges[0].src_memory_id == active_mems[0].id
        assert refines_edges[0].dst_memory_id == hist_mems[0].id

        # Verify temporal as-of queries:
        # Before refinement (Feb 1): returns 3.12
        t_mid = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
        res_past = service.get_as_of("Server A", "python_version", valid_at=t_mid)
        assert res_past is not None
        assert res_past.value_norm == "3.12"

        # After refinement (March 1): returns 3.12.4
        t_after = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
        res_current = service.get_as_of("Server A", "python_version", valid_at=t_after)
        assert res_current is not None
        assert res_current.value_norm == "3.12.4"

    def test_less_specific_version_after_more_specific_is_identical(
        self, service: MemoryService
    ) -> None:
        """3.12.4 followed by 3.12 is IDENTICAL and raises UnsupportedRelationError."""
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 2, 15, 10, 0, tzinfo=UTC)

        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A exact Python is 3.12.4",
            event_time=t1,
        )
        claim1 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12.4",
            extract_confidence=0.95,
        )
        service.ingest_claims(msg1, [claim1], recorded_at=t1)

        msg2 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A Python is 3.12",
            event_time=t2,
        )
        claim2 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        with pytest.raises(UnsupportedRelationError, match="Relation.IDENTICAL"):
            service.ingest_claims(msg2, [claim2], recorded_at=t2)

    def test_peer_version_triggers_replace_not_refine(self, service: MemoryService) -> None:
        """3.12.4 followed by 3.12.5 is CONFLICT, not REFINE, so triggers R-REPLACE."""
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 2, 15, 10, 0, tzinfo=UTC)

        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A Python is 3.12.4",
            event_time=t1,
        )
        claim1 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12.4",
            extract_confidence=0.95,
        )
        service.ingest_claims(msg1, [claim1], recorded_at=t1)

        msg2 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A upgraded to 3.12.5",
            event_time=t2,
        )
        claim2 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12.5",
            extract_confidence=0.95,
        )
        report2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
        assert report2.decisions[0].action == Action.REPLACE
        assert report2.decisions[0].rule_id == "R-REPLACE"

        ent = service.repo.get_entity_by_name("Server A")
        assert ent is not None
        edges = service.repo.get_edges_for_memory(report2.decisions[0].memory_id)
        supersedes_edges = [e for e in edges if e.type == EdgeType.SUPERSEDES]
        assert len(supersedes_edges) == 1
