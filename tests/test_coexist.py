"""Tests for multi-valued attribute coexistence (R-COEXIST).

Governed by docs/RESOLUTION_ENGINE.md, docs/SPECIFICATION.md, and docs/ARCHITECTURE.md.
P2.2 Implementation: Part A.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memorymesh.config import load_engine_config
from memorymesh.db import init_db
from memorymesh.engine.rules import resolve
from memorymesh.errors import UnsupportedRelationError
from memorymesh.models import (
    Action,
    AttributeDef,
    Cardinality,
    Claim,
    EdgeType,
    IngestMessage,
    InsertMemory,
    Memory,
    MemoryStatus,
    NewMemory,
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
def multi_schema() -> AttributeDef:
    return AttributeDef(
        attribute="runs_service",
        description="Services running on host",
        cardinality=Cardinality.MULTI,
        value_type=ValueType.TEXT,
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
            attribute="runs_service",
            description="Services running on host",
            cardinality=Cardinality.MULTI,
            value_type=ValueType.TEXT,
        )
    )
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="python_version",
            description="Python version",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
        )
    )
    return MemoryService(repo)


class TestCoexistEngineRules:
    """Direct unit tests for resolve() with multi-valued attributes."""

    def test_first_claim_on_multi_attribute_inserts(
        self, sample_source: Source, multi_schema: AttributeDef
    ) -> None:
        cfg = load_engine_config()
        t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        rc = ResolvedClaim(
            claim=Claim(
                subject="Server A",
                attribute="runs_service",
                value="api_gateway",
                extract_confidence=0.95,
            ),
            entity_id=1,
            attribute="runs_service",
            value_norm="api_gateway",
            value_display="api_gateway",
            context_key="",
            valid_from=t,
            source=sample_source,
            message_id=1,
        )

        decision = resolve(
            rc=rc,
            chain=[],
            schema=multi_schema,
            cfg=cfg,
            classify=lambda _rc, _m: Relation.UNRELATED,
            dep_targets=[],
            recorded_at=t,
        )
        assert decision.action == Action.INSERT
        assert decision.rule_id == "R-INSERT"
        assert len(decision.mutations) == 1
        assert isinstance(decision.mutations[0], InsertMemory)

    def test_distinct_value_triggers_r_coexist(
        self, sample_source: Source, multi_schema: AttributeDef
    ) -> None:
        cfg = load_engine_config()
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 1, 20, 10, 0, tzinfo=UTC)

        existing_mem = Memory(
            id=10,
            entity_id=1,
            attribute="runs_service",
            value_norm="api_gateway",
            value_display="API Gateway",
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
                attribute="runs_service",
                value="auth_service",
                extract_confidence=0.95,
            ),
            entity_id=1,
            attribute="runs_service",
            value_norm="auth_service",
            value_display="Auth Service",
            context_key="",
            valid_from=t2,
            source=sample_source,
            message_id=2,
        )

        decision = resolve(
            rc=rc,
            chain=[existing_mem],
            schema=multi_schema,
            cfg=cfg,
            classify=lambda _rc, _m: Relation.COMPLEMENTARY,
            dep_targets=[],
            recorded_at=t2,
        )
        assert decision.action == Action.COEXIST
        assert decision.rule_id == "R-COEXIST"
        # No mutations to existing memory
        assert not any(isinstance(m, UpdateMemory) for m in decision.mutations)
        # One mutation inserting new memory
        assert len(decision.mutations) == 1
        assert isinstance(decision.mutations[0], InsertMemory)
        assert decision.mutations[0].memory.status == MemoryStatus.ACTIVE
        assert decision.mutations[0].memory.valid_to is None

    def test_identical_value_on_multi_attribute_raises_unsupported_relation(
        self, sample_source: Source, multi_schema: AttributeDef
    ) -> None:
        cfg = load_engine_config()
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 1, 20, 10, 0, tzinfo=UTC)

        existing_mem = Memory(
            id=10,
            entity_id=1,
            attribute="runs_service",
            value_norm="api_gateway",
            value_display="API Gateway",
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
                attribute="runs_service",
                value="api_gateway",
                extract_confidence=0.95,
            ),
            entity_id=1,
            attribute="runs_service",
            value_norm="api_gateway",
            value_display="API Gateway",
            context_key="",
            valid_from=t2,
            source=sample_source,
            message_id=2,
        )

        with pytest.raises(UnsupportedRelationError, match="Relation.IDENTICAL"):
            resolve(
                rc=rc,
                chain=[existing_mem],
                schema=multi_schema,
                cfg=cfg,
                classify=lambda _rc, _m: Relation.IDENTICAL,
                dep_targets=[],
                recorded_at=t2,
            )


class TestCoexistServiceEndToEnd:
    """End-to-end integration tests using MemoryService."""

    def test_distinct_values_coexist_in_database(self, service: MemoryService) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 1, 20, 10, 0, tzinfo=UTC)

        # Ingest first service
        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs auth_service",
            event_time=t1,
        )
        claim1 = Claim(
            subject="Server A",
            attribute="runs_service",
            value="auth_service",
            extract_confidence=0.95,
        )
        report1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
        assert len(report1.decisions) == 1
        assert report1.decisions[0].action == Action.INSERT
        assert report1.decisions[0].rule_id == "R-INSERT"

        # Ingest second service
        msg2 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs api_gateway",
            event_time=t2,
        )
        claim2 = Claim(
            subject="Server A",
            attribute="runs_service",
            value="api_gateway",
            extract_confidence=0.95,
        )
        report2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
        assert len(report2.decisions) == 1
        assert report2.decisions[0].action == Action.COEXIST
        assert report2.decisions[0].rule_id == "R-COEXIST"

        # Verify both memories are active and open
        ent = service.repo.get_entity_by_name("Server A")
        assert ent is not None
        active_mems = service.repo.find_memories(
            entity_id=int(ent["id"]),
            attribute="runs_service",
        )
        assert len(active_mems) == 2
        values = {m.value_norm for m in active_mems}
        assert values == {"auth_service", "api_gateway"}
        for m in active_mems:
            assert m.status == MemoryStatus.ACTIVE
            assert m.valid_to is None
            assert m.closed_at is None

        # Verify no supersedes edges exist between them
        edges0 = service.repo.get_edges_for_memory(active_mems[0].id)
        edges1 = service.repo.get_edges_for_memory(active_mems[1].id)
        assert not any(e.type == EdgeType.SUPERSEDES for e in edges0)
        assert not any(e.type == EdgeType.SUPERSEDES for e in edges1)

    def test_duplicate_value_rejected_as_unsupported_and_does_not_insert_duplicate(
        self, service: MemoryService
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 1, 20, 10, 0, tzinfo=UTC)

        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs auth_service",
            event_time=t1,
        )
        claim1 = Claim(
            subject="Server A",
            attribute="runs_service",
            value="auth_service",
            extract_confidence=0.95,
        )
        service.ingest_claims(msg1, [claim1], recorded_at=t1)

        # Second different message with identical claim value
        msg2 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A still runs auth_service",
            event_time=t2,
        )
        claim2 = Claim(
            subject="Server A",
            attribute="runs_service",
            value="auth_service",
            extract_confidence=0.95,
        )
        with pytest.raises(UnsupportedRelationError, match="Relation.IDENTICAL"):
            service.ingest_claims(msg2, [claim2], recorded_at=t2)

        # Verify no duplicate memory was inserted in DB
        ent = service.repo.get_entity_by_name("Server A")
        assert ent is not None
        mems = service.repo.find_memories(
            entity_id=int(ent["id"]),
            attribute="runs_service",
        )
        assert len(mems) == 1
        assert mems[0].status == MemoryStatus.ACTIVE
        assert mems[0].valid_to is None
        assert mems[0].closed_at is None

    def test_legacy_unnormalized_multi_valued_memory_duplicate_detection(
        self, service: MemoryService
    ) -> None:
        """Verify classifier detects equivalence against legacy unnormalized memory values."""
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 1, 20, 10, 0, tzinfo=UTC)

        # Create entity, initial message, and insert legacy unnormalized memory
        entity_id = service.repo.insert_entity(canonical_name="Server A", created_at=t1)
        msg_id = service.repo.insert_message(
            source_id="sysadmin_priya",
            text="Initial legacy message",
            event_time=t1,
            ingested_at=t1,
            text_hash="legacy_hash_1",
        )
        legacy_mem = NewMemory(
            entity_id=entity_id,
            attribute="runs_service",
            value_norm="  API   Gateway  ",
            value_display="  API   Gateway  ",
            context={},
            context_key="",
            valid_from=t1,
            valid_to=None,
            closed_at=None,
            status=MemoryStatus.ACTIVE,
            confidence=1.0,
            raw_text="Server A runs API Gateway",
            message_id=msg_id,
            source_id="sysadmin_priya",
        )
        service.repo.insert_memory(legacy_mem, recorded_at=t1)

        # Incoming normalized claim "api_gateway" / "api gateway"
        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs api gateway",
            event_time=t2,
        )
        claim = Claim(
            subject="Server A",
            attribute="runs_service",
            value="api gateway",
            extract_confidence=0.95,
        )
        with pytest.raises(UnsupportedRelationError, match="Relation.IDENTICAL"):
            service.ingest_claims(msg, [claim], recorded_at=t2)

        # No duplicate memory inserted, original memory remains active
        mems = service.repo.find_memories(
            entity_id=entity_id,
            attribute="runs_service",
        )
        assert len(mems) == 1
        assert mems[0].status == MemoryStatus.ACTIVE
        assert len(mems) == 1
        assert mems[0].status == MemoryStatus.ACTIVE

    def test_context_separation_on_multi_attribute(self, service: MemoryService) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 1, 20, 10, 0, tzinfo=UTC)

        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs redis in staging",
            event_time=t1,
        )
        claim1 = Claim(
            subject="Server A",
            attribute="runs_service",
            value="redis",
            context={"env": "staging"},
            extract_confidence=0.95,
        )
        report1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
        assert report1.decisions[0].action == Action.INSERT

        # In production context, redis is an independent insert
        msg2 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs redis in prod",
            event_time=t2,
        )
        claim2 = Claim(
            subject="Server A",
            attribute="runs_service",
            value="redis",
            context={"env": "prod"},
            extract_confidence=0.95,
        )
        report2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
        assert report2.decisions[0].action == Action.INSERT

        ent = service.repo.get_entity_by_name("Server A")
        assert ent is not None
        mems_staging = service.repo.find_memories(
            entity_id=int(ent["id"]),
            attribute="runs_service",
            context_key="env=staging",
        )
        assert len(mems_staging) == 1

        mems_prod = service.repo.find_memories(
            entity_id=int(ent["id"]),
            attribute="runs_service",
            context_key="env=prod",
        )
        assert len(mems_prod) == 1

    def test_single_valued_attribute_still_replaces(self, service: MemoryService) -> None:
        """Verify single-valued attribute replaces rather than coexists."""
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 3, 20, 10, 0, tzinfo=UTC)

        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A Python is 3.10",
            event_time=t1,
        )
        claim1 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.10",
            extract_confidence=0.95,
        )
        service.ingest_claims(msg1, [claim1], recorded_at=t1)

        msg2 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A Python upgraded to 3.12",
            event_time=t2,
        )
        claim2 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
        assert report2.decisions[0].action == Action.REPLACE
        assert report2.decisions[0].rule_id == "R-REPLACE"

        # Verify old memory is historical and new memory is active
        ent = service.repo.get_entity_by_name("Server A")
        assert ent is not None
        all_mems = service.repo.get_memory_chain(
            entity_id=int(ent["id"]),
            attribute="python_version",
        )
        assert len(all_mems) == 2
        active_mems = [m for m in all_mems if m.status == MemoryStatus.ACTIVE]
        hist_mems = [m for m in all_mems if m.status == MemoryStatus.HISTORICAL]
        assert len(active_mems) == 1
        assert active_mems[0].value_norm == "3.12"
        assert len(hist_mems) == 1
        assert hist_mems[0].value_norm == "3.10"
