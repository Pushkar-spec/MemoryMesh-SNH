"""Integration and persistence tests for Evidence and Confidence in MemoryMesh.

Governed by docs/SPECIFICATION.md Section 3, docs/ARCHITECTURE.md, and docs/RESOLUTION_ENGINE.md.
P3.1 Implementation: Evidence & Confidence Service Integration Tests.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memorymesh.db import init_db
from memorymesh.errors import ForeignKeyConstraintError, MemoryNotFoundError, UnknownSourceError
from memorymesh.models import (
    Action,
    AttributeDef,
    Cardinality,
    Claim,
    IngestMessage,
    MemoryStatus,
    NewEvidence,
    NewMemory,
    Source,
    ValueType,
)
from memorymesh.repository import Repository
from memorymesh.service import MemoryService


@pytest.fixture
def repo() -> Repository:
    conn = init_db(":memory:")
    repository = Repository(conn)
    repository.insert_source(
        Source(
            id="sysadmin_priya",
            name="Priya Sharma",
            type="admin",
            reliability=0.90,
            notes="Senior Admin",
        )
    )
    repository.insert_source(
        Source(
            id="wiki",
            name="Internal Wiki",
            type="wiki",
            reliability=0.75,
            notes="Documentation Wiki",
        )
    )
    repository.insert_source(
        Source(
            id="dev_chat",
            name="Slack Chat",
            type="chat",
            reliability=0.50,
            notes="Engineering Chat",
        )
    )
    repository.insert_attribute_schema(
        AttributeDef(
            attribute="python_version",
            description="Python runtime version",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
            half_life_days=365.0,
        )
    )
    repository.insert_attribute_schema(
        AttributeDef(
            attribute="runs_service",
            description="Services running on host",
            cardinality=Cardinality.MULTI,
            value_type=ValueType.TEXT,
        )
    )
    return repository


@pytest.fixture
def service(repo: Repository) -> MemoryService:
    return MemoryService(repo)


class TestEvidencePersistence:
    """Direct repository persistence tests for Evidence records."""

    def test_evidence_persistence_and_association(self, repo: Repository) -> None:
        t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        entity_id = repo.insert_entity("Server A", created_at=t)
        msg_id = repo.insert_message("sysadmin_priya", "Runs Python 3.10", t, t, "hash1")

        mem = NewMemory(
            entity_id=entity_id,
            attribute="python_version",
            value_norm="3.10",
            value_display="Python 3.10",
            context={},
            context_key="",
            valid_from=t,
            valid_to=None,
            closed_at=None,
            status=MemoryStatus.ACTIVE,
            confidence=0.855,
            raw_text="Runs Python 3.10",
            message_id=msg_id,
            source_id="sysadmin_priya",
        )
        mem_id = repo.insert_memory(mem, recorded_at=t)

        ev = NewEvidence(
            source_id="sysadmin_priya",
            message_id=msg_id,
            strength=0.855,
            observed_at=t,
            span="Runs Python 3.10",
        )
        ev_id = repo.insert_evidence(mem_id, ev)
        assert ev_id > 0

        # Retrieve via get_evidence and get_evidence_for_memory
        stored_ev = repo.get_evidence(ev_id)
        assert stored_ev is not None
        assert stored_ev.id == ev_id
        assert stored_ev.memory_id == mem_id
        assert stored_ev.source_id == "sysadmin_priya"
        assert stored_ev.message_id == msg_id
        assert stored_ev.strength == pytest.approx(0.855)
        assert stored_ev.span == "Runs Python 3.10"

        evidence_list = repo.get_evidence_for_memory(mem_id)
        assert len(evidence_list) == 1
        assert evidence_list[0].id == ev_id

    def test_invalid_foreign_key_references_rejected(self, repo: Repository) -> None:
        t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        entity_id = repo.insert_entity("Server A", created_at=t)
        msg_id = repo.insert_message("sysadmin_priya", "Runs Python 3.10", t, t, "hash1")

        mem = NewMemory(
            entity_id=entity_id,
            attribute="python_version",
            value_norm="3.10",
            value_display="Python 3.10",
            context={},
            context_key="",
            valid_from=t,
            valid_to=None,
            closed_at=None,
            status=MemoryStatus.ACTIVE,
            confidence=0.855,
            raw_text="Runs Python 3.10",
            message_id=msg_id,
            source_id="sysadmin_priya",
        )
        mem_id = repo.insert_memory(mem, recorded_at=t)

        # 1. Non-existent memory_id
        with pytest.raises(ForeignKeyConstraintError):
            repo.insert_evidence(
                99999,
                NewEvidence(
                    source_id="sysadmin_priya",
                    message_id=msg_id,
                    strength=0.855,
                    observed_at=t,
                ),
            )

        # 2. Non-existent source_id
        with pytest.raises(ForeignKeyConstraintError):
            repo.insert_evidence(
                mem_id,
                NewEvidence(
                    source_id="unknown_ghost_source",
                    message_id=msg_id,
                    strength=0.855,
                    observed_at=t,
                ),
            )

        # 3. Non-existent message_id
        with pytest.raises(ForeignKeyConstraintError):
            repo.insert_evidence(
                mem_id,
                NewEvidence(
                    source_id="sysadmin_priya",
                    message_id=88888,
                    strength=0.855,
                    observed_at=t,
                ),
            )

    def test_duplicate_evidence_idempotency(self, repo: Repository) -> None:
        t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        entity_id = repo.insert_entity("Server A", created_at=t)
        msg_id = repo.insert_message("sysadmin_priya", "Runs Python 3.10", t, t, "hash1")

        mem_id = repo.insert_memory(
            NewMemory(
                entity_id=entity_id,
                attribute="python_version",
                value_norm="3.10",
                value_display="Python 3.10",
                context={},
                context_key="",
                valid_from=t,
                valid_to=None,
                closed_at=None,
                status=MemoryStatus.ACTIVE,
                confidence=0.855,
                raw_text="Runs Python 3.10",
                message_id=msg_id,
                source_id="sysadmin_priya",
            ),
            recorded_at=t,
        )

        ev = NewEvidence(
            source_id="sysadmin_priya",
            message_id=msg_id,
            strength=0.855,
            observed_at=t,
            span="exact span",
        )

        # First insert returns ID
        id1 = repo.insert_evidence(mem_id, ev)
        assert id1 > 0

        # Exact duplicate insert returns identical ID without creating duplicate row
        id2 = repo.insert_evidence(mem_id, ev)
        assert id2 == id1

        evidence_rows = repo.get_evidence_for_memory(mem_id)
        assert len(evidence_rows) == 1


class TestServiceEvidenceAndConfidence:
    """Service layer integration tests for evidence addition and confidence updates."""

    def test_service_add_evidence_updates_confidence_atomically(
        self, service: MemoryService
    ) -> None:
        t1 = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 2, 2, 10, 0, tzinfo=UTC)

        # 1. Ingest initial claim Server A Python 3.10 from priya
        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs Python 3.10.",
            event_time=t1,
        )
        claim1 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.10",
            extract_confidence=0.95,
        )
        report1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
        assert report1.decisions[0].action == Action.INSERT
        mem_id = report1.decisions[0].memory_id
        assert mem_id is not None

        mem1 = service.repo.get_memory(mem_id)
        assert mem1 is not None
        assert mem1.confidence == pytest.approx(0.855, abs=0.001)

        # 2. Add second evidence from wiki via service.add_evidence
        msg2_id = service.repo.insert_message(
            source_id="wiki",
            text="Server A Python runtime is 3.10",
            event_time=t2,
            ingested_at=t2,
            text_hash="wiki_hash_1",
        )
        new_ev = NewEvidence(
            source_id="wiki",
            message_id=msg2_id,
            strength=0.7125,  # 0.75 * 0.95
            observed_at=t2,
            span="Python runtime is 3.10",
        )
        persisted_ev = service.add_evidence(mem_id, new_ev)
        assert persisted_ev.memory_id == mem_id
        assert persisted_ev.source_id == "wiki"

        # 3. Verify memory confidence was updated to 0.958
        updated_mem = service.repo.get_memory(mem_id)
        assert updated_mem is not None
        assert updated_mem.confidence == pytest.approx(0.958, abs=0.001)

        # 4. Verify evidence list has 2 rows
        ev_list = service.get_memory_evidence(mem_id)
        assert len(ev_list) == 2
        sources = {e.source_id for e in ev_list}
        assert sources == {"sysadmin_priya", "wiki"}

    def test_service_recompute_confidence_from_persisted_evidence(
        self, service: MemoryService
    ) -> None:
        t = datetime(2026, 1, 10, tzinfo=UTC)
        entity_id = service.repo.insert_entity("Server A", created_at=t)
        msg1 = service.repo.insert_message("sysadmin_priya", "m1", t, t, "h1")
        msg2 = service.repo.insert_message("wiki", "m2", t, t, "h2")

        mem_id = service.repo.insert_memory(
            NewMemory(
                entity_id=entity_id,
                attribute="python_version",
                value_norm="3.10",
                value_display="3.10",
                context={},
                context_key="",
                valid_from=t,
                valid_to=None,
                closed_at=None,
                status=MemoryStatus.ACTIVE,
                confidence=0.5,  # Stale / outdated confidence
                message_id=msg1,
                source_id="sysadmin_priya",
            ),
            recorded_at=t,
        )

        service.repo.insert_evidence(
            mem_id,
            NewEvidence(source_id="sysadmin_priya", message_id=msg1, strength=0.855, observed_at=t),
        )
        service.repo.insert_evidence(
            mem_id,
            NewEvidence(source_id="wiki", message_id=msg2, strength=0.7125, observed_at=t),
        )

        recomputed_conf = service.recompute_confidence(mem_id)
        assert recomputed_conf == pytest.approx(0.958, abs=0.001)

        mem = service.repo.get_memory(mem_id)
        assert mem is not None
        assert mem.confidence == pytest.approx(0.958, abs=0.001)

    def test_service_add_evidence_invalid_memory_raises_not_found(
        self, service: MemoryService
    ) -> None:
        t = datetime(2026, 1, 10, tzinfo=UTC)
        ev = NewEvidence(
            source_id="sysadmin_priya",
            message_id=1,
            strength=0.855,
            observed_at=t,
        )
        with pytest.raises(MemoryNotFoundError, match="Memory with ID 99999 does not exist"):
            service.add_evidence(99999, ev)

    def test_service_add_evidence_unknown_source_raises_unknown_source(
        self, service: MemoryService
    ) -> None:
        t = datetime(2026, 1, 10, tzinfo=UTC)
        entity_id = service.repo.insert_entity("Server A", created_at=t)
        msg_id = service.repo.insert_message("sysadmin_priya", "m1", t, t, "h1")
        mem_id = service.repo.insert_memory(
            NewMemory(
                entity_id=entity_id,
                attribute="python_version",
                value_norm="3.10",
                value_display="3.10",
                context={},
                context_key="",
                valid_from=t,
                valid_to=None,
                closed_at=None,
                status=MemoryStatus.ACTIVE,
                confidence=0.855,
                message_id=msg_id,
                source_id="sysadmin_priya",
            ),
            recorded_at=t,
        )

        ev = NewEvidence(
            source_id="nonexistent_source_xyz",
            message_id=msg_id,
            strength=0.5,
            observed_at=t,
        )
        with pytest.raises(UnknownSourceError, match="nonexistent_source_xyz"):
            service.add_evidence(mem_id, ev)


class TestHeroScenarioEndToEndCorroboration:
    """End-to-end tests reproducing Hero Scenario rows m1, m2, and duplicate handling."""

    def test_hero_scenario_m1_m2_corroboration_flow(self, service: MemoryService) -> None:
        t1 = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 2, 2, 10, 0, tzinfo=UTC)

        # m1: "Server A runs Python 3.10." (priya 0.90, 10 Jan) -> R-INSERT, conf = 0.855
        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs Python 3.10.",
            event_time=t1,
        )
        claim1 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.10",
            extract_confidence=0.95,
        )
        rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
        assert len(rep1.decisions) == 1
        assert rep1.decisions[0].rule_id == "R-INSERT"
        assert rep1.decisions[0].scores["new_score"] == pytest.approx(0.855, abs=0.001)

        m1_id = rep1.decisions[0].memory_id
        assert m1_id is not None
        mem1 = service.repo.get_memory(m1_id)
        assert mem1 is not None
        assert mem1.confidence == pytest.approx(0.855, abs=0.001)

        # m2: "Runtime for Server A: Python 3.10" (wiki 0.75, 2 Feb) -> R-CORROBORATE, conf = 0.958
        msg2 = IngestMessage(
            source_id="wiki",
            text="Runtime for Server A: Python 3.10",
            event_time=t2,
        )
        claim2 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.10",
            extract_confidence=0.95,
        )
        rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
        assert len(rep2.decisions) == 1
        assert rep2.decisions[0].action == Action.CORROBORATE
        assert rep2.decisions[0].rule_id == "R-CORROBORATE"
        assert rep2.decisions[0].scores["confidence"] == pytest.approx(0.958, abs=0.001)

        # Exactly one active memory remains in the database
        ent = service.repo.get_entity_by_name("Server A")
        assert ent is not None
        mems = service.repo.find_memories(entity_id=int(ent["id"]), attribute="python_version")
        assert len(mems) == 1
        assert mems[0].id == m1_id
        assert mems[0].confidence == pytest.approx(0.958, abs=0.001)

        # Two evidence rows associated with the memory
        ev_list = service.repo.get_evidence_for_memory(m1_id)
        assert len(ev_list) == 2
        assert {e.source_id for e in ev_list} == {"sysadmin_priya", "wiki"}

    def test_same_source_duplicate_claim_r_dup_same_source(self, service: MemoryService) -> None:
        t1 = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 1, 20, 10, 0, tzinfo=UTC)

        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs Python 3.10.",
            event_time=t1,
        )
        claim1 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.10",
            extract_confidence=0.95,
        )
        rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
        m1_id = rep1.decisions[0].memory_id
        assert m1_id is not None

        # Same source makes same claim in a different message
        msg2 = IngestMessage(
            source_id="sysadmin_priya",
            text="Re-confirming Server A runs Python 3.10.",
            event_time=t2,
        )
        claim2 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.10",
            extract_confidence=0.95,
        )
        rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
        assert rep2.decisions[0].action == Action.CORROBORATE
        assert rep2.decisions[0].rule_id == "R-DUP-SAME-SOURCE"
        # Confidence remains at initial 0.855 without boost
        assert rep2.decisions[0].scores["confidence"] == pytest.approx(0.855, abs=0.001)

        mem = service.repo.get_memory(m1_id)
        assert mem is not None
        assert mem.confidence == pytest.approx(0.855, abs=0.001)

        # Evidence row was recorded
        ev_list = service.repo.get_evidence_for_memory(m1_id)
        assert len(ev_list) == 2

    def test_corroborate_on_multi_valued_attribute(self, service: MemoryService) -> None:
        t1 = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 1, 20, 10, 0, tzinfo=UTC)

        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs redis.",
            event_time=t1,
        )
        claim1 = Claim(
            subject="Server A",
            attribute="runs_service",
            value="redis",
            extract_confidence=0.95,
        )
        rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
        m1_id = rep1.decisions[0].memory_id
        assert m1_id is not None

        # Wiki corroborates redis on multi-valued attribute
        msg2 = IngestMessage(
            source_id="wiki",
            text="Redis runs on Server A.",
            event_time=t2,
        )
        claim2 = Claim(
            subject="Server A",
            attribute="runs_service",
            value="redis",
            extract_confidence=0.95,
        )
        rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
        assert rep2.decisions[0].action == Action.CORROBORATE
        assert rep2.decisions[0].rule_id == "R-CORROBORATE"

        # Exactly 1 memory exists for redis, with confidence 0.958
        ent = service.repo.get_entity_by_name("Server A")
        assert ent is not None
        mems = service.repo.find_memories(entity_id=int(ent["id"]), attribute="runs_service")
        assert len(mems) == 1
        assert mems[0].confidence == pytest.approx(0.958, abs=0.001)


class TestEvidenceAcrossAllResolutionDecisions:
    """Verify that every applicable memory creation or corroboration persists correct evidence."""

    def test_evidence_persisted_for_r_replace(self, service: MemoryService) -> None:
        t1 = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 2, 10, 10, 0, tzinfo=UTC)

        msg1 = IngestMessage(source_id="sysadmin_priya", text="Python 3.10", event_time=t1)
        c1 = Claim(
            subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
        )
        rep1 = service.ingest_claims(msg1, [c1], recorded_at=t1)
        m1_id = rep1.decisions[0].memory_id
        assert m1_id is not None

        # Verify initial evidence for M1
        ev1_list = service.get_memory_evidence(m1_id)
        assert len(ev1_list) == 1
        assert ev1_list[0].memory_id == m1_id
        assert ev1_list[0].source_id == "sysadmin_priya"
        assert ev1_list[0].strength == pytest.approx(0.855, abs=0.001)

        # Newer conflicting claim triggers R-REPLACE
        msg2 = IngestMessage(source_id="wiki", text="Python 3.12", event_time=t2)
        c2 = Claim(
            subject="Server A", attribute="python_version", value="3.12", extract_confidence=0.95
        )
        rep2 = service.ingest_claims(msg2, [c2], recorded_at=t2)
        assert rep2.decisions[0].rule_id == "R-REPLACE"
        m2_id = rep2.decisions[0].memory_id
        assert m2_id is not None
        assert m2_id != m1_id

        # Replacement memory M2 must have its own evidence persisted
        ev2_list = service.get_memory_evidence(m2_id)
        assert len(ev2_list) == 1
        assert ev2_list[0].memory_id == m2_id
        assert ev2_list[0].source_id == "wiki"
        assert ev2_list[0].strength == pytest.approx(0.7125, abs=0.001)

        # Closed old memory M1 must still retain its original evidence intact (never overwritten)
        ev1_list_after = service.get_memory_evidence(m1_id)
        assert len(ev1_list_after) == 1
        assert ev1_list_after[0].id == ev1_list[0].id

    def test_evidence_persisted_for_r_coexist(self, service: MemoryService) -> None:
        t1 = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)

        msg1 = IngestMessage(source_id="sysadmin_priya", text="Runs nginx", event_time=t1)
        c1 = Claim(
            subject="Server A", attribute="runs_service", value="nginx", extract_confidence=0.95
        )
        rep1 = service.ingest_claims(msg1, [c1], recorded_at=t1)
        m1_id = rep1.decisions[0].memory_id
        assert m1_id is not None

        # Distinct value on multi-valued attribute triggers R-COEXIST
        msg2 = IngestMessage(source_id="dev_chat", text="Runs postgres", event_time=t2)
        c2 = Claim(
            subject="Server A", attribute="runs_service", value="postgres", extract_confidence=0.90
        )
        rep2 = service.ingest_claims(msg2, [c2], recorded_at=t2)
        assert rep2.decisions[0].rule_id == "R-COEXIST"
        m2_id = rep2.decisions[0].memory_id
        assert m2_id is not None

        # Both memories must have their corresponding evidence
        ev1_list = service.get_memory_evidence(m1_id)
        assert len(ev1_list) == 1
        assert ev1_list[0].source_id == "sysadmin_priya"

        ev2_list = service.get_memory_evidence(m2_id)
        assert len(ev2_list) == 1
        assert ev2_list[0].source_id == "dev_chat"
        # dev_chat (0.50) * extract (0.90) = 0.45
        assert ev2_list[0].strength == pytest.approx(0.45, abs=0.001)

    def test_evidence_persisted_for_r_refine(self, service: MemoryService) -> None:
        t1 = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 2, 10, 10, 0, tzinfo=UTC)

        msg1 = IngestMessage(source_id="wiki", text="Python 3.12", event_time=t1)
        c1 = Claim(
            subject="Server A", attribute="python_version", value="3.12", extract_confidence=0.95
        )
        rep1 = service.ingest_claims(msg1, [c1], recorded_at=t1)
        m1_id = rep1.decisions[0].memory_id
        assert m1_id is not None

        # More specific patch version triggers R-REFINE
        msg2 = IngestMessage(source_id="sysadmin_priya", text="Python 3.12.4", event_time=t2)
        c2 = Claim(
            subject="Server A", attribute="python_version", value="3.12.4", extract_confidence=0.95
        )
        rep2 = service.ingest_claims(msg2, [c2], recorded_at=t2)
        assert rep2.decisions[0].rule_id == "R-REFINE"
        m2_id = rep2.decisions[0].memory_id
        assert m2_id is not None

        # Both memories have their own evidence preserved
        ev1_list = service.get_memory_evidence(m1_id)
        assert len(ev1_list) == 1
        assert ev1_list[0].source_id == "wiki"

        ev2_list = service.get_memory_evidence(m2_id)
        assert len(ev2_list) == 1
        assert ev2_list[0].source_id == "sysadmin_priya"
        assert ev2_list[0].strength == pytest.approx(0.855, abs=0.001)

    def test_recompute_matches_base_confidence_across_all_decisions(
        self, service: MemoryService
    ) -> None:
        """Verify evidence persistence and recomputation across all six resolution decisions."""
        t1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        t2 = datetime(2026, 1, 15, 0, 0, tzinfo=UTC)
        t3 = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
        t4 = datetime(2026, 2, 15, 0, 0, tzinfo=UTC)
        t5 = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
        t6 = datetime(2026, 3, 15, 0, 0, tzinfo=UTC)

        # ---------------------------------------------------------------------
        # 1. R-INSERT: Initial claim for Server A python_version
        # ---------------------------------------------------------------------
        msg1 = IngestMessage(source_id="sysadmin_priya", text="Python 3.10", event_time=t1)
        c1 = Claim(
            subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
        )
        rep1 = service.ingest_claims(msg1, [c1], recorded_at=t1)
        assert rep1.decisions[0].rule_id == "R-INSERT"
        m1_id = rep1.decisions[0].memory_id
        assert m1_id is not None

        mem1 = service.repo.get_memory(m1_id)
        assert mem1 is not None
        # priya (0.90) * extract (0.95) = 0.855
        assert mem1.confidence == pytest.approx(0.855, abs=0.0001)

        # Evidence records check
        ev1 = service.get_memory_evidence(m1_id)
        assert len(ev1) == 1
        assert ev1[0].source_id == "sysadmin_priya"
        assert ev1[0].strength == pytest.approx(0.855, abs=0.0001)

        # Recomputed confidence matches stored base confidence
        assert service.recompute_confidence(m1_id) == pytest.approx(mem1.confidence, abs=0.0001)

        # ---------------------------------------------------------------------
        # 2. R-CORROBORATE: Different source (wiki) corroborates M1
        # ---------------------------------------------------------------------
        msg2 = IngestMessage(source_id="wiki", text="Python 3.10", event_time=t2)
        c2 = Claim(
            subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
        )
        rep2 = service.ingest_claims(msg2, [c2], recorded_at=t2)
        assert rep2.decisions[0].rule_id == "R-CORROBORATE"

        mem1_corrob = service.repo.get_memory(m1_id)
        assert mem1_corrob is not None
        # 1 - (1 - 0.855) * (1 - 0.7125) = 0.9583125
        assert mem1_corrob.confidence == pytest.approx(0.9583, abs=0.001)

        # Evidence rows check (both priya and wiki preserved)
        ev1_corrob = service.get_memory_evidence(m1_id)
        assert len(ev1_corrob) == 2
        assert {e.source_id for e in ev1_corrob} == {"sysadmin_priya", "wiki"}

        # Recomputed confidence matches stored base confidence
        assert service.recompute_confidence(m1_id) == pytest.approx(
            mem1_corrob.confidence, abs=0.0001
        )

        # ---------------------------------------------------------------------
        # 3. R-DUP-SAME-SOURCE: Repeated statement from wiki (same source)
        # ---------------------------------------------------------------------
        msg3 = IngestMessage(source_id="wiki", text="Confirm Python 3.10", event_time=t3)
        c3 = Claim(
            subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
        )
        rep3 = service.ingest_claims(msg3, [c3], recorded_at=t3)
        assert rep3.decisions[0].rule_id == "R-DUP-SAME-SOURCE"

        mem1_dup = service.repo.get_memory(m1_id)
        assert mem1_dup is not None
        # Same source does NOT boost confidence
        assert mem1_dup.confidence == pytest.approx(mem1_corrob.confidence, abs=0.0001)

        # 3 evidence rows recorded (audit provenance preserved)
        ev1_dup = service.get_memory_evidence(m1_id)
        assert len(ev1_dup) == 3

        # Recomputed confidence preserves base confidence exactly
        assert service.recompute_confidence(m1_id) == pytest.approx(mem1_dup.confidence, abs=0.0001)

        # ---------------------------------------------------------------------
        # 4. R-REPLACE: Newer conflicting claim replaces M1
        # ---------------------------------------------------------------------
        msg4 = IngestMessage(source_id="wiki", text="Migrated to Python 3.12", event_time=t4)
        c4 = Claim(
            subject="Server A", attribute="python_version", value="3.12", extract_confidence=0.95
        )
        rep4 = service.ingest_claims(msg4, [c4], recorded_at=t4)
        assert rep4.decisions[0].rule_id == "R-REPLACE"
        m2_id = rep4.decisions[0].memory_id
        assert m2_id is not None
        assert m2_id != m1_id

        mem2 = service.repo.get_memory(m2_id)
        assert mem2 is not None
        # wiki (0.75) * extract (0.95) = 0.7125
        assert mem2.confidence == pytest.approx(0.7125, abs=0.0001)

        # Evidence for M2 persisted
        ev2 = service.get_memory_evidence(m2_id)
        assert len(ev2) == 1
        assert ev2[0].source_id == "wiki"
        assert ev2[0].strength == pytest.approx(0.7125, abs=0.0001)

        # Existing evidence for replaced M1 is preserved intact
        ev1_after_replace = service.get_memory_evidence(m1_id)
        assert len(ev1_after_replace) == 3

        # Recomputed confidence for both matches stored base confidence
        assert service.recompute_confidence(m2_id) == pytest.approx(mem2.confidence, abs=0.0001)
        assert service.recompute_confidence(m1_id) == pytest.approx(mem1_dup.confidence, abs=0.0001)

        # ---------------------------------------------------------------------
        # 5. R-REFINE: More specific patch version refines M2
        # ---------------------------------------------------------------------
        msg5 = IngestMessage(source_id="sysadmin_priya", text="Python 3.12.4", event_time=t5)
        c5 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12.4",
            extract_confidence=0.95,
        )
        rep5 = service.ingest_claims(msg5, [c5], recorded_at=t5)
        assert rep5.decisions[0].rule_id == "R-REFINE"
        m3_id = rep5.decisions[0].memory_id
        assert m3_id is not None

        mem3 = service.repo.get_memory(m3_id)
        assert mem3 is not None
        # priya (0.90) * extract (0.95) = 0.855
        assert mem3.confidence == pytest.approx(0.855, abs=0.0001)

        # Evidence for refined memory persisted
        ev3 = service.get_memory_evidence(m3_id)
        assert len(ev3) == 1
        assert ev3[0].source_id == "sysadmin_priya"
        assert ev3[0].strength == pytest.approx(0.855, abs=0.0001)

        # Prior evidence for refined memory M2 is preserved intact
        assert len(service.get_memory_evidence(m2_id)) == 1

        # Recomputed confidence matches stored base confidence
        assert service.recompute_confidence(m3_id) == pytest.approx(mem3.confidence, abs=0.0001)

        # ---------------------------------------------------------------------
        # 6. R-COEXIST: Distinct values on multi-valued attribute
        # ---------------------------------------------------------------------
        # Initial insertion on runs_service
        msg6a = IngestMessage(source_id="sysadmin_priya", text="Runs redis", event_time=t6)
        c6a = Claim(
            subject="Server A", attribute="runs_service", value="redis", extract_confidence=0.95
        )
        rep6a = service.ingest_claims(msg6a, [c6a], recorded_at=t6)
        assert rep6a.decisions[0].rule_id == "R-INSERT"
        m4_id = rep6a.decisions[0].memory_id
        assert m4_id is not None

        # Coexisting distinct service on runs_service
        msg6b = IngestMessage(source_id="dev_chat", text="Runs postgres", event_time=t6)
        c6b = Claim(
            subject="Server A",
            attribute="runs_service",
            value="postgres",
            extract_confidence=0.90,
        )
        rep6b = service.ingest_claims(msg6b, [c6b], recorded_at=t6)
        assert rep6b.decisions[0].rule_id == "R-COEXIST"
        m5_id = rep6b.decisions[0].memory_id
        assert m5_id is not None

        mem5 = service.repo.get_memory(m5_id)
        assert mem5 is not None
        # dev_chat (0.50) * extract (0.90) = 0.45
        assert mem5.confidence == pytest.approx(0.45, abs=0.0001)

        # Evidence for coexisting memory persisted
        ev5 = service.get_memory_evidence(m5_id)
        assert len(ev5) == 1
        assert ev5[0].source_id == "dev_chat"
        assert ev5[0].strength == pytest.approx(0.45, abs=0.0001)

        # Evidence for first coexisting memory M4 preserved intact
        assert len(service.get_memory_evidence(m4_id)) == 1

        # Recomputed confidence matches stored base confidence
        assert service.recompute_confidence(m5_id) == pytest.approx(mem5.confidence, abs=0.0001)
        assert service.recompute_confidence(m4_id) == pytest.approx(
            service.repo.get_memory(m4_id).confidence,
            abs=0.0001,  # type: ignore[union-attr]
        )


class TestBaseConfidenceIntegrityAndDecayProtection:
    """Verify base confidence is protected from repeated queries, decay, and review mutations."""

    def test_repeated_queries_do_not_mutate_base_confidence(self, service: MemoryService) -> None:
        # Register an attribute with 14-day half-life
        service.repo.insert_attribute_schema(
            AttributeDef(
                attribute="service_status",
                description="Status",
                cardinality=Cardinality.SINGLE,
                value_type=ValueType.ENUM,
                half_life_days=14.0,
                allowed_values=["up", "down"],
            )
        )
        t_valid = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        msg = IngestMessage(source_id="sysadmin_priya", text="Status is up", event_time=t_valid)
        c = Claim(
            subject="Server A", attribute="service_status", value="up", extract_confidence=0.95
        )
        rep = service.ingest_claims(msg, [c], recorded_at=t_valid)
        mem_id = rep.decisions[0].memory_id
        assert mem_id is not None

        initial_mem = service.repo.get_memory(mem_id)
        assert initial_mem is not None
        base_conf = initial_mem.confidence
        assert base_conf == pytest.approx(0.855, abs=0.001)

        # 1. Query at t_valid + 14 days (decay = 0.5)
        t_14d = datetime(2026, 1, 15, 0, 0, tzinfo=UTC)
        res1 = service.current("Server A", "service_status", now=t_14d)
        assert len(res1.items) == 1
        # Effective confidence is halved
        assert res1.items[0].effective_confidence == pytest.approx(base_conf * 0.5, abs=0.001)
        # Base confidence in memory view remains 0.855
        assert res1.items[0].memory.confidence == pytest.approx(base_conf, abs=0.001)

        # 2. Query again 100 times at t_valid + 28 days (decay = 0.25)
        t_28d = datetime(2026, 1, 29, 0, 0, tzinfo=UTC)
        for _ in range(10):
            res2 = service.current("Server A", "service_status", now=t_28d)
            assert res2.items[0].effective_confidence == pytest.approx(base_conf * 0.25, abs=0.001)

        # 3. Verify database persisted memory confidence has NOT changed
        persisted_mem = service.repo.get_memory(mem_id)
        assert persisted_mem is not None
        assert persisted_mem.confidence == pytest.approx(base_conf, abs=0.001)

    def test_review_penalty_applies_only_to_effective_confidence(
        self, service: MemoryService
    ) -> None:
        t = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        msg = IngestMessage(source_id="sysadmin_priya", text="Python 3.10", event_time=t)
        c = Claim(
            subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
        )
        rep = service.ingest_claims(msg, [c], recorded_at=t)
        mem_id = rep.decisions[0].memory_id
        assert mem_id is not None

        # Flag memory for review
        service.repo.update_memory(mem_id, {"needs_review": True, "review_reason": "flagged"})

        # Memory base confidence remains 0.855
        mem = service.repo.get_memory(mem_id)
        assert mem is not None
        assert mem.confidence == pytest.approx(0.855, abs=0.001)
        assert mem.needs_review is True

        # Query effective confidence applies 0.50 review penalty
        res = service.current("Server A", "python_version", now=t)
        assert len(res.items) == 1
        assert res.items[0].effective_confidence == pytest.approx(0.855 * 0.50, abs=0.001)
        assert res.items[0].memory.confidence == pytest.approx(0.855, abs=0.001)

        # Recomputing base confidence still produces 0.855 (not penalized)
        recomputed = service.recompute_confidence(mem_id)
        assert recomputed == pytest.approx(0.855, abs=0.001)

    def test_historical_retrieval_effective_confidence_uses_caller_reference_time(
        self, service: MemoryService
    ) -> None:
        # Register decaying attribute
        service.repo.insert_attribute_schema(
            AttributeDef(
                attribute="decay_attr",
                description="Decays",
                cardinality=Cardinality.SINGLE,
                value_type=ValueType.TEXT,
                half_life_days=10.0,
            )
        )
        t_valid = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        msg = IngestMessage(source_id="sysadmin_priya", text="Initial val", event_time=t_valid)
        c = Claim(subject="Server A", attribute="decay_attr", value="v1", extract_confidence=0.95)
        rep = service.ingest_claims(msg, [c], recorded_at=t_valid)
        mem_id = rep.decisions[0].memory_id
        assert mem_id is not None

        # At t_valid + 10 days: decay factor is 0.5
        t_hist_10d = datetime(2026, 1, 11, 0, 0, tzinfo=UTC)
        # Even if "now" is in 2027, historical retrieval as-of t_hist_10d uses t_hist_10d
        t_future = datetime(2027, 1, 1, 0, 0, tzinfo=UTC)
        res_hist = service.current(
            "Server A",
            "decay_attr",
            valid_at=t_hist_10d,
            now=t_future,
        )
        assert len(res_hist.items) == 1
        assert res_hist.items[0].effective_confidence == pytest.approx(0.855 * 0.5, abs=0.001)
        assert res_hist.items[0].is_stale is False

        # At t_valid + 100 days: decay factor is 2^(-10) = 0.0009765625
        t_hist_100d = datetime(2026, 4, 11, 0, 0, tzinfo=UTC)
        res_hist_stale = service.current(
            "Server A",
            "decay_attr",
            valid_at=t_hist_100d,
            now=t_future,
        )
        assert len(res_hist_stale.items) == 1
        assert res_hist_stale.items[0].is_stale is True

    def test_configured_staleness_threshold_drives_retrieval(self, service: MemoryService) -> None:
        t = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        msg = IngestMessage(source_id="sysadmin_priya", text="Python 3.10", event_time=t)
        c = Claim(
            subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
        )
        service.ingest_claims(msg, [c], recorded_at=t)

        # Baseline: base confidence is 0.855, effective confidence is 0.855
        # Under default configured stale_floor = 0.30 -> not stale
        res_default = service.current("Server A", "python_version", now=t)
        assert res_default.items[0].is_stale is False

        # Alter engine configuration to strict stale_floor = 0.90 -> is_stale becomes True
        service.cfg.engine.thresholds.stale_floor = 0.90
        res_strict = service.current("Server A", "python_version", now=t)
        assert res_strict.items[0].is_stale is True

        # Alter engine configuration to permissive stale_floor = 0.50 -> is_stale becomes False
        service.cfg.engine.thresholds.stale_floor = 0.50
        res_permissive = service.current("Server A", "python_version", now=t)
        assert res_permissive.items[0].is_stale is False

    def test_get_effective_confidence_clock_and_decay_semantics(self, repo: Repository) -> None:
        """Verify get_effective_confidence with explicit/omitted timestamps and clock."""
        # Attribute with 14-day half-life
        repo.insert_attribute_schema(
            AttributeDef(
                attribute="exp_status",
                description="Status",
                cardinality=Cardinality.SINGLE,
                value_type=ValueType.ENUM,
                half_life_days=14.0,
                allowed_values=["up", "down"],
            )
        )
        t_valid = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        current_time = datetime(2026, 1, 15, 0, 0, tzinfo=UTC)  # 14 days later (1 half-life)

        # Service with injected clock
        service_clock = MemoryService(repo, clock=lambda: current_time)

        msg = IngestMessage(source_id="sysadmin_priya", text="Status up", event_time=t_valid)
        c = Claim(subject="Server A", attribute="exp_status", value="up", extract_confidence=0.95)
        rep = service_clock.ingest_claims(msg, [c], recorded_at=t_valid)
        mem_id = rep.decisions[0].memory_id
        assert mem_id is not None

        mem = repo.get_memory(mem_id)
        assert mem is not None
        base_conf = mem.confidence
        assert base_conf == pytest.approx(0.855, abs=0.001)

        # 1. Explicit reference timestamp is respected
        t_28d = datetime(2026, 1, 29, 0, 0, tzinfo=UTC)  # 28 days = 2 half-lives
        eff_explicit = service_clock.get_effective_confidence(mem_id, at=t_28d)
        assert eff_explicit == pytest.approx(base_conf * 0.25, abs=0.001)

        # 2. Omitted reference timestamp uses the injected clock (current_time = 14 days)
        eff_omitted = service_clock.get_effective_confidence(mem_id, at=None)
        assert eff_omitted == pytest.approx(base_conf * 0.50, abs=0.001)

        # 3. Old memory with configured half-life receives expected decay (3 half-lives = 42 days)
        t_42d = datetime(2026, 2, 12, 0, 0, tzinfo=UTC)
        eff_42d = service_clock.get_effective_confidence(mem_id, at=t_42d)
        assert eff_42d == pytest.approx(base_conf * 0.125, abs=0.001)

        # 4. Calling get_effective_confidence does NOT modify stored base confidence in DB
        persisted_after = repo.get_memory(mem_id)
        assert persisted_after is not None
        assert persisted_after.confidence == pytest.approx(base_conf, abs=0.0001)

    def test_get_effective_confidence_rejects_naive_injected_clock(self, repo: Repository) -> None:
        """Verify get_effective_confidence raises ValueError when injected clock is naive."""
        from datetime import tzinfo

        t_valid = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        naive_clock_service = MemoryService(repo, clock=lambda: datetime(2026, 1, 15, 0, 0))

        msg = IngestMessage(source_id="sysadmin_priya", text="Python 3.10", event_time=t_valid)
        c = Claim(
            subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
        )
        rep = naive_clock_service.ingest_claims(msg, [c], recorded_at=t_valid)
        mem_id = rep.decisions[0].memory_id
        assert mem_id is not None

        # 1. Naive datetime from clock must raise ValueError
        with pytest.raises(ValueError, match="Clock must return a timezone-aware datetime"):
            naive_clock_service.get_effective_confidence(mem_id, at=None)

        # 2. Tzinfo exists but utcoffset() returns None
        class NullOffsetTz(tzinfo):
            def utcoffset(self, dt: datetime | None) -> None:
                return None

            def tzname(self, dt: datetime | None) -> str | None:
                return "NullOffset"

            def dst(self, dt: datetime | None) -> None:
                return None

        null_offset_service = MemoryService(
            repo, clock=lambda: datetime(2026, 1, 15, 0, 0, tzinfo=NullOffsetTz())
        )
        with pytest.raises(ValueError, match="Clock must return a timezone-aware datetime"):
            null_offset_service.get_effective_confidence(mem_id, at=None)

        # 3. Supplying an explicit timezone-aware timestamp bypasses the clock and succeeds
        eff_valid = naive_clock_service.get_effective_confidence(mem_id, at=t_valid)
        assert eff_valid == pytest.approx(0.855, abs=0.001)
        eff_14d = naive_clock_service.get_effective_confidence(
            mem_id, at=datetime(2026, 1, 15, 0, 0, tzinfo=UTC)
        )
        assert eff_14d == pytest.approx(0.855 * (2.0 ** (-14.0 / 365.0)), abs=0.001)
