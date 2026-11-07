"""Tests for deterministic two-tier entity resolution and alias safety.

Governed by docs/ARCHITECTURE.md Section 3 (Stage 2), docs/ROADMAP_AND_TASKS.md (T2.4),
and config/entities.yaml.
P2.2 Implementation: Part C.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memorymesh.db import init_db
from memorymesh.engine.entity import (
    EntityCandidate,
    EntityResolutionAction,
    normalize_entity_name,
    resolve_entity,
)
from memorymesh.errors import AliasCollisionError
from memorymesh.models import (
    Action,
    AttributeDef,
    AttrRef,
    Cardinality,
    Claim,
    EdgeType,
    IngestMessage,
    Source,
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


class TestEntityNormalization:
    """Tests for pure entity name normalization."""

    def test_whitespace_and_casing_normalization(self) -> None:
        assert normalize_entity_name("Server A") == "server a"
        assert normalize_entity_name("  server a  ") == "server a"
        assert normalize_entity_name("Server \t  A") == "server a"
        assert normalize_entity_name("SRV-A") == "srv-a"
        assert normalize_entity_name("  The   Main   API   Box  ") == "the main api box"


class TestEntityCorePureLogic:
    """Pure logic tests for engine/entity.py without any database or clock dependencies."""

    def test_pure_resolution_works_without_database(self) -> None:
        candidates = [
            EntityCandidate(
                canonical_name="Server A",
                entity_type="server",
                aliases=["srv-a", "server-a", "the main API box"],
                db_id=1,
            ),
            EntityCandidate(
                canonical_name="Library X",
                entity_type="library",
                aliases=["libx", "lib-x"],
                db_id=2,
            ),
        ]

        # 1. Exact canonical match
        res_canon = resolve_entity("Server A", candidates)
        assert res_canon.action == EntityResolutionAction.EXACT_MATCH
        assert res_canon.matched_entity_id == 1
        assert res_canon.canonical_name == "Server A"
        assert res_canon.needs_review is False

        # 1b. Exact alias match
        res_alias = resolve_entity("srv-a", candidates)
        assert res_alias.action == EntityResolutionAction.EXACT_MATCH
        assert res_alias.matched_entity_id == 1
        assert res_alias.canonical_name == "Server A"
        assert res_alias.needs_review is False

        # Case-insensitive whitespace match
        res_ws = resolve_entity("  The   Main   API   Box  ", candidates)
        assert res_ws.action == EntityResolutionAction.EXACT_MATCH
        assert res_ws.matched_entity_id == 1

        # High-confidence single fuzzy match (score >= 90)
        res_fuzzy = resolve_entity("Server A box", candidates)
        assert res_fuzzy.action == EntityResolutionAction.FUZZY_MATCH
        assert res_fuzzy.matched_entity_id == 1
        assert res_fuzzy.score >= 90.0
        assert res_fuzzy.needs_review is False

        # Single review-level candidate (75 <= score < 90)
        res_review = resolve_entity("Server B", candidates)
        assert res_review.action == EntityResolutionAction.REVIEW_MATCH
        assert res_review.matched_entity_id is None
        assert res_review.needs_review is True
        assert 75.0 <= res_review.score < 90.0

        # Below review threshold (< 75)
        res_unknown = resolve_entity("Cluster Z", candidates)
        assert res_unknown.action == EntityResolutionAction.UNKNOWN
        assert res_unknown.matched_entity_id is None
        assert res_unknown.needs_review is False
        assert res_unknown.score < 75.0

    def test_two_entities_sharing_same_alias_is_ambiguous(self) -> None:
        """2. Two entities sharing the same normalized alias produce an ambiguous result."""
        candidates = [
            EntityCandidate(
                canonical_name="Server A",
                entity_type="server",
                aliases=["shared-box"],
                db_id=1,
            ),
            EntityCandidate(
                canonical_name="Server B",
                entity_type="server",
                aliases=["shared-box"],
                db_id=2,
            ),
        ]

        res = resolve_entity("shared-box", candidates)
        assert res.action == EntityResolutionAction.AMBIGUOUS_MATCH
        assert res.matched_entity_id is None
        assert res.needs_review is True
        assert res.canonical_name == "shared-box"

    def test_exact_canonical_and_alias_collision_across_entities_is_ambiguous(self) -> None:
        """3. Exact canonical-name/alias collision across entities produces ambiguous result."""
        candidates = [
            EntityCandidate(
                canonical_name="Server Alpha",
                entity_type="server",
                aliases=["alpha-node"],
                db_id=1,
            ),
            EntityCandidate(
                canonical_name="Alpha-Node",
                entity_type="server",
                aliases=[],
                db_id=2,
            ),
        ]

        res = resolve_entity("alpha-node", candidates)
        assert res.action == EntityResolutionAction.AMBIGUOUS_MATCH
        assert res.matched_entity_id is None
        assert res.needs_review is True

    def test_reversing_candidate_order_does_not_change_exact_collision_result(self) -> None:
        """4. Reversing candidate order does not change the result."""
        cand_a = EntityCandidate(
            canonical_name="Server One",
            entity_type="server",
            aliases=["box-x"],
            db_id=1,
        )
        cand_b = EntityCandidate(
            canonical_name="Server Two",
            entity_type="server",
            aliases=["box-x"],
            db_id=2,
        )

        res_forward = resolve_entity("box-x", [cand_a, cand_b])
        res_reverse = resolve_entity("box-x", [cand_b, cand_a])

        assert res_forward.action == EntityResolutionAction.AMBIGUOUS_MATCH
        assert res_reverse.action == EntityResolutionAction.AMBIGUOUS_MATCH
        assert res_forward.matched_entity_id is None
        assert res_reverse.matched_entity_id is None
        assert res_forward.needs_review is True
        assert res_reverse.needs_review is True

    def test_equal_score_candidates_above_threshold_trigger_ambiguous_review(self) -> None:
        """Equal top scores above auto_match threshold must create review entity, not auto-merge."""
        candidates = [
            EntityCandidate(
                canonical_name="Worker Alpha",
                entity_type="service",
                aliases=["worker-a"],
                db_id=10,
            ),
            EntityCandidate(
                canonical_name="Worker Beta",
                entity_type="service",
                aliases=["worker-b"],
                db_id=20,
            ),
        ]

        # "Worker" scores identically against both (score 100.0 >= 90.0)
        res = resolve_entity("Worker", candidates, auto_match_threshold=90.0)
        assert res.action == EntityResolutionAction.AMBIGUOUS_MATCH
        assert res.matched_entity_id is None
        assert res.needs_review is True
        assert res.canonical_name == "Worker"

    def test_equal_score_candidates_within_review_band_trigger_ambiguous_review(self) -> None:
        """Equal top scores within review band must create review entity without auto-merge."""
        candidates = [
            EntityCandidate(
                canonical_name="Payment Server Alpha",
                entity_type="service",
                aliases=[],
                db_id=1,
            ),
            EntityCandidate(
                canonical_name="Payment Server Beta",
                entity_type="service",
                aliases=[],
                db_id=2,
            ),
        ]

        # "Payment Server Node" scores identically (84.85) against both in review band [75, 90)
        res = resolve_entity("Payment Server Node", candidates, auto_match_threshold=90.0)
        assert res.action == EntityResolutionAction.AMBIGUOUS_MATCH
        assert res.matched_entity_id is None
        assert res.needs_review is True
        assert 75.0 <= res.score < 90.0

    def test_empty_candidates_returns_unknown(self) -> None:
        res = resolve_entity("New System", [])
        assert res.action == EntityResolutionAction.UNKNOWN
        assert res.matched_entity_id is None
        assert res.needs_review is False


class TestEntityRepositoryAliasSafety:
    """Tests for repository-level alias ownership preservation and collision detection."""

    def test_alias_cannot_be_reassigned_to_different_entity(self) -> None:
        """Verify inserting an alias already owned by Entity A cannot reassign it to Entity B."""
        conn = init_db(":memory:")
        repo = Repository(conn)
        t = datetime(2026, 1, 1, tzinfo=UTC)

        id1 = repo.insert_entity("Server Alpha", created_at=t)
        id2 = repo.insert_entity("Server Beta", created_at=t)

        # Entity 1 claims alias "main-box"
        success = repo.insert_entity_alias("main-box", id1)
        assert success is True

        # Idempotent call for Entity 1 succeeds
        assert repo.insert_entity_alias("main-box", id1) is False

        # Attempt to transfer alias to Entity 2 raises AliasCollisionError
        with pytest.raises(AliasCollisionError, match="already assigned to entity 1"):
            repo.insert_entity_alias("main-box", id2)

        # Verify alias ownership remained unchanged
        ent = repo.get_entity_by_alias("main-box")
        assert ent is not None
        assert ent["id"] == id1


class TestEntityServiceIntegration:
    """End-to-end integration tests using MemoryService for entity resolution."""

    def test_alias_in_claim_resolves_to_canonical_entity(self, service: MemoryService) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 2, 15, 10, 0, tzinfo=UTC)

        # Ingest claim using alias "srv-a" (from config/entities.yaml)
        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="srv-a runs Python 3.10",
            event_time=t1,
        )
        claim1 = Claim(
            subject="srv-a",
            attribute="python_version",
            value="3.10",
            extract_confidence=0.95,
        )
        report1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
        assert report1.decisions[0].action == Action.INSERT

        # Ingest claim using canonical "Server A"
        msg2 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A upgraded to Python 3.12",
            event_time=t2,
        )
        claim2 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
        # Because both resolved to the same entity, the second replaces the first
        assert report2.decisions[0].action == Action.REPLACE
        assert report2.decisions[0].rule_id == "R-REPLACE"

        # Verify only one canonical entity exists in database
        ents = service.repo.list_entities()
        assert len(ents) == 1
        assert ents[0]["canonical_name"] == "Server A"
        assert ents[0]["needs_review"] == 0

        # Verify retrieval works using either canonical name or alias
        mem_canon = service.get_current("Server A", "python_version", now=t2)
        assert mem_canon is not None
        assert mem_canon.value_norm == "3.12"

        mem_alias = service.get_current("srv-a", "python_version", now=t2)
        assert mem_alias is not None
        assert mem_alias.value_norm == "3.12"

    def test_fuzzy_high_confidence_attaches_to_existing_entity(
        self, service: MemoryService
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 2, 15, 10, 0, tzinfo=UTC)

        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs Python 3.10",
            event_time=t1,
        )
        claim1 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.10",
            extract_confidence=0.95,
        )
        service.ingest_claims(msg1, [claim1], recorded_at=t1)

        # "Server A box" matches "Server A" with score >= 90
        msg2 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A box upgraded to Python 3.12",
            event_time=t2,
        )
        claim2 = Claim(
            subject="Server A box",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
        assert report2.decisions[0].action == Action.REPLACE

        # Only one entity exists in database
        ents = service.repo.list_entities()
        assert len(ents) == 1
        assert ents[0]["canonical_name"] == "Server A"

    def test_service_exact_collision_creates_review_entity_without_reassigning_alias(
        self, service: MemoryService
    ) -> None:
        """5. The service does not reassign original alias and flags the new entity for review."""
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 2, 15, 10, 0, tzinfo=UTC)

        # Pre-seed Entity 1 with alias "shared-node"
        e1_id = service.repo.insert_entity("Cluster Alpha", entity_type="server", created_at=t1)
        service.repo.insert_entity_alias("shared-node", e1_id)

        # Pre-seed Entity 2 whose canonical name collides with Entity 1's alias
        e2_id = service.repo.insert_entity("Shared-Node", entity_type="server", created_at=t1)

        # Ingest claim with subject "shared-node" (exact collision across Candidate 1 & Candidate 2)
        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="shared-node Python is 3.12",
            event_time=t2,
        )
        claim = Claim(
            subject="shared-node",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )

        # Ingesting should NOT raise AliasCollisionError; it creates a separate review entity
        report = service.ingest_claims(msg, [claim], recorded_at=t2)
        assert report.decisions[0].action == Action.INSERT

        # Verify a new entity was created and flagged with needs_review=1
        new_ent_id = report.decisions[0].memory_id
        assert new_ent_id is not None
        mem = service.repo.get_memory(new_ent_id)
        assert mem is not None

        new_ent = service.repo.get_entity(mem.entity_id)
        assert new_ent is not None
        assert new_ent["id"] not in (e1_id, e2_id)
        assert new_ent["needs_review"] == 1

        # Verify alias "shared-node" remained safely owned by Entity 1
        owner = service.repo.get_entity_by_alias("shared-node")
        assert owner is not None
        assert owner["id"] == e1_id

    def test_ambiguous_fuzzy_match_creates_flagged_entity_in_db(
        self, service: MemoryService
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 2, 15, 10, 0, tzinfo=UTC)

        # Pre-seed two database entities that tie on a common token
        service.repo.insert_entity("Worker Alpha", entity_type="service", created_at=t1)
        service.repo.insert_entity("Worker Beta", entity_type="service", created_at=t1)

        # Ingest claim with ambiguous subject "Worker"
        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Worker runs Python 3.12",
            event_time=t2,
        )
        claim = Claim(
            subject="Worker",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report = service.ingest_claims(msg, [claim], recorded_at=t2)
        assert report.decisions[0].action == Action.INSERT

        # Verify a separate entity "Worker" was created with needs_review=1
        worker_ent = service.repo.get_entity_by_name("Worker")
        assert worker_ent is not None
        assert worker_ent["needs_review"] == 1

        # Total 3 entities now exist
        assert len(service.repo.list_entities()) == 3

    def test_review_level_match_creates_flagged_entity_in_db(self, service: MemoryService) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 2, 15, 10, 0, tzinfo=UTC)

        # Ingest Server A
        msg1 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs Python 3.10",
            event_time=t1,
        )
        claim1 = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.10",
            extract_confidence=0.95,
        )
        service.ingest_claims(msg1, [claim1], recorded_at=t1)

        # Ingest "Server B" (score ~87.5 in review band [75, 90))
        msg2 = IngestMessage(
            source_id="sysadmin_priya",
            text="Server B runs Python 3.12",
            event_time=t2,
        )
        claim2 = Claim(
            subject="Server B",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
        assert report2.decisions[0].action == Action.INSERT

        ent_b = service.repo.get_entity_by_name("Server B")
        assert ent_b is not None
        assert ent_b["needs_review"] == 1
        assert len(service.repo.list_entities()) == 2

    def test_dependency_alias_resolution(self, service: MemoryService) -> None:
        """Verify dependency target specified via alias resolves to canonical entity."""
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 2, 15, 10, 0, tzinfo=UTC)

        # Ingest Python version on Server A
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
        service.ingest_claims(msg1, [claim1], recorded_at=t1)

        # Ingest Library X depending on "srv-a"
        msg2 = IngestMessage(
            source_id="sysadmin_priya",
            text="Library X python_version is 3.12 depending on srv-a",
            event_time=t2,
        )
        claim2 = Claim(
            subject="Library X",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
            depends_on=[AttrRef(entity="srv-a", attribute="python_version")],
        )
        report2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
        assert report2.decisions[0].action == Action.INSERT

        new_mem_id = report2.decisions[0].memory_id
        assert new_mem_id is not None
        edges = service.repo.get_edges_for_memory(new_mem_id)
        dep_edges = [e for e in edges if e.type == EdgeType.DEPENDS_ON]
        assert len(dep_edges) == 1
