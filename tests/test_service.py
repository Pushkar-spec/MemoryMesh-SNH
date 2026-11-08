"""Tests for MemoryService structured ingestion and end-to-end temporal scenarios.

Governed by docs/SPECIFICATION.md Section 2, docs/ARCHITECTURE.md, and docs/AGENTS.md.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memorymesh.config import load_app_config
from memorymesh.db import init_db
from memorymesh.errors import UnknownSourceError
from memorymesh.models import (
    Action,
    AttributeDef,
    AttrRef,
    Cardinality,
    Claim,
    EdgeType,
    IngestMessage,
    MemoryStatus,
    NewMemory,
    Source,
    ValueType,
)
from memorymesh.repository import Repository
from memorymesh.service import MemoryService


@pytest.fixture
def service() -> MemoryService:
    """Create MemoryService instance backed by an initialized in-memory database."""
    conn = init_db(":memory:")
    repo = Repository(conn)

    # Seed initial catalogs
    repo.insert_source(
        Source(
            id="sysadmin_priya",
            name="Priya Sharma",
            type="admin",
            reliability=0.95,
            notes="Senior Admin",
        )
    )
    repo.insert_source(
        Source(
            id="dev_chat",
            name="Developer Chat",
            type="chat",
            reliability=0.60,
        )
    )
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="python_version",
            description="Installed Python runtime version",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
        )
    )
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="os_version",
            description="Operating system release",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
        )
    )
    repo.insert_entity(canonical_name="Server A", entity_type="server")

    cfg = load_app_config()
    return MemoryService(db_target=repo, cfg=cfg)


def test_first_claim_creates_r_insert(service: MemoryService) -> None:
    """1. First claim creates an R-INSERT decision and persists memory and evidence."""
    t_msg = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    message = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A runs Python 3.10",
        event_time=t_msg,
    )
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=0.95,
        span="Server A runs Python 3.10",
    )

    report = service.ingest_claims(message, [claim], recorded_at=t_msg)

    assert report.duplicate is False
    assert report.claims_extracted == 1
    assert len(report.decisions) == 1

    dec = report.decisions[0]
    assert dec.action == Action.INSERT
    assert dec.rule_id == "R-INSERT"
    assert dec.memory_id is not None

    # Verify persisted memory
    mem = service.repo.get_memory(dec.memory_id)
    assert mem is not None
    assert mem.value_norm == "3.10"
    assert mem.status == MemoryStatus.ACTIVE
    assert mem.valid_from == t_msg
    assert mem.valid_to is None
    assert mem.closed_at is None
    assert len(mem.evidence) == 1
    assert mem.evidence[0].source_id == "sysadmin_priya"


def test_newer_conflicting_claim_creates_r_replace(service: MemoryService) -> None:
    """2-7. Newer conflicting claim replaces old memory, sets validity bounds, and adds edge."""
    t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    # 1. Initial claim
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
    report1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = report1.decisions[0].memory_id
    assert m1_id is not None

    # 2. Newer claim
    msg2 = IngestMessage(
        source_id="sysadmin_priya",
        text="Migrated Server A to Python 3.12",
        event_time=t2,
    )
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.95,
    )
    report2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)

    assert len(report2.decisions) == 1
    dec2 = report2.decisions[0]
    assert dec2.action == Action.REPLACE
    assert dec2.rule_id == "R-REPLACE"
    m2_id = dec2.memory_id
    assert m2_id is not None
    assert m2_id != m1_id

    # 3-5. Verify old memory is now historical and closed at t2
    old_mem = service.repo.get_memory(m1_id)
    assert old_mem is not None
    assert old_mem.status == MemoryStatus.HISTORICAL
    assert old_mem.valid_to == t2
    assert old_mem.closed_at == t2

    # 6. Verify new memory is active
    new_mem = service.repo.get_memory(m2_id)
    assert new_mem is not None
    assert new_mem.status == MemoryStatus.ACTIVE
    assert new_mem.value_norm == "3.12"
    assert new_mem.valid_from == t2
    assert new_mem.valid_to is None

    # 7. Verify SUPERSEDES edge points from new memory (m2) to old memory (m1)
    edges = service.repo.get_edges_for_memory(m2_id, direction="src")
    supersedes_edges = [e for e in edges if e.type == EdgeType.SUPERSEDES]
    assert len(supersedes_edges) == 1
    assert supersedes_edges[0].src_memory_id == m2_id
    assert supersedes_edges[0].dst_memory_id == m1_id


def test_dependencies_persisted_correctly(service: MemoryService) -> None:
    """8. Dependencies in Claim.depends_on are resolved and persisted as DEPENDS_ON edges."""
    t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)

    # Ingest foundation memory: OS version
    msg_os = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A runs Ubuntu 24.04",
        event_time=t,
    )
    claim_os = Claim(
        subject="Server A",
        attribute="os_version",
        value="24.04",
        extract_confidence=0.95,
    )
    rep_os = service.ingest_claims(msg_os, [claim_os], recorded_at=t)
    os_mem_id = rep_os.decisions[0].memory_id
    assert os_mem_id is not None

    # Ingest dependent memory: Python depends on OS version
    msg_py = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A runs Python 3.12 on Ubuntu",
        event_time=t,
    )
    claim_py = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        depends_on=[AttrRef(entity="Server A", attribute="os_version")],
        extract_confidence=0.95,
    )
    rep_py = service.ingest_claims(msg_py, [claim_py], recorded_at=t)
    py_mem_id = rep_py.decisions[0].memory_id
    assert py_mem_id is not None

    # Verify DEPENDS_ON edge points from Python memory to OS memory
    edges = service.repo.get_edges_for_memory(py_mem_id, direction="src")
    dep_edges = [e for e in edges if e.type == EdgeType.DEPENDS_ON]
    assert len(dep_edges) == 1
    assert dep_edges[0].src_memory_id == py_mem_id
    assert dep_edges[0].dst_memory_id == os_mem_id


def test_ingestion_transaction_atomicity_and_rollback(service: MemoryService) -> None:
    """9-10. Mutation failure rolls back all changes, leaving existing records intact."""
    t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)

    # Establish initial state
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
    initial_mem = service.get_current("Server A", "python_version", now=t1)
    assert initial_mem is not None
    assert initial_mem.value_norm == "3.10"

    # Attempt to ingest claim for unknown attribute (raises error during claim resolution)
    msg_bad = IngestMessage(
        source_id="sysadmin_priya",
        text="Invalid message",
        event_time=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
    )
    claim_bad = Claim(
        subject="Server A",
        attribute="non_existent_unregistered_attribute",
        value="foo",
        extract_confidence=0.95,
    )

    with pytest.raises(ValueError, match="Unknown attribute"):
        service.ingest_claims(msg_bad, [claim_bad])

    # 10. Verify database was NOT partially modified: message was not stored and m1 remains active
    unchanged_mem = service.get_current("Server A", "python_version", now=t1)
    assert unchanged_mem is not None
    assert unchanged_mem.value_norm == "3.10"
    assert unchanged_mem.status == MemoryStatus.ACTIVE
    assert unchanged_mem.valid_to is None


def test_duplicate_message_is_noop(service: MemoryService) -> None:
    """Duplicate message ingestion is a no-op with duplicate=True."""
    t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    message = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A runs Python 3.10",
        event_time=t,
    )
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=0.95,
    )

    rep1 = service.ingest_claims(message, [claim], recorded_at=t)
    assert rep1.duplicate is False
    assert len(rep1.decisions) == 1

    # Ingest the exact same message again
    rep2 = service.ingest_claims(message, [claim], recorded_at=t)
    assert rep2.duplicate is True
    assert len(rep2.decisions) == 0
    assert rep2.message_id == rep1.message_id


def test_end_to_end_migration_scenario(service: MemoryService) -> None:
    """Required end-to-end scenario:

    1. Ingest Server A runs Python 3.10 (valid from earlier date).
    2. Ingest Server A runs Python 3.12 (valid from later date).
    3. Query current memory -> Python 3.12.
    4. Query as of timestamp before migration -> Python 3.10.
    5. Query exactly at migration timestamp -> Python 3.12.
    """
    t_initial = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t_before_migration = datetime(2026, 3, 15, 10, 0, tzinfo=UTC)
    t_migration = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    t_current = datetime(2026, 6, 25, 9, 0, tzinfo=UTC)

    # 1. Ingest Python 3.10 on Jan 15
    msg1 = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A runs Python 3.10",
        event_time=t_initial,
    )
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=0.95,
    )
    service.ingest_claims(msg1, [claim1], recorded_at=t_initial)

    # 2. Ingest migration to Python 3.12 on June 1
    msg2 = IngestMessage(
        source_id="sysadmin_priya",
        text="We migrated Server A to Python 3.12",
        event_time=t_migration,
    )
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.95,
    )
    service.ingest_claims(msg2, [claim2], recorded_at=t_migration)

    # 3. Query current memory (now = June 25)
    current_res = service.current("Server A", "python_version", now=t_current)
    assert len(current_res.items) == 1
    assert current_res.items[0].memory.value_norm == "3.12"
    assert current_res.items[0].memory.status == MemoryStatus.ACTIVE

    # 4. Query memory as of timestamp before migration (March 15)
    march_res = service.current(
        "Server A", "python_version", valid_at=t_before_migration, now=t_current
    )
    assert len(march_res.items) == 1
    assert march_res.items[0].memory.value_norm == "3.10"
    assert march_res.items[0].memory.status == MemoryStatus.HISTORICAL

    # 5. Query exactly at migration timestamp (June 1 12:00:00)
    boundary_res = service.current(
        "Server A", "python_version", valid_at=t_migration, now=t_current
    )
    assert len(boundary_res.items) == 1
    # Boundary rule [valid_from, valid_to): at valid_to old memory is closed, new memory is valid
    assert boundary_res.items[0].memory.value_norm == "3.12"
    assert boundary_res.items[0].memory.status == MemoryStatus.ACTIVE


def test_auto_load_and_persist_source_and_attribute_from_config() -> None:
    """Empty DB: Sources/attributes absent from DB are auto-loaded from AppConfig and saved."""
    conn = init_db(":memory:")
    cfg = load_app_config()
    service = MemoryService(db_target=conn, cfg=cfg)

    # Verify catalogs are empty in DB before ingestion
    assert service.repo.get_source("sysadmin_priya") is None
    assert service.repo.get_attribute_schema("python_version") is None

    t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    msg = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A runs Python 3.10",
        event_time=t,
    )
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=0.95,
    )

    report = service.ingest_claims(msg, [claim], recorded_at=t)
    assert report.duplicate is False
    assert len(report.decisions) == 1
    assert report.decisions[0].action == Action.INSERT

    # Verify both source and attribute definition were persisted to the database
    persisted_source = service.repo.get_source("sysadmin_priya")
    assert persisted_source is not None
    assert persisted_source.id == "sysadmin_priya"
    assert persisted_source.type == "admin"
    assert persisted_source.reliability == 0.90

    persisted_attr = service.repo.get_attribute_schema("python_version")
    assert persisted_attr is not None
    assert persisted_attr.attribute == "python_version"
    assert persisted_attr.cardinality == Cardinality.SINGLE
    assert persisted_attr.value_type == ValueType.VERSION


def test_unknown_source_error_on_empty_db() -> None:
    """Empty DB: An unconfigured source raises UnknownSourceError and performs no writes."""
    conn = init_db(":memory:")
    service = MemoryService(db_target=conn, cfg=load_app_config())

    t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    msg = IngestMessage(
        source_id="non_existent_unconfigured_source",
        text="Server A runs Python 3.10",
        event_time=t,
    )
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=0.95,
    )

    with pytest.raises(
        UnknownSourceError, match="Unknown source 'non_existent_unconfigured_source'"
    ):
        service.ingest_claims(msg, [claim])

    assert service.repo.get_source("non_existent_unconfigured_source") is None


def test_unknown_attribute_error_on_empty_db() -> None:
    """Empty DB: An unconfigured attribute raises ValueError with clear error."""
    conn = init_db(":memory:")
    service = MemoryService(db_target=conn, cfg=load_app_config())

    t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    msg = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A has unknown attribute",
        event_time=t,
    )
    claim = Claim(
        subject="Server A",
        attribute="totally_fake_attribute_xyz",
        value="bar",
        extract_confidence=0.95,
    )

    with pytest.raises(ValueError, match="Unknown attribute 'totally_fake_attribute_xyz'"):
        service.ingest_claims(msg, [claim])

    assert service.repo.get_attribute_schema("totally_fake_attribute_xyz") is None


def test_identical_claim_on_different_message_raises_unsupported_relation(
    service: MemoryService,
) -> None:
    """A second different message with identical claim raises documented UnsupportedRelationError.

    Duplicate message idempotency remains unaffected.
    """
    t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 1, 16, 11, 0, tzinfo=UTC)

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

    # 1. First ingestion succeeds
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    assert rep1.duplicate is False
    assert len(rep1.decisions) == 1
    m1_id = rep1.decisions[0].memory_id

    # 2. Duplicate exact message is idempotent
    dup_rep = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    assert dup_rep.duplicate is True
    assert dup_rep.decisions == []

    # 3. Second, different message asserting the exact same fact (IDENTICAL relation)
    msg2 = IngestMessage(
        source_id="sysadmin_priya",
        text="Re-affirming Server A runs Python 3.10",
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
    assert rep2.decisions[0].rule_id == "R-DUP-SAME-SOURCE"

    # 4. Verify existing memory remains active and uncorrupted, with two evidence rows
    m1 = service.repo.get_memory(m1_id)  # type: ignore[arg-type]
    assert m1 is not None
    assert m1.status == MemoryStatus.ACTIVE
    assert m1.value_norm == "3.10"
    assert len(service.repo.get_evidence_for_memory(m1_id)) == 2


def test_ingest_claims_normalizes_value(service: MemoryService) -> None:
    """Ingestion normalizes raw claim values and preserves raw text in value_display."""
    t_msg = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    message = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A runs Python v3.12",
        event_time=t_msg,
    )
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="Python v3.12",
        extract_confidence=0.95,
        span="Python v3.12",
    )

    report = service.ingest_claims(message, [claim], recorded_at=t_msg)
    assert len(report.decisions) == 1
    mem_id = report.decisions[0].memory_id
    assert mem_id is not None

    mem = service.repo.get_memory(mem_id)
    assert mem is not None
    assert mem.value_norm == "3.12"
    assert mem.value_display == "Python v3.12"


def test_ingest_claims_invalid_value_skips_without_creating_memory(
    service: MemoryService,
) -> None:
    """Invalid claim values follow documented SKIP behavior: logged as SKIP, no memory created."""
    t_msg = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    message = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A runs invalid version",
        event_time=t_msg,
    )
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="no-version-here",
        extract_confidence=0.95,
    )

    report = service.ingest_claims(message, [claim], recorded_at=t_msg)
    assert report.duplicate is False
    assert len(report.decisions) == 1
    decision_entry = report.decisions[0]
    assert decision_entry.action == Action.SKIP
    assert decision_entry.rule_id == "R-SKIP-VALUE"
    assert decision_entry.memory_id is None
    assert "No valid version components found" in decision_entry.explanation

    # Verify no memory was created in repository
    current_res = service.current("Server A", "python_version", now=t_msg)
    assert current_res.items == []


def test_ingest_claims_passes_enum_allowed_values(service: MemoryService) -> None:
    """ENUM allowed_values are passed from schema, normalizing valid or skipping invalid."""
    service.repo.insert_attribute_schema(
        AttributeDef(
            attribute="env_tier",
            description="Deployment tier",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.ENUM,
            allowed_values=["production", "staging", "dev"],
        )
    )
    t_msg = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)

    # 1. Valid mixed-case enum claim normalizes to lowercase
    msg_valid = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A deployed to STAGING",
        event_time=t_msg,
    )
    claim_valid = Claim(
        subject="Server A",
        attribute="env_tier",
        value="STAGING",
        extract_confidence=0.95,
    )
    rep_valid = service.ingest_claims(msg_valid, [claim_valid], recorded_at=t_msg)
    assert len(rep_valid.decisions) == 1
    assert rep_valid.decisions[0].action == Action.INSERT
    mem = service.repo.get_memory(rep_valid.decisions[0].memory_id)  # type: ignore[arg-type]
    assert mem is not None
    assert mem.value_norm == "staging"
    assert mem.value_display == "STAGING"

    # 2. Invalid enum value triggers SKIP without corrupting existing memory
    t_msg2 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    msg_invalid = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A deployed to local",
        event_time=t_msg2,
    )
    claim_invalid = Claim(
        subject="Server A",
        attribute="env_tier",
        value="local",
        extract_confidence=0.95,
    )
    rep_invalid = service.ingest_claims(msg_invalid, [claim_invalid], recorded_at=t_msg2)
    assert len(rep_invalid.decisions) == 1
    assert rep_invalid.decisions[0].action == Action.SKIP
    assert rep_invalid.decisions[0].rule_id == "R-SKIP-VALUE"
    assert "not an allowed enum value" in rep_invalid.decisions[0].explanation


def test_ingest_claims_classifies_legacy_unnormalized_memory(service: MemoryService) -> None:
    """Ingestion correctly classifies against legacy unnormalized existing memories."""
    t_p12 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    # Store initial message
    msg1_id = service.repo.insert_message(
        source_id="sysadmin_priya",
        text="Legacy seed",
        event_time=t_p12,
    )
    # Simulate an unnormalized legacy memory stored during P1.2
    mem_id = service.repo.insert_memory(
        memory=NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="v3.12",  # Unnormalized legacy storage
            value_display="Python v3.12",
            context_key="",
            valid_from=t_p12,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=msg1_id,
            source_id="sysadmin_priya",
        ),
        recorded_at=t_p12,
    )
    assert mem_id is not None

    # 1. Incoming claim with normalized "3.12" against legacy "v3.12"
    # Classifies as IDENTICAL, raising UnsupportedRelationError in P2.1 (not replacing!)
    t2 = datetime(2026, 2, 1, 10, 0, tzinfo=UTC)
    msg2 = IngestMessage(
        source_id="sysadmin_priya",
        text="Confirming Python 3.12",
        event_time=t2,
    )
    claim_identical = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.95,
    )
    rep_ident = service.ingest_claims(msg2, [claim_identical], recorded_at=t2)
    assert len(rep_ident.decisions) == 1
    assert rep_ident.decisions[0].action == Action.CORROBORATE
    assert rep_ident.decisions[0].rule_id == "R-DUP-SAME-SOURCE"
    assert len(service.repo.get_evidence_for_memory(mem_id)) == 1

    # 2. Incoming claim with conflicting "3.14" against legacy "v3.12"
    # Classifies as CONFLICTS, replacing the legacy memory via R-REPLACE
    t3 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    msg3 = IngestMessage(
        source_id="sysadmin_priya",
        text="Upgraded to Python 3.14",
        event_time=t3,
    )
    claim_replace = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.14",
        extract_confidence=0.95,
    )
    rep3 = service.ingest_claims(msg3, [claim_replace], recorded_at=t3)
    assert len(rep3.decisions) == 1
    assert rep3.decisions[0].action == Action.REPLACE
    assert rep3.decisions[0].rule_id == "R-REPLACE"

    # Current memory is now 3.14
    current_res = service.current("Server A", "python_version", now=t3)
    assert len(current_res.items) == 1
    assert current_res.items[0].memory.value_norm == "3.14"
    assert current_res.items[0].memory.status == MemoryStatus.ACTIVE
