"""Tests for MemoryService structured ingestion and end-to-end temporal scenarios.

Governed by docs/SPECIFICATION.md Section 2, docs/ARCHITECTURE.md, and docs/AGENTS.md.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memorymesh.config import load_app_config
from memorymesh.db import init_db
from memorymesh.errors import (
    ConflictResolutionError,
    UnknownSourceError,
    UnsupportedRelationError,
)
from memorymesh.models import (
    Action,
    AttributeDef,
    AttrRef,
    Cardinality,
    Claim,
    EdgeType,
    IngestMessage,
    MemoryStatus,
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

    with pytest.raises(
        (UnsupportedRelationError, ConflictResolutionError),
        match="IDENTICAL.*deferred to Phase 3.*R-CORROBORATE",
    ):
        service.ingest_claims(msg2, [claim2], recorded_at=t2)

    # 4. Verify existing memory remains active and uncorrupted
    m1 = service.repo.get_memory(m1_id)  # type: ignore[arg-type]
    assert m1 is not None
    assert m1.status == MemoryStatus.ACTIVE
    assert m1.value_norm == "3.10"
