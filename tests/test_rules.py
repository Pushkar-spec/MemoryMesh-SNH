"""Tests for the deterministic resolution rules (R-INSERT, R-REPLACE).

Governed by docs/RESOLUTION_ENGINE.md, docs/SPECIFICATION.md Section 1.1, and docs/AGENTS.md.
All engine tests are pure functions using hand-built Pydantic objects without a database.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memorymesh.config import load_engine_config
from memorymesh.engine.rules import resolve
from memorymesh.errors import ConflictResolutionError, InvalidValidityWindowError
from memorymesh.models import (
    Action,
    AddEdge,
    AttributeDef,
    Cardinality,
    Claim,
    EdgeType,
    InsertMemory,
    Memory,
    MemoryStatus,
    Relation,
    ResolvedClaim,
    Source,
    UpdateMemory,
    ValueType,
)


@pytest.fixture
def sample_attribute() -> AttributeDef:
    """Provide standard single-valued version attribute definition."""
    return AttributeDef(
        attribute="python_version",
        description="Installed Python version",
        cardinality=Cardinality.SINGLE,
        value_type=ValueType.VERSION,
    )


@pytest.fixture
def sample_source() -> Source:
    """Provide standard admin source definition."""
    return Source(
        id="sysadmin_priya",
        name="Priya Sharma",
        type="admin",
        reliability=0.95,
        notes="Senior Admin",
    )


def test_r_insert_first_claim(sample_attribute: AttributeDef, sample_source: Source) -> None:
    """R-INSERT: First claim with no candidate memories produces an INSERT decision."""
    cfg = load_engine_config()
    t_valid = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t_recorded = datetime(2026, 1, 15, 10, 5, tzinfo=UTC)

    rc = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.10",
            extract_confidence=0.95,
            span="Server A runs Python 3.10",
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t_valid,
        valid_to=None,
        source=sample_source,
        message_id=101,
    )

    decision = resolve(
        rc=rc,
        chain=[],
        schema=sample_attribute,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t_recorded,
    )

    assert decision.action == Action.INSERT
    assert decision.rule_id == "R-INSERT"
    assert "no existing candidate" in decision.explanation.lower()
    assert len(decision.mutations) == 1
    assert decision.affected_memory_ids == []

    mutation = decision.mutations[0]
    assert isinstance(mutation, InsertMemory)
    assert mutation.memory.entity_id == 1
    assert mutation.memory.attribute == "python_version"
    assert mutation.memory.value_norm == "3.10"
    assert mutation.memory.value_display == "Python 3.10"
    assert mutation.memory.valid_from == t_valid
    assert mutation.memory.valid_to is None
    assert mutation.memory.closed_at is None
    assert mutation.memory.status == MemoryStatus.ACTIVE
    assert mutation.memory.confidence == 1.0
    assert mutation.evidence.source_id == "sysadmin_priya"
    assert mutation.evidence.strength == 1.0


def test_r_insert_with_dependency_targets(
    sample_attribute: AttributeDef, sample_source: Source
) -> None:
    """R-INSERT: Produces DEPENDS_ON edge mutations when dep_targets are present."""
    cfg = load_engine_config()
    t_valid = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t_recorded = datetime(2026, 1, 15, 10, 5, tzinfo=UTC)

    rc = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.10",
            extract_confidence=0.95,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t_valid,
        source=sample_source,
        message_id=101,
    )

    decision = resolve(
        rc=rc,
        chain=[],
        schema=sample_attribute,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[42, 43],
        recorded_at=t_recorded,
    )

    assert decision.action == Action.INSERT
    # Must have 1 InsertMemory and 2 AddEdge mutations
    assert len(decision.mutations) == 3
    assert isinstance(decision.mutations[0], InsertMemory)

    edge1 = decision.mutations[1]
    assert isinstance(edge1, AddEdge)
    assert edge1.src == "$new"
    assert edge1.dst == 42
    assert edge1.type == EdgeType.DEPENDS_ON

    edge2 = decision.mutations[2]
    assert isinstance(edge2, AddEdge)
    assert edge2.src == "$new"
    assert edge2.dst == 43
    assert edge2.type == EdgeType.DEPENDS_ON


def test_r_replace_closes_window_and_creates_edge(
    sample_attribute: AttributeDef, sample_source: Source
) -> None:
    """R-REPLACE: Newer conflicting single-valued claim closes old memory and supersedes."""
    cfg = load_engine_config()
    t_old = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t_new = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    t_recorded = datetime(2026, 6, 1, 12, 5, tzinfo=UTC)

    existing_mem = Memory(
        id=1,
        entity_id=1,
        entity_name="Server A",
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t_old,
        valid_to=None,
        recorded_at=t_old,
        closed_at=None,
        status=MemoryStatus.ACTIVE,
        confidence=1.0,
        message_id=101,
        source_id="sysadmin_priya",
    )

    rc_new = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
            span="Migrated Server A to Python 3.12",
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t_new,
        source=sample_source,
        message_id=102,
    )

    decision = resolve(
        rc=rc_new,
        chain=[existing_mem],
        schema=sample_attribute,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t_recorded,
    )

    assert decision.action == Action.REPLACE
    assert decision.rule_id == "R-REPLACE"
    assert decision.affected_memory_ids == [1]
    assert "replaces" in decision.explanation.lower()

    # Mutations: Insert new memory, update old memory, add supersedes edge
    assert len(decision.mutations) == 3

    insert_mut = decision.mutations[0]
    assert isinstance(insert_mut, InsertMemory)
    assert insert_mut.memory.value_norm == "3.12"
    assert insert_mut.memory.status == MemoryStatus.ACTIVE
    assert insert_mut.memory.valid_from == t_new

    update_mut = decision.mutations[1]
    assert isinstance(update_mut, UpdateMemory)
    assert update_mut.memory_id == 1
    assert update_mut.fields["status"] == MemoryStatus.HISTORICAL
    assert update_mut.fields["valid_to"] == t_new
    assert update_mut.fields["closed_at"] == t_recorded

    edge_mut = decision.mutations[2]
    assert isinstance(edge_mut, AddEdge)
    assert edge_mut.src == "$new"
    assert edge_mut.dst == 1
    assert edge_mut.type == EdgeType.SUPERSEDES


def test_r_replace_not_newer_claim_rejected(
    sample_attribute: AttributeDef, sample_source: Source
) -> None:
    """R-REPLACE is rejected when the new conflicting claim is older than existing memory."""
    cfg = load_engine_config()
    t_existing = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    t_older_claim = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)

    existing_mem = Memory(
        id=1,
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t_existing,
        recorded_at=t_existing,
        status=MemoryStatus.ACTIVE,
        confidence=1.0,
        message_id=101,
        source_id="sysadmin_priya",
    )

    rc_older = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.10",
            extract_confidence=0.95,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t_older_claim,
        source=sample_source,
        message_id=102,
    )

    # An older conflicting claim cannot replace existing newer memory
    with pytest.raises(ConflictResolutionError, match="not newer"):
        resolve(
            rc=rc_older,
            chain=[existing_mem],
            schema=sample_attribute,
            cfg=cfg,
            classify=lambda _rc, _m: Relation.CONFLICTS,
            dep_targets=[],
            recorded_at=datetime(2026, 6, 1, 12, 5, tzinfo=UTC),
        )


def test_same_time_validity_boundary_does_not_replace(
    sample_attribute: AttributeDef, sample_source: Source
) -> None:
    """Equal valid_from timestamp does not allow R-REPLACE."""
    cfg = load_engine_config()
    t_same = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    existing_mem = Memory(
        id=1,
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t_same,
        recorded_at=t_same,
        status=MemoryStatus.ACTIVE,
        confidence=1.0,
        message_id=101,
        source_id="sysadmin_priya",
    )

    rc_same = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t_same,
        source=sample_source,
        message_id=102,
    )

    with pytest.raises(ConflictResolutionError, match="not newer"):
        resolve(
            rc=rc_same,
            chain=[existing_mem],
            schema=sample_attribute,
            cfg=cfg,
            classify=lambda _rc, _m: Relation.CONFLICTS,
            dep_targets=[],
            recorded_at=datetime(2026, 6, 1, 12, 5, tzinfo=UTC),
        )


def test_invalid_validity_window_inputs(
    sample_attribute: AttributeDef, sample_source: Source
) -> None:
    """Validate that valid_to <= valid_from raises InvalidValidityWindowError."""
    cfg = load_engine_config()
    t_start = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    t_end = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)  # Prior to start!

    rc_invalid = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t_start,
        valid_to=t_end,
        source=sample_source,
        message_id=102,
    )

    with pytest.raises(
        InvalidValidityWindowError, match="valid_to must be greater than valid_from"
    ):
        resolve(
            rc=rc_invalid,
            chain=[],
            schema=sample_attribute,
            cfg=cfg,
            classify=lambda _rc, _m: Relation.CONFLICTS,
            dep_targets=[],
            recorded_at=datetime(2026, 6, 1, 12, 5, tzinfo=UTC),
        )


def test_no_candidate_open_ended_after_closed_window(
    sample_attribute: AttributeDef, sample_source: Source
) -> None:
    """No candidate memory covers rc.valid_from when prior window is closed; gives R-INSERT."""
    cfg = load_engine_config()
    t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 3, 15, 10, 0, tzinfo=UTC)
    t3 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    closed_mem = Memory(
        id=1,
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t1,
        valid_to=t2,
        recorded_at=t1,
        closed_at=t2,
        status=MemoryStatus.HISTORICAL,
        confidence=1.0,
        message_id=101,
        source_id="sysadmin_priya",
    )

    # Claim is for t3, which is well after t2 (previous window closed)
    rc_later = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t3,
        source=sample_source,
        message_id=102,
    )

    decision = resolve(
        rc=rc_later,
        chain=[closed_mem],
        schema=sample_attribute,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t3,
    )

    assert decision.action == Action.INSERT
    assert decision.rule_id == "R-INSERT"
    assert len(decision.mutations) == 1
    assert isinstance(decision.mutations[0], InsertMemory)
    assert decision.mutations[0].memory.valid_from == t3


def test_multi_cardinality_conflicting_claim_blocks_replace(
    sample_source: Source,
) -> None:
    """A conflicting claim for a MULTI attribute must never trigger R-REPLACE."""
    multi_attr = AttributeDef(
        attribute="runs_service",
        description="Services running on host",
        cardinality=Cardinality.MULTI,
        value_type=ValueType.TEXT,
    )
    cfg = load_engine_config()
    t_old = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t_new = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    existing_mem = Memory(
        id=1,
        entity_id=1,
        attribute="runs_service",
        value_norm="api_gateway",
        value_display="API Gateway",
        context_key="",
        valid_from=t_old,
        recorded_at=t_old,
        status=MemoryStatus.ACTIVE,
        confidence=1.0,
        message_id=101,
        source_id="sysadmin_priya",
    )

    rc_new = ResolvedClaim(
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
        valid_from=t_new,
        source=sample_source,
        message_id=102,
    )

    with pytest.raises(ConflictResolutionError, match="Multi-valued attribute cardinality"):
        resolve(
            rc=rc_new,
            chain=[existing_mem],
            schema=multi_attr,
            cfg=cfg,
            classify=lambda _rc, _m: Relation.CONFLICTS,
            dep_targets=[],
            recorded_at=t_new,
        )
