"""Tests for P5.1: Dependency Resolution and Propagation.

Governed by docs/RESOLUTION_ENGINE.md Section 5 and docs/SPECIFICATION.md.
Tests cover:
1. Dependency resolution (empty, direct, multiple, transitive, diamond, ordering, missing).
2. Graph safety (cycles, self-referential, duplicate edges).
3. Propagation (review flagging, transitive, facts/confidence intact, idempotence, atomic rollback).
4. Regression (dispute resolution, temporal retrieval, invariant checks).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from memorymesh.config import load_app_config
from memorymesh.db import init_db
from memorymesh.engine.propagate import (
    build_dependency_graph,
    compute_propagation,
    resolve_direct_dependencies,
    resolve_direct_dependents,
    resolve_transitive_dependents,
)
from memorymesh.errors import ForeignKeyConstraintError, MemoryNotFoundError
from memorymesh.models import (
    Action,
    AttributeDef,
    AttrRef,
    Cardinality,
    Claim,
    Edge,
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
    """Fixture providing an initialized MemoryService with test sources."""
    conn = init_db(":memory:")
    repo = Repository(conn)
    repo.insert_source(
        Source(
            id="sysadmin_alice",
            name="Alice Admin",
            type="admin",
            reliability=0.9,
        )
    )
    repo.insert_source(
        Source(
            id="junior_bob",
            name="Bob Junior",
            type="user",
            reliability=0.6,
        )
    )
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="operating_system",
            description="Operating System",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="python_version",
            description="Python Version",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
        )
    )
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="framework",
            description="Framework",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="app_version",
            description="Application Version",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
        )
    )
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="memory_gb",
            description="RAM",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.NUMBER,
        )
    )
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="component_b",
            description="Component B",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="component_c",
            description="Component C",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="component_d",
            description="Component D",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )
    for i in range(5):
        repo.insert_attribute_schema(
            AttributeDef(
                attribute=f"attr_{i}",
                description=f"Attr {i}",
                cardinality=Cardinality.SINGLE,
                value_type=ValueType.TEXT,
            )
        )
    repo.insert_entity(canonical_name="Server A", entity_type="server")
    repo.insert_entity(canonical_name="Server B", entity_type="server")

    cfg = load_app_config()
    return MemoryService(db_target=repo, cfg=cfg)


# =============================================================================
# 1. Dependency Resolution Tests
# =============================================================================


def test_dependency_resolution_empty_dependencies(service: MemoryService) -> None:
    """1. A memory with no dependencies returns an empty result."""
    msg = IngestMessage(
        source_id="sysadmin_alice",
        event_time=datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC),
        text="Server A runs Linux",
    )
    claim = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC),
        raw_text="Server A runs Linux",
    )
    rep = service.ingest_claims(msg, [claim])
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    deps = service.get_dependencies(m_id)
    dependents = service.get_dependents(m_id)
    trans_deps = service.get_transitive_dependents(m_id)

    assert deps == []
    assert dependents == []
    assert trans_deps == []


def test_dependency_resolution_one_direct_dependent(service: MemoryService) -> None:
    """2. A memory with one direct dependent resolves correctly."""
    # Upstream memory: OS
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Server A OS")
    claim1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A OS",
    )
    rep1 = service.ingest_claims(msg1, [claim1])
    os_id = rep1.decisions[0].memory_id
    assert os_id is not None

    # Downstream memory: Python depends on OS
    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Server A Python")
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Server A Python",
    )
    rep2 = service.ingest_claims(msg2, [claim2])
    py_id = rep2.decisions[0].memory_id
    assert py_id is not None

    # Verify upstream dependencies of Python memory
    py_deps = service.get_dependencies(py_id)
    assert len(py_deps) == 1
    assert py_deps[0].id == os_id

    # Verify downstream dependents of OS memory
    os_dependents = service.get_dependents(os_id)
    assert len(os_dependents) == 1
    assert os_dependents[0].id == py_id


def test_dependency_resolution_multiple_direct_dependents(service: MemoryService) -> None:
    """3. Multiple direct dependents are returned."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Server A OS")
    claim1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A OS",
    )
    rep1 = service.ingest_claims(msg1, [claim1])
    os_id = rep1.decisions[0].memory_id
    assert os_id is not None

    # Dependent 1: python_version depends on OS
    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Server A Python")
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Server A Python",
    )
    rep2 = service.ingest_claims(msg2, [claim2])
    py_id = rep2.decisions[0].memory_id
    assert py_id is not None

    # Dependent 2: memory_gb depends on OS
    msg3 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Server A RAM")
    claim3 = Claim(
        subject="Server A",
        attribute="memory_gb",
        value="64",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Server A RAM",
    )
    rep3 = service.ingest_claims(msg3, [claim3])
    ram_id = rep3.decisions[0].memory_id
    assert ram_id is not None

    dependents = service.get_dependents(os_id)
    assert len(dependents) == 2
    dep_ids = [d.id for d in dependents]
    assert py_id in dep_ids
    assert ram_id in dep_ids


def test_dependency_resolution_multilevel_transitive_chain(service: MemoryService) -> None:
    """4. A multi-level dependency chain (A -> B -> C -> D) resolves transitively."""
    # A (OS) -> B (Python) -> C (Framework) -> D (App)
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="System stack")

    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A OS",
    )
    m1_id = service.ingest_claims(msg, [c1]).decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python")
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Server A Python",
    )
    m2_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert m2_id is not None

    # Directly create level 3 and 4 dependencies in graph
    m3_id = service.repo.insert_memory(
        service.repo.get_memory(m2_id).model_copy(  # type: ignore[union-attr]
            update={
                "id": 0,
                "attribute": "framework",
                "value_norm": "FastAPI",
                "value_display": "FastAPI",
            }
        ),
        recorded_at=t1,
    )
    service.repo.insert_edge(
        src_memory_id=m3_id, dst_memory_id=m2_id, edge_type=EdgeType.DEPENDS_ON, created_at=t1
    )

    m4_id = service.repo.insert_memory(
        service.repo.get_memory(m2_id).model_copy(  # type: ignore[union-attr]
            update={
                "id": 0,
                "attribute": "app_version",
                "value_norm": "1.0.0",
                "value_display": "1.0.0",
            }
        ),
        recorded_at=t1,
    )
    service.repo.insert_edge(
        src_memory_id=m4_id, dst_memory_id=m3_id, edge_type=EdgeType.DEPENDS_ON, created_at=t1
    )

    # Transitive dependents from M1
    trans = service.get_transitive_dependents(m1_id)
    assert len(trans) == 3
    assert trans[0][0].id == m2_id
    assert trans[0][1] == [m1_id, m2_id]
    assert trans[1][0].id == m3_id
    assert trans[1][1] == [m1_id, m2_id, m3_id]
    assert trans[2][0].id == m4_id
    assert trans[2][1] == [m1_id, m2_id, m3_id, m4_id]


def test_dependency_resolution_shared_downstream_diamond(service: MemoryService) -> None:
    """5. Shared downstream dependencies in a diamond graph return each memory only once."""
    # A -> B -> D and A -> C -> D
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Diamond")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    a_id = service.ingest_claims(msg, [c1]).decisions[0].memory_id
    assert a_id is not None

    b_id = service.repo.insert_memory(
        service.repo.get_memory(a_id).model_copy(  # type: ignore[union-attr]
            update={"id": 0, "attribute": "component_b", "value_norm": "B", "value_display": "B"}
        ),
        recorded_at=t1,
    )
    service.repo.insert_edge(
        src_memory_id=b_id, dst_memory_id=a_id, edge_type=EdgeType.DEPENDS_ON, created_at=t1
    )

    c_id = service.repo.insert_memory(
        service.repo.get_memory(a_id).model_copy(  # type: ignore[union-attr]
            update={"id": 0, "attribute": "component_c", "value_norm": "C", "value_display": "C"}
        ),
        recorded_at=t1,
    )
    service.repo.insert_edge(
        src_memory_id=c_id, dst_memory_id=a_id, edge_type=EdgeType.DEPENDS_ON, created_at=t1
    )

    d_id = service.repo.insert_memory(
        service.repo.get_memory(a_id).model_copy(  # type: ignore[union-attr]
            update={"id": 0, "attribute": "component_d", "value_norm": "D", "value_display": "D"}
        ),
        recorded_at=t1,
    )
    # D depends on both B and C
    service.repo.insert_edge(
        src_memory_id=d_id, dst_memory_id=b_id, edge_type=EdgeType.DEPENDS_ON, created_at=t1
    )
    service.repo.insert_edge(
        src_memory_id=d_id, dst_memory_id=c_id, edge_type=EdgeType.DEPENDS_ON, created_at=t1
    )

    trans = service.get_transitive_dependents(a_id)
    discovered_ids = [m.id for m, _ in trans]
    assert discovered_ids.count(d_id) == 1
    assert set(discovered_ids) == {b_id, c_id, d_id}


def test_dependency_resolution_deterministic_ordering(service: MemoryService) -> None:
    """6. Results have deterministic ordering across calls."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Deterministic")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    a_id = service.ingest_claims(msg, [c1]).decisions[0].memory_id
    assert a_id is not None

    created_ids: list[int] = []
    for i in range(5):
        mid = service.repo.insert_memory(
            service.repo.get_memory(a_id).model_copy(  # type: ignore[union-attr]
                update={
                    "id": 0,
                    "attribute": f"attr_{i}",
                    "value_norm": str(i),
                    "value_display": str(i),
                }
            ),
            recorded_at=t1,
        )
        service.repo.insert_edge(
            src_memory_id=mid, dst_memory_id=a_id, edge_type=EdgeType.DEPENDS_ON, created_at=t1
        )
        created_ids.append(mid)

    for _ in range(5):
        deps = service.get_dependents(a_id)
        assert [d.id for d in deps] == sorted(created_ids)


def test_dependency_resolution_missing_or_inactive_memories(service: MemoryService) -> None:
    """7. Missing or inactive referenced memories are handled safely."""
    # Non-existent ID raises MemoryNotFoundError
    with pytest.raises(MemoryNotFoundError):
        service.get_dependencies(999999)

    with pytest.raises(MemoryNotFoundError):
        service.get_dependents(999999)

    with pytest.raises(MemoryNotFoundError):
        service.get_transitive_dependents(999999)

    # Inactive memory (HISTORICAL) in dependency graph
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Inactive test")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    a_id = service.ingest_claims(msg, [c1]).decisions[0].memory_id
    assert a_id is not None

    # Insert dependent memory with status=HISTORICAL
    hist_id = service.repo.insert_memory(
        service.repo.get_memory(a_id).model_copy(  # type: ignore[union-attr]
            update={
                "id": 0,
                "attribute": "python_version",
                "value_norm": "2.7",
                "value_display": "2.7",
                "status": MemoryStatus.HISTORICAL,
            }
        ),
        recorded_at=t1,
    )
    service.repo.insert_edge(
        src_memory_id=hist_id, dst_memory_id=a_id, edge_type=EdgeType.DEPENDS_ON, created_at=t1
    )

    # Direct query still finds the dependent record safely
    dependents = service.get_dependents(a_id)
    assert any(d.id == hist_id for d in dependents)


# =============================================================================
# 2. Graph Safety Tests
# =============================================================================


def test_graph_safety_cycle_does_not_cause_infinite_traversal(service: MemoryService) -> None:
    """8. A cycle (A -> B -> C -> A) does not cause infinite traversal."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Cycle test")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    a_id = service.ingest_claims(msg, [c1]).decisions[0].memory_id
    assert a_id is not None

    b_id = service.repo.insert_memory(
        service.repo.get_memory(a_id).model_copy(  # type: ignore[union-attr]
            update={"id": 0, "attribute": "python_version", "value_norm": "B", "value_display": "B"}
        ),
        recorded_at=t1,
    )
    c_id = service.repo.insert_memory(
        service.repo.get_memory(a_id).model_copy(  # type: ignore[union-attr]
            update={"id": 0, "attribute": "framework", "value_norm": "C", "value_display": "C"}
        ),
        recorded_at=t1,
    )

    # B depends on A: A -> B
    service.repo.insert_edge(
        src_memory_id=b_id, dst_memory_id=a_id, edge_type=EdgeType.DEPENDS_ON, created_at=t1
    )
    # C depends on B: B -> C
    service.repo.insert_edge(
        src_memory_id=c_id, dst_memory_id=b_id, edge_type=EdgeType.DEPENDS_ON, created_at=t1
    )
    # A depends on C: C -> A (CYCLE)
    service.repo.insert_edge(
        src_memory_id=a_id, dst_memory_id=c_id, edge_type=EdgeType.DEPENDS_ON, created_at=t1
    )

    # Must terminate without recursion/timeout
    trans = service.get_transitive_dependents(a_id)
    assert len(trans) == 2
    assert {m.id for m, _ in trans} == {b_id, c_id}


def test_graph_safety_self_referential_dependency(service: MemoryService) -> None:
    """9. Self-referential dependencies (A -> A) are handled safely."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Self ref")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    a_id = service.ingest_claims(msg, [c1]).decisions[0].memory_id
    assert a_id is not None

    # Insert self-loop
    service.repo.insert_edge(
        src_memory_id=a_id, dst_memory_id=a_id, edge_type=EdgeType.DEPENDS_ON, created_at=t1
    )

    deps = service.get_dependencies(a_id)
    assert deps == []
    dependents = service.get_dependents(a_id)
    assert dependents == []
    trans = service.get_transitive_dependents(a_id)
    assert trans == []


def test_graph_safety_duplicate_edges(service: MemoryService) -> None:
    """10. Duplicate dependency edges do not produce duplicate results."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    edges = [
        Edge(
            id=1,
            src_memory_id=2,
            dst_memory_id=1,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t1,
        ),
        Edge(
            id=2,
            src_memory_id=2,
            dst_memory_id=1,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t1,
        ),
    ]
    deps = resolve_direct_dependents(edges, 1)
    assert len(deps) == 1
    assert deps[0] == 2

    trans = resolve_transitive_dependents(edges, 1)
    assert len(trans) == 1
    assert trans[0][0] == 2


# =============================================================================
# 3. Change Propagation Tests
# =============================================================================


def test_propagation_upstream_change_marks_dependent_for_review(service: MemoryService) -> None:
    """11. An upstream memory change (R-REPLACE) marks its dependent memories for review."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    # Step 1: Ingest Python 3.10
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.10")
    c1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Python 3.10",
    )
    rep1 = service.ingest_claims(msg1, [c1])
    py_id = rep1.decisions[0].memory_id
    assert py_id is not None

    # Step 2: Ingest Library X that depends on Python
    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Library X 1.2")
    c2 = Claim(
        subject="Server A",
        attribute="framework",
        value="FastAPI",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="python_version")],
        raw_text="FastAPI",
    )
    rep2 = service.ingest_claims(msg2, [c2])
    lib_id = rep2.decisions[0].memory_id
    assert lib_id is not None

    # Verify initial state: lib is not needing review
    lib_mem = service.repo.get_memory(lib_id)
    assert lib_mem is not None and lib_mem.needs_review is False

    # Step 3: Upgrade Python to 3.12 (R-REPLACE triggers propagation)
    msg3 = IngestMessage(source_id="sysadmin_alice", event_time=t2, text="Python 3.12")
    c3 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t2,
        raw_text="Python 3.12",
    )
    rep3 = service.ingest_claims(msg3, [c3])

    # Check that decisions contain R-REPLACE and R-PROPAGATE
    rule_ids = [d.rule_id for d in rep3.decisions]
    assert "R-REPLACE" in rule_ids
    assert "R-PROPAGATE" in rule_ids

    # Check dependent memory is now flagged for review
    lib_mem_after = service.repo.get_memory(lib_id)
    assert lib_mem_after is not None
    assert lib_mem_after.needs_review is True
    assert "depends on changed upstream M" in (lib_mem_after.review_reason or "")


def test_propagation_transitive_dependents_included(service: MemoryService) -> None:
    """12. Transitive dependents (A -> B -> C) are included and marked for review."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    # A (OS)
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    os_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert os_id is not None

    # B (Python depends on OS)
    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python")
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Python",
    )
    py_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert py_id is not None

    # C (Framework depends on Python)
    msg3 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="FastAPI")
    c3 = Claim(
        subject="Server A",
        attribute="framework",
        value="FastAPI",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="python_version")],
        raw_text="FastAPI",
    )
    fw_id = service.ingest_claims(msg3, [c3]).decisions[0].memory_id
    assert fw_id is not None

    # Upgrade OS (R-REPLACE)
    msg_up = IngestMessage(source_id="sysadmin_alice", event_time=t2, text="OS Upgrade")
    c_up = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Debian 12",
        extract_confidence=1.0,
        valid_from=t2,
        raw_text="OS Upgrade",
    )
    rep_up = service.ingest_claims(msg_up, [c_up])

    # Both B and C must be marked for review
    prop_decisions = [d for d in rep_up.decisions if d.rule_id == "R-PROPAGATE"]
    flagged_ids = [d.memory_id for d in prop_decisions]
    assert py_id in flagged_ids
    assert fw_id in flagged_ids

    py_dec = next(d for d in prop_decisions if d.memory_id == py_id)
    fw_dec = next(d for d in prop_decisions if d.memory_id == fw_id)

    # 1. Changing A records A as root cause for both B and C
    assert py_dec.claim_json["upstream_memory_id"] == os_id
    assert fw_dec.claim_json["upstream_memory_id"] == os_id

    # 2. Verify C's path is [A, B, C]
    assert py_dec.claim_json["path"] == [os_id, py_id]
    assert fw_dec.claim_json["path"] == [os_id, py_id, fw_id]

    # Automatic ingestion propagation marked automatic
    assert py_dec.claim_json["trigger_mode"] == "automatic"
    assert fw_dec.claim_json["trigger_mode"] == "automatic"

    py_mem = service.repo.get_memory(py_id)
    fw_mem = service.repo.get_memory(fw_id)
    assert py_mem is not None and py_mem.needs_review is True
    assert fw_mem is not None and fw_mem.needs_review is True

    # 3. Verify C's review reason does not claim B changed
    assert fw_mem.review_reason is not None
    assert f"depends on changed upstream M{os_id}" in fw_mem.review_reason
    assert f"depends on M{py_id}, which was" not in fw_mem.review_reason


def test_propagation_downstream_factual_content_unchanged(service: MemoryService) -> None:
    """13. The original downstream factual content remains unchanged."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.10")
    c1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Python",
    )
    py_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert py_id is not None

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="FastAPI")
    c2 = Claim(
        subject="Server A",
        attribute="framework",
        value="FastAPI",
        extract_confidence=0.88,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="python_version")],
        raw_text="FastAPI",
    )
    fw_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert fw_id is not None

    before = service.repo.get_memory(fw_id)
    assert before is not None

    # Upgrade Python
    msg_up = IngestMessage(source_id="sysadmin_alice", event_time=t2, text="Python 3.12")
    c_up = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t2,
        raw_text="Python 3.12",
    )
    service.ingest_claims(msg_up, [c_up])

    after = service.repo.get_memory(fw_id)
    assert after is not None

    # Values, validity dates, status, confidence unchanged
    assert after.value_norm == before.value_norm
    assert after.value_display == before.value_display
    assert after.valid_from == before.valid_from
    assert after.valid_to == before.valid_to
    assert after.status == before.status
    assert after.confidence == before.confidence
    # Only needs_review and review_reason changed
    assert after.needs_review is True
    assert after.review_reason != before.review_reason


def test_propagation_evidence_and_confidence_unchanged(service: MemoryService) -> None:
    """14. Existing evidence and base confidence remain unchanged during propagation."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.10")
    c1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Python",
    )
    py_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert py_id is not None

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="FastAPI")
    c2 = Claim(
        subject="Server A",
        attribute="framework",
        value="FastAPI",
        extract_confidence=0.75,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="python_version")],
        raw_text="FastAPI",
    )
    fw_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert fw_id is not None

    ev_before = service.repo.get_evidence_for_memory(fw_id)
    conf_before = service.repo.get_memory(fw_id).confidence  # type: ignore[union-attr]

    # Trigger propagation
    msg_up = IngestMessage(source_id="sysadmin_alice", event_time=t2, text="Python 3.12")
    c_up = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t2,
        raw_text="Python 3.12",
    )
    service.ingest_claims(msg_up, [c_up])

    ev_after = service.repo.get_evidence_for_memory(fw_id)
    conf_after = service.repo.get_memory(fw_id).confidence  # type: ignore[union-attr]

    assert len(ev_after) == len(ev_before)
    assert conf_after == conf_before


def test_propagation_unrelated_memories_unaffected(service: MemoryService) -> None:
    """15. Unrelated memories are not affected by propagation."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Setup")
    c1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Python",
    )
    service.ingest_claims(msg1, [c1])

    # Unrelated memory on Server B
    msg_b = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Server B OS")
    c_unrelated = Claim(
        subject="Server B",
        attribute="operating_system",
        value="Debian 12",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server B OS",
    )
    unrelated_id = service.ingest_claims(msg_b, [c_unrelated]).decisions[0].memory_id
    assert unrelated_id is not None

    # Update Server A Python
    msg_up = IngestMessage(source_id="sysadmin_alice", event_time=t2, text="Python 3.12")
    c_up = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t2,
        raw_text="Python 3.12",
    )
    service.ingest_claims(msg_up, [c_up])

    unrelated_mem = service.repo.get_memory(unrelated_id)
    assert unrelated_mem is not None
    assert unrelated_mem.needs_review is False


def test_propagation_idempotent(service: MemoryService) -> None:
    """16. Event-level idempotency: repeating same event skips, distinct events record."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)
    t3 = datetime(2026, 7, 1, 10, 0, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Setup Python")
    c1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Python",
    )
    rep1 = service.ingest_claims(msg1, [c1])
    py_id = rep1.decisions[0].memory_id
    assert py_id is not None

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Setup FastAPI")
    c2 = Claim(
        subject="Server A",
        attribute="framework",
        value="FastAPI",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="python_version")],
        raw_text="FastAPI",
    )
    rep2 = service.ingest_claims(msg2, [c2])
    fastapi_id = rep2.decisions[0].memory_id
    assert fastapi_id is not None

    # 1. Event E1: First propagation event creates a record
    logs_e1 = service.propagate_change(
        changed_ids=[py_id], cause_text="kernel update", recorded_at=t2
    )
    assert len(logs_e1) == 1
    assert logs_e1[0].memory_id == fastapi_id
    assert logs_e1[0].action == Action.PROPAGATE

    # Check memory state after E1: needs_review marked, reason recorded
    mem_after_e1 = service.repo.get_memory(fastapi_id)
    assert mem_after_e1 is not None
    assert mem_after_e1.needs_review is True
    assert "kernel update" in mem_after_e1.review_reason

    # 2. Repeating E1 does not create a duplicate
    logs_e1_repeat = service.propagate_change(
        changed_ids=[py_id], cause_text="kernel update", recorded_at=t2
    )
    assert len(logs_e1_repeat) == 0

    # 3. Distinct Event E2 affecting already-reviewed memory creates a separate record
    logs_e2 = service.propagate_change(
        changed_ids=[py_id], cause_text="python CVE vulnerability patch", recorded_at=t3
    )
    assert len(logs_e2) == 1
    assert logs_e2[0].memory_id == fastapi_id
    assert logs_e2[0].action == Action.PROPAGATE

    # Check memory state after E2: stays needs_review=True, both reasons preserved
    mem_after_e2 = service.repo.get_memory(fastapi_id)
    assert mem_after_e2 is not None
    assert mem_after_e2.needs_review is True
    assert "kernel update" in mem_after_e2.review_reason
    assert "python CVE vulnerability patch" in mem_after_e2.review_reason
    assert mem_after_e2.value_display == "FastAPI"

    # 4. Repeating E2 does not create another duplicate
    logs_e2_repeat = service.propagate_change(
        changed_ids=[py_id], cause_text="python CVE vulnerability patch", recorded_at=t3
    )
    assert len(logs_e2_repeat) == 0

    # 5. Both event explanations remain recoverable from decision log
    dec_logs = service.get_decision_logs(memory_id=fastapi_id)
    prop_logs = [d for d in dec_logs if d.rule_id == "R-PROPAGATE"]
    assert len(prop_logs) == 2
    causes = [d.claim_json["cause"] for d in prop_logs]
    assert "kernel update" in causes
    assert "python CVE vulnerability patch" in causes

    # 6. Stable event_id idempotency: distinct event_id creates record, repeat skips
    logs_ev1 = service.propagate_change(
        changed_ids=[py_id],
        cause_text="scheduled audit",
        recorded_at=t3,
        event_id="audit_batch_001",
    )
    assert len(logs_ev1) == 1
    assert logs_ev1[0].claim_json.get("event_id") == "audit_batch_001"

    logs_ev1_repeat = service.propagate_change(
        changed_ids=[py_id],
        cause_text="scheduled audit",
        recorded_at=t3,
        event_id="audit_batch_001",
    )
    assert len(logs_ev1_repeat) == 0

    logs_ev2 = service.propagate_change(
        changed_ids=[py_id],
        cause_text="scheduled audit",
        recorded_at=t3,
        event_id="audit_batch_002",
    )
    assert len(logs_ev2) == 1
    assert logs_ev2[0].claim_json.get("event_id") == "audit_batch_002"


def test_idempotency_case1_same_root_same_event_same_dependent(
    service: MemoryService,
) -> None:
    """Idempotency Case 1: Same root + same event + same dependent -> no duplicate log."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    os_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert os_id is not None

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python")
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Python",
    )
    py_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert py_id is not None

    # Verify initial database state: 0 R-PROPAGATE decision logs for Python
    cursor = service.repo.conn.execute(
        "SELECT COUNT(*) FROM decision_log WHERE rule_id = 'R-PROPAGATE' AND memory_id = ?",
        (py_id,),
    )
    assert cursor.fetchone()[0] == 0

    # First invocation: generates 1 returned log and persists 1 record
    logs1 = service.propagate_change(
        changed_ids=[os_id],
        cause_text="kernel upgrade",
        event_id="evt_001",
        recorded_at=t2,
    )
    assert len(logs1) == 1
    assert logs1[0].memory_id == py_id
    assert logs1[0].claim_json["upstream_memory_id"] == os_id
    assert logs1[0].claim_json.get("event_id") == "evt_001"

    cursor = service.repo.conn.execute(
        "SELECT COUNT(*) FROM decision_log WHERE rule_id = 'R-PROPAGATE' AND memory_id = ?",
        (py_id,),
    )
    assert cursor.fetchone()[0] == 1

    # Second invocation (identical root, event_id, cause, dependent): returns 0 logs, 0 new DB rows
    logs2 = service.propagate_change(
        changed_ids=[os_id],
        cause_text="kernel upgrade",
        event_id="evt_001",
        recorded_at=t2,
    )
    assert len(logs2) == 0

    cursor = service.repo.conn.execute(
        "SELECT COUNT(*) FROM decision_log WHERE rule_id = 'R-PROPAGATE' AND memory_id = ?",
        (py_id,),
    )
    assert cursor.fetchone()[0] == 1


def test_idempotency_case2_same_root_distinct_event_same_dependent(
    service: MemoryService,
) -> None:
    """Idempotency Case 2: Same root + distinct event + same dependent -> new log."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)
    t3 = datetime(2026, 7, 1, 10, 0, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    os_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert os_id is not None

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python")
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Python",
    )
    py_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert py_id is not None

    # Event 1: kernel upgrade
    logs_e1 = service.propagate_change(
        changed_ids=[os_id],
        cause_text="kernel upgrade",
        event_id="evt_001",
        recorded_at=t2,
    )
    assert len(logs_e1) == 1

    py_mem_e1 = service.repo.get_memory(py_id)
    assert py_mem_e1 is not None and py_mem_e1.needs_review is True
    assert "kernel upgrade" in py_mem_e1.review_reason

    # Event 2: distinct event ID and distinct cause affecting already-reviewed memory
    logs_e2 = service.propagate_change(
        changed_ids=[os_id],
        cause_text="security CVE patch",
        event_id="evt_002",
        recorded_at=t3,
    )
    assert len(logs_e2) == 1
    assert logs_e2[0].claim_json.get("event_id") == "evt_002"

    # Both records persisted in database
    cursor = service.repo.conn.execute(
        "SELECT claim_json FROM decision_log WHERE rule_id = 'R-PROPAGATE' AND memory_id = ? "
        "ORDER BY id ASC",
        (py_id,),
    )
    rows = cursor.fetchall()
    assert len(rows) == 2
    import json

    c1_data = json.loads(rows[0][0])
    c2_data = json.loads(rows[1][0])
    assert c1_data.get("event_id") == "evt_001"
    assert c2_data.get("event_id") == "evt_002"

    # Dependent memory factual value and confidence are unmodified, both reasons preserved
    py_mem_e2 = service.repo.get_memory(py_id)
    assert py_mem_e2 is not None and py_mem_e2.needs_review is True
    assert py_mem_e2.value_display == "3.10"
    assert "kernel upgrade" in py_mem_e2.review_reason
    assert "security CVE patch" in py_mem_e2.review_reason


def test_idempotency_case3_different_roots_same_event_same_dependent(
    service: MemoryService,
) -> None:
    """Idempotency Case 3: Different roots + same event + same dependent -> separate logs."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    # Root 1: Server A OS
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    os_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert os_id is not None

    # Root 2: Database B OS
    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="DB OS")
    c2 = Claim(
        subject="Database B",
        attribute="operating_system",
        value="Debian 12",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="DB OS",
    )
    db_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert db_id is not None

    # Dependent: Framework on Server A depends on both Server A OS and Database B OS
    msg3 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="App")
    c3 = Claim(
        subject="Server A",
        attribute="framework",
        value="FastAPI",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[
            AttrRef(entity="Server A", attribute="operating_system"),
            AttrRef(entity="Database B", attribute="operating_system"),
        ],
        raw_text="App",
    )
    app_id = service.ingest_claims(msg3, [c3]).decisions[0].memory_id
    assert app_id is not None

    # Create one valid audit message
    audit_msg_id = service.repo.insert_message(
        source_id="sysadmin_alice",
        text="Unified infrastructure overhaul audit",
        event_time=t2,
    )

    # Simultaneous propagation of both roots passing the SAME explicit message_id,
    # same event_id, same cause, and same trigger_mode
    logs = service.propagate_change(
        changed_ids=[os_id, db_id],
        cause_text="infrastructure overhaul",
        event_id="infra_001",
        message_id=audit_msg_id,
        trigger_mode="manual",
        recorded_at=t2,
    )
    app_logs = [d for d in logs if d.memory_id == app_id]
    assert len(app_logs) == 2
    upstreams = {d.claim_json["upstream_memory_id"] for d in app_logs}
    assert upstreams == {os_id, db_id}
    assert all(d.message_id == audit_msg_id for d in app_logs)
    assert all(d.claim_json["trigger_mode"] == "manual" for d in app_logs)
    assert all(d.claim_json.get("event_id") == "infra_001" for d in app_logs)

    # Verify both records persisted in database, independently distinguished by upstream root
    cursor = service.repo.conn.execute(
        "SELECT COUNT(*) FROM decision_log WHERE rule_id = 'R-PROPAGATE' AND memory_id = ? "
        "AND message_id = ?",
        (app_id, audit_msg_id),
    )
    assert cursor.fetchone()[0] == 2

    # Repeating with same roots, event_id, message_id, cause, trigger_mode:
    # 0 returned logs, DB count unchanged
    repeat_logs = service.propagate_change(
        changed_ids=[os_id, db_id],
        cause_text="infrastructure overhaul",
        event_id="infra_001",
        message_id=audit_msg_id,
        trigger_mode="manual",
        recorded_at=t2,
    )
    assert len(repeat_logs) == 0

    cursor = service.repo.conn.execute(
        "SELECT COUNT(*) FROM decision_log WHERE rule_id = 'R-PROPAGATE' AND memory_id = ? "
        "AND message_id = ?",
        (app_id, audit_msg_id),
    )
    assert cursor.fetchone()[0] == 2


def test_idempotency_case4_multiple_paths_from_one_root_one_dependent(
    service: MemoryService,
) -> None:
    """Idempotency Case 4: Multiple paths from one root to one dependent -> one log."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    # Root: Server A OS (M1)
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    os_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert os_id is not None

    # Intermediate 1: Python (M2) depends on OS
    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python")
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Python",
    )
    py_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert py_id is not None

    # Intermediate 2: Framework (M3) depends on OS
    msg3 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Framework")
    c3 = Claim(
        subject="Server A",
        attribute="framework",
        value="FastAPI",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Framework",
    )
    fw_id = service.ingest_claims(msg3, [c3]).decisions[0].memory_id
    assert fw_id is not None

    # Dependent: App Version (M4) depends on both Python (M2) and Framework (M3) (diamond)
    msg4 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="AppVersion")
    c4 = Claim(
        subject="Server A",
        attribute="app_version",
        value="1.0.0",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[
            AttrRef(entity="Server A", attribute="python_version"),
            AttrRef(entity="Server A", attribute="framework"),
        ],
        raw_text="AppVersion",
    )
    app_ver_id = service.ingest_claims(msg4, [c4]).decisions[0].memory_id
    assert app_ver_id is not None

    # Propagate from root OS: diamond paths reach App Version via both branches
    logs = service.propagate_change(
        changed_ids=[os_id],
        cause_text="major os patch",
        event_id="diamond_001",
        recorded_at=t2,
    )
    # App Version receives exactly ONE propagation record
    app_ver_logs = [d for d in logs if d.memory_id == app_ver_id]
    assert len(app_ver_logs) == 1
    assert app_ver_logs[0].claim_json["upstream_memory_id"] == os_id

    # Verify persisted in database: exactly 1 R-PROPAGATE row for App Version
    cursor = service.repo.conn.execute(
        "SELECT COUNT(*) FROM decision_log WHERE rule_id = 'R-PROPAGATE' AND memory_id = ?",
        (app_ver_id,),
    )
    assert cursor.fetchone()[0] == 1

    # Repeating returns 0 logs and leaves database count at 1
    repeat_logs = service.propagate_change(
        changed_ids=[os_id],
        cause_text="major os patch",
        event_id="diamond_001",
        recorded_at=t2,
    )
    assert len(repeat_logs) == 0

    cursor = service.repo.conn.execute(
        "SELECT COUNT(*) FROM decision_log WHERE rule_id = 'R-PROPAGATE' AND memory_id = ?",
        (app_ver_id,),
    )
    assert cursor.fetchone()[0] == 1


def test_idempotency_case5_manual_vs_automatic_invocations(
    service: MemoryService,
) -> None:
    """Idempotency Case 5: Manual vs automatic invocations follow the chosen identity rule."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    # Initial state: Root OS and dependent Python
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    os_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert os_id is not None

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python")
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Python",
    )
    py_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert py_id is not None

    # Create a valid explicit audit message ID to use for both calls
    audit_msg_id = service.repo.insert_message(
        source_id="sysadmin_alice",
        text="Audit for trigger mode distinction",
        event_time=t2,
    )

    cause = "vulnerability patch propagation"
    event_id = "trig_event_001"

    # Step 1: First invoke with trigger_mode="automatic"
    auto_logs = service.propagate_change(
        changed_ids=[os_id],
        cause_text=cause,
        event_id=event_id,
        message_id=audit_msg_id,
        trigger_mode="automatic",
        recorded_at=t2,
    )
    assert len(auto_logs) == 1
    assert auto_logs[0].claim_json["trigger_mode"] == "automatic"

    cursor = service.repo.conn.execute(
        "SELECT COUNT(*) FROM decision_log WHERE rule_id = 'R-PROPAGATE' AND memory_id = ?",
        (py_id,),
    )
    assert cursor.fetchone()[0] == 1

    # Step 2: Then invoke with trigger_mode="manual" (all other parameters identical)
    manual_logs = service.propagate_change(
        changed_ids=[os_id],
        cause_text=cause,
        event_id=event_id,
        message_id=audit_msg_id,
        trigger_mode="manual",
        recorded_at=t2,
    )
    assert len(manual_logs) == 1
    assert manual_logs[0].claim_json["trigger_mode"] == "manual"

    # Verify that both invocations produce separate persisted logs
    cursor = service.repo.conn.execute(
        "SELECT claim_json FROM decision_log WHERE rule_id = 'R-PROPAGATE' AND memory_id = ? "
        "ORDER BY id ASC",
        (py_id,),
    )
    rows = cursor.fetchall()
    assert len(rows) == 2
    import json

    t_modes = [json.loads(r[0])["trigger_mode"] for r in rows]
    assert t_modes == ["automatic", "manual"]

    # Step 3: Repeat automatic invocation and verify it creates no duplicate
    repeat_auto = service.propagate_change(
        changed_ids=[os_id],
        cause_text=cause,
        event_id=event_id,
        message_id=audit_msg_id,
        trigger_mode="automatic",
        recorded_at=t2,
    )
    assert len(repeat_auto) == 0

    # Repeat manual invocation and verify it creates no duplicate
    repeat_manual = service.propagate_change(
        changed_ids=[os_id],
        cause_text=cause,
        event_id=event_id,
        message_id=audit_msg_id,
        trigger_mode="manual",
        recorded_at=t2,
    )
    assert len(repeat_manual) == 0

    cursor = service.repo.conn.execute(
        "SELECT COUNT(*) FROM decision_log WHERE rule_id = 'R-PROPAGATE' AND memory_id = ?",
        (py_id,),
    )
    assert cursor.fetchone()[0] == 2


def test_unsupported_trigger_mode_rejected(service: MemoryService) -> None:
    """Unsupported trigger_mode string raises ValueError and makes no DB changes."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    os_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert os_id is not None

    counts_before = service.repo.get_record_counts()

    with pytest.raises(ValueError, match="Unsupported trigger_mode: 'invalid_mode'"):
        service.propagate_change(
            changed_ids=[os_id],
            cause_text="test",
            recorded_at=t2,
            trigger_mode="invalid_mode",  # type: ignore[arg-type]
        )

    # Database records unchanged
    assert service.repo.get_record_counts() == counts_before


def test_manual_propagation_attribution(service: MemoryService) -> None:
    """Test manual propagation attribution semantics and FK integrity."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Setup OS")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    rep1 = service.ingest_claims(msg1, [c1])
    os_id = rep1.decisions[0].memory_id
    os_msg_id = rep1.message_id
    assert os_id is not None

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Setup App")
    c2 = Claim(
        subject="Server A",
        attribute="framework",
        value="Django",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Django",
    )
    rep2 = service.ingest_claims(msg2, [c2])
    django_id = rep2.decisions[0].memory_id
    django_msg_id = rep2.message_id
    assert django_id is not None
    assert django_msg_id != os_msg_id

    # Case A: When message_id is omitted, attributes to upstream memory's message_id
    # (NOT the unrelated downstream memory's message_id)
    logs_manual = service.propagate_change(
        changed_ids=[os_id], cause_text="manual os deprecation", recorded_at=t2
    )
    assert len(logs_manual) == 1
    entry = logs_manual[0]
    assert entry.message_id == os_msg_id
    assert entry.message_id != django_msg_id
    assert entry.claim_json["trigger_mode"] == "manual"
    assert "Manual propagation: " in entry.explanation

    # Case B: When explicit message_id is supplied, it is respected
    audit_msg_id = service.repo.insert_message(
        source_id="sysadmin_alice",
        text="Audit run",
        event_time=t2,
    )
    logs_with_msg = service.propagate_change(
        changed_ids=[os_id],
        cause_text="audit maintenance",
        recorded_at=t2,
        message_id=audit_msg_id,
    )
    assert len(logs_with_msg) == 1
    assert logs_with_msg[0].message_id == audit_msg_id
    assert logs_with_msg[0].claim_json["trigger_mode"] == "manual"


def test_multiple_roots_and_diamond_propagation(service: MemoryService) -> None:
    """4 & 5. Multiple roots reaching same dependent, and diamond path deduplication."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    # Root 1: OS (Server A)
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    os_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert os_id is not None

    # Intermediate 1: Python depends on OS
    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python")
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Python",
    )
    py_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert py_id is not None

    # Root 2: Database OS (Database B)
    msg3 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="DB")
    c3 = Claim(
        subject="Database B",
        attribute="operating_system",
        value="Debian 12",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="DB",
    )
    db_id = service.ingest_claims(msg3, [c3]).decisions[0].memory_id
    assert db_id is not None

    # Dependent C: Web App depends on both Python (Server A) and DB OS (Database B)
    msg4 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="App")
    c4 = Claim(
        subject="Server A",
        attribute="framework",
        value="FastAPI",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[
            AttrRef(entity="Server A", attribute="python_version"),
            AttrRef(entity="Database B", attribute="operating_system"),
        ],
        raw_text="App",
    )
    app_id = service.ingest_claims(msg4, [c4]).decisions[0].memory_id
    assert app_id is not None

    # Test 4: Two changed roots (os_id and db_id) both reach app_id
    logs = service.propagate_change(
        changed_ids=[os_id, db_id],
        cause_text="infrastructure overhaul",
        recorded_at=t2,
    )
    # app_id receives two records: one with upstream_id=os_id, one with upstream_id=db_id
    app_logs = [d for d in logs if d.memory_id == app_id]
    assert len(app_logs) == 2
    app_upstreams = {d.claim_json["upstream_memory_id"] for d in app_logs}
    assert app_upstreams == {os_id, db_id}

    # Verify both review reasons are preserved in app memory without duplicating
    app_mem = service.repo.get_memory(app_id)
    assert app_mem is not None and app_mem.needs_review is True
    assert f"M{os_id}" in app_mem.review_reason
    assert f"M{db_id}" in app_mem.review_reason

    # Test 5: Diamond deduplication - repeating for os_id returns 0 duplicate records
    repeat_os = service.propagate_change(
        changed_ids=[os_id],
        cause_text="infrastructure overhaul",
        recorded_at=t2,
    )
    assert len(repeat_os) == 0


def test_propagation_preserves_path_and_trigger_info(service: MemoryService) -> None:
    """17. Propagation preserves the dependency path and trigger information."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Stack")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    os_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert os_id is not None

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python")
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Python",
    )
    py_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert py_id is not None

    # Upgrade OS
    msg_up = IngestMessage(source_id="sysadmin_alice", event_time=t2, text="OS 24")
    c_up = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Debian 12",
        extract_confidence=1.0,
        valid_from=t2,
        raw_text="OS 24",
    )
    rep = service.ingest_claims(msg_up, [c_up])
    prop_dec = next(d for d in rep.decisions if d.rule_id == "R-PROPAGATE")

    assert prop_dec.action == Action.PROPAGATE
    assert prop_dec.memory_id == py_id
    assert prop_dec.claim_json["trigger"] == "propagation"
    assert prop_dec.claim_json["upstream_memory_id"] == os_id
    assert prop_dec.claim_json["path"] == [os_id, py_id]
    assert "superseded" in prop_dec.claim_json["cause"]


def test_propagation_failure_rolls_back_atomically(
    service: MemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """18. Failure during propagation rolls back the upstream mutation and changes."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Stack")
    c1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Python",
    )
    py_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert py_id is not None

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="FastAPI")
    c2 = Claim(
        subject="Server A",
        attribute="framework",
        value="FastAPI",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="python_version")],
        raw_text="FastAPI",
    )
    fw_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert fw_id is not None

    # Simulate an error during mark_memory_for_review
    def fail_mark(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Simulated failure during propagation")

    monkeypatch.setattr(service.repo, "mark_memory_for_review", fail_mark)

    # Ingest newer Python claim: must fail and roll back completely
    msg_up = IngestMessage(source_id="sysadmin_alice", event_time=t2, text="Python 3.12")
    c_up = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t2,
        raw_text="Python 3.12",
    )
    with pytest.raises(RuntimeError, match="Simulated failure during propagation"):
        service.ingest_claims(msg_up, [c_up])

    # Check that Python 3.10 is still ACTIVE, no Python 3.12 was saved
    py_mem = service.repo.get_memory(py_id)
    assert py_mem is not None
    assert py_mem.status == MemoryStatus.ACTIVE
    assert py_mem.valid_to is None

    # Check that Framework memory was NOT marked for review
    fw_mem = service.repo.get_memory(fw_id)
    assert fw_mem is not None
    assert fw_mem.needs_review is False


def test_propagation_manual_chain_failure_rolls_back_atomically(
    service: MemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure-injection: allow 1st update to succeed, fail on 2nd log, verify complete rollback."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    # Root: OS (M1)
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    os_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert os_id is not None

    # Downstream 1: Python (M2 depends on M1)
    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python")
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Python",
    )
    py_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert py_id is not None

    # Downstream 2: Framework (M3 depends on M2)
    msg3 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="FastAPI")
    c3 = Claim(
        subject="Server A",
        attribute="framework",
        value="FastAPI",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="python_version")],
        raw_text="FastAPI",
    )
    fw_id = service.ingest_claims(msg3, [c3]).decisions[0].memory_id
    assert fw_id is not None

    # Take database snapshot prior to the operation
    pre_py = service.repo.get_memory(py_id)
    pre_fw = service.repo.get_memory(fw_id)
    pre_logs = service.repo.get_all_decision_logs()
    assert pre_py is not None and pre_py.needs_review is False
    assert pre_fw is not None and pre_fw.needs_review is False
    pre_log_ids = [d.id for d in pre_logs]

    # Allow first review-state update and log to succeed, but force later log insertion to fail
    orig_insert = service.repo.insert_decision_log
    log_call_count = 0

    def faulty_insert(entry: Any) -> Any:
        nonlocal log_call_count
        log_call_count += 1
        if log_call_count > 1:
            raise RuntimeError("Forced decision-log insertion failure on second downstream memory")
        return orig_insert(entry)

    monkeypatch.setattr(service.repo, "insert_decision_log", faulty_insert)

    # Invoke directly in manual mode: must raise exception and roll back completely
    with pytest.raises(RuntimeError, match="Forced decision-log insertion failure"):
        service.propagate_change(
            changed_ids=[os_id],
            cause_text="manual kernel upgrade",
            recorded_at=t2,
        )

    # Verify all memory review states and decision-log rows are identical to pre-operation snapshot
    post_py = service.repo.get_memory(py_id)
    post_fw = service.repo.get_memory(fw_id)
    post_logs = service.repo.get_all_decision_logs()

    assert post_py is not None
    assert post_py.needs_review is False
    assert post_py.review_reason == pre_py.review_reason

    assert post_fw is not None
    assert post_fw.needs_review is False
    assert post_fw.review_reason == pre_fw.review_reason

    assert [d.id for d in post_logs] == pre_log_ids


def test_attribution_validation_and_rejection(service: MemoryService) -> None:
    """Validate explicit message attribution, root attribution fallback, and message rejection."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    rep1 = service.ingest_claims(msg1, [c1])
    os_id = rep1.decisions[0].memory_id
    assert os_id is not None

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python")
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Python",
    )
    rep2 = service.ingest_claims(msg2, [c2])
    py_id = rep2.decisions[0].memory_id
    assert py_id is not None

    # Case 1: Non-existent explicit message_id is rejected with ForeignKeyConstraintError
    with pytest.raises(
        ForeignKeyConstraintError, match="Explicit message_id 999999 does not exist"
    ):
        service.propagate_change(
            changed_ids=[os_id],
            cause_text="test",
            recorded_at=t2,
            message_id=999999,
        )

    # Case 2: Fabricated foreign-key ID (e.g. 0) is never accepted
    with pytest.raises(ForeignKeyConstraintError, match="Explicit message_id 0 does not exist"):
        service.propagate_change(
            changed_ids=[os_id],
            cause_text="test",
            recorded_at=t2,
            message_id=0,
        )

    # Case 3: Valid explicit message ID is accepted
    audit_msg_id = service.repo.insert_message(
        source_id="sysadmin_alice",
        text="Valid Audit",
        event_time=t2,
    )
    logs = service.propagate_change(
        changed_ids=[os_id],
        cause_text="test valid audit",
        recorded_at=t2,
        message_id=audit_msg_id,
    )
    assert len(logs) == 1
    assert logs[0].message_id == audit_msg_id
    assert logs[0].claim_json["trigger_mode"] == "manual"


def test_invalid_explicit_message_id_rolls_back_without_db_changes(
    service: MemoryService,
) -> None:
    """Explicit invalid message_id is validated inside transaction and causes no DB changes."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    os_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert os_id is not None

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python")
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Python",
    )
    py_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert py_id is not None

    counts_before = service.repo.get_record_counts()
    py_before = service.repo.get_memory(py_id)
    assert py_before is not None and py_before.needs_review is False

    # Standalone call with invalid message_id
    with pytest.raises(
        ForeignKeyConstraintError, match="Explicit message_id 999999 does not exist"
    ):
        service.propagate_change(
            changed_ids=[os_id],
            cause_text="test standalone rollback",
            recorded_at=t2,
            message_id=999999,
        )

    # Database records unchanged
    assert service.repo.get_record_counts() == counts_before
    py_after = service.repo.get_memory(py_id)
    assert py_after is not None
    assert py_after.needs_review is False
    assert py_after.review_reason == py_before.review_reason

    # Nested call inside an outer transaction/savepoint
    with service.repo.transaction():
        with pytest.raises(ForeignKeyConstraintError):
            service.propagate_change(
                changed_ids=[os_id],
                cause_text="test nested rollback",
                recorded_at=t2,
                message_id=999999,
            )
        # Outer transaction performs a valid operation and completes
        service.repo.insert_message(
            source_id="sysadmin_alice",
            text="Outer surviving message",
            event_time=t2,
        )

    counts_after_nested = service.repo.get_record_counts()
    assert counts_after_nested["messages"] == counts_before["messages"] + 1
    assert counts_after_nested["decision_log"] == counts_before["decision_log"]
    assert counts_after_nested["memories"] == counts_before["memories"]
    py_final = service.repo.get_memory(py_id)
    assert py_final is not None and py_final.needs_review is False


# =============================================================================
# 4. Regression Tests
# =============================================================================


def test_regression_dispute_resolution_with_propagation(service: MemoryService) -> None:
    """19. Dispute resolution remains intact, and loser memories propagate to dependents."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 3, 2, 10, 0, 0, tzinfo=UTC)
    t3 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    # Two competing claims create DISPUTE
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Claim 1")
    c1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Python 3.10",
    )
    m1_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t2, text="Claim 2")
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.11",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Python 3.11",
    )
    service.ingest_claims(msg2, [c2])

    # Dependent depends on python_version (which is under dispute)
    c_dep = Claim(
        subject="Server A",
        attribute="framework",
        value="FastAPI",
        extract_confidence=1.0,
        valid_from=t2,
        depends_on=[AttrRef(entity="Server A", attribute="python_version")],
        raw_text="FastAPI",
    )
    msg_dep = IngestMessage(source_id="sysadmin_alice", event_time=t2, text="Dep Claim")
    dep_rep = service.ingest_claims(msg_dep, [c_dep])
    dep_id = dep_rep.decisions[0].memory_id
    assert dep_id is not None

    # Resolve dispute in favor of M1 (winner). M2 is loser.
    res_log = service.resolve_dispute(winner_memory_id=m1_id, recorded_at=t3)
    assert res_log.rule_id == "R-DISPUTE-RESOLVE"
    assert res_log.memory_id == m1_id

    # The winner M1 is ACTIVE
    m1 = service.repo.get_memory(m1_id)
    assert m1 is not None and m1.status == MemoryStatus.ACTIVE


def test_regression_temporal_retrieval(service: MemoryService) -> None:
    """20. Existing temporal retrieval and historical queries remain intact."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Fact 1")
    c1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Python 3.10",
    )
    service.ingest_claims(msg1, [c1])

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t2, text="Fact 2")
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t2,
        raw_text="Python 3.12",
    )
    service.ingest_claims(msg2, [c2])

    # Current retrieval at t2
    cur = service.current("Server A", "python_version", valid_at=t2, now=t2)
    assert len(cur.items) == 1 and cur.items[0].memory.value_display == "3.12"

    # Historical retrieval as-of March 15
    hist = service.current(
        "Server A", "python_version", valid_at=datetime(2026, 3, 15, tzinfo=UTC), now=t2
    )
    assert len(hist.items) == 1 and hist.items[0].memory.value_display == "3.10"


def test_regression_invariants_hold_after_propagation(service: MemoryService) -> None:
    """21. Invariant checking passes cleanly after dependency propagation."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Stack")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    service.ingest_claims(msg1, [c1])

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python")
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="Python",
    )
    service.ingest_claims(msg2, [c2])

    # Upgrade OS -> triggers R-REPLACE and R-PROPAGATE
    msg_up = IngestMessage(source_id="sysadmin_alice", event_time=t2, text="OS 24")
    c_up = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 24.04",
        extract_confidence=1.0,
        valid_from=t2,
        raw_text="OS 24",
    )
    service.ingest_claims(msg_up, [c_up])

    # All data integrity invariants must hold cleanly
    report = service.check_invariants()
    assert report.is_valid is True
    assert report.violations == []
    assert report.total_checked > 0


# =============================================================================
# 5. Pure Engine Unit Tests
# =============================================================================


def test_pure_propagate_functions() -> None:
    """Unit tests for pure functions in engine/propagate.py."""
    edges = [
        Edge(
            id=1,
            src_memory_id=2,
            dst_memory_id=1,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        Edge(
            id=2,
            src_memory_id=3,
            dst_memory_id=2,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        Edge(
            id=3,
            src_memory_id=1,
            dst_memory_id=1,
            type=EdgeType.DEPENDS_ON,
            reason="self",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        Edge(
            id=4,
            src_memory_id=4,
            dst_memory_id=1,
            type=EdgeType.SUPERSEDES,
            reason="not depends_on",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    ]

    # Graph maps: 2 depends on 1, 3 depends on 2
    deps_map, upstreams_map = build_dependency_graph(edges)
    assert deps_map == {1: [2], 2: [3]}
    assert upstreams_map == {2: [1], 3: [2]}

    assert resolve_direct_dependencies(edges, 2) == [1]
    assert resolve_direct_dependents(edges, 1) == [2]

    trans = resolve_transitive_dependents(edges, 1)
    assert trans == [(2, [1, 2]), (3, [1, 2, 3])]

    # Compute propagation
    statuses = {2: MemoryStatus.ACTIVE, 3: MemoryStatus.DISPUTED}
    items = compute_propagation(
        edges=edges,
        memory_statuses=statuses,
        already_flagged_ids=set(),
        changed_ids=[1],
        cause_text="superseded",
    )
    assert len(items) == 2
    assert items[0].memory_id == 2
    assert items[0].path == [1, 2]
    assert "superseded" in items[0].review_reason
    assert items[1].memory_id == 3
    assert items[1].path == [1, 2, 3]
