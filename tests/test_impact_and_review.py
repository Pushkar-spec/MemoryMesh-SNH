"""Tests for P5.2: Impact Analysis and Review Acknowledgement.

Governed by docs/SPECIFICATION.md Section 2 and docs/RESOLUTION_ENGINE.md Section 6.3.
Tests cover:
1. Pure engine compute_impact_paths (trees, chains, diamond, cycles, boundaries).
2. Service impact analysis:
   - Root with no dependents
   - Direct dependent
   - Multi-level dependency chain
   - Diamond dependency graph
   - Multiple paths to same dependent (deduplicated memories, multiple paths preserved)
   - Dependency cycle safety
   - Missing root memory error handling
   - Correct dependency path direction
   - Correct memory status and needs_review reporting
   - Deterministic output
   - Read-only safety (zero database mutations)
3. Service review acknowledgement:
   - Successful acknowledgement of needs_review memory
   - Flag cleared and review_reason cleared
   - Recorded in audit/decision log with Action.ACKNOWLEDGE and R-ACKNOWLEDGE
   - Invariance of factual value, evidence, confidence, validity intervals, status, and edges
   - Independent isolation (acknowledging one memory does not acknowledge downstream dependents)
   - Rejection when memory is not marked for review (ValueError)
   - Nonexistent memory handling (MemoryNotFoundError)
   - Timezone-aware timestamp validation and UTC normalization
   - Idempotent repeated acknowledgement (returns existing log, no duplicate rows)
   - Atomic rollback on transaction failure
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from memorymesh.config import load_app_config
from memorymesh.db import init_db
from memorymesh.engine.propagate import (
    compute_impact_paths,
)
from memorymesh.errors import MemoryNotFoundError
from memorymesh.models import (
    Action,
    AttributeDef,
    AttrRef,
    Cardinality,
    Claim,
    DecisionLogEntry,
    Edge,
    EdgeType,
    IngestMessage,
    Source,
    ValueType,
)
from memorymesh.repository import Repository
from memorymesh.service import MemoryService


@pytest.fixture
def service() -> MemoryService:
    """Fixture providing an initialized MemoryService with test sources and schemas."""
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
            attribute="database_engine",
            description="Database Engine",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )

    cfg = load_app_config()
    return MemoryService(db_target=repo, cfg=cfg)


# =============================================================================
# 1. Pure Engine Unit Tests: compute_impact_paths
# =============================================================================


def test_pure_compute_impact_paths_empty_and_disconnected() -> None:
    """Engine: Empty edges or disconnected nodes produce empty impact paths."""
    assert compute_impact_paths([], 1) == []

    # Edge exists between 2 and 3, but root 1 has no dependents
    edge = Edge(
        id=1,
        src_memory_id=3,
        dst_memory_id=2,
        type=EdgeType.DEPENDS_ON,
        reason="",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert compute_impact_paths([edge], 1) == []


def test_pure_compute_impact_paths_diamond_and_cycles() -> None:
    """Engine: Correctly discovers diamond paths and handles cycles without infinite loops."""
    # Diamond graph: 2 -> 1, 3 -> 1, 4 -> 2, 4 -> 3
    # plus a cycle: 2 -> 4 (cycle 2 -> 4 -> 2)
    # plus self loop: 1 -> 1
    t = datetime(2026, 1, 1, tzinfo=UTC)
    edges = [
        Edge(
            id=1,
            src_memory_id=2,
            dst_memory_id=1,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=2,
            src_memory_id=3,
            dst_memory_id=1,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=3,
            src_memory_id=4,
            dst_memory_id=2,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=4,
            src_memory_id=4,
            dst_memory_id=3,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=5,
            src_memory_id=2,
            dst_memory_id=4,
            type=EdgeType.DEPENDS_ON,
            reason="cycle",
            created_at=t,
        ),
        Edge(
            id=6,
            src_memory_id=1,
            dst_memory_id=1,
            type=EdgeType.DEPENDS_ON,
            reason="self",
            created_at=t,
        ),
    ]

    res = compute_impact_paths(edges, root_id=1)
    dep_ids = [r[0] for r in res]
    assert dep_ids == [2, 3, 4]  # root 1 is not in dep_ids

    # Check 4 has two paths from 1: [1, 2, 4] and [1, 3, 4]
    item_4 = next(r for r in res if r[0] == 4)
    assert item_4[1] in ([1, 2, 4], [1, 3, 4])
    assert [1, 2, 4] in item_4[2]
    assert [1, 3, 4] in item_4[2]
    assert item_4[3] is False


def test_pure_compute_impact_paths_bounds_and_truncation() -> None:
    """Engine: When alternate paths exceed limit, paths_truncated is set to True."""
    t = datetime(2026, 1, 1, tzinfo=UTC)
    # Diamond graph: 1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4
    edges = [
        Edge(
            id=1,
            src_memory_id=2,
            dst_memory_id=1,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=2,
            src_memory_id=3,
            dst_memory_id=1,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=3,
            src_memory_id=4,
            dst_memory_id=2,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=4,
            src_memory_id=4,
            dst_memory_id=3,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
    ]
    # Set max_paths_per_dependent=1 so node 4 has truncated alternate paths
    res = compute_impact_paths(edges, root_id=1, max_paths_per_dependent=1)
    item_4 = next(r for r in res if r[0] == 4)
    assert len(item_4[2]) == 1
    assert item_4[3] is True


@pytest.mark.parametrize("invalid_val", [0, -1, -10, True, False, 1.5, "10"])
def test_pure_compute_impact_paths_invalid_max_paths(invalid_val: Any) -> None:
    """compute_impact_paths rejects non-positive or non-integer max_paths_per_dependent."""
    with pytest.raises(ValueError, match="max_paths_per_dependent must be a positive integer"):
        compute_impact_paths([], 1, max_paths_per_dependent=invalid_val)


@pytest.mark.parametrize("invalid_val", [0, -1, -50, True, False, 2.5, "500"])
def test_pure_compute_impact_paths_invalid_max_budget(invalid_val: Any) -> None:
    """compute_impact_paths rejects non-positive or non-integer max_search_budget."""
    with pytest.raises(ValueError, match="max_search_budget must be a positive integer"):
        compute_impact_paths([], 1, max_search_budget=invalid_val)


def test_pure_compute_impact_paths_parent_reconstruction_long_chain() -> None:
    """Phase 1 parent tracking accurately reconstructs shortest paths along long chains."""
    t = datetime(2026, 1, 1, tzinfo=UTC)
    # Linear chain of 60 edges (nodes 1..61)
    # Edge: src=dep, dst=upstream
    edges = [
        Edge(
            id=i,
            src_memory_id=i + 1,
            dst_memory_id=i,
            type=EdgeType.DEPENDS_ON,
            reason="chain",
            created_at=t,
        )
        for i in range(1, 61)
    ]

    res = compute_impact_paths(edges, root_id=1)
    assert len(res) == 60

    # Verify every node k has primary_path = [1, 2, ..., k]
    for dep_id, primary_path, all_paths, truncated in res:
        expected = list(range(1, dep_id + 1))
        assert primary_path == expected
        assert all_paths == [expected]
        assert truncated is False


def test_pure_compute_impact_paths_shortest_path_accuracy() -> None:
    """Parent reconstruction accurately preserves the true shortest path among multiple routes."""
    t = datetime(2026, 1, 1, tzinfo=UTC)
    # Multiple paths from 1 to 4:
    # 1-hop: 4 -> 1
    # 2-hop: 5 -> 1, 4 -> 5
    # 3-hop: 2 -> 1, 3 -> 2, 4 -> 3
    edges = [
        Edge(
            id=1,
            src_memory_id=4,
            dst_memory_id=1,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=2,
            src_memory_id=5,
            dst_memory_id=1,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=3,
            src_memory_id=4,
            dst_memory_id=5,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=4,
            src_memory_id=2,
            dst_memory_id=1,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=5,
            src_memory_id=3,
            dst_memory_id=2,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=6,
            src_memory_id=4,
            dst_memory_id=3,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
    ]

    res = compute_impact_paths(edges, root_id=1)
    item_4 = next(r for r in res if r[0] == 4)

    # Primary path must be the shortest (1-hop) [1, 4]
    assert item_4[1] == [1, 4]
    # all_paths must contain all 3 routes sorted by length
    assert item_4[2] == [[1, 4], [1, 5, 4], [1, 2, 3, 4]]
    assert item_4[3] is False


def test_pure_compute_impact_paths_high_outdegree_small_budget() -> None:
    """High-outdegree graphs bound candidate path expansions and preserve complete reachability."""
    t = datetime(2026, 1, 1, tzinfo=UTC)
    # Root 1 has 30 direct dependents (2..31)
    # Each node 2..31 has a child (32..61)
    edges: list[Edge] = []
    edge_id = 1
    for d in range(2, 32):
        edges.append(
            Edge(
                id=edge_id,
                src_memory_id=d,
                dst_memory_id=1,
                type=EdgeType.DEPENDS_ON,
                reason="",
                created_at=t,
            )
        )
        edge_id += 1
        edges.append(
            Edge(
                id=edge_id,
                src_memory_id=d + 30,
                dst_memory_id=d,
                type=EdgeType.DEPENDS_ON,
                reason="",
                created_at=t,
            )
        )
        edge_id += 1

    # Run with small search budget of 5 candidate expansions
    res = compute_impact_paths(edges, root_id=1, max_search_budget=5)

    # Complete reachability must be preserved: all 60 dependents discovered in Phase 1
    assert len(res) == 60
    discovered_ids = {r[0] for r in res}
    assert discovered_ids == set(range(2, 62))

    # Because budget (5) was far less than candidate edges, truncation flag is accurately set
    truncated_count = sum(1 for r in res if r[3] is True)
    assert truncated_count > 0


def test_pure_compute_impact_paths_deterministic_across_repeated_runs() -> None:
    """compute_impact_paths returns identical, deterministic results across repeated runs."""
    t = datetime(2026, 1, 1, tzinfo=UTC)
    # Diamond graph with cross edges
    edges = [
        Edge(
            id=1,
            src_memory_id=2,
            dst_memory_id=1,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=2,
            src_memory_id=3,
            dst_memory_id=1,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=3,
            src_memory_id=4,
            dst_memory_id=2,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=4,
            src_memory_id=4,
            dst_memory_id=3,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=5,
            src_memory_id=5,
            dst_memory_id=4,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
        Edge(
            id=6,
            src_memory_id=5,
            dst_memory_id=2,
            type=EdgeType.DEPENDS_ON,
            reason="",
            created_at=t,
        ),
    ]

    baseline = compute_impact_paths(edges, root_id=1)
    for _ in range(5):
        assert compute_impact_paths(edges, root_id=1) == baseline


# =============================================================================
# 2. Service Impact Analysis Tests
# =============================================================================


def test_impact_missing_root_memory_raises_error(service: MemoryService) -> None:
    """Impact analysis on non-existent memory ID raises MemoryNotFoundError."""
    with pytest.raises(MemoryNotFoundError, match="Memory with id=999999 not found"):
        service.impact(999999)


def test_impact_root_with_no_dependents(service: MemoryService) -> None:
    """Impact analysis on memory with zero dependents returns empty list."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Standalone OS")
    claim = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Standalone OS",
    )
    rep = service.ingest_claims(msg, [claim])
    os_id = rep.decisions[0].memory_id
    assert os_id is not None

    impact_items = service.impact(os_id)
    assert impact_items == []


def test_impact_direct_and_transitive_chain(service: MemoryService) -> None:
    """Impact analysis traces multi-level chain: OS (M1) -> Python (M2) -> Framework (M3)."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)

    # M1: OS
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

    # M2: Python depends on OS
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

    # M3: Framework depends on Python
    msg3 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Framework")
    c3 = Claim(
        subject="Server A",
        attribute="framework",
        value="FastAPI",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="python_version")],
        raw_text="Framework",
    )
    fw_id = service.ingest_claims(msg3, [c3]).decisions[0].memory_id
    assert fw_id is not None

    # Analyze impact from root OS (os_id)
    impact_items = service.impact(os_id)
    assert len(impact_items) == 2

    # Check root memory is NOT included in its own dependents
    assert all(item.memory.memory.id != os_id for item in impact_items)

    # Item 1: Python (direct dependent)
    item_py = impact_items[0]
    assert item_py.memory.memory.id == py_id
    assert item_py.path == [os_id, py_id]
    assert item_py.all_paths == [[os_id, py_id]]
    assert "Direct dependent" in item_py.reason
    assert "status=active" in item_py.reason
    assert "needs_review=False" in item_py.reason

    # Item 2: Framework (transitive dependent)
    item_fw = impact_items[1]
    assert item_fw.memory.memory.id == fw_id
    assert item_fw.path == [os_id, py_id, fw_id]
    assert item_fw.all_paths == [[os_id, py_id, fw_id]]
    assert "Transitive dependent" in item_fw.reason
    assert f"M{os_id} -> M{py_id} -> M{fw_id}" in item_fw.reason


def test_impact_diamond_graph_deduplicates_and_preserves_multiple_paths(
    service: MemoryService,
) -> None:
    """Diamond graph: Root -> B1 & B2 -> D preserves both paths without duplicate entries."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)

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

    # Intermediate 2: Database (M3) depends on OS
    msg3 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="DB")
    c3 = Claim(
        subject="Server A",
        attribute="database_engine",
        value="PostgreSQL",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
        raw_text="DB",
    )
    db_id = service.ingest_claims(msg3, [c3]).decisions[0].memory_id
    assert db_id is not None

    # Dependent: App (M4) depends on both Python and Database (Diamond)
    msg4 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="App")
    c4 = Claim(
        subject="Server A",
        attribute="app_version",
        value="1.0.0",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[
            AttrRef(entity="Server A", attribute="python_version"),
            AttrRef(entity="Server A", attribute="database_engine"),
        ],
        raw_text="App",
    )
    app_id = service.ingest_claims(msg4, [c4]).decisions[0].memory_id
    assert app_id is not None

    impact_items = service.impact(os_id)
    # Exactly 3 items: py_id, db_id, app_id (app_id is NOT duplicated)
    assert len(impact_items) == 3
    affected_ids = [item.memory.memory.id for item in impact_items]
    assert affected_ids == [py_id, db_id, app_id]

    app_item = impact_items[2]
    assert app_item.memory.memory.id == app_id
    # Both diamond paths are preserved
    assert len(app_item.all_paths) == 2
    assert [os_id, py_id, app_id] in app_item.all_paths
    assert [os_id, db_id, app_id] in app_item.all_paths
    assert "also reachable via" in app_item.reason


def test_impact_cycle_safety(service: MemoryService) -> None:
    """Impact analysis terminates cleanly and safely when a dependency cycle exists."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)

    # Node 1: OS
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    m1_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert m1_id is not None

    # Node 2: Python depends on OS
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
    m2_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert m2_id is not None

    # Node 3: Framework depends on Python
    msg3 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Framework")
    c3 = Claim(
        subject="Server A",
        attribute="framework",
        value="Django",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="python_version")],
        raw_text="Framework",
    )
    m3_id = service.ingest_claims(msg3, [c3]).decisions[0].memory_id
    assert m3_id is not None

    # Artificially inject a cycle: M2 depends on M3 (M2 -> M3 -> M2)
    service.repo.insert_edge(
        src_memory_id=m2_id,
        dst_memory_id=m3_id,
        edge_type=EdgeType.DEPENDS_ON,
        reason="synthetic cycle test",
        created_at=t1,
    )

    # Must terminate without recursion error or infinite loop
    impact_items = service.impact(m1_id)
    assert len(impact_items) == 2
    ids = [item.memory.memory.id for item in impact_items]
    assert ids == [m2_id, m3_id]


def test_impact_read_only_and_deterministic(service: MemoryService) -> None:
    """Impact analysis is strictly read-only and produces deterministic outputs."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)

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
    service.ingest_claims(msg2, [c2])

    counts_before = service.repo.get_record_counts()

    run1 = service.impact(os_id)
    run2 = service.impact(os_id)

    # Identical deterministic output
    assert len(run1) == len(run2) == 1
    assert run1[0].memory.memory.id == run2[0].memory.memory.id
    assert run1[0].path == run2[0].path
    assert run1[0].reason == run2[0].reason

    # Zero database modifications
    counts_after = service.repo.get_record_counts()
    assert counts_before == counts_after


def test_impact_reflects_propagation_and_review_state(service: MemoryService) -> None:
    """Impact analysis incorporates review flags and R-PROPAGATE audit history."""
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

    # Upgrade OS -> triggers R-REPLACE and automatic R-PROPAGATE
    msg_up = IngestMessage(source_id="sysadmin_alice", event_time=t2, text="OS 24")
    c_up = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Debian 12",
        extract_confidence=1.0,
        valid_from=t2,
        raw_text="OS 24",
    )
    service.ingest_claims(msg_up, [c_up])

    impact_items = service.impact(os_id)
    assert len(impact_items) == 1
    item = impact_items[0]
    assert item.memory.memory.id == py_id
    assert item.memory.memory.needs_review is True
    assert "needs_review=True" in item.reason
    assert "propagated via R-PROPAGATE" in item.reason


# =============================================================================
# 3. Service Review Acknowledgement Tests
# =============================================================================


def test_acknowledge_review_successful(service: MemoryService) -> None:
    """Successfully acknowledge a memory marked needs_review and verify audit logging."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)
    t3 = datetime(2026, 6, 2, 12, 0, 0, tzinfo=UTC)

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

    # Trigger propagation on OS upgrade
    service.propagate_change(changed_ids=[os_id], cause_text="kernel update", recorded_at=t2)
    py_mem = service.repo.get_memory(py_id)
    assert py_mem is not None and py_mem.needs_review is True
    assert "kernel update" in py_mem.review_reason

    # Snapshot state prior to acknowledgement
    val_before = py_mem.value_display
    conf_before = py_mem.confidence
    valid_from_before = py_mem.valid_from
    valid_to_before = py_mem.valid_to
    status_before = py_mem.status
    edges_before = service.repo.get_edges_for_memory(py_id)
    ev_before = service.repo.get_evidence_for_memory(py_id)

    # Acknowledge review
    ack_log = service.acknowledge_review(
        memory_id=py_id,
        reviewed_by="sysadmin_alice",
        explanation="Verified Python 3.10 works fine with new kernel",
        recorded_at=t3,
    )

    # 1. Verify returned audit log
    assert ack_log.action == Action.ACKNOWLEDGE
    assert ack_log.rule_id == "R-ACKNOWLEDGE"
    assert ack_log.memory_id == py_id
    assert ack_log.affected_memory_ids == [py_id]
    assert ack_log.claim_json["reviewed_by"] == "sysadmin_alice"
    assert "kernel update" in ack_log.claim_json["cleared_reason"]
    assert ack_log.created_at == t3

    # 2. Verify needs_review is cleared in database
    py_after = service.repo.get_memory(py_id)
    assert py_after is not None
    assert py_after.needs_review is False
    assert py_after.review_reason == ""

    # 3. Verify factual data, confidence, evidence, intervals, and edges are 100% unchanged
    assert py_after.value_display == val_before
    assert py_after.confidence == conf_before
    assert py_after.valid_from == valid_from_before
    assert py_after.valid_to == valid_to_before
    assert py_after.status == status_before
    assert service.repo.get_edges_for_memory(py_id) == edges_before
    assert service.repo.get_evidence_for_memory(py_id) == ev_before


def test_acknowledge_review_does_not_clear_dependent_flags(service: MemoryService) -> None:
    """Acknowledging one memory must not clear review flags for downstream dependents."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)

    # OS (M1) -> Python (M2) -> Framework (M3)
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

    msg3 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Framework")
    c3 = Claim(
        subject="Server A",
        attribute="framework",
        value="FastAPI",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="python_version")],
        raw_text="Framework",
    )
    fw_id = service.ingest_claims(msg3, [c3]).decisions[0].memory_id
    assert fw_id is not None

    # Flag both downstream memories
    service.propagate_change(changed_ids=[os_id], cause_text="os upgrade", recorded_at=t2)
    assert service.repo.get_memory(py_id).needs_review is True
    assert service.repo.get_memory(fw_id).needs_review is True

    # Acknowledge only Python (py_id)
    service.acknowledge_review(py_id, recorded_at=t2)

    # Python is cleared, but Framework MUST REMAIN needs_review=True
    assert service.repo.get_memory(py_id).needs_review is False
    fw_mem = service.repo.get_memory(fw_id)
    assert fw_mem is not None
    assert fw_mem.needs_review is True
    assert "os upgrade" in fw_mem.review_reason


def test_acknowledge_review_non_review_memory_raises_error(service: MemoryService) -> None:
    """Cannot silently acknowledge a memory that is not flagged for review."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
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
    assert service.repo.get_memory(os_id).needs_review is False

    with pytest.raises(ValueError, match=f"Memory M{os_id} is not marked for review"):
        service.acknowledge_review(os_id)


def test_acknowledge_review_nonexistent_memory_raises_error(service: MemoryService) -> None:
    """Acknowledging non-existent memory ID raises MemoryNotFoundError."""
    with pytest.raises(MemoryNotFoundError, match="Memory with ID 999999 does not exist"):
        service.acknowledge_review(999999)


def test_acknowledge_review_timestamp_validation(service: MemoryService) -> None:
    """Naive timestamps are rejected; timezone-aware non-UTC timestamps are normalized."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    claim = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    os_id = service.ingest_claims(msg, [claim]).decisions[0].memory_id
    assert os_id is not None
    service.repo.mark_memory_for_review(os_id, "manual flag")

    # Naive datetime rejected
    naive_t = datetime(2026, 6, 1, 10, 0, 0)
    with pytest.raises(ValueError, match="Recorded timestamp must be a timezone-aware datetime"):
        service.acknowledge_review(os_id, recorded_at=naive_t)

    # Timezone with non-zero offset normalized to UTC
    tz_offset = timedelta(hours=5, minutes=30)
    from datetime import timezone as dt_timezone

    aware_t = datetime(2026, 6, 1, 15, 30, 0, tzinfo=dt_timezone(tz_offset))
    log = service.acknowledge_review(os_id, recorded_at=aware_t)
    assert log.created_at == datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)


def test_acknowledge_review_idempotent_repeat(service: MemoryService) -> None:
    """Repeating review acknowledgement is idempotent and does not create duplicate logs."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    claim = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    os_id = service.ingest_claims(msg, [claim]).decisions[0].memory_id
    assert os_id is not None
    service.repo.mark_memory_for_review(os_id, "test review flag")

    # First acknowledgement
    log1 = service.acknowledge_review(os_id, reviewed_by="alice")
    cursor = service.repo.conn.execute(
        "SELECT COUNT(*) FROM decision_log WHERE rule_id = 'R-ACKNOWLEDGE' AND memory_id = ?",
        (os_id,),
    )
    assert cursor.fetchone()[0] == 1

    # Second acknowledgement (idempotent repeat)
    log2 = service.acknowledge_review(os_id, reviewed_by="alice")
    assert log2.id == log1.id

    cursor = service.repo.conn.execute(
        "SELECT COUNT(*) FROM decision_log WHERE rule_id = 'R-ACKNOWLEDGE' AND memory_id = ?",
        (os_id,),
    )
    assert cursor.fetchone()[0] == 1


def test_acknowledge_review_atomic_rollback_on_failure(
    service: MemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure during acknowledgement audit write rolls back memory review state changes."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    claim = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    os_id = service.ingest_claims(msg, [claim]).decisions[0].memory_id
    assert os_id is not None
    service.repo.mark_memory_for_review(os_id, "critical check required")

    def fail_insert(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Simulated crash during decision log insertion")

    monkeypatch.setattr(service.repo, "insert_decision_log", fail_insert)

    with pytest.raises(RuntimeError, match="Simulated crash during decision log insertion"):
        service.acknowledge_review(os_id)

    # Memory state must remain rolled back: needs_review must still be True!
    mem_after = service.repo.get_memory(os_id)
    assert mem_after is not None
    assert mem_after.needs_review is True
    assert mem_after.review_reason == "critical check required"


def test_impact_chain_longer_than_50_edges(service: MemoryService) -> None:
    """Impact analysis must not truncate chains longer than 50 edges.

    Builds a linear dependency chain of 55 edges (56 memories: M1 -> M2 -> ... -> M56).
    Verifies that all 55 downstream dependents are discovered, and the 56th memory
    is present at the end of the chain.
    """
    t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    mem_ids: list[int] = []
    for i in range(1, 57):
        msg = IngestMessage(source_id="sysadmin_alice", event_time=t0, text=f"Step {i}")
        claim = Claim(
            subject=f"srv{i:02d}",
            attribute="operating_system",
            value=f"OS_{i}",
            extract_confidence=0.9,
            valid_from=t0,
            raw_text=f"srv{i:02d}",
        )
        report = service.ingest_claims(msg, [claim])
        m_id = report.decisions[0].memory_id
        assert m_id is not None
        mem_ids.append(m_id)

    # Link in chain: mem_ids[i] depends on mem_ids[i-1] (55 edges)
    # Edge: src = dependent = mem_ids[i], dst = dependency = mem_ids[i-1]
    for i in range(1, len(mem_ids)):
        service.repo.insert_edge(
            src_memory_id=mem_ids[i],
            dst_memory_id=mem_ids[i - 1],
            edge_type=EdgeType.DEPENDS_ON,
            reason="chain",
            created_at=t0,
        )

    # Analyze impact from root (mem_ids[0])
    impact_items = service.impact(mem_ids[0])

    # Must contain exactly 55 dependents
    assert len(impact_items) == 55

    # Final dependent (mem_ids[55], i.e. 56th memory) must be present
    final_dep = next(it for it in impact_items if it.memory.memory.id == mem_ids[-1])
    assert len(final_dep.path) == 56
    assert final_dep.path[0] == mem_ids[0]
    assert final_dep.path[-1] == mem_ids[-1]
    assert final_dep.paths_truncated is False


def test_acknowledge_review_not_confused_after_reflagging(service: MemoryService) -> None:
    """An old acknowledgement must not be mistaken for a new acknowledgement if re-flagged.

    Lifecycle:
    1. Memory is flagged for review.
    2. Acknowledged by alice -> R-ACKNOWLEDGE log 1.
    3. Re-flagged via propagation -> R-PROPAGATE log 2, needs_review=True.
    4. Acknowledging again must generate a new R-ACKNOWLEDGE log 3, distinct from log 1.
    5. Repeating acknowledgement immediately without re-flagging is idempotent and returns log 3.
    """
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    claim = Claim(
        subject="Server B",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    m_id = service.ingest_claims(msg, [claim]).decisions[0].memory_id
    assert m_id is not None

    # Step 1: Flag for review
    service.repo.mark_memory_for_review(m_id, "first issue")

    # Step 2: First acknowledgement
    log1 = service.acknowledge_review(m_id, reviewed_by="alice")
    assert log1.rule_id == "R-ACKNOWLEDGE"
    assert log1.claim_json.get("reviewed_by") == "alice"

    mem = service.repo.get_memory(m_id)
    assert mem is not None
    assert mem.needs_review is False

    # Step 3: Re-flag memory via propagation
    service.repo.mark_memory_for_review(m_id, "second upstream issue")
    prop_log = DecisionLogEntry(
        id=0,
        message_id=mem.message_id,
        claim_json={"trigger": "propagation", "upstream_memory_id": 999, "cause": "disputed"},
        action=Action.PROPAGATE,
        rule_id="R-PROPAGATE",
        memory_id=m_id,
        affected_memory_ids=[m_id],
        scores={},
        explanation="Re-flagged due to upstream change",
        created_at=t1 + timedelta(hours=1),
    )
    service.repo.insert_decision_log(prop_log)

    mem_reflagged = service.repo.get_memory(m_id)
    assert mem_reflagged is not None
    assert mem_reflagged.needs_review is True

    # Step 4: Acknowledge review again
    log2 = service.acknowledge_review(m_id, reviewed_by="bob")
    assert log2.id != log1.id
    assert log2.rule_id == "R-ACKNOWLEDGE"
    assert log2.claim_json.get("reviewed_by") == "bob"

    # Step 5: Repeat acknowledgement immediately without re-flagging -> idempotent
    log3 = service.acknowledge_review(m_id, reviewed_by="bob")
    assert log3.id == log2.id


def test_acknowledge_review_nested_transaction_rollback(service: MemoryService) -> None:
    """Outer transaction rollback rolls back acknowledgement changes (savepoint support)."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    claim = Claim(
        subject="Server C",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="OS",
    )
    m_id = service.ingest_claims(msg, [claim]).decisions[0].memory_id
    assert m_id is not None
    service.repo.mark_memory_for_review(m_id, "check needed")

    with pytest.raises(RuntimeError, match="Simulated outer failure"):
        with service.repo.transaction():
            service.acknowledge_review(m_id, reviewed_by="alice")
            # Outer block fails
            raise RuntimeError("Simulated outer failure")

    # State must be rolled back: needs_review must remain True
    mem = service.repo.get_memory(m_id)
    assert mem is not None
    assert mem.needs_review is True
    assert mem.review_reason == "check needed"

    # Decision log entry must not be committed
    ack_log = service.repo.get_current_acknowledgement_log(m_id)
    assert ack_log is None


def test_reviewer_attribution_persisted_and_factual_invariance(service: MemoryService) -> None:
    """Reviewer attribution is persisted in the database and factual memory fields are immutable."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    claim = Claim(
        subject="Server D",
        attribute="operating_system",
        value="Ubuntu 22.04",
        extract_confidence=0.85,
        valid_from=t1,
        raw_text="OS",
    )
    m_id = service.ingest_claims(msg, [claim]).decisions[0].memory_id
    assert m_id is not None
    service.repo.mark_memory_for_review(m_id, "security audit needed")

    mem_before = service.repo.get_memory(m_id)
    assert mem_before is not None

    ack_log = service.acknowledge_review(
        m_id,
        reviewed_by="SecOps_Lead",
        explanation="Verified signed hash",
    )
    assert ack_log.claim_json["reviewed_by"] == "SecOps_Lead"

    # Check persistence directly in database
    cursor = service.repo.conn.execute(
        "SELECT claim_json, explanation FROM decision_log WHERE id = ?",
        (ack_log.id,),
    )
    row = cursor.fetchone()
    assert row is not None
    persisted_claim = row[0]
    assert '"reviewed_by": "SecOps_Lead"' in persisted_claim
    assert "Verified signed hash" in row[1]

    # Verify factual immutability
    mem_after = service.repo.get_memory(m_id)
    assert mem_after is not None
    assert mem_after.value_norm == mem_before.value_norm
    assert mem_after.value_display == mem_before.value_display
    assert mem_after.confidence == mem_before.confidence
    assert mem_after.valid_from == mem_before.valid_from
    assert mem_after.valid_to == mem_before.valid_to
    assert mem_after.status == mem_before.status
    assert mem_after.needs_review is False
    assert mem_after.review_reason == ""
