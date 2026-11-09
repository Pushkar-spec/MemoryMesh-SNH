"""Tests for Context Scopes and Backfill (P6.1)."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from memorymesh.config import load_app_config
from memorymesh.db import init_db
from memorymesh.errors import ConflictResolutionError, InvalidValidityWindowError
from memorymesh.models import (
    Action,
    AttrRef,
    Claim,
    EdgeType,
    IngestMessage,
    MemoryStatus,
)
from memorymesh.retrieval.retrieve import build_context_key, normalize_context
from memorymesh.service import MemoryService


@pytest.fixture
def service() -> Generator[MemoryService, None, None]:
    """Isolated in-memory MemoryService loaded with standard configuration."""
    conn = init_db(":memory:")
    cfg = load_app_config()
    svc = MemoryService(db_target=conn, cfg=cfg)
    try:
        yield svc
    finally:
        conn.close()


def test_context_isolation_same_entity_attribute_different_contexts(
    service: MemoryService,
) -> None:
    """Claims for the same entity and attribute in different contexts do not conflict."""
    t1 = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)

    # Ingest Project A runtime: 3.10
    msg_a = IngestMessage(
        source_id="sysadmin_priya", text="Project A uses Python 3.10", event_time=t1
    )
    claim_a = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        valid_from=t1,
        context={"project": "Project A"},
        extract_confidence=0.95,
    )
    rep_a = service.ingest_claims(msg_a, [claim_a], recorded_at=t1)
    assert len(rep_a.decisions) == 1
    assert rep_a.decisions[0].rule_id == "R-INSERT"
    assert rep_a.decisions[0].action == Action.INSERT

    # Ingest Project B runtime: 3.12 (same entity, same attribute, different context)
    msg_b = IngestMessage(
        source_id="sysadmin_priya", text="Project B uses Python 3.12", event_time=t1
    )
    claim_b = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=t1,
        context={"project": "Project B"},
        extract_confidence=0.95,
    )
    rep_b = service.ingest_claims(msg_b, [claim_b], recorded_at=t1)
    assert len(rep_b.decisions) == 1
    assert rep_b.decisions[0].rule_id == "R-INSERT"
    assert rep_b.decisions[0].action == Action.INSERT

    # Query with Project A context
    res_a = service.current("Server A", "python_version", context={"project": "Project A"}, now=t1)
    assert len(res_a.items) == 1
    assert res_a.items[0].memory.value_norm == "3.10"
    assert res_a.items[0].scope_match == "exact"
    assert res_a.items[0].memory.context == {"project": "Project A"}
    assert res_a.items[0].memory.context_key == '{"project":"Project A"}'

    # Query with Project B context
    res_b = service.current("Server A", "python_version", context={"project": "Project B"}, now=t1)
    assert len(res_b.items) == 1
    assert res_b.items[0].memory.value_norm == "3.12"
    assert res_b.items[0].scope_match == "exact"
    assert res_b.items[0].memory.context == {"project": "Project B"}
    assert res_b.items[0].memory.context_key == '{"project":"Project B"}'


def test_context_conflict_within_same_context(service: MemoryService) -> None:
    """Conflicting claims within the same context trigger normal replacement."""
    t1 = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    ctx = {"project": "Project A"}

    # Project A initial claim: 3.10
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Initial 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        valid_from=t1,
        context=ctx,
        extract_confidence=0.95,
    )
    service.ingest_claims(msg1, [claim1], recorded_at=t1)

    # Project A newer update: 3.12
    msg2 = IngestMessage(source_id="sysadmin_priya", text="Upgrade to 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=t2,
        context=ctx,
        extract_confidence=0.95,
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    assert len(rep2.decisions) == 1
    assert rep2.decisions[0].rule_id == "R-REPLACE"
    assert rep2.decisions[0].action == Action.REPLACE

    # Verify Project A now has 3.12 as current
    res = service.current("Server A", "python_version", context=ctx, now=t2)
    assert len(res.items) == 1
    assert res.items[0].memory.value_norm == "3.12"
    assert res.items[0].memory.status == MemoryStatus.ACTIVE


def test_context_default_behavior_when_no_context_supplied(service: MemoryService) -> None:
    """Claims without context default to global context (context_key='')."""
    t1 = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)

    msg = IngestMessage(source_id="sysadmin_priya", text="Default global claim", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=t1,
        extract_confidence=0.95,
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t1)
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    mem = service.repo.get_memory(m_id)
    assert mem is not None
    assert mem.context == {}
    assert mem.context_key == ""

    # Query with None and {} both resolve to the global memory
    res_none = service.current("Server A", "python_version", context=None, now=t1)
    assert len(res_none.items) == 1
    assert res_none.items[0].memory.id == m_id
    assert res_none.items[0].scope_match == "exact"

    res_empty = service.current("Server A", "python_version", context={}, now=t1)
    assert len(res_empty.items) == 1
    assert res_empty.items[0].memory.id == m_id
    assert res_empty.items[0].scope_match == "exact"


def test_context_specificity_and_inherited_global_fallback(service: MemoryService) -> None:
    """Scope matching selects highest specificity match, falling back to global."""
    t1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 20, 0, 0, tzinfo=UTC)

    # 1. Global memory: Python 3.12
    msg_global = IngestMessage(source_id="sysadmin_priya", text="Global runtime", event_time=t1)
    claim_global = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=t1,
        context={},
        extract_confidence=0.95,
    )
    service.ingest_claims(msg_global, [claim_global], recorded_at=t1)

    # 2. Staging memory: Python 3.11
    msg_staging = IngestMessage(source_id="sysadmin_priya", text="Staging runtime", event_time=t2)
    claim_staging = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.11",
        valid_from=t2,
        context={"environment": "staging"},
        extract_confidence=0.95,
    )
    service.ingest_claims(msg_staging, [claim_staging], recorded_at=t2)

    t_now = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)

    # Query staging: exact match returns 3.11 (specificity 1 > specificity 0)
    res_staging = service.current(
        "Server A", "python_version", context={"environment": "staging"}, now=t_now
    )
    assert len(res_staging.items) == 1
    assert res_staging.items[0].memory.value_norm == "3.11"
    assert res_staging.items[0].scope_match == "exact"

    # Query production: no production memory exists, falls back to inherited global 3.12
    res_prod = service.current(
        "Server A", "python_version", context={"environment": "production"}, now=t_now
    )
    assert len(res_prod.items) == 1
    assert res_prod.items[0].memory.value_norm == "3.12"
    assert res_prod.items[0].scope_match == "inherited_global"

    # Query global: returns 3.12
    res_global = service.current("Server A", "python_version", context={}, now=t_now)
    assert len(res_global.items) == 1
    assert res_global.items[0].memory.value_norm == "3.12"
    assert res_global.items[0].scope_match == "exact"


def test_context_persistence_after_db_reopen(tmp_path: Path) -> None:
    """Context information persists faithfully across database reconnection."""
    db_file = tmp_path / "context_test.db"
    conn = init_db(str(db_file))
    cfg = load_app_config()
    svc = MemoryService(db_target=conn, cfg=cfg)

    t = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_priya", text="Context message", event_time=t)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.11",
        valid_from=t,
        context={"env": "qa", "region": "us-east"},
        extract_confidence=0.95,
    )
    svc.ingest_claims(msg, [claim], recorded_at=t)
    conn.close()

    # Reopen connection
    conn2 = sqlite3.connect(str(db_file))
    conn2.row_factory = sqlite3.Row
    svc2 = MemoryService(db_target=conn2, cfg=cfg)
    try:
        res = svc2.current(
            "Server A", "python_version", context={"env": "qa", "region": "us-east"}, now=t
        )
        assert len(res.items) == 1
        mem = res.items[0].memory
        assert mem.value_norm == "3.11"
        assert mem.context == {"env": "qa", "region": "us-east"}
        assert mem.context_key == '{"env":"qa","region":"us-east"}'
        assert res.items[0].scope_match == "exact"
    finally:
        conn2.close()


def test_context_aware_historical_retrieval(service: MemoryService) -> None:
    """Historical retrieval operates within the requested context scope."""
    t1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    ctx = {"project": "Alpha"}

    msg1 = IngestMessage(source_id="sysadmin_priya", text="Alpha 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        valid_from=t1,
        context=ctx,
        extract_confidence=0.95,
    )
    service.ingest_claims(msg1, [claim1], recorded_at=t1)

    msg2 = IngestMessage(source_id="sysadmin_priya", text="Alpha 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=t2,
        context=ctx,
        extract_confidence=0.95,
    )
    service.ingest_claims(msg2, [claim2], recorded_at=t2)

    # Historical query as-of March 1 in Alpha context
    t_hist = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    res_hist = service.current("Server A", "python_version", context=ctx, valid_at=t_hist, now=t2)
    assert len(res_hist.items) == 1
    assert res_hist.items[0].memory.value_norm == "3.10"
    assert res_hist.items[0].memory.status == MemoryStatus.HISTORICAL

    # Current query in Alpha context
    res_curr = service.current("Server A", "python_version", context=ctx, now=t2)
    assert len(res_curr.items) == 1
    assert res_curr.items[0].memory.value_norm == "3.12"
    assert res_curr.items[0].memory.status == MemoryStatus.ACTIVE


def test_context_aware_decision_and_explanation(service: MemoryService) -> None:
    """Decision logging and explain notes reflect the memory's context scope."""
    t = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_priya", text="Scoped claim", event_time=t)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=t,
        context={"environment": "staging"},
        extract_confidence=0.95,
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t)
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    exp = service.explain(m_id)
    assert any('Context scope: \'{"environment":"staging"}\'' in note for note in exp.notes)

    # Decision logs verify claim_json retains context
    logs = service.get_decision_logs(m_id)
    assert len(logs) >= 1
    assert logs[0].claim_json["context"] == {"environment": "staging"}


def test_cross_context_dependency_and_propagation(service: MemoryService) -> None:
    """Scoped memory can depend on inherited global memory; upstream changes propagate."""
    t1 = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)

    # Global runtime memory
    msg_py = IngestMessage(source_id="sysadmin_priya", text="Global Python 3.12", event_time=t1)
    claim_py = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=t1,
        extract_confidence=0.95,
    )
    rep_py = service.ingest_claims(msg_py, [claim_py], recorded_at=t1)
    m_py_id = rep_py.decisions[0].memory_id
    assert m_py_id is not None

    # Staging dependent service
    msg_svc = IngestMessage(
        source_id="sysadmin_priya",
        text="Staging API gateway on Server A",
        event_time=t1,
    )
    claim_svc = Claim(
        subject="Server A",
        attribute="runs_service",
        value="api_gateway",
        valid_from=t1,
        context={"environment": "staging"},
        depends_on=[
            AttrRef(entity="Server A", attribute="python_version", extract_confidence=0.95)
        ],
        extract_confidence=0.95,
    )
    rep_svc = service.ingest_claims(msg_svc, [claim_svc], recorded_at=t1)
    m_svc_id = rep_svc.decisions[0].memory_id
    assert m_svc_id is not None

    # Verify cross-context DEPENDS_ON edge: M_svc (staging) -> M_py (global)
    edges = service.repo.get_edges_for_memory(m_svc_id, direction="outgoing")
    dep_edges = [e for e in edges if e.type == EdgeType.DEPENDS_ON]
    assert len(dep_edges) == 1
    assert dep_edges[0].dst_memory_id == m_py_id

    # Propagate change on M_py
    prop_logs = service.propagate_change(
        changed_ids=[m_py_id],
        cause_text="CVE vulnerability in runtime",
        trigger_mode="manual",
    )
    assert len(prop_logs) == 1
    assert prop_logs[0].memory_id == m_svc_id

    down_mem = service.repo.get_memory(m_svc_id)
    assert down_mem is not None
    assert down_mem.needs_review is True
    assert "CVE vulnerability in runtime" in down_mem.review_reason


def test_context_ablation_use_context_false(service: MemoryService) -> None:
    """Ablating use_context forces context_key to '' so staging and prod facts collide."""
    service.cfg.engine.ablation.use_context = False

    t1 = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)

    # Ingest Staging 3.11
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Staging 3.11", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.11",
        valid_from=t1,
        context={"environment": "staging"},
        extract_confidence=0.95,
    )
    service.ingest_claims(msg1, [claim1], recorded_at=t1)

    # Ingest Production 3.12 (with use_context=False, this replaces 3.11 because both are global)
    msg2 = IngestMessage(source_id="sysadmin_priya", text="Prod 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=t2,
        context={"environment": "production"},
        extract_confidence=0.95,
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    assert len(rep2.decisions) == 1
    assert rep2.decisions[0].rule_id == "R-REPLACE"
    assert rep2.decisions[0].action == Action.REPLACE


def test_backfill_historical_claim_valid_vs_recorded_time_distinct(
    service: MemoryService,
) -> None:
    """Ingesting a historical claim records valid_from from the past and recorded_at from now."""
    t_past = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
    t_now = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)

    msg = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A was provisioned with Python 3.8 back in 2024",
        event_time=t_past,
    )
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.8",
        valid_from=t_past,
        extract_confidence=0.95,
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t_now)
    assert len(rep.decisions) == 1
    assert rep.decisions[0].rule_id == "R-INSERT"
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    mem = service.repo.get_memory(m_id)
    assert mem is not None
    assert mem.valid_from == t_past
    assert mem.recorded_at == t_now


def test_backfill_older_claim_after_newer_claim_exists(service: MemoryService) -> None:
    """Ingesting an older claim after a newer claim already exists triggers R-BACKFILL."""
    t1_old = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    t2_new = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t_ingest = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)

    # 1. Establish newer memory first (Python 3.12 starting 2026-01-01)
    msg_new = IngestMessage(source_id="sysadmin_priya", text="Newer fact", event_time=t2_new)
    claim_new = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=t2_new,
        extract_confidence=0.95,
    )
    rep_new = service.ingest_claims(msg_new, [claim_new], recorded_at=t_ingest)
    assert rep_new.decisions[0].rule_id == "R-INSERT"
    m_new_id = rep_new.decisions[0].memory_id
    assert m_new_id is not None

    # 2. Backfill older memory (Python 3.10 valid 2025-01-01)
    msg_old = IngestMessage(
        source_id="sysadmin_priya",
        text="Older fact discovered later",
        event_time=t1_old,
    )
    claim_old = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        valid_from=t1_old,
        extract_confidence=0.95,
    )
    rep_old = service.ingest_claims(msg_old, [claim_old], recorded_at=t_ingest)
    assert len(rep_old.decisions) == 1
    assert rep_old.decisions[0].rule_id == "R-BACKFILL"
    assert rep_old.decisions[0].action == Action.BACKFILL

    m_backfill_id = rep_old.decisions[0].memory_id
    assert m_backfill_id is not None
    m_backfill = service.repo.get_memory(m_backfill_id)
    assert m_backfill is not None
    assert m_backfill.status == MemoryStatus.HISTORICAL
    assert m_backfill.valid_from == t1_old
    assert m_backfill.valid_to == t2_new
    assert m_backfill.closed_at == t_ingest

    # 3. Newer memory remains ACTIVE and open-ended
    mem_new_after = service.repo.get_memory(m_new_id)
    assert mem_new_after is not None
    assert mem_new_after.status == MemoryStatus.ACTIVE
    assert mem_new_after.valid_from == t2_new
    assert mem_new_after.valid_to is None

    # 4. Point-in-time retrieval
    t_mid = datetime(2025, 6, 1, 0, 0, tzinfo=UTC)
    m_as_of = service.get_as_of("Server A", "python_version", valid_at=t_mid)
    assert m_as_of is not None
    assert m_as_of.value_norm == "3.10"
    assert m_as_of.status == MemoryStatus.HISTORICAL

    m_cur = service.get_as_of("Server A", "python_version", valid_at=t_ingest)
    assert m_cur is not None
    assert m_cur.value_norm == "3.12"
    assert m_cur.status == MemoryStatus.ACTIVE


def test_backfill_multiple_historical_changes_chronological_order(
    service: MemoryService,
) -> None:
    """Three historical changes ingested in chronological order create contiguous intervals."""
    t1 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    t3 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t_ingest = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)

    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Python 3.8", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.8",
                valid_from=t1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_ingest,
    )
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Python 3.10", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t2,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_ingest,
    )
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Python 3.12", event_time=t3),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t3,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_ingest,
    )

    m1 = service.get_as_of("Server A", "python_version", valid_at=datetime(2024, 6, 1, tzinfo=UTC))
    m2 = service.get_as_of("Server A", "python_version", valid_at=datetime(2025, 6, 1, tzinfo=UTC))
    m3 = service.get_as_of("Server A", "python_version", valid_at=datetime(2026, 6, 1, tzinfo=UTC))

    assert m1 is not None and m1.value_norm == "3.8" and m1.status == MemoryStatus.HISTORICAL
    assert m2 is not None and m2.value_norm == "3.10" and m2.status == MemoryStatus.HISTORICAL
    assert m3 is not None and m3.value_norm == "3.12" and m3.status == MemoryStatus.ACTIVE

    report = service.check_invariants()
    assert report.is_valid is True


def test_backfill_multiple_historical_changes_reverse_order(service: MemoryService) -> None:
    """Three historical changes ingested in reverse order yield identical intervals."""
    t1 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    t3 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t_ingest = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)

    # Ingest t3, then t2, then t1
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Python 3.12", event_time=t3),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t3,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_ingest,
    )
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Python 3.10", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t2,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_ingest,
    )
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Python 3.8", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.8",
                valid_from=t1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_ingest,
    )

    m1 = service.get_as_of("Server A", "python_version", valid_at=datetime(2024, 6, 1, tzinfo=UTC))
    m2 = service.get_as_of("Server A", "python_version", valid_at=datetime(2025, 6, 1, tzinfo=UTC))
    m3 = service.get_as_of("Server A", "python_version", valid_at=datetime(2026, 6, 1, tzinfo=UTC))

    assert m1 is not None and m1.value_norm == "3.8" and m1.status == MemoryStatus.HISTORICAL
    assert m2 is not None and m2.value_norm == "3.10" and m2.status == MemoryStatus.HISTORICAL
    assert m3 is not None and m3.value_norm == "3.12" and m3.status == MemoryStatus.ACTIVE

    report = service.check_invariants()
    assert report.is_valid is True


def test_backfill_with_explicit_valid_to(service: MemoryService) -> None:
    """Historical claim with explicit valid_to creates a bounded interval leaving a gap."""
    t_next = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    t_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t_end = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    t_ingest = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)

    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="v3.12", event_time=t_next),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t_next,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_ingest,
    )
    rep = service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="v3.10", event_time=t_start),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t_start,
                valid_to=t_end,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_ingest,
    )
    assert rep.decisions[0].rule_id == "R-BACKFILL"
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    mem = service.repo.get_memory(m_id)
    assert mem is not None
    assert mem.valid_from == t_start
    assert mem.valid_to == t_end

    # Gap at April 1 returns None
    m_gap = service.get_as_of(
        "Server A", "python_version", valid_at=datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    )
    assert m_gap is None


def test_backfill_weak_claim_rejected(service: MemoryService) -> None:
    """A low-trust backfill claim is rejected via the R-BACKFILL trust gate."""
    t_newer = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    t_older = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t_ingest = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)

    # Newer fact from authoritative admin (confidence 0.9025)
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="v3.12", event_time=t_newer),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t_newer,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_ingest,
    )

    # Weak claim from intern (strength 0.30 * 0.40 = 0.12 < 0.9025 * 0.5)
    msg_weak = IngestMessage(source_id="intern_chat", text="v2.7 maybe", event_time=t_older)
    claim_weak = Claim(
        subject="Server A",
        attribute="python_version",
        value="2.7",
        valid_from=t_older,
        extract_confidence=0.40,
    )
    rep_weak = service.ingest_claims(msg_weak, [claim_weak], recorded_at=t_ingest)
    assert rep_weak.decisions[0].rule_id == "R-REJECT"
    assert rep_weak.decisions[0].action == Action.REJECT

    chain = service.repo.get_memory_chain(1, "python_version", include_rejected=False)
    assert len(chain) == 1
    assert chain[0].value_norm == "3.12"


def test_backfill_invariants_pass(service: MemoryService) -> None:
    """Running check_invariants() passes after complex interleaved backfills."""
    t_today = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)
    t1 = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="New", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t2,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Old", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )

    report = service.check_invariants()
    assert report.is_valid is True


def test_backfill_transaction_rollback_on_failure(service: MemoryService) -> None:
    """A batch error during backfill rolls back all changes atomically."""
    t_today = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)
    t_newer = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    t_older = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="v3.12", event_time=t_newer),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t_newer,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )

    claims = [
        Claim(
            subject="Server A",
            attribute="python_version",
            value="3.10",
            valid_from=t_older,
            extract_confidence=0.95,
        ),
        Claim(
            subject="Server A",
            attribute="nonexistent_attribute",
            value="fail",
            valid_from=t_older,
            extract_confidence=0.95,
        ),
    ]

    with pytest.raises(ValueError, match="Unknown attribute 'nonexistent_attribute'"):
        service.ingest_claims(
            IngestMessage(source_id="sysadmin_priya", text="Batch", event_time=t_older),
            claims,
            recorded_at=t_today,
        )

    chain = service.repo.get_memory_chain(1, "python_version")
    assert len(chain) == 1
    assert chain[0].value_norm == "3.12"


def test_backfill_trust_score_significantly_later_memory(service: MemoryService) -> None:
    """Later memory evaluated at inception baseline avoids backwards negative decay anomalies."""
    t_today = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)
    t_distant_future = datetime(2028, 1, 1, 0, 0, tzinfo=UTC)
    t_old = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya", text="Future baseline", event_time=t_distant_future
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.14",
                valid_from=t_distant_future,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )

    # 1. Low-trust intern backfill claim is rejected
    msg_intern = IngestMessage(
        source_id="intern_chat", text="Ancient intern claim", event_time=t_old
    )
    claim_intern = Claim(
        subject="Server A",
        attribute="python_version",
        value="2.7",
        valid_from=t_old,
        extract_confidence=0.40,
    )
    rep_weak = service.ingest_claims(msg_intern, [claim_intern], recorded_at=t_today)
    assert rep_weak.decisions[0].rule_id == "R-REJECT"
    assert rep_weak.decisions[0].action == Action.REJECT

    # 2. Authoritative admin backfill claim is accepted
    msg_admin = IngestMessage(
        source_id="sysadmin_priya", text="Ancient admin claim", event_time=t_old
    )
    claim_admin = Claim(
        subject="Server A",
        attribute="python_version",
        value="2.7",
        valid_from=t_old,
        extract_confidence=0.95,
    )
    rep_strong = service.ingest_claims(msg_admin, [claim_admin], recorded_at=t_today)
    assert rep_strong.decisions[0].rule_id == "R-BACKFILL"
    assert rep_strong.decisions[0].action == Action.BACKFILL

    report = service.check_invariants()
    assert report.is_valid is True


def test_backfill_chain_ordering_nearest_subsequent_boundary(service: MemoryService) -> None:
    """Backfill chooses the nearest subsequent boundary when multiple later memories exist."""
    t_today = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    t3 = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)

    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M1", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M2", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.11",
                valid_from=t2,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M3", event_time=t3),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t3,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )

    t_early = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    rep_early = service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Early fact", event_time=t_early),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.8",
                valid_from=t_early,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    assert rep_early.decisions[0].rule_id == "R-BACKFILL"
    m_early_id = rep_early.decisions[0].memory_id
    assert m_early_id is not None

    mem_early = service.repo.get_memory(m_early_id)
    assert mem_early is not None
    assert mem_early.valid_from == t_early
    # Selects nearest subsequent boundary (t1), not t2 or t3
    assert mem_early.valid_to == t1

    report = service.check_invariants()
    assert report.is_valid is True


def test_backfill_with_no_later_memory(service: MemoryService) -> None:
    """A claim with an earlier valid_from but no later memory in chain produces R-INSERT."""
    t_today = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)
    t_past = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    rep = service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Historical start", event_time=t_past),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.8",
                valid_from=t_past,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    assert rep.decisions[0].rule_id == "R-INSERT"
    assert rep.decisions[0].action == Action.INSERT

    report = service.check_invariants()
    assert report.is_valid is True


def test_backfill_explicit_valid_to_later_than_next_memory(service: MemoryService) -> None:
    """Explicit valid_to extending past next memory is capped by later.valid_from."""
    t_today = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)
    t_later = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    t_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t_overshooting_end = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)

    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Later fact", event_time=t_later),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t_later,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )

    rep = service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Overshooting claim", event_time=t_start),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t_start,
                valid_to=t_overshooting_end,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    assert rep.decisions[0].rule_id == "R-BACKFILL"
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    mem = service.repo.get_memory(m_id)
    assert mem is not None
    assert mem.valid_to == t_later

    report = service.check_invariants()
    assert report.is_valid is True


def test_backfill_into_already_occupied_historical_window(service: MemoryService) -> None:
    """Claim inside an already-closed historical window does not rewrite history."""
    t_today = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)
    t1 = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2025, 6, 1, 0, 0, tzinfo=UTC)
    t3 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

    # 1. Establish M1: [2025-01-01, 2025-06-01) HISTORICAL (value="3.8")
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M1", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.8",
                valid_from=t1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    # M2: [2025-06-01, None) -> closes M1 at 2025-06-01
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M2", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t2,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    # M3: [2026-01-01, None) -> closes M2 at 2026-01-01
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M3", event_time=t3),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t3,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )

    # Claim inside M1's historical window [2025-01-01, 2025-06-01) with conflicting value
    t_inside = datetime(2025, 3, 1, 0, 0, tzinfo=UTC)
    rep_conflict = service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya", text="Contradicting history", event_time=t_inside
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.9",
                valid_from=t_inside,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    # Reaches branch C: Conflict with history only -> R-DISPUTE
    # Does NOT rewrite history automatically
    assert rep_conflict.decisions[0].rule_id == "R-DISPUTE"
    assert rep_conflict.decisions[0].action == Action.DISPUTE

    # The new disputed memory is bounded by the historical window's upper limit (t2)
    m_dispute_id = rep_conflict.decisions[0].memory_id
    assert m_dispute_id is not None
    m_dispute = service.repo.get_memory(m_dispute_id)
    assert m_dispute is not None
    assert m_dispute.valid_from == t_inside
    assert m_dispute.valid_to == t2
    assert m_dispute.status == MemoryStatus.DISPUTED

    # Existing M1 is placed in dispute but retains its original valid_from/valid_to window
    chain = service.repo.get_memory_chain(1, "python_version", include_rejected=True)
    m1_updated = next(m for m in chain if m.value_norm == "3.8")
    assert m1_updated.status == MemoryStatus.DISPUTED
    assert m1_updated.valid_from == t1
    assert m1_updated.valid_to == t2

    # M2 and M3 remain untouched
    m2_curr = next(m for m in chain if m.value_norm == "3.10")
    assert m2_curr.status == MemoryStatus.HISTORICAL
    assert m2_curr.valid_from == t2
    assert m2_curr.valid_to == t3

    m3_curr = next(m for m in chain if m.value_norm == "3.12")
    assert m3_curr.status == MemoryStatus.ACTIVE
    assert m3_curr.valid_from == t3
    assert m3_curr.valid_to is None

    report = service.check_invariants()
    assert report.is_valid is True


def test_backfill_into_genuine_gap_between_historical_intervals(service: MemoryService) -> None:
    """Backfilling into a gap between closed historical intervals fills only that gap."""
    t_today = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)
    t1_start = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    t1_end = datetime(2025, 3, 1, 0, 0, tzinfo=UTC)
    t2_start = datetime(2025, 7, 1, 0, 0, tzinfo=UTC)
    t2_end = datetime(2025, 9, 1, 0, 0, tzinfo=UTC)
    t_active = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

    # M1 with explicit valid_to
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M1", event_time=t1_start),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.8",
                valid_from=t1_start,
                valid_to=t1_end,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    # M2 with explicit valid_to
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M2", event_time=t2_start),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t2_start,
                valid_to=t2_end,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    # M3 active
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M3", event_time=t_active),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t_active,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )

    # Backfill into gap [2025-03-01, 2025-07-01)
    t_gap = datetime(2025, 4, 1, 0, 0, tzinfo=UTC)
    rep_gap = service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Fill gap", event_time=t_gap),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.9",
                valid_from=t_gap,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    assert rep_gap.decisions[0].rule_id == "R-BACKFILL"
    m_id = rep_gap.decisions[0].memory_id
    assert m_id is not None

    mem = service.repo.get_memory(m_id)
    assert mem is not None
    assert mem.valid_from == t_gap
    # Fills gap up to start of next memory (M2 at t2_start)
    assert mem.valid_to == t2_start
    assert mem.status == MemoryStatus.HISTORICAL

    report = service.check_invariants()
    assert report.is_valid is True


def test_backfill_identical_claim_inside_existing_historical_interval(
    service: MemoryService,
) -> None:
    """An identical claim falling inside an existing historical interval corroborates."""
    t_today = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)
    t1 = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2025, 6, 1, 0, 0, tzinfo=UTC)

    # 1. Establish M1: [2025-01-01, 2025-06-01) with "3.8"
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M1", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.8",
                valid_from=t1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    # 2. Establish M2: [2025-06-01, None) with "3.10" -> closes M1 at 2025-06-01
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M2", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t2,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )

    # 3. Ingest identical claim at 2025-03-01 with value "3.8" from another source (ci_monitor)
    t_mid = datetime(2025, 3, 1, 0, 0, tzinfo=UTC)
    rep_corrob = service.ingest_claims(
        IngestMessage(source_id="ci_monitor", text="Corroborating past fact", event_time=t_mid),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.8",
                valid_from=t_mid,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    assert rep_corrob.decisions[0].rule_id == "R-CORROBORATE"
    assert rep_corrob.decisions[0].action == Action.CORROBORATE

    # No new memory inserted
    chain = service.repo.get_memory_chain(1, "python_version", include_rejected=True)
    assert len(chain) == 2

    # M1 remains HISTORICAL
    m1 = next(m for m in chain if m.value_norm == "3.8")
    assert m1.status == MemoryStatus.HISTORICAL
    assert m1.valid_from == t1
    assert m1.valid_to == t2

    report = service.check_invariants()
    assert report.is_valid is True


def test_backfill_weak_conflicting_claim_inside_existing_historical_interval(
    service: MemoryService,
) -> None:
    """A weak conflicting claim inside a historical interval is rejected with R-REJECT."""
    t_today = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)
    t1 = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2025, 6, 1, 0, 0, tzinfo=UTC)

    # 1. Establish M1: [2025-01-01, 2025-06-01) with "3.8"
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M1", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.8",
                valid_from=t1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    # 2. Establish M2: [2025-06-01, None) with "3.10"
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M2", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t2,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )

    # 3. Weak conflicting claim inside M1's window
    t_mid = datetime(2025, 3, 1, 0, 0, tzinfo=UTC)
    msg_weak = IngestMessage(source_id="intern_chat", text="I think it was 2.7", event_time=t_mid)
    claim_weak = Claim(
        subject="Server A",
        attribute="python_version",
        value="2.7",
        valid_from=t_mid,
        extract_confidence=0.40,
    )
    rep_weak = service.ingest_claims(msg_weak, [claim_weak], recorded_at=t_today)
    assert rep_weak.decisions[0].rule_id == "R-REJECT"
    assert rep_weak.decisions[0].action == Action.REJECT

    # M1 remains HISTORICAL and untouched
    chain = service.repo.get_memory_chain(1, "python_version", include_rejected=False)
    assert len(chain) == 2
    m1 = next(m for m in chain if m.value_norm == "3.8")
    assert m1.status == MemoryStatus.HISTORICAL

    report = service.check_invariants()
    assert report.is_valid is True


def test_context_equally_specific_incomparable_matching_contexts(
    service: MemoryService,
) -> None:
    """Equally specific incomparable contexts are flagged as ambiguous for single-valued slots."""
    t = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)

    # Ingest for project=A
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Project A runtime", event_time=t),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t,
                context={"project": "A"},
                extract_confidence=0.95,
            )
        ],
        recorded_at=t,
    )
    # Ingest for environment=staging
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Staging runtime", event_time=t),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.11",
                valid_from=t,
                context={"environment": "staging"},
                extract_confidence=0.95,
            )
        ],
        recorded_at=t,
    )

    # Query with combined context: {"project": "A", "environment": "staging"}
    query_ctx = {"project": "A", "environment": "staging"}
    res = service.current("Server A", "python_version", context=query_ctx, now=t)

    assert res.is_ambiguous is True
    assert res.ambiguity_reason is not None
    assert "Ambiguous context match" in res.ambiguity_reason
    assert len(res.items) == 2

    # Both items must be labelled inherited_partial
    assert all(it.scope_match == "inherited_partial" for it in res.items)

    # get_as_of returns None rather than silently choosing a candidate
    val = service.get_as_of("Server A", "python_version", valid_at=t, context=query_ctx)
    assert val is None


def test_context_query_one_exact_match(service: MemoryService) -> None:
    """An exact context match is labelled exact and is unambiguous."""
    t = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    ctx = {"project": "A", "environment": "staging"}

    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Exact scoped claim", event_time=t),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t,
                context=ctx,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t,
    )

    res = service.current("Server A", "python_version", context=ctx, now=t)
    assert res.is_ambiguous is False
    assert len(res.items) == 1
    assert res.items[0].scope_match == "exact"
    assert res.items[0].memory.value_norm == "3.12"

    mem = service.get_as_of("Server A", "python_version", valid_at=t, context=ctx)
    assert mem is not None
    assert mem.value_norm == "3.12"


def test_context_query_partial_inherited_match(service: MemoryService) -> None:
    """A partial inherited scope is labelled inherited_partial, not inherited_global."""
    t = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)

    # Global memory (specificity 0)
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Global fact", event_time=t),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t,
    )
    # Staging memory (specificity 1)
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Staging fact", event_time=t),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.11",
                valid_from=t,
                context={"environment": "staging"},
                extract_confidence=0.95,
            )
        ],
        recorded_at=t,
    )

    # Query with context {"project": "A", "environment": "staging"}
    query_ctx = {"project": "A", "environment": "staging"}
    res = service.current("Server A", "python_version", context=query_ctx, now=t)
    assert res.is_ambiguous is False
    assert len(res.items) == 1
    assert res.items[0].memory.value_norm == "3.11"
    # Crucial: labelled inherited_partial, NOT inherited_global
    assert res.items[0].scope_match == "inherited_partial"


def test_context_query_only_global_fallback(service: MemoryService) -> None:
    """A query falling back to a genuinely global memory is labelled inherited_global."""
    t = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)

    # Global memory only
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Global fact", event_time=t),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t,
    )

    query_ctx = {"project": "A", "environment": "staging"}
    res = service.current("Server A", "python_version", context=query_ctx, now=t)
    assert res.is_ambiguous is False
    assert len(res.items) == 1
    assert res.items[0].memory.value_norm == "3.10"
    # Crucial: labelled inherited_global
    assert res.items[0].scope_match == "inherited_global"


def test_context_single_valued_conflicting_matches(service: MemoryService) -> None:
    """Single-valued conflicting matches across incomparable scopes flag ambiguity."""
    t = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)

    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M1", event_time=t),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t,
                context={"project": "Alpha"},
                extract_confidence=0.95,
            )
        ],
        recorded_at=t,
    )
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M2", event_time=t),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t,
                context={"cluster": "us-east"},
                extract_confidence=0.95,
            )
        ],
        recorded_at=t,
    )

    query_ctx = {"project": "Alpha", "cluster": "us-east"}
    res = service.current("Server A", "python_version", context=query_ctx, now=t)
    assert res.is_ambiguous is True
    assert len(res.items) == 2
    vals = {it.memory.value_norm for it in res.items}
    assert vals == {"3.10", "3.12"}
    assert service.get_as_of("Server A", "python_version", valid_at=t, context=query_ctx) is None


def test_context_multi_valued_matching_memories(service: MemoryService) -> None:
    """Multi-valued attributes allow coexisting items across scopes without ambiguity."""
    t = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)

    # runs_service is multi-valued
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Service A", event_time=t),
        [
            Claim(
                subject="Server A",
                attribute="runs_service",
                value="api_gateway",
                valid_from=t,
                context={"project": "Alpha"},
                extract_confidence=0.95,
            )
        ],
        recorded_at=t,
    )
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Service B", event_time=t),
        [
            Claim(
                subject="Server A",
                attribute="runs_service",
                value="auth_service",
                valid_from=t,
                context={"environment": "staging"},
                extract_confidence=0.95,
            )
        ],
        recorded_at=t,
    )

    query_ctx = {"project": "Alpha", "environment": "staging"}
    res = service.current("Server A", "runs_service", context=query_ctx, now=t)
    # Multi-valued coexistence is expected, not ambiguous
    assert res.is_ambiguous is False
    assert len(res.items) == 2
    vals = {it.memory.value_norm for it in res.items}
    assert vals == {"api_gateway", "auth_service"}


def test_context_key_collision_regression_delimiters() -> None:
    """Delimiter characters (semicolons, equals signs) in context keys/values do not collide."""
    ctx_a = {"a": "b;c=d"}
    ctx_b = {"a": "b", "c": "d"}
    ctx_c = {"a=b": "c"}
    ctx_d = {"a": "b=c"}

    key_a = build_context_key(ctx_a)
    key_b = build_context_key(ctx_b)
    key_c = build_context_key(ctx_c)
    key_d = build_context_key(ctx_d)

    keys = {key_a, key_b, key_c, key_d}
    # All four must be strictly distinct
    assert len(keys) == 4


def test_context_key_normalization_and_sorting() -> None:
    """Whitespace is stripped, keys sorted, and identical contexts produce identical keys."""
    ctx1 = {"  z  ": " 1 ", " a ": " 2 "}
    ctx2 = {"a": "2", "z": "1"}
    assert normalize_context(ctx1) == {"a": "2", "z": "1"}
    assert build_context_key(ctx1) == '{"a":"2","z":"1"}'
    assert build_context_key(ctx1) == build_context_key(ctx2)


def test_context_key_duplicate_keys_rejected() -> None:
    """Keys that collide after whitespace normalization are rejected with ValueError."""
    ctx = {"env ": "staging", "env": "production"}
    with pytest.raises(ValueError, match="Duplicate context key"):
        build_context_key(ctx)


def test_context_key_empty_keys_and_values_rejected() -> None:
    """Empty or blank keys and values are rejected; empty context produces empty string."""
    with pytest.raises(ValueError, match="Context key cannot be empty"):
        build_context_key({"": "value"})

    with pytest.raises(ValueError, match="Context key cannot be empty"):
        build_context_key({"   ": "value"})

    with pytest.raises(ValueError, match="Context value for key 'env' cannot be empty"):
        build_context_key({"env": ""})

    with pytest.raises(ValueError, match="Context value for key 'env' cannot be empty"):
        build_context_key({"env": "   "})

    # Global context
    assert build_context_key({}) == ""
    assert build_context_key(None) == ""


def test_context_case_sensitivity() -> None:
    """Context keys and values are strictly case-sensitive."""
    # Differing casing in keys are distinct and non-colliding
    ctx_case = {"env": "prod", "Env": "staging"}
    norm = normalize_context(ctx_case)
    assert norm == {"env": "prod", "Env": "staging"}
    key = build_context_key(ctx_case)
    assert key == '{"Env":"staging","env":"prod"}'

    # Differing casing in values produces distinct context keys
    key_lower = build_context_key({"env": "prod"})
    key_upper = build_context_key({"env": "PROD"})
    assert key_lower != key_upper
    assert key_lower == '{"env":"prod"}'
    assert key_upper == '{"env":"PROD"}'


def test_legacy_context_key_canonicalization_safety() -> None:
    """_canonicalize_context_key safely parses valid legacy strings and rejects invalid ones."""
    from memorymesh.repository import _canonicalize_context_key

    # None and empty
    assert _canonicalize_context_key(None) is None
    assert _canonicalize_context_key("") == ""

    # Already canonical JSON
    assert _canonicalize_context_key('{"env":"prod"}') == '{"env":"prod"}'

    # Valid legacy strings
    assert _canonicalize_context_key("env=staging") == '{"env":"staging"}'
    assert _canonicalize_context_key("env=staging;region=us-east") == (
        '{"env":"staging","region":"us-east"}'
    )
    assert _canonicalize_context_key("region=us-east;env=staging") == (
        '{"env":"staging","region":"us-east"}'
    )

    # Duplicate legacy keys fail safely
    with pytest.raises(ValueError, match="Duplicate legacy context key"):
        _canonicalize_context_key("env=staging;env=prod")

    # Malformed legacy token without '='
    with pytest.raises(ValueError, match="missing '=' delimiter"):
        _canonicalize_context_key("env=staging;malformed_token")

    # Empty key or value
    with pytest.raises(ValueError, match="key and value must be non-empty"):
        _canonicalize_context_key("=staging")

    with pytest.raises(ValueError, match="key and value must be non-empty"):
        _canonicalize_context_key("env=")

    # Empty tokens must be rejected, not discarded
    with pytest.raises(ValueError, match="Empty token in legacy context key"):
        _canonicalize_context_key("env=prod;;region=us")

    with pytest.raises(ValueError, match="Empty token in legacy context key"):
        _canonicalize_context_key("env=prod;")

    # Malformed canonical JSON
    with pytest.raises(ValueError, match="Malformed canonical JSON"):
        _canonicalize_context_key("{")

    # Duplicate keys in JSON (differing values)
    with pytest.raises(ValueError, match="Duplicate context key in JSON: 'environment'"):
        _canonicalize_context_key('{"environment":"production","environment":"staging"}')

    # Duplicate keys in JSON (identical values)
    with pytest.raises(ValueError, match="Duplicate context key in JSON: 'env'"):
        _canonicalize_context_key('{"env":"prod","env":"prod"}')

    # Duplicate keys in JSON with varying whitespace
    with pytest.raises(ValueError, match="Duplicate context key in JSON: 'env'"):
        _canonicalize_context_key('{"env":"prod"," env ":"staging"}')

    # Empty key in JSON
    with pytest.raises(ValueError, match="Context key cannot be empty"):
        _canonicalize_context_key('{"":"value"}')

    # Empty or whitespace value in JSON
    with pytest.raises(ValueError, match="Context value for key 'env' cannot be empty"):
        _canonicalize_context_key('{"env":""}')

    with pytest.raises(ValueError, match="Context value for key 'env' cannot be empty"):
        _canonicalize_context_key('{"env":"   "}')

    # Non-string value in JSON
    with pytest.raises(ValueError, match="must be strings"):
        _canonicalize_context_key('{"env":123}')

    # Legacy strings without a valid key=value structure
    with pytest.raises(ValueError, match="Invalid context key 'malformed'"):
        _canonicalize_context_key("malformed")


def test_historical_identical_vs_refines_difference(service: MemoryService) -> None:
    """Demonstrates that IDENTICAL corroborates evidence while REFINES produces R-REFINE.

    Adjusts the coarse memory window [t1, t_refine) and establishes [t_refine, t2) for
    the refined memory, eliminating competing values and preserving temporal invariants.
    """
    t_today = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)
    t1 = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2025, 6, 1, 0, 0, tzinfo=UTC)

    # 1. Ingest original historical memory: M1 [2025-01-01, 2025-06-01) with '3.10'
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M1", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    # Ingest newer active memory: M2 [2025-06-01, None) with '3.12' -> closes M1 at 2025-06-01
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M2", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t2,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )

    # Ingest IDENTICAL claim: corroborates M1 without creating a new memory
    t_corrob = datetime(2025, 2, 1, 0, 0, tzinfo=UTC)
    rep_ident = service.ingest_claims(
        IngestMessage(source_id="ci_monitor", text="Identical fact", event_time=t_corrob),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t_corrob,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    assert rep_ident.decisions[0].rule_id == "R-CORROBORATE"
    assert rep_ident.decisions[0].action == Action.CORROBORATE
    chain_after_ident = service.repo.get_memory_chain(1, "python_version")
    assert len(chain_after_ident) == 2  # No new memory created

    # 2. Ingest more specific historical refinement: M3 [2025-03-01, 2025-06-01) with '3.10.4'
    t_refine = datetime(2025, 3, 1, 0, 0, tzinfo=UTC)
    rep_refine = service.ingest_claims(
        IngestMessage(source_id="ci_monitor", text="Refined fact", event_time=t_refine),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10.4",
                valid_from=t_refine,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    assert rep_refine.decisions[0].rule_id == "R-REFINE"
    assert rep_refine.decisions[0].action == Action.REFINE
    new_mem_id = rep_refine.decisions[0].memory_id
    assert new_mem_id is not None

    refined_mem = service.repo.get_memory(new_mem_id)
    assert refined_mem is not None
    assert refined_mem.value_norm == "3.10.4"
    assert refined_mem.status == MemoryStatus.HISTORICAL
    assert refined_mem.valid_from == t_refine
    assert refined_mem.valid_to == t2  # Bounded by original historical window

    # Verify M1 was closed at t_refine, eliminating overlap between M1 and refined_mem
    m1 = service.repo.get_memory(1)
    assert m1 is not None
    assert m1.status == MemoryStatus.HISTORICAL
    assert m1.valid_from == t1
    assert m1.valid_to == t_refine  # Truncated to period before refinement

    # 3. Query before refinement's valid_from (2025-02-01)
    t_before = datetime(2025, 2, 1, 0, 0, tzinfo=UTC)
    res_before = service.current("Server A", "python_version", valid_at=t_before, now=t_today)
    assert len(res_before.items) == 1
    assert res_before.items[0].memory.value_norm == "3.10"
    assert res_before.is_ambiguous is False
    as_of_before = service.get_as_of("Server A", "python_version", valid_at=t_before)
    assert as_of_before is not None
    assert as_of_before.value_norm == "3.10"

    # 4. Query after refinement's valid_from (2025-04-01)
    t_after = datetime(2025, 4, 1, 0, 0, tzinfo=UTC)
    res_after = service.current("Server A", "python_version", valid_at=t_after, now=t_today)
    assert len(res_after.items) == 1
    assert res_after.items[0].memory.value_norm == "3.10.4"
    assert res_after.is_ambiguous is False
    as_of_after = service.get_as_of("Server A", "python_version", valid_at=t_after)
    assert as_of_after is not None
    assert as_of_after.value_norm == "3.10.4"

    # 5. Confirm no competing values: neither query reports multiple or ambiguous items
    assert len(res_before.items) == 1
    assert len(res_after.items) == 1

    # 6. Confirm historical provenance remains explainable
    exp_refined = service.explain(new_mem_id)
    assert any(e.type == EdgeType.REFINES and e.dst_memory_id == m1.id for e in exp_refined.edges)
    exp_m1 = service.explain(m1.id)
    assert exp_m1.items[0].memory.value_norm == "3.10"
    assert any(e.type == EdgeType.REFINES and e.src_memory_id == new_mem_id for e in exp_m1.edges)

    # 7. Run service.check_invariants() successfully
    report = service.check_invariants()
    assert report.is_valid is True


def test_dispute_clamping_ordinary_active_vs_historical(service: MemoryService) -> None:
    """Ordinary disputes preserve open-ended windows; historical disputes are clamped."""
    t_today = datetime(2026, 6, 25, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t1_end = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
    t_conflict = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)

    # Ingest ACTIVE memory with finite valid_to [2026-01-01, 2026-08-01)
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Active with window", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="deploy_day",
                value="friday",
                valid_from=t1,
                valid_to=t1_end,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    # Ordinary conflicting claim with open-ended valid_to=None
    rep_ord = service.ingest_claims(
        IngestMessage(source_id="dev_chat", text="Conflicting deploy day", event_time=t_conflict),
        [
            Claim(
                subject="Server A",
                attribute="deploy_day",
                value="monday",
                valid_from=t_conflict,
                valid_to=None,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_today,
    )
    assert rep_ord.decisions[0].rule_id == "R-DISPUTE"
    m_dispute_id = rep_ord.decisions[0].memory_id
    assert m_dispute_id is not None
    m_dispute = service.repo.get_memory(m_dispute_id)
    assert m_dispute is not None
    # Crucial: ordinary dispute does NOT clamp to candidate's 2026-08-01
    assert m_dispute.valid_to is None
    assert m_dispute.status == MemoryStatus.DISPUTED

    report = service.check_invariants()
    assert report.is_valid is True


def test_partial_historical_refine_three_way_split_and_bitemporal_reconstruction(
    service: MemoryService,
) -> None:
    """Partial historical refinement splits into before, refined, and continuation segments.

    Deterministic scenario:
    - Target: Python 3.10 valid [2026-01-01, 2026-06-01), recorded at 2026-01-15 (T1).
    - Active later fact: Python 3.12 valid [2026-06-01, +inf), recorded at 2026-06-01.
    - Partial refinement: Python 3.10.4 valid [2026-03-01, 2026-04-01), recorded at 2026-07-01 (T2).

    Verifies:
    1. Three-way interval partition:
       - M_orig: [2026-01-01, 2026-03-01) with '3.10'
       - M_refined: [2026-03-01, 2026-04-01) with '3.10.4'
       - M_cont: [2026-04-01, 2026-06-01) with '3.10'
    2. Zero validity gap across [2026-01-01, 2026-06-01).
    3. Half-open intervals: single-valued slot never returns competing memories.
    4. Query coverage:
       - Before refinement: valid_at=2026-02-15 -> '3.10'
       - Refinement start boundary: valid_at=2026-03-01 -> '3.10.4'
       - Inside refinement: valid_at=2026-03-15 -> '3.10.4'
       - Refinement end boundary / continuation start: valid_at=2026-04-01 -> '3.10'
       - Inside continuation: valid_at=2026-05-15 -> '3.10'
       - Original interval end boundary: valid_at=2026-06-01 -> '3.12'
       - After original interval: valid_at=2026-07-01 -> '3.12'
    5. Bitemporal belief reconstruction:
       - Belief time T_belief_1 (2026-02-01, prior to refinement recorded at T2):
         A query for valid_at=2026-03-15 as of T_belief_1 reconstructs the earlier belief ('3.10').
       - Belief time T_belief_2 (2026-07-15, after refinement recorded at T2):
         A query for valid_at=2026-03-15 as of T_belief_2 returns refined belief ('3.10.4').
       - Provenance in decision_log and edges:
         - DecisionLogEntry for refinement links to refined memory and has action REFINE.
         - Edges: M_refined REFINES M_orig; M_cont SUPERSEDES M_orig.
    6. System invariant check: service.check_invariants() is 100% valid.
    """
    t_jan1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t_mar1 = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    t_apr1 = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    t_jun1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)

    t_rec1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t_rec_later = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    t_rec2 = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)

    # 1. Ingest original fact: Python 3.10 starting Jan 1 (active until June)
    rep1 = service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs Python 3.10",
            event_time=t_jan1,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t_jan1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_rec1,
    )
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None
    m1 = service.repo.get_memory(m1_id)
    assert m1 is not None
    assert m1.value_norm == "3.10"
    assert m1.valid_from == t_jan1
    assert m1.status == MemoryStatus.ACTIVE

    # 2. Ingest newer fact: Python 3.12 starting Jun 1 -> supersedes M1
    # into HISTORICAL [Jan 1, Jun 1)
    rep_later = service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya",
            text="Server A upgraded to Python 3.12 in June",
            event_time=t_jun1,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t_jun1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_rec_later,
    )
    assert rep_later.decisions[0].rule_id == "R-REPLACE"
    m1_hist = service.repo.get_memory(m1_id)
    assert m1_hist is not None
    assert m1_hist.status == MemoryStatus.HISTORICAL
    assert m1_hist.valid_to == t_jun1

    # 3. Ingest partial historical refinement: Python 3.10.4 for [Mar 1, Apr 1) at t_rec2
    rep_refine = service.ingest_claims(
        IngestMessage(
            source_id="ci_monitor",
            text="CI confirms Python 3.10.4 was used from March to April",
            event_time=t_mar1,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10.4",
                valid_from=t_mar1,
                valid_to=t_apr1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_rec2,
    )
    decision = rep_refine.decisions[0]
    assert decision.rule_id == "R-REFINE"
    assert decision.action == Action.REFINE
    m_refined_id = decision.memory_id
    assert m_refined_id is not None

    # Inspect the 3 segments:
    # Segment 1 (M1 updated): [Jan 1, Mar 1) with '3.10'
    m1_updated = service.repo.get_memory(m1_id)
    assert m1_updated is not None
    assert m1_updated.value_norm == "3.10"
    assert m1_updated.valid_from == t_jan1
    assert m1_updated.valid_to == t_mar1
    assert m1_updated.closed_at == t_rec2

    # Segment 2 (M_refined): [Mar 1, Apr 1) with '3.10.4'
    m_refined = service.repo.get_memory(m_refined_id)
    assert m_refined is not None
    assert m_refined.value_norm == "3.10.4"
    assert m_refined.valid_from == t_mar1
    assert m_refined.valid_to == t_apr1
    assert m_refined.status == MemoryStatus.HISTORICAL

    # Segment 3 (M_cont): [Apr 1, Jun 1) with '3.10'
    chain = service.repo.get_memory_chain(1, "python_version")
    continuation_mems = [
        m for m in chain if m.id not in (m1_id, m_refined_id) and m.value_norm == "3.10"
    ]
    assert len(continuation_mems) == 1
    m_cont = continuation_mems[0]
    assert m_cont.valid_from == t_apr1
    assert m_cont.valid_to == t_jun1
    assert m_cont.status == MemoryStatus.HISTORICAL
    assert m_cont.confidence == m1.confidence

    # 4. Query coverage & half-open interval checks across the timeline:
    t_query_before = datetime(2026, 2, 15, 0, 0, tzinfo=UTC)
    t_query_boundary_start = t_mar1
    t_query_inside = datetime(2026, 3, 15, 0, 0, tzinfo=UTC)
    t_query_boundary_mid = t_apr1
    t_query_continuation = datetime(2026, 5, 15, 0, 0, tzinfo=UTC)
    t_query_boundary_end = t_jun1
    t_query_after = datetime(2026, 7, 15, 0, 0, tzinfo=UTC)

    # (a) Query before refinement starts: returns '3.10'
    res = service.current("Server A", "python_version", valid_at=t_query_before, now=t_rec2)
    assert len(res.items) == 1
    assert res.items[0].memory.value_norm == "3.10"
    assert res.items[0].memory.id == m1_id

    # (b) Query at boundary valid_from (2026-03-01): returns '3.10.4'
    res = service.current("Server A", "python_version", valid_at=t_query_boundary_start, now=t_rec2)
    assert len(res.items) == 1
    assert res.items[0].memory.value_norm == "3.10.4"
    assert res.items[0].memory.id == m_refined_id

    # (c) Query during refinement (2026-03-15): returns '3.10.4'
    res = service.current("Server A", "python_version", valid_at=t_query_inside, now=t_rec2)
    assert len(res.items) == 1
    assert res.items[0].memory.value_norm == "3.10.4"
    assert res.items[0].memory.id == m_refined_id

    # (d) Query at refinement end boundary / continuation start (2026-04-01): returns '3.10'
    res = service.current("Server A", "python_version", valid_at=t_query_boundary_mid, now=t_rec2)
    assert len(res.items) == 1
    assert res.items[0].memory.value_norm == "3.10"
    assert res.items[0].memory.id == m_cont.id

    # (e) Query inside continuation segment (2026-05-15): returns '3.10'
    res = service.current("Server A", "python_version", valid_at=t_query_continuation, now=t_rec2)
    assert len(res.items) == 1
    assert res.items[0].memory.value_norm == "3.10"
    assert res.items[0].memory.id == m_cont.id

    # (f) Query at end of original interval (2026-06-01): transitions seamlessly to '3.12'
    res = service.current("Server A", "python_version", valid_at=t_query_boundary_end, now=t_rec2)
    assert len(res.items) == 1
    assert res.items[0].memory.value_norm == "3.12"

    # (g) Query after original interval (2026-07-15): returns '3.12'
    res = service.current("Server A", "python_version", valid_at=t_query_after, now=t_rec2)
    assert len(res.items) == 1
    assert res.items[0].memory.value_norm == "3.12"

    # 5. Bitemporal belief reconstruction:
    # What was believed on 2026-02-01 (before refinement at t_rec2) about valid_at=2026-03-15?
    t_belief_pre = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    as_of_pre = service.get_as_of(
        "Server A", "python_version", valid_at=t_query_inside, known_at=t_belief_pre
    )
    assert as_of_pre is not None
    assert as_of_pre.value_norm == "3.10"
    assert as_of_pre.id == m1_id

    # What was believed on 2026-07-15 (after refinement at t_rec2) about valid_at=2026-03-15?
    t_belief_post = datetime(2026, 7, 15, 0, 0, tzinfo=UTC)
    as_of_post = service.get_as_of(
        "Server A", "python_version", valid_at=t_query_inside, known_at=t_belief_post
    )
    assert as_of_post is not None
    assert as_of_post.value_norm == "3.10.4"
    assert as_of_post.id == m_refined_id

    # 6. Provenance & Edge verification:
    exp_refined = service.explain(m_refined_id)
    assert any(e.type == EdgeType.REFINES and e.dst_memory_id == m1_id for e in exp_refined.edges)
    exp_cont = service.explain(m_cont.id)
    assert any(e.type == EdgeType.SUPERSEDES and e.dst_memory_id == m1_id for e in exp_cont.edges)

    # Assert exact source and destination IDs from persisted edges in repository
    all_refined_edges = service.repo.get_edges_for_memory(m_refined_id, direction="src")
    refines_edges = [e for e in all_refined_edges if e.type == EdgeType.REFINES]
    assert len(refines_edges) == 1
    assert refines_edges[0].src_memory_id == m_refined_id
    assert refines_edges[0].dst_memory_id == m1_id

    all_cont_edges = service.repo.get_edges_for_memory(m_cont.id, direction="src")
    supersedes_edges = [e for e in all_cont_edges if e.type == EdgeType.SUPERSEDES]
    assert len(supersedes_edges) == 1
    assert supersedes_edges[0].src_memory_id == m_cont.id
    assert supersedes_edges[0].dst_memory_id == m1_id

    # 7. Check Invariants
    inv_report = service.check_invariants()
    assert inv_report.is_valid is True
    assert len(inv_report.violations) == 0


def test_continuation_relationship_exact_edge_endpoints(service: MemoryService) -> None:
    """Verify multiple inserted memories resolve exact relationship edge endpoints.

    Tests that:
    1. Refined memory -> original memory via REFINES (src=m_refined.id, dst=target.id).
    2. Continuation memory -> original memory via SUPERSEDES (src=m_cont.id, dst=target.id).
    3. No self-loops or swapped endpoint identities exist.
    """
    t1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    t_rec = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    t_ref_start = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    t_ref_end = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    t_ref_rec = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)

    # Ingest M1 (Python 3.10)
    rep1 = service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M1", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_rec,
    )
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # Replace M1 with M2 (Python 3.12 at June 1) -> M1 becomes HISTORICAL [Jan 1, Jun 1)
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="M2", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t2,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_rec,
    )

    # Ingest partial refinement [March 1, April 1) -> produces M_refined and M_cont
    rep_ref = service.ingest_claims(
        IngestMessage(source_id="ci_monitor", text="M_ref", event_time=t_ref_start),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10.4",
                valid_from=t_ref_start,
                valid_to=t_ref_end,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_ref_rec,
    )
    m_refined_id = rep_ref.decisions[0].memory_id
    assert m_refined_id is not None

    chain = service.repo.get_memory_chain(1, "python_version")
    cont_candidates = [
        m for m in chain if m.id not in (m1_id, m_refined_id) and m.value_norm == "3.10"
    ]
    assert len(cont_candidates) == 1
    m_cont_id = cont_candidates[0].id
    assert m_cont_id != m_refined_id
    assert m_cont_id != m1_id

    # Retrieve all edges directly from repository
    edges_from_refined = service.repo.get_edges_for_memory(m_refined_id, direction="src")
    edges_from_cont = service.repo.get_edges_for_memory(m_cont_id, direction="src")

    # Assert exact endpoint IDs for REFINES
    refines_edges = [e for e in edges_from_refined if e.type == EdgeType.REFINES]
    assert len(refines_edges) == 1
    assert refines_edges[0].src_memory_id == m_refined_id
    assert refines_edges[0].dst_memory_id == m1_id

    # Assert exact endpoint IDs for SUPERSEDES
    supersedes_edges = [e for e in edges_from_cont if e.type == EdgeType.SUPERSEDES]
    assert len(supersedes_edges) == 1
    assert supersedes_edges[0].src_memory_id == m_cont_id
    assert supersedes_edges[0].dst_memory_id == m1_id

    # Verify no edge endpoints are swapped
    assert not any(
        e.src_memory_id == m1_id and e.dst_memory_id == m_refined_id for e in edges_from_refined
    )
    assert not any(
        e.src_memory_id == m1_id and e.dst_memory_id == m_cont_id for e in edges_from_cont
    )


def test_continuation_preserves_multiple_original_evidences_and_confidence(
    service: MemoryService,
) -> None:
    """Historical continuation memory preserves all original evidence records and confidence.

    Scenario:
    - Initial fact M1 for Server A python_version='3.10' from sysadmin_priya at t1=2026-01-01
      (strength=0.855).
    - Corroborating fact from ci_monitor at t1_corrob=2026-01-10 (strength=0.700).
      M1 now has 2 distinct evidence records with different sources, strengths, and observed_at.
      Confidence is noisy-OR aggregated.
    - Upgraded to Python 3.12 at t2=2026-06-01, moving M1 to HISTORICAL [2026-01-01, 2026-06-01).
    - Partial historical refinement at t_refine_rec=2026-07-01:
      CI reports Python 3.10.4 used [2026-03-01, 2026-04-01) from dev_chat (strength=0.900).

    Verifies:
    1. Original memory M1 retains both original evidence records intact:
       - Record 1: sysadmin_priya, strength=0.855, observed_at=2026-01-01.
       - Record 2: ci_monitor, strength=0.700, observed_at=2026-01-10.
    2. Continuation memory preserves BOTH original evidence records:
       - No fabricated observation time (observed_at are 2026-01-01 and 2026-01-10, NOT 2026-07-01).
       - No conversion of aggregate confidence to single strength (strengths are 0.855 and 0.700).
       - Distinct database rows associated with continuation_mem.id.
    3. Continuation memory confidence is noisy-OR consistent:
       - confidence matches M1.confidence within tolerance.
    4. Refined memory retains only its own new evidence:
       - dev_chat, strength=0.900, observed_at=2026-07-01.
    5. Original memory and continuation memory are both fully explainable.
    6. System invariant audit: 100% valid.
    """
    t1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t1_corrob = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    t_refine_start = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    t_refine_end = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    t_refine_rec = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)

    # 1. Ingest initial fact: source sysadmin_priya, reliability 0.90 * extract 0.95 = 0.855
    rep1 = service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya",
            text="Initial provision: Server A runs Python 3.10",
            event_time=t1,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t1,
                extract_confidence=0.95,
                span="Server A runs Python 3.10",
            )
        ],
        recorded_at=t1,
    )
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # 2. Corroborate fact from second source: ci_monitor (reliability 0.85 * 0.90)
    rep_corrob = service.ingest_claims(
        IngestMessage(
            source_id="ci_monitor",
            text="CI job corroborates: Server A runs Python 3.10",
            event_time=t1_corrob,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t1_corrob,
                extract_confidence=0.90,
                span="Server A runs Python 3.10",
            )
        ],
        recorded_at=t1_corrob,
    )
    assert rep_corrob.decisions[0].rule_id == "R-CORROBORATE"

    # Verify M1 now has 2 distinct evidence records
    evs_m1 = service.repo.get_evidence_for_memory(m1_id)
    assert len(evs_m1) == 2
    sources_m1 = {e.source_id for e in evs_m1}
    assert sources_m1 == {"sysadmin_priya", "ci_monitor"}
    obs_m1 = {e.observed_at for e in evs_m1}
    assert obs_m1 == {t1, t1_corrob}
    m1_before_refine = service.repo.get_memory(m1_id)
    assert m1_before_refine is not None
    m1_conf_expected = m1_before_refine.confidence

    # 3. Supercede M1 with M2 at June 1 -> M1 is now HISTORICAL [Jan 1, Jun 1)
    service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya",
            text="Upgrade: Server A runs Python 3.12",
            event_time=t2,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t2,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t2,
    )

    # 4. Partial historical refinement at July 1 from dev_chat (third source)
    rep_refine = service.ingest_claims(
        IngestMessage(
            source_id="dev_chat",
            text="Dev chat confirms Python 3.10.4 in March",
            event_time=t_refine_start,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10.4",
                valid_from=t_refine_start,
                valid_to=t_refine_end,
                extract_confidence=0.90,
                span="Python 3.10.4 in March",
            )
        ],
        recorded_at=t_refine_rec,
    )
    assert rep_refine.decisions[0].rule_id == "R-REFINE"
    m_refined_id = rep_refine.decisions[0].memory_id
    assert m_refined_id is not None

    # Retrieve continuation memory
    chain = service.repo.get_memory_chain(1, "python_version")
    cont_mems = [m for m in chain if m.id not in (m1_id, m_refined_id) and m.value_norm == "3.10"]
    assert len(cont_mems) == 1
    m_cont = cont_mems[0]

    # Verify M1 still has its original 2 evidence records
    evs_m1_post = service.repo.get_evidence_for_memory(m1_id)
    assert len(evs_m1_post) == 2
    assert {e.source_id for e in evs_m1_post} == {"sysadmin_priya", "ci_monitor"}

    # Verify continuation memory has BOTH original evidence records cloned
    evs_cont = service.repo.get_evidence_for_memory(m_cont.id)
    assert len(evs_cont) == 2
    sources_cont = {e.source_id for e in evs_cont}
    assert sources_cont == {"sysadmin_priya", "ci_monitor"}

    # Verify timestamps and strengths are preserved, NOT fabricated at refinement time
    obs_cont = {e.observed_at for e in evs_cont}
    assert obs_cont == {t1, t1_corrob}
    assert t_refine_rec not in obs_cont  # NOT fabricated at refinement time!

    strengths_m1 = sorted([e.strength for e in evs_m1])
    strengths_cont = sorted([e.strength for e in evs_cont])
    assert pytest.approx(strengths_cont) == strengths_m1

    # Verify continuation confidence matches original confidence
    assert pytest.approx(m_cont.confidence, abs=1e-4) == m1_conf_expected

    # Verify refined memory has only its own evidence
    evs_refined = service.repo.get_evidence_for_memory(m_refined_id)
    assert len(evs_refined) == 1
    assert evs_refined[0].source_id == "dev_chat"
    assert evs_refined[0].observed_at == t_refine_rec

    # Verify explainability of all three memories
    exp_m1 = service.explain(m1_id)
    assert len(exp_m1.items[0].memory.evidence) == 2
    assert any("Supported by 2 evidence record(s)" in n for n in exp_m1.notes)

    exp_cont = service.explain(m_cont.id)
    assert len(exp_cont.items[0].memory.evidence) == 2
    assert any("Supported by 2 evidence record(s)" in n for n in exp_cont.notes)

    exp_ref = service.explain(m_refined_id)
    assert len(exp_ref.items[0].memory.evidence) == 1
    assert any("Supported by 1 evidence record(s)" in n for n in exp_ref.notes)

    # Verify invariants
    report = service.check_invariants()
    assert report.is_valid is True
    assert len(report.violations) == 0


def test_historical_refine_rejects_and_rolls_back_when_target_lacks_evidence(
    service: MemoryService,
) -> None:
    """Historical refinement on memory without evidence is rejected with transaction rollback.

    Scenario:
    - Insert a historical memory directly into DB with NO evidence records
      (unprovenanced/corrupted).
    - Attempt a partial historical refinement via service.ingest_claims().

    Verifies:
    1. Ingestion raises ConflictResolutionError explicitly.
    2. Atomic rollback:
       - Target memory in DB remains completely unchanged (valid_to, closed_at, status).
       - IngestMessage is rolled back.
       - No refined memory or continuation memory is inserted.
       - No edges or decision log entries are inserted.
    3. Invariant check:
       - No new invariant violations introduced.
       - Original database state preserved.
    """
    t_jan1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t_mar1 = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    t_apr1 = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    t_jun1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    t_rec = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)

    # Ingest M1 and supersede with M2 so M1 becomes a HISTORICAL memory
    rep1 = service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya",
            text="Initial: Server A runs Python 3.10",
            event_time=t_jan1,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t_jan1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_jan1,
    )
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    rep2 = service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya",
            text="Upgrade: Server A runs Python 3.12",
            event_time=t_jun1,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t_jun1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_jun1,
    )
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    # Simulate unprovenanced/corrupted historical memory by removing evidence records
    service.repo.conn.execute("DELETE FROM evidence WHERE memory_id = ?", (m1_id,))
    service.repo.conn.commit()

    # Confirm M1 has NO evidence records
    assert len(service.repo.get_evidence_for_memory(m1_id)) == 0

    m1_before = service.repo.get_memory(m1_id)
    assert m1_before is not None
    assert m1_before.status == MemoryStatus.HISTORICAL
    assert m1_before.valid_to == t_jun1

    # Attempt partial historical refinement
    msg = IngestMessage(
        source_id="ci_monitor",
        text="CI reports refined Python 3.10.4",
        event_time=t_mar1,
    )
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10.4",
        valid_from=t_mar1,
        valid_to=t_apr1,
        extract_confidence=0.95,
    )

    with pytest.raises(ConflictResolutionError, match="target has no evidence records"):
        service.ingest_claims(msg, [claim], recorded_at=t_rec)

    # Assert atomic transaction rollback:
    # 1. Target memory remains completely unchanged
    m1_after = service.repo.get_memory(m1_id)
    assert m1_after is not None
    assert m1_after.valid_to == t_jun1  # NOT truncated to t_mar1
    assert m1_after.closed_at == m1_before.closed_at
    assert m1_after.status == MemoryStatus.HISTORICAL

    # 2. No new memories inserted (only M1 and M2)
    chain = service.repo.get_memory_chain(m1_after.entity_id, "python_version")
    assert len(chain) == 2
    assert [m.id for m in chain] == [m1_id, m2_id]

    # 3. No new edges inserted (only the SUPERSEDES edge from M2 to M1 exists)
    edges = service.repo.get_edges_for_memory(m1_id)
    assert len(edges) == 1
    assert edges[0].src_memory_id == m2_id and edges[0].dst_memory_id == m1_id

    # 4. IngestMessage rolled back (ci_monitor's message was not persisted)
    msg_hash = hashlib.sha256(msg.text.encode("utf-8")).hexdigest()
    assert service.repo.get_message_by_hash(msg.source_id, msg_hash, msg.event_time) is None

    # 5. Normal multi-evidence continuation still working on another slot/entity
    t_alt_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t_alt_end = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Alt M1", event_time=t_alt_start),
        [
            Claim(
                subject="Server B",
                attribute="python_version",
                value="3.10",
                valid_from=t_alt_start,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_alt_start,
    )
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Alt M2", event_time=t_alt_end),
        [
            Claim(
                subject="Server B",
                attribute="python_version",
                value="3.12",
                valid_from=t_alt_end,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_alt_end,
    )
    rep_alt_ref = service.ingest_claims(
        IngestMessage(source_id="ci_monitor", text="Alt Refined", event_time=t_mar1),
        [
            Claim(
                subject="Server B",
                attribute="python_version",
                value="3.10.4",
                valid_from=t_mar1,
                valid_to=t_apr1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_rec,
    )
    assert rep_alt_ref.decisions[0].rule_id == "R-REFINE"
    chain_b = service.repo.get_memory_chain(2, "python_version")
    assert len(chain_b) == 4  # before, refined, continuation, active M2

    # Invariants audit:
    # Only the pre-existing corrupted M1 (whose evidence was deleted)
    # violates INV-CONFIDENCE-CONSISTENT; no temporal or relationship
    # invariants are violated and rollback was clean.
    report = service.check_invariants()
    assert len(report.violations) == 1
    assert report.violations[0].invariant_id == "INV-CONFIDENCE-CONSISTENT"
    assert report.violations[0].affected_ids == [m1_id]


def test_historical_dispute_temporal_invariants_and_metadata(
    service: MemoryService,
) -> None:
    """Historical memory entering dispute produces bounded valid_to and non-null closed_at.

    Verifies:
    1. Newly created disputed memory inherits historical upper bound as finite valid_to.
    2. closed_at is set to recorded_at (bitemporal known closure),
       satisfying INV-CLOSED-AT-REQUIRED.
    3. Existing historical memory status transitions to DISPUTED with valid_to/closed_at preserved.
    4. Invariant checker reports zero temporal or confidence violations.
    """
    t_jan1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t_mar1 = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    t_jun1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    t_recorded = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)

    # 1. Ingest initial claim M1
    rep1 = service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya",
            text="Initial: Server A runs Python 3.10",
            event_time=t_jan1,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t_jan1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_jan1,
    )
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # 2. Supersede M1 with M2 at June 1 -> M1 becomes HISTORICAL [Jan 1, Jun 1)
    rep2 = service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya",
            text="Upgrade: Server A runs Python 3.12",
            event_time=t_jun1,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t_jun1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_jun1,
    )
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    m1_hist = service.repo.get_memory(m1_id)
    assert m1_hist is not None
    assert m1_hist.status == MemoryStatus.HISTORICAL
    assert m1_hist.valid_from == t_jan1
    assert m1_hist.valid_to == t_jun1
    assert m1_hist.closed_at == t_jun1

    # 3. Conflicting claim arrives for historical window: Python 3.8 at March 1
    # Reliability 0.90 * extract 0.90 = 0.810 (above reject threshold)
    rep_dispute = service.ingest_claims(
        IngestMessage(
            source_id="ci_monitor",
            text="Audit log says Server A ran Python 3.8 in March",
            event_time=t_mar1,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.8",
                valid_from=t_mar1,
                extract_confidence=0.90,
            )
        ],
        recorded_at=t_recorded,
    )
    assert rep_dispute.decisions[0].rule_id == "R-DISPUTE"
    assert rep_dispute.decisions[0].action == Action.DISPUTE
    m_dispute_id = rep_dispute.decisions[0].memory_id
    assert m_dispute_id is not None

    # 4. Assert newly created disputed memory temporal metadata:
    # - valid_from is March 1
    # - valid_to is clamped to M1's historical upper bound (June 1)
    # - closed_at is recorded_at (NOT None!)
    # - status is DISPUTED
    m_dispute = service.repo.get_memory(m_dispute_id)
    assert m_dispute is not None
    assert m_dispute.status == MemoryStatus.DISPUTED
    assert m_dispute.valid_from == t_mar1
    assert m_dispute.valid_to == t_jun1
    assert m_dispute.closed_at == t_recorded
    assert m_dispute.closed_at is not None

    # 5. Assert existing historical memory M1 metadata:
    # - status transitions to DISPUTED
    # - valid_from, valid_to, and closed_at preserved
    m1_after = service.repo.get_memory(m1_id)
    assert m1_after is not None
    assert m1_after.status == MemoryStatus.DISPUTED
    assert m1_after.valid_from == t_jan1
    assert m1_after.valid_to == t_jun1
    assert m1_after.closed_at == t_jun1

    # 6. Assert active M2 remains untouched
    m2_after = service.repo.get_memory(m2_id)
    assert m2_after is not None
    assert m2_after.status == MemoryStatus.ACTIVE
    assert m2_after.valid_from == t_jun1
    assert m2_after.valid_to is None
    assert m2_after.closed_at is None

    # 7. Invariant check: zero violations across database
    report = service.check_invariants()
    assert report.is_valid is True
    assert len(report.violations) == 0


def test_active_memory_dispute_temporal_metadata_open_ended(
    service: MemoryService,
) -> None:
    """Active memory dispute retains open-ended windows (valid_to=None, closed_at=None)."""
    t_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t_conflict = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    t_recorded = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)

    # Ingest active memory with open-ended validity
    rep1 = service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya",
            text="Server A scheduled deploy day is friday",
            event_time=t_start,
        ),
        [
            Claim(
                subject="Server A",
                attribute="deploy_day",
                value="friday",
                valid_from=t_start,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_start,
    )
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # Ingest conflicting claim with open-ended validity (source dev_chat reliability 0.60)
    rep2 = service.ingest_claims(
        IngestMessage(
            source_id="dev_chat",
            text="Team chat says deploy day is monday",
            event_time=t_conflict,
        ),
        [
            Claim(
                subject="Server A",
                attribute="deploy_day",
                value="monday",
                valid_from=t_conflict,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_recorded,
    )
    assert rep2.decisions[0].rule_id == "R-DISPUTE"
    m_dispute_id = rep2.decisions[0].memory_id
    assert m_dispute_id is not None

    m_dispute = service.repo.get_memory(m_dispute_id)
    assert m_dispute is not None
    assert m_dispute.status == MemoryStatus.DISPUTED
    assert m_dispute.valid_to is None
    assert m_dispute.closed_at is None

    m1_after = service.repo.get_memory(m1_id)
    assert m1_after is not None
    assert m1_after.status == MemoryStatus.DISPUTED
    assert m1_after.valid_to is None
    assert m1_after.closed_at is None

    report = service.check_invariants()
    assert report.is_valid is True
    assert len(report.violations) == 0


def test_historical_dispute_invalid_temporal_window_rolls_back_transaction(
    service: MemoryService,
) -> None:
    """Historical dispute with invalid validity window raises error and rolls back cleanly."""
    t_jan1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t_jun1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    t_recorded = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)

    # Ingest initial claim M1 and supersede with M2
    rep1 = service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Initial", event_time=t_jan1),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                valid_from=t_jan1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_jan1,
    )
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    service.ingest_claims(
        IngestMessage(source_id="sysadmin_priya", text="Upgrade", event_time=t_jun1),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                valid_from=t_jun1,
                extract_confidence=0.95,
            )
        ],
        recorded_at=t_jun1,
    )

    # Ingest invalid claim whose valid_to <= valid_from (e.g. inverted bounds)
    t_invalid_from = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    t_invalid_to = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)

    msg = IngestMessage(
        source_id="ci_monitor",
        text="Invalid inverted interval claim",
        event_time=t_invalid_from,
    )
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.8",
        valid_from=t_invalid_from,
        valid_to=t_invalid_to,
        extract_confidence=0.90,
    )

    with pytest.raises(InvalidValidityWindowError):
        service.ingest_claims(msg, [claim], recorded_at=t_recorded)

    # Assert atomic rollback:
    # 1. M1 remains HISTORICAL (NOT DISPUTED)
    m1_after = service.repo.get_memory(m1_id)
    assert m1_after is not None
    assert m1_after.status == MemoryStatus.HISTORICAL
    assert m1_after.valid_to == t_jun1
    assert m1_after.closed_at == t_jun1

    # 2. Ingest message rolled back
    msg_hash = hashlib.sha256(msg.text.encode("utf-8")).hexdigest()
    assert service.repo.get_message_by_hash(msg.source_id, msg_hash, msg.event_time) is None

    # 3. No new edges inserted
    edges = service.repo.get_edges_for_memory(m1_id)
    assert len(edges) == 1
    assert edges[0].type == EdgeType.SUPERSEDES

    # 4. Invariant check is clean
    report = service.check_invariants()
    assert report.is_valid is True
    assert len(report.violations) == 0
