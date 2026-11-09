"""Tests for P5.3: G1 Structured Hero Demo.

Governed by docs/SPECIFICATION.md Section 2 and docs/RESOLUTION_ENGINE.md.
Tests verify:
1. Initial administrator claim is accepted (R-INSERT).
2. Administrator update replaces the previous current value (R-REPLACE).
3. Intern's low-trust claim is rejected (R-REJECT).
4. Current retrieval returns 3.12.
5. Historical retrieval returns 3.10.
6. Explanation includes the relevant decision and provenance.
7. Dependency propagation flags the dependent memory.
8. Impact analysis reports the expected dependent and path.
9. Acknowledgement clears only the selected review flag.
10. Acknowledgement preserves the dependent memory's factual and temporal fields.
11. The demo is deterministic and repeatable across multiple runs.
12. CLI entry point run_demo() exits with code 0.
"""

from __future__ import annotations

import io
from collections.abc import Generator
from datetime import UTC, datetime

import pytest

from memorymesh.config import load_app_config
from memorymesh.db import init_db
from memorymesh.models import (
    Action,
    Claim,
    EdgeType,
    IngestMessage,
    MemoryStatus,
)
from memorymesh.service import MemoryService
from scripts.demo_hero_scenario import DEMO_CLOCK_REF, run_demo


@pytest.fixture
def hero_service() -> Generator[MemoryService, None, None]:
    """Fixture providing an isolated in-memory MemoryService loaded with app config."""
    conn = init_db(":memory:")
    cfg = load_app_config()
    service = MemoryService(db_target=conn, cfg=cfg)
    try:
        yield service
    finally:
        conn.close()


def test_hero_demo_step_by_step_verification(hero_service: MemoryService) -> None:
    """Verify each stage of the G1 Hero Demo scenario independently and thoroughly."""
    service = hero_service

    # =========================================================================
    # Stage 1: Initial Administrator Claim
    # =========================================================================
    t1 = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    msg1 = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A provisioned with Python 3.10 runtime",
        event_time=t1,
    )
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        valid_from=t1,
        extract_confidence=0.95,
    )

    report1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    assert len(report1.decisions) == 1
    dec1 = report1.decisions[0]
    assert dec1.rule_id == "R-INSERT"
    assert dec1.action == Action.INSERT

    m1_id = dec1.memory_id
    assert m1_id is not None
    m1 = service.repo.get_memory(m1_id)
    assert m1 is not None
    assert m1.value_norm == "3.10"
    assert m1.status == MemoryStatus.ACTIVE
    assert m1.valid_from == t1
    assert m1.valid_to is None
    assert pytest.approx(m1.confidence, rel=1e-3) == 0.855

    # =========================================================================
    # Stage 2: Administrator Update (Trust-Aware Replacement)
    # =========================================================================
    t2 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    msg2 = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A upgraded to Python 3.12",
        event_time=t2,
    )
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=t2,
        extract_confidence=0.95,
    )

    report2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    assert len(report2.decisions) == 1
    dec2 = report2.decisions[0]
    assert dec2.rule_id == "R-REPLACE"
    assert dec2.action == Action.REPLACE

    m2_id = dec2.memory_id
    assert m2_id is not None
    assert m2_id != m1_id

    m2 = service.repo.get_memory(m2_id)
    assert m2 is not None
    assert m2.value_norm == "3.12"
    assert m2.status == MemoryStatus.ACTIVE
    assert m2.valid_from == t2
    assert m2.valid_to is None

    # Verify M1 is closed as historical
    m1_updated = service.repo.get_memory(m1_id)
    assert m1_updated is not None
    assert m1_updated.status == MemoryStatus.HISTORICAL
    assert m1_updated.valid_to == t2

    # Verify supersedes edge from M2 -> M1
    edges_m2 = service.repo.get_edges_for_memory(m2_id, direction="outgoing")
    supersedes_edges = [e for e in edges_m2 if e.type == EdgeType.SUPERSEDES]
    assert len(supersedes_edges) == 1
    assert supersedes_edges[0].dst_memory_id == m1_id

    # =========================================================================
    # Stage 3: Lower-Trust Conflicting Claim Rejection
    # =========================================================================
    t3 = datetime(2026, 6, 5, 0, 0, tzinfo=UTC)
    msg3 = IngestMessage(
        source_id="intern_chat",
        text="Pretty sure Server A runs Python 3.9",
        event_time=t3,
    )
    claim3 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.9",
        valid_from=t3,
        extract_confidence=0.70,
    )

    report3 = service.ingest_claims(msg3, [claim3], recorded_at=t3)
    assert len(report3.decisions) == 1
    dec3 = report3.decisions[0]
    assert dec3.rule_id == "R-REJECT"
    assert dec3.action == Action.REJECT
    assert "below reject threshold" in dec3.explanation

    # Current memory M2 must remain unchanged and active
    m2_check = service.repo.get_memory(m2_id)
    assert m2_check is not None
    assert m2_check.value_norm == "3.12"
    assert m2_check.status == MemoryStatus.ACTIVE

    # =========================================================================
    # Stage 4: Temporal Retrieval
    # =========================================================================
    t_curr = datetime(2026, 6, 15, 0, 0, tzinfo=UTC)
    t_hist = datetime(2026, 3, 15, 0, 0, tzinfo=UTC)

    # Current query at t_curr
    res_curr = service.current("Server A", "python_version", now=t_curr)
    assert len(res_curr.items) == 1
    assert res_curr.items[0].memory.id == m2_id
    assert res_curr.items[0].memory.value_norm == "3.12"
    assert res_curr.items[0].memory.status == MemoryStatus.ACTIVE

    # Historical query at t_hist (between t1 and t2)
    res_hist = service.current("Server A", "python_version", valid_at=t_hist, now=t_curr)
    assert len(res_hist.items) == 1
    assert res_hist.items[0].memory.id == m1_id
    assert res_hist.items[0].memory.value_norm == "3.10"
    assert res_hist.items[0].memory.status == MemoryStatus.HISTORICAL

    # Also check convenience methods get_current and get_as_of
    mem_curr = service.get_current("Server A", "python_version", now=t_curr)
    assert mem_curr is not None
    assert mem_curr.value_norm == "3.12"

    mem_hist = service.get_as_of("Server A", "python_version", valid_at=t_hist)
    assert mem_hist is not None
    assert mem_hist.value_norm == "3.10"

    # =========================================================================
    # Stage 5: Decision Explainability and Provenance
    # =========================================================================
    exp = service.explain(m2_id)
    assert exp.items[0].memory.id == m2_id
    assert "sysadmin_priya" in exp.items[0].source_names
    assert any("ACTIVE" in n for n in exp.notes)
    assert any("0.8550" in n for n in exp.notes)
    assert any(d.rule_id == "R-REPLACE" for d in exp.decisions)
    assert any(e.type == EdgeType.SUPERSEDES for e in exp.edges)

    # =========================================================================
    # Stage 6: Multi-Level Dependency Ingestion and Change Propagation
    # =========================================================================
    msg_dep1 = IngestMessage(
        source_id="sysadmin_priya",
        text="Deploying API Gateway on Server A",
        event_time=t1,
    )
    claim_dep1 = Claim(
        subject="Server A",
        attribute="runs_service",
        value="api_gateway",
        valid_from=t1,
        extract_confidence=0.95,
    )
    rep_dep1 = service.ingest_claims(msg_dep1, [claim_dep1], recorded_at=t1)
    m_dep1_id = rep_dep1.decisions[0].memory_id
    assert m_dep1_id is not None

    msg_dep2 = IngestMessage(
        source_id="sysadmin_priya",
        text="Service status is operational up",
        event_time=t1,
    )
    claim_dep2 = Claim(
        subject="Server A",
        attribute="service_status",
        value="up",
        valid_from=t1,
        extract_confidence=0.95,
    )
    rep_dep2 = service.ingest_claims(msg_dep2, [claim_dep2], recorded_at=t1)
    m_dep2_id = rep_dep2.decisions[0].memory_id
    assert m_dep2_id is not None

    # Establish dependency chain: M_dep2 -> M_dep1 -> M2
    service.repo.insert_edge(
        src_memory_id=m_dep1_id,
        dst_memory_id=m2_id,
        edge_type=EdgeType.DEPENDS_ON,
        reason="API Gateway requires Python 3.12 runtime",
        created_at=t2,
    )
    service.repo.insert_edge(
        src_memory_id=m_dep2_id,
        dst_memory_id=m_dep1_id,
        edge_type=EdgeType.DEPENDS_ON,
        reason="Service status depends on API Gateway uptime",
        created_at=t2,
    )

    # Before propagation: neither is flagged
    assert service.repo.get_memory(m_dep1_id).needs_review is False
    assert service.repo.get_memory(m_dep2_id).needs_review is False

    # Trigger upstream change propagation
    t4 = datetime(2026, 6, 20, 0, 0, tzinfo=UTC)
    audit_msg_id = service.repo.insert_message(
        source_id="sysadmin_priya",
        text="Security scan alert: CVE-2026-4444 advisory",
        event_time=t4,
    )
    prop_logs = service.propagate_change(
        changed_ids=[m2_id],
        cause_text="CVE-2026-4444 security advisory",
        message_id=audit_msg_id,
        recorded_at=t4,
        trigger_mode="manual",
    )
    assert len(prop_logs) == 2
    assert all(log.rule_id == "R-PROPAGATE" for log in prop_logs)

    # Verify both downstream memories are flagged for review
    m_dep1_flagged = service.repo.get_memory(m_dep1_id)
    assert m_dep1_flagged is not None
    assert m_dep1_flagged.needs_review is True
    assert "CVE-2026-4444" in m_dep1_flagged.review_reason
    assert f"M{m2_id}" in m_dep1_flagged.review_reason

    m_dep2_flagged = service.repo.get_memory(m_dep2_id)
    assert m_dep2_flagged is not None
    assert m_dep2_flagged.needs_review is True
    assert "CVE-2026-4444" in m_dep2_flagged.review_reason
    assert f"M{m2_id} -> M{m_dep1_id} -> M{m_dep2_id}" in m_dep2_flagged.review_reason

    # =========================================================================
    # Stage 7: Downstream Impact Analysis
    # =========================================================================
    impact_items = service.impact(m2_id)
    assert len(impact_items) == 2

    # Direct dependent: M_dep1
    item_direct = next(it for it in impact_items if it.memory.memory.id == m_dep1_id)
    assert item_direct.path == [m2_id, m_dep1_id]
    assert item_direct.memory.memory.needs_review is True
    assert "Direct dependent" in item_direct.reason
    assert item_direct.paths_truncated is False

    # Transitive dependent: M_dep2
    item_trans = next(it for it in impact_items if it.memory.memory.id == m_dep2_id)
    assert item_trans.path == [m2_id, m_dep1_id, m_dep2_id]
    assert item_trans.memory.memory.needs_review is True
    assert "Transitive dependent" in item_trans.reason
    assert item_trans.paths_truncated is False

    # =========================================================================
    # Stage 8: Review Acknowledgement and Independent Isolation
    # =========================================================================
    t5 = datetime(2026, 6, 21, 0, 0, tzinfo=UTC)
    ack_log = service.acknowledge_review(
        m_dep1_id,
        recorded_at=t5,
        reviewed_by="SecOps_Alice",
        explanation="Verified CVE-2026-4444 patch applied to runtime",
    )
    assert ack_log.rule_id == "R-ACKNOWLEDGE"
    assert ack_log.action == Action.ACKNOWLEDGE
    assert ack_log.claim_json.get("reviewed_by") == "SecOps_Alice"

    # M_dep1 review flag cleared and factual data completely immutable
    m_dep1_ack = service.repo.get_memory(m_dep1_id)
    assert m_dep1_ack is not None
    assert m_dep1_ack.needs_review is False
    assert m_dep1_ack.review_reason == ""
    assert m_dep1_ack.value_norm == m_dep1_flagged.value_norm
    assert m_dep1_ack.confidence == m_dep1_flagged.confidence
    assert m_dep1_ack.valid_from == m_dep1_flagged.valid_from
    assert m_dep1_ack.valid_to == m_dep1_flagged.valid_to
    assert m_dep1_ack.status == m_dep1_flagged.status

    # Independent isolation: M_dep2 remains under review!
    m_dep2_ack = service.repo.get_memory(m_dep2_id)
    assert m_dep2_ack is not None
    assert m_dep2_ack.needs_review is True
    assert m_dep2_ack.review_reason == m_dep2_flagged.review_reason


def test_hero_demo_run_demo_executable() -> None:
    """Verify scripts/demo_hero_scenario.py executes end-to-end with exit code 0."""
    buf = io.StringIO()
    exit_code = run_demo(stream=buf)
    output = buf.getvalue()

    assert exit_code == 0
    assert "MemoryMesh: G1 Structured Hero Scenario Demonstration (P5.3)" in output
    assert "STEP 1: Initial Administrator Claim" in output
    assert "STEP 2: Administrator Update" in output
    assert "STEP 3: Lower-Trust Conflicting Claim Rejection" in output
    assert "STEP 4: Temporal Retrieval" in output
    assert "STEP 5: Decision Explainability and Provenance" in output
    assert "STEP 6: Dependency Setup and Downstream Propagation" in output
    assert "STEP 7: Downstream Impact Analysis" in output
    assert "STEP 8: Review Acknowledgement and Independent Isolation" in output
    assert "Hero scenario demonstration completed successfully." in output


def test_hero_demo_deterministic_repeatability() -> None:
    """Verify that multiple consecutive runs of the demo yield identical results."""
    buf1 = io.StringIO()
    code1 = run_demo(stream=buf1)
    out1 = buf1.getvalue()

    buf2 = io.StringIO()
    code2 = run_demo(stream=buf2)
    out2 = buf2.getvalue()

    assert code1 == 0 and code2 == 0
    # Compare complete raw outputs without filtering any lines
    assert out1 == out2
    # Verify the fixed reference clock timestamp is present in the output
    ref_ts_str = DEMO_CLOCK_REF.isoformat()
    assert ref_ts_str in out1
    assert f"effective confidence at {ref_ts_str} is" in out1
