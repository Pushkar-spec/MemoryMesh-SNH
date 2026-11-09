"""Comprehensive test suite for P4.1 Decision Log Retrieval, Explain Service, and Invariants.

Tests:
1. Decision log retrieval (by memory_id subject/affected, all decisions, ordering, limits).
2. Explain service across states (ACTIVE, HISTORICAL, DISPUTED, REJECTED, needs_review).
3. Explain service read-only verification (no mutations, no new logs).
4. Invariant checks on clean database (all 8 invariants hold).
5. Invariant detection for corrupted data (temporal, overlap, confidence, FK, schema).
6. Idempotent ingestion verification.
7. Invariant checker read-only verification.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from memorymesh.config import load_app_config
from memorymesh.db import init_db
from memorymesh.engine.invariants import (
    INV_CLOSED_AT_REQUIRED,
    INV_VALID_INTERVAL,
    TemporalViolation,
    check_intervals_overlap,
    validate_temporal_window,
)
from memorymesh.errors import MemoryNotFoundError
from memorymesh.models import (
    Claim,
    IngestMessage,
    Memory,
    MemoryStatus,
)
from memorymesh.service import MemoryService


@pytest.fixture
def test_clock() -> datetime:
    return datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def service(test_clock: datetime) -> MemoryService:
    conn = init_db(":memory:")
    cfg = load_app_config()
    current_time = test_clock
    return MemoryService(conn, cfg=cfg, clock=lambda: current_time)


# =============================================================================
# Decision Log Retrieval Tests
# =============================================================================


def test_decision_log_retrieval_by_memory_id_and_affected(
    service: MemoryService, test_clock: datetime
) -> None:
    t1 = test_clock
    t2 = t1 + timedelta(hours=1)

    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A upgraded to Python 3.11",
        event_time=t2,
    )
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.11",
        extract_confidence=1.0,
        valid_from=t2,
        raw_text="Server A upgraded to Python 3.11",
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    # Memory 1 decisions should include:
    # 1. INSERT of M1 (primary subject)
    # 2. REPLACE by M2 (M1 listed in affected_memory_ids)
    m1_logs = service.get_decision_logs(memory_id=m1_id)
    assert len(m1_logs) == 2
    assert m1_logs[0].rule_id == "R-INSERT"
    assert m1_logs[1].rule_id == "R-REPLACE"
    assert m1_logs[0].created_at <= m1_logs[1].created_at

    # Memory 2 decisions:
    m2_logs = service.get_decision_logs(memory_id=m2_id)
    assert len(m2_logs) == 1
    assert m2_logs[0].rule_id == "R-REPLACE"
    assert m2_logs[0].memory_id == m2_id


def test_decision_log_retrieval_all_and_limits(
    service: MemoryService, test_clock: datetime
) -> None:
    t1 = test_clock
    t2 = t1 + timedelta(hours=1)

    msg1 = IngestMessage(source_id="dev_chat", text="Fact 1", event_time=t1)
    claim1 = Claim(
        subject="Host 1",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Fact 1",
    )
    service.ingest_claims(msg1, [claim1], recorded_at=t1)

    msg2 = IngestMessage(source_id="dev_chat", text="Fact 2", event_time=t2)
    claim2 = Claim(
        subject="Host 2",
        attribute="python_version",
        value="3.11",
        extract_confidence=1.0,
        valid_from=t2,
        raw_text="Fact 2",
    )
    service.ingest_claims(msg2, [claim2], recorded_at=t2)

    all_logs = service.get_decision_logs()
    assert len(all_logs) == 2

    limited = service.get_decision_logs(limit=1)
    assert len(limited) == 1
    assert limited[0].id == all_logs[0].id


def test_decision_log_retrieval_memory_specific_with_limit(
    service: MemoryService, test_clock: datetime
) -> None:
    t1 = test_clock
    t2 = t1 + timedelta(hours=1)

    msg1 = IngestMessage(source_id="dev_chat", text="Fact 1", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="sysadmin_priya", text="Fact 2", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.11",
        extract_confidence=1.0,
        valid_from=t2,
    )
    service.ingest_claims(msg2, [claim2], recorded_at=t2)

    # M1 has 2 decisions: INSERT (id=1) and REPLACE (id=2)
    all_m1_logs = service.get_decision_logs(memory_id=m1_id)
    assert len(all_m1_logs) == 2

    # Query with limit=1: must return exactly 1 decision (the earliest)
    limited_m1_logs = service.get_decision_logs(memory_id=m1_id, limit=1)
    assert len(limited_m1_logs) == 1
    assert limited_m1_logs[0].id == all_m1_logs[0].id

    # Query with limit=1, offset=1: must return the second decision
    offset_m1_logs = service.get_decision_logs(memory_id=m1_id, limit=1, offset=1)
    assert len(offset_m1_logs) == 1
    assert offset_m1_logs[0].id == all_m1_logs[1].id

    # Query with limit=None, offset=1: must return all decisions starting at index 1
    offset_only_m1_logs = service.get_decision_logs(memory_id=m1_id, limit=None, offset=1)
    assert len(offset_only_m1_logs) == 1
    assert offset_only_m1_logs[0].id == all_m1_logs[1].id

    # Also verify repository method directly with limit=None, offset=1
    repo_offset_only = service.repo.get_decision_logs_for_memory(m1_id, limit=None, offset=1)
    assert len(repo_offset_only) == 1
    assert repo_offset_only[0].id == all_m1_logs[1].id


def test_decision_log_retrieval_nonexistent_memory(service: MemoryService) -> None:
    with pytest.raises(MemoryNotFoundError, match="Memory with id=99999 not found"):
        service.get_decision_logs(memory_id=99999)


# =============================================================================
# Explain Service Tests
# =============================================================================


def test_explain_active_memory(service: MemoryService, test_clock: datetime) -> None:
    t1 = test_clock
    msg = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t1)
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    qr = service.explain(m_id)
    assert qr.plan.intent == "why"
    assert qr.plan.memory_id == m_id
    assert qr.plan.entity == "Server A"
    assert qr.plan.attribute == "python_version"

    assert len(qr.items) == 1
    view = qr.items[0]
    assert view.memory.status == MemoryStatus.ACTIVE
    assert view.effective_confidence > 0.0
    assert "dev_chat" in view.source_names
    assert len(qr.rejected_claims) == 0
    assert len(qr.disputes) == 0
    assert len(qr.decisions) == 1
    assert any("ACTIVE" in n for n in qr.notes)
    assert any("dev_chat" in n for n in qr.notes)


def test_explain_historical_superseded_memory(service: MemoryService, test_clock: datetime) -> None:
    t1 = test_clock
    t2 = t1 + timedelta(hours=1)

    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A upgraded to Python 3.11",
        event_time=t2,
    )
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.11",
        extract_confidence=1.0,
        valid_from=t2,
        raw_text="Server A upgraded to Python 3.11",
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    qr = service.explain(m1_id)
    assert qr.items[0].memory.status == MemoryStatus.HISTORICAL
    assert len(qr.edges) == 1
    assert qr.edges[0].type.value == "supersedes"
    assert any("HISTORICAL" in n for n in qr.notes)
    assert any("supersedes" in n for n in qr.notes)


def test_explain_disputed_memory(service: MemoryService, test_clock: datetime) -> None:
    t1 = test_clock
    t2 = t1 + timedelta(hours=1)

    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # Conflicting claim at same trust level triggers DISPUTE
    msg2 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    qr = service.explain(m1_id)
    assert qr.items[0].memory.status == MemoryStatus.DISPUTED
    assert len(qr.disputes) == 1
    assert any("DISPUTED" in n for n in qr.notes)
    assert any("dispute" in n.lower() for n in qr.notes)


def test_explain_rejected_memory(service: MemoryService, test_clock: datetime) -> None:
    t1 = test_clock
    t2 = t1 + timedelta(hours=1)
    t3 = t1 + timedelta(hours=2)

    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    # Resolve dispute in favor of M1 (M2 becomes REJECTED)
    service.resolve_dispute(
        winner_memory_id=m1_id,
        recorded_at=t3,
    )

    qr = service.explain(m2_id)
    assert qr.items[0].memory.status == MemoryStatus.REJECTED
    assert len(qr.rejected_claims) == 1
    assert qr.rejected_claims[0].memory.id == m2_id
    assert any("REJECTED" in n for n in qr.notes)


def test_explain_strictly_read_only(service: MemoryService, test_clock: datetime) -> None:
    t1 = test_clock
    msg = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t1)
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    def get_counts() -> dict[str, int]:
        counts: dict[str, int] = {}
        for tbl in ["memories", "edges", "decision_log", "evidence", "messages"]:
            row = service.repo.conn.execute(f"SELECT COUNT(*) AS c FROM {tbl}").fetchone()
            counts[tbl] = int(row["c"])
        return counts

    counts_before = get_counts()

    # Call explain twice
    qr1 = service.explain(m_id)
    qr2 = service.explain(m_id)

    counts_after = get_counts()
    assert counts_before == counts_after
    assert qr1.notes == qr2.notes
    assert len(qr1.decisions) == len(qr2.decisions)


# =============================================================================
# Invariant Checker Tests
# =============================================================================


def test_invariant_checker_clean_database(service: MemoryService, test_clock: datetime) -> None:
    t1 = test_clock
    msg = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )
    service.ingest_claims(msg, [claim], recorded_at=t1)

    rep = service.check_invariants()
    assert rep.is_valid is True
    assert len(rep.violations) == 0
    assert rep.total_checked > 0


def test_invariant_checker_detects_closed_at_missing(
    service: MemoryService, test_clock: datetime
) -> None:
    t1 = test_clock
    msg = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )
    service.ingest_claims(msg, [claim], recorded_at=t1)

    # Bypass CHECK constraints to inject corruption: valid_to set but closed_at NULL
    service.repo.conn.execute("PRAGMA ignore_check_constraints=ON")
    service.repo.conn.execute(
        "UPDATE memories SET valid_to = '2026-06-02T00:00:00Z', closed_at = NULL WHERE id = 1"
    )

    rep = service.check_invariants()
    assert rep.is_valid is False
    v_ids = [v.invariant_id for v in rep.violations]
    assert "INV-CLOSED-AT-REQUIRED" in v_ids
    v = next(v for v in rep.violations if v.invariant_id == "INV-CLOSED-AT-REQUIRED")
    assert 1 in v.affected_ids


def test_invariant_checker_detects_invalid_interval(
    service: MemoryService, test_clock: datetime
) -> None:
    t1 = test_clock
    msg = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )
    service.ingest_claims(msg, [claim], recorded_at=t1)

    # Bypass CHECK constraints to inject corruption: valid_to <= valid_from
    service.repo.conn.execute("PRAGMA ignore_check_constraints=ON")
    service.repo.conn.execute(
        "UPDATE memories SET valid_from = '2026-06-05T00:00:00Z', "
        "valid_to = '2026-06-01T00:00:00Z', closed_at = '2026-06-01T00:00:00Z' WHERE id = 1"
    )

    rep = service.check_invariants()
    assert rep.is_valid is False
    v_ids = [v.invariant_id for v in rep.violations]
    assert "INV-VALID-INTERVAL" in v_ids


def test_invariant_checker_detects_overlapping_active_windows(
    service: MemoryService, test_clock: datetime
) -> None:
    t1 = test_clock
    msg = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )
    service.ingest_claims(msg, [claim], recorded_at=t1)

    # Directly insert second active memory with overlapping window
    # for same entity & single attribute
    row = service.repo.conn.execute("SELECT * FROM memories WHERE id = 1").fetchone()
    service.repo.conn.execute(
        """
        INSERT INTO memories (
            entity_id, attribute, value_norm, value_display, context_json, context_key,
            valid_from, valid_to, recorded_at, closed_at, status, confidence,
            needs_review, review_reason, raw_text, message_id, source_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, 0, '', ?, ?, ?)
        """,
        (
            row["entity_id"],
            row["attribute"],
            "3.12",
            "3.12",
            row["context_json"],
            row["context_key"],
            "2026-06-01T00:00:00Z",
            None,
            "2026-06-01T00:00:00Z",
            None,
            row["confidence"],
            "raw",
            row["message_id"],
            row["source_id"],
        ),
    )

    rep = service.check_invariants()
    assert rep.is_valid is False
    v_ids = [v.invariant_id for v in rep.violations]
    assert "INV-NO-ACTIVE-OVERLAP" in v_ids


def test_invariant_checker_detects_confidence_mismatch(
    service: MemoryService, test_clock: datetime
) -> None:
    t1 = test_clock
    msg = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )
    service.ingest_claims(msg, [claim], recorded_at=t1)

    # Tamper with base confidence
    service.repo.conn.execute("UPDATE memories SET confidence = 0.9999 WHERE id = 1")

    rep = service.check_invariants()
    assert rep.is_valid is False
    v_ids = [v.invariant_id for v in rep.violations]
    assert "INV-CONFIDENCE-CONSISTENT" in v_ids


def test_invariant_checker_confidence_empty_evidence(
    service: MemoryService, test_clock: datetime
) -> None:
    t1 = test_clock
    msg = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )
    service.ingest_claims(msg, [claim], recorded_at=t1)

    # Remove evidence records for memory 1
    service.repo.conn.execute("DELETE FROM evidence WHERE memory_id = 1")

    # Case 1: stored confidence is nonzero (0.50) without evidence -> violation
    rep1 = service.check_invariants()
    assert rep1.is_valid is False
    v_ids = [v.invariant_id for v in rep1.violations]
    assert "INV-CONFIDENCE-CONSISTENT" in v_ids
    v = next(v for v in rep1.violations if v.invariant_id == "INV-CONFIDENCE-CONSISTENT")
    assert v.details["expected_confidence"] == 0.0

    # Case 2: stored confidence is 0.0 without evidence -> valid
    service.repo.conn.execute("UPDATE memories SET confidence = 0.0 WHERE id = 1")
    rep2 = service.check_invariants()
    assert rep2.is_valid is True


def test_invariant_checker_idempotent_ingestion(
    service: MemoryService, test_clock: datetime
) -> None:
    t1 = test_clock
    msg = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )

    # First ingestion
    rep1 = service.ingest_claims(msg, [claim], recorded_at=t1)
    assert rep1.duplicate is False
    assert len(rep1.decisions) == 1

    def capture_db_state() -> dict[str, list[dict[str, Any]]]:
        tables = ["messages", "memories", "evidence", "edges", "decision_log"]
        state: dict[str, list[dict[str, Any]]] = {}
        for tbl in tables:
            rows = service.repo.conn.execute(f"SELECT * FROM {tbl} ORDER BY id ASC").fetchall()
            state[tbl] = [dict(r) for r in rows]
        return state

    state_before = capture_db_state()

    # Second ingestion of identical message and claim
    rep2 = service.ingest_claims(msg, [claim], recorded_at=t1)
    assert rep2.duplicate is True
    assert len(rep2.decisions) == 0

    state_after = capture_db_state()
    # Detailed row-by-row state comparison across all tables
    assert state_before == state_after

    # Invariants should be completely valid
    rep = service.check_invariants()
    assert rep.is_valid is True


def test_invariant_checker_backfilled_rejected_memory(
    service: MemoryService, test_clock: datetime
) -> None:
    # Backfilled scenario: valid_from in past, recorded_at later
    t_valid = test_clock
    t_recorded = test_clock + timedelta(days=5)

    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t_valid)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t_valid,
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t_recorded)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.12", event_time=t_valid)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
        valid_from=t_valid,
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t_recorded + timedelta(hours=1))
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    # Resolve dispute so M2 is REJECTED
    service.resolve_dispute(
        winner_memory_id=m1_id,
        recorded_at=t_recorded + timedelta(hours=2),
    )

    # Invariant checker must verify INV-REJECTED-EXCLUDED on the backfilled rejected memory M2
    rep = service.check_invariants()
    assert rep.is_valid is True

    # Confirm that if include_rejected=True, M2 WOULD be returned
    # (proving the test specifically verifies the status filter, not temporal exclusion)
    m2 = service.repo.get_memory(m2_id)
    assert m2 is not None
    assert m2.status == MemoryStatus.REJECTED
    with_rejected = service.repo.find_memories(
        entity_id=m2.entity_id,
        attribute="python_version",
        valid_at=t_valid,
        known_at=t_recorded + timedelta(hours=2),
        include_rejected=True,
    )
    assert any(m.id == m2_id for m in with_rejected)


def test_invariant_checker_rejects_naive_recorded_at(
    service: MemoryService, test_clock: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    t1 = test_clock
    msg1 = IngestMessage(source_id="dev_chat", text="Fact 1", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="dev_chat", text="Fact 2", event_time=t1)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t1 + timedelta(hours=1))
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    service.resolve_dispute(
        winner_memory_id=m1_id,
        recorded_at=t1 + timedelta(hours=2),
    )

    orig_get_all = service.repo.get_all_memories

    def mock_get_all(include_evidence: bool = True) -> list[Memory]:
        mems = orig_get_all(include_evidence=include_evidence)
        for m in mems:
            if m.id == m2_id:
                m.recorded_at = datetime(2026, 6, 1, 12, 0, 0)  # naive datetime
        return mems

    monkeypatch.setattr(service.repo, "get_all_memories", mock_get_all)

    with pytest.raises(ValueError, match="invalid timezone-naive recorded_at"):
        service.check_invariants()


def test_invariant_checker_total_checked_definition(
    service: MemoryService, test_clock: datetime
) -> None:
    t1 = test_clock
    msg = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )
    service.ingest_claims(msg, [claim], recorded_at=t1)

    rep = service.check_invariants()
    counts = service.repo.get_record_counts()

    # Must count exactly the 5 primary domain tables
    assert set(counts.keys()) == {
        "memories",
        "messages",
        "edges",
        "decision_log",
        "evidence",
    }

    expected_total = sum(counts.values())
    assert rep.total_checked == expected_total
    assert rep.total_checked > 0

    # Ensure catalog/reference tables are excluded from total_checked
    catalog_tables = ["sources", "attribute_schema", "entities", "entity_aliases"]
    catalog_counts = sum(
        int(service.repo.conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0])
        for tbl in catalog_tables
    )
    assert catalog_counts > 0
    # total_checked must NOT include the catalog counts
    assert rep.total_checked < expected_total + catalog_counts


def test_invariant_checker_detects_dangling_foreign_keys(
    service: MemoryService, test_clock: datetime
) -> None:
    t1 = test_clock
    msg = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )
    service.ingest_claims(msg, [claim], recorded_at=t1)

    # Insert edge referencing nonexistent dst_memory_id=99999
    service.repo.conn.execute("PRAGMA foreign_keys=OFF")
    service.repo.conn.execute(
        """
        INSERT INTO edges (src_memory_id, dst_memory_id, type, reason, created_at)
        VALUES (1, 99999, 'supersedes', 'test edge', '2026-06-01T00:00:00Z')
        """
    )

    rep = service.check_invariants()
    assert rep.is_valid is False
    v_ids = [v.invariant_id for v in rep.violations]
    assert "INV-FOREIGN-KEY-INTEGRITY" in v_ids


def test_invariant_checker_detects_schema_integrity_violations(
    service: MemoryService, test_clock: datetime
) -> None:
    t1 = test_clock
    msg = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )
    service.ingest_claims(msg, [claim], recorded_at=t1)

    # Insert memory referencing non-existent entity_id=99999
    service.repo.conn.execute("PRAGMA foreign_keys=OFF")
    service.repo.conn.execute(
        """
        INSERT INTO memories (
            entity_id, attribute, value_norm, value_display, context_json, context_key,
            valid_from, valid_to, recorded_at, closed_at, status, confidence,
            needs_review, review_reason, raw_text, message_id, source_id
        ) VALUES (99999, 'python_version', '3.10', '3.10', '{}', '', '2026-06-01T00:00:00Z',
            NULL, '2026-06-01T00:00:00Z', NULL, 'active', 0.5, 0, '', 'raw', 1, 'dev_chat')
        """
    )

    rep = service.check_invariants()
    assert rep.is_valid is False
    v_ids = [v.invariant_id for v in rep.violations]
    assert "INV-SOURCE-ATTRIBUTE-INTEGRITY" in v_ids


def test_invariant_checker_is_strictly_read_only(
    service: MemoryService, test_clock: datetime
) -> None:
    t1 = test_clock
    msg = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )
    service.ingest_claims(msg, [claim], recorded_at=t1)

    def get_db_fingerprint() -> list[tuple[str, int]]:
        tables = [
            "memories",
            "edges",
            "decision_log",
            "evidence",
            "messages",
            "entities",
            "sources",
        ]
        res: list[tuple[str, int]] = []
        for tbl in tables:
            row = service.repo.conn.execute(f"SELECT COUNT(*) AS c FROM {tbl}").fetchone()
            res.append((tbl, int(row["c"])))
        return res

    fp_before = get_db_fingerprint()
    rep1 = service.check_invariants()
    rep2 = service.check_invariants()
    fp_after = get_db_fingerprint()

    assert fp_before == fp_after
    assert rep1.is_valid == rep2.is_valid
    assert rep1.total_checked == rep2.total_checked
    assert len(rep1.violations) == len(rep2.violations)


def test_invariant_checker_delegates_to_pure_engine_functions(
    service: MemoryService, test_clock: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    t1 = test_clock
    msg = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t1,
        raw_text="Server A runs Python 3.10",
    )
    report = service.ingest_claims(msg, [claim], recorded_at=t1)
    assert report.decisions and report.decisions[0].memory_id is not None
    mem_id = report.decisions[0].memory_id

    # Insert a second active memory for the single-valued attribute using captured memory ID
    row = service.repo.conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
    assert row is not None
    service.repo.conn.execute(
        """
        INSERT INTO memories (
            entity_id, attribute, value_norm, value_display, context_json, context_key,
            valid_from, valid_to, recorded_at, closed_at, status, confidence,
            needs_review, review_reason, raw_text, message_id, source_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, 0, '', ?, ?, ?)
        """,
        (
            row["entity_id"],
            row["attribute"],
            "3.12",
            "3.12",
            row["context_json"],
            row["context_key"],
            "2026-06-01T00:00:00Z",
            None,
            "2026-06-01T00:00:00Z",
            None,
            row["confidence"],
            "raw",
            row["message_id"],
            row["source_id"],
        ),
    )

    temporal_called = 0
    overlap_called = 0

    import memorymesh.service as service_module

    orig_validate_temporal = service_module.validate_temporal_window
    orig_check_overlap = service_module.check_intervals_overlap

    def spy_validate_temporal(*args: Any, **kwargs: Any) -> list[TemporalViolation]:
        nonlocal temporal_called
        temporal_called += 1
        return orig_validate_temporal(*args, **kwargs)

    def spy_check_overlap(*args: Any, **kwargs: Any) -> bool:
        nonlocal overlap_called
        overlap_called += 1
        return orig_check_overlap(*args, **kwargs)

    monkeypatch.setattr(service_module, "validate_temporal_window", spy_validate_temporal)
    monkeypatch.setattr(service_module, "check_intervals_overlap", spy_check_overlap)

    rep = service.check_invariants()
    assert temporal_called >= 2
    assert overlap_called >= 1
    assert rep.is_valid is False
    v_ids = [v.invariant_id for v in rep.violations]
    assert "INV-NO-ACTIVE-OVERLAP" in v_ids


def test_pure_validate_temporal_window() -> None:
    t1 = datetime(2026, 1, 1, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, tzinfo=UTC)

    assert validate_temporal_window(t1, None, None) == []
    assert validate_temporal_window(t1, t2, t2) == []

    v1 = validate_temporal_window(t1, t2, None)
    assert len(v1) == 1
    assert v1[0].code == INV_CLOSED_AT_REQUIRED
    assert "closed_at is NULL" in v1[0].description

    v2 = validate_temporal_window(t2, t1, t2)
    assert len(v2) == 1
    assert v2[0].code == INV_VALID_INTERVAL
    assert "strictly greater" in v2[0].description


def test_pure_check_intervals_overlap() -> None:
    t1 = datetime(2026, 1, 1, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, tzinfo=UTC)
    t3 = datetime(2026, 9, 1, tzinfo=UTC)

    assert check_intervals_overlap(t1, t2, t2, t3) is False
    assert check_intervals_overlap(t1, t3, t2, t3) is True
    assert check_intervals_overlap(t1, None, t2, None) is True
    assert check_intervals_overlap(t1, t2, t3, None) is False
