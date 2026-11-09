"""Comprehensive test suite for P3.3 Dispute Resolution Service.

Tests:
1. Dispute identification and validation.
2. Resolution behavior across older, newer, and contemporaneous losers.
3. Multi-memory dispute group resolution.
4. Preservation of evidence, history, and original decision logs.
5. Retrieval integration (service.disputes, QueryResult.disputes, get_current safety).
6. Atomicity and rollback on partial failure.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from memorymesh.config import load_app_config
from memorymesh.db import init_db
from memorymesh.errors import MemoryNotFoundError
from memorymesh.models import (
    Action,
    AttributeDef,
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
    conn = init_db(":memory:")
    repo = Repository(conn)

    repo.insert_source(
        Source(
            id="priya",
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

    app_cfg = load_app_config()
    return MemoryService(repo, app_cfg)


def test_identify_valid_unresolved_dispute(service: MemoryService) -> None:
    """Dispute group identifies all participating memories connected by contradicts edges."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

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

    group1 = service.repo.get_dispute_group(m1_id)
    group2 = service.repo.get_dispute_group(m2_id)
    assert len(group1) == 2
    assert len(group2) == 2
    assert {m.id for m in group1} == {m1_id, m2_id}
    assert {m.id for m in group2} == {m1_id, m2_id}


def test_reject_invalid_winner_id(service: MemoryService) -> None:
    """Resolving a non-existent memory ID raises MemoryNotFoundError."""
    with pytest.raises(MemoryNotFoundError, match="Memory with ID 99999 does not exist"):
        service.resolve_dispute(99999)


def test_reject_non_disputed_winner(service: MemoryService) -> None:
    """Resolving a memory that is ACTIVE (not DISPUTED) raises ValueError."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    msg1 = IngestMessage(source_id="priya", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    mem = service.repo.get_memory(m1_id)
    assert mem is not None
    assert mem.status == MemoryStatus.ACTIVE

    with pytest.raises(ValueError, match="expected 'disputed'"):
        service.resolve_dispute(m1_id)


def test_reject_isolated_disputed_memory(service: MemoryService) -> None:
    """A disputed memory with no contradiction edges / group < 2 raises ValueError."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
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

    # Manually transition memory to disputed without contradiction edges
    service.repo.update_memory(m1_id, {"status": MemoryStatus.DISPUTED})

    with pytest.raises(ValueError, match="does not belong to an active multi-memory dispute group"):
        service.resolve_dispute(m1_id)


def test_resolve_dispute_older_loser_becomes_historical(service: MemoryService) -> None:
    """When winner is newer than loser, older loser becomes HISTORICAL with valid_to set."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)
    t_resolve = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)

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

    # Resolve in favor of newer claim m2
    log = service.resolve_dispute(m2_id, recorded_at=t_resolve)
    assert log.rule_id == "R-DISPUTE-RESOLVE"
    assert log.action == Action.DISPUTE
    assert log.memory_id == m2_id
    assert set(log.affected_memory_ids) == {m1_id, m2_id}

    winner = service.repo.get_memory(m2_id)
    loser = service.repo.get_memory(m1_id)
    assert winner is not None
    assert loser is not None

    # Winner is ACTIVE
    assert winner.status == MemoryStatus.ACTIVE
    assert winner.valid_from == t2
    assert winner.valid_to is None

    # Loser is HISTORICAL with valid_to == winner.valid_from
    assert loser.status == MemoryStatus.HISTORICAL
    assert loser.valid_from == t1
    assert loser.valid_to == winner.valid_from
    assert loser.closed_at == t_resolve

    # Supersedes edge created from winner to older loser
    edges = service.repo.get_edges_for_memory(m2_id, direction="src")
    supersedes_edges = [
        e for e in edges if e.type == EdgeType.SUPERSEDES and e.dst_memory_id == m1_id
    ]
    assert len(supersedes_edges) == 1

    # Evidence preserved
    assert len(winner.evidence) > 0
    assert len(loser.evidence) > 0


def test_resolve_dispute_newer_loser_becomes_rejected(service: MemoryService) -> None:
    """When winner is older than loser, newer loser becomes REJECTED."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)
    t_resolve = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)

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

    # Resolve in favor of m1 (the older claim)
    log = service.resolve_dispute(m1_id, recorded_at=t_resolve)
    assert log.rule_id == "R-DISPUTE-RESOLVE"

    winner = service.repo.get_memory(m1_id)
    loser = service.repo.get_memory(m2_id)
    assert winner is not None
    assert loser is not None

    assert winner.status == MemoryStatus.ACTIVE
    assert loser.status == MemoryStatus.REJECTED
    assert loser.closed_at == t_resolve


def test_resolve_dispute_contemporaneous_loser_becomes_rejected(
    service: MemoryService,
) -> None:
    """When winner and loser have identical valid_from, loser becomes REJECTED."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t_resolve = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)

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

    # Same valid_from, conflicting value
    msg2 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.12", event_time=t1)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t1)
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    service.resolve_dispute(m1_id, recorded_at=t_resolve)
    winner = service.repo.get_memory(m1_id)
    loser = service.repo.get_memory(m2_id)
    assert winner is not None
    assert loser is not None

    assert winner.status == MemoryStatus.ACTIVE
    assert loser.status == MemoryStatus.REJECTED
    assert loser.closed_at == t_resolve


def test_resolve_dispute_multi_memory_group(service: MemoryService) -> None:
    """Group of 3 memories resolves winner to ACTIVE, older to HISTORICAL, newer to REJECTED."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)
    t3 = datetime(2026, 6, 3, 10, 0, tzinfo=UTC)
    t_resolve = datetime(2026, 6, 10, 10, 0, tzinfo=UTC)

    # Ingest 3 conflicting claims
    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    m1_id = service.ingest_claims(msg1, [claim1], recorded_at=t1).decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.11", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.11",
        extract_confidence=0.60,
    )
    m2_id = service.ingest_claims(msg2, [claim2], recorded_at=t2).decisions[0].memory_id
    assert m2_id is not None

    msg3 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.12", event_time=t3)
    claim3 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
    )
    m3_id = service.ingest_claims(msg3, [claim3], recorded_at=t3).decisions[0].memory_id
    assert m3_id is not None

    # Check 3-memory dispute group
    group = service.repo.get_dispute_group(m2_id)
    assert len(group) == 3
    assert {m.id for m in group} == {m1_id, m2_id, m3_id}

    # Resolve in favor of m2 (t2)
    service.resolve_dispute(m2_id, recorded_at=t_resolve)

    mem1 = service.repo.get_memory(m1_id)
    mem2 = service.repo.get_memory(m2_id)
    mem3 = service.repo.get_memory(m3_id)
    assert mem1 is not None
    assert mem2 is not None
    assert mem3 is not None

    # m2 is ACTIVE
    assert mem2.status == MemoryStatus.ACTIVE
    # m1 is older (t1 < t2) -> HISTORICAL with valid_to == t2
    assert mem1.status == MemoryStatus.HISTORICAL
    assert mem1.valid_to == t2
    # m3 is newer (t3 > t2) -> REJECTED
    assert mem3.status == MemoryStatus.REJECTED


def test_already_resolved_dispute_cannot_be_re_resolved(service: MemoryService) -> None:
    """Once resolved, calling resolve_dispute on winner or loser raises ValueError."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    m1_id = service.ingest_claims(msg1, [claim1], recorded_at=t1).decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
    )
    m2_id = service.ingest_claims(msg2, [claim2], recorded_at=t2).decisions[0].memory_id
    assert m2_id is not None

    service.resolve_dispute(m2_id)

    # Attempting to re-resolve winner
    with pytest.raises(ValueError, match="expected 'disputed'"):
        service.resolve_dispute(m2_id)

    # Attempting to resolve loser
    with pytest.raises(ValueError, match="expected 'disputed'"):
        service.resolve_dispute(m1_id)


def test_unrelated_memories_remain_untouched(service: MemoryService) -> None:
    """Dispute resolution on one entity/attribute does not touch other entities or attributes."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    # Server A has a dispute on python_version
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

    # Server B has an active memory on python_version
    msg3 = IngestMessage(source_id="priya", text="Server B runs Python 3.11", event_time=t1)
    claim3 = Claim(
        subject="Server B",
        attribute="python_version",
        value="3.11",
        extract_confidence=1.0,
    )
    rep3 = service.ingest_claims(msg3, [claim3], recorded_at=t1)
    m3_id = rep3.decisions[0].memory_id
    assert m3_id is not None

    service.resolve_dispute(m2_id)

    mem_b = service.repo.get_memory(m3_id)
    assert mem_b is not None
    assert mem_b.status == MemoryStatus.ACTIVE
    assert mem_b.valid_to is None


def test_original_decision_log_preserved(service: MemoryService) -> None:
    """Original R-DISPUTE decision log is preserved alongside new R-DISPUTE-RESOLVE log."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    service.ingest_claims(msg1, [claim1], recorded_at=t1)

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

    service.resolve_dispute(m2_id)

    # Check decision logs for msg2
    logs = service.repo.get_decision_logs_for_message(rep2.message_id)
    rule_ids = [entry.rule_id for entry in logs]
    assert "R-DISPUTE" in rule_ids
    assert "R-DISPUTE-RESOLVE" in rule_ids


def test_retrieval_integration_before_and_after_resolution(service: MemoryService) -> None:
    """Disputes populate QueryResult.disputes before resolution and clear after resolution."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)
    t_now = datetime(2026, 6, 5, 10, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    m1_id = service.ingest_claims(msg1, [claim1], recorded_at=t1).decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
    )
    m2_id = service.ingest_claims(msg2, [claim2], recorded_at=t2).decisions[0].memory_id
    assert m2_id is not None

    # Before resolution
    # 1. service.disputes() returns the dispute pair
    dispute_list = service.disputes("Server A")
    assert len(dispute_list) == 1
    assert {dispute_list[0].a.memory.id, dispute_list[0].b.memory.id} == {m1_id, m2_id}

    # 2. service.current() has disputes populated
    res_before = service.current("Server A", "python_version", now=t_now)
    assert len(res_before.disputes) == 1
    assert len(res_before.items) == 2

    # 3. get_current() is dispute-safe and returns None
    assert service.get_current("Server A", "python_version", now=t_now) is None

    # Resolve dispute
    service.resolve_dispute(m2_id, recorded_at=t_now)

    # After resolution
    # 1. service.disputes() is empty
    assert len(service.disputes("Server A")) == 0

    # 2. service.current() disputes is empty
    res_after = service.current("Server A", "python_version", now=t_now)
    assert len(res_after.disputes) == 0

    # 3. get_current() returns the winner
    current_mem = service.get_current("Server A", "python_version", now=t_now)
    assert current_mem is not None
    assert current_mem.id == m2_id
    assert current_mem.value_norm == "3.12"

    # 4. get_as_of() for older time returns older loser as historical
    older_mem = service.get_as_of("Server A", "python_version", valid_at=t1)
    assert older_mem is not None
    assert older_mem.id == m1_id
    assert older_mem.value_norm == "3.10"
    assert older_mem.status == MemoryStatus.HISTORICAL


def test_atomicity_and_rollback_on_failure(service: MemoryService) -> None:
    """If an error occurs during decision logging, all memory updates roll back."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    m1_id = service.ingest_claims(msg1, [claim1], recorded_at=t1).decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
    )
    m2_id = service.ingest_claims(msg2, [claim2], recorded_at=t2).decisions[0].memory_id
    assert m2_id is not None

    # Patch repo.insert_decision_log to raise an error
    with patch.object(
        service.repo, "insert_decision_log", side_effect=RuntimeError("Database failure")
    ):
        with pytest.raises(RuntimeError, match="Database failure"):
            service.resolve_dispute(m2_id)

    # Verify rollback of winner status, loser status & validity fields,
    # supersedes edges, and decision log entries
    mem1 = service.repo.get_memory(m1_id)
    mem2 = service.repo.get_memory(m2_id)
    assert mem1 is not None
    assert mem2 is not None

    # 1. Winner status remains DISPUTED
    assert mem2.status == MemoryStatus.DISPUTED
    assert mem2.valid_to is None
    assert mem2.closed_at is None

    # 2. Loser status and validity fields remain untouched
    assert mem1.status == MemoryStatus.DISPUTED
    assert mem1.valid_to is None
    assert mem1.closed_at is None

    # 3. No SUPERSEDES edge was created
    edges = service.repo.get_edges_for_memory(m2_id, direction="src")
    supersedes_edges = [e for e in edges if e.type == EdgeType.SUPERSEDES]
    assert len(supersedes_edges) == 0

    # 4. No R-DISPUTE-RESOLVE decision-log entry was created
    cursor = service.repo.conn.execute(
        "SELECT COUNT(*) FROM decision_log WHERE rule_id = 'R-DISPUTE-RESOLVE'"
    )
    assert cursor.fetchone()[0] == 0


def test_second_resolution_attempt_is_rejected_without_state_change_or_log(
    service: MemoryService,
) -> None:
    """Repeated resolution attempt after a successful resolution creates no extra log or change."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    m1_id = service.ingest_claims(msg1, [claim1], recorded_at=t1).decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
    )
    m2_id = service.ingest_claims(msg2, [claim2], recorded_at=t2).decisions[0].memory_id
    assert m2_id is not None

    # First resolution succeeds
    service.resolve_dispute(m2_id)

    # Count decision logs in database
    cursor = service.repo.conn.execute("SELECT COUNT(*) FROM decision_log")
    log_count_before = cursor.fetchone()[0]

    # Attempting to resolve m2_id again must raise ValueError
    with pytest.raises(ValueError, match="expected 'disputed'"):
        service.resolve_dispute(m2_id)

    # Attempting to resolve m1_id must also raise ValueError
    with pytest.raises(ValueError, match="expected 'disputed'"):
        service.resolve_dispute(m1_id)

    # Verify no additional decision log rows were written
    cursor = service.repo.conn.execute("SELECT COUNT(*) FROM decision_log")
    log_count_after = cursor.fetchone()[0]
    assert log_count_after == log_count_before

    # Verify memory states remain unchanged
    m1 = service.repo.get_memory(m1_id)
    m2 = service.repo.get_memory(m2_id)
    assert m1 is not None and m1.status == MemoryStatus.HISTORICAL
    assert m2 is not None and m2.status == MemoryStatus.ACTIVE


def test_competing_resolution_attempts_prevent_stale_double_resolution(
    service: MemoryService,
) -> None:
    """Two competing callers targeting the same dispute cannot both succeed using stale state."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    m1_id = service.ingest_claims(msg1, [claim1], recorded_at=t1).decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
    )
    m2_id = service.ingest_claims(msg2, [claim2], recorded_at=t2).decisions[0].memory_id
    assert m2_id is not None

    # Caller 1 resolves in favor of m1_id
    log1 = service.resolve_dispute(m1_id)
    assert log1.rule_id == "R-DISPUTE-RESOLVE"
    assert log1.memory_id == m1_id

    # Caller 2 immediately attempts to resolve in favor of m2_id based on previously observed state
    with pytest.raises(ValueError, match="expected 'disputed'"):
        service.resolve_dispute(m2_id)

    # Winner chosen by Caller 1 remains authoritative
    winner = service.repo.get_memory(m1_id)
    loser = service.repo.get_memory(m2_id)
    assert winner is not None and winner.status == MemoryStatus.ACTIVE
    assert loser is not None and loser.status == MemoryStatus.REJECTED


def test_resolve_dispute_rejects_explicit_naive_recorded_at(
    service: MemoryService,
) -> None:
    """Explicit naive recorded_at timestamp is rejected with ValueError."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    m1_id = service.ingest_claims(msg1, [claim1], recorded_at=t1).decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
    )
    m2_id = service.ingest_claims(msg2, [claim2], recorded_at=t2).decisions[0].memory_id
    assert m2_id is not None

    naive_recorded_at = datetime(2026, 6, 5, 12, 0)  # No tzinfo
    with pytest.raises(ValueError, match="timezone-aware datetime"):
        service.resolve_dispute(m2_id, recorded_at=naive_recorded_at)


def test_resolve_dispute_rejects_naive_injected_clock() -> None:
    """Injected clock returning naive datetime is rejected with ValueError."""
    conn = init_db(":memory:")
    repo = Repository(conn)
    repo.insert_source(Source(id="dev_chat", name="Dev Chat", type="chat", reliability=0.60))
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="python_version",
            description="Python version",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
        )
    )
    app_cfg = load_app_config()

    # Clock returns timezone-naive datetime
    naive_clock = lambda: datetime(2026, 6, 5, 12, 0)  # noqa: E731
    service = MemoryService(repo, app_cfg, clock=naive_clock)

    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    m1_id = service.ingest_claims(msg1, [claim1], recorded_at=t1).decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
    )
    m2_id = service.ingest_claims(msg2, [claim2], recorded_at=t2).decisions[0].memory_id
    assert m2_id is not None

    with pytest.raises(ValueError, match="timezone-aware datetime"):
        service.resolve_dispute(m2_id)


def test_resolve_dispute_normalizes_non_utc_offset_to_utc(
    service: MemoryService,
) -> None:
    """Explicit timezone-aware datetime with non-UTC offset is normalized to UTC."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    m1_id = service.ingest_claims(msg1, [claim1], recorded_at=t1).decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
    )
    m2_id = service.ingest_claims(msg2, [claim2], recorded_at=t2).decisions[0].memory_id
    assert m2_id is not None

    # UTC+5:30: 2026-06-05 17:30:00+05:30 == 2026-06-05 12:00:00Z
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    ist_time = datetime(2026, 6, 5, 17, 30, tzinfo=ist_tz)
    expected_utc = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)

    log = service.resolve_dispute(m2_id, recorded_at=ist_time)

    # 1. Decision log created_at is normalized to UTC
    assert log.created_at == expected_utc
    assert log.created_at.tzinfo == UTC

    # 2. Loser closed_at is normalized to UTC
    loser = service.repo.get_memory(m1_id)
    assert loser is not None
    assert loser.closed_at == expected_utc
    assert loser.closed_at.tzinfo == UTC

    # 3. SUPERSEDES edge created_at is normalized to UTC
    edges = service.repo.get_edges_for_memory(m2_id, direction="src")
    supersedes = [e for e in edges if e.type == EdgeType.SUPERSEDES]
    assert len(supersedes) == 1
    assert supersedes[0].created_at == expected_utc
    assert supersedes[0].created_at.tzinfo == UTC


def test_resolve_dispute_accepts_utc_recorded_at(
    service: MemoryService,
) -> None:
    """Explicit UTC timezone-aware datetime continues to work cleanly."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)
    t_resolve = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    m1_id = service.ingest_claims(msg1, [claim1], recorded_at=t1).decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="dev_chat", text="Server A runs Python 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
    )
    m2_id = service.ingest_claims(msg2, [claim2], recorded_at=t2).decisions[0].memory_id
    assert m2_id is not None

    log = service.resolve_dispute(m2_id, recorded_at=t_resolve)
    assert log.created_at == t_resolve

    loser = service.repo.get_memory(m1_id)
    assert loser is not None
    assert loser.closed_at == t_resolve
