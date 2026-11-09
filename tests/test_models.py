"""Tests for MemoryMesh domain models and data shapes."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from memorymesh.models import (
    Action,
    AddEdge,
    AddEvidence,
    AttrRef,
    Cardinality,
    Claim,
    Decision,
    DecisionLogEntry,
    EdgeType,
    Evidence,
    FinalAnswer,
    IngestMessage,
    IngestReport,
    InsertMemory,
    Memory,
    MemoryStatus,
    MemoryView,
    NewEvidence,
    NewMemory,
    Polarity,
    QueryPlan,
    QueryResult,
    RecomputeConfidence,
    Relation,
    ResolvedClaim,
    Source,
    UpdateMemory,
    ValueType,
    format_datetime_utc,
)


def test_enum_members_and_values() -> None:
    """Verify all StrEnum types expose expected specification values."""
    assert Cardinality.SINGLE == "single"
    assert Cardinality.MULTI == "multi"

    assert ValueType.VERSION == "version"
    assert ValueType.ENUM == "enum"
    assert ValueType.TEXT == "text"
    assert ValueType.NUMBER == "number"
    assert ValueType.DATE == "date"

    assert MemoryStatus.ACTIVE == "active"
    assert MemoryStatus.HISTORICAL == "historical"
    assert MemoryStatus.REJECTED == "rejected"
    assert MemoryStatus.DISPUTED == "disputed"

    assert Polarity.ASSERT == "assert"
    assert Polarity.RETRACT == "retract"

    assert Relation.IDENTICAL == "identical"
    assert Relation.REFINES == "refines"
    assert Relation.CONFLICTS == "conflicts"
    assert Relation.COMPLEMENTARY == "complementary"
    assert Relation.UNRELATED == "unrelated"

    assert Action.INSERT == "INSERT"
    assert Action.CORROBORATE == "CORROBORATE"
    assert Action.REFINE == "REFINE"
    assert Action.COEXIST == "COEXIST"
    assert Action.REPLACE == "REPLACE"
    assert Action.BACKFILL == "BACKFILL"
    assert Action.REJECT == "REJECT"
    assert Action.DISPUTE == "DISPUTE"
    assert Action.RETRACT == "RETRACT"
    assert Action.SKIP == "SKIP"
    assert Action.PROPAGATE == "PROPAGATE"
    assert Action.ACKNOWLEDGE == "ACKNOWLEDGE"

    assert EdgeType.SUPERSEDES == "supersedes"
    assert EdgeType.REFINES == "refines"
    assert EdgeType.CONTRADICTS == "contradicts"
    assert EdgeType.DEPENDS_ON == "depends_on"


def test_datetime_serialization_format() -> None:
    """Verify timestamps serialize to YYYY-MM-DDTHH:MM:SSZ UTC strings in JSON."""
    dt = datetime(2026, 6, 25, 9, 30, 45, tzinfo=UTC)
    msg = IngestMessage(source_id="test", text="hello", event_time=dt)

    raw_json = msg.model_dump_json()
    data = json.loads(raw_json)
    assert data["event_time"] == "2026-06-25T09:30:45Z"

    # Also test naive datetime treated as UTC
    naive_dt = datetime(2026, 3, 15, 12, 0, 0)
    assert format_datetime_utc(naive_dt) == "2026-03-15T12:00:00Z"

    msg2 = IngestMessage(source_id="test2", text="world", event_time=naive_dt)
    assert json.loads(msg2.model_dump_json())["event_time"] == "2026-03-15T12:00:00Z"


def test_source_model_validation() -> None:
    """Verify Source model field constraints."""
    source = Source(
        id="sysadmin_priya",
        name="Priya",
        type="admin",
        reliability=0.90,
    )
    assert source.reliability == 0.90
    assert source.notes == ""

    # Out of range reliability
    with pytest.raises(ValidationError):
        Source(id="bad", name="Bad", type="admin", reliability=-0.1)

    with pytest.raises(ValidationError):
        Source(id="bad", name="Bad", type="admin", reliability=1.1)

    # Invalid source type
    with pytest.raises(ValidationError):
        Source(id="bad", name="Bad", type="invalid_type", reliability=0.5)  # type: ignore[arg-type]


def test_claim_model_defaults_and_validation() -> None:
    """Verify Claim default factories and constraints."""
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.95,
    )
    assert claim.polarity == Polarity.ASSERT
    assert claim.valid_from is None
    assert claim.valid_to is None
    assert claim.context == {}
    assert claim.depends_on == []
    assert claim.span == ""

    # Verify mutable default isolation
    claim2 = Claim(
        subject="Server B",
        attribute="python_version",
        value="3.10",
        extract_confidence=0.8,
    )
    claim.context["env"] = "staging"
    claim.depends_on.append(AttrRef(entity="Server A", attribute="os_release"))
    assert claim2.context == {}
    assert claim2.depends_on == []


def test_resolved_claim_roundtrip() -> None:
    """Verify ResolvedClaim structure and round-trip serialization."""
    src = Source(id="priya", name="Priya", type="admin", reliability=0.9)
    claim = Claim(
        subject="Server A", attribute="python_version", value="3.12", extract_confidence=0.9
    )
    now = datetime(2026, 6, 25, 9, 0, 0, tzinfo=UTC)

    rc = ResolvedClaim(
        claim=claim,
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="3.12",
        context_key="env=prod",
        valid_from=now,
        valid_to=None,
        source=src,
        message_id=42,
    )

    dumped = rc.model_dump_json()
    loaded = ResolvedClaim.model_validate_json(dumped)
    assert loaded.entity_id == 1
    assert loaded.value_norm == "3.12"
    assert loaded.source.id == "priya"


def test_stored_memory_and_evidence() -> None:
    """Verify Memory and Evidence models with nested relations."""
    now = datetime(2026, 6, 25, 9, 0, 0, tzinfo=UTC)
    ev = Evidence(
        id=10,
        memory_id=1,
        source_id="priya",
        message_id=101,
        strength=0.855,
        observed_at=now,
        span="Python 3.12 installed",
    )

    mem = Memory(
        id=1,
        entity_id=1,
        entity_name="Server A",
        attribute="python_version",
        value_norm="3.12",
        value_display="3.12",
        valid_from=now,
        recorded_at=now,
        status=MemoryStatus.ACTIVE,
        confidence=0.855,
        message_id=101,
        source_id="priya",
        evidence=[ev],
    )

    assert mem.status == MemoryStatus.ACTIVE
    assert mem.needs_review is False
    assert len(mem.evidence) == 1
    assert mem.evidence[0].strength == 0.855


def test_mutation_union_types() -> None:
    """Verify that Mutation union supports all specified mutation types."""
    now = datetime(2026, 6, 25, 9, 0, 0, tzinfo=UTC)

    new_mem = NewMemory(
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="3.12",
        context_key="",
        valid_from=now,
        status=MemoryStatus.ACTIVE,
        confidence=0.9,
        message_id=1,
        source_id="priya",
    )
    new_ev = NewEvidence(
        source_id="priya",
        message_id=1,
        strength=0.9,
        observed_at=now,
    )

    mutations = [
        InsertMemory(memory=new_mem, evidence=new_ev),
        UpdateMemory(memory_id=1, fields={"status": "historical", "valid_to": now}),
        AddEvidence(memory_id=1, evidence=new_ev),
        AddEdge(src="$new", dst=1, type=EdgeType.SUPERSEDES, reason="newer admin assertion"),
        AddEdge(src=2, dst=1, type=EdgeType.DEPENDS_ON),
        RecomputeConfidence(memory_id="$new"),
        RecomputeConfidence(memory_id=1),
    ]

    dec = Decision(
        action=Action.REPLACE,
        rule_id="R-REPLACE",
        explanation="Admin replaces earlier version",
        scores={"new_score": 0.9, "old_score": 0.8},
        mutations=mutations,
        affected_memory_ids=[1],
    )

    dumped = dec.model_dump_json()
    reparsed = Decision.model_validate_json(dumped)
    assert len(reparsed.mutations) == 7
    assert reparsed.action == Action.REPLACE
    assert reparsed.rule_id == "R-REPLACE"


def test_add_edge_literal_new_and_int() -> None:
    """Verify AddEdge src and dst accept int or literal '$new', rejecting others."""
    # Valid cases
    e1 = AddEdge(src="$new", dst=2, type=EdgeType.SUPERSEDES)
    assert e1.src == "$new"
    assert e1.dst == 2

    e2 = AddEdge(src=1, dst="$new", type=EdgeType.DEPENDS_ON)
    assert e2.src == 1
    assert e2.dst == "$new"

    e3 = AddEdge(src=1, dst=2, type=EdgeType.REFINES)
    assert e3.src == 1
    assert e3.dst == 2

    # Invalid literal
    with pytest.raises(ValidationError):
        AddEdge(src="invalid_id", dst=2, type=EdgeType.SUPERSEDES)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        AddEdge(src=1, dst="other", type=EdgeType.SUPERSEDES)  # type: ignore[arg-type]


def test_query_and_answer_models() -> None:
    """Verify QueryPlan, MemoryView, FinalAnswer, and QueryResult serialization."""
    now = datetime(2026, 6, 25, 9, 0, 0, tzinfo=UTC)
    mem = Memory(
        id=1,
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="3.12",
        valid_from=now,
        recorded_at=now,
        status=MemoryStatus.ACTIVE,
        confidence=0.855,
        message_id=1,
        source_id="priya",
    )

    mv = MemoryView(
        memory=mem,
        effective_confidence=0.855,
        is_stale=False,
        source_names=["Priya (Sysadmin)"],
        scope_match="exact",
    )

    plan = QueryPlan(intent="current", entity="Server A", attribute="python_version")
    qr = QueryResult(plan=plan, items=[mv])
    assert len(qr.items) == 1
    assert qr.items[0].effective_confidence == 0.855

    ans = FinalAnswer(
        value="3.12",
        conflict_flagged=False,
        explanation="Priya confirmed 3.12",
        cited_memory_ids=[1],
    )
    assert ans.value == "3.12"
    assert ans.cited_memory_ids == [1]


def test_decision_log_and_ingest_report() -> None:
    """Verify DecisionLogEntry and IngestReport."""
    now = datetime(2026, 6, 25, 9, 0, 0, tzinfo=UTC)
    entry = DecisionLogEntry(
        id=1,
        message_id=10,
        claim_json={"subject": "Server A", "attribute": "python_version", "value": "3.12"},
        action=Action.INSERT,
        rule_id="R-INSERT",
        memory_id=1,
        scores={"new_score": 0.855},
        explanation="First claim in slot",
        created_at=now,
    )

    rep = IngestReport(
        message_id=10,
        duplicate=False,
        claims_extracted=1,
        decisions=[entry],
    )
    assert rep.message_id == 10
    assert rep.claims_extracted == 1
    assert len(rep.decisions) == 1
    assert rep.decisions[0].rule_id == "R-INSERT"
