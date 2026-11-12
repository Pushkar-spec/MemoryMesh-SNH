"""Tests for P7.3: Query Parser and Answer Composer.

Governed by docs/ARCHITECTURE.md Section 4, docs/SPECIFICATION.md Section 3,
and MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P7.3).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from memorymesh.db import init_db
from memorymesh.llm.answer_composer import AnswerComposer
from memorymesh.llm.client import LLMClient
from memorymesh.llm.providers import FakeProvider
from memorymesh.llm.query_parser import QueryParser
from memorymesh.models import (
    AskResult,
    AttributeDef,
    AttrRef,
    Cardinality,
    Claim,
    IngestMessage,
    ParsedQuery,
    QueryIntent,
    QueryType,
    Source,
    TemporalIntent,
    ValueType,
)
from memorymesh.repository import Repository
from memorymesh.service import MemoryService


class CrashingProvider:
    """Simulated provider that fails with transport/connection errors."""

    def complete(self, *args: Any, **kwargs: Any) -> str:
        raise ConnectionError("Simulated LLM network failure")


class SequenceProvider:
    """Mock provider returning a sequence of predefined responses across successive calls."""

    def __init__(self, sequence: list[str]) -> None:
        self.sequence = list(sequence)
        self.idx = 0
        self.calls: list[dict[str, Any]] = []

    def complete(self, system: str, user: str, *, model: str | None = None) -> str:
        self.calls.append({"system": system, "user": user})
        if self.idx < len(self.sequence):
            res = self.sequence[self.idx]
            self.idx += 1
            return res
        return self.sequence[-1]


@pytest.fixture
def repo() -> Repository:
    conn = init_db(":memory:")
    r = Repository(conn)
    r.insert_source(
        Source(
            id="sysadmin_alice",
            name="Alice Admin",
            type="admin",
            reliability=0.95,
        )
    )
    r.insert_source(
        Source(
            id="junior_bob",
            name="Bob Junior",
            type="user",
            reliability=0.60,
        )
    )
    r.insert_attribute_schema(
        AttributeDef(
            attribute="operating_system",
            description="Operating System",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )
    r.insert_attribute_schema(
        AttributeDef(
            attribute="python_version",
            description="Python Version",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
        )
    )
    r.insert_attribute_schema(
        AttributeDef(
            attribute="runs_service",
            description="Running microservices",
            cardinality=Cardinality.MULTI,
            value_type=ValueType.TEXT,
        )
    )
    r.insert_attribute_schema(
        AttributeDef(
            attribute="database_engine",
            description="Database Engine",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )
    return r


# =============================================================================
# 1. Query Parser Tests
# =============================================================================


def test_query_parser_current_value_question(repo: Repository) -> None:
    """1. Current-value question is parsed into a structured entity, attribute, and intent."""
    response_payload = {
        "entity": "Server A",
        "attribute": "python_version",
        "context": {},
        "intent": "current",
        "target_date": None,
        "reasoning": "User is asking for Server A's current Python version",
    }
    fake_provider = FakeProvider(default_response=json.dumps(response_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    parser = QueryParser(
        llm_client=client,
        attributes=repo.list_attribute_schemas(),
        known_entities=["Server A", "Server B"],
    )

    query = parser.parse("What Python version is Server A currently running?")
    assert query.entity == "Server A"
    assert query.attribute == "python_version"
    assert query.intent == QueryIntent.CURRENT
    assert query.temporal_intent == TemporalIntent.CURRENT
    assert query.is_historical is False
    assert query.is_why is False
    assert query.is_impact is False
    assert query.original_question == "What Python version is Server A currently running?"


def test_query_parser_historical_question_with_explicit_date(repo: Repository) -> None:
    """2. Historical question with explicit date parses into AS_OF intent and valid_at."""
    response_payload = {
        "entity": "Server A",
        "attribute": "operating_system",
        "context": {},
        "intent": "as_of",
        "target_date": "2026-03-01T00:00:00Z",
        "reasoning": "Past point-in-time OS query",
    }
    fake_provider = FakeProvider(default_response=json.dumps(response_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    parser = QueryParser(
        llm_client=client,
        attributes=repo.list_attribute_schemas(),
        known_entities=["Server A"],
    )

    query = parser.parse("What OS was Server A running on 2026-03-01?")
    assert query.entity == "Server A"
    assert query.attribute == "operating_system"
    assert query.intent == QueryIntent.AS_OF
    assert query.temporal_intent == TemporalIntent.AS_OF
    assert query.is_historical is True
    assert query.valid_at == datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)


def test_query_parser_why_value_is_believed(repo: Repository) -> None:
    """3. Question asking why a value is believed maps to WHY intent and EXPLANATION type."""
    response_payload = {
        "entity": "Server A",
        "attribute": "operating_system",
        "context": {},
        "intent": "why",
        "target_date": None,
        "reasoning": "User asks for justification and evidence",
    }
    fake_provider = FakeProvider(default_response=json.dumps(response_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    parser = QueryParser(
        llm_client=client,
        attributes=repo.list_attribute_schemas(),
        known_entities=["Server A"],
    )

    query = parser.parse("Why do we believe Server A runs Ubuntu?")
    assert query.entity == "Server A"
    assert query.attribute == "operating_system"
    assert query.intent == QueryIntent.WHY
    assert query.query_type == QueryType.EXPLANATION
    assert query.is_why is True


def test_query_parser_dependent_memories_or_impact(repo: Repository) -> None:
    """4. Question asking about dependencies maps to IMPACT intent."""
    response_payload = {
        "entity": "Server A",
        "attribute": "operating_system",
        "context": {},
        "intent": "impact",
        "target_date": None,
        "reasoning": "User asks what depends on Server A's OS",
    }
    fake_provider = FakeProvider(default_response=json.dumps(response_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    parser = QueryParser(
        llm_client=client,
        attributes=repo.list_attribute_schemas(),
        known_entities=["Server A"],
    )

    query = parser.parse("What depends on Server A's operating system?")
    assert query.entity == "Server A"
    assert query.attribute == "operating_system"
    assert query.intent == QueryIntent.IMPACT
    assert query.query_type == QueryType.DEPENDENCY_IMPACT
    assert query.is_impact is True


def test_query_parser_context_scope(repo: Repository) -> None:
    """5. Question containing an environment context extracts structured context."""
    response_payload = {
        "entity": "Database B",
        "attribute": "database_engine",
        "context": {"env": "prod"},
        "intent": "current",
        "target_date": None,
        "reasoning": "Query constrained to prod environment",
    }
    fake_provider = FakeProvider(default_response=json.dumps(response_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    parser = QueryParser(
        llm_client=client,
        attributes=repo.list_attribute_schemas(),
        known_entities=["Database B"],
    )

    query = parser.parse("What database engine is Database B using in prod?")
    assert query.entity == "Database B"
    assert query.attribute == "database_engine"
    assert query.context == {"env": "prod"}


def test_query_parser_invalid_llm_response_falls_back_deterministically(
    repo: Repository,
) -> None:
    """6. Invalid non-JSON LLM response falls back cleanly to deterministic parsing."""
    fake_provider = FakeProvider(default_response="This is plain text with no JSON!")
    client = LLMClient(provider=fake_provider, repo=repo)
    parser = QueryParser(
        llm_client=client,
        attributes=repo.list_attribute_schemas(),
        known_entities=["Server A"],
    )

    query = parser.parse("What Python version is Server A currently running?")
    assert query.entity == "Server A"
    assert query.attribute == "python_version"
    assert query.intent == QueryIntent.CURRENT
    assert query.original_question == "What Python version is Server A currently running?"


def test_query_parser_provider_failure_falls_back_deterministically(repo: Repository) -> None:
    """7. Provider transport crash falls back safely to deterministic parsing."""
    client = LLMClient(provider=CrashingProvider(), repo=repo)
    parser = QueryParser(
        llm_client=client,
        attributes=repo.list_attribute_schemas(),
        known_entities=["Server A"],
    )

    query = parser.parse("Why do we believe Server A runs Ubuntu?")
    assert query.entity == "Server A"
    assert query.intent == QueryIntent.WHY
    assert query.is_why is True


def test_query_parser_no_llm_configured(repo: Repository) -> None:
    """8. QueryParser works completely without an LLM."""
    parser = QueryParser(
        llm_client=None,
        attributes=repo.list_attribute_schemas(),
        known_entities=["Server A"],
    )

    query = parser.parse("What Python version is Server A running on 2026-06-01?")
    assert query.entity == "Server A"
    assert query.attribute == "python_version"
    assert query.intent == QueryIntent.AS_OF
    assert query.is_historical is True
    assert query.valid_at == datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)


# =============================================================================
# 2. Answer Composer Tests
# =============================================================================


def test_answer_composer_single_memory(repo: Repository) -> None:
    """1. Composes answer grounded in a single retrieved memory."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    rep = service.ingest_claims(msg, [c])
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    compose_payload = {
        "answer_text": f"Server A is running Python 3.12 [M{m_id}].",
        "value": "3.12",
        "values": ["3.12"],
        "referenced_memory_ids": [m_id],
        "explanation": "Reported by sysadmin_alice with confidence 0.95.",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = FakeProvider(default_response=json.dumps(compose_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    assert ans.value == "3.12"
    assert ans.referenced_memory_ids == [m_id]
    assert ans.answer_text == f"Server A's python_version is 3.12 [M{m_id}]."
    assert ans.is_uncertain is False
    assert ans.no_relevant_memory is False
    assert ans.answer_text == f"Server A's python_version is 3.12 [M{m_id}]."
    assert ans.explanation == "Reported by sysadmin_alice with confidence 0.95."


def test_answer_composer_multiple_supporting_memories(repo: Repository) -> None:
    """2. Composes answer for multi-valued attribute referencing multiple memories."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Services")
    c1 = Claim(
        subject="Server A",
        attribute="runs_service",
        value="auth-api",
        extract_confidence=1.0,
        valid_from=t1,
    )
    c2 = Claim(
        subject="Server A",
        attribute="runs_service",
        value="web-gateway",
        extract_confidence=1.0,
        valid_from=t1,
    )
    rep = service.ingest_claims(msg, [c1, c2])
    m1_id = rep.decisions[0].memory_id
    m2_id = rep.decisions[1].memory_id
    assert m1_id is not None and m2_id is not None

    query_res = service.current(entity="Server A", attribute="runs_service", now=t1)
    parsed = ParsedQuery(
        original_question="What services does Server A run?",
        entity="Server A",
        attribute="runs_service",
    )

    compose_payload = {
        "answer_text": f"Server A runs auth-api [M{m1_id}] and web-gateway [M{m2_id}].",
        "value": "auth-api",
        "values": ["auth-api", "web-gateway"],
        "referenced_memory_ids": [m1_id, m2_id],
        "explanation": "Both services are actively coexisting.",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = FakeProvider(default_response=json.dumps(compose_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What services does Server A run?", parsed, query_res)
    assert set(ans.referenced_memory_ids) == {m1_id, m2_id}
    assert "auth-api" in ans.values
    assert "web-gateway" in ans.values
    assert ans.answer_text == (
        f"Server A's runs_service has 2 active values: auth-api [M{m1_id}], web-gateway [M{m2_id}]."
    )


def test_answer_composer_historical_answer(repo: Repository) -> None:
    """3. Historical answer respects temporal boundaries and cites historical memories."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)

    # March: Ubuntu 22.04
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS 22")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Debian 12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m1_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert m1_id is not None

    # June: Upgraded to CentOS 9
    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t2, text="OS upgrade")
    c2 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="CentOS 9",
        extract_confidence=1.0,
        valid_from=t2,
    )
    m2_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert m2_id is not None

    # Query as of April (when Debian 12 was active)
    t_query = datetime(2026, 4, 1, 10, 0, 0, tzinfo=UTC)
    query_res = service.current(
        entity="Server A",
        attribute="operating_system",
        valid_at=t_query,
        now=t2,
    )
    parsed = ParsedQuery(
        original_question="What OS was Server A running on April 1st?",
        entity="Server A",
        attribute="operating_system",
        intent=QueryIntent.AS_OF,
        valid_at=t_query,
    )

    compose_payload = {
        "answer_text": f"On 2026-04-01, Server A was running Debian 12 [M{m1_id}].",
        "value": "Debian 12",
        "values": ["Debian 12"],
        "referenced_memory_ids": [m1_id],
        "explanation": "Valid from 2026-03-01 until 2026-06-01.",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = FakeProvider(default_response=json.dumps(compose_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What OS was Server A running on April 1st?", parsed, query_res)
    assert ans.value == "Debian 12"
    assert ans.referenced_memory_ids == [m1_id]
    assert m2_id not in ans.referenced_memory_ids


def test_answer_composer_no_relevant_memory_found(repo: Repository) -> None:
    """4. When no memory is found, composer explicitly states insufficient evidence."""
    service = MemoryService(db_target=repo)
    t_now = datetime(2026, 6, 1, tzinfo=UTC)
    query_res = service.current(entity="Server A", attribute="python_version", now=t_now)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    composer = AnswerComposer(llm_client=None)  # deterministic or with LLM
    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    assert ans.no_relevant_memory is True
    assert ans.abstained is True
    assert ans.referenced_memory_ids == []
    assert ans.answer_text == "Entity 'Server A' not found."


def test_answer_composer_conflicting_or_disputed_evidence(repo: Repository) -> None:
    """5. When memories are in dispute, composer flags conflict and communicates uncertainty."""
    t1 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)

    # Two equal-reliability sources report conflicting values simultaneously
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Claim Alice")
    c1 = Claim(
        subject="Server A",
        attribute="database_engine",
        value="PostgreSQL 15",
        extract_confidence=0.95,
        valid_from=t1,
    )
    service.ingest_claims(msg1, [c1])

    # Second admin disputes
    repo.insert_source(Source(id="admin_dave", name="Dave", type="admin", reliability=0.95))
    msg2 = IngestMessage(source_id="admin_dave", event_time=t1, text="Claim Dave")
    c2 = Claim(
        subject="Server A",
        attribute="database_engine",
        value="MySQL 8",
        extract_confidence=0.95,
        valid_from=t1,
    )
    service.ingest_claims(msg2, [c2])

    query_res = service.current(entity="Server A", attribute="database_engine", now=t1)
    assert len(query_res.disputes) == 1

    parsed = ParsedQuery(
        original_question="What database engine does Server A run?",
        entity="Server A",
        attribute="database_engine",
    )

    composer = AnswerComposer(llm_client=None)
    ans = composer.compose("What database engine does Server A run?", parsed, query_res)
    assert ans.conflict_flagged is True
    assert ans.is_uncertain is True
    d = query_res.disputes[0]
    expected_ans = (
        f"Active dispute for {d.a.memory.entity_name} {d.a.memory.attribute}: "
        f"Memory M{d.a.memory.id} ('{d.a.memory.value_display}') contradicts "
        f"Memory M{d.b.memory.id} ('{d.b.memory.value_display}')."
    )
    assert ans.answer_text == expected_ans
    assert ans.explanation == "Conflicting claims exist for this information."


def test_answer_composer_memory_marked_needs_review(repo: Repository) -> None:
    """6. Memory marked needs_review flags uncertainty and mentions review reason."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Debian 12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    # Flag for review
    repo.mark_memory_for_review(m_id, "Kernel panic reported")

    query_res = service.current(entity="Server A", attribute="operating_system", now=t1)
    parsed = ParsedQuery(
        original_question="What OS is Server A running?",
        entity="Server A",
        attribute="operating_system",
    )

    composer = AnswerComposer(llm_client=None)
    ans = composer.compose("What OS is Server A running?", parsed, query_res)
    assert ans.needs_review_flagged is True
    assert ans.is_uncertain is True
    assert ans.answer_text == (
        f"Server A's operating_system is Debian 12 [M{m_id}]. "
        f"Note: Memory M{m_id} is flagged for review."
    )
    assert "This memory is flagged for review." in ans.explanation


def test_answer_composer_unsupported_memory_id_is_sanitized(repo: Repository) -> None:
    """7. Hallucinated or unsupported memory IDs are stripped from citations and text."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    # LLM hallucinates non-existent memory ID 9999
    compose_payload = {
        "answer_text": f"Server A runs Python 3.12 [M{m_id}] and also [M9999].",
        "value": "3.12",
        "values": ["3.12"],
        "referenced_memory_ids": [m_id, 9999],
        "explanation": "Grounding test",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = FakeProvider(default_response=json.dumps(compose_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    # 9999 must not be cited, and factual support falls back to deterministic answer
    assert 9999 not in ans.referenced_memory_ids
    assert ans.referenced_memory_ids == [m_id]
    assert "[M9999]" not in ans.answer_text
    assert ans.value == "3.12"
    assert "3.12" in ans.answer_text


def test_answer_composer_unsupported_factual_claim_prevented(repo: Repository) -> None:
    """8. Empty query results prevent hallucinations even if LLM invents facts."""
    service = MemoryService(db_target=repo)
    t_now = datetime(2026, 6, 1, tzinfo=UTC)
    query_res = service.current(entity="Server A", attribute="python_version", now=t_now)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    # Empty context must automatically return abstention without executing LLM call
    composer = AnswerComposer(llm_client=LLMClient(provider=CrashingProvider(), repo=repo))
    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    assert ans.no_relevant_memory is True
    assert ans.abstained is True
    assert ans.referenced_memory_ids == []
    assert ans.answer_text == "Entity 'Server A' not found."


def test_answer_composer_invalid_json_falls_back_deterministically(repo: Repository) -> None:
    """9. Malformed JSON from provider falls back safely to deterministic answer."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    fake_provider = FakeProvider(default_response="Not JSON at all!")
    client = LLMClient(provider=fake_provider, repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    assert ans.value == "3.12"
    assert ans.referenced_memory_ids == [m_id]
    assert ans.answer_text == f"Server A's python_version is 3.12 [M{m_id}]."


def test_answer_composer_provider_failure_falls_back_deterministically(repo: Repository) -> None:
    """10. Network failure falls back safely to deterministic answer without crashing."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    client = LLMClient(provider=CrashingProvider(), repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    assert ans.value == "3.12"
    assert ans.referenced_memory_ids == [m_id]
    assert ans.answer_text == f"Server A's python_version is 3.12 [M{m_id}]."


def test_answer_composer_no_llm_configured(repo: Repository) -> None:
    """11. Works completely without an LLM configured."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    composer = AnswerComposer(llm_client=None)
    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    assert ans.value == "3.12"
    assert ans.referenced_memory_ids == [m_id]
    assert ans.answer_text == f"Server A's python_version is 3.12 [M{m_id}]."


# =============================================================================
# 3. Service Integration Tests
# =============================================================================


def test_service_ask_end_to_end_flow(repo: Repository) -> None:
    """1. Natural-language question flows through parsing, retrieval, and composition."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)

    # Ingest fact
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    # Configure mock provider for parse and compose
    parse_resp = {
        "entity": "Server A",
        "attribute": "python_version",
        "context": {},
        "intent": "current",
        "target_date": None,
        "reasoning": "current python version",
    }
    compose_resp = {
        "answer_text": f"Server A is currently running Python 3.12 [M{m_id}].",
        "value": "3.12",
        "values": ["3.12"],
        "referenced_memory_ids": [m_id],
        "explanation": "Verified in current memory store.",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = SequenceProvider(sequence=[json.dumps(parse_resp), json.dumps(compose_resp)])
    service_with_llm = MemoryService(db_target=repo, llm=fake_provider)

    ask_res = service_with_llm.ask("What Python version is Server A currently running?")
    assert isinstance(ask_res, AskResult)
    assert ask_res.question == "What Python version is Server A currently running?"
    assert ask_res.answer.value == "3.12"
    assert ask_res.answer.cited_memory_ids == [m_id]
    assert len(ask_res.result.items) == 1


def test_service_ask_parser_cannot_override_retrieval_results(repo: Repository) -> None:
    """2. The parser cannot override retrieval results; answers are grounded in retrieved facts."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)

    # Real memory is 3.12
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    # User asks a leading question assuming Python 3.9
    parse_resp = {
        "entity": "Server A",
        "attribute": "python_version",
        "context": {},
        "intent": "current",
        "target_date": None,
        "reasoning": "Lookup python_version",
    }
    compose_resp = {
        "answer_text": f"Server A is actually running Python 3.12 [M{m_id}], not 3.9.",
        "value": "3.12",
        "values": ["3.12"],
        "referenced_memory_ids": [m_id],
        "explanation": "Retrieved factual memory M1.",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = SequenceProvider(sequence=[json.dumps(parse_resp), json.dumps(compose_resp)])
    service_with_llm = MemoryService(db_target=repo, llm=fake_provider)

    ask_res = service_with_llm.ask("Why is Server A running Python 3.9?")
    assert ask_res.answer.value == "3.12"
    assert ask_res.result.items[0].memory.value_display == "3.12"


def test_service_ask_composer_cannot_override_engine_decisions(repo: Repository) -> None:
    """3. Composer cannot unilaterally pick a winner or hide an unresolved dispute."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)

    # Disputed database engine
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="PG 15")
    c1 = Claim(
        subject="Server A",
        attribute="database_engine",
        value="PostgreSQL 15",
        extract_confidence=0.95,
        valid_from=t1,
    )
    service.ingest_claims(msg1, [c1])

    repo.insert_source(Source(id="admin_dave", name="Dave", type="admin", reliability=0.95))
    msg2 = IngestMessage(source_id="admin_dave", event_time=t1, text="MySQL 8")
    c2 = Claim(
        subject="Server A",
        attribute="database_engine",
        value="MySQL 8",
        extract_confidence=0.95,
        valid_from=t1,
    )
    service.ingest_claims(msg2, [c2])

    parse_resp = {
        "entity": "Server A",
        "attribute": "database_engine",
        "context": {},
        "intent": "current",
        "target_date": None,
    }
    # LLM attempts to claim PostgreSQL 15 won without a dispute
    compose_resp = {
        "answer_text": "Server A is definitely running PostgreSQL 15.",
        "value": "PostgreSQL 15",
        "referenced_memory_ids": [1],
        "is_uncertain": False,
    }
    fake_provider = SequenceProvider(sequence=[json.dumps(parse_resp), json.dumps(compose_resp)])
    service_with_llm = MemoryService(db_target=repo, llm=fake_provider)

    ask_res = service_with_llm.ask("What database engine is Server A running?")
    # Engine dispute overrides LLM certainty
    assert ask_res.answer.conflict_flagged is True
    assert (
        "dispute" in ask_res.answer.explanation.lower()
        or "dispute" in ask_res.answer.answer_text.lower()
    )


def test_service_ask_historical_questions_use_temporal_retrieval(repo: Repository) -> None:
    """4. Historical queries correctly respect valid_at half-open interval."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)

    # March: Debian 12
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Debian")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Debian 12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m1_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id

    # June: CentOS 9
    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t2, text="CentOS")
    c2 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="CentOS 9",
        extract_confidence=1.0,
        valid_from=t2,
    )
    m2_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id

    # Ask with explicit as_of April 1st
    t_april = datetime(2026, 4, 1, 10, 0, 0, tzinfo=UTC)
    ask_res = service.ask("What OS was Server A running?", as_of=t_april)
    assert ask_res.answer.value == "Debian 12"
    assert ask_res.answer.cited_memory_ids == [m1_id]
    assert m2_id not in ask_res.answer.cited_memory_ids


def test_service_ask_does_not_mutate_memories(repo: Repository) -> None:
    """5. Answering a question is strictly read-only and causes zero database mutations."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    service.ingest_claims(msg, [c])

    # Record counts before
    mems_before = repo.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    ev_before = repo.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    dec_before = repo.conn.execute("SELECT COUNT(*) FROM decision_log").fetchone()[0]
    edges_before = repo.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    # Ask questions
    service.ask("What Python version is Server A running?")
    service.ask("Why do we believe Server A runs Python 3.12?")
    service.ask("What was Server A running in 2025?")

    # Verify counts after
    mems_after = repo.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    ev_after = repo.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    dec_after = repo.conn.execute("SELECT COUNT(*) FROM decision_log").fetchone()[0]
    edges_after = repo.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    assert mems_before == mems_after
    assert ev_before == ev_after
    assert dec_before == dec_after
    assert edges_before == edges_after


def test_service_ask_no_llm_operation_safe(repo: Repository) -> None:
    """6. Complete ask operation runs deterministically and safely with no LLM."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo, llm=None)  # No LLM configured
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    ask_res = service.ask("What Python version is Server A currently running?")
    assert ask_res.answer.value == "3.12"
    assert ask_res.answer.cited_memory_ids == [m_id]
    assert ask_res.answer.abstained is False


def test_service_ask_why_and_impact_intents(repo: Repository) -> None:
    """7. Verifies explainability and impact questions retrieve supporting graph structures."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo, llm=None)

    # 1. OS memory
    msg1 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Debian 12")
    c1 = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Debian 12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    os_id = service.ingest_claims(msg1, [c1]).decisions[0].memory_id
    assert os_id is not None

    # 2. Python memory dependent on OS
    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
        depends_on=[AttrRef(entity="Server A", attribute="operating_system")],
    )
    py_id = service.ingest_claims(msg2, [c2]).decisions[0].memory_id
    assert py_id is not None

    # Test "why" question
    why_res = service.ask("Why does Server A run Debian 12?")
    assert why_res.result.plan.intent == "why"
    assert len(why_res.result.decisions) >= 1
    assert why_res.answer.cited_memory_ids == [os_id]

    # Test "impact" question
    impact_res = service.ask("What depends on Server A's operating system?")
    assert impact_res.result.plan.intent == "impact"
    assert len(impact_res.result.impact) >= 1
    assert impact_res.result.impact[0].memory.memory.id == py_id


def test_answer_composer_contradictory_factual_answer_falls_back_deterministically(
    repo: Repository,
) -> None:
    """Audit: Valid memory ID with contradictory factual answer falls back to memory truth."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    # LLM outputs valid cited memory ID m_id, but asserts contradictory factual value "3.14"
    compose_payload = {
        "answer_text": f"Server A runs Python 3.14 [M{m_id}].",
        "value": "3.14",
        "values": ["3.14"],
        "referenced_memory_ids": [m_id],
        "explanation": "Contradictory claim",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = FakeProvider(default_response=json.dumps(compose_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    # Contradiction must be rejected: authoritative factual value from memory is preserved
    assert ans.value == "3.12"
    assert "3.12" in ans.answer_text
    assert "3.14" not in ans.answer_text
    assert ans.referenced_memory_ids == [m_id]


def test_answer_composer_unsupported_additional_claim_falls_back_deterministically(
    repo: Repository,
) -> None:
    """Audit: Valid memory ID with unsupported additional factual claim falls back safely."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    # LLM hallucinates an additional ungrounded claim ("AWS us-east-1")
    compose_payload = {
        "answer_text": f"Server A runs Python 3.12 [M{m_id}] and is hosted in AWS us-east-1.",
        "value": "3.12",
        "values": ["3.12", "AWS us-east-1"],
        "referenced_memory_ids": [m_id],
        "explanation": "Extra claim test",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = FakeProvider(default_response=json.dumps(compose_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    # Unsupported claim must trigger safe deterministic compose
    assert ans.value == "3.12"
    assert ans.values == ["3.12"]
    assert "AWS us-east-1" not in ans.answer_text
    assert "3.12" in ans.answer_text


def test_bitemporal_valid_time_versus_belief_time_audit(repo: Repository) -> None:
    """Audit: Distinguish valid_at (when true) vs known_at (when MemoryMesh recorded/believed).

    Scenario:
    - Server A ran Python 3.10 starting March 1 (2026-03-01).
    - It migrated to Python 3.12 on April 1 (2026-04-01).
    - MemoryMesh learned about the migration on April 5 (2026-04-05).
    """
    t_mar1 = datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)
    t_apr1 = datetime(2026, 4, 1, 0, 0, 0, tzinfo=UTC)
    t_apr5 = datetime(2026, 4, 5, 0, 0, 0, tzinfo=UTC)
    t_apr6 = datetime(2026, 4, 6, 0, 0, 0, tzinfo=UTC)

    service = MemoryService(db_target=repo, clock=lambda: t_apr6)

    # 1. On March 1: Server A runs Python 3.10, valid_from = March 1
    msg1 = IngestMessage(
        source_id="sysadmin_alice", event_time=t_mar1, text="Server A running 3.10"
    )
    c1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
        valid_from=t_mar1,
    )
    rep1 = service.ingest_claims(msg1, [c1], recorded_at=t_mar1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # 2. On April 5: MemoryMesh learns Server A migrated to Python 3.12 on April 1
    msg2 = IngestMessage(
        source_id="sysadmin_alice", event_time=t_apr1, text="Server A migrated to 3.12"
    )
    c2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t_apr1,
    )
    rep2 = service.ingest_claims(msg2, [c2], recorded_at=t_apr5)
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    # Query A: "What version was Server A running on March 15?"
    # Must query VALID TIME: valid_at = March 15, known_at is not restricted
    t_mar15 = datetime(2026, 3, 15, 0, 0, 0, tzinfo=UTC)
    res_a = service.ask("What version was Server A running on March 15?")
    assert res_a.result.plan.valid_at == t_mar15
    assert res_a.result.plan.known_at != t_mar15  # Not a belief-time query on March 15
    assert res_a.answer.value == "3.10"
    assert "3.10" in res_a.answer.answer_text
    assert res_a.answer.cited_memory_ids == [m1_id]

    # Query B: "What did MemoryMesh believe on March 15?"
    # Must query KNOWN TIME: known_at = March 15 (excludes migration recorded on April 5)
    res_b = service.ask("What did MemoryMesh believe on March 15?")
    assert res_b.result.plan.known_at == t_mar15
    assert res_b.answer.value == "3.10"
    assert "3.10" in res_b.answer.answer_text
    assert m2_id not in res_b.answer.cited_memory_ids
    assert res_b.answer.cited_memory_ids == [m1_id]

    # Query C: Current state after April 5 returns Python 3.12
    res_c = service.ask("What Python version is Server A running?")
    assert res_c.answer.value == "3.12"
    assert "3.12" in res_c.answer.answer_text
    assert res_c.answer.cited_memory_ids == [m2_id]


def test_answer_composer_correct_value_plus_invented_date_falls_back(repo: Repository) -> None:
    """Audit: Correct memory value plus invented date falls back to deterministic answer."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    # LLM hallucinates an ungrounded date (2021-05-10) not present in context
    compose_payload = {
        "answer_text": f"Server A runs Python 3.12 [M{m_id}] and was deployed on 2021-05-10.",
        "value": "3.12",
        "values": ["3.12"],
        "referenced_memory_ids": [m_id],
        "unsupported_claims": [],  # Intentionally omitted by LLM
        "explanation": "Invented date test",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = FakeProvider(default_response=json.dumps(compose_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    # The complete answer text must NOT retain the invented date
    assert "2021-05-10" not in ans.answer_text
    assert "3.12" in ans.answer_text
    assert ans.value == "3.12"
    assert ans.answer_text == f"Server A's python_version is 3.12 [M{m_id}]."


def test_answer_composer_correct_value_plus_invented_entity_or_location_falls_back(
    repo: Repository,
) -> None:
    """Audit: Correct memory value plus invented location/entity falls back deterministically."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    # LLM hallucinates invented location ("datacenter-dc9") and entity ("Client Corp")
    compose_payload = {
        "answer_text": (
            f"Server A runs Python 3.12 [M{m_id}] located at datacenter-dc9 for Client Corp."
        ),
        "value": "3.12",
        "values": ["3.12"],
        "referenced_memory_ids": [m_id],
        "unsupported_claims": [],
        "explanation": "Invented location test",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = FakeProvider(default_response=json.dumps(compose_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    # The complete answer text must NOT retain the invented entity or location
    assert "datacenter-dc9" not in ans.answer_text
    assert "Client Corp" not in ans.answer_text
    assert "3.12" in ans.answer_text
    assert ans.value == "3.12"
    assert ans.answer_text == f"Server A's python_version is 3.12 [M{m_id}]."


def test_independent_bitemporal_dimensions_audit(repo: Repository) -> None:
    """Audit: Verify independent valid_at and known_at dimensions.

    Required scenario:
    - Recorded at: March 10.
    - Valid from: April 1.
    - Value: Python 3.12.
    Evaluate at March 15:
    - Query 1: What MemoryMesh knew on March 15 about the planned April 1 change retrieves 3.12.
    - Query 2: What was actually valid on March 15 does not return 3.12 as active.
    """
    t_mar10 = datetime(2026, 3, 10, 0, 0, 0, tzinfo=UTC)
    t_mar15 = datetime(2026, 3, 15, 0, 0, 0, tzinfo=UTC)
    t_apr1 = datetime(2026, 4, 1, 0, 0, 0, tzinfo=UTC)

    service = MemoryService(db_target=repo, clock=lambda: t_mar15)

    # Ingest future-dated change recorded on March 10, valid from April 1
    msg = IngestMessage(
        source_id="sysadmin_alice",
        event_time=t_mar10,
        text="Server A scheduled to upgrade to Python 3.12 on April 1",
    )
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t_apr1,
    )
    rep = service.ingest_claims(msg, [c], recorded_at=t_mar10)
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    # Query 1: Natural language query specifying both dimensions:
    # "What did MemoryMesh know on March 15 about the planned April 1 change?"
    # Must preserve known_at = March 15 and valid_at = April 1 independently
    res1 = service.ask("What did MemoryMesh know on March 15 about the planned April 1 change?")
    assert res1.result.plan.valid_at == t_apr1
    assert res1.result.plan.known_at == t_mar15
    assert res1.answer.value == "3.12"
    assert "3.12" in res1.answer.answer_text
    assert res1.answer.cited_memory_ids == [m_id]

    # Also test via explicit parameter overrides on service.ask()
    res1_explicit = service.ask(
        "What was the python version?",
        valid_at=t_apr1,
        known_at=t_mar15,
    )
    assert res1_explicit.result.plan.valid_at == t_apr1
    assert res1_explicit.result.plan.known_at == t_mar15
    assert res1_explicit.answer.value == "3.12"
    assert res1_explicit.answer.cited_memory_ids == [m_id]

    # Query 2: Query asking what version was actually valid on March 15:
    # At March 15, Python 3.12 is NOT valid (valid_from = April 1 > March 15)
    res2 = service.ask("What version was Server A running on March 15?")
    assert res2.result.plan.valid_at == t_mar15
    # Python 3.12 must NOT be returned as active on March 15
    assert res2.answer.value != "3.12"
    assert res2.answer.abstained is True or m_id not in res2.answer.cited_memory_ids


def test_answer_composer_correct_version_plus_invented_security_claim_enforces_deterministic_answer(
    repo: Repository,
) -> None:
    """Audit: Correct version plus invented security claim enforces deterministic answer text."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    # LLM hallucinates an invented security assertion
    compose_payload = {
        "answer_text": (
            f"Server A runs Python 3.12 [M{m_id}] and is the most secure server "
            "in the organization."
        ),
        "value": "3.12",
        "values": ["3.12"],
        "referenced_memory_ids": [m_id],
        "unsupported_claims": [],
        "explanation": "Security rating verified as maximum level.",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = FakeProvider(default_response=json.dumps(compose_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    assert ans.answer_text == f"Server A's python_version is 3.12 [M{m_id}]."
    assert "secure" not in ans.answer_text
    assert "organization" not in ans.answer_text
    assert "secure" not in ans.explanation
    assert ans.value == "3.12"
    assert ans.referenced_memory_ids == [m_id]


def test_answer_composer_correct_version_plus_causal_explanation_enforces_deterministic(
    repo: Repository,
) -> None:
    """Audit: Correct version plus invented causal explanation enforces deterministic text."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    # LLM hallucinates an invented causal explanation
    compose_payload = {
        "answer_text": (
            f"Server A upgraded to Python 3.12 [M{m_id}] because Python 3.10 had a "
            "critical compiler vulnerability."
        ),
        "value": "3.12",
        "values": ["3.12"],
        "referenced_memory_ids": [m_id],
        "unsupported_claims": [],
        "explanation": "Upgraded due to critical compiler vulnerability in the previous release.",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = FakeProvider(default_response=json.dumps(compose_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    assert ans.answer_text == f"Server A's python_version is 3.12 [M{m_id}]."
    assert "compiler" not in ans.answer_text
    assert "vulnerability" not in ans.answer_text
    assert "compiler" not in ans.explanation
    assert "vulnerability" not in ans.explanation
    assert ans.value == "3.12"
    assert ans.referenced_memory_ids == [m_id]


def test_answer_composer_correct_version_plus_operational_status_enforces_deterministic(
    repo: Repository,
) -> None:
    """Audit: Correct version plus invented operational status enforces deterministic text."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    # LLM hallucinates invented operational metrics
    compose_payload = {
        "answer_text": (
            f"Server A is currently running Python 3.12 [M{m_id}] with 99.99% operational "
            "uptime in production."
        ),
        "value": "3.12",
        "values": ["3.12"],
        "referenced_memory_ids": [m_id],
        "unsupported_claims": [],
        "explanation": "High availability production deployment maintains four-nines uptime.",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = FakeProvider(default_response=json.dumps(compose_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    assert ans.answer_text == f"Server A's python_version is 3.12 [M{m_id}]."
    assert "uptime" not in ans.answer_text
    assert "production" not in ans.answer_text
    assert "uptime" not in ans.explanation
    assert ans.value == "3.12"
    assert ans.referenced_memory_ids == [m_id]


def test_answer_composer_correct_version_plus_organization_claim_enforces_deterministic(
    repo: Repository,
) -> None:
    """Audit: Correct version plus organization-level claim enforces deterministic text."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    # LLM hallucinates corporate/organizational policies
    compose_payload = {
        "answer_text": (
            f"Server A runs Python 3.12 [M{m_id}] as mandated across all EMEA engineering teams."
        ),
        "value": "3.12",
        "values": ["3.12"],
        "referenced_memory_ids": [m_id],
        "unsupported_claims": [],
        "explanation": "Mandated corporate standard across all EMEA engineering teams.",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = FakeProvider(default_response=json.dumps(compose_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    assert ans.answer_text == f"Server A's python_version is 3.12 [M{m_id}]."
    assert "emea" not in ans.answer_text
    assert "mandated" not in ans.answer_text
    assert "emea" not in ans.explanation
    assert ans.value == "3.12"
    assert ans.referenced_memory_ids == [m_id]


def test_answer_composer_no_relevant_memories_for_existing_entity(repo: Repository) -> None:
    """When an entity exists but no memories match criteria, composer returns deterministic note."""
    service = MemoryService(db_target=repo)
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Debian 12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    service.ingest_claims(msg, [c])

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    composer = AnswerComposer(llm_client=None)
    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    assert ans.no_relevant_memory is True
    assert ans.abstained is True
    assert ans.referenced_memory_ids == []
    assert ans.answer_text == "No memory valid at the requested time."


def test_explanation_source_and_outage_does_not_permit_invented_causal_explanation(
    repo: Repository,
) -> None:
    """Audit: Evidence with source and outage does not permit an invented causal explanation."""
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="incident_status",
            description="Incident Status",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="server outage")
    c = Claim(
        subject="Server A",
        attribute="incident_status",
        value="server outage",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="incident_status", now=t1)
    parsed = ParsedQuery(
        original_question="What is the incident status of Server A?",
        entity="Server A",
        attribute="incident_status",
    )

    # LLM hallucinates an invented causal explanation: 'sysadmin_alice caused the server outage.'
    compose_payload = {
        "answer_text": f"Server A's incident_status is server outage [M{m_id}].",
        "value": "server outage",
        "values": ["server outage"],
        "referenced_memory_ids": [m_id],
        "unsupported_claims": [],
        "explanation": "sysadmin_alice caused the server outage.",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = FakeProvider(default_response=json.dumps(compose_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What is the incident status of Server A?", parsed, query_res)
    assert ans.answer_text == f"Server A's incident_status is server outage [M{m_id}]."
    # Causation must never be inferred or permitted in the explanation
    assert "caused" not in ans.explanation
    assert ans.explanation == "Reported by sysadmin_alice with confidence 0.95."


def test_explanation_entity_and_status_does_not_permit_invented_causal_relationship(
    repo: Repository,
) -> None:
    """Audit: Evidence with entity and status does not permit an invented causal relationship."""
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="operational_status",
            description="Operational Status",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="degraded")
    c = Claim(
        subject="Server A",
        attribute="operational_status",
        value="degraded",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="operational_status", now=t1)
    parsed = ParsedQuery(
        original_question="What is Server A's operational status?",
        entity="Server A",
        attribute="operational_status",
    )

    # LLM hallucinates an invented causal explanation
    compose_payload = {
        "answer_text": f"Server A's operational_status is degraded [M{m_id}].",
        "value": "degraded",
        "values": ["degraded"],
        "referenced_memory_ids": [m_id],
        "unsupported_claims": [],
        "explanation": "Server A was degraded because of hardware failure and disk corruption.",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = FakeProvider(default_response=json.dumps(compose_payload))
    client = LLMClient(provider=fake_provider, repo=repo)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What is Server A's operational status?", parsed, query_res)
    assert ans.answer_text == f"Server A's operational_status is degraded [M{m_id}]."
    assert "because" not in ans.explanation
    assert "hardware failure" not in ans.explanation
    assert "corruption" not in ans.explanation
    assert ans.explanation == "Reported by sysadmin_alice with confidence 0.95."


def test_explanation_structured_provenance_produces_correct_deterministic_explanation(
    repo: Repository,
) -> None:
    """Audit: Structured provenance fields produce the correct deterministic explanation."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    composer = AnswerComposer(llm_client=None)
    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    assert ans.explanation == "Reported by sysadmin_alice with confidence 0.95."
    assert ans.answer_text == f"Server A's python_version is 3.12 [M{m_id}]."


def test_explanation_missing_provenance_produces_no_invented_explanation() -> None:
    """Audit: Missing provenance fields produce no invented explanation (empty string)."""
    from memorymesh.models import Memory, MemoryStatus, MemoryView, QueryPlan, QueryResult

    m = Memory(
        id=101,
        entity_id=1,
        entity_name="Server A",
        attribute="python_version",
        value_norm="3.12",
        value_display="3.12",
        valid_from=datetime(2026, 3, 1, tzinfo=UTC),
        recorded_at=datetime(2026, 3, 1, tzinfo=UTC),
        status=MemoryStatus.ACTIVE,
        confidence=1.0,
        message_id=1,
        source_id="",
        evidence=[],
    )
    view = MemoryView(
        memory=m,
        effective_confidence=0.0,
        is_stale=False,
        scope_match="exact",
    )
    query_res = QueryResult(plan=QueryPlan(intent=QueryIntent.CURRENT), items=[view])
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    # Even if LLM invents an explanation, it must be omitted because provenance is missing
    compose_payload = {
        "answer_text": "Server A's python_version is 3.12 [M101].",
        "value": "3.12",
        "values": ["3.12"],
        "referenced_memory_ids": [101],
        "unsupported_claims": [],
        "explanation": "Retrieved from internal administrator logs.",
        "is_uncertain": False,
        "no_relevant_memory": False,
    }
    fake_provider = FakeProvider(default_response=json.dumps(compose_payload))
    client = LLMClient(provider=fake_provider)
    composer = AnswerComposer(llm_client=client)

    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    assert ans.explanation == ""
    assert "administrator" not in ans.explanation
    assert ans.answer_text == "Server A's python_version is 3.12 [M101]."


def test_explanation_disputed_and_needs_review_indicators_remain_intact(
    repo: Repository,
) -> None:
    """Audit: Disputed and needs_review indicators remain intact in deterministic explanation."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)

    # 1. Test needs_review
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Debian 12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None
    repo.mark_memory_for_review(m_id, "Kernel panic reported")

    query_res = service.current(entity="Server A", attribute="operating_system", now=t1)
    parsed = ParsedQuery(
        original_question="What OS is Server A running?",
        entity="Server A",
        attribute="operating_system",
    )

    composer = AnswerComposer(llm_client=None)
    ans = composer.compose("What OS is Server A running?", parsed, query_res)
    assert ans.needs_review_flagged is True
    assert ans.is_uncertain is True
    assert "This memory is flagged for review." in ans.explanation
    assert "Kernel panic" not in ans.explanation
    assert "Reported by sysadmin_alice with confidence 0.47." in ans.explanation

    # 2. Test dispute
    repo.insert_source(Source(id="admin_dave", name="Dave", type="admin", reliability=0.95))
    msg2 = IngestMessage(source_id="admin_dave", event_time=t1, text="Claim Dave")
    c2 = Claim(
        subject="Server A",
        attribute="database_engine",
        value="MySQL 8",
        extract_confidence=1.0,
        valid_from=t1,
    )
    service.ingest_claims(msg2, [c2])

    msg3 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Claim Alice")
    c3 = Claim(
        subject="Server A",
        attribute="database_engine",
        value="PostgreSQL 15",
        extract_confidence=1.0,
        valid_from=t1,
    )
    service.ingest_claims(msg3, [c3])

    query_res_disp = service.current(entity="Server A", attribute="database_engine", now=t1)
    ans_disp = composer.compose(
        "What database engine does Server A run?",
        parsed,
        query_res_disp,
    )
    assert ans_disp.conflict_flagged is True
    assert ans_disp.is_uncertain is True
    assert ans_disp.explanation == "Conflicting claims exist for this information."


def test_review_explanation_arbitrary_accusation_does_not_appear(repo: Repository) -> None:
    """Regression 1 & 7: Arbitrary review reason accusation does not appear in either

    answer_text or explanation, while remaining intact in the underlying database record.
    """
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Debian 12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    accusation = "Admin Eve intentionally sabotaged the server to delete audit logs"
    repo.mark_memory_for_review(m_id, accusation)

    query_res = service.current(entity="Server A", attribute="operating_system", now=t1)
    parsed = ParsedQuery(
        original_question="What OS is Server A running?",
        entity="Server A",
        attribute="operating_system",
    )

    composer = AnswerComposer(llm_client=None)
    ans = composer.compose("What OS is Server A running?", parsed, query_res)
    # The accusation must NEVER appear in either user-facing field
    assert "sabotaged" not in ans.explanation
    assert "audit logs" not in ans.explanation
    assert "Admin Eve" not in ans.explanation
    assert "sabotaged" not in ans.answer_text
    assert "audit logs" not in ans.answer_text
    assert "Admin Eve" not in ans.answer_text

    # Fixed review indicators and flags must remain intact
    assert ans.needs_review_flagged is True
    assert ans.is_uncertain is True
    assert "This memory is flagged for review." in ans.explanation
    assert f"Note: Memory M{m_id} is flagged for review." in ans.answer_text

    # But the original review reason remains intact in the database and provenance
    stored_mem = repo.get_memory(m_id)
    assert stored_mem is not None
    assert stored_mem.review_reason == accusation
    explain_res = service.explain(m_id)
    assert explain_res.items[0].memory.review_reason == accusation


def test_dispute_explanation_arbitrary_unsupported_causal_claim_does_not_appear(
    repo: Repository,
) -> None:
    """Regression 2 & 7: Distinctive arbitrary dispute causal claim does not appear in

    either answer_text or explanation, while the original edge_reason remains preserved
    and accessible in the underlying records via repository and service APIs.
    """
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    repo.insert_source(Source(id="admin_dave", name="Dave", type="admin", reliability=0.95))

    msg1 = IngestMessage(source_id="admin_dave", event_time=t1, text="MySQL 8")
    c1 = Claim(
        subject="Server A",
        attribute="database_engine",
        value="MySQL 8",
        extract_confidence=1.0,
        valid_from=t1,
    )
    service.ingest_claims(msg1, [c1])

    msg2 = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="PostgreSQL 15")
    c2 = Claim(
        subject="Server A",
        attribute="database_engine",
        value="PostgreSQL 15",
        extract_confidence=1.0,
        valid_from=t1,
    )
    service.ingest_claims(msg2, [c2])

    distinctive_reason = "Alice sabotaged configuration because Bob caused corruption"
    repo.conn.execute(
        "UPDATE edges SET reason = ? WHERE type = 'contradicts'",
        (distinctive_reason,),
    )
    repo.conn.commit()

    # 1. Verify original dispute edge reason is preserved in repository and service APIs
    disputes = service.disputes(entity="Server A")
    assert len(disputes) == 1
    assert disputes[0].edge_reason == distinctive_reason

    ent = repo.get_entity_by_name("Server A")
    assert ent is not None
    disputed_pairs = repo.find_disputed_pairs(entity_id=int(ent["id"]))
    assert len(disputed_pairs) == 1
    assert disputed_pairs[0][2] == distinctive_reason

    # 2. Verify answer composition never leaks the arbitrary causal dispute reason
    query_res = service.current(entity="Server A", attribute="database_engine", now=t1)
    parsed = ParsedQuery(
        original_question="What database engine does Server A run?",
        entity="Server A",
        attribute="database_engine",
    )

    composer = AnswerComposer(llm_client=None)
    ans = composer.compose("What database engine does Server A run?", parsed, query_res)

    # Distinctive causal claim must NEVER appear in explanation or answer_text
    assert "sabotaged" not in ans.explanation
    assert "because" not in ans.explanation
    assert "caused corruption" not in ans.explanation
    assert "sabotaged" not in ans.answer_text
    assert "because" not in ans.answer_text
    assert "caused corruption" not in ans.answer_text

    # Fixed dispute indicator and boolean flags must remain intact
    assert ans.explanation == "Conflicting claims exist for this information."
    assert ans.conflict_flagged is True
    assert ans.is_uncertain is True

    # 3. Re-verify original edge_reason is still stored and returned unchanged
    re_checked_disputes = service.disputes(entity="Server A")
    assert re_checked_disputes[0].edge_reason == distinctive_reason


def test_fixed_review_indicator_appears_when_needs_review_is_true(repo: Repository) -> None:
    """Regression 3: The fixed review indicator appears when needs_review=True."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="OS")
    c = Claim(
        subject="Server A",
        attribute="operating_system",
        value="Debian 12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None
    repo.mark_memory_for_review(m_id, "Flagged by integrity checker")

    query_res = service.current(entity="Server A", attribute="operating_system", now=t1)
    parsed = ParsedQuery(
        original_question="What OS is Server A running?",
        entity="Server A",
        attribute="operating_system",
    )

    composer = AnswerComposer(llm_client=None)
    ans = composer.compose("What OS is Server A running?", parsed, query_res)
    assert ans.needs_review_flagged is True
    assert "This memory is flagged for review." in ans.explanation


def test_fixed_dispute_indicator_appears_for_active_disputes(repo: Repository) -> None:
    """Regression 4: The fixed dispute indicator appears for active disputes."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    repo.insert_source(Source(id="admin_dave", name="Dave", type="admin", reliability=0.95))

    service.ingest_claims(
        IngestMessage(source_id="admin_dave", event_time=t1, text="MySQL 8"),
        [
            Claim(
                subject="Server A",
                attribute="database_engine",
                value="MySQL 8",
                extract_confidence=1.0,
                valid_from=t1,
            )
        ],
    )
    service.ingest_claims(
        IngestMessage(source_id="sysadmin_alice", event_time=t1, text="PostgreSQL 15"),
        [
            Claim(
                subject="Server A",
                attribute="database_engine",
                value="PostgreSQL 15",
                extract_confidence=1.0,
                valid_from=t1,
            )
        ],
    )

    query_res = service.current(entity="Server A", attribute="database_engine", now=t1)
    parsed = ParsedQuery(
        original_question="What database engine does Server A run?",
        entity="Server A",
        attribute="database_engine",
    )

    composer = AnswerComposer(llm_client=None)
    ans = composer.compose("What database engine does Server A run?", parsed, query_res)
    assert ans.conflict_flagged is True
    assert ans.explanation == "Conflicting claims exist for this information."


def test_source_and_confidence_explanations_still_work(repo: Repository) -> None:
    """Regression 5: Source and confidence explanations still work correctly."""
    t1 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
    service = MemoryService(db_target=repo)
    msg = IngestMessage(source_id="sysadmin_alice", event_time=t1, text="Python 3.12")
    c = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
        valid_from=t1,
    )
    m_id = service.ingest_claims(msg, [c]).decisions[0].memory_id
    assert m_id is not None

    query_res = service.current(entity="Server A", attribute="python_version", now=t1)
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    composer = AnswerComposer(llm_client=None)
    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    assert ans.explanation == "Reported by sysadmin_alice with confidence 0.95."
    assert ans.answer_text == f"Server A's python_version is 3.12 [M{m_id}]."


def test_missing_provenance_does_not_produce_invented_explanation_regression() -> None:
    """Regression 6: Missing provenance fields do not produce invented explanations."""
    from memorymesh.models import Memory, MemoryStatus, MemoryView, QueryPlan, QueryResult

    m = Memory(
        id=202,
        entity_id=1,
        entity_name="Server A",
        attribute="python_version",
        value_norm="3.12",
        value_display="3.12",
        valid_from=datetime(2026, 3, 1, tzinfo=UTC),
        recorded_at=datetime(2026, 3, 1, tzinfo=UTC),
        status=MemoryStatus.ACTIVE,
        confidence=1.0,
        message_id=1,
        source_id="",
        evidence=[],
    )
    view = MemoryView(
        memory=m,
        effective_confidence=0.0,
        is_stale=False,
        scope_match="exact",
    )
    query_res = QueryResult(plan=QueryPlan(intent=QueryIntent.CURRENT), items=[view])
    parsed = ParsedQuery(
        original_question="What Python version is Server A running?",
        entity="Server A",
        attribute="python_version",
    )

    composer = AnswerComposer(llm_client=None)
    ans = composer.compose("What Python version is Server A running?", parsed, query_res)
    assert ans.explanation == ""
    assert ans.answer_text == "Server A's python_version is 3.12 [M202]."
