"""End-to-End Natural-Language Ingestion and Question-Answering Integration Tests (P7.4).

Governed by:
- docs/EVALUATION_AND_DEMO.md (Hero Scenario hero-001)
- docs/SPECIFICATION.md Section 2 (Service API)
- MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P7.4)

Verifies:
1. Full natural-language ingestion via service.ingest():
   - Extraction of structured claims candidates via ClaimExtractor.
   - Pydantic validation of extracted claim models.
   - Normalization of entity, attribute, and values.
   - Deterministic engine authority (LLM never decides REPLACED, REJECTED, DISPUTED, or COEXIST).
   - Atomic persistence of memories, evidence, edges, and decision logs.
   - Idempotent repeated ingestion.
2. Complete Hero Scenario hero-001:
   - Ingests raw messages m1-m8 and verifies all 8 expected decisions:
     {m1: R-INSERT, m2: R-CORROBORATE, m3: R-INSERT, m4: R-INSERT,
      m5: R-REPLACE, m6: R-REJECT, m7: R-DISPUTE, m8: R-INSERT}
   - Verifies natural-language questions q1-q7:
     q1 (current): "What Python version does Server A use?" -> 3.12 (forbidden: 3.10, 3.9)
     q2 (as_of): "What Python version did Server A use in March?" -> 3.10
     q3 (why): "Why do you believe Server A is on Python 3.12?" -> sysadmin_priya, R-REPLACE
     q4 (impact): "What might break because of the Python migration on Server A?"
                  -> dependents: [("Library X", "compatibility_note")], needs_review: True
     q5 (current/conflicts): "Which day does Server A deploy?" -> conflict: True, [fri, mon]
     q6 (current staging): "What Python version does Server A use in staging?" -> 3.11
     q7 (history): "Which Python versions has Server A had?" -> [3.10, 3.12], forbid: 3.9
3. Pipeline edge cases and failure modes:
   - Provider failure fallback in extraction and query parsing.
   - Idempotency on repeated raw message ingestion.
   - Independent bitemporal valid_at vs known_at querying.
   - Abstention on unanswerable questions.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from memorymesh.config import load_app_config
from memorymesh.db import init_db
from memorymesh.llm.providers import LLMProvider
from memorymesh.models import (
    Action,
    IngestMessage,
)
from memorymesh.service import MemoryService


class HeroScenarioScriptedProvider(LLMProvider):
    """Zero-network scripted LLM provider for the canonical Hero Scenario hero-001."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append({"system": system, "user": user, "model": model})

        # =====================================================================
        # 1. Extraction Prompts (prompt_id="extract_claims")
        # =====================================================================
        if "Server A runs Python 3.10." in user:
            return json.dumps(
                {
                    "claims": [
                        {
                            "subject": "Server A",
                            "attribute": "python_version",
                            "value": "3.10",
                            "polarity": "assert",
                            "valid_from": "2026-01-10T09:00:00Z",
                            "extract_confidence": 0.95,
                            "span": "Server A runs Python 3.10.",
                        }
                    ]
                }
            )

        if "Runtime for Server A: Python 3.10" in user:
            return json.dumps(
                {
                    "claims": [
                        {
                            "subject": "Server A",
                            "attribute": "python_version",
                            "value": "3.10",
                            "polarity": "assert",
                            "valid_from": "2026-02-02T11:30:00Z",
                            "extract_confidence": 0.90,
                            "span": "Python 3.10",
                        }
                    ]
                }
            )

        if "Server A deploys happen on Fridays." in user:
            return json.dumps(
                {
                    "claims": [
                        {
                            "subject": "Server A",
                            "attribute": "deploy_day",
                            "value": "friday",
                            "polarity": "assert",
                            "valid_from": "2026-02-10T10:00:00Z",
                            "extract_confidence": 0.95,
                            "span": "Fridays",
                        }
                    ]
                }
            )

        if "Library X is deployed on Server A" in user:
            return json.dumps(
                {
                    "claims": [
                        {
                            "subject": "Library X",
                            "attribute": "compatibility_note",
                            "value": "works fine with Python 3.10 runtime on Server A",
                            "polarity": "assert",
                            "valid_from": "2026-03-05T14:00:00Z",
                            "depends_on": [
                                {
                                    "entity": "Server A",
                                    "attribute": "python_version",
                                }
                            ],
                            "extract_confidence": 0.95,
                            "span": "works fine with its Python 3.10 runtime.",
                        }
                    ]
                }
            )

        if "We migrated Server A to Python 3.12 today." in user:
            return json.dumps(
                {
                    "claims": [
                        {
                            "subject": "Server A",
                            "attribute": "python_version",
                            "value": "3.12",
                            "previous_value": "3.10",
                            "polarity": "assert",
                            "valid_from": "2026-06-01T08:00:00Z",
                            "extract_confidence": 0.95,
                            "span": "migrated Server A to Python 3.12 today.",
                        }
                    ]
                }
            )

        if "I think Server A is still on Python 3.9?" in user:
            return json.dumps(
                {
                    "claims": [
                        {
                            "subject": "Server A",
                            "attribute": "python_version",
                            "value": "3.9",
                            "polarity": "assert",
                            "valid_from": "2026-06-03T16:20:00Z",
                            "extract_confidence": 0.70,
                            "span": "Python 3.9",
                        }
                    ]
                }
            )

        if "FYI deploys for Server A are on Mondays now." in user:
            return json.dumps(
                {
                    "claims": [
                        {
                            "subject": "Server A",
                            "attribute": "deploy_day",
                            "value": "monday",
                            "polarity": "assert",
                            "valid_from": "2026-06-15T10:10:00Z",
                            "extract_confidence": 0.90,
                            "span": "Mondays",
                        }
                    ]
                }
            )

        if "In staging, Server A still runs Python 3.11." in user:
            return json.dumps(
                {
                    "claims": [
                        {
                            "subject": "Server A",
                            "attribute": "python_version",
                            "value": "3.11",
                            "context": {"env": "staging"},
                            "polarity": "assert",
                            "valid_from": "2026-06-20T13:00:00Z",
                            "extract_confidence": 0.90,
                            "span": "Python 3.11",
                        }
                    ]
                }
            )

        # =====================================================================
        # 2. Query Parsing Prompts (prompt_id="parse_query")
        # =====================================================================
        if "What Python version does Server A use?" in user:
            return json.dumps(
                {
                    "entity": "Server A",
                    "attribute": "python_version",
                    "intent": "current",
                    "temporal_intent": "current",
                    "query_type": "fact_lookup",
                    "context": {},
                    "is_historical": False,
                }
            )

        if "What Python version did Server A use in March?" in user:
            return json.dumps(
                {
                    "entity": "Server A",
                    "attribute": "python_version",
                    "intent": "as_of",
                    "temporal_intent": "as_of",
                    "query_type": "fact_lookup",
                    "valid_at": "2026-03-15T00:00:00Z",
                    "date_str": "March 2026",
                    "is_historical": True,
                }
            )

        if "Why do you believe Server A is on Python 3.12?" in user:
            return json.dumps(
                {
                    "entity": "Server A",
                    "attribute": "python_version",
                    "intent": "why",
                    "temporal_intent": "current",
                    "query_type": "explanation",
                    "is_why": True,
                }
            )

        if "What might break because of the Python migration on Server A?" in user:
            return json.dumps(
                {
                    "entity": "Server A",
                    "attribute": "python_version",
                    "intent": "impact",
                    "temporal_intent": "current",
                    "query_type": "dependency_impact",
                    "is_impact": True,
                }
            )

        if "What Python version does Server Z use?" in user:
            return json.dumps(
                {
                    "entity": "Server Z",
                    "attribute": "python_version",
                    "intent": "current",
                    "temporal_intent": "current",
                    "query_type": "fact_lookup",
                    "context": {},
                    "is_historical": False,
                }
            )

        if "Which day does Server A deploy?" in user:
            return json.dumps(
                {
                    "entity": "Server A",
                    "attribute": "deploy_day",
                    "intent": "current",
                    "temporal_intent": "current",
                    "query_type": "fact_lookup",
                }
            )

        if "What Python version does Server A use in staging?" in user:
            return json.dumps(
                {
                    "entity": "Server A",
                    "attribute": "python_version",
                    "context": {"env": "staging"},
                    "intent": "current",
                    "temporal_intent": "current",
                    "query_type": "fact_lookup",
                }
            )

        if "Which Python versions has Server A had?" in user:
            return json.dumps(
                {
                    "entity": "Server A",
                    "attribute": "python_version",
                    "intent": "history",
                    "temporal_intent": "history",
                    "query_type": "timeline_history",
                    "is_historical": True,
                }
            )

        # =====================================================================
        # 3. Answer Composition Prompts (prompt_id="compose_answer")
        # =====================================================================
        # Generic composition response honoring deterministic grounding
        return json.dumps(
            {
                "answer_text": "Processed response.",
                "value": None,
                "values": [],
                "referenced_memory_ids": [],
                "explanation": "",
                "unsupported_claims": [],
                "no_relevant_memory": False,
            }
        )


@pytest.fixture
def nl_service() -> MemoryService:
    """Fixture providing MemoryService backed by isolated in-memory DB and scripted provider."""
    conn = init_db(":memory:")
    cfg = load_app_config()
    provider = HeroScenarioScriptedProvider()
    clock_ref = datetime(2026, 6, 25, 9, 0, 0, tzinfo=UTC)
    service = MemoryService(
        db_target=conn,
        cfg=cfg,
        llm=provider,
        clock=lambda: clock_ref,
    )
    return service


def test_hero_001_end_to_end_natural_language_scenario(nl_service: MemoryService) -> None:
    """Test full hero-001 scenario through natural-language ingest and ask APIs."""
    service = nl_service

    # -------------------------------------------------------------------------
    # Part 1: Ingest all 8 raw natural-language messages (m1 to m8)
    # -------------------------------------------------------------------------
    raw_messages = [
        # m1
        IngestMessage(
            source_id="sysadmin_priya",
            event_time=datetime(2026, 1, 10, 9, 0, tzinfo=UTC),
            text="Server A runs Python 3.10.",
        ),
        # m2
        IngestMessage(
            source_id="wiki",
            event_time=datetime(2026, 2, 2, 11, 30, tzinfo=UTC),
            text="Runtime for Server A: Python 3.10 (see infra page).",
        ),
        # m3
        IngestMessage(
            source_id="sysadmin_priya",
            event_time=datetime(2026, 2, 10, 10, 0, tzinfo=UTC),
            text="Server A deploys happen on Fridays.",
        ),
        # m4
        IngestMessage(
            source_id="sysadmin_priya",
            event_time=datetime(2026, 3, 5, 14, 0, tzinfo=UTC),
            text="Library X is deployed on Server A and works fine with its Python 3.10 runtime.",
        ),
        # m5
        IngestMessage(
            source_id="sysadmin_priya",
            event_time=datetime(2026, 6, 1, 8, 0, tzinfo=UTC),
            text="We migrated Server A to Python 3.12 today.",
        ),
        # m6
        IngestMessage(
            source_id="intern_chat",
            event_time=datetime(2026, 6, 3, 16, 20, tzinfo=UTC),
            text="I think Server A is still on Python 3.9?",
        ),
        # m7
        IngestMessage(
            source_id="dev_chat",
            event_time=datetime(2026, 6, 15, 10, 10, tzinfo=UTC),
            text="FYI deploys for Server A are on Mondays now.",
        ),
        # m8
        IngestMessage(
            source_id="dev_chat",
            event_time=datetime(2026, 6, 20, 13, 0, tzinfo=UTC),
            text="In staging, Server A still runs Python 3.11.",
        ),
    ]

    expected_rules = [
        ("m1", "R-INSERT", Action.INSERT),
        ("m2", "R-CORROBORATE", Action.CORROBORATE),
        ("m3", "R-INSERT", Action.INSERT),
        ("m4", "R-INSERT", Action.INSERT),
        ("m5", "R-REPLACE", Action.REPLACE),
        ("m6", "R-REJECT", Action.REJECT),
        ("m7", "R-DISPUTE", Action.DISPUTE),
        ("m8", "R-INSERT", Action.INSERT),
    ]

    reports = []
    for (m_id, exp_rule, exp_action), msg in zip(expected_rules, raw_messages, strict=True):
        report = service.ingest(msg)
        reports.append((m_id, report))

        assert not report.duplicate, f"{m_id} unexpectedly marked duplicate"
        assert report.claims_extracted == 1, f"{m_id} expected 1 extracted claim"
        assert len(report.decisions) >= 1, f"{m_id} expected at least 1 decision"

        dec = report.decisions[0]
        assert dec.rule_id == exp_rule, (
            f"{m_id}: expected rule {exp_rule}, got {dec.rule_id} ({dec.explanation})"
        )
        assert dec.action == exp_action, f"{m_id}: expected action {exp_action}, got {dec.action}"

        if m_id == "m5":
            assert len(report.decisions) == 2, "m5 should have R-REPLACE and downstream R-PROPAGATE"
            assert report.decisions[1].rule_id == "R-PROPAGATE"
            assert report.decisions[1].action == Action.PROPAGATE

    # -------------------------------------------------------------------------
    # Part 2: Verify all 7 natural language questions (q1 to q7)
    # -------------------------------------------------------------------------
    now_ref = datetime(2026, 6, 25, 9, 0, tzinfo=UTC)

    # q1: Current Python version -> 3.12 (forbidden: 3.10, 3.9)
    q1_res = service.ask("What Python version does Server A use?", reference_time=now_ref)
    assert q1_res.answer.value == "3.12"
    assert "3.12" in q1_res.answer.answer_text
    assert "3.10" not in q1_res.answer.answer_text
    assert "3.9" not in q1_res.answer.answer_text
    assert len(q1_res.answer.cited_memory_ids) >= 1

    # q2: Historical Python version as of March -> 3.10 (forbidden: 3.12, 3.9)
    q2_res = service.ask("What Python version did Server A use in March?", reference_time=now_ref)
    assert q2_res.answer.value == "3.10"
    assert "3.10" in q2_res.answer.answer_text
    assert "3.12" not in q2_res.answer.answer_text
    assert "3.9" not in q2_res.answer.answer_text
    assert len(q2_res.answer.cited_memory_ids) >= 1

    # q3: Explanation for Python 3.12 -> cited_source: sysadmin_priya, rule_id: R-REPLACE
    q3_res = service.ask("Why do you believe Server A is on Python 3.12?", reference_time=now_ref)
    assert q3_res.result.decisions, "Explanation should include decision history"
    assert any(d.rule_id == "R-REPLACE" for d in q3_res.result.decisions)
    assert "sysadmin_priya" in q3_res.result.items[0].source_names
    assert (
        "sysadmin_priya" in q3_res.answer.explanation
        or "sysadmin_priya" in q3_res.result.items[0].source_names
    )
    assert "R-REPLACE" in q3_res.answer.explanation or any(
        d.rule_id == "R-REPLACE" for d in q3_res.result.decisions
    )

    # q4: Impact analysis -> Library X compatibility_note, needs_review_flagged: True
    q4_res = service.ask(
        "What might break because of the Python migration on Server A?", reference_time=now_ref
    )
    assert q4_res.answer.needs_review_flagged is True
    assert len(q4_res.result.impact) >= 1
    dep_mem = q4_res.result.impact[0].memory.memory
    assert dep_mem.entity_name == "Library X"
    assert dep_mem.attribute == "compatibility_note"
    assert dep_mem.needs_review is True
    assert "Library X" in q4_res.answer.answer_text

    # q5: Deploys conflict -> conflict_flagged: True, values: ["friday", "monday"]
    q5_res = service.ask("Which day does Server A deploy?", reference_time=now_ref)
    assert q5_res.answer.conflict_flagged is True
    assert "friday" in [v.lower() for v in q5_res.answer.values]
    assert "monday" in [v.lower() for v in q5_res.answer.values]
    assert len(q5_res.result.disputes) >= 1

    # q6: Python version in staging -> 3.11
    q6_res = service.ask(
        "What Python version does Server A use in staging?", reference_time=now_ref
    )
    assert q6_res.answer.value == "3.11"
    assert "3.11" in q6_res.answer.answer_text

    # q7: History of Python versions -> values: ["3.10", "3.12"], forbidden: ["3.9"]
    q7_res = service.ask("Which Python versions has Server A had?", reference_time=now_ref)
    norm_values = [v.strip() for v in q7_res.answer.values]
    assert "3.10" in norm_values
    assert "3.12" in norm_values
    assert "3.9" not in norm_values
    assert "3.9" not in q7_res.answer.answer_text

    # -------------------------------------------------------------------------
    # Part 3: Verify system invariants across persisted records
    # -------------------------------------------------------------------------
    inv_report = service.check_invariants()
    assert inv_report.violations == []


def test_nl_ingest_idempotency_and_no_duplicate_processing(
    nl_service: MemoryService,
) -> None:
    """Verifies that re-ingesting identical natural language messages is completely idempotent."""
    service = nl_service

    msg = IngestMessage(
        source_id="sysadmin_priya",
        event_time=datetime(2026, 1, 10, 9, 0, tzinfo=UTC),
        text="Server A runs Python 3.10.",
    )

    # 1. First ingest
    rep1 = service.ingest(msg)
    assert rep1.duplicate is False
    assert rep1.claims_extracted == 1
    assert len(rep1.decisions) == 1

    # Record initial counts
    counts_before = service.repo.get_record_counts()

    # 2. Second ingest of exact same message
    rep2 = service.ingest(msg)
    assert rep2.duplicate is True
    assert rep2.claims_extracted == 0
    assert rep2.decisions == []
    assert rep2.message_id == rep1.message_id

    # Verify database state was completely unmutated
    counts_after = service.repo.get_record_counts()
    assert counts_before == counts_after


def test_nl_ingest_extraction_reporting_and_failure_modes() -> None:
    """Verifies explicit failure/success reporting in IngestReport across extraction modes.

    Tests:
    1. Successful extraction with claims -> success=True, error=None, claims_extracted > 0.
    2. Successful extraction with zero claims -> success=True, error=None, claims_extracted=0.
    3. Provider/extraction failure with raise_on_error=False -> success=False, error is set.
    4. Provider/extraction failure with raise_on_error=True -> raises exception.
    5. Confirmation that failed extraction does not create memories or decision records.
    """
    conn = init_db(":memory:")
    cfg = load_app_config()

    # 1. Successful extraction with claims
    valid_payload = {
        "claims": [
            {
                "subject": "Server A",
                "attribute": "python_version",
                "value": "3.12",
                "polarity": "assert",
                "valid_from": "2026-06-01T10:00:00Z",
                "extract_confidence": 0.95,
                "span": "Server A runs Python 3.12",
            }
        ]
    }

    class SuccessWithClaimsProvider(LLMProvider):
        def complete(self, *args: Any, **kwargs: Any) -> str:
            return json.dumps(valid_payload)

    svc1 = MemoryService(db_target=conn, cfg=cfg, llm=SuccessWithClaimsProvider())
    msg1 = IngestMessage(
        source_id="sysadmin_priya",
        event_time=datetime(2026, 6, 1, 10, 0, tzinfo=UTC),
        text="Server A runs Python 3.12.",
    )
    rep1 = svc1.ingest(msg1)
    assert rep1.success is True
    assert rep1.error is None
    assert rep1.claims_extracted == 1
    assert len(rep1.decisions) == 1
    assert len(svc1.repo.get_all_memories()) == 1

    # 2. Successful extraction with zero claims
    empty_payload: dict[str, list[Any]] = {"claims": []}

    class SuccessZeroClaimsProvider(LLMProvider):
        def complete(self, *args: Any, **kwargs: Any) -> str:
            return json.dumps(empty_payload)

    conn2 = init_db(":memory:")
    svc2 = MemoryService(db_target=conn2, cfg=cfg, llm=SuccessZeroClaimsProvider())
    msg2 = IngestMessage(
        source_id="sysadmin_priya",
        event_time=datetime(2026, 6, 2, 10, 0, tzinfo=UTC),
        text="I wonder how Server A is doing today?",
    )
    rep2 = svc2.ingest(msg2)
    assert rep2.success is True
    assert rep2.error is None
    assert rep2.claims_extracted == 0
    assert rep2.decisions == []
    # Legitimate 0-claim message was stored in messages table, but 0 memories/decisions created
    assert rep2.message_id > 0
    assert len(svc2.repo.get_all_memories()) == 0
    assert len(svc2.repo.get_all_decision_logs()) == 0

    # 3. Provider/extraction failure with raise_on_error=False
    class FailingProvider(LLMProvider):
        def complete(self, *args: Any, **kwargs: Any) -> str:
            raise ConnectionError("Simulated LLM connection timeout")

    conn3 = init_db(":memory:")
    svc3 = MemoryService(db_target=conn3, cfg=cfg, llm=FailingProvider())
    msg3 = IngestMessage(
        source_id="sysadmin_priya",
        event_time=datetime(2026, 6, 3, 10, 0, tzinfo=UTC),
        text="Server A runs Python 3.12.",
    )
    rep3 = svc3.ingest(msg3, raise_on_error=False)
    assert rep3.success is False
    assert rep3.error is not None
    assert "Extraction failed" in rep3.error
    assert rep3.claims_extracted == 0
    assert rep3.decisions == []
    assert rep3.message_id == 0

    # 5. Confirmation that failed extraction does NOT create messages, memories, or decision records
    assert len(svc3.repo.get_all_memories()) == 0
    assert len(svc3.repo.get_all_decision_logs()) == 0
    assert svc3.repo.get_record_counts()["messages"] == 0

    # 4. Provider/extraction failure with raise_on_error=True
    with pytest.raises(ConnectionError):
        svc3.ingest(msg3, raise_on_error=True)
    # Still zero records
    assert len(svc3.repo.get_all_memories()) == 0
    assert len(svc3.repo.get_all_decision_logs()) == 0


def test_q5_deployment_conflict_deterministic_regression() -> None:
    """Verifies that 'Which day does Server A deploy?' resolves deterministically with conflict.

    Expected behavior:
    1. Query is parsed with QueryIntent.CURRENT (facts for current deploy day).
    2. Both Friday and Monday values are returned.
    3. The conflict is identified by existing deterministic memory/dispute state.
    4. conflict_flagged=True.
    5. Correct memory citations (both conflicting memory IDs) are included.
    6. The LLM does not independently invent or decide that a conflict exists.
    """
    conn = init_db(":memory:")
    cfg = load_app_config()

    # Create service with no LLM to prove pure deterministic authority
    service = MemoryService(db_target=conn, cfg=cfg, llm=None)

    # Ingest Friday deploy (sysadmin_priya)
    t1 = datetime(2026, 2, 10, 10, 0, tzinfo=UTC)
    m3_msg = IngestMessage(
        source_id="sysadmin_priya",
        event_time=t1,
        text="Server A deploys happen on Fridays.",
    )
    from memorymesh.models import Claim

    rep_m3 = service.ingest_claims(
        m3_msg,
        [
            Claim(
                subject="Server A",
                attribute="deploy_day",
                value="friday",
                valid_from=t1,
                extract_confidence=0.95,
            )
        ],
    )
    m3_id = rep_m3.decisions[0].memory_id
    assert rep_m3.decisions[0].rule_id == "R-INSERT"

    # Ingest Monday deploy (dev_chat -> R-DISPUTE)
    t2 = datetime(2026, 6, 15, 10, 10, tzinfo=UTC)
    m7_msg = IngestMessage(
        source_id="dev_chat",
        event_time=t2,
        text="FYI deploys for Server A are on Mondays now.",
    )
    rep_m7 = service.ingest_claims(
        m7_msg,
        [
            Claim(
                subject="Server A",
                attribute="deploy_day",
                value="monday",
                valid_from=t2,
                extract_confidence=0.90,
            )
        ],
    )
    m7_id = rep_m7.decisions[0].memory_id
    assert rep_m7.decisions[0].rule_id == "R-DISPUTE"

    # Ask the question: "Which day does Server A deploy?"
    now_ref = datetime(2026, 6, 25, 9, 0, tzinfo=UTC)
    ask_res = service.ask("Which day does Server A deploy?", reference_time=now_ref)

    # 1. Clarify actual parsed QueryIntent
    assert ask_res.result.plan.intent in ("current", "CURRENT")
    assert ask_res.result.plan.entity == "Server A"
    assert ask_res.result.plan.attribute == "deploy_day"

    # 2. Both Friday and Monday returned in values
    returned_values = [v.lower().strip() for v in ask_res.answer.values]
    assert "friday" in returned_values
    assert "monday" in returned_values

    # 3. Conflict is identified by deterministic repository dispute state
    assert len(ask_res.result.disputes) >= 1
    assert ask_res.answer.conflict_flagged is True

    # 4. Correct memory citations included
    assert m3_id in ask_res.answer.cited_memory_ids
    assert m7_id in ask_res.answer.cited_memory_ids

    # 5. Deterministic explanation
    assert "Conflicting claims exist for this information." in ask_res.answer.explanation


def test_nl_ask_independent_bitemporal_evaluation(nl_service: MemoryService) -> None:
    """Verifies independent evaluation of valid_at and known_at in natural language questions."""
    service = nl_service

    # Setup late-arriving knowledge:
    # Event happened at t_event (2026-03-01), but system learned it at t_known (2026-05-01)
    t_event = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t_known = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)

    msg = IngestMessage(
        source_id="sysadmin_priya",
        event_time=t_event,
        text="Server A runs Python 3.10.",
    )
    service.ingest(msg, recorded_at=t_known)

    # 1. Belief-time before recorded_at (as known on 2026-04-01): system did not know it yet
    t_before_known = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    res_unknown = service.ask(
        "What Python version does Server A use?",
        known_at=t_before_known,
    )
    assert res_unknown.answer.abstained is True

    # 2. As known after recorded_at (as known on 2026-05-15): system knows the fact
    t_after_known = datetime(2026, 5, 15, 10, 0, tzinfo=UTC)
    res_known = service.ask(
        "What Python version does Server A use?",
        known_at=t_after_known,
    )
    assert res_known.answer.value == "3.10"
    assert res_known.answer.abstained is False


def test_nl_ask_abstention_on_unanswerable_questions(nl_service: MemoryService) -> None:
    """Verifies abstention when an asked question has no relevant memory in the database."""
    service = nl_service

    # Ingest Server A fact
    service.ingest(
        IngestMessage(
            source_id="sysadmin_priya",
            event_time=datetime(2026, 1, 10, 9, 0, tzinfo=UTC),
            text="Server A runs Python 3.10.",
        )
    )

    # Ask about non-existent entity Server Z
    res = service.ask("What Python version does Server Z use?")
    assert res.answer.abstained is True
    assert res.answer.value is None
    assert any(
        phrase in res.answer.answer_text
        for phrase in ("Server Z", "No target entity", "No relevant memory")
    )


def test_nl_ingest_multi_claim_message() -> None:
    """Verifies atomic ingestion of multi-claim natural-language messages."""
    conn = init_db(":memory:")
    cfg = load_app_config()

    multi_claim_payload = {
        "claims": [
            {
                "subject": "Server A",
                "attribute": "python_version",
                "value": "3.12",
                "polarity": "assert",
                "valid_from": "2026-06-01T10:00:00Z",
                "extract_confidence": 0.95,
                "span": "Server A runs Python 3.12",
            },
            {
                "subject": "Server A",
                "attribute": "deploy_day",
                "value": "friday",
                "polarity": "assert",
                "valid_from": "2026-06-01T10:00:00Z",
                "extract_confidence": 0.90,
                "span": "deploys happen on Fridays",
            },
        ]
    }

    class MultiClaimProvider(LLMProvider):
        def complete(self, *args: Any, **kwargs: Any) -> str:
            return json.dumps(multi_claim_payload)

    service = MemoryService(
        db_target=conn,
        cfg=cfg,
        llm=MultiClaimProvider(),
    )

    msg = IngestMessage(
        source_id="sysadmin_priya",
        event_time=datetime(2026, 6, 1, 10, 0, tzinfo=UTC),
        text="Server A runs Python 3.12 and deploys happen on Fridays.",
    )

    report = service.ingest(msg)
    assert report.duplicate is False
    assert report.claims_extracted == 2
    assert len(report.decisions) == 2
    assert report.decisions[0].rule_id == "R-INSERT"
    assert report.decisions[1].rule_id == "R-INSERT"

    # Both memories should now be active
    py_mem = service.get_current("Server A", "python_version", now=datetime(2026, 6, 2, tzinfo=UTC))
    assert py_mem is not None
    assert py_mem.value_norm == "3.12"

    deploy_mem = service.get_current("Server A", "deploy_day", now=datetime(2026, 6, 2, tzinfo=UTC))
    assert deploy_mem is not None
    assert deploy_mem.value_norm == "friday"


def test_nl_ingest_value_normalization_and_skip() -> None:
    """Verifies that un-normalizable values trigger R-SKIP-VALUE without creating corrupt memory."""
    conn = init_db(":memory:")
    cfg = load_app_config()

    invalid_val_payload = {
        "claims": [
            {
                "subject": "Server A",
                "attribute": "deploy_day",
                "value": "on an arbitrary future someday",  # Invalid enum value
                "polarity": "assert",
                "valid_from": "2026-06-01T10:00:00Z",
                "extract_confidence": 0.95,
                "span": "on an arbitrary future someday",
            }
        ]
    }

    class InvalidValProvider(LLMProvider):
        def complete(self, *args: Any, **kwargs: Any) -> str:
            return json.dumps(invalid_val_payload)

    service = MemoryService(db_target=conn, cfg=cfg, llm=InvalidValProvider())

    msg = IngestMessage(
        source_id="sysadmin_priya",
        event_time=datetime(2026, 6, 1, 10, 0, tzinfo=UTC),
        text="Server A deploys happen on an arbitrary future someday.",
    )

    report = service.ingest(msg)
    assert report.claims_extracted == 1
    assert len(report.decisions) == 1
    assert report.decisions[0].rule_id == "R-SKIP-VALUE"
    assert report.decisions[0].action == Action.SKIP

    # Verify no memory was created for deploy_day
    all_mems = service.repo.get_all_memories()
    assert len(all_mems) == 0


def test_hero_001_pure_deterministic_fallback_no_llm(nl_service: MemoryService) -> None:
    """Verifies that question answering on hero scenario works deterministically without LLM."""
    service = nl_service

    # Ingest the 8 hero scenario messages
    t_m1 = datetime(2026, 1, 10, 9, 0, tzinfo=UTC)
    service.ingest(
        IngestMessage(
            source_id="sysadmin_priya", event_time=t_m1, text="Server A runs Python 3.10."
        )
    )
    t_m2 = datetime(2026, 2, 2, 11, 30, tzinfo=UTC)
    service.ingest(
        IngestMessage(
            source_id="wiki",
            event_time=t_m2,
            text="Runtime for Server A: Python 3.10 (see infra page).",
        )
    )
    t_m3 = datetime(2026, 2, 10, 10, 0, tzinfo=UTC)
    service.ingest(
        IngestMessage(
            source_id="sysadmin_priya", event_time=t_m3, text="Server A deploys happen on Fridays."
        )
    )
    t_m4 = datetime(2026, 3, 5, 14, 0, tzinfo=UTC)
    service.ingest(
        IngestMessage(
            source_id="sysadmin_priya",
            event_time=t_m4,
            text="Library X is deployed on Server A and works fine with its Python 3.10 runtime.",
        )
    )
    t_m5 = datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
    service.ingest(
        IngestMessage(
            source_id="sysadmin_priya",
            event_time=t_m5,
            text="We migrated Server A to Python 3.12 today.",
        )
    )
    t_m6 = datetime(2026, 6, 3, 16, 20, tzinfo=UTC)
    service.ingest(
        IngestMessage(
            source_id="intern_chat",
            event_time=t_m6,
            text="I think Server A is still on Python 3.9?",
        )
    )
    t_m7 = datetime(2026, 6, 15, 10, 10, tzinfo=UTC)
    service.ingest(
        IngestMessage(
            source_id="dev_chat",
            event_time=t_m7,
            text="FYI deploys for Server A are on Mondays now.",
        )
    )
    t_m8 = datetime(2026, 6, 20, 13, 0, tzinfo=UTC)
    service.ingest(
        IngestMessage(
            source_id="dev_chat",
            event_time=t_m8,
            text="In staging, Server A still runs Python 3.11.",
        )
    )

    # Create a no-LLM service instance pointing to the exact same database
    no_llm_service = MemoryService(
        db_target=service.repo.conn,
        cfg=service.cfg,
        llm=None,  # No LLM configured!
        clock=service.clock,
    )

    now_ref = datetime(2026, 6, 25, 9, 0, tzinfo=UTC)

    # q1: Current Python version -> 3.12
    q1 = no_llm_service.ask("What Python version does Server A use?", reference_time=now_ref)
    assert q1.answer.value == "3.12"
    assert "3.12" in q1.answer.answer_text

    # q2: Historical Python version as of March -> 3.10
    q2 = no_llm_service.ask(
        "What Python version did Server A use in March?", reference_time=now_ref
    )
    assert q2.answer.value == "3.10"
    assert "3.10" in q2.answer.answer_text

    # q3: Explanation for Python 3.12 -> rule_id: R-REPLACE
    q3 = no_llm_service.ask(
        "Why do you believe Server A is on Python 3.12?", reference_time=now_ref
    )
    assert q3.result.decisions
    assert any(d.rule_id == "R-REPLACE" for d in q3.result.decisions)

    # q4: Impact -> Library X, needs_review_flagged=True
    q4 = no_llm_service.ask(
        "What might break because of the Python migration on Server A?", reference_time=now_ref
    )
    assert q4.answer.needs_review_flagged is True
    assert len(q4.result.impact) >= 1
    assert q4.result.impact[0].memory.memory.entity_name == "Library X"

    # q5: Deploys conflict -> conflict_flagged: True, values: ["friday", "monday"]
    q5 = no_llm_service.ask("Which day does Server A deploy?", reference_time=now_ref)
    assert q5.answer.conflict_flagged is True
    assert "friday" in [v.lower() for v in q5.answer.values]
    assert "monday" in [v.lower() for v in q5.answer.values]

    # q6: Python version in staging -> 3.11
    q6 = no_llm_service.ask(
        "What Python version does Server A use in staging?", reference_time=now_ref
    )
    assert q6.answer.value == "3.11"
    assert "3.11" in q6.answer.answer_text

    # q7: History of Python versions -> [3.10, 3.12]
    q7 = no_llm_service.ask("Which Python versions has Server A had?", reference_time=now_ref)
    norm_vals = [v.strip() for v in q7.answer.values]
    assert "3.10" in norm_vals
    assert "3.12" in norm_vals
    assert "3.9" not in norm_vals
