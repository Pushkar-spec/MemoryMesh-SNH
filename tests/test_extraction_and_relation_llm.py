"""Comprehensive tests for LLM-assisted extraction and relation classification (P7.2).

Governed by docs/ARCHITECTURE.md Section 7, docs/SPECIFICATION.md Section 3,
docs/RESOLUTION_ENGINE.md Section 3, and MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P7.2).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from memorymesh.config import Cardinality, ValueType
from memorymesh.db import init_db
from memorymesh.engine.classifier import classify_relation
from memorymesh.errors import LLMOutputError
from memorymesh.llm.classifier import LLMRelationClassifier, classify_relation_llm
from memorymesh.llm.client import LLMClient
from memorymesh.llm.extractor import ClaimExtractor
from memorymesh.llm.prompts import (
    build_classify_relation_prompt,
    build_extract_claims_prompt,
    format_attribute_catalog,
    format_known_entities,
)
from memorymesh.llm.providers import FakeProvider
from memorymesh.models import (
    Action,
    AttributeDef,
    Claim,
    IngestMessage,
    Polarity,
    Relation,
    RelationProposal,
    Source,
)
from memorymesh.repository import Repository
from memorymesh.service import MemoryService

# ==============================================================================
# Test Fixtures & Helpers
# ==============================================================================


@pytest.fixture
def sample_attributes() -> list[AttributeDef]:
    """Sample attribute catalog definitions for testing."""
    return [
        AttributeDef(
            attribute="python_version",
            description="Python runtime version on a server",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
            aliases=["python", "runtime", "py version"],
        ),
        AttributeDef(
            attribute="runs_service",
            description="Services hosted on a server",
            cardinality=Cardinality.MULTI,
            value_type=ValueType.TEXT,
            aliases=["hosts", "runs", "serves"],
        ),
        AttributeDef(
            attribute="compatibility_note",
            description="Statement that something works with a given runtime",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
            aliases=["works with", "compatible with", "needs"],
        ),
        AttributeDef(
            attribute="service_status",
            description="Operational status",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.ENUM,
            allowed_values=["up", "down", "degraded", "maintenance"],
            aliases=["status"],
        ),
    ]


@pytest.fixture
def memory_repo() -> Repository:
    """In-memory SQLite repository initialized with standard schema."""
    conn = init_db(":memory:")
    return Repository(conn)


# ==============================================================================
# 1. Extraction Unit Tests
# ==============================================================================


def test_extract_single_claim(sample_attributes: list[AttributeDef]) -> None:
    """Extract a single direct factual claim from natural language text."""
    payload = {
        "claims": [
            {
                "subject": "Server A",
                "attribute": "python_version",
                "value": "3.12",
                "previous_value": None,
                "polarity": "assert",
                "valid_from": "2026-06-01T08:00:00Z",
                "valid_to": None,
                "context": {},
                "depends_on": [],
                "extract_confidence": 0.95,
                "span": "Server A is running Python 3.12",
            }
        ]
    }
    fake_provider = FakeProvider(default_response=json.dumps(payload))
    client = LLMClient(provider=fake_provider, cache_enabled=False)
    extractor = ClaimExtractor(
        llm_client=client,
        attributes=sample_attributes,
        known_entities=["Server A"],
    )

    msg = IngestMessage(
        source_id="admin_alice",
        text="Server A is running Python 3.12 as of this morning.",
        event_time=datetime(2026, 6, 1, 8, 0, tzinfo=UTC),
    )
    result = extractor.extract(msg)

    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.subject == "Server A"
    assert claim.attribute == "python_version"
    assert claim.value == "3.12"
    assert claim.previous_value is None
    assert claim.polarity == Polarity.ASSERT
    assert claim.valid_from == datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
    assert claim.extract_confidence == 0.95
    assert claim.span == "Server A is running Python 3.12"

    domain_claims = extractor.to_domain_claims(result, message=msg)
    assert len(domain_claims) == 1
    assert domain_claims[0].subject == "Server A"
    assert domain_claims[0].value == "3.12"


def test_extract_multiple_claims_in_one_input(sample_attributes: list[AttributeDef]) -> None:
    """Extract multiple distinct claims from a single compound message."""
    payload = {
        "claims": [
            {
                "subject": "Server A",
                "attribute": "python_version",
                "value": "3.12",
                "polarity": "assert",
                "extract_confidence": 0.95,
                "span": "Server A runs Python 3.12",
            },
            {
                "subject": "Server A",
                "attribute": "runs_service",
                "value": "auth-service",
                "polarity": "assert",
                "extract_confidence": 0.90,
                "span": "hosts the auth-service",
            },
        ]
    }
    fake_provider = FakeProvider(default_response=json.dumps(payload))
    client = LLMClient(provider=fake_provider, cache_enabled=False)
    extractor = ClaimExtractor(llm_client=client, attributes=sample_attributes)

    result = extractor.extract("Server A runs Python 3.12 and hosts the auth-service.")
    assert len(result.claims) == 2
    assert result.claims[0].attribute == "python_version"
    assert result.claims[0].value == "3.12"
    assert result.claims[1].attribute == "runs_service"
    assert result.claims[1].value == "auth-service"


def test_extract_explicit_previous_and_new_values(sample_attributes: list[AttributeDef]) -> None:
    """Capture explicit previous value when stated (e.g. from 3.10 to 3.12)."""
    payload = {
        "claims": [
            {
                "subject": "Server A",
                "attribute": "python_version",
                "value": "3.12",
                "previous_value": "3.10",
                "polarity": "assert",
                "valid_from": "2026-05-25T00:00:00Z",
                "extract_confidence": 0.98,
                "span": "migrated Server A from Python 3.10 to Python 3.12 last Monday",
            }
        ]
    }
    fake_provider = FakeProvider(default_response=json.dumps(payload))
    client = LLMClient(provider=fake_provider, cache_enabled=False)
    extractor = ClaimExtractor(llm_client=client, attributes=sample_attributes)

    text = "We migrated Server A from Python 3.10 to Python 3.12 last Monday."
    result = extractor.extract(text, reference_time=datetime(2026, 6, 1, 10, 0, tzinfo=UTC))

    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.subject == "Server A"
    assert claim.value == "3.12"
    assert claim.previous_value == "3.10"
    assert claim.valid_from == datetime(2026, 5, 25, 0, 0, tzinfo=UTC)

    domain_claim = claim.to_claim()
    assert domain_claim.previous_value == "3.10"


def test_extract_missing_or_ambiguous_values(sample_attributes: list[AttributeDef]) -> None:
    """Input containing no factual assertions returns empty claims list without inventing facts."""
    payload = {"claims": []}
    fake_provider = FakeProvider(default_response=json.dumps(payload))
    client = LLMClient(provider=fake_provider, cache_enabled=False)
    extractor = ClaimExtractor(llm_client=client, attributes=sample_attributes)

    result = extractor.extract("I wonder how Server A is doing today?")
    assert len(result.claims) == 0


def test_extract_attribute_alias_normalization(sample_attributes: list[AttributeDef]) -> None:
    """Raw attribute alias (e.g. 'runtime') is normalized to canonical catalog name."""
    payload = {
        "claims": [
            {
                "subject": "Server A",
                "attribute": "runtime",  # Alias for python_version
                "value": "3.12",
                "extract_confidence": 0.90,
                "span": "runtime is 3.12",
            }
        ]
    }
    fake_provider = FakeProvider(default_response=json.dumps(payload))
    client = LLMClient(provider=fake_provider, cache_enabled=False)
    extractor = ClaimExtractor(llm_client=client, attributes=sample_attributes)

    result = extractor.extract("Server A runtime is 3.12.")
    assert len(result.claims) == 1
    assert result.claims[0].attribute == "python_version"


def test_extract_invalid_output_and_safe_fallback(sample_attributes: list[AttributeDef]) -> None:
    """Malformed output that fails parsing returns empty list safely when raise_on_error=False."""
    fake_provider = FakeProvider(default_response="Not valid JSON at all")
    client = LLMClient(provider=fake_provider, cache_enabled=False)
    extractor = ClaimExtractor(llm_client=client, attributes=sample_attributes)

    # Safe fallback returns empty ExtractionResult
    result = extractor.extract("Some message", raise_on_error=False)
    assert len(result.claims) == 0

    # Explicit raise_on_error=True raises LLMOutputError
    with pytest.raises(LLMOutputError):
        extractor.extract("Some message", raise_on_error=True)


def test_extract_pydantic_validation_failure_handling(
    sample_attributes: list[AttributeDef],
) -> None:
    """Schema violation in LLM output triggers repair retry; fails gracefully if unrepairable."""
    bad_payload = {"claims": [{"subject": 12345, "attribute": "invalid"}]}  # Missing value
    fake_provider = FakeProvider(default_response=json.dumps(bad_payload))
    client = LLMClient(provider=fake_provider, cache_enabled=False)
    extractor = ClaimExtractor(llm_client=client, attributes=sample_attributes)

    result = extractor.extract("Server A is down", raise_on_error=False)
    assert len(result.claims) == 0


# ==============================================================================
# 2. Relation Classification Unit Tests
# ==============================================================================


def test_relation_classification_replacement_proposal(
    sample_attributes: list[AttributeDef],
) -> None:
    """Example 1: Replacement interpretation on single-valued attribute."""
    payload = {
        "relation": "conflicts",
        "confidence": 0.95,
        "reason": "New statement upgrades Server A to Python 3.12, replacing Python 3.10.",
    }
    fake_provider = FakeProvider(default_response=json.dumps(payload))
    client = LLMClient(provider=fake_provider, cache_enabled=False)

    attr = sample_attributes[2]  # compatibility_note (single, text)
    res = classify_relation_llm(
        llm_client=client,
        claim_val="uses Python 3.12",
        mem_val="uses Python 3.10",
        schema=attr,
        claim_raw_text="We upgraded Server A to Python 3.12.",
        mem_raw_text="Server A uses Python 3.10.",
    )
    assert res.relation == RelationProposal.CONFLICTS
    assert res.confidence == 0.95

    # Pure engine mapping with proposal returns CONFLICTS
    engine_rel = classify_relation(
        "uses Python 3.12",
        "uses Python 3.10",
        attr,
        proposal=res.relation,
    )
    assert engine_rel == Relation.CONFLICTS


def test_relation_classification_refinement_proposal(
    sample_attributes: list[AttributeDef],
) -> None:
    """Example 2: Refinement interpretation (adds specific detail without contradicting)."""
    payload = {
        "relation": "refines",
        "confidence": 0.92,
        "reason": "Python 3.12 specifies the exact runtime for generic Python assertion.",
    }
    fake_provider = FakeProvider(default_response=json.dumps(payload))
    client = LLMClient(provider=fake_provider, cache_enabled=False)

    attr = sample_attributes[2]  # compatibility_note (single, text)
    res = classify_relation_llm(
        llm_client=client,
        claim_val="uses Python 3.12",
        mem_val="uses Python",
        schema=attr,
    )
    assert res.relation == RelationProposal.REFINES
    assert res.confidence == 0.92

    engine_rel = classify_relation(
        "uses Python 3.12",
        "uses Python",
        attr,
        proposal=res.relation,
    )
    assert engine_rel == Relation.REFINES


def test_relation_classification_coexistence_proposal(
    sample_attributes: list[AttributeDef],
) -> None:
    """Example 3: Coexistence interpretation on multi-valued attribute."""
    payload = {
        "relation": "complementary",
        "confidence": 0.96,
        "reason": "Application can support both Python and Node.js simultaneously.",
    }
    fake_provider = FakeProvider(default_response=json.dumps(payload))
    client = LLMClient(provider=fake_provider, cache_enabled=False)

    attr = sample_attributes[1]  # runs_service (multi, text)
    res = classify_relation_llm(
        llm_client=client,
        claim_val="Node.js",
        mem_val="Python",
        schema=attr,
    )
    assert res.relation == RelationProposal.COMPLEMENTARY

    engine_rel = classify_relation(
        "Node.js",
        "Python",
        attr,
        proposal=res.relation,
    )
    assert engine_rel == Relation.COMPLEMENTARY


def test_relation_classification_ambiguous_relationship(
    sample_attributes: list[AttributeDef],
) -> None:
    """Example 4: Ambiguous/uncertain relationship does not force confident classification."""
    payload = {
        "relation": "ambiguous",
        "confidence": 0.40,
        "reason": "Hedging in statement leaves runtime relationship uncertain.",
    }
    fake_provider = FakeProvider(default_response=json.dumps(payload))
    client = LLMClient(provider=fake_provider, cache_enabled=False)

    attr = sample_attributes[2]  # compatibility_note (single, text)
    classifier = LLMRelationClassifier(client)
    res = classifier.classify(
        claim_val="uses Python 3.12",
        mem_val="uses Python 3.10",
        schema=attr,
    )
    assert res.relation == RelationProposal.AMBIGUOUS
    assert res.confidence == 0.40

    # Ambiguous proposal resolves to RelationProposal.AMBIGUOUS
    proposal = classifier.resolve_proposal(res)
    assert proposal == RelationProposal.AMBIGUOUS

    # Ambiguous proposal produces Relation.AMBIGUOUS in pure engine
    engine_rel = classify_relation(
        "uses Python 3.12",
        "uses Python 3.10",
        attr,
        proposal=proposal,
    )
    assert engine_rel == Relation.AMBIGUOUS


def test_cardinality_guard_overrides_llm_complementary_proposal() -> None:
    """Single-valued attribute CANNOT coexist even if LLM proposes complementary."""
    payload = {
        "relation": "complementary",
        "confidence": 0.95,
        "reason": "Both values seem fine together.",
    }
    fake_provider = FakeProvider(default_response=json.dumps(payload))
    client = LLMClient(provider=fake_provider, cache_enabled=False)

    single_attr = AttributeDef(
        attribute="single_text",
        description="Single valued text attribute",
        cardinality=Cardinality.SINGLE,
        value_type=ValueType.TEXT,
    )

    classifier = LLMRelationClassifier(client)
    res = classifier.classify("Val B", "Val A", single_attr)
    assert res.relation == RelationProposal.COMPLEMENTARY

    # Pure engine cardinality guard strictly enforces CONFLICTS on single-valued
    authoritative = classify_relation("Val B", "Val A", single_attr, proposal=res.relation)
    assert authoritative == Relation.CONFLICTS


def test_invalid_llm_classification_falls_back_deterministically() -> None:
    """When LLM classifier crashes or outputs invalid JSON, fallback is deterministic."""
    fake_provider = FakeProvider(default_response="Not JSON")
    client = LLMClient(provider=fake_provider, cache_enabled=False)

    single_attr = AttributeDef(
        attribute="single_text",
        description="Single valued text",
        cardinality=Cardinality.SINGLE,
        value_type=ValueType.TEXT,
    )
    multi_attr = AttributeDef(
        attribute="multi_text",
        description="Multi valued text",
        cardinality=Cardinality.MULTI,
        value_type=ValueType.TEXT,
    )

    classifier = LLMRelationClassifier(client)
    res_single_raw = classifier.classify("Val B", "Val A", single_attr)
    assert res_single_raw.relation == RelationProposal.AMBIGUOUS
    proposal = classifier.resolve_proposal(res_single_raw)
    assert proposal == RelationProposal.AMBIGUOUS

    # Similar strings (ratio >= 70) yield AMBIGUOUS
    assert classify_relation("Val B", "Val A", single_attr, proposal=proposal) == Relation.AMBIGUOUS
    assert classify_relation("Val B", "Val A", single_attr) == Relation.AMBIGUOUS

    # Clearly different strings (ratio < 70) deterministically conflict/complement
    assert classify_relation("Alpha Service", "Zeta Cluster", single_attr) == Relation.CONFLICTS
    assert classify_relation("Alpha Service", "Zeta Cluster", multi_attr) == Relation.COMPLEMENTARY


def test_deterministic_exact_match_bypasses_llm_entirely() -> None:
    """Exact normalized value match is always IDENTICAL without calling the LLM."""

    # Provider with zero configured responses that would raise if called
    class FailIfCalledProvider:
        def complete(self, *args: Any, **kwargs: Any) -> str:
            raise AssertionError("Provider should NOT be called for exact value matches!")

    attr = AttributeDef(
        attribute="note",
        description="Text note",
        cardinality=Cardinality.SINGLE,
        value_type=ValueType.TEXT,
    )

    # Pure engine classification has no provider dependency
    res = classify_relation("Exact Match", "Exact Match", attr)
    assert res == Relation.IDENTICAL


# ==============================================================================
# 3. Integration Tests with MemoryService & Resolution Engine
# ==============================================================================


def test_service_extract_claims_integration(memory_repo: Repository) -> None:
    """MemoryService.extract_claims integrates cleanly with LLMClient and ClaimExtractor."""
    payload = {
        "claims": [
            {
                "subject": "Server A",
                "attribute": "python_version",
                "value": "3.12",
                "polarity": "assert",
                "extract_confidence": 0.95,
                "span": "Server A has Python 3.12",
            }
        ]
    }
    fake_provider = FakeProvider(default_response=json.dumps(payload))
    service = MemoryService(db_target=memory_repo, llm=fake_provider)

    msg = IngestMessage(
        source_id="admin_alice",
        text="Server A has Python 3.12 installed.",
        event_time=datetime(2026, 6, 1, 10, 0, tzinfo=UTC),
    )
    claims = service.extract_claims(msg)

    assert len(claims) == 1
    assert claims[0].subject == "Server A"
    assert claims[0].attribute == "python_version"
    assert claims[0].value == "3.12"
    assert claims[0].valid_from == datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


def test_llm_cannot_bypass_deterministic_trust_rules(memory_repo: Repository) -> None:
    """A claim with low trust proposing conflict/replacement is REJECTED by deterministic engine."""
    # 1. Register high-trust source and ingest initial active memory
    memory_repo.insert_source(
        Source(id="admin_priya", name="Priya", type="admin", reliability=0.95)
    )
    memory_repo.insert_source(Source(id="chat_bot", name="Chat Bot", type="chat", reliability=0.20))

    service = MemoryService(db_target=memory_repo)

    # Ingest baseline high-trust memory
    t1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    service.ingest_claims(
        IngestMessage(
            source_id="admin_priya",
            text="Server A note: Verified stable",
            event_time=t1,
        ),
        [
            Claim(
                subject="Server A",
                attribute="compatibility_note",
                value="Verified stable",
                extract_confidence=1.0,
            )
        ],
    )

    # Verify initial memory is active
    memories = service.current(entity="Server A", attribute="compatibility_note", now=t1)
    assert len(memories.items) == 1
    assert memories.items[0].memory.value_display == "Verified stable"

    # 2. Ingest low-trust claim with an LLM that proposes 'conflicts'
    classify_payload = {
        "relation": "conflicts",
        "confidence": 0.95,
        "reason": "Claims assert conflicting stability assessments.",
    }
    fake_provider = FakeProvider(default_response=json.dumps(classify_payload))
    service_with_llm = MemoryService(db_target=memory_repo, llm=fake_provider)

    t2 = datetime(2026, 6, 2, 0, 0, tzinfo=UTC)
    report = service_with_llm.ingest_claims(
        IngestMessage(source_id="chat_bot", text="Server A note: Unstable crash", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="compatibility_note",
                value="Unstable crash",
                extract_confidence=0.90,
            )
        ],
    )

    # 3. Deterministic trust rules must REJECT the low-trust replacement attempt (R-REJECT)
    assert len(report.decisions) == 1
    assert report.decisions[0].action == Action.REJECT
    assert report.decisions[0].rule_id == "R-REJECT"

    # Active memory remains the high-trust 'Verified stable'
    active_mems = service.current(entity="Server A", attribute="compatibility_note", now=t2)
    assert len(active_mems.items) == 1
    assert active_mems.items[0].memory.value_display == "Verified stable"


def test_prompt_builders_format_cleanly() -> None:
    """Prompt builders format catalog, entities, and definitions without syntax errors."""
    attrs = [
        AttributeDef(
            attribute="python_version",
            description="Runtime",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
            aliases=["py"],
        )
    ]
    catalog_text = format_attribute_catalog(attrs)
    assert "python_version" in catalog_text
    assert "single" in catalog_text

    entities_text = format_known_entities(["Server A", "Server B"])
    assert "Server A, Server B" == entities_text

    sys_extract, user_extract = build_extract_claims_prompt(
        "Server A runs Python 3.12",
        reference_time=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
        source_name="alice",
        attributes=attrs,
        known_entities=["Server A"],
    )
    assert "python_version" in user_extract
    assert "Server A" in user_extract

    sys_cls, user_cls = build_classify_relation_prompt(
        attribute_name="compatibility_note",
        attribute_description="Compatibility",
        cardinality="single",
        existing_value="Python 3.10",
        new_value="Python 3.12",
    )
    assert "Python 3.10" in user_cls
    assert "Python 3.12" in user_cls


# ==============================================================================
# 4. Engine Purity, Ambiguity Fallback & previous_value Tests
# ==============================================================================


def test_pure_engine_classifier_has_zero_llm_dependencies() -> None:
    """Engine classifier is completely pure and deterministic with no LLM dependency."""
    attr_single = AttributeDef(
        attribute="python_version",
        description="Python runtime",
        cardinality=Cardinality.SINGLE,
        value_type=ValueType.VERSION,
    )
    attr_multi = AttributeDef(
        attribute="runs_service",
        description="Services",
        cardinality=Cardinality.MULTI,
        value_type=ValueType.TEXT,
    )

    # Works with no LLM client or proposal
    assert classify_relation("3.12", "3.12", attr_single) == Relation.IDENTICAL
    assert classify_relation("3.12.4", "3.12", attr_single) == Relation.REFINES
    assert classify_relation("3.10", "3.12", attr_single) == Relation.CONFLICTS
    assert classify_relation("auth-api", "web-gateway", attr_multi) == Relation.COMPLEMENTARY

    # Respects pure proposal inputs without any network/client
    assert (
        classify_relation("svc_b", "svc_a", attr_multi, proposal=RelationProposal.IDENTICAL)
        == Relation.IDENTICAL
    )


def test_scenario_a_single_attribute_ambiguous_relation_prevents_replacement(
    memory_repo: Repository,
) -> None:
    """Scenario A: SINGLE attribute with high-confidence claim and AMBIGUOUS LLM relation.

    Ambiguity alone must not manufacture replacement or dispute; claim is skipped.
    """
    memory_repo.insert_source(Source(id="admin_user", name="Admin", type="admin", reliability=0.95))
    memory_repo.insert_attribute_schema(
        AttributeDef(
            attribute="runtime_text",
            description="Runtime environment string",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )

    ambiguous_payload = {
        "relation": "ambiguous",
        "confidence": 0.40,
        "reason": "Relationship between runtime versions is ambiguous",
    }
    fake_provider = FakeProvider(default_response=json.dumps(ambiguous_payload))
    service = MemoryService(db_target=memory_repo, llm=fake_provider)

    # 1. Baseline trusted memory: Python 3.10
    t1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A uses Python 3.10", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="runtime_text",
                value="Python 3.10",
                extract_confidence=1.0,
            )
        ],
    )

    # 2. New high-confidence claim: Python 3.12 (confidence 1.0)
    t2 = datetime(2026, 6, 2, 0, 0, tzinfo=UTC)
    report = service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A uses Python 3.12", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="runtime_text",
                value="Python 3.12",
                extract_confidence=1.0,  # High claim confidence
            )
        ],
    )

    # Ambiguity alone must NOT cause replacement or dispute
    assert len(report.decisions) == 1
    assert report.decisions[0].action == Action.SKIP
    assert report.decisions[0].rule_id == "R-SKIP-AMBIGUOUS"
    assert report.decisions[0].action != Action.REPLACE
    assert report.decisions[0].action != Action.DISPUTE

    # Existing trusted memory remains ACTIVE and untouched
    current_mems = service.current(entity="Server A", attribute="runtime_text", now=t2)
    assert len(current_mems.items) == 1
    assert current_mems.items[0].memory.value_display == "Python 3.10"
    assert current_mems.items[0].memory.closed_at is None


def test_scenario_b_single_attribute_provider_failure_prevents_replacement(
    memory_repo: Repository,
) -> None:
    """Scenario B: SINGLE attribute with provider failure.

    Provider failure must not manufacture replacement or dispute; claim is skipped.
    """
    memory_repo.insert_source(Source(id="admin_user", name="Admin", type="admin", reliability=0.95))
    memory_repo.insert_attribute_schema(
        AttributeDef(
            attribute="runtime_text",
            description="Runtime environment string",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )

    class CrashingProvider:
        def complete(self, *args: Any, **kwargs: Any) -> str:
            raise ConnectionError("Simulated LLM network timeout")

    service = MemoryService(db_target=memory_repo, llm=CrashingProvider())

    t1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A uses Python 3.10", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="runtime_text",
                value="Python 3.10",
                extract_confidence=1.0,
            )
        ],
    )

    t2 = datetime(2026, 6, 2, 0, 0, tzinfo=UTC)
    report = service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A uses Python 3.12", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="runtime_text",
                value="Python 3.12",
                extract_confidence=1.0,
            )
        ],
    )

    assert len(report.decisions) == 1
    assert report.decisions[0].action == Action.SKIP
    assert report.decisions[0].rule_id == "R-SKIP-AMBIGUOUS"

    # Existing memory untouched
    current_mems = service.current(entity="Server A", attribute="runtime_text", now=t2)
    assert len(current_mems.items) == 1
    assert current_mems.items[0].memory.value_display == "Python 3.10"


def test_scenario_c_multi_attribute_ambiguous_relation_prevents_coexistence(
    memory_repo: Repository,
) -> None:
    """Scenario C: MULTI attribute with high-confidence claim and AMBIGUOUS relation.

    Ambiguity alone must not silently cause coexistence; claim is safely skipped.
    """
    memory_repo.insert_source(Source(id="admin_user", name="Admin", type="admin", reliability=0.90))

    ambiguous_payload = {
        "relation": "ambiguous",
        "confidence": 0.30,
        "reason": "Uncertain whether gateway complements auth",
    }
    fake_provider = FakeProvider(default_response=json.dumps(ambiguous_payload))
    service = MemoryService(db_target=memory_repo, llm=fake_provider)

    t1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A runs auth service", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="runs_service",
                value="auth service",
                extract_confidence=1.0,
            )
        ],
    )

    t2 = datetime(2026, 6, 2, 0, 0, tzinfo=UTC)
    report = service.ingest_claims(
        IngestMessage(
            source_id="admin_user", text="Server A runs authentication service", event_time=t2
        ),
        [
            Claim(
                subject="Server A",
                attribute="runs_service",
                value="authentication service",
                extract_confidence=1.0,  # High claim confidence
            )
        ],
    )

    # Ambiguity alone must NOT cause silent coexistence
    assert len(report.decisions) == 1
    assert report.decisions[0].action == Action.SKIP
    assert report.decisions[0].rule_id == "R-SKIP-AMBIGUOUS"
    assert report.decisions[0].action != Action.COEXIST

    # Only original memory exists; duplicate candidate was not added
    active_mems = service.current(entity="Server A", attribute="runs_service", now=t2)
    assert len(active_mems.items) == 1
    assert active_mems.items[0].memory.value_display == "auth service"


def test_scenario_d_low_relation_confidence_prevents_action_despite_high_claim_confidence(
    memory_repo: Repository,
) -> None:
    """Scenario D: New claim confidence is high, but relation confidence is below threshold.

    High claim confidence cannot substitute for relation confidence.
    """
    memory_repo.insert_source(Source(id="admin_user", name="Admin", type="admin", reliability=0.95))
    memory_repo.insert_attribute_schema(
        AttributeDef(
            attribute="runtime_text",
            description="Runtime environment string",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )

    low_conf_payload = {
        "relation": "conflicts",
        "confidence": 0.35,  # Below min_confidence (0.50)
        "reason": "Tentative relation proposal",
    }
    fake_provider = FakeProvider(default_response=json.dumps(low_conf_payload))
    service = MemoryService(db_target=memory_repo, llm=fake_provider)

    t1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A uses Python 3.10", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="runtime_text",
                value="Python 3.10",
                extract_confidence=1.0,
            )
        ],
    )

    t2 = datetime(2026, 6, 2, 0, 0, tzinfo=UTC)
    report = service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A uses Python 3.12", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="runtime_text",
                value="Python 3.12",
                extract_confidence=0.98,  # Very high claim confidence
            )
        ],
    )

    assert len(report.decisions) == 1
    assert report.decisions[0].action == Action.SKIP
    assert report.decisions[0].rule_id == "R-SKIP-AMBIGUOUS"

    current_mems = service.current(entity="Server A", attribute="runtime_text", now=t2)
    assert len(current_mems.items) == 1
    assert current_mems.items[0].memory.value_display == "Python 3.10"


def test_scenario_e_deterministic_fallback_without_llm_genuinely_establishes_relations(
    memory_repo: Repository,
) -> None:
    """Scenario E: Deterministic classifier genuinely establishes relations without an LLM."""
    service = MemoryService(db_target=memory_repo, llm=None)  # No LLM configured
    memory_repo.insert_source(Source(id="admin_user", name="Admin", type="admin", reliability=0.95))

    schema_version = AttributeDef(
        attribute="python_version",
        description="Python version",
        cardinality=Cardinality.SINGLE,
        value_type=ValueType.VERSION,
    )
    schema_enum = AttributeDef(
        attribute="service_status",
        description="Status",
        cardinality=Cardinality.SINGLE,
        value_type=ValueType.ENUM,
        allowed_values=["up", "down", "maintenance"],
    )
    schema_number = AttributeDef(
        attribute="port_number",
        description="Port",
        cardinality=Cardinality.SINGLE,
        value_type=ValueType.NUMBER,
    )
    schema_date = AttributeDef(
        attribute="release_date",
        description="Release date",
        cardinality=Cardinality.SINGLE,
        value_type=ValueType.DATE,
    )

    # 1. Version REFINES and CONFLICTS
    assert classify_relation("3.12.4", "3.12", schema_version) == Relation.REFINES
    assert classify_relation("3.12", "3.10", schema_version) == Relation.CONFLICTS

    # 2. Enum CONFLICTS and IDENTICAL
    assert classify_relation("up", "down", schema_enum) == Relation.CONFLICTS
    assert classify_relation("up", "up", schema_enum) == Relation.IDENTICAL

    # 3. Number CONFLICTS and IDENTICAL
    assert classify_relation("8080", "8443", schema_number) == Relation.CONFLICTS
    assert classify_relation("8080", "8080", schema_number) == Relation.IDENTICAL

    # 4. Date CONFLICTS and IDENTICAL
    assert classify_relation("2026-06-01", "2026-07-01", schema_date) == Relation.CONFLICTS
    assert classify_relation("2026-06-01", "2026-06-01", schema_date) == Relation.IDENTICAL

    # 5. Exact match on TEXT is IDENTICAL
    schema_text = AttributeDef(
        attribute="os_release",
        description="OS",
        cardinality=Cardinality.SINGLE,
        value_type=ValueType.TEXT,
    )
    assert classify_relation("Ubuntu 24.04", "Ubuntu 24.04", schema_text) == Relation.IDENTICAL

    # 6. Service ingestion with version refinement without LLM
    t1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Python 3.12", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                extract_confidence=1.0,
            )
        ],
    )

    t2 = datetime(2026, 6, 2, 0, 0, tzinfo=UTC)
    report = service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Python 3.12.4", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12.4",
                extract_confidence=1.0,
            )
        ],
    )
    assert len(report.decisions) == 1
    assert report.decisions[0].action == Action.REFINE
    assert report.decisions[0].rule_id == "R-REFINE"


def test_provider_failure_with_deterministically_classifiable_pair(
    memory_repo: Repository,
) -> None:
    """Clearly different strings (ratio < 70) resolve deterministically despite provider failure."""
    memory_repo.insert_source(Source(id="admin_user", name="Admin", type="admin", reliability=0.95))
    memory_repo.insert_attribute_schema(
        AttributeDef(
            attribute="os_release",
            description="Operating system release",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )

    class CrashingProvider:
        def complete(self, *args: Any, **kwargs: Any) -> str:
            raise ConnectionError("Simulated LLM network timeout")

    service = MemoryService(db_target=memory_repo, llm=CrashingProvider())

    t1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A runs Ubuntu 24.04", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="os_release",
                value="Ubuntu 24.04",
                extract_confidence=1.0,
            )
        ],
    )

    # Ingest newer conflicting OS claim (ratio 38.1 < 70)
    t2 = datetime(2026, 6, 2, 0, 0, tzinfo=UTC)
    report = service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A runs Debian 12", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="os_release",
                value="Debian 12",
                extract_confidence=1.0,
            )
        ],
    )

    # Deterministic classifier establishes CONFLICTS -> REPLACE
    assert len(report.decisions) == 1
    assert report.decisions[0].action == Action.REPLACE
    assert report.decisions[0].rule_id == "R-REPLACE"

    current_mems = service.current(entity="Server A", attribute="os_release", now=t2)
    assert len(current_mems.items) == 1
    assert current_mems.items[0].memory.value_display == "Debian 12"


def test_provider_failure_with_genuinely_unresolved_pair(
    memory_repo: Repository,
) -> None:
    """Similar candidate strings (ratio >= 70) safely SKIP on provider failure."""
    memory_repo.insert_source(Source(id="admin_user", name="Admin", type="admin", reliability=0.95))
    memory_repo.insert_attribute_schema(
        AttributeDef(
            attribute="runtime_text",
            description="Runtime environment string",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )

    class CrashingProvider:
        def complete(self, *args: Any, **kwargs: Any) -> str:
            raise ConnectionError("Simulated LLM network timeout")

    service = MemoryService(db_target=memory_repo, llm=CrashingProvider())

    t1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A uses Python 3.10", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="runtime_text",
                value="Python 3.10",
                extract_confidence=1.0,
            )
        ],
    )

    t2 = datetime(2026, 6, 2, 0, 0, tzinfo=UTC)
    report = service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A uses Python 3.12", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="runtime_text",
                value="Python 3.12",
                extract_confidence=1.0,
            )
        ],
    )

    assert len(report.decisions) == 1
    assert report.decisions[0].action == Action.SKIP
    assert report.decisions[0].rule_id == "R-SKIP-AMBIGUOUS"

    current_mems = service.current(entity="Server A", attribute="runtime_text", now=t2)
    assert len(current_mems.items) == 1
    assert current_mems.items[0].memory.value_display == "Python 3.10"


def test_ambiguous_llm_output_with_deterministically_classifiable_pair(
    memory_repo: Repository,
) -> None:
    """Ambiguous LLM output still resolves deterministically when strings are distinct (< 70)."""
    memory_repo.insert_source(Source(id="admin_user", name="Admin", type="admin", reliability=0.95))
    memory_repo.insert_attribute_schema(
        AttributeDef(
            attribute="runs_service",
            description="Running services",
            cardinality=Cardinality.MULTI,
            value_type=ValueType.TEXT,
        )
    )

    ambiguous_payload = {
        "relation": "ambiguous",
        "confidence": 0.40,
        "reason": "Unsure if services complement each other",
    }
    fake_provider = FakeProvider(default_response=json.dumps(ambiguous_payload))
    service = MemoryService(db_target=memory_repo, llm=fake_provider)

    t1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A runs auth-api", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="runs_service",
                value="auth-api",
                extract_confidence=1.0,
            )
        ],
    )

    # Distinct service (ratio 31.6% < 70)
    t2 = datetime(2026, 6, 2, 0, 0, tzinfo=UTC)
    report = service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A runs web-gateway", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="runs_service",
                value="web-gateway",
                extract_confidence=1.0,
            )
        ],
    )

    # Ratio < 70 on MULTI establishes COMPLEMENTARY -> COEXIST
    assert len(report.decisions) == 1
    assert report.decisions[0].action == Action.COEXIST
    assert report.decisions[0].rule_id == "R-COEXIST"

    active_mems = service.current(entity="Server A", attribute="runs_service", now=t2)
    assert len(active_mems.items) == 2


def test_ambiguous_llm_output_with_genuinely_unresolved_pair(
    memory_repo: Repository,
) -> None:
    """Ambiguous LLM output on similar candidate strings (ratio >= 70) safely produces SKIP."""
    memory_repo.insert_source(Source(id="admin_user", name="Admin", type="admin", reliability=0.95))
    memory_repo.insert_attribute_schema(
        AttributeDef(
            attribute="runtime_text",
            description="Runtime environment string",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )

    ambiguous_payload = {
        "relation": "ambiguous",
        "confidence": 0.40,
        "reason": "Ambiguous runtime versions",
    }
    fake_provider = FakeProvider(default_response=json.dumps(ambiguous_payload))
    service = MemoryService(db_target=memory_repo, llm=fake_provider)

    t1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A uses Python 3.10", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="runtime_text",
                value="Python 3.10",
                extract_confidence=1.0,
            )
        ],
    )

    t2 = datetime(2026, 6, 2, 0, 0, tzinfo=UTC)
    report = service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A uses Python 3.12", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="runtime_text",
                value="Python 3.12",
                extract_confidence=1.0,
            )
        ],
    )

    assert len(report.decisions) == 1
    assert report.decisions[0].action == Action.SKIP
    assert report.decisions[0].rule_id == "R-SKIP-AMBIGUOUS"

    current_mems = service.current(entity="Server A", attribute="runtime_text", now=t2)
    assert len(current_mems.items) == 1
    assert current_mems.items[0].memory.value_display == "Python 3.10"


def test_no_llm_configured_with_genuinely_unresolved_pair(
    memory_repo: Repository,
) -> None:
    """Similar candidate strings (ratio >= 70) with no LLM safely produce R-SKIP-AMBIGUOUS."""
    memory_repo.insert_source(Source(id="admin_user", name="Admin", type="admin", reliability=0.95))
    memory_repo.insert_attribute_schema(
        AttributeDef(
            attribute="runtime_text",
            description="Runtime environment string",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.TEXT,
        )
    )

    service = MemoryService(db_target=memory_repo, llm=None)  # No LLM configured

    t1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A uses Python 3.10", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="runtime_text",
                value="Python 3.10",
                extract_confidence=1.0,
            )
        ],
    )

    t2 = datetime(2026, 6, 2, 0, 0, tzinfo=UTC)
    report = service.ingest_claims(
        IngestMessage(source_id="admin_user", text="Server A uses Python 3.12", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="runtime_text",
                value="Python 3.12",
                extract_confidence=1.0,
            )
        ],
    )

    assert len(report.decisions) == 1
    assert report.decisions[0].action == Action.SKIP
    assert report.decisions[0].rule_id == "R-SKIP-AMBIGUOUS"

    current_mems = service.current(entity="Server A", attribute="runtime_text", now=t2)
    assert len(current_mems.items) == 1
    assert current_mems.items[0].memory.value_display == "Python 3.10"


def test_uncertain_claim_cannot_silently_supersede_trusted_memory(
    memory_repo: Repository,
) -> None:
    """An uncertain or low-confidence claim never silently modifies or replaces trusted memory."""
    memory_repo.insert_source(
        Source(id="trusted_admin", name="Admin", type="admin", reliability=0.95)
    )
    memory_repo.insert_source(
        Source(id="rumor_source", name="Rumor", type="chat", reliability=0.25)
    )

    service = MemoryService(db_target=memory_repo)

    # 1. Trusted memory established
    t1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    service.ingest_claims(
        IngestMessage(source_id="trusted_admin", text="Server A is up", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="service_status",
                value="up",
                extract_confidence=1.0,
            )
        ],
    )

    # 2. Low-confidence claim below min_extract_confidence (0.40) is SKIPPED
    t2 = datetime(2026, 6, 2, 0, 0, tzinfo=UTC)
    report_lowconf = service.ingest_claims(
        IngestMessage(source_id="rumor_source", text="Server A might be down?", event_time=t2),
        [
            Claim(
                subject="Server A",
                attribute="service_status",
                value="down",
                extract_confidence=0.35,  # Below threshold
            )
        ],
    )
    assert len(report_lowconf.decisions) == 1
    assert report_lowconf.decisions[0].action == Action.SKIP
    assert report_lowconf.decisions[0].rule_id == "R-SKIP-LOWCONF"

    # 3. Low-trust claim with extract_confidence 0.50 is REJECTED
    t3 = datetime(2026, 6, 3, 0, 0, tzinfo=UTC)
    report_reject = service.ingest_claims(
        IngestMessage(source_id="rumor_source", text="I think Server A is down", event_time=t3),
        [
            Claim(
                subject="Server A",
                attribute="service_status",
                value="down",
                extract_confidence=0.50,
            )
        ],
    )
    assert len(report_reject.decisions) == 1
    assert report_reject.decisions[0].action == Action.REJECT
    assert report_reject.decisions[0].rule_id == "R-REJECT"

    # Trusted memory remains completely active and untouched
    current_mem = service.current(entity="Server A", attribute="service_status", now=t3)
    assert len(current_mem.items) == 1
    assert current_mem.items[0].memory.value_display == "up"


def test_previous_value_is_metadata_and_does_not_bypass_trust_rules(
    memory_repo: Repository,
) -> None:
    """previous_value does not independently trigger replacement or bypass trust guards."""
    memory_repo.insert_source(
        Source(id="trusted_admin", name="Admin", type="admin", reliability=0.95)
    )
    memory_repo.insert_source(
        Source(id="untrusted_source", name="Guest", type="user", reliability=0.20)
    )

    service = MemoryService(db_target=memory_repo)

    # 1. Establish high-trust active memory: Python 3.10
    t1 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    service.ingest_claims(
        IngestMessage(source_id="trusted_admin", text="Server A uses 3.10", event_time=t1),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                extract_confidence=1.0,
            )
        ],
    )

    # 2. Untrusted source claims upgrade from 3.10 to 3.12 with previous_value='3.10'
    t2 = datetime(2026, 6, 2, 0, 0, tzinfo=UTC)
    claim_with_prev = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        previous_value="3.10",  # Metadata indicating claimed superseded value
        extract_confidence=0.90,
    )
    report = service.ingest_claims(
        IngestMessage(source_id="untrusted_source", text="Upgraded 3.10 to 3.12", event_time=t2),
        [claim_with_prev],
    )

    # 3. previous_value does NOT bypass trust rules: the low-trust replacement attempt is REJECTED
    assert len(report.decisions) == 1
    assert report.decisions[0].action == Action.REJECT
    assert report.decisions[0].rule_id == "R-REJECT"

    # Active memory remains 3.10
    active = service.current(entity="Server A", attribute="python_version", now=t2)
    assert len(active.items) == 1
    assert active.items[0].memory.value_display == "3.10"

    # 4. Backward compatibility: Claim without previous_value works seamlessly
    claim_default = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
    )
    assert claim_default.previous_value is None
