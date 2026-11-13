"""Tests for LLM-assisted entity-resolution tie-break (Stretch P7.5).

Governed by:
- docs/ARCHITECTURE.md Section 3 (Stage 2)
- docs/SPECIFICATION.md Section 2 & 3
- MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P7.5)

Verifies:
1. Deterministic precedence: exact match, deterministically resolved fuzzy match,
   and new entities never invoke the LLM.
2. Tie-break execution: ambiguous candidate sets invoke LLM tie-break only when enabled.
3. Successful tie-break: valid high-confidence proposal resolves to the existing candidate
   without creating duplicate entities or modifying unrelated entities/aliases.
4. Safe failure behavior: low confidence, abstention, invalid candidate IDs, malformed JSON,
   validation errors, and provider exceptions fall back safely to deterministic review entities.
5. Regression coverage: similar but distinct entities are never silently merged.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from memorymesh.config import AppConfig, load_app_config
from memorymesh.db import init_db
from memorymesh.llm.client import LLMClient
from memorymesh.llm.providers import FakeProvider
from memorymesh.models import (
    Action,
    AttributeDef,
    Cardinality,
    Claim,
    EntityTiebreakResult,
    IngestMessage,
    Source,
    ValueType,
)
from memorymesh.repository import Repository
from memorymesh.service import MemoryService


@pytest.fixture
def base_config() -> AppConfig:
    return load_app_config()


@pytest.fixture
def sample_source() -> Source:
    return Source(
        id="sysadmin_priya",
        name="Priya Sharma",
        type="admin",
        reliability=0.95,
        notes="Senior Admin",
    )


@pytest.fixture
def repo(sample_source: Source) -> Repository:
    conn = init_db(":memory:")
    repository = Repository(conn)
    repository.insert_source(sample_source)
    repository.insert_attribute_schema(
        AttributeDef(
            attribute="python_version",
            description="Python runtime version",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
        )
    )
    return repository


def make_service_with_fake_llm(
    repo: Repository,
    cfg: AppConfig,
    provider: FakeProvider,
) -> MemoryService:
    client = LLMClient(
        provider=provider,
        repo=repo,
        config=cfg.engine.llm,
    )
    return MemoryService(repo, cfg=cfg, llm=client)


# ==============================================================================
# Model and Validation Unit Tests
# ==============================================================================


class TestEntityTiebreakModels:
    """Unit tests for EntityTiebreakResult Pydantic schema and validation rules."""

    def test_valid_proposal(self) -> None:
        res = EntityTiebreakResult(
            selected_entity_id=42,
            confidence=0.95,
            explanation="Matches Worker Alpha",
            abstained=False,
        )
        assert res.selected_entity_id == 42
        assert res.confidence == 0.95
        assert res.abstained is False

    def test_valid_abstention(self) -> None:
        res = EntityTiebreakResult(
            selected_entity_id=None,
            confidence=0.0,
            explanation="Cannot determine",
            abstained=True,
        )
        assert res.selected_entity_id is None
        assert res.abstained is True

    def test_abstained_with_selected_id_fails_validation(self) -> None:
        with pytest.raises(ValidationError):
            EntityTiebreakResult(
                selected_entity_id=42,
                confidence=0.9,
                explanation="Conflicting",
                abstained=True,
            )

    def test_confidence_out_of_range_fails_validation(self) -> None:
        with pytest.raises(ValidationError):
            EntityTiebreakResult(
                selected_entity_id=1,
                confidence=1.5,
                abstained=False,
            )
        with pytest.raises(ValidationError):
            EntityTiebreakResult(
                selected_entity_id=1,
                confidence=-0.1,
                abstained=False,
            )


# ==============================================================================
# 1. Deterministic Behavior (LLM Never Invoked)
# ==============================================================================


class TestDeterministicResolutionPrecedence:
    """Verifies that deterministic tiers always precede LLM tie-breaking."""

    def test_exact_match_does_not_invoke_llm(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        e1_id = repo.insert_entity("Server A", entity_type="server", created_at=t1)
        repo.insert_entity_alias("server a", e1_id)

        fake_provider = FakeProvider()
        service = make_service_with_fake_llm(repo, base_config, fake_provider)

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Server A runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report = service.ingest_claims(msg, [claim], recorded_at=t1)
        assert report.decisions[0].action == Action.INSERT

        # Deterministic exact match must not invoke LLM
        assert len(fake_provider.calls) == 0

    def test_deterministically_resolved_fuzzy_match_does_not_invoke_llm(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        e1_id = repo.insert_entity("Server Production Node", entity_type="server", created_at=t1)
        repo.insert_entity_alias("server production node", e1_id)

        fake_provider = FakeProvider()
        service = make_service_with_fake_llm(repo, base_config, fake_provider)

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Server Production Node Box runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="Server Production Node Box",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report = service.ingest_claims(msg, [claim], recorded_at=t1)
        assert report.decisions[0].action == Action.INSERT

        # High confidence fuzzy match (>= 90%) must not invoke LLM
        assert len(fake_provider.calls) == 0

    def test_new_entity_with_no_candidates_does_not_invoke_llm(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        fake_provider = FakeProvider()
        service = make_service_with_fake_llm(repo, base_config, fake_provider)

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="CompletelyUnknownCluster runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="CompletelyUnknownCluster",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report = service.ingest_claims(msg, [claim], recorded_at=t1)
        assert report.decisions[0].action == Action.INSERT

        # Unknown entity with no candidates must not invoke LLM
        assert len(fake_provider.calls) == 0

    def test_existing_ambiguous_candidate_set_invokes_tiebreak_when_enabled(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        e1_id = repo.insert_entity("Worker Alpha", entity_type="service", created_at=t1)
        e2_id = repo.insert_entity("Worker Beta", entity_type="service", created_at=t1)
        repo.insert_entity_alias("worker alpha", e1_id)
        repo.insert_entity_alias("worker beta", e2_id)

        fake_provider = FakeProvider(
            default_response=json.dumps(
                {
                    "selected_entity_id": e1_id,
                    "confidence": 0.95,
                    "explanation": "Refers to Worker Alpha",
                    "abstained": False,
                }
            )
        )
        service = make_service_with_fake_llm(repo, base_config, fake_provider)

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Worker runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="Worker",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        service.ingest_claims(msg, [claim], recorded_at=t1)

        # Ambiguous fuzzy tie MUST invoke the LLM tiebreak
        assert len(fake_provider.calls) == 1
        user_call = fake_provider.calls[0]["user"]
        assert "Worker" in user_call
        assert "Worker Alpha" in user_call
        assert "Worker Beta" in user_call


# ==============================================================================
# 2. Successful LLM Proposal
# ==============================================================================


class TestSuccessfulLLMProposal:
    """Verifies that valid, high-confidence LLM proposals are accepted safely."""

    def test_valid_high_confidence_proposal_accepted(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        e1_id = repo.insert_entity("Worker Alpha", entity_type="service", created_at=t1)
        e2_id = repo.insert_entity("Worker Beta", entity_type="service", created_at=t1)
        repo.insert_entity_alias("worker alpha", e1_id)
        repo.insert_entity_alias("worker beta", e2_id)

        fake_provider = FakeProvider(
            default_response=json.dumps(
                {
                    "selected_entity_id": e1_id,
                    "confidence": 0.95,
                    "explanation": "Subject refers to primary Worker Alpha",
                    "abstained": False,
                }
            )
        )
        service = make_service_with_fake_llm(repo, base_config, fake_provider)

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Worker runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="Worker",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report = service.ingest_claims(msg, [claim], recorded_at=t1)
        mem_id = report.decisions[0].memory_id
        assert mem_id is not None
        mem = repo.get_memory(mem_id)
        assert mem is not None

        # Accepted candidate ID is exactly e1_id
        assert mem.entity_id == e1_id

        # No 3rd entity was created; exactly 2 entities exist
        assert len(repo.list_entities()) == 2

    def test_accepted_candidate_id_is_strictly_from_supplied_set(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        e1_id = repo.insert_entity("Cluster East", entity_type="cluster", created_at=t1)
        e2_id = repo.insert_entity("Cluster West", entity_type="cluster", created_at=t1)
        repo.insert_entity_alias("cluster east", e1_id)
        repo.insert_entity_alias("cluster west", e2_id)

        fake_provider = FakeProvider(
            default_response=json.dumps(
                {
                    "selected_entity_id": e2_id,
                    "confidence": 0.88,
                    "explanation": "Context refers to West cluster",
                    "abstained": False,
                }
            )
        )
        service = make_service_with_fake_llm(repo, base_config, fake_provider)

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Cluster runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="Cluster",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report = service.ingest_claims(msg, [claim], recorded_at=t1)
        mem_id = report.decisions[0].memory_id
        assert mem_id is not None
        mem = repo.get_memory(mem_id)
        assert mem is not None

        # Selected ID is strictly e2_id from supplied {e1_id, e2_id}
        assert mem.entity_id == e2_id

    def test_no_unrelated_entity_or_alias_modified(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        e1_id = repo.insert_entity("Worker Alpha", entity_type="service", created_at=t1)
        e2_id = repo.insert_entity("Worker Beta", entity_type="service", created_at=t1)
        repo.insert_entity_alias("worker alpha", e1_id)
        repo.insert_entity_alias("worker beta", e2_id)

        fake_provider = FakeProvider(
            default_response=json.dumps(
                {
                    "selected_entity_id": e1_id,
                    "confidence": 0.95,
                    "explanation": "Refers to Worker Alpha",
                    "abstained": False,
                }
            )
        )
        service = make_service_with_fake_llm(repo, base_config, fake_provider)

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Worker runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="Worker",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        service.ingest_claims(msg, [claim], recorded_at=t1)

        # Worker Beta remains completely unchanged
        beta_ent = repo.get_entity(e2_id)
        assert beta_ent is not None
        assert beta_ent["canonical_name"] == "Worker Beta"
        assert beta_ent["needs_review"] == 0

        # Worker Beta's alias was not stolen
        owner = repo.get_entity_by_alias("worker beta")
        assert owner is not None
        assert owner["id"] == e2_id


# ==============================================================================
# 3. Safe Failure Behavior (Fallback to Deterministic Review Entity)
# ==============================================================================


class TestSafeFailureBehavior:
    """Verifies that all failure conditions fall back safely to deterministic review entities."""

    def test_low_confidence_proposal_remains_unresolved(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        e1_id = repo.insert_entity("Worker Alpha", entity_type="service", created_at=t1)
        e2_id = repo.insert_entity("Worker Beta", entity_type="service", created_at=t1)
        repo.insert_entity_alias("worker alpha", e1_id)
        repo.insert_entity_alias("worker beta", e2_id)

        # Confidence 0.65 is below configured threshold 0.80
        fake_provider = FakeProvider(
            default_response=json.dumps(
                {
                    "selected_entity_id": e1_id,
                    "confidence": 0.65,
                    "explanation": "Uncertain match",
                    "abstained": False,
                }
            )
        )
        service = make_service_with_fake_llm(repo, base_config, fake_provider)

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Worker runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="Worker",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report = service.ingest_claims(msg, [claim], recorded_at=t1)
        mem_id = report.decisions[0].memory_id
        assert mem_id is not None
        mem = repo.get_memory(mem_id)
        assert mem is not None

        # Low confidence falls back: creates separate review entity "Worker"
        worker_ent = repo.get_entity(mem.entity_id)
        assert worker_ent is not None
        assert worker_ent["id"] not in (e1_id, e2_id)
        assert worker_ent["canonical_name"] == "Worker"
        assert worker_ent["needs_review"] == 1
        assert len(repo.list_entities()) == 3

    def test_explicit_abstention_remains_unresolved(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        e1_id = repo.insert_entity("Worker Alpha", entity_type="service", created_at=t1)
        e2_id = repo.insert_entity("Worker Beta", entity_type="service", created_at=t1)
        repo.insert_entity_alias("worker alpha", e1_id)
        repo.insert_entity_alias("worker beta", e2_id)

        fake_provider = FakeProvider(
            default_response=json.dumps(
                {
                    "selected_entity_id": None,
                    "confidence": 0.0,
                    "explanation": "Cannot determine between alpha and beta",
                    "abstained": True,
                }
            )
        )
        service = make_service_with_fake_llm(repo, base_config, fake_provider)

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Worker runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="Worker",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report = service.ingest_claims(msg, [claim], recorded_at=t1)
        mem_id = report.decisions[0].memory_id
        assert mem_id is not None
        mem = repo.get_memory(mem_id)
        assert mem is not None

        # Abstention falls back: creates separate review entity
        worker_ent = repo.get_entity(mem.entity_id)
        assert worker_ent is not None
        assert worker_ent["id"] not in (e1_id, e2_id)
        assert worker_ent["needs_review"] == 1

    def test_invalid_candidate_id_is_rejected(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        e1_id = repo.insert_entity("Worker Alpha", entity_type="service", created_at=t1)
        e2_id = repo.insert_entity("Worker Beta", entity_type="service", created_at=t1)
        repo.insert_entity_alias("worker alpha", e1_id)
        repo.insert_entity_alias("worker beta", e2_id)

        # ID 999 is hallucinated / not in supplied candidate set {e1_id, e2_id}
        fake_provider = FakeProvider(
            default_response=json.dumps(
                {
                    "selected_entity_id": 999,
                    "confidence": 0.99,
                    "explanation": "Invented entity ID",
                    "abstained": False,
                }
            )
        )
        service = make_service_with_fake_llm(repo, base_config, fake_provider)

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Worker runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="Worker",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report = service.ingest_claims(msg, [claim], recorded_at=t1)
        mem_id = report.decisions[0].memory_id
        assert mem_id is not None
        mem = repo.get_memory(mem_id)
        assert mem is not None

        # Rejected invalid ID: fallback creates separate review entity
        worker_ent = repo.get_entity(mem.entity_id)
        assert worker_ent is not None
        assert worker_ent["id"] not in (e1_id, e2_id, 999)
        assert worker_ent["needs_review"] == 1

    def test_malformed_json_remains_unresolved(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        e1_id = repo.insert_entity("Worker Alpha", entity_type="service", created_at=t1)
        e2_id = repo.insert_entity("Worker Beta", entity_type="service", created_at=t1)
        repo.insert_entity_alias("worker alpha", e1_id)
        repo.insert_entity_alias("worker beta", e2_id)

        fake_provider = FakeProvider(default_response="MALFORMED_OUTPUT_NOT_JSON {{{")
        service = make_service_with_fake_llm(repo, base_config, fake_provider)

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Worker runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="Worker",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report = service.ingest_claims(msg, [claim], recorded_at=t1)
        mem_id = report.decisions[0].memory_id
        assert mem_id is not None
        mem = repo.get_memory(mem_id)
        assert mem is not None

        # Malformed JSON safely falls back to review entity
        worker_ent = repo.get_entity(mem.entity_id)
        assert worker_ent is not None
        assert worker_ent["id"] not in (e1_id, e2_id)
        assert worker_ent["needs_review"] == 1

    def test_pydantic_validation_failure_remains_unresolved(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        e1_id = repo.insert_entity("Worker Alpha", entity_type="service", created_at=t1)
        e2_id = repo.insert_entity("Worker Beta", entity_type="service", created_at=t1)
        repo.insert_entity_alias("worker alpha", e1_id)
        repo.insert_entity_alias("worker beta", e2_id)

        # Contradictory: abstained is True but selected_entity_id is provided
        fake_provider = FakeProvider(
            default_response=json.dumps(
                {
                    "selected_entity_id": e1_id,
                    "confidence": 0.95,
                    "explanation": "Contradictory proposal",
                    "abstained": True,
                }
            )
        )
        service = make_service_with_fake_llm(repo, base_config, fake_provider)

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Worker runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="Worker",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report = service.ingest_claims(msg, [claim], recorded_at=t1)
        mem_id = report.decisions[0].memory_id
        assert mem_id is not None
        mem = repo.get_memory(mem_id)
        assert mem is not None

        # Schema validation error safely falls back to review entity
        worker_ent = repo.get_entity(mem.entity_id)
        assert worker_ent is not None
        assert worker_ent["id"] not in (e1_id, e2_id)
        assert worker_ent["needs_review"] == 1

    def test_provider_exception_remains_unresolved(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        e1_id = repo.insert_entity("Worker Alpha", entity_type="service", created_at=t1)
        e2_id = repo.insert_entity("Worker Beta", entity_type="service", created_at=t1)
        repo.insert_entity_alias("worker alpha", e1_id)
        repo.insert_entity_alias("worker beta", e2_id)

        class ErrorProvider:
            def complete(self, *args: Any, **kwargs: Any) -> str:
                raise RuntimeError("LLM connection timeout")

        service = MemoryService(
            repo,
            cfg=base_config,
            llm=LLMClient(
                provider=ErrorProvider(),  # type: ignore[arg-type]
                repo=repo,
                config=base_config.engine.llm,
            ),
        )

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Worker runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="Worker",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report = service.ingest_claims(msg, [claim], recorded_at=t1)
        mem_id = report.decisions[0].memory_id
        assert mem_id is not None
        mem = repo.get_memory(mem_id)
        assert mem is not None

        # Provider exception safely caught; falls back to review entity
        worker_ent = repo.get_entity(mem.entity_id)
        assert worker_ent is not None
        assert worker_ent["id"] not in (e1_id, e2_id)
        assert worker_ent["needs_review"] == 1

    def test_unexpected_programming_error_is_not_silently_swallowed(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        e1_id = repo.insert_entity("Worker Alpha", entity_type="service", created_at=t1)
        e2_id = repo.insert_entity("Worker Beta", entity_type="service", created_at=t1)
        repo.insert_entity_alias("worker alpha", e1_id)
        repo.insert_entity_alias("worker beta", e2_id)

        class BuggyProvider:
            def complete(self, *args: Any, **kwargs: Any) -> str:
                raise TypeError("Unexpected code bug in tiebreak")

        service = MemoryService(
            repo,
            cfg=base_config,
            llm=LLMClient(
                provider=BuggyProvider(),  # type: ignore[arg-type]
                repo=repo,
                config=base_config.engine.llm,
            ),
        )

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Worker runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="Worker",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        # Programming bug must propagate and not be swallowed as an ambiguous fallback
        with pytest.raises(TypeError, match="Unexpected code bug in tiebreak"):
            service.ingest_claims(msg, [claim], recorded_at=t1)

    def test_provider_exception_logging_redacts_credentials(
        self,
        repo: Repository,
        base_config: AppConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        e1_id = repo.insert_entity("Worker Alpha", entity_type="service", created_at=t1)
        e2_id = repo.insert_entity("Worker Beta", entity_type="service", created_at=t1)
        repo.insert_entity_alias("worker alpha", e1_id)
        repo.insert_entity_alias("worker beta", e2_id)

        class LeakyCredentialProvider:
            def complete(self, *args: Any, **kwargs: Any) -> str:
                raise RuntimeError(
                    "Anthropic API error with key sk-ant-api03-1234567890abcdef "
                    "and token bearer secret123456"
                )

        service = MemoryService(
            repo,
            cfg=base_config,
            llm=LLMClient(
                provider=LeakyCredentialProvider(),  # type: ignore[arg-type]
                repo=repo,
                config=base_config.engine.llm,
            ),
        )

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Worker runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="Worker",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )

        with caplog.at_level(logging.WARNING):
            service.ingest_claims(msg, [claim], recorded_at=t1)

        # Assert no sensitive credentials leaked into logs
        log_text = caplog.text
        assert "sk-ant-api03" not in log_text
        assert "secret123456" not in log_text
        assert "[REDACTED]" in log_text

    def test_llm_disabled_preserves_existing_deterministic_behavior(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        e1_id = repo.insert_entity("Worker Alpha", entity_type="service", created_at=t1)
        e2_id = repo.insert_entity("Worker Beta", entity_type="service", created_at=t1)
        repo.insert_entity_alias("worker alpha", e1_id)
        repo.insert_entity_alias("worker beta", e2_id)

        # llm=None explicitly disables LLM tiebreak
        service = MemoryService(repo, cfg=base_config, llm=None)

        msg = IngestMessage(
            source_id="sysadmin_priya",
            text="Worker runs Python 3.12",
            event_time=t1,
        )
        claim = Claim(
            subject="Worker",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report = service.ingest_claims(msg, [claim], recorded_at=t1)
        mem_id = report.decisions[0].memory_id
        assert mem_id is not None
        mem = repo.get_memory(mem_id)
        assert mem is not None

        worker_ent = repo.get_entity(mem.entity_id)
        assert worker_ent is not None
        assert worker_ent["id"] not in (e1_id, e2_id)
        assert worker_ent["needs_review"] == 1
        assert len(repo.list_entities()) == 3


# ==============================================================================
# 4. Regression & Safety Invariant Tests
# ==============================================================================


class TestTiebreakRegressionAndSafety:
    """Verifies that similar but distinct entities are NEVER silently merged."""

    def test_similar_distinct_entities_never_silently_merged(
        self,
        repo: Repository,
        base_config: AppConfig,
    ) -> None:
        """Two distinct entities (e.g. Database Primary vs Database Replica)

        must never be merged when an ambiguous reference 'Database' is encountered
        and the tie-break fails or abstains.
        """
        t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 2, 1, 10, 0, tzinfo=UTC)

        e_primary = repo.insert_entity("Database Primary", entity_type="server", created_at=t1)
        e_replica = repo.insert_entity("Database Replica", entity_type="server", created_at=t1)
        repo.insert_entity_alias("database primary", e_primary)
        repo.insert_entity_alias("database replica", e_replica)

        # Ingest a memory for primary
        msg_p = IngestMessage(
            source_id="sysadmin_priya",
            text="Database Primary runs Python 3.10",
            event_time=t1,
        )
        claim_p = Claim(
            subject="Database Primary",
            attribute="python_version",
            value="3.10",
            extract_confidence=0.95,
        )
        service_init = MemoryService(repo, cfg=base_config, llm=None)
        service_init.ingest_claims(msg_p, [claim_p], recorded_at=t1)

        # Now ambiguous subject "Database" with abstaining LLM
        fake_provider = FakeProvider(
            default_response=json.dumps(
                {
                    "selected_entity_id": None,
                    "confidence": 0.0,
                    "explanation": "Cannot distinguish between primary and replica",
                    "abstained": True,
                }
            )
        )
        service = make_service_with_fake_llm(repo, base_config, fake_provider)

        msg_ambig = IngestMessage(
            source_id="sysadmin_priya",
            text="Database runs Python 3.12",
            event_time=t2,
        )
        claim_ambig = Claim(
            subject="Database",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        )
        report_ambig = service.ingest_claims(msg_ambig, [claim_ambig], recorded_at=t2)

        # Ingestion must NOT supersede or overwrite Database Primary
        mem_ambig_id = report_ambig.decisions[0].memory_id
        assert mem_ambig_id is not None
        mem_ambig = repo.get_memory(mem_ambig_id)
        assert mem_ambig is not None

        # Verify Database Primary and Database Replica are intact and distinct
        primary_mem = service.get_current("Database Primary", "python_version", now=t2)
        assert primary_mem is not None
        assert primary_mem.value_norm == "3.10"
        assert primary_mem.entity_id == e_primary

        # Ambiguous memory is isolated on its own review entity
        assert mem_ambig.entity_id not in (e_primary, e_replica)
        ambig_ent = repo.get_entity(mem_ambig.entity_id)
        assert ambig_ent is not None
        assert ambig_ent["needs_review"] == 1

        # Aliases are uncorrupted
        assert repo.get_entity_by_alias("database primary")["id"] == e_primary
        assert repo.get_entity_by_alias("database replica")["id"] == e_replica
