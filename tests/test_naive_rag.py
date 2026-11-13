"""Unit and integration tests for the Naive RAG baseline (P8.1).

Governed by:
- docs/EVALUATION_AND_DEMO.md Section 1, 4, 5
- MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P8.1)

Verifies:
1. FastEmbed embedding generation and cosine similarity top-5 ranking.
2. Empty message collection and fewer than 5 messages behavior.
3. Message formatting with timestamps and source names.
4. Prompt construction using baseline_answer template.
5. Answer generation and FinalAnswer schema validation.
6. Fallback handling on provider failure or malformed JSON.
7. Complete independence from MemoryMesh resolution engine.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest

from baseline.naive_rag import NaiveRAGBaseline
from eval.metrics import evaluate_question
from memorymesh.config import load_app_config
from memorymesh.errors import BaselineEmbeddingError
from memorymesh.llm.client import LLMClient
from memorymesh.llm.prompts import (
    BASELINE_ANSWER_SYSTEM,
    build_baseline_answer_prompt,
    format_baseline_messages,
)
from memorymesh.llm.providers import LLMProvider
from memorymesh.models import FinalAnswer


class MockProvider(LLMProvider):
    """Mock LLM provider for baseline answer testing."""

    def __init__(self, response_json: dict[str, Any] | None = None, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = fail
        self.response_json = response_json or {
            "value": "3.12",
            "values": ["3.10", "3.12"],
            "conflict_flagged": False,
            "needs_review_flagged": False,
            "abstained": False,
            "explanation": "Priya stated Server A runs Python 3.12.",
            "cited_memory_ids": [],
            "answer_text": "Server A runs Python 3.12.",
        }

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append({"system": system, "user": user, "model": model})
        if self.fail:
            raise RuntimeError("Provider connection timeout")
        return json.dumps(self.response_json)


class DummyEmbeddingModel:
    """Deterministic mock embedding model for fast non-ONNX testing."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Produce simple deterministic vectors
        vectors: list[list[float]] = []
        for text in texts:
            t = text.lower()
            v = [
                float("python" in t),
                float("3.12" in t),
                float("server a" in t),
                float("deploy" in t),
            ]
            vectors.append(v)
        return vectors


def test_naive_rag_retrieval_ranking() -> None:
    """Verifies that NaiveRAGBaseline ranks relevant messages highest via cosine similarity."""
    baseline = NaiveRAGBaseline(embedding_model=DummyEmbeddingModel())
    messages = [
        {
            "id": "m1",
            "t": "2026-01-10T09:00:00Z",
            "source": "priya",
            "text": "Server A runs Python 3.10",
        },
        {
            "id": "m2",
            "t": "2026-06-01T08:00:00Z",
            "source": "priya",
            "text": "Server A runs Python 3.12",
        },
        {
            "id": "m3",
            "t": "2026-02-10T10:00:00Z",
            "source": "priya",
            "text": "Database backup completed",
        },
    ]
    baseline.index_messages(messages)

    results = baseline.retrieve("What Python 3.12 version on Server A?", top_k=2)
    assert len(results) == 2
    # m2 has python, 3.12, server a -> highest score
    assert results[0][0]["id"] == "m2"
    assert results[1][0]["id"] == "m1"


def test_naive_rag_fewer_than_k_and_empty() -> None:
    """Verifies retrieval when index has fewer messages than top_k or is empty."""
    baseline = NaiveRAGBaseline(embedding_model=DummyEmbeddingModel())
    assert baseline.retrieve("Python?", top_k=5) == []

    baseline.index_messages(
        [{"id": "m1", "t": "2026-01-10T09:00:00Z", "source": "priya", "text": "Only one"}]
    )
    res = baseline.retrieve("Only one", top_k=5)
    assert len(res) == 1
    assert res[0][0]["id"] == "m1"


def test_format_baseline_messages() -> None:
    """Verifies raw message formatting with ISO timestamps and sources."""
    msgs = [
        {
            "id": "m1",
            "source": "priya",
            "t": "2026-01-10T09:00:00Z",
            "text": "Server A runs Python 3.10",
        },
        {"id": "m2", "source": "wiki", "t": "2026-02-02T11:30:00Z", "text": "Infra wiki note"},
    ]
    formatted = format_baseline_messages(msgs)
    assert "[m1] (2026-01-10T09:00:00Z, source: priya): Server A runs Python 3.10" in formatted
    assert "[m2] (2026-02-02T11:30:00Z, source: wiki): Infra wiki note" in formatted


def test_build_baseline_answer_prompt() -> None:
    """Verifies prompt construction with question, retrieved messages, and reference time."""
    msgs = [{"id": "m1", "source": "priya", "t": "2026-01-10T09:00:00Z", "text": "Hello"}]
    ref_time = datetime(2026, 6, 25, 9, 0, tzinfo=UTC)
    sys_prompt, user_prompt = build_baseline_answer_prompt(
        question="What runs?",
        messages=msgs,
        reference_time=ref_time,
    )
    assert sys_prompt == BASELINE_ANSWER_SYSTEM
    assert "2026-06-25T09:00:00Z" in user_prompt
    assert "What runs?" in user_prompt
    assert "[m1] (2026-01-10T09:00:00Z, source: priya): Hello" in user_prompt


def test_naive_rag_ask_success() -> None:
    """Verifies end-to-end question answering via NaiveRAGBaseline returning FinalAnswer."""
    mock_provider = MockProvider()
    cfg = load_app_config()
    llm_client = LLMClient(provider=mock_provider, config=cfg.engine.llm)
    baseline = NaiveRAGBaseline(
        llm_client=llm_client,
        embedding_model=DummyEmbeddingModel(),
        model="claude-sonnet-5-5",
    )

    baseline.index_messages(
        [
            {
                "id": "m1",
                "t": "2026-01-10T09:00:00Z",
                "source": "priya",
                "text": "Server A runs Python 3.10.",
            },
            {
                "id": "m2",
                "t": "2026-06-01T08:00:00Z",
                "source": "priya",
                "text": "Migrated Server A to Python 3.12.",
            },
        ]
    )

    ans = baseline.ask("What Python version does Server A use?")
    assert isinstance(ans, FinalAnswer)
    assert ans.value == "3.12"
    assert ans.answer_text == "Server A runs Python 3.12."
    assert len(mock_provider.calls) == 1
    # Verify model tier used is answer_model (claude-sonnet-5-5)
    assert mock_provider.calls[0]["model"] == "claude-sonnet-5-5"


def test_naive_rag_ask_provider_failure_fallback() -> None:
    """Verifies fallback abstention when LLM provider fails."""
    mock_provider = MockProvider(fail=True)
    cfg = load_app_config()
    llm_client = LLMClient(provider=mock_provider, config=cfg.engine.llm)
    baseline = NaiveRAGBaseline(
        llm_client=llm_client,
        embedding_model=DummyEmbeddingModel(),
    )
    baseline.index_messages(
        [{"id": "m1", "t": "2026-01-10T09:00:00Z", "source": "priya", "text": "Server A"}]
    )

    ans = baseline.ask("What Python version?")
    assert ans.abstained is True
    assert "Baseline answer generation failed" in ans.explanation


def test_naive_rag_empty_collection_abstains() -> None:
    """Verifies baseline abstains without calling LLM when no messages are indexed."""
    mock_provider = MockProvider()
    cfg = load_app_config()
    llm_client = LLMClient(provider=mock_provider, config=cfg.engine.llm)
    baseline = NaiveRAGBaseline(
        llm_client=llm_client,
        embedding_model=DummyEmbeddingModel(),
    )
    ans = baseline.ask("Any memories?")
    assert ans.abstained is True
    assert len(mock_provider.calls) == 0


def test_naive_rag_independence_from_resolution_engine() -> None:
    """Verifies baseline operates without importing or invoking resolution engine rules."""
    baseline = NaiveRAGBaseline(embedding_model=DummyEmbeddingModel())
    # Verify baseline has no repository, no engine, no entity resolver
    assert not hasattr(baseline, "repo")
    assert not hasattr(baseline, "engine")
    assert not hasattr(baseline, "rules")


def test_fastembed_loading_failure_raises_typed_error() -> None:
    """Verifies that failure to import or load FastEmbed raises BaselineEmbeddingError."""
    baseline = NaiveRAGBaseline()
    with patch(
        "fastembed.TextEmbedding",
        side_effect=ImportError("Weights unavailable or not installed"),
    ):
        with pytest.raises(BaselineEmbeddingError) as exc_info:
            _ = baseline.embedding_model
        assert "FastEmbed embedding model is not available" in str(exc_info.value)


def test_normal_b0_mode_never_silently_switches_to_lexical() -> None:
    """Verifies that normal FastEmbed B0 baseline raises BaselineEmbeddingError on failure

    and never silently substitutes lexical/token similarity.
    """

    class BrokenEmbeddingModel:
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("ONNX runtime missing or model weights download failed")

    baseline = NaiveRAGBaseline(
        embedding_model=BrokenEmbeddingModel(),
        retrieval_method="fastembed",
    )
    messages = [
        {"id": "m1", "text": "Server A runs Python 3.10"},
        {"id": "m2", "text": "Server A migrated to Python 3.12"},
    ]

    # Must raise BaselineEmbeddingError, not fallback to lexical
    with pytest.raises(BaselineEmbeddingError) as exc_info:
        baseline.index_messages(messages)
    assert "FastEmbed embedding generation failed" in str(exc_info.value)
    assert baseline.embeddings is None


def test_explicit_fallback_mode_identified_separately() -> None:
    """Verifies explicit fallback is separately identified and never reported as FastEmbed."""
    baseline = NaiveRAGBaseline(retrieval_method="lexical")
    assert baseline.retrieval_method == "lexical"

    messages = [
        {"id": "m1", "text": "Server A runs Python 3.10"},
        {"id": "m2", "text": "Server A runs Python 3.12"},
    ]
    baseline.index_messages(messages)
    results = baseline.retrieve("Python 3.12", top_k=1)
    assert len(results) == 1
    assert results[0][0]["id"] == "m2"

    with pytest.raises(ValueError, match="Unsupported retrieval method"):
        NaiveRAGBaseline(retrieval_method="unknown_method")


def test_naive_rag_hero_q1_rag_failure() -> None:
    """Verifies naive RAG failure on hero scenario question q1.

    Indexed messages:
    - m1 (Priya, Jan 10): "Server A runs Python 3.10."
    - m2 (Wiki, Feb 2): "Runtime for Server A: Python 3.10 (see infra page)."
    - m5 (Priya, Jun 1): "We migrated Server A to Python 3.12 today."
    - m6 (Intern, Jun 3): "I think Server A is still on Python 3.9?"

    Demonstrates:
    1. Retrieval returns conflicting raw messages (m1, m2, m5, m6).
    2. Context passed to answer generation contains superseded versions (3.10)
       and rumors (3.9) alongside the current migration (3.12).
    3. When evaluated against the canonical expectation (value="3.12",
       forbidden=["3.10", "3.9"]), an answer selecting the stale version (3.10)
       fails current accuracy and is flagged as stale.
    """
    mock_provider = MockProvider(
        response_json={
            "value": "3.10",
            "values": ["3.10"],
            "conflict_flagged": False,
            "needs_review_flagged": False,
            "abstained": False,
            "explanation": "Server A runs Python 3.10 according to Priya and wiki.",
            "cited_memory_ids": [],
            "answer_text": "Server A runs Python 3.10.",
        }
    )
    cfg = load_app_config()
    llm_client = LLMClient(provider=mock_provider, config=cfg.engine.llm)
    baseline = NaiveRAGBaseline(
        llm_client=llm_client,
        embedding_model=DummyEmbeddingModel(),
    )
    hero_messages = [
        {
            "id": "m1",
            "t": "2026-01-10T09:00:00Z",
            "source": "sysadmin_priya",
            "text": "Server A runs Python 3.10.",
        },
        {
            "id": "m2",
            "t": "2026-02-02T11:30:00Z",
            "source": "wiki",
            "text": "Runtime for Server A: Python 3.10 (see infra page).",
        },
        {
            "id": "m5",
            "t": "2026-06-01T08:00:00Z",
            "source": "sysadmin_priya",
            "text": "We migrated Server A to Python 3.12 today.",
        },
        {
            "id": "m6",
            "t": "2026-06-03T16:20:00Z",
            "source": "intern_chat",
            "text": "I think Server A is still on Python 3.9?",
        },
    ]
    baseline.index_messages(hero_messages)

    # 1. Retrieval audit: verify which messages are actually retrieved
    retrieved = baseline.retrieve("What Python version does Server A use?", top_k=4)
    retrieved_ids = [item[0]["id"] for item in retrieved]
    retrieved_texts = [item[0]["text"] for item in retrieved]

    assert len(retrieved) == 4
    assert "m1" in retrieved_ids  # Stale 3.10
    assert "m2" in retrieved_ids  # Stale 3.10
    assert "m5" in retrieved_ids  # Current 3.12
    assert "m6" in retrieved_ids  # Rumor 3.9
    assert any("3.10" in t for t in retrieved_texts)
    assert any("3.12" in t for t in retrieved_texts)

    # 2. Context verification: prompt receives conflicting versions
    sys_prompt, user_prompt = build_baseline_answer_prompt(
        question="What Python version does Server A use?",
        messages=[item[0] for item in retrieved],
        reference_time=datetime(2026, 6, 25, 9, 0, tzinfo=UTC),
    )
    assert "[m1]" in user_prompt
    assert "[m5]" in user_prompt
    assert "Python 3.10" in user_prompt
    assert "Python 3.12" in user_prompt

    # 3. Answer-generation step: deterministic mock generates stale answer (3.10)
    ans = baseline.ask(
        "What Python version does Server A use?",
        reference_time=datetime(2026, 6, 25, 9, 0, tzinfo=UTC),
    )
    assert ans.value == "3.10"

    # 4. Metric evaluation against canonical expectation
    q1 = {
        "id": "q1",
        "type": "current",
        "text": "What Python version does Server A use?",
        "expect": {"value": "3.12", "forbidden": ["3.10", "3.9"]},
    }
    ans_dict = {
        "value": ans.value,
        "values": ans.values,
        "answer_text": ans.answer_text,
    }
    result = evaluate_question("hero-001", q1, ans_dict, is_baseline=True)

    # 5. Exact failure assertions
    assert result.is_current_correct is False
    assert result.is_stale is True


def test_naive_rag_staging_context_contamination() -> None:
    """Verifies that naive RAG contaminates production queries with staging context (m8).

    Unlike MemoryMesh which isolates chains by context (e.g. {'env': 'staging'}),
    naive RAG indexes raw messages globally. When querying for the production Python
    version, staging messages mentioning Python on Server A are retrieved into context.
    """
    baseline = NaiveRAGBaseline(embedding_model=DummyEmbeddingModel())
    messages = [
        {
            "id": "m1",
            "t": "2026-01-10T09:00:00Z",
            "source": "sysadmin_priya",
            "text": "Server A runs Python 3.10.",
        },
        {
            "id": "m5",
            "t": "2026-06-01T08:00:00Z",
            "source": "sysadmin_priya",
            "text": "We migrated Server A to Python 3.12 today.",
        },
        {
            "id": "m8",
            "t": "2026-06-20T13:00:00Z",
            "source": "dev_chat",
            "text": "In staging, Server A still runs Python 3.11.",
        },
    ]
    baseline.index_messages(messages)

    retrieved = baseline.retrieve("What Python version does Server A use?", top_k=3)
    retrieved_ids = [item[0]["id"] for item in retrieved]

    # Staging message m8 is retrieved into the production query context
    assert "m8" in retrieved_ids
    assert any("staging" in item[0]["text"] for item in retrieved)
    assert any("3.11" in item[0]["text"] for item in retrieved)
