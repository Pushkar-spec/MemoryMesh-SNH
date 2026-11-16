"""Unit and integration tests for GET /api/llm/calls route.

Governed by docs/SPECIFICATION.md Section 7.2.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from memorymesh.models import LLMCall, LLMCallStatus


def test_get_llm_calls_and_filters(client: TestClient) -> None:
    """Test GET /api/llm/calls with status and prompt_id filtering."""
    service = client.app.state.service  # type: ignore[attr-defined]

    # Seed mock LLM calls into repository
    call1 = LLMCall(
        id="call-001",
        timestamp=datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC),
        prompt_id="extract_claims",
        provider="fake",
        model="fake-fast",
        prompt_summary="Extract claims from message",
        response_raw='{"claims": []}',
        parsed_json='{"claims": []}',
        latency_ms=12.5,
        status=LLMCallStatus.SUCCESS,
        error=None,
    )
    call2 = LLMCall(
        id="call-002",
        timestamp=datetime(2026, 6, 1, 10, 1, 0, tzinfo=UTC),
        prompt_id="answer_query",
        provider="fake",
        model="fake-answer",
        prompt_summary="Compose final answer",
        response_raw="Error from provider",
        parsed_json=None,
        latency_ms=45.0,
        status=LLMCallStatus.ERROR,
        error="Simulated provider error",
    )
    service.repo.insert_llm_call(call1)
    service.repo.insert_llm_call(call2)

    # Fetch all
    res = client.get("/api/llm/calls")
    assert res.status_code == 200
    calls = res.json()
    assert len(calls) >= 2

    # Filter by prompt_id
    res_prompt = client.get("/api/llm/calls?prompt_id=extract_claims")
    assert res_prompt.status_code == 200
    assert all(c["prompt_id"] == "extract_claims" for c in res_prompt.json())

    # Filter by status
    res_status = client.get("/api/llm/calls?status=error")
    assert res_status.status_code == 200
    assert all(c["status"] == "error" for c in res_status.json())

    # Verify parsed_json is deserialized
    item1 = next(c for c in calls if c["id"] == "call-001")
    assert item1["parsed_json"] == {"claims": []}
