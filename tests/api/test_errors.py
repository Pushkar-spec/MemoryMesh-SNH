"""Unit tests verifying uniform error format and status code mappings.

Governed by docs/SPECIFICATION.md Section 7.1.
All errors follow: {"error": {"code": str, "message": str}}
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from memorymesh.errors import (
    ConflictResolutionError,
    LLMError,
    UnknownSourceError,
)


def test_error_not_found(client: TestClient) -> None:
    """Test 404 maps to {"error": {"code": "not_found", "message": "..."}}."""
    res = client.get("/api/memories/999999")
    assert res.status_code == 404
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "not_found"
    assert "not found" in data["error"]["message"].lower()


def test_error_validation_payload(client: TestClient) -> None:
    """Test 422 validation error on malformed JSON payload."""
    res = client.post("/api/ingest", json={"invalid": "payload"})
    assert res.status_code == 422
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "validation_error"


def test_error_unknown_source(client: TestClient) -> None:
    """Test 422 unknown_source code when source does not exist."""
    with patch(
        "memorymesh.service.MemoryService.ingest",
        side_effect=UnknownSourceError("Source 'nonexistent_bot' is not registered."),
    ):
        res = client.post(
            "/api/ingest",
            json={
                "source_id": "nonexistent_bot",
                "event_time": "2026-06-01T10:00:00Z",
                "text": "Server A runs Python 3.12.",
            },
        )
        assert res.status_code == 422
        data = res.json()
        assert data["error"]["code"] == "unknown_source"
        assert "nonexistent_bot" in data["error"]["message"]


def test_error_conflict(client: TestClient) -> None:
    """Test 409 conflict when conflict cannot be resolved."""
    with patch(
        "memorymesh.service.MemoryService.resolve_dispute",
        side_effect=ConflictResolutionError("Dispute resolution failed."),
    ):
        res = client.post("/api/disputes/resolve", json={"winner_memory_id": 123})
        assert res.status_code == 409
        data = res.json()
        assert data["error"]["code"] == "conflict"


def test_error_llm_unavailable(client: TestClient) -> None:
    """Test 502 bad gateway when LLM provider fails."""
    with patch(
        "memorymesh.service.MemoryService.ask_with_compare",
        side_effect=LLMError("Anthropic API rate limit exceeded."),
    ):
        res = client.post("/api/ask", json={"question": "What version?"})
        assert res.status_code == 502
        data = res.json()
        assert data["error"]["code"] == "llm_unavailable"


def test_error_internal_server_error(client: TestClient) -> None:
    """Test 500 internal error does not leak secrets or stack traces."""
    secret_token = "SUPER_SECRET_TOKEN_DO_NOT_LEAK"
    with patch(
        "memorymesh.service.MemoryService.get_catalog",
        side_effect=RuntimeError(f"Database crash with secret: {secret_token}"),
    ):
        res = client.get("/api/catalog")
        assert res.status_code == 500
        data = res.json()
        assert data["error"]["code"] == "internal"
        assert secret_token not in data["error"]["message"]
        assert "internal server error" in data["error"]["message"].lower()


def test_error_unrouted_path_404(client: TestClient) -> None:
    """Test Starlette 404 for unrouted paths follows uniform error envelope."""
    res = client.get("/api/completely_nonexistent_route")
    assert res.status_code == 404
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "not_found"
    assert data["error"]["message"] == "Not Found"


def test_error_method_not_allowed_405(client: TestClient) -> None:
    """Test Starlette 405 Method Not Allowed follows uniform error envelope."""
    res = client.post("/api/health")
    assert res.status_code == 405
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "method_not_allowed"
    assert data["error"]["message"] == "Method Not Allowed"
