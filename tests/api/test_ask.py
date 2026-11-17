"""Unit and integration tests for POST /api/ask route.

Governed by docs/SPECIFICATION.md Section 7.2.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_ask_basic_query(client: TestClient) -> None:
    """Test POST /api/ask returns AskResponse with mesh result."""
    # Ingest a memory
    client.post(
        "/api/ingest",
        json={
            "source_id": "sysadmin_priya",
            "event_time": "2026-06-01T10:00:00Z",
            "text": "Server A runs Python 3.12.",
            "mode": "structured",
            "claims": [
                {
                    "subject": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "polarity": "assert",
                    "extract_confidence": 0.90,
                }
            ],
        },
    )

    payload = {
        "question": "What version of Python is running on Server A?",
        "now": "2026-06-01T12:00:00Z",
        "compare": False,
    }
    response = client.post("/api/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "mesh" in data
    assert data["mesh"]["question"] == payload["question"]
    assert "result" in data["mesh"]
    assert "answer" in data["mesh"]
    assert data["baseline"] is None
    assert data["baseline_retrieved"] == []


def test_ask_with_naive_rag_compare(client: TestClient) -> None:
    """Test POST /api/ask with compare=True returns both MemoryMesh and baseline answers."""
    # Ingest message
    client.post(
        "/api/ingest",
        json={
            "source_id": "sysadmin_priya",
            "event_time": "2026-06-01T10:00:00Z",
            "text": "Server A was upgraded to Python 3.12 by Priya.",
            "mode": "structured",
            "claims": [
                {
                    "subject": "Server A",
                    "attribute": "python_version",
                    "value": "3.12",
                    "polarity": "assert",
                    "extract_confidence": 0.90,
                }
            ],
        },
    )

    payload = {
        "question": "What is the Python version on Server A?",
        "now": "2026-06-01T12:00:00Z",
        "compare": True,
    }
    response = client.post("/api/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "mesh" in data
    assert data["baseline"] is not None
    assert isinstance(data["baseline_retrieved"], list)
    assert len(data["baseline_retrieved"]) >= 1
    hit = data["baseline_retrieved"][0]
    assert "message_id" in hit
    assert "source_name" in hit
    assert "score" in hit


def test_ask_now_timezone_validation(client: TestClient) -> None:
    """Test POST /api/ask rejects timezone-naive now timestamps."""
    payload_naive = {
        "question": "What version of Python?",
        "now": "2026-06-01T12:00:00",
    }
    res_naive = client.post("/api/ask", json=payload_naive)
    assert res_naive.status_code == 422
    data = res_naive.json()
    assert data["error"]["code"] == "validation_error"
    assert "timezone-aware" in data["error"]["message"].lower()

    # Valid non-UTC offset works
    payload_offset = {
        "question": "What version of Python?",
        "now": "2026-06-01T14:00:00+02:00",
    }
    res_offset = client.post("/api/ask", json=payload_offset)
    assert res_offset.status_code == 200


def test_ask_sequential_success_then_failure(client: TestClient) -> None:
    """Verify first ask succeeds and second ask fails with validation error."""
    # First question succeeds
    res_success = client.post(
        "/api/ask",
        json={"question": "What version?", "now": "2026-06-01T12:00:00Z"},
    )
    assert res_success.status_code == 200
    assert "mesh" in res_success.json()

    # Second question fails validation
    res_fail = client.post("/api/ask", json={"question": "   ", "now": "invalid-time"})
    assert res_fail.status_code == 422
    assert res_fail.json()["error"]["code"] == "validation_error"
