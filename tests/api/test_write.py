"""Unit and integration tests for write-path API routes.

Governed by docs/SPECIFICATION.md Section 7.2.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_ingest_structured_claims(client: TestClient) -> None:
    """Test POST /api/ingest with mode='structured' and explicit claims."""
    payload = {
        "source_id": "sysadmin_priya",
        "event_time": "2026-06-01T10:00:00Z",
        "text": "Server A was upgraded to Python 3.12.",
        "meta": {},
        "mode": "structured",
        "claims": [
            {
                "subject": "Server A",
                "attribute": "python_version",
                "value": "3.12",
                "polarity": "assert",
                "extract_confidence": 0.95,
                "context": {},
                "depends_on": [],
                "span": "upgraded to Python 3.12",
            }
        ],
    }

    response = client.post("/api/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["claims_extracted"] == 1
    assert data["duplicate"] is False
    assert len(data["decisions"]) >= 1
    assert data["decisions"][0]["action"] == "INSERT"


def test_acknowledge_memory_review(client: TestClient) -> None:
    """Test POST /api/memories/{id}/acknowledge."""
    # First ingest a claim
    ingest_payload = {
        "source_id": "sysadmin_priya",
        "event_time": "2026-06-01T10:00:00Z",
        "text": "Server A runs Python 3.11.",
        "mode": "structured",
        "claims": [
            {
                "subject": "Server A",
                "attribute": "python_version",
                "value": "3.11",
                "polarity": "assert",
                "extract_confidence": 0.95,
            }
        ],
    }
    ingest_res = client.post("/api/ingest", json=ingest_payload)
    assert ingest_res.status_code == 200
    mem_id = ingest_res.json()["decisions"][0]["memory_id"]

    # Flag it for review via repository directly
    service = client.app.state.service  # type: ignore[attr-defined]
    service.repo.flag_memory_review(mem_id, "Test review needed")

    # Verify flagged in memory view
    mem_res = client.get(f"/api/memories/{mem_id}")
    assert mem_res.status_code == 200
    assert mem_res.json()["memory"]["needs_review"] is True

    # Acknowledge review via API
    ack_res = client.post(f"/api/memories/{mem_id}/acknowledge")
    assert ack_res.status_code == 200
    ack_data = ack_res.json()
    assert ack_data["id"] == mem_id
    assert ack_data["needs_review"] is False

    # Check that GET now reflects acknowledged state
    mem_res2 = client.get(f"/api/memories/{mem_id}")
    assert mem_res2.json()["memory"]["needs_review"] is False


def test_acknowledge_nonexistent_memory(client: TestClient) -> None:
    """Test POST /api/memories/{id}/acknowledge with nonexistent ID returns 404."""
    response = client.post("/api/memories/99999/acknowledge")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "not_found"


def test_resolve_dispute(client: TestClient) -> None:
    """Test POST /api/disputes/resolve."""
    # Ingest two conflicting claims from same-reliability sources to trigger DISPUTE
    msg1 = {
        "source_id": "ci_monitor",
        "event_time": "2026-06-01T10:00:00Z",
        "text": "Server A deploy day is Tuesday.",
        "mode": "structured",
        "claims": [
            {
                "subject": "Server A",
                "attribute": "deploy_day",
                "value": "tuesday",
                "polarity": "assert",
                "extract_confidence": 1.0,
            }
        ],
    }
    msg2 = {
        "source_id": "dev_chat",
        "event_time": "2026-06-01T10:05:00Z",
        "text": "Server A deploy day is Thursday.",
        "mode": "structured",
        "claims": [
            {
                "subject": "Server A",
                "attribute": "deploy_day",
                "value": "thursday",
                "polarity": "assert",
                "extract_confidence": 1.0,
            }
        ],
    }
    r1 = client.post("/api/ingest", json=msg1)
    r2 = client.post("/api/ingest", json=msg2)
    assert r1.status_code == 200
    assert r2.status_code == 200

    mem1_id = r1.json()["decisions"][0]["memory_id"]
    mem2_id = r2.json()["decisions"][0]["memory_id"]

    # Verify dispute list shows the dispute
    disputes_res = client.get("/api/disputes")
    assert disputes_res.status_code == 200
    disputes = disputes_res.json()
    assert len(disputes) >= 1

    # Resolve dispute picking mem2 as winner
    resolve_res = client.post("/api/disputes/resolve", json={"winner_memory_id": mem2_id})
    assert resolve_res.status_code == 200
    decision = resolve_res.json()
    assert decision["rule_id"] == "R-DISPUTE-RESOLVE"
    assert decision["memory_id"] == mem2_id

    # Verify mem2 is active and mem1 is historical
    m1 = client.get(f"/api/memories/{mem1_id}").json()
    m2 = client.get(f"/api/memories/{mem2_id}").json()
    assert m2["memory"]["status"] == "active"
    assert m1["memory"]["status"] == "historical"


def test_demo_reset(client: TestClient) -> None:
    """Test POST /api/demo/reset."""
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
    assert len(client.get("/api/memories").json()) >= 1

    # Call reset
    res = client.post("/api/demo/reset")
    assert res.status_code == 200
    assert res.json() == {"ok": True}

    # Verify empty memories
    assert len(client.get("/api/memories").json()) == 0
