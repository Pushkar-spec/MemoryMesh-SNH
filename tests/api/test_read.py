"""Unit and integration tests for read-path API routes.

Governed by docs/SPECIFICATION.md Section 7.2.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _seed_sample_memories(client: TestClient) -> dict[str, int]:
    """Helper to seed sample memories into the test database."""
    res1 = client.post(
        "/api/ingest",
        json={
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
                    "extract_confidence": 0.90,
                }
            ],
        },
    )
    res2 = client.post(
        "/api/ingest",
        json={
            "source_id": "sysadmin_priya",
            "event_time": "2026-06-01T11:00:00Z",
            "text": "Server A was upgraded to Python 3.12.",
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
    return {
        "m1": res1.json()["decisions"][0]["memory_id"],
        "m2": res2.json()["decisions"][0]["memory_id"],
    }


def test_get_health(client: TestClient) -> None:
    """Test GET /api/health returns status, llm_provider, and db."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["llm_provider"] == "fake"
    assert "db" in data


def test_get_catalog(client: TestClient) -> None:
    """Test GET /api/catalog returns sources, attributes, entities, thresholds, ablation."""
    response = client.get("/api/catalog")
    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) >= 1
    assert any(s["id"] == "sysadmin_priya" for s in data["sources"])
    assert len(data["attributes"]) >= 1
    assert any(a["attribute"] == "python_version" for a in data["attributes"])
    assert len(data["entities"]) >= 1
    assert any(e["name"] == "Server A" for e in data["entities"])
    assert "stale_floor" in data["thresholds"]
    assert "use_temporal" in data["ablation"]


def test_list_memories_and_filters(client: TestClient) -> None:
    """Test GET /api/memories with filtering and pagination."""
    ids = _seed_sample_memories(client)

    # List all
    res = client.get("/api/memories")
    assert res.status_code == 200
    mems = res.json()
    assert len(mems) >= 2

    # Filter by entity
    res_ent = client.get("/api/memories?entity=Server A")
    assert res_ent.status_code == 200
    assert len(res_ent.json()) >= 2

    # Filter by attribute
    res_attr = client.get("/api/memories?attribute=python_version")
    assert res_attr.status_code == 200
    assert len(res_attr.json()) >= 2

    # Filter by status: active vs historical
    res_active = client.get("/api/memories?status=active")
    assert res_active.status_code == 200
    active_mems = res_active.json()
    assert any(m["memory"]["id"] == ids["m2"] for m in active_mems)

    # Test status case-insensitivity
    res_active_upper = client.get("/api/memories?status=ACTIVE")
    assert res_active_upper.status_code == 200
    assert any(m["memory"]["id"] == ids["m2"] for m in res_active_upper.json())

    # Test invalid status returns 422 validation_error
    res_invalid_status = client.get("/api/memories?status=unknown_status")
    assert res_invalid_status.status_code == 422
    assert res_invalid_status.json()["error"]["code"] == "validation_error"

    res_hist = client.get("/api/memories?status=historical")
    assert res_hist.status_code == 200
    hist_mems = res_hist.json()
    assert any(m["memory"]["id"] == ids["m1"] for m in hist_mems)

    # Pagination
    res_page = client.get("/api/memories?limit=1&offset=0")
    assert res_page.status_code == 200
    assert len(res_page.json()) == 1


def test_now_query_parameter_validation(client: TestClient) -> None:
    """Test now query parameter accepts UTC and offset datetimes and rejects naive."""
    # Valid UTC
    res_utc = client.get("/api/memories?now=2026-06-01T12:00:00Z")
    assert res_utc.status_code == 200

    # Valid non-UTC offset
    res_offset = client.get("/api/memories?now=2026-06-01T14:00:00%2B02:00")
    assert res_offset.status_code == 200

    # Timezone-naive datetime rejected with 422
    res_naive = client.get("/api/memories?now=2026-06-01T12:00:00")
    assert res_naive.status_code == 422
    data_naive = res_naive.json()
    assert data_naive["error"]["code"] == "validation_error"
    assert "timezone-aware" in data_naive["error"]["message"].lower()

    # Invalid datetime syntax rejected with 422
    res_bad = client.get("/api/memories?now=not-a-datetime")
    assert res_bad.status_code == 422
    assert res_bad.json()["error"]["code"] == "validation_error"


def test_get_memory_detail_and_404(client: TestClient) -> None:
    """Test GET /api/memories/{id} returns MemoryView or 404."""
    ids = _seed_sample_memories(client)
    res = client.get(f"/api/memories/{ids['m2']}")
    assert res.status_code == 200
    data = res.json()
    assert data["memory"]["id"] == ids["m2"]
    assert data["memory"]["value_norm"] == "3.12"
    assert "effective_confidence" in data
    assert "is_stale" in data

    # 404 test
    res404 = client.get("/api/memories/99999")
    assert res404.status_code == 404
    assert res404.json()["error"]["code"] == "not_found"


def test_explain_memory_and_404(client: TestClient) -> None:
    """Test GET /api/memories/{id}/explain returns QueryResult."""
    ids = _seed_sample_memories(client)
    res = client.get(f"/api/memories/{ids['m2']}/explain")
    assert res.status_code == 200
    data = res.json()
    assert "plan" in data
    assert "items" in data
    assert any(item["memory"]["id"] == ids["m2"] for item in data["items"])

    res404 = client.get("/api/memories/99999/explain")
    assert res404.status_code == 404
    assert res404.json()["error"]["code"] == "not_found"


def test_memory_impact_and_404(client: TestClient) -> None:
    """Test GET /api/memories/{id}/impact returns list[ImpactItem]."""
    ids = _seed_sample_memories(client)
    res = client.get(f"/api/memories/{ids['m2']}/impact")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    res404 = client.get("/api/memories/99999/impact")
    assert res404.status_code == 404
    assert res404.json()["error"]["code"] == "not_found"


def test_get_timeline(client: TestClient) -> None:
    """Test GET /api/timeline returns SlotTimeline lanes."""
    _seed_sample_memories(client)
    res = client.get("/api/timeline?entity=Server A&attribute=python_version")
    assert res.status_code == 200
    timelines = res.json()
    assert len(timelines) >= 1
    slot = timelines[0]
    assert slot["entity"] == "Server A"
    assert slot["attribute"] == "python_version"
    assert len(slot["memories"]) >= 1


def test_get_graph(client: TestClient) -> None:
    """Test GET /api/graph returns GraphNode and GraphEdge."""
    ids = _seed_sample_memories(client)
    res = client.get("/api/graph")
    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert "edges" in data

    node_ids = {n["id"] for n in data["nodes"]}
    assert any(nid.startswith("entity:") for nid in node_ids)
    assert f"mem:{ids['m1']}" in node_ids
    assert f"mem:{ids['m2']}" in node_ids

    # Check 'about' and 'supersedes' edges
    edge_types = {e["type"] for e in data["edges"]}
    assert "about" in edge_types
    assert "supersedes" in edge_types


def test_get_decisions(client: TestClient) -> None:
    """Test GET /api/decisions returns DecisionLogEntry list."""
    _seed_sample_memories(client)
    res = client.get("/api/decisions")
    assert res.status_code == 200
    decisions = res.json()
    assert len(decisions) >= 2

    # Filter by rule_id
    res_rule = client.get("/api/decisions?rule_id=R-INSERT")
    assert res_rule.status_code == 200
    assert all(d["rule_id"] == "R-INSERT" for d in res_rule.json())
