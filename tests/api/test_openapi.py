"""Tests verifying OpenAPI schema generation and docs availability.

Governed by docs/SPECIFICATION.md Section 7.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_openapi_json(client: TestClient) -> None:
    """Test GET /openapi.json serves valid OpenAPI specification."""
    res = client.get("/openapi.json")
    assert res.status_code == 200
    data = res.json()
    assert data["info"]["title"] == "MemoryMesh API"
    assert data["info"]["version"] == "0.1.0"
    paths = data["paths"]

    # Verify all required endpoints from SPECIFICATION.md Section 7.2 are present
    expected_endpoints = [
        ("/api/ingest", "post"),
        ("/api/memories/{id}/acknowledge", "post"),
        ("/api/disputes/resolve", "post"),
        ("/api/demo/reset", "post"),
        ("/api/health", "get"),
        ("/api/catalog", "get"),
        ("/api/memories", "get"),
        ("/api/memories/{id}", "get"),
        ("/api/memories/{id}/explain", "get"),
        ("/api/memories/{id}/impact", "get"),
        ("/api/timeline", "get"),
        ("/api/graph", "get"),
        ("/api/disputes", "get"),
        ("/api/decisions", "get"),
        ("/api/llm/calls", "get"),
        ("/api/ask", "post"),
        ("/api/eval/latest", "get"),
        ("/api/eval/latest/report", "get"),
    ]

    for path, method in expected_endpoints:
        assert path in paths, f"Missing endpoint in OpenAPI schema: {path}"
        assert method in paths[path], f"Missing {method.upper()} for endpoint: {path}"


def test_swagger_docs_html(client: TestClient) -> None:
    """Test GET /docs returns interactive Swagger UI HTML."""
    res = client.get("/docs")
    assert res.status_code == 200
    assert "swagger-ui" in res.text.lower() or "html" in res.headers["content-type"]
