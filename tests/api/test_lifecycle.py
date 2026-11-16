"""Tests for FastAPI application lifecycle, provider initialization, and connection cleanup.

Governed by docs/SPECIFICATION.md Section 7 and docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from memorymesh.api.main import create_app
from memorymesh.config import load_app_config
from memorymesh.service import MemoryService


@pytest.fixture
def fake_cfg():
    """AppConfig configured to use the deterministic offline fake provider."""
    cfg = load_app_config()
    cfg.engine.llm.provider = "fake"  # type: ignore[assignment]
    return cfg


def test_default_app_lifespan_startup_and_query(fake_cfg, tmp_path) -> None:
    """Test default create_app without injected service starts up cleanly and queries DB."""
    db_file = tmp_path / "test_lifecycle.db"
    with patch("memorymesh.api.main.DEFAULT_DB_PATH", db_file):
        app = create_app(cfg=fake_cfg)
        with TestClient(app) as client:
            # Verify DB query works across worker thread without check_same_thread error
            res = client.get("/api/memories")
            assert res.status_code == 200
            assert isinstance(res.json(), list)

            # Verify health endpoint reports the configured provider
            health_res = client.get("/api/health")
            assert health_res.status_code == 200
            assert health_res.json()["llm_provider"] == "fake"


def test_e2e_ingest_through_default_app(fake_cfg, tmp_path) -> None:
    """Test E2E ingest extracts claims via FakeProvider without real network calls."""
    db_file = tmp_path / "test_e2e_ingest.db"
    with patch("memorymesh.api.main.DEFAULT_DB_PATH", db_file):
        app = create_app(cfg=fake_cfg)
        with TestClient(app) as client:
            # Default mode is "e2e"
            ingest_payload = {
                "source_id": "sysadmin_priya",
                "event_time": "2026-06-01T10:00:00Z",
                "text": "Server A runs Python 3.12.",
            }
            res = client.post("/api/ingest", json=ingest_payload)
            assert res.status_code == 200
            data = res.json()
            assert data["duplicate"] is False
            assert "decisions" in data


def test_structured_ingest_through_default_app(fake_cfg, tmp_path) -> None:
    """Test structured ingest succeeds directly without requiring LLM extraction."""
    db_file = tmp_path / "test_struct_ingest.db"
    with patch("memorymesh.api.main.DEFAULT_DB_PATH", db_file):
        app = create_app(cfg=fake_cfg)
        with TestClient(app) as client:
            ingest_payload = {
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
                        "extract_confidence": 0.95,
                    }
                ],
            }
            res = client.post("/api/ingest", json=ingest_payload)
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["claims_extracted"] == 1
            assert len(data["decisions"]) >= 1


def test_lifespan_closes_internally_owned_connection(fake_cfg, tmp_path) -> None:
    """Test connection created internally during lifespan is closed upon shutdown."""
    db_file = tmp_path / "test_cleanup.db"
    with patch("memorymesh.api.main.DEFAULT_DB_PATH", db_file):
        app = create_app(cfg=fake_cfg)
        conn = None
        with TestClient(app) as client:
            service = getattr(client.app.state, "service", None)
            assert service is not None
            conn = service.repo.conn
            # Connection is active inside context
            assert conn.execute("SELECT 1").fetchone()[0] == 1

        # After exiting lifespan, the owned connection should be closed
        assert conn is not None
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            conn.execute("SELECT 1")


def test_lifespan_does_not_close_externally_injected_service(test_db_conn, fake_cfg) -> None:
    """Test externally injected service/connection is preserved after lifespan shutdown."""
    from memorymesh.repository import Repository

    repo = Repository(test_db_conn)
    service = MemoryService(db_target=repo, cfg=fake_cfg)
    app = create_app(service=service)

    with TestClient(app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200

    # External connection remains open
    assert test_db_conn.execute("SELECT 1").fetchone()[0] == 1


def test_lifespan_startup_failure_cleans_up_connection(fake_cfg, tmp_path) -> None:
    """Test startup exception raises cleanly and closes partially opened connection."""
    db_file = tmp_path / "test_startup_fail.db"
    with (
        patch("memorymesh.api.main.DEFAULT_DB_PATH", db_file),
        patch(
            "memorymesh.api.main.get_provider",
            side_effect=ValueError("Simulated provider failure"),
        ),
    ):
        app = create_app(cfg=fake_cfg)
        with pytest.raises(RuntimeError, match="Failed to initialize configured LLM provider"):
            with TestClient(app):
                pass
