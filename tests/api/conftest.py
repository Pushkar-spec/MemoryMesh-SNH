"""Test fixtures and setup for MemoryMesh FastAPI backend test suite.

Governed by docs/SPECIFICATION.md Section 7.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from memorymesh.api.main import create_app
from memorymesh.config import AppConfig, load_app_config
from memorymesh.db import init_db
from memorymesh.llm.client import LLMClient
from memorymesh.llm.providers import FakeProvider
from memorymesh.repository import Repository
from memorymesh.service import MemoryService


@pytest.fixture
def fake_llm() -> FakeProvider:
    """Deterministic offline FakeProvider."""
    return FakeProvider()


@pytest.fixture
def app_config() -> AppConfig:
    """Canonical AppConfig with fake provider override for API testing."""
    cfg = load_app_config()
    cfg.engine.llm.provider = "fake"  # type: ignore[assignment]
    return cfg


@pytest.fixture
def test_db_conn() -> Generator[sqlite3.Connection, None, None]:
    """In-memory SQLite database connection initialized with canonical schema."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def memory_service(
    test_db_conn: sqlite3.Connection,
    app_config: AppConfig,
    fake_llm: FakeProvider,
) -> MemoryService:
    """Isolated MemoryService instance for testing."""
    repo = Repository(test_db_conn)
    llm_client = LLMClient(provider=fake_llm, repo=repo, config=app_config.engine.llm)
    return MemoryService(
        db_target=repo,
        cfg=app_config,
        llm=llm_client,
        clock=lambda: datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC),
    )


@pytest.fixture
def client(memory_service: MemoryService) -> Generator[TestClient, None, None]:
    """FastAPI TestClient with injected MemoryService."""
    app = create_app(service=memory_service)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
