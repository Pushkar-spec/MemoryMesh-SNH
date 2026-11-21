"""Integration tests for SSE scenario replay endpoint GET /api/demo/replay.

Governed by docs/SPECIFICATION.md Section 7.2 and docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from memorymesh.api.main import create_app
from memorymesh.api.routers import demo
from memorymesh.config import load_app_config
from memorymesh.db import get_connection, init_db
from memorymesh.llm.client import LLMClient
from memorymesh.llm.providers import FakeProvider
from memorymesh.models import Claim, IngestMessage
from memorymesh.repository import Repository
from memorymesh.service import MemoryService


@pytest.fixture
def replay_client() -> Generator[TestClient, None, None]:
    """TestClient configured with in-memory database and fake LLM."""
    conn = get_connection(":memory:", check_same_thread=False)
    init_db(conn)
    repo = Repository(conn)
    cfg = load_app_config()
    cfg.engine.llm.provider = "fake"  # type: ignore[assignment]
    llm = LLMClient(provider=FakeProvider(), repo=repo, config=cfg.engine.llm)
    service = MemoryService(db_target=repo, cfg=cfg, llm=llm)
    app = create_app(service=service)
    app.state.service = service
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    conn.close()


def _parse_sse_events(raw_lines: list[str]) -> list[tuple[str, dict]]:
    """Parse raw SSE line chunks into (event_type, payload_dict) pairs."""
    events: list[tuple[str, dict]] = []
    current_event: str | None = None
    data_lines: list[str] = []

    for line in raw_lines:
        line = line.strip()
        if not line:
            if current_event and data_lines:
                try:
                    payload = json.loads("\n".join(data_lines))
                    events.append((current_event, payload))
                except Exception:
                    pass
            current_event = None
            data_lines = []
        elif line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())

    if current_event and data_lines:
        try:
            payload = json.loads("\n".join(data_lines))
            events.append((current_event, payload))
        except Exception:
            pass

    return events


def test_replay_hero_scenario_sequential(replay_client: TestClient) -> None:
    """Verify GET /api/demo/replay streams sequential decisions and completes."""
    # Ensure clean database via documented POST reset
    reset_res = replay_client.post("/api/demo/reset")
    assert reset_res.status_code == 200

    with replay_client.stream(
        "GET", "/api/demo/replay?scenario=hero-001&mode=structured&delay_ms=0"
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        lines = list(response.iter_lines())
        events = _parse_sse_events(lines)

    assert len(events) >= 9, f"Expected at least 9 SSE events, got {len(events)}"

    # First 8 events must be message_ingested
    ingested_events = [e for e in events if e[0] == "message_ingested"]
    assert len(ingested_events) == 8

    # Verify canonical decision rules in order
    rules_per_msg = []
    for _, report in ingested_events:
        assert report["success"] is True
        rules = [d["rule_id"] for d in report.get("decisions", [])]
        rules_per_msg.append(rules)

    # m1: R-INSERT
    assert "R-INSERT" in rules_per_msg[0]
    # m2: R-CORROBORATE
    assert "R-CORROBORATE" in rules_per_msg[1]
    # m3: R-INSERT
    assert "R-INSERT" in rules_per_msg[2]
    # m5: R-REPLACE
    assert "R-REPLACE" in rules_per_msg[4]
    # m6: R-REJECT
    assert "R-REJECT" in rules_per_msg[5]
    # m7: R-DISPUTE
    assert "R-DISPUTE" in rules_per_msg[6]

    # Final event must be 'done' with 8 messages
    done_events = [e for e in events if e[0] == "done"]
    assert len(done_events) == 1
    assert done_events[0][1]["messages"] == 8


def test_replay_nonexistent_scenario_emits_error(replay_client: TestClient) -> None:
    """Verify replay yields error event when scenario cannot be located."""
    with replay_client.stream(
        "GET", "/api/demo/replay?scenario=nonexistent_xyz&delay_ms=0"
    ) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())
        events = _parse_sse_events(lines)

    assert len(events) == 1
    event_type, payload = events[0]
    assert event_type == "error"
    assert "not found" in payload.get("error", "").lower()


def test_post_reset_endpoint_performs_reset(replay_client: TestClient) -> None:
    """Verify explicit POST /api/demo/reset clears prior database state."""
    service = replay_client.app.state.service  # type: ignore[attr-defined]

    # Pre-populate a source and memory
    im = IngestMessage(
        source_id="sysadmin_priya",
        event_time=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        text="Pre-existing test message",
    )
    c = Claim(
        subject="PreExistingServer",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
    )
    service.ingest_claims(message=im, claims=[c])
    assert len(service.repo.get_all_memories()) >= 1

    # Call POST /api/demo/reset
    res = replay_client.post("/api/demo/reset")
    assert res.status_code == 200
    assert res.json() == {"ok": True}

    # Verify that the pre-existing server is gone because database was reset
    memories = service.repo.get_all_memories()
    test_entities = [m for m in memories if m.entity_name == "PreExistingServer"]
    assert len(test_entities) == 0


def test_replay_default_does_not_reset_database(replay_client: TestClient) -> None:
    """Verify GET /api/demo/replay without reset parameter does NOT delete data."""
    service = replay_client.app.state.service  # type: ignore[attr-defined]

    # Pre-populate a source and memory
    im = IngestMessage(
        source_id="sysadmin_priya",
        event_time=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        text="Pre-existing test message",
    )
    c = Claim(
        subject="PreservedServer",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
    )
    service.ingest_claims(message=im, claims=[c])
    assert len(service.repo.get_all_memories()) >= 1

    # Run replay without reset query param
    with replay_client.stream("GET", "/api/demo/replay?scenario=hero-001&delay_ms=0") as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())
        events = _parse_sse_events(lines)

    done_events = [e for e in events if e[0] == "done"]
    assert len(done_events) == 1

    # Verify that the pre-existing server STILL EXISTS because reset was not performed
    memories = service.repo.get_all_memories()
    test_entities = [m for m in memories if m.entity_name == "PreservedServer"]
    assert len(test_entities) >= 1


def test_replay_legacy_reset_param_cannot_delete_data(replay_client: TestClient) -> None:
    """Verify GET /api/demo/replay with legacy reset=true cannot delete database records."""
    service = replay_client.app.state.service  # type: ignore[attr-defined]

    # Pre-populate a source and memory
    im = IngestMessage(
        source_id="sysadmin_priya",
        event_time=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        text="Pre-existing test message for legacy reset param check",
    )
    c = Claim(
        subject="PreservedLegacyServer",
        attribute="python_version",
        value="3.12",
        extract_confidence=1.0,
    )
    service.ingest_claims(message=im, claims=[c])
    assert len(service.repo.get_all_memories()) >= 1

    # Explicitly pass legacy reset=true to GET endpoint
    with replay_client.stream(
        "GET", "/api/demo/replay?scenario=hero-001&delay_ms=0&reset=true"
    ) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())
        events = _parse_sse_events(lines)

    done_events = [e for e in events if e[0] == "done"]
    assert len(done_events) == 1

    # Memory MUST still exist: GET replay is strictly read-only regardless of params
    memories = service.repo.get_all_memories()
    test_entities = [m for m in memories if m.entity_name == "PreservedLegacyServer"]
    assert len(test_entities) >= 1


def test_replay_overlapping_concurrent_requests_rejected(
    replay_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify two overlapping replay requests cannot run concurrently.

    Uses deterministic event synchronization rather than arbitrary sleeps:
    instruments try_acquire to signal when the first request has acquired the lock,
    ensuring the second request arrives while the lock is genuinely held.
    """
    app = replay_client.app
    first_events: list[tuple[str, dict]] = []
    second_events: list[tuple[str, dict]] = []
    lock_acquired_by_first = threading.Event()

    real_try_acquire = demo._replay_lock.try_acquire

    def instrumented_try_acquire() -> bool:
        acquired = real_try_acquire()
        if acquired:
            lock_acquired_by_first.set()
        return acquired

    monkeypatch.setattr(demo._replay_lock, "try_acquire", instrumented_try_acquire)

    def run_first_replay() -> None:
        with TestClient(app, raise_server_exceptions=False) as client1:
            with client1.stream("GET", "/api/demo/replay?scenario=hero-001&delay_ms=250") as resp:
                assert resp.status_code == 200
                lines = list(resp.iter_lines())
                first_events.extend(_parse_sse_events(lines))

    def run_second_replay() -> None:
        # Deterministically wait until request 1 has acquired the lock
        assert lock_acquired_by_first.wait(timeout=5.0), "First replay failed to acquire lock"
        with TestClient(app, raise_server_exceptions=False) as client2:
            with client2.stream("GET", "/api/demo/replay?scenario=hero-001&delay_ms=0") as resp:
                assert resp.status_code == 200
                lines = list(resp.iter_lines())
                second_events.extend(_parse_sse_events(lines))

    t1 = threading.Thread(target=run_first_replay)
    t2 = threading.Thread(target=run_second_replay)

    t1.start()
    t2.start()

    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    # Assert both threads have actually terminated (not timed out)
    assert not t1.is_alive(), "First replay thread timed out"
    assert not t2.is_alive(), "Second replay thread timed out"

    # Second request must have received the concurrency rejection error
    assert len(second_events) == 1
    assert second_events[0][0] == "error"
    assert "already active" in second_events[0][1].get("error", "").lower()

    # First request must have proceeded and completed all 8 messages
    first_done = [e for e in first_events if e[0] == "done"]
    assert len(first_done) == 1
    assert first_done[0][1]["messages"] == 8

    # After first finishes, a subsequent replay request can start cleanly
    with replay_client.stream("GET", "/api/demo/replay?scenario=hero-001&delay_ms=0") as third_resp:
        assert third_resp.status_code == 200
        third_lines = list(third_resp.iter_lines())
        third_events = _parse_sse_events(third_lines)

    third_done = [e for e in third_events if e[0] == "done"]
    assert len(third_done) == 1
    assert third_done[0][1]["messages"] == 8


@pytest.mark.asyncio
async def test_replay_lock_released_on_generator_cancellation(
    replay_client: TestClient,
) -> None:
    """Verify replay lock is released when generator is cancelled during active replay.

    Directly exercises the streaming response's async body iterator to avoid
    synchronous TestClient response buffering. Proves that:
    1. Lock is acquired when generator starts.
    2. Replay is still in progress (not completed) when cancellation occurs.
    3. Generator closure (aclose) releases the lock.
    4. A subsequent replay can immediately acquire the lock and complete.
    """
    service = replay_client.app.state.service  # type: ignore[attr-defined]

    scope = {"type": "http", "method": "GET", "path": "/api/demo/replay"}
    mock_request = Request(scope)
    mock_request.is_disconnected = AsyncMock(return_value=False)

    assert not demo._replay_lock.is_locked

    # Call endpoint directly to obtain EventSourceResponse and its async generator
    response = await demo.replay_scenario(
        request=mock_request,
        service=service,
        scenario="hero-001",
        mode="structured",
        delay_ms=0,
    )
    generator = response.body_iterator

    # Step 1: Advance by 1 event (message #1 ingested)
    first_event = await generator.__anext__()
    assert first_event.event == "message_ingested"

    # Step 2: Prove lock is acquired while replay is actively in progress
    assert demo._replay_lock.is_locked, "Lock must be held while stream is active"

    # Step 3: Explicitly interrupt / close the generator mid-stream
    await generator.aclose()

    # Step 4: Prove lock is released immediately upon generator cancellation
    assert not demo._replay_lock.is_locked, "Lock must be released on generator closure"

    # Step 5: Prove subsequent replay can acquire the lock and run to completion
    subsequent_response = await demo.replay_scenario(
        request=mock_request,
        service=service,
        scenario="hero-001",
        mode="structured",
        delay_ms=0,
    )
    events = [event async for event in subsequent_response.body_iterator]
    assert len(events) == 9  # 8 message_ingested + 1 done
    assert events[-1].event == "done"

    # Lock must be released after completion
    assert not demo._replay_lock.is_locked


@pytest.mark.asyncio
async def test_replay_lock_released_on_client_disconnect_flag(
    replay_client: TestClient,
) -> None:
    """Verify replay loop terminates and releases lock when is_disconnected is detected."""
    service = replay_client.app.state.service  # type: ignore[attr-defined]
    scope = {"type": "http", "method": "GET", "path": "/api/demo/replay"}
    mock_request = Request(scope)

    # Disconnect after message #1
    call_count = 0

    async def mock_is_disconnected() -> bool:
        nonlocal call_count
        call_count += 1
        return call_count > 1

    mock_request.is_disconnected = mock_is_disconnected

    response = await demo.replay_scenario(
        request=mock_request,
        service=service,
        scenario="hero-001",
        mode="structured",
        delay_ms=0,
    )
    events = [event async for event in response.body_iterator]
    # Replay terminates early due to disconnect and yields done with partial count
    assert any(e.event == "done" for e in events)
    assert not demo._replay_lock.is_locked


def test_replay_lock_released_on_completion_and_failure(replay_client: TestClient) -> None:
    """Verify replay lock is released reliably after completion and after failure."""
    assert not demo._replay_lock.is_locked

    # Case 1: Failure during scenario loading (non-existent scenario)
    with replay_client.stream(
        "GET", "/api/demo/replay?scenario=bad_scenario_123&delay_ms=0"
    ) as resp:
        events = _parse_sse_events(list(resp.iter_lines()))
        assert events[0][0] == "error"

    # Lock must be released
    assert not demo._replay_lock.is_locked

    # Case 2: Normal completion
    with replay_client.stream("GET", "/api/demo/replay?scenario=hero-001&delay_ms=0") as resp:
        events = _parse_sse_events(list(resp.iter_lines()))
        done = [e for e in events if e[0] == "done"]
        assert len(done) == 1

    # Lock must be released
    assert not demo._replay_lock.is_locked
