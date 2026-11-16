"""Concurrency and single-process ingest serialization tests.

Governed by docs/ARCHITECTURE.md Section 4b.
"""

from __future__ import annotations

import concurrent.futures
import threading
from datetime import UTC, datetime
from typing import Any

from fastapi.testclient import TestClient

from memorymesh.models import Claim, IngestMessage
from memorymesh.service import MemoryService


def test_concurrent_ingest_serialized(client: TestClient) -> None:
    """Verify concurrent ingest requests are properly serialized by the ingest lock.

    All requests must succeed without SQLite concurrency/locking errors,
    and all messages must be persisted consistently in the database.
    """
    num_requests = 10

    def send_ingest(index: int) -> int:
        payload = {
            "source_id": "sysadmin_priya",
            "event_time": f"2026-06-01T10:{index:02d}:00Z",
            "text": f"Server {index} runs Python 3.12.",
            "mode": "structured",
            "claims": [
                {
                    "subject": f"Server {index}",
                    "attribute": "python_version",
                    "value": "3.12",
                    "polarity": "assert",
                    "extract_confidence": 0.95,
                }
            ],
        }
        res = client.post("/api/ingest", json=payload)
        assert res.status_code == 200, f"Request {index} failed: {res.text}"
        data = res.json()
        assert data["success"] is True
        return data["message_id"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(send_ingest, i) for i in range(num_requests)]
        msg_ids = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(msg_ids) == num_requests
    assert len(set(msg_ids)) == num_requests

    # Verify all memories are present in repository
    mems_res = client.get("/api/memories?limit=500")
    assert mems_res.status_code == 200
    mems = mems_res.json()
    assert len(mems) >= num_requests


class _ObservableLock:
    """Wrapper around a threading lock that records lock acquisition attempts."""

    def __init__(self, target: Any) -> None:
        self._target = target
        self.acquire_attempted = threading.Event()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        self.acquire_attempted.set()
        return self._target.acquire(blocking, timeout)

    def release(self) -> None:
        self._target.release()

    def __enter__(self) -> bool:
        self.acquire_attempted.set()
        return self._target.__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        return self._target.__exit__(exc_type, exc_val, exc_tb)


def test_read_write_isolation_commit(
    client: TestClient,
    memory_service: MemoryService,
) -> None:
    """Verify concurrent reads cannot observe uncommitted writes and wait until commit.

    1. Writer thread acquires the shared ingest lock and opens a write transaction.
    2. Concurrent reader thread requests data for the entity being written.
    3. Reader reaches and attempts lock acquisition, and is blocked while writer holds lock.
    4. Upon commit and lock release, reader proceeds and sees the committed data.
    """
    real_lock = memory_service._ingest_lock
    obs_lock = _ObservableLock(real_lock)
    memory_service._ingest_lock = obs_lock  # type: ignore[assignment]

    write_active = threading.Event()
    allow_commit = threading.Event()
    read_completed = threading.Event()
    read_result: list[Any] = []

    def writer_action() -> None:
        with memory_service._ingest_lock:
            with memory_service.repo.transaction():
                msg = IngestMessage(
                    id=101,
                    source_id="sysadmin_priya",
                    text="Server Commit Test runs Python 3.12",
                    event_time=datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC),
                    ingested_at=datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC),
                )
                claim = Claim(
                    id=101,
                    message_id=101,
                    subject="Server Commit Test",
                    attribute="python_version",
                    value="3.12",
                    polarity="assert",
                    extract_confidence=0.95,
                )
                memory_service.ingest_claims(msg, [claim])
                write_active.set()
                assert allow_commit.wait(timeout=5.0)

    def reader_action() -> None:
        res = client.get("/api/memories?entity=Server Commit Test")
        read_result.append(res)
        read_completed.set()

    try:
        writer_t = threading.Thread(target=writer_action)
        writer_t.start()

        assert write_active.wait(timeout=5.0)

        # Clear event so it strictly captures the reader's lock acquisition attempt
        obs_lock.acquire_attempted.clear()

        reader_t = threading.Thread(target=reader_action)
        reader_t.start()

        # 1. Establish that the reader has explicitly attempted lock acquisition
        assert obs_lock.acquire_attempted.wait(timeout=2.0), (
            "Reader never attempted lock acquisition"
        )

        # 2. Assert reader is blocked and cannot complete while writer holds lock
        assert not read_completed.is_set(), "Reader completed while write transaction held lock"
        assert reader_t.is_alive(), "Reader thread terminated prematurely"

        allow_commit.set()
        writer_t.join(timeout=5.0)

        assert read_completed.wait(timeout=5.0)
        reader_t.join(timeout=5.0)

        assert len(read_result) == 1
        res = read_result[0]
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        assert data[0]["memory"]["attribute"] == "python_version"
        assert data[0]["memory"]["value_norm"] == "3.12"
    finally:
        memory_service._ingest_lock = real_lock


def test_read_write_isolation_rollback(
    client: TestClient,
    memory_service: MemoryService,
) -> None:
    """Verify concurrent reads cannot observe rolled-back data (zero dirty reads).

    1. Writer thread acquires the shared ingest lock and opens a write transaction.
    2. Writer mutates the database inside the uncommitted transaction.
    3. Concurrent reader thread requests data for the entity being written.
    4. Reader reaches and attempts lock acquisition, and is blocked while writer holds lock.
    5. Writer triggers a rollback and releases the lock.
    6. Reader unblocks, queries the database, and sees 0 memories (no dirty read).
    """
    real_lock = memory_service._ingest_lock
    obs_lock = _ObservableLock(real_lock)
    memory_service._ingest_lock = obs_lock  # type: ignore[assignment]

    write_active = threading.Event()
    allow_rollback = threading.Event()
    read_completed = threading.Event()
    read_result: list[Any] = []

    def writer_action() -> None:
        try:
            with memory_service._ingest_lock:
                with memory_service.repo.transaction():
                    msg = IngestMessage(
                        id=102,
                        source_id="sysadmin_priya",
                        text="Server Rollback Test runs Python 3.12",
                        event_time=datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC),
                        ingested_at=datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC),
                    )
                    claim = Claim(
                        id=102,
                        message_id=102,
                        subject="Server Rollback Test",
                        attribute="python_version",
                        value="3.12",
                        polarity="assert",
                        extract_confidence=0.95,
                    )
                    memory_service.ingest_claims(msg, [claim])
                    write_active.set()
                    assert allow_rollback.wait(timeout=5.0)
                    raise RuntimeError("Trigger transaction rollback")
        except RuntimeError:
            pass

    def reader_action() -> None:
        res = client.get("/api/memories?entity=Server Rollback Test")
        read_result.append(res)
        read_completed.set()

    try:
        writer_t = threading.Thread(target=writer_action)
        writer_t.start()

        assert write_active.wait(timeout=5.0)

        # Clear event so it strictly captures the reader's lock acquisition attempt
        obs_lock.acquire_attempted.clear()

        reader_t = threading.Thread(target=reader_action)
        reader_t.start()

        # 1. Establish that the reader has explicitly attempted lock acquisition
        assert obs_lock.acquire_attempted.wait(timeout=2.0), (
            "Reader never attempted lock acquisition"
        )

        # 2. Assert reader is blocked and cannot complete while writer holds lock
        assert not read_completed.is_set(), "Reader completed while write transaction held lock"
        assert reader_t.is_alive(), "Reader thread terminated prematurely"

        allow_rollback.set()
        writer_t.join(timeout=5.0)

        assert read_completed.wait(timeout=5.0)
        reader_t.join(timeout=5.0)

        assert len(read_result) == 1
        res = read_result[0]
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 0
    finally:
        memory_service._ingest_lock = real_lock
