"""Tests for the structured temporal memory CLI demonstration script.

Governed by docs/SPECIFICATION.md Section 2 and docs/ROADMAP_AND_TASKS.md (T1.5).
Verifies:
  - Demonstration completes with exit code 0.
  - Produces deterministic, reproducible output.
  - Exercises MemoryService.ingest_claims and retrieval.
  - Operates completely offline without modifying any disk application database.
"""

from __future__ import annotations

import io
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from memorymesh.db import init_db
from memorymesh.service import MemoryService
from scripts.demo_temporal_memory import run_demo


def test_demo_execution_success() -> None:
    """The CLI demo executes successfully, returning exit code 0 and valid summary."""
    stream = io.StringIO()
    exit_code = run_demo(stream=stream)

    assert exit_code == 0
    output = stream.getvalue()

    assert "MemoryMesh: Structured Temporal Memory Demonstration (P1.3)" in output
    assert "STEP 1: Ingest Initial Claim" in output
    assert "Rule ID     : R-INSERT" in output
    assert "STEP 2: Ingest Newer Conflicting Claim" in output
    assert "Rule ID     : R-REPLACE" in output
    assert "Current Version    : Python 3.12 (expected: 3.12)" in output
    assert "As-Of March 15     : Python 3.10 (expected: 3.10)" in output
    assert "Demonstration completed successfully." in output


def test_demo_deterministic_output() -> None:
    """Successive runs of the demo produce identical, deterministic output."""
    stream1 = io.StringIO()
    stream2 = io.StringIO()

    code1 = run_demo(stream=stream1)
    code2 = run_demo(stream=stream2)

    assert code1 == 0
    assert code2 == 0
    assert stream1.getvalue() == stream2.getvalue()
    assert len(stream1.getvalue()) > 500


def test_demo_does_not_modify_disk_database(tmp_path: Path) -> None:
    """The demo uses an in-memory database and never touches data/memorymesh.db."""
    default_db = Path("data/memorymesh.db")
    existed_before = default_db.exists()
    mtime_before = default_db.stat().st_mtime if existed_before else None

    stream = io.StringIO()
    exit_code = run_demo(stream=stream)

    assert exit_code == 0
    if not existed_before:
        assert not default_db.exists()
    else:
        assert default_db.stat().st_mtime == mtime_before


def test_demo_subprocess_invocation() -> None:
    """The demo script can be invoked as a standalone CLI command via python."""
    result = subprocess.run(
        [sys.executable, "scripts/demo_temporal_memory.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Demonstration completed successfully." in result.stdout
    assert "Current Version    : Python 3.12" in result.stdout
    assert "As-Of March 15     : Python 3.10" in result.stdout


def test_demo_exercises_service_and_retrieval() -> None:
    """The demo invokes the actual MemoryService.ingest_claims and retrieval methods."""
    stream = io.StringIO()
    orig_ingest = MemoryService.ingest_claims
    orig_current = MemoryService.current
    with (
        patch.object(
            MemoryService, "ingest_claims", autospec=True, side_effect=orig_ingest
        ) as spy_ingest,
        patch.object(
            MemoryService, "current", autospec=True, side_effect=orig_current
        ) as spy_current,
    ):
        exit_code = run_demo(stream=stream)

        assert exit_code == 0
        assert spy_ingest.call_count == 2
        assert spy_current.call_count == 2

        # Verify claim 1 details: args are (self, message, claims)
        call1_args = spy_ingest.call_args_list[0]
        claims1 = call1_args[0][2]
        assert len(claims1) == 1
        assert claims1[0].value == "3.10"

        # Verify claim 2 details: args are (self, message, claims)
        call2_args = spy_ingest.call_args_list[1]
        claims2 = call2_args[0][2]
        assert len(claims2) == 1
        assert claims2[0].value == "3.12"


def test_demo_fails_when_first_decision_not_r_insert() -> None:
    """Demo exits with code 1 when the first decision is not R-INSERT."""
    orig_ingest = MemoryService.ingest_claims

    def fake_ingest(self, msg, claims, recorded_at=None):
        rep = orig_ingest(self, msg, claims, recorded_at=recorded_at)
        if claims[0].value == "3.10" and rep.decisions:
            corrupted_dec = rep.decisions[0].model_copy(update={"rule_id": "R-OTHER"})
            return rep.model_copy(update={"decisions": [corrupted_dec]})
        return rep

    stream = io.StringIO()
    with patch.object(MemoryService, "ingest_claims", autospec=True, side_effect=fake_ingest):
        exit_code = run_demo(stream=stream)

    assert exit_code == 1
    output = stream.getvalue()
    assert "Step 1 validation failed" in output
    assert "Demonstration completed successfully." not in output


def test_demo_fails_when_second_decision_not_r_replace() -> None:
    """Demo exits with code 1 when the second decision is not R-REPLACE."""
    orig_ingest = MemoryService.ingest_claims

    def fake_ingest(self, msg, claims, recorded_at=None):
        rep = orig_ingest(self, msg, claims, recorded_at=recorded_at)
        if claims[0].value == "3.12" and rep.decisions:
            corrupted_dec = rep.decisions[0].model_copy(update={"rule_id": "R-OTHER"})
            return rep.model_copy(update={"decisions": [corrupted_dec]})
        return rep

    stream = io.StringIO()
    with patch.object(MemoryService, "ingest_claims", autospec=True, side_effect=fake_ingest):
        exit_code = run_demo(stream=stream)

    assert exit_code == 1
    output = stream.getvalue()
    assert "Step 2 validation failed" in output
    assert "Demonstration completed successfully." not in output


def test_demo_fails_when_current_retrieval_mismatches() -> None:
    """Demo exits with code 1 when current retrieval does not return Python 3.12."""
    orig_current = MemoryService.current

    def fake_current(self, entity, attr, context=None, valid_at=None, known_at=None, now=None):
        res = orig_current(
            self, entity, attr, context=context, valid_at=valid_at, known_at=known_at, now=now
        )
        if valid_at is None and res.items:
            corrupted_mem = res.items[0].memory.model_copy(update={"value_norm": "9.99"})
            corrupted_view = res.items[0].model_copy(update={"memory": corrupted_mem})
            return res.model_copy(update={"items": [corrupted_view]})
        return res

    stream = io.StringIO()
    with patch.object(MemoryService, "current", autospec=True, side_effect=fake_current):
        exit_code = run_demo(stream=stream)

    assert exit_code == 1
    output = stream.getvalue()
    assert "Step 3 validation failed" in output
    assert "Demonstration completed successfully." not in output


def test_demo_fails_when_historical_retrieval_mismatches() -> None:
    """Demo exits with code 1 when historical retrieval does not return Python 3.10."""
    orig_current = MemoryService.current

    def fake_current(self, entity, attr, context=None, valid_at=None, known_at=None, now=None):
        res = orig_current(
            self, entity, attr, context=context, valid_at=valid_at, known_at=known_at, now=now
        )
        if valid_at is not None and res.items:
            corrupted_mem = res.items[0].memory.model_copy(update={"value_norm": "9.99"})
            corrupted_view = res.items[0].model_copy(update={"memory": corrupted_mem})
            return res.model_copy(update={"items": [corrupted_view]})
        return res

    stream = io.StringIO()
    with patch.object(MemoryService, "current", autospec=True, side_effect=fake_current):
        exit_code = run_demo(stream=stream)

    assert exit_code == 1
    output = stream.getvalue()
    assert "Step 4 validation failed" in output
    assert "Demonstration completed successfully." not in output


def test_demo_closes_connection_on_success_and_error() -> None:
    """The in-memory database connection is closed on success, error, and exception."""
    captured_connections: list[sqlite3.Connection] = []

    def spy_init_db(target):
        conn = init_db(target)
        captured_connections.append(conn)
        return conn

    # 1. Normal successful run
    with patch("scripts.demo_temporal_memory.init_db", side_effect=spy_init_db):
        exit_code = run_demo(stream=io.StringIO())
        assert exit_code == 0
        conn1 = captured_connections[-1]
        with pytest.raises(sqlite3.ProgrammingError, match="Cannot operate on a closed database"):
            conn1.execute("SELECT 1")

    # 2. Run with early return on failure
    with (
        patch("scripts.demo_temporal_memory.init_db", side_effect=spy_init_db),
        patch.object(
            MemoryService,
            "ingest_claims",
            side_effect=RuntimeError("Simulated unexpected failure"),
        ),
    ):
        exit_code = run_demo(stream=io.StringIO())
        assert exit_code == 1
        conn2 = captured_connections[-1]
        with pytest.raises(sqlite3.ProgrammingError, match="Cannot operate on a closed database"):
            conn2.execute("SELECT 1")
