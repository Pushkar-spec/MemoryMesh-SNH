"""Tests for SQLite connection, schema initialization, and reset functionality."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memorymesh.db import (
    _reset_connection_tables,
    connection_scope,
    get_connection,
    get_schema_path,
    get_schema_sql,
    init_db,
    reset_db,
)

EXPECTED_TABLES = {
    "sources",
    "entities",
    "entity_aliases",
    "attribute_schema",
    "messages",
    "memories",
    "evidence",
    "edges",
    "decision_log",
    "llm_calls",
}

EXPECTED_INDEXES = {
    "idx_mem_lookup",
    "idx_mem_valid",
    "idx_ev_mem",
    "idx_edge_dst",
    "idx_edge_src",
}


def test_schema_path_and_loading() -> None:
    """Verify schema.sql is found and read independently of working directory."""
    path = get_schema_path()
    assert path.is_file()
    assert path.name == "schema.sql"

    sql = get_schema_sql()
    assert "CREATE TABLE IF NOT EXISTS sources" in sql
    assert "CREATE TABLE IF NOT EXISTS memories" in sql


def test_schema_loading_independent_of_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify schema loading still succeeds even when current working directory changes."""
    monkeypatch.chdir(tmp_path)
    sql = get_schema_sql()
    assert "CREATE TABLE IF NOT EXISTS sources" in sql


def test_init_db_in_memory() -> None:
    """Verify in-memory database initializes with all 10 tables and indexes."""
    conn = init_db(":memory:")
    try:
        cursor = conn.cursor()
        tables = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        }
        assert EXPECTED_TABLES.issubset(tables)

        indexes = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        }
        assert EXPECTED_INDEXES.issubset(indexes)
    finally:
        conn.close()


def test_init_db_twice_is_safe(tmp_path: Path) -> None:
    """Verify that calling init_db repeatedly does not raise errors or duplicate tables."""
    db_file = tmp_path / "test_twice.db"
    conn1 = init_db(db_file)
    conn1.close()

    # Second initialization should succeed safely
    conn2 = init_db(db_file)
    cursor = conn2.cursor()
    tables = {
        row[0]
        for row in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        ).fetchall()
    }
    assert EXPECTED_TABLES.issubset(tables)
    conn2.close()


def test_foreign_keys_enforced_and_reject_violations(tmp_path: Path) -> None:
    """Verify foreign key enforcement is active and rejects orphaned rows."""
    db_file = tmp_path / "test_fk.db"
    conn = init_db(db_file)

    try:
        # Verify foreign keys pragma is ON
        fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert fk == 1

        # Attempt inserting a message pointing to a nonexistent source_id
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO messages (source_id, text, text_hash, event_time, ingested_at)
                VALUES (
                    'nonexistent_source',
                    'hello',
                    'hash123',
                    '2026-06-25T09:00:00Z',
                    '2026-06-25T09:00:00Z'
                );
                """
            )
    finally:
        conn.close()


def test_wal_mode_enabled_for_file_db(tmp_path: Path) -> None:
    """Verify that file-backed databases enable WAL mode."""
    db_file = tmp_path / "test_wal.db"
    conn = init_db(db_file)
    try:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        conn.close()


def test_in_memory_journal_mode() -> None:
    """Verify in-memory database initializes without error and does not force WAL mode."""
    conn = init_db(":memory:")
    try:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() in ("memory", "off")
    finally:
        conn.close()


def test_connection_scope_context_manager(tmp_path: Path) -> None:
    """Verify connection_scope provides an active connection and closes it cleanly."""
    db_file = tmp_path / "test_scope.db"
    init_db(db_file).close()

    with connection_scope(db_file) as conn:
        assert isinstance(conn, sqlite3.Connection)
        row = conn.execute("SELECT 1 AS val;").fetchone()
        assert row["val"] == 1

    # After exiting with block, connection should be closed
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1;")


def test_reset_db_clears_data_file(tmp_path: Path) -> None:
    """Verify reset_db deletes old data and recreates empty schema on a file path."""
    db_file = tmp_path / "test_reset.db"
    conn = init_db(db_file)

    # Insert test entity
    conn.execute(
        """
        INSERT INTO entities (canonical_name, type, created_at)
        VALUES ('Server A', 'server', '2026-01-01T00:00:00Z');
        """
    )
    conn.commit()
    conn.close()

    # Reset
    conn2 = reset_db(db_file)
    try:
        count = conn2.execute("SELECT count(*) FROM entities;").fetchone()[0]
        assert count == 0
        # Tables still exist
        tables = {
            row[0]
            for row in conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        }
        assert EXPECTED_TABLES.issubset(tables)
    finally:
        conn2.close()


def test_reset_db_on_connection() -> None:
    """Verify reset_db clears tables when passed an open connection."""
    conn = init_db(":memory:")
    conn.execute(
        """
        INSERT INTO entities (canonical_name, type, created_at)
        VALUES ('Server B', 'server', '2026-01-01T00:00:00Z');
        """
    )
    conn.commit()
    assert conn.execute("SELECT count(*) FROM entities;").fetchone()[0] == 1

    # Reset on connection
    reset_db(conn)
    assert conn.execute("SELECT count(*) FROM entities;").fetchone()[0] == 0
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        ).fetchall()
    }
    assert EXPECTED_TABLES.issubset(tables)
    conn.close()


def test_reset_with_foreign_key_related_records() -> None:
    """1. Verify reset succeeds on database containing interrelated parent/child rows."""
    conn = init_db(":memory:")
    try:
        # Insert parent source
        conn.execute(
            """
            INSERT INTO sources (id, name, type, reliability, notes)
            VALUES ('src_slack', 'Slack Chat', 'chat', 0.8, 'Slack source');
            """
        )
        # Insert child message referencing source
        conn.execute(
            """
            INSERT INTO messages (id, source_id, text, text_hash, event_time, ingested_at)
            VALUES (
                1, 'src_slack', 'msg text', 'hash1',
                '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z'
            );
            """
        )
        # Insert attribute schema
        conn.execute(
            """
            INSERT INTO attribute_schema (
                attribute, description, cardinality, value_type, half_life_days, allowed_values_json
            )
            VALUES ('status', 'Current server status', 'single', 'text', 30.0, '[]');
            """
        )
        # Insert entities
        conn.execute(
            """
            INSERT INTO entities (id, canonical_name, type, created_at)
            VALUES (1, 'Alpha Server', 'server', '2026-01-01T00:00:00Z'),
                   (2, 'Beta Server', 'server', '2026-01-01T00:00:00Z');
            """
        )
        # Insert memory referencing entity, attribute, message, source
        conn.execute(
            """
            INSERT INTO memories (
                id, entity_id, attribute, value_norm, value_display,
                valid_from, recorded_at, status, confidence, message_id, source_id
            )
            VALUES (
                10, 1, 'status', 'active', 'Active',
                '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 'active', 0.9, 1, 'src_slack'
            ),
            (
                20, 2, 'status', 'inactive', 'Inactive',
                '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 'active', 0.9, 1, 'src_slack'
            );
            """
        )
        # Insert evidence referencing memory, source, message
        conn.execute(
            """
            INSERT INTO evidence (memory_id, source_id, message_id, strength, observed_at, span)
            VALUES (10, 'src_slack', 1, 0.9, '2026-01-01T00:00:00Z', 'status is active');
            """
        )
        # Insert edge referencing memories
        conn.execute(
            """
            INSERT INTO edges (
                src_memory_id, dst_memory_id, type, reason, created_at
            )
            VALUES (20, 10, 'supersedes', 'updated status', '2026-01-01T00:00:00Z');
            """
        )
        conn.commit()

        # Reset connection with related records
        reset_db(conn)

        # Confirm all tables empty and schema intact
        assert conn.execute("SELECT count(*) FROM sources;").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM messages;").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM entities;").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM memories;").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM evidence;").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM edges;").fetchone()[0] == 0

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        }
        assert EXPECTED_TABLES.issubset(tables)
    finally:
        conn.close()


def test_reset_when_transaction_is_active() -> None:
    """2. Verify reset cleanly handles an active uncommitted transaction."""
    conn = init_db(":memory:")
    try:
        # Start transaction with uncommitted insert
        conn.execute(
            """
            INSERT INTO entities (canonical_name, type, created_at)
            VALUES ('Uncommitted Server', 'server', '2026-01-01T00:00:00Z');
            """
        )
        assert conn.in_transaction is True

        # Reset should abort/rollback uncommitted work and reset cleanly
        reset_db(conn)

        assert conn.in_transaction is False
        assert conn.execute("SELECT count(*) FROM entities;").fetchone()[0] == 0
    finally:
        conn.close()


def test_foreign_key_enforcement_after_reset() -> None:
    """3. Verify foreign key enforcement is active after reset and rejects orphans."""
    conn = init_db(":memory:")
    try:
        reset_db(conn)

        fk_val = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert fk_val == 1

        # Attempt inserting a child message without valid source
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO messages (source_id, text, text_hash, event_time, ingested_at)
                VALUES (
                    'nonexistent_src',
                    'text',
                    'hash_orphaned',
                    '2026-01-01T00:00:00Z',
                    '2026-01-01T00:00:00Z'
                );
                """
            )
    finally:
        conn.close()


def test_reset_with_second_open_connection(tmp_path: Path) -> None:
    """4. Verify WAL snapshot isolation and reset behavior with active read transaction.

    Expected sequence:
    1. Open two connections to the same WAL database.
    2. Insert and commit a test entity.
    3. Start a read transaction on connection 2 and read the entity.
    4. Reset the database through connection 3.
    5. Verify reset completes safely.
    6. Verify connection 2 can finish its existing transaction (retains initial snapshot).
    7. Start a new transaction on connection 2.
    8. Verify it now sees the empty database.
    9. Verify the complete schema still exists.
    """
    db_file = tmp_path / "concurrent_wal_snapshot.db"

    # 1. Open two connections to the same WAL database
    conn1 = init_db(db_file)
    conn2 = get_connection(db_file)

    # 2. Insert and commit a test entity
    conn1.execute(
        """
        INSERT INTO entities (canonical_name, type, created_at)
        VALUES ('Active Server', 'server', '2026-01-01T00:00:00Z');
        """
    )
    conn1.commit()

    # 3. Start a read transaction on connection 2 and read the entity
    conn2.execute("BEGIN;")
    initial_count = conn2.execute("SELECT count(*) FROM entities;").fetchone()[0]
    assert initial_count == 1

    try:
        # 4. Reset the database through connection 3
        conn3 = reset_db(db_file)
        try:
            # 5. Verify reset completes safely
            reset_count = conn3.execute("SELECT count(*) FROM entities;").fetchone()[0]
            assert reset_count == 0

            # 6. Verify connection 2 can finish its existing transaction (sees snapshot)
            snap_count = conn2.execute("SELECT count(*) FROM entities;").fetchone()[0]
            assert snap_count == 1
            conn2.commit()

            # 7. Start a new transaction on connection 2
            conn2.execute("BEGIN;")

            # 8. Verify it now sees the empty database
            new_count = conn2.execute("SELECT count(*) FROM entities;").fetchone()[0]
            assert new_count == 0
            conn2.commit()

            # 9. Verify the complete schema still exists
            tables = {
                row[0]
                for row in conn2.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                ).fetchall()
            }
            assert EXPECTED_TABLES.issubset(tables)
        finally:
            conn3.close()
    finally:
        if conn2.in_transaction:
            conn2.rollback()
        conn2.close()
        conn1.close()


def test_reset_rejected_by_lock_preserves_database(tmp_path: Path) -> None:
    """4b. Verify that when reset is rejected due to a lock, the database remains intact."""
    db_file = tmp_path / "locked_db.db"
    conn1 = init_db(db_file)
    conn1.execute(
        """
        INSERT INTO entities (canonical_name, type, created_at)
        VALUES ('Locked Server', 'server', '2026-01-01T00:00:00Z');
        """
    )
    conn1.commit()

    # conn2 acquires an immediate exclusive write lock
    conn2 = get_connection(db_file)
    conn2.execute("BEGIN IMMEDIATE;")

    try:
        # Attempting reset while conn2 holds the write lock must fail with OperationalError
        # (Using a short timeout connection to avoid prolonged lock wait)
        conn_locked = sqlite3.connect(str(db_file), timeout=0.01)
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            reset_db(conn_locked)
        conn_locked.close()
    finally:
        conn2.rollback()
        conn2.close()

    # Confirm original database remains completely intact and usable
    assert conn1.execute("SELECT count(*) FROM entities;").fetchone()[0] == 1
    tables = {
        row[0]
        for row in conn1.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        ).fetchall()
    }
    assert EXPECTED_TABLES.issubset(tables)
    conn1.close()


def test_reset_file_backed_wal_database(tmp_path: Path) -> None:
    """5. Verify resetting a file-backed WAL database maintains WAL and clears data."""
    db_file = tmp_path / "wal_test.db"
    conn = init_db(db_file)
    try:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"

        conn.execute(
            """
            INSERT INTO entities (canonical_name, type, created_at)
            VALUES ('Persisted Entity', 'server', '2026-01-01T00:00:00Z');
            """
        )
        conn.commit()
    finally:
        conn.close()

    # Reset file-backed database
    conn_reset = reset_db(db_file)
    try:
        mode_after = conn_reset.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode_after.lower() == "wal"
        assert conn_reset.execute("SELECT count(*) FROM entities;").fetchone()[0] == 0

        tables = {
            row[0]
            for row in conn_reset.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        }
        assert EXPECTED_TABLES.issubset(tables)

        indexes = {
            row[0]
            for row in conn_reset.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        }
        assert EXPECTED_INDEXES.issubset(indexes)
    finally:
        conn_reset.close()


def test_repeated_reset_and_initialization(tmp_path: Path) -> None:
    """6. Verify repeated reset and init calls on both file-backed and connection targets."""
    db_file = tmp_path / "repeated_cycles.db"

    # Repeated file-backed cycles
    for i in range(3):
        conn = init_db(db_file)
        conn.execute(
            f"INSERT INTO entities (canonical_name, type, created_at) "
            f"VALUES ('Server_{i}', 'server', '2026-01-01T00:00:00Z');"
        )
        conn.commit()
        conn.close()

        conn_reset = reset_db(db_file)
        assert conn_reset.execute("SELECT count(*) FROM entities;").fetchone()[0] == 0
        conn_reset.close()

    # Repeated on an active in-memory connection
    conn_mem = init_db(":memory:")
    try:
        for i in range(3):
            conn_mem.execute(
                f"INSERT INTO entities (canonical_name, type, created_at) "
                f"VALUES ('Mem_{i}', 'server', '2026-01-01T00:00:00Z');"
            )
            conn_mem.commit()
            assert conn_mem.execute("SELECT count(*) FROM entities;").fetchone()[0] == 1

            reset_db(conn_mem)
            assert conn_mem.execute("SELECT count(*) FROM entities;").fetchone()[0] == 0
            init_db(conn_mem)
            assert conn_mem.execute("SELECT count(*) FROM entities;").fetchone()[0] == 0
    finally:
        conn_mem.close()


class FailingTeardownConnection(sqlite3.Connection):
    """Custom SQLite connection subclass for simulating teardown failure scenarios."""

    fail_on_drop_table: str | None = None
    fail_on_cleanup: bool = False

    def cursor(self, *args, **kwargs) -> sqlite3.Cursor:
        real_cur = super().cursor(*args, **kwargs)
        parent = self

        class InterceptingCursor:
            def __init__(self, cur: sqlite3.Cursor) -> None:
                self._cur = cur

            def execute(self, sql: str, *a, **kw) -> sqlite3.Cursor:
                if (
                    parent.fail_on_drop_table
                    and "DROP TABLE" in sql
                    and parent.fail_on_drop_table in sql
                ):
                    raise sqlite3.OperationalError(
                        f"Simulated drop failure for {parent.fail_on_drop_table} table"
                    )
                return self._cur.execute(sql, *a, **kw)

            def fetchall(self) -> list:
                return self._cur.fetchall()

            def __getattr__(self, name: str):
                return getattr(self._cur, name)

        return InterceptingCursor(real_cur)  # type: ignore[return-value]

    def execute(self, sql: str, *args, **kwargs) -> sqlite3.Cursor:
        if self.fail_on_cleanup and "PRAGMA foreign_keys = ON" in sql:
            raise sqlite3.OperationalError("Simulated secondary cleanup failure")
        return super().execute(sql, *args, **kwargs)


def test_teardown_rollback_on_failure() -> None:
    """7. Verify teardown failure rolls back dropped tables and restores foreign keys."""
    conn = sqlite3.connect(":memory:", factory=FailingTeardownConnection)
    conn = init_db(conn)
    try:
        # Populate initial entity
        conn.execute(
            """
            INSERT INTO entities (canonical_name, type, created_at)
            VALUES ('Preserved Server', 'server', '2026-01-01T00:00:00Z');
            """
        )
        conn.commit()

        initial_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        }
        assert EXPECTED_TABLES.issubset(initial_tables)

        # Trigger simulated failure when dropping 'messages' table midway through teardown
        conn.fail_on_drop_table = "messages"

        # Reset must fail with the simulated error
        with pytest.raises(
            sqlite3.OperationalError, match="Simulated drop failure for messages table"
        ):
            _reset_connection_tables(conn)

        conn.fail_on_drop_table = None

        # 1. Teardown transaction was rolled back
        assert conn.in_transaction is False

        # 2. Database is NOT left with a partially dropped schema; all tables intact
        remaining_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        }
        assert remaining_tables == initial_tables

        # Data in entities remains intact
        assert conn.execute("SELECT count(*) FROM entities;").fetchone()[0] == 1

        # 3. Foreign key enforcement is restored
        assert conn.execute("PRAGMA foreign_keys;").fetchone()[0] == 1
    finally:
        conn.close()


def test_teardown_preserves_original_exception_when_cleanup_fails() -> None:
    """8. Verify original teardown exception is preserved if cleanup also raises an error."""
    conn = sqlite3.connect(":memory:", factory=FailingTeardownConnection)
    conn = init_db(conn)
    try:
        conn.fail_on_drop_table = "messages"
        conn.fail_on_cleanup = True

        with pytest.raises(
            sqlite3.OperationalError, match="Simulated drop failure for messages table"
        ):
            _reset_connection_tables(conn)
    finally:
        conn.fail_on_cleanup = False
        conn.close()
