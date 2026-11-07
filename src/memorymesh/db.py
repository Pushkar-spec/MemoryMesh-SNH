"""SQLite database connection, schema initialization, and reset foundation.

Governed by docs/SPECIFICATION.md Section 3 and AGENTS.md.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

DEFAULT_DB_PATH = "data/memorymesh.db"


def get_schema_path() -> Path:
    """Return the absolute filesystem path to schema.sql."""
    return Path(__file__).resolve().parent / "schema.sql"


def get_schema_sql() -> str:
    """Read the canonical SQLite schema definition from schema.sql."""
    path = get_schema_path()
    if not path.is_file():
        raise FileNotFoundError(f"Database schema file not found at {path}")
    return path.read_text(encoding="utf-8")


def is_in_memory(db_path: str | Path) -> bool:
    """Determine whether a database target string represents an in-memory database."""
    path_str = str(db_path)
    return path_str == ":memory:" or "mode=memory" in path_str


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled and row factory configured.

    Args:
        db_path: Filesystem path to SQLite file or ':memory:' for transient tests.

    Returns:
        An open sqlite3.Connection with Row factory and foreign keys active.
    """
    path_str = str(db_path)
    if not is_in_memory(path_str):
        Path(path_str).resolve().parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path_str)
    conn.row_factory = sqlite3.Row

    # Foreign keys must be enabled per connection in SQLite
    conn.execute("PRAGMA foreign_keys = ON;")
    fk_row = conn.execute("PRAGMA foreign_keys;").fetchone()
    if not fk_row or not fk_row[0]:
        conn.close()
        raise RuntimeError("Failed to enable foreign keys for SQLite connection.")

    # Enable WAL mode for file-backed databases
    if not is_in_memory(path_str):
        conn.execute("PRAGMA journal_mode = WAL;")

    return conn


@contextmanager
def connection_scope(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager providing an open SQLite connection that closes automatically on exit."""
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(target: str | Path | sqlite3.Connection = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Execute schema.sql to initialize all tables and indexes.

    Safe for repeated executions (uses IF NOT EXISTS constructs).

    Args:
        target: A database path (file or :memory:) or an existing sqlite3.Connection.

    Returns:
        The initialized sqlite3.Connection.
    """
    schema_sql = get_schema_sql()

    if isinstance(target, sqlite3.Connection):
        conn = target
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(schema_sql)
        return conn

    conn = get_connection(target)
    conn.executescript(schema_sql)
    return conn


def _reset_connection_tables(conn: sqlite3.Connection) -> None:
    """Drop all user tables and views on an open SQLite connection transactionally.

    Safety considerations:
    1. Active transactions: SQLite PRAGMA foreign_keys cannot be changed while a
       transaction is active (conn.in_transaction is True). Any active transaction
       is explicitly rolled back so that PRAGMA foreign_keys = OFF takes effect.
    2. Foreign key enforcement: PRAGMA foreign_keys is disabled outside an active
       transaction and is guaranteed to be restored in a finally block.
    3. Explicit teardown transaction: Teardown begins an explicit transaction (BEGIN)
       and commits only after every view and table DROP succeeds. If any DROP fails,
       the teardown transaction is rolled back so no partially dropped schema remains.
    4. Exception preservation: If cleanup encounters an error while restoring foreign
       keys or rolling back, the original teardown exception is preserved.
    """
    if conn.in_transaction:
        conn.rollback()

    # Disable foreign keys outside an active transaction
    conn.execute("PRAGMA foreign_keys = OFF;")

    teardown_error: BaseException | None = None
    try:
        conn.execute("BEGIN;")
        cursor = conn.cursor()

        # Drop views first if any exist
        views = [
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='view' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        ]
        for view in views:
            cursor.execute(f'DROP VIEW IF EXISTS "{view}";')

        tables = [
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        ]
        for table in tables:
            cursor.execute(f'DROP TABLE IF EXISTS "{table}";')

        conn.commit()
    except BaseException as exc:
        teardown_error = exc
        try:
            if conn.in_transaction:
                conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            if conn.in_transaction:
                conn.rollback()
            conn.execute("PRAGMA foreign_keys = ON;")
        except Exception:
            if teardown_error is None:
                raise


def reset_db(target: str | Path | sqlite3.Connection = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Destructively recreate a clean database matching the canonical schema.

    Respects SQLite connection and transaction semantics without manual deletion
    of database files or companion -wal / -shm / -journal files. Operates safely
    even when other connections to the database file are currently open.

    If an active transaction is open on an existing connection, it is explicitly
    rolled back to ensure PRAGMA foreign_keys = OFF takes effect. Foreign key
    enforcement is restored on completion or error.

    Args:
        target: A database path (file or :memory:) or an existing sqlite3.Connection.

    Returns:
        A fresh, initialized sqlite3.Connection with all canonical tables and indexes.
    """
    if isinstance(target, sqlite3.Connection):
        conn = target
        _reset_connection_tables(conn)
        return init_db(conn)

    path_str = str(target)
    if is_in_memory(path_str):
        return init_db(":memory:")

    # File-backed path: open connection and recreate schema in-place via SQL semantics
    conn = get_connection(path_str)
    try:
        _reset_connection_tables(conn)
        return init_db(conn)
    except Exception:
        conn.close()
        raise
