"""SQLite Repository layer for MemoryMesh storage operations.

Governed by docs/SPECIFICATION.md Section 3, docs/AGENTS.md, and docs/ROADMAP_AND_TASKS.md (T1.1).
All SQL queries live exclusively within this module.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Literal

from memorymesh.errors import (
    AliasCollisionError,
    ForeignKeyConstraintError,
    InvalidUpdateFieldError,
    MemoryNotFoundError,
    RepositoryError,
)
from memorymesh.models import (
    Action,
    AttributeDef,
    Cardinality,
    DecisionLogEntry,
    Edge,
    EdgeType,
    Evidence,
    Memory,
    MemoryStatus,
    NewEvidence,
    NewMemory,
    Source,
    ValueType,
    format_datetime_utc,
)

ALLOWED_MEMORY_UPDATE_FIELDS = frozenset(
    {
        "status",
        "valid_to",
        "closed_at",
        "confidence",
        "needs_review",
        "review_reason",
        "value_norm",
        "value_display",
    }
)


def parse_db_datetime(val: str | None) -> datetime | None:
    """Parse an ISO-8601 UTC string from SQLite into a timezone-aware datetime."""
    if val is None:
        return None
    dt = datetime.fromisoformat(val)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def validate_memory_update_field(key: str, val: Any) -> Any:
    """Validate a memory update field value against domain constraints.

    Returns the serialized value suitable for SQLite parameter binding.
    """
    if key == "status":
        if isinstance(val, MemoryStatus):
            return val.value
        if isinstance(val, str) and val in {s.value for s in MemoryStatus}:
            return val
        raise InvalidUpdateFieldError(
            f"Invalid status '{val}'. Must be one of {[s.value for s in MemoryStatus]}."
        )

    if key in ("valid_to", "closed_at"):
        if val is None:
            return None
        if not isinstance(val, datetime):
            raise InvalidUpdateFieldError(
                f"Field '{key}' must be a datetime or None, got {type(val).__name__}."
            )
        return format_datetime_utc(val)

    if key == "confidence":
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise InvalidUpdateFieldError(
                f"Field 'confidence' must be numeric, got {type(val).__name__}."
            )
        num_val = float(val)
        if not (0.0 <= num_val <= 1.0):
            raise InvalidUpdateFieldError(
                f"Field 'confidence' must be between 0.0 and 1.0, got {num_val}."
            )
        return num_val

    if key == "needs_review":
        if not isinstance(val, (bool, int)) or (isinstance(val, int) and val not in (0, 1)):
            raise InvalidUpdateFieldError(
                f"Field 'needs_review' must be a boolean, got {type(val).__name__}."
            )
        return 1 if val else 0

    if key in ("review_reason", "value_norm", "value_display"):
        if not isinstance(val, str):
            raise InvalidUpdateFieldError(
                f"Field '{key}' must be a string, got {type(val).__name__}."
            )
        return val

    return val


def _reject_duplicate_object_pairs(pairs: list[tuple[Any, Any]]) -> dict[str, str]:
    """JSON object pairs hook rejecting duplicate keys, empty/whitespace keys or values,

    and non-string types.
    """
    seen: dict[str, str] = {}
    for k, v in pairs:
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(
                f"Context key '{k}' and value '{v}' must be strings, got "
                f"{type(k).__name__} and {type(v).__name__}."
            )
        k_clean = k.strip()
        v_clean = v.strip()
        if not k_clean:
            raise ValueError("Context key cannot be empty or whitespace.")
        if not v_clean:
            raise ValueError(f"Context value for key '{k_clean}' cannot be empty or whitespace.")
        if k_clean in seen:
            raise ValueError(
                f"Duplicate context key in JSON: '{k_clean}'. Duplicate keys are not allowed."
            )
        seen[k_clean] = v_clean
    return seen


def _canonicalize_context_key(key: str | None) -> str | None:
    """Normalize legacy 'k=v;k=v' context key strings to canonical JSON context keys.

    Safely converts valid legacy semicolon-delimited key=value strings into canonical
    sorted compact JSON context keys for repository queries.

    Limitations and Safety Invariants:
        - Rejects duplicate keys in JSON objects during parsing via object_pairs_hook.
        - Rejects malformed JSON strings starting with '{' that fail JSON decoding.
        - Rejects empty legacy tokens rather than discarding them.
        - Rejects non-JSON strings without valid 'key=value' structure (e.g. 'malformed').
        - Rejects tokens with ambiguous multiple '=' delimiters.
        - Rejects empty keys, empty values, and duplicate keys.
    """
    if key is None or not key:
        return key

    if key.startswith("{"):
        try:
            parsed = json.loads(key, object_pairs_hook=_reject_duplicate_object_pairs)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed canonical JSON context key: '{key}'") from exc
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Malformed canonical JSON context key: '{key}'") from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Malformed canonical JSON context key: '{key}'. "
                "Must be a dict of string keys/values."
            )
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    if "=" not in key:
        raise ValueError(
            f"Invalid context key '{key}': "
            "must be valid canonical JSON or valid legacy 'key=value' format."
        )

    raw_tokens = key.split(";")
    cleaned: dict[str, str] = {}
    for raw_token in raw_tokens:
        token = raw_token.strip()
        if not token:
            raise ValueError(
                f"Empty token in legacy context key '{key}'. Malformed delimiters are not allowed."
            )
        if "=" not in token:
            raise ValueError(
                f"Malformed legacy context key token '{token}' in '{key}': missing '=' delimiter."
            )
        if token.count("=") > 1:
            raise ValueError(
                f"Ambiguous legacy context key token '{token}' in '{key}': multiple '=' delimiters."
            )
        k, v = token.split("=", 1)
        k_clean = k.strip()
        v_clean = v.strip()
        if not k_clean or not v_clean:
            raise ValueError(
                f"Malformed legacy context key token '{token}' in '{key}': "
                "key and value must be non-empty."
            )
        if k_clean in cleaned:
            raise ValueError(
                f"Duplicate legacy context key '{k_clean}' in '{key}'. "
                "Lossless canonical conversion is impossible."
            )
        cleaned[k_clean] = v_clean

    if not cleaned:
        return ""
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class Repository:
    """Encapsulates all SQLite database interactions with typed domain models."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Execute operations within an atomic SQLite transaction.

        Supports nested calls via SQLite SAVEPOINTS. Commits on successful exit,
        rolls back on exception, and preserves connection usability and foreign-key enforcement.
        """
        nested = self.conn.in_transaction
        sp_id = f"sp_{uuid.uuid4().hex}"
        if nested:
            self.conn.execute(f"SAVEPOINT {sp_id};")
        else:
            self.conn.execute("BEGIN IMMEDIATE;")

        try:
            yield
            if nested:
                self.conn.execute(f"RELEASE SAVEPOINT {sp_id};")
            else:
                self.conn.commit()
        except BaseException:
            if nested:
                self.conn.execute(f"ROLLBACK TO SAVEPOINT {sp_id};")
                self.conn.execute(f"RELEASE SAVEPOINT {sp_id};")
            else:
                if self.conn.in_transaction:
                    self.conn.rollback()
            raise

    # =========================================================================
    # Memory Operations
    # =========================================================================

    def insert_memory(
        self,
        memory: NewMemory | Memory,
        recorded_at: datetime | None = None,
    ) -> int:
        """Insert a new memory record and return its generated integer ID."""
        eff_recorded_at = recorded_at or getattr(memory, "recorded_at", None) or datetime.now(UTC)
        valid_from_str = format_datetime_utc(memory.valid_from)
        valid_to_str = format_datetime_utc(memory.valid_to) if memory.valid_to else None
        recorded_at_str = format_datetime_utc(eff_recorded_at)
        closed_at_str = format_datetime_utc(memory.closed_at) if memory.closed_at else None
        context_json = json.dumps(memory.context or {})
        status_str = memory.status.value if hasattr(memory.status, "value") else str(memory.status)

        sql = """
        INSERT INTO memories (
            entity_id, attribute, value_norm, value_display,
            context_json, context_key, valid_from, valid_to,
            recorded_at, closed_at, status, confidence,
            needs_review, review_reason, raw_text, message_id, source_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            memory.entity_id,
            memory.attribute,
            memory.value_norm,
            memory.value_display,
            context_json,
            memory.context_key,
            valid_from_str,
            valid_to_str,
            recorded_at_str,
            closed_at_str,
            status_str,
            float(memory.confidence),
            1 if getattr(memory, "needs_review", False) else 0,
            getattr(memory, "review_reason", ""),
            getattr(memory, "raw_text", ""),
            memory.message_id,
            memory.source_id,
        )

        try:
            cursor = self.conn.execute(sql, params)
            return cursor.lastrowid or 0
        except sqlite3.IntegrityError as exc:
            if "FOREIGN KEY" in str(exc).upper():
                raise ForeignKeyConstraintError(
                    f"Foreign key constraint failed on memory insertion: {exc}"
                ) from exc
            raise RepositoryError(f"Database integrity error: {exc}") from exc

    def get_memory(self, memory_id: int, include_evidence: bool = True) -> Memory | None:
        """Retrieve a Memory by ID, optionally populating its associated evidence list."""
        sql = """
        SELECT m.*, e.canonical_name AS entity_name
        FROM memories m
        LEFT JOIN entities e ON m.entity_id = e.id
        WHERE m.id = ?
        """
        cursor = self.conn.execute(sql, (memory_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_memory(row, include_evidence=include_evidence)

    def update_memory(
        self,
        memory_id: int,
        fields: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Update existing memory fields using an explicit allowlist and parameterized SQL.

        Validates resulting validity-window state (valid_to requires closed_at) before
        executing the update, ensuring database invariants are preserved.
        """
        updates: dict[str, Any] = {}
        if fields:
            updates.update(fields)
        if kwargs:
            updates.update(kwargs)

        if not updates:
            return

        # Validate against explicit allowlist to prevent arbitrary SQL injection
        disallowed = set(updates.keys()) - ALLOWED_MEMORY_UPDATE_FIELDS
        if disallowed:
            raise InvalidUpdateFieldError(
                f"Field(s) {sorted(disallowed)} are not allowed for memory updates. "
                f"Allowed fields: {sorted(ALLOWED_MEMORY_UPDATE_FIELDS)}"
            )

        # Validate field-level types and values first
        serialized_updates: dict[str, Any] = {}
        for key in sorted(updates.keys()):
            serialized_updates[key] = validate_memory_update_field(key, updates[key])

        # Retrieve existing memory row to verify existence and check resulting state
        row = self.conn.execute(
            "SELECT valid_to, closed_at FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            raise MemoryNotFoundError(f"Memory with ID {memory_id} does not exist.")

        # If either valid_to or closed_at is being updated, validate resulting combined state
        if "valid_to" in updates or "closed_at" in updates:
            resulting_valid_to = (
                serialized_updates["valid_to"] if "valid_to" in updates else row["valid_to"]
            )
            resulting_closed_at = (
                serialized_updates["closed_at"] if "closed_at" in updates else row["closed_at"]
            )

            # Invariant: valid_to IS NOT NULL implies closed_at IS NOT NULL
            if resulting_valid_to is not None and resulting_closed_at is None:
                raise InvalidUpdateFieldError(
                    f"Resulting memory state is invalid: valid_to is set ({resulting_valid_to}) "
                    "but closed_at is NULL."
                )

        set_clauses: list[str] = [f"{k} = ?" for k in sorted(serialized_updates.keys())]
        params: list[Any] = [serialized_updates[k] for k in sorted(serialized_updates.keys())]
        params.append(memory_id)

        sql = f"UPDATE memories SET {', '.join(set_clauses)} WHERE id = ?"
        self.conn.execute(sql, params)

    def close_memory(
        self,
        memory_id: int,
        valid_to: datetime,
        closed_at: datetime,
        status: MemoryStatus | str = MemoryStatus.HISTORICAL,
    ) -> None:
        """Close an active memory's validity window and update its lifecycle status."""
        self.update_memory(
            memory_id,
            {
                "status": status,
                "valid_to": valid_to,
                "closed_at": closed_at,
            },
        )

    def flag_memory_review(self, memory_id: int, reason: str) -> None:
        """Flag a memory as needing review with a specified reason."""
        self.update_memory(
            memory_id,
            {
                "needs_review": True,
                "review_reason": reason,
            },
        )

    def update_memory_confidence(self, memory_id: int, confidence: float) -> None:
        """Update a memory's persisted base confidence score."""
        conf_val = float(confidence)
        if not (0.0 <= conf_val <= 1.0):
            raise RepositoryError(f"Confidence must be between 0.0 and 1.0, got {conf_val}.")
        self.update_memory(memory_id, {"confidence": conf_val})

    def get_memory_chain(
        self,
        entity_id: int,
        attribute: str,
        context_key: str = "",
        include_rejected: bool = False,
    ) -> list[Memory]:
        """Retrieve all memories for a slot ordered by valid_from ASC."""
        sql = """
        SELECT m.*, e.canonical_name AS entity_name
        FROM memories m
        LEFT JOIN entities e ON m.entity_id = e.id
        WHERE m.entity_id = ? AND m.attribute = ? AND m.context_key = ?
        """
        params: list[Any] = [entity_id, attribute, _canonicalize_context_key(context_key) or ""]
        if not include_rejected:
            sql += " AND m.status != 'rejected'"

        sql += " ORDER BY m.valid_from ASC, m.id ASC"

        cursor = self.conn.execute(sql, params)
        rows = cursor.fetchall()
        return [self._row_to_memory(r, include_evidence=True) for r in rows]

    def find_memories(
        self,
        entity_id: int,
        attribute: str,
        *,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        context_key: str | None = None,
        include_rejected: bool = False,
    ) -> list[Memory]:
        """Find memories matching temporal criteria, ordered valid_from DESC, id DESC."""
        clauses = ["m.entity_id = ?", "m.attribute = ?"]
        params: list[Any] = [entity_id, attribute]

        if not include_rejected:
            clauses.append("m.status != 'rejected'")

        if context_key is not None:
            clauses.append("m.context_key = ?")
            params.append(_canonicalize_context_key(context_key))

        if known_at is not None:
            known_str = format_datetime_utc(known_at)
            clauses.append("m.recorded_at <= ?")
            params.append(known_str)

        if valid_at is not None:
            valid_str = format_datetime_utc(valid_at)
            clauses.append("m.valid_from <= ?")
            params.append(valid_str)

            if known_at is not None:
                clauses.append("(m.valid_to IS NULL OR m.valid_to > ? OR m.closed_at > ?)")
                params.append(valid_str)
                params.append(known_str)
            else:
                clauses.append("(m.valid_to IS NULL OR m.valid_to > ?)")
                params.append(valid_str)

        sql = f"""
        SELECT m.*, e.canonical_name AS entity_name
        FROM memories m
        LEFT JOIN entities e ON m.entity_id = e.id
        WHERE {" AND ".join(clauses)}
        ORDER BY m.valid_from DESC, m.id DESC
        """
        cursor = self.conn.execute(sql, params)
        rows = cursor.fetchall()
        return [self._row_to_memory(r, include_evidence=True) for r in rows]

    def _row_to_memory(self, row: sqlite3.Row, include_evidence: bool = True) -> Memory:
        """Convert a SQLite Row into a typed Memory model."""
        memory_id = int(row["id"])
        evidence_list: list[Evidence] = []
        if include_evidence:
            evidence_list = self.get_evidence_for_memory(memory_id)

        return Memory(
            id=memory_id,
            entity_id=int(row["entity_id"]),
            entity_name=str(row["entity_name"] or ""),
            attribute=str(row["attribute"]),
            value_norm=str(row["value_norm"]),
            value_display=str(row["value_display"]),
            context=json.loads(row["context_json"] or "{}"),
            context_key=str(row["context_key"] or ""),
            valid_from=parse_db_datetime(row["valid_from"]),  # type: ignore[arg-type]
            valid_to=parse_db_datetime(row["valid_to"]),
            recorded_at=parse_db_datetime(row["recorded_at"]),  # type: ignore[arg-type]
            closed_at=parse_db_datetime(row["closed_at"]),
            status=MemoryStatus(row["status"]),
            confidence=float(row["confidence"]),
            needs_review=bool(row["needs_review"]),
            review_reason=str(row["review_reason"] or ""),
            raw_text=str(row["raw_text"] or ""),
            message_id=int(row["message_id"]),
            source_id=str(row["source_id"]),
            evidence=evidence_list,
        )

    # =========================================================================
    # Evidence Operations
    # =========================================================================

    def insert_evidence(self, memory_id: int, evidence: NewEvidence | Evidence) -> int:
        """Insert an evidence row associated with a memory and return its ID.

        If an identical evidence row for (memory_id, source_id, message_id, span) already
        exists, returns the existing evidence ID without creating a duplicate.
        """
        str_val = float(evidence.strength)
        if not (0.0 <= str_val <= 1.0):
            raise RepositoryError(f"Evidence strength must be between 0.0 and 1.0, got {str_val}.")

        # Idempotency check: preserve existing evidence and avoid duplicate records
        existing = self.conn.execute(
            """
            SELECT id FROM evidence
            WHERE memory_id = ? AND source_id = ? AND message_id = ? AND span = ?
            """,
            (memory_id, evidence.source_id, evidence.message_id, evidence.span or ""),
        ).fetchone()
        if existing is not None:
            return int(existing["id"])

        sql = """
        INSERT INTO evidence (memory_id, source_id, message_id, strength, observed_at, span)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        observed_str = format_datetime_utc(evidence.observed_at)
        params = (
            memory_id,
            evidence.source_id,
            evidence.message_id,
            str_val,
            observed_str,
            evidence.span or "",
        )
        try:
            cursor = self.conn.execute(sql, params)
            return cursor.lastrowid or 0
        except sqlite3.IntegrityError as exc:
            if "FOREIGN KEY" in str(exc).upper():
                raise ForeignKeyConstraintError(
                    f"Foreign key constraint failed on evidence insertion: {exc}"
                ) from exc
            raise RepositoryError(f"Database integrity error: {exc}") from exc

    def get_evidence(self, evidence_id: int) -> Evidence | None:
        """Retrieve a single evidence row by ID."""
        sql = "SELECT * FROM evidence WHERE id = ?"
        row = self.conn.execute(sql, (evidence_id,)).fetchone()
        if row is None:
            return None
        return Evidence(
            id=int(row["id"]),
            memory_id=int(row["memory_id"]),
            source_id=str(row["source_id"]),
            message_id=int(row["message_id"]),
            strength=float(row["strength"]),
            observed_at=parse_db_datetime(row["observed_at"]),  # type: ignore[arg-type]
            span=str(row["span"] or ""),
        )

    def get_evidence_for_memory(self, memory_id: int) -> list[Evidence]:
        """Retrieve all evidence rows linked to a specific memory ID."""
        sql = "SELECT * FROM evidence WHERE memory_id = ? ORDER BY id ASC"
        cursor = self.conn.execute(sql, (memory_id,))
        rows = cursor.fetchall()
        return [
            Evidence(
                id=int(r["id"]),
                memory_id=int(r["memory_id"]),
                source_id=str(r["source_id"]),
                message_id=int(r["message_id"]),
                strength=float(r["strength"]),
                observed_at=parse_db_datetime(r["observed_at"]),  # type: ignore[arg-type]
                span=str(r["span"] or ""),
            )
            for r in rows
        ]

    # =========================================================================
    # Edge Operations
    # =========================================================================

    def insert_edge(
        self,
        src_memory_id: int,
        dst_memory_id: int,
        edge_type: EdgeType | str,
        reason: str = "",
        created_at: datetime | None = None,
    ) -> int:
        """Insert a relationship edge between two memories."""
        type_str = edge_type.value if hasattr(edge_type, "value") else str(edge_type)
        created_str = format_datetime_utc(created_at or datetime.now(UTC))

        sql = """
        INSERT INTO edges (src_memory_id, dst_memory_id, type, reason, created_at)
        VALUES (?, ?, ?, ?, ?)
        """
        params = (src_memory_id, dst_memory_id, type_str, reason, created_str)
        try:
            cursor = self.conn.execute(sql, params)
            return cursor.lastrowid or 0
        except sqlite3.IntegrityError as exc:
            if "FOREIGN KEY" in str(exc).upper():
                raise ForeignKeyConstraintError(
                    f"Foreign key constraint failed on edge insertion: {exc}"
                ) from exc
            raise RepositoryError(f"Database integrity error: {exc}") from exc

    def get_edges_for_memory(
        self,
        memory_id: int,
        direction: Literal["src", "dst", "both"] = "both",
    ) -> list[Edge]:
        """Retrieve relationship edges where memory_id is source, destination, or either."""
        if direction == "src":
            sql = "SELECT * FROM edges WHERE src_memory_id = ? ORDER BY id ASC"
            cursor = self.conn.execute(sql, (memory_id,))
        elif direction == "dst":
            sql = "SELECT * FROM edges WHERE dst_memory_id = ? ORDER BY id ASC"
            cursor = self.conn.execute(sql, (memory_id,))
        else:
            sql = "SELECT * FROM edges WHERE src_memory_id = ? OR dst_memory_id = ? ORDER BY id ASC"
            cursor = self.conn.execute(sql, (memory_id, memory_id))

        rows = cursor.fetchall()
        return [self._row_to_edge(r) for r in rows]

    def _row_to_edge(self, r: sqlite3.Row) -> Edge:
        """Convert a database row into an Edge domain object."""
        return Edge(
            id=int(r["id"]),
            src_memory_id=int(r["src_memory_id"]),
            dst_memory_id=int(r["dst_memory_id"]),
            type=EdgeType(r["type"]),
            reason=str(r["reason"] or ""),
            created_at=parse_db_datetime(r["created_at"]),  # type: ignore[arg-type]
        )

    def get_dependency_edges(
        self,
        memory_id: int,
        direction: Literal["dependencies", "dependents", "both"] = "both",
    ) -> list[Edge]:
        """Retrieve DEPENDS_ON relationship edges.

        Convention:
            dependent --DEPENDS_ON--> upstream
            src_memory_id = dependent (downstream)
            dst_memory_id = upstream (foundation)

        Args:
            memory_id: Target memory ID.
            direction:
                - 'dependencies': memories that memory_id depends on
                - 'dependents': memories that depend on memory_id
                - 'both': either direction
        """
        if direction == "dependencies":
            sql = (
                "SELECT * FROM edges "
                "WHERE src_memory_id = ? AND type = 'depends_on' "
                "ORDER BY id ASC"
            )
            cursor = self.conn.execute(sql, (memory_id,))
        elif direction == "dependents":
            sql = (
                "SELECT * FROM edges "
                "WHERE dst_memory_id = ? AND type = 'depends_on' "
                "ORDER BY id ASC"
            )
            cursor = self.conn.execute(sql, (memory_id,))
        else:
            sql = (
                "SELECT * FROM edges "
                "WHERE (src_memory_id = ? OR dst_memory_id = ?) AND type = 'depends_on' "
                "ORDER BY id ASC"
            )
            cursor = self.conn.execute(sql, (memory_id, memory_id))

        rows = cursor.fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_transitive_dependency_edges(
        self,
        root_ids: Sequence[int],
        direction: Literal["downstream", "upstream"] = "downstream",
    ) -> list[Edge]:
        """Retrieve transitive DEPENDS_ON edges reachable from root_ids.

        Uses recursive CTE with automatic edge deduplication across cycles.

        Args:
            root_ids: Starting memory IDs.
            direction:
                - 'downstream': trace dependents of root_ids
                - 'upstream': trace foundations of root_ids
        """
        if not root_ids:
            return []

        id_list = list(root_ids)
        placeholders = ",".join("?" for _ in id_list)

        if direction == "downstream":
            sql = f"""
            WITH RECURSIVE dep_graph(edge_id, src_id, dst_id) AS (
                SELECT id, src_memory_id, dst_memory_id
                FROM edges
                WHERE dst_memory_id IN ({placeholders}) AND type = 'depends_on'
                UNION
                SELECT e.id, e.src_memory_id, e.dst_memory_id
                FROM edges e
                JOIN dep_graph dg ON e.dst_memory_id = dg.src_id
                WHERE e.type = 'depends_on'
            )
            SELECT e.*
            FROM edges e
            JOIN (SELECT DISTINCT edge_id FROM dep_graph) dg ON e.id = dg.edge_id
            ORDER BY e.id ASC;
            """
        else:
            sql = f"""
            WITH RECURSIVE dep_graph(edge_id, src_id, dst_id) AS (
                SELECT id, src_memory_id, dst_memory_id
                FROM edges
                WHERE src_memory_id IN ({placeholders}) AND type = 'depends_on'
                UNION
                SELECT e.id, e.src_memory_id, e.dst_memory_id
                FROM edges e
                JOIN dep_graph dg ON e.src_memory_id = dg.dst_id
                WHERE e.type = 'depends_on'
            )
            SELECT e.*
            FROM edges e
            JOIN (SELECT DISTINCT edge_id FROM dep_graph) dg ON e.id = dg.edge_id
            ORDER BY e.id ASC;
            """

        cursor = self.conn.execute(sql, id_list)
        rows = cursor.fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_memories_by_ids(
        self,
        memory_ids: Sequence[int],
        include_evidence: bool = False,
    ) -> list[Memory]:
        """Retrieve memories matching the given IDs in deterministic ID order."""
        if not memory_ids:
            return []

        id_list = sorted(set(memory_ids))
        placeholders = ",".join("?" for _ in id_list)
        sql = f"""
        SELECT m.*, e.canonical_name AS entity_name
        FROM memories m
        LEFT JOIN entities e ON m.entity_id = e.id
        WHERE m.id IN ({placeholders})
        ORDER BY m.id ASC
        """
        rows = self.conn.execute(sql, id_list).fetchall()
        return [self._row_to_memory(r, include_evidence=include_evidence) for r in rows]

    def mark_memory_for_review(self, memory_id: int, review_reason: str) -> None:
        """Mark a memory as requiring review due to dependency propagation."""
        sql = "UPDATE memories SET needs_review = 1, review_reason = ? WHERE id = ?"
        self.conn.execute(sql, (review_reason, memory_id))

    def acknowledge_memory_review(self, memory_id: int) -> None:
        """Clear review flag and review reason for an acknowledged memory."""
        sql = "UPDATE memories SET needs_review = 0, review_reason = '' WHERE id = ?"
        self.conn.execute(sql, (memory_id,))

    def get_superseded_ancestor_values(self, memory_ids: list[int] | set[int]) -> set[str]:
        """Retrieve historical values superseded by memory_ids through the supersession chain.

        Recursively traces edges where type='supersedes' (src_memory_id -> dst_memory_id),
        returning distinct normalized values of historical memories reachable from input IDs.
        """
        if not memory_ids:
            return set()

        id_list = list(memory_ids)
        placeholders = ",".join("?" for _ in id_list)
        sql = f"""
        WITH RECURSIVE supersession_chain(ancestor_id) AS (
            SELECT dst_memory_id
            FROM edges
            WHERE src_memory_id IN ({placeholders}) AND type = 'supersedes'
            UNION
            SELECT e.dst_memory_id
            FROM edges e
            JOIN supersession_chain sc ON e.src_memory_id = sc.ancestor_id
            WHERE e.type = 'supersedes'
        )
        SELECT DISTINCT m.value_norm
        FROM memories m
        JOIN supersession_chain sc ON m.id = sc.ancestor_id
        WHERE m.status = 'historical';
        """
        cursor = self.conn.execute(sql, id_list)
        return {str(row[0]) for row in cursor.fetchall()}

    def get_dispute_group(self, memory_id: int) -> list[Memory]:
        """Retrieve the complete connected component of disputed memories for memory_id.

        Traverses edges of type 'contradicts' starting from memory_id, collecting all
        memories that share the same entity_id, attribute, and context_key and have
        status == MemoryStatus.DISPUTED.
        """
        root = self.get_memory(memory_id)
        if root is None or root.status != MemoryStatus.DISPUTED:
            return []

        visited: set[int] = {root.id}
        queue: list[Memory] = [root]
        group: list[Memory] = [root]

        while queue:
            curr = queue.pop(0)
            sql = """
            SELECT src_memory_id, dst_memory_id
            FROM edges
            WHERE (src_memory_id = ? OR dst_memory_id = ?)
              AND type = 'contradicts'
            """
            cursor = self.conn.execute(sql, (curr.id, curr.id))
            for row in cursor.fetchall():
                nbr_id = int(
                    row["dst_memory_id"]
                    if row["src_memory_id"] == curr.id
                    else row["src_memory_id"]
                )
                if nbr_id not in visited:
                    visited.add(nbr_id)
                    nbr = self.get_memory(nbr_id)
                    if (
                        nbr is not None
                        and nbr.status == MemoryStatus.DISPUTED
                        and nbr.entity_id == root.entity_id
                        and nbr.attribute == root.attribute
                        and nbr.context_key == root.context_key
                    ):
                        queue.append(nbr)
                        group.append(nbr)

        return group

    def find_disputed_pairs(
        self,
        entity_id: int | None = None,
        attribute: str | None = None,
        context_key: str | None = None,
    ) -> list[tuple[Memory, Memory, str]]:
        """Retrieve unique pairs of memories connected by active contradicts edges.

        Both memories in each returned pair must currently have status == 'disputed'
        and identical entity_id, attribute, and context_key.
        """
        clauses = [
            "e.type = 'contradicts'",
            "ma.status = 'disputed'",
            "mb.status = 'disputed'",
            "ma.entity_id = mb.entity_id",
            "ma.attribute = mb.attribute",
            "ma.context_key = mb.context_key",
        ]
        params: list[Any] = []

        if entity_id is not None:
            clauses.append("ma.entity_id = ?")
            params.append(entity_id)
        if attribute is not None:
            clauses.append("ma.attribute = ?")
            params.append(attribute)
        if context_key is not None:
            clauses.append("ma.context_key = ?")
            params.append(_canonicalize_context_key(context_key))

        sql = f"""
        SELECT
            e.id AS edge_id,
            e.src_memory_id,
            e.dst_memory_id,
            e.reason AS edge_reason
        FROM edges e
        JOIN memories ma ON e.src_memory_id = ma.id
        JOIN memories mb ON e.dst_memory_id = mb.id
        WHERE {" AND ".join(clauses)}
        ORDER BY e.id ASC
        """
        cursor = self.conn.execute(sql, params)
        rows = cursor.fetchall()

        seen_pairs: set[tuple[int, int]] = set()
        pairs: list[tuple[Memory, Memory, str]] = []
        for row in rows:
            id_a = int(row["src_memory_id"])
            id_b = int(row["dst_memory_id"])
            pair_key = (min(id_a, id_b), max(id_a, id_b))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            mem_a = self.get_memory(pair_key[0])
            mem_b = self.get_memory(pair_key[1])
            if mem_a is not None and mem_b is not None:
                pairs.append((mem_a, mem_b, str(row["edge_reason"] or "")))

        return pairs

    # =========================================================================
    # Seeding and Infrastructure Operations
    # =========================================================================

    def insert_source(self, source: Source) -> None:
        """Insert or update an information source definition using UPSERT."""
        sql = """
        INSERT INTO sources (id, name, type, reliability, notes)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            type = excluded.type,
            reliability = excluded.reliability,
            notes = excluded.notes
        """
        self.conn.execute(
            sql,
            (source.id, source.name, source.type, float(source.reliability), source.notes),
        )

    def insert_entity(
        self,
        canonical_name: str,
        entity_type: str = "thing",
        needs_review: bool = False,
        created_at: datetime | None = None,
    ) -> int:
        """Insert a new entity record and return its ID."""
        sql = """
        INSERT INTO entities (canonical_name, type, needs_review, created_at)
        VALUES (?, ?, ?, ?)
        """
        created_str = format_datetime_utc(created_at or datetime.now(UTC))
        cursor = self.conn.execute(
            sql,
            (canonical_name, entity_type, 1 if needs_review else 0, created_str),
        )
        return cursor.lastrowid or 0

    def insert_attribute_schema(self, attribute_def: AttributeDef) -> None:
        """Insert or update an attribute catalog definition using UPSERT."""
        sql = """
        INSERT INTO attribute_schema (
            attribute, description, cardinality, value_type, half_life_days, allowed_values_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(attribute) DO UPDATE SET
            description = excluded.description,
            cardinality = excluded.cardinality,
            value_type = excluded.value_type,
            half_life_days = excluded.half_life_days,
            allowed_values_json = excluded.allowed_values_json
        """
        self.conn.execute(
            sql,
            (
                attribute_def.attribute,
                attribute_def.description,
                attribute_def.cardinality.value,
                attribute_def.value_type.value,
                attribute_def.half_life_days,
                json.dumps(attribute_def.allowed_values),
            ),
        )

    def insert_message(
        self,
        source_id: str,
        text: str,
        event_time: datetime,
        ingested_at: datetime | None = None,
        text_hash: str | None = None,
    ) -> int:
        """Insert an incoming raw message record and return its ID."""
        eff_ingested = ingested_at or event_time
        computed_hash = text_hash or hashlib.sha256(text.encode("utf-8")).hexdigest()

        sql = """
        INSERT INTO messages (source_id, text, text_hash, event_time, ingested_at)
        VALUES (?, ?, ?, ?, ?)
        """
        cursor = self.conn.execute(
            sql,
            (
                source_id,
                text,
                computed_hash,
                format_datetime_utc(event_time),
                format_datetime_utc(eff_ingested),
            ),
        )
        return cursor.lastrowid or 0

    def get_source(self, source_id: str) -> Source | None:
        """Retrieve a source definition from the database by ID."""
        sql = "SELECT * FROM sources WHERE id = ?"
        row = self.conn.execute(sql, (source_id,)).fetchone()
        if row is None:
            return None
        return Source(
            id=str(row["id"]),
            name=str(row["name"]),
            type=row["type"],
            reliability=float(row["reliability"]),
            notes=str(row["notes"] or ""),
        )

    def get_entity(self, entity_id: int) -> dict[str, Any] | None:
        """Find an entity record by integer ID."""
        sql = "SELECT * FROM entities WHERE id = ?"
        row = self.conn.execute(sql, (entity_id,)).fetchone()
        return dict(row) if row else None

    def get_entity_by_name(self, name: str) -> dict[str, Any] | None:
        """Find an entity by canonical_name with case-insensitive fallback."""
        sql = "SELECT * FROM entities WHERE canonical_name = ?"
        row = self.conn.execute(sql, (name,)).fetchone()
        if row is None:
            sql_ci = "SELECT * FROM entities WHERE LOWER(canonical_name) = LOWER(?)"
            row = self.conn.execute(sql_ci, (name,)).fetchone()
        return dict(row) if row else None

    def insert_entity_alias(self, alias_norm: str, entity_id: int) -> bool:
        """Insert a normalized entity alias mapping preserving existing ownership.

        If the alias already points to `entity_id`, this is an idempotent no-op.
        If the alias is already owned by a different entity, an AliasCollisionError
        is raised to prevent silent transfer.
        """
        row = self.conn.execute(
            "SELECT entity_id FROM entity_aliases WHERE alias_norm = ?",
            (alias_norm,),
        ).fetchone()

        if row is not None:
            existing_id = int(row["entity_id"])
            if existing_id != entity_id:
                raise AliasCollisionError(
                    f"Alias '{alias_norm}' is already assigned to entity {existing_id}; "
                    f"cannot reassign to entity {entity_id}."
                )
            return False

        self.conn.execute(
            "INSERT INTO entity_aliases (alias_norm, entity_id) VALUES (?, ?)",
            (alias_norm, entity_id),
        )
        return True

    def get_entity_by_alias(self, alias_norm: str) -> dict[str, Any] | None:
        """Find an entity record via its normalized alias in entity_aliases table."""
        sql = """
        SELECT e.* FROM entities e
        JOIN entity_aliases a ON e.id = a.entity_id
        WHERE a.alias_norm = ?
        """
        row = self.conn.execute(sql, (alias_norm,)).fetchone()
        return dict(row) if row else None

    def list_entities(self) -> list[dict[str, Any]]:
        """Retrieve all entity records ordered by ID."""
        sql = "SELECT * FROM entities ORDER BY id ASC"
        rows = self.conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def list_entity_aliases(self) -> list[dict[str, Any]]:
        """Retrieve all entity alias records."""
        sql = "SELECT * FROM entity_aliases"
        rows = self.conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def get_attribute_schema(self, attribute: str) -> AttributeDef | None:
        """Retrieve an attribute catalog definition from the database."""
        sql = "SELECT * FROM attribute_schema WHERE attribute = ?"
        row = self.conn.execute(sql, (attribute,)).fetchone()
        if row is None:
            return None
        return AttributeDef(
            attribute=str(row["attribute"]),
            description=str(row["description"]),
            cardinality=Cardinality(row["cardinality"]),
            value_type=ValueType(row["value_type"]),
            half_life_days=(
                float(row["half_life_days"]) if row["half_life_days"] is not None else None
            ),
            allowed_values=json.loads(row["allowed_values_json"]),
        )

    def get_message_by_hash(
        self, source_id: str, text_hash: str, event_time: datetime
    ) -> int | None:
        """Find an existing message by (source_id, text_hash, event_time)."""
        sql = """
        SELECT id FROM messages
        WHERE source_id = ? AND text_hash = ? AND event_time = ?
        """
        cursor = self.conn.execute(sql, (source_id, text_hash, format_datetime_utc(event_time)))
        row = cursor.fetchone()
        return int(row["id"]) if row else None

    def get_message(self, message_id: int) -> dict[str, Any] | None:
        """Retrieve a message record by primary key ID."""
        sql = "SELECT * FROM messages WHERE id = ?"
        row = self.conn.execute(sql, (message_id,)).fetchone()
        return dict(row) if row else None

    def insert_decision_log(self, entry: DecisionLogEntry) -> DecisionLogEntry:
        """Insert a decision log row and return the persisted entry with its generated ID."""
        sql = """
        INSERT INTO decision_log (
            message_id, claim_json, action, rule_id,
            memory_id, affected_ids_json, scores_json, explanation, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        claim_json_str = json.dumps(entry.claim_json)
        affected_ids_str = json.dumps(entry.affected_memory_ids)
        scores_str = json.dumps(entry.scores)
        created_at_str = format_datetime_utc(entry.created_at)
        action_str = entry.action.value if hasattr(entry.action, "value") else str(entry.action)

        cursor = self.conn.execute(
            sql,
            (
                entry.message_id,
                claim_json_str,
                action_str,
                entry.rule_id,
                entry.memory_id,
                affected_ids_str,
                scores_str,
                entry.explanation,
                created_at_str,
            ),
        )
        persisted_id = cursor.lastrowid or 0
        return entry.model_copy(update={"id": persisted_id})

    def has_propagation_decision_log(
        self,
        memory_id: int,
        upstream_id: int,
        message_id: int,
        cause: str | None = None,
        event_id: str | None = None,
        trigger_mode: str | None = None,
    ) -> bool:
        """Check if an R-PROPAGATE decision log already exists for a specific propagation event.

        Identity of a propagation event:
        - memory_id: the dependent memory flagged for review
        - upstream_id: the changed root memory causing the propagation
        - message_id: the message attributed to this event
        - trigger_mode (if provided): invocation mode ("automatic" vs "manual")
        - event_id (if provided): explicit stable event identifier
        - cause (if event_id omitted): the cause text characterizing this event
        """
        conditions = [
            "rule_id = 'R-PROPAGATE'",
            "memory_id = ?",
            "message_id = ?",
            "json_extract(claim_json, '$.upstream_memory_id') = ?",
        ]
        params: list[Any] = [memory_id, message_id, upstream_id]

        if trigger_mode is not None:
            conditions.append("json_extract(claim_json, '$.trigger_mode') = ?")
            params.append(trigger_mode)

        if event_id is not None:
            conditions.append("json_extract(claim_json, '$.event_id') = ?")
            params.append(event_id)
        elif cause is not None:
            conditions.append("json_extract(claim_json, '$.cause') = ?")
            params.append(cause)

        where_clause = " AND ".join(conditions)
        sql = f"SELECT 1 FROM decision_log WHERE {where_clause} LIMIT 1"
        row = self.conn.execute(sql, params).fetchone()
        return row is not None

    def get_latest_acknowledgement_log(self, memory_id: int) -> DecisionLogEntry | None:
        """Retrieve the most recent R-ACKNOWLEDGE decision log entry for a memory."""
        sql = """
        SELECT * FROM decision_log
        WHERE rule_id = 'R-ACKNOWLEDGE' AND memory_id = ?
        ORDER BY id DESC LIMIT 1
        """
        row = self.conn.execute(sql, (memory_id,)).fetchone()
        return self._row_to_decision_log(row) if row else None

    def get_current_acknowledgement_log(self, memory_id: int) -> DecisionLogEntry | None:
        """Retrieve the active R-ACKNOWLEDGE decision log entry for a memory.

        Returns the log entry only if the most recent review-lifecycle decision
        (between R-PROPAGATE and R-ACKNOWLEDGE) is R-ACKNOWLEDGE.
        If the memory was subsequently re-flagged by R-PROPAGATE, returns None
        so that an old acknowledgement is never mistaken for the current state.
        """
        sql = """
        SELECT * FROM decision_log
        WHERE memory_id = ? AND rule_id IN ('R-PROPAGATE', 'R-ACKNOWLEDGE')
        ORDER BY id DESC LIMIT 1
        """
        row = self.conn.execute(sql, (memory_id,)).fetchone()
        if row and row["rule_id"] == "R-ACKNOWLEDGE":
            return self._row_to_decision_log(row)
        return None

    def _row_to_decision_log(self, r: sqlite3.Row) -> DecisionLogEntry:
        """Convert a database row into a DecisionLogEntry domain object."""
        return DecisionLogEntry(
            id=int(r["id"]),
            message_id=int(r["message_id"]),
            claim_json=json.loads(r["claim_json"]),
            action=Action(r["action"]),
            rule_id=str(r["rule_id"]),
            memory_id=int(r["memory_id"]) if r["memory_id"] is not None else None,
            affected_memory_ids=json.loads(r["affected_ids_json"]),
            scores=json.loads(r["scores_json"]),
            explanation=str(r["explanation"]),
            created_at=parse_db_datetime(r["created_at"]),  # type: ignore[arg-type]
        )

    def get_decision_logs_for_message(self, message_id: int) -> list[DecisionLogEntry]:
        """Retrieve decision log entries for a given message ID."""
        sql = "SELECT * FROM decision_log WHERE message_id = ? ORDER BY id ASC"
        cursor = self.conn.execute(sql, (message_id,))
        rows = cursor.fetchall()
        return [self._row_to_decision_log(r) for r in rows]

    def get_decision_log(self, decision_id: int) -> DecisionLogEntry | None:
        """Retrieve a single decision log entry by ID."""
        sql = "SELECT * FROM decision_log WHERE id = ?"
        row = self.conn.execute(sql, (decision_id,)).fetchone()
        return self._row_to_decision_log(row) if row else None

    def get_decision_logs_for_memory(
        self,
        memory_id: int,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[DecisionLogEntry]:
        """Retrieve decision logs associated with a memory as primary or affected target."""
        base_sql = """
        SELECT DISTINCT d.*
        FROM decision_log d
        WHERE d.memory_id = ?
           OR EXISTS (
               SELECT 1 FROM json_each(d.affected_ids_json) WHERE value = ?
           )
        ORDER BY d.created_at ASC, d.id ASC
        """
        if limit is not None:
            sql = f"{base_sql}\nLIMIT ? OFFSET ?"
            cursor = self.conn.execute(sql, (memory_id, memory_id, limit, offset))
        elif offset > 0:
            sql = f"{base_sql}\nLIMIT -1 OFFSET ?"
            cursor = self.conn.execute(sql, (memory_id, memory_id, offset))
        else:
            cursor = self.conn.execute(base_sql, (memory_id, memory_id))
        rows = cursor.fetchall()
        return [self._row_to_decision_log(r) for r in rows]

    def get_all_decision_logs(
        self,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[DecisionLogEntry]:
        """Retrieve all decision log entries ordered chronologically with stable tie-breaker."""
        if limit is not None:
            sql = "SELECT * FROM decision_log ORDER BY created_at ASC, id ASC LIMIT ? OFFSET ?"
            cursor = self.conn.execute(sql, (limit, offset))
        elif offset > 0:
            sql = "SELECT * FROM decision_log ORDER BY created_at ASC, id ASC LIMIT -1 OFFSET ?"
            cursor = self.conn.execute(sql, (offset,))
        else:
            sql = "SELECT * FROM decision_log ORDER BY created_at ASC, id ASC"
            cursor = self.conn.execute(sql)
        rows = cursor.fetchall()
        return [self._row_to_decision_log(r) for r in rows]

    def get_all_memories(self, include_evidence: bool = True) -> list[Memory]:
        """Retrieve all memories in the database ordered by id ASC."""
        sql = """
        SELECT m.*, e.canonical_name AS entity_name
        FROM memories m
        LEFT JOIN entities e ON m.entity_id = e.id
        ORDER BY m.id ASC
        """
        rows = self.conn.execute(sql).fetchall()
        return [self._row_to_memory(r, include_evidence=include_evidence) for r in rows]

    def find_temporal_window_violations(self) -> list[dict[str, Any]]:
        """Find memories violating valid_to/closed_at constraints."""
        sql = """
        SELECT id, valid_from, valid_to, closed_at
        FROM memories
        WHERE (valid_to IS NOT NULL AND closed_at IS NULL)
           OR (valid_to IS NOT NULL AND valid_to <= valid_from)
        ORDER BY id ASC
        """
        rows = self.conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def find_overlapping_active_windows(self) -> list[dict[str, Any]]:
        """Find pairs of active single-valued memories with overlapping validity windows."""
        sql = """
        SELECT
            m1.id AS id1,
            m2.id AS id2,
            m1.entity_id,
            m1.attribute,
            m1.context_key,
            m1.valid_from AS valid_from1,
            m1.valid_to AS valid_to1,
            m2.valid_from AS valid_from2,
            m2.valid_to AS valid_to2
        FROM memories m1
        JOIN memories m2 ON (
            m1.entity_id = m2.entity_id
            AND m1.attribute = m2.attribute
            AND m1.context_key = m2.context_key
            AND m1.id < m2.id
        )
        JOIN attribute_schema a ON m1.attribute = a.attribute
        WHERE a.cardinality = 'single'
          AND m1.status = 'active'
          AND m2.status = 'active'
          AND m1.valid_from < COALESCE(m2.valid_to, '9999-12-31T23:59:59Z')
          AND m2.valid_from < COALESCE(m1.valid_to, '9999-12-31T23:59:59Z')
        ORDER BY m1.id ASC, m2.id ASC
        """
        rows = self.conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def find_dangling_references(self) -> list[dict[str, Any]]:
        """Find dangling or broken references across relationships and decision logs."""
        violations: list[dict[str, Any]] = []

        # Dangling edges
        sql_edges = """
        SELECT id, src_memory_id, dst_memory_id
        FROM edges
        WHERE src_memory_id NOT IN (SELECT id FROM memories)
           OR dst_memory_id NOT IN (SELECT id FROM memories)
        """
        for r in self.conn.execute(sql_edges).fetchall():
            violations.append(
                {
                    "type": "dangling_edge",
                    "id": int(r["id"]),
                    "src_memory_id": int(r["src_memory_id"]),
                    "dst_memory_id": int(r["dst_memory_id"]),
                }
            )

        # Dangling decision_log memory_id
        sql_decisions = """
        SELECT id, memory_id
        FROM decision_log
        WHERE memory_id IS NOT NULL
          AND memory_id NOT IN (SELECT id FROM memories)
        """
        for r in self.conn.execute(sql_decisions).fetchall():
            violations.append(
                {
                    "type": "dangling_decision_memory",
                    "id": int(r["id"]),
                    "memory_id": int(r["memory_id"]),
                }
            )

        # Dangling decision_log affected_ids
        sql_affected = """
        SELECT DISTINCT d.id, j.value AS affected_id
        FROM decision_log d, json_each(d.affected_ids_json) j
        WHERE j.value NOT IN (SELECT id FROM memories)
        """
        for r in self.conn.execute(sql_affected).fetchall():
            violations.append(
                {
                    "type": "dangling_decision_affected",
                    "id": int(r["id"]),
                    "affected_id": int(r["affected_id"]),
                }
            )

        # Dangling evidence memory_id
        sql_evidence = """
        SELECT id, memory_id
        FROM evidence
        WHERE memory_id NOT IN (SELECT id FROM memories)
        """
        for r in self.conn.execute(sql_evidence).fetchall():
            violations.append(
                {
                    "type": "dangling_evidence_memory",
                    "id": int(r["id"]),
                    "memory_id": int(r["memory_id"]),
                }
            )

        return violations

    def find_schema_integrity_violations(self) -> list[dict[str, Any]]:
        """Find references to non-existent entities, attributes, or sources."""
        violations: list[dict[str, Any]] = []

        # Memories referencing non-existent entities
        sql_ent = """
        SELECT id, entity_id
        FROM memories
        WHERE entity_id NOT IN (SELECT id FROM entities)
        """
        for r in self.conn.execute(sql_ent).fetchall():
            violations.append(
                {
                    "type": "invalid_entity",
                    "id": int(r["id"]),
                    "entity_id": int(r["entity_id"]),
                }
            )

        # Memories referencing non-existent attributes
        sql_attr = """
        SELECT id, attribute
        FROM memories
        WHERE attribute NOT IN (SELECT attribute FROM attribute_schema)
        """
        for r in self.conn.execute(sql_attr).fetchall():
            violations.append(
                {
                    "type": "invalid_attribute",
                    "id": int(r["id"]),
                    "attribute": str(r["attribute"]),
                }
            )

        # Memories referencing non-existent sources
        sql_src = """
        SELECT id, source_id
        FROM memories
        WHERE source_id IS NOT NULL
          AND source_id NOT IN (SELECT id FROM sources)
        """
        for r in self.conn.execute(sql_src).fetchall():
            violations.append(
                {
                    "type": "invalid_memory_source",
                    "id": int(r["id"]),
                    "source_id": str(r["source_id"]),
                }
            )

        # Evidence referencing non-existent sources
        sql_ev_src = """
        SELECT id, memory_id, source_id
        FROM evidence
        WHERE source_id NOT IN (SELECT id FROM sources)
        """
        for r in self.conn.execute(sql_ev_src).fetchall():
            violations.append(
                {
                    "type": "invalid_evidence_source",
                    "id": int(r["id"]),
                    "memory_id": int(r["memory_id"]),
                    "source_id": str(r["source_id"]),
                }
            )

        return violations

    def get_message_integrity_records(self) -> list[dict[str, Any]]:
        """Retrieve message identification records for deduplication integrity checks."""
        sql = "SELECT id, source_id, text_hash, event_time FROM messages ORDER BY id ASC"
        cursor = self.conn.execute(sql)
        return [dict(r) for r in cursor.fetchall()]

    def get_record_counts(self) -> dict[str, int]:
        """Return row counts for primary domain tables audited by invariant checks.

        Evaluates primary transactional and audit tables:
        - memories
        - messages
        - edges
        - decision_log
        - evidence

        Excludes static catalog and reference tables (sources, attribute_schema,
        entities, entity_aliases), which serve as reference targets rather than
        audited assertions.
        """
        counts: dict[str, int] = {}
        for tbl in ["memories", "messages", "edges", "decision_log", "evidence"]:
            cursor = self.conn.execute(f"SELECT COUNT(*) AS c FROM {tbl}")
            row = cursor.fetchone()
            counts[tbl] = int(row["c"]) if row else 0
        return counts
