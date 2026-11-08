"""SQLite Repository layer for MemoryMesh storage operations.

Governed by docs/SPECIFICATION.md Section 3, docs/AGENTS.md, and docs/ROADMAP_AND_TASKS.md (T1.1).
All SQL queries live exclusively within this module.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Generator
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
        params: list[Any] = [entity_id, attribute, context_key]
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
            params.append(context_key)

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
        return [
            Edge(
                id=int(r["id"]),
                src_memory_id=int(r["src_memory_id"]),
                dst_memory_id=int(r["dst_memory_id"]),
                type=EdgeType(r["type"]),
                reason=str(r["reason"] or ""),
                created_at=parse_db_datetime(r["created_at"]),  # type: ignore[arg-type]
            )
            for r in rows
        ]

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

    def get_decision_logs_for_message(self, message_id: int) -> list[DecisionLogEntry]:
        """Retrieve decision log entries for a given message ID."""
        sql = "SELECT * FROM decision_log WHERE message_id = ? ORDER BY id ASC"
        cursor = self.conn.execute(sql, (message_id,))
        rows = cursor.fetchall()
        return [
            DecisionLogEntry(
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
            for r in rows
        ]
