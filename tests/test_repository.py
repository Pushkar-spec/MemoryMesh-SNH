"""Tests for the SQLite Repository layer and transaction management.

Governed by docs/SPECIFICATION.md Section 3, docs/AGENTS.md, and docs/ROADMAP_AND_TASKS.md (T1.1).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from memorymesh.db import init_db
from memorymesh.errors import (
    ForeignKeyConstraintError,
    InvalidUpdateFieldError,
    MemoryNotFoundError,
)
from memorymesh.models import (
    AttributeDef,
    Cardinality,
    EdgeType,
    MemoryStatus,
    NewEvidence,
    NewMemory,
    Source,
    ValueType,
)
from memorymesh.repository import Repository


@pytest.fixture
def repo() -> Repository:
    """Provide a fresh in-memory Repository with initialized schema and seed records."""
    conn = init_db(":memory:")
    repository = Repository(conn)

    # Seed required foreign key records without writing SQL in tests
    repository.insert_source(
        Source(
            id="sysadmin_priya",
            name="Priya Sharma",
            type="admin",
            reliability=0.95,
            notes="Senior Infrastructure Engineer",
        )
    )
    repository.insert_source(
        Source(
            id="dev_chat",
            name="Developer Slack",
            type="chat",
            reliability=0.60,
            notes="General channel",
        )
    )
    repository.insert_attribute_schema(
        AttributeDef(
            attribute="python_version",
            description="Installed Python runtime version",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
        )
    )
    repository.insert_entity(canonical_name="server_a", entity_type="server")
    repository.insert_message(
        source_id="sysadmin_priya",
        text="Server A runs Python 3.10",
        event_time=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
    )
    return repository


def test_memory_insert_and_get_by_id(repo: Repository) -> None:
    """Verify inserting a memory persists it and allows retrieval by ID."""
    new_mem = NewMemory(
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context={"env": "prod"},
        context_key="env=prod",
        valid_from=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
        valid_to=None,
        closed_at=None,
        status=MemoryStatus.ACTIVE,
        confidence=0.95,
        raw_text="Server A runs Python 3.10",
        message_id=1,
        source_id="sysadmin_priya",
    )
    recorded_at = datetime(2026, 1, 15, 10, 5, tzinfo=UTC)
    mem_id = repo.insert_memory(new_mem, recorded_at=recorded_at)
    assert mem_id == 1

    retrieved = repo.get_memory(mem_id)
    assert retrieved is not None
    assert retrieved.id == 1
    assert retrieved.entity_id == 1
    assert retrieved.entity_name == "server_a"
    assert retrieved.attribute == "python_version"
    assert retrieved.value_norm == "3.10"
    assert retrieved.value_display == "Python 3.10"
    assert retrieved.context == {"env": "prod"}
    assert retrieved.context_key == "env=prod"
    assert retrieved.valid_from == datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    assert retrieved.valid_to is None
    assert retrieved.recorded_at == recorded_at
    assert retrieved.closed_at is None
    assert retrieved.status == MemoryStatus.ACTIVE
    assert retrieved.confidence == 0.95
    assert retrieved.needs_review is False
    assert retrieved.message_id == 1
    assert retrieved.source_id == "sysadmin_priya"


def test_get_memory_nonexistent_returns_none(repo: Repository) -> None:
    """Verify get_memory returns None when ID does not exist."""
    assert repo.get_memory(9999) is None


def test_memory_field_updates_typed_methods(repo: Repository) -> None:
    """Verify updating memory fields using typed methods."""
    new_mem = NewMemory(
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
        status=MemoryStatus.ACTIVE,
        confidence=0.95,
        message_id=1,
        source_id="sysadmin_priya",
    )
    mem_id = repo.insert_memory(new_mem, recorded_at=datetime(2026, 1, 15, 10, 0, tzinfo=UTC))

    # Close memory window and transition to HISTORICAL
    valid_to = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    closed_at = datetime(2026, 6, 1, 12, 5, tzinfo=UTC)
    repo.close_memory(mem_id, valid_to=valid_to, closed_at=closed_at)

    updated = repo.get_memory(mem_id)
    assert updated is not None
    assert updated.status == MemoryStatus.HISTORICAL
    assert updated.valid_to == valid_to
    assert updated.closed_at == closed_at

    # Flag for review
    repo.flag_memory_review(mem_id, reason="Dependency updated")
    flagged = repo.get_memory(mem_id)
    assert flagged is not None
    assert flagged.needs_review is True
    assert flagged.review_reason == "Dependency updated"


def test_memory_field_update_dict_allowlist_validation(repo: Repository) -> None:
    """Verify update_memory validates field names against an explicit allowlist."""
    new_mem = NewMemory(
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
        status=MemoryStatus.ACTIVE,
        confidence=0.95,
        message_id=1,
        source_id="sysadmin_priya",
    )
    mem_id = repo.insert_memory(new_mem, recorded_at=datetime(2026, 1, 15, 10, 0, tzinfo=UTC))

    # Valid update
    repo.update_memory(mem_id, {"confidence": 0.99})
    assert repo.get_memory(mem_id).confidence == 0.99

    # Disallowed field construction rejected
    with pytest.raises(InvalidUpdateFieldError, match="not allowed for memory updates"):
        repo.update_memory(mem_id, {"arbitrary_injected_column": "value"})

    # Non-existent memory update raises MemoryNotFoundError
    with pytest.raises(MemoryNotFoundError):
        repo.update_memory(8888, {"status": MemoryStatus.DISPUTED})


def test_evidence_insert_and_retrieval(repo: Repository) -> None:
    """Verify inserting evidence and retrieving evidence associated with a memory."""
    new_mem = NewMemory(
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
        status=MemoryStatus.ACTIVE,
        confidence=0.95,
        message_id=1,
        source_id="sysadmin_priya",
    )
    mem_id = repo.insert_memory(new_mem, recorded_at=datetime(2026, 1, 15, 10, 0, tzinfo=UTC))

    ev1 = NewEvidence(
        source_id="sysadmin_priya",
        message_id=1,
        strength=0.95,
        observed_at=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
        span="Python 3.10",
    )
    ev_id1 = repo.insert_evidence(mem_id, ev1)
    assert ev_id1 == 1

    ev2 = NewEvidence(
        source_id="dev_chat",
        message_id=1,
        strength=0.60,
        observed_at=datetime(2026, 1, 16, 9, 0, tzinfo=UTC),
        span="runs 3.10",
    )
    ev_id2 = repo.insert_evidence(mem_id, ev2)
    assert ev_id2 == 2

    evidence_list = repo.get_evidence_for_memory(mem_id)
    assert len(evidence_list) == 2
    assert evidence_list[0].id == ev_id1
    assert evidence_list[0].source_id == "sysadmin_priya"
    assert evidence_list[0].strength == 0.95
    assert evidence_list[1].id == ev_id2
    assert evidence_list[1].source_id == "dev_chat"
    assert evidence_list[1].strength == 0.60

    # Verify get_memory also attaches evidence
    mem = repo.get_memory(mem_id)
    assert mem is not None
    assert len(mem.evidence) == 2


def test_edge_insert_and_retrieval(repo: Repository) -> None:
    """Verify inserting relationship edges between memories and retrieving them."""
    mem1_id = repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.10",
            value_display="Python 3.10",
            context_key="",
            valid_from=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
            status=MemoryStatus.HISTORICAL,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
    )
    mem2_id = repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.12",
            value_display="Python 3.12",
            context_key="",
            valid_from=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=datetime(2026, 6, 1, 12, 5, tzinfo=UTC),
    )

    edge_id = repo.insert_edge(
        src_memory_id=mem2_id,
        dst_memory_id=mem1_id,
        edge_type=EdgeType.SUPERSEDES,
        reason="Server migrated from 3.10 to 3.12",
        created_at=datetime(2026, 6, 1, 12, 5, tzinfo=UTC),
    )
    assert edge_id == 1

    edges_out = repo.get_edges_for_memory(mem2_id, direction="src")
    assert len(edges_out) == 1
    assert edges_out[0].id == edge_id
    assert edges_out[0].src_memory_id == mem2_id
    assert edges_out[0].dst_memory_id == mem1_id
    assert edges_out[0].type == EdgeType.SUPERSEDES
    assert edges_out[0].reason == "Server migrated from 3.10 to 3.12"

    edges_in = repo.get_edges_for_memory(mem1_id, direction="dst")
    assert len(edges_in) == 1
    assert edges_in[0].src_memory_id == mem2_id


def test_transaction_commit_persists_changes(repo: Repository) -> None:
    """Verify that operations within a transaction commit successfully."""
    with repo.transaction():
        mem_id = repo.insert_memory(
            NewMemory(
                entity_id=1,
                attribute="python_version",
                value_norm="3.10",
                value_display="Python 3.10",
                context_key="",
                valid_from=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
                status=MemoryStatus.ACTIVE,
                confidence=0.95,
                message_id=1,
                source_id="sysadmin_priya",
            ),
            recorded_at=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
        )

    # Memory must exist after exiting the context manager
    assert repo.get_memory(mem_id) is not None


def test_transaction_rollback_on_exception(repo: Repository) -> None:
    """Verify that an exception inside a transaction rolls back uncommitted changes."""
    with pytest.raises(RuntimeError, match="simulated failure"):
        with repo.transaction():
            repo.insert_memory(
                NewMemory(
                    entity_id=1,
                    attribute="python_version",
                    value_norm="3.10",
                    value_display="Python 3.10",
                    context_key="",
                    valid_from=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
                    status=MemoryStatus.ACTIVE,
                    confidence=0.95,
                    message_id=1,
                    source_id="sysadmin_priya",
                ),
                recorded_at=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
            )
            raise RuntimeError("simulated failure")

    # Inserted memory must NOT exist
    assert repo.get_memory(1) is None


def test_database_remains_usable_after_rollback(repo: Repository) -> None:
    """Verify the database connection remains completely usable after an abort/rollback."""
    try:
        with repo.transaction():
            repo.insert_memory(
                NewMemory(
                    entity_id=1,
                    attribute="python_version",
                    value_norm="3.10",
                    value_display="Python 3.10",
                    context_key="",
                    valid_from=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
                    status=MemoryStatus.ACTIVE,
                    confidence=0.95,
                    message_id=1,
                    source_id="sysadmin_priya",
                ),
                recorded_at=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
            )
            raise ValueError("aborted operation")
    except ValueError:
        pass

    # Connection should execute subsequent transactions without errors
    with repo.transaction():
        valid_id = repo.insert_memory(
            NewMemory(
                entity_id=1,
                attribute="python_version",
                value_norm="3.12",
                value_display="Python 3.12",
                context_key="",
                valid_from=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
                status=MemoryStatus.ACTIVE,
                confidence=0.95,
                message_id=1,
                source_id="sysadmin_priya",
            ),
            recorded_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
        )

    saved = repo.get_memory(valid_id)
    assert saved is not None
    assert saved.value_norm == "3.12"


def test_foreign_key_enforcement(repo: Repository) -> None:
    """Verify foreign key enforcement rejects references to non-existent entities or attributes."""
    # Attempt inserting a memory with an invalid entity_id
    invalid_mem = NewMemory(
        entity_id=9999,  # Does not exist
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
        status=MemoryStatus.ACTIVE,
        confidence=0.95,
        message_id=1,
        source_id="sysadmin_priya",
    )
    with pytest.raises((ForeignKeyConstraintError, sqlite3.IntegrityError)):
        repo.insert_memory(invalid_mem, recorded_at=datetime(2026, 1, 15, 10, 0, tzinfo=UTC))


def test_get_memory_chain(repo: Repository) -> None:
    """Verify get_memory_chain retrieves all non-rejected memories sorted by valid_from."""
    t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.10",
            value_display="Python 3.10",
            context_key="",
            valid_from=t1,
            valid_to=t2,
            closed_at=t2,
            status=MemoryStatus.HISTORICAL,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t1,
    )

    repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.12",
            value_display="Python 3.12",
            context_key="",
            valid_from=t2,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t2,
    )

    chain = repo.get_memory_chain(entity_id=1, attribute="python_version")
    assert len(chain) == 2
    assert chain[0].value_norm == "3.10"
    assert chain[1].value_norm == "3.12"
    assert chain[0].valid_from < chain[1].valid_from


def test_insert_source_upsert_preserves_dependents(repo: Repository) -> None:
    """Verify that insert_source UPSERT does not recreate rows and preserves related records."""
    t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)

    # 1. Insert initial memory and evidence referencing sysadmin_priya
    mem_id = repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.10",
            value_display="Python 3.10",
            context_key="",
            valid_from=t,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t,
    )
    ev_id = repo.insert_evidence(
        mem_id,
        NewEvidence(
            source_id="sysadmin_priya",
            message_id=1,
            strength=0.95,
            observed_at=t,
            span="Python 3.10",
        ),
    )

    # 2. Call insert_source again with updated information on the same source id
    repo.insert_source(
        Source(
            id="sysadmin_priya",
            name="Priya Sharma (Staff Engineer)",
            type="admin",
            reliability=0.99,
            notes="Promoted and updated notes",
        )
    )

    # 3. Verify existing memory and evidence records remain intact
    mem = repo.get_memory(mem_id)
    assert mem is not None
    assert mem.source_id == "sysadmin_priya"
    assert mem.id == mem_id

    ev_list = repo.get_evidence_for_memory(mem_id)
    assert len(ev_list) == 1
    assert ev_list[0].id == ev_id
    assert ev_list[0].source_id == "sysadmin_priya"

    # 4. Verify source fields were updated in place
    row = repo.conn.execute("SELECT * FROM sources WHERE id = 'sysadmin_priya'").fetchone()
    assert row is not None
    assert row["name"] == "Priya Sharma (Staff Engineer)"
    assert row["reliability"] == 0.99
    assert row["notes"] == "Promoted and updated notes"


def test_insert_attribute_schema_upsert_preserves_dependents(repo: Repository) -> None:
    """Verify that insert_attribute_schema UPSERT updates in place and preserves memories."""
    t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)

    mem_id = repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.10",
            value_display="Python 3.10",
            context_key="",
            valid_from=t,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t,
    )

    # Update attribute schema in place
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="python_version",
            description="Active Python version (updated description)",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
            half_life_days=365.0,
            allowed_values=["3.10", "3.11", "3.12"],
        )
    )

    # Dependent memory must remain intact
    mem = repo.get_memory(mem_id)
    assert mem is not None
    assert mem.attribute == "python_version"

    # Schema row must reflect updated fields
    row = repo.conn.execute(
        "SELECT * FROM attribute_schema WHERE attribute = 'python_version'"
    ).fetchone()
    assert row is not None
    assert row["description"] == "Active Python version (updated description)"
    assert row["half_life_days"] == 365.0


def test_nested_transactions(repo: Repository) -> None:
    """Verify nested transaction behaviors: commit, inner rollback, and outer rollback."""
    t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)

    # A. Successful nested transaction commits both operations
    with repo.transaction():
        m1 = repo.insert_memory(
            NewMemory(
                entity_id=1,
                attribute="python_version",
                value_norm="3.10",
                value_display="Python 3.10",
                context_key="",
                valid_from=t,
                status=MemoryStatus.ACTIVE,
                confidence=0.95,
                message_id=1,
                source_id="sysadmin_priya",
            ),
            recorded_at=t,
        )
        with repo.transaction():
            m2 = repo.insert_memory(
                NewMemory(
                    entity_id=1,
                    attribute="python_version",
                    value_norm="3.11",
                    value_display="Python 3.11",
                    context_key="",
                    valid_from=t,
                    status=MemoryStatus.ACTIVE,
                    confidence=0.95,
                    message_id=1,
                    source_id="sysadmin_priya",
                ),
                recorded_at=t,
            )

    assert repo.get_memory(m1) is not None
    assert repo.get_memory(m2) is not None

    # B. Inner transaction failure and rollback while outer transaction continues
    with repo.transaction():
        m3 = repo.insert_memory(
            NewMemory(
                entity_id=1,
                attribute="python_version",
                value_norm="3.12",
                value_display="Python 3.12",
                context_key="",
                valid_from=t,
                status=MemoryStatus.ACTIVE,
                confidence=0.95,
                message_id=1,
                source_id="sysadmin_priya",
            ),
            recorded_at=t,
        )
        try:
            with repo.transaction():
                repo.insert_memory(
                    NewMemory(
                        entity_id=1,
                        attribute="python_version",
                        value_norm="3.13-failed",
                        value_display="Python 3.13",
                        context_key="",
                        valid_from=t,
                        status=MemoryStatus.ACTIVE,
                        confidence=0.95,
                        message_id=1,
                        source_id="sysadmin_priya",
                    ),
                    recorded_at=t,
                )
                raise RuntimeError("inner abort")
        except RuntimeError:
            pass

        m4 = repo.insert_memory(
            NewMemory(
                entity_id=1,
                attribute="python_version",
                value_norm="3.14",
                value_display="Python 3.14",
                context_key="",
                valid_from=t,
                status=MemoryStatus.ACTIVE,
                confidence=0.95,
                message_id=1,
                source_id="sysadmin_priya",
            ),
            recorded_at=t,
        )

    assert repo.get_memory(m3) is not None
    assert repo.get_memory(m4) is not None
    # Failed inner memory must not exist
    chain_values = [m.value_norm for m in repo.get_memory_chain(1, "python_version")]
    assert "3.13-failed" not in chain_values

    # C. Successful inner transaction followed by outer rollback
    try:
        with repo.transaction():
            m_outer = repo.insert_memory(
                NewMemory(
                    entity_id=1,
                    attribute="python_version",
                    value_norm="3.15-outer",
                    value_display="Python 3.15",
                    context_key="",
                    valid_from=t,
                    status=MemoryStatus.ACTIVE,
                    confidence=0.95,
                    message_id=1,
                    source_id="sysadmin_priya",
                ),
                recorded_at=t,
            )
            with repo.transaction():
                m_inner = repo.insert_memory(
                    NewMemory(
                        entity_id=1,
                        attribute="python_version",
                        value_norm="3.16-inner",
                        value_display="Python 3.16",
                        context_key="",
                        valid_from=t,
                        status=MemoryStatus.ACTIVE,
                        confidence=0.95,
                        message_id=1,
                        source_id="sysadmin_priya",
                    ),
                    recorded_at=t,
                )
            raise ValueError("outer failure after inner success")
    except ValueError:
        pass

    assert repo.get_memory(m_outer) is None
    assert repo.get_memory(m_inner) is None

    # D. Connection remains usable after all nested transaction tests
    with repo.transaction():
        m_final = repo.insert_memory(
            NewMemory(
                entity_id=1,
                attribute="python_version",
                value_norm="3.17-final",
                value_display="Python 3.17",
                context_key="",
                valid_from=t,
                status=MemoryStatus.ACTIVE,
                confidence=0.95,
                message_id=1,
                source_id="sysadmin_priya",
            ),
            recorded_at=t,
        )
    assert repo.get_memory(m_final) is not None


def test_update_memory_value_validations(repo: Repository) -> None:
    """Verify that update_memory validates value types and ranges for all allowed fields."""
    t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    mem_id = repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.10",
            value_display="Python 3.10",
            context_key="",
            valid_from=t,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t,
    )

    # 1. Status validation
    with pytest.raises(InvalidUpdateFieldError, match="Invalid status"):
        repo.update_memory(mem_id, {"status": "unsupported_status"})

    # 2. Confidence validation
    with pytest.raises(InvalidUpdateFieldError, match="between 0.0 and 1.0"):
        repo.update_memory(mem_id, {"confidence": 1.5})
    with pytest.raises(InvalidUpdateFieldError, match="between 0.0 and 1.0"):
        repo.update_memory(mem_id, {"confidence": -0.1})
    with pytest.raises(InvalidUpdateFieldError, match="must be numeric"):
        repo.update_memory(mem_id, {"confidence": "high"})

    # 3. Datetime validations
    with pytest.raises(InvalidUpdateFieldError, match="must be a datetime"):
        repo.update_memory(mem_id, {"valid_to": "not-a-datetime"})
    with pytest.raises(InvalidUpdateFieldError, match="must be a datetime"):
        repo.update_memory(mem_id, {"closed_at": 123456})

    # 4. Inconsistent valid_to / closed_at update
    with pytest.raises(InvalidUpdateFieldError, match="closed_at is NULL"):
        repo.update_memory(mem_id, {"valid_to": t, "closed_at": None})

    # 5. needs_review boolean validation
    with pytest.raises(InvalidUpdateFieldError, match="must be a boolean"):
        repo.update_memory(mem_id, {"needs_review": "yes"})

    # 6. review_reason string validation
    with pytest.raises(InvalidUpdateFieldError, match="must be a string"):
        repo.update_memory(mem_id, {"review_reason": 42})


def test_update_memory_resulting_validity_window_consistency(repo: Repository) -> None:
    """Verify that update_memory validates the resulting state of valid_to and closed_at."""
    t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 3, 15, 10, 0, tzinfo=UTC)
    t3 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    # Initially open-ended memory: valid_to is None, closed_at is None
    mem_id = repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.10",
            value_display="Python 3.10",
            context_key="",
            valid_from=t1,
            valid_to=None,
            closed_at=None,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t1,
    )

    # 1. Setting valid_to without closed_at when existing closed_at is NULL must fail
    with pytest.raises(InvalidUpdateFieldError, match="closed_at is NULL"):
        repo.update_memory(mem_id, {"valid_to": t2})

    # Verify original record left completely unchanged
    mem = repo.get_memory(mem_id)
    assert mem is not None
    assert mem.valid_to is None
    assert mem.closed_at is None

    # 2. Updating both fields to valid non-NULL values must pass
    repo.update_memory(mem_id, {"valid_to": t2, "closed_at": t2})
    mem = repo.get_memory(mem_id)
    assert mem is not None
    assert mem.valid_to == t2
    assert mem.closed_at == t2

    # 3. Setting valid_to without closed_at when existing closed_at is already populated passes
    repo.update_memory(mem_id, {"valid_to": t3})
    mem = repo.get_memory(mem_id)
    assert mem is not None
    assert mem.valid_to == t3
    assert mem.closed_at == t2

    # 4. Setting closed_at=None when existing valid_to is non-NULL must fail
    with pytest.raises(InvalidUpdateFieldError, match="closed_at is NULL"):
        repo.update_memory(mem_id, {"closed_at": None})

    # Verify original record left completely unchanged
    mem = repo.get_memory(mem_id)
    assert mem is not None
    assert mem.valid_to == t3
    assert mem.closed_at == t2

    # 5. Updating both fields to NULL must pass
    repo.update_memory(mem_id, {"valid_to": None, "closed_at": None})
    mem = repo.get_memory(mem_id)
    assert mem is not None
    assert mem.valid_to is None
    assert mem.closed_at is None
