"""Tests for temporal memory retrieval and validity boundaries.

Governed by docs/SPECIFICATION.md Section 4.1 and docs/ROADMAP_AND_TASKS.md (T1.3).
Rule: valid_from <= requested_time < valid_to (half-open validity interval).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memorymesh.db import init_db
from memorymesh.models import (
    AttributeDef,
    Cardinality,
    MemoryStatus,
    NewMemory,
    Source,
    ValueType,
)
from memorymesh.repository import Repository
from memorymesh.retrieval.retrieve import (
    retrieve_as_of,
    retrieve_current,
    retrieve_valid_memories,
)


@pytest.fixture
def repo() -> Repository:
    """Provide a seeded in-memory Repository with entity, source, and attribute."""
    conn = init_db(":memory:")
    repository = Repository(conn)

    repository.insert_source(
        Source(
            id="sysadmin_priya",
            name="Priya Sharma",
            type="admin",
            reliability=0.95,
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
    repository.insert_entity(canonical_name="Server A", entity_type="server")
    repository.insert_message(
        source_id="sysadmin_priya",
        text="Setup message",
        event_time=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
    )
    return repository


def test_retrieve_active_memory(repo: Repository) -> None:
    """1. Retrieving an active memory returns the open-ended record."""
    t_start = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t_now = datetime(2026, 6, 25, 9, 0, tzinfo=UTC)

    mem_id = repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.12",
            value_display="Python 3.12",
            context_key="",
            valid_from=t_start,
            valid_to=None,
            closed_at=None,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t_start,
    )

    mem = retrieve_current(repo, entity_id=1, attribute="python_version", now=t_now)
    assert mem is not None
    assert mem.id == mem_id
    assert mem.value_norm == "3.12"
    assert mem.status == MemoryStatus.ACTIVE


def test_retrieve_historical_memory_as_of(repo: Repository) -> None:
    """2. Retrieving an older historical memory using an as-of timestamp."""
    t_start = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t_migration = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    t_query_march = datetime(2026, 3, 15, 10, 0, tzinfo=UTC)

    # Historical memory valid from Jan 15 to June 1
    m1_id = repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.10",
            value_display="Python 3.10",
            context_key="",
            valid_from=t_start,
            valid_to=t_migration,
            closed_at=t_migration,
            status=MemoryStatus.HISTORICAL,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t_start,
    )

    # Active memory valid from June 1 onwards
    repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.12",
            value_display="Python 3.12",
            context_key="",
            valid_from=t_migration,
            valid_to=None,
            closed_at=None,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t_migration,
    )

    # Query as of March 15 should return the historical Python 3.10 memory
    mem_march = retrieve_as_of(
        repo, entity_id=1, attribute="python_version", valid_at=t_query_march
    )
    assert mem_march is not None
    assert mem_march.id == m1_id
    assert mem_march.value_norm == "3.10"
    assert mem_march.status == MemoryStatus.HISTORICAL


def test_retrieve_newer_memory_at_current_time(repo: Repository) -> None:
    """3. Retrieving the newer memory using a current timestamp."""
    t_start = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t_migration = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    t_now = datetime(2026, 6, 25, 9, 0, tzinfo=UTC)

    repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.10",
            value_display="Python 3.10",
            context_key="",
            valid_from=t_start,
            valid_to=t_migration,
            closed_at=t_migration,
            status=MemoryStatus.HISTORICAL,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t_start,
    )

    m2_id = repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.12",
            value_display="Python 3.12",
            context_key="",
            valid_from=t_migration,
            valid_to=None,
            closed_at=None,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t_migration,
    )

    mem_current = retrieve_current(repo, entity_id=1, attribute="python_version", now=t_now)
    assert mem_current is not None
    assert mem_current.id == m2_id
    assert mem_current.value_norm == "3.12"
    assert mem_current.status == MemoryStatus.ACTIVE


def test_exact_validity_boundary_returns_new_memory(repo: Repository) -> None:
    """4. Exact validity boundary: requested_time == valid_to returns new memory, not old."""
    t_start = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t_migration = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    m1_id = repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.10",
            value_display="Python 3.10",
            context_key="",
            valid_from=t_start,
            valid_to=t_migration,  # Closed at t_migration
            closed_at=t_migration,
            status=MemoryStatus.HISTORICAL,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t_start,
    )

    m2_id = repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.12",
            value_display="Python 3.12",
            context_key="",
            valid_from=t_migration,  # Starts at t_migration
            valid_to=None,
            closed_at=None,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t_migration,
    )

    # Query exactly at t_migration
    mem_boundary = retrieve_as_of(
        repo, entity_id=1, attribute="python_version", valid_at=t_migration
    )
    assert mem_boundary is not None
    # Must be new memory M2, NOT old memory M1
    assert mem_boundary.id == m2_id
    assert mem_boundary.id != m1_id
    assert mem_boundary.value_norm == "3.12"


def test_rejected_memories_excluded(repo: Repository) -> None:
    """5. Rejected memories must not be returned by temporal retrieval."""
    t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)

    repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.9",
            value_display="Python 3.9",
            context_key="",
            valid_from=t,
            valid_to=None,
            closed_at=None,
            status=MemoryStatus.REJECTED,
            confidence=0.30,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t,
    )

    mem = retrieve_current(repo, entity_id=1, attribute="python_version", now=t)
    assert mem is None

    memories = retrieve_valid_memories(
        repo, entity_id=1, attribute="python_version", valid_at=t, include_rejected=False
    )
    assert len(memories) == 0


def test_context_filtering(repo: Repository) -> None:
    """6. Context filtering separates global from environment-specific memories."""
    t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)

    # Global memory: context_key = ""
    repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.12",
            value_display="Python 3.12 (Global)",
            context={},
            context_key="",
            valid_from=t,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t,
    )

    # Staging memory: context_key = "environment=staging"
    repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.11",
            value_display="Python 3.11 (Staging)",
            context={"environment": "staging"},
            context_key="environment=staging",
            valid_from=t,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t,
    )

    # Query with staging context
    staging_mem = retrieve_current(
        repo,
        entity_id=1,
        attribute="python_version",
        now=t,
        context={"environment": "staging"},
    )
    assert staging_mem is not None
    assert staging_mem.value_norm == "3.11"
    assert staging_mem.context_key == "environment=staging"

    # Query without context (global)
    global_mem = retrieve_current(repo, entity_id=1, attribute="python_version", now=t)
    assert global_mem is not None
    assert global_mem.value_norm == "3.12"
    assert global_mem.context_key == ""


def test_no_matching_memory_returns_none(repo: Repository) -> None:
    """7. No matching memory returns None / empty list."""
    t_query = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    t_memory = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)

    repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.10",
            value_display="Python 3.10",
            context_key="",
            valid_from=t_memory,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t_memory,
    )

    # Querying before memory validity start returns None
    mem = retrieve_as_of(repo, entity_id=1, attribute="python_version", valid_at=t_query)
    assert mem is None

    empty_list = retrieve_valid_memories(
        repo, entity_id=1, attribute="python_version", valid_at=t_query
    )
    assert empty_list == []


def test_bitemporal_late_arriving_information(repo: Repository) -> None:
    """8. Bitemporal retrieval handles late-arriving information.

    - Python 3.10 valid and recorded in Jan.
    - In July, learned Python 3.12 installed June 1 (M1 closed valid_to=June, closed_at=July).
    - Querying June with known_at in June returns Python 3.10 (closure was unknown in June).
    - Querying June with known_at in July returns Python 3.12 (closure now known).
    - Querying with known_at before recorded_at excludes the memory entirely.
    """
    t_jan = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t_june = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    t_july = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)

    t_query_june = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
    t_belief_june = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    t_belief_july = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    t_pre_jan = datetime(2025, 12, 1, 0, 0, tzinfo=UTC)

    # 1. Memory 1: Python 3.10 recorded in Jan, later closed in July effective June 1
    m1_id = repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.10",
            value_display="Python 3.10",
            context_key="",
            valid_from=t_jan,
            valid_to=t_june,
            closed_at=t_july,
            status=MemoryStatus.HISTORICAL,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t_jan,
    )

    # 2. Memory 2: Python 3.12 recorded in July, valid starting June 1
    m2_id = repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.12",
            value_display="Python 3.12",
            context_key="",
            valid_from=t_june,
            valid_to=None,
            closed_at=None,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t_july,
    )

    # 3. Query June 15 with belief time in June (before July ingestion)
    # Closure of M1 happened in July, so in June we believed M1 was still valid
    mem_june_belief = retrieve_as_of(
        repo,
        entity_id=1,
        attribute="python_version",
        valid_at=t_query_june,
        known_at=t_belief_june,
    )
    assert mem_june_belief is not None
    assert mem_june_belief.id == m1_id
    assert mem_june_belief.value_norm == "3.10"

    # 4. Query June 15 with belief time in July (after July ingestion)
    # M1 closure is now known; M2 is known and active for June 15
    mem_july_belief = retrieve_as_of(
        repo,
        entity_id=1,
        attribute="python_version",
        valid_at=t_query_june,
        known_at=t_belief_july,
    )
    assert mem_july_belief is not None
    assert mem_july_belief.id == m2_id
    assert mem_july_belief.value_norm == "3.12"

    # 5. Query with belief time prior to recorded_at excludes the memory
    mem_pre_record = retrieve_as_of(
        repo,
        entity_id=1,
        attribute="python_version",
        valid_at=t_query_june,
        known_at=t_pre_jan,
    )
    assert mem_pre_record is None


def test_retrieve_as_of_defaults_known_at_to_valid_at(repo: Repository) -> None:
    """retrieve_as_of() defaults known_at to valid_at when omitted.

    - M1 (Python 3.10) valid Jan 15, recorded Jan 15, closed in July
      (valid_to=June 1, closed_at=July 1).
    - M2 (Python 3.12) valid June 1, recorded July 1.
    - retrieve_as_of(valid_at=June 15, known_at=None) defaults known_at to June 15:
      M2 (recorded July 1) is excluded; M1 was still open -> returns M1 (Python 3.10).
    - retrieve_as_of(valid_at=June 15, known_at=July 15) preserves known_at=July 15:
      M2 (recorded July 1) is visible -> returns M2 (Python 3.12).
    """
    t_jan = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t_june = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    t_july = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)

    t_query_june = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
    t_explicit_july = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)

    # 1. M1: Python 3.10 recorded in Jan, closed in July effective June 1
    m1_id = repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.10",
            value_display="Python 3.10",
            context_key="",
            valid_from=t_jan,
            valid_to=t_june,
            closed_at=t_july,
            status=MemoryStatus.HISTORICAL,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t_jan,
    )

    # 2. M2: Python 3.12 recorded in July, valid starting June 1
    m2_id = repo.insert_memory(
        NewMemory(
            entity_id=1,
            attribute="python_version",
            value_norm="3.12",
            value_display="Python 3.12",
            context_key="",
            valid_from=t_june,
            valid_to=None,
            closed_at=None,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            message_id=1,
            source_id="sysadmin_priya",
        ),
        recorded_at=t_july,
    )

    # 3. Omitting known_at defaults knowledge time to valid_at (June 15)
    # Late-arriving memory M2 (recorded July 1) is excluded; M1 closure was not yet recorded in June
    mem_default_known = retrieve_as_of(
        repo,
        entity_id=1,
        attribute="python_version",
        valid_at=t_query_june,
    )
    assert mem_default_known is not None
    assert mem_default_known.id == m1_id
    assert mem_default_known.value_norm == "3.10"

    # 4. Explicit known_at=July 15 is preserved
    # Late-arriving memory M2 is now known and valid for June 15
    mem_explicit_known = retrieve_as_of(
        repo,
        entity_id=1,
        attribute="python_version",
        valid_at=t_query_june,
        known_at=t_explicit_july,
    )
    assert mem_explicit_known is not None
    assert mem_explicit_known.id == m2_id
    assert mem_explicit_known.value_norm == "3.12"
