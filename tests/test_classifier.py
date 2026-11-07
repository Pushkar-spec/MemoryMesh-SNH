"""Tests for deterministic relation classification.

Governed by docs/SPECIFICATION.md Section 1.1, docs/ARCHITECTURE.md,
and docs/ROADMAP_AND_TASKS.md (T2.2).
Covers relation classification rules across all five ValueTypes and cardinality settings.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memorymesh.engine.classifier import classify_claim_memory, classify_relation
from memorymesh.models import (
    AttributeDef,
    Cardinality,
    Claim,
    Memory,
    MemoryStatus,
    Relation,
    ResolvedClaim,
    Source,
    ValueType,
)

# =============================================================================
# Helper Fixtures & Schemas
# =============================================================================


def make_schema(
    attribute: str,
    cardinality: Cardinality = Cardinality.SINGLE,
    value_type: ValueType = ValueType.TEXT,
) -> AttributeDef:
    """Helper creating an AttributeDef."""
    return AttributeDef(
        attribute=attribute,
        description="Test attribute",
        cardinality=cardinality,
        value_type=value_type,
    )


# =============================================================================
# VERSION Classification Tests
# =============================================================================


@pytest.mark.parametrize(
    ("claim_val", "mem_val", "expected"),
    [
        # Identical versions
        ("3.12", "3.12", Relation.IDENTICAL),
        ("3.12.0", "3.12.0", Relation.IDENTICAL),
        ("2.7.18", "2.7.18", Relation.IDENTICAL),
        # More specific version -> REFINES
        ("3.12.4", "3.12", Relation.REFINES),
        ("3.12.1", "3.12", Relation.REFINES),
        ("3.10.2", "3", Relation.REFINES),
        ("1.0.0.1", "1.0.0", Relation.REFINES),
        # Less specific version -> IDENTICAL (compatible)
        ("3.12", "3.12.4", Relation.IDENTICAL),
        ("3", "3.10.2", Relation.IDENTICAL),
        ("1.0", "1.0.5", Relation.IDENTICAL),
        # Conflicting versions (single-valued)
        ("3.12", "3.10", Relation.CONFLICTS),
        ("3.12.1", "3.12.2", Relation.CONFLICTS),
        ("2.7", "3.10", Relation.CONFLICTS),
        ("4.0", "3.12", Relation.CONFLICTS),
    ],
)
def test_version_classification(claim_val: str, mem_val: str, expected: Relation) -> None:
    """Version values classify as IDENTICAL, REFINES, or CONFLICTS under single cardinality."""
    schema = make_schema("python_version", Cardinality.SINGLE, ValueType.VERSION)
    assert classify_relation(claim_val, mem_val, schema) == expected


def test_version_classification_multi_valued_complementary() -> None:
    """Conflicting versions on multi-valued attributes classify as COMPLEMENTARY."""
    schema = make_schema("supported_python", Cardinality.MULTI, ValueType.VERSION)
    # Differing versions can coexist on multi-valued attributes
    assert classify_relation("3.12", "3.10", schema) == Relation.COMPLEMENTARY
    # Identical versions still classify as IDENTICAL
    assert classify_relation("3.12", "3.12", schema) == Relation.IDENTICAL


# =============================================================================
# ENUM Classification Tests
# =============================================================================


def test_enum_classification() -> None:
    """Enum values classify as IDENTICAL when equal and CONFLICTS when distinct (single)."""
    schema = make_schema("service_status", Cardinality.SINGLE, ValueType.ENUM)
    assert classify_relation("up", "up", schema) == Relation.IDENTICAL
    assert classify_relation("up", "down", schema) == Relation.CONFLICTS
    assert classify_relation("maintenance", "degraded", schema) == Relation.CONFLICTS


def test_enum_classification_multi_valued() -> None:
    """Distinct enum values classify as COMPLEMENTARY under multi-valued cardinality."""
    schema = make_schema("service_tags", Cardinality.MULTI, ValueType.ENUM)
    assert classify_relation("primary", "secondary", schema) == Relation.COMPLEMENTARY
    assert classify_relation("primary", "primary", schema) == Relation.IDENTICAL


# =============================================================================
# NUMBER Classification Tests
# =============================================================================


def test_number_classification() -> None:
    """Number values classify as IDENTICAL when equal and CONFLICTS when distinct (single)."""
    schema = make_schema("port_number", Cardinality.SINGLE, ValueType.NUMBER)
    assert classify_relation("8080", "8080", schema) == Relation.IDENTICAL
    assert classify_relation("8080", "8443", schema) == Relation.CONFLICTS
    assert classify_relation("10.5", "10.5", schema) == Relation.IDENTICAL
    assert classify_relation("10.5", "10.6", schema) == Relation.CONFLICTS


def test_number_classification_multi_valued() -> None:
    """Distinct number values classify as COMPLEMENTARY on multi-valued attributes."""
    schema = make_schema("open_ports", Cardinality.MULTI, ValueType.NUMBER)
    assert classify_relation("80", "443", schema) == Relation.COMPLEMENTARY
    assert classify_relation("80", "80", schema) == Relation.IDENTICAL


# =============================================================================
# DATE Classification Tests
# =============================================================================


def test_date_classification() -> None:
    """Date values classify as IDENTICAL when equal and CONFLICTS when distinct (single)."""
    schema = make_schema("release_date", Cardinality.SINGLE, ValueType.DATE)
    assert classify_relation("2026-06-01", "2026-06-01", schema) == Relation.IDENTICAL
    assert classify_relation("2026-06-01", "2026-07-01", schema) == Relation.CONFLICTS


def test_date_classification_multi_valued() -> None:
    """Distinct date values classify as COMPLEMENTARY on multi-valued attributes."""
    schema = make_schema("audit_dates", Cardinality.MULTI, ValueType.DATE)
    assert classify_relation("2026-01-01", "2026-06-01", schema) == Relation.COMPLEMENTARY
    assert classify_relation("2026-01-01", "2026-01-01", schema) == Relation.IDENTICAL


# =============================================================================
# TEXT Classification Tests
# =============================================================================


def test_text_classification() -> None:
    """Text values classify as IDENTICAL when equal and CONFLICTS when distinct (single)."""
    schema = make_schema("os_release", Cardinality.SINGLE, ValueType.TEXT)
    assert classify_relation("ubuntu 24.04", "ubuntu 24.04", schema) == Relation.IDENTICAL
    assert classify_relation("ubuntu 24.04", "debian 12", schema) == Relation.CONFLICTS


def test_text_classification_multi_valued() -> None:
    """Distinct text values classify as COMPLEMENTARY on multi-valued attributes."""
    schema = make_schema("runs_service", Cardinality.MULTI, ValueType.TEXT)
    assert classify_relation("auth-api", "web-gateway", schema) == Relation.COMPLEMENTARY
    assert classify_relation("auth-api", "auth-api", schema) == Relation.IDENTICAL


# =============================================================================
# classify_claim_memory & UNRELATED Tests
# =============================================================================


def test_classify_claim_memory_unrelated_entity_or_attribute() -> None:
    """Claims across different entities or different attributes classify as UNRELATED."""
    schema_py = make_schema("python_version", Cardinality.SINGLE, ValueType.VERSION)
    src = Source(id="src1", name="Source 1", type="admin", reliability=0.95)
    t = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

    claim_server_a = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=1.0,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t,
        source=src,
        message_id=1,
    )

    mem_server_b = Memory(
        id=10,
        entity_id=2,  # Different entity
        entity_name="Server B",
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t,
        recorded_at=t,
        status=MemoryStatus.ACTIVE,
        confidence=0.95,
        message_id=1,
        source_id="src1",
    )

    mem_different_attr = Memory(
        id=11,
        entity_id=1,
        entity_name="Server A",
        attribute="os_version",  # Different attribute
        value_norm="24.04",
        value_display="Ubuntu 24.04",
        context_key="",
        valid_from=t,
        recorded_at=t,
        status=MemoryStatus.ACTIVE,
        confidence=0.95,
        message_id=1,
        source_id="src1",
    )

    # Different entity -> UNRELATED
    assert classify_claim_memory(claim_server_a, mem_server_b, schema_py) == Relation.UNRELATED

    # Different attribute -> UNRELATED
    assert (
        classify_claim_memory(claim_server_a, mem_different_attr, schema_py) == Relation.UNRELATED
    )

    # Matching entity and attribute -> delegates to value relation
    mem_same_slot = mem_server_b.model_copy(update={"entity_id": 1, "entity_name": "Server A"})
    assert classify_claim_memory(claim_server_a, mem_same_slot, schema_py) == Relation.IDENTICAL


# =============================================================================
# Legacy Memory Normalization Regression Tests (P2.1 Review Corrections)
# =============================================================================


def test_legacy_unnormalized_memory_classification() -> None:
    """Classification normalizes claim and memory values so legacy records classify accurately."""
    schema_version = make_schema("python_version", Cardinality.SINGLE, ValueType.VERSION)
    schema_number = make_schema("port_number", Cardinality.SINGLE, ValueType.NUMBER)
    schema_enum = AttributeDef(
        attribute="service_status",
        description="Service operational state",
        cardinality=Cardinality.SINGLE,
        value_type=ValueType.ENUM,
        allowed_values=["up", "down", "maintenance"],
    )

    # 1. Existing memory value 'v3.12', incoming claim value '3.12' -> IDENTICAL
    assert classify_relation("3.12", "v3.12", schema_version) == Relation.IDENTICAL

    # 2. Existing memory value 'Python 3.12', incoming claim value 'v3.12' -> IDENTICAL
    assert classify_relation("v3.12", "Python 3.12", schema_version) == Relation.IDENTICAL

    # 3. Existing memory value '10.0', incoming claim value '10' for NUMBER -> IDENTICAL
    assert classify_relation("10", "10.0", schema_number) == Relation.IDENTICAL

    # 4. Existing memory value 'UP', incoming claim value 'up' for ENUM -> IDENTICAL
    assert classify_relation("up", "UP", schema_enum) == Relation.IDENTICAL


# =============================================================================
# Determinism Tests
# =============================================================================


def test_classification_determinism() -> None:
    """Repeated calls with identical arguments always return the exact same Relation output."""
    schema = make_schema("python_version", Cardinality.SINGLE, ValueType.VERSION)
    for _ in range(50):
        assert classify_relation("3.12.4", "3.12", schema) == Relation.REFINES
        assert classify_relation("3.12", "3.12.4", schema) == Relation.IDENTICAL
        assert classify_relation("3.10", "3.12", schema) == Relation.CONFLICTS
        assert classify_relation("3.12", "3.12", schema) == Relation.IDENTICAL
