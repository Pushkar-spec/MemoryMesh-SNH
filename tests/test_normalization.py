"""Tests for deterministic value normalization.

Governed by docs/SPECIFICATION.md Section 6, docs/ARCHITECTURE.md,
and docs/ROADMAP_AND_TASKS.md (T2.1).
Covers all five ValueTypes: VERSION, ENUM, TEXT, NUMBER, DATE.
"""

from __future__ import annotations

import pytest

from memorymesh.engine.normalize import (
    normalize_date,
    normalize_enum,
    normalize_number,
    normalize_text,
    normalize_value,
    normalize_version,
)
from memorymesh.errors import NormalizationError
from memorymesh.models import ValueType

# =============================================================================
# VERSION Normalization Tests
# =============================================================================


@pytest.mark.parametrize(
    ("raw_input", "expected"),
    [
        ("3.12", "3.12"),
        ("v3.12", "3.12"),
        ("V3.12", "3.12"),
        ("Python 3.12", "3.12"),
        ("Python v3.12", "3.12"),
        ("Version 3.12.4", "3.12.4"),
        ("  v1.0.0  ", "1.0.0"),
        ("2.7.18", "2.7.18"),
        ("v10", "10"),
        ("3", "3"),
        ("3.10.0", "3.10.0"),
        ("release 1.2.3.4.5", "1.2.3.4.5"),
    ],
)
def test_normalize_version_valid(raw_input: str, expected: str) -> None:
    """Valid version representations normalize to their canonical component string."""
    assert normalize_version(raw_input) == expected
    assert normalize_value(raw_input, ValueType.VERSION) == expected


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        "   ",
        "abc",
        "python",
        "v",
        "V",
        "...",
        "---",
    ],
)
def test_normalize_version_invalid(invalid_input: str) -> None:
    """Invalid version representations raise NormalizationError."""
    with pytest.raises(NormalizationError, match="No valid version components found|empty"):
        normalize_version(invalid_input)


# =============================================================================
# ENUM Normalization Tests
# =============================================================================

DAYS_CATALOG = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]
STATUS_CATALOG = ["up", "down", "degraded", "maintenance"]


@pytest.mark.parametrize(
    ("raw_input", "catalog", "expected"),
    [
        ("friday", DAYS_CATALOG, "friday"),
        ("Friday", DAYS_CATALOG, "friday"),
        ("FRIDAY", DAYS_CATALOG, "friday"),
        ("Fridays", DAYS_CATALOG, "friday"),
        ("fridays", DAYS_CATALOG, "friday"),
        ("on Friday", DAYS_CATALOG, "friday"),
        ("on Fridays", DAYS_CATALOG, "friday"),
        ("  monday  ", DAYS_CATALOG, "monday"),
        ("Mondays", DAYS_CATALOG, "monday"),
        ("on Monday", DAYS_CATALOG, "monday"),
        ("UP", STATUS_CATALOG, "up"),
        ("down", STATUS_CATALOG, "down"),
        ("  Degraded  ", STATUS_CATALOG, "degraded"),
        ("maintenance", STATUS_CATALOG, "maintenance"),
    ],
)
def test_normalize_enum_valid(raw_input: str, catalog: list[str], expected: str) -> None:
    """Valid enum inputs normalize to catalog casing handling case, 'on ', and plurals."""
    assert normalize_enum(raw_input, allowed_values=catalog) == expected
    assert normalize_value(raw_input, ValueType.ENUM, allowed_values=catalog) == expected


@pytest.mark.parametrize(
    ("invalid_input", "catalog"),
    [
        ("unknown", DAYS_CATALOG),
        ("someday", DAYS_CATALOG),
        ("offline", STATUS_CATALOG),
        ("", STATUS_CATALOG),
        ("   ", STATUS_CATALOG),
        ("on ", DAYS_CATALOG),
    ],
)
def test_normalize_enum_invalid(invalid_input: str, catalog: list[str]) -> None:
    """Enum values outside the allowed catalog or empty raise NormalizationError."""
    with pytest.raises(NormalizationError, match="not an allowed enum value|empty"):
        normalize_enum(invalid_input, allowed_values=catalog)


def test_normalize_enum_mixed_case_catalog() -> None:
    """Mixed-case allowed values canonicalize strictly to lowercase output."""
    mixed_catalog = ["Up", "DOWN", "Degraded", "Friday"]
    assert normalize_enum("UP", allowed_values=mixed_catalog) == "up"
    assert normalize_enum("up", allowed_values=mixed_catalog) == "up"
    assert normalize_enum("Down", allowed_values=mixed_catalog) == "down"
    assert normalize_enum("on Friday", allowed_values=mixed_catalog) == "friday"
    assert normalize_enum("Fridays", allowed_values=mixed_catalog) == "friday"


def test_normalize_enum_explicitly_empty_allowed_values() -> None:
    """An explicitly supplied empty allowed_values list must not bypass validation."""
    with pytest.raises(NormalizationError, match="not an allowed enum value"):
        normalize_enum("up", allowed_values=[])
    with pytest.raises(NormalizationError, match="not an allowed enum value"):
        normalize_enum("friday", allowed_values=[])


def test_normalize_enum_without_catalog() -> None:
    """Enum normalization without an explicit catalog falls back to cleaned lowercase."""
    assert normalize_enum("  CustomValue  ") == "customvalue"
    assert normalize_enum("on CustomValue") == "customvalue"


# =============================================================================
# TEXT Normalization Tests
# =============================================================================


@pytest.mark.parametrize(
    ("raw_input", "expected"),
    [
        ("Ubuntu 24.04 LTS", "ubuntu 24.04 lts"),
        ("  Ubuntu   24.04   LTS  ", "ubuntu 24.04 lts"),
        ("API Gateway\nService", "api gateway service"),
        ("SingleWord", "singleword"),
        ("   ", ""),
        ("", ""),
    ],
)
def test_normalize_text(raw_input: str, expected: str) -> None:
    """Text normalization trims, lowercases, and collapses internal whitespace."""
    assert normalize_text(raw_input) == expected
    assert normalize_value(raw_input, ValueType.TEXT) == expected


# =============================================================================
# NUMBER Normalization Tests
# =============================================================================


@pytest.mark.parametrize(
    ("raw_input", "expected"),
    [
        ("10", "10"),
        ("10.0", "10"),
        ("10.00", "10"),
        ("  10  ", "10"),
        ("10.5", "10.5"),
        ("10.50", "10.5"),
        ("0.0", "0"),
        ("0", "0"),
        ("1,000", "1000"),
        ("1,000,000", "1000000"),
        ("1,000 GB", "1000"),
        ("1,000.50 GB", "1000.5"),
        ("16 GB", "16"),
        ("100 ms", "100"),
        ("-5", "-5"),
        ("-5.2", "-5.2"),
        ("-5.20", "-5.2"),
        ("-1000", "-1000"),
        ("+42", "42"),
        ("0.005", "0.005"),
        ("1e3", "1000"),
    ],
)
def test_normalize_number_valid(raw_input: str, expected: str) -> None:
    """Equivalent numeric representations canonicalize identically without precision loss."""
    assert normalize_number(raw_input) == expected
    assert normalize_value(raw_input, ValueType.NUMBER) == expected


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        "   ",
        "abc",
        "abc123xyz",
        "1,2,3",
        "10 GB and 20 GB",
        "1,000,00",
        "no-digits-here",
        "--",
    ],
)
def test_normalize_number_invalid(invalid_input: str) -> None:
    """Invalid numeric inputs raise NormalizationError."""
    with pytest.raises(NormalizationError):
        normalize_number(invalid_input)


# =============================================================================
# DATE Normalization Tests
# =============================================================================


@pytest.mark.parametrize(
    ("raw_input", "expected"),
    [
        ("2026-06-01", "2026-06-01"),
        ("2026/06/01", "2026-06-01"),
        ("2026-6-1", "2026-06-01"),
        ("  2026-06-01  ", "2026-06-01"),
        ("2026-06-01T00:00:00Z", "2026-06-01"),
        ("2026-06-01T15:30:00Z", "2026-06-01"),
        ("2026-06-01T15:30:00+00:00", "2026-06-01"),
        ("2026-06-01T17:30:00+02:00", "2026-06-01"),
        ("2026-06-01T10:30:00-05:00", "2026-06-01"),
        # UTC conversions that cross midnight (previous date or next date)
        ("2026-06-01T01:30:00+04:00", "2026-05-31"),
        ("2026-06-01T22:30:00-04:00", "2026-06-02"),
        ("2026-01-01T02:00:00+05:30", "2025-12-31"),
        ("2026-12-31T20:00:00-06:00", "2027-01-01"),
    ],
)
def test_normalize_date_valid(raw_input: str, expected: str) -> None:
    """Valid ISO date/timestamp inputs normalize strictly to UTC ISO date format (YYYY-MM-DD)."""
    assert normalize_date(raw_input) == expected
    assert normalize_value(raw_input, ValueType.DATE) == expected


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        "   ",
        "2026-06-01T15:30:00",  # Timezone-naive timestamp
        "2026-06-01 15:30:00",  # Timezone-naive datetime
        "2026-02-31",  # Invalid calendar day
        "2026-04-31",  # April has 30 days
        "2026-13-01",  # Invalid month
        "01/02/03",  # Ambiguous
        "12-05-2026",  # Non-ISO
        "June 1, 2026",  # Non-ISO
        "not-a-date",
        "yesterday",
    ],
)
def test_normalize_date_invalid(invalid_input: str) -> None:
    """Invalid or ambiguous dates explicitly raise NormalizationError rather than guessing."""
    with pytest.raises(NormalizationError):
        normalize_date(invalid_input)


# =============================================================================
# ValueType Dispatcher Test
# =============================================================================


def test_normalize_value_unsupported_type() -> None:
    """Unknown value type raises NormalizationError or ValueError."""
    with pytest.raises(ValueError):
        normalize_value("foo", "unsupported_type")  # type: ignore[arg-type]
