"""Value normalization module for MemoryMesh.

Governed by docs/SPECIFICATION.md Section 6, docs/ARCHITECTURE.md,
and docs/RESOLUTION_ENGINE.md.
Implements canonical task P2.1 (T2.1): deterministic value normalization for all five ValueTypes:
  - VERSION: Extract version string, strip leading 'v'/'V' or prefix, preserve sub-versions.
  - ENUM: Lowercase, strip whitespace, strip prefix 'on ', de-pluralize, and validate.
  - TEXT: Collapse internal whitespace, strip ends, lowercase for value_norm.
  - NUMBER: Decimal canonicalization, unit-stripping, avoiding floating point precision errors.
  - DATE: Parse ISO dates (YYYY-MM-DD or UTC datetime), reject ambiguous/invalid calendar dates.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

from memorymesh.errors import NormalizationError
from memorymesh.models import ValueType


def normalize_version(value: str) -> str:
    """Normalize a version string by extracting the canonical numeric version components.

    Examples:
      - 'v3.12' -> '3.12'
      - 'V3.12' -> '3.12'
      - 'Python 3.12' -> '3.12'
      - '  v1.0.0-alpha  ' -> '1.0.0'
      - '3' -> '3'

    Raises:
      NormalizationError: When no valid version component can be extracted or input is empty.
    """
    cleaned = value.strip()
    if not cleaned:
        raise NormalizationError("Cannot normalize empty version string.")

    # Match numeric version components with optional 'v'/'V' prefix or after boundary/word
    match = re.search(r"(?:(?<=[^\w.])|[vV]|\b)(\d+(?:\.\d+)*)", cleaned)
    if not match:
        raise NormalizationError(f"No valid version components found in '{value}'.")

    return match.group(1)


def normalize_enum(value: str, allowed_values: list[str] | None = None) -> str:
    """Normalize an enum string using lowercase, prefix stripping, and de-pluralization.

    Examples:
      - 'Friday' -> 'friday'
      - 'on Friday' -> 'friday'
      - 'Fridays' -> 'friday'
      - '  UP  ' -> 'up'

    Raises:
      NormalizationError: When value is empty or outside the configured allowed_values catalog.
    """
    cleaned = value.strip().lower()
    if not cleaned:
        raise NormalizationError("Cannot normalize empty enum string.")

    # Strip prefix 'on ' (e.g. 'on Friday' -> 'friday')
    if cleaned.startswith("on "):
        cleaned = cleaned[3:].strip()

    if allowed_values is not None:
        # Normalize allowed catalog values to lowercase for case-insensitive matching
        allowed_set: set[str] = {
            allowed.strip().lower() for allowed in allowed_values if allowed.strip()
        }

        # 1. Exact match (case-insensitive) -> canonical lowercase output
        if cleaned in allowed_set:
            return cleaned

        # 2. De-pluralization: e.g. 'fridays' -> 'friday', 'mondays' -> 'monday'
        if cleaned.endswith("s") and cleaned[:-1] in allowed_set:
            return cleaned[:-1]

        raise NormalizationError(
            f"Value '{value}' is not an allowed enum value. Allowed: {allowed_values}"
        )

    return cleaned


def normalize_text(value: str) -> str:
    """Normalize a free-text string by stripping and collapsing internal whitespace.

    Examples:
      - '  Ubuntu   24.04 LTS  ' -> 'ubuntu 24.04 lts'
      - 'API Gateway' -> 'api gateway'
    """
    return " ".join(value.strip().split()).lower()


NUMBER_PATTERN = re.compile(
    r"^([+-]?(?:(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?)|(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?))"
    r"(?:\s*([a-zA-Z%°/][a-zA-Z0-9_°/%-]*))?$"
)


def normalize_number(value: str) -> str:
    """Normalize a numeric representation to a canonical Decimal string without precision loss.

    Strictly parses numeric representation:
      - Strips valid unit suffix (e.g. 'GB', 'MB', '%').
      - Rejects malformed comma grouping (e.g. '1,2,3').
      - Rejects multiple numbers (e.g. '10 GB and 20 GB').
      - Rejects arbitrary text containing numbers (e.g. 'abc123xyz').

    Examples:
      - '10' -> '10'
      - '10.0' -> '10'
      - '10.00' -> '10'
      - '10.5' -> '10.5'
      - '1,000 GB' -> '1000'
      - '1,000.50 GB' -> '1000.5'
      - '-5.2' -> '-5.2'

    Raises:
      NormalizationError: When no valid numeric representation can be extracted or input is invalid.
    """
    cleaned = value.strip()
    if not cleaned:
        raise NormalizationError("Cannot normalize empty number string.")

    match = NUMBER_PATTERN.match(cleaned)
    if not match:
        raise NormalizationError(
            f"Invalid numeric input '{value}'. Expected complete numeric representation."
        )

    raw_num = match.group(1).replace(",", "")
    try:
        d = Decimal(raw_num)
    except (InvalidOperation, ValueError) as exc:
        raise NormalizationError(f"Invalid numeric input '{value}': {exc}") from exc

    if d == 0:
        return "0"

    norm = d.to_integral() if d == d.to_integral() else d.normalize()
    return f"{norm:f}"


def normalize_date(value: str) -> str:
    """Normalize a date string to canonical ISO date format (YYYY-MM-DD).

    Accepts:
      - Documented ISO date values: 'YYYY-MM-DD' or 'YYYY/MM/DD'
      - Timezone-aware timestamps (converts to UTC and reduces to UTC calendar date)

    Rejects:
      - Timezone-naive timestamps (e.g. '2026-06-01T15:30:00')
      - Invalid calendar dates (e.g. '2026-02-31')
      - Ambiguous formats (e.g. '01/02/03', 'yesterday')

    Examples:
      - '2026-06-01' -> '2026-06-01'
      - '2026/06/01' -> '2026-06-01'
      - '2026-06-01T00:00:00Z' -> '2026-06-01'
      - '2026-06-01T15:30:00Z' -> '2026-06-01'
      - '2026-06-01T17:30:00+02:00' -> '2026-06-01'
      - '2026-06-01T01:30:00+04:00' -> '2026-05-31'
      - '2026-06-01T22:30:00-04:00' -> '2026-06-02'

    Raises:
      NormalizationError: When date is invalid, timezone-naive, or in an ambiguous format.
    """
    cleaned = value.strip()
    if not cleaned:
        raise NormalizationError("Cannot normalize empty date string.")

    # 1. Standard calendar date: YYYY-MM-DD or YYYY/MM/DD
    date_match = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", cleaned)
    if date_match:
        year, month, day = (
            int(date_match.group(1)),
            int(date_match.group(2)),
            int(date_match.group(3)),
        )
        try:
            d = datetime(year, month, day)
            return d.strftime("%Y-%m-%d")
        except ValueError as exc:
            raise NormalizationError(f"Invalid calendar date '{value}': {exc}") from exc

    # 2. ISO timestamp: must be timezone-aware
    try:
        dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NormalizationError(
            f"Invalid or ambiguous date format '{value}'. Expected ISO date format (YYYY-MM-DD)."
        ) from exc

    if dt.tzinfo is None:
        raise NormalizationError(
            f"Timezone-naive timestamp '{value}' is not supported; timezone required."
        )

    dt_utc = dt.astimezone(UTC)
    return dt_utc.strftime("%Y-%m-%d")


def normalize_value(
    value: str,
    value_type: ValueType | str,
    *,
    allowed_values: list[str] | None = None,
) -> str:
    """Normalize a raw claim value according to the specified ValueType."""
    vt = ValueType(value_type) if not isinstance(value_type, ValueType) else value_type

    if vt == ValueType.VERSION:
        return normalize_version(value)
    if vt == ValueType.ENUM:
        return normalize_enum(value, allowed_values=allowed_values)
    if vt == ValueType.TEXT:
        return normalize_text(value)
    if vt == ValueType.NUMBER:
        return normalize_number(value)
    if vt == ValueType.DATE:
        return normalize_date(value)

    raise NormalizationError(f"Unsupported value type: {vt}")
