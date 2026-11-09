"""Pure functional invariant checks for MemoryMesh.

Governed by docs/AGENTS.md Section 3D, docs/ARCHITECTURE.md, and docs/SPECIFICATION.md.
Contains pure validation functions with no database access and no clock access.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import NamedTuple

from memorymesh.engine.confidence import EvidenceLike, calculate_base_confidence

INV_CLOSED_AT_REQUIRED = "INV-CLOSED-AT-REQUIRED"
INV_VALID_INTERVAL = "INV-VALID-INTERVAL"


class TemporalViolation(NamedTuple):
    code: str
    description: str


def validate_temporal_window(
    valid_from: datetime,
    valid_to: datetime | None,
    closed_at: datetime | None,
) -> list[TemporalViolation]:
    """Validate temporal validity invariants for a memory record.

    Invariants:
    1. valid_to IS NOT NULL implies closed_at IS NOT NULL (INV-CLOSED-AT-REQUIRED).
    2. valid_to > valid_from whenever valid_to is set (INV-VALID-INTERVAL).

    Returns:
        List of TemporalViolation(code, description), empty if valid.
    """
    violations: list[TemporalViolation] = []
    if valid_to is not None and closed_at is None:
        violations.append(
            TemporalViolation(
                code=INV_CLOSED_AT_REQUIRED,
                description="valid_to is set but closed_at is NULL",
            )
        )
    if valid_to is not None and valid_to <= valid_from:
        violations.append(
            TemporalViolation(
                code=INV_VALID_INTERVAL,
                description=(
                    f"valid_to ({valid_to.isoformat()}) must be strictly greater "
                    f"than valid_from ({valid_from.isoformat()})"
                ),
            )
        )
    return violations


def check_intervals_overlap(
    start1: datetime,
    end1: datetime | None,
    start2: datetime,
    end2: datetime | None,
) -> bool:
    """Check whether two half-open temporal intervals [start1, end1) and [start2, end2) overlap.

    None represents an open-ended interval (+infinity).
    """
    latest_start = max(start1, start2)
    if end1 is None and end2 is None:
        return True
    if end1 is None:
        return latest_start < end2  # type: ignore[operator]
    if end2 is None:
        return latest_start < end1
    earliest_end = min(end1, end2)
    return latest_start < earliest_end


def validate_confidence_recomputation(
    stored_confidence: float,
    evidence: Sequence[EvidenceLike],
    *,
    max_confidence: float = 0.99,
    tolerance: float = 1e-4,
) -> float | None:
    """Verify that stored confidence matches noisy-OR calculation from evidence.

    Returns:
        Expected confidence if there is a mismatch exceeding tolerance, else None.
    """
    expected = calculate_base_confidence(evidence, max_confidence=max_confidence)
    if abs(stored_confidence - expected) > tolerance:
        return expected
    return None
