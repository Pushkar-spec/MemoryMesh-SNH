"""Pure deterministic confidence and evidence calculation engine for MemoryMesh.

Governed by docs/RESOLUTION_ENGINE.md Section 2, docs/SPECIFICATION.md Section 1.1,
and docs/ARCHITECTURE.md (ADR-006).

Functional Core:
  - Pure functions only.
  - No database access or repository imports.
  - No network calls or LLM invocations.
  - No datetime.now() invocations; reference times must be passed explicitly.
  - Deterministic calculations based on evidence records.

Formulas:
  strength(e)           = source.reliability * claim.extract_confidence
  base_confidence(m)    = min(max_confidence,
                              1 - PRODUCT_s(1 - max_e(strength(s, e))))
  decay(m, at)          = 0.5 ^ (age_days / half_life_days) if half_life_days else 1.0
  effective_conf(m, at) = base_confidence * decay * (review_penalty if needs_review else 1.0)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol


class EvidenceLike(Protocol):
    """Protocol matching Evidence and NewEvidence models for confidence calculations."""

    @property
    def source_id(self) -> str: ...

    @property
    def strength(self) -> float: ...


def calculate_strength(source_reliability: float, extract_confidence: float) -> float:
    """Calculate the strength of an individual piece of evidence.

    Formula:
      strength = source.reliability * claim.extract_confidence

    Args:
        source_reliability: Reliability rating of the source in [0.0, 1.0].
        extract_confidence: Model extraction confidence score in [0.0, 1.0].

    Returns:
        Combined strength in [0.0, 1.0].

    Raises:
        ValueError: If either input is outside [0.0, 1.0].
    """
    if not (0.0 <= source_reliability <= 1.0):
        raise ValueError(
            f"source_reliability must be between 0.0 and 1.0, got {source_reliability}."
        )
    if not (0.0 <= extract_confidence <= 1.0):
        raise ValueError(
            f"extract_confidence must be between 0.0 and 1.0, got {extract_confidence}."
        )

    return float(source_reliability * extract_confidence)


def calculate_base_confidence(
    evidence: list[EvidenceLike],
    max_confidence: float = 0.99,
) -> float:
    """Calculate the base confidence of a memory from its associated evidence.

    Combines independent evidence using the noisy-OR model across distinct sources.
    Multiple evidence records from the same source take the maximum strength among
    that source's statements (a source cannot corroborate itself).

    Formula:
      base_conf = min(max_confidence, 1 - PRODUCT_s(1 - max_e(strength(s, e))))

    Args:
        evidence: List of evidence items supporting the memory.
        max_confidence: Ceiling cap on confidence (default 0.99). Must be in (0.0, 1.0].

    Returns:
        Base confidence score in [0.0, max_confidence]. Returns 0.0 if evidence list is empty.

    Raises:
        ValueError: If max_confidence is invalid or any evidence item has an invalid strength.
    """
    if not (0.0 < max_confidence <= 1.0):
        raise ValueError(
            f"max_confidence must be strictly positive and <= 1.0, got {max_confidence}."
        )

    if not evidence:
        return 0.0

    # Group evidence by distinct source_id, keeping the maximum strength per source
    source_max_strengths: dict[str, float] = {}
    for ev in evidence:
        s = float(ev.strength)
        if not (0.0 <= s <= 1.0):
            raise ValueError(f"Evidence strength must be between 0.0 and 1.0, got {s}.")
        source_id = ev.source_id
        if source_id not in source_max_strengths or s > source_max_strengths[source_id]:
            source_max_strengths[source_id] = s

    # Noisy-OR product across distinct sources
    product = 1.0
    for strength in source_max_strengths.values():
        product *= 1.0 - strength

    combined = 1.0 - product
    # Cap at max_confidence
    return min(max_confidence, max(0.0, combined))


def calculate_decay(
    valid_from: datetime,
    at: datetime,
    half_life_days: float | None = None,
) -> float:
    """Calculate temporal exponential decay factor for a memory at reference time 'at'.

    Formula:
      decay = 2 ** (-delta_t / half_life_days) if half_life_days is set else 1.0
      where delta_t = max(0.0, (at - valid_from) in days)

    Semantics:
      - If at is earlier than valid_from, decay is 1.0.
      - If at equals valid_from, decay is 1.0.
      - Future valid_from values do not produce zero confidence.
      - If half_life_days is None, facts do not decay (decay is 1.0).
      - Zero or negative half_life_days raises ValueError.

    Args:
        valid_from: Start of memory validity window (must be timezone-aware).
        at: Query reference timestamp (must be timezone-aware).
        half_life_days: Number of days for confidence to halve, or None if facts do not decay.

    Returns:
        Decay multiplier factor in (0.0, 1.0].

    Raises:
        ValueError: If timestamps are timezone-naive, lack a UTC offset, or half_life_days
            is non-positive.
    """
    if (
        valid_from.tzinfo is None
        or valid_from.utcoffset() is None
        or at.tzinfo is None
        or at.utcoffset() is None
    ):
        raise ValueError(
            "Timestamps (valid_from, at) must be timezone-aware with a defined UTC offset."
        )

    if half_life_days is None:
        return 1.0

    if half_life_days <= 0.0:
        raise ValueError(f"half_life_days must be strictly positive (> 0.0), got {half_life_days}.")

    valid_from_utc = valid_from.astimezone(UTC)
    at_utc = at.astimezone(UTC)

    if at_utc <= valid_from_utc:
        return 1.0

    delta_seconds = (at_utc - valid_from_utc).total_seconds()
    age_days = delta_seconds / 86400.0

    factor = 2.0 ** (-age_days / half_life_days)
    return max(0.0, min(1.0, factor))


def calculate_effective_confidence(
    base_conf: float,
    decay_factor: float = 1.0,
    needs_review: bool = False,
    review_penalty: float = 0.5,
) -> float:
    """Calculate effective confidence after applying temporal decay and review penalty.

    Formula:
      effective_conf = base_confidence * decay * (review_penalty if needs_review else 1.0)

    Args:
        base_conf: Base confidence score in [0.0, 1.0].
        decay_factor: Temporal decay factor in [0.0, 1.0].
        needs_review: True if memory is flagged for human review.
        review_penalty: Multiplier applied when needs_review is True (default 0.5).

    Returns:
        Effective confidence in [0.0, 1.0].

    Raises:
        ValueError: If any multiplier is outside [0.0, 1.0].
    """
    if not (0.0 <= base_conf <= 1.0):
        raise ValueError(f"base_conf must be between 0.0 and 1.0, got {base_conf}.")
    if not (0.0 <= decay_factor <= 1.0):
        raise ValueError(f"decay_factor must be between 0.0 and 1.0, got {decay_factor}.")
    if not (0.0 <= review_penalty <= 1.0):
        raise ValueError(f"review_penalty must be between 0.0 and 1.0, got {review_penalty}.")

    penalty = review_penalty if needs_review else 1.0
    effective = base_conf * decay_factor * penalty
    return max(0.0, min(1.0, effective))


def is_stale(
    effective_conf: float,
    stale_floor: float,
) -> bool:
    """Determine whether an effective confidence score falls below the staleness floor.

    Args:
        effective_conf: Effective confidence score in [0.0, 1.0].
        stale_floor: Threshold below which a memory is considered stale, in [0.0, 1.0].

    Returns:
        True if effective_conf < stale_floor, False otherwise.

    Raises:
        ValueError: If effective_conf or stale_floor is outside [0.0, 1.0].
    """
    if not (0.0 <= effective_conf <= 1.0):
        raise ValueError(f"effective_conf must be between 0.0 and 1.0, got {effective_conf}.")
    if not (0.0 <= stale_floor <= 1.0):
        raise ValueError(f"stale_floor must be between 0.0 and 1.0, got {stale_floor}.")

    return bool(effective_conf < stale_floor)


def recompute_memory_confidence(
    evidence: list[EvidenceLike],
    max_confidence: float = 0.99,
) -> float:
    """Recompute base confidence for a memory from its full evidence list.

    Convenience wrapper around calculate_base_confidence.
    """
    return calculate_base_confidence(evidence, max_confidence=max_confidence)
