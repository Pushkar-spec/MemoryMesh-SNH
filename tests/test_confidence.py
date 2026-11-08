"""Unit tests for the pure confidence and evidence calculation engine.

Governed by docs/RESOLUTION_ENGINE.md Section 2, Section 8, and docs/SPECIFICATION.md Section 1.1.
P3.1 Implementation: Confidence Engine Tests.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memorymesh.engine.confidence import (
    calculate_base_confidence,
    calculate_decay,
    calculate_effective_confidence,
    calculate_strength,
    is_stale,
    recompute_memory_confidence,
)
from memorymesh.models import NewEvidence


class TestStrengthCalculation:
    """Tests for strength(e) = source.reliability * claim.extract_confidence."""

    def test_hero_scenario_strengths(self) -> None:
        # Priya (0.90) * direct statement (0.95) -> 0.855
        s1 = calculate_strength(0.90, 0.95)
        assert s1 == pytest.approx(0.855, abs=0.0001)

        # Wiki (0.75) * direct statement (0.95) -> 0.7125
        s2 = calculate_strength(0.75, 0.95)
        assert s2 == pytest.approx(0.7125, abs=0.0001)

        # Intern (0.30) * hedged statement (0.50) -> 0.15
        s3 = calculate_strength(0.30, 0.50)
        assert s3 == pytest.approx(0.15, abs=0.0001)

    def test_strength_boundaries(self) -> None:
        assert calculate_strength(0.0, 0.5) == 0.0
        assert calculate_strength(0.5, 0.0) == 0.0
        assert calculate_strength(1.0, 1.0) == 1.0
        assert calculate_strength(0.0, 0.0) == 0.0

    def test_invalid_strength_inputs_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="source_reliability"):
            calculate_strength(-0.1, 0.95)
        with pytest.raises(ValueError, match="source_reliability"):
            calculate_strength(1.1, 0.95)
        with pytest.raises(ValueError, match="extract_confidence"):
            calculate_strength(0.90, -0.1)
        with pytest.raises(ValueError, match="extract_confidence"):
            calculate_strength(0.90, 1.05)


class TestBaseConfidenceCalculation:
    """Tests for base_confidence(m) with noisy-OR combination and same-source handling."""

    def test_hero_scenario_numbers_and_corroboration(self) -> None:
        t = datetime(2026, 1, 10, tzinfo=UTC)

        # m1: "Server A runs Python 3.10" (priya 0.90, extract 0.95 -> 0.855)
        ev1 = NewEvidence(
            source_id="sysadmin_priya",
            message_id=1,
            strength=0.855,
            observed_at=t,
        )
        conf1 = calculate_base_confidence([ev1])
        assert conf1 == pytest.approx(0.855, abs=0.001)

        # m2: "Runtime for Server A: Python 3.10" (wiki 0.75, extract 0.95 -> 0.7125)
        ev2 = NewEvidence(
            source_id="wiki",
            message_id=2,
            strength=0.7125,
            observed_at=t,
        )
        # 1 - (1 - 0.855) * (1 - 0.7125) = 1 - 0.145 * 0.2875 = 0.9583125
        conf2 = calculate_base_confidence([ev1, ev2])
        assert conf2 == pytest.approx(0.958, abs=0.001)

    def test_same_source_duplicate_does_not_boost_confidence(self) -> None:
        t1 = datetime(2026, 1, 10, tzinfo=UTC)
        t2 = datetime(2026, 1, 15, tzinfo=UTC)

        ev1 = NewEvidence(
            source_id="sysadmin_priya",
            message_id=1,
            strength=0.855,
            observed_at=t1,
        )
        ev2 = NewEvidence(
            source_id="sysadmin_priya",
            message_id=2,
            strength=0.855,
            observed_at=t2,
        )

        conf = calculate_base_confidence([ev1, ev2])
        # Two identical statements from the same source must yield 0.855, not 1 - (1-0.855)^2
        assert conf == pytest.approx(0.855, abs=0.0001)

    def test_same_source_takes_maximum_strength(self) -> None:
        t = datetime(2026, 1, 10, tzinfo=UTC)
        ev_weak = NewEvidence(
            source_id="sysadmin_priya",
            message_id=1,
            strength=0.50,
            observed_at=t,
        )
        ev_strong = NewEvidence(
            source_id="sysadmin_priya",
            message_id=2,
            strength=0.855,
            observed_at=t,
        )

        # Forward and reverse order must both pick the maximum strength 0.855
        assert calculate_base_confidence([ev_weak, ev_strong]) == pytest.approx(0.855, abs=0.0001)
        assert calculate_base_confidence([ev_strong, ev_weak]) == pytest.approx(0.855, abs=0.0001)

    def test_three_distinct_sources_noisy_or(self) -> None:
        t = datetime(2026, 1, 10, tzinfo=UTC)
        ev1 = NewEvidence(source_id="s1", message_id=1, strength=0.5, observed_at=t)
        ev2 = NewEvidence(source_id="s2", message_id=2, strength=0.5, observed_at=t)
        ev3 = NewEvidence(source_id="s3", message_id=3, strength=0.5, observed_at=t)

        # 1 - (1 - 0.5)^3 = 1 - 0.125 = 0.875
        conf = calculate_base_confidence([ev1, ev2, ev3])
        assert conf == pytest.approx(0.875, abs=0.0001)

    def test_confidence_capped_at_max_confidence(self) -> None:
        t = datetime(2026, 1, 10, tzinfo=UTC)
        ev1 = NewEvidence(source_id="s1", message_id=1, strength=0.95, observed_at=t)
        ev2 = NewEvidence(source_id="s2", message_id=2, strength=0.95, observed_at=t)

        # 1 - (0.05)^2 = 0.9975 -> capped at max_confidence 0.99
        conf = calculate_base_confidence([ev1, ev2], max_confidence=0.99)
        assert conf == 0.99

    def test_empty_evidence_returns_zero(self) -> None:
        assert calculate_base_confidence([]) == 0.0

    def test_invalid_evidence_strength_raises(self) -> None:
        t = datetime(2026, 1, 10, tzinfo=UTC)
        ev_invalid = NewEvidence(source_id="s1", message_id=1, strength=1.0, observed_at=t)
        ev_invalid.strength = 1.5  # type: ignore[misc]

        with pytest.raises(ValueError, match="Evidence strength"):
            calculate_base_confidence([ev_invalid])

    def test_recompute_memory_confidence_matches_calculate_base(self) -> None:
        t = datetime(2026, 1, 10, tzinfo=UTC)
        ev1 = NewEvidence(source_id="s1", message_id=1, strength=0.8, observed_at=t)
        ev2 = NewEvidence(source_id="s2", message_id=2, strength=0.6, observed_at=t)

        expected = calculate_base_confidence([ev1, ev2])
        recomputed = recompute_memory_confidence([ev1, ev2])
        assert recomputed == expected

    def test_confidence_is_strictly_deterministic(self) -> None:
        t = datetime(2026, 1, 10, tzinfo=UTC)
        evidence_list = [
            NewEvidence(source_id=f"src_{i}", message_id=i, strength=0.4, observed_at=t)
            for i in range(5)
        ]

        results = [calculate_base_confidence(evidence_list) for _ in range(100)]
        assert len(set(results)) == 1


class TestDecayAndEffectiveConfidence:
    """Tests for calculate_decay, calculate_effective_confidence, and is_stale."""

    def test_decay_without_half_life(self) -> None:
        t1 = datetime(2026, 1, 1, tzinfo=UTC)
        t2 = datetime(2026, 6, 1, tzinfo=UTC)

        assert calculate_decay(t1, t2, half_life_days=None) == 1.0

    def test_decay_zero_or_negative_half_life_rejected(self) -> None:
        t1 = datetime(2026, 1, 1, tzinfo=UTC)
        t2 = datetime(2026, 6, 1, tzinfo=UTC)

        with pytest.raises(ValueError, match="half_life_days must be strictly positive"):
            calculate_decay(t1, t2, half_life_days=0.0)

        with pytest.raises(ValueError, match="half_life_days must be strictly positive"):
            calculate_decay(t1, t2, half_life_days=-5.0)

    def test_decay_future_or_same_timestamp(self) -> None:
        t = datetime(2026, 1, 1, tzinfo=UTC)
        t_past = datetime(2025, 12, 1, tzinfo=UTC)

        # at == valid_from must return 1.0
        assert calculate_decay(t, t, half_life_days=30.0) == 1.0
        # at < valid_from (future memory validity) must return 1.0, not zero
        assert calculate_decay(t, t_past, half_life_days=30.0) == 1.0

    def test_decay_timezone_awareness_and_conversion(self) -> None:
        from datetime import timedelta, timezone, tzinfo

        # 1. Naive timestamps must be rejected
        t_naive = datetime(2026, 1, 1)
        t_aware = datetime(2026, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="Timestamps .* must be timezone-aware"):
            calculate_decay(t_naive, t_aware, half_life_days=30.0)
        with pytest.raises(ValueError, match="Timestamps .* must be timezone-aware"):
            calculate_decay(t_aware, t_naive, half_life_days=30.0)

        # 2. Tzinfo exists but utcoffset() returns None
        class UnspecifiedTz(tzinfo):
            def utcoffset(self, dt: datetime | None) -> None:
                return None

            def tzname(self, dt: datetime | None) -> str | None:
                return "Unspecified"

            def dst(self, dt: datetime | None) -> None:
                return None

        t_unspecified = datetime(2026, 1, 1, tzinfo=UnspecifiedTz())
        assert t_unspecified.tzinfo is not None
        assert t_unspecified.utcoffset() is None

        with pytest.raises(ValueError, match="Timestamps .* must be timezone-aware"):
            calculate_decay(t_unspecified, t_aware, half_life_days=30.0)
        with pytest.raises(ValueError, match="Timestamps .* must be timezone-aware"):
            calculate_decay(t_aware, t_unspecified, half_life_days=30.0)

        # 3. Different offsets normalized consistently to UTC
        tz_plus5 = timezone(timedelta(hours=5))
        # 2026-01-31T05:00:00+05:00 is exactly 2026-01-31T00:00:00 UTC (30 days later)
        t_plus5 = datetime(2026, 1, 31, 5, 0, tzinfo=tz_plus5)
        d = calculate_decay(t_aware, t_plus5, half_life_days=30.0)
        assert d == pytest.approx(0.5, abs=0.001)

    def test_decay_exponential_halving(self) -> None:
        t1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        t_30d = datetime(2026, 1, 31, 0, 0, tzinfo=UTC)  # exactly 30 days
        t_60d = datetime(2026, 3, 2, 0, 0, tzinfo=UTC)  # exactly 60 days

        d30 = calculate_decay(t1, t_30d, half_life_days=30.0)
        assert d30 == pytest.approx(0.5, abs=0.001)

        d60 = calculate_decay(t1, t_60d, half_life_days=30.0)
        assert d60 == pytest.approx(0.25, abs=0.001)

    def test_effective_confidence_with_review_penalty(self) -> None:
        # Worked example row m5 in RESOLUTION_ENGINE.md:
        # Flagged memory effective conf = 0.855 * 0.5 = 0.4275
        eff = calculate_effective_confidence(
            base_conf=0.855,
            decay_factor=1.0,
            needs_review=True,
            review_penalty=0.50,
        )
        assert eff == pytest.approx(0.4275, abs=0.0001)

    def test_effective_confidence_without_review_penalty(self) -> None:
        eff = calculate_effective_confidence(
            base_conf=0.855,
            decay_factor=1.0,
            needs_review=False,
            review_penalty=0.50,
        )
        assert eff == pytest.approx(0.855, abs=0.0001)

    def test_effective_confidence_with_both_decay_and_penalty(self) -> None:
        eff = calculate_effective_confidence(
            base_conf=0.80,
            decay_factor=0.50,
            needs_review=True,
            review_penalty=0.50,
        )
        # 0.80 * 0.50 * 0.50 = 0.20
        assert eff == pytest.approx(0.20, abs=0.0001)

    def test_is_stale_floor(self) -> None:
        assert is_stale(0.29, stale_floor=0.30) is True
        assert is_stale(0.30, stale_floor=0.30) is False
        assert is_stale(0.31, stale_floor=0.30) is False

    def test_is_stale_with_different_configured_thresholds(self) -> None:
        # Default configured stale_floor = 0.30
        assert is_stale(0.25, stale_floor=0.30) is True
        assert is_stale(0.35, stale_floor=0.30) is False

        # Strict configured stale_floor = 0.60
        assert is_stale(0.50, stale_floor=0.60) is True
        assert is_stale(0.65, stale_floor=0.60) is False

        # Permissive configured stale_floor = 0.10
        assert is_stale(0.15, stale_floor=0.10) is False
        assert is_stale(0.05, stale_floor=0.10) is True

    def test_effective_confidence_input_validation(self) -> None:
        with pytest.raises(ValueError, match="base_conf"):
            calculate_effective_confidence(-0.1)
        with pytest.raises(ValueError, match="base_conf"):
            calculate_effective_confidence(1.1)
        with pytest.raises(ValueError, match="decay_factor"):
            calculate_effective_confidence(0.8, decay_factor=-0.1)
        with pytest.raises(ValueError, match="decay_factor"):
            calculate_effective_confidence(0.8, decay_factor=1.1)
        with pytest.raises(ValueError, match="review_penalty"):
            calculate_effective_confidence(0.8, review_penalty=-0.1)
        with pytest.raises(ValueError, match="review_penalty"):
            calculate_effective_confidence(0.8, review_penalty=1.1)

    def test_is_stale_input_validation(self) -> None:
        with pytest.raises(ValueError, match="effective_conf"):
            is_stale(-0.1, stale_floor=0.30)
        with pytest.raises(ValueError, match="effective_conf"):
            is_stale(1.1, stale_floor=0.30)
        with pytest.raises(ValueError, match="stale_floor"):
            is_stale(0.5, stale_floor=-0.1)
        with pytest.raises(ValueError, match="stale_floor"):
            is_stale(0.5, stale_floor=1.1)

    def test_calculate_base_confidence_max_confidence_validation(self) -> None:
        with pytest.raises(ValueError, match="max_confidence"):
            calculate_base_confidence([], max_confidence=0.0)
        with pytest.raises(ValueError, match="max_confidence"):
            calculate_base_confidence([], max_confidence=-0.5)
        with pytest.raises(ValueError, match="max_confidence"):
            calculate_base_confidence([], max_confidence=1.1)
