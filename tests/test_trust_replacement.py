"""Tests for P3.2 Trust Replacement, Low-Trust Rejection, Dispute Guards, Revert Protection.

Governed by docs/RESOLUTION_ENGINE.md Section 4.1 B/C, docs/SPECIFICATION.md Section 1.1 & 2.1,
and docs/ARCHITECTURE.md.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memorymesh.config import load_app_config, load_engine_config
from memorymesh.db import init_db
from memorymesh.engine.rules import resolve
from memorymesh.models import (
    Action,
    AddEdge,
    AttributeDef,
    Cardinality,
    Claim,
    EdgeType,
    IngestMessage,
    InsertMemory,
    Memory,
    MemoryStatus,
    Relation,
    ResolvedClaim,
    Source,
    UpdateMemory,
    ValueType,
)
from memorymesh.repository import Repository
from memorymesh.service import MemoryService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def version_attr() -> AttributeDef:
    """Single-valued version attribute."""
    return AttributeDef(
        attribute="python_version",
        description="Installed Python runtime version",
        cardinality=Cardinality.SINGLE,
        value_type=ValueType.VERSION,
    )


@pytest.fixture
def high_trust_source() -> Source:
    """High-reliability admin source (0.95)."""
    return Source(
        id="sysadmin_priya",
        name="Priya Sharma",
        type="admin",
        reliability=0.95,
        notes="Senior Admin",
    )


@pytest.fixture
def med_trust_source() -> Source:
    """Medium-reliability chat source (0.60)."""
    return Source(
        id="dev_chat",
        name="Developer Chat",
        type="chat",
        reliability=0.60,
    )


@pytest.fixture
def low_trust_source() -> Source:
    """Low-reliability intern source (0.30)."""
    return Source(
        id="intern_dan",
        name="Dan Intern",
        type="user",
        reliability=0.30,
    )


@pytest.fixture
def service() -> MemoryService:
    """Create MemoryService instance with seeded catalogs."""
    conn = init_db(":memory:")
    repo = Repository(conn)

    repo.insert_source(
        Source(
            id="sysadmin_priya",
            name="Priya Sharma",
            type="admin",
            reliability=0.95,
            notes="Senior Admin",
        )
    )
    repo.insert_source(
        Source(
            id="dev_chat",
            name="Developer Chat",
            type="chat",
            reliability=0.60,
        )
    )
    repo.insert_source(
        Source(
            id="intern_dan",
            name="Dan Intern",
            type="user",
            reliability=0.30,
        )
    )
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="python_version",
            description="Installed Python runtime version",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
        )
    )

    app_cfg = load_app_config()
    return MemoryService(repo, app_cfg)


# ---------------------------------------------------------------------------
# Pure Engine Tests: R-SKIP-LOWCONF
# ---------------------------------------------------------------------------


def test_r_skip_lowconf(version_attr: AttributeDef, high_trust_source: Source) -> None:
    """Claims with extract_confidence < min_extract_confidence are skipped with R-SKIP-LOWCONF."""
    cfg = load_engine_config()
    t_valid = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)

    rc = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.30,  # Below min_extract_confidence (0.40)
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t_valid,
        source=high_trust_source,
        message_id=101,
    )

    decision = resolve(
        rc=rc,
        chain=[],
        schema=version_attr,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t_valid,
    )

    assert decision.action == Action.SKIP
    assert decision.rule_id == "R-SKIP-LOWCONF"
    assert "min_extract_confidence" in decision.explanation
    assert len(decision.mutations) == 0


# ---------------------------------------------------------------------------
# Pure Engine Tests: Trust-Based Replacement (R-REPLACE)
# ---------------------------------------------------------------------------


def test_r_replace_stronger_claim_satisfies_margin(
    version_attr: AttributeDef,
    high_trust_source: Source,
    med_trust_source: Source,
) -> None:
    """Stronger newer claim satisfies replace_margin (new_score >= old * margin) -> R-REPLACE."""
    cfg = load_engine_config()
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    # Existing memory: med trust (0.60), confidence 0.80 -> old_score = 0.80
    active_mem = Memory(
        id=1,
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t1,
        recorded_at=t1,
        status=MemoryStatus.ACTIVE,
        confidence=0.80,
        message_id=101,
        source_id=med_trust_source.id,
    )

    # Newer claim: high trust (0.95), extract 0.90 -> s_new = 0.855
    # old_score * replace_margin = 0.80 * 0.80 = 0.64 <= 0.855
    rc_newer = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.90,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t2,
        source=high_trust_source,
        message_id=102,
    )

    decision = resolve(
        rc=rc_newer,
        chain=[active_mem],
        schema=version_attr,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t2,
    )

    assert decision.action == Action.REPLACE
    assert decision.rule_id == "R-REPLACE"
    assert decision.scores["new_score"] == pytest.approx(0.855, rel=1e-3)
    assert decision.affected_memory_ids == [1]

    # Verify mutations:
    # 1. Update old memory: status -> HISTORICAL, valid_to -> rc.valid_from
    update_muts = [m for m in decision.mutations if isinstance(m, UpdateMemory)]
    assert len(update_muts) == 1
    assert update_muts[0].memory_id == 1
    assert update_muts[0].fields["status"] == MemoryStatus.HISTORICAL
    assert update_muts[0].fields["valid_to"] == t2
    assert update_muts[0].fields["closed_at"] == t2

    # 2. Insert new memory: status -> ACTIVE
    insert_muts = [m for m in decision.mutations if isinstance(m, InsertMemory)]
    assert len(insert_muts) == 1
    assert insert_muts[0].memory.status == MemoryStatus.ACTIVE
    assert insert_muts[0].memory.value_norm == "3.12"

    # 3. Add edge: SUPERSEDES ($new -> 1)
    edge_muts = [m for m in decision.mutations if isinstance(m, AddEdge)]
    assert len(edge_muts) == 1
    assert edge_muts[0].src == "$new"
    assert edge_muts[0].dst == 1
    assert edge_muts[0].type == EdgeType.SUPERSEDES


def test_r_replace_threshold_boundaries(
    version_attr: AttributeDef,
) -> None:
    """Exact threshold boundary tests for replace_margin: exact match replaces, below disputes."""
    cfg = load_engine_config()
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    perfect_source = Source(id="root", name="Root", type="admin", reliability=1.0)
    active_mem = Memory(
        id=1,
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t1,
        recorded_at=t1,
        status=MemoryStatus.ACTIVE,
        confidence=1.0,
        message_id=101,
        source_id="root",
    )

    # replace_margin is 0.80.
    # At exact threshold: s_new = 0.80 -> new_score >= old_score * margin -> R-REPLACE
    rc_boundary = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.80,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t2,
        source=perfect_source,
        message_id=102,
    )

    dec_boundary = resolve(
        rc=rc_boundary,
        chain=[active_mem],
        schema=version_attr,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t2,
    )
    assert dec_boundary.action == Action.REPLACE
    assert dec_boundary.rule_id == "R-REPLACE"

    # Just below threshold: s_new = 0.799 -> new_score < old_score * margin -> R-DISPUTE
    rc_just_below = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.799,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t2,
        source=perfect_source,
        message_id=103,
    )

    dec_just_below = resolve(
        rc=rc_just_below,
        chain=[active_mem],
        schema=version_attr,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t2,
    )
    assert dec_just_below.action == Action.DISPUTE
    assert dec_just_below.rule_id == "R-DISPUTE"


# ---------------------------------------------------------------------------
# Pure Engine Tests: Low-Trust Rejection (R-REJECT)
# ---------------------------------------------------------------------------


def test_r_reject_low_trust_conflicting_claim(
    version_attr: AttributeDef,
    high_trust_source: Source,
    low_trust_source: Source,
) -> None:
    """Weaker claim below reject_ratio (new_score < old_score * reject_ratio) produces R-REJECT."""
    cfg = load_engine_config()
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    # Active memory: confidence 1.0 -> old_score = 1.0
    active_mem = Memory(
        id=1,
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t1,
        recorded_at=t1,
        status=MemoryStatus.ACTIVE,
        confidence=1.0,
        message_id=101,
        source_id=high_trust_source.id,
    )

    # Low trust claim: intern (0.30), extract 0.80 -> s_new = 0.24
    # reject_threshold = 1.0 * 0.50 = 0.50. Since 0.24 < 0.50 -> R-REJECT
    rc_low = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.80,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t2,
        source=low_trust_source,
        message_id=102,
    )

    decision = resolve(
        rc=rc_low,
        chain=[active_mem],
        schema=version_attr,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t2,
    )

    assert decision.action == Action.REJECT
    assert decision.rule_id == "R-REJECT"
    assert "below reject threshold" in decision.explanation
    assert decision.scores["new_score"] == pytest.approx(0.24, rel=1e-3)
    assert decision.scores["old_score"] == pytest.approx(1.0, rel=1e-3)

    # Existing active memory must NOT be modified
    update_muts = [m for m in decision.mutations if isinstance(m, UpdateMemory)]
    assert len(update_muts) == 0

    # Rejected memory stored with status REJECTED
    insert_muts = [m for m in decision.mutations if isinstance(m, InsertMemory)]
    assert len(insert_muts) == 1
    assert insert_muts[0].memory.status == MemoryStatus.REJECTED
    assert insert_muts[0].memory.value_norm == "3.12"

    # Edge CONTRADICTS from $new to live memory
    edge_muts = [m for m in decision.mutations if isinstance(m, AddEdge)]
    assert len(edge_muts) == 1
    assert edge_muts[0].src == "$new"
    assert edge_muts[0].dst == 1
    assert edge_muts[0].type == EdgeType.CONTRADICTS


def test_r_reject_threshold_boundary(version_attr: AttributeDef) -> None:
    """Boundary test: new_score == old_score * reject_ratio is DISPUTE, not REJECT."""
    cfg = load_engine_config()
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    perfect_source = Source(id="root", name="Root", type="admin", reliability=1.0)
    active_mem = Memory(
        id=1,
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t1,
        recorded_at=t1,
        status=MemoryStatus.ACTIVE,
        confidence=1.0,
        message_id=101,
        source_id="root",
    )

    # reject_ratio = 0.50.
    # At exact threshold: s_new = 0.50 -> not strictly less than 0.50 -> R-DISPUTE
    rc_boundary = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.50,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t2,
        source=perfect_source,
        message_id=102,
    )

    dec_boundary = resolve(
        rc=rc_boundary,
        chain=[active_mem],
        schema=version_attr,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t2,
    )
    assert dec_boundary.action == Action.DISPUTE
    assert dec_boundary.rule_id == "R-DISPUTE"

    # Just below threshold: s_new = 0.499 -> strictly less than 0.50 -> R-REJECT
    rc_just_below = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.499,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t2,
        source=perfect_source,
        message_id=103,
    )

    dec_just_below = resolve(
        rc=rc_just_below,
        chain=[active_mem],
        schema=version_attr,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t2,
    )
    assert dec_just_below.action == Action.REJECT
    assert dec_just_below.rule_id == "R-REJECT"


# ---------------------------------------------------------------------------
# Pure Engine Tests: Dispute Guards (R-DISPUTE)
# ---------------------------------------------------------------------------


def test_r_dispute_middle_band_marks_both_disputed(
    version_attr: AttributeDef,
    med_trust_source: Source,
) -> None:
    """Score in middle band (old * reject_ratio <= new_score < old * margin) -> R-DISPUTE."""
    cfg = load_engine_config()
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    # Active memory: confidence 0.60 -> old_score = 0.60
    # reject_threshold = 0.60 * 0.50 = 0.30
    # replace_threshold = 0.60 * 0.80 = 0.48
    active_mem = Memory(
        id=1,
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t1,
        recorded_at=t1,
        status=MemoryStatus.ACTIVE,
        confidence=0.60,
        message_id=101,
        source_id=med_trust_source.id,
    )

    # Middle band claim: extract 0.60, med_trust 0.60 -> s_new = 0.36
    # 0.30 <= 0.36 < 0.48 -> R-DISPUTE
    rc_mid = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.60,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t2,
        source=med_trust_source,
        message_id=102,
    )

    decision = resolve(
        rc=rc_mid,
        chain=[active_mem],
        schema=version_attr,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t2,
    )

    assert decision.action == Action.DISPUTE
    assert decision.rule_id == "R-DISPUTE"
    assert decision.affected_memory_ids == [1]

    # Existing memory updated to DISPUTED
    update_muts = [m for m in decision.mutations if isinstance(m, UpdateMemory)]
    assert len(update_muts) == 1
    assert update_muts[0].memory_id == 1
    assert update_muts[0].fields["status"] == MemoryStatus.DISPUTED

    # New memory inserted as DISPUTED
    insert_muts = [m for m in decision.mutations if isinstance(m, InsertMemory)]
    assert len(insert_muts) == 1
    assert insert_muts[0].memory.status == MemoryStatus.DISPUTED
    assert insert_muts[0].memory.value_norm == "3.12"

    # Edge CONTRADICTS created between new memory and old memory
    edge_muts = [m for m in decision.mutations if isinstance(m, AddEdge)]
    assert len(edge_muts) == 1
    assert edge_muts[0].src == "$new"
    assert edge_muts[0].dst == 1
    assert edge_muts[0].type == EdgeType.CONTRADICTS


# ---------------------------------------------------------------------------
# Pure Engine Tests: Revert Protection (revert_margin)
# ---------------------------------------------------------------------------


def test_revert_guard_detects_superseded_history_and_applies_revert_margin(
    version_attr: AttributeDef,
    high_trust_source: Source,
) -> None:
    """When conflicting claim matches superseded HISTORICAL value, revert_margin is used."""
    cfg = load_engine_config()
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)
    t3 = datetime(2026, 6, 3, 10, 0, tzinfo=UTC)

    # 1. Historical memory for 3.10, superseded at t2
    hist_mem = Memory(
        id=1,
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t1,
        valid_to=t2,
        recorded_at=t1,
        closed_at=t2,
        status=MemoryStatus.HISTORICAL,
        confidence=1.0,
        message_id=101,
        source_id=high_trust_source.id,
    )

    # 2. Currently active memory for 3.12 (score = 0.90)
    perfect_source = Source(id="root", name="Root", type="admin", reliability=1.0)
    active_mem = Memory(
        id=2,
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t2,
        recorded_at=t2,
        status=MemoryStatus.ACTIVE,
        confidence=0.90,
        message_id=102,
        source_id="root",
    )

    # 3. New claim attempting to revert back to "3.10"
    # Old score = 0.90.
    # Normal replace_margin = 0.80 -> 0.90 * 0.80 = 0.72.
    # Revert margin = 1.00 -> 0.90 * 1.00 = 0.90.
    # Claim score = 0.85.
    # Under normal margin, 0.85 >= 0.72 would REPLACE.
    # Under revert margin, 0.85 < 0.90, so replacement is BLOCKED -> R-DISPUTE!
    rc_revert_blocked = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.10",
            extract_confidence=0.85,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t3,
        source=perfect_source,
        message_id=103,
    )

    dec_blocked = resolve(
        rc=rc_revert_blocked,
        chain=[hist_mem, active_mem],
        schema=version_attr,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t3,
    )

    assert dec_blocked.action == Action.DISPUTE
    assert dec_blocked.rule_id == "R-DISPUTE"
    assert "revert guard" in dec_blocked.explanation.lower()
    assert dec_blocked.scores["margin"] == 1.00

    # If incoming claim is strong enough to satisfy revert_margin (s_new = 0.95 >= 0.90 * 1.00):
    rc_revert_allowed = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.10",
            extract_confidence=0.95,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t3,
        source=perfect_source,
        message_id=104,
    )

    dec_allowed = resolve(
        rc=rc_revert_allowed,
        chain=[hist_mem, active_mem],
        schema=version_attr,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t3,
    )

    assert dec_allowed.action == Action.REPLACE
    assert dec_allowed.rule_id == "R-REPLACE"
    assert "revert" in dec_allowed.explanation.lower()


# ---------------------------------------------------------------------------
# Pure Engine Tests: Conflict with History Only
# ---------------------------------------------------------------------------


def test_conflict_with_history_only_never_rewrites_history(
    version_attr: AttributeDef,
    high_trust_source: Source,
    low_trust_source: Source,
) -> None:
    """When a new claim conflicts within a closed historical window and no live memory covers it."""
    cfg = load_engine_config()
    t1 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 2, 1, 10, 0, tzinfo=UTC)
    t_inside = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)

    hist_mem = Memory(
        id=1,
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t1,
        valid_to=t2,
        recorded_at=t1,
        closed_at=t2,
        status=MemoryStatus.HISTORICAL,
        confidence=1.0,
        message_id=101,
        source_id=high_trust_source.id,
    )

    # 1. Weak claim -> R-REJECT (0.30 * 0.60 = 0.18 < 1.0 * 0.50 = 0.50)
    rc_weak = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.60,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t_inside,
        source=low_trust_source,
        message_id=102,
    )

    dec_weak = resolve(
        rc=rc_weak,
        chain=[hist_mem],
        schema=version_attr,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t2,
    )
    assert dec_weak.action == Action.REJECT
    assert dec_weak.rule_id == "R-REJECT"

    # 2. Moderate/strong claim -> R-DISPUTE (never R-REPLACE)
    rc_strong = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.95,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t_inside,
        source=high_trust_source,  # 0.95 * 0.95 = 0.9025 >= 0.50
        message_id=103,
    )

    dec_strong = resolve(
        rc=rc_strong,
        chain=[hist_mem],
        schema=version_attr,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t2,
    )
    assert dec_strong.action == Action.DISPUTE
    assert dec_strong.rule_id == "R-DISPUTE"


# ---------------------------------------------------------------------------
# Pure Engine Tests: Ablation Flags
# ---------------------------------------------------------------------------


def test_ablation_use_trust_flag_disables_trust_differences(
    version_attr: AttributeDef,
    low_trust_source: Source,
) -> None:
    """When ablation.use_trust = False, scores evaluate to 1.0 (capped at max_confidence)."""
    cfg = load_engine_config()
    cfg.ablation.use_trust = False

    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    active_mem = Memory(
        id=1,
        entity_id=1,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        valid_from=t1,
        recorded_at=t1,
        status=MemoryStatus.ACTIVE,
        confidence=1.0,
        message_id=101,
        source_id="sysadmin_priya",
    )

    # Normally low_trust_source would be rejected (0.30 * 0.80 = 0.24 < 0.50).
    # With use_trust = False, scores are 1.0 (capped at 0.99), newer claim replaces memory.
    rc = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="python_version",
            value="3.12",
            extract_confidence=0.80,
        ),
        entity_id=1,
        attribute="python_version",
        value_norm="3.12",
        value_display="Python 3.12",
        context_key="",
        valid_from=t2,
        source=low_trust_source,
        message_id=102,
    )

    decision = resolve(
        rc=rc,
        chain=[active_mem],
        schema=version_attr,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t2,
    )
    assert decision.action == Action.REPLACE
    assert decision.rule_id == "R-REPLACE"
    assert decision.scores["new_score"] == pytest.approx(cfg.thresholds.max_confidence, rel=1e-3)
    assert decision.scores["old_score"] == 1.0


def test_temporal_decay_affects_old_memory_score(
    high_trust_source: Source,
    med_trust_source: Source,
) -> None:
    """When an attribute has half_life_days, older memory effective score decays with time."""
    decay_attr = AttributeDef(
        attribute="cpu_governor",
        description="CPU frequency scaling governor",
        cardinality=Cardinality.SINGLE,
        value_type=ValueType.TEXT,
        half_life_days=30.0,
    )
    cfg = load_engine_config()
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = t1 + timedelta(days=30)  # Exactly 1 half-life later

    # Old memory at t1 with confidence 1.0
    old_mem = Memory(
        id=1,
        entity_id=1,
        attribute="cpu_governor",
        value_norm="performance",
        value_display="performance",
        context_key="",
        valid_from=t1,
        recorded_at=t1,
        status=MemoryStatus.ACTIVE,
        confidence=1.0,
        message_id=101,
        source_id=high_trust_source.id,
    )

    # Claim at t2: score = 0.60 * 0.75 = 0.45.
    # Without decay: old_score = 1.0, replace_threshold = 0.80 (0.45 would reject).
    # With 1 half-life decay: decay = 0.5, old_score = 0.50.
    # replace_threshold = 0.50 * 0.80 = 0.40.
    # Since 0.45 >= 0.40, the decayed memory is replaced!
    rc = ResolvedClaim(
        claim=Claim(
            subject="Server A",
            attribute="cpu_governor",
            value="powersave",
            extract_confidence=0.75,
        ),
        entity_id=1,
        attribute="cpu_governor",
        value_norm="powersave",
        value_display="powersave",
        context_key="",
        valid_from=t2,
        source=med_trust_source,
        message_id=102,
    )

    dec_with_decay = resolve(
        rc=rc,
        chain=[old_mem],
        schema=decay_attr,
        cfg=cfg,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t2,
    )
    assert dec_with_decay.action == Action.REPLACE
    assert dec_with_decay.rule_id == "R-REPLACE"
    assert dec_with_decay.scores["old_score"] == pytest.approx(0.50, rel=1e-3)

    # When use_temporal = False, decay is not applied (old_score remains 1.0)
    cfg_no_temp = load_engine_config()
    cfg_no_temp.ablation.use_temporal = False
    dec_no_decay = resolve(
        rc=rc,
        chain=[old_mem],
        schema=decay_attr,
        cfg=cfg_no_temp,
        classify=lambda _rc, _m: Relation.CONFLICTS,
        dep_targets=[],
        recorded_at=t2,
    )
    assert dec_no_decay.action == Action.REJECT
    assert dec_no_decay.rule_id == "R-REJECT"
    assert dec_no_decay.scores["old_score"] == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Service-Level Integration Tests (End-to-End Ingestion, Database, Retrieval)
# ---------------------------------------------------------------------------


def test_service_trust_replacement_e2e(service: MemoryService) -> None:
    """End-to-end trust-based replacement closes window and creates supersedes edge."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    # 1. Ingest initial claim
    msg1 = IngestMessage(
        source_id="dev_chat",
        text="Server A runs Python 3.10",
        event_time=t1,
    )
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=0.80,
    )
    report1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    assert report1.decisions[0].action == Action.INSERT
    mem1_id = report1.decisions[0].memory_id

    # 2. Ingest stronger conflicting claim
    msg2 = IngestMessage(
        source_id="sysadmin_priya",
        text="Upgraded Server A to Python 3.12",
        event_time=t2,
    )
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.95,
    )
    report2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    assert report2.decisions[0].action == Action.REPLACE
    assert report2.decisions[0].rule_id == "R-REPLACE"
    mem2_id = report2.decisions[0].memory_id
    assert mem2_id is not None

    # Current retrieval returns the new memory
    cur = service.get_current("Server A", "python_version", now=t2)
    assert cur is not None
    assert cur.value_norm == "3.12"
    assert cur.status == MemoryStatus.ACTIVE

    # Historical retrieval returns old memory at t1
    at_t1 = service.get_as_of("Server A", "python_version", valid_at=t1)
    assert at_t1 is not None
    assert at_t1.value_norm == "3.10"

    # Check edges in database
    edges = service.repo.get_edges_for_memory(mem2_id, direction="src")
    assert any(e.type == EdgeType.SUPERSEDES and e.dst_memory_id == mem1_id for e in edges)


def test_service_low_trust_rejection_e2e(service: MemoryService) -> None:
    """Low-trust conflicting claim is rejected, stored as REJECTED, and excluded from current."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    # 1. High trust memory established
    msg1 = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A runs Python 3.10",
        event_time=t1,
    )
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=0.95,
    )
    report1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    assert report1.decisions[0].action == Action.INSERT
    mem1_id = report1.decisions[0].memory_id

    # 2. Low trust claim ingested
    msg2 = IngestMessage(
        source_id="intern_dan",  # reliability = 0.30 -> score = 0.30 * 0.70 = 0.21 < 0.9025 * 0.50
        text="Server A has Python 3.12 installed",
        event_time=t2,
    )
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.70,
    )
    report2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    assert report2.decisions[0].action == Action.REJECT
    assert report2.decisions[0].rule_id == "R-REJECT"
    rejected_mem_id = report2.decisions[0].memory_id
    assert rejected_mem_id is not None

    # Verify rejected memory is persisted in repository with REJECTED status
    rejected_mem = service.repo.get_memory(rejected_mem_id)
    assert rejected_mem is not None
    assert rejected_mem.status == MemoryStatus.REJECTED

    # Current retrieval still returns original memory (NOT rejected one)
    cur = service.get_current("Server A", "python_version", now=t2)
    assert cur is not None
    assert cur.id == mem1_id
    assert cur.value_norm == "3.10"
    assert cur.status == MemoryStatus.ACTIVE

    # Contradicts edge exists from rejected memory to active memory
    edges = service.repo.get_edges_for_memory(rejected_mem_id, direction="src")
    assert any(e.type == EdgeType.CONTRADICTS and e.dst_memory_id == mem1_id for e in edges)


def test_service_dispute_flow_e2e(service: MemoryService) -> None:
    """Middle-band conflict marks both existing and new memory as DISPUTED, visible in current."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    # 1. Dev chat initial claim (0.60 * 1.0 = 0.60)
    msg1 = IngestMessage(
        source_id="dev_chat",
        text="Server A runs Python 3.10",
        event_time=t1,
    )
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=1.0,
    )
    report1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    mem1_id = report1.decisions[0].memory_id

    # 2. Conflicting claim in middle band (0.60 * 0.60 = 0.36)
    # reject_threshold = 0.60 * 0.50 = 0.30
    # replace_threshold = 0.60 * 0.80 = 0.48
    # Since 0.30 <= 0.36 < 0.48 -> R-DISPUTE
    msg2 = IngestMessage(
        source_id="dev_chat",
        text="Actually Server A has Python 3.12",
        event_time=t2,
    )
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.60,
    )
    report2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    assert report2.decisions[0].action == Action.DISPUTE
    assert report2.decisions[0].rule_id == "R-DISPUTE"
    mem2_id = report2.decisions[0].memory_id

    # Check database statuses
    mem1 = service.repo.get_memory(mem1_id)
    mem2 = service.repo.get_memory(mem2_id)
    assert mem1.status == MemoryStatus.DISPUTED
    assert mem2.status == MemoryStatus.DISPUTED

    # Current retrieval includes disputed memories per specification
    res = service.current("Server A", "python_version", now=t2)
    val_norms = {item.memory.value_norm for item in res.items}
    assert "3.10" in val_norms
    assert "3.12" in val_norms


def test_service_revert_guard_e2e(service: MemoryService) -> None:
    """Reverting to a superseded historical value requires revert_margin (1.00)."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)
    t3 = datetime(2026, 6, 3, 10, 0, tzinfo=UTC)
    t4 = datetime(2026, 6, 4, 10, 0, tzinfo=UTC)

    # 1. Priya sets python_version = 3.10
    service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya",
            text="Initial Server A install is Python 3.10",
            event_time=t1,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                extract_confidence=0.95,
            )
        ],
        recorded_at=t1,
    )

    # 2. Priya upgrades python_version = 3.12 (replaces 3.10)
    service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya",
            text="Upgraded Server A to Python 3.12",
            event_time=t2,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                extract_confidence=0.95,
            )
        ],
        recorded_at=t2,
    )
    cur = service.get_current("Server A", "python_version", now=t2)
    assert cur.value_norm == "3.12"

    # 3. Dev chat claims python_version is "3.10" again (revert attempt!)
    # Dev chat score = 0.60 * 0.95 = 0.57.
    # Active memory score = 0.95 * 0.95 = 0.9025.
    # Revert margin = 1.00. 0.57 < 0.9025 * 1.00 -> blocked!
    # Also 0.57 >= 0.9025 * 0.50 = 0.45125, so it lands in R-DISPUTE.
    rep3 = service.ingest_claims(
        IngestMessage(
            source_id="dev_chat",
            text="I thought Server A is on Python 3.10",
            event_time=t3,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                extract_confidence=0.95,
            )
        ],
        recorded_at=t3,
    )
    assert rep3.decisions[0].action == Action.DISPUTE
    assert rep3.decisions[0].rule_id == "R-DISPUTE"
    assert "revert guard" in rep3.decisions[0].explanation.lower()

    # 4. Priya confirms the revert with high trust (score = 0.95 * 1.0 = 0.95 >= 0.9025 * 1.00)
    rep4 = service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya",
            text="Rolled back Server A to Python 3.10 due to bug",
            event_time=t4,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                extract_confidence=1.00,
            )
        ],
        recorded_at=t4,
    )
    assert rep4.decisions[0].action == Action.REPLACE
    assert rep4.decisions[0].rule_id == "R-REPLACE"
    assert "revert" in rep4.decisions[0].explanation.lower()


def test_service_min_extract_confidence_skip_e2e(service: MemoryService) -> None:
    """Claims below min_extract_confidence are skipped (R-SKIP-LOWCONF), creating no memory."""
    t = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)

    msg = IngestMessage(
        source_id="sysadmin_priya",
        text="Vaguely heard Server A might run Python 3.12",
        event_time=t,
    )
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        extract_confidence=0.30,  # Below min_extract_confidence (0.40)
    )

    report = service.ingest_claims(msg, [claim], recorded_at=t)
    assert len(report.decisions) == 1
    dec = report.decisions[0]
    assert dec.action == Action.SKIP
    assert dec.rule_id == "R-SKIP-LOWCONF"
    assert dec.memory_id is None
    assert "min_extract_confidence" in dec.explanation

    # No memory exists in repository
    cur = service.get_current("Server A", "python_version", now=t)
    assert cur is None


# ---------------------------------------------------------------------------
# Dispute Safety for get_current()
# ---------------------------------------------------------------------------


def test_get_current_dispute_safety(service: MemoryService) -> None:
    """get_current() returns a Memory only when there is exactly one unambiguous ACTIVE result."""
    t0 = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    # 1. No matching memory -> returns None
    assert service.get_current("Server A", "python_version", now=t0) is None

    # 2. Exactly one ACTIVE memory -> returns that memory
    service.ingest_claims(
        IngestMessage(
            source_id="dev_chat",
            text="Server A runs Python 3.10",
            event_time=t1,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                extract_confidence=1.0,
            )
        ],
        recorded_at=t1,
    )
    active_cur = service.get_current("Server A", "python_version", now=t1)
    assert active_cur is not None
    assert active_cur.value_norm == "3.10"
    assert active_cur.status == MemoryStatus.ACTIVE

    # 3. Two conflicting memories in DISPUTE -> get_current() returns None
    service.ingest_claims(
        IngestMessage(
            source_id="dev_chat",
            text="Actually Server A runs Python 3.12",
            event_time=t2,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                extract_confidence=0.60,
            )
        ],
        recorded_at=t2,
    )
    assert service.get_current("Server A", "python_version", now=t2) is None

    # Detailed query still returns both disputed memories
    detailed = service.current("Server A", "python_version", now=t2)
    assert len(detailed.items) == 2
    assert all(item.memory.status == MemoryStatus.DISPUTED for item in detailed.items)
    disputed_values = {item.memory.value_norm for item in detailed.items}
    assert disputed_values == {"3.10", "3.12"}

    # 4. Single disputed memory (e.g. after one side was modified) -> returns None
    service.repo.update_memory(detailed.items[0].memory.id, {"status": MemoryStatus.REJECTED})
    # Now only 1 memory is visible in validity window, but its status is DISPUTED
    single_disp = service.current("Server A", "python_version", now=t2)
    assert len(single_disp.items) == 1
    assert single_disp.items[0].memory.status == MemoryStatus.DISPUTED
    assert service.get_current("Server A", "python_version", now=t2) is None


# ---------------------------------------------------------------------------
# Supersession-Chain-Aware Revert Protection Tests
# ---------------------------------------------------------------------------


def test_multi_step_supersession_revert_e2e(service: MemoryService) -> None:
    """Reverting to an older ancestor across replacements (A -> B -> C -> A) triggers guard."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)
    t3 = datetime(2026, 6, 3, 10, 0, tzinfo=UTC)
    t4 = datetime(2026, 6, 4, 10, 0, tzinfo=UTC)

    # 1. Priya installs Python 3.10
    service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya",
            text="Server A has Python 3.10",
            event_time=t1,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                extract_confidence=0.95,
            )
        ],
        recorded_at=t1,
    )

    # 2. Priya upgrades to Python 3.11 (supersedes 3.10)
    service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya",
            text="Server A upgraded to Python 3.11",
            event_time=t2,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.11",
                extract_confidence=0.95,
            )
        ],
        recorded_at=t2,
    )

    # 3. Priya upgrades to Python 3.12 (supersedes 3.11)
    service.ingest_claims(
        IngestMessage(
            source_id="sysadmin_priya",
            text="Server A upgraded to Python 3.12",
            event_time=t3,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                extract_confidence=0.95,
            )
        ],
        recorded_at=t3,
    )
    cur = service.get_current("Server A", "python_version", now=t3)
    assert cur is not None
    assert cur.value_norm == "3.12"

    # 4. Dev chat claims Python is 3.10 (multi-step revert: 3.12 -> 3.11 -> 3.10)
    # Dev chat score = 0.60 * 0.95 = 0.57.
    # Active memory score = 0.95 * 0.95 = 0.9025.
    # Revert margin = 1.00 -> 0.57 < 0.9025 * 1.00 -> BLOCKED by revert guard!
    rep4 = service.ingest_claims(
        IngestMessage(
            source_id="dev_chat",
            text="Server A is running Python 3.10 again",
            event_time=t4,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                extract_confidence=0.95,
            )
        ],
        recorded_at=t4,
    )
    assert rep4.decisions[0].action == Action.DISPUTE
    assert rep4.decisions[0].rule_id == "R-DISPUTE"
    assert "revert guard" in rep4.decisions[0].explanation.lower()
    assert rep4.decisions[0].scores["margin"] == 1.00


def test_unrelated_historical_backfill_does_not_trigger_revert_guard_e2e(
    service: MemoryService,
) -> None:
    """An unrelated historical memory without a supersedes edge does not trigger revert guard."""
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t3 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)

    # Directly insert an unrelated historical memory for "3.10" (e.g. from an ancient backfill)
    # with NO supersedes edge connecting it to the active memory
    from memorymesh.models import NewEvidence, NewMemory

    entity_id = service._resolve_and_persist_entity("Server A", t0)
    msg_id = service.repo.insert_message(
        source_id="sysadmin_priya",
        text="Ancient backfill: Server A had Python 3.10",
        event_time=t0,
    )
    backfill_mem = NewMemory(
        entity_id=entity_id,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        message_id=msg_id,
        source_id="sysadmin_priya",
        valid_from=t0,
        valid_to=t1,
        closed_at=t1,
        status=MemoryStatus.HISTORICAL,
        confidence=0.80,
    )
    backfill_ev = NewEvidence(
        source_id="sysadmin_priya",
        message_id=msg_id,
        strength=0.80,
        observed_at=t1,
    )
    backfill_id = service.repo.insert_memory(backfill_mem, recorded_at=t1)
    service.repo.insert_evidence(backfill_id, backfill_ev)

    # Ingest active memory: 3.12 with med_trust (0.60 * 0.80 = 0.48)
    service.ingest_claims(
        IngestMessage(
            source_id="dev_chat",
            text="Server A runs Python 3.12",
            event_time=t2,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                extract_confidence=0.80,
            )
        ],
        recorded_at=t2,
    )
    cur = service.get_current("Server A", "python_version", now=t2)
    assert cur is not None
    assert cur.value_norm == "3.12"

    # Ingest new claim: 3.10 with score 0.42 (chat 0.60 * 0.70 = 0.42)
    # Active score = 0.48.
    # Normal replace_margin = 0.80 -> 0.48 * 0.80 = 0.384.
    # If revert_margin were mistakenly applied (1.00), 0.42 < 0.48 would BLOCK replacement.
    # Because 3.10 was NEVER superseded by the active memory, normal replace_margin applies!
    # 0.42 >= 0.384 -> REPLACEMENT SUCCEEDS!
    rep = service.ingest_claims(
        IngestMessage(
            source_id="dev_chat",
            text="Reconfigured Server A to Python 3.10",
            event_time=t3,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                extract_confidence=0.70,
            )
        ],
        recorded_at=t3,
    )
    assert rep.decisions[0].action == Action.REPLACE
    assert rep.decisions[0].rule_id == "R-REPLACE"
    assert rep.decisions[0].scores["margin"] == 0.80
    assert "replace_margin" in rep.decisions[0].explanation


def test_retracted_memory_does_not_trigger_revert_guard_e2e(
    service: MemoryService,
) -> None:
    """A retracted memory without supersedes edges does not count as a superseded ancestor."""
    t1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 10, 0, tzinfo=UTC)
    t3 = datetime(2026, 6, 3, 10, 0, tzinfo=UTC)

    # Insert a retracted historical memory directly (no supersedes edge)
    from memorymesh.models import NewMemory

    entity_id = service._resolve_and_persist_entity("Server A", t1)
    msg_id = service.repo.insert_message(
        source_id="sysadmin_priya",
        text="Retracted: Server A had Python 3.10",
        event_time=t1,
    )
    retracted_mem = NewMemory(
        entity_id=entity_id,
        attribute="python_version",
        value_norm="3.10",
        value_display="Python 3.10",
        context_key="",
        message_id=msg_id,
        source_id="sysadmin_priya",
        valid_from=t1,
        valid_to=t2,
        closed_at=t2,
        status=MemoryStatus.HISTORICAL,
        confidence=0.80,
    )
    service.repo.insert_memory(retracted_mem, recorded_at=t2)

    # Ingest active memory: 3.12 (score = 0.60 * 0.80 = 0.48)
    service.ingest_claims(
        IngestMessage(
            source_id="dev_chat",
            text="Server A has Python 3.12",
            event_time=t2,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.12",
                extract_confidence=0.80,
            )
        ],
        recorded_at=t2,
    )

    # Claim for 3.10 with score 0.42 >= 0.48 * 0.80 = 0.384
    # Replaces normally under replace_margin = 0.80
    rep = service.ingest_claims(
        IngestMessage(
            source_id="dev_chat",
            text="Server A has Python 3.10",
            event_time=t3,
        ),
        [
            Claim(
                subject="Server A",
                attribute="python_version",
                value="3.10",
                extract_confidence=0.70,
            )
        ],
        recorded_at=t3,
    )
    assert rep.decisions[0].action == Action.REPLACE
    assert rep.decisions[0].rule_id == "R-REPLACE"
    assert rep.decisions[0].scores["margin"] == 0.80
