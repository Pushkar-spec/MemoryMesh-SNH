"""P6.2 Regression and verification test suite.

Covers:
1. Retraction via service API (service.retract and service.retract_memory alias).
2. Retraction via claim ingestion (Polarity.RETRACT producing R-RETRACT / SKIP / REJECT).
3. Lifecycle enforcement (rejecting invalid status transitions).
4. Temporal window constraints (valid_to > valid_from).
5. Evidence preservation during and after retraction.
6. Downstream dependency propagation on retraction (R-PROPAGATE).
7. Bitemporal belief-time reconstruction and queries (known_at, valid_at, as-of boundaries).
8. Multi-valued attribute retraction (test_r_retract_closes_multi).
9. Idempotency across operations (retraction, duplicate claims, edge insertion, propagation).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone, tzinfo

import pytest

from memorymesh.config import load_app_config
from memorymesh.db import init_db
from memorymesh.errors import (
    InvalidLifecycleTransitionError,
    InvalidValidityWindowError,
    MemoryNotFoundError,
)
from memorymesh.models import (
    Action,
    AttributeDef,
    AttrRef,
    Cardinality,
    Claim,
    DecisionLogEntry,
    EdgeType,
    IngestMessage,
    MemoryStatus,
    Polarity,
    Source,
    ValueType,
)
from memorymesh.repository import Repository
from memorymesh.retrieval.retrieve import retrieve_as_of, retrieve_valid_memories
from memorymesh.service import MemoryService


@pytest.fixture
def service() -> MemoryService:
    """Create MemoryService instance backed by an initialized in-memory database."""
    conn = init_db(":memory:")
    repo = Repository(conn)

    # Seed initial test sources
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
            id="monitoring_agent",
            name="Infra Monitor",
            type="agent",
            reliability=0.90,
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
            id="untrusted_feed",
            name="Untrusted Feed",
            type="tool",
            reliability=0.20,
        )
    )

    # Seed schemas
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="python_version",
            description="Installed Python runtime version",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
        )
    )
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="os_version",
            description="Operating system release",
            cardinality=Cardinality.SINGLE,
            value_type=ValueType.VERSION,
        )
    )
    repo.insert_attribute_schema(
        AttributeDef(
            attribute="runs_service",
            description="Services currently executed by host",
            cardinality=Cardinality.MULTI,
            value_type=ValueType.TEXT,
        )
    )

    repo.insert_entity(canonical_name="Server A", entity_type="server")

    cfg = load_app_config()
    return MemoryService(db_target=repo, cfg=cfg)


# =============================================================================
# 1. Service Retraction Tests
# =============================================================================


def test_service_retract_active_memory(service: MemoryService) -> None:
    """Retracting an active memory transitions it to HISTORICAL with valid_to and closed_at."""
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    msg = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A has Python 3.10",
        event_time=t1,
    )
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=0.95,
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t1)
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    # Retract the memory
    log = service.retract(
        memory_id=m_id,
        reason="Server A Python decommissioned",
        valid_to=t2,
        recorded_at=t2,
    )

    assert log.rule_id == "R-RETRACT"
    assert log.action == Action.RETRACT
    assert log.memory_id == m_id
    assert m_id in log.affected_memory_ids

    # Verify memory state in repository
    m = service.repo.get_memory(m_id)
    assert m is not None
    assert m.status == MemoryStatus.HISTORICAL
    assert m.valid_to == t2
    assert m.closed_at == t2
    assert m.value_norm == "3.10"


def test_service_retract_memory_alias(service: MemoryService) -> None:
    """service.retract_memory is an exact alias of service.retract."""
    assert service.retract_memory == service.retract

    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    msg = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A has Python 3.10",
        event_time=t1,
    )
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=0.95,
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t1)
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    log = service.retract_memory(m_id, valid_to=t2, recorded_at=t2)
    assert log.rule_id == "R-RETRACT"
    m = service.repo.get_memory(m_id)
    assert m is not None
    assert m.status == MemoryStatus.HISTORICAL


def test_service_retract_disputed_memory(service: MemoryService) -> None:
    """Retracting a memory currently in DISPUTED status transitions it to HISTORICAL."""
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    t3 = datetime(2026, 3, 5, 10, 0, tzinfo=UTC)

    # Ingest from source 1
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.90
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # Ingest conflicting from source 2 at identical event_time -> DISPUTE
    msg2 = IngestMessage(
        source_id="monitoring_agent", text="Server A has Python 3.12", event_time=t1
    )
    claim2 = Claim(
        subject="Server A", attribute="python_version", value="3.12", extract_confidence=0.90
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    assert rep2.decisions[0].rule_id == "R-DISPUTE"

    m1 = service.repo.get_memory(m1_id)
    assert m1 is not None and m1.status == MemoryStatus.DISPUTED

    # Retract m1
    log = service.retract(m1_id, valid_to=t3, recorded_at=t3)
    assert log.rule_id == "R-RETRACT"
    m1_after = service.repo.get_memory(m1_id)
    assert m1_after is not None
    assert m1_after.status == MemoryStatus.HISTORICAL
    assert m1_after.valid_to == t3


def test_service_retract_nonexistent_raises(service: MemoryService) -> None:
    """Retracting a non-existent memory ID raises MemoryNotFoundError."""
    t = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    with pytest.raises(MemoryNotFoundError, match="Memory M99999 not found"):
        service.retract(99999, valid_to=t, recorded_at=t)


def test_service_retract_invalid_lifecycle_rejected(service: MemoryService) -> None:
    """Attempting to retract a REJECTED memory raises InvalidLifecycleTransitionError."""
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)

    # Ingest strong claim
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
    )
    service.ingest_claims(msg1, [claim1], recorded_at=t1)

    # Ingest very weak conflicting claim -> R-REJECT
    msg2 = IngestMessage(source_id="untrusted_feed", text="Server A has Python 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A", attribute="python_version", value="3.12", extract_confidence=0.50
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    assert rep2.decisions[0].rule_id == "R-REJECT"
    rejected_id = rep2.decisions[0].memory_id
    assert rejected_id is not None

    with pytest.raises(InvalidLifecycleTransitionError, match="status 'rejected'"):
        service.retract(rejected_id, valid_to=t2, recorded_at=t2)


def test_service_retract_invalid_lifecycle_superseded(service: MemoryService) -> None:
    """Attempting to retract an already-superseded HISTORICAL memory raises error."""
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    t3 = datetime(2026, 3, 3, 10, 0, tzinfo=UTC)

    # M1
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.90
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # M2 replaces M1 -> M1 becomes HISTORICAL via R-REPLACE
    msg2 = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A", attribute="python_version", value="3.12", extract_confidence=0.95
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    assert rep2.decisions[0].rule_id == "R-REPLACE"

    m1 = service.repo.get_memory(m1_id)
    assert m1 is not None and m1.status == MemoryStatus.HISTORICAL

    with pytest.raises(InvalidLifecycleTransitionError, match="already superseded or closed"):
        service.retract(m1_id, valid_to=t3, recorded_at=t3)


def test_service_retract_idempotency(service: MemoryService) -> None:
    """Repeated retraction calls on an already retracted memory return the existing log entry."""
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    msg = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t1)
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    # First retraction
    log1 = service.retract(m_id, reason="First retraction", valid_to=t2, recorded_at=t2)
    assert log1.rule_id == "R-RETRACT"

    initial_log_count = len(service.repo.get_decision_logs_for_memory(m_id))

    # Second retraction (identical call)
    log2 = service.retract(m_id, reason="Second retraction call", valid_to=t2, recorded_at=t2)
    assert log2.id == log1.id
    assert log2.rule_id == "R-RETRACT"

    # Verify no duplicate decision log entry was inserted
    after_log_count = len(service.repo.get_decision_logs_for_memory(m_id))
    assert after_log_count == initial_log_count


def test_service_retract_invalid_valid_to(service: MemoryService) -> None:
    """valid_to <= valid_from raises InvalidValidityWindowError."""
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t_invalid = datetime(2026, 2, 1, 10, 0, tzinfo=UTC)

    msg = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t1)
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    with pytest.raises(InvalidValidityWindowError, match="must be strictly greater"):
        service.retract(m_id, valid_to=t_invalid, recorded_at=t1)

    # Equal timestamps also invalid
    with pytest.raises(InvalidValidityWindowError, match="must be strictly greater"):
        service.retract(m_id, valid_to=t1, recorded_at=t1)


def test_service_retract_naive_datetime_raises(service: MemoryService) -> None:
    """Passing a timezone-naive datetime raises ValueError."""
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t1)
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    naive_dt = datetime(2026, 6, 1, 12, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        service.retract(m_id, recorded_at=naive_dt)

    with pytest.raises(ValueError, match="timezone-aware"):
        service.retract(m_id, valid_to=naive_dt)


def test_service_retract_evidence_preservation(service: MemoryService) -> None:
    """Retraction preserves all supporting evidence records intact."""
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    msg = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=0.95,
        span="Python 3.10",
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t1)
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    ev_before = service.get_memory_evidence(m_id)
    assert len(ev_before) >= 1
    assert ev_before[0].source_id == "sysadmin_priya"

    service.retract(m_id, valid_to=t2, recorded_at=t2)

    ev_after = service.get_memory_evidence(m_id)
    assert len(ev_after) == len(ev_before)
    assert ev_after[0].source_id == ev_before[0].source_id
    assert ev_after[0].span == ev_before[0].span


def test_service_retract_triggers_propagation(service: MemoryService) -> None:
    """Retracting an upstream root memory triggers R-PROPAGATE for active downstream dependents."""
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    t3 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    # Upstream: Server A python_version = 3.10
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    py_id = rep1.decisions[0].memory_id
    assert py_id is not None

    # Downstream: Server A runs_service = billing_app (depends on python_version)
    msg2 = IngestMessage(
        source_id="sysadmin_priya", text="Server A runs billing_app", event_time=t2
    )
    claim2 = Claim(
        subject="Server A",
        attribute="runs_service",
        value="billing_app",
        extract_confidence=0.90,
        depends_on=[AttrRef(entity="Server A", attribute="python_version")],
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    svc_id = rep2.decisions[0].memory_id
    assert svc_id is not None

    # Verify dependent is active and not flagged
    svc_mem = service.repo.get_memory(svc_id)
    assert svc_mem is not None
    assert svc_mem.needs_review is False

    # Retract upstream python_version
    service.retract(py_id, valid_to=t3, recorded_at=t3)

    # Downstream memory should now be flagged needs_review=True
    svc_after = service.repo.get_memory(svc_id)
    assert svc_after is not None
    assert svc_after.needs_review is True
    assert "retracted" in svc_after.review_reason


# =============================================================================
# 2. Claim-Level Retraction Tests (Polarity.RETRACT)
# =============================================================================


def test_claim_retract_single_valued(service: MemoryService) -> None:
    """Ingesting a claim with Polarity.RETRACT closes matching active memory via R-RETRACT."""
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m_id = rep1.decisions[0].memory_id
    assert m_id is not None

    # Ingest retraction claim
    msg2 = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A no longer has Python 3.10",
        event_time=t2,
    )
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        polarity=Polarity.RETRACT,
        extract_confidence=0.95,
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    assert len(rep2.decisions) >= 1
    assert rep2.decisions[0].rule_id == "R-RETRACT"
    assert rep2.decisions[0].action == Action.RETRACT

    m = service.repo.get_memory(m_id)
    assert m is not None
    assert m.status == MemoryStatus.HISTORICAL
    assert m.valid_to == t2
    assert m.closed_at == t2


def test_r_retract_closes_multi(service: MemoryService) -> None:
    """Section 4.4 & 4.2: Retraction closes only the matching multi-valued memory."""
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    t3 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    # Ingest service 1: auth_service
    msg1 = IngestMessage(
        source_id="sysadmin_priya", text="Server A runs auth_service", event_time=t1
    )
    claim1 = Claim(
        subject="Server A", attribute="runs_service", value="auth_service", extract_confidence=0.95
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    auth_id = rep1.decisions[0].memory_id
    assert auth_id is not None

    # Ingest service 2: billing_service (coexists)
    msg2 = IngestMessage(
        source_id="sysadmin_priya", text="Server A runs billing_service", event_time=t2
    )
    claim2 = Claim(
        subject="Server A",
        attribute="runs_service",
        value="billing_service",
        extract_confidence=0.95,
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    billing_id = rep2.decisions[0].memory_id
    assert billing_id is not None
    assert rep2.decisions[0].rule_id == "R-COEXIST"

    # Retract auth_service only
    msg3 = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A no longer runs auth_service",
        event_time=t3,
    )
    claim3 = Claim(
        subject="Server A",
        attribute="runs_service",
        value="auth_service",
        polarity=Polarity.RETRACT,
        extract_confidence=0.95,
    )
    rep3 = service.ingest_claims(msg3, [claim3], recorded_at=t3)
    assert rep3.decisions[0].rule_id == "R-RETRACT"
    assert rep3.decisions[0].action == Action.RETRACT

    # auth_service memory is HISTORICAL
    auth_mem = service.repo.get_memory(auth_id)
    assert auth_mem is not None
    assert auth_mem.status == MemoryStatus.HISTORICAL
    assert auth_mem.valid_to == t3

    # billing_service memory is still ACTIVE
    billing_mem = service.repo.get_memory(billing_id)
    assert billing_mem is not None
    assert billing_mem.status == MemoryStatus.ACTIVE
    assert billing_mem.valid_to is None


def test_claim_retract_no_target_skips(service: MemoryService) -> None:
    """Retraction with nothing to retract produces R-SKIP-NOTARGET."""
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)

    msg = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A no longer runs payments_daemon",
        event_time=t1,
    )
    claim = Claim(
        subject="Server A",
        attribute="runs_service",
        value="payments_daemon",
        polarity=Polarity.RETRACT,
        extract_confidence=0.95,
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t1)
    assert len(rep.decisions) == 1
    assert rep.decisions[0].rule_id == "R-SKIP-NOTARGET"
    assert rep.decisions[0].action == Action.SKIP


def test_claim_retract_weak_source_rejects(service: MemoryService) -> None:
    """Retraction too weak produces R-REJECT-RETRACT without mutating target memory."""
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m_id = rep1.decisions[0].memory_id
    assert m_id is not None

    # Weak retraction from untrusted source
    msg2 = IngestMessage(
        source_id="untrusted_feed",
        text="Server A no longer has Python 3.10",
        event_time=t2,
    )
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        polarity=Polarity.RETRACT,
        extract_confidence=0.50,
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    assert len(rep2.decisions) == 1
    assert rep2.decisions[0].rule_id == "R-REJECT-RETRACT"
    assert rep2.decisions[0].action == Action.REJECT

    # Target memory remains ACTIVE
    m = service.repo.get_memory(m_id)
    assert m is not None
    assert m.status == MemoryStatus.ACTIVE


def test_claim_retract_invalid_window_raises(service: MemoryService) -> None:
    """Retraction claim with valid_from <= target.valid_from raises InvalidValidityWindowError."""
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
    )
    service.ingest_claims(msg1, [claim1], recorded_at=t1)

    # Claim with valid_from equal to target.valid_from
    msg2 = IngestMessage(
        source_id="sysadmin_priya", text="Server A no longer has Python 3.10", event_time=t1
    )
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        polarity=Polarity.RETRACT,
        valid_from=t1,
        extract_confidence=0.95,
    )
    with pytest.raises(InvalidValidityWindowError, match="must be strictly greater"):
        service.ingest_claims(msg2, [claim2], recorded_at=t1)


# =============================================================================
# 3. Belief-Time Query Tests (known_at)
# =============================================================================


def test_known_at_ignores_later_closure(service: MemoryService) -> None:
    """Belief-time query before closed_at still sees the memory as valid/open."""
    t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t_retract = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    t_query_inside = datetime(2026, 3, 15, 10, 0, tzinfo=UTC)
    t_belief_before = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    t_belief_after = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)

    # Ingest Python 3.10 in January
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m_id = rep1.decisions[0].memory_id
    assert m_id is not None

    # Retract in June
    service.retract(m_id, valid_to=t_retract, recorded_at=t_retract)

    # Query with belief time BEFORE retraction (April 1)
    # The retraction had not occurred yet as of April 1
    res_before = service.current(
        "Server A",
        "python_version",
        valid_at=t_query_inside,
        known_at=t_belief_before,
        now=t_belief_after,
    )
    assert len(res_before.items) == 1
    assert res_before.items[0].memory.id == m_id
    assert res_before.items[0].memory.value_norm == "3.10"

    # Query with valid_at after retraction window (June 15) as known before retraction (April 1):
    # At April 1, memory was open-ended (valid_to was unknown/unset), so it was believed valid!
    t_query_june = datetime(2026, 6, 15, 10, 0, tzinfo=UTC)
    res_june_pre = service.current(
        "Server A",
        "python_version",
        valid_at=t_query_june,
        known_at=t_belief_before,
        now=t_belief_after,
    )
    assert len(res_june_pre.items) == 1
    assert res_june_pre.items[0].memory.id == m_id

    # Query with belief time AFTER retraction (July 1):
    # Retraction is now known: valid_to is June 1, so June 15 is outside the validity window
    res_june_post = service.current(
        "Server A",
        "python_version",
        valid_at=t_query_june,
        known_at=t_belief_after,
        now=t_belief_after,
    )
    assert len(res_june_post.items) == 0


def test_as_of_boundary(service: MemoryService) -> None:
    """valid_at == valid_to returns newer memory (half-open [valid_from, valid_to))."""
    t1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    t_now = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)

    # M1 valid Jan 1
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.90
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # M2 replaces M1 at June 1
    msg2 = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A", attribute="python_version", value="3.12", extract_confidence=0.95
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    # Query exactly at boundary t2 (June 1)
    res_boundary = service.current("Server A", "python_version", valid_at=t2, now=t_now)
    assert len(res_boundary.items) == 1
    assert res_boundary.items[0].memory.id == m2_id
    assert res_boundary.items[0].memory.value_norm == "3.12"

    # Query 1 second before t2 -> returns M1
    t_just_before = t2 - timedelta(seconds=1)
    res_before = service.current("Server A", "python_version", valid_at=t_just_before, now=t_now)
    assert len(res_before.items) == 1
    assert res_before.items[0].memory.id == m1_id
    assert res_before.items[0].memory.value_norm == "3.10"


def test_get_current_with_known_at(service: MemoryService) -> None:
    """service.get_current respects known_at for belief-time queries."""
    t1 = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    t_query = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t_after = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    service.retract(m1_id, valid_to=t2, recorded_at=t2)

    # Known at April 1 (before retraction recorded)
    m_pre = service.get_current("Server A", "python_version", now=t_query, known_at=t_query)
    assert m_pre is not None
    assert m_pre.id == m1_id

    # Current at July 1 (retracted memory is no longer active at t_after)
    m_post = service.get_current("Server A", "python_version", now=t_after, known_at=t_after)
    assert m_post is None


# =============================================================================
# 4. Idempotency Tests
# =============================================================================


def test_duplicate_claim_ingest_idempotency(service: MemoryService) -> None:
    """Duplicate message ingestion returns duplicate=True and does not duplicate memories."""
    t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t)
    claim = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
    )

    rep1 = service.ingest_claims(msg, [claim], recorded_at=t)
    assert rep1.duplicate is False
    assert len(rep1.decisions) == 1

    # Ingest identical message again
    rep2 = service.ingest_claims(msg, [claim], recorded_at=t)
    assert rep2.duplicate is True
    assert rep2.message_id == rep1.message_id
    assert len(rep2.decisions) == 0

    # Ensure memories table has exactly 1 entry
    ent = service.repo.get_entity_by_name("Server A")
    assert ent is not None
    mems = service.repo.find_memories(entity_id=int(ent["id"]), attribute="python_version")
    assert len(mems) == 1


def test_edge_insertion_idempotency(service: MemoryService) -> None:
    """Inserting the same edge twice returns the existing edge ID without error."""
    t = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    msg = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t)
    claim = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t)
    m1_id = rep.decisions[0].memory_id
    assert m1_id is not None

    # Create dummy second memory
    from memorymesh.models import NewMemory

    ent = service.repo.get_entity_by_name("Server A")
    assert ent is not None
    m2_id = service.repo.insert_memory(
        NewMemory(
            entity_id=int(ent["id"]),
            attribute="runs_service",
            value_norm="api",
            value_display="api",
            context_key="",
            valid_from=t,
            message_id=rep.message_id,
            source_id="sysadmin_priya",
            status=MemoryStatus.ACTIVE,
            confidence=0.9,
        ),
        recorded_at=t,
    )

    edge_id_1 = service.repo.insert_edge(
        src_memory_id=m2_id,
        dst_memory_id=m1_id,
        edge_type=EdgeType.DEPENDS_ON,
        reason="test dependency",
        created_at=t,
    )
    assert edge_id_1 > 0

    # Insert identical edge again -> returns same edge ID
    edge_id_2 = service.repo.insert_edge(
        src_memory_id=m2_id,
        dst_memory_id=m1_id,
        edge_type=EdgeType.DEPENDS_ON,
        reason="test dependency",
        created_at=t,
    )
    assert edge_id_2 == edge_id_1


def test_service_retract_transaction_rollback_on_failure(service: MemoryService) -> None:
    """Failure during retraction rolls back all updates atomically, preserving memory state."""
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    msg = IngestMessage(
        source_id="sysadmin_priya",
        text="Server A has Python 3.10",
        event_time=t1,
    )
    claim = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        extract_confidence=0.95,
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t1)
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    original_insert_log = service.repo.insert_decision_log

    def fail_insert_log(entry: DecisionLogEntry) -> DecisionLogEntry:
        if entry.rule_id == "R-RETRACT":
            raise RuntimeError("Simulated crash during decision logging")
        return original_insert_log(entry)

    service.repo.insert_decision_log = fail_insert_log  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Simulated crash during decision logging"):
        service.retract(m_id, valid_to=t2, recorded_at=t2)

    service.repo.insert_decision_log = original_insert_log  # type: ignore[method-assign]

    # Memory state must remain completely unchanged (ACTIVE, valid_to=None, closed_at=None)
    m_after = service.repo.get_memory(m_id)
    assert m_after is not None
    assert m_after.status == MemoryStatus.ACTIVE
    assert m_after.valid_to is None
    assert m_after.closed_at is None


# =============================================================================
# 5. Targeted Corrections Regression Tests (Issue 1, 2, 3)
# =============================================================================


def test_disputed_memory_retraction_lifecycle_and_resolution(
    service: MemoryService,
) -> None:
    """Issue 1: Retracting a disputed memory leaves counterpart in DISPUTED with review flag.

    Verifies:
    1. Retraction of M1 transitions M1 to HISTORICAL with valid_to and closed_at.
    2. Counterpart M2 is NOT automatically promoted to ACTIVE; remains DISPUTED.
    3. M2 is flagged needs_review=True with explanatory review_reason.
    4. M2 is not orphaned: surfaces in find_disputed_pairs() and service.disputes().
    5. service.get_current() returns None (no active authoritative winner).
    6. Operator can resolve the dispute in favor of M2 via service.resolve_dispute().
    7. Winner M2 becomes ACTIVE, needs_review is cleared, and SUPERSEDES edge connects to M1.
    8. check_invariants() reports zero violations.
    """
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    t3 = datetime(2026, 3, 5, 10, 0, tzinfo=UTC)
    t4 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)

    # Ingest M1: Python 3.10
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.90
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # Ingest M2: Python 3.12 at same event_time -> triggers R-DISPUTE
    msg2 = IngestMessage(
        source_id="monitoring_agent", text="Server A has Python 3.12", event_time=t1
    )
    claim2 = Claim(
        subject="Server A", attribute="python_version", value="3.12", extract_confidence=0.90
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None
    assert rep2.decisions[0].rule_id == "R-DISPUTE"

    m1 = service.repo.get_memory(m1_id)
    m2 = service.repo.get_memory(m2_id)
    assert m1 is not None and m1.status == MemoryStatus.DISPUTED
    assert m2 is not None and m2.status == MemoryStatus.DISPUTED

    # Retract M1 at t3
    log = service.retract(m1_id, reason="M1 retracted by admin", valid_to=t3, recorded_at=t3)
    assert log.rule_id == "R-RETRACT"

    # M1 transitions to HISTORICAL
    m1_after = service.repo.get_memory(m1_id)
    assert m1_after is not None
    assert m1_after.status == MemoryStatus.HISTORICAL
    assert m1_after.valid_to == t3
    assert m1_after.closed_at == t3

    # M2 is NOT automatically promoted to ACTIVE; remains DISPUTED
    m2_after = service.repo.get_memory(m2_id)
    assert m2_after is not None
    assert m2_after.status == MemoryStatus.DISPUTED
    # M2 is flagged with needs_review=True
    assert m2_after.needs_review is True
    assert f"Conflicting memory M{m1_id} was retracted" in m2_after.review_reason

    # Current retrieval does NOT return M2 as active fact (no authoritative winner yet)
    current_mem = service.get_current("Server A", "python_version", now=t3)
    assert current_mem is None

    # M2 is NOT orphaned: find_disputed_pairs surfaces the dispute
    pairs = service.repo.find_disputed_pairs()
    assert len(pairs) >= 1
    assert any({p[0].id, p[1].id} == {m1_id, m2_id} for p in pairs)

    # service.disputes surfaces it as well
    disputes = service.disputes("Server A")
    assert len(disputes) >= 1
    assert any({d.a.memory.id, d.b.memory.id} == {m1_id, m2_id} for d in disputes)

    # Operator resolves the dispute in favor of M2
    res_log = service.resolve_dispute(winner_memory_id=m2_id, recorded_at=t4)
    assert res_log.rule_id == "R-DISPUTE-RESOLVE"
    assert res_log.memory_id == m2_id

    # Winner M2 is promoted to ACTIVE, review flag is cleared
    m2_resolved = service.repo.get_memory(m2_id)
    assert m2_resolved is not None
    assert m2_resolved.status == MemoryStatus.ACTIVE
    assert m2_resolved.needs_review is False
    assert m2_resolved.review_reason == ""

    # SUPERSEDES edge inserted between winner M2 and retracted counterpart M1
    edges = service.repo.get_edges_for_memory(m2_id, direction="src")
    supersedes_edges = [
        e for e in edges if e.type == EdgeType.SUPERSEDES and e.dst_memory_id == m1_id
    ]
    assert len(supersedes_edges) == 1

    # Invariants report zero violations
    inv = service.check_invariants()
    assert inv.is_valid is True
    assert len(inv.violations) == 0


def test_retroactive_retraction_belief_time(service: MemoryService) -> None:
    """Issue 2: Genuine retroactive retraction exercises belief-time closure reconstruction.

    Scenario:
    - Learn Jan 10 (valid_from = Jan 10, recorded_at = Jan 10)
    - Retraction recorded June 1 (recorded_at = June 1), real-world invalid Feb 1 (valid_to = Feb 1)
    - Query target: valid_at = March 15.
      Notice: valid_to <= valid_at (Feb 1 <= March 15), so event-time validity ended.
    - Case A: known_at = April 1 (< June 1). Retraction unknown at April 1 (closed_at > known_at).
      Reconstructs valid_to = None. Must return Python 3.10.
    - Case B: known_at = July 1 (> June 1). Retraction known at July 1 (closed_at <= known_at).
      valid_to = Feb 1. March 15 >= Feb 1, so must NOT return Python 3.10.
    """
    t_jan10 = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
    t_feb1 = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    t_mar15 = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
    t_apr1 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    t_jun1 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    t_jul1 = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)

    # Ingest Python 3.10 on Jan 10
    msg = IngestMessage(
        source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t_jan10
    )
    claim = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.95
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t_jan10)
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    # Retract on June 1 with retroactive valid_to = Feb 1
    service.retract(m_id, valid_to=t_feb1, recorded_at=t_jun1)

    # Verify persisted record
    persisted = service.repo.get_memory(m_id)
    assert persisted is not None
    assert persisted.status == MemoryStatus.HISTORICAL
    assert persisted.valid_to == t_feb1
    assert persisted.closed_at == t_jun1

    # Case A: Belief time known_at = April 1 (< June 1)
    # The retraction was recorded on June 1, so on April 1 it was completely unknown.
    # Reconstructed valid_to is None. Because valid_from (Jan 10) <= March 15 and valid_to is None,
    # the query MUST return Python 3.10.
    res_a = service.current(
        "Server A",
        "python_version",
        valid_at=t_mar15,
        known_at=t_apr1,
        now=t_jul1,
    )
    assert len(res_a.items) == 1
    assert res_a.items[0].memory.id == m_id
    assert res_a.items[0].memory.value_norm == "3.10"

    mem_curr_a = service.get_current(
        "Server A",
        "python_version",
        now=t_mar15,
        known_at=t_apr1,
    )
    assert mem_curr_a is not None
    assert mem_curr_a.id == m_id
    assert mem_curr_a.value_norm == "3.10"

    # Case B: Belief time known_at = July 1 (> June 1)
    # The retraction is now known. valid_to is Feb 1. Since valid_at (March 15) >= valid_to (Feb 1),
    # the memory is no longer valid at March 15. The query MUST NOT return Python 3.10.
    res_b = service.current(
        "Server A",
        "python_version",
        valid_at=t_mar15,
        known_at=t_jul1,
        now=t_jul1,
    )
    assert len(res_b.items) == 0

    mem_curr_b = service.get_current(
        "Server A",
        "python_version",
        now=t_mar15,
        known_at=t_jul1,
    )
    assert mem_curr_b is None


def test_belief_time_reconstructs_later_rejected_dispute_loser(
    service: MemoryService,
) -> None:
    """Issue 3: Belief-time queries reconstruct later-rejected dispute losers as DISPUTED.

    When a dispute between M1 and M2 is resolved at t_resolve (June 1) in favor of M1,
    M2 is rejected with closed_at = June 1.
    - Querying at known_at = July 1 (or current belief): M2 is excluded as REJECTED.
    - Querying at known_at = April 1 (< June 1): M2 was not yet rejected at April 1;
      its effective status was DISPUTED and find_disputed_pairs(known_at=April 1) reconstructs
      the active dispute pair (M1, M2).
    """
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    t_apr1 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    t_jun1 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    t_jul1 = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)

    # Ingest M1: Python 3.10
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Server A has Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.90
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # Ingest M2: Python 3.12 at same event_time -> triggers R-DISPUTE
    msg2 = IngestMessage(
        source_id="monitoring_agent", text="Server A has Python 3.12", event_time=t1
    )
    claim2 = Claim(
        subject="Server A", attribute="python_version", value="3.12", extract_confidence=0.90
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    # Resolve dispute at June 1 in favor of M1 (M2 becomes REJECTED with closed_at=June 1)
    service.resolve_dispute(winner_memory_id=m1_id, recorded_at=t_jun1)

    m1_db = service.repo.get_memory(m1_id)
    m2_db = service.repo.get_memory(m2_id)
    assert m1_db is not None and m1_db.status == MemoryStatus.ACTIVE
    assert m2_db is not None and m2_db.status == MemoryStatus.REJECTED
    assert m2_db.closed_at == t_jun1

    # Current query at July 1: M2 is excluded as REJECTED
    curr_jul = service.current("Server A", "python_version", now=t_jul1, known_at=t_jul1)
    assert len(curr_jul.items) == 1
    assert curr_jul.items[0].memory.id == m1_id
    assert len(curr_jul.disputes) == 0

    pairs_curr = service.repo.find_disputed_pairs()
    assert len(pairs_curr) == 0

    # Belief-time query at April 1 (< June 1):
    # Dispute had NOT been resolved yet at April 1!
    m2_apr = service.repo.get_memory(m2_id, known_at=t_apr1)
    assert m2_apr is not None
    assert m2_apr.status == MemoryStatus.DISPUTED
    assert m2_apr.closed_at is None

    # find_disputed_pairs at April 1 reconstructs the active dispute
    pairs_apr = service.repo.find_disputed_pairs(known_at=t_apr1)
    assert len(pairs_apr) == 1
    assert {pairs_apr[0][0].id, pairs_apr[0][1].id} == {m1_id, m2_id}

    # service.current at April 1 reflects the dispute
    curr_apr = service.current(
        "Server A",
        "python_version",
        valid_at=t1,
        known_at=t_apr1,
        now=t_jul1,
    )
    assert len(curr_apr.disputes) == 1
    assert {curr_apr.disputes[0].a.memory.id, curr_apr.disputes[0].b.memory.id} == {m1_id, m2_id}

    # Invariants are valid
    assert service.check_invariants().is_valid is True


def test_historical_dispute_before_retraction(service: MemoryService) -> None:
    """Test A: Dispute before retraction reconstructs both memories as DISPUTED.

    When M1 and M2 enter dispute at March 1, and M1 is retracted at March 5,
    querying at belief time March 3 (before retraction) must reconstruct both
    memories as DISPUTED with closed_at=None and valid_to=None. The dispute pair
    must be visible and no authoritative current winner returned.
    """
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t_query = datetime(2026, 3, 3, 12, 0, tzinfo=UTC)
    t_retract = datetime(2026, 3, 5, 14, 0, tzinfo=UTC)

    # Ingest M1: Python 3.11 at March 1
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Python 3.11", event_time=t1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.11", extract_confidence=0.90
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # Ingest M2: Python 3.12 at March 1 -> triggers R-DISPUTE
    msg2 = IngestMessage(source_id="monitoring_agent", text="Python 3.12", event_time=t1)
    claim2 = Claim(
        subject="Server A", attribute="python_version", value="3.12", extract_confidence=0.90
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t1)
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    # Both are DISPUTED initially
    m1 = service.repo.get_memory(m1_id)
    m2 = service.repo.get_memory(m2_id)
    assert m1 is not None and m1.status == MemoryStatus.DISPUTED
    assert m2 is not None and m2.status == MemoryStatus.DISPUTED

    # Retract M1 at March 5 -> M1 becomes HISTORICAL in DB
    service.retract(m1_id, reason="Admin error", recorded_at=t_retract)

    m1_db = service.repo.get_memory(m1_id)
    assert m1_db is not None
    assert m1_db.status == MemoryStatus.HISTORICAL
    assert m1_db.closed_at == t_retract

    # Query at belief time March 3 (prior to retraction):
    # Both M1 and M2 must be reconstructed as DISPUTED with open intervals.
    m1_b = service.repo.get_memory(m1_id, known_at=t_query)
    m2_b = service.repo.get_memory(m2_id, known_at=t_query)
    assert m1_b is not None
    assert m1_b.status == MemoryStatus.DISPUTED
    assert m1_b.closed_at is None
    assert m1_b.valid_to is None
    assert m1_b.needs_review is True

    assert m2_b is not None
    assert m2_b.status == MemoryStatus.DISPUTED
    assert m2_b.closed_at is None
    assert m2_b.valid_to is None
    assert m2_b.needs_review is True

    # Dispute pair is visible at earlier belief time
    pairs = service.repo.find_disputed_pairs(known_at=t_query)
    assert len(pairs) == 1
    assert {pairs[0][0].id, pairs[0][1].id} == {m1_id, m2_id}

    disputes = service.disputes(known_at=t_query)
    assert len(disputes) == 1
    assert {disputes[0].a.memory.id, disputes[0].b.memory.id} == {m1_id, m2_id}

    # No authoritative current winner returned when under dispute
    winner = service.get_current("Server A", "python_version", now=t_query, known_at=t_query)
    assert winner is None

    # Invariants are satisfied
    assert service.check_invariants().is_valid is True


def test_belief_after_retraction(service: MemoryService) -> None:
    """Test B: Belief after retraction preserves retracted memory and actionable counterpart.

    When M1 is retracted at March 5, querying at March 7 shows M1 as HISTORICAL
    and M2 as DISPUTED with needs_review=True. The dispute pair remains reviewable
    and counterpart M2 can be resolved to ACTIVE.
    """
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t_retract = datetime(2026, 3, 5, 14, 0, tzinfo=UTC)
    t_after = datetime(2026, 3, 7, 12, 0, tzinfo=UTC)
    t_resolve = datetime(2026, 3, 8, 9, 0, tzinfo=UTC)

    # Ingest M1 and M2 into dispute at March 1
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Python 3.11", event_time=t1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.11", extract_confidence=0.90
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="monitoring_agent", text="Python 3.12", event_time=t1)
    claim2 = Claim(
        subject="Server A", attribute="python_version", value="3.12", extract_confidence=0.90
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t1)
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    # Retract M1 at March 5
    service.retract(m1_id, reason="Erroneous claim", recorded_at=t_retract)

    # Query at belief time March 7 (> March 5)
    m1_after = service.repo.get_memory(m1_id, known_at=t_after)
    m2_after = service.repo.get_memory(m2_id, known_at=t_after)

    assert m1_after is not None
    assert m1_after.status == MemoryStatus.HISTORICAL
    assert m1_after.closed_at == t_retract
    assert m1_after.valid_to == t_retract

    assert m2_after is not None
    assert m2_after.status == MemoryStatus.DISPUTED
    assert m2_after.needs_review is True

    # Dispute pair remains reviewable
    pairs = service.repo.find_disputed_pairs(known_at=t_after)
    assert len(pairs) == 1
    assert {pairs[0][0].id, pairs[0][1].id} == {m1_id, m2_id}

    # Actionable: resolving dispute in favor of M2 succeeds
    service.resolve_dispute(winner_memory_id=m2_id, recorded_at=t_resolve)
    m2_resolved = service.repo.get_memory(m2_id)
    assert m2_resolved is not None
    assert m2_resolved.status == MemoryStatus.ACTIVE

    # Invariants are valid
    assert service.check_invariants().is_valid is True


def test_ordinary_historical_memory_not_disputed(service: MemoryService) -> None:
    """Test C: Ordinary historical memories are not reconstructed as DISPUTED.

    When M1 is superseded by M2 via R-REPLACE, M1 becomes HISTORICAL without
    ever being in a dispute. It must remain ACTIVE before replacement and
    HISTORICAL after replacement, never DISPUTED, across belief-time queries.
    """
    t_jan1 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    t_apr1 = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    t_jun1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t_jul1 = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)

    # Ingest M1: Python 3.10 at Jan 1 (ACTIVE)
    msg1 = IngestMessage(source_id="dev_chat", text="Python 3.10", event_time=t_jan1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.70
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t_jan1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    m1_init = service.repo.get_memory(m1_id)
    assert m1_init is not None and m1_init.status == MemoryStatus.ACTIVE

    # Ingest M2: Python 3.11 at June 1 with higher score -> R-REPLACE
    msg2 = IngestMessage(source_id="sysadmin_priya", text="Python 3.11", event_time=t_jun1)
    claim2 = Claim(
        subject="Server A", attribute="python_version", value="3.11", extract_confidence=0.95
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t_jun1)
    assert rep2.decisions[0].action == Action.REPLACE
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    # DB state: M1 is HISTORICAL, M2 is ACTIVE
    m1_db = service.repo.get_memory(m1_id)
    assert m1_db is not None
    assert m1_db.status == MemoryStatus.HISTORICAL
    assert m1_db.closed_at == t_jun1

    # Query at April 1 (before replacement):
    # M1 is an ordinary historical memory created by replacement; at April 1 (before
    # replacement on June 1), it must be reconstructed as ACTIVE with open intervals
    # (closed_at=None, valid_to=None), and NEVER as DISPUTED.
    m1_apr = service.repo.get_memory(m1_id, known_at=t_apr1)
    assert m1_apr is not None
    assert m1_apr.status == MemoryStatus.ACTIVE
    assert m1_apr.closed_at is None
    assert m1_apr.valid_to is None
    assert m1_apr.status != MemoryStatus.DISPUTED

    # No disputes exist at April 1
    pairs_apr = service.repo.find_disputed_pairs(known_at=t_apr1)
    assert len(pairs_apr) == 0
    disputes_apr = service.disputes(known_at=t_apr1)
    assert len(disputes_apr) == 0

    # Query at July 1 (after replacement):
    # M1 is HISTORICAL, M2 is ACTIVE, neither is DISPUTED
    m1_jul = service.repo.get_memory(m1_id, known_at=t_jul1)
    m2_jul = service.repo.get_memory(m2_id, known_at=t_jul1)
    assert m1_jul is not None
    assert m1_jul.status == MemoryStatus.HISTORICAL
    assert m2_jul is not None
    assert m2_jul.status == MemoryStatus.ACTIVE

    pairs_jul = service.repo.find_disputed_pairs(known_at=t_jul1)
    assert len(pairs_jul) == 0
    disputes_jul = service.disputes(known_at=t_jul1)
    assert len(disputes_jul) == 0

    # Invariants are valid
    assert service.check_invariants().is_valid is True


def test_resolved_dispute_belief_time_lifecycle(service: MemoryService) -> None:
    """Test D: Belief-time lifecycle queries across three distinct operational windows.

    Lifecycle:
    - March 1: M1 and M2 enter dispute (both DISPUTED).
    - March 5: M1 is retracted (M1 -> HISTORICAL, M2 remains DISPUTED).
    - March 10: Dispute resolved in favor of M2 (M2 -> ACTIVE).

    Windows:
    - Window 1 (March 3): before retraction -> both DISPUTED, pair visible, winner is None.
    - Window 2 (March 7): between retraction and resolution -> M1 HISTORICAL, M2 DISPUTED,
      pair visible, winner is None.
    - Window 3 (March 15): after resolution -> M1 HISTORICAL, M2 ACTIVE, pair empty,
      winner is M2 ("3.12").
    """
    t_mar1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t_mar3 = datetime(2026, 3, 3, 12, 0, tzinfo=UTC)
    t_mar5 = datetime(2026, 3, 5, 14, 0, tzinfo=UTC)
    t_mar7 = datetime(2026, 3, 7, 12, 0, tzinfo=UTC)
    t_mar10 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    t_mar15 = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)

    # Ingest M1: Python 3.11 at March 1
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Python 3.11", event_time=t_mar1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.11", extract_confidence=0.90
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t_mar1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # Ingest M2: Python 3.12 at March 1 -> triggers R-DISPUTE
    msg2 = IngestMessage(source_id="monitoring_agent", text="Python 3.12", event_time=t_mar1)
    claim2 = Claim(
        subject="Server A", attribute="python_version", value="3.12", extract_confidence=0.90
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t_mar1)
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    # Retract M1 on March 5
    service.retract(m1_id, reason="Admin retraction", recorded_at=t_mar5)

    # Resolve dispute on March 10 in favor of M2
    service.resolve_dispute(winner_memory_id=m2_id, recorded_at=t_mar10)

    # =========================================================================
    # Window 1: March 3 (before retraction)
    # =========================================================================
    m1_w1 = service.repo.get_memory(m1_id, known_at=t_mar3)
    m2_w1 = service.repo.get_memory(m2_id, known_at=t_mar3)
    assert m1_w1 is not None
    assert m1_w1.status == MemoryStatus.DISPUTED
    assert m1_w1.closed_at is None
    assert m1_w1.valid_to is None

    assert m2_w1 is not None
    assert m2_w1.status == MemoryStatus.DISPUTED
    assert m2_w1.closed_at is None
    assert m2_w1.valid_to is None

    pairs_w1 = service.repo.find_disputed_pairs(known_at=t_mar3)
    assert len(pairs_w1) == 1
    assert {pairs_w1[0][0].id, pairs_w1[0][1].id} == {m1_id, m2_id}

    curr_w1 = service.get_current("Server A", "python_version", now=t_mar3, known_at=t_mar3)
    assert curr_w1 is None

    # =========================================================================
    # Window 2: March 7 (between retraction and resolution)
    # =========================================================================
    m1_w2 = service.repo.get_memory(m1_id, known_at=t_mar7)
    m2_w2 = service.repo.get_memory(m2_id, known_at=t_mar7)
    assert m1_w2 is not None
    assert m1_w2.status == MemoryStatus.HISTORICAL
    assert m1_w2.closed_at == t_mar5
    assert m1_w2.valid_to == t_mar5

    assert m2_w2 is not None
    assert m2_w2.status == MemoryStatus.DISPUTED
    assert m2_w2.needs_review is True

    pairs_w2 = service.repo.find_disputed_pairs(known_at=t_mar7)
    assert len(pairs_w2) == 1
    assert {pairs_w2[0][0].id, pairs_w2[0][1].id} == {m1_id, m2_id}

    curr_w2 = service.get_current("Server A", "python_version", now=t_mar7, known_at=t_mar7)
    assert curr_w2 is None

    # =========================================================================
    # Window 3: March 15 (after resolution)
    # =========================================================================
    m1_w3 = service.repo.get_memory(m1_id, known_at=t_mar15)
    m2_w3 = service.repo.get_memory(m2_id, known_at=t_mar15)
    assert m1_w3 is not None
    assert m1_w3.status == MemoryStatus.HISTORICAL
    assert m1_w3.closed_at == t_mar5

    assert m2_w3 is not None
    assert m2_w3.status == MemoryStatus.ACTIVE
    assert m2_w3.closed_at is None

    pairs_w3 = service.repo.find_disputed_pairs(known_at=t_mar15)
    assert len(pairs_w3) == 0

    curr_w3 = service.get_current("Server A", "python_version", now=t_mar15, known_at=t_mar15)
    assert curr_w3 is not None
    assert curr_w3.id == m2_id
    assert curr_w3.value_display == "3.12"

    # Invariants are valid
    assert service.check_invariants().is_valid is True


def test_known_at_rejects_naive_timestamp(service: MemoryService) -> None:
    """Naive known_at is rejected with ValueError across repository, service, and retrieval."""
    t_aware = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t_naive = datetime(2026, 3, 1, 10, 0)

    # Class with tzinfo where utcoffset returns None
    class UnspecifiedTz(tzinfo):
        def utcoffset(self, dt: datetime | None) -> None:
            return None

        def tzname(self, dt: datetime | None) -> str | None:
            return None

        def dst(self, dt: datetime | None) -> timedelta | None:
            return None

    t_unspecified = datetime(2026, 3, 1, 10, 0, tzinfo=UnspecifiedTz())

    # Ingest a sample memory
    msg = IngestMessage(source_id="sysadmin_priya", text="Python 3.10", event_time=t_aware)
    claim = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.90
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t_aware)
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    # 1. Repository.get_memory
    with pytest.raises(ValueError, match="known_at must be a timezone-aware datetime"):
        service.repo.get_memory(m_id, known_at=t_naive)
    with pytest.raises(ValueError, match="known_at must be a timezone-aware datetime"):
        service.repo.get_memory(m_id, known_at=t_unspecified)

    # 2. Repository.find_memories
    with pytest.raises(ValueError, match="known_at must be a timezone-aware datetime"):
        service.repo.find_memories(1, "python_version", known_at=t_naive)
    with pytest.raises(ValueError, match="known_at must be a timezone-aware datetime"):
        service.repo.find_memories(1, "python_version", known_at=t_unspecified)

    # 3. Repository.find_disputed_pairs
    with pytest.raises(ValueError, match="known_at must be a timezone-aware datetime"):
        service.repo.find_disputed_pairs(known_at=t_naive)
    with pytest.raises(ValueError, match="known_at must be a timezone-aware datetime"):
        service.repo.find_disputed_pairs(known_at=t_unspecified)

    # 4. Repository decision log checks
    with pytest.raises(ValueError, match="before_or_at must be a timezone-aware datetime"):
        service.repo.has_dispute_decision_log(m_id, before_or_at=t_naive)
    with pytest.raises(ValueError, match="before_or_at must be a timezone-aware datetime"):
        service.repo.has_dispute_resolution_log(m_id, before_or_at=t_naive)

    # 5. Service.current and get_current
    with pytest.raises(ValueError, match="known_at must be a timezone-aware datetime"):
        service.current("Server A", "python_version", now=t_aware, known_at=t_naive)
    with pytest.raises(ValueError, match="known_at must be a timezone-aware datetime"):
        service.get_current("Server A", "python_version", now=t_aware, known_at=t_naive)

    # 6. Service.disputes
    with pytest.raises(ValueError, match="known_at must be a timezone-aware datetime"):
        service.disputes(known_at=t_naive)

    # 7. Retrieval helpers
    with pytest.raises(ValueError, match="known_at must be a timezone-aware datetime"):
        retrieve_valid_memories(
            service.repo, 1, "python_version", valid_at=t_aware, known_at=t_naive
        )
    with pytest.raises(ValueError, match="known_at must be a timezone-aware datetime"):
        retrieve_as_of(service.repo, 1, "python_version", valid_at=t_aware, known_at=t_naive)


def test_known_at_timezone_aware_non_utc_normalized(service: MemoryService) -> None:
    """Timezone-aware non-UTC known_at is correctly normalized to UTC without semantic drift."""
    tz_ist = timezone(timedelta(hours=5, minutes=30))  # UTC+5:30
    tz_est = timezone(timedelta(hours=-5))  # UTC-5:00

    t_mar1_utc = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t_mar5_utc = datetime(2026, 3, 5, 14, 0, tzinfo=UTC)

    # Ingest M1: Python 3.11 at March 1
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Python 3.11", event_time=t_mar1_utc)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.11", extract_confidence=0.90
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t_mar1_utc)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # Ingest M2: Python 3.12 at March 1 -> triggers R-DISPUTE
    msg2 = IngestMessage(source_id="monitoring_agent", text="Python 3.12", event_time=t_mar1_utc)
    claim2 = Claim(
        subject="Server A", attribute="python_version", value="3.12", extract_confidence=0.90
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t_mar1_utc)
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    # Retract M1 on March 5 (14:00 UTC)
    service.retract(m1_id, reason="Admin retraction", recorded_at=t_mar5_utc)

    # Query at March 3 17:30 IST (+05:30) == March 3 12:00 UTC (before retraction)
    t_query_ist = datetime(2026, 3, 3, 17, 30, tzinfo=tz_ist)
    m1_ist = service.repo.get_memory(m1_id, known_at=t_query_ist)
    m2_ist = service.repo.get_memory(m2_id, known_at=t_query_ist)
    assert m1_ist is not None and m1_ist.status == MemoryStatus.DISPUTED
    assert m2_ist is not None and m2_ist.status == MemoryStatus.DISPUTED

    pairs_ist = service.repo.find_disputed_pairs(known_at=t_query_ist)
    assert len(pairs_ist) == 1
    assert {pairs_ist[0][0].id, pairs_ist[0][1].id} == {m1_id, m2_id}

    disputes_ist = service.disputes(known_at=t_query_ist)
    assert len(disputes_ist) == 1

    curr_ist = service.get_current(
        "Server A", "python_version", now=t_mar5_utc, known_at=t_query_ist
    )
    assert curr_ist is None

    # Query at March 7 07:00 EST (-05:00) == March 7 12:00 UTC (after retraction)
    t_query_est = datetime(2026, 3, 7, 7, 0, tzinfo=tz_est)
    m1_est = service.repo.get_memory(m1_id, known_at=t_query_est)
    m2_est = service.repo.get_memory(m2_id, known_at=t_query_est)
    assert m1_est is not None and m1_est.status == MemoryStatus.HISTORICAL
    assert m2_est is not None and m2_est.status == MemoryStatus.DISPUTED

    pairs_est = service.repo.find_disputed_pairs(known_at=t_query_est)
    assert len(pairs_est) == 1
    assert {pairs_est[0][0].id, pairs_est[0][1].id} == {m1_id, m2_id}

    # Invariants are valid
    assert service.check_invariants().is_valid is True


# =============================================================================
# 9. Ordinary Historical Reconstruction Regression Tests (P6.2 Final Correction)
# =============================================================================


def test_ordinary_replacement_belief_time(service: MemoryService) -> None:
    """Test A: Ordinary replacement belief-time reconstruction.

    - Ingest Python 3.10 at T1 (ACTIVE).
    - Ingest Python 3.12 at T2 (T2 > T1), triggering R-REPLACE and closing M1.
    - Query with known_at = T_query (T1 < T_query < T2):
      M1 is reconstructed as ACTIVE, closed_at=None, valid_to=None.
      service.get_current("Server A", "python_version", now=T_query, known_at=T_query) returns M1.
    - Query with known_at = T_post (T_post > T2):
      M1 is HISTORICAL (closed_at=T2, valid_to=T2).
      M2 is ACTIVE (closed_at=None).
      service.get_current("Server A", "python_version", now=T_post, known_at=T_post) returns M2.
    """
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t_query = datetime(2026, 3, 3, 12, 0, tzinfo=UTC)
    t2 = datetime(2026, 3, 5, 14, 0, tzinfo=UTC)
    t_post = datetime(2026, 3, 7, 12, 0, tzinfo=UTC)

    msg1 = IngestMessage(source_id="sysadmin_priya", text="Python 3.10", event_time=t1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.90
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    msg2 = IngestMessage(source_id="sysadmin_priya", text="Python 3.12", event_time=t2)
    claim2 = Claim(
        subject="Server A", attribute="python_version", value="3.12", extract_confidence=0.95
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t2)
    assert rep2.decisions[0].action == Action.REPLACE
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    # As of T_query (before replacement was known):
    m1_pre = service.repo.get_memory(m1_id, known_at=t_query)
    assert m1_pre is not None
    assert m1_pre.status == MemoryStatus.ACTIVE
    assert m1_pre.closed_at is None
    assert m1_pre.valid_to is None

    curr_pre = service.get_current("Server A", "python_version", now=t_query, known_at=t_query)
    assert curr_pre is not None
    assert curr_pre.id == m1_id
    assert curr_pre.value_display == "3.10"

    # As of T_post (after replacement):
    m1_post = service.repo.get_memory(m1_id, known_at=t_post)
    m2_post = service.repo.get_memory(m2_id, known_at=t_post)
    assert m1_post is not None
    assert m1_post.status == MemoryStatus.HISTORICAL
    assert m1_post.closed_at == t2
    assert m1_post.valid_to == t2

    assert m2_post is not None
    assert m2_post.status == MemoryStatus.ACTIVE
    assert m2_post.closed_at is None

    curr_post = service.get_current("Server A", "python_version", now=t_post, known_at=t_post)
    assert curr_post is not None
    assert curr_post.id == m2_id
    assert curr_post.value_display == "3.12"

    assert service.check_invariants().is_valid is True


def test_ordinary_service_retraction_belief_time(service: MemoryService) -> None:
    """Test B: Ordinary service-level retraction belief-time reconstruction.

    - Ingest active memory at T1.
    - Retract via service.retract at T2 (T2 > T1).
    - Query with known_at = T_query (T1 < T_query < T2):
      Memory is reconstructed as ACTIVE, closed_at=None, valid_to=None.
      service.get_current returns the active memory.
    - Query with known_at = T_post (T_post > T2):
      Memory is HISTORICAL with closed_at=T2, valid_to=T2.
      service.get_current returns None.
    """
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t_query = datetime(2026, 3, 3, 12, 0, tzinfo=UTC)
    t2 = datetime(2026, 3, 5, 14, 0, tzinfo=UTC)
    t_post = datetime(2026, 3, 7, 12, 0, tzinfo=UTC)

    msg = IngestMessage(source_id="sysadmin_priya", text="Python 3.10", event_time=t1)
    claim = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.90
    )
    rep = service.ingest_claims(msg, [claim], recorded_at=t1)
    m_id = rep.decisions[0].memory_id
    assert m_id is not None

    # Retract at T2
    retract_rep = service.retract(m_id, reason="Erroneous entry", recorded_at=t2)
    assert retract_rep.memory_id == m_id
    assert retract_rep.action == Action.RETRACT

    # As of T_query (before retraction was known):
    m_pre = service.repo.get_memory(m_id, known_at=t_query)
    assert m_pre is not None
    assert m_pre.status == MemoryStatus.ACTIVE
    assert m_pre.closed_at is None
    assert m_pre.valid_to is None

    curr_pre = service.get_current("Server A", "python_version", now=t_query, known_at=t_query)
    assert curr_pre is not None
    assert curr_pre.id == m_id
    assert curr_pre.value_display == "3.10"

    # As of T_post (after retraction):
    m_post = service.repo.get_memory(m_id, known_at=t_post)
    assert m_post is not None
    assert m_post.status == MemoryStatus.HISTORICAL
    assert m_post.closed_at == t2
    assert m_post.valid_to == t2

    curr_post = service.get_current("Server A", "python_version", now=t_post, known_at=t_post)
    assert curr_post is None

    assert service.check_invariants().is_valid is True


def test_ordinary_historical_backfill_not_converted_to_active(service: MemoryService) -> None:
    """Test D: Historical backfill memories are NOT reconstructed as ACTIVE.

    Verifies that historical memories created through backfill (R-BACKFILL) remain
    HISTORICAL and are not converted to ACTIVE merely because closed_at is later
    than known_at.
    """
    t_old = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    t_new = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t_ingest_new = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t_ingest_old = datetime(2026, 6, 5, 10, 0, tzinfo=UTC)
    t_query = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)

    # 1. Establish newer active memory (Python 3.12 starting 2026-01-01)
    msg_new = IngestMessage(source_id="sysadmin_priya", text="Python 3.12", event_time=t_new)
    claim_new = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=t_new,
        extract_confidence=0.95,
    )
    rep_new = service.ingest_claims(msg_new, [claim_new], recorded_at=t_ingest_new)
    m_new_id = rep_new.decisions[0].memory_id
    assert m_new_id is not None

    # 2. Ingest older claim (Python 3.10 valid 2025-01-01) -> triggers R-BACKFILL
    msg_old = IngestMessage(source_id="sysadmin_priya", text="Python 3.10", event_time=t_old)
    claim_old = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        valid_from=t_old,
        extract_confidence=0.95,
    )
    rep_old = service.ingest_claims(msg_old, [claim_old], recorded_at=t_ingest_old)
    assert len(rep_old.decisions) == 1
    assert rep_old.decisions[0].rule_id == "R-BACKFILL"
    m_backfill_id = rep_old.decisions[0].memory_id
    assert m_backfill_id is not None

    # Check repository backfill identification
    assert service.repo.is_backfill_memory(m_backfill_id) is True
    assert service.repo.is_backfill_memory(m_new_id) is False

    # Backfill memory in DB has status HISTORICAL and closed_at = t_ingest_old
    m_backfill_db = service.repo.get_memory(m_backfill_id)
    assert m_backfill_db is not None
    assert m_backfill_db.status == MemoryStatus.HISTORICAL
    assert m_backfill_db.closed_at == t_ingest_old

    # Query with known_at = June 10 (after backfill):
    m_backfill_query = service.repo.get_memory(m_backfill_id, known_at=t_query)
    assert m_backfill_query is not None
    assert m_backfill_query.status == MemoryStatus.HISTORICAL
    assert m_backfill_query.status != MemoryStatus.ACTIVE
    assert m_backfill_query.closed_at == t_ingest_old

    # Even if closed_at were simulated to be strictly in the future of known_at,
    # is_backfill_memory ensures it is never reconstructed as ACTIVE.
    with service.repo.transaction():
        service.repo.conn.execute(
            "UPDATE memories SET closed_at = '2026-12-31 00:00:00+00:00' WHERE id = ?",
            (m_backfill_id,),
        )
    m_sim = service.repo.get_memory(m_backfill_id, known_at=t_query)
    assert m_sim is not None
    assert m_sim.status == MemoryStatus.HISTORICAL
    assert m_sim.status != MemoryStatus.ACTIVE

    assert service.check_invariants().is_valid is True


def test_historical_refinement_belief_time_reconstruction(service: MemoryService) -> None:
    """Historical refinement of a closed memory preserves HISTORICAL status before refinement.

    Lifecycle:
    1. Jan 1: M1 ingested (Python 3.10, valid_from=Jan 1, ACTIVE).
    2. June 1: M2 ingested (Python 3.12, valid_from=June 1, ACTIVE).
       M1 becomes HISTORICAL via R-REPLACE with valid_to=June 1, closed_at=June 1.
    3. Oct 1: Refined historical claim arrives for [March 1, May 1) (Python 3.10.4).
       Triggers R-REFINE (historical refinement).
       M1 (before segment) updated to valid_to=March 1, closed_at=Oct 1.
       M3 (refined segment) inserted for [March 1, May 1), status=HISTORICAL.
       M4 (continuation segment) inserted for [May 1, June 1), status=HISTORICAL.

    Queries:
    - Query known_at = August 1 (after replacement at June 1, but before refinement at Oct 1):
      - M1 remains HISTORICAL (never incorrectly reconstructed as ACTIVE).
      - M1 closed_at == June 1 and valid_to == June 1 (reflects what was known on August 1).
      - M3 and M4 do not exist yet (recorded_at > August 1).
      - Point-in-time retrieval at April 15 returns M1 (Python 3.10).
    - Query known_at = November 1 (after refinement at Oct 1):
      - M1 is HISTORICAL with closed_at == Oct 1, valid_to == March 1.
      - M3 (Python 3.10.4) is visible over [March 1, May 1).
      - Point-in-time retrieval at April 15 returns M3 (Python 3.10.4).
    - Query known_at = April 1 (before replacement at June 1):
      - M1 is reconstructed as ACTIVE with closed_at=None, valid_to=None.
    """
    t_jan1 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    t_mar1 = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    t_apr1 = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    t_apr15 = datetime(2026, 4, 15, 12, 0, tzinfo=UTC)
    t_may1 = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    t_jun1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t_aug1 = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    t_oct1 = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
    t_nov1 = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)

    # 1. Ingest M1: Python 3.10 on Jan 1
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Python 3.10", event_time=t_jan1)
    claim1 = Claim(
        subject="Server A", attribute="python_version", value="3.10", extract_confidence=0.90
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t_jan1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # 2. Ingest M2: Python 3.12 on June 1 -> R-REPLACE closes M1
    msg2 = IngestMessage(source_id="sysadmin_priya", text="Python 3.12", event_time=t_jun1)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=t_jun1,
        extract_confidence=0.95,
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t_jun1)
    assert rep2.decisions[0].action == Action.REPLACE
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    m1_db_mid = service.repo.get_memory(m1_id)
    assert m1_db_mid is not None
    assert m1_db_mid.status == MemoryStatus.HISTORICAL
    assert m1_db_mid.valid_to == t_jun1
    assert m1_db_mid.closed_at == t_jun1

    # 3. Ingest historical refinement on Oct 1: Python 3.10.4 over [March 1, May 1)
    msg3 = IngestMessage(source_id="sysadmin_priya", text="Refined to 3.10.4", event_time=t_mar1)
    claim3 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10.4",
        valid_from=t_mar1,
        valid_to=t_may1,
        extract_confidence=0.95,
    )
    rep3 = service.ingest_claims(msg3, [claim3], recorded_at=t_oct1)
    assert rep3.decisions[0].action == Action.REFINE
    assert rep3.decisions[0].rule_id == "R-REFINE"
    m3_id = rep3.decisions[0].memory_id
    assert m3_id is not None

    # Verify DB state after refinement:
    # M1 valid_to shortened to March 1, closed_at updated to Oct 1
    m1_db_post = service.repo.get_memory(m1_id)
    assert m1_db_post is not None
    assert m1_db_post.status == MemoryStatus.HISTORICAL
    assert m1_db_post.valid_to == t_mar1
    assert m1_db_post.closed_at == t_oct1

    # =========================================================================
    # Window A: Query known_at = August 1 (between June 1 replacement & Oct 1 refinement)
    # =========================================================================
    # M1 must remain HISTORICAL, and must NOT be reconstructed as ACTIVE
    m1_aug = service.repo.get_memory(m1_id, known_at=t_aug1)
    assert m1_aug is not None
    assert m1_aug.status == MemoryStatus.HISTORICAL
    assert m1_aug.status != MemoryStatus.ACTIVE

    # closed_at and valid_to must reflect what was known on August 1 (pre-refinement values)
    assert m1_aug.closed_at == t_jun1
    assert m1_aug.valid_to == t_jun1

    # M2 is ACTIVE as of August 1
    m2_aug = service.repo.get_memory(m2_id, known_at=t_aug1)
    assert m2_aug is not None
    assert m2_aug.status == MemoryStatus.ACTIVE

    # Refined segment M3 does not exist yet as of August 1
    m3_aug = service.repo.get_memory(m3_id, known_at=t_aug1)
    assert m3_aug is None

    # Point-in-time retrieval at April 15 as known on August 1 returns M1 (Python 3.10)
    res_aug = service.current(
        "Server A", "python_version", valid_at=t_apr15, known_at=t_aug1, now=t_aug1
    )
    assert len(res_aug.items) == 1
    assert res_aug.items[0].memory.id == m1_id
    assert res_aug.items[0].memory.value_norm == "3.10"

    # get_current at August 1 returns M2 (Python 3.12)
    curr_aug = service.get_current("Server A", "python_version", now=t_aug1, known_at=t_aug1)
    assert curr_aug is not None
    assert curr_aug.id == m2_id

    # =========================================================================
    # Window B: Query known_at = November 1 (after historical refinement at Oct 1)
    # =========================================================================
    m1_nov = service.repo.get_memory(m1_id, known_at=t_nov1)
    assert m1_nov is not None
    assert m1_nov.status == MemoryStatus.HISTORICAL
    assert m1_nov.valid_to == t_mar1
    assert m1_nov.closed_at == t_oct1

    # Refined segment M3 is now visible as HISTORICAL
    m3_nov = service.repo.get_memory(m3_id, known_at=t_nov1)
    assert m3_nov is not None
    assert m3_nov.status == MemoryStatus.HISTORICAL
    assert m3_nov.value_norm == "3.10.4"
    assert m3_nov.valid_from == t_mar1
    assert m3_nov.valid_to == t_may1

    # Point-in-time retrieval at April 15 as known on November 1 returns M3 (Python 3.10.4)
    res_nov = service.current(
        "Server A", "python_version", valid_at=t_apr15, known_at=t_nov1, now=t_nov1
    )
    assert len(res_nov.items) == 1
    assert res_nov.items[0].memory.id == m3_id
    assert res_nov.items[0].memory.value_norm == "3.10.4"

    # Point-in-time retrieval at Feb 15 returns M1 (before segment)
    t_feb15 = datetime(2026, 2, 15, 12, 0, tzinfo=UTC)
    res_feb = service.current(
        "Server A", "python_version", valid_at=t_feb15, known_at=t_nov1, now=t_nov1
    )
    assert len(res_feb.items) == 1
    assert res_feb.items[0].memory.id == m1_id
    assert res_feb.items[0].memory.value_norm == "3.10"

    # =========================================================================
    # Window C: Query known_at = April 1 (before replacement at June 1)
    # =========================================================================
    m1_pre = service.repo.get_memory(m1_id, known_at=t_apr1)
    assert m1_pre is not None
    assert m1_pre.status == MemoryStatus.ACTIVE
    assert m1_pre.closed_at is None
    assert m1_pre.valid_to is None

    curr_pre = service.get_current("Server A", "python_version", now=t_apr1, known_at=t_apr1)
    assert curr_pre is not None
    assert curr_pre.id == m1_id
    assert curr_pre.value_display == "3.10"

    assert service.check_invariants().is_valid is True


def test_dispute_resolution_never_reconstructs_already_historical_memory_as_active(
    service: MemoryService,
) -> None:
    """A memory that became HISTORICAL before entering a dispute is not reconstructed as ACTIVE.

    Lifecycle:
    1. Jan 1: M1 ingested (Python 3.10, ACTIVE).
    2. Feb 1: M2 ingested (Python 3.12, ACTIVE).
       R-REPLACE closes M1. M1 becomes HISTORICAL (valid_to=Feb 1, closed_at=Feb 1).
    3. March 1: M3 arrives for [Jan 1, Feb 1) with close confidence to M1.
       Triggers historical dispute! M1 and M3 enter DISPUTED status.
    4. April 1: Dispute resolved in favor of M3.
       R-DISPUTE-RESOLVE references M1 in affected_memory_ids / loser_memory_ids.

    Reconstruction verification:
    - known_at = Feb 15 (between Feb 1 replacement and March 1 dispute):
      M1 was HISTORICAL, closed at Feb 1. Must NOT be reconstructed as ACTIVE.
    - known_at = March 15 (during historical dispute):
      M1 was DISPUTED.
    - known_at = April 15 (after dispute resolution):
      M1 remains closed/historical/rejected, NOT ACTIVE.
    - known_at = Jan 15 (before Feb 1 replacement):
      M1 was truly ACTIVE, reconstructed with closed_at=None, valid_to=None.
    """
    t_jan1 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    t_jan15 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    t_feb1 = datetime(2026, 2, 1, 10, 0, tzinfo=UTC)
    t_feb15 = datetime(2026, 2, 15, 12, 0, tzinfo=UTC)
    t_mar1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    t_mar15 = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
    t_apr1 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    t_apr15 = datetime(2026, 4, 15, 12, 0, tzinfo=UTC)

    # 1. Jan 1: M1 ingested (Python 3.10, ACTIVE)
    msg1 = IngestMessage(source_id="sysadmin_priya", text="Python 3.10", event_time=t_jan1)
    claim1 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.10",
        valid_from=t_jan1,
        extract_confidence=0.90,
    )
    rep1 = service.ingest_claims(msg1, [claim1], recorded_at=t_jan1)
    m1_id = rep1.decisions[0].memory_id
    assert m1_id is not None

    # 2. Feb 1: M2 ingested (Python 3.12, ACTIVE) -> R-REPLACE closes M1
    msg2 = IngestMessage(source_id="sysadmin_priya", text="Python 3.12", event_time=t_feb1)
    claim2 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.12",
        valid_from=t_feb1,
        extract_confidence=0.95,
    )
    rep2 = service.ingest_claims(msg2, [claim2], recorded_at=t_feb1)
    assert rep2.decisions[0].action == Action.REPLACE
    m2_id = rep2.decisions[0].memory_id
    assert m2_id is not None

    m1_post_rep = service.repo.get_memory(m1_id)
    assert m1_post_rep is not None
    assert m1_post_rep.status == MemoryStatus.HISTORICAL
    assert m1_post_rep.closed_at == t_feb1
    assert m1_post_rep.valid_to == t_feb1

    # 3. March 1: Historical claim arrives for [Jan 1, Feb 1) with close confidence
    msg3 = IngestMessage(source_id="ci_monitor", text="Python 3.8", event_time=t_jan1)
    claim3 = Claim(
        subject="Server A",
        attribute="python_version",
        value="3.8",
        valid_from=t_jan1,
        extract_confidence=0.90,
    )
    rep3 = service.ingest_claims(msg3, [claim3], recorded_at=t_mar1)
    assert rep3.decisions[0].action == Action.DISPUTE
    m3_id = rep3.decisions[0].memory_id
    assert m3_id is not None

    # 4. April 1: Resolve dispute in favor of M3
    res_log = service.resolve_dispute(winner_memory_id=m3_id, recorded_at=t_apr1)
    assert res_log.rule_id == "R-DISPUTE-RESOLVE"
    assert m1_id in res_log.affected_memory_ids

    # Query 1: known_at = Feb 15 (between Feb 1 replacement and March 1 dispute)
    # M1 MUST be HISTORICAL, and must NOT be reconstructed as ACTIVE
    m1_feb = service.repo.get_memory(m1_id, known_at=t_feb15)
    assert m1_feb is not None
    assert m1_feb.status == MemoryStatus.HISTORICAL
    assert m1_feb.status != MemoryStatus.ACTIVE
    assert m1_feb.closed_at == t_feb1
    assert m1_feb.valid_to == t_feb1

    # Query 2: known_at = March 15 (during historical dispute)
    m1_mar = service.repo.get_memory(m1_id, known_at=t_mar15)
    assert m1_mar is not None
    assert m1_mar.status == MemoryStatus.DISPUTED

    # Query 3: known_at = April 15 (after dispute resolution)
    m1_apr = service.repo.get_memory(m1_id, known_at=t_apr15)
    assert m1_apr is not None
    assert m1_apr.status == MemoryStatus.REJECTED
    assert m1_apr.closed_at == t_apr1
    assert m1_apr.valid_to == t_feb1

    # Query 4: known_at = Jan 15 (before Feb 1 replacement)
    # M1 was truly ACTIVE on Jan 15
    m1_jan = service.repo.get_memory(m1_id, known_at=t_jan15)
    assert m1_jan is not None
    assert m1_jan.status == MemoryStatus.ACTIVE
    assert m1_jan.closed_at is None
    assert m1_jan.valid_to is None

    assert service.check_invariants().is_valid is True
