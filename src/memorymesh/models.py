"""Core domain data models for MemoryMesh.

Governed by docs/SPECIFICATION.md Section 1. All models are Pydantic v2.
Timestamps are UTC datetime in Python, serialized to ISO-8601 UTC strings
matching YYYY-MM-DDTHH:MM:SSZ in SQLite and JSON.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer

# ==============================================================================
# Base Model with Timestamp Serialization
# ==============================================================================


def format_datetime_utc(dt: datetime) -> str:
    """Format datetime as YYYY-MM-DDTHH:MM:SSZ string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class MeshBaseModel(BaseModel):
    """Base model enforcing YYYY-MM-DDTHH:MM:SSZ datetime serialization for JSON."""

    @field_serializer("*", mode="wrap", check_fields=False)
    def serialize_datetime(self, value: Any, handler: Any) -> Any:
        if isinstance(value, datetime):
            return format_datetime_utc(value)
        return handler(value)


# ==============================================================================
# Enums
# ==============================================================================


class Cardinality(StrEnum):
    """Cardinality of an attribute."""

    SINGLE = "single"
    MULTI = "multi"


class ValueType(StrEnum):
    """Value types supported by normalizers and classifiers."""

    VERSION = "version"
    ENUM = "enum"
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"


class MemoryStatus(StrEnum):
    """Lifecycle statuses of a memory."""

    ACTIVE = "active"
    HISTORICAL = "historical"
    REJECTED = "rejected"
    DISPUTED = "disputed"


class Polarity(StrEnum):
    """Polarity of a claim."""

    ASSERT = "assert"
    RETRACT = "retract"


class Relation(StrEnum):
    """Relation between a claim value and an existing memory value."""

    IDENTICAL = "identical"  # same value (or less specific but compatible)
    REFINES = "refines"  # new value is a more specific version of old
    CONFLICTS = "conflicts"  # incompatible values on a single-valued attribute
    COMPLEMENTARY = "complementary"  # both can be true (multi-valued or additive)
    UNRELATED = "unrelated"


class Action(StrEnum):
    """Actions decided by the resolution engine."""

    INSERT = "INSERT"
    CORROBORATE = "CORROBORATE"
    REFINE = "REFINE"
    COEXIST = "COEXIST"
    REPLACE = "REPLACE"
    BACKFILL = "BACKFILL"
    REJECT = "REJECT"
    DISPUTE = "DISPUTE"
    RETRACT = "RETRACT"
    SKIP = "SKIP"
    PROPAGATE = "PROPAGATE"
    ACKNOWLEDGE = "ACKNOWLEDGE"


class EdgeType(StrEnum):
    """Typed relationship between memories."""

    SUPERSEDES = "supersedes"
    REFINES = "refines"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"


# ==============================================================================
# Catalog & Configuration Objects
# ==============================================================================


class Source(MeshBaseModel):
    """Information source definition and reliability rating."""

    id: str  # slug, e.g. "sysadmin_priya"
    name: str
    type: Literal["admin", "wiki", "tool", "chat", "user", "agent"]
    reliability: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class AttributeDef(MeshBaseModel):
    """Attribute catalog schema definition."""

    attribute: str  # snake_case, e.g. "python_version"
    description: str
    cardinality: Cardinality
    value_type: ValueType
    half_life_days: float | None = None  # None = facts never decay
    allowed_values: list[str] = Field(default_factory=list)  # for ENUM
    aliases: list[str] = Field(default_factory=list)  # phrases that map to this attribute


# ==============================================================================
# Extraction Models
# ==============================================================================


class AttrRef(MeshBaseModel):
    """Reference to an entity's attribute for dependency tracking."""

    entity: str
    attribute: str


class Claim(MeshBaseModel):
    """Raw output of the extractor, before resolution."""

    subject: str
    attribute: str
    value: str
    polarity: Polarity = Polarity.ASSERT
    valid_from: datetime | None = None  # None -> message event_time
    valid_to: datetime | None = None  # rare: "was true until March"
    context: dict[str, str] = Field(default_factory=dict)  # e.g. {"environment": "staging"}
    depends_on: list[AttrRef] = Field(default_factory=list)
    extract_confidence: float = Field(ge=0.0, le=1.0)
    span: str = ""  # exact substring of the message supporting the claim


class IngestMessage(MeshBaseModel):
    """Incoming raw text message for ingestion."""

    source_id: str
    text: str
    event_time: datetime  # when the message was written (world time)


class ResolvedClaim(MeshBaseModel):
    """Claim with normalized value, resolved entity, and context key."""

    claim: Claim
    entity_id: int
    attribute: str
    value_norm: str  # normalized for comparison
    value_display: str  # original-looking value for UI
    context_key: str  # canonical "k=v;k=v" sorted, "" for global
    valid_from: datetime  # filled in
    valid_to: datetime | None = None
    source: Source
    message_id: int


# ==============================================================================
# Stored Objects
# ==============================================================================


class Evidence(MeshBaseModel):
    """Stored link between a memory and the source/message that stated it."""

    id: int
    memory_id: int
    source_id: str
    message_id: int
    strength: float  # source.reliability * extract_confidence
    observed_at: datetime
    span: str = ""


class Memory(MeshBaseModel):
    """Core memory item stored in the database."""

    id: int
    entity_id: int
    entity_name: str = ""
    attribute: str
    value_norm: str
    value_display: str
    context: dict[str, str] = Field(default_factory=dict)
    context_key: str = ""
    valid_from: datetime
    valid_to: datetime | None = None
    recorded_at: datetime  # when the system learned it
    closed_at: datetime | None = None  # when the system learned it stopped being true
    status: MemoryStatus
    confidence: float  # base confidence derived from evidence
    needs_review: bool = False
    review_reason: str = ""
    raw_text: str = ""
    message_id: int
    source_id: str  # first/primary source
    evidence: list[Evidence] = Field(default_factory=list)


class Edge(MeshBaseModel):
    """Directed edge between two memories."""

    id: int
    src_memory_id: int
    dst_memory_id: int
    type: EdgeType
    reason: str = ""
    created_at: datetime


class DecisionLogEntry(MeshBaseModel):
    """Audit log entry for an ingestion or resolution decision."""

    id: int
    message_id: int
    claim_json: dict[str, Any]
    action: Action
    rule_id: str  # e.g. "R-REPLACE"
    memory_id: int | None = None  # primary memory affected/created
    affected_memory_ids: list[int] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)  # new_score, old_score, thresholds used
    explanation: str
    created_at: datetime


# ==============================================================================
# Engine I/O Models
# ==============================================================================


class NewMemory(MeshBaseModel):
    """Data required to insert a new memory record."""

    entity_id: int
    attribute: str
    value_norm: str
    value_display: str
    context: dict[str, str] = Field(default_factory=dict)
    context_key: str
    valid_from: datetime
    valid_to: datetime | None = None
    closed_at: datetime | None = None
    status: MemoryStatus
    confidence: float
    raw_text: str = ""
    message_id: int
    source_id: str


class NewEvidence(MeshBaseModel):
    """Data required to add an evidence link."""

    source_id: str
    message_id: int
    strength: float
    observed_at: datetime
    span: str = ""


class InsertMemory(MeshBaseModel):
    """Mutation to insert a new memory along with initial evidence."""

    memory: NewMemory
    evidence: NewEvidence


class UpdateMemory(MeshBaseModel):
    """Mutation to update fields on an existing memory."""

    memory_id: int
    fields: dict[str, Any] = Field(default_factory=dict)


class AddEvidence(MeshBaseModel):
    """Mutation to attach corroborating evidence to a memory."""

    memory_id: int
    evidence: NewEvidence


class AddEdge(MeshBaseModel):
    """Mutation to add an edge. $new indicates the newly inserted memory."""

    src: int | Literal["$new"]
    dst: int | Literal["$new"]
    type: EdgeType
    reason: str = ""


class RecomputeConfidence(MeshBaseModel):
    """Mutation instructing the confidence score to be recalculated."""

    memory_id: int | Literal["$new"]


Mutation = InsertMemory | UpdateMemory | AddEvidence | AddEdge | RecomputeConfidence


class Decision(MeshBaseModel):
    """Outcome of resolving a claim against candidate memories."""

    action: Action
    rule_id: str
    explanation: str
    scores: dict[str, float] = Field(default_factory=dict)
    mutations: list[Mutation] = Field(default_factory=list)
    affected_memory_ids: list[int] = Field(default_factory=list)


# ==============================================================================
# Query and Answer Models
# ==============================================================================


class QueryPlan(MeshBaseModel):
    """Parsed query intent and retrieval criteria."""

    intent: Literal["current", "as_of", "history", "why", "impact", "conflicts"]
    entity: str | None = None
    attribute: str | None = None
    valid_at: datetime | None = None
    known_at: datetime | None = None
    context: dict[str, str] = Field(default_factory=dict)
    memory_id: int | None = None


class MemoryView(MeshBaseModel):
    """Presentation view of a memory with computed effective confidence."""

    memory: Memory
    effective_confidence: float  # confidence * decay * review penalty
    is_stale: bool  # effective_confidence below stale floor
    source_names: list[str] = Field(default_factory=list)
    scope_match: Literal["exact", "inherited_global"]


class ImpactItem(MeshBaseModel):
    """Item flagged by impact analysis traversal."""

    memory: MemoryView
    path: list[int] = Field(default_factory=list)  # memory IDs from changed memory to dependent
    reason: str
    all_paths: list[list[int]] = Field(default_factory=list)
    paths_truncated: bool = False


class DisputeView(MeshBaseModel):
    """Two conflicting active memories in dispute."""

    a: MemoryView
    b: MemoryView
    edge_reason: str = ""


class QueryResult(MeshBaseModel):
    """Result of a query containing retrieved memories, disputes, and impact."""

    plan: QueryPlan
    items: list[MemoryView] = Field(default_factory=list)
    rejected_claims: list[MemoryView] = Field(default_factory=list)
    disputes: list[DisputeView] = Field(default_factory=list)
    impact: list[ImpactItem] = Field(default_factory=list)
    decisions: list[DecisionLogEntry] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class FinalAnswer(MeshBaseModel):
    """Final natural-language answer schema shared with naive-RAG baseline."""

    value: str | None = None
    values: list[str] = Field(default_factory=list)  # for multi-valued / history answers
    conflict_flagged: bool = False
    needs_review_flagged: bool = False
    abstained: bool = False
    explanation: str = ""
    cited_memory_ids: list[int] = Field(default_factory=list)


class AskResult(MeshBaseModel):
    """Top-level ask response containing query details and composed answer."""

    question: str
    result: QueryResult
    answer: FinalAnswer


class IngestReport(MeshBaseModel):
    """Report returned after ingesting a message."""

    message_id: int
    duplicate: bool = False
    claims_extracted: int
    decisions: list[DecisionLogEntry] = Field(default_factory=list)


class InvariantViolation(MeshBaseModel):
    """Structured violation reported by the invariant checker."""

    invariant_id: str
    description: str
    affected_ids: list[int] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class InvariantReport(MeshBaseModel):
    """Structured outcome of running data integrity invariant checks.

    total_checked represents the total count of primary domain records evaluated
    across memories, messages, edges, decision_log, and evidence. Reference and
    catalog tables (sources, attribute_schema, entities, entity_aliases) serve as
    reference targets and are excluded from this count.
    """

    is_valid: bool
    total_checked: int = 0
    violations: list[InvariantViolation] = Field(default_factory=list)
