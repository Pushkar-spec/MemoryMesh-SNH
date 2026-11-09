# MemoryMesh: Data Models & API Specification

Governs **data shapes**. Behavior is governed by `RESOLUTION_ENGINE.md`.

## 1. Core Data Models (`src/memorymesh/models.py`)

All structures are Pydantic v2. Timestamps are `datetime` in UTC in Python and serialized as `YYYY-MM-DDTHH:MM:SSZ` strings in SQLite and JSON.

```python
from __future__ import annotations
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

# ---------- Enums ----------
class Cardinality(StrEnum):
    SINGLE = "single"
    MULTI = "multi"

class ValueType(StrEnum):
    VERSION = "version"
    ENUM = "enum"
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"

class MemoryStatus(StrEnum):
    ACTIVE = "active"
    HISTORICAL = "historical"
    REJECTED = "rejected"
    DISPUTED = "disputed"

class Polarity(StrEnum):
    ASSERT = "assert"
    RETRACT = "retract"

class Relation(StrEnum):
    IDENTICAL = "identical"          # same value (or less specific but compatible)
    REFINES = "refines"              # new value is a more specific version of old
    CONFLICTS = "conflicts"          # incompatible values on a single-valued attribute
    COMPLEMENTARY = "complementary"  # both can be true (multi-valued or additive)
    UNRELATED = "unrelated"

class Action(StrEnum):
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

class EdgeType(StrEnum):
    SUPERSEDES = "supersedes"
    REFINES = "refines"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"

# ---------- Catalog & configuration objects ----------
class Source(BaseModel):
    id: str                                  # slug, e.g. "sysadmin_priya"
    name: str
    type: Literal["admin", "wiki", "tool", "chat", "user", "agent"]
    reliability: float = Field(ge=0.0, le=1.0)
    notes: str = ""

class AttributeDef(BaseModel):
    attribute: str                           # snake_case, e.g. "python_version"
    description: str
    cardinality: Cardinality
    value_type: ValueType
    half_life_days: Optional[float] = None   # None = facts never decay
    allowed_values: list[str] = []           # for ENUM
    aliases: list[str] = []                  # phrases that map to this attribute (extractor hint)

# ---------- Extraction ----------
class AttrRef(BaseModel):
    entity: str
    attribute: str

class Claim(BaseModel):
    """Raw output of the extractor, before resolution."""
    subject: str
    attribute: str
    value: str
    polarity: Polarity = Polarity.ASSERT
    valid_from: Optional[datetime] = None    # None -> message event_time
    valid_to: Optional[datetime] = None      # rare: "was true until March"
    context: dict[str, str] = {}             # e.g. {"environment": "staging"}
    depends_on: list[AttrRef] = []
    extract_confidence: float = Field(ge=0.0, le=1.0)
    span: str = ""                           # exact substring of the message supporting the claim

class IngestMessage(BaseModel):
    source_id: str
    text: str
    event_time: datetime                     # when the message was written (world time)

class ResolvedClaim(BaseModel):
    claim: Claim
    entity_id: int
    attribute: str
    value_norm: str                          # normalized for comparison
    value_display: str                       # original-looking value for UI
    context_key: str                         # canonical "k=v;k=v" sorted, "" for global
    valid_from: datetime                     # filled in
    valid_to: Optional[datetime]
    source: Source
    message_id: int

# ---------- Stored objects ----------
class Evidence(BaseModel):
    id: int
    memory_id: int
    source_id: str
    message_id: int
    strength: float                          # source.reliability * extract_confidence
    observed_at: datetime
    span: str = ""

class Memory(BaseModel):
    id: int
    entity_id: int
    entity_name: str = ""
    attribute: str
    value_norm: str
    value_display: str
    context: dict[str, str] = {}
    context_key: str = ""
    valid_from: datetime
    valid_to: Optional[datetime] = None
    recorded_at: datetime                    # when the system learned it
    closed_at: Optional[datetime] = None     # when the system learned it stopped being true
    status: MemoryStatus
    confidence: float                        # base confidence derived from evidence
    needs_review: bool = False
    review_reason: str = ""
    raw_text: str = ""
    message_id: int
    source_id: str                           # first/primary source
    evidence: list[Evidence] = []

class Edge(BaseModel):
    id: int
    src_memory_id: int
    dst_memory_id: int
    type: EdgeType
    reason: str = ""
    created_at: datetime

class DecisionLogEntry(BaseModel):
    id: int
    message_id: int
    claim_json: dict[str, Any]
    action: Action
    rule_id: str                             # e.g. "R-REPLACE"
    memory_id: Optional[int]                 # primary memory affected/created
    affected_memory_ids: list[int] = []
    scores: dict[str, float] = {}            # new_score, old_score, thresholds used
    explanation: str
    created_at: datetime
```

### 1.1 Engine I/O models (pure core)

```python
class NewMemory(BaseModel):
    entity_id: int
    attribute: str
    value_norm: str
    value_display: str
    context: dict[str, str]
    context_key: str
    valid_from: datetime
    valid_to: Optional[datetime]
    closed_at: Optional[datetime]
    status: MemoryStatus
    confidence: float
    raw_text: str
    message_id: int
    source_id: str

class NewEvidence(BaseModel):
    source_id: str
    message_id: int
    strength: float
    observed_at: datetime
    span: str = ""

# Mutations. "$new" as an ID means "the memory inserted by this Decision".
class InsertMemory(BaseModel):    memory: NewMemory; evidence: NewEvidence
class UpdateMemory(BaseModel):    memory_id: int; fields: dict[str, Any]   # status, valid_to, closed_at, needs_review, review_reason
class AddEvidence(BaseModel):     memory_id: int; evidence: NewEvidence
class AddEdge(BaseModel):         src: int | Literal["$new"]; dst: int | Literal["$new"]; type: EdgeType; reason: str
class RecomputeConfidence(BaseModel): memory_id: int | Literal["$new"]
Mutation = InsertMemory | UpdateMemory | AddEvidence | AddEdge | RecomputeConfidence

class Decision(BaseModel):
    action: Action
    rule_id: str
    explanation: str
    scores: dict[str, float] = {}
    mutations: list[Mutation] = []
    affected_memory_ids: list[int] = []      # memories whose validity/status changed (feeds propagation)
```

### 1.2 Query & answer models

```python
class QueryPlan(BaseModel):
    intent: Literal["current", "as_of", "history", "why", "impact", "conflicts"]
    entity: Optional[str] = None
    attribute: Optional[str] = None
    valid_at: Optional[datetime] = None
    known_at: Optional[datetime] = None
    context: dict[str, str] = {}
    memory_id: Optional[int] = None

class MemoryView(BaseModel):
    memory: Memory
    effective_confidence: float              # confidence * decay * review penalty
    is_stale: bool                           # effective_confidence below stale floor
    source_names: list[str]
    scope_match: Literal["exact", "inherited_global"]

class ImpactItem(BaseModel):
    memory: MemoryView
    path: list[int]                          # memory IDs from changed memory to this dependent
    reason: str

class DisputeView(BaseModel):
    a: MemoryView
    b: MemoryView
    edge_reason: str

class QueryResult(BaseModel):
    plan: QueryPlan
    items: list[MemoryView] = []             # resolved answers (usually 0 or 1 for single-valued)
    rejected_claims: list[MemoryView] = []   # surfaced for history/why intents only
    disputes: list[DisputeView] = []
    impact: list[ImpactItem] = []
    decisions: list[DecisionLogEntry] = []
    edges: list[Edge] = []
    notes: list[str] = []                    # e.g. "no memory valid at that time"

class InvariantViolation(BaseModel):
    invariant_id: str                        # e.g. "INV-NO-ACTIVE-OVERLAP"
    description: str
    affected_ids: list[int] = []
    details: dict[str, Any] = {}

class InvariantReport(BaseModel):
    is_valid: bool
    total_checked: int                       # total primary records (memories, messages, edges, decisions, evidence) audited; excludes catalog/reference tables
    violations: list[InvariantViolation] = []

class FinalAnswer(BaseModel):
    """Identical schema for MemoryMesh and the naive-RAG baseline (fair grading)."""
    value: Optional[str] = None
    values: list[str] = []                   # for multi-valued / history answers
    conflict_flagged: bool = False
    needs_review_flagged: bool = False
    abstained: bool = False
    explanation: str = ""
    cited_memory_ids: list[int] = []

class AskResult(BaseModel):
    question: str
    result: QueryResult
    answer: FinalAnswer

class IngestReport(BaseModel):
    message_id: int
    duplicate: bool = False
    claims_extracted: int
    decisions: list[DecisionLogEntry]
```

---

## 2. Service API (`src/memorymesh/service.py`)

The FastAPI layer, the eval harness, and scripts all use this facade. Nothing else is public. The React app never touches it directly; it calls the REST contract in section 7.

```python
class MemoryService:
    def __init__(self, db_path: str, cfg: AppConfig, llm: LLMClient | None = None): ...

    # Write
    # recorded_at = when the system learned it. Defaults to message.event_time (simulated live feed);
    # pass explicitly to model late-arriving information (belief-time queries depend on it).
    def ingest(self, message: IngestMessage, *, recorded_at: datetime | None = None) -> IngestReport:   # extract + resolve + apply
    def ingest_claims(self, message: IngestMessage, claims: list[Claim],
                      *, recorded_at: datetime | None = None) -> IngestReport:                         # bypasses LLM (structured mode)
    def resolve_dispute(self, winner_memory_id: int) -> DecisionLogEntry:           # human action
    def acknowledge_review(self, memory_id: int) -> None:                           # clear needs_review

    # Read
    def current(self, entity: str, attribute: str, context: dict[str, str] | None = None,
                valid_at: datetime | None = None, known_at: datetime | None = None, *, now: datetime) -> QueryResult
    def history(self, entity: str, attribute: str, context: dict[str, str] | None = None) -> QueryResult
    def explain(self, memory_id: int) -> QueryResult                                # evidence + decisions + edges
    def get_decision_logs(self, memory_id: int | None = None, limit: int | None = None,
                          offset: int = 0) -> list[DecisionLogEntry]
    def check_invariants(self) -> InvariantReport                                  # read-only integrity checks
    def impact(self, memory_id: int) -> list[ImpactItem]
    def disputes(self, entity: str | None = None) -> list[DisputeView]
    def ask(self, question: str, *, now: datetime) -> AskResult                     # parse -> retrieve -> compose
    def graph(self) -> "networkx.MultiDiGraph"

    # Admin
    def reset(self) -> None
    def load_catalogs(self) -> None                                                 # sources + attributes from YAML
```

`now` is always passed explicitly (see AGENTS.md time rules). The UI passes the real time; tests and the demo pass a fixed "demo now" (e.g. `2026-06-25T09:00:00Z`).

---

## 3. SQLite Schema (`src/memorymesh/schema.sql`)

```sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sources (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,
    reliability REAL NOT NULL CHECK (reliability BETWEEN 0 AND 1),
    notes       TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS entities (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL UNIQUE,
    type           TEXT NOT NULL DEFAULT 'thing',
    needs_review   INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_norm TEXT PRIMARY KEY,                    -- lowercase, punctuation stripped
    entity_id  INTEGER NOT NULL REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS attribute_schema (
    attribute           TEXT PRIMARY KEY,
    description         TEXT NOT NULL,
    cardinality         TEXT NOT NULL CHECK (cardinality IN ('single','multi')),
    value_type          TEXT NOT NULL CHECK (value_type IN ('version','enum','text','number','date')),
    half_life_days      REAL,
    allowed_values_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL REFERENCES sources(id),
    text        TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    event_time  TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    UNIQUE (source_id, text_hash, event_time)
);

CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id     INTEGER NOT NULL REFERENCES entities(id),
    attribute     TEXT NOT NULL REFERENCES attribute_schema(attribute),
    value_norm    TEXT NOT NULL,
    value_display TEXT NOT NULL,
    context_json  TEXT NOT NULL DEFAULT '{}',
    context_key   TEXT NOT NULL DEFAULT '',
    valid_from    TEXT NOT NULL,
    valid_to      TEXT,
    recorded_at   TEXT NOT NULL,
    closed_at     TEXT,
    status        TEXT NOT NULL CHECK (status IN ('active','historical','rejected','disputed')),
    confidence    REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    needs_review  INTEGER NOT NULL DEFAULT 0,
    review_reason TEXT NOT NULL DEFAULT '',
    raw_text      TEXT NOT NULL DEFAULT '',
    message_id    INTEGER NOT NULL REFERENCES messages(id),
    source_id     TEXT NOT NULL REFERENCES sources(id),
    CHECK (valid_to IS NULL OR closed_at IS NOT NULL),
    CHECK (valid_to IS NULL OR valid_to > valid_from)
);
CREATE INDEX IF NOT EXISTS idx_mem_lookup ON memories(entity_id, attribute, context_key, status);
CREATE INDEX IF NOT EXISTS idx_mem_valid  ON memories(valid_from, valid_to);

CREATE TABLE IF NOT EXISTS evidence (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    source_id   TEXT NOT NULL REFERENCES sources(id),
    message_id  INTEGER NOT NULL REFERENCES messages(id),
    strength    REAL NOT NULL CHECK (strength BETWEEN 0 AND 1),
    observed_at TEXT NOT NULL,
    span        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_ev_mem ON evidence(memory_id);

CREATE TABLE IF NOT EXISTS edges (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    src_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    dst_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    type          TEXT NOT NULL CHECK (type IN ('supersedes','refines','contradicts','depends_on')),
    reason        TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    UNIQUE (src_memory_id, dst_memory_id, type)
);
CREATE INDEX IF NOT EXISTS idx_edge_dst ON edges(dst_memory_id, type);
CREATE INDEX IF NOT EXISTS idx_edge_src ON edges(src_memory_id, type);

CREATE TABLE IF NOT EXISTS decision_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id          INTEGER NOT NULL REFERENCES messages(id),
    claim_json          TEXT NOT NULL,
    action              TEXT NOT NULL,
    rule_id             TEXT NOT NULL,
    memory_id           INTEGER REFERENCES memories(id),
    affected_ids_json   TEXT NOT NULL DEFAULT '[]',
    scores_json         TEXT NOT NULL DEFAULT '{}',
    explanation         TEXT NOT NULL,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id             TEXT PRIMARY KEY,
    timestamp      TEXT NOT NULL,
    prompt_id      TEXT NOT NULL,           -- extract_claims | classify_relation | parse_query | compose_answer | ...
    provider       TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_summary TEXT NOT NULL,
    response_raw   TEXT NOT NULL,
    parsed_json    TEXT,
    latency_ms     REAL NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('success','retry','error','cache_hit')),
    error          TEXT
);
```

---

## 4. Canonical Queries

### 4.1 "What is valid at time V, as believed at time K?"
```sql
SELECT m.*
FROM memories m
WHERE m.entity_id = :entity_id
  AND m.attribute = :attribute
  AND m.status != 'rejected'
  AND m.recorded_at <= :known_at
  AND m.valid_from  <= :valid_at
  AND (m.valid_to IS NULL OR m.valid_to > :valid_at OR m.closed_at > :known_at);
```
Defaults: `valid_at = known_at = now`. Scope filtering and specificity ranking happen in Python (`retrieval/context.py`), not SQL.

### 4.2 Candidate chain for the engine
```sql
SELECT * FROM memories
WHERE entity_id = :entity_id AND attribute = :attribute AND context_key = :context_key
  AND status != 'rejected'
ORDER BY valid_from ASC, id ASC;
```

### 4.3 Dependents of a memory (one hop)
```sql
SELECT src_memory_id FROM edges WHERE dst_memory_id = :id AND type = 'depends_on';
```

---

## 5. Config File Formats

### `config/engine.yaml`
```yaml
thresholds:
  min_extract_confidence: 0.40   # below -> SKIP (R-SKIP-LOWCONF)
  replace_margin: 0.80           # new_score >= old_score * margin -> REPLACE
  reject_ratio: 0.50             # new_score <  old_score * ratio  -> REJECT
  revert_margin: 1.00            # stricter replace margin when the new value equals a superseded value
  max_confidence: 0.99
  review_penalty: 0.50           # multiplier on confidence while needs_review
  stale_floor: 0.30              # effective confidence below this -> is_stale
  dispute_autoresolve_ratio: 0.50  # stretch S3
fuzzy:
  entity_auto_match: 90          # rapidfuzz token_set_ratio
  entity_tiebreak_min: 75
  text_duplicate_candidate: 70   # min ratio before asking the LLM classifier
ablation:                        # all true in production; eval flips them one at a time
  use_temporal: true
  use_trust: true
  use_cardinality: true
  use_context: true
  use_propagation: true
llm:
  provider: anthropic            # anthropic | gemini | fake  (env LLM_PROVIDER overrides)
  fast_model: claude-haiku-5-5   # extractor, classifier, parser  (verify against provider docs)
  answer_model: claude-sonnet-5-5
  concurrency: 3
  cache: true
```

### `config/sources.yaml`
```yaml
sources:
  - {id: sysadmin_priya, name: "Priya (Sysadmin)", type: admin, reliability: 0.90}
  - {id: ci_monitor,     name: "CI/Monitoring",    type: tool,  reliability: 0.85}
  - {id: wiki,           name: "Infra Wiki",       type: wiki,  reliability: 0.75}
  - {id: dev_chat,       name: "Dev Chat",         type: chat,  reliability: 0.50}
  - {id: intern_chat,    name: "Intern (Chat)",    type: chat,  reliability: 0.30}
```

### `config/attributes.yaml`
```yaml
attributes:
  - {attribute: python_version, description: "Python runtime version on a server", cardinality: single, value_type: version,
     aliases: ["python", "runtime", "py version"]}
  - {attribute: os_release, description: "Operating system and release", cardinality: single, value_type: text,
     aliases: ["os", "operating system", "ubuntu"]}
  - {attribute: owner_team, description: "Team that owns the system", cardinality: single, value_type: text,
     aliases: ["owned by", "owner"]}
  - {attribute: deploy_day, description: "Weekday on which deploys happen", cardinality: single, value_type: enum,
     allowed_values: [monday, tuesday, wednesday, thursday, friday, saturday, sunday], aliases: ["deploys on", "release day"]}
  - {attribute: service_status, description: "Operational status", cardinality: single, value_type: enum, half_life_days: 14,
     allowed_values: [up, down, degraded, maintenance], aliases: ["status", "is down", "is up"]}
  - {attribute: runs_service, description: "Services hosted on a server", cardinality: multi, value_type: text,
     aliases: ["hosts", "runs", "serves"]}
  - {attribute: compatibility_note, description: "Statement that something works with a given runtime", cardinality: single, value_type: text,
     aliases: ["works with", "compatible with", "needs"]}
```

### `config/entities.yaml` (curated aliases for the demo)
```yaml
entities:
  - {name: "Server A", type: server, aliases: ["srv-a", "server-a", "the main API box"]}
  - {name: "Library X", type: library, aliases: ["libx", "lib-x"]}
```

---

## 6. Value Normalization (`ingest/normalize.py`)

| `value_type` | Normalization | Invalid -> |
| :--- | :--- | :--- |
| `version` | Extract first `\d+(\.\d+)*` after trimming; `"v3.12"`, `"Python 3.12"` -> `"3.12"` | `SKIP` (`R-SKIP-VALUE`) |
| `enum` | Lowercase, strip, strip plural/`on `, e.g. `"Fridays"` -> `"friday"`; must be in `allowed_values` | `SKIP` |
| `text` | Collapse whitespace, lowercase for `value_norm`; keep original in `value_display` | never invalid |
| `number` | `float()` after stripping units; keep unit in display | `SKIP` |
| `date` | Parse to ISO date | `SKIP` |

**Context key:** sorted `"k=v"` pairs joined by `;`, lowercase. `{}` -> `""` (global scope).

---

## 7. REST API Contract (required; the React frontend depends on it)

Routes are thin: validate input, call `MemoryService`, return Pydantic models. **No logic in handlers.** Base path `/api`. JSON uses the timestamp format `YYYY-MM-DDTHH:MM:SSZ` (a custom Pydantic serializer enforces it). OpenAPI is served at `/openapi.json`; the frontend generates its types from it (`make types`).

### 7.1 Conventions

| Topic | Rule |
| :--- | :--- |
| Errors | `{"error": {"code": "string", "message": "string"}}`; codes: `not_found`, `validation_error`, `unknown_source`, `llm_unavailable`, `conflict`, `internal` |
| Status codes | 200/201 success, 404 not found, 422 validation, 502 LLM provider failure, 500 internal |
| `now` | Endpoints that depend on "now" accept an optional `now` (ISO) query/body field; default is server time. The UI exposes a **demo clock** in the header and sends it |
| Pagination | `limit` (default 100, max 500) and `offset` on list endpoints |
| Provider status | `GET /api/health` reports `llm_provider` (`anthropic`/`fake`...) so the UI can show an "offline demo mode" badge |

### 7.2 Endpoints

**Write**

| Method & Path | Body | Returns |
| :--- | :--- | :--- |
| `POST /api/ingest` | `IngestMessage` + optional `recorded_at`, optional `mode: "e2e"\|"structured"` with `claims` | `IngestReport` |
| `POST /api/memories/{id}/acknowledge` | none | `Memory` |
| `POST /api/disputes/resolve` | `{"winner_memory_id": int}` | `DecisionLogEntry` |
| `POST /api/demo/reset` | none | `{"ok": true}` |

**Read**

| Method & Path | Query / Body | Returns |
| :--- | :--- | :--- |
| `GET /api/health` | | `{"status": "ok", "llm_provider": str, "db": str}` |
| `GET /api/catalog` | | `{"sources": [Source], "attributes": [AttributeDef], "entities": [...], "thresholds": {...}, "ablation": {...}}` (powers the Settings page; read-only) |
| `GET /api/memories` | `entity, attribute, status, needs_review, limit, offset` | `list[MemoryView]` |
| `GET /api/memories/{id}` | | `MemoryView` |
| `GET /api/memories/{id}/explain` | | `QueryResult` (evidence, decisions, edges) |
| `GET /api/memories/{id}/impact` | | `list[ImpactItem]` |
| `GET /api/timeline` | `entity, attribute` (optional filters) | `list[SlotTimeline]` where `SlotTimeline = {entity, attribute, context_key, memories: [MemoryView], rejected: [MemoryView]}` |
| `GET /api/graph` | `entity` (optional) | `{"nodes": [GraphNode], "edges": [GraphEdge]}` (see 7.3) |
| `GET /api/disputes` | `entity` (optional) | `list[DisputeView]` |
| `GET /api/decisions` | `rule_id, message_id, limit, offset` | `list[DecisionLogEntry]` |
| `GET /api/llm/calls` | `prompt_id, status, limit` | `list[LLMCallLog]` |
| `POST /api/ask` | `{"question": str, "now": ISO?, "compare": bool}` | `AskResponse` (see 7.3) |
| `GET /api/eval/latest` | | `EvalSummary` (metrics table as JSON, run metadata) or 404 |
| `GET /api/eval/latest/report` | | `text/markdown` |

**Streaming (SSE, the only one)**

| Method & Path | Query | Events |
| :--- | :--- | :--- |
| `GET /api/demo/replay` | `scenario=hero-001`, `mode=structured\|e2e`, `delay_ms=800`, `reset=true` | `message_ingested` (data: `IngestReport`), `error` (data: error object), `done` (data: `{"messages": int}`) |

### 7.3 Additional response models

```python
class GraphNode(BaseModel):
    id: str                       # "entity:3" or "mem:12"
    kind: Literal["entity", "memory"]
    label: str
    status: Optional[MemoryStatus] = None
    confidence: Optional[float] = None
    needs_review: bool = False
    meta: dict[str, Any] = {}     # value, validity window, sources (for the side panel)

class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: Literal["about", "supersedes", "refines", "contradicts", "depends_on"]
    reason: str = ""

class BaselineHit(BaseModel):
    message_id: int
    source_name: str
    event_time: datetime
    text: str
    score: float

class AskResponse(BaseModel):
    mesh: AskResult                              # MemoryMesh answer + QueryResult
    baseline: Optional[FinalAnswer] = None       # present when compare=true
    baseline_retrieved: list[BaselineHit] = []

class LLMCallLog(BaseModel):
    id: str; timestamp: datetime; prompt_id: str; provider: str; model: str
    prompt_summary: str; response_raw: str; parsed_json: Optional[dict[str, Any]] = None
    latency_ms: float; status: Literal["success", "retry", "error", "cache_hit"]; error: Optional[str] = None

class EvalSummary(BaseModel):
    run_id: str; created_at: datetime; mode: str
    systems: list[str]                           # ["B0", "MM", "A1", ...]
    metrics: dict[str, dict[str, Optional[float]]]   # metric -> system -> value (None = n/a)
    scenarios: int; questions: int
```

### 7.4 Backend layout for the API (`src/memorymesh/api/`)

```text
api/
├── main.py          # app factory, lifespan (creates MemoryService), CORS, error handlers, router includes
├── deps.py          # get_service() dependency
├── errors.py        # exception -> {"error": {...}} mapping
└── routers/
    ├── write.py     # ingest, acknowledge, resolve, reset
    ├── read.py      # memories, timeline, graph, disputes, decisions, catalog, health
    ├── ask.py       # ask (+ baseline compare)
    ├── llm.py       # llm calls
    ├── eval.py      # eval results
    └── demo.py      # SSE replay
```

### 7.5 Frontend layout (`frontend/src/`)

```text
src/
├── main.tsx, App.tsx          # router + layout (sidebar, header with demo clock and provider badge)
├── api/
│   ├── types.ts               # GENERATED (make types), never edit by hand
│   ├── client.ts              # fetch wrapper, error mapping
│   ├── hooks.ts               # TanStack Query hooks per endpoint
│   └── replay.ts              # EventSource wrapper for /api/demo/replay
├── pages/
│   ├── IngestPage.tsx         MemoriesPage.tsx   GraphPage.tsx      TimelinePage.tsx
│   ├── AskPage.tsx            ReviewPage.tsx     DecisionsPage.tsx  InspectorPage.tsx
│   └── EvaluationPage.tsx     SettingsPage.tsx
├── components/                # StatusBadge, ConfidenceBar, DecisionCard, ProvenancePanel, ValidityBar, AnswerCard, EmptyState, ErrorState
└── lib/format.ts              # all date/number formatting
```
