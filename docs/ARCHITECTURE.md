# MemoryMesh: System Architecture

## 1. High-Level Concept

MemoryMesh is a **write-time reconciliation** memory. Where a plain vector store keeps every message and lets the model sort it out at answer time, MemoryMesh resolves conflicts when information arrives, stores the outcome as a **bitemporal knowledge graph**, and answers questions from the resolved state.

```mermaid
flowchart TD
    subgraph Input["Input"]
        Msg["Raw message + source + event time"]
    end

    subgraph Ingest["Ingest Pipeline (write path)"]
        Ext["1. Extractor (LLM)"]
        Norm["2. Normalizer + Entity Resolver"]
        Cand["3. Candidate Finder (SQL)"]
        Cls["4. Relation Classifier (rules, LLM for text)"]
        Eng["5. Resolution Engine (PURE)"]
        Apply["6. Apply Mutations (1 transaction)"]
        Prop["7. Dependency Propagator"]
        Log["8. Decision Log"]
    end

    subgraph Store["SQLite (source of truth)"]
        T1[("memories")]
        T2[("evidence")]
        T3[("edges")]
        T4[("decision_log")]
        T5[("messages / entities / sources")]
    end

    subgraph Query["Query Path (read path)"]
        QP["Query Parser (LLM)"]
        Ret["Retriever: validity + context + confidence"]
        Gr["NetworkX graph: impact / why"]
        Comp["Answer Composer (LLM, facts only)"]
    end

    UI["React UI (via FastAPI /api)"]

    Msg --> Ext --> Norm --> Cand --> Cls --> Eng --> Apply --> Prop --> Log
    Apply --> Store
    Prop --> Store
    Log --> Store
    Cand --> Store
    UI --> Msg
    UI --> QP --> Ret --> Comp --> UI
    Ret --> Store
    Ret --> Gr --> Store
```

---

## 2. Design Pattern: Functional Core, Imperative Shell

```mermaid
flowchart LR
    subgraph Shell["Imperative shell (I/O)"]
        S1["service.py"]
        S2["repository.py"]
        S3["llm/client.py"]
    end
    subgraph Core["Functional core (pure, no I/O)"]
        C1["engine/rules.py: resolve()"]
        C2["engine/confidence.py"]
        C3["engine/propagate.py"]
        C4["engine/classifier.py (rules part)"]
    end
    S1 -->|"claim + candidates + config"| C1
    C1 -->|"Decision (list of Mutations)"| S1
    S1 -->|"apply in one transaction"| S2
```

Why: the part that must be **correct and explainable** is testable with plain Python objects and no database, clock, or network. A bug in a decision becomes a failing unit test, not a mystery.

---

## 3. Write Path Contracts

Each stage has one job and a typed input/output (models in `SPECIFICATION.md`).

| # | Stage | Input | Output | Rules |
| :-: | :--- | :--- | :--- | :--- |
| 1 | Extractor | `IngestMessage` | `list[Claim]` | LLM, constrained to attribute catalog; may return `[]` |
| 2 | Normalizer / Entity Resolver | `Claim` | `ResolvedClaim` or `SKIP` | Normalize value by `value_type`; resolve subject to `entity_id`; unknown attribute -> `SKIP` |
| 3 | Candidate Finder | `ResolvedClaim` | `chain: list[Memory]` | All non-rejected memories for `(entity, attribute, context_key)` sorted by `valid_from`, with evidence |
| 4 | Relation Classifier | `(ResolvedClaim, Memory)` | `Relation` | Deterministic by value type; LLM only for `text` values |
| 5 | Resolution Engine | claim, chain, schema, source, config | `Decision` | Pure. Rules in `RESOLUTION_ENGINE.md` |
| 6 | Apply | `Decision` | DB state | Single transaction; inserts, updates, evidence, edges |
| 7 | Propagator | changed memory IDs + graph | `list[Mutation]` | BFS over reversed `depends_on` edges; flags `needs_review` |
| 8 | Decision Log | `Decision` + propagation | `decision_log` row | Always written, even for `SKIP` |

**Idempotency:** `messages` has a unique key on `(source_id, text_hash, event_time)`. A duplicate returns the earlier `IngestReport` and changes nothing.

**Ordering:** messages are processed in the order given, but the engine reasons by `valid_from`, not arrival order. Late-arriving older facts become `BACKFILL` (see engine doc).

---

## 4. Read Path

| Query intent | Parser output | Retrieval behavior |
| :--- | :--- | :--- |
| `current` | entity, attribute, context | Memories valid *now*, scope-matched, not rejected; most specific scope wins |
| `as_of` | + `valid_at` | Same, evaluated at `valid_at` |
| `history` | entity, attribute | All non-rejected memories ordered by `valid_from`; rejected claims returned separately as "rejected claims" |
| `why` | memory or entity+attribute | Evidence rows, source reliabilities, linked decision log entries, edges |
| `impact` | memory or entity+attribute | Reverse `depends_on` traversal (descendants) with reasons |
| `conflicts` | optional entity | All `disputed` memories grouped by contradiction pair |

**Belief-time queries** (optional parameter `known_at`): "what did we *believe* on 2 June?" uses `recorded_at <= known_at` and treats `valid_to` as unset when `closed_at > known_at`. See `SPECIFICATION.md` section 4.

**Composer boundary:** the LLM receives only the `QueryResult` JSON. It may not add facts. Cited memory IDs are validated against the IDs it was given; unknown IDs are stripped and logged.

---

## 4b. API Layer & Frontend Boundary

```mermaid
flowchart LR
    subgraph Browser["Browser (React + TS)"]
        Pages["Pages"] --> Hooks["TanStack Query hooks (src/api)"]
    end
    Hooks -->|"JSON / SSE over /api"| Routers
    subgraph Backend["FastAPI"]
        Routers["Routers (validate only)"] --> Svc["MemoryService"]
    end
    Svc --> Core["Engine + Repository + LLM client"]
    Backend -. "OpenAPI schema" .-> Types["openapi-typescript -> src/api/types.ts"]
    Types -.-> Hooks
```

- **Single process owner:** the API process owns the SQLite connection and the `MemoryService` instance (created at startup via FastAPI lifespan). Single-writer is acceptable; use a lock around ingest.
- **Contract-first:** Pydantic models in `models.py` are the API schemas. Change a model, regenerate types, fix the TypeScript compile errors; the compiler lists what the UI must change.
- **Streaming:** only one place uses SSE, the scenario **replay** (`GET /api/demo/replay`), so the UI can show decision cards appearing message by message. Everything else is plain request/response.
- **CORS / dev proxy:** Vite proxies `/api` to `localhost:8000`; the API also allows `http://localhost:5173` for direct calls.
- **Errors:** every failure returns `{"error": {"code": "...", "message": "..."}}` with an appropriate HTTP status; the UI shows `message`.

---

## 5. Memory Status State Machine

```mermaid
stateDiagram-v2
    [*] --> active: R-INSERT / R-COEXIST / R-REPLACE (new)
    [*] --> historical: R-BACKFILL (new older fact)
    [*] --> rejected: R-REJECT
    [*] --> disputed: R-DISPUTE (new)
    active --> historical: replaced / refined / truncated
    active --> disputed: R-DISPUTE (existing side)
    disputed --> active: resolve_dispute(winner)
    disputed --> rejected: resolve_dispute(loser)
    rejected --> [*]
    historical --> [*]
```

`needs_review` is an **orthogonal flag**, not a status: a memory can be `active` and `needs_review` (a dependency changed underneath it).

---

## 6. Graph Model

SQLite stores relational rows; NetworkX is a **view** built from them for traversal and visualization.

- **Nodes:** `entity:{id}` and `mem:{id}` with attributes (status, confidence, value, validity).
- **Edges:**
  - `about`: `mem -> entity` (implicit, from `entity_id`)
  - `supersedes`: new memory -> old memory
  - `refines`: refined memory -> coarser memory
  - `contradicts`: new memory -> existing memory (stored once, treated as symmetric)
  - `depends_on`: dependent memory -> the memory it assumes

Impact analysis = `networkx.ancestors(G_depends, changed_node)` on the subgraph of `depends_on` edges (edges point dependent -> foundation, so dependents are ancestors).

---

## 7. LLM Wrapper and Client Infrastructure (P7.1)

```mermaid
flowchart TD
    Task["Caller (extractor / classifier / parser / composer)"] --> Cache{"Cache lookup: SHA-256(provider, model, system, user, schema, max_tokens)"}
    Cache -- Hit --> Ret["Return cached validated model copy"]
    Cache -- Miss --> Sem["Acquire Semaphore(concurrency)"]
    Sem --> Prov["Provider complete() (Anthropic / Gemini / Fake)"]
    Prov --> Release["Release Semaphore"]
    Release --> Parse["Extract JSON & Pydantic validate"]
    Parse -- OK --> LogSuccess["Write llm_calls row (status=success)"]
    LogSuccess --> Store["Store in process-local cache"]
    Store --> Ret
    Parse -- Fail --> LogRetry["Write llm_calls row (status=retry)"]
    LogRetry --> Repair["Construct repair directive with schema & error details"]
    Repair --> SemRetry["Acquire Semaphore(concurrency)"]
    SemRetry --> ProvRetry["Provider complete() repair call"]
    ProvRetry --> ReleaseRetry["Release Semaphore"]
    ReleaseRetry --> ParseRetry["Extract JSON & Pydantic validate"]
    ParseRetry -- OK --> LogSuccessRetry["Write llm_calls row (status=success)"]
    LogSuccessRetry --> Store
    ParseRetry -- Fail --> LogError["Write llm_calls row (status=error)"]
    LogError --> RaiseError["Raise LLMOutputError"]
```

### Core Components

1. **`LLMClient.complete_json()`**:
   - Centralized, typed API accepting `prompt_id`, `system`, `user`, `response_model: type[T]`, `model`, `max_tokens`.
   - Returns a validated Pydantic model instance `T`.
   - Agnostic to underlying providers via the `LLMProvider` protocol (`complete(system, user, *, model, max_tokens) -> str`).

2. **JSON Extraction and Validation (`extract_json_payload`)**:
   - Supports plain JSON objects/arrays, surrounding whitespace, and Markdown fences (```` ```json ```` or ```` ``` ````).
   - Handles nested structures cleanly via bracket depth tracking without fragile regular expression truncation.
   - Validates extracted dictionaries strictly against caller-supplied Pydantic models.

3. **Exactly One Retry with Repair Directive**:
   - Triggered when provider output fails JSON extraction or Pydantic validation.
   - Initial call is logged with `status="retry"`.
   - Sends a targeted repair directive containing the validation error details and schema specification.
   - If the repair attempt fails again, logs `status="error"` and raises `LLMOutputError`. Indefinite retries are prevented.

4. **Deterministic SHA-256 Cache**:
   - Keyed on `SHA-256(provider, model, system, user, schema_json, max_tokens)`.
   - In-memory, process-local storage (`dict[str, BaseModel]`) protected by a thread lock. Resets on process restart.
   - Cache hits return a deep copy of the validated model without acquiring concurrency slots or logging calls.
   - Failed responses are never cached.

5. **Concurrency Control**:
   - Enforced synchronously via `threading.Semaphore` with configurable limits (`concurrency`, default 3).
   - Concurrency slots are acquired strictly during provider completions and released immediately in a `try...finally` block.
   - Cache hits bypass semaphore acquisition entirely.

6. **Audit Logging (`llm_calls` Table) and Sensitive Data Sanitization**:
   - Every physical provider invocation (initial and repair) is recorded in SQLite via `Repository.insert_llm_call()`.
   - Logs timestamp, prompt ID, provider, model, sanitized prompt summary, raw response, parsed JSON, latency (ms), status (`success`, `retry`, `error`), and error details.
   - Sensitive credentials (API keys, bearer tokens, passwords, private keys, authorization headers) are automatically sanitized and redacted across all logged fields and error diagnostics.
   - Persisted diagnostic information is strictly bounded (`prompt_summary` <= 120 chars, `response_raw` <= 1000 chars, `parsed_json` <= 1000 chars, `error` <= 500 chars).
   - `LLMOutputError` automatically sanitizes `raw_response` and `details` to ensure credentials are never leaked to callers or application logs.
   - **Provider Exception Security Note**: When an external provider raises a transport or network exception, `LLMClient` re-raises the original provider exception unchanged to preserve provider exception contracts and avoid conflating transport failures with output errors. The persisted database record in `llm_calls.error` is sanitized and redacted, but the caller-facing exception object itself retains whatever exception string the external provider created.

7. **Offline Testing with `FakeProvider`**:
   - Tests execute with zero network access and deterministic responses using `FakeProvider` or `SequenceProvider`.
   - Offline verification:
     ```bash
     export LLM_PROVIDER=fake
     make smoke
     pytest tests/test_llm_client.py
     ```

---

## 8. Failure Modes & Handling

| Failure | Handling |
| :--- | :--- |
| LLM returns invalid JSON twice | `LLMOutputError`; message stored, `decision_log` row with `R-SKIP-LLMFAIL`; ingestion continues with next message |
| Extractor invents an unknown attribute | `R-SKIP-ATTR`; never auto-create attributes |
| Entity cannot be matched confidently | New entity created with `needs_review=1`; shown in UI |
| Classifier LLM unavailable for `text` values | Default relation `CONFLICTS`; explanation says "classifier unavailable" |
| DB write error mid-claim | Transaction rolls back; claim reported as failed; DB remains consistent |
| Two systems write at once | Out of scope; single-writer SQLite with WAL |

---

## 9. Architecture Decision Records

| ID | Decision | Rationale | Trade-off |
| :--- | :--- | :--- | :--- |
| **ADR-001** | SQLite + NetworkX, not Neo4j | Zero setup; temporal queries are plain SQL; graph is small | No native graph queries; fine at demo scale |
| **ADR-002** | Pure decision engine | Testable, explainable, repeatable | Slightly more plumbing (Decision/Mutation objects) |
| **ADR-003** | No LangGraph / agent framework | The pipeline is linear and deterministic; the only human-in-loop step (`resolve_dispute`) is a single service call | Less "agentic" branding; much less to debug |
| **ADR-004** | React + TypeScript + Vite frontend | Polished, interactive demo (graph, timeline, side-by-side compare); types generated from the API | More moving parts than a Python-only UI; mitigated by a strict API contract and generated types |
| **ADR-005** | Vectorless write path (SQL + rapidfuzz) | Entity/attribute matching is structured; embeddings add install risk and opacity | Fuzzy aliases need curated alias table or LLM tie-break |
| **ADR-006** | Confidence derived from evidence (noisy-OR), recomputed not mutated | Auditable; corroboration cannot be double-counted | Slightly more compute per write |
| **ADR-007** | One row per claim with `valid_to`/`closed_at`, not full row versioning | Simple, still supports belief-time queries | Approximates full bitemporal history; documented limits |
| **ADR-008** | Naive-RAG baseline uses `fastembed` dense retrieval | A fair baseline must be a standard RAG, not a strawman | Extra dependency used only in eval |
| **ADR-009** | FastAPI layer is required, with no logic of its own | The React app needs an HTTP contract; keeping routes thin preserves ADR-002 | One more layer to maintain; routes are trivial by rule |

---

## 10. Non-Goals (Explicit)

- Future-dated / planned facts (e.g. "will migrate on 1 December")
- Multi-user concurrent writes
- Arbitrary open-ended schemas (attributes come from `config/attributes.yaml`)
- Automatic learning of source reliability from outcomes (reliabilities are configured and visible)
- Fine-tuned NLI models (optional stretch only)
