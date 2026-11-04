# MemoryMesh — Master Implementation Plan

**Canonical plan: 10 phases, 36 implementation batches.**

This document is the single source of truth for the implementation sequence. Use the phase counts and batch IDs exactly as written here. Do not use earlier conflicting counts.

## 1. Canonical phase and batch count

| Phase | Name | Batches |
|---|---|---:|
| P0 | Foundation and setup | 4 |
| P1 | Temporal memory core | 3 |
| P2 | Cardinality and classification | 2 |
| P3 | Evidence, confidence and trust | 3 |
| P4 | Decision logging and explainability | 1 |
| P5 | Dependencies and impact | 3 |
| P6 | Context, backfill and retraction | 2 |
| P7 | LLM integration | 5 |
| P8 | Baseline and evaluation | 4 |
| P9 | API and React UI | 7 |
| P10 | Polish | 2 |
| **Total** | | **36** |

The phase counts add up to 36: 4 + 3 + 2 + 3 + 1 + 3 + 2 + 5 + 4 + 7 + 2 = 36.

## 2. Implementation batches

### P0 — Foundation and setup (4 batches)

- **P0.1 — Repository and development tooling**
  - Establish the repository structure, Python project configuration and dependencies.
  - Configure Makefile targets, Git hygiene, environment examples, Ruff and Pytest.
  - Acceptance: setup instructions work; lint/test commands are available; no application feature logic is added prematurely.
- **P0.2 — Configuration and typed settings**
  - Add YAML configuration files and typed configuration loading.
  - Keep tunable engine thresholds in configuration rather than hardcoding them.
  - Acceptance: configuration loads with validation and sensible documented defaults.
- **P0.3 — SQLite foundation and domain models**
  - Implement the SQLite schema, connection/database setup, WAL and foreign-key configuration.
  - Add Pydantic v2 domain models and repository foundations as specified.
  - Acceptance: schema and model tests pass; SQLite remains the source of truth.
- **P0.4 — LLM provider foundation and frontend scaffold**
  - Add provider interfaces and implementations: Anthropic provider, Gemini stub and FakeProvider; include a smoke test.
  - Scaffold React 18, TypeScript strict, Vite, Tailwind, React Router, TanStack Query and React Flow.
  - Add frontend proxy and required `web`, `web-check` and `types` commands.
  - Acceptance: provider smoke test works without requiring a real network call when using FakeProvider; frontend starts and type/check commands run.

### P1 — Temporal memory core (3 batches)

- **P1.1 — Repository, transactions and basic decision rules**
  - Implement repository methods for memories, evidence and edges, plus transaction management.
  - Implement pure `R-INSERT` and `R-REPLACE` rules.
  - Acceptance: rule tests cover insertion, replacement and validity-window boundaries.
- **P1.2 — Temporal retrieval and atomic structured ingestion**
  - Implement current and as-of retrieval.
  - Implement `service.ingest_claims` and atomic application of structured decisions.
  - Acceptance: the Server A Python 3.10 → 3.12 scenario returns 3.12 currently and 3.10 for the earlier period; `valid_at == valid_to` follows the specification.
- **P1.3 — CLI demonstration**
  - Add a small CLI demonstration of the structured temporal-memory flow.
  - This is a supplementary implementation batch for usability; it is not an original numbered task in the initial roadmap.
  - Acceptance: the demo runs from documented commands and prints deterministic structured output.

### P2 — Cardinality and classification (2 batches)

- **P2.1 — Normalization and deterministic relation classification**
  - Implement value normalization and deterministic relation classification.
  - Acceptance: equivalent values normalize consistently; classifier tests cover supported relation types.
- **P2.2 — Multi-valued attributes, refinement and entity resolution**
  - Implement multi-valued coexistence, refinement and entity resolution.
  - Acceptance: single-valued and multi-valued attributes behave differently as specified; refinement and entity-resolution tests pass.

### P3 — Evidence, confidence and trust (3 batches)

- **P3.1 — Evidence and confidence calculations**
  - Implement evidence strength, noisy-OR aggregation, confidence calculations/decay where specified, corroboration and duplicate-source handling.
  - Acceptance: calculations match the examples and formulas in `RESOLUTION_ENGINE.md`; repeated evidence from the same source is handled according to the specification.
- **P3.2 — Trust-based replacement, rejection, dispute and revert guards**
  - Implement trust-aware replacement, rejection and dispute decisions, confidence margins, and protections against unsafe reverts/history conflicts.
  - Acceptance: high-trust and low-trust claims produce the specified outcomes; edge cases are covered by tests.
- **P3.3 — Dispute resolution service**
  - Implement `service.resolve_dispute`.
  - Acceptance: dispute resolution updates state and audit history atomically and follows the documented rules.

### P4 — Decision logging and explainability (1 batch)

- **P4.1 — Decision log, explain service and invariant checks**
  - Log every claim decision, including SKIPs, in the same transaction as its mutations.
  - Implement `service.explain(memory_id)` with evidence, decision history, rules and relevant edges.
  - Add invariant checks to service tests.
  - Acceptance: explanations identify the supporting evidence/rule/reasoning; all documented data-integrity invariants are tested.

### P5 — Dependencies and impact (3 batches)

- **P5.1 — Dependency resolution and propagation**
  - Resolve `depends_on` references to memory IDs, create dependency edges, and implement propagation in the engine/service.
  - Log propagation decisions using the documented rule.
  - Acceptance: changes to a depended-on memory flag affected dependent memories for review.
- **P5.2 — Impact analysis and review acknowledgement**
  - Implement read-only `service.impact(memory_id)` traversal with paths.
  - Implement `service.acknowledge_review`.
  - Acceptance: impact results show the dependency path; acknowledgement updates review state without changing historical facts.
- **P5.3 — G1 structured hero demonstration**
  - Build the structured hero-scenario demonstration in `scripts/seed_demo.py`.
  - Acceptance: the demo prints the specified q1–q7 answers and demonstrates temporal retrieval, explanation and dependency impact.

### P6 — Context, backfill and retraction (2 batches)

- **P6.1 — Context scopes and backfill**
  - Implement context-key canonicalization, scope matching/specificity and `R-BACKFILL`.
  - Acceptance: context-specific queries select the right memory; the documented staging example returns 3.11.
- **P6.2 — Retraction, belief-time queries and idempotency**
  - Implement `R-RETRACT`, `R-SKIP-NOTARGET`, `known_at` queries, `recorded_at` ingestion parameters and idempotent ingestion.
  - Acceptance: retractions and no-target cases are logged; belief-time queries distinguish what was believed from what was recorded; repeated identical ingestion is a no-op.

### P7 — LLM integration (5 batches)

- **P7.1 — LLM client infrastructure**
  - Implement the provider-agnostic `LLMClient.complete_json()` layer, concurrency limit, SHA-256 cache, JSON extraction, exactly one retry, typed errors and `llm_calls` logging.
  - Acceptance: FakeProvider tests run with zero network access; malformed output and retry behavior are tested.
- **P7.2 — Extraction prompts and LLM-assisted relation classification**
  - Implement prompts from `PROMPTS.md`, extraction and post-processing, plus LLM-backed classification for text values with deterministic fallbacks.
  - Acceptance: outputs are schema-validated; the LLM suggests/classifies but never decides REPLACE, REJECT, DISPUTE or COEXIST.
- **P7.3 — Query parser and answer composer**
  - Implement query parsing and answer composition with post-validation, including removal of unsupported/hallucinated citations.
  - Acceptance: parser/composer tests validate structured outputs and citation references.
- **P7.4 — End-to-end natural-language ingest and ask**
  - Implement full `service.ingest` and `service.ask` flows.
  - Acceptance: the hero scenario matches the structured engine's expected decisions and answers.
- **P7.5 — Entity-resolution LLM tie-break (stretch)**
  - Add an LLM tie-break for ambiguous entity matches only if the core system is stable and the documented constraints are preserved.
  - Acceptance: deterministic fallback remains available; no resolution-engine authority is delegated to the LLM.

### P8 — Baseline and evaluation (4 batches)

- **P8.1 — Naive RAG baseline and evaluation runner**
  - Implement the fastembed top-5 naive RAG baseline and `baseline_answer`.
  - Implement the evaluation runner, metrics and supported ablation flags.
  - Acceptance: both MemoryMesh and baseline can be evaluated with repeatable outputs.
- **P8.2 — Evaluation scenarios: core update and trust cases**
  - Add scenarios for simple updates, corroboration, low-trust rejection, disputes and multi-valued attributes.
  - Acceptance: scenario coverage and expected outcomes are explicit and reproducible.
- **P8.3 — Evaluation scenarios: context, ordering, dependencies and time**
  - Add scenarios for context, out-of-order events, dependencies and time-travel queries.
  - Acceptance: all required categories are represented; include designed-to-lose cases for honest comparison.
- **P8.4 — Evaluation report and G2 gate**
  - Generate `report.md` and `results.json`; complete the G2 evaluation gate.
  - Acceptance: report contains metrics, scenario coverage and baseline comparison; results can be regenerated from the documented command.

### P9 — API and React UI (7 batches)

- **P9.1 — FastAPI backend**
  - Implement the API per `SPECIFICATION.md`, including lifespan/service creation, CORS, error handling, routers, OpenAPI and tests.
  - Acceptance: API tests pass and OpenAPI schema is generated.
- **P9.2 — Frontend shell and API client**
  - Implement the app shell, sidebar, header, demo clock, provider badge, reset action, API client, TanStack Query hooks and shared components.
  - Acceptance: frontend uses only the API layer for HTTP and has loading/error/empty states.
- **P9.3 — Ingest, memories, provenance and Ask**
  - Implement ingest and decision cards, memory browsing/filters, provenance drawer, Ask UI and naive-RAG comparison.
  - Acceptance: these flows work against the API and show citations/provenance.
- **P9.4 — Graph and timeline**
  - Implement the React Flow dependency graph and timeline with validity bars and rejected-memory markers.
  - Acceptance: graph and timeline reflect backend data; no business rules are implemented in the browser.
- **P9.5 — Review and decision history**
  - Implement dispute review/winner selection, review acknowledgement and decision-log filtering.
  - Acceptance: actions call backend endpoints and refresh relevant query data.
- **P9.6 — SSE replay**
  - Implement the replay endpoint and frontend replay flow for the hero scenario.
  - Acceptance: replay events are delivered in order and the UI handles connection/error states.
- **P9.7 — LLM Inspector, Evaluation and Settings**
  - Implement the LLM Inspector, Evaluation and Settings pages.
  - Acceptance: pages show backend-backed data and complete loading/error/empty states; settings do not expose secrets.

### P10 — Polish (2 batches)

- **P10.1 — Documentation, clean-clone setup and final technical checks**
  - Finalize README quick start, `.env.example`, setup instructions, clean Git history and secret checks.
  - Run the full required check suite.
  - Acceptance: a clean clone can be set up using documented commands; `make check` passes; no secrets are committed.
- **P10.2 — Demo assets and pitch**
  - Record the screencast, prepare slides with actual evaluation results, and rehearse the pitch.
  - Acceptance: demo is reproducible and the pitch fits within three minutes.

## 3. Rules for every coding-agent batch

1. Work on exactly one batch ID at a time.
2. Read `AGENTS.md` and the relevant architecture, specification, resolution-engine, prompt and roadmap documentation before editing.
3. Treat `RESOLUTION_ENGINE.md` as authoritative for behavior and `SPECIFICATION.md` as authoritative for data shapes.
4. Write or update tests before implementation. Do not weaken tests or change expected behavior without documented justification.
5. Make the smallest change that completes the batch. Do not implement future batches or add unapproved dependencies/features.
6. Keep engine logic pure; keep SQL in `repository.py`; keep HTTP orchestration in service/API layers as documented.
7. Run the relevant tests and `make check` where feasible. The project owner independently runs the reported commands.
8. Report: batch ID, files changed, behavior implemented, tests/commands and results, assumptions, known limitations, and work deliberately not done.
9. Do not proceed to the next batch until the supervisor reviews and approves the current batch.
10. Commit each approved batch using a Conventional Commit message.

## 4. Architectural constraints that must not be violated

- SQLite is the source of truth. NetworkX is a graph built from SQLite data, not a replacement database.
- Do not add Neo4j, a vector database, LangChain, LangGraph or an ORM.
- The engine must not call the LLM. The LLM must not make final REPLACE, REJECT, DISPUTE or COEXIST decisions.
- All SQL belongs in `repository.py`; do not put SQL in routes or engine modules.
- Keep thresholds in configuration, not hardcoded constants.
- Pass time values into pure engine/retrieval functions; do not call `datetime.now()` inside them.
- Every decision, including SKIP, must be auditable.
- Preserve the documented data-integrity invariants and idempotency requirements.
- Frontend business logic stays on the backend; frontend HTTP calls go through `src/api/` and TanStack Query.
- Use strict TypeScript and generated OpenAPI types; do not introduce another UI or state-management framework.

## 5. Execution gates

- **G1 — After P5.3:** structured hero demo must demonstrate temporal facts, explanation and dependency impact.
- **G2 — After P8.4:** evaluation report and machine-readable results must be generated and reviewed.
- **Final gate — After P10.2:** full checks pass, demo is reproducible, evaluation claims match actual results, and pitch is rehearsed.

## 6. Scope-cut order if time becomes limited

Cut in this order, only after considering the impact:
1. Settings page and draggable as-of timeline cursor.
2. Entity tie-break LLM and dispute auto-resolution stretch work.
3. Decay/half-life support if not essential to the documented core.
4. Evaluation UI (retain a static report/screenshot).
5. `known_at` belief-time queries.
6. SSE replay (replace with a deterministic load-scenario action).
7. Ablations A4/A5 (retain A1–A3).
8. Reduce evaluation scenarios from 40 to 25 while retaining every category.

Do not cut:
- Validity windows and `R-REPLACE`.
- Trust-based rejection and dispute handling.
- Decision log and explainability.
- Dependency impact/review flagging.
- Naive-RAG comparison on the hero scenario and the required additional scenarios.
- LLM Inspector page.
- REST API.

## 7. Current execution position

Start at **P0.1**. Do not skip or merge batches unless the supervisor explicitly revises this canonical plan. The next batch is only issued after the previous batch is reviewed and approved.

---

**Consistency rule:** If any earlier message or draft conflicts with this document, this document's 36-batch breakdown is authoritative. In particular, P0 contains exactly four batches: P0.1, P0.2, P0.3 and P0.4.
