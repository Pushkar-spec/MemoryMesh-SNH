# MemoryMesh: Evaluation, Ablation & Demo Plan

The claim to prove: **resolving memory at write time beats retrieving raw messages at answer time** on facts that change, conflict, and carry different trust. All numbers in the final pitch must come from `make eval`, never from this document. Targets below are hypotheses to be measured, not results.

---

## 1. Systems Under Test

| ID | System | Description |
| :--- | :--- | :--- |
| **B0** | Naive RAG baseline | Embed all messages (`fastembed`), top-5 retrieval, LLM answers via `baseline_answer` |
| **MM** | MemoryMesh (full) | Full engine, all flags on, answers via `compose_answer` |
| **A1** | MM without temporal | `use_temporal: false` |
| **A2** | MM without trust | `use_trust: false` |
| **A3** | MM without cardinality | `use_cardinality: false` |
| **A4** | MM without context | `use_context: false` |
| **A5** | MM without propagation | `use_propagation: false` |

Both B0 and MM answer through the same `FinalAnswer` schema and the same model tier so grading is deterministic and fair.

### Evaluation modes

| Mode | LLM | Purpose |
| :--- | :--- | :--- |
| `structured` | none (claims provided in the scenario) | Tests the **engine** deterministically; runs in CI; used for ablations A1-A5 |
| `e2e` | extractor + parser + composer | Tests the **whole system** against B0; headline comparison |
| `engine-only answers` | none (deterministic composer) | Secondary table that removes composer effects |

---

## 2. Scenario Format (`eval/scenarios/*.yaml`)

```yaml
id: replace-001
category: simple_update
description: "Admin migrates a server; later query asks current and past."
now: "2026-06-25T09:00:00Z"
messages:
  - {id: m1, t: "2026-01-10T09:00:00Z", source: sysadmin_priya, text: "Server A runs Python 3.10."}
  - {id: m2, t: "2026-06-01T08:00:00Z", source: sysadmin_priya, text: "We migrated Server A to Python 3.12 today."}
structured_claims:             # used in `structured` mode (bypasses LLM)
  m1: [{subject: "Server A", attribute: python_version, value: "3.10", extract_confidence: 0.95}]
  m2: [{subject: "Server A", attribute: python_version, value: "3.12", valid_from: "2026-06-01T00:00:00Z", extract_confidence: 0.95}]
expected_decisions:            # optional engine-level checks
  m1: R-INSERT
  m2: R-REPLACE
questions:
  - id: q1
    type: current
    text: "What Python version does Server A use?"
    plan: {intent: current, entity: "Server A", attribute: python_version}   # used in structured mode
    expect: {value: "3.12", forbidden: ["3.10"]}
  - id: q2
    type: as_of
    text: "What Python version did Server A use in March?"
    plan: {intent: as_of, entity: "Server A", attribute: python_version, valid_at: "2026-03-15T12:00:00Z"}
    expect: {value: "3.10", forbidden: ["3.12"]}
```

### `expect` keys

| Key | Meaning |
| :--- | :--- |
| `value` | Exact normalized value the answer's `value` must equal |
| `values` | Ordered list for history/multi-valued answers |
| `forbidden` | Values that count as **stale/wrong** if the answer's `value` or `values` contains them |
| `conflict_flagged` | Expected boolean (default false) |
| `needs_review_flagged` | Expected boolean |
| `abstain` | Expected `abstained` boolean |
| `dependents` | For `impact`: set of `(entity, attribute)` pairs that must be found |
| `cited_source` | For `why`: source ID that must appear among cited evidence |
| `rule_id` | For `why`: rule ID that must appear in the explanation trace |

---

## 3. Scenario Catalog (40 scenarios, ~100 questions)

| Category | Count | What it tests |
| :--- | :-: | :--- |
| `simple_update` | 6 | Replace with validity windows; current + as-of |
| `corroboration` | 5 | Duplicates and paraphrases raise confidence, no duplicate memories |
| `low_trust_reject` | 5 | Weak contradicting claim must not overwrite |
| `dispute` | 4 | Mid-trust conflict must surface as a dispute, not a silent pick |
| `multi_valued` | 4 | Hosting several services; coexistence; retraction |
| `context` | 4 | Staging vs production; specificity |
| `out_of_order` | 4 | Late-arriving older facts; backfill; no history rewrite |
| `dependency` | 5 | Single-hop and transitive impact; flag on foundation change |
| `time_travel` | 3 | As-of, boundaries, and `known_at` (belief-time) |
| **Total** | **40** | |

**Designed-to-lose cases (at least 3, mandatory):** include them and report them honestly.
1. A genuinely correct claim from a low-trust source gets rejected (known limitation).
2. A paraphrased entity alias not present in `entities.yaml` ("the box in rack 4") creates a new entity and misses the link.
3. An ambiguous hedged message where both systems fail.

**Write scenarios before tuning thresholds.** Freeze `config/engine.yaml` thresholds before the first full run; record any later change in the results file.

---

## 4. Metrics (`eval/metrics.py`)

| Metric | Definition |
| :--- | :--- |
| **Current accuracy** | % of `current` questions where `answer.value == expect.value` |
| **As-of accuracy** | Same for `as_of` and `time_travel` questions |
| **History accuracy** | % of `history` questions where `answer.values == expect.values` (order-sensitive) |
| **Stale-answer rate** | % of current/as-of questions where the answer contains a `forbidden` value (lower is better) |
| **Conflict recall** | Of questions with `expect.conflict_flagged = true`, % flagged |
| **Conflict false-alarm rate** | Of questions with `expect.conflict_flagged = false`, % flagged anyway |
| **Impact recall / precision** | Found dependents vs expected dependents |
| **Provenance correctness** | % of `why` answers citing the expected source and rule |
| **Abstention accuracy** | % correct on `abstain` expectations |
| **Engine decision accuracy** | % of messages whose rule ID equals `expected_decisions` (structured mode) |
| **Cost** | LLM calls and total latency per question (from `llm_calls`) |

B0 has no rule IDs or dependents: its Impact, Provenance, and Decision metrics are reported as "n/a" rather than zero, and noted in the table footer.

### Output
`make eval` writes `eval/results/<timestamp>/` containing `results.json`, `report.md` (the comparison table), and per-question traces. It also prints one summary table:

```text
Metric                    B0 (naive RAG)   MM (full)   A1 -temporal   A2 -trust   A3 -card.   A4 -context   A5 -propag.
Current accuracy              ...             ...          ...            ...         ...          ...           ...
As-of accuracy                ...
Stale-answer rate             ...
Conflict recall               ...
Impact recall                 ...
Provenance correctness        ...
```
Fill from the run. Do not hand-edit.

### Fairness checklist
- Same questions, same message order, same LLM tier for B0 and MM answers.
- B0 is told to flag conflicts.
- B0 receives top-5 with timestamps and source names (strongest reasonable baseline).
- Temperature 0; results cached by prompt hash; report that LLM variance was not measured (or run 3 times and report mean and range, if time permits).

---

## 5. Hero Scenario (`eval/scenarios/demo_hero.yaml`)

This is both the demo script data and scenario `hero-001`.

```yaml
id: hero-001
category: hero
now: "2026-06-25T09:00:00Z"
messages:
  - {id: m1, t: "2026-01-10T09:00:00Z", source: sysadmin_priya, text: "Server A runs Python 3.10."}
  - {id: m2, t: "2026-02-02T11:30:00Z", source: wiki,           text: "Runtime for Server A: Python 3.10 (see infra page)."}
  - {id: m3, t: "2026-02-10T10:00:00Z", source: sysadmin_priya, text: "Server A deploys happen on Fridays."}
  - {id: m4, t: "2026-03-05T14:00:00Z", source: sysadmin_priya, text: "Library X is deployed on Server A and works fine with its Python 3.10 runtime."}
  - {id: m5, t: "2026-06-01T08:00:00Z", source: sysadmin_priya, text: "We migrated Server A to Python 3.12 today."}
  - {id: m6, t: "2026-06-03T16:20:00Z", source: intern_chat,    text: "I think Server A is still on Python 3.9?"}
  - {id: m7, t: "2026-06-15T10:10:00Z", source: dev_chat,       text: "FYI deploys for Server A are on Mondays now."}
  - {id: m8, t: "2026-06-20T13:00:00Z", source: dev_chat,       text: "In staging, Server A still runs Python 3.11."}
expected_decisions: {m1: R-INSERT, m2: R-CORROBORATE, m3: R-INSERT, m4: R-INSERT, m5: R-REPLACE, m6: R-REJECT, m7: R-DISPUTE, m8: R-INSERT}
questions:
  - {id: q1, type: current, text: "What Python version does Server A use?",                        expect: {value: "3.12", forbidden: ["3.10", "3.9"]}}
  - {id: q2, type: as_of,   text: "What Python version did Server A use in March?",                expect: {value: "3.10", forbidden: ["3.12", "3.9"]}}
  - {id: q3, type: why,     text: "Why do you believe Server A is on Python 3.12?",                expect: {cited_source: sysadmin_priya, rule_id: R-REPLACE}}
  - {id: q4, type: impact,  text: "What might break because of the Python migration on Server A?", expect: {dependents: [["Library X", "compatibility_note"]], needs_review_flagged: true}}
  - {id: q5, type: current, text: "Which day does Server A deploy?",                               expect: {conflict_flagged: true, values: ["friday", "monday"]}}
  - {id: q6, type: current, text: "What Python version does Server A use in staging?",             expect: {value: "3.11"}}
  - {id: q7, type: history, text: "Which Python versions has Server A had?",                       expect: {values: ["3.10", "3.12"], forbidden: ["3.9"]}}
```

**Expected baseline weakness (hypothesis to verify, not a claim):** naive RAG retrieves the 3.10, 3.12, and 3.9 messages together; the most recent one is the intern's hedged message, so q1 is at risk. Report what actually happens.

---

## 6. UI Demo Application (React)

React + TypeScript app in `frontend/`, built against the REST contract in `SPECIFICATION.md` section 7. Pages are routes in React Router; data comes only from TanStack Query hooks.

| Page (route) | Contents | Main endpoints |
| :--- | :--- | :--- |
| **Ingest** `/` | Message box, source dropdown, timestamp picker, **Ingest** button; after each ingest a **decision card** (action badge, rule ID, scores, explanation, flagged dependents). A **Replay hero scenario** button streams messages m1-m8 with decision cards appearing one by one | `POST /api/ingest`, `GET /api/demo/replay` (SSE), `GET /api/catalog` |
| **Memories** `/memories` | Filterable table: entity, attribute, value, status badge, confidence bar, validity window, sources, needs-review icon; row click opens the provenance drawer | `GET /api/memories`, `/explain` |
| **Graph** `/graph` | React Flow canvas: entity and memory nodes (color by status), edge styles by type (`supersedes` solid, `contradicts` dashed red, `depends_on` dotted blue); click a node for the provenance panel | `GET /api/graph` |
| **Timeline** `/timeline` | One lane per `(entity, attribute, context)`; bars = validity windows (open-ended bars fade out to the right); disputed amber; rejected claims as hollow markers; a draggable **"as of" cursor** that highlights what was valid at that time | `GET /api/timeline` |
| **Ask** `/ask` | Question box, optional as-of hint; answer card with `[M12]` citations that open the provenance drawer; **"Compare with naive RAG"** toggle shows two columns plus the baseline's retrieved messages | `POST /api/ask` |
| **Review** `/review` | Disputes with **Pick winner** buttons; `needs_review` memories with **Acknowledge** | `GET /api/disputes`, `POST /api/disputes/resolve`, `POST /api/memories/{id}/acknowledge` |
| **Decisions** `/decisions` | Full audit trail, filter by rule ID | `GET /api/decisions` |
| **LLM Inspector** `/inspector` | Table of calls: prompt id, summary, raw response, parsed JSON, latency, status (cache hits marked) | `GET /api/llm/calls` |
| **Evaluation** `/evaluation` | The metrics table from the latest `make eval` run (with n/a handled), plus the markdown report | `GET /api/eval/latest` |
| **Settings** `/settings` | Read-only source reliabilities, thresholds, ablation flags, attribute catalog | `GET /api/catalog` |

**Global layout:** left sidebar navigation; header with the **demo clock** (editable date-time sent as `now`), LLM provider badge from `/api/health` (shows "offline demo mode" for `fake`), and a **Reset demo** button.

**Design requirements**
- Fixed status colors everywhere: active green, historical gray, disputed amber, rejected red, needs-review blue.
- Any memory ID shown (e.g. `M12`) is clickable and opens the same **provenance drawer** (evidence, decisions, edges) from any page.
- Reliabilities and thresholds are always reachable in one click (Settings), never hidden.
- Every page implements loading, error, and empty states.
- Desktop-first; must be legible when projected at 1280x720.

---

## 7. The 3-Minute Pitch Script

### Minute 1: The Problem (0:00 - 1:00)
- **Speaker:** "AI assistants remember with a notebook that only ever adds lines. Ours says Python 3.10, then Python 3.12, then a rumor says 3.9. What does it believe? Nobody knows, because memory was never resolved."
- **Action:** Show **B0 (naive RAG)** on the Ask page. Ask "What Python version does Server A use?" with all hero messages loaded. Show the answer and the mixed retrieved messages.
- **Speaker:** "It saw all three and guessed."

### Minute 2: MemoryMesh Resolves (1:00 - 2:15)
- **Action:** Reset, then ingest messages m1-m8 live (or at speed). After each, flash the decision card:
  - m2 -> *Corroborated, confidence 0.86 -> 0.96*
  - m5 -> *Replaced 3.10, window closed, 1 dependent flagged*
  - m6 -> *Rejected: new 0.15 vs existing 0.86*
  - m7 -> *Disputed: sources disagree*
- **Action:** Show Graph and Timeline pages updating.
- **Action:** Ask q1-q4. Show the answers with citations; open "why" for q3 showing the rule ID and numbers.
- **Action:** Ask q5; show the dispute, both positions shown.

### Minute 3: Proof & Close (2:15 - 3:00)
- **Action:** Open the Evaluation page: the comparison table from `make eval`, and the ablation row ("remove trust: low-trust rumor wins; remove temporal: stale answers").
- **Action:** Open the LLM Inspector: "The LLM only reads and writes language. The decisions are deterministic code you can audit."
- **Close:** "Memory that knows what changed, who said it, and what it affects."

### Backup plan
- Record a full screencast (with audio) the night before.
- `make seed` loads the hero scenario in structured mode with **no API key** so the demo works offline; the fake provider answers with the deterministic composer.
- Keep a pre-run `eval/results/latest/report.md` screenshot in the slides.

---

## 8. Likely Judge Questions

| Question | Prepared answer |
| :--- | :--- |
| "Why not just prompt an LLM with all messages?" | We compare against exactly that (B0) on the same questions; results in the table. Also: no persistent state, no audit trail, no consistency across sessions. |
| "Where do source reliabilities come from?" | Configured and visible; the engine is parameterized by them. Learning them from outcomes is future work (links to TrustChain-style feedback). |
| "What if a low-trust source is right?" | It is stored as `rejected` with its evidence, and corroboration by a second source or a human `resolve_dispute` can change the outcome. Shown as a designed-to-lose case. |
| "How do you detect conflicts in free text?" | Deterministic for versions/enums/numbers/dates; LLM classifier only for text values, with a conservative fallback. |
| "How does it scale?" | Chains are indexed per `(entity, attribute, context)`; only one chain is loaded per claim. Graph view is built on demand. |
| "What if entity names differ?" | Alias table, fuzzy match, optional LLM tie-break; unresolved names create flagged entities rather than silent merges. |
