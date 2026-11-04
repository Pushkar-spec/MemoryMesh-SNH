# MemoryMesh: Vibecoding Guide

How to build this entire project with an AI coding agent without losing control of it. Written for someone new to the domain: you do not need to understand every line, but you **must** be able to check that each step works.

---

## 1. The Core Loop (use for every task)

```text
1. Pick ONE task ID from ROADMAP_AND_TASKS.md
2. Give the agent the task + the docs it must read
3. Agent writes tests first, then code
4. You run `make check` yourself (do not trust "it should work")
5. You check the task's acceptance line manually when it says "manual"
6. Commit. Then next task.
```

Rules that save you hours:
- **One task per session/prompt.** Big prompts produce big, wrong rewrites.
- **Tests are your eyes.** If you cannot read the code, you can still read test names and see them go green.
- **Never accept "I updated the tests so they pass."** Tests change only when the *spec* changes.
- **Docs are the source of truth.** If the agent wants to behave differently from `RESOLUTION_ENGINE.md`, the docs change first (by you), then the code.
- **Commit after every green task.** If the agent breaks something, `git restore` and retry with a tighter prompt.

---

## 2. Session Setup

1. Put this whole doc pack in the repo (`README.md`, `AGENTS.md`, `docs/`).
2. Open your agent (Claude Code, Cursor, Copilot agent, etc.) in the repo root.
3. Paste the **Kickoff Prompt** below once per new session.

### Kickoff Prompt (paste at the start of each session)
```text
You are building MemoryMesh. Before doing anything:
1. Read AGENTS.md fully.
2. Read docs/ARCHITECTURE.md, docs/SPECIFICATION.md, and docs/RESOLUTION_ENGINE.md.
3. Read the task list in docs/ROADMAP_AND_TASKS.md.

Rules: follow AGENTS.md strictly. Work on exactly one task ID that I give you. Write tests first.
Run `make check` before reporting. Do not add dependencies, files, or features outside the task.
If something in the docs is ambiguous, choose the simplest option, list it under "Assumptions", and continue.
Report: files changed, tests added, assumptions, and anything not done.

Confirm you have read the docs by summarizing, in 5 lines, the difference between the engine (pure) and the service (I/O).
```

---

## 3. Per-Phase Prompts (copy and paste)

Replace nothing; just paste. Each prompt assumes the kickoff prompt has been given.

### P0: Setup
```text
Task T0.1 to T0.4. Create the repo skeleton exactly as in README.md "Repository layout" (empty files are fine for modules not yet needed).
Implement pyproject.toml with dependencies: pydantic>=2, pyyaml, networkx, rapidfuzz, streamlit, pyvis, plotly, pandas, httpx,
anthropic, pytest, ruff, and optional extras: api (fastapi, uvicorn), baseline (fastembed). Create Makefile targets from AGENTS.md section 6.
Implement config.py, schema.sql, db.py, and models.py exactly as in docs/SPECIFICATION.md. Add tests for config loading and model round-trips.
```
```text
Task T0.5. Implement llm/providers.py with AnthropicProvider, a GeminiProvider stub that raises NotImplementedError, and FakeProvider (docs/PROMPTS.md section 8).
Implement scripts/smoke_test_llm.py that sends a tiny JSON-only prompt and prints parsed JSON or a clear error. Read keys from .env via environment variables.
```

### P1: Temporal core
```text
Task T1.1 and T1.2. Implement repository.py methods needed for memories, evidence, edges and a transaction context manager. Then implement engine/rules.py
with only R-INSERT and R-REPLACE (assume every source has strength 1.0 for now). resolve() must be a pure function with the signature in docs/RESOLUTION_ENGINE.md.
Write tests with hand-built Memory objects. No database in engine tests.
```
```text
Task T1.3 and T1.4. Implement retrieval current/as-of using the SQL in docs/SPECIFICATION.md section 4.1, and service.ingest_claims that applies a Decision atomically.
Add a test for the Python 3.10 -> 3.12 example including the boundary case valid_at == valid_to.
```

### P2: Cardinality
```text
Task T2.1 to T2.3. Implement ingest/normalize.py, the rule-based part of engine/classifier.py, and the multi-valued path plus R-REFINE in engine/rules.py
exactly as in docs/RESOLUTION_ENGINE.md sections 3 and 4. Table-driven tests for normalization.
```
```text
Task T2.4. Implement entity resolution tiers 1 (normalized alias) and 2 (rapidfuzz using thresholds from config). Unknown names create a new entity with needs_review=1. No LLM yet.
```

### P3: Confidence and trust
```text
Task T3.1 and T3.2. Implement engine/confidence.py with the formulas in docs/RESOLUTION_ENGINE.md section 2 and R-CORROBORATE / R-DUP-SAME-SOURCE.
Your tests must reproduce these exact numbers: 0.855 for the first admin claim and 0.958 after wiki corroboration (tolerance 0.001).
```
```text
Task T3.3 to T3.5. Add trust-based R-REPLACE margins, R-REJECT, R-DISPUTE, the revert guard, history-conflict handling, and service.resolve_dispute.
Reproduce the worked example rows m5, m6, m7 in docs/RESOLUTION_ENGINE.md section 8 as tests with the stated numbers.
```

### P4: Decision log and explain
```text
Task T4.1 to T4.3. Ensure EVERY claim, including skipped ones, writes a decision_log row in the same transaction as its mutations.
Implement service.explain and a reusable invariant checker (AGENTS.md section 3D) that all service tests call at the end.
```

### P5: Dependencies
```text
Task T5.1 to T5.4. Resolve depends_on references to the memory valid at the claim's valid_from, emit depends_on edges, implement engine/propagate.py as in
docs/RESOLUTION_ENGINE.md section 5, log R-PROPAGATE per flagged memory, implement service.impact and acknowledge_review.
Then write scripts/seed_demo.py that ingests the hero scenario in structured mode and prints answers to q1-q7 (gate G1).
```

### P6: Context, backfill, retract
```text
Task T6.1 to T6.5. Implement context keys and scope specificity (docs/RESOLUTION_ENGINE.md 6.1), R-BACKFILL, R-RETRACT, belief-time queries (known_at, recorded_at),
and idempotent ingest. One test per item, named as in the required tests table.
```

### P7: LLM layer
```text
Task T7.1. Implement llm/client.py per docs/ARCHITECTURE.md section 7 and docs/PROMPTS.md section 7: semaphore, SHA-256 cache, JSON extraction,
Pydantic validation, one retry with the repair directive, llm_calls logging. Test with FakeProvider including an invalid-JSON-then-valid case.
```
```text
Task T7.2 to T7.5. Create prompt files from docs/PROMPTS.md. Implement extractor with post-processing, the LLM-backed text classifier with fallbacks,
the query parser, the composer with citation validation, and service.ingest / service.ask. Run the hero scenario end to end and compare decisions to expected_decisions.
```

### P8: Evaluation
```text
Task T8.1 and T8.2. Implement baseline/naive_rag.py (fastembed top-5 + baseline_answer prompt, same FinalAnswer schema) and eval/run_eval.py with
structured, e2e and ablation modes and eval/metrics.py exactly as in docs/EVALUATION_AND_DEMO.md. Start with only the hero scenario.
```
```text
Task T8.3. Write scenarios for the category <CATEGORY> (see docs/EVALUATION_AND_DEMO.md section 3) in the YAML format of section 2. Include structured_claims and expected_decisions.
Use different entities and sources than the hero scenario. Do not tune thresholds.
```
(Run T8.3 once per category.)

### P9: UI
```text
Task T9.1 and T9.2. Create the Streamlit multipage app in app/ that holds a MemoryService in st.session_state. Implement the Ingest page (with decision card)
and Memory Table page as in docs/EVALUATION_AND_DEMO.md section 6. Call only the MemoryService public API.
```
(Repeat for T9.3 to T9.6, naming the pages.)

### P10: Polish
```text
Task T10.1. Verify the README quick start from a clean clone: new venv, install, make reset, make seed, make app. Fix any step that fails and update README.md.
```

---

## 4. Reviewing AI-Generated Code (checklist)

Before accepting any task, check:

- [ ] `make check` is green **and you ran it yourself**
- [ ] New behavior has tests named after the rule/task
- [ ] No SQL outside `repository.py`
- [ ] No `datetime.now()` in `engine/` or `retrieval/`
- [ ] No magic numbers (thresholds read from config)
- [ ] LLM is not choosing replace/reject/dispute
- [ ] Every code path writes a decision-log row
- [ ] No new dependency, no new top-level file, no scope beyond the task
- [ ] No test was weakened or deleted
- [ ] Commit message includes the task ID

---

## 5. Debugging Playbook

| Symptom | Likely cause | What to ask the agent |
| :--- | :--- | :--- |
| Wrong answer for "current" | Validity window not closed, or `valid_at` boundary off by one | "Print the memory rows for this slot and explain why each is or is not valid at T." |
| Everything gets replaced | Trust/threshold config not applied; `use_trust` false | "Print new_score, old_score, thresholds for this decision." |
| Duplicates keep growing | Idempotency or `R-CORROBORATE` path not hit; classifier returns CONFLICTS for equal values | "Show normalized values being compared." |
| Entity splits (`Server A` vs `srv-a`) | Alias table missing, fuzzy threshold too high | "Show resolver tier decisions for these names." |
| LLM output fails validation | Prompt drift or extra prose | "Show raw response from llm_calls and the validation error; do not loosen the schema." |
| Dependency not flagged | `depends_on` target not resolved at claim time | "Show dep_targets resolved for this message." |
| Streamlit state confusion | Service recreated on rerun | "Store the service in st.session_state and reuse." |
| Agent rewrites many files | Prompt too broad | `git restore .`; re-prompt with a single task ID and the "do not touch other files" instruction |

When stuck, the most useful instruction is: **"Do not change code yet. Explain what happens step by step for this input and show the values at each step."**

---

## 6. Learn-As-You-Go (study only what the current phase needs)

| Phase | Concept to understand (30-60 min each) |
| :--- | :--- |
| P0 | Virtual environments, `pyproject.toml`, pytest basics |
| P1 | Pydantic models; SQL `SELECT/INSERT/UPDATE/WHERE`; ISO timestamps and UTC |
| P1-P3 | **Bitemporal data** (valid time vs record time); read Martin Fowler's "Bitemporal History" |
| P2 | Cardinality (one vs many values); string normalization |
| P3 | Probability basics: combining independent evidence (noisy-OR) |
| P4 | Audit logging; why explanations must be generated by code, not guessed |
| P5 | Directed graphs, BFS, transitive dependencies; NetworkX basics |
| P7 | Structured LLM output (JSON schemas, validation, retry); few-shot prompting |
| P8 | Evaluation design: baselines, ablations, fair comparison |
| P9 | Streamlit basics and `st.session_state` |

---

## 7. Glossary

| Term | Plain meaning |
| :--- | :--- |
| **Memory** | One stored claim: who/what, attribute, value, when true, who said it, how sure |
| **Slot** | One entity + attribute + context, e.g. "Server A's python_version in global scope" |
| **Valid time** | When something was true in the real world |
| **Record time** | When the system learned it |
| **Bitemporal** | Tracking both valid time and record time |
| **Cardinality** | Whether an attribute holds one value (python_version) or many (runs_service) |
| **Provenance** | Where a fact came from (which source, which message) |
| **Evidence** | One source's statement supporting a memory |
| **Noisy-OR** | Combining independent evidence so more sources raise confidence without exceeding 1 |
| **Corroboration** | A different source repeating the same fact |
| **Supersede** | A newer fact replaces an older one; the old one becomes historical |
| **Dispute** | Two conflicting facts that cannot be resolved automatically; both are shown |
| **Propagation** | Flagging things that depended on a fact that changed |
| **Functional core** | Pure decision code with no database, clock, or network |
| **Ablation** | Turning one feature off to measure how much it mattered |
| **Naive RAG** | Embed messages, fetch the nearest few, let an LLM answer; no conflict handling |
