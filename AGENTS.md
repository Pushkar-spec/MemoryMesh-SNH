# MemoryMesh: Agent Coding Guidelines & Context

This file defines instructions, boundaries, and conventions for any AI coding agent working on **MemoryMesh** (hackathon track SNH-AI-03). Read it fully at the start of every session.

---

## 1. Mission & North Star

Build a long-term memory system that treats memory as **changing, conflicting, and expiring**. When new information arrives, the system decides whether to **replace, refine, coexist, or reject** relative to existing memory, and records why.

**Acceptance test (must always pass):**
1. `"Server A runs Python 3.10"` (admin), then `"We migrated Server A to Python 3.12"` (admin), then `"I think it's still 3.9?"` (intern).
2. Query *current* -> `3.12`. Query *as of March* -> `3.10`. Query *why* -> cites the admin message and rule `R-REPLACE`. Query *impact* -> lists dependent memories flagged `needs_review`. The intern claim is `rejected`.

### Design principle: LLMs read and write words; code makes decisions
| Task | Owner |
| :--- | :--- |
| Turn messy text into structured claims | LLM (extractor) |
| Judge whether two **free-text** values conflict | LLM (classifier), only for `text` value types |
| Parse a user question into a query plan | LLM (query parser) |
| Write the natural-language answer | LLM (composer), from retrieved facts only |
| Replace / refine / coexist / reject / dispute | **Deterministic code** (`engine/rules.py`) |
| Validity windows, confidence, dependencies, audit log | **Deterministic code** |

If you are about to let an LLM decide something in the second block, stop.

---

## 2. Tech Stack & Environment

| Component | Choice | Notes |
| :--- | :--- | :--- |
| Language | Python 3.11+ | Type hints everywhere; `from __future__ import annotations` |
| Models | Pydantic v2 | All boundaries validated |
| Storage | SQLite3 (stdlib) | One file `data/memorymesh.db`; WAL mode; foreign keys ON |
| Graph | NetworkX | Built **from** SQLite rows on demand; SQLite is the source of truth |
| Fuzzy matching | `rapidfuzz` | Entity resolution tier 2 |
| LLM | Provider-agnostic wrapper | Default `anthropic`; also `gemini`, `fake`. Verify current model IDs in provider docs |
| Baseline embeddings | `fastembed` | Used **only** by the naive-RAG baseline; no PyTorch |
| UI | Streamlit + pyvis + plotly | Multipage app calling `MemoryService` directly |
| API (optional) | FastAPI | Thin wrapper, build last, first thing to cut |
| Quality | `pytest`, `ruff` | `make check` must pass before every commit |
| Config | YAML in `config/` | No magic numbers in code |

---

## 3. Strict Coding Conventions

### A. Architecture rules
- **Functional core, imperative shell.** `engine/*` contains pure functions: inputs in, `Decision` out. No DB, no network, no `datetime.now()`. `service.py` and `repository.py` do the I/O.
- **All SQL lives in `repository.py`.** No SQL strings anywhere else.
- **One transaction per claim.** Applying a `Decision` (mutations + decision log + propagation) is atomic. On exception, roll back and log.
- **Every decision is logged.** Every claim processed (including `SKIP`) writes a `decision_log` row with `rule_id`, `explanation`, and numeric `scores`. No silent paths.
- **Thresholds come from `config/engine.yaml`.** Never hardcode `0.8`, `0.5`, etc.
- **The engine never calls the LLM directly.** The relation classifier receives an injected callable for text conflicts.

### B. Time rules
- Store and compare all timestamps as **UTC ISO-8601 strings in exactly one format**: `YYYY-MM-DDTHH:MM:SSZ`. Lexicographic order equals chronological order; tests depend on this.
- Never call `datetime.now()` inside `engine/` or `retrieval/`; pass `now` / `valid_at` / `known_at` in. This makes tests deterministic.
- `valid_from` defaults to the message's `event_time` when the claim states no date.

### C. LLM calls (`src/memorymesh/llm/client.py`)
- **All LLM calls go through `LLMClient.complete_json()`.** No direct SDK calls elsewhere.
- Temperature `0`. Request pure JSON; strip code fences if returned (see `docs/PROMPTS.md` section 7).
- Validate every response against a Pydantic model. On JSON or validation failure, **retry exactly once** with the repair directive, then raise `LLMOutputError`.
- Wrap calls in `asyncio.Semaphore(3)` (or a thread lock if sync) and cache by SHA-256 of `(provider, model, system, user)`.
- Log every call into `llm_calls` (timestamp, prompt id, prompt summary, raw response, parsed JSON, latency, status). The UI Inspector reads this table.
- `LLM_PROVIDER=fake` must run the whole test suite with **zero network access**.

### D. Data integrity invariants (enforced by tests)
1. `valid_to IS NOT NULL` implies `closed_at IS NOT NULL`.
2. `valid_to > valid_from` whenever `valid_to` is set.
3. For a **single-valued** attribute and one `context_key`, no two non-rejected, non-disputed memories have overlapping validity windows.
4. A `rejected` memory is never returned by retrieval (but is visible in history/audit views).
5. `confidence` is always recomputable from `evidence` rows alone.
6. Re-ingesting the same message (same source, text, event_time) is a no-op.

### E. Style
- `ruff` clean, line length 100. Public functions get a one-line docstring. No dead code, no commented-out blocks.
- Names: `snake_case` functions, `PascalCase` classes, `UPPER_CASE` constants. Rule IDs are strings like `"R-REPLACE"`.
- Errors: raise typed exceptions from `memorymesh.errors`; never `except Exception: pass`.
- Logging: stdlib `logging`; never `print` outside scripts.

### F. Testing
- Every rule in `RESOLUTION_ENGINE.md` has at least one test named after its rule ID (`test_r_replace_...`).
- Engine tests are pure and use hand-built `Memory` objects; no DB needed.
- Service tests use an in-memory SQLite (`:memory:`) and the `fake` LLM.
- Eval scenarios in `eval/scenarios/` double as integration tests in `structured` mode.

### G. Privacy & Security
- Never commit `.env`, API keys, or `data/`.
- Demo data must be synthetic. Do not ingest real personal data.
- Do not log full API keys or full prompts containing secrets; the call log stores a summary plus the raw response only.

---

## 4. Forbidden Actions

- Do **not** add dependencies not listed in `pyproject.toml` without writing the reason in the commit message.
- Do **not** introduce Neo4j, vector databases, LangChain, LangGraph, or ORMs. See ADRs in `docs/ARCHITECTURE.md`.
- Do **not** let an LLM choose between REPLACE / REJECT / DISPUTE / COEXIST.
- Do **not** change a threshold, rule, table, or model shape without updating the governing document **in the same commit**.
- Do **not** expand scope beyond the current phase in `docs/ROADMAP_AND_TASKS.md`.
- Do **not** weaken or delete a failing test to make a build pass. Fix the cause, or ask the human.
- Do **not** mutate rows by hand-written SQL outside `repository.py`.

---

## 5. Working Protocol

1. Identify the current **task ID** (e.g. `T3.2`) from `docs/ROADMAP_AND_TASKS.md`.
2. Read the referenced sections of the governing documents.
3. Write or update tests first when the task changes behavior.
4. Implement the smallest change that satisfies the task's **Definition of Done**.
5. Run `make check`. Fix until green.
6. Commit with a Conventional Commit message: `feat(engine): implement R-REPLACE [T3.2]`.
7. Report: what changed, tests added, anything ambiguous you decided, anything you did **not** do.

**When unsure:** do not guess silently. Choose the simplest option consistent with the docs, record it under "Assumptions" in your report, and continue.

---

## 6. Common Commands

```bash
make setup     # venv + editable install + dev tools
make reset     # delete and recreate data/memorymesh.db
make seed      # load config + hero scenario (structured mode, no LLM)
make test      # pytest
make lint      # ruff check + ruff format --check
make check     # lint + test
make app       # streamlit run app/Home.py
make eval      # python eval/run_eval.py --mode structured && --mode e2e
make smoke     # python scripts/smoke_test_llm.py  (Go/No-Go gate)
```

---

## 7. Key Directory Structure

See `README.md` section "Repository layout". The package root is `src/memorymesh/`. Only `repository.py` touches SQL; only `llm/client.py` touches an LLM SDK; only `engine/` decides outcomes.
