# MemoryMesh: Conflict-Aware Long-Term AI Memory

> Hackathon track: **SNH-AI-03**. Domain focus: **IT infrastructure knowledge** (servers, libraries, services, teams).

Most agent memory is `store -> retrieve -> generate`. It breaks the moment facts change, disagree, or come from sources of different quality. MemoryMesh decides **at write time** whether new information should **replace, refine, coexist with, or be rejected against** what is already known, and it keeps the reasons.

## The one-sentence acceptance test

> Feed the system `"Server A runs Python 3.10"`, later `"We migrated Server A to Python 3.12"`, then a low-trust `"I think it's still 3.9?"`. Ask *"what is it now?"*, *"what was it in March?"*, *"why do you believe that?"* and *"what might break?"*. All four answers must be correct, cited, and explained.

If that works, the project works. Everything else is polish.

## What the system does

| Capability | How |
| :--- | :--- |
| Temporal validity | Every memory has `valid_from` / `valid_to` plus `recorded_at` / `closed_at` (bitemporal) |
| Source provenance | Every memory links to raw messages and sources through an `evidence` table |
| Confidence | Computed from evidence (noisy-OR over independent sources), never hand-edited |
| Contradictions | Typed `contradicts` edges, `disputed` status, surfaced explicitly at query time |
| Context | `{environment: staging}` scopes; most specific scope wins |
| Dependencies | `depends_on` edges; changing a fact flags everything built on it |
| Explainability | Every decision is written to `decision_log` with a rule ID and numeric scores |

## Quick start

```bash
git clone <repo> && cd memorymesh
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                   # add your LLM API key (or set LLM_PROVIDER=fake)
make reset && make seed                                # create DB and load the hero demo scenario
make app                                               # Streamlit UI on http://localhost:8501
make eval                                              # run benchmark vs naive RAG
make check                                             # lint + tests (run before every commit)
```

## Documentation map

Read in this order. When documents disagree: **RESOLUTION_ENGINE.md governs behavior**, **SPECIFICATION.md governs data shapes**.

| File | Purpose |
| :--- | :--- |
| [`AGENTS.md`](AGENTS.md) | Rules for the AI coding agent: stack, conventions, forbidden actions |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Components, data flow, design decisions (ADRs) |
| [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) | Pydantic models, SQLite schema, service API, config formats |
| [`docs/RESOLUTION_ENGINE.md`](docs/RESOLUTION_ENGINE.md) | The decision rules, formulas, worked numeric examples |
| [`docs/PROMPTS.md`](docs/PROMPTS.md) | Every LLM prompt, with JSON contracts |
| [`docs/EVALUATION_AND_DEMO.md`](docs/EVALUATION_AND_DEMO.md) | Scenarios, metrics, baseline, ablations, 3-minute pitch |
| [`docs/ROADMAP_AND_TASKS.md`](docs/ROADMAP_AND_TASKS.md) | Phases, task IDs, acceptance tests, cut order |
| [`docs/VIBECODING_GUIDE.md`](docs/VIBECODING_GUIDE.md) | How to drive an AI coding agent through this project, with copy-paste prompts |

## What changed compared with the DemoToSkill document template

| Template file | Decision | Reason |
| :--- | :--- | :--- |
| `DEMO_AND_ABLATION.md` | Renamed `EVALUATION_AND_DEMO.md` | Measurement against a baseline is the core proof here, not a UI mutation test |
| `PROMPTS.md` | Kept | LLM is used for extraction, text conflicts, query parsing, answers |
| *(new)* `RESOLUTION_ENGINE.md` | Added | The rule set is the product; it deserves its own contract |
| *(new)* `VIBECODING_GUIDE.md` | Added | The project is fully agent-built; the workflow must be written down |
| `.agent/skills/...` | Removed | `AGENTS.md` is enough at this size |
| LangGraph, Playwright, FTS5 heal memory, React frontend | Removed | The pipeline is linear and deterministic; see ADR-003 and ADR-004 |

## Repository layout

```text
memorymesh/
├── README.md
├── AGENTS.md
├── Makefile
├── pyproject.toml
├── .env.example
├── docs/                      # the eight planning documents above (minus README/AGENTS)
├── config/
│   ├── engine.yaml            # thresholds and ablation flags
│   ├── sources.yaml           # source reliabilities
│   └── attributes.yaml        # attribute catalog (cardinality, value types)
├── src/memorymesh/
│   ├── models.py              # Pydantic models (SPECIFICATION section 1)
│   ├── config.py              # YAML loading into typed config
│   ├── db.py                  # connection, schema creation
│   ├── schema.sql
│   ├── repository.py          # ALL SQL lives here
│   ├── service.py             # MemoryService facade (public API)
│   ├── ingest/
│   │   ├── pipeline.py        # message -> claims -> decisions -> mutations
│   │   ├── extractor.py       # LLM claim extraction
│   │   ├── normalize.py       # value normalization per value_type
│   │   └── entity_resolver.py
│   ├── engine/                # PURE functions, no I/O
│   │   ├── classifier.py      # relation classification
│   │   ├── rules.py           # resolve(): the decision engine
│   │   ├── confidence.py
│   │   └── propagate.py
│   ├── retrieval/
│   │   ├── retrieve.py        # current / as-of / history / why
│   │   ├── context.py         # scope matching and specificity
│   │   └── graph.py           # NetworkX builder, impact analysis
│   ├── llm/
│   │   ├── client.py          # wrapper: cache, semaphore, retry, call log
│   │   ├── providers.py       # anthropic / gemini / fake
│   │   ├── query_parser.py
│   │   ├── composer.py
│   │   └── prompts/*.md
│   ├── baseline/naive_rag.py
│   └── api/main.py            # optional thin FastAPI wrapper
├── app/                       # Streamlit multipage UI
├── eval/
│   ├── scenarios/*.yaml
│   ├── run_eval.py
│   └── metrics.py
├── scripts/                   # smoke_test_llm.py, seed_demo.py, reset_db.py
├── tests/
└── data/                      # gitignored: memorymesh.db, llm cache
```
