SHELL = cmd.exe
.PHONY: help setup test lint check api web types web-check eval smoke reset seed

help:
	@echo MemoryMesh Development Commands
	@echo ===============================
	@echo Active commands:
	@echo   make setup     - Install package in editable mode with development dependencies
	@echo   make test      - Run test suite with pytest
	@echo   make lint      - Run ruff check and format verification
	@echo   make check     - Run backend stage checks (lint + test)
	@echo   make smoke     - [P0.4] Run LLM provider smoke test
	@echo   make web       - [P0.4] Start Vite React dev server
	@echo   make web-check - [P0.4] Run frontend verification (eslint, tsc, vite build)
	@echo   make types     - [P0.4/P9] Regenerate TypeScript API types from OpenAPI
	@echo.
	@echo Pending future phases:
	@echo   make api       - [P9] Start FastAPI backend server
	@echo   make eval      - [P8] Run evaluation pipeline against baseline
	@echo   make reset     - [P0.3] Reset database and recreate schema
	@echo   make seed      - [P5] Seed demo scenario

setup:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

# For P0.1, check runs backend lint and test.
# In P0.4+ after frontend scaffolding, this will also include web-check.
check: lint test

# ==============================================================================
# Pending implementation commands (deferred to respective roadmap phases)
# ==============================================================================

api:
	@echo ERROR: 'make api' is pending P9 (FastAPI implementation).
	@exit 1

web:
	npm --prefix frontend run dev

types:
	@echo NOTE: 'make types' requires the FastAPI backend to be running at http://localhost:8000.
	npx --prefix frontend openapi-typescript http://localhost:8000/openapi.json -o frontend/src/api/types.ts

web-check:
	npm --prefix frontend run lint
	npx --prefix frontend tsc -p frontend/tsconfig.app.json --noEmit
	npm --prefix frontend run build

smoke:
	python scripts/smoke_test_llm.py

eval:
	@echo ERROR: 'make eval' is pending P8 (Evaluation suite).
	@exit 1

reset:
	@echo ERROR: 'make reset' is pending P0.3 (Database schema and db.py).
	@exit 1

seed:
	@echo ERROR: 'make seed' is pending P5 (Hero scenario seeding script).
	@exit 1

