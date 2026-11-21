"""FastAPI application factory and server entrypoint for MemoryMesh.

Governed by docs/SPECIFICATION.md Section 7 and docs/ARCHITECTURE.md.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from memorymesh.api.errors import register_error_handlers
from memorymesh.api.routers import ask, demo, llm, read, write
from memorymesh.api.routers import eval as eval_router
from memorymesh.config import AppConfig, load_app_config
from memorymesh.db import DEFAULT_DB_PATH, init_db
from memorymesh.llm.client import LLMClient
from memorymesh.llm.providers import get_provider
from memorymesh.repository import Repository
from memorymesh.service import MemoryService


def create_app(
    service: MemoryService | None = None,
    cfg: AppConfig | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        service: Optional pre-configured MemoryService instance for dependency
            injection in testing and embedded scenarios. When omitted, the service
            and its dependencies are initialized during lifespan startup.
        cfg: Optional AppConfig override when constructing the default service.
    """
    owned_service = False

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        nonlocal owned_service
        if service is not None:
            app.state.service = service
            owned_service = False
        elif getattr(app.state, "service", None) is None:
            app_cfg = cfg or load_app_config()
            conn = None
            try:
                conn = init_db(DEFAULT_DB_PATH, check_same_thread=False)
                repo = Repository(conn)
                provider_name = (
                    app_cfg.engine.llm.provider.value
                    if hasattr(app_cfg.engine.llm.provider, "value")
                    else str(app_cfg.engine.llm.provider)
                )
                llm_provider = get_provider(provider_name)
                llm_client = LLMClient(
                    provider=llm_provider,
                    repo=repo,
                    config=app_cfg.engine.llm,
                )
                app.state.service = MemoryService(
                    db_target=repo,
                    cfg=app_cfg,
                    llm=llm_client,
                )
                owned_service = True
            except Exception as exc:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                raise RuntimeError(
                    f"Failed to initialize configured LLM provider '{provider_name}': {exc}"
                ) from exc

        try:
            yield
        finally:
            if owned_service:
                svc = getattr(app.state, "service", None)
                if svc is not None and hasattr(svc, "close"):
                    svc.close()
                elif svc is not None and hasattr(svc, "repo") and hasattr(svc.repo, "conn"):
                    try:
                        svc.repo.conn.close()
                    except Exception:
                        pass

    app = FastAPI(
        title="MemoryMesh API",
        description="Conflict-aware long-term memory system for LLM applications.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS configuration for local React frontend development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Uniform error handlers
    register_error_handlers(app)

    # Include REST routers under /api
    app.include_router(write.router, prefix="/api")
    app.include_router(read.router, prefix="/api")
    app.include_router(ask.router, prefix="/api")
    app.include_router(llm.router, prefix="/api")
    app.include_router(eval_router.router, prefix="/api")
    app.include_router(demo.router, prefix="/api")

    return app


app = create_app()
