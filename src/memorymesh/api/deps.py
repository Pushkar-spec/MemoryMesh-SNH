"""FastAPI dependency providers for MemoryMesh API.

Governed by docs/SPECIFICATION.md Section 7.
"""

from __future__ import annotations

from typing import cast

from fastapi import Request

from memorymesh.service import MemoryService


def get_service(request: Request) -> MemoryService:
    """Retrieve the singleton MemoryService instance bound to the FastAPI application state."""
    service = getattr(request.app.state, "service", None)
    if service is None or not isinstance(service, MemoryService):
        raise RuntimeError("MemoryService is not initialized on app.state.")
    return cast(MemoryService, service)
