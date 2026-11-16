"""MemoryMesh REST API package.

Governed by docs/SPECIFICATION.md Section 7.
"""

from __future__ import annotations

from memorymesh.api.main import app, create_app

__all__ = ["app", "create_app"]
