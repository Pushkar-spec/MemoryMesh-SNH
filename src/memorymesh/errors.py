"""Typed domain exceptions for MemoryMesh.

Governed by docs/AGENTS.md Section 3E.
"""

from __future__ import annotations


class MemoryMeshError(Exception):
    """Base exception for all MemoryMesh domain and runtime errors."""


class RepositoryError(MemoryMeshError):
    """Base exception for repository operations."""


class MemoryNotFoundError(RepositoryError):
    """Raised when an operation references a memory ID that does not exist."""


class InvalidUpdateFieldError(RepositoryError, ValueError):
    """Raised when an update contains field names not in the allowed allowlist."""


class ForeignKeyConstraintError(RepositoryError):
    """Raised when an operation violates database foreign key constraints."""


class EngineError(MemoryMeshError):
    """Base exception for resolution engine errors."""


class InvalidValidityWindowError(EngineError, ValueError):
    """Raised when validity window dates violate logical or database constraints."""


class ConflictResolutionError(EngineError, ValueError):
    """Raised when a conflict cannot be resolved under the active resolution rules."""
