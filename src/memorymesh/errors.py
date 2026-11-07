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


class UnknownSourceError(MemoryMeshError):
    """Raised when an operation references a source not registered in the system."""


class EngineError(MemoryMeshError):
    """Base exception for resolution engine errors."""


class InvalidValidityWindowError(EngineError, ValueError):
    """Raised when validity window dates violate logical or database constraints."""


class ConflictResolutionError(EngineError, ValueError):
    """Raised when a conflict cannot be resolved under the active resolution rules."""


class UnsupportedRelationError(ConflictResolutionError, NotImplementedError):
    """Raised when a claim-memory relation (e.g., IDENTICAL) is not supported in Phase 1."""
