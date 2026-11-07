"""Deterministic relation classifier for MemoryMesh.

Governed by docs/SPECIFICATION.md Section 1.1, docs/ARCHITECTURE.md,
and docs/ROADMAP_AND_TASKS.md (T2.2).
Implements canonical task P2.1: deterministic relation classification between a new claim
and existing memory:
  - IDENTICAL: Same normalized value, or less-specific compatible version (e.g. 3 vs 3.12).
  - REFINES: More-specific compatible version (e.g. 3.12.4 vs 3.12).
  - CONFLICTS: Incompatible values on single-valued attributes.
  - COMPLEMENTARY: Incompatible or distinct values on multi-valued attributes that can coexist.
  - UNRELATED: Claims across different entities or different attributes.
"""

from __future__ import annotations

from memorymesh.engine.normalize import normalize_value
from memorymesh.errors import NormalizationError
from memorymesh.models import AttributeDef, Cardinality, Memory, Relation, ResolvedClaim, ValueType


def _classify_version_relation(
    claim_val: str,
    mem_val: str,
    cardinality: Cardinality,
) -> Relation:
    """Classify relationship between two normalized version strings."""
    try:
        claim_parts = [int(p) for p in claim_val.split(".")]
        mem_parts = [int(p) for p in mem_val.split(".")]
    except ValueError:
        if claim_val == mem_val:
            return Relation.IDENTICAL
        if claim_val.startswith(mem_val + "."):
            return Relation.REFINES
        if mem_val.startswith(claim_val + "."):
            return Relation.IDENTICAL
        return Relation.CONFLICTS if cardinality == Cardinality.SINGLE else Relation.COMPLEMENTARY

    # Equal components
    if claim_parts == mem_parts:
        return Relation.IDENTICAL

    # More specific version: claim has mem as prefix and has more components
    # e.g., existing memory is 3.12, new claim is 3.12.4
    if len(claim_parts) > len(mem_parts) and claim_parts[: len(mem_parts)] == mem_parts:
        return Relation.REFINES

    # Less specific version: mem has claim as prefix and claim has fewer components
    # e.g., existing memory is 3.12.4, new claim is 3.12 (or existing is 3.10.1, claim is 3)
    if len(claim_parts) < len(mem_parts) and mem_parts[: len(claim_parts)] == claim_parts:
        return Relation.IDENTICAL

    # Incompatible / differing version components
    return Relation.CONFLICTS if cardinality == Cardinality.SINGLE else Relation.COMPLEMENTARY


def classify_relation(
    claim_val: str,
    mem_val: str,
    schema: AttributeDef,
) -> Relation:
    """Classify deterministic relation between claim value and existing memory.

    Normalizes both claim and memory values using the attribute's ValueType and
    allowed_values to properly handle legacy, unnormalized existing memories.
    """
    try:
        norm_claim = normalize_value(
            claim_val,
            schema.value_type,
            allowed_values=schema.allowed_values,
        )
    except NormalizationError:
        norm_claim = claim_val

    try:
        norm_mem = normalize_value(
            mem_val,
            schema.value_type,
            allowed_values=schema.allowed_values,
        )
    except NormalizationError:
        norm_mem = mem_val

    # 1. Exact normalized value equality is always IDENTICAL
    if norm_claim == norm_mem:
        return Relation.IDENTICAL

    # 2. Version-specific relation rules
    if schema.value_type == ValueType.VERSION:
        return _classify_version_relation(norm_claim, norm_mem, schema.cardinality)

    # 3. For single-valued attributes, differing values represent a conflict
    if schema.cardinality == Cardinality.SINGLE:
        return Relation.CONFLICTS

    # 4. For multi-valued attributes, differing values coexist (complementary)
    return Relation.COMPLEMENTARY


def classify_claim_memory(
    claim: ResolvedClaim,
    memory: Memory,
    schema: AttributeDef,
) -> Relation:
    """Classify relation between a ResolvedClaim and an existing Memory record."""
    if claim.entity_id != memory.entity_id or claim.attribute != memory.attribute:
        return Relation.UNRELATED

    return classify_relation(claim.value_norm, memory.value_norm, schema)
