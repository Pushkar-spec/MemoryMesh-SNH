"""Deterministic relation classifier for MemoryMesh.

Governed by docs/SPECIFICATION.md Section 1.1, docs/ARCHITECTURE.md,
docs/RESOLUTION_ENGINE.md Section 3, and
MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P2.1, P7.2).

Pure functional core: zero I/O, zero network, zero LLM provider dependencies.
Classifies the relation between a new claim value and existing memory value:
  - IDENTICAL: Same normalized value, or less-specific compatible version (e.g. 3 vs 3.12).
  - REFINES: More-specific compatible version (e.g. 3.12.4 vs 3.12).
  - CONFLICTS: Incompatible values on single-valued attributes.
  - COMPLEMENTARY: Incompatible or distinct values on multi-valued attributes that can coexist.
  - UNRELATED: Claims across different entities or different attributes.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from memorymesh.engine.normalize import normalize_value
from memorymesh.errors import NormalizationError
from memorymesh.models import (
    AttributeDef,
    Cardinality,
    Memory,
    Relation,
    RelationProposal,
    ResolvedClaim,
    ValueType,
)


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
    proposal: RelationProposal | None = None,
    *,
    text_duplicate_candidate: int = 70,
) -> Relation:
    """Classify relation between claim value and existing memory.

    Pure, deterministic function with zero I/O or provider dependencies.
    Normalizes both claim and memory values using the attribute's ValueType and
    allowed_values to properly handle legacy, unnormalized existing memories.
    If a validated relation proposal is provided by the orchestration layer,
    applies strict deterministic cardinality and invariant guards.
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

    # 1. Exact normalized value equality is ALWAYS IDENTICAL (deterministic invariant)
    if norm_claim == norm_mem:
        return Relation.IDENTICAL

    # 2. Version-specific relation rules (pure deterministic)
    if schema.value_type == ValueType.VERSION:
        return _classify_version_relation(norm_claim, norm_mem, schema.cardinality)

    # 3. Text attributes: deterministic fuzzy matching vs LLM proposal
    if schema.value_type == ValueType.TEXT:
        ratio = fuzz.ratio(norm_claim, norm_mem)
        # Clearly different strings below duplicate threshold are deterministically classifiable
        if ratio < text_duplicate_candidate:
            return (
                Relation.CONFLICTS
                if schema.cardinality == Cardinality.SINGLE
                else Relation.COMPLEMENTARY
            )

        # Strings meeting or exceeding threshold are duplicate/similarity candidates
        if proposal is not None and proposal != RelationProposal.AMBIGUOUS:
            if proposal == RelationProposal.IDENTICAL:
                return Relation.IDENTICAL
            if proposal == RelationProposal.REFINES:
                return Relation.REFINES
            if proposal == RelationProposal.CONFLICTS:
                if schema.cardinality == Cardinality.SINGLE:
                    return Relation.CONFLICTS
                return Relation.COMPLEMENTARY
            if proposal == RelationProposal.COMPLEMENTARY:
                if schema.cardinality == Cardinality.SINGLE:
                    return Relation.CONFLICTS
                return Relation.COMPLEMENTARY

        # Unresolved similar strings without a decisive LLM proposal are AMBIGUOUS
        return Relation.AMBIGUOUS

    # 4. If a validated relation proposal is provided for other types
    if proposal is not None:
        if proposal == RelationProposal.AMBIGUOUS:
            return Relation.AMBIGUOUS
        if proposal == RelationProposal.IDENTICAL:
            return Relation.IDENTICAL
        if proposal == RelationProposal.REFINES:
            return Relation.REFINES
        if proposal == RelationProposal.CONFLICTS:
            if schema.cardinality == Cardinality.SINGLE:
                return Relation.CONFLICTS
            return Relation.COMPLEMENTARY
        if proposal == RelationProposal.COMPLEMENTARY:
            if schema.cardinality == Cardinality.SINGLE:
                return Relation.CONFLICTS
            return Relation.COMPLEMENTARY

    # 5. Pure deterministic classification based on schema cardinality (ENUM, NUMBER, DATE)
    if schema.cardinality == Cardinality.SINGLE:
        return Relation.CONFLICTS

    return Relation.COMPLEMENTARY


def classify_claim_memory(
    claim: ResolvedClaim,
    memory: Memory,
    schema: AttributeDef,
    proposal: RelationProposal | None = None,
    *,
    text_duplicate_candidate: int = 70,
) -> Relation:
    """Classify relation between a ResolvedClaim and an existing Memory record."""
    if claim.entity_id != memory.entity_id or claim.attribute != memory.attribute:
        return Relation.UNRELATED

    return classify_relation(
        claim.value_norm,
        memory.value_norm,
        schema,
        proposal=proposal,
        text_duplicate_candidate=text_duplicate_candidate,
    )
