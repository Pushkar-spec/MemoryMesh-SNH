"""LLM-assisted relation classification for MemoryMesh (P7.2).

Governed by docs/ARCHITECTURE.md Section 7, docs/RESOLUTION_ENGINE.md Section 3,
and MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P7.2).
"""

from __future__ import annotations

import logging

from memorymesh.llm.client import LLMClient
from memorymesh.llm.prompts import build_classify_relation_prompt
from memorymesh.models import (
    AttributeDef,
    Relation,
    RelationClassificationResult,
    RelationProposal,
)

logger = logging.getLogger(__name__)


class LLMRelationClassifier:
    """Interprets natural-language relationships between a new claim and existing memory.

    Emits structured relation proposals validated by Pydantic. Proposes relationships
    such as replacement, refinement, coexistence, or ambiguity, but never acts as the
    final authority. All final memory decisions remain governed by deterministic guards
    and resolution engine rules.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        min_confidence: float = 0.5,
        default_model: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.min_confidence = min_confidence
        self.default_model = default_model

    def classify(
        self,
        claim_val: str,
        mem_val: str,
        schema: AttributeDef,
        *,
        claim_raw_text: str = "",
        mem_raw_text: str = "",
        model: str | None = None,
    ) -> RelationClassificationResult:
        """Call the LLM to classify the natural-language relation between two statements.

        Args:
            claim_val: New claim value.
            mem_val: Existing memory value.
            schema: AttributeDef schema definition.
            claim_raw_text: Optional original message or evidence span of the claim.
            mem_raw_text: Optional original raw statement of the existing memory.
            model: Optional model override.

        Returns:
            RelationClassificationResult: Validated relation proposal, confidence, and reason.
        """
        card_str = (
            schema.cardinality.value
            if hasattr(schema.cardinality, "value")
            else str(schema.cardinality)
        )
        system_prompt, user_prompt = build_classify_relation_prompt(
            attribute_name=schema.attribute,
            attribute_description=schema.description,
            cardinality=card_str,
            existing_value=mem_val,
            new_value=claim_val,
            existing_raw_text=mem_raw_text,
            new_raw_text=claim_raw_text,
        )

        eff_model = model or self.default_model

        try:
            return self.llm_client.complete_json(
                prompt_id="classify_relation",
                system=system_prompt,
                user=user_prompt,
                response_model=RelationClassificationResult,
                model=eff_model,
            )
        except Exception as exc:
            logger.warning("LLM relation classification failed: %s", exc)
            return RelationClassificationResult(
                relation=RelationProposal.AMBIGUOUS,
                confidence=0.0,
                reason=f"LLM classification unavailable: {exc}",
            )

    def resolve_proposal(
        self,
        result: RelationClassificationResult,
    ) -> RelationProposal:
        """Resolve an LLM classification result into an actionable or AMBIGUOUS proposal.

        Returns RelationProposal.AMBIGUOUS when:
        1. The LLM returned AMBIGUOUS.
        2. The LLM confidence is below min_confidence.
        3. The provider call failed or produced malformed output.

        Returning RelationProposal.AMBIGUOUS signals that the relation cannot be asserted with
        confidence, preventing false replacement or coexistence.
        """
        if result.relation == RelationProposal.AMBIGUOUS or result.confidence < self.min_confidence:
            return RelationProposal.AMBIGUOUS
        return result.relation

    def to_engine_relation(
        self,
        result: RelationClassificationResult,
        claim_val: str,
        mem_val: str,
        schema: AttributeDef,
    ) -> Relation:
        """Map an LLM classification result to an engine Relation via deterministic rules."""
        from memorymesh.engine.classifier import classify_relation

        proposal = self.resolve_proposal(result)
        return classify_relation(claim_val, mem_val, schema, proposal=proposal)


def classify_relation_llm(
    llm_client: LLMClient,
    claim_val: str,
    mem_val: str,
    schema: AttributeDef,
    *,
    claim_raw_text: str = "",
    mem_raw_text: str = "",
    min_confidence: float = 0.5,
    model: str | None = None,
) -> RelationClassificationResult:
    """Call the LLM relation classifier directly and return the rich structured proposal."""
    classifier = LLMRelationClassifier(llm_client=llm_client, min_confidence=min_confidence)
    return classifier.classify(
        claim_val=claim_val,
        mem_val=mem_val,
        schema=schema,
        claim_raw_text=claim_raw_text,
        mem_raw_text=mem_raw_text,
        model=model,
    )
