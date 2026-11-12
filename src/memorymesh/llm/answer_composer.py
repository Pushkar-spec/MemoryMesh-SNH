"""LLM-assisted answer composition with strict grounding and citation validation (P7.3).

Governed by docs/ARCHITECTURE.md Section 4, docs/SPECIFICATION.md Section 3,
and MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P7.3).
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from pydantic import Field

from memorymesh.llm.client import LLMClient
from memorymesh.llm.prompts import build_compose_answer_prompt
from memorymesh.models import (
    ComposedAnswer,
    MemoryView,
    MeshBaseModel,
    ParsedQuery,
    QueryResult,
)

logger = logging.getLogger(__name__)


class LLMAnswerComposeResult(MeshBaseModel):
    """Raw structured output schema from the answer composer LLM prompt."""

    answer_text: str = ""
    value: str | None = None
    values: list[str] = Field(default_factory=list)
    referenced_memory_ids: list[int] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    explanation: str = ""
    is_uncertain: bool = False
    no_relevant_memory: bool = False


class AnswerComposer:
    """Composes grounded natural-language answers strictly from retrieved memories."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        default_model: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.default_model = default_model

    def _extract_allowed_memory_ids(self, result: QueryResult) -> set[int]:
        """Collect all genuine memory IDs present in the retrieved QueryResult."""
        allowed: set[int] = set()
        for item in result.items:
            allowed.add(item.memory.id)
        for d in result.disputes:
            allowed.add(d.a.memory.id)
            allowed.add(d.b.memory.id)
        for imp in result.impact:
            allowed.add(imp.memory.memory.id)
            for path_id in imp.path:
                allowed.add(path_id)
        for dec in result.decisions:
            if dec.memory_id is not None:
                allowed.add(dec.memory_id)
            if dec.affected_memory_ids:
                allowed.update(dec.affected_memory_ids)
        return allowed

    def _extract_allowed_values(self, result: QueryResult) -> set[str]:
        """Collect all genuine normalized values present in the retrieved QueryResult."""
        allowed: set[str] = set()
        for item in result.items:
            m = item.memory
            if m.value_display:
                allowed.add(m.value_display.strip().lower())
            if m.value_norm:
                allowed.add(m.value_norm.strip().lower())
        for d in result.disputes:
            for m in (d.a.memory, d.b.memory):
                if m.value_display:
                    allowed.add(m.value_display.strip().lower())
                if m.value_norm:
                    allowed.add(m.value_norm.strip().lower())
        return allowed

    def _build_single_memory_explanation(self, item: MemoryView) -> str:
        """Construct a safe, deterministic explanation strictly from structured provenance fields.

        Governed by P7.3: Explanations must be constructed using structured provenance fields.
        Never infer causation, motivation, responsibility, or other relationships.
        If a safe explanation cannot be constructed (no source, confidence, or review), omit it.
        """
        m = item.memory
        source = m.source_id.strip() if m.source_id else None
        if not source and m.evidence and m.evidence[0].source_id:
            source = m.evidence[0].source_id.strip() or None

        conf = item.effective_confidence

        parts: list[str] = []
        if source and conf is not None and conf > 0.0:
            parts.append(f"Reported by {source} with confidence {conf:.2f}.")
        elif source:
            parts.append(f"Reported by {source}.")
        elif conf is not None and conf > 0.0:
            parts.append(f"Confidence {conf:.2f}.")

        if m.needs_review:
            parts.append("This memory is flagged for review.")

        return " ".join(parts)

    def _deterministic_compose(
        self,
        question: str,
        query: ParsedQuery,
        result: QueryResult,
    ) -> ComposedAnswer:
        """Compose a safe, factual answer deterministically from retrieved memory records."""
        # 1. Scope ambiguity
        if result.is_ambiguous:
            reason = result.ambiguity_reason or "Ambiguous context scope detected."
            return ComposedAnswer(
                answer_text=reason,
                explanation=reason,
                is_uncertain=True,
                is_complete=False,
                conflict_flagged=True,
                abstained=True,
            )

        # 2. Active disputes
        if result.disputes:
            d = result.disputes[0]
            ans = (
                f"Active dispute for {d.a.memory.entity_name} {d.a.memory.attribute}: "
                f"Memory M{d.a.memory.id} ('{d.a.memory.value_display}') contradicts "
                f"Memory M{d.b.memory.id} ('{d.b.memory.value_display}')."
            )
            return ComposedAnswer(
                answer_text=ans,
                values=[d.a.memory.value_display, d.b.memory.value_display],
                referenced_memory_ids=[d.a.memory.id, d.b.memory.id],
                explanation="Conflicting claims exist for this information.",
                is_uncertain=True,
                is_complete=False,
                conflict_flagged=True,
            )

        # 3. Empty results (no memories found)
        if not result.items:
            note = (
                result.notes[0] if result.notes else f"No relevant memory found for '{question}'."
            )
            return ComposedAnswer(
                answer_text=note,
                explanation=(
                    "No relevant memory matched the entity, attribute, and temporal criteria."
                ),
                no_relevant_memory=True,
                abstained=True,
                is_uncertain=True,
                is_complete=False,
            )

        # 4. Single-memory answer
        if len(result.items) == 1:
            item = result.items[0]
            m = item.memory
            val = m.value_display
            ans = f"{m.entity_name}'s {m.attribute} is {val} [M{m.id}]."

            needs_review = m.needs_review
            is_uncertain = needs_review
            explanation = self._build_single_memory_explanation(item)
            if needs_review:
                ans += f" Note: Memory M{m.id} is flagged for review."

            return ComposedAnswer(
                answer_text=ans,
                value=val,
                values=[val],
                referenced_memory_ids=[m.id],
                explanation=explanation,
                is_uncertain=is_uncertain,
                is_complete=not needs_review,
                needs_review_flagged=needs_review,
            )

        # 5. Multiple memories (multi-valued or historical timeline)
        m0 = result.items[0].memory
        vals = [it.memory.value_display for it in result.items]
        ids = [it.memory.id for it in result.items]
        needs_review = any(it.memory.needs_review for it in result.items)

        if query.intent == "history":
            hist_parts = [
                f"{it.memory.value_display} [M{it.memory.id}] "
                f"(from {it.memory.valid_from.strftime('%Y-%m-%d')})"
                for it in result.items
            ]
            ans = (
                f"Historical values for {m0.entity_name}'s {m0.attribute}: {', '.join(hist_parts)}."
            )
        else:
            val_citations = [f"{it.memory.value_display} [M{it.memory.id}]" for it in result.items]
            ans = (
                f"{m0.entity_name}'s {m0.attribute} has {len(vals)} active values: "
                f"{', '.join(val_citations)}."
            )

        explanation = f"Retrieved {len(result.items)} memory records."
        if needs_review:
            explanation += " This memory is flagged for review."

        return ComposedAnswer(
            answer_text=ans,
            value=vals[0],
            values=vals,
            referenced_memory_ids=ids,
            explanation=explanation,
            is_uncertain=needs_review,
            is_complete=not needs_review,
            needs_review_flagged=needs_review,
        )

    def compose(
        self,
        question: str,
        query: ParsedQuery,
        result: QueryResult,
        *,
        reference_time: datetime | None = None,
        model: str | None = None,
    ) -> ComposedAnswer:
        """Compose a structured, grounded answer from a query and retrieved memories."""
        if reference_time is None:
            reference_time = datetime.now(UTC)
        elif reference_time.tzinfo is None or reference_time.utcoffset() is None:
            reference_time = reference_time.replace(tzinfo=UTC)

        # Authoritative factual answer and explanation are ALWAYS deterministic
        det_answer = self._deterministic_compose(question, query, result)

        allowed_ids = self._extract_allowed_memory_ids(result)

        # When no memories or disputes exist, strictly avoid LLM call to prevent hallucinations
        if not result.items and not result.disputes and not result.impact:
            return det_answer

        # Fallback if no LLMClient configured
        if self.llm_client is None:
            return det_answer

        intent_str = query.intent.value if hasattr(query.intent, "value") else str(query.intent)
        system_prompt, user_prompt = build_compose_answer_prompt(
            question,
            result,
            intent=intent_str,
            reference_time=reference_time,
        )

        try:
            raw_res = self.llm_client.complete_json(
                prompt_id="compose_answer",
                system=system_prompt,
                user=user_prompt,
                response_model=LLMAnswerComposeResult,
                model=model,
            )
        except Exception as e:
            logger.warning(
                "AnswerComposer LLM call failed; falling back to deterministic answer: %s", e
            )
            return det_answer

        # Grounding & citation sanity verification on LLM response
        has_invalid_id = any(mid not in allowed_ids for mid in raw_res.referenced_memory_ids)
        text_citation_ids = [int(m) for m in re.findall(r"\[M(\d+)\]", raw_res.answer_text)]
        if any(mid not in allowed_ids for mid in text_citation_ids):
            has_invalid_id = True

        if has_invalid_id:
            logger.warning("AnswerComposer detected invalid memory IDs in LLM response")
            return det_answer

        if raw_res.unsupported_claims:
            logger.warning("AnswerComposer detected unsupported claims in LLM output")
            return det_answer

        if raw_res.no_relevant_memory and result.items:
            logger.warning("AnswerComposer detected LLM contradiction of engine retrieval")
            return det_answer

        if result.items:
            allowed_vals = self._extract_allowed_values(result)
            if raw_res.value is not None and raw_res.value.strip().lower() not in allowed_vals:
                logger.warning("AnswerComposer value contradicts retrieved memories")
                return det_answer
            if raw_res.values:
                for v in raw_res.values:
                    if v.strip().lower() not in allowed_vals:
                        logger.warning("AnswerComposer values contain ungrounded value")
                        return det_answer

        # Final answer: Both factual answer statement AND explanation are strictly deterministic
        # constructed from structured provenance fields.
        return ComposedAnswer(
            answer_text=det_answer.answer_text,
            value=det_answer.value,
            values=det_answer.values,
            referenced_memory_ids=det_answer.referenced_memory_ids,
            explanation=det_answer.explanation,
            is_uncertain=det_answer.is_uncertain,
            is_complete=det_answer.is_complete,
            no_relevant_memory=det_answer.no_relevant_memory,
            conflict_flagged=det_answer.conflict_flagged,
            needs_review_flagged=det_answer.needs_review_flagged,
            abstained=det_answer.abstained,
        )
