"""LLM-assisted structured claim extraction (P7.2).

Governed by docs/ARCHITECTURE.md Section 7, docs/SPECIFICATION.md Section 3,
and MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P7.2).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from memorymesh.config import AttributeConfig
from memorymesh.errors import LLMOutputError
from memorymesh.llm.client import LLMClient
from memorymesh.llm.prompts import build_extract_claims_prompt
from memorymesh.models import (
    AttributeDef,
    Claim,
    ExtractedClaim,
    ExtractionResult,
    IngestMessage,
)

logger = logging.getLogger(__name__)


class ClaimExtractor:
    """Extracts structured claim candidates from natural language messages using LLMClient."""

    def __init__(
        self,
        llm_client: LLMClient,
        attributes: list[AttributeDef] | list[AttributeConfig] | None = None,
        known_entities: list[str] | None = None,
        default_model: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.attributes = list(attributes) if attributes else []
        self.known_entities = list(known_entities) if known_entities else []
        self.default_model = default_model

        # Build alias lookup map for attribute normalization
        self._attr_lookup: dict[str, str] = {}
        for a in self.attributes:
            canonical = a.attribute
            self._attr_lookup[canonical.lower()] = canonical
            self._attr_lookup[canonical.replace("_", " ").lower()] = canonical
            if hasattr(a, "aliases") and a.aliases:
                for alias in a.aliases:
                    self._attr_lookup[alias.strip().lower()] = canonical

    def _normalize_attribute_name(self, raw_attr: str) -> str:
        """Resolve a raw or aliased attribute name to its canonical catalog attribute name."""
        cleaned = raw_attr.strip().lower()
        if cleaned in self._attr_lookup:
            return self._attr_lookup[cleaned]
        # Also try underscore version
        with_underscores = cleaned.replace(" ", "_")
        if with_underscores in self._attr_lookup:
            return self._attr_lookup[with_underscores]
        return raw_attr.strip()

    def _post_process_claim(
        self,
        candidate: ExtractedClaim,
        raw_message_text: str,
    ) -> ExtractedClaim | None:
        """Validate and post-process an extracted claim candidate."""
        # 1. Require non-empty subject and value
        subject = candidate.subject.strip()
        value = candidate.value.strip()
        if not subject or not value:
            return None

        # 2. Canonicalize attribute name
        canonical_attr = self._normalize_attribute_name(candidate.attribute)

        # 3. Clean and verify previous_value
        prev_val = candidate.previous_value.strip() if candidate.previous_value else None
        if prev_val == "":
            prev_val = None

        # 4. Normalize timestamps to timezone-aware UTC
        valid_from = candidate.valid_from
        if valid_from is not None:
            if valid_from.tzinfo is None or valid_from.utcoffset() is None:
                valid_from = valid_from.replace(tzinfo=UTC)
            else:
                valid_from = valid_from.astimezone(UTC)

        valid_to = candidate.valid_to
        if valid_to is not None:
            if valid_to.tzinfo is None or valid_to.utcoffset() is None:
                valid_to = valid_to.replace(tzinfo=UTC)
            else:
                valid_to = valid_to.astimezone(UTC)

        # 5. Bound extract_confidence
        confidence = max(0.0, min(1.0, float(candidate.extract_confidence)))

        # 6. Verify and sanitize evidence span
        span = candidate.span.strip() if candidate.span else ""
        if not span or span not in raw_message_text:
            # Fall back to matching value or full text
            if value in raw_message_text:
                span = value
            else:
                span = raw_message_text.strip()

        return ExtractedClaim(
            subject=subject,
            attribute=canonical_attr,
            value=value,
            previous_value=prev_val,
            polarity=candidate.polarity,
            valid_from=valid_from,
            valid_to=valid_to,
            context=dict(candidate.context),
            depends_on=list(candidate.depends_on),
            extract_confidence=confidence,
            span=span,
        )

    def extract(
        self,
        message: IngestMessage | str,
        *,
        reference_time: datetime | None = None,
        source_name: str = "user",
        model: str | None = None,
        raise_on_error: bool = False,
    ) -> ExtractionResult:
        """Extract structured claim candidates from natural language text.

        Args:
            message: IngestMessage or raw text string.
            reference_time: Reference timestamp for relative date resolution.
            source_name: Identifier of the source.
            model: Optional model override.
            raise_on_error: If True, re-raise LLMOutputError; if False, return empty result.

        Returns:
            ExtractionResult: Validated and post-processed claims.
        """
        if isinstance(message, IngestMessage):
            text = message.text
            ref_time = message.event_time
            src = message.source_id
        else:
            text = str(message)
            ref_time = reference_time or datetime.now(UTC)
            src = source_name

        system_prompt, user_prompt = build_extract_claims_prompt(
            message_text=text,
            reference_time=ref_time,
            source_name=src,
            attributes=self.attributes,
            known_entities=self.known_entities,
        )

        eff_model = model or self.default_model

        try:
            raw_result = self.llm_client.complete_json(
                prompt_id="extract_claims",
                system=system_prompt,
                user=user_prompt,
                response_model=ExtractionResult,
                model=eff_model,
            )
        except LLMOutputError as exc:
            logger.warning("LLM extraction failed after repair retry: %s", exc)
            if raise_on_error:
                raise
            return ExtractionResult(claims=[], error=str(exc))
        except Exception as exc:
            logger.warning("LLM extraction provider failure: %s", exc)
            if raise_on_error:
                raise
            return ExtractionResult(claims=[], error=str(exc))

        # Post-process all extracted claims
        sanitized_claims: list[ExtractedClaim] = []
        for raw_claim in raw_result.claims:
            processed = self._post_process_claim(raw_claim, text)
            if processed is not None:
                sanitized_claims.append(processed)

        return ExtractionResult(claims=sanitized_claims)

    def to_domain_claims(
        self,
        result: ExtractionResult,
        message: IngestMessage | None = None,
    ) -> list[Claim]:
        """Convert an ExtractionResult into domain Claim models ready for ingestion."""
        domain_claims: list[Claim] = []
        for extracted in result.claims:
            claim = extracted.to_claim()
            # If valid_from was not explicitly stated, fall back to message event_time
            if claim.valid_from is None and message is not None:
                claim.valid_from = (
                    message.event_time.astimezone(UTC)
                    if message.event_time.tzinfo is not None
                    else message.event_time.replace(tzinfo=UTC)
                )
            domain_claims.append(claim)
        return domain_claims

    def extract_claims_for_ingest(
        self,
        message: IngestMessage,
        *,
        model: str | None = None,
        raise_on_error: bool = False,
    ) -> list[Claim]:
        """Convenience method: extract candidates and convert directly to domain Claims."""
        result = self.extract(
            message,
            model=model,
            raise_on_error=raise_on_error,
        )
        return self.to_domain_claims(result, message=message)
