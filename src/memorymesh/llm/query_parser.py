"""LLM-assisted natural-language query parser (P7.3).

Governed by docs/ARCHITECTURE.md Section 4, docs/SPECIFICATION.md Section 3,
and MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P7.3).
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from pydantic import Field

from memorymesh.config import AttributeConfig
from memorymesh.llm.client import LLMClient
from memorymesh.llm.prompts import build_parse_query_prompt
from memorymesh.models import (
    AttributeDef,
    MeshBaseModel,
    ParsedQuery,
    QueryIntent,
    QueryType,
    TemporalIntent,
)

logger = logging.getLogger(__name__)


class LLMQueryParseResult(MeshBaseModel):
    """Raw structured output schema from the query parsing LLM prompt."""

    entity: str | None = None
    attribute: str | None = None
    context: dict[str, str] = Field(default_factory=dict)
    intent: str = "current"
    valid_at: str | None = None
    known_at: str | None = None
    target_date: str | None = None
    reasoning: str = ""


MONTH_LOOKUP: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def _is_belief_time_query(text: str) -> bool:
    """Return True if the question asks about MemoryMesh's knowledge or beliefs."""
    q_low = text.lower()
    return bool(re.search(r"\b(believe|believed|belief|know|knew|known|recorded)\b", q_low))


class QueryParser:
    """Interprets natural-language user questions into structured retrieval queries."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        attributes: list[AttributeDef] | list[AttributeConfig] | None = None,
        known_entities: list[str] | None = None,
        default_model: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.attributes = list(attributes) if attributes else []
        self.known_entities = list(known_entities) if known_entities else []
        self.default_model = default_model

        # Build attribute lookup dictionary for normalization
        self._attr_lookup: dict[str, str] = {}
        for a in self.attributes:
            canonical = a.attribute
            self._attr_lookup[canonical.lower()] = canonical
            self._attr_lookup[canonical.replace("_", " ").lower()] = canonical
            self._attr_lookup[canonical.replace("_", "-").lower()] = canonical
            if canonical == "operating_system":
                self._attr_lookup["os"] = canonical
            elif canonical == "python_version":
                self._attr_lookup["python"] = canonical
            elif canonical == "deploy_day":
                self._attr_lookup["deploy"] = canonical
                self._attr_lookup["deploys"] = canonical
            if hasattr(a, "aliases") and a.aliases:
                for alias in a.aliases:
                    self._attr_lookup[alias.strip().lower()] = canonical
                    self._attr_lookup[alias.strip().replace("_", " ").lower()] = canonical

    def _normalize_attribute_name(self, raw_attr: str | None) -> str | None:
        """Resolve a raw or aliased attribute name to its canonical catalog attribute name."""
        if not raw_attr:
            return None
        cleaned = raw_attr.strip().lower()
        if cleaned in self._attr_lookup:
            return self._attr_lookup[cleaned]
        with_underscores = cleaned.replace(" ", "_").replace("-", "_")
        if with_underscores in self._attr_lookup:
            return self._attr_lookup[with_underscores]

        # Partial matching across canonical attributes
        for key, canonical in self._attr_lookup.items():
            if key in cleaned or cleaned in key:
                return canonical
        return raw_attr.strip()

    def _normalize_entity_name(self, raw_entity: str | None) -> str | None:
        """Resolve an entity name against known entities if a close match exists."""
        if not raw_entity:
            return None
        cleaned = raw_entity.strip()
        lower_cleaned = cleaned.lower()

        # Exact match
        for known in self.known_entities:
            if known.lower() == lower_cleaned:
                return known

        # Substring match
        for known in self.known_entities:
            if known.lower() in lower_cleaned or lower_cleaned in known.lower():
                return known

        return cleaned

    def _parse_date_or_natural(
        self,
        date_str: str | None,
        default_year: int = 2026,
    ) -> tuple[datetime | None, str | None]:
        """Parse an ISO date, timestamp, or natural English date into a UTC datetime."""
        if not date_str:
            return None, None
        cleaned = date_str.strip()

        # 1. Standard ISO formats
        try:
            dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
            if dt.tzinfo is None or dt.utcoffset() is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC), cleaned
        except Exception:
            pass

        # 2. YYYY-MM-DD
        m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", cleaned)
        if m:
            try:
                y, mon, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return datetime(y, mon, d, 0, 0, 0, tzinfo=UTC), m.group(0)
            except Exception:
                pass

        # 3. Natural date: Month Day [Year] (e.g. "March 15", "March 15th", "March 15, 2026")
        mon_names = "|".join(MONTH_LOOKUP.keys())
        nat_pattern1 = rf"\b({mon_names})[a-z]*\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b"
        m1 = re.search(nat_pattern1, cleaned, re.IGNORECASE)
        if m1:
            try:
                mon_str = m1.group(1).lower()
                mon = MONTH_LOOKUP[mon_str]
                day = int(m1.group(2))
                yr = int(m1.group(3)) if m1.group(3) else default_year
                return datetime(yr, mon, day, 0, 0, 0, tzinfo=UTC), m1.group(0)
            except Exception:
                pass

        # 4. Natural date: Day Month [Year] (e.g. "15th of March", "15 March 2026")
        nat_pattern2 = (
            rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({mon_names})[a-z]*(?:,?\s+(\d{{4}}))?\b"
        )
        m2 = re.search(nat_pattern2, cleaned, re.IGNORECASE)
        if m2:
            try:
                day = int(m2.group(1))
                mon_str = m2.group(2).lower()
                mon = MONTH_LOOKUP[mon_str]
                yr = int(m2.group(3)) if m2.group(3) else default_year
                return datetime(yr, mon, day, 0, 0, 0, tzinfo=UTC), m2.group(0)
            except Exception:
                pass

        # 5. Standalone month: [in / during / as of] Month [Year]
        nat_pattern3 = rf"\b(?:in\s+|during\s+|as\s+of\s+)?({mon_names})[a-z]*(?:\s+(\d{{4}}))?\b"
        m3 = re.search(nat_pattern3, cleaned, re.IGNORECASE)
        if m3:
            try:
                mon_str = m3.group(1).lower()
                mon = MONTH_LOOKUP[mon_str]
                yr = int(m3.group(2)) if m3.group(2) else default_year
                return datetime(yr, mon, 15, 0, 0, 0, tzinfo=UTC), m3.group(0).strip()
            except Exception:
                pass

        return None, None

    def _parse_iso_date(self, date_str: str | None) -> datetime | None:
        """Parse an ISO date or timestamp into a timezone-aware UTC datetime."""
        dt, _ = self._parse_date_or_natural(date_str)
        return dt

    def _parse_bitemporal_dates(
        self,
        question: str,
        default_year: int = 2026,
    ) -> tuple[datetime | None, datetime | None, str | None]:
        """Extract valid_at and known_at dates independently from text."""
        q_lower = question.lower()
        mon_names = "|".join(MONTH_LOOKUP.keys())

        # Collect all matches with start positions and parsed datetimes
        found_dates: list[tuple[int, int, datetime, str]] = []

        # 1. ISO format: YYYY-MM-DD
        for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", question):
            try:
                y, mon, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                dt = datetime(y, mon, d, 0, 0, 0, tzinfo=UTC)
                found_dates.append((m.start(), m.end(), dt, m.group(0)))
            except Exception:
                pass

        # 2. Natural date: Month Day [Year]
        nat_pattern1 = rf"\b({mon_names})[a-z]*\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b"
        for m in re.finditer(nat_pattern1, question, re.IGNORECASE):
            try:
                mon = MONTH_LOOKUP[m.group(1).lower()]
                day = int(m.group(2))
                yr = int(m.group(3)) if m.group(3) else default_year
                dt = datetime(yr, mon, day, 0, 0, 0, tzinfo=UTC)
                found_dates.append((m.start(), m.end(), dt, m.group(0)))
            except Exception:
                pass

        # 3. Natural date: Day Month [Year]
        nat_pattern2 = (
            rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({mon_names})[a-z]*(?:,?\s+(\d{{4}}))?\b"
        )
        for m in re.finditer(nat_pattern2, question, re.IGNORECASE):
            try:
                day = int(m.group(1))
                mon = MONTH_LOOKUP[m.group(2).lower()]
                yr = int(m.group(3)) if m.group(3) else default_year
                dt = datetime(yr, mon, day, 0, 0, 0, tzinfo=UTC)
                found_dates.append((m.start(), m.end(), dt, m.group(0)))
            except Exception:
                pass

        # 4. Standalone month: [in / during / as of] Month [Year]
        nat_pattern3 = rf"\b(?:in\s+|during\s+|as\s+of\s+)?({mon_names})[a-z]*(?:\s+(\d{{4}}))?\b"
        for m in re.finditer(nat_pattern3, question, re.IGNORECASE):
            # Skip if this overlaps with any already identified date span
            if any(m.start() < end and m.end() > start for start, end, _, _ in found_dates):
                continue
            try:
                mon = MONTH_LOOKUP[m.group(1).lower()]
                yr = int(m.group(2)) if m.group(2) else default_year
                dt = datetime(yr, mon, 15, 0, 0, 0, tzinfo=UTC)
                found_dates.append((m.start(), m.end(), dt, m.group(0).strip()))
            except Exception:
                pass

        # Deduplicate matches by span
        unique_dates: list[tuple[int, int, datetime, str]] = []
        seen_spans: set[tuple[int, int]] = set()
        for item in sorted(found_dates, key=lambda x: x[0]):
            span = (item[0], item[1])
            if span not in seen_spans:
                seen_spans.add(span)
                unique_dates.append(item)

        if not unique_dates:
            return None, None, None

        if len(unique_dates) == 1:
            _, _, dt, text_str = unique_dates[0]
            if _is_belief_time_query(q_lower):
                return None, dt, text_str
            return dt, None, text_str

        # If 2 or more dates are found: assign based on surrounding context
        valid_at: datetime | None = None
        known_at: datetime | None = None
        date_strs: list[str] = []

        for start, _end, dt, dstr in unique_dates:
            date_strs.append(dstr)
            prefix = q_lower[max(0, start - 35) : start]
            if any(
                w in prefix for w in ("know", "knew", "known", "believe", "believed", "recorded")
            ):
                if known_at is None:
                    known_at = dt
                    continue
            if any(
                w in prefix
                for w in ("valid", "running", "planned", "effective", "for", "about", "change")
            ):
                if valid_at is None:
                    valid_at = dt
                    continue
            if known_at is None and _is_belief_time_query(prefix):
                known_at = dt
            elif valid_at is None:
                valid_at = dt
            elif known_at is None:
                known_at = dt

        return valid_at, known_at, " / ".join(date_strs)

    def _deterministic_parse(
        self,
        question: str,
        reference_time: datetime,
    ) -> ParsedQuery:
        """Safe deterministic query interpretation when an LLM is unavailable or fails."""
        q_lower = question.lower()

        # 1. Detect query intent
        intent = QueryIntent.CURRENT
        query_type = QueryType.FACT_LOOKUP
        temporal_intent = TemporalIntent.CURRENT
        is_historical = False
        is_why = False
        is_impact = False

        if any(
            w in q_lower for w in ("why", "how do we know", "who said", "evidence", "provenance")
        ):
            intent = QueryIntent.WHY
            query_type = QueryType.EXPLANATION
            is_why = True
        elif any(w in q_lower for w in ("depend", "impact", "blast radius", "affected", "break")):
            intent = QueryIntent.IMPACT
            query_type = QueryType.DEPENDENCY_IMPACT
            is_impact = True
        elif any(w in q_lower for w in ("dispute", "conflict", "contradict")):
            intent = QueryIntent.CONFLICTS
            query_type = QueryType.DISPUTE_CHECK
        elif any(
            w in q_lower
            for w in ("history", "timeline", "past", "previously", "previous", "had", "versions")
        ):
            intent = QueryIntent.HISTORY
            query_type = QueryType.TIMELINE_HISTORY
            temporal_intent = TemporalIntent.HISTORY
            is_historical = True

        # 2. Check for explicit date/time (bitemporal: valid_at vs known_at)
        valid_at, known_at, date_str = self._parse_bitemporal_dates(question, reference_time.year)
        if valid_at is not None or known_at is not None:
            intent = QueryIntent.AS_OF
            temporal_intent = TemporalIntent.AS_OF
            is_historical = True

        # 3. Match known entity
        matched_entity: str | None = None
        for ent in sorted(self.known_entities, key=len, reverse=True):
            if ent.lower() in q_lower:
                matched_entity = ent
                break
        if matched_entity is None:
            m_ent = re.search(r"\b(server\s+[a-z0-9_-]+)\b", question, re.IGNORECASE)
            if m_ent:
                matched_entity = m_ent.group(1).title()

        # 4. Match attribute
        matched_attr: str | None = None
        for key in sorted(self._attr_lookup.keys(), key=len, reverse=True):
            if re.search(r"\b" + re.escape(key) + r"\b", q_lower):
                matched_attr = self._attr_lookup[key]
                break

        # 5. Extract context hints (e.g. env=prod, in prod)
        context: dict[str, str] = {}
        env_match = re.search(
            r"\b(?:in|env=)\s*(prod(?:uction)?|staging|dev(?:elopment)?)\b", q_lower
        )
        if env_match:
            val = env_match.group(1)
            context["env"] = "prod" if "prod" in val else ("staging" if "staging" in val else "dev")

        return ParsedQuery(
            original_question=question,
            entity=matched_entity,
            attribute=matched_attr,
            context=context,
            intent=intent,
            temporal_intent=temporal_intent,
            query_type=query_type,
            valid_at=valid_at,
            known_at=known_at,
            date_str=date_str,
            is_historical=is_historical,
            is_why=is_why,
            is_impact=is_impact,
            confidence=0.7,
            reasoning="Deterministic rule-based query parsing",
        )

    def parse(
        self,
        question: str,
        *,
        reference_time: datetime | None = None,
        model: str | None = None,
    ) -> ParsedQuery:
        """Parse a natural-language question into a structured query."""
        if reference_time is None:
            reference_time = datetime.now(UTC)
        elif reference_time.tzinfo is None or reference_time.utcoffset() is None:
            reference_time = reference_time.replace(tzinfo=UTC)

        # Fallback if no LLM configured
        if self.llm_client is None:
            return self._deterministic_parse(question, reference_time)

        system_prompt, user_prompt = build_parse_query_prompt(
            question,
            reference_time=reference_time,
            attributes=self.attributes,
            known_entities=self.known_entities,
        )

        try:
            raw_res = self.llm_client.complete_json(
                prompt_id="parse_query",
                system=system_prompt,
                user=user_prompt,
                response_model=LLMQueryParseResult,
                model=model,
            )
        except Exception as e:
            logger.warning(
                "QueryParser LLM call failed; falling back to deterministic parsing: %s", e
            )
            return self._deterministic_parse(question, reference_time)

        # Normalize attribute and entity
        norm_entity = self._normalize_entity_name(raw_res.entity)
        norm_attr = self._normalize_attribute_name(raw_res.attribute)

        # Parse bitemporal dates
        valid_at: datetime | None = None
        known_at: datetime | None = None
        date_str: str | None = None

        if raw_res.valid_at:
            valid_at, date_str = self._parse_date_or_natural(raw_res.valid_at, reference_time.year)
        if raw_res.known_at:
            known_at, date_str = self._parse_date_or_natural(raw_res.known_at, reference_time.year)
        if raw_res.target_date and not valid_at and not known_at:
            parsed_dt, date_str = self._parse_date_or_natural(
                raw_res.target_date, reference_time.year
            )
            if parsed_dt is not None:
                if _is_belief_time_query(question):
                    known_at = parsed_dt
                else:
                    valid_at = parsed_dt

        # Fallback to question text if LLM omitted either or both dates
        nat_valid, nat_known, nat_dstr = self._parse_bitemporal_dates(question, reference_time.year)
        if valid_at is None and nat_valid is not None:
            valid_at = nat_valid
        if known_at is None and nat_known is not None:
            known_at = nat_known
        if not date_str:
            date_str = nat_dstr

        # Map raw intent string to QueryIntent enum
        raw_intent = (raw_res.intent or "current").strip().lower()
        intent = QueryIntent.CURRENT
        for qi in QueryIntent:
            if qi.value == raw_intent:
                intent = qi
                break

        # If a valid past date or belief date is given, override to AS_OF if current
        if (valid_at is not None or known_at is not None) and intent == QueryIntent.CURRENT:
            intent = QueryIntent.AS_OF

        # Infer temporal intent and query type
        is_why = intent == QueryIntent.WHY
        is_impact = intent == QueryIntent.IMPACT
        is_historical = (
            intent in (QueryIntent.AS_OF, QueryIntent.HISTORY)
            or valid_at is not None
            or known_at is not None
        )

        temporal_intent = TemporalIntent.CURRENT
        if intent == QueryIntent.AS_OF or valid_at is not None or known_at is not None:
            temporal_intent = TemporalIntent.AS_OF
        elif intent == QueryIntent.HISTORY:
            temporal_intent = TemporalIntent.HISTORY

        query_type = QueryType.FACT_LOOKUP
        if is_why:
            query_type = QueryType.EXPLANATION
        elif is_impact:
            query_type = QueryType.DEPENDENCY_IMPACT
        elif intent == QueryIntent.CONFLICTS:
            query_type = QueryType.DISPUTE_CHECK
        elif intent == QueryIntent.HISTORY:
            query_type = QueryType.TIMELINE_HISTORY

        # Clean context dict
        cleaned_ctx = {}
        if isinstance(raw_res.context, dict):
            for k, v in raw_res.context.items():
                if k and v:
                    cleaned_ctx[str(k).strip()] = str(v).strip()

        return ParsedQuery(
            original_question=question,
            entity=norm_entity,
            attribute=norm_attr,
            context=cleaned_ctx,
            intent=intent,
            temporal_intent=temporal_intent,
            query_type=query_type,
            valid_at=valid_at,
            known_at=known_at,
            date_str=date_str or raw_res.target_date,
            is_historical=is_historical,
            is_why=is_why,
            is_impact=is_impact,
            confidence=1.0,
            reasoning=raw_res.reasoning or "LLM structured interpretation",
        )
