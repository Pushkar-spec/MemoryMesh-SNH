"""LLM provider foundation, client infrastructure, prompts, and extraction (P7.1, P7.2, P7.3)."""

from __future__ import annotations

from memorymesh.llm.answer_composer import AnswerComposer
from memorymesh.llm.classifier import LLMRelationClassifier, classify_relation_llm
from memorymesh.llm.client import (
    LLMClient,
    extract_json_payload,
    sanitize_sensitive_text,
)
from memorymesh.llm.extractor import ClaimExtractor
from memorymesh.llm.prompts import (
    build_classify_relation_prompt,
    build_compose_answer_prompt,
    build_extract_claims_prompt,
    build_parse_query_prompt,
    format_attribute_catalog,
    format_known_entities,
    format_retrieved_context_for_composer,
)
from memorymesh.llm.providers import (
    AnthropicProvider,
    FakeProvider,
    GeminiProvider,
    LLMProvider,
    get_provider,
)
from memorymesh.llm.query_parser import QueryParser
from memorymesh.models import (
    ComposedAnswer,
    ExtractedClaim,
    ExtractionResult,
    ParsedQuery,
    QueryIntent,
    QueryType,
    RelationClassificationResult,
    RelationProposal,
    TemporalIntent,
)

__all__ = [
    "AnswerComposer",
    "AnthropicProvider",
    "ClaimExtractor",
    "ComposedAnswer",
    "ExtractedClaim",
    "ExtractionResult",
    "FakeProvider",
    "GeminiProvider",
    "LLMClient",
    "LLMProvider",
    "LLMRelationClassifier",
    "ParsedQuery",
    "QueryIntent",
    "QueryParser",
    "QueryType",
    "RelationClassificationResult",
    "RelationProposal",
    "TemporalIntent",
    "build_classify_relation_prompt",
    "build_compose_answer_prompt",
    "build_extract_claims_prompt",
    "build_parse_query_prompt",
    "classify_relation_llm",
    "extract_json_payload",
    "format_attribute_catalog",
    "format_known_entities",
    "format_retrieved_context_for_composer",
    "get_provider",
    "sanitize_sensitive_text",
]
