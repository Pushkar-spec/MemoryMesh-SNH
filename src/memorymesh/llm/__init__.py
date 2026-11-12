"""LLM provider foundation, client infrastructure, prompts, and extraction (P7.1, P7.2)."""

from __future__ import annotations

from memorymesh.llm.classifier import LLMRelationClassifier, classify_relation_llm
from memorymesh.llm.client import (
    LLMClient,
    extract_json_payload,
    sanitize_sensitive_text,
)
from memorymesh.llm.extractor import ClaimExtractor
from memorymesh.llm.prompts import (
    build_classify_relation_prompt,
    build_extract_claims_prompt,
    format_attribute_catalog,
    format_known_entities,
)
from memorymesh.llm.providers import (
    AnthropicProvider,
    FakeProvider,
    GeminiProvider,
    LLMProvider,
    get_provider,
)
from memorymesh.models import (
    ExtractedClaim,
    ExtractionResult,
    RelationClassificationResult,
    RelationProposal,
)

__all__ = [
    "AnthropicProvider",
    "ClaimExtractor",
    "ExtractedClaim",
    "ExtractionResult",
    "FakeProvider",
    "GeminiProvider",
    "LLMClient",
    "LLMProvider",
    "LLMRelationClassifier",
    "RelationClassificationResult",
    "RelationProposal",
    "build_classify_relation_prompt",
    "build_extract_claims_prompt",
    "classify_relation_llm",
    "extract_json_payload",
    "format_attribute_catalog",
    "format_known_entities",
    "get_provider",
    "sanitize_sensitive_text",
]
