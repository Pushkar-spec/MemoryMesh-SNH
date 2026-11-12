"""Centralized, provider-agnostic LLM client infrastructure.

Governed by docs/ARCHITECTURE.md Section 7, docs/SPECIFICATION.md Section 3,
and MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P7.1).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from memorymesh.config import LLMConfig
from memorymesh.errors import LLMOutputError
from memorymesh.llm.providers import LLMProvider
from memorymesh.models import LLMCall, LLMCallStatus
from memorymesh.repository import Repository

T = TypeVar("T", bound=BaseModel)

SENSITIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Authorization / Bearer tokens
    (re.compile(r"(?i)\b(bearer\s*[:=]?\s*)['\"]?[A-Za-z0-9_\-\.]{6,}['\"]?"), r"\1[REDACTED]"),
    # Anthropic API keys
    (re.compile(r"\b(sk-ant-[A-Za-z0-9_\-]{8,})\b"), "[REDACTED]"),
    # OpenAI and generic sk- keys
    (re.compile(r"\b(sk-[A-Za-z0-9_\-]{10,})\b"), "[REDACTED]"),
    # Google AI / Gemini API keys
    (re.compile(r"\b(AIza[0-9A-Za-z_\-]{30,})\b"), "[REDACTED]"),
    # Key-value assignments and JSON fields: api_key=..., "api_key": "...", etc.
    (
        re.compile(
            r"""(?i)(['"]?(?:api[_-]?key|secret(?:[_-]?key)?|password|passwd|auth[_-]?token|access[_-]?token|token)['"]?\s*[:=]\s*['"]?)[^\s'",;&]{4,}"""
        ),
        r"\1[REDACTED]",
    ),
    # Basic auth credentials in URLs
    (re.compile(r"(https?://)([^:]+):([^@]+)@"), r"\1[REDACTED]:[REDACTED]@"),
    # Private keys
    (
        re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----"),
        "[REDACTED_PRIVATE_KEY]",
    ),
]


def sanitize_sensitive_text(text: str | None, max_len: int | None = None) -> str:
    """Sanitize sensitive credentials and bound text length for safe logging and error reporting."""
    if not text:
        return ""

    sanitized = text

    # 1. Scrub exact matches of known environment variable secrets (if set)
    for env_var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        val = os.environ.get(env_var)
        if val and len(val) >= 6 and val in sanitized:
            sanitized = sanitized.replace(val, "[REDACTED]")

    # Also scrub any environment variable ending with _API_KEY, _SECRET, _TOKEN
    for key, val in os.environ.items():
        if (
            key.endswith(("_API_KEY", "_SECRET", "_TOKEN"))
            and val
            and len(val) >= 8
            and val in sanitized
        ):
            sanitized = sanitized.replace(val, "[REDACTED]")

    # 2. Scrub regex patterns
    for pattern, replacement in SENSITIVE_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)

    # 3. Bound text length if max_len is specified
    if max_len is not None and len(sanitized) > max_len:
        sanitized = sanitized[:max_len] + "... [truncated]"

    return sanitized


def extract_json_payload(text: str) -> Any:
    """Extract and parse JSON from raw LLM output text.

    Supports:
    - Pure JSON strings (objects or arrays)
    - JSON surrounded by whitespace or introductory/trailing prose
    - Markdown fenced code blocks (```json ... ``` or ``` ... ```)
    - Nested JSON objects and arrays without regex truncation
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Cannot extract JSON from empty text.")

    # 1. Try parsing directly (fast path for clean outputs)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2. Check for markdown code blocks: ```json ... ``` or ``` ... ```
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    matches = fence_pattern.findall(cleaned)
    for candidate in matches:
        cand_str = candidate.strip()
        if cand_str:
            try:
                return json.loads(cand_str)
            except Exception:
                continue

    # 3. Look for balanced JSON object {...} or array [...] in the text
    # Scan from the first brace/bracket tracking depth, respecting string quotes and escapes
    first_brace = cleaned.find("{")
    first_bracket = cleaned.find("[")

    start_idx = -1
    open_char = ""
    close_char = ""

    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start_idx = first_brace
        open_char = "{"
        close_char = "}"
    elif first_bracket != -1:
        start_idx = first_bracket
        open_char = "["
        close_char = "]"

    if start_idx != -1:
        depth = 0
        in_string = False
        escape = False
        end_idx = -1

        for i in range(start_idx, len(cleaned)):
            ch = cleaned[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                if in_string:
                    escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == open_char:
                    depth += 1
                elif ch == close_char:
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break

        if end_idx != -1:
            candidate = cleaned[start_idx:end_idx]
            try:
                return json.loads(candidate)
            except Exception:
                pass

    raise ValueError(f"Failed to extract valid JSON payload from text: {text[:200]!r}")


class LLMClient:
    """Centralized LLM client providing validated JSON completions.

    Features:
    - Provider-agnostic completion layer via LLMProvider protocol
    - Extraction of clean or Markdown-fenced JSON
    - Validation against caller-provided Pydantic models
    - Exactly one repair retry with schema guidance on invalid JSON or validation failures
    - Deterministic SHA-256 caching of validated responses (in-memory, process-local)
    - Synchronous concurrency control via threading.Semaphore (slot released for cache hits)
    - Structured audit logging into the SQLite llm_calls table via Repository
    """

    def __init__(
        self,
        provider: LLMProvider,
        repo: Repository | None = None,
        config: LLMConfig | None = None,
        *,
        default_model: str | None = None,
        max_concurrency: int | None = None,
        cache_enabled: bool | None = None,
    ) -> None:
        self.provider = provider
        self.repo = repo
        self.config = config

        if config is not None:
            self.default_model = default_model or config.fast_model
            self.max_concurrency = (
                max_concurrency if max_concurrency is not None else config.concurrency
            )
            self.cache_enabled = cache_enabled if cache_enabled is not None else config.cache
        else:
            self.default_model = default_model or "claude-haiku-5-5"
            self.max_concurrency = max_concurrency if max_concurrency is not None else 3
            self.cache_enabled = cache_enabled if cache_enabled is not None else True

        self._semaphore = threading.Semaphore(self.max_concurrency)
        # Process-local in-memory cache: maps SHA-256 cache key to validated BaseModel instance.
        # Note: This cache is in-memory and process-local; it resets on process restart.
        self._cache: dict[str, BaseModel] = {}
        self._cache_lock = threading.Lock()

    def _get_provider_name(self) -> str:
        """Derive an identifier for the injected LLM provider."""
        if hasattr(self.provider, "name") and isinstance(self.provider.name, str):
            return self.provider.name
        return self.provider.__class__.__name__

    def _summarize_prompt(self, user: str, system: str = "", max_len: int = 120) -> str:
        """Produce a sanitized single-line summary of prompt text for logging."""
        prompt_text = f"[{system}] {user}" if system else user
        sanitized = sanitize_sensitive_text(prompt_text)
        one_line = " ".join(sanitized.split())
        if len(one_line) > max_len:
            return one_line[:max_len] + "... [truncated]"
        return one_line

    def clear_cache(self) -> None:
        """Clear all entries from the in-memory response cache."""
        with self._cache_lock:
            self._cache.clear()

    def compute_cache_key(
        self,
        provider_name: str,
        model: str,
        system: str,
        user: str,
        response_model: type[BaseModel],
        max_tokens: int,
    ) -> str:
        """Compute a deterministic SHA-256 cache key for an LLM request."""
        schema_dict = response_model.model_json_schema()
        key_payload = {
            "provider": provider_name,
            "model": model,
            "system": system,
            "user": user,
            "schema": schema_dict,
            "max_tokens": max_tokens,
        }
        serialized = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _log_call(
        self,
        prompt_id: str,
        provider: str,
        model: str,
        prompt_summary: str,
        response_raw: str,
        parsed_json: str | None,
        latency_ms: float,
        status: LLMCallStatus,
        error: str | None,
        timestamp: datetime,
    ) -> None:
        """Record an LLM call in the repository's llm_calls table if repo is configured.

        Sanitizes sensitive credentials and bounds length for diagnostic persistence.
        """
        if self.repo is None:
            return

        call_id = f"call_{uuid.uuid4().hex[:16]}"
        safe_summary = sanitize_sensitive_text(prompt_summary, max_len=120)
        safe_raw = sanitize_sensitive_text(response_raw, max_len=1000)
        safe_parsed = (
            sanitize_sensitive_text(parsed_json, max_len=1000) if parsed_json is not None else None
        )
        safe_error = sanitize_sensitive_text(error, max_len=500) if error is not None else None

        call = LLMCall(
            id=call_id,
            timestamp=timestamp,
            prompt_id=prompt_id,
            provider=provider,
            model=model,
            prompt_summary=safe_summary,
            response_raw=safe_raw,
            parsed_json=safe_parsed,
            latency_ms=latency_ms,
            status=status,
            error=safe_error,
        )
        try:
            self.repo.insert_llm_call(call)
        except Exception:
            # Audit logging failure should not abort the caller unless repository invariants fail
            pass

    def complete_json(
        self,
        prompt_id: str,
        system: str,
        user: str,
        response_model: type[T],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> T:
        """Execute a completion request and return a validated Pydantic model instance.

        Workflow:
        1. Check response cache. If present, return copy without acquiring semaphore or logging.
        2. Acquire concurrency semaphore and call provider.
        3. Extract JSON and validate against response_model.
        4. If valid, log success, cache, and return.
        5. If invalid JSON or validation error, log retry, make exactly one repair call.
        6. If repair succeeds, log success, cache, and return.
        7. If repair fails, log error and raise LLMOutputError.
        """
        provider_name = self._get_provider_name()
        eff_model = model or self.default_model

        # 1. Cache lookup (zero concurrency slot acquisition for cache hits)
        cache_key = self.compute_cache_key(
            provider_name=provider_name,
            model=eff_model,
            system=system,
            user=user,
            response_model=response_model,
            max_tokens=max_tokens,
        )

        if self.cache_enabled and use_cache:
            with self._cache_lock:
                cached_item = self._cache.get(cache_key)
                if cached_item is not None and isinstance(cached_item, response_model):
                    return cached_item.model_copy(deep=True)  # type: ignore[return-value]

        # 2. First provider call with concurrency control
        call_time = datetime.now(UTC)
        start_time = time.perf_counter()
        raw_response = ""

        try:
            with self._semaphore:
                raw_response = self.provider.complete(
                    system=system,
                    user=user,
                    model=eff_model,
                    max_tokens=max_tokens,
                )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self._log_call(
                prompt_id=prompt_id,
                provider=provider_name,
                model=eff_model,
                prompt_summary=self._summarize_prompt(user, system=system),
                response_raw="",
                parsed_json=None,
                latency_ms=latency_ms,
                status=LLMCallStatus.ERROR,
                error=str(exc),
                timestamp=call_time,
            )
            raise

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 3. Parse and validate first response
        validation_error: Exception | None = None
        parsed_data: Any = None
        try:
            parsed_data = extract_json_payload(raw_response)
            validated_instance = response_model.model_validate(parsed_data)
            # Success on initial attempt
            self._log_call(
                prompt_id=prompt_id,
                provider=provider_name,
                model=eff_model,
                prompt_summary=self._summarize_prompt(user, system=system),
                response_raw=raw_response,
                parsed_json=json.dumps(parsed_data)
                if isinstance(parsed_data, (dict, list))
                else None,
                latency_ms=latency_ms,
                status=LLMCallStatus.SUCCESS,
                error=None,
                timestamp=call_time,
            )
            if self.cache_enabled and use_cache:
                with self._cache_lock:
                    self._cache[cache_key] = validated_instance.model_copy(deep=True)
            return validated_instance
        except (ValueError, ValidationError) as exc:
            validation_error = exc

        # 4. First attempt failed: log as RETRY and execute exactly one repair call
        self._log_call(
            prompt_id=prompt_id,
            provider=provider_name,
            model=eff_model,
            prompt_summary=self._summarize_prompt(user, system=system),
            response_raw=raw_response,
            parsed_json=json.dumps(parsed_data) if isinstance(parsed_data, (dict, list)) else None,
            latency_ms=latency_ms,
            status=LLMCallStatus.RETRY,
            error=str(validation_error),
            timestamp=call_time,
        )

        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        repair_user_prompt = (
            f"{user}\n\n"
            f"[REPAIR DIRECTIVE]\n"
            f"Your previous response was invalid JSON or failed schema validation.\n"
            f"Error details: {validation_error}\n\n"
            f"Previous output:\n{raw_response}\n\n"
            f"Please output ONLY a valid JSON object matching the required schema below:\n"
            f"{schema_json}\n"
            f"Do not include any explanation or markdown formatting other than the JSON object."
        )

        repair_call_time = datetime.now(UTC)
        repair_start_time = time.perf_counter()
        repair_raw_response = ""

        try:
            with self._semaphore:
                repair_raw_response = self.provider.complete(
                    system=system,
                    user=repair_user_prompt,
                    model=eff_model,
                    max_tokens=max_tokens,
                )
        except Exception as exc:
            repair_latency_ms = (time.perf_counter() - repair_start_time) * 1000.0
            self._log_call(
                prompt_id=prompt_id,
                provider=provider_name,
                model=eff_model,
                prompt_summary=self._summarize_prompt(repair_user_prompt, system=system),
                response_raw="",
                parsed_json=None,
                latency_ms=repair_latency_ms,
                status=LLMCallStatus.ERROR,
                error=str(exc),
                timestamp=repair_call_time,
            )
            raise

        repair_latency_ms = (time.perf_counter() - repair_start_time) * 1000.0

        # 5. Parse and validate repair response
        repair_parsed_data: Any = None
        try:
            repair_parsed_data = extract_json_payload(repair_raw_response)
            validated_instance = response_model.model_validate(repair_parsed_data)
            # Success on repair
            self._log_call(
                prompt_id=prompt_id,
                provider=provider_name,
                model=eff_model,
                prompt_summary=self._summarize_prompt(repair_user_prompt, system=system),
                response_raw=repair_raw_response,
                parsed_json=json.dumps(repair_parsed_data)
                if isinstance(repair_parsed_data, (dict, list))
                else None,
                latency_ms=repair_latency_ms,
                status=LLMCallStatus.SUCCESS,
                error=None,
                timestamp=repair_call_time,
            )
            if self.cache_enabled and use_cache:
                with self._cache_lock:
                    self._cache[cache_key] = validated_instance.model_copy(deep=True)
            return validated_instance
        except (ValueError, ValidationError) as second_exc:
            self._log_call(
                prompt_id=prompt_id,
                provider=provider_name,
                model=eff_model,
                prompt_summary=self._summarize_prompt(repair_user_prompt, system=system),
                response_raw=repair_raw_response,
                parsed_json=json.dumps(repair_parsed_data)
                if isinstance(repair_parsed_data, (dict, list))
                else None,
                latency_ms=repair_latency_ms,
                status=LLMCallStatus.ERROR,
                error=str(second_exc),
                timestamp=repair_call_time,
            )
            raise LLMOutputError(
                f"LLM output failed parsing or validation after 1 repair attempt: {second_exc}",
                raw_response=repair_raw_response,
                details=str(second_exc),
            ) from second_exc
