"""Comprehensive tests for LLMClient centralized infrastructure (P7.1).

Governed by docs/ARCHITECTURE.md Section 7, docs/SPECIFICATION.md Section 3,
and MEMORYMESH_MASTER_IMPLEMENTATION_PLAN_CANONICAL.md (P7.1).
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from pydantic import BaseModel, Field

from memorymesh.db import init_db
from memorymesh.errors import LLMOutputError
from memorymesh.llm.client import LLMClient, extract_json_payload
from memorymesh.llm.providers import FakeProvider, LLMProvider
from memorymesh.models import LLMCallStatus
from memorymesh.repository import Repository

# ==============================================================================
# Sample Response Schemas for Testing
# ==============================================================================


class SimpleSample(BaseModel):
    name: str
    count: int


class NestedSample(BaseModel):
    title: str
    details: dict[str, Any]
    items: list[str] = Field(default_factory=list)


class StrictSample(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    label: str


# ==============================================================================
# Helper Mock Providers
# ==============================================================================


class SequenceProvider:
    """Mock provider returning a programmed sequence of responses."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "model": model,
                "max_tokens": max_tokens,
            }
        )
        if not self.responses:
            return '{"name": "default", "count": 0}'
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class ConcurrencyTrackingProvider:
    """Mock provider that tracks concurrent in-flight invocations."""

    def __init__(self, delay_sec: float = 0.05) -> None:
        self.delay_sec = delay_sec
        self.lock = threading.Lock()
        self.current_in_flight = 0
        self.max_observed_in_flight = 0
        self.total_calls = 0

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 1024,
    ) -> str:
        with self.lock:
            self.total_calls += 1
            self.current_in_flight += 1
            if self.current_in_flight > self.max_observed_in_flight:
                self.max_observed_in_flight = self.current_in_flight

        time.sleep(self.delay_sec)

        with self.lock:
            self.current_in_flight -= 1

        return '{"name": "concurrency_test", "count": 1}'


# ==============================================================================
# 1. JSON Parsing Tests
# ==============================================================================


def test_extract_json_plain_object() -> None:
    raw = '{"name": "Alpha", "count": 10}'
    data = extract_json_payload(raw)
    assert data == {"name": "Alpha", "count": 10}


def test_extract_json_with_whitespace() -> None:
    raw = '   \n\n\t{"name": "Beta", "count": 20}\n  \t '
    data = extract_json_payload(raw)
    assert data == {"name": "Beta", "count": 20}


def test_extract_json_markdown_fence_with_language() -> None:
    raw = """Here is your output:
```json
{
  "name": "Gamma",
  "count": 30
}
```
Hope this helps!"""
    data = extract_json_payload(raw)
    assert data == {"name": "Gamma", "count": 30}


def test_extract_json_markdown_fence_without_language() -> None:
    raw = """```
{
  "name": "Delta",
  "count": 40
}
```"""
    data = extract_json_payload(raw)
    assert data == {"name": "Delta", "count": 40}


def test_extract_json_nested_structure() -> None:
    raw = """Response preamble:
{
  "title": "NestedTest",
  "details": {
    "key1": "val1",
    "inner": {"escaped_quote": "hello \\"world\\""}
  },
  "items": ["a", "b", "c"]
}
trailing thoughts."""
    data = extract_json_payload(raw)
    assert data["title"] == "NestedTest"
    assert data["details"]["key1"] == "val1"
    assert data["items"] == ["a", "b", "c"]


def test_extract_json_malformed_raises_value_error() -> None:
    raw = "This is not JSON at all."
    with pytest.raises(ValueError, match="Failed to extract valid JSON payload"):
        extract_json_payload(raw)


def test_extract_json_empty_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty text"):
        extract_json_payload("   \n ")


# ==============================================================================
# 2. Retry Logic Tests
# ==============================================================================


def test_retry_invalid_json_then_valid_succeeds() -> None:
    """Provider returning malformed JSON on call 1 and valid JSON on call 2 succeeds."""
    provider = SequenceProvider(
        [
            "I forgot to return JSON!",
            '{"name": "Repaired", "count": 99}',
        ]
    )
    client = LLMClient(provider=provider, cache_enabled=False)
    result = client.complete_json(
        prompt_id="test_prompt",
        system="System prompt",
        user="User prompt",
        response_model=SimpleSample,
    )
    assert result.name == "Repaired"
    assert result.count == 99
    assert len(provider.calls) == 2
    # Second call must contain repair directive
    assert "[REPAIR DIRECTIVE]" in provider.calls[1]["user"]
    assert "System prompt" in provider.calls[1]["system"]


def test_retry_schema_invalid_then_valid_succeeds() -> None:
    """Provider returning valid JSON with wrong schema types on call 1 and valid on call 2."""
    provider = SequenceProvider(
        [
            '{"score": 5.5, "label": "Too high"}',  # score must be <= 1.0
            '{"score": 0.85, "label": "Valid"}',
        ]
    )
    client = LLMClient(provider=provider, cache_enabled=False)
    result = client.complete_json(
        prompt_id="strict_test",
        system="System",
        user="Evaluate",
        response_model=StrictSample,
    )
    assert result.score == 0.85
    assert result.label == "Valid"
    assert len(provider.calls) == 2
    assert "[REPAIR DIRECTIVE]" in provider.calls[1]["user"]


def test_retry_fails_twice_raises_llm_output_error() -> None:
    """When output fails validation after the repair attempt, LLMOutputError is raised."""
    provider = SequenceProvider(
        [
            "Bad output 1",
            "Bad output 2",
        ]
    )
    client = LLMClient(provider=provider, cache_enabled=False)
    with pytest.raises(LLMOutputError) as exc_info:
        client.complete_json(
            prompt_id="fail_test",
            system="System",
            user="Prompt",
            response_model=SimpleSample,
        )
    assert "failed parsing or validation after 1 repair attempt" in str(exc_info.value)
    assert exc_info.value.raw_response == "Bad output 2"
    assert len(provider.calls) == 2


def test_provider_called_once_when_first_response_valid() -> None:
    """Clean initial output results in exactly 1 provider call."""
    provider = FakeProvider(default_response='{"name": "DirectSuccess", "count": 1}')
    client = LLMClient(provider=provider, cache_enabled=False)
    res = client.complete_json(
        prompt_id="clean_test",
        system="Sys",
        user="User",
        response_model=SimpleSample,
    )
    assert res.name == "DirectSuccess"
    assert len(provider.calls) == 1


# ==============================================================================
# 3. SHA-256 Cache Tests
# ==============================================================================


def test_cache_hit_returns_cached_result_without_provider_call() -> None:
    """Identical requests return the cached result and call the provider only once."""
    provider = SequenceProvider(['{"name": "CachedValue", "count": 100}'])
    client = LLMClient(provider=provider, cache_enabled=True)

    res1 = client.complete_json("p1", "sys", "usr", SimpleSample)
    assert res1.name == "CachedValue"
    assert len(provider.calls) == 1

    # Second identical call
    res2 = client.complete_json("p1", "sys", "usr", SimpleSample)
    assert res2.name == "CachedValue"
    assert len(provider.calls) == 1  # No additional provider call


def test_cache_miss_on_different_model() -> None:
    provider = FakeProvider(default_response='{"name": "ModelDiff", "count": 1}')
    client = LLMClient(provider=provider, cache_enabled=True)

    client.complete_json("p1", "sys", "usr", SimpleSample, model="model-a")
    assert len(provider.calls) == 1

    client.complete_json("p1", "sys", "usr", SimpleSample, model="model-b")
    assert len(provider.calls) == 2


def test_cache_miss_on_different_prompt() -> None:
    provider = FakeProvider(default_response='{"name": "PromptDiff", "count": 1}')
    client = LLMClient(provider=provider, cache_enabled=True)

    client.complete_json("p1", "sys1", "usr1", SimpleSample)
    assert len(provider.calls) == 1

    client.complete_json("p1", "sys1", "usr2", SimpleSample)
    assert len(provider.calls) == 2


def test_cache_miss_on_different_schema() -> None:
    provider = FakeProvider(
        responses={
            "Simple": '{"name": "A", "count": 1}',
            "Strict": '{"score": 0.5, "label": "B"}',
        }
    )
    client = LLMClient(provider=provider, cache_enabled=True)

    client.complete_json("p1", "sys", "Simple", SimpleSample)
    assert len(provider.calls) == 1

    client.complete_json("p1", "sys", "Strict", StrictSample)
    assert len(provider.calls) == 2


def test_cache_miss_on_different_max_tokens() -> None:
    provider = FakeProvider(default_response='{"name": "TokensDiff", "count": 1}')
    client = LLMClient(provider=provider, cache_enabled=True)

    client.complete_json("p1", "sys", "usr", SimpleSample, max_tokens=500)
    assert len(provider.calls) == 1

    client.complete_json("p1", "sys", "usr", SimpleSample, max_tokens=1000)
    assert len(provider.calls) == 2


def test_failed_output_is_not_cached() -> None:
    """Failed responses must not populate the cache."""
    provider = SequenceProvider(["Bad response 1", "Bad response 2"])
    client = LLMClient(provider=provider, cache_enabled=True)

    with pytest.raises(LLMOutputError):
        client.complete_json("p1", "sys", "usr", SimpleSample)

    assert len(client._cache) == 0


# ==============================================================================
# 4. Concurrency Control Tests
# ==============================================================================


def test_concurrency_limit_is_respected() -> None:
    """Configured max concurrency is never exceeded across concurrent threads."""
    provider = ConcurrencyTrackingProvider(delay_sec=0.04)
    # Set concurrency to 2
    client = LLMClient(provider=provider, max_concurrency=2, cache_enabled=False)

    threads: list[threading.Thread] = []
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            client.complete_json(f"thread_{i}", "sys", f"user_{i}", SimpleSample)
        except Exception as exc:
            errors.append(exc)

    for i in range(6):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert not errors
    assert provider.total_calls == 6
    assert provider.max_observed_in_flight <= 2


def test_cache_hits_do_not_acquire_concurrency_slot() -> None:
    """When requests hit cache, concurrency semaphore is not acquired."""
    provider = FakeProvider(default_response='{"name": "CacheSlot", "count": 1}')
    client = LLMClient(provider=provider, max_concurrency=1, cache_enabled=True)

    # First call populates cache
    client.complete_json("p1", "sys", "usr", SimpleSample)

    # Drain semaphore to 0 to simulate active lock
    acquired = client._semaphore.acquire(blocking=False)
    assert acquired is True

    # Cache hit should return successfully even though semaphore is at 0
    cached_res = client.complete_json("p1", "sys", "usr", SimpleSample)
    assert cached_res.name == "CacheSlot"

    # Release semaphore
    client._semaphore.release()


def test_concurrency_slot_released_after_provider_exception() -> None:
    """When a provider crashes, the semaphore slot is reliably released."""

    class CrashProvider:
        def complete(self, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError("Provider network failure")

    client = LLMClient(provider=CrashProvider(), max_concurrency=1, cache_enabled=False)

    with pytest.raises(RuntimeError, match="Provider network failure"):
        client.complete_json("p1", "sys", "usr", SimpleSample)

    # Verify semaphore is fully released and can be acquired immediately
    can_acquire = client._semaphore.acquire(blocking=False)
    assert can_acquire is True
    client._semaphore.release()


# ==============================================================================
# 5. SQLite Logging (llm_calls) Tests
# ==============================================================================


def test_logging_successful_provider_call() -> None:
    conn = init_db(":memory:")
    repo = Repository(conn)
    provider = FakeProvider(default_response='{"name": "LogSuccess", "count": 5}')
    client = LLMClient(provider=provider, repo=repo, cache_enabled=False)

    client.complete_json("prompt_extract", "Sys prompt", "User query", SimpleSample)

    logs = repo.get_llm_calls()
    assert len(logs) == 1
    entry = logs[0]
    assert entry.prompt_id == "prompt_extract"
    assert entry.provider == "FakeProvider"
    assert entry.status == LLMCallStatus.SUCCESS
    assert entry.error is None
    assert "LogSuccess" in entry.response_raw
    assert entry.latency_ms >= 0


def test_logging_repair_attempt_logged_separately() -> None:
    conn = init_db(":memory:")
    repo = Repository(conn)
    provider = SequenceProvider(
        [
            "Bad format",
            '{"name": "RepairedLog", "count": 7}',
        ]
    )
    client = LLMClient(provider=provider, repo=repo, cache_enabled=False)

    client.complete_json("repair_test", "Sys", "User", SimpleSample)

    logs = repo.get_llm_calls()
    assert len(logs) == 2
    # Call 1 was logged as RETRY
    assert logs[0].status == LLMCallStatus.RETRY
    assert logs[0].error is not None
    # Call 2 was logged as SUCCESS
    assert logs[1].status == LLMCallStatus.SUCCESS
    assert logs[1].error is None
    assert "RepairedLog" in logs[1].response_raw


def test_logging_failed_call_logged() -> None:
    conn = init_db(":memory:")
    repo = Repository(conn)
    provider = SequenceProvider(["Bad 1", "Bad 2"])
    client = LLMClient(provider=provider, repo=repo, cache_enabled=False)

    with pytest.raises(LLMOutputError):
        client.complete_json("fail_log", "Sys", "User", SimpleSample)

    logs = repo.get_llm_calls()
    assert len(logs) == 2
    assert logs[0].status == LLMCallStatus.RETRY
    assert logs[1].status == LLMCallStatus.ERROR
    assert logs[1].error is not None


def test_cache_hit_does_not_create_llm_call_log() -> None:
    conn = init_db(":memory:")
    repo = Repository(conn)
    provider = FakeProvider(default_response='{"name": "CachedLog", "count": 8}')
    client = LLMClient(provider=provider, repo=repo, cache_enabled=True)

    # First call
    client.complete_json("p1", "sys", "user", SimpleSample)
    assert len(repo.get_llm_calls()) == 1

    # Second call (cache hit)
    client.complete_json("p1", "sys", "user", SimpleSample)
    # Must still have only 1 log entry (no provider invocation)
    assert len(repo.get_llm_calls()) == 1


# ==============================================================================
# 6. Provider Independence Tests
# ==============================================================================


def test_fake_provider_zero_network_access() -> None:
    """FakeProvider functions completely offline without any network or credentials."""
    provider = FakeProvider(default_response='{"name": "Offline", "count": 10}')
    client = LLMClient(provider=provider)
    res = client.complete_json("offline", "sys", "user", SimpleSample)
    assert res.name == "Offline"
    assert res.count == 10


def test_custom_provider_protocol_conformance() -> None:
    """Client interacts strictly with the LLMProvider protocol."""

    class CustomDuckProvider:
        name = "DuckProvider"

        def complete(
            self,
            system: str,
            user: str,
            *,
            model: str,
            max_tokens: int = 1024,
        ) -> str:
            return '{"name": "Quack", "count": 42}'

    provider: LLMProvider = CustomDuckProvider()
    client = LLMClient(provider=provider)
    res = client.complete_json("duck", "sys", "user", SimpleSample)
    assert res.name == "Quack"
    assert res.count == 42
    assert client._get_provider_name() == "DuckProvider"


# ==============================================================================
# 7. Security and Redaction Tests
# ==============================================================================


def test_secrets_redacted_from_logging_and_errors() -> None:
    """Sentinel secrets in prompt and response are redacted from llm_calls and errors."""
    conn = init_db(":memory:")
    repo = Repository(conn)

    sentinel_api_key = "sk-ant-SENTINEL_API_KEY_9999"
    sentinel_bearer = "SENTINEL_BEARER_8888"
    sentinel_password = "SENTINEL_PASSWORD_7777"

    prompt_with_secrets = (
        f"Extract facts: api_key={sentinel_api_key} and Bearer {sentinel_bearer} "
        f"password={sentinel_password}"
    )
    raw_response_with_secrets = (
        f"Here is your result with api_key={sentinel_api_key} and invalid JSON {{"
    )

    provider = SequenceProvider(
        [
            raw_response_with_secrets,
            f"Still invalid with Bearer {sentinel_bearer} {{",
        ]
    )
    client = LLMClient(provider=provider, repo=repo, cache_enabled=False)

    with pytest.raises(LLMOutputError) as exc_info:
        client.complete_json(
            "security_test",
            "System prompt",
            prompt_with_secrets,
            SimpleSample,
        )

    # 1. Verify secrets do not appear in LLMOutputError diagnostics
    error_obj = exc_info.value
    assert sentinel_api_key not in error_obj.raw_response
    assert sentinel_bearer not in error_obj.raw_response
    assert sentinel_password not in error_obj.raw_response
    assert sentinel_api_key not in error_obj.details
    assert sentinel_bearer not in error_obj.details
    assert sentinel_password not in str(error_obj)

    # 2. Verify secrets do not appear in persisted llm_calls rows
    logs = repo.get_llm_calls(prompt_id="security_test")
    assert len(logs) == 2
    for log in logs:
        for secret in (sentinel_api_key, sentinel_bearer, sentinel_password):
            assert secret not in log.prompt_summary
            assert secret not in log.response_raw
            if log.parsed_json:
                assert secret not in log.parsed_json
            if log.error:
                assert secret not in log.error

        # Ensure redaction indicators are present
        assert "[REDACTED]" in log.prompt_summary or "[truncated]" in log.prompt_summary
        assert "[REDACTED]" in log.response_raw or "[truncated]" in log.response_raw


# ==============================================================================
# 8. Provider / Transport Failure Logging Tests
# ==============================================================================


def test_provider_transport_failure_logging_and_concurrency() -> None:
    """Provider/transport exception logs single error, releases semaphore, and does not retry."""
    conn = init_db(":memory:")
    repo = Repository(conn)

    sentinel_secret = "sk-ant-TRANSPORT_CRASH_SECRET_5555"

    class TransportCrashProvider:
        def __init__(self) -> None:
            self.invocations = 0

        def complete(
            self,
            system: str,
            user: str,
            *,
            model: str,
            max_tokens: int = 1024,
        ) -> str:
            self.invocations += 1
            # Simulate a transport failure containing an auth header/token in the message
            raise ConnectionError(
                f"Connection reset by peer during TLS handshake for "
                f"Authorization: Bearer {sentinel_secret}"
            )

    crash_provider = TransportCrashProvider()
    client = LLMClient(provider=crash_provider, repo=repo, max_concurrency=2, cache_enabled=False)

    # 1. Provider exception must NOT be converted into LLMOutputError
    with pytest.raises(ConnectionError, match="Connection reset by peer") as exc_info:
        client.complete_json(
            "transport_test",
            "System",
            f"Fetch with key api_key={sentinel_secret}",
            SimpleSample,
        )

    # Demonstrate that the caller receives the original provider exception unchanged,
    # which retains whatever credential the external provider put in the exception message:
    assert sentinel_secret in str(exc_info.value)

    # 2. Exactly one provider invocation occurred (no output repair retry)
    assert crash_provider.invocations == 1

    # 3. Exactly one llm_calls row was created with status 'error'
    logs = repo.get_llm_calls(prompt_id="transport_test")
    assert len(logs) == 1
    call_log = logs[0]
    assert call_log.status == LLMCallStatus.ERROR
    assert call_log.response_raw == ""
    assert call_log.parsed_json is None
    assert call_log.latency_ms >= 0

    # 4. Error category and message are safely recorded without credentials
    assert call_log.error is not None
    assert "Connection reset by peer" in call_log.error
    assert sentinel_secret not in call_log.error
    assert sentinel_secret not in call_log.prompt_summary
    assert "[REDACTED]" in call_log.error

    # 5. Concurrency semaphore was safely released
    can_acquire = client._semaphore.acquire(blocking=False)
    assert can_acquire is True
    client._semaphore.release()


def test_valid_json_with_secrets_sanitized_in_db() -> None:
    """Parsed valid JSON with secrets is sanitized in DB logs while caller gets data."""
    conn = init_db(":memory:")
    repo = Repository(conn)

    sentinel_sys_key = "sk-ant-SENTINEL_SYS_KEY_1111"
    sentinel_user_key = "sk-ant-SENTINEL_USER_KEY_2222"
    sentinel_resp_secret = "sk-ant-SENTINEL_JSON_SECRET_3333"

    class CredentialPayload(BaseModel):
        account: str
        token_value: str

    provider_json = f'{{"account": "admin", "token_value": "api_key={sentinel_resp_secret}"}}'
    provider = FakeProvider(default_response=provider_json)
    client = LLMClient(provider=provider, repo=repo, cache_enabled=False)

    system_prompt = f"System prompt with auth token: Bearer {sentinel_sys_key}"
    user_prompt = f"Get payload with user key: api_key={sentinel_user_key}"

    result = client.complete_json(
        "valid_json_secrets_test",
        system=system_prompt,
        user=user_prompt,
        response_model=CredentialPayload,
    )

    # 1. Pydantic validation passes and caller receives the validated model instance
    assert result.account == "admin"
    assert sentinel_resp_secret in result.token_value

    # 2. Database record in llm_calls has status SUCCESS
    logs = repo.get_llm_calls(prompt_id="valid_json_secrets_test")
    assert len(logs) == 1
    log = logs[0]
    assert log.status == LLMCallStatus.SUCCESS

    # 3. Sentinel credentials do NOT appear in prompt_summary, response_raw, or parsed_json
    for secret in (sentinel_sys_key, sentinel_user_key, sentinel_resp_secret):
        assert secret not in log.prompt_summary
        assert secret not in log.response_raw
        if log.parsed_json:
            assert secret not in log.parsed_json

    # 4. Redaction indicators appear where secrets were scrubbed
    assert "[REDACTED]" in log.prompt_summary
    assert "[REDACTED]" in log.response_raw
    assert log.parsed_json is not None
    assert "[REDACTED]" in log.parsed_json

    # 5. Stored diagnostics are strictly bounded
    assert len(log.prompt_summary) <= 120
    assert len(log.response_raw) <= 1000
    assert len(log.parsed_json) <= 1000
