"""Tests for LLM provider implementations (Anthropic, Gemini, Fake)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memorymesh.llm.providers import (
    AnthropicProvider,
    FakeProvider,
    GeminiProvider,
    get_provider,
)


def test_providers_expose_expected_interface() -> None:
    """1. All three providers expose complete(system, user, *, model, max_tokens)."""
    providers = [
        FakeProvider(),
        GeminiProvider(api_key="mock-key"),
        AnthropicProvider(api_key="mock-key"),
    ]
    for provider in providers:
        assert hasattr(provider, "complete")
        assert callable(provider.complete)


def test_fake_provider_configured_response() -> None:
    """2. FakeProvider returns its configured response based on prompt identifier/text."""
    fake = FakeProvider(
        responses={
            "extract_claims": '{"claims": [{"subject": "server"}]}',
            "classify_relation": '{"relation": "REPLACE"}',
        }
    )
    # Match by key in user text
    resp1 = fake.complete("system", "extract_claims from text", model="fake-model", max_tokens=100)
    assert resp1 == '{"claims": [{"subject": "server"}]}'

    # Match by key in system prompt
    resp2 = fake.complete("classify_relation", "user text", model="fake-model", max_tokens=100)
    assert resp2 == '{"relation": "REPLACE"}'


def test_fake_provider_fallback_response() -> None:
    """3. FakeProvider fallback behavior returns deterministic default (e.g. status ok)."""
    fake = FakeProvider(default_response='{"status": "ok"}')
    resp = fake.complete("arbitrary system", "arbitrary user", model="fake-model", max_tokens=100)
    assert resp == '{"status": "ok"}'


def test_fake_provider_without_api_keys_or_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """4. FakeProvider works completely offline with no environment keys."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    fake = FakeProvider()
    resp = fake.complete("sys", "user", model="fake-model", max_tokens=50)
    assert resp == '{"status": "ok"}'
    assert len(fake.calls) == 1
    assert fake.calls[0]["model"] == "fake-model"


def test_anthropic_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """5. AnthropicProvider reports a missing API key clearly."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    provider = AnthropicProvider(api_key=None)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        provider.complete("system", "user", model="claude-3-haiku", max_tokens=100)


def test_anthropic_successful_mocked_response() -> None:
    """6. AnthropicProvider correctly handles a mocked successful SDK response."""
    provider = AnthropicProvider(api_key="mock-test-key")

    mock_client = MagicMock()
    mock_block = MagicMock()
    mock_block.text = '{"status": "ok"}'
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    mock_client.messages.create.return_value = mock_response

    provider._client = mock_client

    result = provider.complete(
        system="You are a helpful assistant",
        user="Hello",
        model="claude-haiku-5-5",
        max_tokens=256,
    )
    assert result == '{"status": "ok"}'
    mock_client.messages.create.assert_called_once_with(
        model="claude-haiku-5-5",
        max_tokens=256,
        system="You are a helpful assistant",
        messages=[{"role": "user", "content": "Hello"}],
    )


def test_anthropic_handles_api_errors_without_exposing_key() -> None:
    """7. AnthropicProvider handles mocked API errors without exposing secret keys."""
    secret_key = "sk-ant-secret-mock-key-12345"
    provider = AnthropicProvider(api_key=secret_key)

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError(f"Rate limit exceeded with {secret_key}")
    provider._client = mock_client

    with pytest.raises(RuntimeError) as exc_info:
        provider.complete("sys", "usr", model="claude-haiku-5-5", max_tokens=100)

    # Key must NOT be exposed in the error message
    assert secret_key not in str(exc_info.value)
    assert "Anthropic" in str(exc_info.value)


def test_gemini_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """8. GeminiProvider reports a missing API key clearly."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    provider = GeminiProvider(api_key=None)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        provider.complete("system", "user", model="gemini-2.5-flash", max_tokens=100)


def test_gemini_client_lazy_initialization_and_reuse() -> None:
    """9. Verify genai.Client is lazily initialized with the API key and reused."""
    provider = GeminiProvider(api_key="gemini-init-key-123")
    assert provider._client is None

    mock_client_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"status": "ok"}'
    mock_client_instance.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client_instance) as mock_client_cls:
        # First call initializes the client
        res1 = provider.complete("sys 1", "usr 1", model="gemini-2.5-flash", max_tokens=100)
        assert res1 == '{"status": "ok"}'
        mock_client_cls.assert_called_once_with(api_key="gemini-init-key-123")

        # Second call reuses the existing client instance (does not call constructor again)
        res2 = provider.complete("sys 2", "usr 2", model="gemini-2.5-flash", max_tokens=200)
        assert res2 == '{"status": "ok"}'
        assert mock_client_cls.call_count == 1

        # Verify generate_content arguments on the second call
        assert mock_client_instance.models.generate_content.call_count == 2
        _, kwargs2 = mock_client_instance.models.generate_content.call_args
        assert kwargs2["model"] == "gemini-2.5-flash"
        assert kwargs2["contents"] == "usr 2"
        assert kwargs2["config"].system_instruction == "sys 2"
        assert kwargs2["config"].max_output_tokens == 200


def test_gemini_client_initialization_error_redacts_key() -> None:
    """10. Verify errors raised during genai.Client construction redact the API key."""
    secret_key = "secret-client-init-key-456"
    provider = GeminiProvider(api_key=secret_key)

    with patch("google.genai.Client", side_effect=Exception(f"Invalid key: {secret_key}")):
        with pytest.raises(RuntimeError) as exc_info:
            provider.complete("sys", "usr", model="gemini-2.5-flash")

        assert secret_key not in str(exc_info.value)
        assert "[REDACTED]" in str(exc_info.value)
        assert "Gemini provider error" in str(exc_info.value)


def test_gemini_client_initialization_and_call() -> None:
    """11. GeminiProvider generates content with correct arguments on an injected client."""
    provider = GeminiProvider(api_key="gemini-mock-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"status": "ok"}'
    mock_client.models.generate_content.return_value = mock_response

    provider._client = mock_client

    result = provider.complete(
        system="System instruction text",
        user="User prompt text",
        model="gemini-2.5-flash",
        max_tokens=512,
    )
    assert result == '{"status": "ok"}'

    # Verify generate_content arguments
    mock_client.models.generate_content.assert_called_once()
    _, kwargs = mock_client.models.generate_content.call_args
    assert kwargs["model"] == "gemini-2.5-flash"
    assert kwargs["contents"] == "User prompt text"
    assert kwargs["config"].system_instruction == "System instruction text"
    assert kwargs["config"].max_output_tokens == 512


def test_gemini_empty_response_handling() -> None:
    """10. GeminiProvider handles empty or None response text gracefully."""
    provider = GeminiProvider(api_key="gemini-mock-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = None
    mock_client.models.generate_content.return_value = mock_response

    provider._client = mock_client

    result = provider.complete("sys", "usr", model="gemini-2.5-flash", max_tokens=100)
    assert result == ""


def test_gemini_handles_api_errors_without_exposing_key() -> None:
    """11. GeminiProvider handles API errors without exposing credentials in logs."""
    secret_key = "gemini-super-secret-key-9999"
    provider = GeminiProvider(api_key=secret_key)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError(
        f"API quota exceeded with key: {secret_key}"
    )
    provider._client = mock_client

    with pytest.raises(RuntimeError) as exc_info:
        provider.complete("sys", "usr", model="gemini-2.5-flash", max_tokens=100)

    # Key must NOT be exposed
    assert secret_key not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)
    assert "Gemini provider error" in str(exc_info.value)


def test_get_provider_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """12. Verify get_provider resolves provider instances by name and env var."""
    p_fake = get_provider("fake")
    assert isinstance(p_fake, FakeProvider)

    p_gemini = get_provider("gemini", api_key="gem-key")
    assert isinstance(p_gemini, GeminiProvider)
    assert p_gemini.api_key == "gem-key"

    # Test resolution via LLM_PROVIDER env variable
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    p_env_gemini = get_provider()
    assert isinstance(p_env_gemini, GeminiProvider)

    p_anthropic = get_provider("anthropic", api_key="dummy")
    assert isinstance(p_anthropic, AnthropicProvider)

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_provider("unsupported_provider")


def test_smoke_test_model_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """13. Verify model resolution for smoke test under various providers and env vars."""
    from scripts.smoke_test_llm import resolve_model_name

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    # When engine config specifies an Anthropic model but provider is Gemini, use Gemini default
    assert resolve_model_name("gemini", "claude-haiku-5-5") == "gemini-2.5-flash"

    # When engine config specifies a Gemini model, preserve it
    assert resolve_model_name("gemini", "gemini-1.5-pro") == "gemini-1.5-pro"

    # Explicit GEMINI_MODEL override
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-flash")
    assert resolve_model_name("gemini", "claude-haiku-5-5") == "gemini-custom-flash"

    # Generic LLM_MODEL override takes highest precedence
    monkeypatch.setenv("LLM_MODEL", "global-model-override")
    assert resolve_model_name("gemini", "claude-haiku-5-5") == "global-model-override"
    assert resolve_model_name("anthropic", "claude-haiku-5-5") == "global-model-override"
    assert resolve_model_name("fake", "claude-haiku-5-5") == "global-model-override"
