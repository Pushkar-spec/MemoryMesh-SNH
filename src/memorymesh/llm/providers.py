"""LLM Provider abstractions and adapters (Anthropic, Gemini, Fake).

Governed by docs/ARCHITECTURE.md Section 7 and AGENTS.md.
"""

from __future__ import annotations

import os
from typing import Any, Protocol


class LLMProvider(Protocol):
    """Protocol defining the standard provider completion interface."""

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 1024,
    ) -> str:
        """Execute a text completion request and return the raw response string."""
        ...


class AnthropicProvider:
    """Production provider connecting to the Anthropic Claude Messages API."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self._client: Any = None

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 1024,
    ) -> str:
        """Call Anthropic messages.create and return the raw output text."""
        effective_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not effective_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Please configure the "
                "ANTHROPIC_API_KEY environment variable."
            )

        try:
            if self._client is None:
                import anthropic

                self._client = anthropic.Anthropic(api_key=effective_key)

            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text_blocks: list[str] = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_blocks.append(block.text)
            return "".join(text_blocks)
        except Exception as exc:
            # Protect sensitive credentials by scrubbing the key from error messages
            err_msg = str(exc)
            if effective_key:
                err_msg = err_msg.replace(effective_key, "[REDACTED]")
            raise RuntimeError(f"Anthropic provider error: {err_msg}") from None


class GeminiProvider:
    """Production provider connecting to the Google Gemini API via google-genai."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self._client: Any = None

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 1024,
    ) -> str:
        """Call Google Gemini generate_content and return the raw output text."""
        effective_key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not effective_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please configure the "
                "GEMINI_API_KEY environment variable."
            )

        try:
            if self._client is None:
                from google import genai

                self._client = genai.Client(api_key=effective_key)

            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            )
            response = self._client.models.generate_content(
                model=model,
                contents=user,
                config=config,
            )
            return response.text or ""
        except Exception as exc:
            # Protect sensitive credentials by scrubbing the key from error messages
            err_msg = str(exc)
            if effective_key:
                err_msg = err_msg.replace(effective_key, "[REDACTED]")
            raise RuntimeError(f"Gemini provider error: {err_msg}") from None


class FakeProvider:
    """Deterministic, offline fake provider for tests and development."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_response: str = '{"status": "ok"}',
    ) -> None:
        self.responses: dict[str, str] = dict(responses) if responses else {}
        self.default_response = default_response
        self.calls: list[dict[str, Any]] = []

    def set_response(self, key: str, response: str) -> None:
        """Register a scripted response for a prompt key."""
        self.responses[key] = response

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 1024,
    ) -> str:
        """Return scripted or fallback response deterministically without network access."""
        self.calls.append(
            {
                "system": system,
                "user": user,
                "model": model,
                "max_tokens": max_tokens,
            }
        )

        # Check for scripted response match by key in user text or system prompt
        for key, value in self.responses.items():
            if key in user or key in system:
                return value

        return self.default_response


def get_provider(name: str | None = None, **kwargs: Any) -> LLMProvider:
    """Factory creating an LLMProvider instance by name.

    Args:
        name: Provider name ('anthropic', 'gemini', 'fake').
              Defaults to LLM_PROVIDER environment variable or 'fake'.
        **kwargs: Optional provider constructor arguments (e.g. api_key, responses).

    Returns:
        An instance conforming to LLMProvider.
    """
    selected = (name or os.environ.get("LLM_PROVIDER") or "fake").strip().lower()

    if selected == "anthropic":
        return AnthropicProvider(**kwargs)
    if selected == "gemini":
        return GeminiProvider(**kwargs)
    if selected == "fake":
        return FakeProvider(**kwargs)

    raise ValueError(
        f"Unknown LLM provider: '{selected}'. Supported providers are: "
        "'anthropic', 'gemini', 'fake'."
    )
