"""Smoke test script verifying the configured LLM provider returns valid JSON.

Governed by docs/AGENTS.md, docs/ROADMAP_AND_TASKS.md (Gate G0).
"""

from __future__ import annotations

import json
import os
import re
import sys

from memorymesh.config import load_engine_config
from memorymesh.llm.providers import get_provider


def extract_json_text(raw_text: str) -> str:
    """Strip markdown code fences if returned by the LLM."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def resolve_model_name(provider_name: str, config_model: str) -> str:
    """Resolve the model name for the selected provider with explicit precedence.

    Precedence:
    1. Generic model override: LLM_MODEL environment variable.
    2. Provider-specific override: GEMINI_MODEL / ANTHROPIC_MODEL environment variables.
    3. Configured engine model if provider-compatible.
    4. Default provider model fallback.
    """
    if os.environ.get("LLM_MODEL"):
        return os.environ["LLM_MODEL"]

    if provider_name == "gemini":
        if os.environ.get("GEMINI_MODEL"):
            return os.environ["GEMINI_MODEL"]
        if config_model.startswith("gemini"):
            return config_model
        return "gemini-2.5-flash"

    if provider_name == "anthropic":
        if os.environ.get("ANTHROPIC_MODEL"):
            return os.environ["ANTHROPIC_MODEL"]
        return config_model

    return config_model


def run_smoke_test() -> int:
    """Execute LLM smoke test against configured provider and print JSON result."""
    try:
        config = load_engine_config()
    except Exception as exc:
        print(f"[ERROR] Failed to load engine configuration: {exc}", file=sys.stderr)
        return 1

    provider_name = os.environ.get("LLM_PROVIDER") or config.llm.provider.value
    model_name = resolve_model_name(provider_name, config.llm.fast_model)

    print(f"Running LLM smoke test with provider '{provider_name}' (model: {model_name})...")

    try:
        provider = get_provider(provider_name)
    except Exception as exc:
        print(f"[ERROR] Could not initialize provider '{provider_name}': {exc}", file=sys.stderr)
        return 1

    system_prompt = (
        "You are an assistant that outputs strictly valid JSON only. "
        "Do not include any prose, markdown code fences, or explanation."
    )
    user_prompt = 'Return a JSON object with exactly: {"status": "ok"}'

    try:
        raw_output = provider.complete(
            system=system_prompt,
            user=user_prompt,
            model=model_name,
            max_tokens=100,
        )
    except Exception as exc:
        # Never print API keys
        err_msg = str(exc)
        print(f"[ERROR] Provider completion failed: {err_msg}", file=sys.stderr)
        return 2

    cleaned_text = extract_json_text(raw_output)

    try:
        parsed_json = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        print(
            f"[ERROR] Response could not be parsed as JSON: {exc}\n"
            f"Raw output received:\n{raw_output}",
            file=sys.stderr,
        )
        return 3

    if not isinstance(parsed_json, dict) or parsed_json.get("status") != "ok":
        print(
            f"[ERROR] JSON response does not match expected status object: {parsed_json}",
            file=sys.stderr,
        )
        return 4

    print("Smoke test SUCCESS! Valid JSON response received:")
    print(json.dumps(parsed_json, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
