"""
llm.py
======

Provider-independent LLM layer for AI Civilization (Day 5).

The simulation must NOT know or care which model provider is in use. This module
owns every detail of talking to an LLM and exposes a single, stable entry point:

    get_decision(prompt: str) -> dict   # {"action": ..., "reason": ...}

Swap providers by changing PROVIDER below. "ollama" (local, default) and
"gemini" (cloud) are both implemented; "openai" is a stub that future days can
fill in without touching the simulation (main.py / world.py).

Design
------
    get_decision()                <- public, provider-agnostic
        -> _get_ollama_decision() <- private, provider-specific (local default)
        -> _gemini_decision()     <- private, provider-specific (cloud)
        -> _openai_decision()     <- (future)

Robustness contract: get_decision ALWAYS returns a usable decision dict. Any
provider failure (network error, quota, bad/missing JSON, invalid action)
degrades gracefully to a safe `rest` — the simulation can never crash on model
output. The set of allowed actions is sourced from world.VALID_ACTIONS so there
is a single source of truth for the action vocabulary.
"""

import json
import os
import sys
from typing import Any

from dotenv import load_dotenv

from world import VALID_ACTIONS

# Active provider. Change this one line to switch backends.
# Supported today: "ollama" (default, local), "gemini" (cloud). Future: "openai".
PROVIDER = "ollama"

# Ollama-specific config (only consulted when PROVIDER == "ollama").
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"

# Gemini-specific config (only consulted when PROVIDER == "gemini").
GEMINI_MODEL = "gemini-2.5-flash"

# Returned whenever a provider is unreachable or returns anything untrustworthy.
FALLBACK_DECISION = {
    "action": "rest",
    "reason": "Invalid or unavailable LLM response; defaulting to rest.",
}

# Cached Gemini client so we don't rebuild it on every call.
_gemini_client: Any = None


def get_decision(prompt: str) -> dict[str, Any]:
    """Provider-independent decision entry point.

    Dispatches to the configured PROVIDER and returns a validated decision dict
    of the form {"action": <valid action>, "reason": <str>}. Always succeeds:
    failures fall back to a safe `rest`.
    """
    if PROVIDER == "ollama":
        return _get_ollama_decision(prompt)

    if PROVIDER == "gemini":
        return _gemini_decision(prompt)

    # Future providers plug in here without any change to the simulation:
    #   if PROVIDER == "openai":
    #       return _openai_decision(prompt)

    raise ValueError(f"Unknown PROVIDER: {PROVIDER!r}")


def _validate_decision(data: Any) -> dict[str, Any]:
    """Coerce a parsed object into a trusted decision dict, or fall back.

    Shared by all providers: enforces that `action` is one of VALID_ACTIONS and
    keeps only the fields we trust (reason is optional/logging-only).
    """
    if not isinstance(data, dict):
        return dict(FALLBACK_DECISION)
    action = data.get("action")
    if action not in VALID_ACTIONS:
        return dict(FALLBACK_DECISION)
    return {"action": action, "reason": str(data.get("reason", ""))}


def _extract_json(text: str) -> str | None:
    """Pull the first JSON object out of a model response.

    Robust to stray prose or ```json code fences: slice from the first '{' to
    the last '}'. Returns None if no plausible object is present.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


# --- Ollama provider (local, default) --------------------------------------
def _get_ollama_decision(prompt: str) -> dict[str, Any]:
    """Ask a local Ollama model for a decision and return a validated dict.

    Talks to the Ollama HTTP API (OLLAMA_URL) with streaming disabled so the
    full completion arrives in one JSON payload. Reasoning models like qwen3
    often prepend chatter ("Thinking...") before the JSON object, so we lean on
    the shared _extract_json slicer to recover the embedded object. ANY failure
    (network error, non-200, bad JSON, invalid action) degrades to `rest`.
    """
    try:
        import requests  # imported lazily so non-ollama setups don't need it

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()

        # /api/generate returns {"response": "<model text>", ...}.
        text = (response.json().get("response") or "").strip()

        raw = _extract_json(text)
        if raw is None:
            return dict(FALLBACK_DECISION)

        return _validate_decision(json.loads(raw))

    except Exception:
        # Network/HTTP/JSON errors all degrade gracefully to rest.
        return dict(FALLBACK_DECISION)


# --- Gemini provider -------------------------------------------------------
def _get_gemini_client() -> Any:
    """Lazily build and cache the Gemini client.

    The SDK and API key are only required when the Gemini provider is actually
    used, so other providers don't need them. Fails loudly (SystemExit) if the
    key is missing — a misconfiguration, not a recoverable model error.
    """
    global _gemini_client
    if _gemini_client is None:
        from google import genai  # imported lazily so non-gemini setups don't need it

        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            sys.exit(
                "ERROR: GEMINI_API_KEY is not set. "
                "Create ai_civilization/.env with: GEMINI_API_KEY=your_key_here"
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _gemini_decision(prompt: str) -> dict[str, Any]:
    """Ask Gemini for a decision and return a validated dict.

    Requests strict JSON, parses defensively, validates the action. ANY failure
    (SDK/network/quota error, bad JSON, invalid action) degrades to `rest`.
    """
    try:
        from google.genai import types

        client = _get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.7,
            ),
        )

        raw = _extract_json((response.text or "").strip())
        if raw is None:
            return dict(FALLBACK_DECISION)

        return _validate_decision(json.loads(raw))

    except Exception:
        # Network/SDK/JSON errors all degrade gracefully to rest.
        return dict(FALLBACK_DECISION)
