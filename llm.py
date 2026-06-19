"""
llm.py
======

Provider-independent LLM layer for AI Civilization.

The simulation must NOT know or care which model provider is in use. This module
owns every detail of talking to an LLM and exposes two stable entry points:

    get_decision(prompt) -> dict   # {"action": ...,   "reason": ...}   (per-turn; legacy)
    get_strategy(prompt) -> dict   # {"strategy": ..., "target": ..., "reason": ...}

`get_strategy` is the new (Phase 4) primitive: the simulation asks for a
high-level plan only every N turns and executes it in Python in between, so most
turns cost no inference at all. `get_decision` is kept for backward compatibility
(and the test suite) — its signature and behaviour are unchanged.

Swap providers by setting AICIV_PROVIDER (or editing PROVIDER below). Implemented:
  - "ollama" (local, default)
  - "gemini" (cloud)
  - "random" (offline, no model server) — plausible valid output for tests/demos.
"openai" remains a future stub.

Design
------
    get_decision() / get_strategy()      <- public, provider-agnostic
        -> _raw_query()                  <- dispatches ollama/gemini to JSON|None
            -> _query_ollama() / _query_gemini()
        -> _random_decision() / _random_strategy()   <- offline backends

Robustness contract: both entry points ALWAYS return a usable dict. Any provider
failure (network, quota, bad/missing JSON, invalid value) degrades gracefully to
a safe fallback (`rest` / `wander`) — the simulation can never crash on model
output. The allowed vocabularies are sourced from world.VALID_ACTIONS and
strategy.VALID_STRATEGIES so there is a single source of truth for each.
"""

import json
import os
import random
import sys
from typing import Any

from dotenv import load_dotenv

from strategy import DIRECTIONS, VALID_STRATEGIES
from world import VALID_ACTIONS

# Active provider. Defaults to local Ollama; override without editing code via
# the AICIV_PROVIDER env var (e.g. AICIV_PROVIDER=random for offline testing).
PROVIDER = os.getenv("AICIV_PROVIDER", "ollama")

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
FALLBACK_STRATEGY = {
    "strategy": "wander",
    "target": "",
    "reason": "Invalid or unavailable LLM response; defaulting to wander.",
}

# Cached Gemini client so we don't rebuild it on every call.
_gemini_client: Any = None

# Inference accounting (Phase 4 evidence). Every public request increments a
# counter so callers can prove how much strategy caching saved.
_CALL_STATS: dict[str, int] = {"decision": 0, "strategy": 0}


def get_call_stats() -> dict[str, int]:
    """Return a copy of the LLM request counters (decision / strategy)."""
    return dict(_CALL_STATS)


def reset_call_stats() -> None:
    """Zero the LLM request counters (useful between simulations/tests)."""
    _CALL_STATS["decision"] = 0
    _CALL_STATS["strategy"] = 0


# --- Public entry points ---------------------------------------------------
def get_decision(prompt: str) -> dict[str, Any]:
    """Provider-independent per-turn decision (legacy/back-compat entry point).

    Returns a validated {"action": <valid action>, "reason": <str>}. Always
    succeeds; failures fall back to a safe `rest`.
    """
    _CALL_STATS["decision"] += 1
    if PROVIDER == "random":
        return _random_decision(prompt)
    data = _raw_query(prompt)
    return _validate_decision(data) if data is not None else dict(FALLBACK_DECISION)


def get_strategy(prompt: str) -> dict[str, Any]:
    """Provider-independent high-level strategy request (Phase 4 entry point).

    Returns a validated {"strategy": <valid strategy>, "target": <str>,
    "reason": <str>}. Always succeeds; failures fall back to a safe `wander`.
    """
    _CALL_STATS["strategy"] += 1
    if PROVIDER == "random":
        return _random_strategy(prompt)
    data = _raw_query(prompt)
    return _validate_strategy(data) if data is not None else dict(FALLBACK_STRATEGY)


# --- Validation (one place per vocabulary) ---------------------------------
def _validate_decision(data: Any) -> dict[str, Any]:
    """Coerce a parsed object into a trusted decision dict, or fall back."""
    if not isinstance(data, dict):
        return dict(FALLBACK_DECISION)
    action = data.get("action")
    if action not in VALID_ACTIONS:
        return dict(FALLBACK_DECISION)
    return {"action": action, "reason": str(data.get("reason", ""))}


def _validate_strategy(data: Any) -> dict[str, Any]:
    """Coerce a parsed object into a trusted strategy dict, or fall back.

    Tolerates a model that uses "action" instead of "strategy". `target` is kept
    only when it's a sensible direction or non-empty string; it's never trusted
    blindly because the Python executor re-checks it anyway.
    """
    if not isinstance(data, dict):
        return dict(FALLBACK_STRATEGY)
    kind = data.get("strategy", data.get("action"))
    if kind not in VALID_STRATEGIES:
        return dict(FALLBACK_STRATEGY)
    target = str(data.get("target", "")).strip()
    return {"strategy": kind, "target": target, "reason": str(data.get("reason", ""))}


def _extract_json(text: str) -> str | None:
    """Pull the first JSON object out of a model response.

    Robust to stray prose or ```json code fences: slice from the first '{' to
    the last '}'. Returns None if no plausible object is present.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start: end + 1]


# --- Raw provider dispatch (returns parsed JSON dict or None) ---------------
def _raw_query(prompt: str) -> dict[str, Any] | None:
    """Send `prompt` to the configured real provider; return parsed JSON or None.

    Shared by get_decision and get_strategy so the network/parse plumbing lives
    in exactly one place. The "random" provider is handled by the public entry
    points (its output is synthesised, not parsed), so it never reaches here.
    """
    if PROVIDER == "ollama":
        return _query_ollama(prompt)
    if PROVIDER == "gemini":
        return _query_gemini(prompt)
    # Future providers plug in here without touching the simulation:
    #   if PROVIDER == "openai":
    #       return _query_openai(prompt)
    raise ValueError(f"Unknown PROVIDER: {PROVIDER!r}")


# --- Random provider (offline test/demo) -----------------------------------
def _random_decision(prompt: str) -> dict[str, Any]:
    """Pick a plausible valid ACTION without contacting any model.

    Reads two cheap signals from the prompt so it behaves sensibly rather than
    purely at random, then honours VALID_ACTIONS via _validate_decision.
    """
    if "Current Tile: food" in prompt:
        return _validate_decision({"action": "eat", "reason": "Standing on food."})

    direction_to_food = {
        "North: food": "move_north",
        "South: food": "move_south",
        "East: food": "move_east",
        "West: food": "move_west",
    }
    for marker, action in direction_to_food.items():
        if marker in prompt:
            return _validate_decision(
                {"action": action, "reason": "Food spotted in this direction."}
            )

    move_actions = [a for a in VALID_ACTIONS if a.startswith("move_")]
    action = random.choice(move_actions + ["rest"])
    return _validate_decision({"action": action, "reason": "Wandering."})


def _random_strategy(prompt: str) -> dict[str, Any]:
    """Pick a plausible valid STRATEGY without contacting any model.

    Weighted toward 'wander' and 'seek_food' so the personality executor (which
    drives most behaviour) stays clearly visible during offline runs, while still
    occasionally exercising 'explore' and 'rest'.
    """
    kind = random.choice(
        ["wander", "wander", "seek_food", "seek_food", "explore", "rest"]
    )
    target = random.choice(DIRECTIONS) if kind == "explore" else ""
    return _validate_strategy(
        {"strategy": kind, "target": target, "reason": "Random strategy."}
    )


# --- Ollama provider (local, default) --------------------------------------
def _query_ollama(prompt: str) -> dict[str, Any] | None:
    """Ask a local Ollama model and return the parsed JSON object, or None.

    Talks to the Ollama HTTP API (OLLAMA_URL) with streaming disabled so the full
    completion arrives in one payload. Reasoning models like qwen3 often prepend
    chatter before the JSON, so we lean on _extract_json to recover the embedded
    object. ANY failure (network, non-200, bad JSON) returns None; the caller
    decides which fallback to use.
    """
    try:
        import requests  # imported lazily so non-ollama setups don't need it

        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        text = (response.json().get("response") or "").strip()
        raw = _extract_json(text)
        return json.loads(raw) if raw is not None else None
    except Exception:
        return None


# --- Gemini provider -------------------------------------------------------
def _get_gemini_client() -> Any:
    """Lazily build and cache the Gemini client.

    The SDK and API key are only required when the Gemini provider is actually
    used. Fails loudly (SystemExit) if the key is missing — a misconfiguration,
    not a recoverable model error.
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


def _query_gemini(prompt: str) -> dict[str, Any] | None:
    """Ask Gemini and return the parsed JSON object, or None on any failure."""
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
        return json.loads(raw) if raw is not None else None
    except Exception:
        return None
