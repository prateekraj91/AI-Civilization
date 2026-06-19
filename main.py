"""
main.py
=======

Day 3 entry point for AI Civilization — the first real simulation loop.

Responsibilities (Day 3 only):
  1. Build the world, place Alex, spawn food (from Day 2).
  2. For 5 turns: observe -> ask Gemini for a JSON action -> execute -> update
     world_state -> print the turn report.

Gemini must ALWAYS return strict JSON of the form:
    {"action": "move_north", "reason": "Food appears nearby."}
Only the `action` is executed; `reason` is for logging/debugging only.

OUT OF SCOPE for Day 3 (intentionally not implemented):
  memory, hunger, multiple agents, trust/relationships, conversations,
  God Mode, renderer, economy, reputation, alliances, long-term planning.

GEMINI SDK NOTE
---------------
Uses the *new* unified Google Gen AI SDK (`google-genai`, imported as
`from google import genai`). We request JSON via response_mime_type so the
model returns a parseable object rather than free-form prose.
"""

import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from agents import Agent
from world import (
    VALID_ACTIONS,
    create_world,
    execute_action,
    observe,
    place_agent,
    spawn_food,
    world_state,
)

# The Gemini model to use. 2.5-flash is fast/cheap and ideal for this loop.
GEMINI_MODEL = "gemini-2.5-flash"

# Number of turns to simulate (Day 3 spec: exactly 5).
NUM_TURNS = 5

# Module-level Gemini client, initialised in main(). decide() reads it so its
# signature can stay decide(agent, observation) as specified.
client: genai.Client | None = None

# Used whenever Gemini is unreachable or returns anything we can't trust.
FALLBACK_DECISION = {
    "action": "rest",
    "reason": "Invalid or unavailable Gemini response; defaulting to rest.",
}


def make_gemini_client() -> genai.Client:
    """Load the API key from .env and return a configured Gemini client.

    The key is read from the environment (never hard-coded) so secrets stay out
    of source control. Fails loudly with a clear message if the key is missing.
    """
    load_dotenv()  # reads ai_civilization/.env into the environment
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        sys.exit(
            "ERROR: GEMINI_API_KEY is not set. "
            "Create ai_civilization/.env with: GEMINI_API_KEY=your_key_here"
        )

    return genai.Client(api_key=api_key)


def build_prompt(agent: Agent, observation: str) -> str:
    """Construct the instruction sent to Gemini for a single decision.

    Pins the model to the closed action set and demands a strict JSON object so
    the output is machine-parseable rather than free-form text.
    """
    return (
        f"You are {agent.name}, a {agent.personality} agent living on a 10x10 grid.\n"
        f"You can see the four cells adjacent to you:\n"
        f"{observation}\n\n"
        f"Choose EXACTLY ONE action from this list:\n"
        f"  {', '.join(VALID_ACTIONS)}\n\n"
        f"Respond with ONLY a JSON object, no markdown or extra text, shaped exactly:\n"
        f'{{"action": "<one valid action>", "reason": "<short reason>"}}'
    )


def _extract_json(text: str) -> str | None:
    """Pull the first JSON object out of a model response.

    Robust to stray prose or ```json code fences: we slice from the first '{'
    to the last '}'. Returns None if no plausible object is present.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def decide(agent: Agent, observation: str) -> dict[str, Any]:
    """Ask Gemini for the agent's next action and return a validated decision.

    Flow: send the observation -> request strict JSON -> parse safely ->
    validate the action against VALID_ACTIONS. ANY problem (no client, network
    error, bad JSON, missing/invalid action) falls back to a safe `rest`, so the
    simulation loop can never crash on model output.
    """
    if client is None:
        return dict(FALLBACK_DECISION)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=build_prompt(agent, observation),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.7,
            ),
        )

        raw = _extract_json((response.text or "").strip())
        if raw is None:
            return dict(FALLBACK_DECISION)

        data = json.loads(raw)
        action = data.get("action")
        if action not in VALID_ACTIONS:
            return dict(FALLBACK_DECISION)

        # Keep only the fields we trust; reason is optional/logging-only.
        return {"action": action, "reason": str(data.get("reason", ""))}

    except Exception:
        # Network/SDK/JSON errors all degrade gracefully to rest.
        return dict(FALLBACK_DECISION)


def print_turn(turn: int, position: tuple[int, int], observation: str,
               decision: dict[str, Any], result: str) -> None:
    """Print a single turn's report in the Day 3 format."""
    x, y = position
    print("-" * 40)
    print(f"TURN {turn}")
    print()
    print(f"Position: ({x},{y})")
    print()
    print("Observation:")
    print(observation)
    print()
    print("Gemini Decision:")
    print(json.dumps(decision, indent=2))
    print()
    print("Result:")
    print(result)
    print()


def main() -> None:
    global client

    # --- Setup (Day 1 + Day 2) ------------------------------------------
    client = make_gemini_client()
    create_world()

    alex = Agent(
        name="Alex",
        personality="curious and friendly",
        goals={"survive": 8, "wealth": 3, "friendship": 5},
    )
    place_agent(alex, 5, 5)
    spawn_food(5)

    # --- The simulation loop (Day 3) ------------------------------------
    for turn in range(NUM_TURNS):
        world_state["turn"] = turn + 1

        # Capture position BEFORE acting so the report shows where the agent
        # observed from.
        position = alex.position
        observation = observe(alex, world_state)
        decision = decide(alex, observation)
        result = execute_action(alex, decision["action"])

        print_turn(world_state["turn"], position, observation, decision, result)


if __name__ == "__main__":
    main()
