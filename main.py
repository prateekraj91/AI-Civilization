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
    is_dead,
    observe,
    place_agent,
    spawn_food,
    update_hunger,
    world_state,
)

# The Gemini model to use. 2.5-flash is fast/cheap and ideal for this loop.
GEMINI_MODEL = "gemini-2.5-flash"

# Maximum turns to simulate (Day 4: 20 turns, or until the agent starves).
NUM_TURNS = 20

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

    Gives the model the agent's position, hunger, current tile, and nearby
    observations, pins it to the closed action set, and demands strict JSON so
    the output is machine-parseable rather than free-form text.
    """
    x, y = agent.position
    return (
        f"You are {agent.name}, a {agent.personality} agent living on a 10x10 grid.\n\n"
        f"Position: ({x},{y})\n"
        f"Hunger: {agent.hunger} (0 = full, 10 = you starve to death)\n\n"
        f"Observation:\n{observation}\n\n"
        f"Rules:\n"
        f"- You may ONLY choose 'eat' if Current Tile = food.\n"
        f"- If your hunger is high, prioritize survival: move toward food and eat it.\n"
        f"- Choose EXACTLY ONE action from: {', '.join(VALID_ACTIONS)}\n\n"
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


def print_turn(turn: int, position: tuple[int, int], hunger: int,
               observation: str, decision: dict[str, Any], result: str,
               new_hunger: int) -> None:
    """Print a single turn's report in the Day 4 format."""
    x, y = position
    print("-" * 40)
    print(f"TURN {turn}")
    print()
    print(f"Position: ({x},{y})")
    print()
    print(f"Hunger: {hunger}")
    print()
    print("Observation:")
    print()
    print(observation)
    print()
    print("Gemini Decision:")
    print(json.dumps(decision, indent=2))
    print()
    print("Result:")
    print(result)
    print()
    print(f"New Hunger: {new_hunger}")
    print()


def print_death(turn: int, agent: Agent) -> None:
    """Print the final turn header and the starvation message."""
    x, y = agent.position
    print("-" * 40)
    print(f"TURN {turn}")
    print()
    print(f"Position: ({x},{y})")
    print()
    print(f"Hunger: {agent.hunger}")
    print()
    print(f"{agent.name} has died of starvation.")
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

    # --- The survival loop (Day 4) --------------------------------------
    for turn in range(NUM_TURNS):
        world_state["turn"] = turn + 1

        # Time passes first: hunger grows. If it reaches the limit the agent
        # starves before it can act — end the simulation immediately.
        update_hunger(alex)
        if is_dead(alex):
            print_death(world_state["turn"], alex)
            break

        # Capture position/hunger BEFORE acting so the report reflects the state
        # the agent observed and decided from.
        position = alex.position
        hunger_before = alex.hunger
        observation = observe(alex, world_state)
        decision = decide(alex, observation)
        result = execute_action(alex, decision["action"])  # eating lowers hunger

        print_turn(world_state["turn"], position, hunger_before, observation,
                   decision, result, alex.hunger)


if __name__ == "__main__":
    main()
