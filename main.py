"""
main.py
=======

Day 2 entry point for AI Civilization.

Responsibilities (Day 2 only):
  1. Build a 10x10 world grid (the spatial single source of truth).
  2. Place the single agent, "Alex", at (5, 5) through the world layer.
  3. Randomly spawn 5 food items (no overlaps with food or the agent).
  4. Let Alex OBSERVE its four adjacent cells and print the result.

OUT OF SCOPE for Day 2 (intentionally not implemented):
  movement, Gemini decisions, memory logic, hunger logic, multiple agents,
  trust/relationships, conversations, rendering, God Mode.

GEMINI SDK NOTE
---------------
The Day 1 Gemini handshake helpers (`make_gemini_client`, `gemini_test`) are
preserved below but are NOT invoked in the Day 2 turn: Day 2 is purely the grid
+ observation. They use the *new* unified Google Gen AI SDK (`google-genai`,
imported as `from google import genai`).
"""

import os
import sys

from dotenv import load_dotenv
from google import genai

from agents import Agent
from world import (
    create_world,
    observe,
    place_agent,
    spawn_food,
    world_state,
)

# The Gemini model to use. 2.5-flash is fast/cheap and ideal for a hello-world.
GEMINI_MODEL = "gemini-2.5-flash"


def make_gemini_client() -> genai.Client:
    """Load the API key from .env and return a configured Gemini client.

    The key is read from the environment (never hard-coded) so secrets stay out
    of source control. Fails loudly with a clear message if the key is missing.

    NOTE: Preserved from Day 1. Not called during the Day 2 turn.
    """
    load_dotenv()  # reads ai_civilization/.env into the environment
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        sys.exit(
            "ERROR: GEMINI_API_KEY is not set. "
            "Create ai_civilization/.env with: GEMINI_API_KEY=your_key_here"
        )

    # In the new SDK the client carries the credentials/config; there is no
    # module-level global state to configure.
    return genai.Client(api_key=api_key)


def gemini_test(client: genai.Client, agent: Agent) -> str:
    """Ask Gemini to role-play as the given agent and return its reply.

    NOTE: Preserved from Day 1. Not called during the Day 2 turn (no Gemini
    decisions in Day 2).
    """
    prompt = (
        f"You are {agent.name}, a {agent.personality} character. "
        "Say hello and describe yourself in one sentence."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def main() -> None:
    # --- 1. Build the 10x10 world grid ----------------------------------
    create_world()

    # --- 2. Create the single agent -------------------------------------
    alex = Agent(
        name="Alex",
        personality="curious and friendly",
        goals={
            "survive": 8,
            "wealth": 3,
            "friendship": 5,
        },
    )

    # --- 3. Place Alex at (5, 5) THROUGH the world layer -----------------
    # We never write to the grid directly here; place_agent keeps world_state
    # the single source of truth.
    place_agent(alex, 5, 5)

    # --- 4. Spawn 5 food items (no overlap with food or the agent) -------
    spawn_food(5)

    # --- 5. Report the turn + let Alex observe its surroundings ----------
    x, y = alex.position
    print(f"Turn {world_state['turn']}")
    print()
    print(f"Agent: {alex.name}")
    print(f"Position: ({x},{y})")
    print()
    print("Food:")
    print(world_state["food"])
    print()
    print("Observation:")
    print(observe(alex, world_state))


if __name__ == "__main__":
    main()
