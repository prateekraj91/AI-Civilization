"""
main.py
=======

Day 1 entry point for AI Civilization.

Responsibilities (Day 1 only):
  1. Build the world (already provided by world.py).
  2. Create a single agent, "Alex", and register it via the world layer.
  3. Verify the Gemini integration works end-to-end.

OUT OF SCOPE for Day 1 (intentionally not implemented):
  world grid, movement, memory summarization, multiple agents, rendering,
  God Mode.

GEMINI SDK NOTE
---------------
This uses the *new* unified Google Gen AI SDK (`google-genai`, imported as
`from google import genai`). The old `google-generativeai` package with
`genai.configure()` + `genai.GenerativeModel(...)` is deprecated. The new
design is client-based:

    client = genai.Client(api_key=...)
    client.models.generate_content(model=..., contents=...)
"""

import os
import sys

from dotenv import load_dotenv
from google import genai

from agents import Agent
from world import world_state, add_agent

# The Gemini model to use. 2.5-flash is fast/cheap and ideal for a hello-world.
GEMINI_MODEL = "gemini-2.5-flash"


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

    # In the new SDK the client carries the credentials/config; there is no
    # module-level global state to configure.
    return genai.Client(api_key=api_key)


def gemini_test(client: genai.Client, agent: Agent) -> str:
    """Ask Gemini to role-play as the given agent and return its reply.

    NOTE: This reads the agent's data (single source of truth lives in
    world_state, and `agent` is one of those objects) to build the prompt.
    It does not mutate anything — it's a pure read + external call.
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
    # --- 1. Configure the external Gemini integration -------------------
    client = make_gemini_client()

    # --- 2. Create the first agent --------------------------------------
    alex = Agent(
        name="Alex",
        personality="curious and friendly",
        goals={
            "survive": 8,
            "wealth": 3,
            "friendship": 5,
        },
    )

    # --- 3. Register the agent THROUGH the world layer ------------------
    # We never append to world_state directly here; we go through add_agent so
    # the world stays the single source of truth.
    add_agent(alex)

    print(f"Turn {world_state['turn']}: {len(world_state['agents'])} agent(s) in the world.")
    print(f"Created agent: {alex.name} ({alex.personality})")
    print(f"Goals: {alex.goals}\n")

    # --- 4. Verify Gemini works -----------------------------------------
    print("Asking Gemini to introduce Alex...\n")
    reply = gemini_test(client, alex)
    print(f"Gemini says:\n{reply}")


if __name__ == "__main__":
    main()
