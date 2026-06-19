"""
main.py
=======

Days 6-8 entry point for AI Civilization — a MULTI-AGENT simulation.

Three agents (Alex, Bob, Kira) share one world and one food supply. Each turn
they act SEQUENTIALLY (Alex, then Bob, then Kira); food eaten by one vanishes for
the others, so they genuinely compete. Agents perceive their neighbours by name
(Day 7) and remember those sightings in bounded memory (Day 8).

The simulation knows NOTHING about model providers. All AI communication goes
through llm.get_decision(prompt); swapping Ollama for Gemini/random/OpenAI is a
change in llm.py only, never here. Run offline with: AICIV_PROVIDER=random.

Per agent, per turn:
  1. Time passes (hunger += 1); starvation marks the agent dead and skips it.
  2. The agent observes its surroundings (including any neighbours by name).
  3. Social sightings are written to memory (Day 8).
  4. A prompt (state + recent memories + observation) goes to get_decision().
  5. The chosen action is executed against the shared world; the outcome is
     recorded to memory.
  6. The turn is logged.

OUT OF SCOPE (intentionally not implemented yet): conversations, trust /
relationships, reputation, beliefs, God Mode, economy, professions.
"""

import json
import os

from agents import Agent
from llm import PROVIDER, get_decision
from world import (
    VALID_ACTIONS,
    create_world,
    execute_action,
    is_dead,
    mark_dead,
    observe,
    place_agent,
    record_memory,
    record_social_memories,
    render,
    spawn_food,
    update_hunger,
    world_state,
)

# --- Output modes ---------------------------------------------------------
# Two presentation flags that change ONLY what is printed — never the
# simulation itself. DEBUG_MODE (the default) prints a terse per-turn summary:
# turn number, each agent's action, and the food remaining. VERBOSE_MODE prints
# the full detailed report (per-turn map, every observation, decision, result,
# and a final memory summary). Exactly one is active.
#
# Override the default without editing code:  AICIV_OUTPUT=verbose python main.py
_OUTPUT = os.getenv("AICIV_OUTPUT", "debug").lower()
VERBOSE_MODE = _OUTPUT == "verbose"
DEBUG_MODE = not VERBOSE_MODE  # default

# Maximum turns to simulate (or until every agent has starved).
NUM_TURNS = 20

# Food economy. We start with a healthy supply and top it back up whenever it
# runs low, so a multi-agent run stays interesting instead of ending the moment
# the initial food is eaten. Set FOOD_RESPAWN_TO = 0 to disable respawning.
INITIAL_FOOD = 8
FOOD_RESPAWN_TO = 6       # keep at least this many food cells on the map
FOOD_RESPAWN_BATCH = 3    # how many to add when topping up

# The starting cast. Distinct positions and personalities so behaviour and
# logs are easy to tell apart.
AGENT_SPECS = [
    ("Alex", "curious and friendly", {"survive": 8, "wealth": 3, "friendship": 5}, (2, 2)),
    ("Bob", "cautious and territorial", {"survive": 9, "wealth": 5, "friendship": 2}, (7, 7)),
    ("Kira", "bold and competitive", {"survive": 7, "wealth": 6, "friendship": 3}, (2, 7)),
]


def format_memories(memory: list[str]) -> str:
    """Render an agent's recent memories as a bulleted block."""
    if not memory:
        return "Recent Memories:\n- (none yet)"
    lines = "\n".join(f"- {m}" for m in memory)
    return f"Recent Memories:\n{lines}"


def build_prompt(agent: Agent, observation: str) -> str:
    """Construct the decision prompt: state + recent memories + observation.

    Provider-agnostic: this is just a string handed to llm.get_decision(). It
    pins the model to the closed action set and demands strict JSON.
    """
    x, y = agent.position
    return (
        f"You are {agent.name}, a {agent.personality} agent living on a 10x10 grid "
        f"shared with other agents.\n\n"
        f"Position: ({x},{y})\n"
        f"Hunger: {agent.hunger} (0 = full, 10 = you starve to death)\n\n"
        f"{format_memories(agent.memory)}\n\n"
        f"Observation:\n{observation}\n\n"
        f"Rules:\n"
        f"- A direction showing a NAME (e.g. 'North: Bob') is another agent; you "
        f"cannot move onto their cell.\n"
        f"- You may ONLY choose 'eat' if Current Tile = food.\n"
        f"- Food is shared and limited — if you see food and are hungry, race for it.\n"
        f"- Use your recent memories to avoid repeating unhelpful actions.\n"
        f"- Choose EXACTLY ONE action from: {', '.join(VALID_ACTIONS)}\n\n"
        f"Respond with ONLY a JSON object, no markdown or extra text, shaped exactly:\n"
        f'{{"action": "<one valid action>", "reason": "<short reason>"}}'
    )


def maybe_respawn_food() -> None:
    """Top the food supply back up to FOOD_RESPAWN_TO if it has run low."""
    if FOOD_RESPAWN_TO <= 0:
        return
    if len(world_state["food"]) < FOOD_RESPAWN_TO:
        spawn_food(FOOD_RESPAWN_BATCH)


def living_agents() -> list[Agent]:
    """All agents still alive, in turn order."""
    return [a for a in world_state["agents"] if a.alive]


def log_agent_turn(agent: Agent, observation: str, observed: list[str],
                   decision: dict, result: str) -> None:
    """Print one agent's slice of a turn in a clear, scannable format."""
    x, y = agent.position
    print(f"  --- {agent.name} (pos ({x},{y}), hunger {agent.hunger}) ---")
    indented = "\n".join(f"    {line}" for line in observation.splitlines())
    print(indented)
    if observed:
        print(f"    Detected nearby: {', '.join(observed)}")
    print(f"    Decision: {json.dumps(decision)}")
    print(f"    Result: {result}")
    print()


def run_agent_turn(agent: Agent) -> str:
    """Advance a single agent through one turn against the shared world.

    Returns a short label of what the agent did this turn ("move_north",
    "eat", "rest", or "starved") for the terse DEBUG_MODE summary. Printing is
    purely presentational and gated by the output mode; the simulation logic is
    identical regardless of which mode is active.
    """
    # Time passes first: hunger grows. Reaching the limit means the agent
    # starves before it can act.
    update_hunger(agent)
    if is_dead(agent):
        record_memory(agent, "Starved")
        mark_dead(agent)
        if VERBOSE_MODE:
            print(f"  --- {agent.name} ---")
            print(f"    {agent.name} has died of starvation at {agent.position}.")
            print()
        return "starved"

    observation = observe(agent, world_state)
    observed = record_social_memories(agent, world_state)  # Day 8 social memory

    prompt = build_prompt(agent, observation)
    decision = get_decision(prompt)                  # provider-independent
    result = execute_action(agent, decision["action"])  # records a memory

    if VERBOSE_MODE:
        log_agent_turn(agent, observation, observed, decision, result)
    return decision["action"]


def print_final_summary() -> None:
    """Dump survivors, the dead, and every agent's memory at the end."""
    print("=" * 56)
    print("FINAL SUMMARY")
    print("=" * 56)
    survivors = [a.name for a in world_state["agents"] if a.alive]
    casualties = [a.name for a in world_state["agents"] if not a.alive]
    print(f"Survivors:  {', '.join(survivors) or '(none)'}")
    print(f"Casualties: {', '.join(casualties) or '(none)'}")
    print()
    for agent in world_state["agents"]:
        status = "ALIVE" if agent.alive else "DEAD"
        print(f"[{status}] {agent.name} — pos {agent.position}, hunger {agent.hunger}")
        print(format_memories(agent.memory))
        print()


def main() -> None:
    # --- Setup ----------------------------------------------------------
    create_world()
    for name, personality, goals, (x, y) in AGENT_SPECS:
        place_agent(Agent(name=name, personality=personality, goals=goals), x, y)
    spawn_food(INITIAL_FOOD)

    if VERBOSE_MODE:
        print(f"AI Civilization — multi-agent simulation (provider: {PROVIDER})")
        print(f"Agents: {', '.join(a.name for a in world_state['agents'])}")
        print()

    # --- The shared survival loop ---------------------------------------
    for turn in range(1, NUM_TURNS + 1):
        world_state["turn"] = turn

        if VERBOSE_MODE:
            print("=" * 56)
            print(f"TURN {turn}  |  food on map: {len(world_state['food'])}")
            print(render(world_state))
            print()

        # Agents act sequentially and in a stable order. Snapshot the order at
        # the start of the turn so deaths mid-turn don't disturb iteration.
        actions: list[tuple[str, str]] = []
        for agent in [a for a in world_state["agents"] if a.alive]:
            action = run_agent_turn(agent)
            actions.append((agent.name, action))

        # Terse DEBUG_MODE summary: turn, each agent's action, food remaining.
        # Read the food count BEFORE topping up so it reflects what this turn's
        # competition actually left behind.
        if DEBUG_MODE:
            print(f"TURN {turn}")
            for name, action in actions:
                print(f"{name} -> {action}")
            print()
            print(f"Food remaining: {len(world_state['food'])}")
            print()

        maybe_respawn_food()

        if not living_agents():
            if VERBOSE_MODE:
                print("All agents have died. Ending simulation.")
            break

    if VERBOSE_MODE:
        print()
        print_final_summary()


if __name__ == "__main__":
    main()
