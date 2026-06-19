"""
main.py
=======

Day 5 entry point for AI Civilization — provider-independent simulation + memory.

The simulation knows NOTHING about model providers. All AI communication goes
through llm.get_decision(prompt); swapping Gemini for Ollama/OpenAI is a change
in llm.py only, never here.

Per turn:
  1. Time passes (hunger += 1); starvation ends the run immediately.
  2. The agent observes its surroundings.
  3. A prompt (state + recent memories + observation) goes to get_decision().
  4. The chosen action is executed; the outcome is recorded to memory.
  5. The turn (including recent memories) is printed.

OUT OF SCOPE for Day 5 (intentionally not implemented):
  multiple agents, trust, relationships, conversations, beliefs, reputation,
  God Mode, economy, professions.
"""

import json

from agents import Agent
from llm import get_decision
from world import (
    VALID_ACTIONS,
    create_world,
    execute_action,
    is_dead,
    observe,
    place_agent,
    record_memory,
    spawn_food,
    update_hunger,
    world_state,
)

# Maximum turns to simulate (Day 4: 20 turns, or until the agent starves).
NUM_TURNS = 20


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
        f"You are {agent.name}, a {agent.personality} agent living on a 10x10 grid.\n\n"
        f"Position: ({x},{y})\n"
        f"Hunger: {agent.hunger} (0 = full, 10 = you starve to death)\n\n"
        f"{format_memories(agent.memory)}\n\n"
        f"Observation:\n{observation}\n\n"
        f"Rules:\n"
        f"- You may ONLY choose 'eat' if Current Tile = food.\n"
        f"- If your hunger is high, prioritize survival: move toward food and eat it.\n"
        f"- Use your recent memories to avoid repeating unhelpful actions.\n"
        f"- Choose EXACTLY ONE action from: {', '.join(VALID_ACTIONS)}\n\n"
        f"Respond with ONLY a JSON object, no markdown or extra text, shaped exactly:\n"
        f'{{"action": "<one valid action>", "reason": "<short reason>"}}'
    )


def print_turn(turn: int, position: tuple[int, int], hunger: int,
               observation: str, decision: dict, result: str,
               new_hunger: int, memory: list[str]) -> None:
    """Print a single turn's report in the Day 5 format."""
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
    print("Decision:")
    print(json.dumps(decision, indent=2))
    print()
    print("Result:")
    print(result)
    print()
    print(f"New Hunger: {new_hunger}")
    print()
    print(format_memories(memory))
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
    print(format_memories(agent.memory))
    print()


def main() -> None:
    # --- Setup (Day 1 + Day 2) ------------------------------------------
    create_world()

    alex = Agent(
        name="Alex",
        personality="curious and friendly",
        goals={"survive": 8, "wealth": 3, "friendship": 5},
    )
    place_agent(alex, 5, 5)
    spawn_food(5)

    # --- The survival loop (Day 4 + Day 5 memory) -----------------------
    for turn in range(NUM_TURNS):
        world_state["turn"] = turn + 1

        # Time passes first: hunger grows. If it reaches the limit the agent
        # starves before it can act — record it and end immediately.
        update_hunger(alex)
        if is_dead(alex):
            record_memory(alex, "Starved")
            print_death(world_state["turn"], alex)
            break

        # Capture position/hunger BEFORE acting so the report reflects the state
        # the agent observed and decided from.
        position = alex.position
        hunger_before = alex.hunger
        observation = observe(alex, world_state)

        prompt = build_prompt(alex, observation)
        decision = get_decision(prompt)            # provider-independent
        result = execute_action(alex, decision["action"])  # records a memory

        print_turn(world_state["turn"], position, hunger_before, observation,
                   decision, result, alex.hunger, alex.memory)


if __name__ == "__main__":
    main()
