"""
main.py
=======

Entry point for AI Civilization — a multi-agent, personality- and goal-driven
simulation.

Three agents (Alex, Bob, Kira) share one world and one food supply, act
SEQUENTIALLY each turn, perceive their neighbours by name, and remember sightings
in bounded memory. What makes them feel *different* is the behaviour layer:

  - Personality (personality.py) → distinct per-turn instincts (Phase 1).
  - Goals (agents.Agent.goals)   → fed into the strategy prompt (Phase 2).
  - Memory (bounded)             → recent entries fed into the prompt (Phase 3).
  - Strategy (strategy.py)       → the LLM is asked for a high-level plan only
                                   every STRATEGY_INTERVAL turns; in between, the
                                   plan is executed in pure Python (Phase 4). This
                                   cuts inference cost ~5x.

The simulation knows NOTHING about model providers — all AI calls go through
llm.get_strategy(). Run offline with AICIV_PROVIDER=random.

OUT OF SCOPE (intentionally): economies, villages, governments, factions,
religion, crafting, combat, trading, conversations, God Mode.
"""

import os

import conversation
from agents import Agent
from llm import PROVIDER, get_call_stats, get_strategy, reset_call_stats
from strategy import (
    Strategy,
    build_strategy_prompt,
    choose_action,
    format_goals,
    get_personality,
)
from world import (
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
# Presentation only — never affects the simulation. DEBUG_MODE (default) prints a
# terse per-turn summary; VERBOSE_MODE prints the full detailed report.
# Override with: AICIV_OUTPUT=verbose python main.py
_OUTPUT = os.getenv("AICIV_OUTPUT", "debug").lower()
VERBOSE_MODE = _OUTPUT == "verbose"
DEBUG_MODE = not VERBOSE_MODE  # default

# Maximum turns to simulate (or until every agent has starved). Day 9 runs are
# longer so social dynamics (talk + trust) have time to emerge.
NUM_TURNS = 40

# Phase 4: how often (in turns) to refresh an agent's strategy via the LLM.
# Between refreshes the cached strategy is executed in Python — no inference.
STRATEGY_INTERVAL = 5

# Food economy. Day 9 rebalance: the world is deliberately ABUNDANT so a
# reasonably-playing agent survives ~40 turns and lives long enough to socialise.
# On a 10x10 grid this keeps the nearest food usually within a step or two.
# (Scarcity / competition for limited food is Day 11 — not here.)
INITIAL_FOOD = 14
FOOD_RESPAWN_TO = 12      # keep at least this many food cells on the map
FOOD_RESPAWN_BATCH = 5    # how many to add when topping up

# The starting cast. Each agent has a distinct dominant trait so behaviour and
# logs are easy to tell apart: Alex = friendly, Bob = cautious, Kira = independent.
AGENT_SPECS = [
    ("Alex", "friendly and outgoing", {"survive": 7, "friendship": 8, "wealth": 2}, (4, 4)),
    ("Bob", "cautious and territorial", {"survive": 9, "wealth": 5, "friendship": 2}, (6, 4)),
    ("Kira", "independent and competitive", {"survive": 7, "wealth": 8, "friendship": 1}, (4, 6)),
]

# Memory entries worth surfacing in the end-of-run summary (Phase 5).
_IMPORTANT_MEMORY_KEYS = ("Observed", "Ate food", "Starved", "New strategy", "Blocked")


def important_memories(memory: list[str], limit: int = 5) -> list[str]:
    """The most salient recent memories (sightings, meals, strategy changes).

    Falls back to the last few raw memories if nothing notable was recorded, so
    the summary is never empty for an agent that lived.
    """
    notable = [m for m in memory if any(k in m for k in _IMPORTANT_MEMORY_KEYS)]
    chosen = notable or memory
    return chosen[-limit:]


def maybe_respawn_food() -> None:
    """Top the food supply back up to FOOD_RESPAWN_TO if it has run low."""
    if FOOD_RESPAWN_TO <= 0:
        return
    if len(world_state["food"]) < FOOD_RESPAWN_TO:
        spawn_food(FOOD_RESPAWN_BATCH)


def living_agents() -> list[Agent]:
    """All agents still alive, in turn order."""
    return [a for a in world_state["agents"] if a.alive]


def log_agent_turn(agent: Agent, strat: Strategy, refreshed: bool,
                   observation: str, observed: list[str],
                   action: str, note: str, result: str) -> None:
    """Print one agent's slice of a turn (VERBOSE_MODE)."""
    x, y = agent.position
    source = "refreshed via LLM" if refreshed else "cached"
    print(f"  --- {agent.name} (pos ({x},{y}), hunger {agent.hunger}) ---")
    print(f"    Strategy: {strat.label()} ({source})")
    print("\n".join(f"    {line}" for line in observation.splitlines()))
    if observed:
        print(f"    Detected nearby: {', '.join(observed)}")
    print(f"    Action: {action}  ({note})")
    print(f"    Result: {result}")
    print()


def run_agent_turn(agent: Agent, turn: int, strategies: dict[str, Strategy],
                   survived: dict[str, int], counters: dict[str, int]) -> str:
    """Advance one agent through one turn against the shared world.

    Refreshes the agent's strategy via the LLM only when it is missing or stale
    (every STRATEGY_INTERVAL turns); otherwise the cached strategy is executed in
    pure Python. Returns a short action label for the terse summary.
    """
    # Time passes first: hunger grows. Reaching the limit means starvation.
    update_hunger(agent)
    if is_dead(agent):
        # A meal underfoot saves you at the brink: reaching food costs a turn to
        # step on and another to eat, so an agent that arrived at high hunger
        # would otherwise starve one tick before eating. If it is standing on
        # food, it eats now instead of dying.
        if agent.position in world_state["food"]:
            survived[agent.name] = turn
            counters["agent_turns"] += 1
            result = execute_action(agent, "eat")
            if VERBOSE_MODE:
                print(f"  --- {agent.name} ate at the brink (hunger now {agent.hunger}) ---")
                print(f"    {result}\n")
            return "eat"
        record_memory(agent, "Starved")
        mark_dead(agent)
        if VERBOSE_MODE:
            print(f"  --- {agent.name} ---")
            print(f"    {agent.name} has died of starvation at {agent.position}.")
            print()
        return "starved"

    survived[agent.name] = turn
    counters["agent_turns"] += 1

    # Strategy caching (Phase 4): only hit the LLM when due for a refresh. Any
    # message just received rides into this single call so a reply/reaction needs
    # NO extra inference (Day 8).
    strat = strategies.get(agent.name)
    refresh_due = strat is None or (turn - strat.issued_turn) >= STRATEGY_INTERVAL
    incoming = conversation.pending_incoming(agent, turn)
    observation = observe(agent, world_state) if (refresh_due or VERBOSE_MODE) else ""

    refreshed = False
    if refresh_due:
        data = get_strategy(build_strategy_prompt(agent, observation, incoming=incoming))
        strat = Strategy(kind=data["strategy"], target=data.get("target", ""),
                         message=data.get("message", ""), reaction=data.get("reaction", ""),
                         issued_turn=turn)
        strategies[agent.name] = strat
        record_memory(agent, f"New strategy: {strat.label()}")
        refreshed = True

    # Consume any delivered messages and react (deterministic off-refresh, the
    # strategy call's reaction on a refresh turn). No new LLM call either way.
    conversation.process_inbox(agent, refreshed, strat.reaction, turn, world_state)

    # Detection + social memory still happen every turn (Days 7-8 preserved).
    observed = record_social_memories(agent, world_state)

    # Execute the cached strategy in Python (no inference). A talk action is
    # delivered via the conversation layer; everything else mutates the world.
    action, note = choose_action(agent, strat, world_state)
    if action.startswith("talk_to_"):
        result = conversation.handle_talk(agent, action, strat, refreshed, turn, world_state)
    else:
        result = execute_action(agent, action)

    if VERBOSE_MODE:
        log_agent_turn(agent, strat, refreshed, observation, observed, action, note, result)
    return action


def print_agent_summary(survived: dict[str, int]) -> None:
    """Phase 5: per-agent post-run report for easy analysis."""
    print("=" * 56)
    print("AGENT SUMMARY")
    print("=" * 56)
    for agent in world_state["agents"]:
        pers = get_personality(agent)
        print(f"Agent:           {agent.name}")
        print(f"Personality:     {agent.personality} (dominant: {pers.dominant})")
        print(f"Goals:           {format_goals(agent.goals)}")
        print(f"Status:          {'ALIVE' if agent.alive else 'DEAD'}")
        print(f"Turns survived:  {survived.get(agent.name, 0)} / {NUM_TURNS}")
        print("Important memories:")
        for mem in important_memories(agent.memory):
            print(f"  - {mem}")
        print()


def print_inference_savings(counters: dict[str, int]) -> None:
    """Phase 4 evidence: how much strategy caching reduced LLM calls."""
    stats = get_call_stats()
    agent_turns = counters["agent_turns"]
    strat_calls = stats["strategy"]
    saved = agent_turns - strat_calls
    pct = (100 * saved / agent_turns) if agent_turns else 0.0
    print("=" * 56)
    print("INFERENCE COST (strategy caching)")
    print("=" * 56)
    print(f"Agent-turns executed:      {agent_turns}")
    print(f"LLM strategy calls made:   {strat_calls}")
    print(f"Per-turn design would use: {agent_turns} LLM calls")
    print(f"Saved by caching:          {saved} calls (~{pct:.0f}% fewer)")
    print()


def main() -> None:
    # --- Setup ----------------------------------------------------------
    reset_call_stats()
    create_world()
    for name, personality, goals, (x, y) in AGENT_SPECS:
        place_agent(Agent(name=name, personality=personality, goals=goals), x, y)
    spawn_food(INITIAL_FOOD)

    strategies: dict[str, Strategy] = {}
    survived: dict[str, int] = {a.name: 0 for a in world_state["agents"]}
    counters: dict[str, int] = {"agent_turns": 0}

    if VERBOSE_MODE:
        print(f"AI Civilization — personality-driven simulation (provider: {PROVIDER})")
        print(f"Strategy refresh every {STRATEGY_INTERVAL} turns.")
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

        # Snapshot order at turn start so mid-turn deaths don't disturb iteration.
        actions: list[tuple[str, str]] = []
        for agent in [a for a in world_state["agents"] if a.alive]:
            action = run_agent_turn(agent, turn, strategies, survived, counters)
            actions.append((agent.name, action))

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

    # --- End-of-run analysis (both modes) -------------------------------
    print()
    print_agent_summary(survived)
    print_inference_savings(counters)


if __name__ == "__main__":
    main()
