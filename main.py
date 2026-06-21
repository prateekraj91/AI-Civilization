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

# Maximum turns to simulate (or until every agent has starved). Day 9 lengthened
# runs so social dynamics (talk + trust) had time to emerge; Day 11 keeps them
# long (50) so survival pressure has room to force visible eat/starve outcomes.
NUM_TURNS = 50

# Phase 4: how often (in turns) to refresh an agent's strategy via the LLM.
# Between refreshes the cached strategy is executed in Python — no inference.
STRATEGY_INTERVAL = 5

# --- Food economy: scarcity knobs (Day 11) --------------------------------
# These are the ONLY dials for survival pressure — keep them named here, not as
# magic numbers buried in the loop. Day 11 deliberately REVERSES the Day 9
# abundance rebalance: Day 9 flooded the map (INITIAL_FOOD=14, topped back up to
# 12 EVERY turn) so the social layer could emerge with nobody starving. Day 11
# makes food genuinely SCARCE so agents must compete or cooperate — supply is
# tuned BELOW what three hungry agents consume, so not everyone can stay fed.
#
#   knob                  Day 9 (abundant)      Day 11 (scarce, current)
#   INITIAL_FOOD          14                    5      (was 3; raised for contact)
#   respawn rule          top up to 12 / turn   ~1 food every 5 turns
#   placement             scattered             clustered at centre (contention)
#
# Demand vs supply (why someone starves): EAT_RELIEF=7 means each agent needs
# ~1 food per 7 turns; three agents demand ~0.43 food/turn. The drip below
# supplies ~0.20 food/turn — a deliberate deficit, so food runs out under load.
#
# Day 11 contention tune-up: pure scarcity made agents starve alone in separate
# corners (0 talks, 0 trust changes). Two fixes pull them together WITHOUT making
# the world generous: (1) INITIAL_FOOD 3->5 so they live a few turns longer and
# actually meet; (2) FOOD_CLUSTERED — scarce food spawns on the SAME central
# tiles they start near, so they converge and compete instead of dispersing.
INITIAL_FOOD = 5            # food cells on the map at t=0 (was 3; Day 9 was 14)
FOOD_RESPAWN_EVERY = 5      # respawn cadence: add food once every N turns (~0.2/turn)
FOOD_RESPAWN_AMOUNT = 1     # food cells added on each respawn tick
FOOD_RESPAWN_CAP = 5        # never respawn above this many standing food cells; this
                            # only bounds accumulation AFTER agents die — while the
                            # world is contested, demand keeps food well below it.
FOOD_CLUSTERED = True       # spawn food in a central arena (world.FOOD_CLUSTER_RADIUS)
                            # so agents contend over shared tiles, not scatter.

# The starting cast. Each agent has a distinct dominant trait so behaviour and
# logs are easy to tell apart: Alex = friendly, Bob = cautious, Kira = independent.
AGENT_SPECS = [
    ("Alex", "friendly and outgoing", {"survive": 7, "friendship": 8, "wealth": 2}, (4, 4)),
    ("Bob", "cautious and territorial", {"survive": 9, "wealth": 5, "friendship": 2}, (6, 4)),
    ("Kira", "independent and competitive", {"survive": 7, "wealth": 8, "friendship": 1}, (4, 6)),
]

# Memory entries worth surfacing in the end-of-run summary (Phase 5).
_IMPORTANT_MEMORY_KEYS = ("Observed", "Ate food", "Starved", "New strategy", "Blocked",
                          "stole", "Trust in")


def important_memories(memory: list[str], limit: int = 5) -> list[str]:
    """The most salient recent memories (sightings, meals, strategy changes).

    Falls back to the last few raw memories if nothing notable was recorded, so
    the summary is never empty for an agent that lived.
    """
    notable = [m for m in memory if any(k in m for k in _IMPORTANT_MEMORY_KEYS)]
    chosen = notable or memory
    return chosen[-limit:]


def maybe_respawn_food(turn: int) -> None:
    """Day 11 scarcity: drip food onto the map slowly instead of topping it up.

    Day 9 refilled to FOOD_RESPAWN_TO (12) EVERY turn — effectively unlimited
    food. Day 11 replaces that with a slow trickle of FOOD_RESPAWN_AMOUNT every
    FOOD_RESPAWN_EVERY turns (~1 food / 5 turns), which is intentionally SLOWER
    than three agents eat, so food genuinely runs out and they must compete. The
    CAP only stops unbounded accumulation once agents stop eating; while the
    world is contested it almost never binds.
    """
    if FOOD_RESPAWN_EVERY <= 0:
        return
    if turn % FOOD_RESPAWN_EVERY == 0 and len(world_state["food"]) < FOOD_RESPAWN_CAP:
        spawn_food(FOOD_RESPAWN_AMOUNT, cluster=FOOD_CLUSTERED)


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
    elif action.startswith("steal_from_"):
        result = conversation.handle_steal(agent, action, turn, world_state)
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
    spawn_food(INITIAL_FOOD, cluster=FOOD_CLUSTERED)

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

        maybe_respawn_food(turn)

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
