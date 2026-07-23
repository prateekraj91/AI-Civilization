"""
verify_m01.py
=============

Verification harness for V2 milestone M0.1 (heuristic agents — zero-LLM mind).

It proves the milestone's claims by RUNNING the real loop, not by asserting in the
abstract:

  1. Zero LLM calls. A full run with every agent in heuristic cognition leaves the
     llm call counter at exactly 0.
  2. Scale-readiness. ~50 heuristic agents share one (larger) world and run a full
     sim; it completes quickly and a reasonable number SURVIVE — they seek food
     rather than starving from random walking.
  3. Same loop, different mind. The 50 agents are driven by main.run_agent_turn —
     the SAME per-agent turn function the LLM path uses — flipped only by the
     `cognition="heuristic"` flag.

Run (offline, no model server needed):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m01.py

This is a deterministic mechanic check (seeded), NOT a seed-search or a long Qwen
run.
"""

from __future__ import annotations

import random
import time

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from llm import llm
import main
from sim import population
from sim import world
from sim.agents import Agent
from sim.world import spawn_food, world_state

# --- Scale-test world knobs ------------------------------------------------
# A 50-agent population needs more room and more food than the V1 3-agent scarcity
# map. These live ONLY here — the V1 economy in main.py is untouched.
GRID = 20                 # 20x20 = 400 cells: ~12.5% agent density at 50 agents
POP = 50                  # number of heuristic agents
TURNS = 60                # a full run
INITIAL_FOOD = 45         # food cells at t=0
FOOD_PER_TURN = 8         # scattered food added each turn (see demand note below)
FOOD_CAP = 80             # never accumulate past this many standing food cells

# Demand vs supply, so survival is a fair test of the MIND, not a starved economy:
# EAT_RELIEF=7 means each agent needs ~1 food / 7 turns, so 50 agents demand
# ~50/7 = 7.1 food/turn. FOOD_PER_TURN=8 sits just above that — enough that a
# competent forager lives, tight enough that a random walker would not. A mind
# that ignored food would still mostly starve here; that is the point.

# A spread of personalities so behaviour is heterogeneous (curious explorers,
# cautious hoarders, friendly minglers, lone wolves) rather than 50 identical bots.
_PERSONALITIES = [
    "curious and adventurous",
    "cautious and territorial",
    "friendly and outgoing",
    "independent and competitive",
]


def _build_population(rng: random.Random) -> list[Agent]:
    """Place POP heuristic agents on distinct random cells of a fresh GRID world."""
    world.create_world(size=GRID)
    cells = [(x, y) for x in range(GRID) for y in range(GRID)]
    rng.shuffle(cells)
    agents: list[Agent] = []
    for i in range(POP):
        x, y = cells[i]
        a = Agent(
            name=f"H{i:02d}",
            personality=_PERSONALITIES[i % len(_PERSONALITIES)],
            goals={"survive": 8, "wealth": 3, "friendship": 4},
            cognition="heuristic",   # the single M0.1 switch
        )
        world.place_agent(a, x, y)
        agents.append(a)
    spawn_food(INITIAL_FOOD)  # scattered (cluster=False) — 50 agents need elbow room
    return agents


def run() -> None:
    rng = random.Random(7)        # seed the WORLD setup for reproducibility
    random.seed(7)                # seed the global stream the executor/provider draw from

    llm.reset_call_stats()
    agents = _build_population(rng)

    strategies: dict = {}
    survived: dict[str, int] = {a.name: 0 for a in agents}
    counters: dict[str, int] = {"agent_turns": 0}

    start = time.perf_counter()
    for turn in range(1, TURNS + 1):
        world_state["turn"] = turn
        for agent in [a for a in world_state["agents"] if a.alive]:
            main.run_agent_turn(agent, turn, strategies, survived, counters)
        # Simple sustaining drip (this harness owns its own food economy).
        if len(world_state["food"]) < FOOD_CAP:
            spawn_food(FOOD_PER_TURN)
        # Drain the respawn queue announce_death builds, so the population stays at
        # POP living bodies instead of dwindling — a fair scale test of the mind.
        population.process_respawns(turn, world_state)
    elapsed = time.perf_counter() - start

    alive = [a for a in world_state["agents"] if a.alive]
    # Count survival among the ORIGINAL cohort (respawned newcomers are a bonus).
    original_alive = [a for a in agents if a.alive]
    stats = llm.get_call_stats()
    agent_turns = counters["agent_turns"]

    print("=" * 60)
    print(f"M0.1 SCALE VERIFY — {POP} heuristic agents, {GRID}x{GRID} world, {TURNS} turns")
    print("=" * 60)
    print(f"Agent-turns executed:        {agent_turns}")
    print(f"LLM decision calls:          {stats['decision']}")
    print(f"LLM strategy calls:          {stats['strategy']}")
    print(f"Wall-clock:                  {elapsed:.3f}s "
          f"({1000 * elapsed / max(1, agent_turns):.3f} ms / agent-turn)")
    print(f"Living agents at end:        {len(alive)} total "
          f"({len(original_alive)}/{POP} of the original cohort)")
    print(f"Food on map at end:          {len(world_state['food'])}")
    print()

    # --- Assertions ---------------------------------------------------------
    assert stats["strategy"] == 0, f"expected ZERO strategy calls, got {stats['strategy']}"
    assert stats["decision"] == 0, f"expected ZERO decision calls, got {stats['decision']}"
    assert agent_turns > 0, "no agent-turns ran"
    # "A reasonable number survive" — well above the ~half-or-less a random walker
    # manages on this same economy. We require a clear majority of the original 50.
    survival_rate = len(original_alive) / POP
    assert survival_rate >= 0.6, (
        f"only {survival_rate:.0%} of the original cohort survived — the heuristic "
        f"is not foraging well enough (or the food economy needs tuning)")

    print(f"PASS: zero LLM calls, run completed, "
          f"{survival_rate:.0%} of the original cohort survived.")


if __name__ == "__main__":
    run()
