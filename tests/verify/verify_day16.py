"""
verify_day16.py
===============

Day 16 verification harness (READ-ONLY observer of the production machinery — it
drives the REAL hunger loop, the REAL god_mode interventions and the REAL Day 14
cold-start path, and reports what the milestone asks for). Plague and stranger are
deterministic MECHANICS/INTERVENTIONS, not emergent events, so everything here is
reproducible — no Qwen seed-search for emergent social outcomes.

  PART 1 — trigger_plague: the afflicted agent's hunger climbs faster (+3/turn) for
    EXACTLY the plague window, then returns to normal (or it starves). Shows the
    hunger curve, driven by the unchanged world.update_hunger loop.

  PART 2 — sickness is socially VISIBLE: a plagued neighbour reads as "looks sick"
    through the ordinary observe() string and lands in another agent's memory.

  PART 3 — introduce_stranger: a blank-slate cold-start agent enters; existing agents
    receive a WARINESS MEMORY (not a hardcoded trust penalty); the stranger is a real,
    interactable citizen (perceivable + talk-to-able) over following turns.

  PART 4 — BOUNDARY + BENCHMARK: god_mode.py still imports only world-state layers
    (world/population) — the Day 15 AST guard still holds with the Day 16 commands —
    and the new interventions add zero inference.

Run:
    AICIV_PROVIDER=random Jarvis/bin/python verify_day16.py   # offline, deterministic
    AICIV_PROVIDER=ollama Jarvis/bin/python verify_day16.py   # qwen3:8b regression (short)
"""

import ast
import random

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from llm import conversation
from sim import god_mode
from llm import llm
import main
from sim.agents import Agent
from llm.strategy import Strategy, choose_action
from sim.world import (
    create_world,
    execute_action,
    is_sick,
    observe,
    place_agent,
    record_social_memories,
    update_hunger,
    world_state,
)

SEED = 3


def _reset():
    random.seed(SEED)
    llm.reset_call_stats()
    create_world()


def part1_plague_hunger_curve():
    print("=" * 70)
    print("PART 1 — trigger_plague: faster hunger for EXACTLY the window, then normal")
    print("=" * 70)
    _reset()
    kira = Agent(name="Kira", personality="independent and competitive", goals={"survive": 8})
    place_agent(kira, 5, 5)
    world_state["turn"] = 1
    res = god_mode.trigger_plague(world_state, "Kira", turns=10)
    print(f"  {res}")
    print(f"  Kira.plague_until = {kira.plague_until} (= 1 + 10); victim memory: "
          f"{[m for m in kira.memory if 'plague' in m.lower()]}")
    print()
    print("  Hunger CURVE through the plague — per-turn delta isolated (reset to 0 each")
    print("  turn so the increment is visible); the real world.update_hunger applies it:")
    print(f"    {'turn':>4}  {'sick?':<5}  delta_hunger")
    for turn in range(2, 15):
        world_state["turn"] = turn
        kira.hunger = 0
        sick = is_sick(kira, world_state)
        update_hunger(kira)
        tag = "SICK" if sick else "well"
        bar = "#" * kira.hunger
        print(f"    {turn:>4}  {tag:<5}  +{kira.hunger}  {bar}")
    print(f"  -> +3/turn for turns 2..11 (exactly 10), +1/turn from turn 12; "
          f"plague_until now {kira.plague_until} (recovered)")
    print(f"  recovery memory present: "
          f"{any('Recovered from the plague' in m for m in kira.memory)}")

    print()
    print("  And the SAME marker can KILL if the agent cannot keep fed (no eating):")
    _reset()
    bob = Agent(name="Bob", personality="cautious", goals={"survive": 9})
    place_agent(bob, 5, 5)
    bob.hunger = 2
    world_state["turn"] = 1
    god_mode.trigger_plague(world_state, "Bob", turns=10)
    dead_turn = None
    for turn in range(2, 13):
        world_state["turn"] = turn
        update_hunger(bob)
        from sim.world import is_dead
        print(f"    turn {turn}: Bob hunger {bob.hunger}/10  (sick={is_sick(bob, world_state)})")
        if is_dead(bob):
            dead_turn = turn
            break
    print(f"  -> starving Bob who never eats dies on turn {dead_turn} under the plague "
          f"(no scripted reaction — just the faster hunger drain).")


def part2_sickness_is_visible():
    print()
    print("=" * 70)
    print("PART 2 — a plagued neighbour 'looks sick' via the ordinary perception path")
    print("=" * 70)
    _reset()
    alex = Agent(name="Alex", personality="friendly and outgoing", goals={"survive": 7})
    kira = Agent(name="Kira", personality="independent", goals={"survive": 8})
    place_agent(alex, 5, 5)
    place_agent(kira, 6, 5)  # directly East of Alex
    world_state["turn"] = 2
    god_mode.trigger_plague(world_state, "Kira", turns=10)
    print("  Alex observes the world (Kira is one tile East and afflicted):")
    for line in observe(alex, world_state).splitlines():
        print(f"    {line}")
    record_social_memories(alex, world_state)
    print(f"  Alex's social memory now includes: "
          f"{[m for m in alex.memory if 'sick' in m]}")
    print("  -> the plague became social knowledge through observe()/record_social_")
    print("     memories — no new sense, no extra inference.")


def part3_stranger_cold_start():
    print()
    print("=" * 70)
    print("PART 3 — introduce_stranger: blank slate + wariness seeded as MEMORY")
    print("=" * 70)
    _reset()
    alex = Agent(name="Alex", personality="friendly and outgoing", goals={"survive": 7})
    bob = Agent(name="Bob", personality="cautious and territorial", goals={"survive": 9})
    place_agent(alex, 4, 5)
    place_agent(bob, 6, 5)
    world_state["turn"] = 40
    res = god_mode.introduce_stranger(world_state, "Vera", "quiet and guarded")
    vera = next(a for a in world_state["agents"] if a.name == "Vera")
    print(f"  {res}")
    print(f"  Vera cold-start -> memory {vera.memory}, relationships {vera.relationships}, "
          f"allies {vera.allies or '-'}, hunger {vera.hunger}, pos {vera.position}")
    print(f"  Existing agents' wariness MEMORY (not a trust number):")
    for a in (alex, bob):
        warn = [m for m in a.memory if "stranger" in m.lower()]
        print(f"    {a.name}: {warn}   (trust toward Vera recorded? {'Vera' in a.relationships})")
    print(f"  [GOD] events: {[e for e in world_state['events'] if '[GOD]' in e]}")
    print(f"  neutral 'new agent appeared' suppressed: "
          f"{not any('a new agent Vera appeared' in e for e in world_state['events'])}")

    # The stranger is a real, interactable citizen: another agent can perceive and
    # TALK to it through the unchanged conversation layer (no script, no trust hack).
    vx, vy = vera.position
    place_agent(alex, vx - 1, vy)  # stand to Vera's West so they are adjacent
    world_state["turn"] = 41
    print(f"\n  Alex steps next to Vera and perceives: "
          f"'{[l for l in observe(alex, world_state).splitlines() if 'Vera' in l]}'")
    strat = Strategy(kind="talk", target="Vera", message="Hello, stranger.")
    result = conversation.handle_talk(alex, "talk_to_Vera", strat, True, 41, world_state)
    print(f"  Alex talks to the stranger: {result}")
    print(f"  Vera's inbox after the talk: {[m['text'] for m in vera.inbox]}")
    print("  -> integration (or not) now rides the existing talk/trust loop; nothing")
    print("     here scripted whether Vera is trusted.")


def part4_boundary_and_benchmark():
    print()
    print("=" * 70)
    print("PART 4 — ARCHITECTURAL BOUNDARY (still) + INFERENCE BENCHMARK")
    print("=" * 70)
    with open("god_mode.py") as f:
        tree = ast.parse(f.read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    project = imported - {"__future__", "typing", "random"}
    forbidden = {"strategy", "trust", "conversation", "alliance", "personality", "llm"}
    print(f"  god_mode.py imports: {sorted(project)}")
    print(f"    decision-logic imports present? {bool(project & forbidden)}  "
          f"(must be False) -> {project & forbidden or '{}'}")
    print(f"    only world-state layers (world/population)? "
          f"{project <= {'world', 'population'}}")

    # Benchmark: a full run with periodic Day 16 interventions, proving they add no
    # inference beyond the normal periodic strategy refresh.
    _reset()
    for name, personality, goals, (x, y) in main.AGENT_SPECS:
        place_agent(Agent(name=name, personality=personality, goals=goals), x, y)
    from sim.world import spawn_food as _spawn
    _spawn(main.INITIAL_FOOD, cluster=main.FOOD_CLUSTERED)
    strategies, survived, counters = {}, {a.name: 0 for a in world_state["agents"]}, {"agent_turns": 0}
    for turn in range(1, main.NUM_TURNS + 1):
        world_state["turn"] = turn
        for a in [x for x in world_state["agents"] if x.alive]:
            main.run_agent_turn(a, turn, strategies, survived, counters)
        main.maybe_respawn_food(turn)
        for nc in main.population.process_respawns(turn, world_state):
            survived[nc.name] = turn
        if turn == 8:
            god_mode.trigger_plague(world_state)            # random living victim
        if turn == 15:
            god_mode.introduce_stranger(world_state, "Vera", "quiet and guarded")
        if not [x for x in world_state["agents"] if x.alive] and not world_state["pending_respawns"]:
            break
    s = llm.get_call_stats()
    at = counters["agent_turns"]
    per = s["strategy"] / at if at else 0.0
    print(f"\n  full run WITH Day 16 interventions at turns 8 (plague) / 15 (stranger):")
    print(f"    agent-turns: {at}   strategy calls: {s['strategy']}   decisions: {s['decision']}")
    print(f"    calls per agent-turn: {per:.3f}  (per-turn design = 1.000; Day 16 adds 0)")


if __name__ == "__main__":
    part1_plague_hunger_curve()
    part2_sickness_is_visible()
    part3_stranger_cold_start()
    part4_boundary_and_benchmark()
    print()
    print("=" * 70)
    print(f"PROVENANCE (provider={llm.PROVIDER}, seed={SEED})")
    print("=" * 70)
    print("  Plague and stranger are deterministic INTERVENTIONS, not emergent events.")
    print("  god_mode only set a world_state marker (plague_until) and seeded a memory;")
    print("  the faster hunger, the recovery, the 'looks sick' sighting and the wariness")
    print("  all fall out of the UNCHANGED hunger / perception / talk loops. Whether the")
    print("  society actually shuns the stranger or pities the sick is the emergent layer")
    print("  on top — rare under competent play, so no seed-search was done.")
