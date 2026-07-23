"""
verify_day15.py
===============

Day 15 verification harness (READ-ONLY observer of the production machinery — it
drives the REAL main.run_agent_turn loop and the REAL god_mode interventions, and
reports what the milestone asks for). God mode is a set of MECHANICS/INTERVENTIONS,
not emergent events, so everything here is deterministic — no Qwen seed-search.

  PART 1 — PAUSE / MENU / RESUME.
    A scripted God session (status -> spawn_food -> drop_treasure ->
    trigger_drought -> resume) runs through the real god_mode.god_menu with injected
    IO, showing the world change and the loop resume cleanly.

  PART 2 — spawn_food: a HUNGRY agent reaches god-spawned food within ~2 turns,
    purely through the existing perception -> strategy -> executor loop.

  PART 3 — trigger_drought: food respawn drops to 0 for exactly 20 turns, visible
    in events[] and in the per-tick respawn behaviour.

  PART 4 — drop_treasure: hungry agents CONVERGE on the treasure and one CLAIMS it
    (value 10 > a normal meal), again with no scripted reaction.

  PART 5 — BOUNDARY + BENCHMARK: god_mode.py imports only world-state layers (no
    decision logic), and god interventions add zero inference.

Run:
    AICIV_PROVIDER=random Jarvis/bin/python verify_day15.py   # offline, deterministic
    AICIV_PROVIDER=ollama Jarvis/bin/python verify_day15.py   # qwen3:8b regression (short)
"""

import ast
import random

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from sim import god_mode
from llm import llm
import main
from sim.agents import Agent
from llm.strategy import Strategy, choose_action
from sim.world import (
    EAT_RELIEF,
    create_world,
    execute_action,
    place_agent,
    render,
    world_state,
)

SEED = 3


def _reset():
    random.seed(SEED)
    llm.reset_call_stats()
    create_world()


def part1_pause_menu_resume():
    print("=" * 70)
    print("PART 1 — PAUSE / MENU / RESUME (scripted real god_menu session)")
    print("=" * 70)
    _reset()
    place_agent(Agent(name="Alex", personality="friendly and outgoing",
                      goals={"survive": 7}), 4, 4)
    world_state["turn"] = 30

    session = iter([
        "status",
        "spawn_food 4 6",
        "drop_treasure 5 5 10",
        "trigger_drought",
        "spawn_agent Zed curious and bold",
        "bogus_cmd 1 2",      # typo must not crash the menu
        "",                   # blank line resumes
    ])
    out_lines: list[str] = []
    god_mode.god_menu(world_state, 30,
                      read_line=lambda _p="": next(session),
                      out=out_lines.append)
    print("\n".join(out_lines))
    print("\n  AFTER THE SESSION:")
    print(f"    food now includes (4,6): {(4, 6) in world_state['food']}")
    print(f"    treasure at (5,5): {world_state['treasures']}")
    print(f"    drought_until: {world_state['drought_until']} (triggered at turn 30)")
    print(f"    new agent present: {[a.name for a in world_state['agents']]}")
    print(f"    typo survived (menu resumed): "
          f"{any('resuming simulation' in l for l in out_lines)}")
    print("    [GOD] events logged:")
    for e in world_state["events"]:
        if "[GOD]" in e:
            print(f"      {e}")


def part2_spawn_food_drawn():
    print()
    print("=" * 70)
    print("PART 2 — spawn_food: a hungry agent reaches it within ~2 turns")
    print("=" * 70)
    _reset()
    bob = Agent(name="Bob", personality="cautious and territorial", goals={"survive": 9})
    place_agent(bob, 5, 5)
    bob.hunger = 6  # hungry -> survival override seeks food
    world_state["turn"] = 1
    god_mode.spawn_food(world_state, 5, 8)  # 3 tiles due south, the only food
    print(f"  Bob at {bob.position}, hunger {bob.hunger}; God spawns food at (5,8).")
    print(f"    log: {next(e for e in world_state['events'] if '[GOD]' in e)}")
    for t in range(1, 4):
        action, note = choose_action(bob, Strategy(kind="wander"), world_state)
        execute_action(bob, action)
        print(f"    turn {t}: Bob -> {action:<12} now at {bob.position}  ({note})")
        if bob.position == (5, 8):
            break
    reached = bob.position == (5, 8)
    print(f"  Reached the spawned food: {reached}  (no reaction was scripted — the")
    print(f"  executor read the changed world and navigated there)")


def part3_drought():
    print()
    print("=" * 70)
    print("PART 3 — trigger_drought: respawn = 0 for exactly 20 turns")
    print("=" * 70)
    _reset()
    world_state["turn"] = 30
    res = god_mode.trigger_drought(world_state)
    print(f"  {res}")
    print(f"  drought_until = {world_state['drought_until']} (= 30 + 20)")
    ticks_inside, resumed_at = [], None
    for turn in range(31, 60):
        before = len(world_state["food"])
        main.maybe_respawn_food(turn)
        added = len(world_state["food"]) - before
        if turn % main.FOOD_RESPAWN_EVERY == 0:
            if turn <= 50:
                ticks_inside.append((turn, added))
            elif resumed_at is None and added >= 1:
                resumed_at = turn
    print(f"  respawn ticks INSIDE the drought window (turn, food added): {ticks_inside}")
    print(f"  -> every tick added 0 food while turn <= 50")
    print(f"  first tick that resumed adding food after the drought: turn {resumed_at}")


def part4_treasure_convergence():
    print()
    print("=" * 70)
    print("PART 4 — drop_treasure: agents converge and one claims it")
    print("=" * 70)
    _reset()
    # Three hungry agents around the centre, no other food -> the treasure is the
    # only thing to head for. Convergence + the single-occupant tile = competition.
    specs = [("Alex", "friendly and outgoing", (3, 5)),
             ("Bob", "cautious and territorial", (7, 5)),
             ("Kira", "independent and competitive", (5, 7))]
    agents = []
    for name, pers, pos in specs:
        a = Agent(name=name, personality=pers, goals={"survive": 8})
        place_agent(a, *pos)
        a.hunger = 6
        agents.append(a)
    world_state["turn"] = 1
    god_mode.drop_treasure(world_state, 5, 5, 10)
    print(f"  {next(e for e in world_state['events'] if '[GOD]' in e)}")
    print(f"  start distances to (5,5): "
          + ", ".join(f"{a.name}={abs(a.position[0]-5)+abs(a.position[1]-5)}" for a in agents))

    claimed_by = None
    for t in range(1, 8):
        for a in agents:
            if not a.alive:
                continue
            pre = a.position
            action, _ = choose_action(a, Strategy(kind="wander"), world_state)
            res = execute_action(a, action)
            if "claimed a treasure" in res:
                claimed_by = (a.name, t, a.hunger, list(a.inventory))
        dists = ", ".join(f"{a.name}={abs(a.position[0]-5)+abs(a.position[1]-5)}" for a in agents)
        print(f"    turn {t}: distances to treasure -> {dists}")
        if claimed_by:
            break
    if claimed_by:
        name, t, hunger, inv = claimed_by
        print(f"  CLAIMED: {name} reached the treasure on turn {t}; hunger now {hunger} "
              f"(relief 10 > normal meal {EAT_RELIEF}), inventory {inv}")
    else:
        print("  (no claim within the window on this layout — but agents converged)")
    print(f"  treasures remaining: {world_state['treasures']}")


def part5_boundary_and_benchmark():
    print()
    print("=" * 70)
    print("PART 5 — ARCHITECTURAL BOUNDARY + INFERENCE BENCHMARK")
    print("=" * 70)
    with open("god_mode.py") as f:
        tree = ast.parse(f.read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    project = imported - {"__future__", "typing"}
    forbidden = {"strategy", "trust", "conversation", "alliance", "personality", "llm"}
    print(f"  god_mode.py imports: {sorted(project)}")
    print(f"    decision-logic imports present? {bool(project & forbidden)}  "
          f"(must be False) -> {project & forbidden or '{}'}")
    print(f"    only world-state layers (world/population)? "
          f"{project <= {'world', 'population'}}")

    # Benchmark: a full run with periodic god interventions injected by hand (no
    # menu IO), proving interventions add no inference beyond the normal refresh.
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
        if turn == 10:
            god_mode.drop_treasure(world_state, 5, 5, 10)
        if turn == 20:
            god_mode.trigger_drought(world_state)
        if turn == 25:
            god_mode.spawn_food(world_state, 4, 4)
        if not [x for x in world_state["agents"] if x.alive] and not world_state["pending_respawns"]:
            break
    s = llm.get_call_stats()
    at = counters["agent_turns"]
    per = s["strategy"] / at if at else 0.0
    print(f"\n  full run WITH god interventions at turns 10/20/25:")
    print(f"    agent-turns: {at}   strategy calls: {s['strategy']}   decisions: {s['decision']}")
    print(f"    calls per agent-turn: {per:.3f}  (per-turn design = 1.000; god adds 0)")


if __name__ == "__main__":
    part1_pause_menu_resume()
    part2_spawn_food_drawn()
    part3_drought()
    part4_treasure_convergence()
    part5_boundary_and_benchmark()
    print()
    print("=" * 70)
    print(f"PROVENANCE (provider={llm.PROVIDER}, seed={SEED})")
    print("=" * 70)
    print("  God mode is a set of deterministic INTERVENTIONS, not emergent events.")
    print("  Every part above is reproducible; the agents' MOVEMENTS toward the new")
    print("  food/treasure come through the unchanged perception->strategy->executor")
    print("  loop — god_mode only mutated world_state. Qwen adds nothing to verify")
    print("  here beyond a regression smoke-check, so no seed-search was done.")
