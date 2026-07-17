"""
verify_m12.py
=============

Deterministic verification of V2 milestone M1.2: DISCOVERY (unscripted invention).
Phase 1, on top of M1.1 (knowledge diffusion).

Run offline (Ollama OFF, no model server):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m12.py

It re-uses the REAL engine: world setup + the per-turn loop (act, then
knowledge.discover, then knowledge.diffuse) exactly as main.run_simulation runs
them. No LLM calls; discovery is pure state math.

DEMO A — UNSCRIPTED: discovery is a situational roll, not a timer. At a small scale
         (a real gamble) the turn 'fire' is first invented VARIES a lot by seed;
         at 100 agents the whole chain fire -> tools/cooking -> farming unlocks
         over turns that vary by seed.
DEMO B — PREREQUISITES gate it: an agent that knows nothing can only ever invent
         the base item; tools never precede fire, farming never precedes tools.
DEMO C — DISCOVERY -> DIFFUSION: an invented item then spreads via the existing
         M1.1 diffusion (one discoverer -> a rising adoption curve).
DEMO D — ZERO LLM cost, and v1 unregressed: no tech tree -> discovery is a no-op
         drawing no RNG -> a 3-agent run is byte-identical to v1.
"""

from __future__ import annotations

import contextlib
import io
import random

import knowledge
import llm
import main
import population
import world
from agents import Agent
from world import spawn_food, world_state

PERS = ("curious and adventurous", "cautious and territorial",
        "friendly and outgoing", "independent and competitive")


def _build(n: int, seed: int) -> list[Agent]:
    random.seed(seed)
    grid = main.scaled_grid_size(n)
    world.create_world(size=grid)
    cells = [(x, y) for x in range(grid) for y in range(grid)]
    random.Random(seed).shuffle(cells)
    agents = []
    for i in range(n):
        a = Agent(name=f"A{i:03d}", personality=PERS[i % 4], cognition="heuristic",
                  goals={"survive": 8, "wealth": 3, "friendship": 4})
        world.place_agent(a, *cells[i])
        agents.append(a)
    spawn_food(main.scaled_food_cfg(n)["initial"])
    return agents


def _run(n, turns, seed, *, tech_tree=None, sample_item=None, track_first=False):
    """Run the real loop with discovery + diffusion; return curve / first-discovery turns."""
    import cognition
    food = main.scaled_food_cfg(n)
    strategies, survived, counters, tenure = {}, {}, {"agent_turns": 0}, {}
    curve, firsts = [], {}
    with contextlib.redirect_stdout(io.StringIO()):
        for turn in range(1, turns + 1):
            world_state["turn"] = turn
            cognition.update_tiers(world_state, turn, 8, tenure)
            for a in [x for x in world_state["agents"] if x.alive]:
                main.run_agent_turn(a, turn, strategies, survived, counters)
            knowledge.discover(world_state, turn, tech_tree)
            knowledge.diffuse(world_state, turn)
            main._scaled_respawn_food(turn, food)
            population.process_respawns(turn, world_state)
            if track_first:
                for e in world_state["events"]:
                    if "discovered '" in e:
                        item = e.split("discovered '")[1].rstrip("'")
                        firsts.setdefault(item, turn)
            if sample_item is not None:
                knowers = sum(1 for a in world_state["agents"]
                              if a.alive and sample_item in a.knowledge)
                living = sum(1 for a in world_state["agents"] if a.alive)
                curve.append((turn, knowers, living))
    return {"curve": curve, "firsts": firsts}


def demo_a_unscripted() -> None:
    print("=" * 70)
    print("DEMO A — discovery is UNSCRIPTED: the turn it fires varies by run, no timer")
    print("=" * 70)
    print("  small scale (12 agents = a real gamble): first turn 'fire' is invented,")
    print("  by seed:")
    fire_turns = []
    for seed in range(1, 9):
        _build(12, seed)
        r = _run(12, 60, seed, tech_tree=knowledge.TECH_TREE, track_first=True)
        fire_turns.append(r["firsts"].get("fire"))
    print(f"    {dict(zip(range(1, 9), fire_turns))}")
    assert len(set(fire_turns)) >= 4, f"fire turn barely varies ({fire_turns}) — looks scripted"

    print("\n  full chain (100 agents), first-discovery turn per item, by seed:")
    chains = []
    for seed in (1, 2, 3):
        _build(100, seed)
        r = _run(100, 45, seed, tech_tree=knowledge.TECH_TREE, track_first=True)
        chains.append(r["firsts"])
        print(f"    seed {seed}: {r['firsts']}")
    # Every chain must respect order, and downstream turns must differ across seeds.
    for f in chains:
        assert "fire" in f, "fire was never discovered in a 100-agent run"
        if "tools" in f:
            assert f["tools"] >= f["fire"], f
        if "farming" in f:
            assert "tools" in f and f["farming"] >= f["tools"], f
    downstream = [f.get("farming") or f.get("tools") for f in chains]
    assert len(set(downstream)) >= 2, f"downstream turns identical across seeds: {downstream}"
    print("\n  fire's turn swings widely at small scale, and the chain's later turns vary by "
          "seed at scale — discovery is situational, not a fixed schedule.  PASS\n")


def demo_b_prerequisites() -> None:
    print("=" * 70)
    print("DEMO B — prerequisites gate invention (no tools before fire, no farming before tools)")
    print("=" * 70)
    # A single fed, isolated agent that knows NOTHING. With the base rate boosted so we
    # don't need luck, watch what it is ALLOWED to invent, pass by pass.
    saved_base = knowledge.DISCOVERY_BASE
    knowledge.DISCOVERY_BASE = 1.0  # guarantee a roll succeeds, to expose the gate cleanly
    try:
        world.create_world(size=5)
        a = Agent(name="Solo", personality="curious and adventurous", hunger=0)
        world.place_agent(a, 2, 2)
        rng = random.Random(0)

        knowledge.discover(world_state, 1, knowledge.TECH_TREE, rng=rng)
        print(f"  pass 1 (knew nothing): now knows {sorted(a.knowledge)}")
        assert a.knowledge == {"fire"}, "only the no-prereq base item should be inventable first"

        knowledge.discover(world_state, 2, knowledge.TECH_TREE, rng=rng)
        print(f"  pass 2 (knew fire):    now knows {sorted(a.knowledge)}")
        assert "tools" in a.knowledge and "cooking" in a.knowledge, "fire should unlock its branches"
        assert "farming" not in a.knowledge, "farming must NOT appear before tools is known"

        knowledge.discover(world_state, 3, knowledge.TECH_TREE, rng=rng)
        print(f"  pass 3 (knew tools):   now knows {sorted(a.knowledge)}")
        assert "farming" in a.knowledge, "with tools known, farming becomes inventable"
    finally:
        knowledge.DISCOVERY_BASE = saved_base

    # And an agent that never learns fire can NEVER reach the downstream items.
    world.create_world(size=5)
    b = Agent(name="NoFire", personality="curious and adventurous", hunger=0)
    world.place_agent(b, 2, 2)
    no_fire_tree = {k: v for k, v in knowledge.TECH_TREE.items() if k != "fire"}  # fire unreachable
    rng = random.Random(1)
    for turn in range(1, 200):
        knowledge.discover(world_state, turn, no_fire_tree, rng=rng)
    assert not b.knowledge, "with fire unreachable, nothing downstream can ever be invented"
    print("  with fire unreachable, 200 passes invent NOTHING downstream.  PASS\n")


def demo_c_discovery_then_diffusion() -> None:
    print("=" * 70)
    print("DEMO C — a DISCOVERED item then spreads via M1.1 diffusion")
    print("=" * 70)
    _build(100, seed=2)
    r = _run(100, 45, 2, tech_tree=knowledge.TECH_TREE, sample_item="fire", track_first=True)
    fire_turn = r["firsts"]["fire"]
    print(f"  'fire' first invented on turn {fire_turn}; knowers over time:")
    print("  turn | knowers | % living")
    print("  " + "-" * 30)
    for turn, knowers, living in r["curve"]:
        if turn in (fire_turn, fire_turn + 5, fire_turn + 15) or turn % 12 == 0:
            bar = "#" * round(26 * knowers / max(living, 1))
            print(f"  {turn:>4} | {knowers:>7} | {100*knowers/max(living,1):>5.1f}%  {bar}")
    last = r["curve"][-1]
    assert last[1] > 5, f"a discovered item barely spread ({last[1]} knowers)"
    assert last[1] < last[2], "should not saturate to everyone (contact-gated)"
    print(f"\n  one invention -> {last[1]} knowers via contact diffusion (reusing M1.1, no new "
          f"spread code).  PASS\n")


def demo_d_zero_cost_and_v1() -> None:
    print("=" * 70)
    print("DEMO D — discovery adds ZERO LLM calls; no tech tree -> v1 byte-identical")
    print("=" * 70)
    # (1) Diffusion+discovery in isolation: zero model calls of any kind.
    _build(150, seed=4)
    llm.reset_call_stats()
    rng = random.Random(0)
    with contextlib.redirect_stdout(io.StringIO()):
        for turn in range(1, 31):
            world_state["turn"] = turn
            knowledge.discover(world_state, turn, knowledge.TECH_TREE, rng=rng)
            knowledge.diffuse(world_state, turn)
    stats = llm.get_call_stats()
    print(f"  30 discover()+diffuse() passes over 150 agents: LLM calls = {stats}")
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats

    # (2) v1 unregressed: tech_tree=None run == no-knowledge run; and a no-op discover
    # draws no RNG (or it would desync the v1 stream).
    def run(tree):
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, focal_budget=8, tech_tree=tree)
        return buf.getvalue()
    base, no_tree = run(None), run({})
    assert base == no_tree, "an empty tech tree changed the run"
    assert "discovered" not in base, "no-op discovery logged something"

    world.create_world(size=10)
    Agent_a = Agent(name="A", personality="curious and adventurous", hunger=0)
    world.place_agent(Agent_a, 1, 1)
    st0 = random.getstate()
    knowledge.discover(world_state, 1, None)
    assert random.getstate() == st0, "no-tree discover consumed RNG (would desync v1)"
    print("  zero model calls; empty/None tech tree is byte-identical to v1 and draws no "
          "RNG.  PASS\n")


def run() -> None:
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        demo_a_unscripted()
        demo_b_prerequisites()
        demo_c_discovery_then_diffusion()
        demo_d_zero_cost_and_v1()
    finally:
        llm.PROVIDER = saved
    print("=" * 70)
    print("M1.2 VERIFIED: agents invent items unscripted (situational, prereq-gated, "
          "varies by run), the discovery then spreads via M1.1, at zero LLM cost, and a "
          "tree-free v1 run is byte-identical.")
    print("=" * 70)


if __name__ == "__main__":
    run()
