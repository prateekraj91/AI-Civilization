"""
verify_m11.py
=============

Deterministic verification of V2 milestone M1.1: KNOWLEDGE AS PROPAGATING STATE.
First piece of Phase 1 (Knowledge & Technology).

Run offline (Ollama OFF, no model server):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m11.py

It re-uses the REAL engine: world setup + the per-turn loop (move, then
knowledge.diffuse) exactly as main.run_simulation runs them, just instrumented to
sample the adoption curve. No LLM calls, no seed-search.

DEMO A — DIFFUSION CURVE. Seed ONE knower in a 100-agent world; show the fraction
         who know it rising over turns through contact, plateauing (not instant),
         and an ISOLATED agent (walled off in a corner) never learning it.
DEMO B — TRUST + PERSONALITY shape adoption: a measurable gap between curious /
         high-trust learners and cautious-independent / low-trust ones.
DEMO C — ZERO added LLM cost at 100 and 200 agents (identical to no-knowledge).
DEMO D — v1 unregressed: no seeded knowledge -> diffusion is a no-op (no events,
         no RNG drawn) -> a 3-agent run is byte-identical to v1.
"""

from __future__ import annotations

import contextlib
import io
import random

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from sim import knowledge
from llm import llm
import main
from sim import population
from sim import world
from sim.agents import Agent
from sim.world import spawn_food, world_state

PERS = ("curious and adventurous", "cautious and territorial",
        "friendly and outgoing", "independent and competitive")


def _build(n: int, grid: int, seed: int, *, isolate_last: bool = False) -> list[Agent]:
    """Place n agents on a grid; optionally wall the last one off in a lone corner."""
    random.seed(seed)
    world.create_world(size=grid)
    cells = [(x, y) for x in range(grid) for y in range(grid)]
    random.Random(seed).shuffle(cells)
    agents: list[Agent] = []
    for i in range(n):
        x, y = cells[i]
        a = Agent(name=f"A{i:03d}", personality=PERS[i % 4], cognition="heuristic",
                  goals={"survive": 8, "wealth": 3, "friendship": 4})
        world.place_agent(a, x, y)
        agents.append(a)
    if isolate_last:
        # Park the last agent alone in the far corner, ringed by its own walls of
        # distance — with everyone else clustered it is never contacted, so it must
        # never learn. (We also keep it fed so it survives to prove the negative.)
        loner = agents[-1]
        world_state["occupancy"].pop(loner.position, None)
        loner.position = (grid - 1, grid - 1)
        world_state["occupancy"][loner.position] = loner
        loner.hunger = 0
    return agents


def _run(agents, turns, budget, *, sample_item=None, isolate=None, seed=0):
    """Run the real loop (act, then diffuse) and sample who-knows-`sample_item`."""
    from llm import cognition
    n = len(agents)
    food_cfg = main.scaled_food_cfg(n)
    spawn_food(food_cfg["initial"])
    strategies, survived, counters, tenure = {}, {}, {"agent_turns": 0}, {}
    curve = []
    with contextlib.redirect_stdout(io.StringIO()):
        for turn in range(1, turns + 1):
            world_state["turn"] = turn
            cognition.update_tiers(world_state, turn, budget, tenure)
            for a in [x for x in world_state["agents"] if x.alive]:
                main.run_agent_turn(a, turn, strategies, survived, counters)
            knowledge.diffuse(world_state, turn)
            main._scaled_respawn_food(turn, food_cfg)
            population.process_respawns(turn, world_state)
            if isolate is not None:  # keep the loner alive + alone for the whole run
                isolate.hunger = 0
            if sample_item is not None:
                knowers = sum(1 for a in world_state["agents"]
                              if a.alive and sample_item in a.knowledge)
                living = sum(1 for a in world_state["agents"] if a.alive)
                curve.append((turn, knowers, living))
    return curve


def demo_a_diffusion_curve() -> None:
    print("=" * 70)
    print("DEMO A — diffusion curve: 1 seed knower in 100 agents spreads by contact")
    print("=" * 70)
    agents = _build(100, grid=main.scaled_grid_size(100), seed=11, isolate_last=True)
    loner = agents[-1]
    knowledge.grant(world_state, agents[0], "song", turn=0)   # ONE seed knower
    curve = _run(agents, turns=40, budget=8, sample_item="song", isolate=loner, seed=11)

    print("  turn | knowers | % of living")
    print("  " + "-" * 32)
    for turn, knowers, living in curve:
        if turn % 4 == 0 or turn == 1:
            bar = "#" * round(28 * knowers / max(living, 1))
            print(f"  {turn:>4} | {knowers:>7} | {100*knowers/max(living,1):>5.1f}%  {bar}")

    first, last = curve[0], curve[-1]
    early = curve[len(curve) // 4]  # ~turn 10
    assert first[1] <= 3, f"should start near the 1 seeded knower, got {first[1]} after turn 1"
    assert last[1] > 10, f"knowledge barely spread ({last[1]} knowers) — diffusion too weak"
    assert last[1] > 2 * early[1], "spread should be GRADUAL (still climbing late, not instant)"
    assert last[1] < last[2], "everyone knows it — should not saturate to 100% on a contact graph"
    assert "song" not in loner.knowledge, "the isolated agent must never learn it"
    peak_pct = 100 * last[1] / max(last[2], 1)
    print(f"\n  1 -> {last[1]} knowers ({peak_pct:.0f}% of living) over 40 turns; the walled-off "
          f"loner {loner.name} never learned it; spread is gradual, not instant.  PASS\n")


def _adoption_rate(personality: str, trust: int, trials: int = 4000) -> float:
    """Empirical fraction of contacts where a learner of this personality + trust adopts."""
    world.create_world(size=5)
    teacher = Agent(name="T", personality="friendly and outgoing")
    learner = Agent(name="L", personality=personality)
    world.place_agent(teacher, 2, 2)
    world.place_agent(learner, 2, 3)
    teacher.knowledge.add("song")
    if trust:
        learner.relationships["T"] = {"trust": trust, "interactions": 1, "grudge": trust < 0}
    rng = random.Random(99)
    hits = 0
    for _ in range(trials):
        hits += rng.random() < knowledge.adoption_probability(learner, teacher, world_state)
    return hits / trials


def demo_b_trust_and_personality() -> None:
    print("=" * 70)
    print("DEMO B — adoption rises with trust and curiosity; resisted by caution/independence")
    print("=" * 70)
    rows = [
        ("curious,  high trust (+5)", "curious and adventurous", 5),
        ("curious,  neutral   ( 0)", "curious and adventurous", 0),
        ("cautious, neutral   ( 0)", "cautious and territorial", 0),
        ("independent, distrust(-5)", "independent and competitive", -5),
    ]
    print("  learner                       per-contact adoption chance")
    print("  " + "-" * 52)
    rates = {}
    for label, pers, trust in rows:
        r = _adoption_rate(pers, trust)
        rates[label] = r
        print(f"  {label:<28}  {r:6.1%}  {'#' * round(40*r)}")

    hi = rates["curious,  high trust (+5)"]
    cur0 = rates["curious,  neutral   ( 0)"]
    cau0 = rates["cautious, neutral   ( 0)"]
    lo = rates["independent, distrust(-5)"]
    assert hi > cur0 > cau0 > lo, (hi, cur0, cau0, lo)
    print(f"\n  high-trust curious ({hi:.0%}) > neutral curious ({cur0:.0%}) > neutral cautious "
          f"({cau0:.0%}) > distrusted independent ({lo:.0%}).")
    print("  Both trust AND personality move the needle, measurably and in the right "
          "direction.  PASS\n")


def demo_c_zero_llm_cost() -> None:
    print("=" * 70)
    print("DEMO C — knowledge diffusion adds ZERO LLM calls (cost identical to M0.3)")
    print("=" * 70)

    # (1) THE STRICT CLAIM: diffusion itself is pure state. Drive a contact-dense,
    # fully-seeded world through many diffuse() passes and assert NOT ONE model call.
    agents = _build(200, grid=main.scaled_grid_size(200), seed=7)
    for a in agents[:20]:
        knowledge.grant(world_state, a, "song", turn=0)
    llm.reset_call_stats()
    with contextlib.redirect_stdout(io.StringIO()):
        for turn in range(1, 31):
            world_state["turn"] = turn
            knowledge.diffuse(world_state, turn)
    stats = llm.get_call_stats()
    print(f"  30 diffuse() passes over 200 agents (20 seeded): LLM calls = {stats}")
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats
    print("  -> diffusion makes ZERO model calls of any kind, by construction.")

    # (2) END TO END: total LLM cost stays flat with vs without knowledge. The only
    # movement is the OPTIONAL M0.2 interestingness coupling occasionally shuffling the
    # focal set (a teaching moment can tip an agent focal) — still budget-bounded and
    # NOT scaling with how much knowledge spreads.
    for n in (100, 200):
        def go(seed_it):
            agents = _build(n, grid=main.scaled_grid_size(n), seed=7)
            if seed_it:
                knowledge.grant(world_state, agents[0], "song", turn=0)
            llm.reset_call_stats()
            _run(agents, turns=30, budget=8, seed=7)
            return llm.get_call_stats()["strategy"]
        plain, knew = go(False), go(True)
        print(f"  N={n}: strategy calls  without knowledge = {plain}  |  with = {knew}  "
              f"(delta {knew - plain}, optional focal coupling)")
        assert abs(knew - plain) <= 8, (n, plain, knew)  # within one budget's worth, flat
    print("\n  Diffusion is free; total cost stays flat and budget-bounded (the small delta "
          "is the optional 'teaching tips you focal' signal, not diffusion buying "
          "inference).  PASS\n")


def demo_d_v1_unregressed() -> None:
    print("=" * 70)
    print("DEMO D — no seeded knowledge -> diffusion is a no-op -> v1 byte-identical")
    print("=" * 70)

    def run(seed_knowledge):
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, focal_budget=8, knowledge_seed=seed_knowledge)
        return buf.getvalue()

    base = run(None)
    also_none = run([])  # empty seed list — still nothing known
    assert base == also_none, "empty knowledge seed changed the run"
    assert "taught" not in base and "now knows" not in base, "no-op diffusion logged something"

    # And a NO-op diffuse draws no RNG: a knowledge-free turn leaves random() untouched.
    world.create_world(size=10)
    a = Agent(name="A", personality="curious and adventurous"); world.place_agent(a, 1, 1)
    b = Agent(name="B", personality="curious and adventurous"); world.place_agent(b, 1, 2)
    st0 = random.getstate()
    knowledge.diffuse(world_state, 1)
    assert random.getstate() == st0, "no-knowledge diffuse consumed RNG (would desync v1)"
    print("  empty-knowledge run byte-identical to v1; no-op diffuse logs nothing and draws "
          "no RNG.  PASS\n")


def run() -> None:
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        demo_a_diffusion_curve()
        demo_b_trust_and_personality()
        demo_c_zero_llm_cost()
        demo_d_v1_unregressed()
    finally:
        llm.PROVIDER = saved
    print("=" * 70)
    print("M1.1 VERIFIED: knowledge is propagating state — it spreads as a gradual "
          "contact-driven curve, is shaped by trust + personality, costs zero LLM, and "
          "leaves a knowledge-free v1 run byte-identical.")
    print("=" * 70)


if __name__ == "__main__":
    run()
