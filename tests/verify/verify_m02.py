"""
verify_m02.py
=============

Deterministic verification of V2 milestone M0.2: TIERED COGNITION.

Run offline (no model server needed):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m02.py

It demonstrates the three claims M0.2 has to earn — and it does so as a MECHANIC,
not an emergent fluke: no seed-search, no long Qwen run. Every "interesting moment"
below is triggered deterministically (a real world event, fired on purpose) and we
watch the focal set respond.

DEMO A — cost scales with the BUDGET, not the population.
    Run the SAME budget over N = 10, 30, 50 agents. Assert at most `budget` agents
    are focal ("llm") on EVERY turn, and show the LLM strategy-call count stays
    roughly flat as N triples — i.e. inference scales with drama capacity, not crowd
    size. A budget sweep at fixed N shows calls instead track the budget.

DEMO B — the focal set actually TRACKS THE DRAMA.
    A small cast, budget 2. We print the focal set every turn, fire a real theft at
    a chosen agent, and show it gets PROMOTED to focal the moment it's victimised —
    then DEMOTED a few turns later once its life goes quiet (hysteresis holds it
    focal for a minimum tenure first). This is the real result of the milestone.

DEMO C — v1 is unregressed.
    3 agents with a budget >= 3 is byte-identical to the no-tiering (pre-M0.2) path,
    and interestingness ranks an in-conflict agent above a lone wanderer.
"""

from __future__ import annotations

import contextlib
import io
import random

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from sim import alliance
from llm import cognition
from llm import conversation
from llm import llm
import main
from sim import population
from sim import world
from sim.agents import Agent
from sim.world import spawn_food, world_state

_PERSONALITIES = [
    "curious and adventurous",
    "cautious and territorial",
    "friendly and outgoing",
    "independent and competitive",
]


def _build_population(n: int, grid: int, rng: random.Random) -> list[Agent]:
    """Place `n` agents (heuristic baseline) on distinct random cells of a fresh grid."""
    world.create_world(size=grid)
    cells = [(x, y) for x in range(grid) for y in range(grid)]
    rng.shuffle(cells)
    agents: list[Agent] = []
    for i in range(n):
        x, y = cells[i]
        a = Agent(name=f"A{i:02d}", personality=_PERSONALITIES[i % len(_PERSONALITIES)],
                  goals={"survive": 8, "wealth": 3, "friendship": 4},
                  cognition="heuristic")
        world.place_agent(a, x, y)
        agents.append(a)
    return agents


def _tiered_run(n: int, budget: int, turns: int, *, grid: int = 20,
                seed: int = 7) -> dict:
    """Run `turns` of the real loop with M0.2 tiering and return measurements.

    Mirrors main.run_simulation's per-turn order (re-tier, then act, then respawn
    food) but instruments it: it records, for every turn, how many agents are focal
    AFTER the tier update so we can assert the budget is never exceeded.
    """
    rng = random.Random(seed)
    random.seed(seed)
    llm.reset_call_stats()

    agents = _build_population(n, grid, rng)
    spawn_food(int(n * 0.9))

    strategies: dict = {}
    survived: dict[str, int] = {a.name: 0 for a in agents}
    counters: dict[str, int] = {"agent_turns": 0}
    tenure: dict[str, int] = {}

    max_focal = 0
    with contextlib.redirect_stdout(io.StringIO()):
        for turn in range(1, turns + 1):
            world_state["turn"] = turn
            cognition.update_tiers(world_state, turn, budget, tenure)
            living = [a for a in world_state["agents"] if a.alive]
            focal_this_turn = sum(1 for a in living if a.cognition == "llm")
            max_focal = max(max_focal, focal_this_turn)
            assert focal_this_turn <= budget, (turn, focal_this_turn, budget)
            for agent in living:
                main.run_agent_turn(agent, turn, strategies, survived, counters)
            if len(world_state["food"]) < n:
                spawn_food(max(1, n // 6))
            population.process_respawns(turn, world_state)

    stats = llm.get_call_stats()
    return {"n": n, "budget": budget, "turns": turns,
            "strategy_calls": stats["strategy"], "decision_calls": stats["decision"],
            "max_focal": max_focal}


def demo_a_cost_scales_with_budget_not_population() -> None:
    print("=" * 64)
    print("DEMO A — LLM cost scales with the BUDGET, not the population")
    print("=" * 64)

    BUDGET, TURNS = 8, 40
    print(f"Fixed budget = {BUDGET}, turns = {TURNS}. Raising N:")
    print(f"  {'N':>4} | {'max focal/turn':>14} | {'LLM strategy calls':>18} | {'calls/turn':>10}")
    print("  " + "-" * 56)
    results = []
    for n in (10, 30, 50):
        r = _tiered_run(n, BUDGET, TURNS)
        results.append(r)
        print(f"  {n:>4} | {r['max_focal']:>14} | {r['strategy_calls']:>18} | "
              f"{r['strategy_calls'] / TURNS:>10.2f}")
        assert r["max_focal"] <= BUDGET, r

    calls = [r["strategy_calls"] for r in results]
    # Tripling+ the population must NOT meaningfully grow LLM traffic. Allow modest
    # churn slack, but it must be FAR below linear-in-N (which would be ~5x here).
    assert max(calls) <= min(calls) * 1.5 + BUDGET, calls
    print(f"\n  calls stayed ~flat ({min(calls)}..{max(calls)}) while N went 10->50 "
          f"(linear-in-N would be ~{calls[0] * 5}).  PASS\n")

    print(f"Now a BUDGET sweep at fixed N = 40, turns = {TURNS}:")
    print(f"  {'budget':>6} | {'LLM strategy calls':>18}")
    print("  " + "-" * 30)
    sweep = []
    for b in (2, 8, 20):
        r = _tiered_run(40, b, TURNS)
        sweep.append(r["strategy_calls"])
        print(f"  {b:>6} | {r['strategy_calls']:>18}")
    assert sweep[0] < sweep[1] < sweep[2], sweep
    print("\n  calls rise with the budget (and only the budget).  PASS\n")


def _focal_names(state: dict) -> list[str]:
    return sorted(a.name for a in state["agents"] if a.alive and a.cognition == "llm")


def demo_b_focal_set_tracks_drama() -> None:
    print("=" * 64)
    print("DEMO B — the focal set TRACKS THE DRAMA (promote on theft, demote when it moves)")
    print("=" * 64)

    random.seed(3)
    world.create_world(size=10)
    world_state["agents"].clear()
    world_state["food"].clear()
    world_state["events"].clear()

    # 4 agents, budget 2. Two PAIRS in opposite corners. Initially nothing is
    # happening, so the two focal slots fall to Ana & Ben on the tie-break — Victim
    # is NOT focal. We then fire a real THEFT at Victim (it should jump into focal),
    # and later a real BETRAYAL between Ana & Ben (a bigger drama that should pull the
    # focal set back to them, demoting the by-then-quiet Victim). Food keeps everyone
    # alive so what we see is the TIERING moving, never starvation.
    BUDGET = 2
    specs = [
        ("Ana", "friendly and outgoing", (8, 8)),
        ("Ben", "cautious and territorial", (8, 7)),
        ("Victim", "friendly and outgoing", (1, 1)),
        ("Thief", "independent and competitive", (1, 2)),
    ]
    agents = {}
    for name, pers, (x, y) in specs:
        a = Agent(name=name, personality=pers, cognition="heuristic", hunger=0)
        world.place_agent(a, x, y)
        agents[name] = a
    agents["Victim"].inventory.append("food")     # something worth stealing
    agents["Ana"].allies.add("Ben")               # a standing alliance to later betray
    agents["Ben"].allies.add("Ana")
    spawn_food(8)

    strategies: dict = {}
    survived: dict[str, int] = {n: 0 for n in agents}
    counters: dict[str, int] = {"agent_turns": 0}
    tenure: dict[str, int] = {}

    THEFT_TURN, BETRAY_TURN, TURNS = 3, 6, 9
    promoted_turn = demoted_turn = None

    print(f"  budget = {BUDGET}.  theft at turn {THEFT_TURN}, betrayal (elsewhere) at "
          f"turn {BETRAY_TURN}.\n")
    for turn in range(1, TURNS + 1):
        world_state["turn"] = turn

        before = set(_focal_names(world_state))
        cognition.update_tiers(world_state, turn, BUDGET, tenure)
        after = _focal_names(world_state)

        for agent in [a for a in world_state["agents"] if a.alive]:
            with contextlib.redirect_stdout(io.StringIO()):
                main.run_agent_turn(agent, turn, strategies, survived, counters)

        # Fire the scripted drama deterministically at END of turn, so NEXT turn's
        # re-tier sees it (events land one turn before they are perceived, as in v1).
        if turn == THEFT_TURN:
            # Stage a REAL theft: the steal mechanic requires the thief adjacent and
            # the victim standing on a food tile. scan() resolves neighbours by actual
            # position, so pinning the two adjacent (in their quiet corner) is enough.
            victim, thief = agents["Victim"], agents["Thief"]
            victim.position, thief.position = (1, 1), (1, 2)
            if (1, 1) not in world_state["food"]:
                world_state["food"].append((1, 1))   # the food about to be stolen
            conversation.handle_steal(thief, "steal_from_Victim", turn, world_state)
        if turn == BETRAY_TURN:
            alliance.handle_betray(agents["Ana"], "betray_alliance_Ben", turn, world_state)

        tag = ""
        if "Victim" in after and "Victim" not in before:
            tag = "  <- Victim PROMOTED (just robbed)"
            promoted_turn = promoted_turn or turn
        if "Victim" in before and "Victim" not in after:
            tag = "  <- Victim DEMOTED (drama moved on)"
            demoted_turn = demoted_turn or turn
        evt = {THEFT_TURN: "  (theft fires)", BETRAY_TURN: "  (betrayal fires)"}.get(turn, "")
        print(f"  turn {turn:>2}  focal: {', '.join(after):<16}{tag}{evt}")

    print("\n  transition + drama events logged to world_state['events']:")
    for e in world_state["events"]:
        if "focal" in e or "heuristic (routine)" in e or "stole food" in e or "BETRAYED" in e:
            print(f"    {e}")

    assert promoted_turn is not None and promoted_turn > THEFT_TURN, \
        f"Victim was never promoted after the theft (promoted_turn={promoted_turn})"
    assert demoted_turn is not None and demoted_turn > promoted_turn, \
        "Victim was never demoted once the drama moved elsewhere"
    held = demoted_turn - promoted_turn
    assert held >= cognition.MIN_TENURE, (promoted_turn, demoted_turn, cognition.MIN_TENURE)
    print(f"\n  Victim promoted turn {promoted_turn} (right after being robbed), held focal "
          f"{held} turns (>= MIN_TENURE={cognition.MIN_TENURE}), demoted turn {demoted_turn} "
          f"once the betrayal made Ana & Ben the more interesting pair.  PASS\n")


def demo_c_v1_unregressed_and_ranking() -> None:
    print("=" * 64)
    print("DEMO C — v1 unregressed; conflict outranks a lone wanderer")
    print("=" * 64)

    def run(budget):
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(30, focal_budget=budget)
        return buf.getvalue()

    no_tier, tiered = run(None), run(8)
    assert no_tier == tiered, "3-agent budget-8 run diverged from the no-tiering path"
    print("  3 agents, budget 8  ==  no-tiering (pre-M0.2) path: byte-IDENTICAL.  PASS")

    # Interestingness: an in-conflict agent (recent theft) vs a lone, fed wanderer.
    world.create_world(size=10)
    world_state["agents"].clear()
    world_state["food"].clear()
    world_state["events"].clear()
    world_state["turn"] = 5
    conflict = Agent(name="Kira", personality="independent and competitive", hunger=1)
    wanderer = Agent(name="Solo", personality="curious and adventurous", hunger=1)
    wanderer.memory.append("Wandered around.")  # settled, not a blank-slate newcomer
    world.place_agent(conflict, 1, 1)
    world.place_agent(wanderer, 8, 8)
    world_state["events"].append("turn 5: Mallory stole food from Kira")
    conflict.relationships["Mallory"] = {"trust": -5, "interactions": 1, "grudge": True}

    s_conflict = cognition.interestingness(conflict, world_state)
    s_wanderer = cognition.interestingness(wanderer, world_state)
    print(f"  interestingness(in-conflict Kira) = {s_conflict[0]:.1f} ({s_conflict[1]})")
    print(f"  interestingness(lone wanderer Solo) = {s_wanderer[0]:.1f} ({s_wanderer[1]})")
    assert s_conflict[0] > s_wanderer[0], (s_conflict, s_wanderer)
    print("  conflict > wanderer.  PASS\n")


def run() -> None:
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        demo_a_cost_scales_with_budget_not_population()
        demo_b_focal_set_tracks_drama()
        demo_c_v1_unregressed_and_ranking()
    finally:
        llm.PROVIDER = saved
    print("=" * 64)
    print("M0.2 VERIFIED: tiered cognition caps LLM cost at the budget, the focal set "
          "tracks the drama, and v1 is unregressed.")
    print("=" * 64)


if __name__ == "__main__":
    run()
