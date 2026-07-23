"""
verify_m21.py
=============

Deterministic verification of V2 milestone M2.1: SETTLEMENT. Opens Phase 2
(Settlement & Economy), on top of all of Phase 0 + Phase 1 (M1.1 diffusion,
M1.2 discovery, M1.3 tech changes the world / farming food economy).

Run offline (Ollama OFF, no model server):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m21.py

The historical step M2.1 makes: through Phase 1 every agent is a NOMAD. M1.3 let a
knowing population PRODUCE food (farming) into world_state, making a place reliably
worth staying at. M2.1 turns that into the first DURABLE civilizational artifact —
a persistent SETTLEMENT that EMERGES where reliable food makes enough agents stay,
and whose members gain a gentle home-pull. No reliable food -> no settlement; never
a scripted "turn N a village appears".

DEMO A — HEADLINE: matched runs, same seed + population, farming ON vs farming OFF.
         Settlements FORM around food when farming is on (count > 0, clustered at
         food locations) and NONE form when farming is off — settlement is a
         CONSEQUENCE of the food economy, not a scripted event.
DEMO B — EMERGENT, not scripted: across seeds, settlements form at VARYING
         locations and turns (reported).
DEMO C — MEMBERSHIP: an agent that gathers at reliable food joins; an isolated
         nomad never joins.
DEMO D — HOME-PULL: settled members stay tightly clustered around their centre
         (vs sprawling nomads), while a STARVING member still leaves to forage.
DEMO E — ZERO LLM/RNG cost; settlements OFF -> v1 byte-identical.
"""

from __future__ import annotations

import contextlib
import io
import random

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from llm import cognition
from sim import knowledge
from llm import llm
import main
from sim import population
from sim import settlement
from sim import world
from sim.agents import Agent
from llm.strategy import SURVIVAL_HUNGER, choose_action, Strategy
from sim.world import spawn_food, world_state

PERS = ("curious and adventurous", "cautious and territorial",
        "friendly and outgoing", "independent and competitive")


def _matched_run(seed: int, n: int, turns: int, *, farmers_frac: float):
    """A full real-loop run with the settlement system ON; `farmers_frac` seeded farming.

    Mirrors the M1.3 harness (heuristic cast on the scaled economy) but calls
    settlement.update each turn and records every founding (turn, centre).
    """
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
    for a in agents[:int(n * farmers_frac)]:
        a.knowledge.add("farming")
    food_cfg = main.scaled_food_cfg(n)
    spawn_food(food_cfg["initial"])
    strategies, survived, counters, tenure = {}, {}, {"agent_turns": 0}, {}
    founding: list[tuple[int, tuple[int, int]]] = []
    with contextlib.redirect_stdout(io.StringIO()):
        for turn in range(1, turns + 1):
            world_state["turn"] = turn
            cognition.update_tiers(world_state, turn, 8, tenure)
            for a in [x for x in world_state["agents"] if x.alive]:
                main.run_agent_turn(a, turn, strategies, survived, counters)
            knowledge.farm(world_state, turn)
            before = set(world_state["settlements"])
            settlement.update(world_state, turn)
            for sid in sorted(set(world_state["settlements"]) - before):
                founding.append((turn, world_state["settlements"][sid]["center"]))
            main._scaled_respawn_food(turn, food_cfg)
            population.process_respawns(turn, world_state)
    # Snapshot settlements into a plain copy: world_state["settlements"] is a LIVE
    # reference the NEXT _matched_run's create_world would clear out from under us.
    sm = {sid: {"id": s["id"], "center": s["center"],
                "members": set(s["members"]), "founded": s["founded"]}
          for sid, s in world_state["settlements"].items()}
    members = sum(len(s["members"]) for s in sm.values())
    # Mean Chebyshev distance of each settled, living member to ITS settlement centre.
    dists = []
    for a in agents:
        if a.alive and a.settlement is not None:
            c = sm[a.settlement]["center"]
            dists.append(max(abs(a.position[0] - c[0]), abs(a.position[1] - c[1])))
    member_spread = sum(dists) / len(dists) if dists else None
    living = [a for a in agents if a.alive]
    # Sprawl baseline: mean distance of every living agent to the cast's own centroid.
    if living:
        gx = sum(a.position[0] for a in living) / len(living)
        gy = sum(a.position[1] for a in living) / len(living)
        sprawl = sum(max(abs(a.position[0] - gx), abs(a.position[1] - gy))
                     for a in living) / len(living)
    else:
        sprawl = None
    return {"survivors": len(living), "n": n, "n_setts": len(sm), "members": members,
            "founding": founding, "settlements": sm, "agents": agents,
            "member_spread": member_spread, "sprawl": sprawl, "food": list(world_state["food"])}


def demo_a_headline() -> None:
    print("=" * 72)
    print("DEMO A — HEADLINE: settlements form around farmed food, NONE without (matched)")
    print("=" * 72)
    N, TURNS = 100, 60
    for seed in (1, 2):
        farming = _matched_run(seed, N, TURNS, farmers_frac=1.0)
        control = _matched_run(seed, N, TURNS, farmers_frac=0.0)
        fn, cn = farming["n_setts"], control["n_setts"]
        print(f"  seed {seed}:")
        print(f"    farming population: {fn:2d} settlements, {farming['members']:3d}/{N} "
              f"members, {farming['survivors']}/{N} survive")
        print(f"    no-farming control: {cn:2d} settlements, {control['members']:3d}/{N} "
              f"members, {control['survivors']}/{N} survive")
        assert fn >= 5, f"farming should grow many settlements, got {fn}"
        assert cn == 0, f"no reliable food must yield NO settlements, got {cn}"
        # Settlements sit ON the food economy: each centre is near standing food.
        food = set(farming["food"])
        near = sum(1 for s in farming["settlements"].values()
                   if settlement._near_food(s["center"], food, settlement.CLUSTER_RADIUS + 1))
        print(f"      -> {near}/{fn} farming settlements sit on/near standing food")
        assert near >= fn * 0.6, "settlements should cluster at food locations"
    print("\n  Settlements EMERGE around farmed food and never form without it — "
          "settling is a consequence of the food economy, not a scripted event.  PASS\n")


def demo_b_emergent_not_scripted() -> None:
    print("=" * 72)
    print("DEMO B — EMERGENT: across seeds, settlements form at VARYING places & turns")
    print("=" * 72)
    N, TURNS = 100, 60
    first_turns, all_centers = [], []
    for seed in (1, 2, 3, 4):
        r = _matched_run(seed, N, TURNS, farmers_frac=1.0)
        turns = sorted({t for t, _ in r["founding"]})
        centers = [c for _, c in r["founding"]]
        first_turns.append(turns[0] if turns else None)
        all_centers.append(tuple(sorted(centers))[:4])
        print(f"  seed {seed}: {r['n_setts']:2d} settlements; founding turns {turns}; "
              f"first centres {centers[:4]}")
    # The earliest possible founding is SUSTAIN_TURNS (you need that many sustained
    # turns); WHICH turns beyond that, and WHERE, vary by seed — emergence, not a timer.
    assert len(set(all_centers)) > 1, "settlement locations should vary across seeds"
    assert min(t for t in first_turns if t) >= settlement.SUSTAIN_TURNS, \
        "no settlement can form before the sustain window has elapsed"
    print(f"\n  Locations differ every seed and foundings spread across turns "
          f"(floor = SUSTAIN_TURNS={settlement.SUSTAIN_TURNS}) — emergent, not scripted.  PASS\n")


def demo_c_membership() -> None:
    print("=" * 72)
    print("DEMO C — MEMBERSHIP: a gatherer at reliable food joins; an isolated nomad never")
    print("=" * 72)
    world.create_world(size=16)
    # A persistent food plot in one corner; three sustained founders cluster on it.
    plot = [(2, 2), (3, 2), (2, 3), (3, 3)]
    founders = [Agent(name=f"F{i}", personality="cautious and territorial", hunger=0)
                for i in range(3)]
    for a, p in zip(founders, [(2, 2), (3, 2), (2, 3)]):
        world.place_agent(a, *p)
    joiner = Agent(name="Joiner", personality="cautious and territorial", hunger=0)
    world.place_agent(joiner, 12, 12)  # far away at first
    loner = Agent(name="Loner", personality="independent and competitive", hunger=0)
    world.place_agent(loner, 14, 14)   # always isolated, never near the plot

    for turn in range(1, settlement.SUSTAIN_TURNS + 6):
        for p in plot:                 # keep the plot reliably stocked (a farm)
            world.place_food(*p)
        if turn == settlement.SUSTAIN_TURNS + 1:
            world.move_agent(joiner, -1, 0)  # Joiner arrives at the plot edge (11->... )
            joiner.position = (3, 4); world_state["occupancy"][(3, 4)] = joiner
        settlement.update(world_state, turn)

    sid = founders[0].settlement
    print(f"  founders settled into: {sid} (members now "
          f"{sorted(world_state['settlements'][sid]['members'])})")
    print(f"  Joiner.settlement = {joiner.settlement!r}; Loner.settlement = {loner.settlement!r}")
    assert sid is not None and all(f.settlement == sid for f in founders), "founders must settle"
    assert joiner.settlement == sid, "an agent gathering at the reliable food should join"
    assert loner.settlement is None, "an isolated nomad must never join"
    print("  the gatherer joined the settlement; the isolated nomad stayed a nomad.  PASS\n")


def demo_d_home_pull() -> None:
    print("=" * 72)
    print("DEMO D — HOME-PULL: members cluster near centre; a starving member still leaves")
    print("=" * 72)
    # Macro: settled members stay far tighter around their centre than nomads sprawl.
    f = _matched_run(1, 100, 60, farmers_frac=1.0)
    print(f"  mean member distance to settlement centre: {f['member_spread']:.2f} "
          f"(home-pull radius {settlement.HOME_RADIUS})")
    print(f"  mean nomad-equivalent sprawl to cast centroid: {f['sprawl']:.2f}")
    assert f["member_spread"] < f["sprawl"], "settled members should cluster tighter than sprawl"

    # Micro: same settled agent, fed -> pulled home; starving -> overrides to forage.
    world.create_world(size=16)
    a = Agent(name="Cit", personality="curious and adventurous", hunger=0)
    world.place_agent(a, 12, 12)                       # far from its centre at (3, 3)
    world_state["settlements"]["S001"] = {"id": "S001", "center": (3, 3),
                                          "members": {"Cit"}, "founded": 1}
    a.settlement = "S001"
    fed_action, fed_note = choose_action(a, Strategy(kind="explore", target="east"), world_state)
    print(f"  fed member at (12,12), home (3,3): -> {fed_action}  [{fed_note}]")
    assert fed_action in ("move_north", "move_west"), "a fed member should drift toward home"
    assert "home-pull" in fed_note

    world.place_food(13, 12)                           # food to the east, away from home
    a.hunger = SURVIVAL_HUNGER + 2                      # now starving
    hungry_action, hungry_note = choose_action(a, Strategy(kind="explore", target="east"), world_state)
    print(f"  STARVING member, food east at (13,12): -> {hungry_action}  [{hungry_note}]")
    assert hungry_action == "move_east", "a starving member must forage outward (survival overrides)"
    assert "home-pull" not in hungry_note
    print("  fed members are pulled home; a starving member still forages outward.  PASS\n")


def demo_e_zero_cost_and_v1() -> None:
    print("=" * 72)
    print("DEMO E — zero LLM/RNG cost; settlements OFF -> v1 byte-identical")
    print("=" * 72)
    # settlement.update in isolation: zero model calls.
    world.create_world(size=16)
    for i in range(12):
        a = Agent(name=f"S{i:02d}", personality="cautious and territorial", hunger=0)
        world.place_agent(a, i % 16, i // 16)
    llm.reset_call_stats()
    with contextlib.redirect_stdout(io.StringIO()):
        for turn in range(1, 31):
            world_state["turn"] = turn
            settlement.update(world_state, turn)
    stats = llm.get_call_stats()
    print(f"  30 settlement.update passes: LLM calls = {stats}")
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats

    # settlement.update draws NO RNG (deterministic threshold, no dice).
    world.create_world(size=12)
    for i in range(6):
        world.place_agent(Agent(name=f"A{i}", personality="cautious and territorial"), i, 0)
    for p in [(0, 0), (1, 0), (2, 0)]:
        world.place_food(*p)
    st0 = random.getstate()
    for turn in range(1, settlement.SUSTAIN_TURNS + 3):
        world_state["turn"] = turn
        settlement.update(world_state, turn)
    assert random.getstate() == st0, "settlement.update consumed RNG (would desync v1)"

    # settlements OFF (default) -> byte-identical to a run with the param absent.
    def run(flag):
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(25, focal_budget=8)
            else:
                main.run_simulation(25, focal_budget=8, settlements=flag)
        return buf.getvalue()
    assert run(None) == run(False), "settlements=False changed the default run"
    print("  zero model calls; update draws no RNG; settlements OFF byte-identical to v1.  PASS\n")


def run() -> None:
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        demo_a_headline()
        demo_b_emergent_not_scripted()
        demo_c_membership()
        demo_d_home_pull()
        demo_e_zero_cost_and_v1()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M2.1 VERIFIED: nomads become a society with PLACES. Settlements EMERGE around "
          "reliable (farmed) food — varying by seed, never without it — members join and "
          "gain a home-pull (survival overrides), at zero LLM/RNG cost, v1 byte-identical. "
          "Phase 2 is open.")
    print("=" * 72)


if __name__ == "__main__":
    run()
