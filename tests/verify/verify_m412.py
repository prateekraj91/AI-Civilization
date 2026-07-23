"""
verify_m412.py
==============

Deterministic verification of V2 milestone M4.12: ERA PROGRESSION — the march of ages.
CLOSES Arc 4 (the road to modernity), on top of M4.11 (metallurgy), M4.10 (writing),
Arc 3 (culture), Arc 2 (revolt), Arc 1 (dynasties) and Phases 0-3.

Run offline (Ollama OFF, no model server, no seed-search):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m412.py

The historical step: the world had scattered techs but no sense of ADVANCEMENT through
ages. M4.12 structures the tech tree into ordered ERAS (Neolithic -> Bronze -> Iron ->
...), each transforming the economy, war and a settlement's appearance — so a
civilization visibly marches from the stone age toward modernity. This closes Arc 4.

HEADLINE 1 — ERAS EMERGE FROM TECH: a settlement advances Neolithic -> Bronze -> Iron as
             its populace masters the gating techs (thresholds binding; advance logged);
             a town that masters metalworking advances while one that does not LAGS. Nothing declared.
HEADLINE 2 — TECH ADVANTAGE COMPOUNDS IN WAR: a SMALLER Iron host beats a LARGER Neolithic
             one in the shared battle math (the era weight shown), and it decides a real
             conquest; same-era falls back to numbers. Knowledge beats numbers on an era curve.
HEADLINE 3 — ADVANCED ECONOMIES OUT-PRODUCE PRIMITIVE ONES: an Iron settlement out-produces
             a Bronze out-produces a Neolithic (the era yield curve shown).
APPEARANCE — the settlement's era is exposed in state and drives era-appropriate building
             rendering (the read-only renderer's town-plan keys its style off the era).
EXTENSIBLE — adding a higher era is a pure DATA addition (a 4th era slots in with no new machinery).
COST       — zero added LLM; --eras off byte-identical; deterministic/reproducible.
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from sim import eras
from sim import knowledge
from llm import llm
import main
from sim import monarchy
from sim import world
from sim.agents import Agent
from sim.world import world_state

NEO = frozenset({"fire", "tools", "farming"})
BRONZE = NEO | {"metalworking"}
IRON = BRONZE | {"weapons", "writing"}


# --- Staging helpers ---------------------------------------------------------
def _fresh() -> None:
    world.create_world()
    world_state["eras_on"] = True


def _settlement(sid, center) -> None:
    world_state["settlements"][sid] = {"id": sid, "center": center, "members": set(), "founded": 0}


def _agent(name, pos, sid="S001", *, knows=None, money=0.0) -> Agent:
    a = Agent(name=name, personality="curious and creative")
    world.place_agent(a, *pos)
    a.hunger, a.age, a.lifespan, a.money, a.settlement = 1, 30, 100, money, sid
    if knows:
        a.knowledge.update(knows)
    if sid is not None and sid in world_state["settlements"]:
        world_state["settlements"][sid]["members"].add(name)
    return a


# --- HEADLINE 1: eras emerge from tech ---------------------------------------
def headline_1_eras_emerge() -> None:
    print("=" * 72)
    print("HEADLINE 1 — ERAS EMERGE FROM TECH (thresholds bind; advances logged; some towns lag)")
    print("=" * 72)

    _fresh(); _settlement("S001", (5, 5))
    a = _agent("A", (5, 5), knows=set(NEO))
    print(f"  a settled populace mastering fire+tools+farming -> {eras.settlement_era(world_state, 'S001')}")
    eras.update(world_state, 1)
    a.knowledge.update({"metalworking"})
    ev = eras.update(world_state, 2)
    print(f"  ...then metalworking -> {eras.settlement_era(world_state, 'S001')}  [{ev[0].split(': ', 1)[1]}]")
    a.knowledge.update({"weapons", "writing"})
    ev = eras.update(world_state, 3)
    print(f"  ...then weapons+writing -> {eras.settlement_era(world_state, 'S001')}  [{ev[0].split(': ', 1)[1]}]")
    assert eras.settlement_era(world_state, "S001") == "Iron Age"

    # Two towns, different tech -> different eras: the advanced one advances, the other LAGS.
    _fresh(); _settlement("S001", (2, 2)); _settlement("S002", (8, 8))
    for n, p in [("Aa", (2, 2)), ("Ab", (2, 3))]:
        _agent(n, p, "S001", knows=set(BRONZE))       # a town that took up metalworking
    for n, p in [("Ba", (8, 8)), ("Bb", (8, 9))]:
        _agent(n, p, "S002", knows=set(NEO))          # a town that did not
    eras.update(world_state, 4)
    print(f"\n  two towns, same turn: S001 (has metalworking) = {eras.settlement_era(world_state, 'S001')}, "
          f"S002 (does not) = {eras.settlement_era(world_state, 'S002')}")
    assert eras.settlement_era(world_state, "S001") == "Bronze Age"
    assert eras.settlement_era(world_state, "S002") == "Neolithic"
    print("  -> era is DERIVED from mastered tech, not declared; advances are earned and seed-varying.")
    print()


# --- HEADLINE 2: tech advantage compounds in war -----------------------------
def headline_2_tech_beats_numbers() -> None:
    print("=" * 72)
    print("HEADLINE 2 — TECH ADVANTAGE COMPOUNDS IN WAR (a smaller advanced host beats a larger primitive)")
    print("=" * 72)

    def battle(att_n, att_tech, def_n, def_tech) -> bool:
        _fresh()
        A = [_agent(f"A{i}", (0, i), sid=None, knows=set(att_tech)) for i in range(att_n)]
        D = [_agent(f"D{i}", (1, i), sid=None, knows=set(def_tech)) for i in range(def_n)]
        won, _, _, _ = monarchy.resolve_battle(world_state, A, D, 1, "a", "d")
        return won

    print(f"  3 IRON vs 5 Neolithic -> attacker wins? {battle(3, IRON, 5, NEO)}   (era weight 1.8 vs 1.0)")
    print(f"  3 BRONZE vs 4 Neolithic -> wins? {battle(3, BRONZE, 4, NEO)}   (1.4 vs 1.0)")
    print(f"  4 IRON vs 4 IRON -> wins? {battle(4, IRON, 4, IRON)}   (same era -> numbers, defender holds a tie)")
    assert battle(3, IRON, 5, NEO) and battle(3, BRONZE, 4, NEO) and not battle(4, IRON, 4, IRON)

    # A real conquest: a small IRON army seizes a larger NEOLITHIC town's militia.
    _fresh(); _settlement("S001", (5, 5))
    for n, p in [("m1", (5, 5)), ("m2", (5, 6)), ("m3", (6, 5)), ("m4", (6, 6)), ("m5", (5, 4))]:
        _agent(n, p, knows=set(NEO))                   # 5 Neolithic militia
    warlord = _agent("Warlord", (4, 4), sid=None, money=30.0)
    for i, p in enumerate([(3, 3), (3, 4), (4, 3)]):
        _agent(f"iron{i}", p, sid=None, knows=set(IRON), money=0.5)   # 3 Iron soldiers to hire
    res = monarchy.attempt_conquest(world_state, warlord, "S001", 5)
    print(f"\n  a real conquest: {res['attackers']} IRON soldiers vs {res['defenders']} Neolithic militia "
          f"-> seized? {res['won']}")
    assert res["won"] and res["attackers"] < res["defenders"]
    print("  -> knowledge beats numbers on the era curve, in the shared battle math (conquest/war/uprising).")
    print()


# --- HEADLINE 3: advanced economies out-produce ------------------------------
def headline_3_economy_curve() -> None:
    print("=" * 72)
    print("HEADLINE 3 — ADVANCED ECONOMIES OUT-PRODUCE PRIMITIVE ONES (the era yield curve)")
    print("=" * 72)

    def food_grown(era_tech) -> int:
        _fresh(); _settlement("S001", (5, 5))
        _agent("F", (5, 5), knows=set(era_tech) | {"farming"})
        rng = random.Random(5)
        total = 0
        for t in range(1, 50):
            knowledge.farm(world_state, t, rng)
            total += len(world_state["food"])
            world_state["food"].clear()
        return total

    neo, bronze, iron = food_grown(NEO), food_grown(BRONZE), food_grown(IRON)
    print(f"  food grown over ~50 turns:  Neolithic {neo}  <  Bronze {bronze}  <  Iron {iron}")
    assert neo < bronze < iron
    print("  -> a more advanced settlement is materially richer; the era lifts the production ceiling.")
    print()


# --- APPEARANCE + EXTENSIBILITY ----------------------------------------------
def appearance_and_extensibility() -> None:
    print("=" * 72)
    print("APPEARANCE + EXTENSIBILITY — era drives the renderer; a new era is a data addition")
    print("=" * 72)

    from renderer.pygame_renderer import build_town_plan
    _fresh(); _settlement("S001", (5, 5))
    _agent("A", (5, 5), knows=set(IRON))
    eras.update(world_state, 1)
    print(f"  era exposed in state for the read-only renderer: world_state['eras']['S001'] = "
          f"{world_state['eras']['S001']}  (style '{eras.building_style(world_state, 'S001')}')")
    assert world_state["eras"]["S001"] == "Iron Age"
    neo_plan = build_town_plan((5, 5), 6, None, (200, 200, 200), False, 10, "neolithic")
    iron_plan = build_town_plan((5, 5), 6, None, (200, 200, 200), False, 10, "iron")
    print(f"  the town-plan keys building style off era: Neolithic huts vs Iron stone "
          f"(stone_wall {iron_plan['stone_wall']}, forge {iron_plan['forge']}); walls differ = "
          f"{neo_plan['buildings'][0]['wall'] != iron_plan['buildings'][0]['wall']}")
    assert iron_plan["stone_wall"] and iron_plan["forge"] and not neo_plan["stone_wall"]
    assert neo_plan["buildings"][0]["wall"] != iron_plan["buildings"][0]["wall"]

    # EXTENSIBILITY: a hypothetical Steel Age slots in as a pure data addition.
    eras.ERAS.append(eras.Era("Steel Age", IRON | {"steelmaking"}, 3.6, "steel"))
    try:
        assert eras.settlement_era(world_state, "S001") == "Iron Age"    # not yet — lacks steelmaking
        next(a for a in world_state["agents"] if a.name == "A").knowledge.add("steelmaking")
        print(f"\n  appended Era('Steel Age', ...) to the ladder -> the town reaches "
              f"{eras.settlement_era(world_state, 'S001')} with no new machinery.")
        assert eras.settlement_era(world_state, "S001") == "Steel Age"
    finally:
        eras.ERAS.pop()
    print("  -> towns look different by age; the road to Medieval/Renaissance/Industrial/Modern is a")
    print("     one-line data addition per era. The engine, not the endpoint, is the deliverable.")
    print()


# --- COST: off byte-identical, deterministic, zero added LLM -----------------
def cost_checks() -> None:
    print("=" * 72)
    print("COST — off byte-identical; seeded runs reproduce; zero added LLM")
    print("=" * 72)

    def run(**kw) -> tuple[str, dict]:
        llm.PROVIDER = "random"
        random.seed(42)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(30, settlements=True, monarchy_on=True, tech_tree=knowledge.TECH_TREE,
                                metallurgy_on=True, writing_on=True, **kw)
        return buf.getvalue(), dict(llm.get_call_stats())

    off, off_calls = run()
    off2, _ = run(eras_on=False)
    assert off == off2
    print("  --eras OFF: byte-identical to the metallurgy+writing run (combat delegates to M4.11, no yield curve)")
    on_a, on_calls = run(eras_on=True)
    on_b, _ = run(eras_on=True)
    assert on_a == on_b
    print("  --eras ON: two seeded runs byte-identical (era is a pure derivation — no RNG)")
    assert on_calls == off_calls
    print(f"  eras added ZERO LLM calls (on={on_calls}, off={off_calls}).")
    print()


if __name__ == "__main__":
    saved = llm.PROVIDER
    try:
        headline_1_eras_emerge()
        headline_2_tech_beats_numbers()
        headline_3_economy_curve()
        appearance_and_extensibility()
        cost_checks()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M4.12 VERIFIED — eras EMERGE from mastered tech; an era advantage COMPOUNDS (an advanced host")
    print("beats a larger primitive one in war, out-produces it in economy); towns VISIBLY evolve by age;")
    print("and the ladder is trivially EXTENSIBLE toward modernity. Arc 4 closes — the civilization marches")
    print("through the ages, and the road to modernity is paved.")
    print("=" * 72)
