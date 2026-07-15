"""
verify_m411.py
==============

Deterministic verification of V2 milestone M4.11: METALLURGY & ARMS — technology
transforms war and work. Second milestone of Arc 4 (road to modernity), on top of
M4.10 (writing), Arc 3 (culture), Arc 2 (revolt), Arc 1 (dynasties) and Phases 0-3.

Run offline (Ollama OFF, no model server, no seed-search):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m411.py

The historical step: M4.10 made tech act on institutions (memory). M4.11 makes it
transform the MATERIAL balance of power: better tools strengthen the economy, and ARMS
multiply force in battle — so KNOWLEDGE starts to beat NUMBERS, and control of the
knowledge of arms becomes politically decisive. Zero LLM.

HEADLINE 1 — METALLURGY EMERGES + STRENGTHENS THE ECONOMY: invented through the existing
             M1.2 discovery with binding prereqs (tools + a settlement with food surplus);
             a metallurgical farmer out-produces an identical neolithic one (yield gap shown).
HEADLINE 2 — KNOWLEDGE BEATS NUMBERS: in the shared battle math a SMALLER armed host beats a
             LARGER unarmed one (the ~1.8x multiplier shown), and equal-armed forces fall back
             to numbers — and it decides a real CONQUEST (a small armed army takes a larger
             unarmed town), so it applies to conquest, war AND uprising.
HEADLINE 3 — ARMS AND THE REVOLT BALANCE (the sharp composition): an armed ruler's garrison
             CRUSHES an unarmed peasant mob (steel beats numbers) — but when the COMMONERS are
             also armed, the mob's numbers win again. Same uprising, three arms-configs, three
             fates: who controls weapon-knowledge decides who can revolt.
COST       — zero added LLM; --metallurgy off byte-identical; deterministic/reproducible.
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any

import knowledge
import llm
import main
import metallurgy
import monarchy
import uprising
import world
from agents import Agent
from world import world_state


# --- Staging helpers ---------------------------------------------------------
def _fresh() -> None:
    world.create_world()
    world_state["metallurgy_on"] = True


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


def _surplus(center=(5, 5)) -> None:
    cx, cy = center
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if (dx, dy) != (0, 0):
                world.place_food(cx + dx, cy + dy)


# --- HEADLINE 1: metallurgy emerges + strengthens the economy ----------------
def headline_1_metallurgy_emerges() -> None:
    print("=" * 72)
    print("HEADLINE 1 — METALLURGY EMERGES + STRENGTHENS THE ECONOMY (prereqs bind; a yield gap)")
    print("=" * 72)

    def can_invent(item, has_prereq, settled, surplus) -> bool:
        _fresh(); _settlement("S001", (5, 5))
        prereq = "tools" if item == "metalworking" else "metalworking"
        a = _agent("S", (5, 5), sid=("S001" if settled else None),
                   knows=({prereq} if has_prereq else set()))
        if surplus:
            _surplus()
        rng = random.Random(0)
        for _ in range(800):
            metallurgy.discover(world_state, 1, rng)
            if item in a.knowledge:
                return True
        return False

    print(f"  metalworking: tools + settlement + surplus -> {can_invent('metalworking', True, True, True)}")
    print(f"  metalworking: NO surplus                   -> {can_invent('metalworking', True, True, False)}")
    print(f"  metalworking: NO prior tech (tools)        -> {can_invent('metalworking', False, True, True)}")
    print(f"  weapons: needs metalworking                -> {can_invent('weapons', True, True, True)}")
    assert can_invent("metalworking", True, True, True) and can_invent("weapons", True, True, True)
    assert not can_invent("metalworking", True, True, False)
    assert not can_invent("metalworking", False, True, True)
    assert not can_invent("weapons", False, True, True)

    def food_grown(metal) -> int:
        _fresh(); _settlement("S001", (5, 5))
        _agent("F", (5, 5), knows=({"farming", "metalworking"} if metal else {"farming"}))
        rng = random.Random(5)
        total = 0
        for t in range(1, 60):
            knowledge.farm(world_state, t, rng)
            total += len(world_state["food"])
            world_state["food"].clear()
        return total

    neo, metal = food_grown(False), food_grown(True)
    print(f"\n  food grown over ~60 turns:  neolithic farmer {neo}  vs  metalworking farmer {metal}")
    assert metal > neo * 1.4
    print("  -> metallurgy is a tech like any other (prereqs binding) and a metallurgical people")
    print("     is materially more productive than a neolithic one.")
    print()


# --- HEADLINE 2: knowledge beats numbers -------------------------------------
def headline_2_knowledge_beats_numbers() -> None:
    print("=" * 72)
    print("HEADLINE 2 — KNOWLEDGE BEATS NUMBERS (the arms multiplier; it decides a real conquest)")
    print("=" * 72)

    def battle(att_n, att_armed, def_n, def_armed) -> bool:
        _fresh()
        A = [_agent(f"A{i}", (0, i), sid=None, knows=({"weapons"} if att_armed else set())) for i in range(att_n)]
        D = [_agent(f"D{i}", (1, i), sid=None, knows=({"weapons"} if def_armed else set())) for i in range(def_n)]
        won, _, _, _ = monarchy.resolve_battle(world_state, A, D, 1, "att", "def")
        return won

    print(f"  3 ARMED vs 4 unarmed -> attacker wins? {battle(3, True, 4, False)}   (a smaller armed host prevails)")
    print(f"  3 unarmed vs 4 unarmed -> wins? {battle(3, False, 4, False)}   (numbers)")
    print(f"  3 ARMED vs 4 ARMED -> wins? {battle(3, True, 4, True)}   (equal arms -> back to numbers)")
    assert battle(3, True, 4, False) and not battle(3, False, 4, False) and not battle(3, True, 4, True)

    # It decides a REAL conquest: a small ARMED bought army takes a larger UNARMED town (militia).
    _fresh(); _settlement("S001", (5, 5))
    for n, p in [("m1", (5, 5)), ("m2", (5, 6)), ("m3", (6, 5)), ("m4", (6, 6)), ("m5", (5, 4))]:
        _agent(n, p)                                   # 5 unarmed militia (the town's defenders)
    aspirant = _agent("Warlord", (4, 4), sid=None, money=30.0)
    for i, p in enumerate([(3, 3), (3, 4), (4, 3)]):
        _agent(f"soldier{i}", p, sid=None, knows={"weapons"}, money=0.5)   # 3 armed mercenaries to hire
    res = monarchy.attempt_conquest(world_state, aspirant, "S001", 5)
    print(f"\n  a real conquest: {res['attackers']} ARMED soldiers vs {res['defenders']} unarmed militia "
          f"-> seized? {res['won']}")
    assert res["won"] and res["attackers"] < res["defenders"]
    print("  -> knowledge beats numbers in the shared battle math, so it tilts conquest, inter-kingdom")
    print("     war AND uprising alike — a smaller armed force overcomes a larger unarmed one.")
    print()


# --- HEADLINE 3: arms and the revolt balance ---------------------------------
def headline_3_revolt_balance() -> None:
    print("=" * 72)
    print("HEADLINE 3 — ARMS AND THE REVOLT BALANCE (who controls weapons decides who can revolt)")
    print("=" * 72)

    def uprising_wins(ruler_armed, mob_armed) -> bool:
        _fresh(); world_state["discontent_on"] = True; world_state["uprising_on"] = True
        _settlement("S001", (5, 5))
        _agent("King", (4, 4), money=0.5)              # drained: no fresh mercs, the garrison decides
        gk = {"weapons"} if ruler_armed else set()
        garr = [_agent(f"g{i}", (3, 3 + i), knows=set(gk)) for i in range(3)]
        world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0,
                                           "garrison": {g.name for g in garr}}
        mk = {"weapons"} if mob_armed else set()
        for i, p in enumerate([(5, 5), (5, 6), (6, 5), (6, 6), (5, 4)]):
            _agent(f"m{i}", p, knows=set(mk))
        world_state["discontent"] = {f"m{i}": 12.0 for i in range(5)}
        res = uprising.update(world_state, 10)
        return bool(res and res[0]["won"])

    r1 = uprising_wins(ruler_armed=True, mob_armed=False)
    r2 = uprising_wins(ruler_armed=True, mob_armed=True)
    r3 = uprising_wins(ruler_armed=False, mob_armed=False)
    print("  a mob of 5 rises against a garrison of 3:")
    print(f"    ruler's guard ARMED, mob unarmed -> mob wins? {r1}   (steel crushes numbers)")
    print(f"    BOTH sides ARMED                 -> mob wins? {r2}   (numbers win again)")
    print(f"    neither armed                    -> mob wins? {r3}   (numbers)")
    assert not r1 and r2 and r3
    print("  -> an armed crown puts down a bare-handed mob (M4.5's counter-revolution gains real teeth);")
    print("     but arm the commoners and the revolt lives again. Control of weapon-knowledge is political.")
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
            main.run_simulation(30, settlements=True, monarchy_on=True,
                                tech_tree=knowledge.TECH_TREE, **kw)
        return buf.getvalue(), dict(llm.get_call_stats())

    off, off_calls = run()
    off2, _ = run(metallurgy_on=False)
    assert off == off2
    print("  --metallurgy OFF: byte-identical to the tech-tree run (no arms -> the battle math is unchanged)")
    on_a, on_calls = run(metallurgy_on=True)
    on_b, _ = run(metallurgy_on=True)
    assert on_a == on_b
    print("  --metallurgy ON: two seeded runs byte-identical (only discovery draws RNG)")
    assert on_calls == off_calls
    print(f"  metallurgy added ZERO LLM calls (on={on_calls}, off={off_calls}).")
    print()


if __name__ == "__main__":
    saved = llm.PROVIDER
    try:
        headline_1_metallurgy_emerges()
        headline_2_knowledge_beats_numbers()
        headline_3_revolt_balance()
        cost_checks()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M4.11 VERIFIED — metallurgy EMERGES as a tech; better tools STRENGTHEN the economy and arms")
    print("MULTIPLY force so knowledge beats numbers (in conquest, war and uprising); and control of")
    print("weapon-knowledge genuinely shifts the REVOLT balance. Technology becomes power — and who")
    print("knows how to make weapons decides who rules.")
    print("=" * 72)
