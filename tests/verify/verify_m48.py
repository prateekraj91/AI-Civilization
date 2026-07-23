"""
verify_m48.py
=============

Deterministic verification of V2 milestone M4.8: RELIGION AS INSTITUTION — shared
belief becomes power. Second milestone of Arc 3 (Belief & Culture), on top of M4.7
(beliefs), Arc 2 (discontent/uprising), Arc 1 (lineage/dynasties) and Phases 0-3.

Run offline (Ollama OFF, no model server, no seed-search):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m48.py

The historical step: M4.7 gave settlements distinct dominant belief sets. M4.8 turns
a coherent shared set into a named FAITH, raises a PROPHET from the devout, and makes
faith touch LEGITIMACY — a ruler ALIGNED with the local faith is trusted, a DEFIANT one
resented. It adds NO new political mechanic: faith only MOVES THE TRUST DIAL the existing
systems read, and the UNCHANGED M4.4 (discontent), M3.5 (breakaway) and M4.5 (uprising)
machinery turns that into consequences. The crown now answers to the altar.

HEADLINE 1 — FAITH EMERGES FROM SHARED BELIEF: a coherent shared belief set crystallises a
             named faith; a fractured town forms none; two towns with the same core share
             one faith, divergent cores form two. Emergent from M4.7, not declared.
HEADLINE 2 — THE PROPHET IS DERIVED: the most devout-and-trusted follower becomes prophet
             (not the richest, not assigned); a faith with no such figure has none.
HEADLINE 3 — FAITH IS LEGITIMACY (the teeth): the SAME king doing the SAME extraction is
             HELD when aligned with the town's faith and LOST to BREAKAWAY when defiant —
             his believing vassal's loyalty eroded by faith through the existing M3.5
             machinery, a prophet deepening it. Same ruler, two alignments, two fates.
COMPOSE    — a conqueror imposing on a HOSTILE-FAITH town (M4.7's conquered "the strong take
             what they want" belief) is harder to hold than a co-faith lord would be.
COST       — zero added LLM; --religion off byte-identical; deterministic/reproducible.
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from sim import beliefs as B
from sim import kingdoms
from llm import llm
import main
from sim import religion
from sim import discontent
from sim import trust
from sim import world
from sim.agents import Agent
from sim.world import world_state

CORE = {B.LAND_PROVIDES, B.STRONGER_TOGETHER}


# --- Staging helpers ---------------------------------------------------------
def _fresh() -> None:
    world.create_world()
    for f in ("beliefs_on", "religion_on", "discontent_on"):
        world_state[f] = True


def _settlement(sid, center) -> None:
    world_state["settlements"][sid] = {"id": sid, "center": center, "members": set(), "founded": 0}


def _agent(name, pos, sid="S001", *, believes=None, money=0.0) -> Agent:
    a = Agent(name=name, personality="friendly and outgoing")
    world.place_agent(a, *pos)
    a.hunger, a.age, a.lifespan, a.money, a.settlement = 1, 30, 100, money, sid
    if sid is not None and sid in world_state["settlements"]:
        world_state["settlements"][sid]["members"].add(name)
    if believes:
        world_state.setdefault("beliefs", {})[name] = set(believes)
    return a


def _find(name) -> Any:
    return next(a for a in world_state["agents"] if a.name == name)


# --- HEADLINE 1: faith emerges from shared belief ----------------------------
def headline_1_faith_emerges() -> None:
    print("=" * 72)
    print("HEADLINE 1 — FAITH EMERGES FROM SHARED BELIEF (coherence crystallises; fracture does not)")
    print("=" * 72)

    _fresh(); _settlement("S001", (5, 5))
    for n, p in [("A", (5, 5)), ("Bb", (5, 6)), ("C", (6, 5)), ("D", (6, 6))]:
        _agent(n, p, believes=CORE)
    religion.form_faiths(world_state, 1)
    f = religion.faith_of_settlement(world_state, "S001")
    print(f"  coherent town (4 share {sorted(CORE)}) -> {f['name']} (followers {len(f['followers'])})")
    assert f is not None and f["core"] == frozenset(CORE)

    _fresh(); _settlement("S001", (5, 5))
    _agent("A", (5, 5), believes={B.LAND_PROVIDES})
    _agent("Bb", (5, 6), believes={B.WORLD_IS_CRUEL})
    _agent("C", (6, 5), believes={B.WEALTH_IS_VIRTUE})
    religion.form_faiths(world_state, 1)
    print(f"  fractured town (all differ)            -> {religion.faith_of_settlement(world_state, 'S001')}")
    assert religion.faith_of_settlement(world_state, "S001") is None

    _fresh(); _settlement("S001", (2, 2)); _settlement("S002", (8, 8))
    for n, p in [("A", (2, 2)), ("Bb", (2, 3)), ("C", (3, 2))]:
        _agent(n, p, "S001", believes=CORE)
    for n, p in [("E", (8, 8)), ("F", (8, 9)), ("G", (9, 8))]:
        _agent(n, p, "S002", believes=CORE)
    religion.form_faiths(world_state, 1)
    print(f"  two towns, SAME core                   -> {len(world_state['faiths'])} faith (shared)")
    assert len(world_state["faiths"]) == 1
    for n in ("E", "F", "G"):
        world_state["beliefs"][n] = {B.WORLD_IS_CRUEL, B.STRONG_TAKE}
    religion.form_faiths(world_state, 2)
    print(f"  two towns, DIVERGENT cores             -> {len(world_state['faiths'])} faiths (distinct)")
    assert len(world_state["faiths"]) == 2
    print("  -> faith is emergent from what a town's populace shares, not declared.")
    print()


# --- HEADLINE 2: the prophet is derived --------------------------------------
def headline_2_prophet_is_derived() -> None:
    print("=" * 72)
    print("HEADLINE 2 — THE PROPHET IS DERIVED (most devout-and-trusted, not richest, not assigned)")
    print("=" * 72)

    _fresh(); _settlement("S001", (5, 5))
    devout = [_agent(n, p, believes=CORE) for n, p in
              [("Pa", (5, 5)), ("Pb", (5, 6)), ("Pc", (6, 5)), ("Pd", (6, 6))]]
    _agent("Croesus", (6, 6), believes=CORE, money=500.0)   # the richest follower — but not trusted
    for a in devout:
        if a.name != "Pb":
            trust.ensure_relationship(a, "Pb")["trust"] = 3   # the flock trusts Pb
    religion.form_faiths(world_state, 1)
    religion.choose_prophets(world_state, 1)
    prophet = religion.faith_of_settlement(world_state, "S001")["prophet"]
    print(f"  flock trusts Pb (Croesus is richest): prophet = {prophet}")
    assert prophet == "Pb"

    _fresh(); _settlement("S001", (5, 5))
    for n, p in [("A", (5, 5)), ("Bb", (5, 6)), ("C", (6, 5))]:
        _agent(n, p, believes=CORE)
    religion.form_faiths(world_state, 1)
    religion.choose_prophets(world_state, 1)
    print(f"  a flock that trusts no one enough:    prophet = "
          f"{religion.faith_of_settlement(world_state, 'S001')['prophet']}")
    assert religion.faith_of_settlement(world_state, "S001")["prophet"] is None
    print("  -> the prophet arises from DEVOTION + the believers' own TRUST; none is forced.")
    print()


# --- HEADLINE 3: faith is legitimacy (the teeth) -----------------------------
def _realm(king_beliefs, with_prophet, town_core=None) -> None:
    """A king ruling a vassal town S002 whose flock holds `town_core` (default CORE)."""
    core = town_core or CORE
    _fresh(); _settlement("S001", (1, 1)); _settlement("S002", (8, 8))
    world_state["tribute_rate"] = 0.25   # <= KING_CONSENT, so tribute itself adds no loyalty backlash
    _agent("King", (1, 1), "S001", believes=king_beliefs, money=50.0)
    _agent("Lord", (8, 8), "S002", believes=core, money=8.0)
    flock = [_agent(n, p, "S002", believes=core, money=8.0) for n, p in [("Va", (8, 7)), ("Vb", (7, 8))]]
    world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
    world_state["monarchs"]["S002"] = {"monarch": "Lord", "since": 0, "garrison": set()}
    world_state["kingdoms"]["King"] = {"king": "King", "home": "S001",
                                       "settlements": {"S001", "S002"}, "vassals": {"S002": "Lord"},
                                       "founded": 0, "discontent": {"Lord": 0}}
    trust.ensure_relationship(_find("Lord"), "King")["trust"] = 2   # loyal fealty at the start
    if with_prophet:
        for a in [_find("Lord")] + flock:
            if a.name != "Va":
                trust.ensure_relationship(a, "Va")["trust"] = 3


def headline_3_faith_is_legitimacy() -> None:
    print("=" * 72)
    print("HEADLINE 3 — FAITH IS LEGITIMACY (aligned king HELD, defiant king LOST to breakaway)")
    print("=" * 72)

    def run_realm(king_beliefs, with_prophet) -> tuple[bool, int]:
        _realm(king_beliefs, with_prophet)
        for t in range(1, 9):
            religion.update(world_state, t)      # faith moves the loyalty dial
            kingdoms.update(world_state, t)       # the UNCHANGED M3.5 machinery reads it
        held = kingdoms.realm_of(world_state, "S002") == "King"
        return held, _find("Lord").relationships["King"]["trust"]

    held_aligned, trust_aligned = run_realm(CORE, with_prophet=False)             # a co-faith king
    held_defiant, trust_defiant = run_realm({B.STRONG_TAKE}, with_prophet=False)  # a defiant king
    held_prophet, trust_prophet = run_realm({B.STRONG_TAKE}, with_prophet=True)   # + an opposing prophet

    print(f"  ALIGNED king (shares the faith):   vassal keeps loyalty {trust_aligned:+d} -> realm held? {held_aligned}")
    print(f"  DEFIANT king (rejects the faith):  vassal loyalty {trust_defiant:+d} -> realm held? {held_defiant}")
    print(f"  DEFIANT king + opposing PROPHET:   vassal loyalty {trust_prophet:+d} -> realm held? {held_prophet}")
    assert held_aligned and not held_defiant
    assert trust_defiant < trust_aligned and trust_prophet <= trust_defiant
    print("  -> the SAME crown, doing the SAME thing, is held by a faith it shares and broken by one")
    print("     it defies — through the existing breakaway machinery; a prophet deepens the rupture.")
    print()

    # The discontent side of the same coin: an aligned monarch's town seethes less than a defiant one's.
    def town_discontent(monarch_aligned) -> tuple[int, float]:
        _fresh(); _settlement("S001", (5, 5))
        _agent("King", (4, 4), believes=(CORE if monarch_aligned else {B.STRONG_TAKE}), money=100.0)
        members = [_agent(n, p, believes=CORE, money=20.0) for n, p in [("A", (5, 5)), ("Bb", (5, 6)), ("C", (6, 5))]]
        world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
        for t in range(1, 8):
            religion.update(world_state, t)
            discontent.update(world_state, t)
        return members[0].relationships.get("King", {}).get("trust", 0), discontent.agent_discontent("A", world_state)

    ta, da = town_discontent(True)
    tu, du = town_discontent(False)
    print(f"  SAME levy — ALIGNED monarch: member loyalty {ta:+d}, discontent {da:.1f}")
    print(f"  SAME levy — DEFIANT monarch: member loyalty {tu:+d}, discontent {du:.1f}")
    assert ta > tu and da < du
    print("  -> faith buffers the discontent of an aligned crown and sharpens it against a defiant one.")
    print()


# --- COMPOSE: a hostile-faith conquered town is harder to hold ---------------
def compose_hostile_faith_town() -> None:
    print("=" * 72)
    print("COMPOSE — a conqueror imposing on a HOSTILE-FAITH town is harder to hold (M4.7 x M4.8)")
    print("=" * 72)

    # A conquered town believes (per M4.7) "the world is cruel / the strong take what they want".
    conquered_core = {B.WORLD_IS_CRUEL, B.STRONG_TAKE}

    def outcome(king_shares_faith) -> bool:
        king_beliefs = conquered_core if king_shares_faith else CORE
        _realm(king_beliefs, with_prophet=False, town_core=conquered_core)
        for t in range(1, 9):
            religion.update(world_state, t)
            kingdoms.update(world_state, t)
        return kingdoms.realm_of(world_state, "S002") == "King"

    held_cofaith = outcome(king_shares_faith=True)
    held_alien = outcome(king_shares_faith=False)
    print(f"  a king who SHARES the conquered town's grim faith holds it? {held_cofaith}")
    print(f"  a king ALIEN to that faith holds it?                        {held_alien}")
    assert held_cofaith and not held_alien
    print("  -> the belief a town formed by SUFFERING (M4.7) now decides who can rule it (M4.8):")
    print("     a conqueror alien to the local faith cannot hold what a co-faith lord would.")
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
            main.run_simulation(30, settlements=True, monarchy_on=True, discontent_on=True,
                                beliefs_on=True, **kw)
        return buf.getvalue(), dict(llm.get_call_stats())

    off, off_calls = run()
    off2, _ = run(religion_on=False)
    assert off == off2
    print("  --religion OFF: byte-identical to the beliefs-only run")
    on_a, on_calls = run(religion_on=True)
    on_b, _ = run(religion_on=True)
    assert on_a == on_b
    print("  --religion ON: two seeded runs byte-identical (faith is deterministic state math)")
    assert on_calls == off_calls
    print(f"  religion added ZERO LLM calls (on={on_calls}, off={off_calls}).")
    print()


if __name__ == "__main__":
    saved = llm.PROVIDER
    try:
        headline_1_faith_emerges()
        headline_2_prophet_is_derived()
        headline_3_faith_is_legitimacy()
        compose_hostile_faith_town()
        cost_checks()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M4.8 VERIFIED — faith genuinely EMERGES from shared belief, the PROPHET arises from")
    print("devotion-and-trust (not assignment), and faith is a real LEGITIMACY force: aligned")
    print("rulers are held and defiant ones undermined through the existing trust/discontent/")
    print("loyalty machinery. The crown answers to the altar — and M4.9 will set cultures at odds.")
    print("=" * 72)
