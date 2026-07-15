"""
verify_m49.py
=============

Deterministic verification of V2 milestone M4.9: CULTURAL IDENTITY & FRICTION —
the imperial problem. CLOSES Arc 3 (Belief & Culture), on top of M4.8 (religion),
M4.7 (beliefs), Arc 2 (discontent/uprising), Arc 1 (lineage) and Phases 0-3.

Run offline (Ollama OFF, no model server, no seed-search):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m49.py

The historical step: you can seize a people in a day but not make them yours for
decades. A settlement's CULTURE is its dominant belief set (M4.7) + faith (M4.8). A
ruler of the SAME culture integrates almost frictionlessly; a ruler ALIEN to the local
culture faces a province that resents him every turn (SUSTAINED loyalty tax), is
likelier to break away or rise — and becomes his only as its CHILDREN slowly grow up in
his culture while the adults keep theirs. Assimilation is a RACE against revolt. M4.9
adds NO new political mechanic: cultural revolt EMERGES from the existing M4.4/M3.5/M4.5
machinery — culture just sustains the trust/discontent dials those systems already read.

HEADLINE 1 — SAME VS FOREIGN CONQUEST (the teeth): conquering a SAME-culture province
             integrates with little extra discontent and HOLDS; conquering a FOREIGN one
             breeds CHRONIC discontent + low loyalty and fractures via the existing
             breakaway machinery. Same conqueror, two cultures, two fates.
HEADLINE 2 — ASSIMILATION TAKES GENERATIONS: under sustained foreign rule the town's
             culture DRIFTS toward the ruler's only as its children assimilate (adults do
             not) — negligible over a few turns, complete over many, at which point the
             fault line FADES. The RACE: a child-rich town assimilates (fault heals) while
             a childless one BREAKS AWAY first (fault fractures). Both outcomes, neither scripted.
HEADLINE 3 — THE MULTI-CULTURAL EMPIRE FRAGMENTS ALONG CULTURAL LINES: a realm spanning
             several cultures loses its FOREIGN provinces to breakaway while its same-culture
             core holds — it fragments along cultural fault lines, not randomly.
COST       — zero added LLM; --culture off byte-identical; deterministic/reproducible.
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any

import beliefs as B
import culture
import discontent
import kingdoms
import llm
import main
import trust
import world
from agents import Agent
from world import world_state

NATIVE = {B.LAND_PROVIDES, B.STRONGER_TOGETHER}       # a prospering town's culture (M4.7)
FOREIGN = {B.STRONG_TAKE, B.WEALTH_IS_VIRTUE}         # an alien conqueror's culture


# --- Staging helpers ---------------------------------------------------------
def _fresh(lineage=False) -> None:
    world.create_world()
    for f in ("beliefs_on", "religion_on", "culture_on", "discontent_on"):
        world_state[f] = True
    world_state["lineage_on"] = lineage


def _settlement(sid, center) -> None:
    world_state["settlements"][sid] = {"id": sid, "center": center, "members": set(), "founded": 0}


def _agent(name, pos, sid="S001", *, believes=None, money=0.0, dependent=False, age=30, parents=()) -> Agent:
    a = Agent(name=name, personality="friendly and outgoing")
    world.place_agent(a, *pos)
    a.hunger, a.age, a.lifespan, a.money = 1, age, 100, money
    a.settlement, a.dependent, a.parents = sid, dependent, parents
    if sid is not None and sid in world_state["settlements"]:
        world_state["settlements"][sid]["members"].add(name)
    if believes is not None:
        world_state.setdefault("beliefs", {})[name] = set(believes)
    return a


def _find(name) -> Any:
    return next(a for a in world_state["agents"] if a.name == name)


def _crown(sid, name) -> None:
    world_state["monarchs"][sid] = {"monarch": name, "since": 0, "garrison": set()}


# --- HEADLINE 1: same vs foreign conquest ------------------------------------
def headline_1_same_vs_foreign() -> None:
    print("=" * 72)
    print("HEADLINE 1 — SAME VS FOREIGN CONQUEST (chronic friction; the province fractures)")
    print("=" * 72)

    def town_discontent(foreign_king):
        _fresh(); _settlement("S001", (5, 5))
        _agent("King", (4, 4), believes=(FOREIGN if foreign_king else NATIVE), money=100.0)
        members = [_agent(n, p, believes=NATIVE, money=20.0)
                   for n, p in [("A", (5, 5)), ("Bb", (5, 6)), ("C", (6, 5))]]
        _crown("S001", "King")
        for t in range(1, 8):
            culture.update(world_state, t)
            discontent.update(world_state, t)
        return members[0].relationships.get("King", {}).get("trust", 0), discontent.agent_discontent("A", world_state)

    ts, ds = town_discontent(False)
    tf, df = town_discontent(True)
    print(f"  SAME-culture conqueror:    member loyalty {ts:+d}, discontent {ds:.1f}")
    print(f"  FOREIGN-culture conqueror: member loyalty {tf:+d}, discontent {df:.1f}   (chronic, every turn)")
    assert tf < ts and df > ds

    def realm_holds(foreign_king):
        _fresh(); _settlement("S001", (1, 1)); _settlement("S002", (8, 8))
        world_state["tribute_rate"] = 0.25
        _agent("King", (1, 1), "S001", believes=(FOREIGN if foreign_king else NATIVE), money=50.0)
        lord = _agent("Lord", (8, 8), "S002", believes=NATIVE, money=8.0)
        for n, p in [("Va", (8, 7)), ("Vb", (7, 8))]:
            _agent(n, p, "S002", believes=NATIVE)
        _crown("S001", "King"); _crown("S002", "Lord")
        world_state["kingdoms"]["King"] = {"king": "King", "home": "S001",
                                           "settlements": {"S001", "S002"}, "vassals": {"S002": "Lord"},
                                           "founded": 0, "discontent": {"Lord": 0}}
        trust.ensure_relationship(lord, "King")["trust"] = 2
        for t in range(1, 9):
            culture.update(world_state, t)
            kingdoms.update(world_state, t)
        return kingdoms.realm_of(world_state, "S002") == "King"

    held_same = realm_holds(False)
    held_foreign = realm_holds(True)
    print(f"\n  a SAME-culture king holds his conquered province?    {held_same}")
    print(f"  a FOREIGN-culture king holds his conquered province? {held_foreign}")
    assert held_same and not held_foreign
    print("  -> same conqueror, two cultures, two fates: the foreign province fractures where the")
    print("     same-culture one holds — chronic friction feeding the existing breakaway machinery.")
    print()


# --- HEADLINE 2: assimilation takes generations ------------------------------
def headline_2_assimilation_takes_generations() -> None:
    print("=" * 72)
    print("HEADLINE 2 — ASSIMILATION TAKES GENERATIONS (children drift; the race with revolt)")
    print("=" * 72)

    # A child-rich foreign-ruled town: its children slowly take on the ruler's culture.
    _fresh(lineage=True); _settlement("S001", (5, 5))
    _agent("King", (4, 4), believes=FOREIGN, money=100.0)
    _agent("Elder", (5, 5), believes=set(NATIVE))
    for n, p in [("K1", (5, 6)), ("K2", (6, 5)), ("K3", (6, 6))]:
        _agent(n, p, believes=set(NATIVE), dependent=True, age=6, parents=("Elder", "X"))
    _crown("S001", "King")
    rng = random.Random(2)
    culture.update(world_state, 1, rng)
    few = culture.assimilation_progress(world_state, "S001")
    for t in range(2, 60):
        culture.update(world_state, t, rng)
    many = culture.assimilation_progress(world_state, "S001")
    print(f"  a foreign-ruled town's assimilation: after 1 turn {few*100:.0f}%  ->  after ~60 turns {many*100:.0f}%")
    print(f"    the child K1's beliefs are now {sorted(world_state['beliefs']['K1'])}")
    print(f"    the adult Elder's are still    {sorted(world_state['beliefs']['Elder'])}")
    assert many > few and world_state["beliefs"]["Elder"] == NATIVE
    faded = not culture.is_foreign_ruled(world_state, "S001")
    print(f"    with the generation assimilated, is the ruler still foreign? {not faded}  (fault line faded: {faded})")
    assert faded
    print("  -> ASSIMILATION won this race: the young grew up in the ruler's culture, the town's")
    print("     signature drifted, and the fault line healed — over generations, not turns.")

    # The other outcome of the race: a childless foreign province BREAKS AWAY before it can assimilate.
    _fresh(); _settlement("S001", (1, 1)); _settlement("S002", (8, 8))
    world_state["tribute_rate"] = 0.25
    _agent("King", (1, 1), "S001", believes=FOREIGN, money=50.0)
    lord = _agent("Lord", (8, 8), "S002", believes=NATIVE, money=8.0)
    for n, p in [("Va", (8, 7)), ("Vb", (7, 8))]:
        _agent(n, p, "S002", believes=NATIVE)     # all adults — no one to assimilate
    _crown("S001", "King"); _crown("S002", "Lord")
    world_state["kingdoms"]["King"] = {"king": "King", "home": "S001",
                                       "settlements": {"S001", "S002"}, "vassals": {"S002": "Lord"},
                                       "founded": 0, "discontent": {"Lord": 0}}
    trust.ensure_relationship(lord, "King")["trust"] = 2
    for t in range(1, 9):
        culture.update(world_state, t)
        kingdoms.update(world_state, t)
    broke = kingdoms.realm_of(world_state, "S002") is None
    print(f"\n  a childless foreign province (no one to assimilate): broke away first? {broke}")
    assert broke
    print("  -> REVOLT won this race: with no children to assimilate, the fault line fractured the")
    print("     realm before the culture could ever drift. Both outcomes, neither scripted.")
    print()


# --- HEADLINE 3: the multi-cultural empire fragments along cultural lines -----
def headline_3_empire_fragments_along_culture() -> None:
    print("=" * 72)
    print("HEADLINE 3 — A MULTI-CULTURAL REALM FRAGMENTS ALONG CULTURAL LINES (not randomly)")
    print("=" * 72)

    # A NATIVE-culture king rules four provinces: two of his own culture (core), two foreign (conquered).
    _fresh(); world_state["tribute_rate"] = 0.25
    homes = {"S001": (1, 1), "S002": (1, 3), "S003": (8, 8), "S004": (8, 6)}
    for sid, c in homes.items():
        _settlement(sid, c)
    # The king holds a static realm (no war chest -> the organic re-conquest loop stays out; we are
    # isolating the CULTURAL breakaway, not re-expansion).
    _agent("King", (1, 1), "S001", believes=NATIVE, money=1.0)
    # Core provinces: the king's own culture.
    _agent("L2", (1, 3), "S002", believes=NATIVE, money=8.0)
    for n, p in [("c2a", (2, 3)), ("c2b", (1, 2))]:
        _agent(n, p, "S002", believes=NATIVE)
    # Foreign provinces: a divergent culture (recall M4.7 — a suffering/conquered town).
    _agent("L3", (8, 8), "S003", believes=FOREIGN, money=8.0)
    for n, p in [("c3a", (8, 7)), ("c3b", (7, 8))]:
        _agent(n, p, "S003", believes=FOREIGN)
    _agent("L4", (8, 6), "S004", believes=FOREIGN, money=8.0)
    for n, p in [("c4a", (8, 5)), ("c4b", (7, 6))]:
        _agent(n, p, "S004", believes=FOREIGN)
    for sid in homes:
        _crown(sid, {"S001": "King", "S002": "L2", "S003": "L3", "S004": "L4"}[sid])
    world_state["kingdoms"]["King"] = {
        "king": "King", "home": "S001",
        "settlements": set(homes), "vassals": {"S002": "L2", "S003": "L3", "S004": "L4"},
        "founded": 0, "discontent": {"L2": 0, "L3": 0, "L4": 0}}
    for lord in ("L2", "L3", "L4"):
        trust.ensure_relationship(_find(lord), "King")["trust"] = 2   # all start loyal

    print("  a realm of four provinces under a NATIVE-culture king:")
    print("    S002 = same culture (core) ; S003, S004 = foreign (conquered, divergent beliefs)")
    for t in range(1, 11):
        culture.update(world_state, t)
        kingdoms.update(world_state, t)

    still = {sid: kingdoms.realm_of(world_state, sid) == "King" for sid in ("S002", "S003", "S004")}
    print(f"\n  after 10 turns — still in the realm?  S002(core)={still['S002']}  "
          f"S003(foreign)={still['S003']}  S004(foreign)={still['S004']}")
    assert still["S002"] and not still["S003"] and not still["S004"]
    print("  -> the realm fragmented along its CULTURAL fault lines: both foreign provinces broke")
    print("     away, the same-culture core held. Empires of many cultures split where they differ.")
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
                                beliefs_on=True, religion_on=True, **kw)
        return buf.getvalue(), dict(llm.get_call_stats())

    off, off_calls = run()
    off2, _ = run(culture_on=False)
    assert off == off2
    print("  --culture OFF: byte-identical to the religion-only run")
    on_a, on_calls = run(culture_on=True)
    on_b, _ = run(culture_on=True)
    assert on_a == on_b
    print("  --culture ON: two seeded runs byte-identical (friction deterministic, assimilation seeded)")
    assert on_calls == off_calls
    print(f"  culture added ZERO LLM calls (on={on_calls}, off={off_calls}).")
    print()


if __name__ == "__main__":
    saved = llm.PROVIDER
    try:
        headline_1_same_vs_foreign()
        headline_2_assimilation_takes_generations()
        headline_3_empire_fragments_along_culture()
        cost_checks()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M4.9 VERIFIED — foreign conquest genuinely breeds chronic unrest a same-culture conquest")
    print("avoids; assimilation truly takes GENERATIONS (via children, not turns); and multi-cultural")
    print("empires fragment along CULTURAL fault lines. Arc 3 closes: the crown learns that")
    print("conquering a people and keeping them are different things.")
    print("=" * 72)
