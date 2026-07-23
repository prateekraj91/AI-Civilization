"""
verify_m43.py
=============

Deterministic verification of V2 milestone M4.3: DYNASTIES — TITLES PASS TO HEIRS.
The CLOSE of Arc 1 (Generations & Dynasties), on top of M4.2 (inheritance of
wealth), M4.1 (birth, childhood, aging) and all of Phases 0-3.

Run offline (Ollama OFF, no model server, no seed-search):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m43.py

The historical step: through M4.2 a dead ruler's WEALTH passed to kin but its CROWN
EVAPORATED — a monarch's realm dissolved the moment he died of old age. M4.3 makes
TITLES dynastic: on ANY death (one shared hook, population.announce_death) the SEAT
passes IMPARTIBLY by primogeniture to the SINGLE eldest heir (the same M4.2 kin-order),
while ALL children still split the gold (M4.2 untouched). The heir inherits the SEAT,
not the LOYALTY — trust is personal and is never copied — so an unknown heir's realm
erodes and BREAKS AWAY through the EXISTING M3.5/M3.6 machinery. Succession is a CRISIS.

HEADLINE 1 — THE CROWN SURVIVES DEATH: a monarch with an heir dies of old age; the
             eldest child is CROWNED (records transferred, coronation logged) and the
             realm (kingdom, vassals, tribute) continues under the heir. CONTRAST: the
             same death pre-M4.3 dissolved the realm. Wealth STILL splits equally among
             ALL children while only the eldest gets the crown — both on one death.
HEADLINE 2 — SUCCESSION IS A CRISIS TEST (loyalty not inherited): a beloved king dies;
             a COLD/unknown heir's vassal loyalty erodes and it BREAKS AWAY via the
             existing machinery; the SAME succession with a TRUSTED heir HOLDS. Same
             realm, two heirs, two fates — dynasties survive on the heir's own standing.
HEADLINE 3 — EXTINCT LINES: a kinless king dies; "line extinguished" logged; the realm
             dissolves into independent settlements; a later aspirant seizes the vacant
             seat by the ordinary conquest machinery.
ORDER      — eldest child preferred; each fallback tier fires only when the closer is
             empty; deterministic age-then-name tiebreak; the child-monarch (regency)
             documented.
MULTILEVEL — an EMPEROR's death passes the imperial throne; a subject-king's death
             passes his subordinate seat — same rules, one level up.
COMPOSE    — zero added LLM; --lineage off byte-identical; deterministic/reproducible;
             M4.2 wealth split unchanged on the same deaths.
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from sim import kingdoms
from sim import lineage
from llm import llm
import main
from sim import monarchy
from sim import population
from sim import trust
from sim import world
from sim.agents import Agent
from sim.world import world_state


# --- Staging helpers ---------------------------------------------------------
def _fresh(pop_cap: int = 12) -> None:
    """A clean lineage-on world with one settlement S001 centred at (5, 5)."""
    world.create_world()
    world_state["lineage_on"] = True
    world_state["lineage"] = {"pop_cap": pop_cap, "birth_seq": 0}
    world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5),
                                          "members": set(), "founded": 0}


def _settlement(sid: str, center: tuple[int, int]) -> None:
    world_state["settlements"][sid] = {"id": sid, "center": center,
                                       "members": set(), "founded": 0}


def _agent(name: str, pos: tuple[int, int], *, money: float = 0.0, stockpile: float = 0.0,
           parents: tuple = (), sid: "str | None" = "S001", dependent: bool = False,
           age: int = 30) -> Agent:
    a = Agent(name=name, personality="friendly and outgoing")
    world.place_agent(a, *pos)
    a.hunger, a.age, a.lifespan = 1, age, 100
    a.money, a.stockpile, a.parents, a.dependent = money, stockpile, parents, dependent
    a.settlement = sid
    if sid is not None and sid in world_state["settlements"]:
        world_state["settlements"][sid]["members"].add(name)
    return a


def _crown(sid: str, monarch: str, garrison: set | None = None) -> None:
    world_state["monarchs"][sid] = {"monarch": monarch, "since": 0,
                                    "garrison": set(garrison or set())}


def _realm(king: str, home: str, settlements: set, vassals: dict) -> None:
    world_state["kingdoms"][king] = {
        "king": king, "home": home, "settlements": set(settlements),
        "vassals": dict(vassals), "founded": 0,
        "discontent": {lord: 0 for lord in vassals.values()}}


# --- HEADLINE 1: the crown survives death ------------------------------------
def headline_1_crown_survives_death() -> None:
    print("=" * 72)
    print("HEADLINE 1 — THE CROWN SURVIVES DEATH (the seat passes; the realm continues)")
    print("=" * 72)

    def build() -> Agent:
        _fresh()
        _settlement("S002", (8, 8))
        king = _agent("Wren", (5, 5), money=40.0, stockpile=8.0, age=62)
        _agent("Aldo", (5, 6), parents=("Wren", "Isla"), age=28)   # ELDEST child
        _agent("Bess", (6, 5), parents=("Wren", "Isla"), age=21)   # younger child
        _agent("Vale", (8, 8), sid="S002")                          # a vassal lord
        _crown("S001", "Wren", {"g1", "g2"})
        _realm("Wren", "S001", {"S001", "S002"}, {"S002": "Vale"})
        return king

    # Pre-M4.3 contrast: with succession suppressed, the dead king's realm is left to
    # dissolve (his crown record is inert; his vassals break from a leaderless king).
    king = build()
    real = lineage.succeed_titles
    lineage.succeed_titles = lambda *a, **k: {"heir": None, "kind": "none", "titles": ""}
    try:
        population.announce_death(king, 80, world_state, cause="old age",
                                  final_memory="Died of old age", note="they died of old age")
    finally:
        lineage.succeed_titles = real
    for t in range(81, 85):
        kingdoms.update(world_state, t)
    pre_holds = kingdoms.realm_of(world_state, "S002") is not None
    print(f"  PRE-M4.3 (succession OFF): King Wren dies of old age -> realm dissolves; "
          f"vassal S002 still in a realm? {pre_holds}")
    assert not pre_holds, "the pre-M4.3 baseline must let the realm dissolve"

    # M4.3: the eldest child is crowned; the realm continues; wealth STILL splits equally.
    king = build()
    aldo = next(a for a in world_state["agents"] if a.name == "Aldo")
    bess = next(a for a in world_state["agents"] if a.name == "Bess")
    print(f"\n  M4.3: King Wren (holding {king.money + king.stockpile:.0f}) dies of old age.")
    print(f"        heirs: Aldo (age {aldo.age}), Bess (age {bess.age}); vassal Vale of S002.")
    population.announce_death(king, 80, world_state, cause="old age",
                              final_memory="Died of old age", note="they died of old age")
    coron = next(e for e in world_state["events"] if "succeeded" in e)
    print(f"    coronation: {coron.split(': ', 1)[1]}")
    assert world_state["monarchs"]["S001"]["monarch"] == "Aldo"
    cyn = world_state["kingdoms"]["Aldo"]
    assert "Wren" not in world_state["kingdoms"]
    assert cyn["settlements"] == {"S001", "S002"} and cyn["vassals"] == {"S002": "Vale"}
    print(f"    the SEAT transferred: monarch of S001 = Aldo; king of the realm "
          f"{sorted(cyn['settlements'])}; vassal lordship of S002 intact under Aldo.")
    # The realm still functions: tribute flows and the structure persists under the heir.
    for t in range(81, 84):
        kingdoms.update(world_state, t)
    assert kingdoms.realm_of(world_state, "S002") == "Aldo", "the realm continues under the heir"
    print(f"    the realm CONTINUES under Aldo (S002 still a vassal after {3} turns).")
    # M4.2 UNTOUCHED: wealth splits EQUALLY among ALL children; only the eldest is crowned.
    print(f"\n    meanwhile the GOLD split EQUALLY (M4.2 untouched): "
          f"Aldo {aldo.money + aldo.stockpile:.1f}, Bess {bess.money + bess.stockpile:.1f}")
    assert aldo.money + aldo.stockpile == bess.money + bess.stockpile == 24.0
    assert not any("Wren succeeded" in e for e in world_state["events"])
    print("    -> ONE death: the eldest gets the CROWN (impartible); ALL children split "
          "the GOLD (partible).")
    print()


# --- HEADLINE 2: succession is a crisis test (loyalty not inherited) ----------
def headline_2_succession_is_a_crisis() -> None:
    print("=" * 72)
    print("HEADLINE 2 — SUCCESSION IS A CRISIS TEST (the heir inherits the seat, not loyalty)")
    print("=" * 72)

    def succeed_with(heir_trust: int) -> tuple[bool, Any]:
        _fresh()
        _settlement("S002", (8, 8))
        king = _agent("Wren", (5, 5), age=62)
        _agent("Aldo", (5, 6), parents=("Wren", "Isla"), age=28)   # the heir
        vassal = _agent("Vale", (8, 8), sid="S002")
        _crown("S001", "Wren")
        _realm("Wren", "S001", {"S001", "S002"}, {"S002": "Vale"})
        # The vassal was loyal to Wren (fealty), and has a PERSONAL standing with the heir.
        trust.ensure_relationship(vassal, "Wren")["trust"] = 3
        trust.ensure_relationship(vassal, "Aldo")["trust"] = heir_trust
        population.announce_death(king, 80, world_state, cause="old age",
                                  final_memory="Died of old age", note="they died of old age")
        # Loyalty is NEVER copied — the heir stands on its own standing.
        assert vassal.relationships["Aldo"]["trust"] == heir_trust
        for t in range(81, 86):       # let the ordinary M3.5 loyalty machinery run
            kingdoms.update(world_state, t)
        held = kingdoms.realm_of(world_state, "S002") == "Aldo"
        return held, vassal

    held_cold, vc = succeed_with(heir_trust=-3)   # an UNKNOWN / distrusted heir
    print(f"  the same beloved King Wren dies; his heir Aldo is a COLD start to the vassal")
    print(f"  (Vale's personal trust in Aldo = {vc.relationships['Aldo']['trust']}, "
          f"NOT the +3 it held for Wren — loyalty is not inherited).")
    print(f"    after 5 turns of the existing machinery: vassal still in Aldo's realm? {held_cold}")
    assert not held_cold, "a cold/distrusted heir must LOSE the vassal via breakaway"
    assert any("BROKE AWAY" in e for e in world_state["events"])
    print(f"    -> {next(e for e in world_state['events'] if 'BROKE AWAY' in e).split(': ', 1)[1]}")

    held_warm, vw = succeed_with(heir_trust=3)    # a KNOWN / trusted heir
    print(f"\n  the SAME succession, but Aldo already has the vassal's PERSONAL trust "
          f"(+{vw.relationships['Aldo']['trust']}):")
    print(f"    after 5 turns: vassal still in Aldo's realm? {held_warm}")
    assert held_warm, "a trusted heir must HOLD the realm"
    print("  -> Same realm, two heirs, two fates: dynasties survive on the HEIR'S OWN")
    print("     standing, not the father's ghost. Succession is a real crisis.")
    print()


# --- HEADLINE 3: extinct lines dissolve into contestable power ----------------
def headline_3_extinct_lines() -> None:
    print("=" * 72)
    print("HEADLINE 3 — EXTINCT LINES (a kinless crown falls vacant, then is re-contested)")
    print("=" * 72)

    _fresh()
    _settlement("S002", (8, 8))
    king = _agent("Wren", (5, 5), age=64)          # NO kin at all
    _agent("Vale", (8, 8), sid="S002")
    _crown("S001", "Wren")
    _realm("Wren", "S001", {"S001", "S002"}, {"S002": "Vale"})
    print("  kinless King Wren dies of old age (no child, parent, or sibling alive).")
    population.announce_death(king, 90, world_state, cause="old age",
                              final_memory="Died of old age", note="they died of old age")
    ext = next(e for e in world_state["events"] if "extinguished" in e)
    print(f"    logged distinctly: {ext.split(': ', 1)[1]}")
    assert "the line of Wren is extinguished" in ext and "lies vacant" in ext
    # The realm dissolves into independent settlements via the existing machinery.
    for t in range(91, 95):
        kingdoms.update(world_state, t)
    assert kingdoms.realm_of(world_state, "S002") is None
    print("    the realm dissolves: vassal settlement S002 is independent again.")
    # The vacant home seat is contestable by ordinary conquest — an aspirant seizes it.
    aspirant = _agent("Zara", (5, 4), money=30.0, sid=None)
    _agent("Merc", (5, 3), money=0.0, sid=None)    # a poor fighter in muster range
    res = monarchy.attempt_conquest(world_state, aspirant, "S001", 96)
    print(f"    a later aspirant Zara musters and seizes the vacant seat of S001: "
          f"won={res['won']} -> monarch of S001 = {world_state['monarchs']['S001']['monarch']}")
    assert res["won"] and world_state["monarchs"]["S001"]["monarch"] == "Zara"
    print("  -> extinct lines dissolve cleanly into contestable power, fought over by the")
    print("     ordinary organic conquest machinery — no scripted succession war.")
    print()


# --- ORDER: eldest-first, tiers bind, child-monarch regency ------------------
def order_and_regency() -> None:
    print("=" * 72)
    print("SUCCESSION ORDER — eldest-first, tiers bind, deterministic tiebreak; regency")
    print("=" * 72)

    def heir_of(seed_kin) -> tuple[str | None, str]:
        _fresh()
        rex = _agent("Rex", (5, 5), parents=("Gpa", "Gma"), age=60)
        _crown("S001", "Rex")
        seed_kin(rex)
        h, kind = lineage._succession_heir(rex, world_state)
        return (h.name if h is not None else None), kind

    name, kind = heir_of(lambda rex: (
        _agent("Ada", (5, 6), parents=("Rex", "M"), age=18),
        _agent("Ben", (6, 5), parents=("Rex", "M"), age=25),   # ELDEST child
        _agent("Gpa", (4, 4), age=80)))                         # a parent, ignored
    print(f"  children present  -> eldest CHILD  : {name} ({kind})")
    assert (name, kind) == ("Ben", "child")

    name, kind = heir_of(lambda rex: (
        _agent("Zed", (5, 6), parents=("Rex", "M"), age=20),
        _agent("Ada", (6, 5), parents=("Rex", "M"), age=20)))   # same age -> name asc
    print(f"  age tie            -> NAME tiebreak : {name} ({kind})  (Ada < Zed)")
    assert (name, kind) == ("Ada", "child")

    name, kind = heir_of(lambda rex: (
        _agent("Gpa", (5, 6), age=82),                          # ELDEST parent
        _agent("Gma", (6, 5), age=78)))
    print(f"  no children        -> eldest PARENT : {name} ({kind})")
    assert (name, kind) == ("Gpa", "parent")

    name, kind = heir_of(lambda rex: (
        _agent("Uma", (5, 6), parents=("Gpa", "Gma"), age=30),
        _agent("Tom", (6, 5), parents=("Gpa", "X"), age=40)))   # ELDEST sibling (shares Gpa)
    print(f"  no child/parent    -> eldest SIBLING: {name} ({kind})")
    assert (name, kind) == ("Tom", "sibling")

    # Child-monarch REGENCY: a dependent heir holds the seat but wields no power.
    _fresh()
    king = _agent("Rex", (5, 5), age=61)
    _agent("Tot", (5, 6), parents=("Rex", "Isla"), dependent=True, age=6)   # a minor heir
    member = _agent("Rich", (4, 4), money=50.0)
    _crown("S001", "Rex", {"g"})
    population.announce_death(king, 70, world_state, cause="old age",
                              final_memory="Died of old age", note="they died of old age")
    reg = next(e for e in world_state["events"] if "succeeded" in e)
    print(f"\n  child-monarch: {reg.split(': ', 1)[1]}")
    assert world_state["monarchs"]["S001"]["monarch"] == "Tot" and "regency" in reg
    before = member.money
    monarchy.levy(world_state, 71)
    tot = next(a for a in world_state["agents"] if a.name == "Tot")
    print(f"    the child regent HOLDS the seat but its powers are dormant: levy took "
          f"{member.money - before:.0f} from a wealthy member; it is never an aspirant "
          f"({tot not in monarchy._eligible_aspirants(world_state, 'S001')}).")
    assert member.money == before
    print("  -> a dependent heir inherits the seat (a historically-real regency); its")
    print("     levy/muster/war stay dormant via the existing is_dependent_child gate.")
    print()


# --- MULTILEVEL: emperor + subject-king succession ---------------------------
def multilevel_succession() -> None:
    print("=" * 72)
    print("MULTI-LEVEL — an emperor's throne and a subject-king's seat both pass by blood")
    print("=" * 72)

    _fresh()
    _settlement("S002", (9, 9))
    emp = _agent("Emp", (5, 5), age=63)
    _agent("Ehe", (5, 6), parents=("Emp", "Isla"), age=25)     # emperor's heir
    sky = _agent("Sky", (9, 9), sid="S002", age=58)            # a subject-king
    _crown("S001", "Emp"); _crown("S002", "Sky")
    _realm("Emp", "S001", {"S001"}, {})
    _realm("Sky", "S002", {"S002"}, {})
    world_state["empires"]["Emp"] = {"emperor": "Emp", "subject_kings": {"Sky": {"since": 0}},
                                     "founded": 0, "discontent": {"Sky": 0}}
    print("  Emperor Emp rules an empire with subject-king Sky. Emp dies of old age.")
    population.announce_death(emp, 100, world_state, cause="old age",
                              final_memory="Died of old age", note="they died of old age")
    assert "Ehe" in world_state["empires"] and "Emp" not in world_state["empires"]
    ehe = world_state["empires"]["Ehe"]
    print(f"    imperial throne passes: emperor = {ehe['emperor']}; subject-kings "
          f"{sorted(ehe['subject_kings'])}; his own realm re-keyed to "
          f"{world_state['kingdoms']['Ehe']['king']}.")
    assert ehe["subject_kings"] == {"Sky": {"since": 0}} and ehe["discontent"] == {"Sky": 0}
    assert world_state["monarchs"]["S001"]["monarch"] == "Ehe"

    _agent("Ski", (8, 9), parents=("Sky", "Nel"), age=22)      # subject-king's heir
    print("\n  Now subject-king Sky dies of old age.")
    population.announce_death(sky, 101, world_state, cause="old age",
                              final_memory="Died of old age", note="they died of old age")
    ehe = world_state["empires"]["Ehe"]
    print(f"    the subordinate seat passes: subject-kings now {sorted(ehe['subject_kings'])}; "
          f"Sky's realm re-keyed to {world_state['kingdoms']['Ski']['king']}.")
    assert "Ski" in ehe["subject_kings"] and "Sky" not in ehe["subject_kings"]
    assert world_state["kingdoms"]["Ski"]["king"] == "Ski"
    print("  -> the same primogeniture rules bind at every level of the feudal hierarchy.")
    print()


# --- COMPOSE: off byte-identical, deterministic, zero LLM --------------------
def compose_checks() -> None:
    print("=" * 72)
    print("COMPOSE — off byte-identical; seeded runs reproduce; zero LLM")
    print("=" * 72)

    def run(**kw) -> str:
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(30, **kw)
        return buf.getvalue()

    assert run() == run(lineage_on=False)
    print("  --lineage OFF: byte-identical to the current default run (no succession fires)")
    assert run(lineage_on=True) == run(lineage_on=True)
    print("  --lineage ON: two seeded runs byte-identical (succession draws no RNG)")

    # Zero added LLM across a batch of successions.
    llm.reset_call_stats()
    for _ in range(3):
        _fresh()
        king = _agent("Rex", (5, 5), money=20.0, age=60)
        _agent("Cyn", (5, 6), parents=("Rex", "Isla"), age=20)
        _crown("S001", "Rex"); _realm("Rex", "S001", {"S001"}, {})
        population.announce_death(king, 5, world_state, cause="old age",
                                  final_memory="Died", note="they died")
    calls = llm.get_call_stats()
    assert calls["strategy"] == 0 and calls["decision"] == 0
    print(f"  succession over repeated deaths added ZERO LLM calls "
          f"(strategy={calls['strategy']}, decision={calls['decision']}).")
    print()


if __name__ == "__main__":
    saved = llm.PROVIDER
    try:
        headline_1_crown_survives_death()
        headline_2_succession_is_a_crisis()
        headline_3_extinct_lines()
        order_and_regency()
        multilevel_succession()
        compose_checks()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M4.3 VERIFIED — crowns pass down bloodlines; loyalty must be re-earned by each")
    print("heir; extinct lines dissolve into contestable power. Arc 1 closes: the House")
    print("becomes real, and history gains dynasties that rise, rule, and fall by heirs.")
    print("=" * 72)
