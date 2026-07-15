"""
verify_m413.py
==============

Deterministic verification of V2 milestone M4.13: RELATIONS & TREATIES — kingdoms gain
a second verb. First milestone of Arc 5 (Diplomacy & the Interstate System), on top of
Arc 4 (eras/metallurgy/writing), Arc 3 (culture), Arc 2 (revolt), Arc 1 (dynasties) and
Phases 0-3.

Run offline (Ollama OFF, no model server, no seed-search):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m413.py

The historical step: kingdoms had exactly ONE interaction — opportunistic WAR (M3.6).
M4.13 gives every kingdom-PAIR a STANCE derived from history, and TREATIES — non-aggression
pacts and defensive alliances — that CONSTRAIN the existing war machinery. War stops being
a kingdom's only language. Zero LLM; stance is deterministic integer math.

HEADLINE 1 — STANCE EMERGES FROM HISTORY: two kingdoms that warred are hostile; two sharing
             culture/faith trend friendly; a broken treaty sours a pair; stance decays toward
             neutral over quiet time. Derived from state, not assigned.
HEADLINE 2 — DIPLOMACY PREVENTS WAR: a non-aggression pact between two kingdoms the M3.6 loop
             WOULD have fought prevents the war while it holds; when the pact breaks (souring),
             the war fires. Same pair, pact vs no pact, two outcomes.
HEADLINE 3 — ALLIANCES SHIFT WARS: an attacker who beats a lone defender LOSES when the
             defender's ALLY musters to the defence (combined hosts); and an ally whose honour
             has lapsed FAILS to answer, leaving the defender to fall — conditional, not ironclad.
COMPOSE    — a cold heir (M4.3) does NOT renew his father's pact: on his succession the pact
             lapses and the war it had prevented reopens.
COST       — zero added LLM; --diplomacy off byte-identical; deterministic/reproducible.
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any

import beliefs as B
import diplomacy
import empire
import llm
import main
import population
import world
from agents import Agent
from world import world_state


# --- Staging helpers ---------------------------------------------------------
def _fresh() -> None:
    world.create_world(size=60)
    world_state["diplomacy_on"] = True


def _settled(n, p, sid=None, money=0.0, **kw) -> Agent:
    a = Agent(name=n, personality="x")
    world.place_agent(a, *p)
    a.hunger, a.age, a.lifespan, a.money, a.settlement = 1, 30, 100, money, sid
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _mercs(prefix, near, n) -> None:
    for i in range(n):
        _settled(f"{prefix}{i}", (near[0] + i % 2, near[1] + 2), sid=None, money=0.5)


def _realm(king, kmoney, home_c, *, age=30, parents=()) -> None:
    home = f"{king}_home"
    world_state["settlements"][home] = {"id": home, "center": home_c, "members": {king}, "founded": 0}
    _settled(king, home_c, sid=home, money=kmoney, age=age, parents=parents)
    world_state["monarchs"][home] = {"monarch": king, "since": 0, "garrison": set()}
    world_state["kingdoms"][king] = {"king": king, "home": home, "settlements": {home},
                                     "vassals": {}, "founded": 0, "discontent": {}}


# --- HEADLINE 1: stance emerges from history ---------------------------------
def headline_1_stance_from_history() -> None:
    print("=" * 72)
    print("HEADLINE 1 — STANCE EMERGES FROM HISTORY (war cools, culture warms, quiet fades)")
    print("=" * 72)

    _fresh(); _realm("A", 100.0, (10, 10)); _realm("B", 100.0, (18, 10))
    print(f"  two fresh kingdoms A<->B: {diplomacy.stance(world_state, 'A', 'B')}")
    diplomacy.record_war(world_state, "A", "B", 1)
    print(f"  after they go to WAR: {diplomacy.stance(world_state, 'A', 'B')} "
          f"({diplomacy.stance_score(world_state, 'A', 'B'):+d})")
    assert diplomacy.stance(world_state, "A", "B") == "hostile"
    for t in range(2, 12):
        diplomacy.update(world_state, t)
    print(f"  after 10 quiet turns (grievance fades): {diplomacy.stance(world_state, 'A', 'B')}")
    assert diplomacy.stance(world_state, "A", "B") == "neutral"

    # Shared culture warms a pair toward friendly.
    _fresh(); world_state["culture_on"] = True; world_state["beliefs_on"] = True
    _realm("A", 100.0, (10, 10)); _realm("B", 100.0, (18, 10))
    creed = {B.LAND_PROVIDES, B.STRONGER_TOGETHER}
    world_state["beliefs"] = {"A": set(creed), "B": set(creed)}
    for t in range(1, 8):
        diplomacy.update(world_state, t)
    print(f"\n  two kings of ONE culture, after 7 turns: {diplomacy.stance(world_state, 'A', 'B')} "
          f"({diplomacy.stance_score(world_state, 'A', 'B'):+d})")
    assert diplomacy.stance(world_state, "A", "B") == "friendly"
    print("  -> stance is DERIVED from what two crowns have shared and suffered, never assigned.")
    print()


# --- HEADLINE 2: diplomacy prevents war --------------------------------------
def headline_2_diplomacy_prevents_war() -> None:
    print("=" * 72)
    print("HEADLINE 2 — DIPLOMACY PREVENTS WAR (a pact stays the M3.6 loop's hand)")
    print("=" * 72)

    def outcome(with_pact) -> bool:
        _fresh()
        _realm("Rich", 500.0, (10, 10)); _realm("Poor", 50.0, (18, 10))
        _mercs("R", (10, 10), 5); _mercs("P", (18, 10), 2)
        if with_pact:
            world_state["diplomacy"] = {"stance": {("Poor", "Rich"): 3}, "pacts": set(), "alliances": set()}
        diplomacy.update(world_state, 1)
        empire.update(world_state, 1)
        return empire.is_sovereign(world_state, "Poor")   # still free?

    print(f"  strong Rich beside weak Poor — no pact: Poor survives the turn? {outcome(False)}")
    print(f"  the same pair with a NON-AGGRESSION PACT: Poor survives? {outcome(True)}")
    assert not outcome(False) and outcome(True)

    # The pact BREAKS when the stance sours, and the war becomes possible again.
    _fresh()
    _realm("Rich", 500.0, (10, 10)); _realm("Poor", 50.0, (18, 10))
    _mercs("R", (10, 10), 5); _mercs("P", (18, 10), 2)
    world_state["diplomacy"] = {"stance": {("Poor", "Rich"): 3}, "pacts": {("Poor", "Rich")}, "alliances": set()}
    diplomacy.record_war(world_state, "Rich", "Poor", 1)   # a shock sours the pair
    ev = diplomacy.update(world_state, 2)
    print(f"\n  when the stance sours, the pact is BROKEN: {[e for e in ev if 'BROKEN' in e][0]}")
    assert not diplomacy.has_pact(world_state, "Rich", "Poor")
    empire.update(world_state, 2)
    print(f"    ...and the war fires: Poor subjugated? {not empire.is_sovereign(world_state, 'Poor')}")
    assert not empire.is_sovereign(world_state, "Poor")
    print("  -> a pact genuinely PREVENTS a war the machinery would launch; break it and the war returns.")
    print()


# --- HEADLINE 3: alliances shift wars ----------------------------------------
def headline_3_alliances_shift_wars() -> None:
    print("=" * 72)
    print("HEADLINE 3 — ALLIANCES SHIFT WARS (a friend's host joins; a soured ally does not)")
    print("=" * 72)

    def war(alliance_stance) -> tuple[bool, int]:
        _fresh()
        _realm("Rich", 500.0, (10, 10)); _realm("Poor", 50.0, (30, 10)); _realm("Ally", 400.0, (50, 10))
        _mercs("R", (10, 10), 5); _mercs("P", (30, 10), 2); _mercs("Y", (50, 10), 4)
        if alliance_stance is not None:
            world_state["diplomacy"] = {"stance": {("Ally", "Poor"): alliance_stance},
                                        "pacts": set(), "alliances": {("Ally", "Poor")}}
        res = empire.wage_war(world_state, "Rich", "Poor", 1)
        return res["won"], res["def_host"]

    won_lone, def_lone = war(None)
    won_ally, def_ally = war(3)          # an honouring ally
    won_soured, def_soured = war(-5)     # an ally whose honour has lapsed
    print(f"  Rich attacks LONE Poor:            Rich wins? {won_lone}  (hosts vs {def_lone})")
    print(f"  Rich attacks Poor + honouring ALLY: Rich wins? {won_ally}  (defence grew to {def_ally})")
    print(f"  Rich attacks Poor + SOURED ally:   Rich wins? {won_soured}  (ally sat out, defence {def_soured})")
    assert won_lone and not won_ally and won_soured
    assert def_ally > def_lone == def_soured
    print("  -> attacking one member of an alliance means fighting all who HONOUR it — but honour is")
    print("     conditional: a soured ally stays home, so alliances are never ironclad.")
    print()


# --- COMPOSE: a cold heir does not renew his father's pact --------------------
def compose_cold_heir_lapses_the_pact() -> None:
    print("=" * 72)
    print("COMPOSE — a cold heir (M4.3) does not renew his father's pact, reopening the war")
    print("=" * 72)

    _fresh(); world_state["lineage_on"] = True
    _realm("Rich", 500.0, (10, 10))
    _realm("Peace", 50.0, (18, 10), age=62)
    _settled("Heir", (19, 10), sid="Peace_home", age=25, parents=("Peace", "Queen"))
    world_state["settlements"]["Peace_home"]["members"].add("Heir")
    _mercs("R", (10, 10), 5); _mercs("P", (18, 10), 2)
    world_state["diplomacy"] = {"stance": {("Peace", "Rich"): 3}, "pacts": {("Peace", "Rich")}, "alliances": set()}
    diplomacy.update(world_state, 1); empire.update(world_state, 1)
    print(f"  King Peace's pact with Rich holds -> Poor kingdom survives? {empire.is_sovereign(world_state, 'Peace')}")
    assert empire.is_sovereign(world_state, "Peace")

    # King Peace dies of old age; his heir is crowned (M4.3) — but inherits the CROWN, not the pact.
    peace = next(a for a in world_state["agents"] if a.name == "Peace")
    population.announce_death(peace, 2, world_state, cause="old age", final_memory="Died", note="they died")
    assert world_state["monarchs"]["Peace_home"]["monarch"] == "Heir", "the heir is crowned (M4.3)"
    diplomacy.update(world_state, 3)
    print(f"  King Peace dies; heir crowned. The pact was personal -> still pacted? "
          f"{diplomacy.has_pact(world_state, 'Rich', 'Heir')}")
    assert not diplomacy.has_pact(world_state, "Rich", "Heir")
    empire.update(world_state, 3)
    print(f"    ...so Rich attacks the untreatied heir: heir's kingdom subjugated? "
          f"{not empire.is_sovereign(world_state, 'Heir')}")
    assert not empire.is_sovereign(world_state, "Heir")
    print("  -> the crown passes by blood but the alliances do NOT: a new reign must re-earn its peace,")
    print("     and a cold heir reopens the war his father's diplomacy had prevented (echoes M4.3).")
    print()


# --- COST: off byte-identical, deterministic, zero added LLM -----------------
def cost_checks() -> None:
    print("=" * 72)
    print("COST — off byte-identical; seeded runs reproduce; zero added LLM")
    print("=" * 72)

    def run(**kw) -> tuple[str, dict]:
        llm.PROVIDER = "random"
        random.seed(7)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(30, stage="war", **kw)
        return buf.getvalue(), dict(llm.get_call_stats())

    off, off_calls = run()
    off2, _ = run(diplomacy_on=False)
    assert off == off2
    print("  --diplomacy OFF: byte-identical to the staged-war run (the war loop never checks a treaty)")
    on_a, on_calls = run(diplomacy_on=True)
    on_b, _ = run(diplomacy_on=True)
    assert on_a == on_b
    print("  --diplomacy ON: two seeded runs byte-identical (stance is pure integer math — no RNG)")
    assert on_calls == off_calls
    print(f"  diplomacy added ZERO LLM calls (on={on_calls}, off={off_calls}).")
    print()


if __name__ == "__main__":
    saved = llm.PROVIDER
    try:
        headline_1_stance_from_history()
        headline_2_diplomacy_prevents_war()
        headline_3_alliances_shift_wars()
        compose_cold_heir_lapses_the_pact()
        cost_checks()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M4.13 VERIFIED — stance genuinely EMERGES from history; non-aggression pacts genuinely")
    print("PREVENT wars the machinery would launch; and alliances genuinely SHIFT war outcomes while")
    print("staying conditional on honour. Kingdoms gain a second verb — war is no longer their only")
    print("language, and Arc 5 (the interstate system) opens.")
    print("=" * 72)
