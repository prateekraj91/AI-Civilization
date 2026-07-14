"""
verify_m46.py
=============

Deterministic verification of V2 milestone M4.6: THE REVOLUTIONARY — a rising's
leader becomes a legitimate ruler. CLOSES Arc 2 (Revolt & Class Conflict), on top of
M4.5 (uprising), M4.4 (discontent), all of Arc 1 and Phases 0-3.

Run offline (Ollama OFF, no model server, no seed-search):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m46.py

The historical step: M4.5 ends a won rising with the tyrant deposed and the seat
VACANT. M4.6 fills it — the agent who LED the rising emerges as the settlement's new
leader: power seized by FORCE, then LEGITIMISED by CONSENT. He is DERIVED from the
risers (angriest + most trusted by his fellow risers), takes the seat through the
UNCHANGED M3.2 leadership path (a leader, NOT a monarch), and holds it only while
trusted. This closes the cycle the whole project circled — consent (M3.2) -> force
(M3.4) -> revolt (M4.5) -> consent again (M4.6) — and yields the project's second
emergent great-figure: the REVOLUTIONARY, born of grievance (M3.4's conqueror was born
of wealth). And if he later becomes an extractor himself, the same machinery rises
against HIM: the revolution devours its children.

HEADLINE 1 — THE MOB THROWS UP A LEADER: a won rising produces a named revolutionary
             DERIVED from the risers (the angriest COMMONER his fellows most trust —
             not the richest, not arbitrary), who takes the vacant seat by CONSENT (an
             M3.2 leaders record, never a monarch record).
HEADLINE 2 — POWER BY CONSENT, NOT CROWN: the revolutionary holds the seat only while
             trusted — show him LOSING it via M3.2's existing erosion/displacement when
             his following collapses. No permanent crown.
HEADLINE 3 — THE CYCLE CLOSES / THE REVOLUTION DEVOURS ITS CHILDREN: the SAME figure,
             two paths — a revolutionary who rules FAIRLY keeps a calm settlement; one
             who becomes an EXTRACTOR (seizes a crown) breeds fresh discontent and is
             HIMSELF risen against by the same machinery.
COMPOSE    — the FULL CYCLE in one run: a consent-leader -> conquered by FORCE -> the
             people RISE -> a revolutionary rules by CONSENT again.
COST       — zero added LLM; --uprising off byte-identical; deterministic/reproducible.
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any

import discontent
import leadership
import llm
import main
import monarchy
import trust
import uprising
import world
from agents import Agent
from world import world_state


# --- Staging helpers ---------------------------------------------------------
def _fresh() -> None:
    world.create_world()
    world_state["uprising_on"] = True
    world_state["discontent_on"] = True
    world_state["leadership_on"] = True
    world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5),
                                          "members": set(), "founded": 0}


def _agent(name: str, pos: tuple[int, int], *, money: float = 0.0, sid: "str | None" = "S001") -> Agent:
    a = Agent(name=name, personality="friendly and outgoing")
    world.place_agent(a, *pos)
    a.hunger, a.age, a.lifespan = 1, 30, 100
    a.money, a.settlement = money, sid
    if sid is not None and sid in world_state["settlements"]:
        world_state["settlements"][sid]["members"].add(name)
    return a


def _monarch(sid: str, name: str) -> None:
    world_state["monarchs"][sid] = {"monarch": name, "since": 0, "garrison": set()}


def _mercs(positions, money: float = 0.5) -> None:
    for i, p in enumerate(positions):
        a = Agent(name=f"guard{i}", personality="x")
        world.place_agent(a, *p)
        a.hunger, a.age, a.lifespan, a.money, a.settlement = 1, 30, 100, money, None


def _find(name: str) -> Any:
    return next(a for a in world_state["agents"] if a.name == name)


# --- HEADLINE 1: the mob throws up a leader ----------------------------------
def headline_1_the_mob_throws_up_a_leader() -> None:
    print("=" * 72)
    print("HEADLINE 1 — THE MOB THROWS UP A LEADER (derived from the risers, ruling by consent)")
    print("=" * 72)

    _fresh()
    _agent("King", (5, 5), money=0.5)                     # a drained tyrant -> the mob wins on numbers
    for n, p in [("A", (5, 6)), ("B", (6, 5)), ("C", (6, 7))]:
        _agent(n, p, money=0.0)
    rex = _agent("Rex", (7, 5), money=100.0)              # a RICH riser — angry, but cannot lead
    _monarch("S001", "King")
    # Rex is the angriest, but rich; B is the angriest COMMONER, and the one his fellow risers trust.
    world_state["discontent"] = {"A": 12.0, "B": 20.0, "C": 12.0, "Rex": 25.0}
    for n in ("A", "C", "Rex"):
        trust.ensure_relationship(_find(n), "B")["trust"] = 1

    print("  risers: A(disc 12), B(disc 20, trusted by A/C/Rex), C(disc 12), Rex(disc 25 but RICH).")
    res = uprising.update(world_state, 10)[0]
    print(f"    the rising wins; the revolutionary the mob throws up = {res['leader']}")
    print(f"    (NOT Rex — the richest/angriest is excluded; the leader is a COMMONER, derived not assigned)")
    assert res["won"] and res["leader"] == "B"
    assert world_state.get("leaders", {}).get("S001") is None, "the uprising installs no seat itself"
    print(f"    right after the rising: monarch cleared? {'S001' not in world_state['monarchs']}; "
          f"leader record yet? {world_state.get('leaders', {}).get('S001')}")
    # The UNCHANGED M3.2 machinery elects him from the trust the victory earned him — power by CONSENT.
    leadership.update(world_state, 11)
    rec = world_state["leaders"]["S001"]
    print(f"    then M3.2 leadership.update elects him: leader of S001 = {rec['leader']} "
          f"(followers {sorted(rec['followers'])}) — a CONSENT seat, monarch record? "
          f"{'S001' in world_state['monarchs']}")
    assert rec["leader"] == "B" and "S001" not in world_state["monarchs"]
    print("  -> the mob threw up its own leader, and he holds the seat by consent, not a crown.")
    print()


# --- HEADLINE 2: power by consent, not crown ---------------------------------
def headline_2_power_by_consent_not_crown() -> None:
    print("=" * 72)
    print("HEADLINE 2 — POWER BY CONSENT, NOT CROWN (the revolutionary is losable like any leader)")
    print("=" * 72)

    _fresh()
    _agent("King", (5, 5), money=0.5)
    for n, p in [("A", (5, 6)), ("B", (6, 5)), ("C", (6, 7))]:
        _agent(n, p, money=0.0)
    _monarch("S001", "King")
    world_state["discontent"] = {"A": 20.0, "B": 12.0, "C": 12.0}   # A is the angriest -> leads
    uprising.update(world_state, 10)
    leadership.update(world_state, 11)
    print(f"  the revolutionary A rules S001 by consent (followers "
          f"{sorted(world_state['leaders']['S001']['followers'])}).")
    assert world_state["leaders"]["S001"]["leader"] == "A"

    # His following collapses — the people cool on him (trust falls below the M3.2 keep-bar).
    for n in ("B", "C"):
        _find(n).relationships["A"]["trust"] = 0
    leadership.update(world_state, 12)
    print(f"    his following erodes (trust falls below the keep-bar) -> leader of S001 now: "
          f"{world_state['leaders'].get('S001')}")
    assert world_state["leaders"].get("S001") is None
    assert "S001" not in world_state["monarchs"], "he was never a monarch — no permanent grip"
    print("  -> he falls exactly like any M3.2 leader. The revolution gives a following, not a crown.")
    print()


# --- HEADLINE 3: the cycle closes / the revolution devours its children -------
def headline_3_the_revolution_devours_its_children() -> None:
    print("=" * 72)
    print("HEADLINE 3 — THE REVOLUTION DEVOURS ITS CHILDREN (same figure, two paths, two fates)")
    print("=" * 72)

    def revolutionary_then(fate: str) -> bool:
        """Install revolutionary R as leader of S001, then either rule FAIR or seize a CROWN."""
        _fresh()
        r = _agent("R", (5, 5), money=0.5)
        members = [_agent(n, p, money=20.0) for n, p in [("A", (5, 6)), ("C", (6, 5)), ("D", (6, 7))]]
        # R already rules by consent (the survivors of an earlier rising trust him).
        for m in members:
            trust.ensure_relationship(m, "R")["trust"] = 3
        world_state["leaders"]["S001"] = {"leader": "R", "followers": {"A", "C", "D"}, "since": 0}
        if fate == "fair":
            # A fair leader extracts nothing (no monarch/levy) -> members fed, no grievance.
            pass
        else:
            # R SEIZES the crown (M3.4) — now an extractor by FORCE, and the people do not consent to it.
            _monarch("S001", "R")
            for m in members:
                m.relationships["R"] = {"trust": -5}
        for turn in range(1, 13):
            discontent.update(world_state, turn)
        risen = uprising.update(world_state, 13)
        return bool(risen and risen[0]["won"])

    fair_risen = revolutionary_then("fair")
    tyrant_risen = revolutionary_then("crown")
    print(f"  the SAME revolutionary R, two paths after taking power:")
    print(f"    rules FAIRLY (no extraction)      -> his settlement rises against him? {fair_risen}")
    print(f"    becomes an EXTRACTOR (seizes crown)-> his settlement rises against him? {tyrant_risen}")
    assert not fair_risen and tyrant_risen
    assert "S001" not in world_state["monarchs"], "the revolutionary-turned-tyrant is himself deposed"
    print("  -> a fair revolutionary is safe; one who becomes a tyrant is overthrown by the SAME")
    print("     machinery. The wheel turns again — the revolution devours its children.")
    print()


# --- COMPOSE: the full cycle in one run --------------------------------------
def compose_the_full_cycle() -> None:
    print("=" * 72)
    print("COMPOSE — the FULL CYCLE: consent -> force -> revolt -> consent again")
    print("=" * 72)

    _fresh()
    # (1) CONSENT: a trust-leader L rules S001 by the people's trust (2 devoted followers).
    _agent("L", (5, 5), money=2.0)
    f1, f2 = _agent("A", (5, 6)), _agent("B", (4, 5))
    commoners = [_agent(n, p, money=20.0) for n, p in [("C", (6, 5)), ("D", (6, 6)),
                                                       ("E", (5, 7)), ("F", (4, 6))]]
    for f in (f1, f2):
        trust.ensure_relationship(f, "L")["trust"] = 3
    leadership.update(world_state, 1)
    print(f"  (1) CONSENT — {world_state['leaders']['S001']['leader']} leads S001 by trust "
          f"({len(world_state['leaders']['S001']['followers'])} followers).")
    assert world_state["leaders"]["S001"]["leader"] == "L"

    # (2) FORCE: a rich outsider musters an army and CONQUERS S001 (real M3.4 machinery — it beats
    # L's loyal followers, who die defending him; the survivors of the bought army become the garrison).
    aspirant = _agent("Tyr", (4, 4), money=15.0, sid=None)
    _mercs([(3, 3), (3, 4), (4, 3)])                       # a small pool -> a small standing garrison
    res = monarchy.attempt_conquest(world_state, aspirant, "S001", 2)
    print(f"  (2) FORCE — Tyr musters {res['attackers']} fighters vs {res['defenders']} defenders and "
          f"SEIZES S001: monarch now = {world_state['monarchs']['S001']['monarch']} "
          f"(garrison {len(world_state['monarchs']['S001']['garrison'])}).")
    assert res["won"] and world_state["monarchs"]["S001"]["monarch"] == "Tyr"

    # (3) REVOLT: Tyr rules the commoners by force and is distrusted; grievance builds until they RISE
    # (his war chest is spent, so his numbers cannot grow and the mob's numbers decide).
    for c in commoners:
        c.relationships["Tyr"] = {"trust": -5}
    _find("Tyr").money = 0.5
    fired = fired_turn = None
    for turn in range(3, 20):
        discontent.update(world_state, turn)
        r = uprising.update(world_state, turn)
        if r:
            fired, fired_turn = r[0], turn
            break
    print(f"  (3) REVOLT — the people rise on turn {fired_turn}; Tyr is DEPOSED, and the mob throws "
          f"up {fired['leader']}.")
    assert fired is not None and fired["won"] and "S001" not in world_state["monarchs"]

    # (4) CONSENT AGAIN: the revolutionary is elected leader by the survivors' trust.
    leadership.update(world_state, fired_turn + 1)
    rec = world_state["leaders"].get("S001")
    print(f"  (4) CONSENT AGAIN — {rec['leader']} rules S001 by consent (an M3.2 leader, not a crown).")
    assert rec is not None and rec["leader"] == fired["leader"] and "S001" not in world_state["monarchs"]
    print("  -> the wheel turned full circle in one run: consent, force, revolt, and consent again.")
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
            main.run_simulation(30, settlements=True, leadership_on=True, monarchy_on=True,
                                discontent_on=True, **kw)
        return buf.getvalue(), dict(llm.get_call_stats())

    off, off_calls = run()
    off2, _ = run(uprising_on=False)
    assert off == off2
    print("  --uprising OFF: byte-identical to the current institution run")
    on_a, on_calls = run(uprising_on=True)
    on_b, _ = run(uprising_on=True)
    assert on_a == on_b
    print("  --uprising ON: two seeded runs byte-identical (the revolutionary draws no RNG)")
    assert on_calls == off_calls
    print(f"  the revolutionary added ZERO LLM calls (on={on_calls}, off={off_calls}).")
    print()


if __name__ == "__main__":
    saved = llm.PROVIDER
    try:
        headline_1_the_mob_throws_up_a_leader()
        headline_2_power_by_consent_not_crown()
        headline_3_the_revolution_devours_its_children()
        compose_the_full_cycle()
        cost_checks()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M4.6 VERIFIED — the revolutionary genuinely EMERGES from the mob (derived from grievance")
    print("and the risers' own trust, not assigned), rules only by CONSENT (losable like any leader),")
    print("and the cycle truly CLOSES: a revolutionary-turned-tyrant is overthrown by the same")
    print("machinery. Arc 2 closes — power returns to consent, and the wheel turns again.")
    print("=" * 72)
