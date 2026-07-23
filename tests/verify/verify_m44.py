"""
verify_m44.py
=============

Deterministic verification of V2 milestone M4.4: DISCONTENT — the pressure gauge
of CLASS CONFLICT. The OPEN of Arc 2 (Revolt & Class Conflict), on top of all of
Arc 1 (M4.1 lineage, M4.2 inheritance, M4.3 dynasties) and all of Phases 0-3.

Run offline (Ollama OFF, no model server, no seed-search):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m44.py

The historical step: Phase 3 built a pressure engine with NO relief valve —
inequality compounds (M3.1), monarchs levy by force (M3.4), tribute is extracted
up a feudal hierarchy (M3.5), and dynastic heirs inherit crowns they never earned
(M4.3). Until now the oppressed just STARVED QUIETLY. M4.4 gives that pressure a
legible, verified GAUGE — per-agent discontent, derived from EXISTING signals —
so M4.5 can make it BLOW. M4.4 is ONLY the measure: no uprising fires here.

HEADLINE 1 — OPPRESSION RAISES THE GAUGE: the SAME settlement under a heavy-levying
             monarch accumulates high discontent; untaxed (or consensually taxed by
             a trusted M3.3 leader) it stays low. Each driver — levy, hunger-amid-
             plenty, subsistence wage — is individually demonstrable; all off -> flat 0.
HEADLINE 2 — LEGITIMACY BUFFERS GRIEVANCE: the SAME levy by a TRUSTED ruler draws
             materially less discontent than by a DISTRUSTED one. Consent is the
             difference between a tax and a theft.
HEADLINE 3 — GRIEVANCES OUTLAST CAUSES: oppression for N turns then relief -> the
             gauge decays but measurably SLOWER than it rose (asymmetric slopes);
             sustained good conditions eventually return it near 0 (floored).
COMPOSE    — in an integrated tyranny-vs-fair-rule scene the POOREST/most-taxed carry
             the highest discontent and the TYRANT'S settlement shows higher pressure
             than the fair leader's — emergent from the drivers, not assigned.
COST       — zero added LLM; --discontent off byte-identical; deterministic/reproducible.
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from sim import discontent
from sim import labor
from llm import llm
import main
from sim import monarchy
from sim import taxation
from sim import trust
from sim import world
from sim.agents import Agent
from sim.world import world_state


# --- Staging helpers ---------------------------------------------------------
def _fresh() -> None:
    """A clean discontent-on world with one settlement S001 centred at (5, 5)."""
    world.create_world()
    world_state["discontent_on"] = True
    world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5),
                                          "members": set(), "founded": 0}


def _settlement(sid: str, center: tuple[int, int]) -> None:
    world_state["settlements"][sid] = {"id": sid, "center": center,
                                       "members": set(), "founded": 0}


def _agent(name: str, pos: tuple[int, int], *, money: float = 0.0, stockpile: float = 0.0,
           hunger: int = 1, sid: "str | None" = "S001") -> Agent:
    a = Agent(name=name, personality="friendly and outgoing")
    world.place_agent(a, *pos)
    a.hunger, a.age, a.lifespan = hunger, 30, 100
    a.money, a.stockpile, a.settlement = money, stockpile, sid
    if sid is not None and sid in world_state["settlements"]:
        world_state["settlements"][sid]["members"].add(name)
    return a


def _monarch(sid: str, name: str) -> None:
    world_state["monarchs"][sid] = {"monarch": name, "since": 0, "garrison": set()}


def _run(turns: int) -> None:
    for t in range(1, turns + 1):
        discontent.update(world_state, t)


# --- HEADLINE 1: oppression raises the gauge ---------------------------------
def headline_1_oppression_raises_the_gauge() -> None:
    print("=" * 72)
    print("HEADLINE 1 — OPPRESSION RAISES THE GAUGE (and each driver is demonstrable)")
    print("=" * 72)

    # A/B: the SAME settlement, one under a levying monarch, one free — nothing else differs.
    def levied_settlement(with_monarch: bool) -> float:
        _fresh()
        subj = _agent("Sub", (5, 5), money=20.0)      # wealth over the levy threshold, fed
        _agent("Kin", (6, 5), money=60.0)
        if with_monarch:
            _monarch("S001", "Kin")
        _run(10)
        return discontent.agent_discontent("Sub", world_state)

    oppressed = levied_settlement(True)
    free = levied_settlement(False)
    print(f"  same subject, 10 turns:  under a levying monarch -> discontent {oppressed:.1f}")
    print(f"                           free / unruled          -> discontent {free:.1f}")
    assert oppressed > 5 * max(free, 0.01) and free == 0.0
    print("  -> the crown's forced levy is what fills the gauge; remove it and it stays flat.")

    # Consensual tax by a TRUSTED M3.3 leader is NOT extraction — it registers ZERO (consent).
    _fresh()
    sub = _agent("Sub", (5, 5), money=20.0)
    leader = _agent("Led", (6, 5), money=60.0)
    world_state["leadership_on"] = True
    world_state["leaders"]["S001"] = {"leader": "Led", "followers": {"Sub"}, "since": 0}
    trust.ensure_relationship(sub, "Led")["trust"] = 5   # a consenting follower
    _run(10)
    consensual = discontent.agent_discontent("Sub", world_state)
    print(f"\n  the SAME wealth under a CONSENSUAL trust-leader (M3.3, not a monarch): "
          f"discontent {consensual:.1f}")
    assert consensual == 0.0
    print("  -> a consented leader is not 'extraction' at all here: consent, not theft.")

    # Each driver ALONE raises the gauge; ALL off -> flat zero.
    def driver_only(kind: str) -> float:
        _fresh()
        a = _agent("A", (5, 5))
        rich = _agent("R", (6, 5), money=40.0)
        if kind == "deprivation":
            a.hunger = 8                                  # starving beside R's plenty
        elif kind == "exploitation":
            a.money = 20.0                                # fed & solvent; only the wage bites
            world_state.setdefault("employments", []).append(
                {"employer": "R", "worker": "A", "wage": labor.SUBSISTENCE_WAGE, "since": 0})
        elif kind == "extraction":
            a.money = 20.0
            _monarch("S001", "R")
        elif kind == "none":
            a.money = 20.0                                # fed, unemployed, unruled
        _run(6)
        return discontent.agent_discontent("A", world_state)

    dep, exp, ext, non = (driver_only(k) for k in
                          ("deprivation", "exploitation", "extraction", "none"))
    print(f"\n  each driver ALONE over 6 turns:")
    print(f"    deprivation amid plenty : {dep:.1f}")
    print(f"    exploitation (wage)     : {exp:.1f}")
    print(f"    extraction (levy)       : {ext:.1f}")
    print(f"    ALL conditions absent   : {non:.1f}")
    assert dep > 0 and exp > 0 and ext > 0 and non == 0.0
    assert dep > exp and dep > ext, "deprivation amid plenty is the strongest driver"
    print("  -> every driver individually fills the gauge; with none active it never leaves 0.")
    print()


# --- HEADLINE 2: legitimacy buffers grievance --------------------------------
def headline_2_legitimacy_buffers_grievance() -> None:
    print("=" * 72)
    print("HEADLINE 2 — LEGITIMACY BUFFERS GRIEVANCE (consent vs theft, same coin)")
    print("=" * 72)

    def levied_under(heir_trust: int) -> float:
        _fresh()
        sub = _agent("Sub", (5, 5), money=20.0)
        _agent("Kng", (6, 5), money=60.0)
        _monarch("S001", "Kng")
        trust.ensure_relationship(sub, "Kng")["trust"] = heir_trust
        _run(10)
        return discontent.agent_discontent("Sub", world_state)

    hated = levied_under(-5)
    neutral = levied_under(0)
    trusted = levied_under(5)
    print(f"  the SAME monarch takes the SAME levy for 10 turns, differing only in trust:")
    print(f"    distrusted ruler (-5) -> discontent {hated:.1f}")
    print(f"    neutral ruler (0)     -> discontent {neutral:.1f}")
    print(f"    trusted ruler (+5)    -> discontent {trusted:.1f}")
    assert trusted < neutral < hated
    assert trusted < 0.4 * hated
    print("  -> a legitimate (trusted) crown draws a FRACTION of the grievance of a hated one:")
    print("     legitimacy is exactly the difference between a consented tax and a hated levy.")
    print()


# --- HEADLINE 3: grievances outlast their causes -----------------------------
def headline_3_grievances_outlast_causes() -> None:
    print("=" * 72)
    print("HEADLINE 3 — GRIEVANCES OUTLAST CAUSES (rises fast, decays slow, floors at 0)")
    print("=" * 72)

    _fresh()
    poor = _agent("Poor", (5, 5), hunger=8)
    _agent("Rich", (6, 5), money=40.0)

    rise = []
    for t in range(1, 8):
        discontent.update(world_state, t)
        rise.append(discontent.agent_discontent("Poor", world_state))
    peak = rise[-1]
    rise_slope = peak / len(rise)
    print(f"  RISE — 7 turns of hunger amid plenty: {' -> '.join(f'{v:.1f}' for v in rise)}")
    print(f"         average climb {rise_slope:.2f}/turn")

    poor.hunger = 1                                   # relief: the grievance goes silent
    fall = []
    prev = peak
    for t in range(8, 20):
        discontent.update(world_state, t)
        prev = discontent.agent_discontent("Poor", world_state)
        fall.append(prev)
    fall_slope = (peak - fall[10]) / 11
    print(f"  RELIEF — fed again, the gauge EBBS: {peak:.1f} -> {' -> '.join(f'{v:.1f}' for v in fall)}")
    print(f"         average ebb {fall_slope:.2f}/turn")
    assert 0 < fall_slope < rise_slope
    print(f"  -> it FALLS ~{rise_slope / fall_slope:.0f}x slower than it rose: a decade of "
          f"oppression is not erased by one good harvest (hysteresis).")

    for t in range(20, 80):                            # sustained good conditions
        discontent.update(world_state, t)
    floor = discontent.agent_discontent("Poor", world_state)
    print(f"  FLOOR — after sustained relief the gauge returns to {floor:.1f} (a hard floor of 0).")
    assert floor == 0.0
    print()


# --- COMPOSE: emergent — the poor carry the pressure, tyranny outweighs fair rule ---
def compose_pressure_is_emergent() -> None:
    print("=" * 72)
    print("COMPOSE — pressure is EMERGENT: the poor carry it; a tyrant's town outweighs a")
    print("          fair leader's (from the drivers, nothing assigned)")
    print("=" * 72)

    _fresh()
    _settlement("S002", (8, 8))
    # S001 — a TYRANT'S town: a levying monarch, one wealthy commoner, two starving poor.
    _monarch("S001", "Tyr")
    _agent("Tyr", (5, 5), money=120.0)                # the monarch (excluded from own levy)
    _agent("Rich", (5, 6), money=40.0)                # a fed, taxed wealthy commoner
    _agent("Poor1", (6, 5), money=1.0, hunger=8)      # starving amid the plenty next door
    _agent("Poor2", (6, 6), money=1.0, hunger=9)
    # S002 — a FAIR leader's town: consensual M3.3 leadership, everyone fed and solvent.
    world_state["leadership_on"] = True
    world_state["leaders"]["S002"] = {"leader": "Fair", "followers": {"Ann", "Ben"}, "since": 0}
    fair = _agent("Fair", (8, 8), money=40.0, sid="S002")
    ann = _agent("Ann", (8, 9), money=15.0, sid="S002")
    ben = _agent("Ben", (9, 8), money=15.0, sid="S002")
    for f in (ann, ben):
        trust.ensure_relationship(f, "Fair")["trust"] = 5

    # Run the ACTUAL institutions each turn (the monarch levies; the fair leader taxes-and-
    # redistributes) alongside the gauge — the pressure is read off the resulting state.
    world_state["taxation_on"] = True
    for t in range(1, 12):
        monarchy.levy(world_state, t)
        taxation.update(world_state, t)          # the fair leader's consensual M3.3 tax
        discontent.update(world_state, t)

    ranking = sorted(((discontent.agent_discontent(n, world_state), n)
                      for n in ("Tyr", "Rich", "Poor1", "Poor2")), reverse=True)
    print("  S001 (tyranny) per-agent discontent, high to low:")
    for val, name in ranking:
        print(f"    {name:6} {val:5.1f}")
    top_two = {ranking[0][1], ranking[1][1]}
    assert top_two == {"Poor1", "Poor2"}, top_two
    assert discontent.agent_discontent("Tyr", world_state) == 0.0
    print("  -> the two STARVING poor carry the most; the monarch himself carries none.")

    p1 = discontent.settlement_pressure("S001", world_state)
    p2 = discontent.settlement_pressure("S002", world_state)
    d1 = discontent.settlement_discontent("S001", world_state)
    d2 = discontent.settlement_discontent("S002", world_state)
    print(f"\n  SETTLEMENT PRESSURE (resentful faction size / aggregate):")
    print(f"    S001 tyranny     : {p1} resentful, aggregate {d1:.1f}")
    print(f"    S002 fair leader : {p2} resentful, aggregate {d2:.1f}")
    assert p1 > p2 and d1 > d2 and p2 == 0
    print("  -> the tyrant's town seethes; the fair leader's is calm — the number M4.5 will")
    print("     trigger an uprising on, emergent from the class engine, not assigned.")
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
            main.run_simulation(30, settlements=True, labor_on=True, economy_on=True,
                                monarchy_on=True, **kw)
        return buf.getvalue(), dict(llm.get_call_stats())

    off, off_calls = run()
    off2, _ = run(discontent_on=False)
    assert off == off2
    print("  --discontent OFF: byte-identical to the current default institution run")

    on_a, on_calls = run(discontent_on=True)
    on_b, _ = run(discontent_on=True)
    assert on_a == on_b
    print("  --discontent ON: two seeded runs byte-identical (the gauge draws no RNG)")
    assert on_calls == off_calls
    print(f"  the gauge added ZERO LLM calls (on={on_calls}, off={off_calls}).")
    print()


if __name__ == "__main__":
    saved = llm.PROVIDER
    try:
        headline_1_oppression_raises_the_gauge()
        headline_2_legitimacy_buffers_grievance()
        headline_3_grievances_outlast_causes()
        compose_pressure_is_emergent()
        cost_checks()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M4.4 VERIFIED — the class engine gets a pressure gauge. Discontent genuinely")
    print("TRACKS oppression: each driver is demonstrable, legitimacy buffers grievance, and")
    print("grievances outlast their causes. The gauge is DERIVED, not assigned — and M4.5")
    print("will make it BLOW.")
    print("=" * 72)
