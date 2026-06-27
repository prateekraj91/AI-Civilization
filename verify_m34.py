"""
verify_m34.py
=============

Deterministic verification of V2 milestone M3.4: CONQUEST & MONARCHY — power seized by FORCE.
Phase 3 (Institutions), on top of M3.1 (wage labor), M3.2 (legitimate leadership), M3.3
(taxation) and all of Phase 0/1/2.

Run offline (Ollama OFF, no model server, no seed-search, no long Qwen run):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m34.py

The historical step: M3.2 built authority by CONSENT (trust-legitimacy, power granted from below).
M3.4 builds the OTHER source of power: DOMINATION (force, power seized from above). A wealthy agent
converts money into an ARMY of real fighters and seizes a settlement, becoming MONARCH — the dark
climax of the class engine: the rich buy an army and take the crown.

HEADLINE 1 — WEALTH -> FORCE -> CROWN (emergent, not a lookup): a wealthy aspirant SPENDS money to
             muster real fighters and SEIZES a settlement, becoming monarch. The force is bought
             soldiers, not a number compare — a broke aspirant musters nobody.
HEADLINE 2 — FORCE vs LEGITIMACY COLLIDES: a trusted M3.2 leader's loyal FOLLOWERS REPEL a richer
             attacker whose mustered force is smaller; but an overwhelming bought force overcomes
             them. BOTH outcomes — the result turns on mustered force, not on wealth.
DEMO C — MONARCH RULES BY FORCE: a monarch levies wealth WITHOUT consent (contrast M3.3, which
         required a legitimate leader + the consent of the governed).
DEMO D — THE CROWN IS LOSABLE: a stronger later aspirant OVERTHROWS the monarch; succession by force.
DEMO E — WAR IS COSTLY: fighting KILLS real agents on both sides (force is destroyed, not free).
DEMO F — ZERO added LLM/RNG; deterministic + reproducible under seed; monarchy OFF -> v1 identical.
"""

from __future__ import annotations

import contextlib
import io
import random

import leadership
import llm
import main
import monarchy
import world
from agents import Agent
from world import world_state


def _settled(name: str, pos: tuple[int, int], sid: str | None = "S001", **kw) -> Agent:
    """Place a living agent (settled into `sid`, or a roaming outsider when sid is None)."""
    a = Agent(name=name, personality="cautious and territorial")
    world.place_agent(a, *pos)
    a.settlement = sid
    a.hunger = kw.pop("hunger", 0)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _trusts(follower: Agent, leader_name: str, value: int) -> None:
    follower.relationships[leader_name] = {"trust": value, "interactions": 1, "grudge": False}


def _wealth(a: Agent) -> float:
    return a.money + a.stockpile


def _living(name: str) -> Agent:
    return next(a for a in world_state["agents"] if a.name == name)


def _settlement(sid: str, center: tuple[int, int], members: set[str]) -> None:
    world_state["settlements"] = {sid: {"id": sid, "center": center, "members": members, "founded": 0}}


def headline_1_wealth_to_force_to_crown() -> None:
    print("=" * 72)
    print("HEADLINE 1 — WEALTH -> FORCE -> CROWN: a wealthy aspirant buys an army and SEIZES a town")
    print("=" * 72)
    world.create_world(size=14)
    world_state["monarchy_on"] = True; world_state["monarchs"] = {}
    _settlement("S001", (7, 7), {"M1", "M2"})
    _settled("M1", (7, 7), money=1.0); _settled("M2", (7, 8), money=1.0)   # an unled, poor town
    rich = _settled("Rich", (8, 8), sid=None, money=30.0)                   # the aspirant
    for i in range(4):                                                      # poor mercenaries nearby
        _settled(f"Merc{i}", (6 + i % 3, 6), sid=None, money=0.5)
    print(f"  aspirant Rich: wealth {_wealth(rich):.0f}  (could fund {monarchy.max_fighters(rich)} "
          f"fighters at {monarchy.FIGHTER_COST:.0f} each)")
    print(f"  town S001: 2 poor members, NO leader / NO monarch")
    res = monarchy.attempt_conquest(world_state, rich, "S001", 1)
    rec = world_state["monarchs"].get("S001")
    print(f"    -> mustered {res['attackers']} REAL fighters (only 4 mercs were in range), "
          f"spending wealth {30.0:.0f} -> {_wealth(rich):.0f}")
    print(f"    -> fight: {res['attackers']} attackers vs {res['defenders']} militia "
          f"({len(res['att_dead'])}+{len(res['def_dead'])} fell) -> won={res['won']}")
    print(f"    -> monarchs[S001] = monarch {rec['monarch']}, since {rec['since']}, "
          f"garrison {sorted(rec['garrison'])}")
    assert res["won"] and rec["monarch"] == "Rich", "wealth-funded force must seize the town"
    assert res["attackers"] == 4 and _wealth(rich) == 30.0 - 4 * monarchy.FIGHTER_COST, \
        "the force must be REAL fighters paid for with money, not a number compare"

    # A BROKE aspirant can fund NO fighters and conquers nothing — wealth is the hard gate.
    world.create_world(size=14)
    world_state["monarchy_on"] = True; world_state["monarchs"] = {}
    _settlement("S001", (7, 7), {"M1", "M2"})
    _settled("M1", (7, 7), money=1.0); _settled("M2", (7, 8), money=1.0)
    broke = _settled("Broke", (8, 8), sid=None, money=3.0)                  # below the war-chest floor
    for i in range(4):
        _settled(f"Merc{i}", (6 + i % 3, 6), sid=None, money=0.5)
    res2 = monarchy.attempt_conquest(world_state, broke, "S001", 1)
    print(f"  a BROKE aspirant (wealth 3 < war chest {monarchy.MIN_WAR_CHEST:.0f}): "
          f"mustered {res2['attackers']} fighters -> won={res2['won']}")
    assert monarchy.max_fighters(broke) == 0 and not res2["won"], "a broke aspirant can muster nothing"
    print("  -> the crown is bought with REAL soldiers funded by wealth; no money, no army, no "
          "crown.  PASS\n")


def headline_2_force_vs_legitimacy() -> None:
    print("=" * 72)
    print("HEADLINE 2 — FORCE vs LEGITIMACY: loyal followers REPEL a richer-but-smaller attacker;")
    print("              overwhelming bought force overcomes them")
    print("=" * 72)

    def build(n_followers: int, n_mercs: int, aspirant_wealth: float) -> Agent:
        world.create_world(size=16)
        world_state["monarchy_on"] = True; world_state["monarchs"] = {}
        world_state["leadership_on"] = True; world_state["leaders"] = {}
        members = {f"F{i}" for i in range(n_followers)} | {"Chief"}
        _settlement("S001", (8, 8), members)
        _settled("Chief", (8, 8), money=2.0)
        for i in range(n_followers):
            f = _settled(f"F{i}", (8, 9), money=1.0)
            _trusts(f, "Chief", leadership.FORM_TRUST)        # a loyal following (M3.2)
        leadership.update(world_state, 1)
        rich = _settled("Rich", (9, 9), sid=None, money=aspirant_wealth)
        for i in range(n_mercs):
            _settled(f"Merc{i}", (10, 10), sid=None, money=0.5)
        return rich

    # A richer aspirant (wealth 200!) but only 3 mercs in range vs a 4-strong loyal following.
    rich = build(n_followers=4, n_mercs=3, aspirant_wealth=200.0)
    res = monarchy.attempt_conquest(world_state, rich, "S001", 1)
    print(f"  Chief leads 4 loyal followers. Rich (wealth 200 — the WEALTHIEST) musters only "
          f"{res['attackers']} fighters:")
    print(f"    -> {res['attackers']} attackers vs {res['defenders']} loyal defenders "
          f"({res['kind']}) -> won={res['won']}  (LOYALTY REPELS the richer force)")
    assert not res["won"] and "S001" not in world_state["monarchs"], \
        "a trusted leader's followers must repel a smaller bought force, however rich the attacker"

    # Same leader, but an overwhelming bought force (7 fighters > 4 followers) takes the crown.
    rich = build(n_followers=4, n_mercs=7, aspirant_wealth=200.0)
    res2 = monarchy.attempt_conquest(world_state, rich, "S001", 1)
    rec = world_state["monarchs"].get("S001")
    print(f"  same town, Rich now musters {res2['attackers']} fighters:")
    print(f"    -> {res2['attackers']} attackers vs {res2['defenders']} loyal defenders -> "
          f"won={res2['won']} -> MONARCH {rec['monarch']}  (overwhelming FORCE wins)")
    print(f"    -> the trust-leader record SURVIVES (consent persists): leaders[S001] = "
          f"{world_state['leaders']['S001']['leader']} — a powerless figurehead under the crown")
    assert res2["won"] and rec["monarch"] == "Rich", "an overwhelming force must win"
    assert world_state["leaders"]["S001"]["leader"] == "Chief", \
        "conquest rules by force but does not erase consent (the two roles coexist; force takes precedence)"
    print("  -> the outcome turns on MUSTERED FORCE, not wealth: loyalty repels the weak assault,")
    print("     overwhelming wealth wins. A real fight, not a foregone wealth-max.  PASS\n")


def demo_c_monarch_rules_by_force() -> None:
    print("=" * 72)
    print("DEMO C — MONARCH RULES BY FORCE: levies WITHOUT consent (contrast M3.3's consent tax)")
    print("=" * 72)
    world.create_world(size=12)
    world_state["monarchy_on"] = True
    world_state["leaders"] = {}                      # NO legitimate leader, NO trust, NO consent
    world_state["monarchs"] = {"S001": {"monarch": "King", "since": 1, "garrison": {"G1", "G2"}}}
    _settlement("S001", (6, 6), {"Sub1", "Sub2", "King"})
    king = _settled("King", (6, 6), money=5.0)
    sub1 = _settled("Sub1", (6, 7), money=20.0); sub2 = _settled("Sub2", (7, 6), money=15.0)
    print(f"  S001 has a MONARCH (King) but NO trust-leader and NO consent. Subjects: "
          f"Sub1={_wealth(sub1):.0f}, Sub2={_wealth(sub2):.0f}")
    w0 = _wealth(king)
    ev = monarchy.levy(world_state, 2)
    print(f"    -> {ev[0]}")
    print(f"    -> Sub1 {20.0:.0f}->{_wealth(sub1):.0f}, Sub2 {15.0:.0f}->{_wealth(sub2):.0f}; "
          f"King {w0:.0f}->{_wealth(king):.0f} (extracted to the CROWN, not redistributed)")
    assert _wealth(king) > w0 and _wealth(sub1) < 20.0, "a monarch extracts wealth by force"
    assert ev and "no consent" in ev[0], "the monarch's levy needs no consent (vs M3.3)"
    print("  -> rule by force: the levy needs NO legitimate leader and NO consent, and flows to the")
    print("     CROWN (extractive), unlike M3.3's consent-based, redistributive, self-limiting tax.  PASS\n")


def demo_d_crown_is_losable() -> None:
    print("=" * 72)
    print("DEMO D — THE CROWN IS LOSABLE: a stronger later aspirant OVERTHROWS the monarch")
    print("=" * 72)
    world.create_world(size=16)
    world_state["monarchy_on"] = True; world_state["leaders"] = {}
    _settlement("S001", (8, 8), {"Sub"})
    _settled("Sub", (8, 8), money=1.0)
    # An incumbent monarch with a small standing garrison of 2.
    world_state["monarchs"] = {"S001": {"monarch": "OldKing", "since": 1, "garrison": {"G1", "G2"}}}
    _settled("OldKing", (8, 8), sid=None, money=2.0)
    _settled("G1", (8, 9), sid=None, money=0.5); _settled("G2", (9, 8), sid=None, money=0.5)
    # A richer challenger who can muster a bigger army (4 mercs > the 2-strong garrison).
    usurper = _settled("Usurper", (9, 9), sid=None, money=40.0)
    for i in range(4):
        _settled(f"Merc{i}", (10, 10), sid=None, money=0.5)
    print(f"  incumbent MONARCH OldKing defends with a garrison of 2.")
    res = monarchy.attempt_conquest(world_state, usurper, "S001", 5)
    rec = world_state["monarchs"]["S001"]
    print(f"    -> Usurper musters {res['attackers']} fighters vs the {res['defenders']}-strong "
          f"garrison -> won={res['won']}")
    print(f"    -> monarchs[S001] = monarch {rec['monarch']}, since {rec['since']} (succession reset)")
    assert res["won"] and rec["monarch"] == "Usurper" and rec["since"] == 5, \
        "a stronger force must overthrow the monarch (the crown is not permanent)"
    print("  -> no permanent crown: a stronger army takes it by the SAME mechanic. Contested "
          "succession.  PASS\n")


def demo_e_war_is_costly() -> None:
    print("=" * 72)
    print("DEMO E — WAR IS COSTLY: fighting KILLS real agents on both sides")
    print("=" * 72)
    world.create_world(size=16)
    world_state["monarchy_on"] = True; world_state["leadership_on"] = True
    world_state["leaders"] = {}; world_state["monarchs"] = {}
    _settlement("S001", (8, 8), {"Chief", "F0", "F1", "F2", "F3"})
    _settled("Chief", (8, 8), money=2.0)
    for i in range(4):
        f = _settled(f"F{i}", (8, 9), money=1.0); _trusts(f, "Chief", leadership.FORM_TRUST)
    leadership.update(world_state, 1)
    rich = _settled("Rich", (9, 9), sid=None, money=60.0)
    for i in range(6):
        _settled(f"Merc{i}", (10, 10), sid=None, money=0.5)
    alive_before = sum(1 for a in world_state["agents"] if a.alive)
    res = monarchy.attempt_conquest(world_state, rich, "S001", 3)
    alive_after = sum(1 for a in world_state["agents"] if a.alive)
    fallen = res["att_dead"] + res["def_dead"]
    deaths_in_events = [e for e in world_state["events"] if "fell in battle" in e]
    print(f"  a defended assault ({res['attackers']} attackers vs {res['defenders']} defenders):")
    print(f"    -> {len(res['att_dead'])} attackers + {len(res['def_dead'])} defenders FELL "
          f"({sorted(fallen)})")
    print(f"    -> living agents {alive_before} -> {alive_after}; battle DEATH events logged: "
          f"{len(deaths_in_events)}")
    assert alive_after == alive_before - len(fallen) and len(fallen) > 0, "war must kill real agents"
    assert all(not _living_or_none(n) for n in fallen), "the fallen are actually dead"
    assert len(deaths_in_events) == len(fallen), "each battle death is a logged civilizational event"
    print("  -> force is DESTROYED, not free: real agents die on both sides, logged like any "
          "death (respawn queued).  PASS\n")


def _living_or_none(name: str) -> bool:
    a = next((x for x in world_state["agents"] if x.name == name), None)
    return bool(a and a.alive)


def demo_f_zero_cost_determinism_and_v1() -> None:
    print("=" * 72)
    print("DEMO F — ZERO added LLM/RNG; deterministic + reproducible; OFF -> v1 byte-identical")
    print("=" * 72)

    def battle_trajectory() -> list[str]:
        world.create_world(size=16)
        world_state["monarchy_on"] = True; world_state["leadership_on"] = True
        world_state["leaders"] = {}; world_state["monarchs"] = {}
        _settlement("S001", (8, 8), {"Chief", "F0", "F1", "F2"})
        _settled("Chief", (8, 8), money=2.0)
        for i in range(3):
            f = _settled(f"F{i}", (8, 9), money=1.0); _trusts(f, "Chief", leadership.FORM_TRUST)
        leadership.update(world_state, 1)
        _settled("Rich", (9, 9), sid=None, money=60.0)
        for i in range(6):
            _settled(f"Merc{i}", (10, 10), sid=None, money=0.5)
        return [e for e in monarchy.update(world_state, 2)]

    llm.reset_call_stats()
    st0 = random.getstate()
    traj1 = battle_trajectory()
    stats = llm.get_call_stats()
    rng_untouched = random.getstate() == st0
    traj2 = battle_trajectory()
    print(f"  a full monarchy.update battle: LLM calls = {stats}; RNG untouched = {rng_untouched}")
    print(f"  reproducible on re-run = {traj1 == traj2}")
    for e in traj1:
        print(f"    -> {e}")
    assert stats == {"decision": 0, "strategy": 0}, stats
    assert rng_untouched, "monarchy consumed RNG (would desync v1)"
    assert traj1 == traj2, "the battle trajectory must be deterministic/reproducible under seed"

    def run(flag):
        llm.PROVIDER = "random"
        random.seed(43)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(24, focal_budget=8)
            else:
                main.run_simulation(24, focal_budget=8, monarchy_on=flag)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        assert run(None) == run(False), "monarchy_on=False changed the default run"
    finally:
        llm.PROVIDER = saved
    print("  zero model calls; monarchy draws no RNG and is reproducible; OFF byte-identical to v1.  PASS\n")


def run() -> None:
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        headline_1_wealth_to_force_to_crown()
        headline_2_force_vs_legitimacy()
        demo_c_monarch_rules_by_force()
        demo_d_crown_is_losable()
        demo_e_war_is_costly()
        demo_f_zero_cost_determinism_and_v1()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M3.4 VERIFIED: CONQUEST & MONARCHY is the SECOND source of power — DOMINATION by force, "
          "vs M3.2's consent. Military power EMERGES from wealth via REAL fighters: an aspirant "
          "SPENDS money to muster an army (a broke one musters nobody), and SEIZES a settlement to "
          "become MONARCH (HEADLINE 1). The fight turns on MUSTERED FORCE, not wealth — a trusted "
          "leader's loyal followers REPEL a richer-but-smaller attacker, while an overwhelming "
          "bought force wins (HEADLINE 2): force and legitimacy genuinely collide. A monarch rules "
          "by force — levying WITHOUT consent (contrast M3.3's consent tax, DEMO C); the crown is "
          "LOSABLE to a stronger later army (DEMO D); and war is COSTLY — it kills real agents on "
          "both sides (DEMO E). Zero LLM/RNG, deterministic and reproducible under seed, "
          "byte-identical to v1 when off. The dark climax of the class engine: the rich buy an army "
          "and take the crown.")
    print("=" * 72)


if __name__ == "__main__":
    run()
