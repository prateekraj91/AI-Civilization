"""
verify_m36.py
=============

Deterministic verification of V2 milestone M3.6: INTER-KINGDOM WAR & EMPIRE — the CLIMAX of Phase 3
(Institutions), on top of M3.5 (kingdoms & vassalage), M3.4 (conquest & monarchy), M3.3 (taxation),
M3.2 (leadership), M3.1 (wage labor) and all of Phase 0/1/2.

Run offline (Ollama OFF, no model server, no seed-search, no long Qwen run):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m36.py

The historical step: M3.5 built feudal KINGDOMS. M3.6 sets them against EACH OTHER in WAR — whole
feudal hosts clash, and the loser's king is SUBJUGATED into the victor's EMPIRE (a third level of the
same hierarchy: emperor -> subject-king -> vassal-lords -> settlements). And empires are no more
permanent than kingdoms: a subject-king's loyalty is just as CONDITIONAL, so an over-taxing or
weakening emperor FRAGMENTS — power is contingent at EVERY level.

HEADLINE 1 — LOYALTY DECIDES THE WAR: a RICHER kingdom whose vassals are DISLOYAL fields a SMALLER
             loyal host and LOSES to a POORER kingdom whose vassals all muster; flip the same
             kingdom's loyalty and it fields a full host and WINS. Same kingdoms, loyalty flipped,
             opposite results — governance, not wealth, decides war.
HEADLINE 2 — SUBJUGATION -> EMPIRE: a defeated king becomes a subject-king (high-level vassal);
             world_state shows a multi-level empire; tribute cascades through the new level
             (settlement -> lord -> subject-king -> emperor, conserved); the emperor can muster the
             loyal subject-king's whole host for a further war.
HEADLINE 3 — FRAGMENTATION (rise AND fall): an over-taxing emperor loses a subject-king, who BREAKS
             AWAY and reclaims independence (with hysteresis); a fair emperor holds the empire.
DEMO D — WAR IS COSTLY (both armies bleed) + WINNABLE-ASSAULT (no suicidal wars launched).
DEMO E — ZERO added LLM/RNG; deterministic + reproducible under seed; empire OFF -> v1 identical.
"""

from __future__ import annotations

import contextlib
import io
import random

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from sim import empire
from sim import kingdoms
from llm import llm
import main
from sim import world
from sim.agents import Agent
from sim.world import world_state


def _settled(name: str, pos: tuple[int, int], sid: str | None = None, **kw) -> Agent:
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


def _find(name: str) -> Agent:
    return next(a for a in world_state["agents"] if a.name == name)


def _fresh(size: int = 80) -> None:
    world.create_world(size=size)
    for fl in ("monarchy_on", "kingdoms_on", "empire_on", "leadership_on"):
        world_state[fl] = True
    for k in ("leaders", "monarchs", "kingdoms", "empires", "settlements"):
        world_state[k] = {}


def _mercs(prefix: str, near: tuple[int, int], n: int) -> None:
    """A private merc pool of `n` poor agents within muster range of `near` only (well-separated)."""
    for i in range(n):
        _settled(f"{prefix}{i}", (near[0] + (i % 2), near[1] + 2), sid=None, money=0.5)


def _realm(king: str, kmoney: float, home_c: tuple[int, int],
           seats: list[tuple[str, tuple[int, int], str]], vassal_loyal: bool) -> None:
    """A king (monarch of its home seat) with vassal lords ruling far-flung seats (loyal or not)."""
    home = f"{king}_home"
    world_state["settlements"][home] = {"id": home, "center": home_c, "members": {king}, "founded": 0}
    _settled(king, home_c, sid=home, money=kmoney)
    world_state["monarchs"][home] = {"monarch": king, "since": 0, "garrison": set()}
    setts, vassals = {home}, {}
    for sid, c, chief in seats:
        world_state["settlements"][sid] = {"id": sid, "center": c, "members": {chief}, "founded": 0}
        ch = _settled(chief, c, sid=sid, money=40.0)
        _trusts(ch, king, kingdoms.LOYAL_TRUST if vassal_loyal else kingdoms.LOYAL_TRUST - 3)
        world_state["leaders"][sid] = {"leader": chief, "followers": set(), "since": 0}
        vassals[sid] = chief
        setts.add(sid)
    world_state["kingdoms"][king] = {"king": king, "home": home, "settlements": setts,
                                     "vassals": vassals, "founded": 0,
                                     "discontent": {v: 0 for v in vassals.values()}}


def _two_kingdoms(rich_loyal: bool) -> None:
    """Rich (wealthy, 2 vassals — loyalty per flag) adjacent to Poor (modest, 2 LOYAL vassals)."""
    _fresh()
    _realm("Rich", 500.0, (10, 10), [("RV1", (40, 10), "RC1"), ("RV2", (10, 40), "RC2")], rich_loyal)
    _realm("Poor", 50.0, (18, 10), [("PV1", (50, 10), "PC1"), ("PV2", (18, 40), "PC2")], True)
    _mercs("RKm", (10, 10), 4); _mercs("RC1m", (40, 10), 4); _mercs("RC2m", (10, 40), 4)
    _mercs("PKm", (18, 10), 3); _mercs("PC1m", (50, 10), 3); _mercs("PC2m", (18, 40), 3)


# ===========================================================================
def headline_1_loyalty_decides_the_war() -> None:
    print("=" * 72)
    print("HEADLINE 1 — LOYALTY DECIDES THE WAR: same kingdoms, loyalty flipped, opposite results")
    print("=" * 72)
    # Disloyal Rich: richer, but its vassals won't muster -> small host -> LOSES to loyal Poor.
    _two_kingdoms(rich_loyal=False)
    rich, poor = _find("Rich"), _find("Poor")
    hr, hp = empire.imperial_host_size(world_state, rich), empire.imperial_host_size(world_state, poor)
    print(f"  Rich kingdom: wealth {_wealth(rich):.0f}, but its 2 vassals are DISLOYAL -> loyal host {hr}")
    print(f"  Poor kingdom: wealth {_wealth(poor):.0f}, its 2 vassals are LOYAL and muster -> loyal host {hp}")
    print(f"  (Poor opportunistically attacks the richer-but-brittle Rich; neighbours of Poor = "
          f"{empire._kingdom_neighbours(world_state, 'Poor')})")
    r1 = empire.wage_war(world_state, "Poor", "Rich", 1)
    print(f"    -> WAR: Poor's {r1['att_host']} loyal host vs Rich's {r1['def_host']} -> Poor won={r1['won']} "
          f"({len(r1['att_dead'])}+{len(r1['def_dead'])} fell)")
    assert r1["won"] and r1["att_host"] > r1["def_host"], "the POORER but LOYAL kingdom must win"
    assert _wealth(rich) > _wealth(poor), "the LOSER was the RICHER kingdom — wealth did not decide"
    assert "Rich" in world_state["empires"]["Poor"]["subject_kings"], "the defeated richer king is subjugated"
    print("    -> the RICHER kingdom LOST: disloyal vassals withheld their swords (wealth funds, "
          "loyalty fields).")

    # Flip: SAME Rich kingdom, vassals now LOYAL -> full host -> WINS the same matchup.
    _two_kingdoms(rich_loyal=True)
    rich, poor = _find("Rich"), _find("Poor")
    hr, hp = empire.imperial_host_size(world_state, rich), empire.imperial_host_size(world_state, poor)
    print(f"  Now the SAME Rich kingdom (wealth {_wealth(rich):.0f}) with its vassals LOYAL -> host {hr} "
          f"(vs Poor's {hp})")
    r2 = empire.wage_war(world_state, "Rich", "Poor", 1)
    print(f"    -> WAR: Rich's {r2['att_host']} loyal host vs Poor's {r2['def_host']} -> Rich won={r2['won']} "
          f"({len(r2['att_dead'])}+{len(r2['def_dead'])} fell)")
    assert r2["won"] and r2["att_host"] > r2["def_host"], "with loyal vassals the richer kingdom now WINS"
    assert "Poor" in world_state["empires"]["Rich"]["subject_kings"], "now Rich subjugates Poor"
    print("  -> same kingdoms, loyalty flipped, OPPOSITE results: GOVERNANCE, not wealth, decides "
          "the war — a rich tyrant's realm is brittle.  PASS\n")


def headline_2_subjugation_into_empire() -> None:
    print("=" * 72)
    print("HEADLINE 2 — SUBJUGATION -> EMPIRE: a defeated king becomes a multi-level subject-king")
    print("=" * 72)
    _fresh(size=40)
    # Emperor (E) — a subject-king (King) ruling a vassal-lord (Chief) over a rich member (Rich).
    world_state["settlements"]["E"] = {"id": "E", "center": (5, 5), "members": {"Emp"}, "founded": 0}
    emp = _settled("Emp", (5, 5), sid="E", money=10.0)
    world_state["monarchs"]["E"] = {"monarch": "Emp", "since": 0, "garrison": set()}
    world_state["settlements"]["K"] = {"id": "K", "center": (9, 9), "members": {"King"}, "founded": 0}
    king = _settled("King", (9, 9), sid="K", money=10.0)
    world_state["monarchs"]["K"] = {"monarch": "King", "since": 0, "garrison": set()}
    world_state["settlements"]["V"] = {"id": "V", "center": (12, 12), "members": {"Chief", "Rich"}, "founded": 0}
    chief = _settled("Chief", (12, 12), sid="V", money=10.0)
    rich = _settled("Rich", (12, 13), sid="V", money=40.0)
    world_state["leaders"]["V"] = {"leader": "Chief", "followers": {"Rich"}, "since": 0}
    _trusts(chief, "King", kingdoms.LOYAL_TRUST)
    world_state["kingdoms"]["King"] = {"king": "King", "home": "K", "settlements": {"K", "V"},
                                       "vassals": {"V": "Chief"}, "founded": 0, "discontent": {"Chief": 0}}
    world_state["kingdoms"]["Emp"] = {"king": "Emp", "home": "E", "settlements": {"E"},
                                      "vassals": {}, "founded": 0, "discontent": {}}
    # The subjugation event (the emperor has just won the war over King).
    empire._subjugate(world_state, emp, "King", 0)
    _trusts(king, "Emp", kingdoms.LOYAL_TRUST)
    world_state["tribute_rate"] = kingdoms.DEFAULT_KING_SHARE
    world_state["empire_share"] = empire.DEFAULT_EMPIRE_SHARE
    print("  empire structure (a multi-level hierarchy):")
    print(f"    EMPEROR Emp -> subject-king {list(world_state['empires']['Emp']['subject_kings'])} "
          f"-> King's vassal-lords {world_state['kingdoms']['King']['vassals']} -> settlement members")
    assert world_state["empires"]["Emp"]["subject_kings"].get("King") is not None, "King is the emperor's subject-king"
    assert world_state["kingdoms"]["King"]["vassals"] == {"V": "Chief"}, "the subject-king KEEPS his own realm"

    total0 = sum(_wealth(a) for a in world_state["agents"] if a.alive)
    print(f"  tribute cascade (settlement -> lord -> subject-king -> emperor); total wealth {total0:.1f}")
    print(f"    before: Rich {_wealth(rich):.1f}, Chief {_wealth(chief):.1f}, King {_wealth(king):.1f}, "
          f"Emp {_wealth(emp):.1f}")
    kingdoms.tribute(world_state, 1)   # members -> lord -> subject-king (M3.5)
    empire.tribute(world_state, 1)     # subject-king -> emperor (M3.6, the new level)
    total1 = sum(_wealth(a) for a in world_state["agents"] if a.alive)
    print(f"    after:  Rich {_wealth(rich):.1f}, Chief {_wealth(chief):.1f}, King {_wealth(king):.1f}, "
          f"Emp {_wealth(emp):.1f}")
    assert _wealth(emp) > 10.0, "tribute reached the EMPEROR through the new (subject-king) level"
    assert abs(total1 - total0) < 1e-9, "the whole cascade only MOVES wealth — total conserved"
    print(f"    -> wealth conserved across all levels ({total0:.1f} == {total1:.1f}); it concentrates "
          f"at the imperial crown THROUGH the subject-king.")

    # The emperor musters the loyal subject-king's WHOLE host (King + Chief's contingent).
    _mercs("Em", (5, 6), 3); _mercs("Km", (9, 8), 3); _mercs("Cm", (12, 11), 3)
    emp.money = king.money = chief.money = 20.0
    host = empire.imperial_host(world_state, emp, set())
    names = sorted(h.name for h in host)
    print(f"  the emperor calls the imperial muster: {names}")
    assert any(n.startswith("Cm") for n in names), \
        "the emperor mustered the subject-king's vassal contingent too (a WHOLE multi-level host)"
    print("  -> the defeated king is a subject-king in a multi-level EMPIRE; tribute cascades through "
          "the new level and the emperor can field the subject-king's whole host.  PASS\n")


def headline_3_fragmentation() -> None:
    print("=" * 72)
    print("HEADLINE 3 — FRAGMENTATION: an over-taxing emperor loses a subject-king; a fair one holds")
    print("=" * 72)

    def run(share: float, turns: int) -> tuple[list[bool], list[int]]:
        _fresh(size=40)
        world_state["settlements"]["E"] = {"id": "E", "center": (5, 5), "members": {"Emp"}, "founded": 0}
        emp = _settled("Emp", (5, 5), sid="E", money=200.0)
        world_state["monarchs"]["E"] = {"monarch": "Emp", "since": 0, "garrison": set()}
        world_state["settlements"]["K"] = {"id": "K", "center": (9, 9), "members": {"King"}, "founded": 0}
        king = _settled("King", (9, 9), sid="K", money=40.0)
        world_state["monarchs"]["K"] = {"monarch": "King", "since": 0, "garrison": set()}
        world_state["kingdoms"]["King"] = {"king": "King", "home": "K", "settlements": {"K"},
                                           "vassals": {}, "founded": 0, "discontent": {}}
        world_state["kingdoms"]["Emp"] = {"king": "Emp", "home": "E", "settlements": {"E"},
                                          "vassals": {}, "founded": 0, "discontent": {}}
        world_state["empires"]["Emp"] = {"emperor": "Emp", "subject_kings": {"King": {"since": 0}},
                                         "founded": 0, "discontent": {"King": 0}}
        _trusts(king, "Emp", kingdoms.LOYAL_TRUST)
        _mercs("Em", (5, 6), 8)  # a STRONG emperor (so only the over-tax path, not weakening, fires)
        world_state["empire_share"] = share
        in_empire, trace = [], [king.relationships["Emp"]["trust"]]
        for t in range(1, turns + 1):
            king.money = 40.0  # refresh so the imperial levy keeps biting
            empire.tribute(world_state, t)
            empire._check_fragmentation(world_state, t)
            trace.append(king.relationships.get("Emp", {}).get("trust", 0))
            in_empire.append("King" in world_state["empires"].get("Emp", {}).get("subject_kings", {}))
        return in_empire, trace

    grasp, tg = run(0.9, 4)
    print(f"  GRASPING emperor (imperial share 0.9 >> consent {kingdoms.KING_CONSENT}):")
    print(f"    subject-king's trust by turn: {tg}")
    print(f"    King still in the empire after each turn: {grasp}")
    assert grasp[0] is True, "HYSTERESIS: a subject-king must NOT break away on the very first hard turn"
    assert grasp[-1] is False, "a sufficiently disloyal subject-king must BREAK AWAY (reclaim independence)"
    print(f"    -> King BROKE AWAY on turn {grasp.index(False) + 1}, reclaiming his crown and realm")

    fair, tf = run(0.25, 4)
    print(f"  FAIR emperor (imperial share 0.25 <= consent {kingdoms.KING_CONSENT}):")
    print(f"    subject-king's trust by turn: {tf}")
    print(f"    King still in the empire after each turn: {fair}")
    assert all(fair), "a fairly-treated subject-king STAYS in the empire"
    print("  -> same subject-king, two imperial treatments, two fates: empires rise on good "
          "governance and FALL on overreach — no power is permanent.  PASS\n")


def demo_d_costly_and_winnable() -> None:
    print("=" * 72)
    print("DEMO D — WAR IS COSTLY (both armies bleed) + WINNABLE-ASSAULT (no suicidal wars)")
    print("=" * 72)
    _two_kingdoms(rich_loyal=True)
    living0 = sum(1 for a in world_state["agents"] if a.alive)
    r = empire.wage_war(world_state, "Rich", "Poor", 1)
    living1 = sum(1 for a in world_state["agents"] if a.alive)
    print(f"  war Rich->Poor: {len(r['att_dead'])} attacker dead, {len(r['def_dead'])} defender dead; "
          f"living {living0} -> {living1}")
    assert r["att_dead"] and r["def_dead"], "war must kill real agents on BOTH sides"
    assert living1 == living0 - len(r["att_dead"]) - len(r["def_dead"]), "the fallen actually died"

    # Winnable-assault: a kingdom that cannot out-field its neighbour does NOT launch a war.
    _two_kingdoms(rich_loyal=False)  # Rich's host (4) < Poor's (9): Rich must not attack Poor
    rich, poor = _find("Rich"), _find("Poor")
    print(f"  Rich loyal host {empire.imperial_host_size(world_state, rich)} < Poor's "
          f"{empire.imperial_host_size(world_state, poor)} -> Rich must not start a losing war")
    before = len(world_state["events"])
    empire.update(world_state, 2)
    launched_by_rich = [e for e in world_state["events"][before:] if "KING Rich" in e and "war" in e.lower()]
    # Poor (the stronger) DOES attack Rich; Rich (the weaker) never marches on Poor.
    assert not launched_by_rich, "a king does not launch a war it would lose (winnable-assault guard)"
    print(f"  -> only the stronger kingdom (Poor) launched a war; the weaker (Rich) bided its time "
          f"(no suicidal wars).  PASS\n")


def demo_e_zero_cost_determinism_and_v1() -> None:
    print("=" * 72)
    print("DEMO E — ZERO added LLM/RNG; deterministic + reproducible; OFF -> v1 byte-identical")
    print("=" * 72)

    def war_trajectory() -> list[str]:
        _two_kingdoms(rich_loyal=True)
        world_state["empire_share"] = 0.9  # grasp, to also exercise tribute/fragmentation paths
        out = []
        for t in range(1, 4):
            out.extend(empire.update(world_state, t))
        return out

    llm.reset_call_stats()
    st0 = random.getstate()
    traj1 = war_trajectory()
    stats = llm.get_call_stats()
    rng_untouched = random.getstate() == st0
    traj2 = war_trajectory()
    print(f"  a full empire.update arc: LLM calls = {stats}; RNG untouched = {rng_untouched}; "
          f"reproducible = {traj1 == traj2}")
    for e in traj1:
        print(f"    -> {e}")
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats
    assert rng_untouched, "empire consumed RNG (would desync v1)"
    assert traj1 == traj2, "the war trajectory must be deterministic/reproducible under seed"

    def run(flag):
        llm.PROVIDER = "random"
        random.seed(43)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(24, focal_budget=8)
            else:
                main.run_simulation(24, focal_budget=8, empire_on=flag)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        assert run(None) == run(False), "empire_on=False changed the default run"
    finally:
        llm.PROVIDER = saved
    print("  zero model calls; empire draws no RNG and is reproducible; OFF byte-identical to v1.  PASS\n")


def run() -> None:
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        headline_1_loyalty_decides_the_war()
        headline_2_subjugation_into_empire()
        headline_3_fragmentation()
        demo_d_costly_and_winnable()
        demo_e_zero_cost_determinism_and_v1()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M3.6 VERIFIED: INTER-KINGDOM WAR & EMPIRE is the CLIMAX of Phase 3 — feudal KINGDOMS clash "
          "and EMPIRES rise on good governance and fall on overreach. War turns on LOYAL host strength "
          "(king + LOYAL vassals + loyal subject-kings, mustered via the M3.5 machinery), so a RICHER "
          "kingdom with disloyal vassals fields a SMALLER host and LOSES to a POORER kingdom whose "
          "vassals all muster — and flipping that kingdom's loyalty flips the result: governance, not "
          "wealth, decides the war (HEADLINE 1). The defeated king is SUBJUGATED into the victor's "
          "realm as a high-level vassal (a subject-king), forming a multi-level EMPIRE (emperor -> "
          "subject-king -> vassal-lords -> settlements); tribute cascades through the new level "
          "(conserved) and the emperor fields the subject-king's whole host (HEADLINE 2). But a "
          "subject-king's loyalty is just as CONDITIONAL: an over-taxing or weakening emperor loses him "
          "to a breakaway (with hysteresis), while a fair one holds the empire — power is contingent at "
          "EVERY level (HEADLINE 3). War is COSTLY (both armies bleed) and only WINNABLE wars launch "
          "(DEMO D). Zero LLM/RNG, deterministic and reproducible under seed, byte-identical to v1 when "
          "off (DEMO E). The reused fight (monarchy.resolve_battle) and muster (kingdoms.muster_realm) "
          "make M3.6 a thin, faithful CAPSTONE: every power structure Phase 3 built can fall.")
    print("=" * 72)


if __name__ == "__main__":
    run()
