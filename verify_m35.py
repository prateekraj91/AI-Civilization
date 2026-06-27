"""
verify_m35.py
=============

Deterministic verification of V2 milestone M3.5: KINGDOMS & VASSALAGE (feudalism). Phase 3
(Institutions), on top of M3.4 (conquest & monarchy), M3.3 (taxation), M3.2 (leadership),
M3.1 (wage labor) and all of Phase 0/1/2.

Run offline (Ollama OFF, no model server, no seed-search, no long Qwen run):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m35.py

The historical step: M3.4 built a MONARCH who seizes ONE settlement by force. M3.5 is the SCALE-UP —
a monarch conquers NEIGHBOURING settlements into a multi-settlement KINGDOM, a two-level FEUDAL
hierarchy (king -> vassal lords -> their settlements), bound by the feudal bargain (tribute + service
up; protection + autonomy down) with CONDITIONAL loyalty (a pushed vassal can break away).

HEADLINE 1 — KINGDOMS FORM BY CONQUEST into a vassal hierarchy: a monarch conquers a neighbour; the
             conquered local ruler becomes a VASSAL; world_state shows a two-level realm; the realm
             grows to 3+ settlements under one king.
HEADLINE 2 — THE FEUDAL BARGAIN: tribute CASCADES UP (members -> vassal -> king, conserved) and the
             king can MUSTER a loyal vassal's fighters (realm strength = sum of loyal vassals' forces).
HEADLINE 3 — LOYALTY IS CONDITIONAL: an over-taxed vassal's loyalty erodes and it BREAKS AWAY (with
             hysteresis); a fairly-treated vassal stays. Same vassal, two treatments, two fates.
DEMO D — CONSOLIDATION: across turns a realm grows by conquest and the map consolidates (settlements
         under kings rise; independents fall).
DEMO E — ZERO added LLM/RNG; deterministic + reproducible under seed; kingdoms OFF -> v1 identical.
"""

from __future__ import annotations

import contextlib
import io
import random

import economy
import kingdoms
import leadership
import llm
import main
import monarchy
import world
from agents import Agent
from world import world_state


def _settled(name: str, pos: tuple[int, int], sid: str | None = None, **kw) -> Agent:
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


def _find(name: str) -> Agent:
    return next(a for a in world_state["agents"] if a.name == name)


def _settlements(specs: dict[str, tuple]) -> None:
    """specs: sid -> (center, members set)."""
    world_state["settlements"] = {sid: {"id": sid, "center": c, "members": m, "founded": 0}
                                  for sid, (c, m) in specs.items()}


def _make_town(sid: str, center: tuple[int, int], chief: str, followers: list[str],
               member_wealth: float = 1.0) -> None:
    """A settlement with a trust-leader (M3.2): `chief` led by `followers` who trust it."""
    members = {chief} | set(followers)
    world_state["settlements"][sid] = {"id": sid, "center": center, "members": members, "founded": 0}
    _settled(chief, center, sid=sid, money=2.0)
    for i, f in enumerate(followers):
        a = _settled(f, (center[0], center[1] + 1), sid=sid, money=member_wealth)
        _trusts(a, chief, leadership.FORM_TRUST)


def _fresh(size: int = 30) -> None:
    world.create_world(size=size)
    world_state["monarchy_on"] = True
    world_state["kingdoms_on"] = True
    world_state["leadership_on"] = True
    world_state["leaders"] = {}
    world_state["monarchs"] = {}
    world_state["kingdoms"] = {}
    world_state["settlements"] = {}


# ===========================================================================
def headline_1_formation_into_hierarchy() -> None:
    print("=" * 72)
    print("HEADLINE 1 — KINGDOMS FORM BY CONQUEST: a monarch vassalises neighbours into a realm")
    print("=" * 72)
    _fresh()
    # The king: monarch of his home seat S001, wealthy enough to field a royal host.
    world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5), "members": {"King"}, "founded": 0}
    king = _settled("King", (5, 5), sid="S001", money=200.0)
    world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
    # Two neighbouring trust-led towns, each with a chief + 2 loyal followers (loyalty defends).
    _make_town("S002", (9, 9), "Chief2", ["A2", "B2"])
    _make_town("S003", (11, 6), "Chief3", ["A3", "B3"])
    leadership.update(world_state, 0)  # cohere the followings (M3.2)
    # Mercenaries for the king to hire (poor, in range of his seat).
    for i in range(6):
        _settled(f"KM{i}", (4 + i % 3, 4), sid=None, money=0.5)

    print(f"  King is monarch of S001 (wealth {_wealth(king):.0f}). Neighbours: S002 (Chief2 + 2 "
          f"followers), S003 (Chief3 + 2 followers) — both independent, trust-led.")
    r1 = kingdoms.conquer_neighbour(world_state, "King", "S002", 1)
    rec = world_state["kingdoms"]["King"]
    print(f"    -> conquers S002: {r1['host']} royal host vs {r1['defenders']} loyal defenders "
          f"({r1['kind']}) -> won={r1['won']}; Chief2 becomes {('vassal ' + r1['vassal']) if r1['vassal'] else 'n/a'}")
    print(f"    -> realm: king={rec['king']}, settlements={sorted(rec['settlements'])}, "
          f"vassals={rec['vassals']}")
    assert r1["won"] and rec["vassals"].get("S002") == "Chief2", "the conquered local ruler must become a vassal"
    assert "S002" in rec["settlements"], "the conquered settlement joins the realm"
    assert world_state["leaders"]["S002"]["leader"] == "Chief2", \
        "the vassal KEEPS its local leadership (local autonomy preserved, not erased)"

    # Refund the king so it can field a second host (the demo isolates formation, not economy).
    king.money = 200.0
    for i in range(6):
        _settled(f"LM{i}", (10, 5), sid=None, money=0.5)  # mercs near S003
    r2 = kingdoms.conquer_neighbour(world_state, "King", "S003", 2)
    print(f"    -> conquers S003: won={r2['won']}; Chief3 becomes vassal {r2['vassal']}")
    print(f"    -> realm now {len(rec['settlements'])} settlements: {sorted(rec['settlements'])}, "
          f"vassals: {rec['vassals']}")
    assert r2["won"] and len(rec["settlements"]) >= 3, "the realm must grow to 3+ settlements"
    assert set(rec["vassals"].values()) == {"Chief2", "Chief3"}, "a two-level hierarchy of vassal lords"
    print("  -> a two-level FEUDAL KINGDOM (king -> vassal lords -> their settlements) formed by "
          "conquest, the vassals ruling on the king's behalf.  PASS\n")


def headline_2_tribute_and_service() -> None:
    print("=" * 72)
    print("HEADLINE 2 — THE FEUDAL BARGAIN: tribute CASCADES UP + a vassal owes military SERVICE")
    print("=" * 72)
    _fresh()
    # A standing realm: King (S001) with one vassal lord Chief ruling S002, whose members are wealthy.
    world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5), "members": {"King"}, "founded": 0}
    world_state["settlements"]["S002"] = {"id": "S002", "center": (9, 9),
                                          "members": {"Chief", "Rich1", "Rich2"}, "founded": 0}
    king = _settled("King", (5, 5), sid="S001", money=10.0)
    world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
    chief = _settled("Chief", (9, 9), sid="S002", money=10.0)
    rich1 = _settled("Rich1", (9, 10), sid="S002", money=25.0)
    rich2 = _settled("Rich2", (10, 9), sid="S002", money=15.0)
    world_state["leaders"]["S002"] = {"leader": "Chief", "followers": {"Rich1", "Rich2"}, "since": 0}
    world_state["kingdoms"]["King"] = {"king": "King", "home": "S001",
                                       "settlements": {"S001", "S002"}, "vassals": {"S002": "Chief"},
                                       "founded": 0, "discontent": {"Chief": 0}}
    _trusts(chief, "King", kingdoms.LOYAL_TRUST)  # a loyal vassal (sworn fealty)
    world_state["tribute_rate"] = kingdoms.DEFAULT_KING_SHARE  # a fair crown

    total0 = sum(_wealth(a) for a in world_state["agents"] if a.alive)
    w_member0 = _wealth(rich1) + _wealth(rich2)
    w_chief0, w_king0 = _wealth(chief), _wealth(king)
    print(f"  realm: KING (wealth {w_king0:.0f}) <- vassal Chief (wealth {w_chief0:.0f}) <- S002 "
          f"members Rich1={_wealth(rich1):.0f}, Rich2={_wealth(rich2):.0f}")
    ev = kingdoms.tribute(world_state, 1)
    print(f"    -> {ev[0]}")
    print(f"    -> members {w_member0:.1f}->{_wealth(rich1) + _wealth(rich2):.1f}, "
          f"Chief {w_chief0:.1f}->{_wealth(chief):.1f}, KING {w_king0:.1f}->{_wealth(king):.1f}")
    assert _wealth(king) > w_king0 and _wealth(chief) > w_chief0, "tribute flows UP to both vassal and king"
    assert _wealth(rich1) < 25.0 and _wealth(rich2) < 15.0, "the members are levied (bottom of the cascade)"
    total1 = sum(_wealth(a) for a in world_state["agents"] if a.alive)
    assert abs(total1 - total0) < 1e-9, "tribute only MOVES wealth — total conserved across the cascade"
    print(f"  -> wealth conserved across the cascade ({total0:.1f} == {total1:.1f}); it concentrates "
          f"at the crown THROUGH the vassal.")

    # Military service: the king CALLS his loyal vassal — the realm host = king's force + vassal's.
    for i in range(4):
        _settled(f"KM{i}", (4, 4), sid=None, money=0.5)   # mercs near the king
    for i in range(4):
        _settled(f"VM{i}", (9, 8), sid=None, money=0.5)   # mercs near the vassal's seat
    king.money = 20.0  # a war chest for the crown
    host = kingdoms.muster_realm(world_state, king, exclude=set())
    callers = sorted({h.name for h in host})
    print(f"  the king calls the muster: realm host = {len(host)} fighters {callers}")
    assert len(host) > monarchy.max_fighters(_find("KM0")), "the host includes fighters beyond the king's own"
    assert any(h.name.startswith("VM") for h in host), \
        "a LOYAL vassal answers with its own contingent (military service flows up)"

    # A broken-away vassal answers NO call: drop the realm tie and re-muster.
    world_state["kingdoms"]["King"]["vassals"].clear()
    world_state["kingdoms"]["King"]["settlements"].discard("S002")
    for a in world_state["agents"]:  # reset spent merc/king money for a clean second muster
        if a.name.startswith(("KM", "VM")):
            a.money = 0.5
    king.money = 20.0
    host2 = kingdoms.muster_realm(world_state, king, exclude=set())
    print(f"  after Chief breaks away, the king calls again: host = {len(host2)} fighters "
          f"{sorted({h.name for h in host2})} (no vassal contingent)")
    assert not any(h.name.startswith("VM") for h in host2), \
        "a vassal no longer in the realm owes no service — realm strength = SUM of LOYAL vassals' forces"
    print("  -> tribute cascades up and loyal vassals serve; a realm's strength is its loyal "
          "vassals' pooled force.  PASS\n")


def headline_3_conditional_loyalty() -> None:
    print("=" * 72)
    print("HEADLINE 3 — LOYALTY IS CONDITIONAL: a grasping crown loses a vassal; a fair one holds it")
    print("=" * 72)

    def run_realm(tribute_rate: float, turns: int) -> tuple[bool, list[int]]:
        _fresh()
        world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5), "members": {"King"}, "founded": 0}
        world_state["settlements"]["S002"] = {"id": "S002", "center": (9, 9),
                                              "members": {"Chief", "Rich"}, "founded": 0}
        _settled("King", (5, 5), sid="S001", money=10.0)
        world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
        chief = _settled("Chief", (9, 9), sid="S002", money=10.0)
        _settled("Rich", (9, 10), sid="S002", money=25.0)
        world_state["leaders"]["S002"] = {"leader": "Chief", "followers": {"Rich"}, "since": 0}
        world_state["kingdoms"]["King"] = {"king": "King", "home": "S001",
                                           "settlements": {"S001", "S002"}, "vassals": {"S002": "Chief"},
                                           "founded": 0, "discontent": {"Chief": 0}}
        _trusts(chief, "King", kingdoms.LOYAL_TRUST)  # starts loyal (sworn fealty)
        world_state["tribute_rate"] = tribute_rate
        trust_trace = [chief.relationships["King"]["trust"]]
        in_realm = []
        for t in range(1, turns + 1):
            kingdoms.tribute(world_state, t)
            kingdoms._check_breakaways(world_state, t)
            trust_trace.append(chief.relationships.get("King", {}).get("trust", 0))
            in_realm.append("S002" in world_state["kingdoms"].get("King", {}).get("settlements", set()))
        return in_realm, trust_trace

    # GRASPING crown (share 0.9, far above the consent band): the vassal's loyalty erodes and breaks.
    in_realm_hi, trace_hi = run_realm(tribute_rate=0.9, turns=4)
    print(f"  GRASPING king (tribute share 0.9 >> consent {kingdoms.KING_CONSENT}):")
    print(f"    Chief's trust in the king by turn: {trace_hi}")
    print(f"    S002 still in the realm after each turn: {in_realm_hi}")
    assert in_realm_hi[0] is True, "HYSTERESIS: a vassal must NOT break away on the very first hard turn"
    assert in_realm_hi[-1] is False, "a sufficiently disloyal vassal must BREAK AWAY"
    broke_turn = in_realm_hi.index(False) + 1
    print(f"    -> Chief BROKE AWAY on turn {broke_turn} (after sustained disloyalty, not a single dip)")

    # FAIR crown (share 0.25, within consent): no resentment, the vassal stays loyal indefinitely.
    in_realm_lo, trace_lo = run_realm(tribute_rate=0.25, turns=4)
    print(f"  FAIR king (tribute share 0.25 <= consent {kingdoms.KING_CONSENT}):")
    print(f"    Chief's trust in the king by turn: {trace_lo}")
    print(f"    S002 still in the realm after each turn: {in_realm_lo}")
    assert all(in_realm_lo), "a fairly-treated vassal must STAY in the realm"
    print("  -> same vassal, two treatments, two fates: royal power is CONTINGENT on not "
          "over-grasping — feudal loyalty is conditional.  PASS\n")


def demo_d_consolidation() -> None:
    print("=" * 72)
    print("DEMO D — CONSOLIDATION: a realm grows by conquest; the map's independents fall")
    print("=" * 72)
    _fresh()
    # One rich monarch and four weak independent neighbours within reach.
    world_state["settlements"]["S001"] = {"id": "S001", "center": (10, 10), "members": {"King"}, "founded": 0}
    king = _settled("King", (10, 10), sid="S001", money=5000.0)
    world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
    for n, c in [("S002", (13, 13)), ("S003", (7, 13)), ("S004", (13, 7)), ("S005", (7, 7))]:
        _make_town(n, c, f"Chief{n[-1]}", [f"A{n[-1]}", f"B{n[-1]}"])
    leadership.update(world_state, 0)
    # Plenty of mercenaries scattered so the king can keep fielding hosts as the realm grows.
    for i in range(40):
        _settled(f"M{i}", (8 + i % 5, 8 + (i // 5) % 5), sid=None, money=0.5)

    independent0 = sum(1 for s in world_state["settlements"] if kingdoms.realm_of(world_state, s) is None)
    print(f"  start: 5 settlements, {independent0} independent, 0 under any king")
    for t in range(1, 7):
        king.money += 200.0  # tribute/plunder keeps the war chest funded (territorial compounding)
        kingdoms.update(world_state, t)
        rec = world_state["kingdoms"].get("King", {})
        held = len(rec.get("settlements", set()))
        indep = sum(1 for s in world_state["settlements"] if kingdoms.realm_of(world_state, s) is None)
        print(f"    turn {t}: realm holds {held} settlements; {indep} independent remain")
    rec = world_state["kingdoms"]["King"]
    indep_end = sum(1 for s in world_state["settlements"] if kingdoms.realm_of(world_state, s) is None)
    print(f"  end: realm {sorted(rec['settlements'])} ({len(rec['settlements'])} settlements, "
          f"{len(rec['vassals'])} vassals); {indep_end} independent remain")
    assert len(rec["settlements"]) > 1 and indep_end < independent0, \
        "the realm must grow and the count of independent settlements must fall (consolidation)"
    print("  -> pooled wealth funds conquest, the realm grows, the map consolidates: many small "
          "settlements -> fewer, larger kingdoms.  PASS\n")


def demo_e_zero_cost_determinism_and_v1() -> None:
    print("=" * 72)
    print("DEMO E — ZERO added LLM/RNG; deterministic + reproducible; OFF -> v1 byte-identical")
    print("=" * 72)

    def realm_trajectory() -> list[str]:
        _fresh()
        world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5), "members": {"King"}, "founded": 0}
        _settled("King", (5, 5), sid="S001", money=500.0)
        world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
        _make_town("S002", (9, 9), "Chief2", ["A2", "B2"])
        leadership.update(world_state, 0)
        for i in range(8):
            _settled(f"KM{i}", (4 + i % 3, 4), sid=None, money=0.5)
        world_state["tribute_rate"] = 0.9
        out = []
        for t in range(1, 4):
            out.extend(kingdoms.update(world_state, t))
        return out

    llm.reset_call_stats()
    st0 = random.getstate()
    traj1 = realm_trajectory()
    stats = llm.get_call_stats()
    rng_untouched = random.getstate() == st0
    traj2 = realm_trajectory()
    print(f"  a full kingdoms.update arc: LLM calls = {stats}; RNG untouched = {rng_untouched}")
    print(f"  reproducible on re-run = {traj1 == traj2}")
    for e in traj1:
        print(f"    -> {e}")
    assert stats == {"decision": 0, "strategy": 0}, stats
    assert rng_untouched, "kingdoms consumed RNG (would desync v1)"
    assert traj1 == traj2, "the realm trajectory must be deterministic/reproducible under seed"

    def run(flag):
        llm.PROVIDER = "random"
        random.seed(43)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(24, focal_budget=8)
            else:
                main.run_simulation(24, focal_budget=8, kingdoms_on=flag)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        assert run(None) == run(False), "kingdoms_on=False changed the default run"
    finally:
        llm.PROVIDER = saved
    print("  zero model calls; kingdoms draws no RNG and is reproducible; OFF byte-identical to v1.  PASS\n")


def run() -> None:
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        headline_1_formation_into_hierarchy()
        headline_2_tribute_and_service()
        headline_3_conditional_loyalty()
        demo_d_consolidation()
        demo_e_zero_cost_determinism_and_v1()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M3.5 VERIFIED: KINGDOMS & VASSALAGE is the SCALE-UP of M3.4 — single settlements become "
          "multi-settlement FEUDAL KINGDOMS. A monarch conquers NEIGHBOURING settlements (reusing the "
          "M3.4 fight) into a two-level realm (king -> vassal lords -> their settlements), the "
          "conquered local ruler ruling on as a VASSAL with local autonomy (HEADLINE 1). The feudal "
          "bargain runs both ways: TRIBUTE cascades UP (members -> vassal -> king, conserving wealth) "
          "and loyal vassals owe military SERVICE (realm strength = the sum of loyal vassals' forces) "
          "(HEADLINE 2). Loyalty is CONDITIONAL: a grasping crown erodes a vassal's trust until it "
          "BREAKS AWAY (with hysteresis), while a fair crown holds its realm — royal power is "
          "contingent, not absolute (HEADLINE 3). Pooled tribute funds further conquest and the map "
          "CONSOLIDATES (DEMO D). Zero LLM/RNG, deterministic and reproducible under seed, "
          "byte-identical to v1 when off (DEMO E). The feudal kingdoms inter-kingdom war (M3.6) will "
          "set against each other.")
    print("=" * 72)


if __name__ == "__main__":
    run()
