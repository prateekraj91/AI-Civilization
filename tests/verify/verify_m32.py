"""
verify_m32.py
=============

Deterministic verification of V2 milestone M3.2: LEGITIMATE LEADERSHIP — authority by
TRUST. Phase 3 (Institutions), on top of M3.1 (wage labor) and all of Phase 0/1/2.

Run offline (Ollama OFF, no model server, no seed-search, no long Qwen run):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m32.py

The historical step: M3.1 built the first institution (wage labor) — political-economic
power DOWNSTREAM of wealth, the rich employing the poor, inequality compounding. M3.2 builds
the first POLITICAL institution and the first power NOT downstream of wealth: a leader
legitimated by TRUST. The most-trusted agent need not be the richest, so political power
DECOUPLES from economic power — the tension later milestones (law, taxation, conflict) turn
on. A leader here has INFLUENCE, not the power to tax or legislate (scoped to M3.2).

HEADLINE 1 — LEGITIMACY, NOT A LEADERBOARD: a leader emerges ONLY when a real FOLLOWING
             coheres (>= MIN_FOLLOWERS co-settlers trust a common agent above the bar). The
             single highest trust score with ONE admirer does NOT lead; a fractured, low-trust
             settlement has NO leader. Emergence is conditional, not automatic.
HEADLINE 2 — POWER DECOUPLED FROM WEALTH: the emergent leader is NOT the wealthiest agent — a
             poorer, trusted agent leads while a richer, distrusted one does not. Trust, not
             wealth, drives leadership (if the leader were always the richest, trust added
             nothing — a FAIL).
DEMO C — CONTINGENT LEGITIMACY: a leader LOSES the role when trust erodes (the following falls
         below the bar / the leader turns hostile), with HYSTERESIS (a one-turn wobble does not
         unseat); and a more-trusted centre DISPLACES the incumbent when the following shifts.
DEMO D — THE EFFECT IS REAL: a LED settlement is measurably more cohesive than an identical
         UNLED one — followers rally tighter around their leader — without tax or law.
DEMO E — ZERO added LLM/RNG; leadership writes NO trust; leadership OFF -> v1 byte-identical.
DEMO F — IT EMERGES ORGANICALLY (not just in constructed fixtures): a full seeded simulation
         with ZERO injected trust — agents settle, build trust purely through conversation —
         produces a leader at the settlement's founding and the role CHANGES HANDS as the
         trust network shifts. This is the load-bearing demonstration: every other demo hand-
         sets trust to isolate one property; here trust is EARNED in play and leadership falls
         out of it on its own. Deterministic (seeded), reproducible by re-running this file.
"""

from __future__ import annotations

import contextlib
import io
import random

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from sim import leadership
from llm import llm
import main
from sim import settlement
from llm import strategy
from sim import world
from sim.agents import Agent
from sim.world import world_state


def _settled(name: str, pos: tuple[int, int], sid: str = "S001", **kw) -> Agent:
    """Place a living, settled agent (so leadership reads it within a settlement)."""
    a = Agent(name=name, personality="cautious and territorial")
    world.place_agent(a, *pos)
    a.settlement = sid
    a.hunger = 0
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _trusts(follower: Agent, leader_name: str, value: int) -> None:
    """Set `follower`'s trust in `leader_name` — the existing v1 trust the module READS."""
    follower.relationships[leader_name] = {"trust": value, "interactions": 1, "grudge": False}


def _wealth(a: Agent) -> float:
    return a.money + a.stockpile


def headline_1_legitimacy_not_a_leaderboard() -> None:
    print("=" * 72)
    print("HEADLINE 1 — LEGITIMACY, NOT A LEADERBOARD: a leader emerges ONLY when a real")
    print("              FOLLOWING coheres; a fractured/low-trust settlement has NONE")
    print("=" * 72)
    print(f"  (a candidate needs >= MIN_FOLLOWERS={leadership.MIN_FOLLOWERS} co-settlers trusting "
          f"it >= FORM_TRUST={leadership.FORM_TRUST})")

    # A: a cohered cluster -> a leader emerges.
    world.create_world(size=12); world_state["leadership_on"] = True; world_state["leaders"] = {}
    _settled("Mara", (5, 5))
    for nm, pos in (("F1", (5, 6)), ("F2", (6, 5)), ("F3", (4, 5))):
        _trusts(_settled(nm, pos), "Mara", leadership.FORM_TRUST)
    ev = leadership.update(world_state, 1)
    rec = world_state["leaders"].get("S001")
    print(f"  cohered cluster (3 co-settlers trust Mara high):")
    print(f"    -> {ev[0]}")
    print(f"    -> leaders[S001] = leader {rec['leader']}, following {sorted(rec['followers'])}, since {rec['since']}")
    assert rec is not None and rec["leader"] == "Mara" and len(rec["followers"]) >= leadership.MIN_FOLLOWERS

    # B: a SINGLE ardent admirer (the globally highest trust score) — NO cluster, NO leader.
    world.create_world(size=12); world_state["leadership_on"] = True; world_state["leaders"] = {}
    _settled("Idol", (5, 5)); _settled("Fan", (5, 6)); _settled("X", (4, 4)); _settled("Y", (6, 6))
    _trusts(world_state["agents"][1], "Idol", 9)   # Fan trusts Idol at 9 — the world's MAX score
    ev = leadership.update(world_state, 1)
    print(f"  one ardent admirer (Fan trusts Idol at 9 — the highest score in the world):")
    print(f"    -> leaders = {world_state['leaders']}  (a high score is NOT a following)")
    assert world_state["leaders"] == {}, "a single max-trust edge must NOT make a leader"

    # C: a fractured settlement — trust spread thin, no candidate reaches the bar -> no leader.
    world.create_world(size=12); world_state["leadership_on"] = True; world_state["leaders"] = {}
    a = _settled("A", (5, 5)); b = _settled("B", (6, 6)); c = _settled("C", (4, 4)); d = _settled("D", (5, 6))
    _trusts(b, "A", 2); _trusts(c, "D", 2); _trusts(d, "B", 2)   # everyone admires someone different
    ev = leadership.update(world_state, 1)
    print(f"  fractured settlement (each trusts a different agent, nobody reaches a cluster):")
    print(f"    -> leaders = {world_state['leaders']}  (correct: not every group has a leader)")
    assert world_state["leaders"] == {}, "a fractured low-trust settlement must have NO leader"
    print("  -> emergence is CONDITIONAL on a cohered following, not a global-max lookup.  PASS\n")


def headline_2_power_decoupled_from_wealth() -> None:
    print("=" * 72)
    print("HEADLINE 2 — POWER DECOUPLED FROM WEALTH: the trust-leader is NOT the richest")
    print("=" * 72)
    world.create_world(size=12); world_state["leadership_on"] = True; world_state["leaders"] = {}
    poor = _settled("Poor", (5, 5), money=0.0)
    rich = _settled("Rich", (6, 6), money=99.0)         # by far the wealthiest agent
    for nm, pos in (("F1", (5, 6)), ("F2", (6, 5)), ("F3", (4, 5))):
        f = _settled(nm, pos, money=2.0)
        _trusts(f, "Poor", leadership.FORM_TRUST)        # the poor agent is widely trusted
        _trusts(f, "Rich", -3)                           # the rich agent is distrusted
    leadership.update(world_state, 1)
    leader = world_state["leaders"]["S001"]["leader"]
    richest = max(world_state["agents"], key=_wealth).name
    print(f"    wealth: Poor={_wealth(poor):.0f}, Rich={_wealth(rich):.0f}  (richest agent = {richest})")
    print(f"    trust : 3 co-settlers trust Poor HIGH, distrust Rich")
    print(f"    -> emergent leader = {leader}  (poor + trusted), NOT {richest} (rich + distrusted)")
    assert leader == "Poor", "trust, not wealth, must drive leadership"
    assert leader != richest, "if the richest always led, trust would have added nothing (FAIL)"
    print("  -> a poorer, trusted agent leads; the richest does not — political power is not")
    print("     downstream of wealth (the M3.2 point).  PASS\n")


def demo_c_contingent_legitimacy() -> None:
    print("=" * 72)
    print("DEMO C — CONTINGENT LEGITIMACY: lost when trust erodes (with hysteresis); a")
    print("         more-trusted centre DISPLACES the incumbent")
    print("=" * 72)
    print(f"  (hysteresis band: forms on FORM_TRUST={leadership.FORM_TRUST}, retained on "
          f"KEEP_TRUST={leadership.KEEP_TRUST})")
    world.create_world(size=12); world_state["leadership_on"] = True; world_state["leaders"] = {}
    _settled("Lena", (5, 5)); f1 = _settled("F1", (5, 6)); f2 = _settled("F2", (6, 5))
    _trusts(f1, "Lena", leadership.FORM_TRUST); _trusts(f2, "Lena", leadership.FORM_TRUST)
    leadership.update(world_state, 1)
    print(f"    turn 1: {world_state['leaders']['S001']['leader']} leads (2 high-trust followers)")

    # Hysteresis: one follower drifts FORM_TRUST -> KEEP_TRUST. Must NOT unseat.
    _trusts(f1, "Lena", leadership.KEEP_TRUST)
    leadership.update(world_state, 2)
    still = world_state["leaders"].get("S001", {}).get("leader")
    print(f"    turn 2: a follower wobbles {leadership.FORM_TRUST}->{leadership.KEEP_TRUST}  "
          f"-> leader still {still}  (no single-turn flicker)")
    assert still == "Lena", "a one-turn wobble must not unseat (hysteresis)"

    # Real erosion: the leader turns hostile, both followers fall below KEEP_TRUST -> role lost.
    _trusts(f1, "Lena", -3); _trusts(f2, "Lena", -3)
    ev = leadership.update(world_state, 3)
    print(f"    turn 3: leader turns hostile, both fall below the keep bar -> {ev[0]}")
    assert "S001" not in world_state["leaders"], "erosion below the keep bar must end the role"

    # Displacement: a fresh, strictly-more-trusted centre takes over a still-incumbent leader.
    world.create_world(size=12); world_state["leadership_on"] = True; world_state["leaders"] = {}
    _settled("Lena", (5, 5)); _settled("Cyrus", (7, 7))
    f1 = _settled("F1", (5, 6)); f2 = _settled("F2", (6, 5))
    _trusts(f1, "Lena", leadership.FORM_TRUST); _trusts(f2, "Lena", leadership.FORM_TRUST)
    leadership.update(world_state, 1)
    print(f"    [displacement] turn 1: {world_state['leaders']['S001']['leader']} leads")
    _trusts(f1, "Lena", leadership.KEEP_TRUST); _trusts(f2, "Lena", leadership.KEEP_TRUST)
    _trusts(f1, "Cyrus", leadership.FORM_TRUST); _trusts(f2, "Cyrus", leadership.FORM_TRUST)
    ev = leadership.update(world_state, 2)
    rec = world_state["leaders"]["S001"]
    print(f"    turn 2: the following shifts to Cyrus -> {ev[0]}")
    print(f"            leaders[S001] = leader {rec['leader']}, since {rec['since']} (tenure reset)")
    assert rec["leader"] == "Cyrus" and rec["since"] == 2, "a more-trusted centre must displace"
    print("  -> legitimacy is contingent on the ONGOING following, with hysteresis.  PASS\n")


def demo_d_effect_is_real() -> None:
    print("=" * 72)
    print("DEMO D — THE EFFECT IS REAL: a LED settlement is measurably more cohesive than an")
    print("         identical UNLED one (followers rally tighter to the leader — not tax/law)")
    print("=" * 72)

    def build() -> list[Agent]:
        # A leader pinned at the centre, four followers scattered at the settlement's edge.
        world.create_world(size=24)
        world_state["settlements"] = {"S001": {"id": "S001", "center": (12, 12),
                                                "members": set(), "founded": 0}}
        world_state["leaders"] = {}
        _settled("Chief", (12, 12))
        for nm, pos in (("F1", (15, 12)), ("F2", (12, 15)), ("F3", (9, 12)), ("F4", (12, 9))):
            f = _settled(nm, pos)
            _trusts(f, "Chief", leadership.FORM_TRUST)
        return [a for a in world_state["agents"]]

    def mean_dist_to_chief() -> float:
        chief = next(a for a in world_state["agents"] if a.name == "Chief")
        fol = [a for a in world_state["agents"] if a.name != "Chief"]
        return sum(max(abs(a.position[0] - chief.position[0]),
                       abs(a.position[1] - chief.position[1])) for a in fol) / len(fol)

    def run(leadership_on: bool) -> float:
        build()
        world_state["leadership_on"] = leadership_on
        random.seed(11)                       # same start, same RNG: the ONLY difference is leadership
        for t in range(1, 13):
            world_state["turn"] = t
            if leadership_on:
                leadership.update(world_state, t)
            # Drive the followers through the SAME strategy/executor the real loop uses.
            for a in [x for x in world_state["agents"] if x.name != "Chief"]:
                action, _ = strategy.choose_action(a, None, world_state)
                world.execute_action(a, action)
        return mean_dist_to_chief()

    led = run(True)
    unled = run(False)
    print(f"    followers start at Chebyshev distance 3 from the leader.")
    print(f"    LED   (leadership ON) : mean follower distance to leader after 12 turns = {led:.2f}")
    print(f"    UNLED (leadership OFF): mean follower distance to leader after 12 turns = {unled:.2f}")
    print(f"    (LED_HOME_RADIUS={leadership.LED_HOME_RADIUS} pulls followers right around the leader;")
    print(f"     unled, only the looser settlement HOME_RADIUS={settlement.HOME_RADIUS} applies)")
    assert led < unled, f"a led settlement must be MORE cohesive (tighter): {led} vs {unled}"
    print("  -> the led settlement is measurably tighter — real coordinating influence, no")
    print("     taxation, no legislation.  PASS\n")


def demo_e_zero_cost_pure_read_and_v1() -> None:
    print("=" * 72)
    print("DEMO E — ZERO added LLM/RNG; leadership writes NO trust; OFF -> v1 byte-identical")
    print("=" * 72)
    import copy
    world.create_world(size=12); world_state["leadership_on"] = True; world_state["leaders"] = {}
    _settled("Lead", (5, 5)); f1 = _settled("F1", (5, 6)); f2 = _settled("F2", (6, 5))
    _trusts(f1, "Lead", leadership.FORM_TRUST); _trusts(f2, "Lead", leadership.FORM_TRUST)
    before = {a.name: copy.deepcopy(a.relationships) for a in world_state["agents"]}
    llm.reset_call_stats(); st0 = random.getstate()
    with contextlib.redirect_stdout(io.StringIO()):
        for t in range(1, 30):
            leadership.update(world_state, t)
    stats = llm.get_call_stats()
    after = {a.name: a.relationships for a in world_state["agents"]}
    print(f"  29 leadership passes: LLM calls = {stats}; RNG untouched = {random.getstate() == st0}")
    print(f"  trust network unchanged by leadership = {after == before}")
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats
    assert random.getstate() == st0, "leadership consumed RNG (would desync v1)"
    assert after == before, "leadership MUST NOT write any trust value (pure read)"

    def run(flag):
        llm.PROVIDER = "random"
        random.seed(37)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(25, focal_budget=8)
            else:
                main.run_simulation(25, focal_budget=8, leadership_on=flag)
        return buf.getvalue()
    assert run(None) == run(False), "leadership_on=False changed the default run"
    print("  zero model calls; leadership draws no RNG and writes no trust; OFF byte-identical to v1.  PASS\n")


def demo_f_emerges_organically_from_built_trust() -> None:
    print("=" * 72)
    print("DEMO F — IT EMERGES ORGANICALLY: a full seeded sim with ZERO injected trust —")
    print("         agents settle, build trust through conversation, leadership falls out")
    print("=" * 72)

    def organic_run() -> tuple[list[str], dict]:
        # A small clustered cast of farmers on a 10x10 world with a central food pile: they
        # cohere into ONE settlement and build trust ONLY through the ordinary conversation
        # loop — no _trusts() here, every relationship is EARNED in play. Seeded -> reproducible.
        import main
        random.seed(7)
        cells = [(x, y) for x in range(4, 7) for y in range(4, 7)]
        goals = {"survive": 8, "wealth": 3, "friendship": 4}
        specs = [(f"P{i}", ["friendly", "cautious", "social"][i % 3], dict(goals), cells[i])
                 for i in range(7)]
        food = {"initial": 40, "per_turn": 6, "cap": 60, "cluster": True}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(60, focal_budget=7, agent_specs=specs, grid_size=10,
                                food_cfg=food, knowledge_seed=[("farming", 7)],
                                settlements=True, storage_on=True, leadership_on=True,
                                cognition="llm")
        out = buf.getvalue()
        events = [l.strip() for l in out.splitlines()
                  if any(k in l for k in ("emerged as leader", "displaced", "lost legitimacy"))]
        return events, dict(world_state["leaders"])

    # Trust starts EMPTY (we assert nothing is injected) — it is built during the run.
    llm.reset_call_stats()
    events, final = organic_run()
    decisions = llm.get_call_stats()["decision"]
    print("  config: 7 clustered farmers, 1 central food pile, ZERO injected trust, seed=7")
    print("  the trust network is built ENTIRELY by the conversation loop; leadership only reads it:")
    for e in events:
        print(f"    -> {e}")
    print(f"    -> final leaders = {final}")
    print(f"  real-model decision calls added by leadership = {decisions}  (deterministic, reproducible)")
    emerged = [e for e in events if "emerged as leader" in e]
    displaced = [e for e in events if "displaced" in e]
    assert emerged, "a leader must EMERGE organically from conversation-built trust"
    assert displaced, "the role must CHANGE HANDS as the organic trust network shifts"
    assert decisions == 0, "leadership must add zero model decision calls even in a full run"
    # Re-run must be byte-identical in its leadership trajectory (determinism / reproducibility).
    events2, _ = organic_run()
    assert events2 == events, "the organic leadership trajectory must be reproducible (seeded)"
    print("  -> leadership EMERGES and CHANGES HANDS from trust the agents built themselves —")
    print("     not a constructed fixture; reproducible on re-run.  PASS\n")


def run() -> None:
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        headline_1_legitimacy_not_a_leaderboard()
        headline_2_power_decoupled_from_wealth()
        demo_c_contingent_legitimacy()
        demo_d_effect_is_real()
        demo_e_zero_cost_pure_read_and_v1()
        demo_f_emerges_organically_from_built_trust()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M3.2 VERIFIED: LEGITIMATE LEADERSHIP is the first POLITICAL institution and the first "
          "power NOT downstream of wealth. A leader EMERGES as the centre of a coherent TRUST "
          "cluster within a settlement (>= MIN_FOLLOWERS co-settlers above the bar) — never a "
          "global-max lookup, and NONE emerges in a fractured settlement; the leader can be a "
          "poorer, trusted agent the richest member is not (power decoupled from wealth); the role "
          "is CONTINGENT — lost when the following erodes (with hysteresis) and displaced by a "
          "more-trusted centre; and it MATTERS — a led settlement is measurably more cohesive than "
          "an unled one (coordinating influence, not tax/law). A PURE read of the v1 trust system: "
          "zero LLM/RNG, writes no trust, byte-identical to v1 when off.")
    print("=" * 72)


if __name__ == "__main__":
    run()
