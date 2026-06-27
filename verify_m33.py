"""
verify_m33.py
=============

Deterministic verification of V2 milestone M3.3: TAXATION & REDISTRIBUTION — legitimacy acts
on wealth. Phase 3 (Institutions), on top of M3.1 (wage labor) and M3.2 (legitimate leadership)
and all of Phase 0/1/2.

Run offline (Ollama OFF, no model server, no seed-search, no long Qwen run):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m33.py

The historical step: M3.1 built the CLASS ENGINE (wage labor — inequality COMPOUNDS); M3.2 built
the LEGITIMACY ENGINE (a leader legitimated by TRUST, decoupled from wealth). M3.3 is their
COLLISION: a legitimate leader TAXES wealthy followers and REDISTRIBUTES to poor ones — the first
force that BENDS the M3.1 inequality spiral. Scope: taxation + redistribution + a legitimacy
backlash that self-limits over-taxation. NO law/legislation, NO revolt, NO fiat money.

HEADLINE 1 — REDISTRIBUTION BENDS INEQUALITY: matched led settlements, the M3.1 labor spiral
             running in BOTH, taxation the only difference. Within-settlement Gini DROPS and
             stays low with taxation ON; stays HIGH (the spiral persists) with it OFF.
HEADLINE 2 — TAXATION REQUIRES LEGITIMACY: a settlement with NO leader cannot tax — no
             redistribution occurs; only a LED one does. Power downstream of legitimacy.
DEMO C — BACKLASH SELF-LIMITS: the SAME leader, two rates, two fates. MODERATE taxation is
         sustained (no resentment; the poor's gratitude even grows support); OVER-taxation erodes
         the wealthy followers' trust below M3.2's keep bar, the following collapses, and the
         leader LOSES legitimacy (and the power to tax) — consent of the governed.
DEMO D — FLOWS ARE CORRECT: wealth is taxed from the RICHEST followers and redistributed to the
         POOREST; total wealth is CONSERVED; a non-follower (and the leader) is untouched.
DEMO E — ZERO added LLM/RNG; taxation conserves wealth; taxation OFF -> v1 byte-identical.
"""

from __future__ import annotations

import contextlib
import io
import random

import labor
import leadership
import llm
import main
import taxation
import trust
import world
from agents import Agent
from world import world_state


def _settled(name: str, pos: tuple[int, int], sid: str = "S001", **kw) -> Agent:
    """Place a living, settled agent (so leadership/taxation read it within a settlement)."""
    a = Agent(name=name, personality="cautious and territorial")
    world.place_agent(a, *pos)
    a.settlement = sid
    a.hunger = kw.pop("hunger", 0)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _trusts(follower: Agent, leader_name: str, value: int) -> None:
    """Set `follower`'s trust in `leader_name` — the existing v1 trust the institutions use."""
    follower.relationships[leader_name] = {"trust": value, "interactions": 1, "grudge": False}


def _wealth(a: Agent) -> float:
    return a.money + a.stockpile


def _gini(xs: list[float]) -> float:
    """Gini coefficient of non-negative wealth (0 = perfectly equal, ->1 = unequal)."""
    xs = sorted(xs)
    n = len(xs)
    s = sum(xs)
    if s <= 0:
        return 0.0
    return (2 * sum((i + 1) * x for i, x in enumerate(xs))) / (n * s) - (n + 1) / n


def headline_1_redistribution_bends_inequality() -> None:
    print("=" * 72)
    print("HEADLINE 1 — REDISTRIBUTION BENDS INEQUALITY: matched led settlements, the M3.1")
    print("              labor spiral in BOTH, taxation the only difference")
    print("=" * 72)

    def build() -> list[Agent]:
        # A led settlement with the M3.1 class structure: a neutral leader, 2 rich employers
        # (the means of production), 7 poor workers (a desperate underclass). An EMPLOYER'S
        # market (abundant hungry labor, few openings) -> subsistence wages -> employers compound
        # -> inequality PERSISTS. Every follower trusts the leader; nothing is hand-redistributed.
        world.create_world(size=20)
        world_state["leadership_on"] = True
        world_state["leaders"] = {}
        world_state["economy_on"] = True  # so employer surplus past the cap becomes money (spiral)
        world_state["settlements"] = {"S001": {"id": "S001", "center": (10, 10),
                                               "members": set(), "founded": 0}}
        leader = _settled("Chief", (10, 10), money=8.0)  # mid-wealth: neither taxed nor a recipient
        cast = [leader]
        for i in range(2):
            e = _settled(f"Emp{i}", (9 + i, 10), money=30.0)
            e.knowledge.add("farming")                    # a producer skill -> an M3.1 employer
            cast.append(e)
        for i in range(7):
            cast.append(_settled(f"Wkr{i}", (10, 9 + (i % 3)), money=1.0, hunger=6))  # poor + hungry
        for f in cast[1:]:
            _trusts(f, "Chief", leadership.FORM_TRUST)
        return cast

    def run(taxation_on: bool, rate: float = 0.30) -> list[float]:
        cast = build()
        world_state["taxation_on"] = taxation_on
        world_state["tax_rate"] = rate
        curve = [_gini([_wealth(a) for a in cast])]
        for t in range(1, 21):
            leadership.update(world_state, t)   # the leader persists on the (stable) trust cluster
            labor.update(world_state, t)        # the M3.1 spiral: employers capture worker surplus
            if taxation_on:
                taxation.update(world_state, t)  # M3.3: skim the rich, lift the poor
            for a in cast:                       # workers stay at subsistence (the M3.1 treadmill)
                if a.name.startswith("Wkr"):
                    a.hunger = max(a.hunger, 6)
            curve.append(_gini([_wealth(a) for a in cast]))
        return curve

    off = run(taxation_on=False)
    on = run(taxation_on=True)
    print(f"  cast: a leader + 2 rich employers + 7 poor workers; M3.1 labor runs in BOTH arms.")
    print(f"  within-settlement wealth Gini over 20 turns (0 = equal, ->1 = unequal):")
    idx = [0, 4, 8, 12, 16, 20]
    print(f"    turn:           " + "  ".join(f"{t:>5}" for t in idx))
    print(f"    taxation OFF :  " + "  ".join(f"{off[t]:.3f}" for t in idx) + "   (spiral persists, HIGH)")
    print(f"    taxation ON  :  " + "  ".join(f"{on[t]:.3f}" for t in idx) + "   (bent DOWN, stays low)")
    print(f"    -> end: OFF Gini = {off[-1]:.3f}  vs  ON Gini = {on[-1]:.3f}  (taxation bends it "
          f"down {off[-1] - on[-1]:.3f})")
    assert on[-1] < off[-1] - 0.05, f"taxation must measurably lower Gini: on {on[-1]} vs off {off[-1]}"
    assert all(on[t] <= off[t] + 1e-9 for t in range(len(on))), "taxed Gini must never exceed untaxed"
    assert off[-1] > 0.4, f"untaxed inequality should stay high (the spiral), got {off[-1]}"
    print("  -> redistribution measurably BENDS the inequality curve — political legitimacy")
    print("     checking the M3.1 spiral, the milestone's whole point.  PASS\n")


def headline_2_taxation_requires_legitimacy() -> None:
    print("=" * 72)
    print("HEADLINE 2 — TAXATION REQUIRES LEGITIMACY: no leader -> no taxation")
    print("=" * 72)

    def build_unled() -> list[Agent]:
        # Same rich+poor settlement, but FRACTURED: each agent trusts a DIFFERENT co-settler, so
        # no following coheres and M3.2 elects NO leader. Without a legitimate leader, no one has
        # the power to tax.
        world.create_world(size=12)
        world_state["leadership_on"] = True
        world_state["taxation_on"] = True
        world_state["tax_rate"] = 0.30
        world_state["leaders"] = {}
        rich = _settled("Rich", (5, 5), money=40.0)
        a = _settled("A", (6, 6), money=2.0)
        b = _settled("B", (4, 4), money=1.0)
        _trusts(rich, "A", 2); _trusts(a, "B", 2); _trusts(b, "Rich", 2)  # all point different ways
        return [rich, a, b]

    cast = build_unled()
    before = {x.name: _wealth(x) for x in cast}
    leadership.update(world_state, 1)
    ev = taxation.update(world_state, 1)
    after = {x.name: _wealth(x) for x in cast}
    print(f"  fractured settlement (no coherent following): leaders = {world_state['leaders']}")
    print(f"  wealth before = {before}")
    print(f"  wealth after  = {after}")
    print(f"  taxation events = {ev}")
    assert world_state["leaders"] == {}, "a fractured settlement has no leader"
    assert before == after, "with NO leader, NO redistribution may occur"
    assert ev == [], "no leader -> taxation produces no events"

    # Contrast: give the SAME settlement a real following and taxation now flows.
    world.create_world(size=12)
    world_state["leadership_on"] = True; world_state["taxation_on"] = True
    world_state["tax_rate"] = 0.30; world_state["leaders"] = {}
    rich = _settled("Rich", (5, 5), money=40.0)
    led_leader = _settled("Chief", (6, 6), money=8.0)
    poor1 = _settled("P1", (4, 4), money=1.0); poor2 = _settled("P2", (5, 4), money=1.0)
    for f in (rich, poor1, poor2):
        _trusts(f, "Chief", leadership.FORM_TRUST)   # a cohered following around Chief
    leadership.update(world_state, 1)
    pool_before = _wealth(rich)
    ev2 = taxation.update(world_state, 1)
    print(f"  same wealth, now WITH a leader (Chief): {ev2[0]}")
    print(f"    -> Rich {pool_before:.0f} -> {_wealth(rich):.1f}; poor lifted "
          f"{_wealth(poor1):.1f}, {_wealth(poor2):.1f}")
    assert ev2 and _wealth(rich) < pool_before, "a led settlement DOES tax and redistribute"
    print("  -> only a legitimate leader can tax; the power is downstream of legitimacy.  PASS\n")


def demo_c_backlash_self_limits() -> None:
    print("=" * 72)
    print("DEMO C — BACKLASH SELF-LIMITS: the SAME leader, two rates, two fates")
    print("=" * 72)
    print(f"  (consent band: tax <= CONSENT_RATE={taxation.CONSENT_RATE:.0%} draws NO resentment; "
          f"above it each taxed follower withdraws trust)")

    def run_rate(rate: float) -> tuple[list, int, int]:
        # A following dominated by the taxed rich, so their consent is what holds the leader up.
        world.create_world(size=12)
        world_state["leadership_on"] = True; world_state["taxation_on"] = True
        world_state["tax_rate"] = rate; world_state["leaders"] = {}
        _settled("Gov", (5, 5), money=5.0)
        rich = [_settled(f"R{i}", (5 + (i % 2), 6 - (i // 2)), money=40.0) for i in range(3)]
        poor = _settled("Poor", (4, 5), money=1.0)
        for f in rich + [poor]:
            _trusts(f, "Gov", leadership.FORM_TRUST)
        fates = []
        for t in range(1, 5):
            leadership.update(world_state, t)        # re-evaluates legitimacy on current trust
            rec = world_state["leaders"].get("S001")
            fates.append(rec["leader"] if rec else None)
            if rec:
                taxation.update(world_state, t)       # taxes + writes the backlash
        return fates, rich[0].relationships["Gov"]["trust"], poor.relationships["Gov"]["trust"]

    mod_fates, mod_rich_trust, mod_poor_trust = run_rate(0.30)   # moderate, within consent
    print(f"  MODERATE rate 30% (within consent): leader by turn = {mod_fates}")
    print(f"    -> taxed-rich trust in Gov held at {mod_rich_trust} (no resentment); poor trust "
          f"rose to {mod_poor_trust} (gratitude). SUSTAINED.")
    assert all(f == "Gov" for f in mod_fates), "moderate taxation must be sustained"
    assert mod_rich_trust >= leadership.KEEP_TRUST, "moderate taxation must not erode the rich below keep"

    over_fates, over_rich_trust, over_poor_trust = run_rate(0.90)  # tyrannical over-reach
    print(f"  OVER rate 90% (far past consent): leader by turn = {over_fates}")
    print(f"    -> taxed-rich trust in Gov crashed to {over_rich_trust} (< keep "
          f"{leadership.KEEP_TRUST}); the following collapsed and Gov LOST legitimacy -> taxing stopped.")
    assert over_fates[0] == "Gov", "the leader starts legitimate and taxes once"
    assert None in over_fates, "over-taxation must cost the leader its legitimacy"
    assert over_rich_trust < leadership.KEEP_TRUST, "over-taxation must erode the rich below the keep bar"
    print("  -> moderate taxation is tolerated and sustained; tyranny is punished by withdrawal")
    print("     of consent (M3.2's contingency fires). Taxation self-limits.  PASS\n")


def demo_d_flows_are_correct() -> None:
    print("=" * 72)
    print("DEMO D — FLOWS ARE CORRECT: rich->poor among followers; conserved; outsiders untouched")
    print("=" * 72)
    world.create_world(size=12)
    world_state["leadership_on"] = True; world_state["taxation_on"] = True
    world_state["tax_rate"] = 0.30; world_state["leaders"] = {}
    leader = _settled("Chief", (5, 5), money=8.0)
    rich = _settled("Rich", (5, 6), money=50.0)
    poor1 = _settled("Poor1", (6, 5), money=1.0)
    poor2 = _settled("Poor2", (4, 5), money=3.0)
    middle = _settled("Middle", (5, 4), money=7.0)         # a follower, but neither rich nor poor
    outsider = _settled("Outsider", (7, 7), money=99.0)     # settled but NOT a follower (no trust)
    for f in (rich, poor1, poor2, middle):
        _trusts(f, "Chief", leadership.FORM_TRUST)
    leadership.update(world_state, 1)
    everyone = [leader, rich, poor1, poor2, middle, outsider]
    before = {a.name: _wealth(a) for a in everyone}
    total_before = sum(before.values())
    taxation.update(world_state, 1)
    after = {a.name: _wealth(a) for a in everyone}
    print(f"  before: {before}")
    print("  after : {" + ", ".join(f'{k!r}: {v:.2f}' for k, v in after.items()) + "}")
    assert after["Rich"] < before["Rich"], "the rich follower must be TAXED"
    assert after["Poor1"] > before["Poor1"] and after["Poor2"] > before["Poor2"], "the poor must GAIN"
    assert after["Poor1"] - before["Poor1"] > after["Poor2"] - before["Poor2"], \
        "the POOREST (Poor1) must receive the most (need-weighted)"
    assert after["Chief"] == before["Chief"], "the leader taxes nobody for itself (untouched)"
    assert after["Middle"] == before["Middle"], "a mid-wealth follower is neither taxed nor paid"
    assert after["Outsider"] == before["Outsider"], "a non-follower is UNTOUCHED (only followers)"
    assert abs(sum(after.values()) - total_before) < 1e-9, "total wealth must be CONSERVED (redistribution, not minting)"
    print(f"  -> taxed only Rich; lifted Poor1 most then Poor2; leader/middle/outsider untouched;")
    print(f"     total wealth conserved ({total_before:.1f} -> {sum(after.values()):.1f}).  PASS\n")


def demo_e_zero_cost_and_v1() -> None:
    print("=" * 72)
    print("DEMO E — ZERO added LLM/RNG; taxation conserves wealth; OFF -> v1 byte-identical")
    print("=" * 72)
    world.create_world(size=12)
    world_state["leadership_on"] = True; world_state["taxation_on"] = True
    world_state["tax_rate"] = 0.30; world_state["leaders"] = {}
    _settled("Chief", (5, 5), money=8.0)
    rich = _settled("Rich", (5, 6), money=40.0)
    poor1 = _settled("P1", (6, 5), money=1.0); poor2 = _settled("P2", (4, 5), money=1.0)
    for f in (rich, poor1, poor2):
        _trusts(f, "Chief", leadership.FORM_TRUST)
    leadership.update(world_state, 1)
    total0 = sum(_wealth(a) for a in world_state["agents"])
    llm.reset_call_stats()
    st0 = random.getstate()
    with contextlib.redirect_stdout(io.StringIO()):
        for t in range(1, 20):
            taxation.update(world_state, t)
    stats = llm.get_call_stats()
    total1 = sum(_wealth(a) for a in world_state["agents"])
    print(f"  19 taxation passes: LLM calls = {stats}; RNG untouched = {random.getstate() == st0}")
    print(f"  total wealth conserved across all passes = {abs(total1 - total0) < 1e-9} "
          f"({total0:.1f} -> {total1:.1f})")
    assert stats == {"decision": 0, "strategy": 0}, stats
    assert random.getstate() == st0, "taxation consumed RNG (would desync v1)"
    assert abs(total1 - total0) < 1e-9, "taxation must conserve total wealth"

    def run(flag):
        llm.PROVIDER = "random"
        random.seed(41)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(24, focal_budget=8)
            else:
                main.run_simulation(24, focal_budget=8, taxation_on=flag)
        return buf.getvalue()

    assert run(None) == run(False), "taxation_on=False changed the default run"
    print("  zero model calls; taxation draws no RNG and conserves wealth; OFF byte-identical to v1.  PASS\n")


def run() -> None:
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        headline_1_redistribution_bends_inequality()
        headline_2_taxation_requires_legitimacy()
        demo_c_backlash_self_limits()
        demo_d_flows_are_correct()
        demo_e_zero_cost_and_v1()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M3.3 VERIFIED: TAXATION & REDISTRIBUTION is the COLLISION of the M3.1 class engine and "
          "the M3.2 legitimacy engine — the first force that BENDS the inequality spiral. A "
          "legitimate leader taxes its wealthy followers' wealth above a threshold and "
          "redistributes to its poor ones, measurably LOWERING the within-settlement Gini below an "
          "identical untaxed settlement (HEADLINE 1); ONLY a led settlement can tax — power "
          "downstream of legitimacy, not wealth (HEADLINE 2); the backlash SELF-LIMITS taxation — "
          "moderate levies are sustained (consent + the poor's gratitude) while over-taxation "
          "erodes the wealthy followers' trust below the keep bar and the leader LOSES legitimacy "
          "via M3.2's contingency (DEMO C); flows go rich->poor among followers, conserve total "
          "wealth, and leave outsiders untouched (DEMO D). Zero LLM/RNG; writes trust through the "
          "existing logged path; byte-identical to v1 when off. Consent of the governed, emergent.")
    print("=" * 72)


if __name__ == "__main__":
    run()
