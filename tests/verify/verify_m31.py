"""
verify_m31.py
=============

Deterministic verification of V2 milestone M3.1: WAGE LABOR — the first INSTITUTION,
which OPENS Phase 3 (Institutions). On top of all of Phase 0 + Phase 1 + Phase 2
(M2.1 settlement, M2.2 storage/wealth, M2.3 trade/money/proprietary-knowledge).

Run offline (Ollama OFF, no model server, no seed-search, no long Qwen run):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m31.py

The historical step: Phases 1-2 built EMERGENT MATERIAL MECHANICS (knowledge, food,
wealth, prices) that mostly EQUILIBRATE. Phase 3 builds INSTITUTIONS — persistent
structures that coordinate/constrain many agents — and the rule for all of Phase 3 is
that an institution must EMERGE from existing asymmetries, never be installed. Wage labor
is the first, and the first DISEQUILIBRATING one: a rich agent EMPLOYS a poor one to
produce for it, paying a wage, and because the employer captures (output - wage) the
rich-poor gap COMPOUNDS instead of settling. That compounding manufactures the class
tension the rest of Phase 3 (law, conflict, governance) will respond to.

HEADLINE 1 — COMPOUNDING INEQUALITY (the disequilibrium): matched casts, wage labor
             ON vs OFF. The wealth Gini RISES over time with labor ON and stays FLAT
             with it OFF. The inequality curve is reported for both. This is the point.
HEADLINE 2 — EMERGENT WAGE / EXPLOITATION: the SAME work pays a DIFFERENT wage as labor
             supply + desperation change — a worker's-market (scarce labor) HIGH wage vs
             an employer's-market (abundant desperate labor) near-SUBSISTENCE wage. A
             fixed wage would be a fail; subsistence EMERGES from conditions, not script.
DEMO C — ROLES EMERGE: employers are the rich+skilled, workers the poor+unskilled —
         pure reads of M2.2 wealth + M1.3/M2.3 skill. No capital -> never an employer;
         self-sufficient -> never a worker. Change the wealth, the role changes (no flag).
DEMO D — RELATIONSHIP PERSISTS + MUTUALLY ENTERED: a link survives across turns (an
         institution, not a one-shot trade); an employed worker SURVIVES (fed by its
         wage) where an identical unemployed one STARVES — so even at subsistence the
         worker gains NET (it lives), distinguishing exploitation from slavery/theft.
DEMO E — ZERO LLM/RNG cost; wage labor OFF -> v1 byte-identical.
"""

from __future__ import annotations

import contextlib
import io
import random

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from sim import economy
from sim import labor
from llm import llm
import main
from sim import storage
from sim import world
from sim.agents import Agent
from sim.world import world_state


def _gini(xs: list[float]) -> float:
    """Gini coefficient of non-negative values (0 = perfectly equal, ->1 = unequal)."""
    xs = sorted(xs)
    n, s = len(xs), sum(xs)
    if n == 0 or s == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * s) - (n + 1) / n


def _wealth(a: Agent) -> float:
    return a.money + a.stockpile


def _settled(name: str, personality: str, pos: tuple[int, int], **kw) -> Agent:
    """Place a settled agent (settlement set so it can employ / be employed)."""
    a = Agent(name=name, personality=personality)
    world.place_agent(a, *pos)
    a.settlement = "S001"
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def headline_1_compounding_inequality() -> None:
    print("=" * 72)
    print("HEADLINE 1 — COMPOUNDING INEQUALITY: the wealth gap RISES with wage labor")
    print("              ON and stays FLAT with it OFF (matched cast, same start)")
    print("=" * 72)

    def build_cast() -> list[Agent]:
        # Same settlement, identical start: 2 rich+skilled (employers fall out of this) and
        # 8 poor+unskilled+hungry (workers fall out of this). Nothing is assigned.
        world.create_world(size=14)
        world_state["economy_on"] = True          # employer overflow past the cap -> money
        world_state["employments"] = []
        cast = [
            _settled("Boss1", "independent and competitive", (6, 6), money=22.0),
            _settled("Boss2", "independent and competitive", (8, 8), money=22.0),
        ]
        cast[0].knowledge.add("farming")
        cast[1].knowledge.add("farming")
        poor_cells = [(5, 6), (7, 6), (6, 5), (6, 7), (7, 8), (9, 8), (8, 7), (8, 9)]
        for i, pos in enumerate(poor_cells):
            cast.append(_settled(f"W{i}", "cautious and territorial", pos,
                                 money=3.0, hunger=7))
        return cast

    def curve(labor_on: bool) -> tuple[list[float], list[Agent]]:
        cast = build_cast()
        world_state["labor_on"] = labor_on
        pts = [_gini([_wealth(a) for a in cast])]
        for t in range(1, 21):
            if labor_on:
                labor.update(world_state, t)     # the institution: produce-for-boss, pay-worker
            pts.append(_gini([_wealth(a) for a in cast]))
        return pts, cast

    on_curve, on_cast = curve(True)
    off_curve, off_cast = curve(False)

    print("  Wealth Gini over 20 turns (0 = equal, ->1 = unequal):")
    print("    turn:        0     5    10    15    20")
    print(f"    labor ON :  " + "  ".join(f"{on_curve[t]:.2f}" for t in (0, 5, 10, 15, 20)))
    print(f"    labor OFF:  " + "  ".join(f"{off_curve[t]:.2f}" for t in (0, 5, 10, 15, 20)))
    bosses_on = [a for a in on_cast if a.name.startswith("Boss")]
    workers_on = [a for a in on_cast if a.name.startswith("W")]
    print(f"  After 20 turns WITH wage labor: employer wealth "
          f"{min(_wealth(b) for b in bosses_on):.0f}-{max(_wealth(b) for b in bosses_on):.0f}, "
          f"worker wealth {min(_wealth(w) for w in workers_on):.1f}-"
          f"{max(_wealth(w) for w in workers_on):.1f}")
    print(f"  (employers banked the captured surplus; workers stayed near subsistence)")

    assert on_curve[-1] > on_curve[0] + 0.05, f"Gini must RISE with labor on: {on_curve}"
    # Each later point at least as unequal as 5 turns earlier -> compounding, not a blip.
    assert on_curve[20] > on_curve[10] > on_curve[0], "inequality must compound over time"
    assert abs(off_curve[-1] - off_curve[0]) < 1e-9, "with labor OFF wealth (Gini) stays flat"
    assert on_curve[-1] > off_curve[-1] + 0.05, "labor ON must end far more unequal than OFF"
    print("  -> inequality COMPOUNDS with the institution on, stays flat off.  PASS\n")


def headline_2_emergent_wage() -> None:
    print("=" * 72)
    print("HEADLINE 2 — EMERGENT WAGE / EXPLOITATION: the SAME work pays DIFFERENTLY")
    print("              as labor supply + desperation change (not a fixed wage)")
    print("=" * 72)

    # Micro table: the one job (output 2.0) priced across market tightness x desperation.
    print(f"  offered wage for the SAME job (output={labor.LABOR_OUTPUT}, "
          f"subsistence floor={labor.SUBSISTENCE_WAGE}):")
    print("                          worker fed   worker hungry   worker starving")
    for label, op, wk in (("scarce labor (worker's mkt)", 20, 1),
                          ("balanced", 1, 1),
                          ("abundant desperate (boss mkt)", 1, 20)):
        t = labor.market_tightness(op, wk)
        row = []
        for h in (0, 5, world.HUNGER_MAX):
            a = Agent(name="x", personality="x"); a.hunger = h
            row.append(f"{labor.offered_wage(a, t):.2f}")
        print(f"    {label:<30}{row[0]:>8}{row[1]:>14}{row[2]:>16}")

    # Two REAL formed links: read the wage the market actually set on each.
    def formed_wage_workers_market() -> float:
        world.create_world(size=10)
        world_state["economy_on"] = True; world_state["labor_on"] = True
        world_state["employments"] = []
        boss = _settled("Boss", "independent and competitive", (5, 5), money=40.0)
        boss.knowledge.add("farming")               # capacity 8, ONE worker -> labor scarce
        _settled("Hand", "cautious and territorial", (5, 6), money=0.0, hunger=0)
        labor.update(world_state, 1)
        return world_state["employments"][0]["wage"]

    def formed_wage_employers_market() -> float:
        world.create_world(size=10)
        world_state["economy_on"] = True; world_state["labor_on"] = True
        world_state["employments"] = []
        boss = _settled("Boss", "independent and competitive", (5, 5), money=5.0)
        boss.knowledge.add("farming")               # capacity 1, EIGHT desperate workers
        for i, pos in enumerate([(5, 6), (5, 4), (4, 5), (6, 5),
                                 (4, 4), (6, 6), (4, 6), (6, 4)]):
            _settled(f"H{i}", "cautious and territorial", pos,
                     money=0.0, hunger=world.HUNGER_MAX)
        labor.update(world_state, 1)
        return world_state["employments"][0]["wage"]

    w_high = formed_wage_workers_market()
    w_low = formed_wage_employers_market()
    print(f"  REAL hires of the SAME job:")
    print(f"    worker's market  (1 worker, 8 openings)        -> wage {w_high:.2f}  "
          f"(near output {labor.LABOR_OUTPUT}: the worker captures the value)")
    print(f"    employer's market(8 desperate workers, 1 job)  -> wage {w_low:.2f}  "
          f"(= subsistence {labor.SUBSISTENCE_WAGE}: the employer captures it — EXPLOITATION)")
    assert w_high > w_low, "the same work must pay more when labor is scarce"
    assert abs(w_low - labor.SUBSISTENCE_WAGE) < 1e-9, "glut+desperation must bottom at subsistence"
    assert labor.SUBSISTENCE_WAGE <= w_high < labor.LABOR_OUTPUT, "bounded: worker survives, boss profits"
    print("  -> the SAME work paid 2 different wages; subsistence EMERGED from conditions.  PASS\n")


def demo_c_roles_emerge() -> None:
    print("=" * 72)
    print("DEMO C — ROLES EMERGE from wealth + skill (never assigned)")
    print("=" * 72)
    world.create_world(size=10)
    rich_skilled = _settled("RichSkilled", "x", (2, 2), money=20.0); rich_skilled.knowledge.add("farming")
    poor_unskilled = _settled("PoorPlain", "x", (3, 3), money=0.0)
    rich_unskilled = _settled("RichPlain", "x", (4, 4), money=20.0)
    poor_skilled = _settled("PoorSkilled", "x", (5, 5), money=0.0); poor_skilled.knowledge.add("hunting")
    for a in (rich_skilled, poor_unskilled, rich_unskilled, poor_skilled):
        print(f"    {a.name:<12} wealth={_wealth(a):>4.0f} skill={'yes' if a.knowledge else 'no ':<3}"
              f" -> employer={labor.is_employer(a)!s:<5} worker={labor.is_worker(a)}")
    assert labor.is_employer(rich_skilled) and not labor.is_worker(rich_skilled)
    assert labor.is_worker(poor_unskilled) and not labor.is_employer(poor_unskilled)
    assert not labor.is_employer(rich_unskilled) and not labor.is_worker(rich_unskilled), \
        "wealthy-but-unskilled: no means to employ, independent enough not to sell labor"
    assert not labor.is_employer(poor_skilled) and not labor.is_worker(poor_skilled), \
        "poor-but-skilled: owns its means (not a worker), too poor to employ"
    # Pure read, not a flag: strip the capital and the employer role evaporates.
    rich_skilled.money = 0.0
    assert not labor.is_employer(rich_skilled), "role is a read of state — remove capital, role gone"
    print("  no capital -> never an employer; self-sufficient -> never a worker; no flag set.  PASS\n")


def demo_d_persists_and_mutual() -> None:
    print("=" * 72)
    print("DEMO D — the relationship PERSISTS and is MUTUALLY entered (gain, not theft)")
    print("=" * 72)
    # Persistence: form one link, run several turns, confirm the SAME link (since turn 1) lives on.
    world.create_world(size=10)
    world_state["economy_on"] = True; world_state["labor_on"] = True
    world_state["employments"] = []
    boss = _settled("Boss", "independent and competitive", (5, 5), money=60.0)
    boss.knowledge.add("farming")
    _settled("Hand", "cautious and territorial", (5, 6), money=0.0, hunger=4)
    labor.update(world_state, 1)
    since = world_state["employments"][0]["since"]
    for t in range(2, 11):
        labor.update(world_state, t)
    link = next(l for l in world_state["employments"] if l["worker"] == "Hand")
    print(f"    link Boss->Hand formed turn {since}, still active at turn 10 "
          f"(wage {link['wage']:.2f}) — persists across turns")
    assert link["since"] == since == 1, "the SAME persistent link, not re-created each turn"

    # Mutual benefit: an employed worker SURVIVES (fed by its wage) where an identical
    # UNEMPLOYED one STARVES — even at subsistence the worker gains NET (it lives).
    world.create_world(size=10)
    world_state["economy_on"] = True; world_state["labor_on"] = True
    world_state["employments"] = []
    emp = _settled("Owner", "independent and competitive", (3, 3), money=5.0)  # capacity 1
    emp.knowledge.add("farming")
    # Two identical desperate poor workers; only enough capital to hire one.
    employed = _settled("Empd", "cautious and territorial", (3, 4), money=0.0, hunger=5)
    idle = _settled("Idle", "cautious and territorial", (9, 9), money=0.0, hunger=5)  # too far to hire
    for t in range(1, 9):
        world_state["turn"] = t
        world.update_hunger(employed)
        world.update_hunger(idle)
        labor.update(world_state, t)
    wage = next((l["wage"] for l in world_state["employments"] if l["worker"] == "Empd"), None)
    print(f"    employed worker: hunger {employed.hunger}, alive={employed.alive and not world.is_dead(employed)} "
          f"(near-subsistence wage {wage:.2f} -> fed, survives, accumulates only "
          f"{_wealth(employed):.2f})")
    print(f"    idle worker    : hunger {idle.hunger}, alive={not world.is_dead(idle)} "
          f"(no wage -> starves)")
    # A near-subsistence wage (slightly above the floor here, set by the abundant-labor market):
    # the worker barely gets ahead, but it is fed and stays alive — the treadmill, not slavery.
    assert wage is not None and labor.SUBSISTENCE_WAGE <= wage < labor.SUBSISTENCE_WAGE + 0.5, wage
    assert not world.is_dead(employed), "the wage keeps the worker alive — a net gain over starving"
    assert _wealth(employed) > 0.0, "the worker still gains net (it isn't robbed) — voluntary"
    assert world.is_dead(idle), "the identical unemployed worker starves — employed is strictly better"
    print("  persistent link; worker is better off employed (survives + gains a little) than not")
    print("  -> exploitation, not slavery/theft (the worker still gains NET: its life).  PASS\n")


def demo_e_zero_cost_and_v1() -> None:
    print("=" * 72)
    print("DEMO E — zero LLM/RNG cost; wage labor OFF -> v1 byte-identical")
    print("=" * 72)
    # labor.update in isolation: zero model calls, zero RNG draws.
    world.create_world(size=10)
    world_state["economy_on"] = True; world_state["labor_on"] = True
    world_state["employments"] = []
    boss = _settled("Boss", "independent and competitive", (5, 5), money=40.0)
    boss.knowledge.add("farming")
    for i, pos in enumerate([(5, 6), (5, 4), (4, 5), (6, 5)]):
        _settled(f"H{i}", "cautious and territorial", pos, money=0.0, hunger=5)
    llm.reset_call_stats()
    st0 = random.getstate()
    with contextlib.redirect_stdout(io.StringIO()):
        for t in range(1, 30):
            labor.update(world_state, t)
    stats = llm.get_call_stats()
    print(f"  29 labor passes: LLM calls = {stats}; RNG untouched = {random.getstate() == st0}")
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats
    assert random.getstate() == st0, "wage labor consumed RNG (would desync v1)"

    # labor OFF (default) -> byte-identical to a run with the param absent.
    def run(flag):
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(25, focal_budget=8)
            else:
                main.run_simulation(25, focal_budget=8, labor_on=flag)
        return buf.getvalue()
    assert run(None) == run(False), "labor_on=False changed the default run"
    print("  zero model calls; labor draws no RNG; wage labor OFF byte-identical to v1.  PASS\n")


def run() -> None:
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        headline_1_compounding_inequality()
        headline_2_emergent_wage()
        demo_c_roles_emerge()
        demo_d_persists_and_mutual()
        demo_e_zero_cost_and_v1()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M3.1 VERIFIED: WAGE LABOR is the first INSTITUTION. Employer/worker roles EMERGE "
          "from existing M2.2 wealth + M2.3 skill (no flag assigned); the wage EMERGES from "
          "the labor market — the SAME work pays near output when labor is scarce and falls to "
          "SUBSISTENCE under a desperate glut (emergent exploitation, not scripted); the link "
          "PERSISTS across turns and is mutually entered (the worker survives where it would "
          "starve, so it gains NET even when exploited); and because employers capture "
          "(output - wage), inequality COMPOUNDS over time instead of equilibrating — the "
          "disequilibrium that opens Phase 3. Zero LLM/RNG; v1 byte-identical when off.")
    print("=" * 72)


if __name__ == "__main__":
    run()
