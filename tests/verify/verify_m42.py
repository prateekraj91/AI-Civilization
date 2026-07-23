"""
verify_m42.py
=============

Deterministic verification of V2 milestone M4.2: INHERITANCE AT DEATH — the
second milestone of Phase 4 (Generations & Dynasties), on top of M4.1 (birth,
childhood, aging) and all of Phases 0-3.

Run offline (Ollama OFF, no model server, no seed-search):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m42.py

The historical step: through M4.1 a dead agent's wealth VANISHED — every death
erased a life's savings. M4.2 makes death the ENGINE of generational wealth: on
ANY death (old age, starvation, battle — one shared hook, population.announce_death)
the deceased's money + stockpile form an ESTATE that passes PARTIBLY and EQUALLY
down a fixed kin-order (children -> parents -> siblings), escheating to the
settlement's ruler when there is no kin. Wealth is CONSERVED; only movable wealth
moves (titles are M4.3).

HEADLINE 1 — WEALTH FLOWS DOWN GENERATIONS: a wealthy parent dies of old age; the
             children split the estate EQUALLY; conservation shown to the decimal;
             the kin-order shown BINDING (children > parents > siblings, each
             fallback demonstrated); cap-overflow food drops as ground food.
HEADLINE 2 — INHERITANCE COMPOUNDS INEQUALITY ACROSS GENERATIONS (the milestone's
             reason to exist): a long multi-generation A/B — lineage WITH
             inheritance vs the same run with inheritance suppressed (the
             vanish-at-death baseline). Descendant-generation wealth Gini is
             HIGHER with inheritance on; a rich house persists across 2+
             generations while non-heirs start from zero.
HEADLINE 3 — ESCHEAT: a kinless settled agent dies under a monarch; the crown
             absorbs the estate (logged). The same death with no ruler vanishes
             exactly as pre-M4.2.
DEPENDENT   — an inheriting CHILD is a real heir: an orphan with an inherited
             granary outlives an identical orphan with none.
COMPOSE     — zero added LLM calls; --lineage off byte-identical (estate vanishes
             as today); deterministic/reproducible under seed.
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from sim import lineage
from llm import llm
import main
from sim import population
from sim import storage
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


def _agent(name: str, pos: tuple[int, int], *, money: float = 0.0,
           stockpile: float = 0.0, parents: tuple = (), sid: "str | None" = "S001",
           dependent: bool = False) -> Agent:
    """A living agent with wealth + a family link — an estate-builder or an heir."""
    a = Agent(name=name, personality="friendly and outgoing")
    world.place_agent(a, *pos)
    a.hunger, a.age, a.lifespan = 1, 20, 100
    a.money, a.stockpile, a.parents, a.dependent = money, stockpile, parents, dependent
    a.settlement = sid
    if sid is not None:
        world_state["settlements"][sid]["members"].add(name)
    return a


def _gini(values: list[float]) -> float:
    """Standard Gini coefficient of a wealth list (0 = equal, ->1 = concentrated)."""
    xs = sorted(v for v in values)
    n = len(xs)
    total = sum(xs)
    if n == 0 or total <= 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2.0 * cum) / (n * total) - (n + 1) / n


# --- HEADLINE 1: wealth flows down generations, conserved, kin-order binding --
def headline_1_wealth_flows_down() -> None:
    print("=" * 70)
    print("HEADLINE 1 — WEALTH FLOWS DOWN GENERATIONS (equal split, conserved)")
    print("=" * 70)

    _fresh()
    parent = _agent("Ada", (5, 5), money=30.0, stockpile=10.0)
    c1 = _agent("Kade", (5, 6), parents=("Ada", "Ben"))
    c2 = _agent("Lena", (6, 5), parents=("Ada", "Ben"))
    print(f"  Ada dies holding money {parent.money:.2f} + stockpile "
          f"{parent.stockpile:.2f} = estate {parent.money + parent.stockpile:.2f}")
    rec = population.announce_death(parent, 40, world_state, cause="old age",
                                    final_memory="Died of old age",
                                    note="they died of old age")
    print(f"  two children inherit -> Kade: money {c1.money:.2f} / stockpile "
          f"{c1.stockpile:.2f}   Lena: money {c2.money:.2f} / stockpile {c2.stockpile:.2f}")
    assert c1.money == c2.money == 15.0 and c1.stockpile == c2.stockpile == 5.0
    print("  split EQUALLY: each child gets exactly half the money and half the food")

    est = lineage.settle_estate  # re-settle a fresh copy for the accounting record
    _fresh()
    parent = _agent("Ada", (5, 5), money=30.0, stockpile=10.0)
    _agent("Kade", (5, 6), parents=("Ada", "Ben"))
    _agent("Lena", (6, 5), parents=("Ada", "Ben"))
    r = est(parent, 40, world_state)
    print(f"  CONSERVATION to the decimal: estate {r['estate']:.2f} == "
          f"to heirs {r['to_heirs']:.2f} + ground drop {r['ground']:.2f} "
          f"(= {r['to_heirs'] + r['ground']:.2f})")
    assert abs(r["to_heirs"] + r["ground"] - r["estate"]) < 1e-9
    assert parent.money == 0.0 and parent.stockpile == 0.0
    print("  wealth left the corpse (no double-counting), nothing minted, nothing lost")

    # Cap-overflow: food beyond the heir's granary cap drops as ground food.
    _fresh()
    n_food_before = len(world_state["food"])
    parent = _agent("Ada", (5, 5), stockpile=40.0)
    heir = _agent("Kade", (5, 6), parents=("Ada", "Ben"), stockpile=6.0)
    r = est(parent, 41, world_state)
    tiles = len(world_state["food"]) - n_food_before
    print(f"  cap-overflow: sole heir fills granary to {heir.stockpile:.0f}/"
          f"{storage.STORAGE_CAP:.0f}; overflow {r['ground']:.2f} drops as "
          f"{tiles} ground-food tiles (not vanished)")
    assert heir.stockpile == storage.STORAGE_CAP and r["ground"] == 26.0 and tiles == 26

    # Kin-order BINDING: children > parents > siblings, each fallback demonstrated.
    print("  kin-order binds (each tier fires only when the closer tier is empty):")

    def who_inherits(seed_kin) -> tuple[str, list[str]]:
        _fresh()
        ada = _agent("Ada", (5, 5), money=40.0, parents=("Ben", "Cara"))
        seed_kin(ada)
        rr = lineage.settle_estate(ada, 42, world_state)
        heirs = sorted(a.name for a in world_state["agents"] if a.alive and a.money > 0)
        return rr["kind"], heirs

    kind, heirs = who_inherits(lambda ada: (
        _agent("Milo", (4, 4), parents=("Ben", "Ada")),   # a CHILD of Ada
        _agent("Ben", (6, 6)),                            # + a living parent
        _agent("Nell", (3, 3), parents=("Ben", "Cara")))) # + a sibling
    assert kind == "children" and heirs == ["Milo"]
    print(f"    children present  -> {kind:<9} take all  (heirs: {heirs})")

    kind, heirs = who_inherits(lambda ada: (
        _agent("Ben", (6, 6)),                            # a living parent
        _agent("Nell", (3, 3), parents=("Ben", "Cara")))) # + a sibling, but no child
    assert kind == "parents" and heirs == ["Ben"]
    print(f"    no child          -> {kind:<9} take all  (heirs: {heirs})")

    kind, heirs = who_inherits(lambda ada:
        _agent("Nell", (3, 3), parents=("Ben", "Cara")))  # only a sibling
    assert kind == "siblings" and heirs == ["Nell"]
    print(f"    no child/parent   -> {kind:<9} take all  (heirs: {heirs})")
    print()


# --- HEADLINE 2: inheritance COMPOUNDS inequality across generations ----------
_CAST = [
    ("Ada", "cautious and careful",      {"survive": 7, "wealth": 8, "friendship": 2}, (4, 4)),
    ("Ben", "independent and competitive", {"survive": 7, "wealth": 8, "friendship": 1}, (5, 4)),
    ("Cyn", "curious and adventurous",   {"survive": 7, "wealth": 4, "friendship": 4}, (6, 4)),
    ("Dev", "cautious and careful",      {"survive": 8, "wealth": 6, "friendship": 3}, (4, 5)),
    ("Eli", "friendly and outgoing",     {"survive": 7, "wealth": 2, "friendship": 8}, (6, 5)),
    ("Fay", "friendly and kind",         {"survive": 7, "wealth": 2, "friendship": 8}, (4, 6)),
    ("Gus", "independent and competitive", {"survive": 8, "wealth": 7, "friendship": 2}, (5, 6)),
    ("Hal", "friendly and social",       {"survive": 7, "wealth": 3, "friendship": 7}, (6, 6)),
]
_FOUNDERS = {name for name, *_ in _CAST}


def _ancestry_depth(name: str, by_name: dict[str, Any]) -> int:
    """Generation depth: 0 for a founder (no parents), else 1 + max(parent depth)."""
    seen: dict[str, int] = {}

    def depth(n: str) -> int:
        a = by_name.get(n)
        if a is None or not a.parents:
            return 0
        if n in seen:
            return seen[n]
        seen[n] = 1 + max((depth(p) for p in a.parents), default=0)
        return seen[n]

    return depth(name)


def _run(seed: int, turns: int, suppress_inheritance: bool) -> dict:
    """One real end-to-end multi-generation run (heuristic minds, zero LLM).

    `suppress_inheritance` runs the SAME sim with lineage.settle_estate stubbed to
    a no-op (wealth still leaves the corpse — it just vanishes, the pre-M4.2
    baseline). Everything else is identical, so the only difference is whether
    death passes wealth on or erases it.
    """
    real = lineage.settle_estate
    if suppress_inheritance:
        def _vanish(deceased, turn, state):  # matches settle_estate's signature
            deceased.money = 0.0
            deceased.stockpile = 0.0
            return {"estate": 0.0, "kind": "none", "per_heir": 0.0,
                    "to_heirs": 0.0, "ground": 0.0}
        lineage.settle_estate = _vanish
    try:
        llm.PROVIDER = "random"
        random.seed(seed)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(
                turns, agent_specs=_CAST, cognition="heuristic",
                knowledge_seed=[("farming", 8)], settlements=True,
                storage_on=True, economy_on=True, lineage_on=True)
    finally:
        lineage.settle_estate = real

    agents = world_state["agents"]
    by_name = {a.name: a for a in agents}
    living = [a for a in agents if a.alive]
    descendants = [a for a in living if a.parents]
    wealth = {a.name: a.money + a.stockpile for a in descendants}
    return {
        "births": sum(1 for e in world_state["events"] if "was born to" in e),
        "inherit_events": sum(1 for e in world_state["events"] if "inherited" in e),
        "descendants": descendants,
        "desc_wealth": wealth,
        "gini": _gini(list(wealth.values())),
        "by_name": by_name,
        "calls": llm.get_call_stats(),
        "events": list(world_state["events"]),
    }


def headline_2_compounds_inequality() -> None:
    print("=" * 70)
    print("HEADLINE 2 — INHERITANCE COMPOUNDS INEQUALITY ACROSS GENERATIONS")
    print("=" * 70)
    turns, seed = 220, 7
    A = _run(seed, turns, suppress_inheritance=False)   # lineage WITH inheritance
    B = _run(seed, turns, suppress_inheritance=True)     # vanish-at-death baseline
    print(f"  A/B over {turns} turns (seed {seed}), heuristic minds, zero LLM:")
    print(f"    A  inheritance ON : {A['births']:>3} births, "
          f"{A['inherit_events']:>3} inheritance events, "
          f"{len(A['descendants']):>2} living descendants")
    print(f"    B  vanish baseline: {B['births']:>3} births, "
          f"{B['inherit_events']:>3} inheritance events, "
          f"{len(B['descendants']):>2} living descendants")
    print(f"  DESCENDANT-GENERATION WEALTH GINI:  A (inherit) = {A['gini']:.3f}   "
          f"vs   B (vanish) = {B['gini']:.3f}")
    assert A["inherit_events"] > 0, "the inheritance arm must actually pass wealth"
    assert B["inherit_events"] == 0, "the baseline arm must pass no wealth"
    assert A["gini"] > B["gini"], \
        "inheritance must CONCENTRATE descendant wealth (Gini_A > Gini_B)"
    print("  -> inheritance CONCENTRATES descendant wealth: rich houses stay rich,")
    print("     while descendants of poor/kinless lines start from zero.")

    # A concrete rich house persisting across 2+ generations vs a zero-start heir.
    by = A["by_name"]
    ranked = sorted(A["desc_wealth"].items(), key=lambda kv: -kv[1])
    top_name, top_wealth = ranked[0]
    top = by[top_name]
    depth = _ancestry_depth(top_name, by)
    # Trace the ancestral line of the wealthiest heir back to its founder.
    line, cur = [top_name], top
    while cur is not None and cur.parents:
        cur = by.get(cur.parents[0])
        line.append(cur.name if cur is not None else "?")
    # Count generational wealth TRANSFERS down this exact line, read from the
    # durable events log (per-agent memory is bounded and may trim old lines).
    transfers = [
        f"{younger}<-{older}"
        for younger, older in zip(line, line[1:])
        for e in A["events"]
        if f"{younger} inherited" in e and f"from {older}" in e
    ]
    print(f"  richest descendant: {top_name} holds {top_wealth:.2f} "
          f"(generation depth {depth}); line: {' <- '.join(line)}")
    print(f"  wealth cascaded down this line across {len(transfers)} inheritance(s): "
          f"{', '.join(transfers) or '(none in this exact line)'}")
    assert depth >= 2, "the richest house must persist across 2+ generations"
    assert len(transfers) >= 1, \
        "the dynasty's wealth must have flowed down its family line by inheritance"
    poorest = [n for n, w in ranked if w == 0.0]
    assert poorest, "and non-heir descendants start from zero"
    print(f"  meanwhile {len(poorest)} descendant(s) hold 0.00 "
          f"(e.g. {poorest[0]}) — non-heirs start from nothing")
    assert A["calls"]["strategy"] == 0 and A["calls"]["decision"] == 0
    print()


# --- HEADLINE 3: escheat to the crown, else vanish ---------------------------
def headline_3_escheat() -> None:
    print("=" * 70)
    print("HEADLINE 3 — ESCHEAT: the crown profits from a kinless death")
    print("=" * 70)

    _fresh()
    world_state["monarchs"]["S001"] = {"monarch": "Rex", "since": 0, "garrison": set()}
    rex = _agent("Rex", (5, 5))
    loner = _agent("Ada", (6, 6), money=25.0)  # no children, parents, or siblings
    print(f"  kinless settled Ada dies under monarch Rex holding {loner.money:.2f}")
    population.announce_death(loner, 60, world_state, cause="starved")
    print(f"  -> the crown absorbs it: Rex now holds {rex.money:.2f}")
    assert rex.money == 25.0
    assert any("escheated to Rex" in e for e in world_state["events"])
    print("  logged distinctly: " +
          next(e for e in world_state["events"] if "escheated" in e).split(": ", 1)[1])

    _fresh()  # the SAME death with no ruler: the estate vanishes as pre-M4.2.
    _agent("Cato", (4, 4), money=12.0, stockpile=8.0)  # an unrelated wealthy bystander
    loner = _agent("Ada", (6, 6), money=25.0)
    before = sum(a.money + a.stockpile for a in world_state["agents"] if a is not loner)
    population.announce_death(loner, 60, world_state, cause="starved")
    after = sum(a.money + a.stockpile for a in world_state["agents"] if a is not loner)
    assert after == before and loner.money == 0.0
    print(f"  no ruler present (a wealthy bystander is not kin): estate VANISHES as "
          f"before M4.2 — nobody else's wealth moves ({before:.2f} -> {after:.2f})")
    assert any("vanished (no heir)" in e for e in world_state["events"])
    print()


# --- DEPENDENT: an inherited granary keeps a dependent orphan alive -----------
def dependent_heir_survives() -> None:
    print("=" * 70)
    print("DEPENDENT HEIR — an inherited granary lets an orphan outlive its hunger")
    print("=" * 70)

    def orphan_survives(inherit: bool) -> bool:
        _fresh()
        world_state["food"].clear()  # famine: nothing to forage
        orphan = _agent("Kade", (5, 5), parents=("Ada", "Ben"), dependent=True)
        orphan.age, orphan.hunger = 4, 1
        if inherit:
            parent = _agent("Ada", (6, 6), stockpile=18.0, sid=None)
            population.announce_death(parent, 0, world_state, cause="starved")
        for t in range(1, 16):
            world.update_hunger(orphan)
            lineage.update(world_state, t, rng=random.Random(t))
            if orphan.alive and world.is_dead(orphan):
                if not storage.draw_down(orphan):
                    population.announce_death(orphan, t, world_state)
        return orphan.alive

    with_inh = orphan_survives(True)
    without = orphan_survives(False)
    print(f"  orphan WITH an inherited 18.0 granary, 15 foodless turns: "
          f"alive = {with_inh}")
    print(f"  identical orphan WITHOUT inheritance:                    "
          f"alive = {without}")
    assert with_inh and not without, "the inherited granary must be what keeps it alive"
    print("  -> inheritance is a real survival edge — grim, but honest.")
    print()


# --- COMPOSE: off byte-identical, deterministic, zero LLM --------------------
def compose_checks() -> None:
    print("=" * 70)
    print("COMPOSE — off byte-identical; seeded runs reproduce; zero LLM")
    print("=" * 70)

    def run(**kw) -> str:
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(30, **kw)
        return buf.getvalue()

    assert run() == run(lineage_on=False)
    print("  --lineage OFF: byte-identical to the current default run "
          "(estate vanishes as today)")
    assert run(lineage_on=True) == run(lineage_on=True)
    print("  --lineage ON: two seeded runs byte-identical (inheritance draws no RNG)")
    print()


if __name__ == "__main__":
    saved = llm.PROVIDER
    try:
        headline_1_wealth_flows_down()
        headline_2_compounds_inequality()
        headline_3_escheat()
        dependent_heir_survives()
        compose_checks()
    finally:
        llm.PROVIDER = saved
    print("=" * 70)
    print("M4.2 VERIFIED — death stops erasing wealth; history accumulates in families.")
    print("=" * 70)
