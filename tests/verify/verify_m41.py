"""
verify_m41.py
=============

Deterministic verification of V2 milestone M4.1: LINEAGE — birth, childhood,
aging, and family. Opens Phase 4 (Generations & Dynasties), on top of all of
Phases 0-3.

Run offline (Ollama OFF, no model server, no seed-search):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m41.py

The historical step: through Phase 3, agents only die of starvation or battle
and the population is topped up by RESPAWN (blank slates). M4.1 makes life
GENERATIONAL: settled, trusting, fed pairs in a food-surplus settlement bear
children who inherit blended temperament (never knowledge or wealth), are raised
at real cost, age, and die of old age — so the cast TURNS OVER and time itself
becomes meaningful. Wealth inheritance at death is M4.2 and dynastic succession
of titles is M4.3 — deliberately NOT built here.

HEADLINE 1 — FAMILIES EMERGE: in a settled, fed, high-trust world children are
             born, family links recorded, kin-trust seeded — and EVERY gate
             (trust / settlement / fed / surplus / cap) individually binds.
HEADLINE 2 — CHILDREN INHERIT AND LEARN: a child of two curious parents skews
             curious (blend shown, jitter bounded); it knows NOTHING at birth
             but learns through the EXISTING diffusion faster than an adult
             stranger (boost shown); it consumes parental stockpile (drawdown
             shown) and produces nothing until maturity.
HEADLINE 3 — GENERATIONS TURN: in a full 150-turn run the founding cast dies of
             old age at varied lifespans and is REPLACED by descendants —
             population sustained by BIRTHS, respawn silent.
MALTHUS    — abundant food -> population grows toward the cap via births;
             scarce food -> births suppressed, population stagnates/declines.
BACKSTOP   — births impossible + population crashed below the floor -> respawn
             fires; above the floor with births active -> respawn stays silent.
COMPOSE    — zero added LLM calls; --lineage off byte-identical (respawn exactly
             as today); deterministic/reproducible under seed.
"""

from __future__ import annotations

import contextlib
import io
import random

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from sim import knowledge
from sim import lineage
from llm import llm
import main
from sim import population
from sim import trust
from sim import world
from sim.agents import Agent
from llm.strategy import get_personality
from sim.world import world_state


# --- Staging helpers ---------------------------------------------------------
def _fresh(pop_cap: int = 12) -> None:
    """A clean lineage-on world with one settlement S001 centred at (5, 5)."""
    world.create_world()
    world_state["lineage_on"] = True
    world_state["lineage"] = {"pop_cap": pop_cap, "birth_seq": 0}
    world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5),
                                          "members": set(), "founded": 0}


def _settler(name: str, pos: tuple[int, int], personality: str = "friendly and outgoing",
             sid: "str | None" = "S001") -> Agent:
    a = Agent(name=name, personality=personality)
    world.place_agent(a, *pos)
    a.hunger = 0
    a.settlement = sid
    a.age, a.lifespan = 20, 100
    if sid is not None:
        world_state["settlements"][sid]["members"].add(name)
    return a


def _mutual(a: Agent, b: Agent, level: int = lineage.PAIR_TRUST) -> None:
    trust.ensure_relationship(a, b.name)["trust"] = level
    trust.ensure_relationship(b, a.name)["trust"] = level


def _surplus(tiles: int = 4) -> None:
    for x, y in [(4, 4), (6, 6), (5, 3), (3, 5), (7, 5), (5, 7)][:tiles]:
        world.place_food(x, y)


def _births_over(turns: range, seed: int = 3) -> int:
    """Drive the real per-turn lineage update over `turns`; return births."""
    rng = random.Random(seed)
    total = 0
    for t in turns:
        total += len(lineage.update(world_state, t, rng=rng))
    return total


# --- HEADLINE 1: families EMERGE, every gate binds ---------------------------
def headline_1_families_emerge() -> None:
    print("=" * 68)
    print("HEADLINE 1 — FAMILIES EMERGE (and every birth gate binds)")
    print("=" * 68)

    _fresh()
    ada, ben = _settler("Ada", (5, 5)), _settler("Ben", (6, 5))
    _mutual(ada, ben)
    _surplus()
    born = _births_over(range(1, 25))
    assert born >= 1, "settled + trusting + fed + surplus must bear children"
    child = next(a for a in world_state["agents"] if a.parents)
    print(f"  settled/fed/high-trust world over 24 turns -> {born} child(ren)")
    print(f"  family link recorded: {child.name} of {child.parents[0]} + {child.parents[1]}, "
          f"settlement {child.settlement}")
    kt = child.relationships["Ada"]["trust"], ada.relationships[child.name]["trust"]
    assert kt == (lineage.KIN_TRUST, lineage.KIN_TRUST)
    print(f"  kin-trust seeded both ways at {lineage.KIN_TRUST} "
          f"(> pairing bar {lineage.PAIR_TRUST})")
    assert born <= 3, "births must be PACED by the cooldown, not a burst"
    print(f"  births paced: {born} over 24 turns (cooldown {lineage.BIRTH_COOLDOWN})")

    def gate(label, mutate) -> None:
        _fresh()
        a, b = _settler("Ada", (5, 5)), _settler("Ben", (6, 5))
        _mutual(a, b)
        _surplus()
        mutate(a, b)
        n = _births_over(range(1, 25))
        assert n == 0, f"{label} should block ALL births, got {n}"
        print(f"  gate binds: {label:<28} -> 0 births")

    gate("no food surplus", lambda a, b: world_state["food"].clear())
    gate("not settled (nomads)", lambda a, b: (setattr(a, "settlement", None),
                                               setattr(b, "settlement", None)))
    gate("trust below the bar", lambda a, b: _mutual(a, b, lineage.PAIR_TRUST - 1))
    gate("parents not fed", lambda a, b: (setattr(a, "hunger", lineage.FED_HUNGER),
                                          setattr(b, "hunger", lineage.FED_HUNGER)))
    gate("population at the cap", lambda a, b: world_state["lineage"].__setitem__("pop_cap", 2))
    print()


# --- HEADLINE 2: children inherit temperament, EARN knowledge ----------------
def headline_2_inherit_and_learn() -> None:
    print("=" * 68)
    print("HEADLINE 2 — CHILDREN INHERIT (temperament) AND LEARN (knowledge)")
    print("=" * 68)

    _fresh()
    ada = _settler("Ada", (5, 5), personality="curious and adventurous")
    ben = _settler("Ben", (6, 5), personality="curious and adventurous")
    ada.knowledge, ada.stockpile = {"farming"}, 6.0
    _mutual(ada, ben)
    _surplus()
    born = lineage.update(world_state, 1, rng=random.Random(11))
    child = born[0]
    p1, pc = get_personality(ada), get_personality(child)
    print(f"  parents: curiosity {p1.curiosity:.2f} / caution {p1.caution:.2f}")
    print(f"  child:   curiosity {pc.curiosity:.2f} / caution {pc.caution:.2f} "
          f"(dominant: {pc.dominant})")
    assert pc.dominant == "curiosity", "child of two curious parents skews curious"
    for t in ("curiosity", "caution", "friendliness", "independence"):
        mean = getattr(p1, t)  # both parents identical here
        assert abs(getattr(pc, t) - mean) <= lineage.TRAIT_JITTER + 1e-9
    print(f"  jitter bounded: every trait within ±{lineage.TRAIT_JITTER} of the parents' mean")
    assert child.knowledge == set() and child.stockpile == 0.0 and child.money == 0.0
    print("  at birth the child knows NOTHING and owns NOTHING (wealth inheritance = M4.2)")

    # The learning boost, mechanically: same teacher, same personality — the
    # dependent child adopts at CHILD_LEARN_BOOST x the stranger's rate, and its
    # kin-trust in the parent raises it further through the ordinary trust term.
    stranger = _settler("Zed", (1, 1), personality=child.personality, sid=None)
    stranger.age = 30
    p_child = knowledge.adoption_probability(child, ada, world_state)
    p_stranger = knowledge.adoption_probability(stranger, ada, world_state)
    print(f"  adoption chance/contact from parent Ada: child {p_child:.3f} "
          f"vs adult stranger {p_stranger:.3f} "
          f"(boost x{knowledge.CHILD_LEARN_BOOST:.0f} + kin-trust)")
    assert p_child > p_stranger * (knowledge.CHILD_LEARN_BOOST - 0.01)

    # Childhood as lived: fed from the parent's granary (visible drawdown), learns
    # 'farming' through real diffusion contact, produces nothing until maturity.
    rng = random.Random(4)
    start_stock, learned_at, produced = ada.stockpile, None, 0
    for t in range(2, 2 + lineage.CHILDHOOD_TURNS):
        child.hunger = min(world.HUNGER_MAX, child.hunger + 1)   # a child hungers...
        knowledge.diffuse(world_state, t, rng=rng)               # ...and is taught
        produced += sum(1 for name, _ in knowledge.farm(world_state, t, rng=rng)
                        if name == child.name)
        lineage.update(world_state, t, rng=rng)                  # ...and is fed
        if learned_at is None and "farming" in child.knowledge:
            learned_at = t
    assert learned_at is not None, "the child should acquire 'farming' during childhood"
    assert ada.stockpile < start_stock, "raising the child must draw the granary down"
    assert produced == 0, "a dependent must produce nothing, even knowing 'farming'"
    assert not child.dependent, "childhood over — a full agent now"
    print(f"  learned 'farming' from a parent on turn {learned_at} (via M1.1 diffusion)")
    print(f"  Ada's stockpile {start_stock:.1f} -> {ada.stockpile:.1f} "
          f"(children eat their parents' surplus)")
    print(f"  farm tiles produced by the child while dependent: {produced}")
    print(f"  came of age at {lineage.CHILDHOOD_TURNS}: dependent={child.dependent}")
    print()


# --- HEADLINE 3 + MALTHUS: a full run where generations turn -----------------
_CAST = [
    ("Ada", "friendly and outgoing",    {"survive": 7, "friendship": 8, "wealth": 2}, (4, 4)),
    ("Ben", "friendly and cooperative", {"survive": 7, "friendship": 7, "wealth": 3}, (5, 4)),
    ("Cyn", "curious and adventurous",  {"survive": 7, "friendship": 5, "wealth": 3}, (6, 4)),
    ("Dev", "cautious and careful",     {"survive": 9, "friendship": 4, "wealth": 4}, (4, 5)),
    ("Eli", "friendly and outgoing",    {"survive": 7, "friendship": 8, "wealth": 2}, (6, 5)),
    ("Fay", "curious and social",       {"survive": 7, "friendship": 6, "wealth": 3}, (4, 6)),
    ("Gus", "cautious and cooperative", {"survive": 8, "friendship": 5, "wealth": 3}, (5, 6)),
    ("Hal", "friendly and kind",        {"survive": 7, "friendship": 7, "wealth": 2}, (6, 6)),
]
_FOUNDERS = {name for name, *_ in _CAST}


def _full_run(seed: int, turns: int = 150, farming: bool = True) -> dict:
    """One real end-to-end run (heuristic minds, zero LLM): the abundant world
    seeds 'farming' (reliable food -> settlement -> surplus); the scarce world
    seeds nothing (the same cast starves on the map's trickle)."""
    llm.PROVIDER = "random"
    random.seed(seed)
    llm.reset_call_stats()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main.run_simulation(
            turns, agent_specs=_CAST, cognition="heuristic",
            knowledge_seed=[("farming", 8)] if farming else None,
            settlements=True, storage_on=True, lineage_on=True)
    ev = world_state["events"]
    living = [a for a in world_state["agents"] if a.alive]
    return {
        "births": sum(1 for e in ev if "was born to" in e),
        "old_age": sum(1 for e in ev if "died (old age)" in e),
        "starved": sum(1 for e in ev if "died (starved)" in e),
        "respawns": sum(1 for e in ev if "appeared (blank slate)" in e),
        "living": len(living),
        "founders_alive": sum(1 for a in living if a.name in _FOUNDERS),
        "descendants": sum(1 for a in living if a.parents),
        "cap": world_state["lineage"]["pop_cap"],
        "calls": llm.get_call_stats(),
    }


def headline_3_generations_turn_and_malthus() -> None:
    print("=" * 68)
    print("HEADLINE 3 — GENERATIONS TURN (full 150-turn run, seed 7)")
    print("=" * 68)
    r = _full_run(seed=7)
    print(f"  births: {r['births']}   old-age deaths: {r['old_age']}   "
          f"starved: {r['starved']}   respawns: {r['respawns']}")
    print(f"  founders alive at turn 150: {r['founders_alive']} / 8  "
          f"(the founding generation is EXTINCT — died at varied lifespans)")
    print(f"  living population: {r['living']} (cap {r['cap']}), "
          f"of whom {r['descendants']} are DESCENDANTS")
    assert r["founders_alive"] == 0, "founders must age out over 150 turns"
    assert r["old_age"] >= 8, "old age must be the generational scythe"
    assert r["births"] >= 10 and r["descendants"] >= 5, \
        "the population must be sustained by BIRTHS (descendants carry on)"
    assert r["respawns"] == 0, "respawn must stay SILENT while births sustain the world"
    assert r["calls"]["strategy"] == 0 and r["calls"]["decision"] == 0, \
        "the whole generational run must add ZERO LLM calls"
    print("  respawn events: 0 — births, not blank slates, sustain the world")
    print()

    print("=" * 68)
    print("MALTHUS CHECK — abundance grows, scarcity suppresses (A/B, seed 7)")
    print("=" * 68)
    scarce = _full_run(seed=7, farming=False)
    print(f"  A abundant (farming): births {r['births']:>2}, "
          f"population 8 -> {r['living']} (grows toward cap {r['cap']})")
    print(f"  B scarce  (no farms): births {scarce['births']:>2}, "
          f"population 8 -> {scarce['living']} (stagnates/declines; "
          f"starved {scarce['starved']})")
    assert r["living"] > 8, "abundance must GROW the population via births"
    assert scarce["births"] == 0, "scarcity must suppress births entirely"
    assert scarce["living"] <= 8, "a scarce world must not grow"
    print()


# --- BACKSTOP: respawn = extinction insurance only ---------------------------
def backstop_check() -> None:
    print("=" * 68)
    print("BACKSTOP — births primary, respawn extinction-insurance only")
    print("=" * 68)
    # Above the floor with births active: a due respawn is silently DROPPED.
    _fresh()
    for i, pos in enumerate([(1, 1), (3, 1), (5, 1), (7, 1)]):
        _settler(f"A{i}", pos)
    world_state["pending_respawns"] = [10]
    assert population.process_respawns(10, world_state) == []
    print(f"  4 living >= floor {population.TARGET_POPULATION}: "
          f"due respawn dropped — respawn SILENT")

    # Births impossible (no surplus) + crash below the floor: insurance fires.
    world_state["agents"][0].alive = False
    world_state["agents"][1].alive = False
    world_state["food"].clear()
    world_state["pending_respawns"] = [12]
    spawned = population.process_respawns(12, world_state)
    assert len(spawned) == 1
    print(f"  crash to 2 living < floor: respawn FIRES ({spawned[0].name} enters) "
          f"— extinction insurance intact")
    print()


# --- COMPOSE: off byte-identical, deterministic, zero LLM --------------------
def compose_checks() -> None:
    print("=" * 68)
    print("COMPOSE — off byte-identical; seeded runs reproduce; zero LLM")
    print("=" * 68)

    def run(**kw) -> str:
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(30, **kw)
        return buf.getvalue()

    assert run() == run(lineage_on=False)
    print("  --lineage OFF: byte-identical to the current default run "
          "(respawn exactly as today)")
    assert run(lineage_on=True) == run(lineage_on=True)
    print("  --lineage ON: two seeded runs byte-identical (jitter/lifespans "
          "from the seeded sim stream)")
    print()


def run() -> None:
    print("V2 M4.1 verification — LINEAGE: birth, childhood, aging, family")
    print(f"(provider: {llm.PROVIDER}; boundaries: wealth-at-death inheritance = M4.2,")
    print(" dynastic succession of titles = M4.3 — neither built here)\n")
    headline_1_families_emerge()
    headline_2_inherit_and_learn()
    headline_3_generations_turn_and_malthus()
    backstop_check()
    compose_checks()
    print("ALL M4.1 CHECKS PASSED — families emerge from trust+settlement+surplus,")
    print("children inherit temperament but EARN knowledge, and generations TURN.")


if __name__ == "__main__":
    run()
