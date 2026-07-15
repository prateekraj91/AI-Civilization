"""
verify_m410.py
==============

Deterministic verification of V2 milestone M4.10: WRITING & RECORDS — institutional
memory. First milestone of Arc 4 (the Deep Tech Tree: the road to modernity), on top
of Arc 3 (beliefs/religion/culture), Arc 2 (revolt), Arc 1 (dynasties) and Phases 0-3.

Run offline (Ollama OFF, no model server, no seed-search):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m410.py

The historical step: until now everything died with its holder — a ruler's policy at
death, a skill when its last knower died (the knowledge-extinction collapse), history the
moment it scrolled past. WRITING lets recorded things OUTLIVE their makers. A LITERATE
settlement gains persistent LAW, knowledge PRESERVATION and recorded HISTORY, so its
institutions, skills and memory accumulate across generations. Zero LLM — records are STATE.

HEADLINE 1 — WRITING EMERGES AS A TECH: it is invented through the existing M1.2 discovery
             + M1.1 diffusion, with binding prereqs (tools + a settlement with food surplus —
             no surplus/settlement, no writing); a settlement becomes literate. Unscripted.
HEADLINE 2 — INSTITUTIONS OUTLIVE INDIVIDUALS: a literate ruler's tax/levy policy is inscribed;
             the ruler dies and the M4.3 heir INHERITS the written law; the identical succession
             in an ILLITERATE town loses the policy (blank slate). Same event, two outcomes.
HEADLINE 3 — LITERACY CURES KNOWLEDGE EXTINCTION: a literate town whose farming-knowers all die
             RE-TEACHES farming from its records and keeps producing food; an identical illiterate
             town suffers the extinction collapse — the skill is gone, the fields go barren. Writing
             is the cure for the collapse the earlier finding observed.
RECORDS    — a literate settlement accumulates a persistent CHRONICLE of its major events
             (structured entries, zero LLM); an illiterate one keeps no lasting record.
COST       — zero added LLM; --writing off byte-identical; deterministic/reproducible.
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any

import knowledge
import llm
import main
import population
import world
import writing
from agents import Agent
from world import world_state


# --- Staging helpers ---------------------------------------------------------
def _fresh() -> None:
    world.create_world()
    world_state["writing_on"] = True


def _settlement(sid, center) -> None:
    world_state["settlements"][sid] = {"id": sid, "center": center, "members": set(), "founded": 0}


def _agent(name, pos, sid="S001", *, knows=None, hunger=1, age=40, parents=()) -> Agent:
    a = Agent(name=name, personality="curious and creative")
    world.place_agent(a, *pos)
    a.hunger, a.age, a.lifespan, a.settlement, a.parents = hunger, age, 100, sid, parents
    if knows:
        a.knowledge.update(knows)
    if sid is not None and sid in world_state["settlements"]:
        world_state["settlements"][sid]["members"].add(name)
    return a


def _surplus_food(center=(5, 5)) -> None:
    cx, cy = center
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if (dx, dy) != (0, 0):
                world.place_food(cx + dx, cy + dy)


# --- HEADLINE 1: writing emerges as a tech -----------------------------------
def headline_1_writing_emerges() -> None:
    print("=" * 72)
    print("HEADLINE 1 — WRITING EMERGES AS A TECH (prereqs bind; a settlement becomes literate)")
    print("=" * 72)

    def can_invent(has_tools, settled, surplus) -> bool:
        _fresh(); _settlement("S001", (5, 5))
        a = _agent("Scribe", (5, 5), sid=("S001" if settled else None),
                   knows=({"tools"} if has_tools else set()))
        if surplus:
            _surplus_food()
        rng = random.Random(0)
        for _ in range(500):
            writing.discover_writing(world_state, 1, rng)
            if "writing" in a.knowledge:
                return True
        return False

    print(f"  tools + settlement + surplus -> invents writing? {can_invent(True, True, True)}")
    print(f"  NO food surplus              -> invents writing? {can_invent(True, True, False)}")
    print(f"  NO prior tech (tools)        -> invents writing? {can_invent(False, True, True)}")
    print(f"  NO settlement                -> invents writing? {can_invent(True, False, True)}")
    assert can_invent(True, True, True)
    assert not can_invent(True, True, False) and not can_invent(False, True, True) and not can_invent(True, False, True)

    # It then SPREADS through ordinary diffusion, making the settlement literate.
    _fresh(); _settlement("S001", (5, 5))
    _surplus_food()
    scribes = [_agent(n, p, knows={"tools"}) for n, p in
               [("A", (5, 5)), ("B", (5, 6)), ("C", (6, 5)), ("D", (6, 6))]]
    rng = random.Random(3)
    literate_turn = None
    for t in range(1, 40):
        writing.discover_writing(world_state, t, rng)
        knowledge.diffuse(world_state, t, rng)   # writing spreads like any skill
        if writing.is_literate(world_state, "S001") and literate_turn is None:
            literate_turn = t
    n_write = sum(1 for a in scribes if "writing" in a.knowledge)
    print(f"\n  from tools-holding settlers in a surplus town, writing was invented and spread: "
          f"the town became LITERATE by turn {literate_turn} ({n_write}/4 can now write).")
    assert literate_turn is not None
    print("  -> writing is a tech like any other — earned via the existing discovery+diffusion, prereqs binding.")
    print()


# --- HEADLINE 2: institutions outlive individuals ----------------------------
def headline_2_institutions_outlive_individuals() -> None:
    print("=" * 72)
    print("HEADLINE 2 — INSTITUTIONS OUTLIVE INDIVIDUALS (the heir inherits the written law)")
    print("=" * 72)

    def law_after_succession(literate):
        _fresh(); world_state["lineage_on"] = True; _settlement("S001", (5, 5))
        king = _agent("King", (5, 5), knows=({"writing"} if literate else set()), age=60)
        _agent("Heir", (5, 6), knows=({"writing"} if literate else set()), parents=("King", "Q"), age=25)
        world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
        writing.update(world_state, 1)              # King inscribes the law (if literate)
        population.announce_death(king, 2, world_state, cause="old age",
                                  final_memory="Died", note="they died")   # M4.3 succession
        writing.update(world_state, 3)              # the heir inherits it (if literate)
        return writing.written_law(world_state, "S001")

    lit = law_after_succession(True)
    ill = law_after_succession(False)
    print(f"  LITERATE town: King inscribes a tax/levy law; he dies -> the heir's law is")
    print(f"    {lit}")
    print(f"  ILLITERATE town: the identical succession -> written law is {ill} (blank slate)")
    assert lit is not None and lit["set_by"] == "Heir" and lit["inherited_from"] == "King"
    assert ill is None
    print("  -> literacy makes the institution outlive the individual: the heir governs by the")
    print("     inherited framework, where an illiterate realm starts every reign from nothing.")
    print()


# --- HEADLINE 3: literacy cures knowledge extinction --------------------------
def headline_3_literacy_cures_extinction() -> None:
    print("=" * 72)
    print("HEADLINE 3 — LITERACY CURES KNOWLEDGE EXTINCTION (the composition payoff)")
    print("=" * 72)

    def after_wipe(literate):
        """Both towns know farming; the last farmer dies; then we watch whether farming survives and
        the fields keep producing food (the survival proxy — M1.3: farming -> food -> life)."""
        _fresh(); _settlement("S001", (5, 5))
        base = {"tools", "farming"} | ({"writing"} if literate else set())
        master = _agent("Master", (5, 5), knows=base, hunger=1)
        _agent("Heir1", (5, 6), knows=({"writing"} if literate else set()), hunger=1)
        _agent("Heir2", (6, 5), knows=({"writing"} if literate else set()), hunger=1)
        writing.update(world_state, 1)                 # archive farming (if literate)
        # The last living master of farming dies out.
        population.announce_death(master, 2, world_state, cause="old age", final_memory="d", note="d")
        world_state["food"].clear()                    # a barren start — only farming can refill it
        grown = 0
        for t in range(3, 12):
            writing.update(world_state, t)             # re-teach farming from the records (if literate)
            knowledge.farm(world_state, t)             # a living farmer (if any) grows food
            grown += len(world_state["food"])
            world_state["food"].clear()
        knows_farming = any("farming" in a.knowledge for a in world_state["agents"] if a.alive)
        return knows_farming, grown

    lit_knows, lit_food = after_wipe(True)
    ill_knows, ill_food = after_wipe(False)
    print(f"  LITERATE town after its last farmer dies:  farming survives? {lit_knows}, "
          f"food grown over 9 turns = {lit_food}")
    print(f"  ILLITERATE town after its last farmer dies: farming survives? {ill_knows}, "
          f"food grown over 9 turns = {ill_food}  (the fields go barren)")
    assert lit_knows and lit_food > 0
    assert not ill_knows and ill_food == 0
    print("  -> writing is the CURE for the knowledge-extinction collapse: the literate town re-taught")
    print("     farming from its records and kept eating; the illiterate one forgot it forever and starves.")
    print()


# --- RECORDED HISTORY --------------------------------------------------------
def recorded_history() -> None:
    print("=" * 72)
    print("RECORDED HISTORY — a literate settlement keeps a persistent chronicle")
    print("=" * 72)

    def chronicle(literate):
        _fresh(); _settlement("S001", (5, 5))
        _agent("A", (5, 5), knows=({"writing"} if literate else set()))
        _agent("B", (5, 6), knows=({"writing"} if literate else set()))
        # Chronicle entries are the MAJOR events that NAME this settlement (foundings, coronations,
        # uprisings, secessions); faith-scoped events like a prophet's rise name the faith, not the town.
        world_state["events"].append("turn 5: Aldo succeeded Wren as [monarch of S001] (eldest child)")
        world_state["events"].append("turn 5: an UPRISING in S001 — the people rise")
        world_state["events"].append("turn 5: A052 trust in B: 1 -> 2 (talk)")   # minor tick, omitted
        writing.update(world_state, 5)
        return writing.chronicle_of(world_state, "S001")

    lit = chronicle(True)
    print(f"  LITERATE settlement chronicle: {[e['event'] for e in lit]}")
    print(f"  ILLITERATE settlement chronicle: {chronicle(False)}")
    assert len(lit) == 2 and all(("UPRISING" in e["event"] or "succeeded" in e["event"]) for e in lit)
    assert chronicle(False) == []
    print("  -> the literate town records its major history (structured, zero prose); the illiterate")
    print("     town's past vanishes as it happens. This is the substrate the Chronicle (Arc 6) reads.")
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
            main.run_simulation(30, settlements=True, tech_tree=knowledge.TECH_TREE, **kw)
        return buf.getvalue(), dict(llm.get_call_stats())

    off, off_calls = run()
    off2, _ = run(writing_on=False)
    assert off == off2
    print("  --writing OFF: byte-identical to the tech-tree run")
    on_a, on_calls = run(writing_on=True)
    on_b, _ = run(writing_on=True)
    assert on_a == on_b
    print("  --writing ON: two seeded runs byte-identical (records are deterministic state; discovery seeded)")
    assert on_calls == off_calls
    print(f"  writing added ZERO LLM calls (on={on_calls}, off={off_calls}).")
    print()


if __name__ == "__main__":
    saved = llm.PROVIDER
    try:
        headline_1_writing_emerges()
        headline_2_institutions_outlive_individuals()
        headline_3_literacy_cures_extinction()
        recorded_history()
        cost_checks()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M4.10 VERIFIED — writing EMERGES as a tech; institutions (law, knowledge, history) genuinely")
    print("OUTLIVE their makers only where literacy exists; and literacy CURES the knowledge-extinction")
    print("collapse. Memory escapes the individual — civilization gains the power to accumulate.")
    print("=" * 72)
