"""
verify_m47.py
=============

Deterministic verification of V2 milestone M4.7: BELIEFS EMERGE — the inner life.
First milestone of Arc 3 (Belief & Culture), on top of Arc 2 (discontent/uprising/
revolutionary), Arc 1 (lineage/dynasties) and Phases 0-3.

Run offline (Ollama OFF, no model server, no seed-search):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m47.py

The historical step: agents had personalities, knowledge, trust, wealth and politics
— but no IDEAS ABOUT THE WORLD. M4.7 gives the civilization an inner life: short fixed
BELIEF strings that FORM from lived experience and SPREAD by trusted contact like
knowledge. Beliefs are STATE, never generated text (ZERO LLM). Scope is formation +
spread only — no priests, faiths, or legitimacy effects yet (M4.8/M4.9).

HEADLINE 1 — BELIEFS ARE EARNED, NOT ASSIGNED: each formation condition is individually
             demonstrable — abundance -> "the land provides"; starvation -> "the world
             is cruel"; turns under a force ruler -> "the strong take what they want"; a
             producer skill while fed -> "knowledge is power"; and an agent who lives
             through none of them forms NOTHING.
HEADLINE 2 — BELIEFS SPREAD BY TRUST: a belief transmits FAR faster from a trusted
             neighbour than a distrusted one; an isolated agent never acquires one; a
             CONTRADICTORY belief flips the old one only from a trusted source; and the
             curious adopt faster than the independent (reused personality weights).
HEADLINE 3 — PROTO-CULTURES EMERGE: two settlements living different lives grow distinct
             DOMINANT belief sets — a town that PROSPERED believes "the land provides"
             while one that STARVED under a crown believes "the world is cruel / the
             strong take what they want". Nobody assigned a culture; shared experience +
             trust-weighted spread produced one.
CHILDREN   — a dependent child soaks up its parent's belief through the childhood window.
COST       — zero added LLM; --beliefs off byte-identical; deterministic/reproducible.
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any

import beliefs
import llm
import main
import trust
import world
from agents import Agent
from world import world_state


# --- Staging helpers ---------------------------------------------------------
def _fresh() -> None:
    world.create_world()
    world_state["beliefs_on"] = True


def _settlement(sid: str, center: tuple[int, int]) -> None:
    world_state["settlements"][sid] = {"id": sid, "center": center, "members": set(), "founded": 0}


def _agent(name, pos, *, hunger=1, money=0.0, sid="S001", personality="friendly and outgoing",
           knows=None) -> Agent:
    a = Agent(name=name, personality=personality)
    world.place_agent(a, *pos)
    a.hunger, a.age, a.lifespan, a.money, a.settlement = hunger, 30, 100, money, sid
    if knows:
        a.knowledge.update(knows)
    if sid is not None and sid in world_state["settlements"]:
        world_state["settlements"][sid]["members"].add(name)
    return a


def _monarch(sid, name) -> None:
    world_state["monarchs"][sid] = {"monarch": name, "since": 0, "garrison": set()}


# --- HEADLINE 1: beliefs are earned, not assigned ----------------------------
def headline_1_beliefs_are_earned() -> None:
    print("=" * 72)
    print("HEADLINE 1 — BELIEFS ARE EARNED, NOT ASSIGNED (each condition binds; a control forms none)")
    print("=" * 72)

    def lived(build, turns) -> set:
        _fresh()
        _settlement("S001", (5, 5))
        who = build()
        for t in range(1, turns + 1):
            beliefs._update_experience(world_state, t)
        beliefs.form(world_state, turns + 1)
        return beliefs.agent_beliefs(who, world_state)

    fed = lived(lambda: _agent("Fed", (5, 5), hunger=1).name, beliefs.ABUNDANCE_TURNS + 1)
    starved = lived(lambda: _agent("Starve", (5, 5), hunger=8).name, beliefs.HARDSHIP_TURNS + 1)

    def under_crown():
        _agent("King", (6, 5), money=100.0)
        _monarch("S001", "King")
        return _agent("Subj", (5, 5), hunger=1, money=8.0).name
    ruled = lived(under_crown, beliefs.EXTRACTION_TURNS + 1)
    skilled = lived(lambda: _agent("Farmer", (5, 5), hunger=1, knows={"farming"}).name,
                    beliefs.SKILL_FED_TURNS + 1)
    control = lived(lambda: _agent("Meh", (5, 5), hunger=4).name, 15)

    print(f"  sustained abundance   -> {sorted(fed)}")
    print(f"  sustained starvation  -> {sorted(starved)}")
    print(f"  turns under a crown   -> {sorted(ruled)}")
    print(f"  producer skill + fed  -> {sorted(skilled)}")
    print(f"  none of the above     -> {sorted(control)}")
    assert beliefs.LAND_PROVIDES in fed and beliefs.WORLD_IS_CRUEL in starved
    assert beliefs.STRONG_TAKE in ruled and beliefs.KNOWLEDGE_IS_POWER in skilled
    assert control == set()
    print("  -> every belief is earned by a concrete threshold on lived experience; nothing assigned.")
    print()


# --- HEADLINE 2: beliefs spread by trust -------------------------------------
def headline_2_beliefs_spread_by_trust() -> None:
    print("=" * 72)
    print("HEADLINE 2 — BELIEFS SPREAD BY TRUST (trusted fast, distrusted slow, isolated never)")
    print("=" * 72)

    def cohort(trust_val, turns, personality="friendly and outgoing") -> int:
        adopted = 0
        for i in range(30):
            _fresh()
            _settlement("S001", (5, 5))
            _agent(f"S{i}", (5, 5))
            learner = _agent(f"L{i}", (5, 6), personality=personality)
            world_state["beliefs"] = {f"S{i}": {beliefs.LAND_PROVIDES}}
            trust.ensure_relationship(learner, f"S{i}")["trust"] = trust_val
            rng = random.Random(100 + i)
            for t in range(1, turns + 1):
                beliefs.spread(world_state, t, rng)
            adopted += beliefs.LAND_PROVIDES in beliefs.agent_beliefs(f"L{i}", world_state)
        return adopted

    trusted, distrusted = cohort(5, 4), cohort(-5, 4)
    print(f"  a belief adopted within 4 turns:  trusted (+5) {trusted}/30  vs  distrusted (-5) {distrusted}/30")
    assert trusted > distrusted + 8

    curious = cohort(0, 3, "curious and creative")
    independent = cohort(0, 3, "independent and strong-willed")
    print(f"  personality receptivity (3 turns): curious {curious}/30  vs  independent {independent}/30")
    assert curious > independent

    # Isolation: no contact, never adopts.
    _fresh(); _settlement("S001", (5, 5))
    _agent("Src", (1, 1)); _agent("Iso", (9, 9))
    world_state["beliefs"] = {"Src": {beliefs.LAND_PROVIDES}}
    rng = random.Random(1)
    for t in range(1, 30):
        beliefs.spread(world_state, t, rng)
    isolated = beliefs.LAND_PROVIDES in beliefs.agent_beliefs("Iso", world_state)
    print(f"  an isolated agent (no contact) ever adopts? {isolated}")
    assert not isolated

    # Contradiction flip only from a trusted source.
    def flip(trust_val) -> set:
        _fresh(); _settlement("S001", (5, 5))
        _agent("Src", (5, 5)); lrn = _agent("Lrn", (5, 6))
        world_state["beliefs"] = {"Src": {beliefs.WORLD_IS_CRUEL}, "Lrn": {beliefs.LAND_PROVIDES}}
        trust.ensure_relationship(lrn, "Src")["trust"] = trust_val
        rng = random.Random(1)
        for t in range(1, 30):
            beliefs.spread(world_state, t, rng)
        return beliefs.agent_beliefs("Lrn", world_state)

    print(f"  contradiction from a TRUSTED source (+5)  -> {sorted(flip(5))}")
    print(f"  contradiction from an UNTRUSTED source (0)-> {sorted(flip(0))}")
    assert flip(5) == {beliefs.WORLD_IS_CRUEL} and flip(0) == {beliefs.LAND_PROVIDES}
    print("  -> beliefs ride the trust network exactly as knowledge does; a worldview flips only")
    print("     for a trusted mouth. Consent of the believer, not coercion.")
    print()


# --- HEADLINE 3: proto-cultures emerge ---------------------------------------
def headline_3_proto_cultures_emerge() -> None:
    print("=" * 72)
    print("HEADLINE 3 — PROTO-CULTURES EMERGE (two towns, two histories, two belief sets)")
    print("=" * 72)

    _fresh()
    # PROSPER (S001, far west): well-fed, cohesive (high mutual trust -> fast spread + solidarity).
    _settlement("S001", (1, 1))
    prosper = [_agent(n, p, hunger=1, sid="S001") for n, p in
               [("Pa", (1, 1)), ("Pb", (1, 2)), ("Pc", (2, 1)), ("Pd", (2, 2))]]
    for a in prosper:
        for b in prosper:
            if a is not b:
                trust.ensure_relationship(a, b.name)["trust"] = 3
    # SUFFER (S002, far east): starving under a levying crown; modest peer trust (base-rate spread).
    _settlement("S002", (8, 8))
    _agent("Tyrant", (9, 9), money=100.0, sid="S002")
    _monarch("S002", "Tyrant")
    suffer = [_agent(n, p, hunger=8, money=8.0, sid="S002") for n, p in
              [("Sa", (7, 8)), ("Sb", (8, 7)), ("Sc", (8, 8))]]
    for a in suffer:
        for b in suffer:
            if a is not b:
                trust.ensure_relationship(a, b.name)["trust"] = 1

    rng = random.Random(7)
    for turn in range(1, 16):
        beliefs.update(world_state, turn, rng)

    prosper_profile = beliefs.dominant_beliefs("S001", world_state)
    suffer_profile = beliefs.dominant_beliefs("S002", world_state)
    print("  PROSPER town S001 (fed, cohesive):")
    for b, n in prosper_profile:
        print(f"      '{b}' x{n}")
    print("  SUFFER town S002 (starving under a crown):")
    for b, n in suffer_profile:
        print(f"      '{b}' x{n}")
    prosper_set = {b for b, _ in prosper_profile}
    suffer_set = {b for b, _ in suffer_profile}
    assert beliefs.LAND_PROVIDES in prosper_set and beliefs.LAND_PROVIDES not in suffer_set
    assert (beliefs.WORLD_IS_CRUEL in suffer_set or beliefs.STRONG_TAKE in suffer_set)
    assert prosper_set != suffer_set
    print("  -> nobody assigned a culture: what each town SUFFERED (or enjoyed), spread by trust,")
    print("     produced two distinct belief systems side by side.")
    print()


# --- CHILDREN inherit the settlement's beliefs -------------------------------
def children_inherit_beliefs() -> None:
    print("=" * 72)
    print("CHILDREN — a child born into a 'the world is cruel' home grows up believing it")
    print("=" * 72)

    _fresh()
    world_state["lineage_on"] = True
    _settlement("S001", (5, 5))
    _agent("Parent", (5, 5))
    child = _agent("Child", (5, 6))
    child.age, child.dependent, child.parents = 6, True, ("Parent", "Q")
    trust.ensure_relationship(child, "Parent")["trust"] = 4
    world_state["beliefs"] = {"Parent": {beliefs.WORLD_IS_CRUEL}}
    rng = random.Random(3)
    got = None
    for turn in range(1, 20):
        beliefs.spread(world_state, turn, rng)
        if beliefs.WORLD_IS_CRUEL in beliefs.agent_beliefs("Child", world_state):
            got = turn
            break
    print(f"  the dependent child took up its parent's belief by turn {got} (childhood boost).")
    assert got is not None
    print("  -> culture is inherited by UPBRINGING, not by blood.")
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
            main.run_simulation(30, settlements=True, **kw)
        return buf.getvalue(), dict(llm.get_call_stats())

    off, off_calls = run()
    off2, _ = run(beliefs_on=False)
    assert off == off2
    print("  --beliefs OFF: byte-identical to the v1 settlements run")
    on_a, on_calls = run(beliefs_on=True)
    on_b, _ = run(beliefs_on=True)
    assert on_a == on_b
    print("  --beliefs ON: two seeded runs byte-identical (formation deterministic, spread seeded)")
    assert on_calls == off_calls
    print(f"  beliefs added ZERO LLM calls (on={on_calls}, off={off_calls}) — they are STATE, not text.")
    print()


if __name__ == "__main__":
    saved = llm.PROVIDER
    try:
        headline_1_beliefs_are_earned()
        headline_2_beliefs_spread_by_trust()
        headline_3_proto_cultures_emerge()
        children_inherit_beliefs()
        cost_checks()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M4.7 VERIFIED — beliefs genuinely FORM from lived experience (each condition binding,")
    print("nothing assigned), SPREAD by trust exactly as knowledge does, and CLUSTER into proto-")
    print("cultures that differ by what a settlement has SUFFERED. The civilization gains an inner")
    print("life — and M4.8 will turn a shared belief into a faith.")
    print("=" * 72)
