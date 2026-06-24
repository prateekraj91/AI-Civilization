"""
knowledge.py
============

Knowledge as PROPAGATING STATE — V2 milestone M1.1, the first piece of culture.

The idea
--------
Phase 0 built a scalable social-contact network: at 200+ agents they meet, talk,
and build trust through actual proximity. M1.1 puts the first cultural artefact
ONTO that network — a `knowledge` set on each agent (named facts/skills like
"fire" or "food_location_north") that exists beyond any single agent and SPREADS
between them through contact.

Crucially this is CHEAP STATE DIFFUSION, not an LLM call per learner: when a
knower and a non-knower are adjacent (the same N/S/E/W contact channel talk/steal/
ally already use), the item may be adopted with a probability shaped by how much
the LEARNER trusts the teacher and by the learner's personality. Pure Python, zero
inference, so it scales to hundreds of agents at no added model cost.

What it is NOT (kept strictly in scope)
---------------------------------------
No discovery/invention (that is M1.2) — items are SEEDED into agents (at setup or
via a god grant) and this module only moves EXISTING items along the contact graph.
No tech-changes-the-world effects (that is M1.3) — knowing "fire" does nothing to
an agent's behaviour yet; M1.1 is solely about the item spreading correctly and
cheaply. Knowing an item has no behavioural effect, so a run with NO seeded
knowledge is byte-identical to v1 (and `diffuse` consumes zero RNG then — see the
guard in `diffuse`).

Determinism
-----------
Every iteration over a knowledge set is `sorted(...)` so the order of `rng` draws
is independent of Python's per-process string-hash randomisation — two seeded runs
diffuse identically. Transmissions are computed from a SNAPSHOT of who-knows-what
at the start of the turn and applied after, so an item travels at most one hop per
turn (no within-turn chain reactions) and the result never depends on agent order.
"""

from __future__ import annotations

import random
from typing import Any

import world
from strategy import get_personality

# --- Adoption model --------------------------------------------------------
# Base per-contact-per-turn chance a non-knower adopts an item from an adjacent
# knower, BEFORE trust/personality shaping. Tuned so a single seed knower spreads
# as a believable S-curve through a crowd over tens of turns rather than saturating
# in one or two (see verify_m11).
ADOPTION_BASE = 0.25

# Personality of the LEARNER shapes how readily it takes on a new idea: the curious
# soak it up, the cautious and the independent resist. Multiplies the base chance.
PERSONALITY_ADOPT: dict[str, float] = {
    "curiosity": 1.6,
    "friendliness": 1.2,
    "caution": 0.6,
    "independence": 0.5,
}

# Trust shaping: each point of the learner's trust IN THE TEACHER nudges the chance
# up (and a grudge/low trust nudges it down). A linear bump, then clamped — so a
# trusted friend's idea catches far more readily than a distrusted stranger's, and a
# hated rival's barely at all, without ever hitting 0 or 1.
TRUST_COEFF = 0.12
ADOPTION_MIN = 0.02
ADOPTION_MAX = 0.95


def adoption_probability(learner: Any, teacher: Any, state: dict[str, Any]) -> float:
    """Chance `learner` adopts a known item from adjacent `teacher` this turn.

    Shaped by (a) the learner's personality — curious adopts readily, cautious/
    independent resist — and (b) how much the learner TRUSTS the teacher (the same
    relationships[...]['trust'] Phase 0 maintains). Pure read; no LLM, no mutation.
    A brand-new pair (no relationship yet) sits at neutral trust 0, so strangers
    still share at the base rate — just less than trusted friends do.
    """
    pers = get_personality(learner)
    mult = PERSONALITY_ADOPT.get(pers.dominant, 1.0)
    trust = learner.relationships.get(teacher.name, {}).get("trust", 0)
    p = ADOPTION_BASE * mult * (1.0 + TRUST_COEFF * trust)
    return max(ADOPTION_MIN, min(ADOPTION_MAX, p))


def has_any_knowledge(state: dict[str, Any]) -> bool:
    """True if ANY living agent knows anything — the gate that keeps a no-knowledge
    run byte-identical to v1 (and draws ZERO rng): when this is False, `diffuse`
    returns immediately without touching the contact graph or the RNG stream."""
    return any(a.knowledge for a in state["agents"] if a.alive)


def diffuse(state: dict[str, Any], turn: int,
            rng: "random.Random | None" = None) -> list[tuple[str, str, str]]:
    """Spread knowledge one hop along the contact network for this turn.

    For every adjacent (knower, non-knower) pair, the non-knower may ADOPT a held
    item with `adoption_probability`. Transmissions are decided against a snapshot of
    who-knows-what at turn start and applied afterwards, so an item moves at most one
    hop/turn and the outcome is independent of agent iteration order. Each adoption is
    logged to events[] ("turn 12: A052 taught 'fire' to A101") and recorded as a
    memory on both sides. Returns the list of (teacher, item, learner) for callers.

    Cost: O(agents x neighbours) with O(1) neighbour lookup (the M0.3 occupancy
    index) and ZERO LLM calls. Returns [] and draws no RNG when nobody knows anything
    — so a v1 run with no seeded knowledge is unaffected, byte for byte.
    """
    if not has_any_knowledge(state):
        return []
    draw = (rng or random).random

    living = [a for a in state["agents"] if a.alive]
    by_name = {a.name: a for a in living}
    # Snapshot at turn start: decisions never see this turn's own adoptions.
    snapshot = {a.name: frozenset(a.knowledge) for a in living}
    # learner_name -> {item: teacher_name} (the FIRST teacher to land it wins/logs).
    pending: dict[str, dict[str, str]] = {}

    for teacher in living:  # world_state["agents"] order is stable
        t_known = snapshot[teacher.name]
        if not t_known:
            continue
        neighbours = world.adjacent_agents(teacher, state)
        for lname in sorted(neighbours):  # sorted -> hash-seed-independent rng order
            learner = neighbours[lname]
            l_known = snapshot[lname]
            already = pending.setdefault(lname, {})
            for item in sorted(t_known):
                if item in l_known or item in already:
                    continue
                if draw() < adoption_probability(learner, teacher, state):
                    already[item] = teacher.name

    transmissions: list[tuple[str, str, str]] = []
    for lname in sorted(pending):
        learner = by_name[lname]
        for item in sorted(pending[lname]):
            teacher_name = pending[lname][item]
            learner.knowledge.add(item)
            world.record_memory(learner, f"Learned '{item}' from {teacher_name}")
            world.record_memory(by_name[teacher_name], f"Taught '{item}' to {lname}")
            state["events"].append(f"turn {turn}: {teacher_name} taught '{item}' to {lname}")
            transmissions.append((teacher_name, item, lname))
    return transmissions


def grant(state: dict[str, Any], agent: Any, item: str, turn: int) -> None:
    """Seed `item` into one agent's knowledge and log it (used at setup / by god mode).

    The single place an item ENTERS the world from outside the contact graph. From
    here it can only spread by `diffuse` through actual meetings.
    """
    agent.knowledge.add(item)
    world.record_memory(agent, f"Knows '{item}'")
    state["events"].append(f"turn {turn}: {agent.name} now knows '{item}' (seeded)")
