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
from strategy import SURVIVAL_HUNGER, get_personality

# --- Tech tree (M1.2: discovery) -------------------------------------------
# A tiny prerequisite tree, declared as DATA (not branching code): {item -> the
# items that must be known first}. fire is a base discovery (no prereq); it branches
# into tools and cooking; tools unlocks farming. Discovery walks this map, so adding
# an item is a one-line data change, never new control flow. Kept deliberately small
# and generic — M1.2 is about the INVENTION mechanic, not a real tech tree.
#
# This is the canonical tree. It is only ACTIVE in a run that opts in (run_simulation's
# `tech_tree` arg / the --tech-tree CLI flag); the default trio run gets no tree, so
# `discover` is a no-op that draws zero RNG and v1 stays byte-identical.
TECH_TREE: dict[str, frozenset[str]] = {
    "fire": frozenset(),                 # base: discoverable by anyone
    "tools": frozenset({"fire"}),        # needs fire
    "cooking": frozenset({"fire"}),      # needs fire (a sibling branch of tools)
    "farming": frozenset({"tools"}),     # needs tools (and so, transitively, fire)
}

# --- Discovery model (M1.2) ------------------------------------------------
# Base per-agent-per-turn chance of inventing an item whose prereqs are all known,
# BEFORE personality/situation shaping. Small on purpose: discovery must be a rare
# situational roll, never a timer, so the turn it first fires VARIES by run.
DISCOVERY_BASE = 0.02

# Personality shapes the inventor: the curious tinker most; the cautious least. (The
# independent tinker a fair bit — a loner experimenting alone — the friendly less, as
# they spend their spare capacity socialising.) Multiplies the base chance.
DISCOVERY_PERSONALITY: dict[str, float] = {
    "curiosity": 2.0,
    "independence": 1.0,
    "friendliness": 0.8,
    "caution": 0.6,
}

# Situation: only an agent with spare capacity tinkers — a fed agent invents, a hungry
# one is busy surviving and a starving one not at all. Hunger scales the chance down
# to zero at this cutoff, so discovery rides the SAME scarcity pressure the rest of the
# sim runs on (idle/fed -> invent; starving -> never). Tied to SURVIVAL_HUNGER so it
# lines up with where the executor's survival override already takes over.
DISCOVERY_HUNGER_CUTOFF = SURVIVAL_HUNGER

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


def discovery_probability(agent: Any, item: str, state: dict[str, Any]) -> float:
    """Chance `agent` INVENTS `item` this turn, assuming its prereqs are already met.

    Shaped by (a) personality — the curious tinker most, the cautious least — and
    (b) situation: a fed agent has spare capacity to experiment, a starving one does
    not, so the chance scales linearly down to zero at DISCOVERY_HUNGER_CUTOFF. Pure
    read; no LLM, no mutation. NOT a function of the turn number — there is no timer.
    """
    pers = get_personality(agent)
    mult = DISCOVERY_PERSONALITY.get(pers.dominant, 1.0)
    situation = max(0.0, 1.0 - agent.hunger / DISCOVERY_HUNGER_CUTOFF)
    return DISCOVERY_BASE * mult * situation


def discover(state: dict[str, Any], turn: int,
             tree: "dict[str, frozenset[str]] | None" = None,
             rng: "random.Random | None" = None) -> list[tuple[str, str]]:
    """Let agents INVENT items they don't know whose prerequisites they DO know (M1.2).

    For every living agent and every undiscovered item in `tree` whose prereqs the
    agent already knows, roll `discovery_probability`. A success adds the item to that
    agent's knowledge (from where M1.1 `diffuse` spreads it — no new spread code here)
    and logs "turn 34: A052 discovered 'fire'". Prereq checks read a turn-start
    SNAPSHOT, so an agent can't chain fire->tools->farming in a single turn; each item
    is at most one fresh invention per agent per turn.

    Discovery is probabilistic and situational, NEVER a timer: the turn it first fires
    depends on who is fed, curious, and lucky, so it varies by run. Returns the list of
    (agent, item) discovered. A no-op drawing ZERO rng when `tree` is empty/None — so a
    run with no tech tree is byte-identical to v1.

    Cost: O(agents x tree size), ZERO LLM calls.
    """
    if not tree:
        return []
    draw = (rng or random).random

    living = [a for a in state["agents"] if a.alive]
    # Snapshot knowledge so this turn's inventions don't unlock downstream items until
    # the next turn (no within-turn fire->tools->farming cascade) and order can't matter.
    snapshot = {a.name: frozenset(a.knowledge) for a in living}
    discoveries: list[tuple[str, str]] = []

    for agent in living:  # world_state["agents"] order is stable
        known = snapshot[agent.name]
        for item in sorted(tree):  # sorted -> hash-seed-independent rng order
            if item in known or not tree[item] <= known:
                continue  # already known, or prereqs not met -> cannot invent (gated)
            p = discovery_probability(agent, item, state)
            if p > 0.0 and draw() < p:  # starving agents (p == 0) don't even roll
                agent.knowledge.add(item)
                world.record_memory(agent, f"Discovered '{item}'")
                state["events"].append(f"turn {turn}: {agent.name} discovered '{item}'")
                discoveries.append((agent.name, item))
    return discoveries


def grant(state: dict[str, Any], agent: Any, item: str, turn: int) -> None:
    """Seed `item` into one agent's knowledge and log it (used at setup / by god mode).

    The single place an item ENTERS the world from outside the contact graph. From
    here it can only spread by `diffuse` through actual meetings.
    """
    agent.knowledge.add(item)
    world.record_memory(agent, f"Knows '{item}'")
    state["events"].append(f"turn {turn}: {agent.name} now knows '{item}' (seeded)")
