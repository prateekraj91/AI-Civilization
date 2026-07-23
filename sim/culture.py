"""
culture.py
==========

CULTURAL IDENTITY & FRICTION — the imperial problem (V2 milestone M4.9, CLOSES Arc 3: Belief &
Culture). On top of M4.8 (religion), M4.7 (beliefs), Arc 2 (discontent/uprising), Arc 1 (lineage)
and Phases 0-3.

The historical step M4.9 makes — you can seize a people in a day but not make them yours for decades
----------------------------------------------------------------------------------------------------
M4.7 gave settlements distinct belief sets; M4.8 made a shared set a FAITH with legitimacy power.
M4.9 bundles belief + faith into a CULTURAL IDENTITY and creates the deep imperial problem:
conquering a FOREIGN culture breeds CHRONIC unrest and takes GENERATIONS to assimilate. A ruler of
the same culture integrates almost frictionlessly; a ruler alien to the local culture faces a
province that resents him every turn, is likelier to break away or rise — and only becomes his as its
CHILDREN slowly grow up in his culture while the adults keep theirs. Assimilation is a RACE against
revolt, and neither outcome is scripted.

SCOPE — M4.9 is cultural identity + conquest FRICTION + generational ASSIMILATION, and CLOSES Arc 3.
It builds NO separate named "independence movement": cultural revolt EMERGES from the existing
discontent (M4.5) and revolutionary (M4.6) machinery — culture just SUSTAINS the trust/discontent
dials those systems already read (stated boundary). It writes ONLY trust (through the logged
trust.adjust_trust) and, for assimilation, children's beliefs (the M4.7 belief state) — no other
module's code changes; M4.4/M3.5/M4.5 respond on their own.

How it works (emergent; zero LLM; deterministic friction, seeded assimilation)
------------------------------------------------------------------------------
1. CULTURAL IDENTITY (derived): a settlement's culture SIGNATURE is its dominant belief set (the
   beliefs a CULTURE_MAJORITY of its living members hold) — bundling M4.7 beliefs and, through them,
   the M4.8 faith. `same_culture(a, b)` compares two agents by belief overlap (Jaccard >= threshold);
   an agent is FOREIGN to a culture if it shares fewer than CULTURE_SHARE of that culture's signature.
2. CONQUEST FRICTION (the teeth, SUSTAINED): each turn, in a settlement ruled by a FORCE ruler alien
   to its culture, every member still foreign to that ruler withdraws a little trust from him
   (CULTURE_TRUST_STEP, bounded) — a chronic loyalty tax the SAME extraction never levies on a
   same-culture crown. That trust drop is what the UNCHANGED systems read: M4.4 sharpens the province's
   discontent, M3.5 erodes its vassal toward BREAKAWAY, M4.5 tips it toward REVOLT. An empire of many
   cultures carries permanent fault lines.
3. GENERATIONAL ASSIMILATION (composition with M4.1): under sustained foreign rule, dependent CHILDREN
   raised in the town adopt the RULER's beliefs at the childhood rate (ASSIMILATION_RATE) — the
   ruling culture absorbs the young — while ADULTS keep theirs. So the town's signature DRIFTS toward
   the ruler's only as generations turn over; once enough members share his culture the ruler is no
   longer foreign and the friction fades (the fault line healed). Lose the foreign ruler first and the
   town reverts to its own culture, the already-assimilated children keeping the partial drift. A race.
"""

from __future__ import annotations

import math
import random
from typing import Any

from sim import beliefs
from sim import religion
from sim import trust
from sim import world

# --- Tunable constants -------------------------------------------------------
CULTURE_MAJORITY = 0.5   # a belief is part of a settlement's cultural SIGNATURE if this fraction hold it
CULTURE_SHARE = 0.5      # an agent is "native" to a culture if it shares at least this fraction of its signature
CULTURE_SIM_THRESHOLD = 0.5  # same_culture(a, b): Jaccard overlap of their belief sets at/above this

# Conquest friction (deterministic integer trust math, like the M3.5 tribute backlash):
CULTURE_TRUST_STEP = 1   # trust a foreign-ruled member withdraws from its alien ruler per turn (SUSTAINED)
CULTURE_TRUST_BOUND = 6  # ...capped here so culture cannot push trust to absurd values on its own

# Generational assimilation (the ONLY RNG this module draws — gated on culture_on, so off is byte-identical):
ASSIMILATION_RATE = 0.15  # per foreign-ruled dependent CHILD per turn: chance to adopt one ruler belief it lacks


def _living(state: dict[str, Any]) -> dict[str, Any]:
    return {a.name: a for a in state["agents"] if a.alive}


# --- 1. Cultural identity (derived from beliefs) -----------------------------
def culture_signature(state: dict[str, Any], sid: str) -> "frozenset[str]":
    """Settlement `sid`'s cultural SIGNATURE: the beliefs a CULTURE_MAJORITY of its living members
    hold (the dominant set that IS its identity — M4.7 beliefs, and through them the M4.8 faith). Pure read."""
    rec = state.get("settlements", {}).get(sid)
    if rec is None:
        return frozenset()
    living = _living(state)
    n = sum(1 for m in rec["members"] if m in living)
    if n == 0:
        return frozenset()
    need = math.ceil(CULTURE_MAJORITY * n)
    counts = beliefs.belief_counts(sid, state)
    return frozenset(b for b, c in counts.items() if c >= need)


def _shares(held: "set[str]", signature: "frozenset[str]") -> bool:
    """True if `held` covers at least CULTURE_SHARE of `signature` — i.e. is NATIVE to that culture."""
    if not signature:
        return True  # a culture-less town has no foreigners
    return len(held & signature) >= math.ceil(CULTURE_SHARE * len(signature))


def same_culture(a: Any, b: Any, state: dict[str, Any]) -> bool:
    """Do agents `a` and `b` share a culture? Jaccard overlap of their belief sets >= threshold.

    The spec's agent-to-agent comparator: near-identical belief sets = same culture, divergent =
    foreign. Two beliefless agents count as trivially same (no culture to differ over). Pure read."""
    A, B = beliefs.agent_beliefs(a.name, state), beliefs.agent_beliefs(b.name, state)
    if not A and not B:
        return True
    union = len(A | B)
    return union > 0 and len(A & B) / union >= CULTURE_SIM_THRESHOLD


def ruler_is_foreign(state: dict[str, Any], sid: str, ruler: str) -> bool:
    """True if `ruler` is ALIEN to settlement `sid`'s culture (shares < CULTURE_SHARE of its signature)."""
    sig = culture_signature(state, sid)
    if not sig:
        return False
    return not _shares(beliefs.agent_beliefs(ruler, state), sig)


def is_foreign_ruled(state: dict[str, Any], sid: str) -> bool:
    """True if `sid` has a force ruler alien to its culture — a chronic imperial fault line. Pure read."""
    ruler = religion._sovereign(state, sid)
    return ruler is not None and ruler_is_foreign(state, sid, ruler)


def assimilation_progress(state: dict[str, Any], sid: str) -> float:
    """Fraction of `sid`'s living members who now share the RULER's culture — near 0.0 at the conquest
    of a foreign town, rising MONOTONICALLY toward 1.0 as its generations assimilate (and staying high
    after the fault line has healed). 0.0 when there is no ruler with beliefs. Pure read. (The summary
    only DISPLAYS this while `is_foreign_ruled`, so a healed or native town does not show progress.)"""
    ruler = religion._sovereign(state, sid)
    if ruler is None:
        return 0.0
    ruler_bel = beliefs.agent_beliefs(ruler, state)
    if not ruler_bel:
        return 0.0
    rec = state.get("settlements", {}).get(sid)
    living = _living(state)
    members = [m for m in rec["members"] if m in living and m != ruler]
    if not members:
        return 0.0
    assimilated = sum(1 for m in members if _shares(beliefs.agent_beliefs(m, state), frozenset(ruler_bel)))
    return assimilated / len(members)


# --- 2. Conquest friction (SUSTAINED; writes trust only) ---------------------
def apply_friction(state: dict[str, Any], turn: int) -> int:
    """Each turn, members foreign to their settlement's alien ruler withdraw trust from him (M4.9).

    Only fires where the force ruler (`religion._sovereign`) is FOREIGN to the town's culture, and only
    for members still foreign to that ruler (an assimilated member no longer resents him). Writes ONLY
    trust (logged); the UNCHANGED M4.4/M3.5/M4.5 systems turn the sustained loyalty tax into discontent,
    breakaway and revolt. Deterministic (integer steps, ZERO RNG). Returns the number of loyalties moved."""
    living = _living(state)
    moved = 0
    for sid in sorted(state.get("settlements", {})):
        ruler = religion._sovereign(state, sid)
        if ruler is None or ruler not in living or not ruler_is_foreign(state, sid, ruler):
            continue
        ruler_bel = frozenset(beliefs.agent_beliefs(ruler, state))
        rec = state["settlements"][sid]
        for name in sorted(rec["members"]):
            if name == ruler or name not in living:
                continue
            if _shares(beliefs.agent_beliefs(name, state), ruler_bel):
                continue  # a member who has taken on the ruler's culture no longer resents him
            member = living[name]
            cur = member.relationships.get(ruler, {}).get("trust", 0)
            if cur <= -CULTURE_TRUST_BOUND:
                continue  # culture's own push is capped; other systems may move it further
            trust.adjust_trust(member, ruler, -CULTURE_TRUST_STEP,
                               "foreign rule (cultural friction)", turn, state)
            moved += 1
    return moved


# --- 3. Generational assimilation (children adopt the ruler's culture) -------
def assimilate(state: dict[str, Any], turn: int, rng: "random.Random | None" = None) -> list[str]:
    """Under sustained foreign rule, dependent CHILDREN adopt the ruler's beliefs at the childhood rate.

    Only dependent children (the M4.1 gate) in a FOREIGN-ruled settlement are pulled toward the ruler's
    culture — adults keep theirs — so the town's signature drifts only as generations turn over. Adopting
    a ruler belief drops its M4.7 contradiction (the ruling culture is the dominant source). Draws RNG
    (the module's only randomness; gated on culture_on so an off run is byte-identical). Returns events."""
    draw = (rng or random).random
    living = _living(state)
    events: list[str] = []
    for sid in sorted(state.get("settlements", {})):
        ruler = religion._sovereign(state, sid)
        if ruler is None or ruler not in living or not ruler_is_foreign(state, sid, ruler):
            continue
        ruler_bel = beliefs.agent_beliefs(ruler, state)
        if not ruler_bel:
            continue
        rec = state["settlements"][sid]
        for name in sorted(rec["members"]):
            child = living.get(name)
            if child is None or name == ruler or not world.is_dependent_child(child, state):
                continue
            held = state.setdefault("beliefs", {}).setdefault(name, set())
            missing = [b for b in sorted(ruler_bel) if b not in held]
            if not missing or draw() >= ASSIMILATION_RATE:
                continue
            belief = missing[0]
            held.add(belief)
            contra = beliefs.CONTRADICTS.get(belief)
            renounce = ""
            if contra is not None and contra in held:
                held.discard(contra)
                renounce = f" (over '{contra}')"
            world.record_memory(child, f"Raised into {ruler}'s culture: '{belief}'{renounce}")
            events.append(f"turn {turn}: {name} was raised into {ruler}'s culture "
                          f"('{belief}'{renounce})")
    state.setdefault("events", []).extend(events)
    return events


def update(state: dict[str, Any], turn: int, rng: "random.Random | None" = None) -> list[str]:
    """Advance cultural friction + assimilation one turn (M4.9). Runs AFTER religion.update so it reads
    this turn's beliefs/faith. Friction is deterministic; assimilation draws seeded RNG. Caller gates on
    `culture_on`, so an off run never calls this and stays byte-identical. Returns events."""
    moved = apply_friction(state, turn)
    events = assimilate(state, turn, rng)
    if moved:
        events.append(f"turn {turn}: cultural friction moved {moved} loyalties in foreign-ruled towns")
    return events
