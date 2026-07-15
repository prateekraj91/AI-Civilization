"""
beliefs.py
==========

BELIEFS EMERGE — the inner life (V2 milestone M4.7, OPENS Arc 3: Belief & Culture). On top of Arc 2
(M4.4 discontent, M4.5 uprising, M4.6 the revolutionary), Arc 1 (lineage/dynasties) and Phases 0-3.

The historical step M4.7 makes — the civilization gains IDEAS ABOUT THE WORLD
-----------------------------------------------------------------------------
Through Arc 2 agents had personalities, knowledge, trust, wealth and politics — but no BELIEFS: no
stories about what the world IS. Real institutions ran on legitimacy stories, not just trust math.
M4.7 gives the civilization an inner life: short, fixed BELIEF strings that FORM from what an agent
has actually lived through, and SPREAD between agents like knowledge. Beliefs are STATE, never
model-generated text — ZERO LLM.

SCOPE — M4.7 IS BELIEF FORMATION AND SPREAD ONLY. It does NOT build priests, faiths-as-institutions,
ruler-legitimacy effects, or cultural friction — those are M4.8 (religion) and M4.9 (cultural
identity). Beliefs here are inert culture: they are held, they spread, they cluster — but they confer
no mechanical power yet. This module never grants a bonus, changes a vote, or wins a battle; it only
reads lived experience and writes/spreads belief strings, exposing a legible per-agent + per-settlement
readout for M4.8/M4.9 to build on.

Two ways a belief enters a mind
-------------------------------
1. FORMATION FROM EXPERIENCE (deterministic — ZERO RNG). Each belief in the CATALOGUE has a concrete
   FORMATION CONDITION read from lived state (a streak of being fed, a streak of starvation, turns
   spent under a force ruler, deaths witnessed, ...). When an agent's accumulated experience crosses
   the threshold, it COMES TO BELIEVE — earned, never assigned or random. Living your own truth also
   drops its CONTRADICTION (your new reality overrides the old story).
2. SPREAD BY CONTACT (reuses the M1.1 diffusion machinery). Beliefs transmit one hop along the contact
   network with the EXACT adoption probability knowledge uses — `knowledge.adoption_probability`
   (trust-weighted, personality-shaped, child-boosted) — so a trusted neighbour's belief catches far
   more readily than a distrusted stranger's, the curious adopt faster than the independent, and a
   dependent CHILD soaks up its parents' beliefs through the childhood learning window (culture is
   inherited by upbringing, not by blood). A CONTRADICTORY belief FLIPS the old one only when the
   source is trusted enough (>= FLIP_TRUST) — you do not rewrite your worldview for a stranger.

Cost & determinism
------------------
ZERO LLM (beliefs are fixed catalogue strings). Formation is deterministic threshold-math; spread
draws RNG from the seeded stream exactly as knowledge diffusion does (and no-ops drawing no RNG when
nobody holds a belief yet). All per-agent state lives in world_state ("beliefs": {name: set[str]},
"belief_exp": experience counters) — no new Agent field. A run with the system OFF never calls
`update`, so it is byte-identical to v1. Imports knowledge (for the shared adoption model) + economy
(producer skills) + trust + world (one-directional — a higher layer than any of them).
"""

from __future__ import annotations

import random
from typing import Any

import economy
import knowledge
import trust
import world

# --- The belief catalogue (fixed strings — never generated) ------------------
# Each belief is a short worldview an agent can hold. A CONTRADICTION pair is two beliefs that cannot
# be held at once: adopting one drops the other. Kept explicit and small.
LAND_PROVIDES = "the land provides"
WORLD_IS_CRUEL = "the world is cruel"
STRONG_TAKE = "the strong take what they want"
STRONGER_TOGETHER = "we are stronger together"
KNOWLEDGE_IS_POWER = "knowledge is power"
DEAD_WATCH = "the dead watch us"
WEALTH_IS_VIRTUE = "wealth is virtue"
GREED_IS_POISON = "greed is a poison"

CATALOGUE = (LAND_PROVIDES, WORLD_IS_CRUEL, STRONG_TAKE, STRONGER_TOGETHER,
             KNOWLEDGE_IS_POWER, DEAD_WATCH, WEALTH_IS_VIRTUE, GREED_IS_POISON)

# Contradictory pairs (symmetric): holding one is incompatible with the other. Optimism vs despair,
# solidarity vs cynicism, and the moral valence of wealth.
CONTRADICTS: dict[str, str] = {
    LAND_PROVIDES: WORLD_IS_CRUEL,   WORLD_IS_CRUEL: LAND_PROVIDES,
    STRONGER_TOGETHER: STRONG_TAKE,  STRONG_TAKE: STRONGER_TOGETHER,
    WEALTH_IS_VIRTUE: GREED_IS_POISON, GREED_IS_POISON: WEALTH_IS_VIRTUE,
}

# --- Formation thresholds (tunable; each a streak/count of lived experience) --
WELL_FED_HUNGER = 2       # hunger at/below this counts as "fed" for the abundance streak
HARDSHIP_HUNGER = 6       # hunger at/above this counts as "starving" for the hardship streak
WEALTH_RICH = 15.0        # wealth at/above this counts as "prosperous"
PLENTY_WEALTH = 12.0      # a co-settler holding this much is "plenty" for the deprivation check

ABUNDANCE_TURNS = 8       # fed this long -> "the land provides"
HARDSHIP_TURNS = 6        # starving this long -> "the world is cruel"
EXTRACTION_TURNS = 5      # under a force ruler this long -> "the strong take what they want"
SOLIDARITY_TURNS = 6      # in a trusting settled group this long -> "we are stronger together"
SKILL_FED_TURNS = 3       # knows a producer skill AND fed this long -> "knowledge is power"
MANY_DEATHS = 4           # this many deaths witnessed -> "the dead watch us"
RICH_TURNS = 6            # prosperous this long -> "wealth is virtue"
DEPRIVED_TURNS = 5        # hungry amid a neighbour's plenty this long -> "greed is a poison"

# --- Spread model ------------------------------------------------------------
# The per-contact adoption probability is REUSED wholesale from knowledge diffusion
# (`knowledge.adoption_probability`): trust-weighted, personality-shaped, child-boosted. Only this one
# extra rule is belief-specific: a CONTRADICTORY belief overwrites the incumbent one only if the
# learner trusts the source at least this much — a worldview flip needs a trusted mouth, not a stranger's.
FLIP_TRUST = trust.HIGH_THRESHOLD  # = 2


def _beliefs(state: dict[str, Any]) -> dict[str, set]:
    return state.setdefault("beliefs", {})


def _exp(state: dict[str, Any]) -> dict[str, Any]:
    return state.setdefault("belief_exp", {})


def agent_beliefs(name: str, state: dict[str, Any]) -> set:
    """`name`'s current belief set (empty if none) — a pure read."""
    return state.get("beliefs", {}).get(name, set())


# --- Reading lived experience (all pure reads of existing state) -------------
def _wealth(a: Any) -> float:
    return a.money + a.stockpile


def _force_ruler(state: dict[str, Any], sid: str, self_name: str) -> bool:
    """True if a FORCE ruler (a monarch, or a vassal lord in a realm) OTHER than `self_name` rules
    settlement `sid` — the "living under those who take by force" signal (M3.4/M3.5). Pure read."""
    for king in sorted(state.get("kingdoms", {})):
        vassal = state["kingdoms"][king]["vassals"].get(sid)
        if vassal is not None:
            return vassal != self_name
    mon = state.get("monarchs", {}).get(sid)
    return mon is not None and mon["monarch"] != self_name


def _in_trusting_group(agent: Any, state: dict[str, Any]) -> bool:
    """True if `agent` is a dependent child (raised by kin) OR a settled member who trusts >= 2 of its
    co-settlers highly — the lived basis of "we are stronger together". Pure read of trust + settlement."""
    if world.is_dependent_child(agent, state):
        return True
    sid = getattr(agent, "settlement", None)
    if sid is None:
        return False
    rec = state.get("settlements", {}).get(sid)
    if rec is None:
        return False
    trusted = sum(1 for m in rec["members"] if m != agent.name
                  and agent.relationships.get(m, {}).get("trust", 0) >= trust.HIGH_THRESHOLD)
    return trusted >= 2


def _deprived_amid_plenty(agent: Any, state: dict[str, Any]) -> bool:
    """True if `agent` is hungry while a co-settler hoards plenty — the lived basis of "greed is a
    poison" (the same felt local inequality M4.4 reads). Pure read."""
    if agent.hunger < HARDSHIP_HUNGER:
        return False
    sid = getattr(agent, "settlement", None)
    rec = state.get("settlements", {}).get(sid) if sid is not None else None
    if rec is None:
        return False
    living = {a.name: a for a in state["agents"] if a.alive}
    return any(m != agent.name and (o := living.get(m)) is not None and _wealth(o) >= PLENTY_WEALTH
               for m in rec["members"])


def _update_experience(state: dict[str, Any], turn: int) -> None:
    """Advance every living agent's lived-experience counters from THIS turn's state (ZERO RNG).

    Streak counters (fed/hungry/rich) reset when the condition lapses; cumulative counters
    (extracted/solidarity/deprived/deaths) only grow — a life's tally of what it has been through.
    """
    exp = _exp(state)
    living = [a for a in state["agents"] if a.alive]
    names = {a.name for a in living}

    # Deaths witnessed: credit every currently-living agent with the deaths that happened since the
    # last update (monotonic dead-count diff — no fragile event parsing).
    meta = exp.setdefault("__world__", {})
    dead_now = sum(1 for a in state["agents"] if not a.alive)
    new_deaths = 0 if "dead_count" not in meta else max(0, dead_now - meta["dead_count"])
    meta["dead_count"] = dead_now

    # Drop counters for agents no longer living (a respawn reusing a name starts fresh).
    for gone in [n for n in exp if n != "__world__" and n not in names]:
        del exp[gone]

    for a in living:
        c = exp.setdefault(a.name, {"fed": 0, "hungry": 0, "rich": 0,
                                    "extracted": 0, "solidarity": 0, "deprived": 0, "deaths": 0})
        c["fed"] = c["fed"] + 1 if a.hunger <= WELL_FED_HUNGER else 0
        c["hungry"] = c["hungry"] + 1 if a.hunger >= HARDSHIP_HUNGER else 0
        c["rich"] = c["rich"] + 1 if _wealth(a) >= WEALTH_RICH else 0
        if not world.is_dependent_child(a, state) and _force_ruler(state, getattr(a, "settlement", None), a.name):
            c["extracted"] += 1
        if _in_trusting_group(a, state):
            c["solidarity"] += 1
        if _deprived_amid_plenty(a, state):
            c["deprived"] += 1
        c["deaths"] += new_deaths


# --- Formation: experience crosses a threshold -> a belief is BORN -----------
def _formed(counters: dict[str, int], agent: Any) -> list[str]:
    """The beliefs whose formation condition `counters` (+ the agent's skills) currently satisfy."""
    out: list[str] = []
    if counters["fed"] >= ABUNDANCE_TURNS:
        out.append(LAND_PROVIDES)
    if counters["hungry"] >= HARDSHIP_TURNS:
        out.append(WORLD_IS_CRUEL)
    if counters["extracted"] >= EXTRACTION_TURNS:
        out.append(STRONG_TAKE)
    if counters["solidarity"] >= SOLIDARITY_TURNS:
        out.append(STRONGER_TOGETHER)
    if counters["fed"] >= SKILL_FED_TURNS and any(s in agent.knowledge for s in economy.PRODUCER_SKILLS):
        out.append(KNOWLEDGE_IS_POWER)
    if counters["deaths"] >= MANY_DEATHS:
        out.append(DEAD_WATCH)
    if counters["rich"] >= RICH_TURNS:
        out.append(WEALTH_IS_VIRTUE)
    if counters["deprived"] >= DEPRIVED_TURNS:
        out.append(GREED_IS_POISON)
    return out


def form(state: dict[str, Any], turn: int) -> list[str]:
    """Every living agent forms any belief its lived experience now warrants (deterministic, ZERO RNG).

    Adding a belief drops its CONTRADICTION (your own lived truth is the most trusted source, so it
    overrides the old story). Each new belief is logged sparingly ("X came to believe: ..."). Returns
    the event strings.
    """
    beliefs = _beliefs(state)
    exp = _exp(state)
    events: list[str] = []
    for a in sorted((x for x in state["agents"] if x.alive), key=lambda x: x.name):
        counters = exp.get(a.name)
        if counters is None:
            continue
        held = beliefs.setdefault(a.name, set())
        for belief in _formed(counters, a):
            if belief in held:
                continue
            held.add(belief)
            contra = CONTRADICTS.get(belief)
            dropped = ""
            if contra is not None and contra in held:
                held.discard(contra)
                dropped = f" (renouncing '{contra}')"
            ev = f"turn {turn}: {a.name} came to believe '{belief}'{dropped}"
            events.append(ev)
            world.record_memory(a, f"Came to believe '{belief}'{dropped}")
    state.setdefault("events", []).extend(events)
    return events


# --- Spread: one hop along the contact network (reuses the M1.1 model) --------
def _has_any_belief(state: dict[str, Any]) -> bool:
    return any(state.get("beliefs", {}).get(a.name) for a in state["agents"] if a.alive)


def spread(state: dict[str, Any], turn: int, rng: "random.Random | None" = None) -> list[str]:
    """Spread beliefs one hop this turn, TRUST-weighted exactly as knowledge diffuses (M1.1).

    For every adjacent (holder, non-holder) pair the non-holder may ADOPT a belief with
    `knowledge.adoption_probability` (trust + personality + childhood boost — the SAME model). A belief
    that CONTRADICTS one the learner already holds only lands if the learner trusts the source >=
    FLIP_TRUST, and then it FLIPS (the old belief is renounced). Decided against a turn-start snapshot so
    a belief moves at most one hop/turn, order-independent. No-ops drawing NO RNG when nobody believes
    anything yet. Returns the event strings.
    """
    if not _has_any_belief(state):
        return []
    draw = (rng or random).random
    beliefs = _beliefs(state)
    living = [a for a in state["agents"] if a.alive]
    by_name = {a.name: a for a in living}
    snapshot = {a.name: frozenset(beliefs.get(a.name, set())) for a in living}
    # learner -> {belief: teacher} to adopt this turn. A conflicting belief is queued ONLY when the
    # flip is sanctioned (trusted source), and never alongside its own contradiction, so at apply time
    # any queued belief whose contradiction the learner holds is a sanctioned flip -> drop that contra.
    pending: dict[str, dict[str, str]] = {}

    for teacher in living:  # stable world_state order
        t_held = snapshot[teacher.name]
        if not t_held:
            continue
        neighbours = world.adjacent_agents(teacher, state)
        for lname in sorted(neighbours):
            learner = neighbours[lname]
            l_held = snapshot[lname]
            chosen = pending.setdefault(lname, {})
            for belief in sorted(t_held):
                if belief in l_held or belief in chosen:
                    continue
                contra = CONTRADICTS.get(belief)
                # Don't let a learner queue both halves of a contradiction in one turn.
                if contra is not None and contra in chosen:
                    continue
                conflict = contra is not None and contra in l_held
                if conflict and learner.relationships.get(teacher.name, {}).get("trust", 0) < FLIP_TRUST:
                    continue  # a worldview flip needs a trusted source, not a stranger
                if draw() < knowledge.adoption_probability(learner, teacher, state):
                    chosen[belief] = teacher.name

    events: list[str] = []
    for lname in sorted(pending):
        learner = by_name[lname]
        held = beliefs.setdefault(lname, set())
        for belief in sorted(pending[lname]):
            teacher_name = pending[lname][belief]
            held.add(belief)
            renounce = ""
            contra = CONTRADICTS.get(belief)
            if contra is not None and contra in held:  # a sanctioned flip -> renounce the old worldview
                held.discard(contra)
                renounce = f" (renouncing '{contra}')"
            world.record_memory(learner, f"Took up the belief '{belief}' from {teacher_name}{renounce}")
            events.append(f"turn {turn}: {lname} took up '{belief}' from {teacher_name}{renounce}")
    state.setdefault("events", []).extend(events)
    return events


def update(state: dict[str, Any], turn: int, rng: "random.Random | None" = None) -> list[str]:
    """Advance beliefs one turn (M4.7): tally experience -> FORM from it -> SPREAD by contact.

    ZERO LLM. Formation is deterministic; spread draws RNG like knowledge diffusion (and none when
    nobody believes anything yet). Caller gates on `beliefs_on`, so an off run never calls this — no
    "beliefs" key is written — and stays byte-identical. Returns all events logged this turn.
    """
    _update_experience(state, turn)
    events = form(state, turn)
    events += spread(state, turn, rng)
    return events


# --- Derived read-outs (pure reads, for the summary / M4.8-M4.9 / renderer) --
def belief_counts(sid: str, state: dict[str, Any]) -> dict[str, int]:
    """How many of `sid`'s LIVING members hold each belief — the settlement's belief profile."""
    rec = state.get("settlements", {}).get(sid)
    if rec is None:
        return {}
    beliefs = state.get("beliefs", {})
    living = {a.name for a in state["agents"] if a.alive}
    counts: dict[str, int] = {}
    for m in rec["members"]:
        if m in living:
            for b in beliefs.get(m, set()):
                counts[b] = counts.get(b, 0) + 1
    return counts


def dominant_beliefs(sid: str, state: dict[str, Any], top: int = 3) -> list[tuple[str, int]]:
    """The `top` most-held beliefs in settlement `sid` (count desc, then name) — its dominant culture."""
    counts = belief_counts(sid, state)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
