"""
religion.py
===========

RELIGION AS INSTITUTION — shared belief becomes POWER (V2 milestone M4.8, Arc 3: Belief & Culture).
On top of M4.7 (beliefs), Arc 2 (discontent/uprising), Arc 1 (lineage/dynasties) and Phases 0-3.

The historical step M4.8 makes — a shared belief set becomes a FAITH with teeth
-----------------------------------------------------------------------------
M4.7 gave settlements distinct dominant belief sets (proto-cultures from lived experience). M4.8
turns a coherent shared set into a named FAITH, throws up a PROPHET from among the devout, and makes
faith touch LEGITIMACY: a ruler ALIGNED with the local faith is trusted and blessed, one who DEFIES
it is resented — so a conqueror who imposes on a hostile-faith town is harder to hold. The crown now
answers to the altar.

SCOPE — M4.8 is faith FORMATION + prophet EMERGENCE + the legitimacy HOOK, and ONLY that. It does NOT
build THEOCRACY (a prophet BECOMING the political ruler) or cultural friction/assimilation (that is
M4.9). Stated as boundaries below. Faith is not a new political mechanic: it does not vote, tax, or
fight. Its whole effect is to MOVE THE TRUST DIAL the existing systems already read — believers extend
extra trust to an aligned ruler and withdraw it from a defiant one — and the UNCHANGED machinery
converts that into consequences: M4.4 buffers/sharpens the discontent of the SAME extraction (its
`legitimacy_factor` reads trust in the ruler), M3.5 loyalty erodes a defied king toward BREAKAWAY, and
M4.5 rises against a ruler the flock has turned on. So religion.py touches NO other module — it only
writes trust (through the logged `trust.adjust_trust`) and records faith state.

How it works (emergent; zero LLM; deterministic trust math)
-----------------------------------------------------------
1. FAITH FORMS from a coherent shared belief set: when a settlement's LIVING members share a common
   CORE of >= FAITH_CORE_MIN beliefs (each held by at least FAITH_MAJORITY of them) and the congregation
   is at least FAITH_MIN_FOLLOWERS, that core crystallises into a named FAITH. Settlements with the SAME
   core share ONE faith; divergent cores are different faiths; a fractured settlement forms none. The
   name is a deterministic fixed mapping from the core beliefs (no LLM). Faiths are recomputed each turn
   from current beliefs, so a faith SHIFTS or DISSOLVES if its core erodes (M4.7 belief change).
2. THE PROPHET is DERIVED (the third great-figure, after the conqueror M3.4 and the revolutionary M4.6):
   the faith's most DEVOUT (holds the whole core) and most TRUSTED-by-fellow-believers follower — the
   M3.2 trust-cluster logic among co-believers, the M4.6 "derive a figure from state" pattern. Not the
   richest, not assigned. A faith with no sufficiently devout-and-trusted figure has NO prophet (honest).
3. FAITH TOUCHES LEGITIMACY by moving trust only: each turn, for a faith settlement with a force ruler,
   every believer's trust in that ruler is nudged UP if the ruler holds the faith's core (aligned) and
   DOWN if not (defiant). A PROPHET amplifies it (a moral authority for or against the crown). Bounded so
   it cannot run away. That is the entire mechanism — the existing trust/discontent/loyalty systems do
   the rest.
"""

from __future__ import annotations

import math
from typing import Any

import beliefs
import trust
import world

# --- Faith formation thresholds (tunable) ------------------------------------
FAITH_MAJORITY = 0.5      # a belief is CORE if at least this fraction of living members hold it
FAITH_CORE_MIN = 2        # a faith needs a shared core of at least this many beliefs
FAITH_MIN_FOLLOWERS = 3   # ...held by a congregation of at least this many members (a real flock)

# A follower of a faith holds at least this many of its core beliefs (the devout hold all of it).
FOLLOWER_CORE_MIN = 2

# --- Prophet thresholds ------------------------------------------------------
PROPHET_MIN_BACKERS = 2   # a prophet must be trusted (>= HIGH_THRESHOLD) by at least this many co-believers
PROPHET_ALIGN_WEIGHT = 2.0  # devotion (core beliefs held) weight in the prophet score
PROPHET_TRUST_WEIGHT = 1.0  # trust-from-fellow-believers weight in the prophet score

# --- Legitimacy nudges (the teeth — all applied through trust.adjust_trust) --
# Per-turn trust a believer extends to (aligned) or withdraws from (defiant) the ruler of its faith's
# settlement. Small and per-turn so it accrues like the M3.5 tribute backlash; bounded by FAITH_TRUST_
# BOUND so faith can move a ruler from loyal to breakaway-bound without exploding to absurd values.
FAITH_TRUST_STEP = 1
# A PROPHET amplifies the flock's stance: an extra step for/against the ruler when a prophet exists and
# is not the ruler itself (a moral authority blessing or opposing the crown).
PROPHET_TRUST_STEP = 1
# The faith contribution to a believer's trust in a ruler is held within [-BOUND, +BOUND]; beyond it the
# nudge stops (other systems may still move trust further — this only caps FAITH's own push).
FAITH_TRUST_BOUND = 6

# Fixed per-belief epithets -> a deterministic faith name from its core (no LLM).
BELIEF_EPITHET: dict[str, str] = {
    beliefs.LAND_PROVIDES: "the Bountiful Earth",
    beliefs.WORLD_IS_CRUEL: "the Long Trial",
    beliefs.STRONG_TAKE: "the Iron Truth",
    beliefs.STRONGER_TOGETHER: "the Covenant",
    beliefs.KNOWLEDGE_IS_POWER: "the Enlightened",
    beliefs.DEAD_WATCH: "the Watchful Dead",
    beliefs.WEALTH_IS_VIRTUE: "the Golden Path",
    beliefs.GREED_IS_POISON: "the Ascetic Way",
}


def _faith_id(core: "frozenset[str]") -> str:
    """A stable id for a faith with this exact core (order-independent)."""
    return "|".join(sorted(core))


def faith_name(core: "frozenset[str]") -> str:
    """A deterministic display name built from the core beliefs' fixed epithets (no LLM)."""
    return "the Faith of " + " and ".join(BELIEF_EPITHET.get(b, b) for b in sorted(core))


def _living(state: dict[str, Any]) -> dict[str, Any]:
    return {a.name: a for a in state["agents"] if a.alive}


def _realm_king(state: dict[str, Any], sid: str) -> "str | None":
    for king in sorted(state.get("kingdoms", {})):
        if sid in state["kingdoms"][king]["settlements"]:
            return king
    return None


def _sovereign(state: dict[str, Any], sid: str) -> "str | None":
    """The highest FORCE authority over settlement `sid` whose legitimacy faith judges — the realm
    KING if `sid` is in a realm (the conqueror whose loyalty M3.5 breakaway reads), else the MONARCH
    holding it (M3.4, whose levy M4.4/M4.5 read), else the trust-LEADER (M3.2). Pure read."""
    king = _realm_king(state, sid)
    if king is not None:
        return king
    mon = state.get("monarchs", {}).get(sid)
    if mon is not None:
        return mon["monarch"]
    lead = state.get("leaders", {}).get(sid)
    return lead["leader"] if lead is not None else None


# --- 1. Faith formation ------------------------------------------------------
def _settlement_core(state: dict[str, Any], sid: str) -> "frozenset[str]":
    """The shared CORE of settlement `sid`: the beliefs a FAITH_MAJORITY of its living members hold,
    if there are >= FAITH_CORE_MIN of them and a congregation of >= FAITH_MIN_FOLLOWERS; else empty."""
    rec = state.get("settlements", {}).get(sid)
    if rec is None:
        return frozenset()
    living = _living(state)
    n = sum(1 for m in rec["members"] if m in living)
    if n < FAITH_MIN_FOLLOWERS:
        return frozenset()
    need = math.ceil(FAITH_MAJORITY * n)
    counts = beliefs.belief_counts(sid, state)
    core = frozenset(b for b, c in counts.items() if c >= need)
    return core if len(core) >= FAITH_CORE_MIN else frozenset()


def _followers(state: dict[str, Any], sids: "set[str]", core: "frozenset[str]") -> set:
    """Living members of `sids` who hold at least FOLLOWER_CORE_MIN of the core — the congregation."""
    living = _living(state)
    out: set = set()
    for sid in sids:
        rec = state.get("settlements", {}).get(sid)
        if rec is None:
            continue
        for m in rec["members"]:
            if m in living and len(beliefs.agent_beliefs(m, state) & core) >= FOLLOWER_CORE_MIN:
                out.add(m)
    return out


def form_faiths(state: dict[str, Any], turn: int) -> list[str]:
    """Recompute the world's faiths from current beliefs (M4.8). Deterministic, ZERO RNG, ZERO LLM.

    Settlements sharing an identical core belong to ONE faith; a faith persists (keeping its founded
    turn and prophet) while its core still has a qualifying settlement, and DISSOLVES when it does not.
    Returns the events logged (emergence / dissolution)."""
    faiths = state.setdefault("faiths", {})
    # Group qualifying settlements by their (identical) core.
    by_core: dict[frozenset, set] = {}
    for sid in sorted(state.get("settlements", {})):
        core = _settlement_core(state, sid)
        if core:
            by_core.setdefault(core, set()).add(sid)

    events: list[str] = []
    new: dict[str, Any] = {}
    for core, sids in by_core.items():
        fid = _faith_id(core)
        prev = faiths.get(fid)
        founded = prev["founded"] if prev is not None else turn
        prophet = prev["prophet"] if prev is not None else None
        new[fid] = {"core": core, "name": faith_name(core), "settlements": sids,
                    "followers": _followers(state, sids, core), "prophet": prophet, "founded": founded}
        if prev is None:
            ev = f"turn {turn}: {faith_name(core)} took root in {', '.join(sorted(sids))}"
            events.append(ev)
    for fid, faith in faiths.items():
        if fid not in new:
            events.append(f"turn {turn}: {faith['name']} faded (its shared core dissolved)")
    state["faiths"] = new
    state.setdefault("events", []).extend(events)
    return events


# --- 2. Prophet emergence ----------------------------------------------------
def _prophet_of(state: dict[str, Any], faith: dict[str, Any]) -> "str | None":
    """Derive the faith's prophet: the most DEVOUT-and-TRUSTED follower, or None (M4.8).

    A candidate must hold the WHOLE core (fully devout) and be trusted (>= HIGH_THRESHOLD) by at least
    PROPHET_MIN_BACKERS fellow believers (the M3.2 cluster among co-believers). Among those, the highest
    score (devotion + trust-backing), name as the deterministic tiebreak. None if nobody qualifies."""
    core = faith["core"]
    living = _living(state)
    followers = [living[f] for f in sorted(faith["followers"]) if f in living]

    def backing(cand: Any) -> int:
        return sum(1 for o in followers if o.name != cand.name
                   and o.relationships.get(cand.name, {}).get("trust", 0) >= trust.HIGH_THRESHOLD)

    def trust_sum(cand: Any) -> int:
        return sum(o.relationships.get(cand.name, {}).get("trust", 0)
                   for o in followers if o.name != cand.name)

    qualified = [f for f in followers
                 if core.issubset(beliefs.agent_beliefs(f.name, state)) and backing(f) >= PROPHET_MIN_BACKERS]
    if not qualified:
        return None

    def score(cand: Any) -> float:
        align = len(beliefs.agent_beliefs(cand.name, state) & core)
        return PROPHET_ALIGN_WEIGHT * align + PROPHET_TRUST_WEIGHT * trust_sum(cand)

    return sorted(qualified, key=lambda c: (-score(c), c.name))[0].name


def choose_prophets(state: dict[str, Any], turn: int) -> list[str]:
    """Set (or clear) each faith's prophet from the current flock. Logs a new prophet's emergence."""
    events: list[str] = []
    for fid in sorted(state.get("faiths", {})):
        faith = state["faiths"][fid]
        prophet = _prophet_of(state, faith)
        if prophet is not None and prophet != faith.get("prophet"):
            ev = f"turn {turn}: {prophet} arose as prophet of {faith['name']}"
            events.append(ev)
            leader = next((a for a in state["agents"] if a.name == prophet), None)
            if leader is not None:
                world.record_memory(leader, f"Arose as prophet of {faith['name']}")
        faith["prophet"] = prophet
    state.setdefault("events", []).extend(events)
    return events


# --- 3. The legitimacy hook (moves trust only; the rest is existing machinery) -
def is_aligned(state: dict[str, Any], ruler: str, faith: dict[str, Any]) -> bool:
    """True if `ruler`'s OWN beliefs include the faith's whole core — the church blesses this crown."""
    return faith["core"].issubset(beliefs.agent_beliefs(ruler, state))


def apply_legitimacy(state: dict[str, Any], turn: int) -> list[str]:
    """Move believers' trust in their ruler by the ruler's faith-ALIGNMENT (M4.8's teeth).

    For each faith settlement with a force ruler (`_sovereign`): every believer there nudges its trust
    in the ruler UP if the ruler is aligned (holds the core) and DOWN if defiant, amplified by an
    opposing/blessing PROPHET. Bounded by FAITH_TRUST_BOUND. Writes ONLY trust (logged via
    trust.adjust_trust); the UNCHANGED M4.4/M3.5/M4.5 systems turn that trust into discontent,
    breakaway and revolt. Returns the trust-change events (from adjust_trust) — the caller already logs.
    """
    living = _living(state)
    changed = 0
    for fid in sorted(state.get("faiths", {})):
        faith = state["faiths"][fid]
        prophet = faith.get("prophet")
        for sid in sorted(faith["settlements"]):
            ruler = _sovereign(state, sid)
            if ruler is None or ruler not in living:
                continue
            aligned = is_aligned(state, ruler, faith)
            step = FAITH_TRUST_STEP if aligned else -FAITH_TRUST_STEP
            if prophet is not None and prophet != ruler:  # a prophet's moral weight for/against the crown
                step += PROPHET_TRUST_STEP if aligned else -PROPHET_TRUST_STEP
            rec = state.get("settlements", {}).get(sid)
            for name in sorted(faith["followers"]):
                if name == ruler or rec is None or name not in rec["members"] or name not in living:
                    continue
                believer = living[name]
                cur = believer.relationships.get(ruler, {}).get("trust", 0)
                # Cap FAITH's own contribution: stop pushing once past the bound in the step's direction.
                if (step > 0 and cur >= FAITH_TRUST_BOUND) or (step < 0 and cur <= -FAITH_TRUST_BOUND):
                    continue
                reason = (f"faith {'blesses' if aligned else 'condemns'} {ruler}"
                          + ("" if prophet is None or prophet == ruler
                             else f" ({'endorsed' if aligned else 'denounced'} by prophet {prophet})"))
                trust.adjust_trust(believer, ruler, step, reason, turn, state)
                changed += 1
    return [f"turn {turn}: faith moved {changed} loyalties"] if changed else []


def update(state: dict[str, Any], turn: int) -> list[str]:
    """Advance the religion institution one turn (M4.8): form faiths -> choose prophets -> apply the
    legitimacy nudge. ZERO LLM, ZERO RNG. Runs AFTER beliefs.update so it reads this turn's belief
    sets. Caller gates on `religion_on`, so an off run never calls this — no "faiths" key is written —
    and stays byte-identical. Returns events for logging."""
    events = form_faiths(state, turn)
    events += choose_prophets(state, turn)
    events += apply_legitimacy(state, turn)
    return events


# --- Derived read-outs (pure reads, for the summary / M4.9 / renderer) --------
def faith_of_settlement(state: dict[str, Any], sid: str) -> "dict[str, Any] | None":
    """The faith whose core `sid`'s populace holds, or None — a pure read."""
    for fid in sorted(state.get("faiths", {})):
        if sid in state["faiths"][fid]["settlements"]:
            return state["faiths"][fid]
    return None


def ruler_alignment(state: dict[str, Any], sid: str) -> "str | None":
    """'aligned' / 'defiant' / None — how the settlement's ruler stands with its faith (for the summary)."""
    faith = faith_of_settlement(state, sid)
    if faith is None:
        return None
    ruler = _sovereign(state, sid)
    if ruler is None:
        return None
    return "aligned" if is_aligned(state, ruler, faith) else "defiant"
