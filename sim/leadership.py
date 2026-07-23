"""
leadership.py
=============

LEGITIMATE LEADERSHIP — authority by TRUST (V2 milestone M3.2, Phase 3: Institutions).
On top of M3.1 (wage labor) and all of Phase 0 + Phase 1 + Phase 2.

The historical step M3.2 makes
------------------------------
M3.1 built the first INSTITUTION (wage labor) and the first DISEQUILIBRATING engine:
political-economic power downstream of WEALTH — the rich employ the poor and the gap
compounds. M3.2 builds the first POLITICAL institution and, crucially, the first power
that is NOT downstream of wealth: a LEADER legitimated by TRUST. The most-trusted agent
need not be the richest, so political power DECOUPLES from economic power — the tension
that later Phase 3 milestones (law, taxation, conflict) will turn on.

The leader EMERGES from the EXISTING v1 trust system (we READ it, never rebuild it)
------------------------------------------------------------------------------------
Authority is not assigned and is not a global-max lookup. A leader emerges within a
SETTLEMENT only when a real FOLLOWING coheres around a common agent: >= MIN_FOLLOWERS
co-settlers each TRUST that agent at/above the trust bar (trust.HIGH_THRESHOLD). An agent
who merely holds the single highest trust score — one ardent admirer — does NOT lead; a
fractured, low-trust settlement where no cluster coheres has NO leader at all (correct:
not every group has one). The leader is the CENTRE of the coherent cluster (the candidate
the most co-settlers trust), stored as persistent state world_state["leaders"][sid].

Legitimacy is EARNED and CONTINGENT (this is what makes it legitimacy, not a crown)
-----------------------------------------------------------------------------------
Leadership PERSISTS only while the following persists. If trust erodes — the leader turns
hostile, followers die/leave, the cluster falls below the bar — the leader LOSES the role.
HYSTERESIS keeps it from flickering on a single-turn trust wobble: a following FORMS on the
high bar (FORM_TRUST) but an incumbent is RETAINED on a lower bar (KEEP_TRUST), so a
follower drifting from trust 2 -> 1 does not unseat anyone, while real erosion (to <= 0,
e.g. the leader turning hostile) does. A NEW, strictly-more-trusted centre can DISPLACE the
incumbent if the following shifts — leadership is contingent on ONGOING legitimacy.

The minimal real effect (scoped to INFLUENCE — never tax/law)
-------------------------------------------------------------
A leader COORDINATES its following: a follower coheres MORE TIGHTLY around its leader than a
mere settler does around the settlement centre. Concretely, the M2.1 home-pull (strategy
step 3b) targets a follower at the LEADER'S position with a tighter radius (LED_HOME_RADIUS=1
vs settlement.HOME_RADIUS=2). So a LED settlement is measurably more cohesive — its members
cluster around their leader — than an identical UNLED one, with no taxation, no legislation,
no coercion: pure coordinating influence flowing through an existing system. The leader
itself keeps the ordinary settlement home-pull (it is the centre, not its own follower).

Cost & determinism
------------------
ZERO LLM calls and ZERO RNG, and — the load-bearing invariant — leadership writes NO trust
values: it is a PURE READ of agent.relationships plus its own world_state["leaders"] record.
Iteration is over sorted settlement ids and world_state["agents"] order with sorted name
tie-breaks, so the outcome never depends on Python's hash seed. A run with the system OFF
never calls `update` and never has its home-pull retargeted (leaders stays empty), so it is
byte-identical to v1. Imports only `world` (one-directional), keeping the world layer
dependency-free; the leadership record lives only in world_state (no new Agent field).
"""

from __future__ import annotations

from typing import Any

from sim import trust
from sim import world

# --- Tunable constants (documented) ----------------------------------------
# MIN_FOLLOWERS: how many co-settlers must trust a common agent above the bar for a real
# FOLLOWING to cohere around them. Two is the smallest count that reads as a "cluster
# rallying around a centre" rather than one agent simply liking another — and it makes the
# political unit (leader + >= 2 followers) at least three agents, a group, not a pair. This
# is what stops leadership being a global-max lookup: the single most-trusted agent with only
# ONE admirer does not clear it, so authority needs a CLUSTER, not a high score.
MIN_FOLLOWERS = 2

# FORM_TRUST: the trust a co-settler must hold in a candidate to COUNT toward forming a new
# following. Tied to trust.HIGH_THRESHOLD so it reads exactly as the v1 trust system's own
# "high" — a leader emerges from co-settlers who genuinely, highly trust them, reusing the
# existing semantics rather than inventing a new scale.
FORM_TRUST = trust.HIGH_THRESHOLD

# KEEP_TRUST: the LOWER bar at which an INCUMBENT leader retains a follower (hysteresis). A
# following forms on FORM_TRUST but survives on KEEP_TRUST, so a one-turn wobble (trust 2 ->
# 1) does not unseat a leader, while real erosion (trust falling to 0 or negative — e.g. the
# leader turning hostile, a -3 hit) drops the follower below KEEP_TRUST and can end the role.
# The [KEEP_TRUST, FORM_TRUST) band is the hysteresis gap that prevents flicker.
KEEP_TRUST = 1

# LED_HOME_RADIUS: how tightly a FOLLOWER coordinates around its leader — the minimal real
# leadership EFFECT. Smaller than settlement.HOME_RADIUS (2), so a led follower is drawn in
# closer (to the leader's tile) than an ordinary settler is to the settlement centre, making
# a led settlement measurably more cohesive. 1 = right around the leader. INFLUENCE only;
# never taxation or legislation.
LED_HOME_RADIUS = 1


def _living_by_settlement(state: dict[str, Any]) -> dict[str, list[Any]]:
    """Group living, settled agents by settlement id (stable world_state order)."""
    groups: dict[str, list[Any]] = {}
    for a in state["agents"]:
        if a.alive and getattr(a, "settlement", None) is not None:
            groups.setdefault(a.settlement, []).append(a)
    return groups


def _trust_in(follower: Any, name: str) -> int:
    """How much `follower` trusts the agent called `name` — a PURE read (never written)."""
    return follower.relationships.get(name, {}).get("trust", 0)


def _followers_at(members: list[Any], candidate: Any, bar: int) -> set[str]:
    """Names of OTHER members whose trust in `candidate` is >= `bar` (the candidate's following)."""
    return {m.name for m in members
            if m.name != candidate.name and _trust_in(m, candidate.name) >= bar}


def _find(state: dict[str, Any], name: str) -> Any | None:
    """Return the living agent called `name`, or None (small linear scan; no index needed)."""
    for a in state["agents"]:
        if a.alive and a.name == name:
            return a
    return None


def following_target(state: dict[str, Any], agent: Any) -> tuple[tuple[int, int], int] | None:
    """For a FOLLOWER, the (leader_position, LED_HOME_RADIUS) it coordinates around — else None.

    The single read the strategy home-pull uses to apply the leadership effect. Returns None
    (so the ordinary settlement home-pull stands) unless: leadership is active and a leader
    record exists for the agent's settlement, the agent is in that leader's CURRENT following,
    the agent is not the leader itself, and the leader is alive. Pure read — writes nothing.
    """
    sid = getattr(agent, "settlement", None)
    if sid is None:
        return None
    rec = state.get("leaders", {}).get(sid)
    if rec is None:
        return None
    if agent.name == rec["leader"] or agent.name not in rec["followers"]:
        return None
    leader = _find(state, rec["leader"])
    if leader is None:
        return None
    return (leader.position, LED_HOME_RADIUS)


def update(state: dict[str, Any], turn: int) -> list[str]:
    """Advance the leadership institution one turn (ZERO LLM, ZERO RNG, writes NO trust, M3.2).

    For each settlement (sorted ids -> deterministic), read the existing trust network and
    decide its leader:

      * FORM-following of each candidate = co-settlers trusting it >= FORM_TRUST. A candidate
        is ELIGIBLE to lead only with >= MIN_FOLLOWERS such followers (a cohered cluster, not a
        high score). The strongest eligible candidate (most followers, name tie-break) is the
        leader-elect; if none is eligible, the settlement has no elected leader.
      * An INCUMBENT is retained on the lower KEEP_TRUST bar (hysteresis): it stays leader while
        >= MIN_FOLLOWERS co-settlers still trust it >= KEEP_TRUST, UNLESS a strictly-more-trusted
        new centre displaces it (form-following bigger than the incumbent's own). If the
        incumbent's retained following falls below MIN_FOLLOWERS, it loses the role to the
        leader-elect (possibly nobody -> the leadership dissolves).

    Records persist in world_state["leaders"][sid] = {leader, followers, since}; `since` resets
    only when the leader CHANGES (an unchanged leader keeps its founding turn, its `followers`
    refreshed to the current adherents). Returns the event strings logged. Caller gates
    invocation on the `leadership_on` flag, so an off run never calls this (leaders stays empty)
    and stays byte-identical to v1.
    """
    leaders = state["leaders"]
    groups = _living_by_settlement(state)
    events: list[str] = []

    # Settlements that have lost all members (everyone died/left) drop any stale leader record.
    for sid in [s for s in leaders if s not in groups]:
        rec = leaders.pop(sid)
        events.append(f"turn {turn}: leadership of {sid} dissolved ({rec['leader']}'s following is gone)")

    for sid in sorted(groups):
        members = groups[sid]
        # Form-following (high bar) for every candidate; the eligible ones cleared MIN_FOLLOWERS.
        form = {m.name: _followers_at(members, m, FORM_TRUST) for m in members}
        eligible = [m for m in members if len(form[m.name]) >= MIN_FOLLOWERS]
        # Strongest eligible centre: most followers, then alphabetical name (deterministic).
        elect = None
        if eligible:
            elect = sorted(eligible, key=lambda m: (-len(form[m.name]), m.name))[0]

        rec = leaders.get(sid)
        incumbent = next((m for m in members if rec is not None and m.name == rec["leader"]), None)

        if incumbent is not None:
            keep = _followers_at(members, incumbent, KEEP_TRUST)
            if len(keep) >= MIN_FOLLOWERS:
                # Incumbent survives on the lower bar — UNLESS a strictly more-trusted centre exists.
                if (elect is not None and elect.name != incumbent.name
                        and len(form[elect.name]) > len(form[incumbent.name])):
                    leader = elect
                else:
                    leader = incumbent
            else:
                # The incumbent's following has collapsed below even the keep bar -> it falls.
                leader = elect
        else:
            leader = elect

        if leader is None:
            if rec is not None:
                leaders.pop(sid)
                events.append(
                    f"turn {turn}: {rec['leader']} lost legitimacy as leader of {sid} "
                    f"(following fell below {MIN_FOLLOWERS})")
            continue

        # Current adherents = co-settlers still trusting the leader at/above the retain bar.
        adherents = _followers_at(members, leader, KEEP_TRUST)
        if rec is not None and leader.name == rec["leader"]:
            rec["followers"] = adherents  # same leader -> refresh following, keep `since`
        else:
            displaced = rec["leader"] if rec is not None else None
            leaders[sid] = {"leader": leader.name, "followers": adherents, "since": turn}
            if displaced is None:
                events.append(
                    f"turn {turn}: {leader.name} emerged as leader of {sid} "
                    f"(trusted by {len(adherents)} co-settlers)")
                world.record_memory(leader, f"Became leader of {sid} ({len(adherents)} followers)")
            else:
                events.append(
                    f"turn {turn}: {leader.name} displaced {displaced} as leader of {sid} "
                    f"(more-trusted: {len(form[leader.name])} vs {len(form.get(displaced, []))})")
                world.record_memory(leader, f"Became leader of {sid}, displacing {displaced}")

    # Mirror the turn's leadership events into the chronological world log (as settlement.update
    # does for foundings) so emergence/loss/displacement is part of the durable civilizational
    # record, not just the leader's private memory. Off runs never reach here, so v1 is unchanged.
    state["events"].extend(events)
    return events
