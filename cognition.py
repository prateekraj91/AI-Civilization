"""
cognition.py
============

Tiered cognition — the centerpiece of V2 milestone M0.2.

The problem M0.1 left open
--------------------------
M0.1 gave every agent a `cognition` flag ("llm" or "heuristic") behind ONE call
site (`main.run_agent_turn`). Flipping it was a manual, whole-cast switch: either
everyone paid for inference (v1) or nobody did (the zero-LLM heuristic run). M0.2
makes that flip AUTOMATIC and BUDGETED.

The core idea
-------------
Most agents run the cheap heuristic mind. A small, FIXED budget of "focal" slots
run the expensive LLM mind. An agent is promoted to focal ONLY when it is in a
socially/strategically interesting moment, and demoted back when its life goes
routine — so LLM cost scales with DRAMA, not with population.

How it works each turn (all pure-Python, ZERO model calls to decide)
--------------------------------------------------------------------
1. `interestingness(agent, state)` scores every living agent from signals that
   ALREADY EXIST in world_state — no new events are invented:
     - involvement in a recent talk / theft / alliance / betrayal / hostility
       (scanned out of world_state["events"], which is already the social log),
     - a relationship that has moved (trust changed -> new friend or new enemy),
     - being under threat: high hunger or an active plague,
     - being a blank-slate newcomer (cold-start is interesting),
     - proximity to other agents (a lone wanderer in empty space is boring).
2. `update_tiers(...)` ranks the living by score, makes the top `budget` focal
   ("llm") and the rest "heuristic", applies HYSTERESIS so the focal set does not
   thrash every turn, and logs every promotion/demotion to world_state["events"]
   so the focal set's movement is visible.

Composition (why nothing else has to change)
---------------------------------------------
This module only ever WRITES `agent.cognition` and APPENDS event strings. The
dispatch on `agent.cognition` already lives in `main.run_agent_turn` (M0.1), so
focal agents take the v1 LLM strategy path (strategy caching included) and the
rest take the heuristic mind — talk/trust/steal/ally/death/god_mode/renderer are
all oblivious to which tier an agent is in. When `budget >= number of living
agents` every agent is focal and NO transition is logged, so a 3-agent run with a
budget >= 3 is byte-identical to v1.
"""

from __future__ import annotations

import re
from typing import Any

import world

# --- Scoring knobs ---------------------------------------------------------
# How far back (in turns) an event still counts as a "recent" interesting moment.
# Past this window the social drama has faded and stops pulling an agent toward
# focal — which is exactly what lets a quieted agent get demoted again.
RECENT_WINDOW = 5

# Per-event-type base weights, keyed by a marker substring found in the existing
# world_state["events"] strings (see conversation.py / alliance.py / trust.py for
# the exact formats). An agent's event score is the SINGLE strongest recent event
# it is named in (a max, not a sum) so a burst of small talks never out-shouts one
# genuine betrayal, and so the promotion REASON is a single honest label.
_EVENT_WEIGHTS: list[tuple[str, float, str]] = [
    ("BETRAYED", 10.0, "betrayal"),          # "*** X BETRAYED the alliance with Y ***"
    ("stole food from", 8.0, "theft"),       # "X stole food from Y"
    ("formed an ALLIANCE", 7.0, "alliance"), # "X and Y formed an ALLIANCE"
    ("flagged hostility", 6.0, "hostility"), # "X flagged hostility toward Y"
    ("-> hostile", 6.0, "hostility"),        # "X received from Y: \"...\" -> hostile"
    ("proposed an alliance", 5.0, "ally offer"),  # "X proposed an alliance to Y ..."
    ("trust in", 4.0, "relationship shift"), # "X trust in Y: a -> b (reason)"
    ("talked to", 3.0, "talk"),              # "X talked to Y: \"...\""
    ("received from", 3.0, "talk"),          # "X received from Y: \"...\" -> reply"
    ("reply", 3.0, "talk"),                  # "X heard Y reply ..." / "X replied to Y ..."
]

# Under-threat signals, read straight off the agent (no event needed).
HUNGER_THREAT = 6        # hunger at/above which survival pressure makes an agent interesting
W_HUNGER = 2.0           # score per hunger point above HUNGER_THREAT - 1 (so 6->2, 9->8)
W_PLAGUE = 6.0           # an actively sick agent is in a dramatic, decision-heavy spot

# Cold-start: a blank-slate newcomer (no memories, no relationships) deserves a
# look from the expensive mind so its first impressions aren't pure reflex.
W_NEWCOMER = 6.0

# Proximity: a lone wanderer in empty space is boring; an agent amongst others is
# where talk/theft/alliance can actually happen. Counts other living agents within
# PROX_RADIUS (Manhattan), capped so a crowd doesn't dominate genuine drama.
PROX_RADIUS = 2
W_PROX = 1.0
PROX_CAP = 3

# --- Hysteresis knobs ------------------------------------------------------
# A freshly promoted agent stays focal for at least this many turns even if the
# event that promoted it scrolls out of the window — so a one-turn blip doesn't
# cause a promote/demote flap, and the LLM mind gets a few turns to actually act.
MIN_TENURE = 3
# An incumbent focal agent carries this bonus when re-ranked, so a marginally more
# interesting challenger does NOT evict it — only a clearly-more-interesting one
# (score above incumbent + this margin) does. This is the "stays focal until its
# score drops CLEARLY below the cutoff" rule that stops turn-by-turn thrashing.
STICKY_BONUS = 3.0


def _event_turn(event: str) -> int | None:
    """Pull the turn number out of an event string ("turn 12: ..." -> 12)."""
    try:
        return int(event.split(":", 1)[0].split()[1])
    except (IndexError, ValueError):
        return None


# Word tokens in an event line (agent names are whole-word tokens like "Kira",
# "A042"). Splitting once per event and intersecting with the living-names set is
# far cheaper at scale than a per-name regex (which recompiled \bname\b for every
# (event, name) pair — a top-3 cost in the 200-agent profile).
_WORD = re.compile(r"[A-Za-z0-9_]+")


def _names_in(event: str, names: set[str]) -> set[str]:
    """Which of `names` appear (as whole-word tokens) in an event string."""
    return names & set(_WORD.findall(event))


def _recent_event_scores(state: dict[str, Any], names: set[str],
                         turn: int) -> dict[str, tuple[float, str]]:
    """For each living name, its strongest recent event (weight, label).

    Scans world_state["events"] ONCE, from the tail, attributing each recent event
    to every living agent it names — both a thief and its victim are "in" a theft,
    which is correct: a theft makes both parties worth the expensive mind. The
    weight decays linearly with age so a drama that is fading pulls less.
    """
    cutoff = turn - RECENT_WINDOW
    best: dict[str, tuple[float, str]] = {}
    for event in reversed(state.get("events", [])):
        et = _event_turn(event)
        if et is None:
            continue
        if et < cutoff:
            break  # events are chronological; nothing older can be in-window
        present = _names_in(event, names)
        if not present:
            continue
        age = turn - et
        recency = max(0.0, 1.0 - age / (RECENT_WINDOW + 1))
        for marker, weight, label in _EVENT_WEIGHTS:
            if marker in event:
                contribution = weight * recency
                for name in present:
                    if contribution > best.get(name, (0.0, ""))[0]:
                        best[name] = (contribution, label)
                break  # one (strongest-listed) label per event line
    return best


def interestingness(agent: Any, state: dict[str, Any],
                    event_scores: dict[str, tuple[float, str]] | None = None
                    ) -> tuple[float, str]:
    """Score how much `agent` deserves the expensive LLM mind right now.

    Pure-Python and cheap — NO model call decides who is interesting. Returns
    (score, reason) where `reason` names the single strongest contributing signal,
    used to annotate the promotion event so the focal set's movement reads clearly.

    `event_scores` (from `_recent_event_scores`) is passed in when scoring a whole
    cast so the events log is scanned once per turn, not once per agent.
    """
    turn = state.get("turn", 0)
    if event_scores is None:
        event_scores = _recent_event_scores(
            state, {a.name for a in state["agents"] if a.alive}, turn)

    # Each component contributes (score, label); we keep the max as the headline
    # reason but SUM the components so several mild signals can still add up.
    components: list[tuple[float, str]] = []

    components.append(event_scores.get(agent.name, (0.0, "")))

    if agent.hunger >= HUNGER_THREAT:
        components.append(((agent.hunger - HUNGER_THREAT + 1) * W_HUNGER, "high hunger"))
    if world.is_sick(agent, state):
        components.append((W_PLAGUE, "plague"))

    if not agent.memory and not agent.relationships:
        components.append((W_NEWCOMER, "newcomer"))

    nearby = _nearby_count(agent, state)
    if nearby:
        components.append((min(nearby, PROX_CAP) * W_PROX, f"near {nearby} other(s)"))

    total = sum(c[0] for c in components)
    reason = max(components, key=lambda c: c[0])[1] if components else ""
    return total, (reason or "routine")


def _nearby_count(agent: Any, state: dict[str, Any]) -> int:
    """Living OTHER agents within PROX_RADIUS (Manhattan).

    M0.3: queries the occupancy index over the small local diamond of cells around
    the agent — O(PROX_RADIUS^2) (~13 cells at r=2), independent of population —
    instead of scanning all N agents (which made this O(N^2) per turn at scale).
    """
    occ = world.living_agents_by_position(state)
    if not occ:
        return 0
    ax, ay = agent.position
    n = 0
    for dx in range(-PROX_RADIUS, PROX_RADIUS + 1):
        span = PROX_RADIUS - abs(dx)
        for dy in range(-span, span + 1):
            if dx == 0 and dy == 0:
                continue
            if (ax + dx, ay + dy) in occ:
                n += 1
    return n


def update_tiers(state: dict[str, Any], turn: int, budget: int,
                 tenure: dict[str, int]) -> dict[str, tuple[float, str]]:
    """Re-assign the focal (LLM) set for this turn under a fixed budget.

    Ranks living agents by interestingness, makes the top `budget` "llm" and the
    rest "heuristic", applies hysteresis (a minimum focal tenure + an incumbency
    bonus) so the set doesn't thrash, and logs every transition to events[].
    `tenure` is a per-run {name: consecutive focal turns} map the caller keeps
    across turns (the hysteresis memory). Returns the per-agent {name: (score,
    reason)} for inspection/logging by callers (e.g. verify_m02).

    NEVER promotes more than `budget` agents. When `budget >= len(living)` every
    agent is chosen and, since they were already focal in a v1-style run, no
    transition is logged — so the path stays byte-identical to v1.
    """
    living = [a for a in state["agents"] if a.alive]
    names = {a.name for a in living}
    event_scores = _recent_event_scores(state, names, turn)
    scored: dict[str, tuple[float, str]] = {
        a.name: interestingness(a, state, event_scores) for a in living}

    focal_now = {a.name for a in living if getattr(a, "cognition", "llm") == "llm"}

    def effective(a: Any) -> float:
        base = scored[a.name][0]
        return base + (STICKY_BONUS if a.name in focal_now else 0.0)

    # Incumbents still inside their minimum tenure are protected — they keep a slot
    # regardless of score so a just-promoted agent isn't yanked back the next turn.
    protected = {a.name for a in living
                 if 0 < tenure.get(a.name, 0) < MIN_TENURE and a.name in focal_now}
    if len(protected) > budget:  # safety: never let protection exceed the budget
        protected = set(sorted(protected, key=lambda n: (-scored[n][0], n))[:budget])

    slots = max(budget - len(protected), 0)
    challengers = sorted((a for a in living if a.name not in protected),
                         key=lambda a: (-effective(a), a.name))
    chosen = set(protected) | {a.name for a in challengers[:slots]}

    for a in living:
        was_focal = a.name in focal_now
        now_focal = a.name in chosen
        if now_focal:
            a.cognition = "llm"
            tenure[a.name] = tenure.get(a.name, 0) + 1
            if not was_focal:
                state["events"].append(
                    f"turn {turn}: {a.name} promoted to focal ({scored[a.name][1]})")
        else:
            a.cognition = "heuristic"
            if was_focal:
                state["events"].append(
                    f"turn {turn}: {a.name} demoted to heuristic (routine)")
            tenure.pop(a.name, None)

    return scored
