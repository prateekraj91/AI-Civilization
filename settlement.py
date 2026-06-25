"""
settlement.py
=============

SETTLEMENT — V2 milestone M2.1, the first DURABLE civilizational artifact. Opens
Phase 2 (Settlement & Economy) on top of Phase 1's food economy.

The idea (emergent cause, persistent artifact)
----------------------------------------------
Through Phase 1 every agent is a NOMAD: it forages, wanders, and dies or drifts on.
M1.3 changed the world's supply side — a population that knows 'farming' PRODUCES
food into world_state at the tiles its farmers work, so for the first time a place
can be reliably worth staying at. M2.1 is the historical next step: nomads become
SETTLERS around that reliable food.

A settlement is EXPLICIT, persistent state in world_state — because the era arc
(village -> town -> city) that later milestones build needs something durable to
carry forward. But the ACT of settling EMERGES from the food economy, never a
scripted "turn N a village appears": a settlement forms ONLY because farming made a
location worth staying at and enough agents stayed there long enough. No reliable
food -> no sustained cluster -> no settlement.

What "reliable food" means here (no special flag on a tile)
-----------------------------------------------------------
We do not tag a food tile as "farmed" vs "respawned". Reliability is detected
EMERGENTLY through TIME: each turn an agent that is near food has its `settle_streak`
incremented (and reset to 0 the moment it is not). Scattered respawn food is eaten
and not replaced on the same tiles, so a cluster's streak collapses; a maintained
farm plot keeps food next to the same agents turn after turn, so their streaks climb
together. A settlement forms when >= MIN_SETTLERS agents that are spatially clustered
have all SUSTAINED a streak of >= SUSTAIN_TURNS — i.e. enough agents have been fed by
the same place for long enough that staying there is plainly worthwhile.

Scope (kept strictly to M2.1)
-----------------------------
ONLY settlement formation + membership + a gentle home-pull on movement. NO defense/
territory/ownership (Phase 3), NO storage/wealth (M2.2), NO trade (M2.3), NO growth
into towns (later). The home-pull itself lives in strategy.choose_action (so both the
LLM and heuristic minds get it for free); this module owns the state: who is near
reliable food, when a settlement is founded, and who its members are.

Cost & determinism
------------------
ZERO LLM calls and ZERO RNG: formation is a deterministic THRESHOLD on sustained
presence, not a dice roll and not a timer, so it is reproducible and a run with the
system OFF is byte-identical to v1 (the loop simply never calls update). Iteration is
in world_state["agents"] order (stable) with sorted tie-breaks, so the outcome never
depends on Python's hash seed.
"""

from __future__ import annotations

from typing import Any

import world

# --- Tunable constants (documented) ----------------------------------------
# MIN_SETTLERS: how many clustered, sustained agents it takes to FOUND a settlement.
# Three is the smallest count that reads as a "group settling" rather than a pair
# happening to forage the same tile.
MIN_SETTLERS = 3

# FOOD_RADIUS: an agent counts as "near food" (advancing its streak) when a food tile
# sits within this Chebyshev distance. 1 = on or directly around the tile it eats from —
# tight enough that only an agent actually living off the local supply qualifies.
FOOD_RADIUS = 1

# CLUSTER_RADIUS: founders must lie within this Chebyshev distance of the cluster seed
# to be counted as the SAME nascent settlement (and recruits must be within it of an
# existing centre). A small village footprint on the grid.
CLUSTER_RADIUS = 2

# SUSTAIN_TURNS: consecutive turns a founder must have been near food before it can help
# found a settlement. This is the DURATION that turns "passing by food" into "this place
# reliably feeds us" — the emergent test for a reliable source, with no timer on the turn
# number itself (it depends on when agents actually sustain themselves, which varies).
#
# This is the LOAD-BEARING tuning knob for M2.1. It must be LONGER than transient food
# can keep a cluster fed: the world starts with a scattered food dump that briefly feeds
# everyone, so a short window (5-10) lets nomads settle anywhere regardless of farming —
# settlement stops being a consequence of the food economy. Calibrated at 20: only
# CONTINUOUSLY-replenished food (a farm plot grown beside its farmers each turn) keeps a
# cluster's streaks alive that long, so with farming ~13-15 villages emerge around the
# farms and WITHOUT farming ZERO form (see verify_m21 DEMO A). 20 is also the minimum
# possible incubation, so the FIRST village can't appear before turn 20 — honest, not a
# timer: which turns beyond that, where, and how many all vary by seed.
SUSTAIN_TURNS = 20

# HOME_RADIUS: a settled, fed agent drifts back toward its centre once it has wandered
# beyond this Chebyshev distance (the gentle home-pull, applied in strategy). Equal to
# CLUSTER_RADIUS so a member ranges over roughly the settlement's own footprint.
HOME_RADIUS = 2


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Chebyshev (king-move) distance — the natural radius on a square grid."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _near_food(pos: tuple[int, int], food: set[tuple[int, int]], radius: int) -> bool:
    """True if any food tile sits within Chebyshev `radius` of `pos` (pure read)."""
    x, y = pos
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if (x + dx, y + dy) in food:
                return True
    return False


def _new_settlement(state: dict[str, Any], center: tuple[int, int],
                    members: list[Any], turn: int) -> dict[str, Any]:
    """Create + register a persistent settlement record and enrol its founders.

    The single place a settlement ENTERS world_state. Stamps each founder's
    `settlement` field with the new id so the home-pull (strategy) takes effect next
    turn, logs a founding event, and writes a memory on each founder.
    """
    state["settlement_seq"] += 1
    sid = f"S{state['settlement_seq']:03d}"
    record = {
        "id": sid,
        "center": center,
        "members": {a.name for a in members},
        "founded": turn,
    }
    state["settlements"][sid] = record
    for a in members:
        a.settlement = sid
        world.record_memory(a, f"Settled at {sid} {center}")
    names = ", ".join(sorted(record["members"]))
    state["events"].append(
        f"turn {turn}: settlement {sid} founded at {center} by {len(members)} settlers ({names})"
    )
    return record


def _join_settlement(state: dict[str, Any], record: dict[str, Any],
                     agent: Any, turn: int) -> None:
    """Add an existing-nomad `agent` to an already-founded settlement (membership)."""
    record["members"].add(agent.name)
    agent.settlement = record["id"]
    world.record_memory(agent, f"Joined settlement {record['id']}")
    state["events"].append(
        f"turn {turn}: {agent.name} joined settlement {record['id']} at {record['center']}"
    )


def update(state: dict[str, Any], turn: int) -> list[str]:
    """Advance the settlement system one turn (ZERO LLM, ZERO RNG, M2.1).

    Three deterministic passes over the living agents, in three stages:

      1. STREAKS — every agent near food (within FOOD_RADIUS) advances its
         `settle_streak`; everyone else resets to 0. This is the running record of
         who is being reliably fed by a place.
      2. RECRUIT — any nomad standing near an EXISTING settlement (within
         CLUSTER_RADIUS of its centre) AND near food this turn joins it. So a
         settlement gains members as agents gather around its reliable food; an
         isolated nomad far from any settlement never joins.
      3. FOUND — among nomads whose streak has reached SUSTAIN_TURNS, any spatial
         cluster of >= MIN_SETTLERS founds a NEW settlement at the cluster's
         (rounded) centroid. Founders are removed from the candidate pool as each
         settlement forms, so several can form in one turn at different food sites.

    Returns the ids of settlements founded/joined this turn (for callers/logging).
    Caller gates invocation (run_simulation's `settlements` flag), so a run with the
    system off never calls this and stays byte-identical to v1.
    """
    living = [a for a in state["agents"] if a.alive]
    food = set(state["food"])
    touched: list[str] = []

    # 1. Streaks: near reliable food -> climb; otherwise the streak collapses.
    for a in living:
        if _near_food(a.position, food, FOOD_RADIUS):
            a.settle_streak += 1
        else:
            a.settle_streak = 0

    settlements = state["settlements"]

    # 2. Recruit nomads who have gathered at an existing settlement's reliable food.
    for a in living:
        if a.settlement is not None:
            continue
        if not _near_food(a.position, food, FOOD_RADIUS):
            continue  # must be drawn by the reliable food, not merely passing the centre
        for sid in sorted(settlements):  # deterministic: smallest id wins a tie
            if _chebyshev(a.position, settlements[sid]["center"]) <= CLUSTER_RADIUS:
                _join_settlement(state, settlements[sid], a, turn)
                touched.append(sid)
                break

    # 3. Found new settlements from sustained, clustered nomads.
    candidates = [a for a in living
                  if a.settlement is None and a.settle_streak >= SUSTAIN_TURNS]
    used: set[str] = set()
    for seed in candidates:  # world_state order -> stable seed choice
        if seed.name in used:
            continue
        group = [a for a in candidates
                 if a.name not in used
                 and _chebyshev(a.position, seed.position) <= CLUSTER_RADIUS]
        if len(group) < MIN_SETTLERS:
            continue
        cx = round(sum(a.position[0] for a in group) / len(group))
        cy = round(sum(a.position[1] for a in group) / len(group))
        record = _new_settlement(state, (cx, cy), group, turn)
        touched.append(record["id"])
        used.update(a.name for a in group)

    return touched
