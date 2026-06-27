"""
population.py
=============

Day 14: DEATH as something the SOCIETY registers, and RESPAWN of a blank-slate
newcomer that keeps the population bounded.

Death already existed (starvation flips `alive` and frees the cell, see
world.mark_dead). What Day 14 adds is the survivors' REACTION to a death and the
arrival of a new agent some turns later:

  - announce_death() logs a clear DEATH event, writes a bounded memory of the
    death onto every agent that was alive at the time, and queues a respawn.
  - process_respawns() — called once per turn by the main loop — brings in a NEW
    agent RESPAWN_DELAY turns after a death, but only while the living population
    is below TARGET_POPULATION. The newcomer is a genuine social cold-start: a
    fresh Agent with blank memory, empty relationships/allies, hunger reset to 0,
    and a name no living or dead agent already uses (so it can never inherit a
    predecessor's reputation).

Like every social mechanic since Day 8, this adds ZERO new per-turn inference —
it is pure Python bookkeeping over world_state. The newcomer's very first turn
triggers an ordinary strategy refresh (it has no cached strategy yet), exactly
the same call any agent makes; respawn itself never touches the LLM.

The dead are remembered
-----------------------
Death does NOT erase the survivors' relationships toward the deceased. If the
dead agent was an ally or carried a grudge, those records stay on the living
agents (trust_summary still shows them, shared_food_sightings simply skips a dead
ally). That is deliberate — the relationship is a memory the survivor keeps.

Naming choice (documented)
--------------------------
Newcomers get a NEW name drawn from NEWCOMER_SPECS, cycled by respawn_count, with
a numeric suffix appended if that base name is somehow still in use. Names are
NOT reused from the dead, because names are the keys of every relationship/ally
map in the world — reusing "Kira" would silently graft the late Kira's reputation
onto the newcomer and break the cold-start guarantee.

Population rule (documented)
----------------------------
Each death queues exactly one respawn (a deficit of one). A respawn fires only
while living_count < TARGET_POPULATION, so the population is pulled back UP TO the
target and never past it: N agents, one dies (N-1), one enters (N). Surplus
respawns (if the target is already met when one comes due) are dropped, so the
world can never grow unbounded.
"""

from __future__ import annotations

from typing import Any

import world
from agents import Agent

# A dead agent stays gone for this many turns before a newcomer takes its place.
RESPAWN_DELAY = 10

# The population the world is kept topped up to. Respawn never pushes living
# headcount above this — it only refills the deficit a death leaves behind.
TARGET_POPULATION = 3

# The roster newcomers are drawn from, cycled by world_state["respawn_count"].
# Deliberately DISTINCT from the starting cast (Alex/Bob/Kira) so a newcomer's
# name never collides with someone still being remembered. Each spans a different
# dominant trait so the refreshed population still shows varied behaviour.
NEWCOMER_SPECS: list[tuple[str, str, dict[str, int]]] = [
    ("Mira", "curious and adventurous",     {"survive": 7, "wealth": 3, "friendship": 5}),
    ("Otto", "cautious and careful",        {"survive": 9, "wealth": 4, "friendship": 3}),
    ("Zane", "independent and competitive", {"survive": 7, "wealth": 8, "friendship": 1}),
    ("Nova", "friendly and outgoing",       {"survive": 7, "friendship": 8, "wealth": 2}),
]


def living_count(state: dict[str, Any]) -> int:
    """How many agents are currently alive."""
    return sum(1 for a in state["agents"] if a.alive)


def announce_death(agent: Any, turn: int, state: dict[str, Any],
                   cause: str = "starved", *,
                   final_memory: str = "Starved",
                   note: str = "they were starving") -> list[Any]:
    """Register `agent`'s death with the whole society and queue a respawn (Day 14).

    Strengthens the bare Day 6 death (which only flipped `alive`) into an event the
    survivors witness:
      - a clear DEATH line in events[]  ("turn 47: Kira died (starved)");
      - a bounded memory on EVERY agent that was alive at the moment of death
        ("Kira died on turn 47 — they were starving."); the dying agent itself is
        excluded (it is the one dying);
      - the dead agent keeps its own final "Starved" memory for the post-mortem;
      - a respawn is scheduled for turn + RESPAWN_DELAY.

    `cause`, `final_memory` and `note` parameterise the wording so a non-starvation
    death (e.g. M3.4 a fighter falling in battle) reads correctly while every default
    reproduces the original Day 14 strings EXACTLY — so the starvation path (every
    existing caller) is byte-for-byte unchanged.

    Survivors are captured BEFORE mark_dead so the set is exactly "who was alive
    when it happened". Relationships toward the deceased are left untouched — the
    dead are remembered (see module docstring). Returns the survivor list (for
    logging/tests).
    """
    survivors = [a for a in state["agents"] if a.alive and a is not agent]

    # The deceased's own last memory, then the world frees its cell.
    world.record_memory(agent, final_memory)
    world.mark_dead(agent)

    state["events"].append(f"turn {turn}: {agent.name} died ({cause})")

    # Gender-neutral phrasing — we cannot know an agent's gender from a name, so we
    # avoid guessing while keeping the milestone's phrasing.
    for survivor in survivors:
        world.record_memory(
            survivor, f"{agent.name} died on turn {turn} — {note}."
        )

    state.setdefault("pending_respawns", []).append(turn + RESPAWN_DELAY)
    return survivors


def _unique_name(base: str, state: dict[str, Any]) -> str:
    """A name not used by ANY agent in the world (living OR dead).

    Avoiding dead names too is essential: relationships, allies and inboxes are all
    keyed by name, so reusing a deceased agent's name would silently transfer its
    reputation onto the newcomer and break the social cold-start.
    """
    existing = {a.name for a in state["agents"]}
    if base not in existing:
        return base
    i = 2
    while f"{base}{i}" in existing:
        i += 1
    return f"{base}{i}"


def _empty_cell_near_centre(state: dict[str, Any]) -> tuple[int, int] | None:
    """A valid empty cell (no living agent, no food) closest to the grid centre.

    Biasing toward the centre drops the newcomer into the contested arena where the
    others are, so it is immediately observed and reachable rather than stranded in
    an empty corner. Deterministic (distance then coordinate order) so respawn
    placement is reproducible. Returns None only if the whole grid is full — which
    cannot happen with at most TARGET_POPULATION agents on a 10x10 board.
    """
    size = state["size"]
    occupied = {a.position for a in state["agents"] if a.alive}
    occupied |= set(state["food"])
    cx, cy = size // 2, size // 2
    candidates = [
        (x, y)
        for x in range(size)
        for y in range(size)
        if (x, y) not in occupied
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda c: (abs(c[0] - cx) + abs(c[1] - cy), c))
    return candidates[0]


def spawn_blank_agent(name: str, personality: str, turn: int, state: dict[str, Any],
                      goals: dict[str, int] | None = None,
                      pos: tuple[int, int] | None = None,
                      arrival_memory: str | None = None,
                      log_event: bool = True) -> Any | None:
    """Create + place ONE blank-slate agent and notify the survivors (Day 14).

    The single cold-start path, shared by the Day 14 timed respawn (_spawn_newcomer),
    Day 15 God mode (god_mode.spawn_agent) and Day 16 (god_mode.introduce_stranger),
    so every newcomer — respawn, summon or stranger — is built identically: dataclass
    defaults except name/personality/goals (empty memory, empty relationships/allies/
    offers/inbox, hunger 0 — a true social cold start). The name is made unique against
    every agent ever in the world (so it can never inherit a predecessor's reputation).

    `pos` overrides placement (god mode can target a cell); otherwise a valid empty
    cell near the centre is chosen.

    `arrival_memory` is the template written onto each survivor's memory (formatted
    with the newcomer's final unique name); it defaults to the Day 14 neutral arrival
    line. Day 16 passes a wariness line instead ("A stranger, X, arrived. You know
    nothing about them.") so wariness is seeded as MEMORY, never a hardcoded trust
    penalty. `log_event=False` lets a caller suppress the default population event when
    it logs its own (e.g. god_mode's tagged [GOD] line). Returns the new agent, or None
    if no cell is free.
    """
    cell = pos if pos is not None else _empty_cell_near_centre(state)
    if cell is None:
        return None

    unique = _unique_name(name, state)
    newcomer = Agent(name=unique, personality=personality, goals=dict(goals or {}))
    world.place_agent(newcomer, *cell)

    template = arrival_memory or "A new agent, {name}, appeared on turn {turn}."
    for survivor in state["agents"]:
        if survivor.alive and survivor is not newcomer:
            world.record_memory(survivor, template.format(name=unique, turn=turn))
    if log_event:
        state["events"].append(f"turn {turn}: a new agent {unique} appeared (blank slate)")
    return newcomer


def _spawn_newcomer(turn: int, state: dict[str, Any]) -> Any | None:
    """Pick the next roster entry and spawn it as a blank-slate newcomer (Day 14).

    Cycles NEWCOMER_SPECS by respawn_count and delegates the actual creation to the
    shared spawn_blank_agent. Returns the newcomer, or None if no cell was free.
    """
    count = state.get("respawn_count", 0)
    base, personality, goals = NEWCOMER_SPECS[count % len(NEWCOMER_SPECS)]
    newcomer = spawn_blank_agent(base, personality, turn, state, goals=goals)
    if newcomer is not None:
        state["respawn_count"] = count + 1
    return newcomer


def process_respawns(turn: int, state: dict[str, Any]) -> list[Any]:
    """Bring in any newcomers whose respawn has come due this turn (Day 14).

    Called once per turn by the main loop. A queued respawn fires when `turn` has
    reached its due turn AND the living population is below TARGET_POPULATION — the
    second clause is the population bound: surplus respawns are dropped so headcount
    never exceeds the target. A respawn that finds no free cell (effectively never)
    is retried next turn. Returns the list of agents that actually entered.
    """
    queue = state.get("pending_respawns", [])
    if not queue:
        return []

    due = [t for t in queue if turn >= t]
    pending = [t for t in queue if turn < t]
    spawned: list[Any] = []

    for _ in due:
        if living_count(state) >= TARGET_POPULATION:
            continue  # population bound: target already met — drop the respawn
        newcomer = _spawn_newcomer(turn, state)
        if newcomer is None:
            pending.append(turn + 1)  # no cell free this turn — try again next turn
            continue
        spawned.append(newcomer)

    state["pending_respawns"] = pending
    return spawned
