"""
god_mode.py
===========

Day 15: GOD MODE — an interactive CLI for reaching into a running simulation and
changing the WORLD, then watching the agents react on their own.

THE ARCHITECTURAL BOUNDARY (the whole point)
--------------------------------------------
Every function here reads/writes `world_state` and NOTHING else. God mode never
touches agent decision logic, trust math, or the strategy executor — it only
mutates the single source of truth. That is why a god intervention produces a
*reaction* rather than a *scripted action*: we drop food / a drought / a treasure
into the world, and the EXISTING perception -> strategy -> executor loop does the
rest. The agents head for the new food, scatter under the drought, or converge on
the treasure because their normal senses now report a changed world — not because
anything here told them to.

To keep that boundary auditable, this module imports only:
  - `world`      — which OWNS world_state and its low-level mutators, and
  - `population` — the Day 14 blank-slate spawn path, reused so a god-summoned
                   agent is a proper cold-start citizen (same as a respawn).
It imports NO strategy / trust / conversation / alliance / personality / llm.
(The Day 15 regression tests assert exactly this.)

LOGGING
-------
Every intervention appends a clearly-tagged line to world_state["events"], e.g.
  "turn 30: [GOD] drought triggered (20 turns)"
so cause -> effect is legible in the run log.

THE COMMANDS
------------
  spawn_food x y                 add a food tile at (x, y)
  spawn_agent name personality   summon a blank-slate cold-start agent (Day 14 path)
  trigger_drought [turns]        suppress food respawn for N turns (default 20)
  drop_treasure x y [value]      drop a high-value item (default value 10) — it is
                                 mirrored into the food list so the existing nav loop
                                 targets it, but claiming it pays out its full value
                                 (more than a meal) into hunger relief + inventory
  trigger_plague [name]          afflict a random (or named) living agent so it loses
                                 extra hunger per turn for N turns (Day 16) — it
                                 recovers if it survives, dies if it cannot keep fed
  introduce_stranger name pers   add a blank-slate stranger (Day 16); existing agents
                                 get a wariness MEMORY, not a hardcoded trust penalty
  status / help / resume         inspect the world, list commands, continue
"""

from __future__ import annotations

import random
from typing import Any

import population
import world

# Default span of a drought: food respawn is suppressed for this many turns.
DROUGHT_TURNS = 20

# Default worth of a dropped treasure — bigger than EAT_RELIEF (7) so it is more
# desirable than normal food: claiming it relieves more hunger and is worth wealth.
TREASURE_VALUE = 10

# Default span of a plague: the afflicted agent loses extra hunger per turn for this
# many turns (Day 16), then recovers automatically if it survived (see world.update_
# hunger). ~10 turns is long enough to genuinely threaten the agent under scarcity.
PLAGUE_TURNS = 10

# (command usage, one-line help). Single source for both the menu and `help`.
COMMANDS: list[tuple[str, str]] = [
    ("spawn_food x y", "add a food tile at (x, y)"),
    ("spawn_agent name personality...", "summon a blank-slate cold-start agent"),
    ("trigger_drought [turns]", f"stop food respawn for N turns (default {DROUGHT_TURNS})"),
    ("drop_treasure x y [value]", f"drop a high-value item (default value {TREASURE_VALUE})"),
    ("trigger_plague [name]", f"sicken a random/named agent for {PLAGUE_TURNS} turns (faster hunger)"),
    ("introduce_stranger name personality...", "add a blank-slate stranger; others grow wary via memory"),
    ("grant_knowledge name item", "teach an agent a knowledge item (M1.1); it then spreads by contact"),
    ("status", "print the current world state"),
    ("help", "show this command list"),
    ("resume / <blank line>", "leave God mode and continue the simulation"),
]


def _turn(state: dict[str, Any]) -> int:
    """The live turn from world_state, used to stamp every [GOD] log line."""
    return state.get("turn", 0)


def _log(state: dict[str, Any], msg: str) -> str:
    """Append a tagged [GOD] line to events[] and return it (for the CLI echo)."""
    line = f"turn {_turn(state)}: [GOD] {msg}"
    state["events"].append(line)
    return line


# --- Interventions (each mutates world_state ONLY) -------------------------
def spawn_food(state: dict[str, Any], x: int, y: int) -> str:
    """Add a food tile at (x, y). Hungry agents nearby will head for it next turn."""
    size = state["size"]
    if not (0 <= x < size and 0 <= y < size):
        return _log(state, f"spawn_food ({x},{y}) ignored — off the {size}x{size} map")
    if (x, y) in state["food"]:
        return _log(state, f"spawn_food ({x},{y}) ignored — food already there")
    state["food"].append((x, y))
    # Only paint the grid if no living agent stands here; if one does, the food is
    # still recorded and the cell reverts to FOOD when the agent steps off.
    if world.agent_at(x, y, state) is None:
        state["grid"][y][x] = world.FOOD
    return _log(state, f"spawned food at ({x},{y})")


def spawn_agent(state: dict[str, Any], name: str, personality: str,
                goals: dict[str, int] | None = None) -> str:
    """Summon a brand-new agent through the Day 14 blank-slate path (cold start).

    Reuses population.spawn_blank_agent so the newcomer is exactly the kind of
    citizen a respawn produces: empty memory/relationships/allies, hunger 0, a
    unique name, placed on a valid empty cell, and announced to the survivors.
    """
    newcomer = population.spawn_blank_agent(name, personality, _turn(state), state, goals=goals)
    if newcomer is None:
        return _log(state, f"spawn_agent {name} ignored — no empty cell available")
    return _log(state, f"spawned agent {newcomer.name} ({personality}) at {newcomer.position}")


def trigger_drought(state: dict[str, Any], turns: int = DROUGHT_TURNS) -> str:
    """Suppress food respawn for `turns` turns (food respawn rate -> 0).

    Stores the last suppressed turn in world_state["drought_until"]; main's
    maybe_respawn_food reads it and skips respawning while turn <= that value.
    """
    state["drought_until"] = _turn(state) + turns
    return _log(state, f"drought triggered ({turns} turns)")


def drop_treasure(state: dict[str, Any], x: int, y: int,
                  value: int = TREASURE_VALUE) -> str:
    """Drop a high-value item at (x, y) that agents compete for.

    The treasure is recorded in world_state["treasures"] AND mirrored into the food
    list, so the existing perception/navigation loop already routes hungry agents to
    it (zero executor changes). Claiming it (eat) removes it from both and pays out
    its full value as hunger relief plus an inventory entry — strictly more rewarding
    than a normal meal. Only one agent can stand on the tile, so they race for it.
    """
    size = state["size"]
    if not (0 <= x < size and 0 <= y < size):
        return _log(state, f"drop_treasure ({x},{y}) ignored — off the {size}x{size} map")
    if world.treasure_at((x, y), state) is not None:
        return _log(state, f"drop_treasure ({x},{y}) ignored — treasure already there")
    state["treasures"].append({"pos": (x, y), "value": value})
    if (x, y) not in state["food"]:
        state["food"].append((x, y))  # mirror so existing navigation targets it
    if world.agent_at(x, y, state) is None:
        state["grid"][y][x] = world.FOOD
    return _log(state, f"dropped treasure (value {value}) at ({x},{y})")


def trigger_plague(state: dict[str, Any], name: str | None = None,
                   turns: int = PLAGUE_TURNS) -> str:
    """Afflict a random (or named) living agent with a plague for `turns` turns.

    Sets ONLY a world_state marker — the victim's plague_until turn. The existing
    hunger loop (world.update_hunger) reads that marker and drains extra hunger while
    it is in force, and clears it (recovery) once the window elapses with the agent
    still alive. We do NOT touch the agent's decisions: it may starve if it cannot
    keep fed, or pull through on its own. The victim also gets a memory of falling ill
    (informational, like a death notice) so its own prompt reflects the new reality.
    """
    living = [a for a in state["agents"] if a.alive]
    if not living:
        return _log(state, "trigger_plague ignored — no living agent to afflict")
    if name is not None:
        victim = next((a for a in living if a.name == name), None)
        if victim is None:
            return _log(state, f"trigger_plague ignored — no living agent named {name!r}")
    else:
        victim = random.choice(living)
    victim.plague_until = _turn(state) + turns
    world.record_memory(victim, "A plague struck you — you weaken faster now.")
    return _log(state, f"plague struck {victim.name} ({turns} turns)")


def introduce_stranger(state: dict[str, Any], name: str,
                       personality: str = "an unknown newcomer",
                       goals: dict[str, int] | None = None) -> str:
    """Introduce a blank-slate STRANGER through the Day 14/15 cold-start path (Day 16).

    Mechanically identical to spawn_agent (empty memory/relationships/allies, hunger 0,
    unique name) — the difference is purely SOCIAL framing: every existing agent gets a
    wariness MEMORY ("A stranger, X, arrived. You know nothing about them.") instead of
    the neutral arrival line, so distrust is seeded as memory, not a hardcoded trust
    penalty. The stranger then integrates (or not) entirely through the existing talk/
    trust loop — nothing here scripts that. The default population event is suppressed
    in favour of the tagged [GOD] line below.
    """
    newcomer = population.spawn_blank_agent(
        name, personality, _turn(state), state, goals=goals,
        arrival_memory="A stranger, {name}, arrived. You know nothing about them.",
        log_event=False,
    )
    if newcomer is None:
        return _log(state, f"introduce_stranger {name} ignored — no empty cell available")
    return _log(state, f"stranger {newcomer.name} introduced")


def grant_knowledge(state: dict[str, Any], name: str, item: str) -> str:
    """Teach a living agent a knowledge ITEM (M1.1) — a write-only world mutation.

    Adds `item` to the named agent's `knowledge` set and records a memory of it. From
    here the item is just propagating STATE: the existing knowledge.diffuse pass (run
    by the engine each turn) spreads it to in-contact agents on its own — god mode does
    NOT script that, exactly as a dropped food tile lets the agents react themselves.
    Stays inside the boundary: touches only world_state (no strategy/trust/knowledge
    import — the grant is a one-line state write).
    """
    victim = next((a for a in state["agents"] if a.alive and a.name == name), None)
    if victim is None:
        return _log(state, f"grant_knowledge ignored — no living agent named {name!r}")
    victim.knowledge.add(item)
    world.record_memory(victim, f"Knows '{item}'")
    return _log(state, f"granted '{item}' to {victim.name}")


# --- Inspection + CLI ------------------------------------------------------
def status(state: dict[str, Any]) -> str:
    """A human-readable snapshot of the world (no mutation)."""
    lines = [f"--- WORLD STATUS (turn {_turn(state)}) ---"]
    living = [a for a in state["agents"] if a.alive]
    lines.append(f"living agents: {len(living)}")
    for a in living:
        sick = "  SICK(until %d)" % a.plague_until if world.is_sick(a, state) else ""
        lines.append(
            f"  {a.name:<6} pos {a.position}  hunger {a.hunger}/{world.HUNGER_MAX}"
            f"  allies {sorted(a.allies) or '-'}  inv {len(a.inventory)}{sick}"
        )
    lines.append(f"food tiles: {len(state['food'])}  {sorted(state['food'])}")
    lines.append(f"treasures:  {[(t['pos'], t['value']) for t in state['treasures']]}")
    dz = state.get("drought_until", 0)
    lines.append(f"drought:    {'active until turn ' + str(dz) if dz and _turn(state) <= dz else 'none'}")
    lines.append(f"pending respawns (turns): {state.get('pending_respawns', [])}")
    return "\n".join(lines)


def help_text() -> str:
    """The command list, used by both the menu header and the `help` command."""
    width = max(len(usage) for usage, _ in COMMANDS)
    body = "\n".join(f"  {usage:<{width}}  {desc}" for usage, desc in COMMANDS)
    return "GOD MODE commands:\n" + body


def run_command(line: str, state: dict[str, Any], out: Any = print) -> str:
    """Parse and execute ONE God command line; echo + return the result.

    Bad arguments are reported, never raised — the menu must survive typos. Returns
    the result string (so tests can assert on it without capturing stdout).
    """
    parts = line.split()
    if not parts:
        return ""
    cmd, args = parts[0], parts[1:]
    try:
        if cmd == "spawn_food":
            res = spawn_food(state, int(args[0]), int(args[1]))
        elif cmd == "drop_treasure":
            value = int(args[2]) if len(args) > 2 else TREASURE_VALUE
            res = drop_treasure(state, int(args[0]), int(args[1]), value)
        elif cmd == "trigger_drought":
            res = trigger_drought(state, int(args[0]) if args else DROUGHT_TURNS)
        elif cmd == "trigger_plague":
            res = trigger_plague(state, args[0] if args else None)
        elif cmd == "spawn_agent":
            if len(args) < 2:
                res = "usage: spawn_agent <name> <personality...>"
            else:
                res = spawn_agent(state, args[0], " ".join(args[1:]))
        elif cmd == "introduce_stranger":
            if len(args) < 1:
                res = "usage: introduce_stranger <name> [personality...]"
            else:
                pers = " ".join(args[1:]) if len(args) > 1 else "an unknown newcomer"
                res = introduce_stranger(state, args[0], pers)
        elif cmd == "grant_knowledge":
            if len(args) < 2:
                res = "usage: grant_knowledge <name> <item>"
            else:
                res = grant_knowledge(state, args[0], args[1])
        elif cmd == "status":
            res = status(state)
        elif cmd in ("help", "?"):
            res = help_text()
        else:
            res = f"unknown command: {cmd!r} (type 'help')"
    except (ValueError, IndexError):
        res = f"bad arguments for {cmd!r} (type 'help')"
    out(res)
    return res


def god_menu(state: dict[str, Any], turn: int | None = None, *,
             read_line: Any = input, out: Any = print) -> None:
    """Pause into the interactive God menu; a blank line (or 'resume') continues.

    Called between turns by the main loop, so it only ever mutates world_state at a
    clean boundary — the turn loop resumes uncorrupted. `read_line`/`out` are
    injectable so the regression tests can drive a scripted session without real IO.
    """
    if turn is not None:
        state["turn"] = turn  # keep [GOD] log lines stamped with the live turn
    out("")
    out("=" * 56)
    out(f"GOD MODE — paused at turn {_turn(state)}  (blank line resumes)")
    out("=" * 56)
    out(help_text())
    while True:
        try:
            line = read_line("god> ").strip()
        except EOFError:
            break  # piped/empty stdin — treat as resume
        if line in ("", "resume", "quit", "exit"):
            out("[GOD] resuming simulation")
            break
        run_command(line, state, out=out)
