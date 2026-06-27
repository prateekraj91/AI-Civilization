"""
world.py
========

The SINGLE SOURCE OF TRUTH for AI Civilization.

ARCHITECTURE RULE (do not violate)
----------------------------------
`world_state` is the one authoritative description of the entire simulation.

  - Agents READ from world_state to make decisions.
  - Future God Mode will MUTATE world_state.
  - Future renderers will ONLY READ world_state.
  - No component may bypass this: there is no other place the "truth" lives.

Because everything funnels through this dict, the simulation stays
deterministic and inspectable — you can dump `world_state` at any moment and
know the complete state of the world.

The structure is deliberately a plain dict (not a class) for V1 so it is easy
to serialize (JSON), diff, and reason about. It is designed for expansion:
new top-level keys (e.g. "grid", "weather", "economy") can be added later
without breaking existing readers.

DAY 2 ADDITION
--------------
The world now has a 2D grid plus food. The grid is the spatial view; the agent
position and food coordinates are stored in world_state so they remain the
single source of truth. No movement, no decisions — agents can only *observe*.

DAY 6-8 ADDITIONS
-----------------
- Day 6: the world hosts MULTIPLE agents that share one grid and compete for the
  same food. Because food lives in the single `world_state["food"]` list, food
  eaten by one agent vanishes for everyone — competition is automatic.
- Day 7: observe() now reports adjacent agents BY NAME (e.g. "North: Bob"), so
  agents are aware of their neighbours.
- Day 8: record_social_memories() turns that awareness into bounded memory
  entries ("Observed Bob north of me", "Observed Kira near food").

Coordinate convention
---------------------
Positions are (x, y) where x is the column and y is the row, both 0-indexed.
The grid is indexed grid[y][x] (row-major). Compass directions follow screen
coordinates (y increases downward):

    North = y - 1    South = y + 1    East = x + 1    West = x - 1
"""

import random
from typing import Any

# --- Grid constants -------------------------------------------------------
GRID_SIZE = 10  # the world is GRID_SIZE x GRID_SIZE

# Cell kinds. Kept as plain strings so the grid serializes trivially and is
# easy to extend later (just add a new constant).
EMPTY = "empty"
FOOD = "food"
AGENT = "agent"

# The complete, closed set of actions an agent may take (Day 3). Gemini is only
# ever allowed to choose one of these; anything else is rejected by decide().
VALID_ACTIONS = (
    "move_north",
    "move_south",
    "move_east",
    "move_west",
    "eat",
    "rest",
)

# Movement deltas in (dx, dy). Screen coordinates: y increases downward, so
# North decreases y. Matches the compass convention documented above.
_MOVES = {
    "move_north": (0, -1),
    "move_south": (0, 1),
    "move_east": (1, 0),
    "move_west": (-1, 0),
}

# --- Hunger constants (Day 4) --------------------------------------------
HUNGER_MAX = 10        # at this level the agent starves and dies
HUNGER_PER_TURN = 1    # hunger gained each turn
PLAGUE_HUNGER_PER_TURN = 3  # hunger gained each turn while sick (Day 16 plague). A
                            # plagued agent burns through reserves 3x as fast, so it
                            # must eat far more often or starve before it recovers.
EAT_RELIEF = 7         # hunger removed by eating one food (clamped at 0)
                       # Day 9 rebalance: a meal lasts longer so agents spend
                       # fewer turns scrambling and more turns interacting.
                       # (Real scarcity is deferred to Day 11.)

# --- V2 M1.3: technology effects on the hunger/food step ------------------
# A KNOWN tech changes survival outcomes through the existing eat path — gated on
# the item being in agent.knowledge (a plain state read; no import, so world stays
# the lowest layer). An agent that does not know the tech gets no benefit, so a v1
# agent (empty knowledge) eats exactly as before — byte-identical.
#   fire  -> cooking: a cooked meal is worth more, so eating relieves FIRE_EAT_BONUS
#            extra hunger (a fire-knower stretches the same scarce food further).
#   tools -> reach: a tool-user can FORAGE food from an adjacent tile, not only the
#            tile underfoot (see execute_action + strategy's survival override).
# (farming — producing food into the world — lives in knowledge.farm, the food-supply
#  side, since it adds to world_state rather than changing how an agent eats.)
FIRE_EAT_BONUS = 4     # extra hunger relief per meal when the eater knows 'fire'

# --- Memory constants (Day 5, raised Day 8) ------------------------------
MEMORY_LIMIT = 20      # an agent retains only its most recent N memories;
                       # older memories are discarded oldest-first.

# --- Food clustering (Day 11) --------------------------------------------
# When spawn_food(cluster=True), food is confined to a square window of this
# Chebyshev radius around the grid centre instead of scattered across the whole
# map. Day 11 turns this on so scarce food appears on the SAME central tiles the
# agents start near — they converge and contend for it, rather than each starving
# alone in a separate corner. Radius 2 on a 10x10 grid = a 5x5 central arena.
FOOD_CLUSTER_RADIUS = 2

# The authoritative world state. Keep this as the ONLY mutable global.
world_state: dict[str, Any] = {
    "turn": 0,           # current simulation tick
    "agents": [],        # list[Agent] living in the world
    "events": [],        # chronological log of things that have happened
    "size": GRID_SIZE,   # grid is size x size
    "grid": [],          # 2D list[list[str]], grid[y][x] -> cell kind
    "food": [],          # list[tuple[int, int]] of (x, y) food coordinates
    # Day 14 lifecycle. Both live in world_state so the death/respawn machinery
    # stays inspectable and serializable like everything else (single source of
    # truth). `pending_respawns` holds the turns at which a queued newcomer is due
    # (one entry per death, see population.announce_death); `respawn_count` is the
    # running number of newcomers spawned, used to cycle the newcomer roster.
    "pending_respawns": [],  # list[int]: turns at which a respawn becomes due
    "respawn_count": 0,      # how many newcomers have entered so far
    # Day 15 God mode. Both are PLAIN world_state so god_mode.py only ever
    # reads/writes the single source of truth. `treasures` are high-value items the
    # god can drop (each {"pos": (x, y), "value": int}); a treasure's position is
    # ALSO mirrored into `food` so the existing perception/navigation loop targets
    # it with zero executor changes — claiming it (eat) just pays out more.
    # `drought_until` is the last turn through which food respawn is suppressed.
    "treasures": [],         # list[dict]: high-value items dropped by God mode
    "drought_until": 0,      # int: respawn suppressed while turn <= this value
    # M0.3 scaling: an incremental {(x, y): agent} index of where each LIVING agent
    # stands, maintained at the only three places a position/liveness changes
    # (place_agent, move_agent, mark_dead). It makes "who is on this cell?" an O(1)
    # lookup instead of an O(N) rebuild of the whole agent list — the single biggest
    # per-turn cost at 200+ agents, where scan()/collision checks ask it constantly.
    # It is a derived view of agents[]+positions, never an independent truth.
    "occupancy": {},         # dict[tuple[int, int], Agent]: living agent by cell
    # V2 M2.1 settlement: the first DURABLE civilizational artifact. Each settlement is
    # a persistent record {"id", "center": (x, y), "members": set[name], "founded": turn}
    # that EMERGES (settlement.update) when enough agents sustain themselves near reliable
    # (farmed) food, and then carries forward across turns for later eras to build on
    # (storage, trade, towns, governance). Empty for any run with the system off / no
    # reliable food, which keeps the default run byte-identical to v1. Keyed by id for
    # O(1) membership/center lookup (the home-pull and renderer both read it).
    "settlements": {},       # dict[str, dict]: id -> settlement record (persistent)
    "settlement_seq": 0,     # int: monotonic counter so each settlement id is unique
    # V2 M2.2 storage: a single flag mirroring the run's opt-in `storage` setting. The
    # survival-buffer draw-down (storage.draw_down, read in the starvation step) and the
    # read-only wealth overlay in the end-of-run summary both gate on it, so with storage
    # OFF (the default) neither fires and the run stays byte-identical to v1. Surplus
    # ACCUMULATION is gated separately at the call site (run_simulation only calls
    # storage.accumulate when storage is on), so this is purely the read-side switch.
    "storage_on": False,     # bool: M2.2 storage/surplus system enabled for this run
    # V2 M2.3 economy: a single flag mirroring the run's opt-in `economy` setting. Money
    # minting (economy.mint), the trade pass (economy.trade), proprietary-knowledge guarding
    # (knowledge.diffuse skips guarded items), and the money side of the survival buffer
    # (storage.draw_down redeeming money as food) all gate on it. With economy OFF (default)
    # none fire and the run stays byte-identical to v1. Hunting production is NOT gated here —
    # like farming it is gated purely on KNOWING the skill, so it composes the same way.
    "economy_on": False,     # bool: M2.3 trade/money economy enabled for this run
    # V2 M3.1 wage labor — the first INSTITUTION (opens Phase 3). Employment links are
    # PERSISTENT structures (not one-shot trades): each is {employer, worker, wage, since}
    # and survives across turns until the relationship ends (worker quits / employer fires).
    # This persistence is what makes wage labor an institution rather than a market trade.
    # The roles EMERGE from M2.2 wealth + M2.3 skill state (see labor.py) — never assigned.
    # Empty + inert unless the run opts in (labor_on); kept here so the institution rides the
    # single source of truth like every other system and serialises/inspects the same way.
    "employments": [],       # list[dict]: persistent {employer, worker, wage, since} links
    "labor_on": False,       # bool: M3.1 wage-labor institution enabled for this run
    # V2 M3.2 legitimate leadership — the first POLITICAL institution and the first power NOT
    # downstream of wealth. Each record {"leader": name, "followers": set[name], "since": turn}
    # is the persistent CENTRE of a coherent trust cluster within a settlement; it EMERGES
    # (leadership.update) only when >= MIN_FOLLOWERS co-settlers trust a common agent above the
    # trust bar, and is CONTINGENT (lost when that following erodes). A PURE read of the v1 trust
    # system — leadership writes no trust values. Empty + inert unless the run opts in
    # (leadership_on), so a default run never retargets the home-pull and stays byte-identical
    # to v1. Keyed by settlement id (the home-pull and renderer both read it).
    "leaders": {},           # dict[str, dict]: settlement id -> leadership record (persistent)
    "leadership_on": False,  # bool: M3.2 legitimate-leadership institution enabled for this run
}


def add_agent(agent: Any) -> None:
    """Register an agent in the world.

    This helper exists so callers go *through* the world layer instead of
    poking `world_state["agents"]` directly. It gives us one place to later add
    validation, indexing, or God Mode hooks — without changing call sites.
    """
    world_state["agents"].append(agent)


def living_agents_by_position(
    state: dict[str, Any] | None = None,
) -> dict[tuple[int, int], Any]:
    """Map each LIVING agent's position to the agent itself.

    The single helper behind both agent detection (Day 7) and movement collision
    (Day 6): callers ask "who, if anyone, is standing on this cell?" without
    re-scanning the agent list themselves. Dead agents are excluded — they no
    longer occupy space. If two living agents ever shared a cell (they should not,
    because movement forbids it) the later one in the list wins; this is only a
    lookup convenience, not the source of truth for collisions.
    """
    state = world_state if state is None else state
    occ = state.get("occupancy")
    if occ is not None:
        # The maintained index IS the answer at O(1) — see world_state["occupancy"].
        # Returned live (callers only ever .get() from it, never mutate it).
        return occ
    # Fallback for a bare/custom state dict with no maintained index: rebuild from
    # truth, preserving the original O(N) behaviour exactly.
    return {
        agent.position: agent
        for agent in state["agents"]
        if getattr(agent, "alive", True)
    }


def agent_at(x: int, y: int, state: dict[str, Any] | None = None) -> Any | None:
    """Return the living agent standing at (x, y), or None if the cell is free."""
    return living_agents_by_position(state).get((x, y))


# N/S/E/W offsets in a FIXED order, so adjacency walks are deterministic (M1.3).
_ADJ_OFFSETS = ((0, -1), (0, 1), (1, 0), (-1, 0))


def _adjacent_free_food(agent: Any, state: dict[str, Any]) -> tuple[int, int] | None:
    """First adjacent N/S/E/W tile holding FREE food (no agent on it), or None (M1.3).

    The reach a 'tools'-knower forages with: it eats from a neighbouring tile without
    stepping onto it. Free-only (skips a tile another agent stands on) so foraging
    never overlaps the steal mechanic. Deterministic order -> reproducible.
    """
    x, y = agent.position
    food = state["food"]
    occ = state.get("occupancy", {})
    for dx, dy in _ADJ_OFFSETS:
        cell = (x + dx, y + dy)
        if cell in food and cell not in occ:
            return cell
    return None


def place_food(x: int, y: int, state: dict[str, Any] | None = None) -> bool:
    """Add one food tile at (x, y) (M1.3 farming production). Returns True if added.

    The single-cell food adder behind knowledge.farm: keeps the world mutation in the
    world layer (food list + grid stay in sync, source of truth) so callers never poke
    the grid directly. A no-op if food is already there; paints the grid FOOD unless a
    living agent stands on the cell (then the cell reverts to FOOD when it steps off).
    """
    state = world_state if state is None else state
    if (x, y) in state["food"]:
        return False
    state["food"].append((x, y))
    if state["grid"][y][x] != AGENT:
        state["grid"][y][x] = FOOD
    return True


def treasure_at(pos: tuple[int, int],
                state: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return the treasure record sitting on `pos` (Day 15), or None.

    Treasures live in world_state["treasures"] as {"pos", "value"} dicts. This is
    the single lookup behind the eat handler's treasure branch and the renderer.
    """
    state = world_state if state is None else state
    for t in state["treasures"]:
        if t["pos"] == pos:
            return t
    return None


def create_world(size: int = GRID_SIZE) -> list[list[str]]:
    """Build a fresh, empty `size` x `size` world and store it in world_state.

    This is the simulation's reset point and MUST be fully idempotent: calling it
    again has to leave world_state exactly as if the process had just started.
    `world_state` is a module-level singleton, so anything that survives a reset
    leaks into the next run — most insidiously the agent list, which would
    otherwise accumulate dead agents from previous simulations and silently
    inflate counts (turn iteration, food-collision checks, benchmark harnesses).

    Because callers run multiple simulations in one process (test suites, the
    benchmarking harness, future tournaments), every piece of per-simulation
    state is cleared here:
      - agents:  emptied (no carry-over between runs)
      - grid:    regenerated as an all-EMPTY size x size grid
      - food:    emptied (re-seeded afterwards via spawn_food)
      - events:  emptied
      - turn:    reset to 0
      - pending_respawns / respawn_count: cleared (Day 14) — queued newcomers and
        the respawn counter must not leak across simulations, or a fresh run would
        spawn ghosts from a previous one's deaths.

    Lists are cleared in place rather than rebound so any reference already held
    elsewhere keeps pointing at the live (now-empty) collection.
    """
    world_state["size"] = size
    world_state["turn"] = 0
    world_state["grid"] = [[EMPTY for _ in range(size)] for _ in range(size)]
    world_state["agents"].clear()
    world_state["food"].clear()
    world_state["events"].clear()
    world_state["pending_respawns"].clear()
    world_state["respawn_count"] = 0
    world_state["treasures"].clear()
    world_state["drought_until"] = 0
    world_state.setdefault("occupancy", {}).clear()  # M0.3: reset the position index
    # M2.1: settlements are per-simulation, so they must not leak across runs (a stale
    # settlement would falsely home-pull a fresh cast). Cleared like every other state.
    world_state.setdefault("settlements", {}).clear()
    world_state["settlement_seq"] = 0
    # M2.2: the storage flag is per-simulation too — reset OFF so a fresh run is v1 unless
    # the caller (run_simulation) explicitly turns storage on for that run.
    world_state["storage_on"] = False
    # M2.3: same for the economy (trade/money) flag.
    world_state["economy_on"] = False
    # M3.1: employment links are per-simulation persistent state — clear them so a stale
    # boss/worker pairing never leaks across runs — and reset the wage-labor flag OFF.
    world_state.setdefault("employments", []).clear()
    world_state["labor_on"] = False
    # M3.2: leadership records are per-simulation (a stale leader must not leak across runs).
    world_state.setdefault("leaders", {}).clear()
    world_state["leadership_on"] = False
    return world_state["grid"]


def place_agent(agent: Any, x: int, y: int) -> tuple[int, int]:
    """Place `agent` at (x, y): record it on the agent, register it, mark the grid.

    Routes the placement through the world layer so world_state stays the single
    source of truth: the agent's position is mirrored onto the grid as an AGENT
    cell. Registration is idempotent — placing an already-registered agent will
    not add it twice.
    """
    agent.position = (x, y)
    if agent not in world_state["agents"]:
        add_agent(agent)
    world_state["grid"][y][x] = AGENT
    world_state["occupancy"][(x, y)] = agent  # M0.3: keep the position index in sync
    return agent.position


def spawn_food(count: int = 5, *, cluster: bool = False) -> list[tuple[int, int]]:
    """Place up to `count` food cells onto free grid cells.

    Rules enforced:
      - food never overlaps existing food
      - food never overlaps an agent's position

    `cluster` (Day 11): when True, food is restricted to a FOOD_CLUSTER_RADIUS
    window around the grid centre so scarce food lands on contested central tiles
    instead of scattering to the corners. When False (legacy), the whole grid is
    eligible.

    Cells are chosen by shuffling the free candidate pool and taking the first
    `count` — so a full window simply yields fewer placements rather than looping
    forever. Food coordinates are stored in world_state["food"] AND written onto
    the grid so the list (for debugging) and the spatial view agree.
    """
    size = world_state["size"]

    # Cells we must avoid: every agent's position and any food already placed.
    occupied: set[tuple[int, int]] = {a.position for a in world_state["agents"]}
    occupied |= set(world_state["food"])

    if cluster:
        cx, cy = size // 2, size // 2
        r = FOOD_CLUSTER_RADIUS
        candidates = [
            (x, y)
            for x in range(max(0, cx - r), min(size, cx + r + 1))
            for y in range(max(0, cy - r), min(size, cy + r + 1))
        ]
    else:
        candidates = [(x, y) for x in range(size) for y in range(size)]

    free = [c for c in candidates if c not in occupied]
    random.shuffle(free)
    for x, y in free[:count]:
        world_state["food"].append((x, y))
        world_state["grid"][y][x] = FOOD

    return world_state["food"]


# Adjacent-cell offsets shared by observation and social memory, kept in one
# place so the compass order (N, S, E, W) is defined exactly once.
_NEIGHBOURS: dict[str, tuple[int, int]] = {
    "North": (0, -1),
    "South": (0, 1),
    "East": (1, 0),
    "West": (-1, 0),
}


def scan(agent: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Structured perception: what `agent` can sense right now (pure READ).

    Returns a machine-friendly view the Python strategy executor can act on
    without parsing strings::

        {
          "pos": (x, y),
          "on_food": bool,                  # food underfoot
          "cells": {                        # keys: north/south/east/west
            "north": {
              "pos": (x, y) | None,         # None when off-map
              "wall": bool,                 # off the edge of the map
              "food": bool,                 # uneaten food on this cell
              "agent": Agent | None,        # living agent standing here
              "blocked": bool,              # wall OR occupied -> cannot move in
            }, ...
          }
        }

    Cell contents come from the food list and live agent positions, NOT the grid
    array, so a stale AGENT mark can never leak into perception. observe() is a
    thin human-readable wrapper over this.
    """
    x, y = agent.position
    size = state["size"]
    food = set(state["food"])
    occupants = living_agents_by_position(state)

    cells: dict[str, dict[str, Any]] = {}
    for name, (dx, dy) in _NEIGHBOURS.items():
        nx, ny = x + dx, y + dy
        key = name.lower()
        if not (0 <= nx < size and 0 <= ny < size):
            cells[key] = {"pos": None, "wall": True, "food": False,
                          "agent": None, "blocked": True}
            continue
        other = occupants.get((nx, ny))
        if other is agent:
            other = None
        cells[key] = {
            "pos": (nx, ny),
            "wall": False,
            "food": (nx, ny) in food,
            "agent": other,
            "blocked": other is not None,
        }

    return {"pos": (x, y), "on_food": (x, y) in food, "cells": cells}


def adjacent_agents(agent: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Map name -> Agent for every LIVING agent in an adjacent N/S/E/W cell.

    The shared "who can I reach right now?" helper behind both detection and the
    talk action's perception-range rule. Built from scan() so there is no string
    parsing and no second notion of adjacency.
    """
    s = scan(agent, state)
    return {
        cell["agent"].name: cell["agent"]
        for cell in s["cells"].values()
        if cell["agent"] is not None
    }


def visible_food(agent: Any, state: dict[str, Any]) -> set[tuple[int, int]]:
    """The food coordinates `agent` can directly perceive right now (pure READ).

    Built from scan(): the agent's own tile plus its four N/S/E/W neighbours —
    the same perception range it reports in observe(). This is the unit allies
    SHARE under Day 13: each ally contributes the food in ITS window, so a pair
    sees more of a scarce map than either does alone (see alliance.shared_food_
    sightings, which folds the union into the strategy prompt).
    """
    s = scan(agent, state)
    coords: set[tuple[int, int]] = set()
    if s["on_food"]:
        coords.add(s["pos"])
    for cell in s["cells"].values():
        if cell["food"] and cell["pos"] is not None:
            coords.add(cell["pos"])
    return coords


def observe(agent: Any, state: dict[str, Any]) -> str:
    """Human-readable perception string, built on scan() (Day 7 detection).

    Reports the agent's CURRENT tile first, then North/South/East/West. A
    neighbour holding a living agent reports that agent's NAME (e.g. "North:
    Bob"); off-map cells report "wall"; otherwise food/empty.
    """
    s = scan(agent, state)
    lines: list[str] = [
        f"Current Tile: {FOOD if s['on_food'] else EMPTY}",
        "",
    ]
    for name in ("North", "South", "East", "West"):
        cell = s["cells"][name.lower()]
        if cell["wall"]:
            label = "wall"
        elif cell["agent"] is not None:
            # Day 16: a plagued neighbour visibly looks sick, surfaced through the
            # ordinary perception string (no new sense, no extra cost).
            label = cell["agent"].name
            if is_sick(cell["agent"], state):
                label += " (looks sick)"
        elif cell["food"]:
            label = FOOD
        else:
            label = EMPTY
        lines.append(f"{name}: {label}")
    return "\n".join(lines)


def _is_near_food(pos: tuple[int, int], state: dict[str, Any]) -> bool:
    """True if `pos` is on a food cell or directly adjacent (N/S/E/W) to one."""
    x, y = pos
    food = set(state["food"])
    if (x, y) in food:
        return True
    return any((x + dx, y + dy) in food for dx, dy in _NEIGHBOURS.values())


def record_social_memories(agent: Any, state: dict[str, Any]) -> list[str]:
    """Record bounded memories about agents adjacent to `agent` (Day 8).

    For every living agent in an adjacent N/S/E/W cell, store "Observed <name>
    <direction> of me", and additionally "Observed <name> near food" when that
    neighbour is on or beside food. Each entry goes through record_memory, so the
    per-agent cap (MEMORY_LIMIT) and oldest-first eviction apply automatically.

    Returns the names observed this turn (for logging).
    """
    x, y = agent.position
    occupants = living_agents_by_position(state)
    observed: list[str] = []

    for name, (dx, dy) in _NEIGHBOURS.items():
        other = occupants.get((x + dx, y + dy))
        if other is None or other is agent:
            continue
        record_memory(agent, f"Observed {other.name} {name.lower()} of me")
        if _is_near_food(other.position, state):
            record_memory(agent, f"Observed {other.name} near food")
        # Day 16: noticing a sick neighbour is recorded like any other sighting, so
        # the plague becomes social knowledge through the existing memory path.
        if is_sick(other, state):
            record_memory(agent, f"Observed {other.name} looking sick")
        observed.append(other.name)

    return observed


def move_agent(agent: Any, dx: int, dy: int) -> bool:
    """Move `agent` by (dx, dy), keeping the grid in sync. Returns True if moved.

    Two refusal rules (agent stays put, returns False):
      - Boundary: a move that would leave the map.
      - Occupancy (Day 6): another LIVING agent already stands on the target
        cell. Agents cannot stack, which is what forces them to RACE for a
        contested food tile instead of piling onto it.

    When the agent vacates a cell we restore it to FOOD if uneaten food still
    sits there, otherwise EMPTY — so the grid always reflects world_state. The
    destination cell becomes AGENT.
    """
    x, y = agent.position
    nx, ny = x + dx, y + dy
    size = world_state["size"]

    if not (0 <= nx < size and 0 <= ny < size):
        return False  # would leave the map — stay in place

    blocker = agent_at(nx, ny)
    if blocker is not None and blocker is not agent:
        return False  # cell taken by another living agent — stay in place

    # Vacate the old cell. Food is tracked in world_state["food"]; if uneaten
    # food remains here, the cell reverts to FOOD, else EMPTY.
    world_state["grid"][y][x] = FOOD if (x, y) in world_state["food"] else EMPTY
    world_state["occupancy"].pop((x, y), None)  # M0.3: index follows the move

    # Occupy the new cell.
    agent.position = (nx, ny)
    world_state["grid"][ny][nx] = AGENT
    world_state["occupancy"][(nx, ny)] = agent
    return True


def execute_action(agent: Any, action: str) -> str:
    """Apply a validated `action` to the world and return a human-readable result.

    Movement is delegated to move_agent (boundary-safe). `eat` consumes food on
    the agent's current tile if present. `rest` and unknown actions are no-ops.
    All world mutation flows through this layer so world_state stays the single
    source of truth.
    """
    if action in _MOVES:
        dx, dy = _MOVES[action]
        direction = action.split("_")[1]  # "north" / "south" / ...
        if move_agent(agent, dx, dy):
            record_memory(agent, f"Moved {direction}")
            return f"{agent.name} moved {direction}."
        record_memory(agent, f"Blocked moving {direction}")
        return (
            f"{agent.name} tried to move {direction} but was blocked "
            f"(map edge or another agent) and stayed put."
        )

    if action == "eat":
        # The agent may only eat what is on its CURRENT tile. The grid cell
        # stays AGENT (agent on top); only the food record + hunger change.
        #
        # Day 15: a treasure dropped by God mode is checked FIRST. A treasure is
        # mirrored into the food list too (so the existing navigation loop heads for
        # it), but claiming it pays out its full value as hunger relief — more than a
        # normal meal — and goes into the agent's inventory as wealth.
        treasure = treasure_at(agent.position)
        if treasure is not None:
            world_state["treasures"].remove(treasure)
            if agent.position in world_state["food"]:
                world_state["food"].remove(agent.position)
            agent.hunger = max(0, agent.hunger - treasure["value"])
            agent.inventory.append({"treasure": treasure["value"]})
            record_memory(agent, f"Claimed treasure (value {treasure['value']})")
            return f"{agent.name} claimed a treasure (value {treasure['value']})!"
        # M1.3 fire (cooking): a fire-knower extracts more from the same meal.
        know = getattr(agent, "knowledge", ())
        relief = EAT_RELIEF + (FIRE_EAT_BONUS if "fire" in know else 0)
        if agent.position in world_state["food"]:
            world_state["food"].remove(agent.position)
            agent.hunger = max(0, agent.hunger - relief)
            record_memory(agent, "Ate food")
            return f"{agent.name} ate food."
        # M1.3 tools (reach): a tool-user can forage food from an ADJACENT free tile,
        # not only the one underfoot — a real foraging edge, knowers only.
        if "tools" in know:
            cell = _adjacent_free_food(agent, world_state)
            if cell is not None:
                world_state["food"].remove(cell)
                cx, cy = cell
                world_state["grid"][cy][cx] = EMPTY
                agent.hunger = max(0, agent.hunger - relief)
                record_memory(agent, "Foraged nearby food with tools")
                return f"{agent.name} foraged nearby food with tools."
        record_memory(agent, "Tried to eat but found no food")
        return f"{agent.name} tried to eat but there was no food here."

    if action == "rest":
        record_memory(agent, "Rested")
        return f"{agent.name} rested."

    # get_decision() should never pass anything else, but stay safe.
    record_memory(agent, f"Did nothing ({action})")
    return f"{agent.name} did nothing (unknown action: {action})."


def record_memory(agent: Any, text: str) -> list[str]:
    """Append a short event to the agent's memory, keeping only the last N.

    Memory is part of the agent's state (the single source of truth for what it
    has experienced). We trim in place to the most recent MEMORY_LIMIT entries
    so the list can never grow unbounded.
    """
    agent.memory.append(text)
    if len(agent.memory) > MEMORY_LIMIT:
        agent.memory[:] = agent.memory[-MEMORY_LIMIT:]
    return agent.memory


def is_sick(agent: Any, state: dict[str, Any] | None = None) -> bool:
    """True if `agent` is currently afflicted by a plague (Day 16).

    Sick means a god-set plague marker is still in force: the current turn has not
    yet passed agent.plague_until. Pure READ over world_state — the same single
    source of truth god_mode mutated. Used by the hunger loop (extra drain) and by
    perception/social memory (others can SEE that a neighbour looks sick).
    """
    state = world_state if state is None else state
    until = getattr(agent, "plague_until", 0)
    return 0 < until and state["turn"] <= until


def update_hunger(agent: Any) -> int:
    """Advance hunger by one turn, clamped at HUNGER_MAX. Returns the new hunger.

    Called once per turn by the simulation loop. Hunger never exceeds HUNGER_MAX
    (10); reaching it means starvation (see is_dead).

    Day 16 plague: a sick agent (is_sick) loses PLAGUE_HUNGER_PER_TURN instead of
    HUNGER_PER_TURN. When the plague window has fully elapsed and the agent is still
    alive, it RECOVERS here — the marker is cleared and a single recovery memory is
    written — so survival is its own escape from the sickness. god_mode only set the
    marker; the effect and the recovery both fall out of this existing loop, never a
    scripted reaction.
    """
    until = getattr(agent, "plague_until", 0)
    if until and world_state["turn"] > until:
        agent.plague_until = 0  # survived the window — recover
        record_memory(agent, "Recovered from the plague")
    per = PLAGUE_HUNGER_PER_TURN if is_sick(agent) else HUNGER_PER_TURN
    agent.hunger = min(HUNGER_MAX, agent.hunger + per)
    return agent.hunger


def is_dead(agent: Any) -> bool:
    """True if the agent has starved (hunger has reached HUNGER_MAX)."""
    return agent.hunger >= HUNGER_MAX


def mark_dead(agent: Any) -> None:
    """Flag `agent` as dead and free the cell it occupied (Day 6).

    The agent stays in world_state["agents"] so its final state remains
    inspectable, but its grid cell reverts to FOOD (if uneaten food sits there)
    or EMPTY so living agents can move through it and detection no longer sees it.
    """
    agent.alive = False
    x, y = agent.position
    world_state["grid"][y][x] = FOOD if (x, y) in world_state["food"] else EMPTY
    # M0.3: a dead agent no longer occupies space — drop it from the position index
    # (but only if it still holds the cell; a later arrival may already own it).
    if world_state["occupancy"].get((x, y)) is agent:
        world_state["occupancy"].pop((x, y), None)


def render(state: dict[str, Any] | None = None) -> str:
    """Return an ASCII snapshot of the world for logging (Day 6).

    Legend: '.' empty, '*' food, '$' treasure (Day 15), '#' a settlement centre
    (M2.1, read-only overlay), and each living agent's first initial on its cell.
    Built from the food/treasure/settlement records and live agent positions (not the
    grid array) so the picture always matches the authoritative state. The settlement
    overlay is a PURE READ of world_state["settlements"]; it draws nothing for a run
    with no settlements, so the default v1 picture is unchanged.
    """
    state = world_state if state is None else state
    size = state["size"]
    cells = [["." for _ in range(size)] for _ in range(size)]

    # M2.1: settlement centres first (lowest priority — food/treasure/agents overwrite).
    for s in state.get("settlements", {}).values():
        sx, sy = s["center"]
        cells[sy][sx] = "#"
    for fx, fy in state["food"]:
        cells[fy][fx] = "*"
    for t in state["treasures"]:
        tx, ty = t["pos"]
        cells[ty][tx] = "$"
    for agent in state["agents"]:
        if getattr(agent, "alive", True):
            ax, ay = agent.position
            cells[ay][ax] = agent.name[0]

    return "\n".join(" ".join(row) for row in cells)
