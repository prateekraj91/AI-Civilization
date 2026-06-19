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
EAT_RELIEF = 5         # hunger removed by eating one food (clamped at 0)

# The authoritative world state. Keep this as the ONLY mutable global.
world_state: dict[str, Any] = {
    "turn": 0,           # current simulation tick
    "agents": [],        # list[Agent] living in the world
    "events": [],        # chronological log of things that have happened
    "size": GRID_SIZE,   # grid is size x size
    "grid": [],          # 2D list[list[str]], grid[y][x] -> cell kind
    "food": [],          # list[tuple[int, int]] of (x, y) food coordinates
}


def add_agent(agent: Any) -> None:
    """Register an agent in the world.

    This helper exists so callers go *through* the world layer instead of
    poking `world_state["agents"]` directly. It gives us one place to later add
    validation, indexing, or God Mode hooks — without changing call sites.
    """
    world_state["agents"].append(agent)


def create_world(size: int = GRID_SIZE) -> list[list[str]]:
    """Build an empty `size` x `size` grid and store it in world_state.

    Resets the grid and food list so the world starts from a clean slate. The
    grid is the single spatial source of truth; everything else (food, agent
    placement) is written onto it through the helpers below.
    """
    world_state["size"] = size
    world_state["grid"] = [[EMPTY for _ in range(size)] for _ in range(size)]
    world_state["food"] = []
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
    return agent.position


def spawn_food(count: int = 5) -> list[tuple[int, int]]:
    """Randomly place `count` food cells onto the grid.

    Rules enforced:
      - food never overlaps existing food
      - food never overlaps an agent's position

    Food coordinates are stored in world_state["food"] AND written onto the grid
    so both the list (for debugging) and the spatial view agree.
    """
    size = world_state["size"]

    # Cells we must avoid: every agent's position and any food already placed.
    occupied: set[tuple[int, int]] = {a.position for a in world_state["agents"]}
    occupied |= set(world_state["food"])

    placed = 0
    while placed < count:
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        if (x, y) in occupied:
            continue  # collision with agent or food — try again
        occupied.add((x, y))
        world_state["food"].append((x, y))
        world_state["grid"][y][x] = FOOD
        placed += 1

    return world_state["food"]


def observe(agent: Any, world_state: dict[str, Any]) -> str:
    """Inspect the four cells adjacent to `agent` and return a readable string.

    Reports the agent's CURRENT tile first, then North/South/East/West. Cells
    beyond the edge of the map report as "wall". The current tile reads from
    world_state["food"] (not the grid, which shows AGENT where the agent stands)
    so the agent knows whether it is actually standing on food — the signal it
    needs to legitimately choose "eat". This is a pure READ of world_state.
    """
    x, y = agent.position
    size = world_state["size"]
    grid = world_state["grid"]

    # The agent occupies its own cell, so the grid shows AGENT there. The truth
    # of whether food is underfoot lives in the food list.
    current_tile = FOOD if (x, y) in world_state["food"] else EMPTY

    # Ordered so output is always N, S, E, W.
    directions: dict[str, tuple[int, int]] = {
        "North": (x, y - 1),
        "South": (x, y + 1),
        "East": (x + 1, y),
        "West": (x - 1, y),
    }

    lines: list[str] = [f"Current Tile: {current_tile}", ""]
    for name, (nx, ny) in directions.items():
        if 0 <= nx < size and 0 <= ny < size:
            lines.append(f"{name}: {grid[ny][nx]}")
        else:
            lines.append(f"{name}: wall")

    return "\n".join(lines)


def move_agent(agent: Any, dx: int, dy: int) -> bool:
    """Move `agent` by (dx, dy), keeping the grid in sync. Returns True if moved.

    Boundary rule: a move that would leave the map is refused and the agent
    stays put (returns False). When the agent vacates a cell we restore it to
    FOOD if uneaten food still sits there, otherwise EMPTY — so the grid always
    reflects world_state. The destination cell becomes AGENT.
    """
    x, y = agent.position
    nx, ny = x + dx, y + dy
    size = world_state["size"]

    if not (0 <= nx < size and 0 <= ny < size):
        return False  # would leave the map — stay in place

    # Vacate the old cell. Food is tracked in world_state["food"]; if uneaten
    # food remains here, the cell reverts to FOOD, else EMPTY.
    world_state["grid"][y][x] = FOOD if (x, y) in world_state["food"] else EMPTY

    # Occupy the new cell.
    agent.position = (nx, ny)
    world_state["grid"][ny][nx] = AGENT
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
            return f"{agent.name} moved {direction}."
        return f"{agent.name} tried to move {direction} but hit the map edge and stayed put."

    if action == "eat":
        # The agent may only eat what is on its CURRENT tile. The grid cell
        # stays AGENT (agent on top); only the food record + hunger change.
        if agent.position in world_state["food"]:
            world_state["food"].remove(agent.position)
            agent.hunger = max(0, agent.hunger - EAT_RELIEF)
            return f"{agent.name} ate food."
        return f"{agent.name} tried to eat but there was no food here."

    if action == "rest":
        return f"{agent.name} rested."

    # decide() should never pass anything else, but stay safe.
    return f"{agent.name} did nothing (unknown action: {action})."


def update_hunger(agent: Any) -> int:
    """Advance hunger by one turn, clamped at HUNGER_MAX. Returns the new hunger.

    Called once per turn by the simulation loop. Hunger never exceeds
    HUNGER_MAX (10); reaching it means starvation (see is_dead).
    """
    agent.hunger = min(HUNGER_MAX, agent.hunger + HUNGER_PER_TURN)
    return agent.hunger


def is_dead(agent: Any) -> bool:
    """True if the agent has starved (hunger has reached HUNGER_MAX)."""
    return agent.hunger >= HUNGER_MAX
