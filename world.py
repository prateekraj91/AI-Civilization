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

    Looks North/South/East/West of the agent's position. Cells beyond the edge
    of the map report as "wall". This is a pure READ of world_state — it never
    mutates anything (agents only observe).
    """
    x, y = agent.position
    size = world_state["size"]
    grid = world_state["grid"]

    # Ordered so output is always N, S, E, W.
    directions: dict[str, tuple[int, int]] = {
        "North": (x, y - 1),
        "South": (x, y + 1),
        "East": (x + 1, y),
        "West": (x - 1, y),
    }

    lines: list[str] = []
    for name, (nx, ny) in directions.items():
        if 0 <= nx < size and 0 <= ny < size:
            lines.append(f"{name}: {grid[ny][nx]}")
        else:
            lines.append(f"{name}: wall")

    return "\n".join(lines)
