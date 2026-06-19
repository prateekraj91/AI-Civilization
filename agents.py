"""
agents.py
=========

Defines the Agent data model for AI Civilization.

ARCHITECTURE NOTE
-----------------
An Agent is a *plain data container*. It holds no game logic and never mutates
the world directly. Agents only ever READ from `world_state` (the single source
of truth) when deciding what to do. Any change to the world flows back through
the world layer — never by an agent writing to global state.

Keeping Agent as a dataclass (pure data) makes future features trivial to add:
  - serialization / save-load
  - sending agent state to Gemini as context
  - God Mode mutating agents via world_state
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Agent:
    """A single inhabitant of the civilization.

    Fields are intentionally simple/primitive so the whole agent can be
    serialized and embedded inside `world_state` without special handling.
    """

    name: str
    personality: str

    # goals: a weighted map of drives, e.g. {"survive": 8, "wealth": 3}.
    # Higher number = stronger motivation. Used later for decision-making.
    goals: dict[str, int] = field(default_factory=dict)

    # hunger: 0 = full, higher = hungrier. Simple survival stat for now.
    hunger: int = 0

    # position: (x, y). No world grid yet (Day 1), but the field exists so the
    # data model is ready for movement without a future schema migration.
    position: tuple[int, int] = (0, 0)

    # inventory: list of items the agent is carrying.
    inventory: list[Any] = field(default_factory=list)

    # memory: append-only log of what the agent has experienced/observed.
    # No summarization yet (explicitly out of scope for Day 1).
    memory: list[Any] = field(default_factory=list)
