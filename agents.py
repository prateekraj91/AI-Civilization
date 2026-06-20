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

DAY 6 ADDITION
--------------
The world now hosts MULTIPLE agents (Alex, Bob, Kira) that share one grid and
compete for the same food. Each agent carries an explicit `alive` flag so the
simulation loop can skip the dead without removing them from the world (their
final state stays inspectable, which matters for logging and future features
like reputation or post-mortems).
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

    # alive: False once the agent has starved (Day 6). The simulation keeps dead
    # agents in the world for inspection but skips their turns.
    alive: bool = True

    # position: (x, y). No world grid yet (Day 1), but the field exists so the
    # data model is ready for movement without a future schema migration.
    position: tuple[int, int] = (0, 0)

    # inventory: list of items the agent is carrying.
    inventory: list[Any] = field(default_factory=list)

    # memory: append-only log of what the agent has experienced/observed.
    # No summarization yet (explicitly out of scope for Day 1).
    memory: list[Any] = field(default_factory=list)

    # inbox: pending messages from other agents (Day 8 "talk"). Each entry is a
    # dict {"from", "text", "turn", "reply"}. A message sent on turn T is only
    # consumed on a LATER turn, so it lands in the recipient's NEXT decision
    # context (never the same tick). Cleared once consumed.
    inbox: list[Any] = field(default_factory=list)
