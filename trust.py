"""
trust.py
========

Day 9 trust dynamics: each agent forms an opinion of every other agent it has
interacted with, and that opinion shapes future behaviour through the strategy
prompt. It is PURE PYTHON bookkeeping driven entirely by the conversation events
already produced in Day 8 — it adds ZERO LLM calls.

Model
-----
Each agent owns `relationships = {other_name: {"trust": int, "interactions": int}}`,
created lazily on first contact (trust 0, interactions 0).

Trust deltas (applied by conversation.process_inbox from existing events):
  +1  the agent received a non-hostile message (a friendly talk or a reply)
  -3  the agent was the target of a hostile reaction
  interactions += 1 on any talk exchange, either direction.

(Theft-based trust loss is Day 12 and is intentionally NOT implemented here.)

Visibility
----------
Raw numbers are bucketed into low / neutral / high so the model reads intent, not
arithmetic, and the bucketed summary is injected into the strategy prompt.
"""

from __future__ import annotations

from typing import Any

import world

# Trust at/above HIGH reads as "high"; at/below LOW reads as "low"; between is
# "neutral". Small one-off changes stay neutral until a relationship builds.
HIGH_THRESHOLD = 2
LOW_THRESHOLD = -2


def ensure_relationship(agent: Any, other_name: str) -> dict[str, int]:
    """Return agent's relationship record for `other_name`, creating it if new."""
    rel = agent.relationships.get(other_name)
    if rel is None:
        rel = {"trust": 0, "interactions": 0}
        agent.relationships[other_name] = rel
    return rel


def bump_interaction(agent: Any, other_name: str) -> None:
    """Count one interaction between `agent` and `other_name`."""
    ensure_relationship(agent, other_name)["interactions"] += 1


def trust_bucket(value: int) -> str:
    """Map a raw trust number to low / neutral / high."""
    if value >= HIGH_THRESHOLD:
        return "high"
    if value <= LOW_THRESHOLD:
        return "low"
    return "neutral"


def adjust_trust(agent: Any, other_name: str, delta: int, reason: str,
                 turn: int, state: dict[str, Any]) -> int:
    """Apply a trust change, logging it to events and the agent's memory.

    Appends a compact, time-stamped line to world_state["events"] and a bounded
    social-memory entry on the agent whose trust changed. Returns the new trust.
    """
    rel = ensure_relationship(agent, other_name)
    old = rel["trust"]
    rel["trust"] = old + delta
    new = rel["trust"]

    state["events"].append(
        f"turn {turn}: {agent.name} trust in {other_name}: {old} -> {new} ({reason})"
    )
    direction = "rose" if delta > 0 else "fell"
    world.record_memory(
        agent, f"Trust in {other_name} {direction} to {new} ({reason})"
    )
    return new


def trust_summary(agent: Any) -> str:
    """One-line bucketed trust digest for the strategy prompt, or "" if none.

    Example: 'Your trust — Bob: +2 (high), Kira: -3 (low)'.
    """
    if not agent.relationships:
        return ""
    parts = [
        f"{name}: {rel['trust']:+d} ({trust_bucket(rel['trust'])})"
        for name, rel in sorted(agent.relationships.items())
    ]
    return "Your trust — " + ", ".join(parts)
