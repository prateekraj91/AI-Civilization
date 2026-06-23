"""
heuristic.py
============

A real, pure-Python agent MIND (V2 milestone M0.1).

Why this exists
---------------
V1's only "mind" is the LLM: every STRATEGY_INTERVAL turns an agent asks the model
layer (`llm.get_strategy`) for a high-level plan, and `strategy.choose_action`
executes that plan in pure Python in between. That works, but every agent costs
inference, so running MANY agents is expensive.

This module is a second mind that needs ZERO model calls. Given an agent's
perception of `world_state`, it picks a high-level strategy from sensible survival
rules — not at random — and returns it in the **exact same dict shape** that
`llm.get_strategy` returns. So it plugs into the SAME call site the strategy
system already uses (`main.run_agent_turn`), behind a single per-agent flag
(`Agent.cognition`). The rest of the loop — `choose_action`, the conversation /
alliance / trust layers, death/respawn, the renderer — cannot tell which mind is
driving an agent.

Distinct from the `random` provider
------------------------------------
`llm._random_strategy` returns a plausible-but-arbitrary strategy (weighted dice).
This policy is DETERMINISTIC and goal-directed: it reads hunger, what food it can
perceive, what food exists on the map, and personality, and chooses the action a
reasonable survivor would. Combined with `choose_action` (which already owns the
survival override, navigation, and eating), it keeps an agent alive about as well
as competent play allows, given the food actually available.

Not in this milestone (M0.2/M0.3): there is NO tier/promotion system here. Cognition
is a clean per-agent switch, nothing more — heuristic agents never "become" LLM
agents and vice-versa. Scaling to large populations is M0.3; this module only has
to be call-free and cheap enough that doing so is plausible.
"""

from __future__ import annotations

from typing import Any

import world
from strategy import DIRECTIONS, Strategy, choose_action, get_personality

# Hunger at/above which the heuristic mind makes FOOD the priority. Kept a touch
# below strategy.SURVIVAL_HUNGER (5) so the mind starts steering toward food a turn
# BEFORE the executor's hard survival override would force it — the same "head
# toward food soon" band the LLM prompt uses (hunger_line, >= 4).
HEURISTIC_HUNGER: int = 4


def _strat(kind: str, *, target: str = "", reason: str = "") -> dict[str, Any]:
    """Build a strategy dict identical in shape to what llm.get_strategy returns.

    The `message`/`reaction` fields exist only so a heuristic Strategy is a perfect
    structural drop-in for an LLM one; a zero-LLM mind never volunteers speech, so
    they are always empty (talking still happens reactively via choose_action's
    friendly default + the conversation layer, with no inference).
    """
    return {"strategy": kind, "target": target, "message": "", "reaction": "",
            "reason": reason}


def _search_direction(agent: Any, state: dict[str, Any]) -> str:
    """A direction to EXPLORE when hungry with no food known anywhere.

    Biases a searcher toward the centre of the map: food tends to be (and, with
    clustering, always is) nearer the middle than the corners, and an agent pinned
    against an edge has the least new ground in front of it. Once roughly central,
    it rotates by a position/hunger seed so repeated searches sweep different ways
    instead of oscillating on one axis. The executor re-checks the direction is
    actually open, so a blocked pick simply falls through to a roaming default.
    """
    x, y = agent.position
    c = state["size"] // 2
    if abs(x - c) >= abs(y - c) and x != c:
        return "east" if x < c else "west"
    if y != c:
        return "south" if y < c else "north"
    return DIRECTIONS[(x + y + agent.hunger) % len(DIRECTIONS)]


def _near_visible_food(scan: dict[str, Any]) -> bool:
    """True if food is underfoot or in an adjacent cell (this agent's senses)."""
    if scan["on_food"]:
        return True
    return any(cell["food"] for cell in scan["cells"].values())


def decide_strategy(agent: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Pick a high-level strategy from pure-Python rules — NO model call.

    Returns the same {"strategy", "target", "message", "reaction", "reason"} dict
    `llm.get_strategy` returns, so this is a structural drop-in at the strategy
    refresh call site. The rules, in priority:

      1. Hungry (>= HEURISTIC_HUNGER) and food exists somewhere on the map ->
         `seek_food`: head to / eat the nearest food (choose_action navigates).
      2. Hungry but NO food known anywhere -> `explore` toward unseen ground to
         find some, rather than resting while starving.
      3. Well-fed -> defer to PERSONALITY:
           - caution      -> rest when near a food cache, else hug known food
           - curiosity    -> explore
           - friendliness -> wander (choose_action's friendly default seeks company)
           - independence -> avoid others / explore alone
    """
    pers = get_personality(agent)
    scan = world.scan(agent, state)

    if agent.hunger >= HEURISTIC_HUNGER:
        if state["food"]:
            where = "adjacent" if _near_visible_food(scan) else "on the map"
            return _strat("seek_food", reason=f"hungry ({agent.hunger}); food {where}")
        return _strat("explore", target=_search_direction(agent, state),
                      reason=f"hungry ({agent.hunger}); no food in sight -> search")

    dom = pers.dominant
    if dom == "caution":
        if _near_visible_food(scan):
            return _strat("rest", reason="fed & cautious near food -> conserve")
        return _strat("seek_food", reason="fed & cautious -> stay near food")
    if dom == "curiosity":
        return _strat("explore", target=_search_direction(agent, state),
                      reason="fed & curious -> explore")
    if dom == "friendliness":
        return _strat("wander", reason="fed & friendly -> seek company")
    return _strat("avoid", reason="fed & independent -> keep to self")


def decide_action(agent: Any, state: dict[str, Any]) -> tuple[str, str]:
    """The full heuristic policy: perception -> ONE valid action (zero LLM calls).

    A convenience that runs the heuristic mind and then the shared executor in one
    step, returning (action, note) exactly as choose_action does. `main` does not
    call this — it keeps the strategy cached across turns like the LLM path — but it
    is the honest, testable statement of "this mind, given this world, would do X".
    """
    data = decide_strategy(agent, state)
    strat = Strategy(kind=data["strategy"], target=data["target"],
                     issued_turn=state.get("turn", 0))
    return choose_action(agent, strat, state)
