"""
alliance.py
===========

Day 13: ALLIANCES and BETRAYAL — a mutual, two-sided bond between agents, the
information advantage it buys under scarcity, and the permanent rupture that
ending it leaves behind.

Like talk (Day 8) and steal (Day 12), this adds ZERO new per-turn inference. The
decision to ally or betray rides the SAME cached strategy call every agent already
makes every few turns: the strategy executor (strategy.choose_action) emits an
`ally_with_<name>` or `betray_alliance_<name>` action, and the handlers below turn
that into world state. Nothing here ever calls the LLM.

Forming an alliance (mutual, never unilateral)
----------------------------------------------
An alliance only exists if BOTH agents agree, modelled as proposal + acceptance:

  - `agent.ally_offers` holds the names of agents who have proposed an alliance
    TO this agent and are still awaiting its answer.
  - When X plays `ally_with_Y`:
      * if Y already proposed to X (Y in X.ally_offers) -> both have now chosen
        each other, so the alliance FORMS this turn;
      * otherwise X's choice is recorded as a PROPOSAL (X added to Y.ally_offers)
        and nothing forms until Y answers on a later turn.

On formation: both agents gain the other in `agent.allies`, trust rises by
ALLY_TRUST_BONUS (+3) BOTH ways, an ALLIANCE event is logged, and both remember
it. A pair under a grudge (Day 12) on EITHER side can never form an alliance —
`can_ally` refuses it — so a betrayal or theft permanently forecloses allying.

The benefit (mechanically real, not cosmetic)
---------------------------------------------
Allies SHARE food observations. `shared_food_sightings` reports, for an agent,
the food its living allies can currently see that it cannot — and that is folded
straight into the strategy prompt (strategy.build_strategy_prompt). Two scouts
cover more ground than one, which is the concrete reason to ally under scarcity.
The instant an alliance dissolves (betrayal), the sharing stops: `are_allied` is
False, so the next prompt carries none of the ex-ally's sightings.

Betrayal (the one wound bigger than theft)
------------------------------------------
`betray_alliance_<name>` is valid only between current allies. It dissolves the
bond on both sides, drops the betrayed agent's trust in the betrayer by
BETRAYAL_PENALTY (8 — larger than theft's 5) and latches a PERMANENT grudge
(reusing the Day 12 flag), logs a BETRAYAL event, and records it as a major
memory on BOTH agents. Because the grudge is permanent and blocks allying, a
betrayed pair can never ally again.
"""

from __future__ import annotations

from typing import Any

from sim import trust
from sim import world

# Trust both agents gain toward each other when an alliance forms. Positive and
# mutual — the bond is a genuine investment in the relationship.
ALLY_TRUST_BONUS = 3

# Trust the betrayed agent loses in the betrayer. Deliberately larger than
# theft's 5 (trust.THEFT_PENALTY): betraying someone who trusted you enough to
# ally is the worst thing one agent can do to another here. Latched permanent.
BETRAYAL_PENALTY = 8


# --- Relationship predicates ----------------------------------------------
def are_allied(a: Any, b: Any) -> bool:
    """True if `a` and `b` currently hold each other as allies (always mutual)."""
    return b.name in getattr(a, "allies", set()) and a.name in getattr(b, "allies", set())


def _has_grudge(agent: Any, other_name: str) -> bool:
    """Whether `agent` carries a latched grudge toward `other_name` (Day 12)."""
    return bool(agent.relationships.get(other_name, {}).get("grudge"))


def can_ally(a: Any, b: Any) -> bool:
    """Whether `a` and `b` are eligible to form an alliance at all.

    Eligibility (independent of personality/willingness, which the strategy layer
    decides): both alive, not already allied, and NO grudge on EITHER side. The
    grudge rule is what makes a betrayed/robbed pair unable to ever re-ally.
    """
    if not getattr(a, "alive", True) or not getattr(b, "alive", True):
        return False
    if are_allied(a, b):
        return False
    if _has_grudge(a, b.name) or _has_grudge(b, a.name):
        return False
    return True


# --- Shared perception (the alliance benefit) ------------------------------
def _find_agent(state: dict[str, Any], name: str) -> Any | None:
    for agent in state["agents"]:
        if agent.name == name:
            return agent
    return None


def shared_food_sightings(agent: Any, state: dict[str, Any]) -> dict[str, list[tuple[int, int]]]:
    """Food each living ally can see that `agent` cannot — the shared observations.

    Returns {ally_name: [coords...]} for every current ally whose own perception
    (world.visible_food) holds food coordinates `agent` is not already standing
    next to. Empty when the agent has no allies or they see nothing new. Pure READ;
    no LLM call. Folded into the strategy prompt by build_strategy_prompt so the
    benefit is mechanically real, not cosmetic. A dissolved alliance (after a
    betrayal) drops out automatically because the name is no longer in `allies`.
    """
    own = world.visible_food(agent, state)
    out: dict[str, list[tuple[int, int]]] = {}
    for name in sorted(getattr(agent, "allies", set())):
        ally = _find_agent(state, name)
        if ally is None or not getattr(ally, "alive", True):
            continue
        extra = world.visible_food(ally, state) - own
        if extra:
            out[name] = sorted(extra)
    return out


# --- Turn-time action handlers (ride the cached strategy call) -------------
def handle_ally(agent: Any, action: str, turn: int, state: dict[str, Any]) -> str:
    """Execute an `ally_with_<name>` action: propose, or accept into an alliance.

    Range rule mirrors talk/steal: the target must be in an ADJACENT N/S/E/W cell.
    Out of range / dead / grudged / already-allied is a logged no-op (never a
    crash). If the target has ALREADY proposed to us (we are in nobody-can-ally
    limbo no more), the alliance FORMS now; otherwise this records our proposal and
    waits for the target to answer on a later turn — so an alliance is never
    one-sided.
    """
    target_name = action[len("ally_with_"):]
    target = world.adjacent_agents(agent, state).get(target_name)

    if target is None:
        world.record_memory(agent, f"Tried to ally with {target_name} but no one was in reach")
        state["events"].append(
            f"turn {turn}: {agent.name} tried to ally with {target_name} but no one was in reach"
        )
        return f"{agent.name} tried to ally with {target_name} but no one was in reach."

    if not can_ally(agent, target):
        # A grudge on either side (or an existing alliance) blocks it. This is how
        # a betrayed/robbed pair is permanently barred from re-allying.
        reason = "they are already allied" if are_allied(agent, target) else "a grudge stands between them"
        world.record_memory(agent, f"Could not ally with {target_name}: {reason}")
        state["events"].append(
            f"turn {turn}: {agent.name} could not ally with {target_name} ({reason})"
        )
        return f"{agent.name} could not ally with {target_name} ({reason})."

    # Acceptance: the target already offered, so both sides have now chosen.
    if target_name in agent.ally_offers:
        agent.ally_offers.discard(target_name)
        target.ally_offers.discard(agent.name)

        agent.allies.add(target_name)
        target.allies.add(agent.name)

        # +3 trust BOTH ways (a genuine, mutual investment).
        trust.bump_interaction(agent, target_name)
        trust.bump_interaction(target, agent.name)
        trust.adjust_trust(agent, target_name, ALLY_TRUST_BONUS, "alliance formed", turn, state)
        trust.adjust_trust(target, agent.name, ALLY_TRUST_BONUS, "alliance formed", turn, state)

        world.record_memory(agent, f"I allied with {target_name} on turn {turn}.")
        world.record_memory(target, f"I allied with {agent.name} on turn {turn}.")
        state["events"].append(
            f"turn {turn}: {agent.name} and {target_name} formed an ALLIANCE"
        )
        return f"{agent.name} and {target_name} formed an alliance."

    # Otherwise: record our proposal and await the target's answer next turn.
    target.ally_offers.add(agent.name)
    world.record_memory(agent, f"I proposed an alliance to {target_name}.")
    state["events"].append(
        f"turn {turn}: {agent.name} proposed an alliance to {target_name} (awaiting reply)"
    )
    return f"{agent.name} proposed an alliance to {target_name}."


def handle_betray(agent: Any, action: str, turn: int, state: dict[str, Any]) -> str:
    """Execute a `betray_alliance_<name>` action: tear up an existing alliance.

    Valid ONLY between current allies (an alliance is a relationship, so unlike
    forming it needs no adjacency — you can renounce someone from anywhere). On
    success the bond dissolves on both sides, the betrayed agent's trust in the
    betrayer falls by BETRAYAL_PENALTY (8) and latches a PERMANENT grudge, a
    BETRAYAL event is logged, and BOTH agents record it as a major memory. From
    this point the betrayed agent shares no further sightings (alliance gone) and
    the permanent grudge bars the pair from ever allying again.
    """
    ally_name = action[len("betray_alliance_"):]
    betrayed = _find_agent(state, ally_name)

    if ally_name not in getattr(agent, "allies", set()) or betrayed is None:
        world.record_memory(agent, f"Tried to betray {ally_name} but they are not an ally")
        state["events"].append(
            f"turn {turn}: {agent.name} tried to betray {ally_name} but they were not allied"
        )
        return f"{agent.name} tried to betray {ally_name} but they were not allied."

    # Dissolve the bond on BOTH sides and clear any stale offers between them.
    agent.allies.discard(ally_name)
    betrayed.allies.discard(agent.name)
    agent.ally_offers.discard(ally_name)
    betrayed.ally_offers.discard(agent.name)

    # The betrayed agent is the wounded party: trust craters and latches a grudge.
    trust.bump_interaction(betrayed, agent.name)
    trust.adjust_trust(betrayed, agent.name, -BETRAYAL_PENALTY, "betrayal", turn, state,
                       permanent=True)

    # Major memory on BOTH agents — neither forgets a betrayal.
    world.record_memory(agent, f"I BETRAYED my alliance with {ally_name} on turn {turn}.")
    world.record_memory(betrayed, f"{agent.name} BETRAYED our alliance on turn {turn}.")
    state["events"].append(
        f"turn {turn}: *** {agent.name} BETRAYED the alliance with {ally_name} ***"
    )
    return f"{agent.name} betrayed the alliance with {ally_name}."
