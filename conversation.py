"""
conversation.py
===============

The Day 8 "talk" foundation: agents say short messages to adjacent agents, the
recipient reacts (reply / ignore / hostile) on its NEXT turn, and everything is
logged to world_state["events"] and to both agents' bounded memory.

Hard constraint — NO new per-turn inference
--------------------------------------------
Talking must NOT add an LLM call. It reuses the existing strategy-refresh system:

  - The *message* an agent says comes from the SAME strategy LLM call that already
    runs on a refresh turn (Strategy.message). On non-refresh turns the message is
    a deterministic template derived from personality + top goal.
  - The recipient's *reaction* comes from that same strategy call on a refresh
    turn (Strategy.reaction); otherwise it is a deterministic personality rule.
  - Replies are always templated (deterministic) and never themselves trigger a
    reply, so a single exchange can't spiral into an unbounded chain.

Delivery model
--------------
A message sent on turn T is stamped with T and dropped in the recipient's
`inbox`. It is only consumed on a LATER turn (turn strictly greater than T), so it
always arrives in the recipient's *next* decision context, never the same tick.
Trust scoring is intentionally NOT implemented here — that's Day 9. Hostility is
merely logged for that future work.
"""

from __future__ import annotations

from typing import Any

import trust
import world
from strategy import get_personality

# Mirror of llm.VALID_REACTIONS, re-exported for callers/tests of this module.
VALID_REACTIONS = ("reply", "ignore", "hostile")


# --- Deterministic text + reaction (personality-driven, no LLM) ------------
def _top_goal(agent: Any) -> str:
    """The agent's strongest goal name (defaults to 'survive')."""
    if not agent.goals:
        return "survive"
    return max(agent.goals, key=agent.goals.get)


def template_message(agent: Any) -> str:
    """A short opener templated from the speaker's dominant trait + top goal."""
    goal = _top_goal(agent)
    by_trait = {
        "friendliness": f"Hi! Want to team up? I'm focused on {goal}.",
        "curiosity": "Hey — seen anything interesting around here?",
        "caution": "Careful — this area is mine. I'm watching it.",
        "independence": f"This is my patch. I'm chasing {goal}, so keep clear.",
    }
    return by_trait.get(get_personality(agent).dominant, "Hello there.")


def template_reply(agent: Any) -> str:
    """A short reply templated from the replier's dominant trait."""
    by_trait = {
        "friendliness": "Good to meet you! Let's look out for each other.",
        "curiosity": "Interesting — tell me more.",
        "caution": "...fine. Just keep your distance.",
        "independence": "Noted. Don't get in my way.",
    }
    return by_trait.get(get_personality(agent).dominant, "Okay.")


def deterministic_reaction(agent: Any) -> str:
    """Pick a reaction from personality when no LLM reaction is available."""
    by_trait = {
        "friendliness": "reply",
        "curiosity": "reply",
        "caution": "ignore",
        "independence": "hostile",
    }
    return by_trait.get(get_personality(agent).dominant, "ignore")


# --- Inbox plumbing --------------------------------------------------------
def _find_agent(state: dict[str, Any], name: str) -> Any | None:
    for a in state["agents"]:
        if a.name == name:
            return a
    return None


def _deliver(sender_name: str, recipient: Any, text: str, turn: int,
             is_reply: bool) -> None:
    """Append a message to `recipient.inbox`, stamped with the sending turn."""
    recipient.inbox.append(
        {"from": sender_name, "text": text, "turn": turn, "reply": is_reply}
    )


def pending_incoming(agent: Any, turn: int) -> list[str]:
    """Render this turn's CONSUMABLE initial messages as prompt lines.

    Only messages sent on an earlier turn count (so they surface in the *next*
    decision context). Replies are excluded — they need no reaction.
    """
    return [
        f'{e["from"]}: "{e["text"]}"'
        for e in agent.inbox
        if e["turn"] < turn and not e["reply"]
    ]


# --- The two turn-time entry points ---------------------------------------
def handle_talk(agent: Any, action: str, strat: Any, refreshed: bool,
                turn: int, state: dict[str, Any]) -> str:
    """Execute a `talk_to_<name>` action: validate range, deliver, log (Day 8).

    Range rule reuses detection (adjacent N/S/E/W). Out of range → the documented
    no-op. The message text is the strategy's LLM message on the turn it was
    produced, otherwise a personality template — never a new LLM call.
    """
    target_name = action[len("talk_to_"):]
    target = world.adjacent_agents(agent, state).get(target_name)

    if target is None:
        world.record_memory(agent, f"Tried to talk to {target_name} but no one was there")
        state["events"].append(
            f"turn {turn}: {agent.name} tried to talk to {target_name} but no one was there"
        )
        return f"{agent.name} tried to talk to {target_name} but no one was there."

    # Use the LLM-provided message only on the refresh turn that generated it.
    if refreshed and turn == strat.issued_turn and strat.message:
        msg = strat.message
    else:
        msg = template_message(agent)

    _deliver(agent.name, target, msg, turn, is_reply=False)
    world.record_memory(agent, f"I told {target.name}: {msg}")
    state["events"].append(f'turn {turn}: {agent.name} talked to {target.name}: "{msg}"')
    return f'{agent.name} talked to {target.name}: "{msg}"'


def handle_steal(agent: Any, action: str, turn: int,
                 state: dict[str, Any]) -> str:
    """Execute a `steal_from_<name>` action: take a neighbour's food (Day 12).

    Range rule (documented): the victim must be in an ADJACENT N/S/E/W cell — the
    same reach as talk. (Agents can never share a cell, so "adjacent" is the only
    workable contact range.) The theft SUCCEEDS only when the victim is alive and
    is standing on a food tile (the food it holds / is about to eat).

    On success the food transfers to the thief, who eats it immediately (hunger
    relief), and it is removed from the victim's reach. Consequences:
      - events[]   gets a clear THEFT line.
      - the victim remembers the betrayal and its trust in the thief drops by
        trust.THEFT_PENALTY PERMANENTLY (a latched grudge — see trust.adjust_trust).
      - the thief remembers it may be retaliated against.

    Invalid attempts (no one there, dead, or no food to take) are a logged no-op —
    never a crash. Like talk, this rides the existing strategy call and adds NO
    new inference.
    """
    victim_name = action[len("steal_from_"):]
    victim = world.adjacent_agents(agent, state).get(victim_name)

    # --- Validity gates: in range, alive, actually holding food -------------
    if victim is None or not getattr(victim, "alive", True):
        world.record_memory(agent, f"Tried to steal from {victim_name} but no one was there")
        state["events"].append(
            f"turn {turn}: {agent.name} tried to steal from {victim_name} but no one was in reach"
        )
        return f"{agent.name} tried to steal from {victim_name} but no one was in reach."

    if victim.position not in state["food"]:
        world.record_memory(agent, f"Tried to steal from {victim.name} but they had no food")
        state["events"].append(
            f"turn {turn}: {agent.name} tried to steal from {victim.name} but they had no food"
        )
        return f"{agent.name} tried to steal from {victim.name} but there was no food to take."

    # --- Success: transfer the food, thief eats it -------------------------
    state["food"].remove(victim.position)
    agent.hunger = max(0, agent.hunger - world.EAT_RELIEF)

    world.record_memory(agent, f"I stole from {victim.name}. They may retaliate.")
    world.record_memory(victim, f"{agent.name} stole my food on turn {turn}.")
    state["events"].append(f"turn {turn}: {agent.name} stole food from {victim.name}")

    # Permanent, dominant trust hit (bigger than a hostile message's -3).
    trust.bump_interaction(victim, agent.name)
    trust.adjust_trust(victim, agent.name, -trust.THEFT_PENALTY, "theft", turn, state,
                       permanent=True)

    return f"{agent.name} stole food from {victim.name}."


def process_inbox(agent: Any, refreshed: bool, llm_reaction: str,
                  turn: int, state: dict[str, Any]) -> list[tuple[str, str]]:
    """Consume this turn's deliverable messages and react (Day 8).

    Returns a list of (reaction_or_event, other_name) for logging. Messages sent
    this same tick are left in the inbox for next turn. Initial messages get a
    reaction (LLM-chosen on a refresh turn, else a personality rule); replies are
    just acknowledged so the exchange terminates.
    """
    outcomes: list[tuple[str, str]] = []
    remaining: list[Any] = []

    for entry in agent.inbox:
        if entry["turn"] >= turn:
            remaining.append(entry)  # sent this tick → deliver on a later turn
            continue

        sender = entry["from"]
        text = entry["text"]

        if entry["reply"]:
            # A reply to something we said. Acknowledge only — no further chain.
            world.record_memory(agent, f"{sender} replied: {text}")
            state["events"].append(f'turn {turn}: {agent.name} heard {sender} reply: "{text}"')
            # Trust (Day 9): we received a non-hostile reply -> +1.
            trust.bump_interaction(agent, sender)
            trust.adjust_trust(agent, sender, +1, "friendly reply", turn, state)
            outcomes.append(("heard_reply", sender))
            continue

        reaction = llm_reaction if (refreshed and llm_reaction in VALID_REACTIONS) \
            else deterministic_reaction(agent)
        suffix = {
            "reply": " ...I replied",
            "ignore": " ...I ignored them",
            "hostile": " ...hostile",
        }[reaction]
        world.record_memory(agent, f"{sender} said to me: {text}{suffix}")
        state["events"].append(
            f'turn {turn}: {agent.name} received from {sender}: "{text}" -> {reaction}'
        )

        # Trust (Day 9): this is one talk exchange, so count an interaction.
        trust.bump_interaction(agent, sender)
        sender_agent = _find_agent(state, sender)

        if reaction == "hostile":
            # We are hostile toward the sender; the SENDER is the one who
            # "receives" hostility -> their trust in us drops by 3.
            world.record_memory(agent, f"Felt hostile toward {sender}")
            state["events"].append(f"turn {turn}: {agent.name} flagged hostility toward {sender}")
            if sender_agent is not None:
                trust.bump_interaction(sender_agent, agent.name)
                trust.adjust_trust(sender_agent, agent.name, -3, "hostile message", turn, state)
        else:
            # reply/ignore: we received a non-hostile message -> +1 toward sender.
            trust.adjust_trust(agent, sender, +1, "friendly message", turn, state)
            if reaction == "reply" and sender_agent is not None and sender_agent.alive:
                reply_text = template_reply(agent)
                _deliver(agent.name, sender_agent, reply_text, turn, is_reply=True)
                state["events"].append(
                    f'turn {turn}: {agent.name} replied to {sender}: "{reply_text}"'
                )

        outcomes.append((reaction, sender))

    agent.inbox[:] = remaining
    return outcomes
