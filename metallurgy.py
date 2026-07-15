"""
metallurgy.py
=============

METALLURGY & ARMS — technology transforms war and work (V2 milestone M4.11, Arc 4: the road to
modernity). On top of M4.10 (writing), Arc 3 (culture), Arc 2 (revolt), Arc 1 (dynasties) and Phases 0-3.

The historical step M4.11 makes — technology becomes MATERIAL power
------------------------------------------------------------------
M4.10 made tech act on institutions (memory). M4.11 makes it transform the MATERIAL balance of power:
better tools strengthen the ECONOMY, and ARMS multiply FORCE in battle — so KNOWLEDGE starts to beat
NUMBERS, and control of the knowledge of arms becomes politically decisive. A metallurgical people
out-produces a neolithic one, and an armed host beats an equal-sized unarmed one — everywhere battles
resolve (conquest M3.4, war M3.6, AND uprising M4.5).

SCOPE — M4.11 is a metallurgy tech BRANCH with two effects (tool YIELD + armed FORCE multiplier) and
its compositions. It does NOT build full ERA progression (Bronze/Iron eras + visible town evolution) —
that is M4.12 (stated boundary). Metallurgy is a small chain: `metalworking` (better tools, off `tools`)
then `weapons` (arms, off `metalworking`), discovered/spread through the EXISTING M1.2 discovery + M1.1
diffusion, exactly like every other tech. Zero LLM.

The two effects, each composing with an existing system
------------------------------------------------------
1. BETTER TOOLS (economy): a farmer who knows `metalworking` grows food at METALWORK_YIELD_MULT the base
   rate (the boost lives in `knowledge.farm`, gated on the skill + the flag, so a run without metallurgy
   is byte-identical). A metallurgical settlement is materially more productive than a neolithic one.
2. ARMS MULTIPLY FORCE (the teeth): in the shared `monarchy.resolve_battle`, an ARMED combatant (knows
   `weapons`) counts for ARMED_MULTIPLIER of an unarmed head. Since resolve_battle is reused by conquest,
   inter-kingdom war AND uprising, arms tilt EVERY clash. And — the sharpest composition — arms spread by
   TECH DIFFUSION: whoever KNOWS weapons is armed, ruler AND commoner. So an armed garrison crushes an
   unarmed peasant mob (steel beats numbers — M4.5's counter-revolution gains real teeth), BUT a mob that
   ALSO learned weapons fights an armed garrison on even terms and NUMBERS decide again (revolt stays
   alive in advanced societies). WHO is allowed to know weapons determines WHO can resist — arms are
   political. When nobody knows weapons (metallurgy off), every combat weight is 1.0 and battles are
   byte-identical to before.
"""

from __future__ import annotations

import random
from typing import Any

import knowledge
import lineage
import world

# --- The metallurgy tech chain (past `tools`) --------------------------------
METALWORKING = "metalworking"    # better tools (economy) — prereq: tools
WEAPONS = "weapons"              # arms (force multiplier) — prereq: metalworking
# item -> the prior TECH it needs (beyond the shared environmental prereq: a settlement with surplus).
CHAIN: dict[str, str] = {METALWORKING: "tools", WEAPONS: METALWORKING}

# ARMED_MULTIPLIER: how much an armed combatant (knows `weapons`) outweighs an unarmed head in the shared
# battle math. 1.8 -> a smaller armed host beats a larger unarmed one up to ~1.8x its size, so knowledge
# beats numbers without annihilating the value of numbers entirely (two armed sides fall back to a count).
ARMED_MULTIPLIER = 1.8


def is_armed(agent: Any) -> bool:
    """True if `agent` knows how to make/wield weapons — its combat head counts for more. Pure read."""
    return WEAPONS in getattr(agent, "knowledge", ())


def combat_force(units: list[Any]) -> float:
    """The effective FORCE of `units`: each armed combatant counts ARMED_MULTIPLIER, each unarmed 1.0.

    Reused by `monarchy.resolve_battle` for BOTH sides of every clash. When no unit is armed (metallurgy
    off, or a purely neolithic fight) this is exactly `len(units)`, so the battle is byte-identical to before."""
    return sum(ARMED_MULTIPLIER if is_armed(u) else 1.0 for u in units)


def discover(state: dict[str, Any], turn: int, rng: "random.Random | None" = None) -> list[str]:
    """Let eligible agents INVENT the metallurgy chain (M1.2 machinery, ZERO LLM). Returns inventors.

    Eligible for an item = holds its prior tech (metalworking needs `tools`, weapons needs `metalworking`)
    AND is in a SETTLEMENT holding a food SURPLUS (a forge needs a stable, provisioned town). The chance
    reuses `knowledge.discovery_probability`; prereqs read a turn-start snapshot so no within-turn
    tools->metalworking->weapons cascade. Draws RNG from the seeded stream like tech discovery; gated on
    metallurgy_on by the caller, so an off run never invents these and stays byte-identical. Spreads via
    ordinary M1.1 diffusion once known."""
    draw = (rng or random).random
    living = [a for a in state["agents"] if a.alive]
    snapshot = {a.name: frozenset(a.knowledge) for a in living}  # no within-turn chaining
    invented: list[str] = []
    for agent in living:  # stable order
        sid = getattr(agent, "settlement", None)
        if sid is None or not lineage.settlement_surplus(state, sid):
            continue
        known = snapshot[agent.name]
        for item, prereq in CHAIN.items():  # metalworking before weapons — deterministic
            if item in known or prereq not in known:
                continue
            p = knowledge.discovery_probability(agent, item, state)
            if p > 0.0 and draw() < p:
                agent.knowledge.add(item)
                verb = "forged the first WEAPONS" if item == WEAPONS else "mastered METALWORKING"
                world.record_memory(agent, f"{verb} in {sid}")
                state["events"].append(f"turn {turn}: {agent.name} {verb} in {sid}")
                invented.append(agent.name)
    return invented


def update(state: dict[str, Any], turn: int, rng: "random.Random | None" = None) -> list[str]:
    """Advance metallurgy one turn (M4.11): invent the chain (seeded). Spread is the ordinary M1.1
    diffusion; the ECONOMY boost lives in knowledge.farm and the FORCE multiplier in resolve_battle, both
    keyed on the skills. ZERO LLM. Gated on metallurgy_on by the caller, so an off run never calls this
    and stays byte-identical. Returns the inventors' names."""
    return discover(state, turn, rng)


# --- Derived read-outs (pure reads, for the summary / tests) -----------------
def is_metallurgical(state: dict[str, Any], sid: str) -> bool:
    """True if any living member of settlement `sid` knows metalworking (the town has a forge)."""
    rec = state.get("settlements", {}).get(sid)
    if rec is None:
        return False
    return any(a.alive and a.name in rec["members"] and METALWORKING in a.knowledge
               for a in state["agents"])


def is_armed_settlement(state: dict[str, Any], sid: str) -> bool:
    """True if any living member of settlement `sid` knows weapons (the town can field armed fighters)."""
    rec = state.get("settlements", {}).get(sid)
    if rec is None:
        return False
    return any(a.alive and a.name in rec["members"] and WEAPONS in a.knowledge
               for a in state["agents"])
