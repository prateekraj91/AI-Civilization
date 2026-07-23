"""
eras.py
=======

ERA PROGRESSION — the march of ages (V2 milestone M4.12, CLOSES Arc 4: the road to modernity). On top
of M4.11 (metallurgy), M4.10 (writing), Arc 3 (culture), Arc 2 (revolt), Arc 1 (dynasties) and Phases 0-3.

The historical step M4.12 makes — the civilization marches through the ages
-------------------------------------------------------------------------
The world had scattered techs but no sense of ADVANCEMENT through ages. M4.12 structures the tech tree
into ordered ERAS (Neolithic -> Bronze -> Iron -> ...), each transforming the ECONOMY, WAR, and a
settlement's visible APPEARANCE — so a civilization visibly marches from the stone age toward modernity.
This closes Arc 4 and realises the long-standing "stone age to modern" goal.

SCOPE — M4.12 is the ERA MACHINERY + the first THREE eras (Neolithic/Bronze/Iron), built so higher eras
(Medieval/Renaissance/Industrial/Modern) are a pure DATA addition (append to `ERAS` — the "add an era
here" seam is marked). It does NOT build all of history now: the engine + 3 eras prove the design. Zero LLM.

Era is DERIVED from mastered tech (emergent, not declared)
----------------------------------------------------------
A settlement's ERA is the highest rung of the `ERAS` ladder whose required tech-set its populace has
COLLECTIVELY mastered (the union of its living members' knowledge). As the existing discovery/diffusion
spreads techs, a town crosses the next threshold and ADVANCES — logged ("S001 entered the Bronze Age") —
so some towns advance faster than others, seed-varying, nothing scripted.

Each era transforms three things (composing with existing systems)
------------------------------------------------------------------
a. ECONOMY: a settlement's era lifts its farm YIELD along a curve (Bronze > Neolithic, Iron > Bronze) —
   the era generalisation of M4.11's metalworking boost. A more advanced settlement is materially richer.
b. WAR: a soldier's MARTIAL era is a force multiplier in the shared `monarchy.resolve_battle` — bronze
   arms out-fight stone, iron out-fight bronze (the era generalisation of M4.11's armed multiplier). So a
   SMALLER advanced host beats a LARGER primitive one; same-era falls back to numbers. The combat weight
   is keyed on the soldier's OWN martial tech (metalworking/weapons), NOT the full collective era set (a
   soldier need not farm or write), so the intra-town arms distinction of M4.5/M4.11 survives intact.
c. APPEARANCE: the settlement's era is written to `world_state["eras"]` so the READ-ONLY renderer draws
   era-appropriate buildings (Neolithic huts -> Bronze timber+forge -> Iron stone+walls). Towns visibly
   evolve; the renderer only reads the era, never writes it.

Cost & determinism
------------------
ZERO LLM and ZERO RNG — era is a pure derivation over mastered tech. A run with the system OFF never
calls `update` (no "eras" key), the combat weighting delegates to the M4.11 model (byte-identical), and
the yield curve is never consulted, so the run is byte-identical to v1. Imports metallurgy (the M4.11
combat fallback) + world (one-directional).
"""

from __future__ import annotations

from typing import Any, NamedTuple


class Era(NamedTuple):
    name: str
    techs: frozenset          # CUMULATIVE tech-set a settlement's populace must collectively master
    yield_mult: float         # farm-yield multiplier for a settlement in this era (economy)
    style: str                # renderer building-style key (appearance)


# --- The era LADDER (ordered ascending) --------------------------------------
# *** ADD AN ERA HERE *** — appending a tuple is the ONLY change needed to extend toward modernity;
# settlement_era / yield / appearance / advance-logging all read this list, no new machinery. M4.12
# ships the first three; Medieval/Renaissance/Industrial/Modern are a later DATA addition, e.g.:
#   Era("Medieval Age", frozenset({..., "masonry", "cavalry"}), 3.6, "medieval"),
ERAS: list[Era] = [
    Era("Neolithic",  frozenset({"fire", "tools", "farming"}),                                    1.0, "neolithic"),
    Era("Bronze Age", frozenset({"fire", "tools", "farming", "metalworking"}),                    2.0, "bronze"),
    Era("Iron Age",   frozenset({"fire", "tools", "farming", "metalworking", "weapons", "writing"}), 2.8, "iron"),
]

# --- Martial combat weight per soldier (the war effect) ----------------------
# A soldier's fighting weight on the metal era curve — the generalisation of M4.11's armed multiplier.
# Keyed on the soldier's OWN martial tech (weapons -> iron arms, metalworking -> bronze arms, else stone),
# so an armed garrison still out-weights an unarmed mob within one town (M4.11 H3 preserved) AND an iron
# host out-weights a neolithic one between towns (M4.12 H2). When nobody has martial tech, all weights are
# 1.0 and the fight is a plain head count (byte-identical).
NEOLITHIC_COMBAT = 1.0
BRONZE_COMBAT = 1.4
IRON_COMBAT = 1.8


def _era_index_for(known: "set[str] | frozenset[str]") -> int:
    """Highest era index whose required tech-set is a subset of `known`, or -1 (pre-Neolithic)."""
    for i in range(len(ERAS) - 1, -1, -1):
        if ERAS[i].techs <= known:
            return i
    return -1


def _name_index(name: "str | None") -> int:
    return next((i for i, e in enumerate(ERAS) if e.name == name), -1)


def settlement_era_index(state: dict[str, Any], sid: str) -> int:
    """The index of `sid`'s era: the highest rung its living members COLLECTIVELY master (-1 if none)."""
    rec = state.get("settlements", {}).get(sid)
    if rec is None:
        return -1
    known: set[str] = set()
    for a in state["agents"]:
        if a.alive and a.name in rec["members"]:
            known |= a.knowledge
    return _era_index_for(known)


def settlement_era(state: dict[str, Any], sid: str) -> str:
    """`sid`'s era NAME (or 'Pre-Neolithic' below the ladder) — a pure read for the summary/renderer."""
    i = settlement_era_index(state, sid)
    return ERAS[i].name if i >= 0 else "Pre-Neolithic"


def agent_combat_weight(agent: Any) -> float:
    """A single soldier's fighting weight from its OWN martial tech (iron > bronze > stone)."""
    k = getattr(agent, "knowledge", ())
    if "weapons" in k:
        return IRON_COMBAT
    if "metalworking" in k:
        return BRONZE_COMBAT
    return NEOLITHIC_COMBAT


def combat_force(units: list[Any], state: dict[str, Any]) -> float:
    """The effective FORCE of `units` on the era curve — reused by `monarchy.resolve_battle`.

    With eras ON, each soldier counts its martial-era weight, so an advanced host out-weights a larger
    primitive one and same-era falls back to a count. With eras OFF this DELEGATES to the M4.11 model
    (`metallurgy.combat_force` — armed multiplier, or a plain head count when metallurgy is also off), so
    the battle math is byte-identical to before M4.12. Pure read; ZERO RNG."""
    if state.get("eras_on"):
        return sum(agent_combat_weight(u) for u in units)
    from sim import metallurgy
    return metallurgy.combat_force(units)


def yield_mult(state: dict[str, Any], agent: Any) -> float:
    """The farm-yield multiplier for `agent` from its SETTLEMENT's era (the collective advancement lifts
    every farmer's output) — 1.0 for a nomad or a pre-Neolithic town. Pure read."""
    sid = getattr(agent, "settlement", None)
    if sid is None:
        return 1.0
    i = settlement_era_index(state, sid)
    return ERAS[i].yield_mult if i >= 0 else 1.0


def building_style(state: dict[str, Any], sid: str) -> str:
    """The renderer building-style key for `sid`'s era ('neolithic'/'bronze'/'iron'/...). Pure read."""
    i = settlement_era_index(state, sid)
    return ERAS[i].style if i >= 0 else "neolithic"


def update(state: dict[str, Any], turn: int) -> list[str]:
    """Advance the era system one turn (M4.12): recompute each settlement's era, log any ADVANCE, and
    write `world_state["eras"][sid] = era_name` for the renderer/summary. ZERO LLM, ZERO RNG. Caller
    gates on `eras_on`, so an off run never calls this (no 'eras' key) and stays byte-identical. Returns
    the advance events."""
    eras_map = state.setdefault("eras", {})
    events: list[str] = []
    for sid in sorted(state.get("settlements", {})):
        i = settlement_era_index(state, sid)
        name = ERAS[i].name if i >= 0 else None
        prev_i = _name_index(eras_map.get(sid))
        if i > prev_i and i >= 0:  # an ADVANCE up the ladder (never log a regression)
            ev = f"turn {turn}: {sid} entered the {name}"
            events.append(ev)
        eras_map[sid] = name  # always store the current era (the renderer reads this)
    state.setdefault("events", []).extend(events)
    return events
