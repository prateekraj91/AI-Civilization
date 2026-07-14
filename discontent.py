"""
discontent.py
=============

DISCONTENT — the pressure gauge of CLASS CONFLICT (V2 milestone M4.4, opens Arc 2:
Revolt & Class Conflict). On top of all of Arc 1 (M4.1 lineage, M4.2 inheritance, M4.3
dynasties) and all of Phases 0-3 (labor, leadership, taxation, monarchy, kingdoms, empire).

The historical step M4.4 makes — pressure gets a GAUGE (but no relief valve yet)
--------------------------------------------------------------------------------
Phase 3 built a pressure engine with no way to vent: inequality COMPOUNDS (M3.1 wage
labor), monarchs LEVY by force (M3.4), tribute is EXTRACTED up a feudal hierarchy (M3.5),
and dynastic heirs inherit crowns they never earned (M4.3). Until now the oppressed just
STARVE QUIETLY. M4.4 gives that pressure a legible, verified quantity — per-agent
DISCONTENT — so the next milestone can make it BLOW.

SCOPE — M4.4 IS ONLY THE MEASURE. Discontent is built here as a readable number that
TRACKS oppression. NO uprising, revolt, mob, or riot fires in this module — that is M4.5,
which will CONSUME the pressure this module exposes (settlement_pressure). This file must
never take an action against a ruler; it only reads existing state and writes a gauge.

The gauge is DERIVED from EXISTING signals — no new psychology is invented
-------------------------------------------------------------------------
Each turn, for every settled adult, three drivers ADD grievance (magnitudes are the tunable
constants below). All three are pure reads of state other systems already compute:

  1. DEPRIVATION AMID PLENTY (the strongest driver). Being hungry while a settlement-mate
     holds a large stockpile — LOCAL inequality FELT, not an abstract Gini. Scales with how
     hungry the agent is; gated on a wealthy neighbour actually existing (plenty to resent).

  2. EXPLOITATION. Selling labor (M3.1) for a wage near the SUBSISTENCE floor — the closer
     the wage sits to bare survival (the more of the product the employer captures), the
     sharper the grievance. A worker on a tight-market wage well above subsistence resents little.

  3. EXTRACTION, buffered by LEGITIMACY. Being LEVIED by a monarch (M3.4) or paying TRIBUTE
     up a feudal hierarchy (M3.5). Weighted TWO ways:
       * by BURDEN relative to means — a levy taking half a poor agent's wealth stings far
         more than the same coin from a rich one (due / wealth, not the raw sum);
       * by LEGITIMACY — the agent's personal TRUST in whoever rules its settlement. A ruler
         it trusts draws LITTLE grievance from the SAME extraction; a distrusted one draws
         much more. This is the historically exact difference between a CONSENTED tax and a
         HATED levy — legitimacy buffers grievance. (An M3.3 consensual, redistributive tax by
         a trust-leader is not extraction at all here, so it registers ZERO — consent, not theft.)

DECAY WITH HYSTERESIS (grievances outlast their causes)
-------------------------------------------------------
Discontent only decays when NO grievance is active this turn (fed, unlevied-or-consenting,
fairly paid) — and it decays SLOWLY (DECAY per turn), far slower than a live grievance
accumulates. So one good harvest does not erase a decade of oppression: the gauge RISES fast
and FALLS slow (asymmetric slopes), and sustained good conditions eventually return it near 0.
Floored at 0; softly capped at DISCONTENT_CAP so the gauge stays legible and bounded.

SETTLEMENT PRESSURE (derived, the number M4.5 will trigger on)
-------------------------------------------------------------
`settlement_pressure(sid, state)` counts the settlement's living members whose discontent has
crossed RESENTMENT_THRESHOLD — the size of the resentful faction. It is DERIVED on demand from
the per-agent gauge (never stored separately), so it can never drift out of sync with it.

Cost & determinism
------------------
ZERO LLM calls and ZERO new RNG — the gauge is deterministic state-math over a stable (sorted
by name) iteration. Per-agent discontent lives in world_state["discontent"] (a {name: float}
map, no new Agent field), so a run with the system OFF never calls `update`, world_state
carries no "discontent" key, and the run is byte-identical to v1 (the v1 golden master
included). Imports labor + monarchy + kingdoms + world (one-directional, this is a higher
layer than any institution), keeping the world layer dependency-free.
"""

from __future__ import annotations

from typing import Any

import kingdoms
import labor
import monarchy
import world

# --- Tunable constants (documented) ----------------------------------------
# DEPRIVATION_WEIGHT: grievance per turn from hunger amid plenty, at MAXIMAL hunger. Multiplied by
# the agent's hunger fraction (hunger / HUNGER_MAX), so a mildly-hungry agent resents a little and a
# starving one a lot. The STRONGEST driver (highest weight): watching a neighbour's full granary
# while you starve is the sharpest felt injustice — local inequality made visceral.
DEPRIVATION_WEIGHT = 1.5

# HUNGER_PANG: the agent must be at least this hungry before deprivation registers at all — a fed
# agent feels no want no matter how rich its neighbours are (envy of plenty is not itself grievance;
# HUNGER is). Below it the deprivation driver is silent.
HUNGER_PANG = 4

# PLENTY_WEALTH: a settlement-mate must hold at least this liquid wealth (money + stored food) for
# the hunger to be felt AS INJUSTICE ("amid plenty"). Starving in a settlement where everyone is
# equally poor is hardship, not class grievance — it is the VISIBLE hoard next door that galls.
PLENTY_WEALTH = 12.0

# EXPLOITATION_WEIGHT: grievance per turn from a subsistence wage, at the FULL exploitation extreme
# (wage exactly at the SUBSISTENCE floor -> the employer captures the entire surplus). Scaled down by
# how far the wage sits ABOVE subsistence (a worker with real bargaining power resents little).
EXPLOITATION_WEIGHT = 0.8

# EXTRACTION_WEIGHT: grievance per turn from being levied/tributed, at MAXIMAL burden and NEUTRAL
# legitimacy. Multiplied by the burden fraction (<=1) AND the legitimacy factor, so the actual
# increment is usually a fraction of this. High because forced extraction by an unaccountable crown
# is a defining grievance of the class engine.
EXTRACTION_WEIGHT = 3.0

# MEANS_FLOOR: the divisor floor when turning an extraction into a BURDEN fraction (due / max(wealth,
# MEANS_FLOOR)). Without it, a near-broke agent's tiny levy would divide by a tiny wealth and read as
# a crushing burden; the floor keeps "burden relative to means" sane at the bottom.
MEANS_FLOOR = 5.0

# LEGIT_SLOPE / LEGIT_MIN / LEGIT_MAX: how personal TRUST in the ruler buffers extraction grievance.
# legit_factor = clamp(1.0 - LEGIT_SLOPE * trust, LEGIT_MIN, LEGIT_MAX). Neutral trust (0) -> factor
# 1.0 (the raw grievance); a trusted ruler (high +trust) -> factor toward LEGIT_MIN (grievance
# buffered away); a distrusted one (-trust) -> factor toward LEGIT_MAX (grievance amplified). This is
# the "consented tax vs hated levy" law: the SAME coin taken by a loved vs a hated ruler stings very
# differently.
LEGIT_SLOPE = 0.18
LEGIT_MIN = 0.3
LEGIT_MAX = 1.8

# DECAY: how much discontent ebbs per turn WHEN no grievance is active (fully relieved conditions).
# Deliberately far smaller than a live driver's per-turn increment, so the gauge falls MUCH slower
# than it rose — grievances outlast their causes (hysteresis). No decay at all while any grievance
# persists. Floored at 0.
DECAY = 0.3

# RESENTMENT_THRESHOLD: the gauge level at which an agent counts as RESENTFUL — the bar
# settlement_pressure tallies and the point M4.5 will trigger on. Also the (upward) crossing that
# gets a single sparse log line, so the events log reads as notable transitions, not every increment.
RESENTMENT_THRESHOLD = 6.0

# DISCONTENT_CAP: a soft ceiling so the gauge stays legible/bounded (a maximally-oppressed agent
# saturates rather than growing without limit). High enough that ordinary oppression lives below it.
DISCONTENT_CAP = 25.0

# PRESSURE_UPRISING_HINT: a purely LEGIBILITY threshold — the resentful-faction size at which a
# settlement's pressure is worth flagging in the summary. It is NOT a trigger (no uprising fires in
# M4.4; that is M4.5, which will decide its own condition on settlement_pressure). It only marks
# "this many resentful members is a notable concentration" so the world read-out is scannable.
PRESSURE_UPRISING_HINT = 2


def _wealth(a: Any) -> float:
    """An agent's liquid wealth = money + stored food (both food-claims) — the M3.1 class metric."""
    return a.money + a.stockpile


def _measured(state: dict[str, Any]) -> list[Any]:
    """The agents whose discontent is tracked: living, SETTLED adults (sorted by name).

    Nomads have no ruler and no settlement-mates to resent; dependent CHILDREN are outside every
    economic/military system (the M4.1 gate), so neither carries class grievance. Sorted so the
    per-turn iteration — and any events it logs — is deterministic.
    """
    return sorted(
        (a for a in state["agents"]
         if a.alive and a.settlement is not None and not world.is_dependent_child(a, state)),
        key=lambda a: a.name)


# --- The three drivers (each a pure read; each returns a grievance increment) ---
def deprivation(agent: Any, state: dict[str, Any]) -> float:
    """DEPRIVATION AMID PLENTY: hunger felt as injustice beside a wealthy settlement-mate.

    Silent unless the agent is genuinely hungry (>= HUNGER_PANG) AND some CO-SETTLER holds a
    large stockpile (>= PLENTY_WEALTH) — local inequality made visible. Scales with the agent's
    hunger fraction, so the hungrier it is, the sharper the grievance. The strongest driver.
    """
    if agent.hunger < HUNGER_PANG:
        return 0.0
    rec = state.get("settlements", {}).get(agent.settlement)
    if rec is None:
        return 0.0
    living = {a.name: a for a in state["agents"] if a.alive}
    plenty = any(m != agent.name and (o := living.get(m)) is not None and _wealth(o) >= PLENTY_WEALTH
                 for m in rec["members"])
    if not plenty:
        return 0.0
    return DEPRIVATION_WEIGHT * min(1.0, agent.hunger / world.HUNGER_MAX)


def exploitation(agent: Any, state: dict[str, Any]) -> float:
    """EXPLOITATION: selling labor (M3.1) for a wage near the subsistence floor.

    Reads world_state["employments"] for a link where `agent` is the WORKER. Grievance scales with
    the employer's CAPTURE share — (output - wage) / (output - subsistence), 1.0 at a subsistence
    wage, 0.0 at a wage approaching full output. A worker with market leverage (a wage above
    subsistence) resents proportionally less; a subsistence treadmill resents the most.
    """
    link = next((l for l in state.get("employments", []) if l["worker"] == agent.name), None)
    if link is None:
        return 0.0
    span = labor.LABOR_OUTPUT - labor.SUBSISTENCE_WAGE
    if span <= 0:
        return 0.0
    capture = (labor.LABOR_OUTPUT - link["wage"]) / span
    capture = max(0.0, min(1.0, capture))
    return EXPLOITATION_WEIGHT * capture


def _ruler_and_due(agent: Any, state: dict[str, Any]) -> tuple[str | None, float, str]:
    """Who extracts from `agent` this turn, how much they'd take, and the KIND — a pure read.

    Resolves the extraction the agent actually FEELS in its settlement, without double-counting:
      * a VASSAL settlement in a kingdom (M3.5) -> the vassal LORD takes TRIBUTE (the felt feudal
        relation takes precedence);
      * else a MONARCH holds the settlement (M3.4) -> the monarch LEVIES by force;
      * else (an M3.2/M3.3 consensual leader, or no ruler) -> no extraction (returns due 0.0).
    `due` uses the SAME formula the levy/tribute code applies (rate of wealth above the threshold),
    so the gauge reads the exact burden the institution imposes. Never counts the ruler taxing itself,
    and — like the levy/tribute code — never counts a DEPENDENT-CHILD ruler (M4.3 regency), whose
    levy/tribute powers are dormant: no coin is actually taken, so no extraction is felt.
    """
    living = {a.name: a for a in state["agents"] if a.alive}

    def _extracts(ruler_name: str) -> bool:
        # A ruler only extracts if it is alive and NOT a dependent-child regent (whose levy/tribute
        # the institutions skip via the SAME is_dependent_child gate) — else there is no felt take.
        r = living.get(ruler_name)
        return r is not None and not world.is_dependent_child(r, state)

    sid = agent.settlement
    king = kingdoms.realm_of(state, sid)
    if king is not None:
        vassal = state.get("kingdoms", {}).get(king, {}).get("vassals", {}).get(sid)
        if vassal is not None and vassal != agent.name and _extracts(vassal):
            due = kingdoms.TRIBUTE_RATE * max(0.0, _wealth(agent) - kingdoms.TRIBUTE_THRESHOLD)
            return vassal, due, "tribute"
    mon = state.get("monarchs", {}).get(sid)
    if mon is not None and mon["monarch"] != agent.name and _extracts(mon["monarch"]):
        due = monarchy.MONARCH_LEVY_RATE * max(0.0, _wealth(agent) - monarchy.MONARCH_LEVY_THRESHOLD)
        return mon["monarch"], due, "levies"
    return None, 0.0, "none"


def legitimacy_factor(agent: Any, ruler: str) -> float:
    """How much `agent`'s personal TRUST in `ruler` buffers (or amplifies) extraction grievance.

    clamp(1.0 - LEGIT_SLOPE * trust, LEGIT_MIN, LEGIT_MAX): a trusted ruler pulls the factor below 1
    (grievance buffered — consent), a distrusted one above 1 (grievance amplified — resentment).
    Pure read of the v1 trust network.
    """
    trust = agent.relationships.get(ruler, {}).get("trust", 0)
    return max(LEGIT_MIN, min(LEGIT_MAX, 1.0 - LEGIT_SLOPE * trust))


def extraction(agent: Any, state: dict[str, Any]) -> tuple[float, str | None, str]:
    """EXTRACTION: grievance from being levied/tributed, weighted by BURDEN and LEGITIMACY.

    increment = EXTRACTION_WEIGHT * burden * legitimacy_factor, where burden = min(1, due /
    max(wealth, MEANS_FLOOR)) makes the same coin sting a poor agent more than a rich one, and the
    legitimacy factor makes a trusted ruler's take sting far less than a hated one's. Returns
    (increment, ruler_name_or_None, kind) so the caller can name the driver in a crossing log.
    """
    ruler, due, kind = _ruler_and_due(agent, state)
    if ruler is None or due <= 0:
        return 0.0, None, "none"
    burden = min(1.0, due / max(_wealth(agent), MEANS_FLOOR))
    return EXTRACTION_WEIGHT * burden * legitimacy_factor(agent, ruler), ruler, kind


# --- The gauge: accumulate with hysteresis, decay slowly, log crossings sparsely ---
def _crossing_event(agent: Any, turn: int, value: float, contribs: dict[str, float],
                    ruler: str | None, kind: str) -> str:
    """One sparse log line when an agent first crosses into resentment, naming the DOMINANT driver."""
    driver = max(contribs, key=contribs.get)
    if driver == "extraction" and ruler is not None:
        why = f"under {ruler}'s {kind}"
    elif driver == "deprivation":
        why = "with hunger amid plenty"
    else:
        why = "at subsistence wages"
    return f"turn {turn}: {agent.name} seethes {why} (discontent {value:.1f})"


def update(state: dict[str, Any], turn: int) -> list[str]:
    """Advance the discontent gauge one turn (ZERO LLM, ZERO RNG, M4.4). Returns events logged.

    For each settled adult (sorted): sum the three drivers into this turn's grievance G. If G > 0
    the gauge ACCUMULATES (discontent += G) with NO decay — grievance persists while its cause does;
    if G == 0 the gauge DECAYS slowly (‑DECAY, floored at 0) — relief ebbs the resentment away far
    slower than oppression built it (hysteresis). The gauge is soft-capped at DISCONTENT_CAP. An
    agent crossing RESENTMENT_THRESHOLD upward gets ONE sparse log line naming its dominant driver.

    This is a MEASURE ONLY: it never acts on a ruler (that is M4.5). Caller gates on `discontent_on`,
    so an off run never calls this — no "discontent" key is ever written — and stays byte-identical.
    """
    gauge: dict[str, float] = state.setdefault("discontent", {})
    measured = _measured(state)
    names = {a.name for a in measured}
    # Drop gauge entries for agents no longer measured (dead / left / came of age into nothing),
    # so a respawn reusing a name starts fresh and the map cannot grow without bound.
    for gone in [n for n in gauge if n not in names]:
        del gauge[gone]

    events: list[str] = []
    for agent in measured:
        d_dep = deprivation(agent, state)
        d_exp = exploitation(agent, state)
        d_ext, ruler, kind = extraction(agent, state)
        grievance = d_dep + d_exp + d_ext

        prev = gauge.get(agent.name, 0.0)
        if grievance > 0:
            new = min(DISCONTENT_CAP, prev + grievance)   # accumulate; no decay while aggrieved
        else:
            new = max(0.0, prev - DECAY)                  # slow ebb only when fully relieved
        gauge[agent.name] = new

        # Sparse legibility: log ONLY the upward crossing into resentment (not every increment).
        if prev < RESENTMENT_THRESHOLD <= new:
            contribs = {"deprivation": d_dep, "exploitation": d_exp, "extraction": d_ext}
            ev = _crossing_event(agent, turn, new, contribs, ruler, kind)
            events.append(ev)
            world.record_memory(agent, ev.split(": ", 1)[1])
    state.setdefault("events", []).extend(events)
    return events


# --- Derived read-outs (never stored; always recomputed from the gauge) ------
def agent_discontent(name: str, state: dict[str, Any]) -> float:
    """`name`'s current discontent (0.0 if untracked) — a pure read of the gauge."""
    return state.get("discontent", {}).get(name, 0.0)


def settlement_pressure(sid: str, state: dict[str, Any]) -> int:
    """The number of `sid`'s living members whose discontent has crossed RESENTMENT_THRESHOLD.

    The size of the settlement's resentful faction — DERIVED on demand from the per-agent gauge (so
    it can never drift out of sync), and the quantity M4.5 will trigger an uprising on. Pure read.
    """
    rec = state.get("settlements", {}).get(sid)
    if rec is None:
        return 0
    gauge = state.get("discontent", {})
    living = {a.name for a in state["agents"] if a.alive}
    return sum(1 for m in rec["members"]
               if m in living and gauge.get(m, 0.0) >= RESENTMENT_THRESHOLD)


def settlement_discontent(sid: str, state: dict[str, Any]) -> float:
    """Total discontent carried by `sid`'s living members — the settlement's aggregate pressure.

    A companion to the above-threshold COUNT: the summed gauge, useful for ranking a tyrant's
    settlement against a fair leader's at a glance. Pure read, recomputed from the gauge.
    """
    rec = state.get("settlements", {}).get(sid)
    if rec is None:
        return 0.0
    gauge = state.get("discontent", {})
    living = {a.name for a in state["agents"] if a.alive}
    return sum(gauge.get(m, 0.0) for m in rec["members"] if m in living)
