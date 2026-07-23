"""
storage.py
==========

STORAGE & SURPLUS — V2 milestone M2.2, the moment the simulation grows WEALTH.
Builds on M2.1 (settlement) and the whole of Phase 0 + Phase 1's food economy.

The historical step M2.2 makes
------------------------------
M1.3 made farming PRODUCE food; M2.1 made nomads SETTLE around it. In real history
farming's deeper unlock was not just more food but food you could KEEP — a storable
SURPLUS. Stored surplus is what first created wealth, a survival buffer, and (much
later) inequality and trade. M2.2 introduces a PERSONAL stockpile per agent and makes
that wealth a survival buffer: a hungry agent with savings draws on them to live.

What is (and isn't) in scope
----------------------------
ONLY personal storage + surplus accumulation + a survival-buffer effect + wealth
inequality that EMERGES. NO trade/exchange (that is M2.3), NO markets, prices, or
specialization. Wealth is strictly per-agent (a village's wealth, if ever needed, is
just the sum of its members).

Two rules carry the whole milestone
------------------------------------
1. STORING REQUIRES SETTLEMENT. Only a SETTLED agent (agent.settlement is not None)
   banks surplus; a nomad stores nothing. Settlement is what makes durable storage
   possible — this reinforces M2.1's value (a place worth staying at is also a place
   worth keeping a granary). The accumulate pass simply skips nomads.

2. WEALTH IS NEVER ASSIGNED — IT EMERGES. How fast an agent fills its stockpile is a
   product of traits it ALREADY has:
     - PERSONALITY (STORE_PROPENSITY): an independent/competitive (wealth-goal) agent
       hoards; a friendly/sharing one banks far less; cautious/curious sit between.
     - KNOWLEDGE (farming): a farmer PRODUCES food, so it has surplus to bank far more
       often than a non-farmer — it accumulates faster.
   There is no new "richness" stat sprinkled on; wealth VARIES across agents purely as
   a consequence of who they are and what they know (see verify_m22 HEADLINE 1).

3. WEALTH IS A SURVIVAL BUFFER (the immediate consequence). When a member would
   otherwise STARVE (hunger critical, no reachable food) it DRAWS DOWN its stockpile to
   survive instead of dying. In a food shock (god_mode drought) the wealthy weather it
   and the poor die — wealth MATTERS (verify_m22 HEADLINE 2). draw_down is wired into
   the existing starvation step in main.run_agent_turn.

Cost & determinism
------------------
ZERO LLM calls and ZERO RNG: accumulation is pure float state-math on a deterministic
condition (settled + well-fed + beside surplus food), so it is reproducible and a run
with the system OFF is byte-identical to v1 (the loop simply never calls accumulate, and
draw_down is gated on world_state["storage_on"]). Iteration is world_state["agents"]
order (stable). storage.py imports only `world` + `strategy` (one-directional), so the
world layer stays dependency-free.
"""

from __future__ import annotations

from typing import Any

from sim import world
from llm.strategy import get_personality

# --- Tunable constants (documented) ----------------------------------------
# STORAGE_CAP: the ANTI-INFINITY lever and the key tuning knob. A stockpile never
# exceeds this many stored food-units (each unit is one point of future hunger relief,
# so the cap is ~STORAGE_CAP/BUFFER_RELIEF stored meals). It bounds wealth so a hoarder
# plateaus at a finite granary rather than accumulating without limit — and it is what
# makes the rich/poor GAP finite and legible (a hoarder caps out; the poor never get
# near it). Surplus gathered past the cap is simply not stored (no spoilage modelled —
# kept simple). At 20.0 ≈ ~3 stored meals: enough to weather a multi-turn drought, not
# enough to make a settlement immortal.
STORAGE_CAP = 20.0

# STORE_RATE: base food-units banked per QUALIFYING turn (a settled, well-fed agent
# standing by surplus food), before the personality + knowledge multipliers below. Small
# ON PURPOSE: paired with STORAGE_CAP it sets how fast the cap binds. Tuned (0.2) so that
# over a settled life (~30-40 qualifying turns) only genuine HOARDERS (a competitive
# farmer at rate*1.5*2 ≈ 0.6/turn) approach the cap, while the poorest (a friendly
# non-farmer at rate*0.5 ≈ 0.1/turn) accrue a thin reserve — i.e. wealth stays a visible
# GRADIENT across traits rather than everyone saturating the cap and inequality vanishing.
STORE_RATE = 0.2

# STORE_PROPENSITY: the PERSONALITY half of emergent inequality — a multiplier on the
# banking rate keyed by the agent's dominant trait. An independent/competitive agent (the
# wealth-goal hoarder) banks the most; a friendly/sharing one the least; cautious and
# curious sit between. This is read from the SAME Personality the rest of the sim uses —
# never a new per-agent number — so wealth tracks character, not an assignment.
STORE_PROPENSITY: dict[str, float] = {
    "independence": 1.5,   # competitive hoarder — puts surplus away first
    "caution": 1.2,        # territorial saver — keeps a reserve
    "curiosity": 1.0,      # neutral
    "friendliness": 0.5,   # shares/consumes — banks the least
}

# FARMING_BONUS: the KNOWLEDGE half — a farmer PRODUCES food, so it has surplus to bank
# far more often. Multiplies the banking rate for an agent that knows 'farming'. A
# non-farmer accrues at the personality rate alone, so a farmer of the same character is
# reliably richer (and a competitive farmer is the richest agent of all).
FARMING_BONUS = 2.0

# STORE_HUNGER_MAX: a settled agent only banks when its OWN immediate need is met — i.e.
# it is well-fed (hunger <= this). This is the "beyond its immediate hunger need" rule:
# you store the EXCESS, never food you needed to eat now. Below SURVIVAL_HUNGER, so a
# hungry/foraging agent never banks.
STORE_HUNGER_MAX = 3

# STORE_RADIUS: a settled agent must be within this Chebyshev distance of standing food to
# have a surplus to store (you can only bank food you can actually reach). Equal to the
# settlement footprint (CLUSTER_RADIUS/HOME_RADIUS = 2), so a member ranging over its own
# village beside its farm qualifies.
STORE_RADIUS = 2

# --- Survival buffer constants ---------------------------------------------
# BUFFER_RELIEF: how much hunger one drawn-down stored meal removes when an agent is at
# the brink of starvation. Matches a foraged meal (world.EAT_RELIEF = 7) — your granary
# feeds you as well as fresh food would.
BUFFER_RELIEF = 7

# BUFFER_COST: stored food-units consumed to buy that relief. Equal to BUFFER_RELIEF, so a
# stockpile is denominated directly in "hunger relief" — STORAGE_CAP/BUFFER_COST stored
# meals. An agent must hold at least this much to draw a meal: a hoarder (near the cap) can
# weather several drought turns; an agent with thin savings (< one meal) cannot buy even
# one and starves — which is exactly what splits survival by wealth in a food shock.
BUFFER_COST = 7.0


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Chebyshev (king-move) distance — the natural radius on a square grid."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _near_food(pos: tuple[int, int], food: set[tuple[int, int]], radius: int) -> bool:
    """True if any food tile sits within Chebyshev `radius` of `pos` (pure read)."""
    x, y = pos
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if (x + dx, y + dy) in food:
                return True
    return False


def banking_rate(agent: Any) -> float:
    """The food-units `agent` banks on a qualifying turn — its emergent wealth velocity.

    A pure product of traits the agent ALREADY has: the base STORE_RATE, its PERSONALITY
    propensity (hoarder vs sharer), and a FARMING_BONUS if it KNOWS farming (a producer
    has more surplus). No RNG, no assigned number — this is the whole of "wealth emerges
    from who you are and what you know", in one line. Exposed so verify_m22 can score the
    expected ordering against the observed stockpiles.
    """
    pers = get_personality(agent)
    rate = STORE_RATE * STORE_PROPENSITY.get(pers.dominant, 1.0)
    if "farming" in getattr(agent, "knowledge", ()):
        rate *= FARMING_BONUS
    return rate


def accumulate(state: dict[str, Any], turn: int) -> list[tuple[str, float]]:
    """Bank surplus into settled agents' personal stockpiles for this turn (M2.2).

    ZERO LLM, ZERO RNG. One deterministic pass over the living agents. An agent banks
    `banking_rate(agent)` units (capped at STORAGE_CAP) when ALL of:
      - it is SETTLED (agent.settlement is not None) — nomads store nothing (rule 1);
      - it is WELL-FED (hunger <= STORE_HUNGER_MAX) — it stores the EXCESS beyond its own
        immediate need, never food it needed to eat now;
      - it is BESIDE standing food (within STORE_RADIUS) — there is a surplus to store.

    Caller gates invocation on the `storage` flag (run_simulation), so a run with the
    system off never calls this and stays byte-identical to v1. Returns the (name, banked)
    pairs for logging/inspection.
    """
    food = set(state["food"])
    banked: list[tuple[str, float]] = []
    for a in state["agents"]:  # stable world_state order
        if not a.alive or a.settlement is None:
            continue  # rule 1: storing requires settlement
        if world.is_dependent_child(a, state):
            continue  # M4.1: a dependent child gathers/banks nothing (no production)
        if a.hunger > STORE_HUNGER_MAX:
            continue  # not yet fed — its own need comes first, no surplus to bank
        if not _near_food(a.position, food, STORE_RADIUS):
            continue  # no reachable food -> nothing to gather beyond need
        if a.stockpile >= STORAGE_CAP:
            continue  # granary full (cap holds) — surplus past the cap is not stored
        amount = min(banking_rate(a), STORAGE_CAP - a.stockpile)
        a.stockpile += amount
        banked.append((a.name, amount))
    return banked


def draw_down(agent: Any) -> bool:
    """Spend one stored meal to pull `agent` back from starvation (the survival buffer).

    Called from the existing starvation step (main.run_agent_turn) when a member would
    otherwise die with no reachable food. If it can cover BUFFER_COST it consumes that,
    drops its hunger by BUFFER_RELIEF (off the brink), and returns True — it SURVIVES on its
    savings. With thin savings it cannot and returns False — it starves. This is the whole of
    "wealth buffers survival": the rich weather a drought, the poor do not. ZERO RNG.

    M2.3: when the economy is on, MONEY is a food-claim and so is redeemable here too — the
    meal is paid from stored food first, then from money. So an agent that converted its
    surplus to money (or earned money by selling food/knowledge) can still eat in a famine.
    With the economy off, money is ignored and this behaves exactly as in M2.2 (byte-identical).
    """
    economy_on = world.world_state.get("economy_on", False)
    funds = agent.stockpile + (agent.money if economy_on else 0.0)
    if funds < BUFFER_COST:
        return False
    # Spend stored food first, then redeem money (only when the economy backs it).
    from_food = min(agent.stockpile, BUFFER_COST)
    agent.stockpile -= from_food
    if economy_on:
        agent.money -= (BUFFER_COST - from_food)
    agent.hunger = max(0, agent.hunger - BUFFER_RELIEF)
    return True
