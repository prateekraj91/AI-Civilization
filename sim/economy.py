"""
economy.py
==========

TRADE, MONEY & the EMERGENT PRICE — V2 milestone M2.3, which CLOSES Phase 2 (Settlement &
Economy). Builds on M2.1 (settlement), M2.2 (storage/wealth), and all of Phase 0 + Phase 1.

The historical step M2.3 makes
------------------------------
M2.2 created WEALTH and inequality — some agents have far more food/skill than others.
M2.3 makes that asymmetry MOVE: agents TRADE. Trade exists precisely BECAUSE agents differ
(rich/poor, skilled/unskilled, fed/starving); this module lets those differences flow
through VOLUNTARY, mutually-beneficial exchange, at a price that EMERGES from circumstance.

The four pieces (specialization lives in knowledge.py; the rest here)
---------------------------------------------------------------------
1. SPECIALIZATION — a second producer skill, `hunting`, sits beside `farming` (knowledge.py
   TECH_TREE + knowledge.hunt). Two producer types means there is a real skill gap to trade.
2. MONEY — `agent.money`, an emergent FOOD-BACKED currency. It is minted ONLY from food
   surplus that runs past the M2.2 storage cap (see `mint`) — food produced but unstorable —
   so every unit is a claim on real food. It is redeemable as food to survive
   (storage.draw_down) and accepted in trade. It is NOT minted by an authority and NOT fiat:
   AUTHORITY-DECREED / MINTED currency is deferred to Phase 3. Money just gives a unit of
   account so trade isn't stuck on a double coincidence of wants.
3. TRADE — `trade` matches mutually-beneficial exchanges (food<->money, knowledge<->money)
   between adjacent agents. Each side must end up better off BY ITS OWN VALUATION; that is
   what separates trade from THEFT (a hostile act that already exists — untouched here).
   PRICE EMERGES from conditions already tracked: buyer desperation (hunger) pushes it up,
   seller surplus (well past cap) pushes it down, and LOCAL RARITY of a skill sets its worth
   (a skill nobody nearby knows is dear; one everybody knows is cheap). The same good/skill
   trades at DIFFERENT prices in different situations — that variation is the emergence proof.
4. PROPRIETARY KNOWLEDGE — a guarded skill (knowledge.guards, emerging from personality)
   does not diffuse free; it moves ONLY by sale here. Free M1.1 diffusion is untouched for
   non-guarding holders — trade is an ADDITIONAL path, not a replacement.

What this milestone deliberately does NOT build (deferred to Phase 3): WAGE-LABOR / hiring
workers, and MINTED / authority-decreed (fiat) currency. Money here is strictly food-backed.

Cost & determinism
------------------
ZERO LLM calls and ZERO RNG — `mint` and `trade` are deterministic state-math over a stable
iteration (world_state["agents"] order, sorted neighbours, one trade per pair per turn), so a
seeded run reproduces and an economy-OFF run is byte-identical to v1 (the loop never calls
either, knowledge.diffuse's guard-skip is gated off, and money is ignored everywhere). This
module imports only world + storage + knowledge (one-directional), keeping the world layer
dependency-free.
"""

from __future__ import annotations

from typing import Any

from sim import knowledge
from sim import storage
from sim import world

# --- Money (food-backed) ---------------------------------------------------
# 1 unit of food surplus past the storage cap mints this much money. Kept at 1.0 so money is
# denominated in the SAME food-relief units as the stockpile — a money unit is exactly a
# one-point claim on food, which is what makes it redeemable in storage.draw_down.
MONEY_PER_FOOD = 1.0

# --- Food trade ------------------------------------------------------------
# A buyer's marginal value of one food unit (in money) rises with HUNGER — a desperate buyer
# pays more. A fed buyer (hunger 0) values it at 0 and never buys. The coefficient sets how
# steep desperation is.
FOOD_VALUE_COEF = 1.0
# A seller's per-unit RESERVATION falls toward 0 as its stockpile approaches the cap: food it
# can barely keep is near-worthless to hold, so it sells cheap. Scaled by this coefficient.
FOOD_RES_COEF = 1.0
# A seller keeps at least this much food back (its own survival buffer, = one stored meal)
# before any is offered for sale — nobody trades away the food that keeps them alive.
SELLER_KEEP = storage.BUFFER_COST
# Largest food quantity moved in a single pair-trade, so exchange is gradual, not a one-shot
# wealth teleport.
FOOD_LOT = 5.0

# --- Knowledge trade -------------------------------------------------------
# The producer skills — the specializations worth buying when you lack one.
PRODUCER_SKILLS: frozenset[str] = frozenset({"farming", "hunting"})
# Base worth of a skill before rarity/need shaping (in money).
KNOW_BASE = 6.0
# A guarding seller's reservation: it won't surrender exclusivity below this (scaled by
# rarity — the rarer the skill, the more the guard holds out for).
KNOW_GUARD_RES = 2.0
# A buyer with NO producer skill values a producer skill this many times more than a buyer
# who already has one (the skill-gap that makes specialization tradable).
KNOW_NEED_NO_SKILL = 2.0
KNOW_NEED_BASE = 1.0
# Radius over which "local rarity" of a skill is measured (how many nearby already know it).
RARITY_RADIUS = 3


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _has_producer(agent: Any) -> bool:
    """True if the agent already knows a producer skill (farming or hunting)."""
    return any(s in agent.knowledge for s in PRODUCER_SKILLS)


# --- Emergent price functions (pure; shared by trade() and the verifier) ----
def food_value(buyer: Any) -> float:
    """Buyer's marginal value of one food unit, in money — rises with its hunger."""
    return FOOD_VALUE_COEF * buyer.hunger


def food_reservation(seller: Any) -> float:
    """Seller's per-unit reservation, in money — falls to 0 as its larder nears the cap."""
    headroom = max(0.0, storage.STORAGE_CAP - seller.stockpile) / storage.STORAGE_CAP
    return FOOD_RES_COEF * headroom


def food_price(buyer: Any, seller: Any) -> "float | None":
    """Emergent per-unit food price, or None if no mutually-beneficial deal exists.

    The agreed price is the MIDPOINT of the seller's reservation and the buyer's value, so it
    moves with BOTH: a hungrier buyer lifts it, a more-glutted seller lowers it. When the
    buyer values food no more than the seller's reservation there is no trade (returns None).
    """
    v = food_value(buyer)
    r = food_reservation(seller)
    if v <= r:
        return None
    return (v + r) / 2.0


def local_rarity(item: str, center: Any, state: dict[str, Any],
                 radius: int = RARITY_RADIUS, exclude: tuple = ()) -> float:
    """Fraction of OTHER agents near `center` that do NOT know `item` (1.0 = none do).

    The LOCAL scarcity that prices a skill: a skill only the seller holds nearby is dear; one
    everyone around already knows is cheap. Measured over living agents within `radius`,
    excluding `center` itself AND any `exclude` agents (the prospective seller is excluded so
    that "the seller knows it" never, by itself, makes the skill look common). An empty
    neighbourhood -> maximal rarity 1.0.
    """
    cx, cy = center.position
    skip = (center,) + tuple(exclude)
    nearby = [a for a in state["agents"]
              if a.alive and a not in skip and _chebyshev(a.position, (cx, cy)) <= radius]
    if not nearby:
        return 1.0
    knowers = sum(1 for a in nearby if item in a.knowledge)
    return 1.0 - knowers / len(nearby)


def knowledge_value(item: str, buyer: Any, rarity: float) -> float:
    """Buyer's value of learning `item`, in money — scales with local rarity and skill-gap."""
    need = (KNOW_NEED_NO_SKILL
            if (item in PRODUCER_SKILLS and not _has_producer(buyer))
            else KNOW_NEED_BASE)
    return KNOW_BASE * rarity * need


def knowledge_reservation(item: str, rarity: float) -> float:
    """Guarding seller's reservation for `item`, in money — a FIXED floor on exclusivity.

    Deliberately NOT scaled by rarity: a guard won't surrender know-how for near-nothing
    however common it is, while the BUYER's value DOES fall with rarity (knowledge_value). So
    as a skill spreads its value sinks toward this floor and eventually below it — at which
    point knowledge_price returns None (it is too common to be worth the guard's price). That
    crossing is what makes a common skill genuinely un-sellable, not merely cheap.
    """
    return KNOW_GUARD_RES


def knowledge_price(item: str, buyer: Any, rarity: float) -> "float | None":
    """Emergent price to buy `item`, or None if no mutually-beneficial deal exists.

    Midpoint of the guard's reservation and the buyer's value, so the SAME skill costs more
    where it is rare / the buyer badly lacks a producer skill, and little where it is common
    or the buyer already has one — and nothing at all (None) when it is too common to be worth
    the guard's price. This is the emergent price the milestone turns on.
    """
    v = knowledge_value(item, buyer, rarity)
    r = knowledge_reservation(item, rarity)
    if v <= r:
        return None
    return (v + r) / 2.0


# --- Money minting (from food surplus past the cap) ------------------------
def mint(state: dict[str, Any], turn: int) -> list[tuple[str, float]]:
    """Convert each settled agent's food surplus PAST the storage cap into money (M2.3).

    Same qualifying condition as storage.accumulate (settled + well-fed + beside food), but it
    fires only once the stockpile is already FULL (>= STORAGE_CAP): the food the agent keeps
    producing then can't be stored, so instead of being wasted it becomes MONEY — a claim on
    that food. This is the sole source of new money in the sim (it is FOOD-BACKED, never
    minted by decree). ZERO LLM, ZERO RNG. Returns the (name, minted) pairs.
    """
    food = set(state["food"])
    minted: list[tuple[str, float]] = []
    for a in state["agents"]:  # stable world_state order
        if not a.alive or a.settlement is None:
            continue
        if a.hunger > storage.STORE_HUNGER_MAX:
            continue
        if a.stockpile < storage.STORAGE_CAP:
            continue  # only the overflow past a FULL larder becomes money
        if not storage._near_food(a.position, food, storage.STORE_RADIUS):
            continue
        amount = storage.banking_rate(a) * MONEY_PER_FOOD
        a.money += amount
        minted.append((a.name, amount))
    return minted


# --- Payment settlement ----------------------------------------------------
def _settle(buyer: Any, seller: Any, amount: float) -> None:
    """Move `amount` of food-claim value from buyer to seller: money first, then stored food.

    Money and stockpile are both food-claims, so a payer covers a price from whichever it has;
    a payee credited in food (because the buyer had no money) can eat it. Callers ensure the
    buyer can afford `amount` (money + stockpile >= amount).
    """
    pay_money = min(amount, buyer.money)
    buyer.money -= pay_money
    seller.money += pay_money
    rem = amount - pay_money
    pay_food = min(rem, buyer.stockpile)
    buyer.stockpile -= pay_food
    seller.stockpile += pay_food


# --- Candidate trades (each returns (gain, apply) or None) -----------------
def _eval_food(seller: Any, buyer: Any, state: dict[str, Any], turn: int):
    """A food sale from `seller` (surplus) to `buyer` (hungry), paid in money."""
    surplus = seller.stockpile - SELLER_KEEP
    if surplus <= 0:
        return None
    price = food_price(buyer, seller)
    if price is None:
        return None
    qty = min(FOOD_LOT, surplus, buyer.money / price)  # food trades are money-only
    if qty <= 0:
        return None
    cost = price * qty
    gain = (food_value(buyer) - food_reservation(seller)) * qty  # total mutual surplus

    def apply() -> str:
        seller.stockpile -= qty
        buyer.stockpile += qty
        buyer.money -= cost
        seller.money += cost
        msg = (f"turn {turn}: {seller.name} sold {qty:.1f} food to {buyer.name} "
               f"for {cost:.1f} money (price {price:.2f}/unit)")
        world.record_memory(seller, f"Sold {qty:.1f} food to {buyer.name} for {cost:.1f}")
        world.record_memory(buyer, f"Bought {qty:.1f} food from {seller.name} for {cost:.1f}")
        state["events"].append(msg)
        return msg

    return gain, apply


def _eval_knowledge(seller: Any, buyer: Any, state: dict[str, Any], turn: int):
    """Sale of a GUARDED skill from `seller` to a `buyer` who lacks it (the proprietary path).

    Only a skill the seller GUARDS (knowledge.guards — emerges from personality) is sold; a
    freely-taught skill spreads via M1.1 diffusion instead. Picks the seller's most valuable
    sellable skill for this buyer.
    """
    best = None
    for item in sorted(seller.knowledge):
        if item in buyer.knowledge or not knowledge.guards(seller, item):
            continue
        rarity = local_rarity(item, buyer, state, exclude=(seller,))
        price = knowledge_price(item, buyer, rarity)
        if price is None:
            continue
        if buyer.money + buyer.stockpile < price:  # buyer can't afford it
            continue
        gain = knowledge_value(item, buyer, rarity) - knowledge_reservation(item, rarity)
        if best is None or gain > best[0]:
            best = (gain, item, price)
    if best is None:
        return None
    gain, item, price = best

    def apply() -> str:
        buyer.knowledge.add(item)
        _settle(buyer, seller, price)
        msg = (f"turn {turn}: {seller.name} sold knowledge '{item}' to {buyer.name} "
               f"for {price:.1f} money")
        world.record_memory(seller, f"Sold '{item}' to {buyer.name} for {price:.1f}")
        world.record_memory(buyer, f"Bought '{item}' from {seller.name} for {price:.1f}")
        state["events"].append(msg)
        return msg

    return gain, apply


def _best_trade(a: Any, b: Any, state: dict[str, Any], turn: int):
    """Best mutually-beneficial trade between adjacent `a` and `b`, in either direction."""
    candidates = [
        _eval_food(a, b, state, turn), _eval_food(b, a, state, turn),
        _eval_knowledge(a, b, state, turn), _eval_knowledge(b, a, state, turn),
    ]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c[0])  # highest mutual surplus wins


def trade(state: dict[str, Any], turn: int) -> list[str]:
    """One deterministic pass of voluntary exchange across adjacent agent pairs (M2.3).

    For each adjacent pair (processed once, in stable order) the single best mutually-
    beneficial trade — food<->money or guarded-knowledge<->money — is executed, at the
    emergent price. Both sides end up better off by their own valuation (the price sits
    strictly between the seller's reservation and the buyer's value), which is exactly what
    distinguishes trade from theft. ZERO LLM, ZERO RNG. Returns the event strings logged.

    Caller gates invocation on the `economy` flag (run_simulation), so an economy-off run
    never calls this and stays byte-identical to v1.
    """
    # M4.1: dependent children don't trade — full economic agency waits for maturity
    # (the filter is always a no-op when lineage is off -> byte-identical).
    living = [a for a in state["agents"]
              if a.alive and not world.is_dependent_child(a, state)]
    seen: set[tuple[str, str]] = set()
    done: list[str] = []
    for a in living:  # world_state["agents"] order is stable
        neighbours = world.adjacent_agents(a, state)
        for bname in sorted(neighbours):  # deterministic pairing
            b = neighbours[bname]
            key = (a.name, bname) if a.name < bname else (bname, a.name)
            if key in seen:
                continue
            seen.add(key)
            best = _best_trade(a, b, state, turn)
            if best is not None:
                done.append(best[1]())  # apply() -> event string
    return done
