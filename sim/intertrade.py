"""
intertrade.py
=============

TRADE ROUTES & INTERDEPENDENCE — commerce and the peace it may buy (V2 milestone M4.14, Arc 5:
Diplomacy). On top of M4.13 (relations/treaties), Arc 4 (eras), Arc 3 (culture), Arc 2 (revolt),
Arc 1 (dynasties) and all of Phases 0-3.

The historical step M4.14 makes — kingdoms gain a THIRD verb: TRADE
------------------------------------------------------------------
M4.13 gave kingdoms stance + treaties. M4.14 adds TRADE across borders: two peaceful kingdoms with
COMPLEMENTARY surplus and need trade — a food-rich realm sells to a food-poor one at an emergent
price, and BOTH gain (the buyer eats, the seller profits). Trade WARMS the pair's stance (closing the
feedback seam M4.13 left), so commerce can push neutral kingdoms into a pact. And WAR SEVERS trade —
an economic cost of war beyond casualties. Together these let the sim TEST the commercial-peace
hypothesis: do interdependent kingdoms fight less? The answer must EMERGE from the warming + severance
feedback, never be designed — and be reported honestly whichever way it falls.

SCOPE — M4.14 is inter-kingdom TRADE + trade->stance warming + war-severs-trade + the interdependence
MEASUREMENT. It does NOT build anti-hegemon COALITIONS (M4.15) — stated as a boundary. It reuses the
M2.3 price machinery (`economy.food_value`/`food_reservation` and the midpoint rule) lifted to the
realm level; ZERO LLM.

How it works (emergent; zero LLM; deterministic price math)
-----------------------------------------------------------
1. TRADE. Each turn, for every pair of NEIGHBOURING sovereign kingdoms whose stance is not hostile, if
   one realm holds exportable FOOD (its crown's stores) and the other has NEED (hungry members), the
   surplus crown SELLS food to the needy crown at the M2.3 emergent price (midpoint of the buyer's
   value and the seller's reservation). Food flows to the buyer's granary, money to the seller — both
   better off. The per-pair TRADE VOLUME is recorded, and the pair marked an active ROUTE.
2. WARMING (feedback into M4.13). An active route warms the pair's stance in `diplomacy._recompute_
   stance` (it reads this module's routes), like shared kinship does — so sustained commerce trends a
   pair toward friendly / a pact.
3. SEVERANCE. A route that goes hostile (a war, or stance souring) STOPS — logged distinctly; both
   crowns lose the flow the route was generating. War now costs more than blood.
4. MEASUREMENT. Cumulative volume per pair (here) and war count per pair (`diplomacy.war_count`) are
   exposed so a run can COMPARE war frequency among heavily-trading vs isolated pairs — an emergent
   read-out, never a scripted outcome.

Cost & determinism
------------------
ZERO LLM and ZERO RNG — trade is deterministic price math over sorted neighbour pairs. A run with the
system OFF never calls `update` (no "intertrade" key, no route to warm, no severance), so it is
byte-identical to v1. Imports world + economy (the M2.3 price machinery); lazily imports empire (the
kingdom-neighbour read) and diplomacy (stance) — no load-time cycle.
"""

from __future__ import annotations

from typing import Any

from sim import economy
from sim import world

# --- Tunable constants -------------------------------------------------------
INTERTRADE_QTY = 2.0     # food units a crown will move across a route per turn (the caravan size)


def _find(state: dict[str, Any], name: str) -> "Any | None":
    return next((a for a in state["agents"] if a.name == name), None)


def _inter(state: dict[str, Any]) -> dict[str, Any]:
    return state.setdefault("intertrade", {"volume": {}, "routes": set()})


def _realm_need(state: dict[str, Any], king_name: str) -> float:
    """A realm's food NEED: the hunger of its hungriest living member (its most desperate mouth) — the
    buyer's marginal value of imported food. 0 for a well-fed realm. Pure read."""
    rec = state.get("kingdoms", {}).get(king_name)
    if rec is None:
        return 0.0
    living = {a.name: a for a in state["agents"] if a.alive}
    hunger = [living[m].hunger for sid in rec["settlements"]
              for m in state.get("settlements", {}).get(sid, {}).get("members", ())
              if m in living]
    return float(max(hunger)) if hunger else 0.0


def _export_food(king: Any) -> float:
    """A crown's exportable food = the treasury (its own stockpile), built from tribute/levy. Pure read."""
    return king.stockpile


def total_volume(state: dict[str, Any], k1: str, k2: str) -> float:
    """Cumulative trade value that has flowed across this pair's route — the interdependence metric."""
    return _inter(state)["volume"].get(tuple(sorted((k1, k2))), 0.0)


def _price(state: dict[str, Any], buyer: str, seller_agent: Any) -> "float | None":
    """The emergent per-unit food price between two realms, reusing the M2.3 midpoint rule at realm
    scale: midpoint of the buyer realm's VALUE (its need) and the seller crown's RESERVATION (falls as
    its larder fills). None when no mutually-beneficial deal exists (the buyer values food no more than
    the seller's reservation)."""
    v = economy.FOOD_VALUE_COEF * _realm_need(state, buyer)
    r = economy.food_reservation(seller_agent)
    return (v + r) / 2.0 if v > r else None


def update(state: dict[str, Any], turn: int) -> list[str]:
    """Advance inter-kingdom trade one turn (M4.14): move food/wealth across every eligible route, record
    volume + routes (which `diplomacy` reads to warm the pair), and log route openings/severances. ZERO
    LLM, ZERO RNG. Runs BEFORE diplomacy.update so this turn's routes warm this turn's stance. Caller gates
    on `intertrade_on`, so an off run never calls this and stays byte-identical. Returns events."""
    from sim import diplomacy
    from sim import empire

    inter = _inter(state)
    prev_routes = inter["routes"]
    active: set = set()
    events: list[str] = []

    kings = [k for k in sorted(state.get("kingdoms", {}))
             if empire.is_sovereign(state, k) and _find(state, k) is not None]
    seen: set = set()
    for k1 in kings:
        for k2 in empire._kingdom_neighbours(state, k1):
            key = tuple(sorted((k1, k2)))
            if key in seen or k2 not in kings:
                continue
            seen.add(key)
            if diplomacy.stance(state, key[0], key[1]) == "hostile":
                continue  # war / hostility blocks commerce
            a1, a2 = _find(state, key[0]), _find(state, key[1])
            if a1 is None or a2 is None:
                continue
            # The crown with more exportable food SELLS; the other BUYS (complementary surplus/need).
            seller, buyer = (a1, a2) if _export_food(a1) >= _export_food(a2) else (a2, a1)
            price = _price(state, buyer.name, seller)
            if price is None or price <= 0:
                continue
            qty = min(INTERTRADE_QTY, _export_food(seller), buyer.money / price)
            if qty <= 0:
                continue
            cost = price * qty
            seller.stockpile -= qty
            buyer.stockpile += qty          # the needy realm's granary fills...
            buyer.money -= cost
            seller.money += cost            # ...and the surplus realm's treasury profits
            inter["volume"][key] = inter["volume"].get(key, 0.0) + cost
            active.add(key)
            if key not in prev_routes:
                events.append(f"turn {turn}: a TRADE ROUTE opened between {key[0]} and {key[1]} "
                              f"({seller.name} sells food to {buyer.name})")
            world.record_memory(seller, f"Sold food to {buyer.name}'s realm across the border")
            world.record_memory(buyer, f"Bought food from {seller.name}'s realm to feed my people")

    # Severance: a route that was active last turn but is not now (a war / hostility cut it).
    for key in sorted(prev_routes - active):
        if inter["volume"].get(key, 0.0) > 0:
            events.append(f"turn {turn}: the trade route between {key[0]} and {key[1]} was SEVERED "
                          f"— both realms lose the commerce (war costs more than blood)")
    inter["routes"] = active
    state.setdefault("events", []).extend(events)
    return events
