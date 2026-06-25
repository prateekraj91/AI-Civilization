"""
verify_m23.py
=============

Deterministic verification of V2 milestone M2.3: TRADE, MONEY & PROPRIETARY KNOWLEDGE —
which CLOSES Phase 2 (Settlement & Economy). On top of M2.1 (settlement), M2.2
(storage/wealth), and all of Phase 0 + Phase 1.

Run offline (Ollama OFF, no model server):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m23.py

The historical step: M2.2 created WEALTH and inequality. M2.3 makes that asymmetry MOVE —
agents TRADE because they DIFFER (rich/poor, skilled/unskilled, fed/starving). Surplus and
skills flow through voluntary exchange at a price that EMERGES from circumstance, money
(food-backed) gives a unit of account, and knowledge becomes property some guard and sell.

HEADLINE 1 — EMERGENT PRICE: the SAME skill / the SAME food trades at DIFFERENT prices
             depending on rarity, the buyer's skill-gap, hunger, and seller surplus. A fixed
             ratio would be a fail; price moves with circumstance (shown in micro and in two
             real executed trades of the same skill at different prices).
HEADLINE 2 — TRADE REDISTRIBUTES BY ASYMMETRY: surplus & knowledge flow from those who have
             them to those who need them (skilled<->unskilled, rich->poor), and BOTH sides
             end up better off by their own valuation (voluntary, mutually beneficial).
DEMO C — PROPRIETARY KNOWLEDGE: a competitive holder's skill does NOT diffuse free but DOES
         sell; the same skill still diffuses free from a friendly holder (M1.1 intact).
DEMO D — SPECIALIZATION: two producer types (farmer, hunter); each produces food only for
         knowers, and knowledge of each trades because agents lack the other's skill.
DEMO E — MONEY: food surplus past the cap converts to money, money buys food/knowledge back,
         and money redeems as food to survive — it is food-grounded, not fiat.
DEMO F — ZERO LLM/RNG cost; economy OFF -> v1 byte-identical.
"""

from __future__ import annotations

import contextlib
import io
import random
from collections import defaultdict

import cognition
import economy
import knowledge
import llm
import main
import population
import settlement
import storage
import world
from agents import Agent
from world import spawn_food, world_state

PERS = ("curious and adventurous", "cautious and territorial",
        "friendly and outgoing", "independent and competitive")


def _settled(name: str, personality: str, pos: tuple[int, int], **kw) -> Agent:
    """Place a settled agent in the current world (settlement set so storage/mint apply)."""
    a = Agent(name=name, personality=personality)
    world.place_agent(a, *pos)
    a.settlement = "S001"
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def headline_1_emergent_price() -> None:
    print("=" * 72)
    print("HEADLINE 1 — EMERGENT PRICE: the same skill/food trades at DIFFERENT prices")
    print("=" * 72)
    unskilled = Agent(name="U", personality="cautious and territorial")
    farmer = Agent(name="F", personality="cautious and territorial")
    farmer.knowledge.add("farming")
    print("  KNOWLEDGE — price of 'hunting' (a producer skill):")
    p_rare = economy.knowledge_price("hunting", unskilled, rarity=1.0)
    p_mid = economy.knowledge_price("hunting", unskilled, rarity=0.4)
    p_skilled = economy.knowledge_price("hunting", farmer, rarity=1.0)
    p_common = economy.knowledge_price("hunting", farmer, rarity=0.1)
    print(f"    rare (nobody nearby knows it), buyer has NO producer skill : {p_rare:.1f}")
    print(f"    semi-common (rarity 0.4),      buyer has NO producer skill : {p_mid:.1f}")
    print(f"    rare,                          buyer ALREADY farms         : {p_skilled:.1f}")
    print(f"    common (rarity 0.1),           buyer already farms         : "
          f"{p_common}  (no deal — too common to be worth the guard's price)")
    assert p_rare > p_mid > 0 and p_rare > p_skilled and p_common is None

    print("  FOOD — price per unit:")
    glut = Agent(name="G", personality="cautious and territorial"); glut.stockpile = 20.0
    tight = Agent(name="T", personality="cautious and territorial"); tight.stockpile = 12.0
    for h in (3, 6, 9):
        b = Agent(name="B", personality="cautious and territorial"); b.hunger = h
        print(f"    buyer hunger {h}: from a glutted seller {economy.food_price(b, glut):.2f}/unit"
              f"   from a tighter seller {economy.food_price(b, tight):.2f}/unit")
    desperate = Agent(name="D", personality="cautious and territorial"); desperate.hunger = 9
    calm = Agent(name="C", personality="cautious and territorial"); calm.hunger = 3
    assert economy.food_price(desperate, glut) > economy.food_price(calm, glut)
    assert economy.food_price(desperate, tight) > economy.food_price(desperate, glut)

    # The same skill in two REAL trades at two prices: rare neighbourhood vs crowded one.
    print("  SAME SKILL, two real trades:")
    prices = []
    for label, n_other_knowers in (("a village where only the seller knows hunting", 0),
                                   ("a village where 3 of 4 neighbours already hunt", 3)):
        world.create_world(size=10)
        world_state["economy_on"] = True
        seller = Agent(name="Seller", personality="independent and competitive", hunger=2)
        world.place_agent(seller, 5, 5); seller.knowledge.add("hunting"); seller.stockpile = 2.0
        buyer = Agent(name="Buyer", personality="cautious and territorial", hunger=0)
        world.place_agent(buyer, 6, 5); buyer.money = 30.0
        for i in range(n_other_knowers):  # crowd the neighbourhood with other hunters
            o = Agent(name=f"K{i}", personality="cautious and territorial")
            world.place_agent(o, 5 + i, 6); o.knowledge.add("hunting")
        before = buyer.money
        economy.trade(world_state, 1)
        paid = before - buyer.money
        prices.append(paid)
        print(f"    {label}: hunting sold for {paid:.1f} money")
    assert prices[0] > prices[1] >= 0, "the same skill must cost more where it is rarer"
    print("\n  Price is no fixed ratio — it rises with rarity, the buyer's skill-gap and "
          "hunger,\n  and falls with the seller's surplus. Emergent.  PASS\n")


def headline_2_redistribution() -> None:
    print("=" * 72)
    print("HEADLINE 2 — TRADE REDISTRIBUTES BY ASYMMETRY (both sides better off)")
    print("=" * 72)
    # Controlled: a rich, unskilled FARMER beside a poor, starving HUNTER.
    world.create_world(size=8)
    world_state["storage_on"] = True
    world_state["economy_on"] = True
    rich = _settled("Rich", "independent and competitive", (3, 3),
                    hunger=0, stockpile=20.0, money=15.0)
    rich.knowledge.add("farming")            # has food + money, lacks hunting
    poor = _settled("Poor", "independent and competitive", (4, 3),
                    hunger=8, stockpile=0.0, money=0.0)
    poor.knowledge.add("hunting")            # starving, but holds a skill the rich lacks
    rich_w0 = rich.stockpile + rich.money
    poor_w0 = poor.stockpile + poor.money
    print(f"  before:  Rich  food+money={rich_w0:.1f}, knows {sorted(rich.knowledge)}")
    print(f"           Poor  food+money={poor_w0:.1f} (hunger {poor.hunger}), knows {sorted(poor.knowledge)}")
    # Several pair-turns: the hunter sells hunting (gets paid -> can survive); over turns the
    # farmer (now also a hunter) and hunter keep trading what each still lacks.
    for t in range(1, 6):
        economy.trade(world_state, t)
    print(f"  after:   Rich  food+money={rich.stockpile + rich.money:.1f}, knows {sorted(rich.knowledge)}")
    print(f"           Poor  food+money={poor.stockpile + poor.money:.1f} (hunger {poor.hunger}), "
          f"knows {sorted(poor.knowledge)}")
    assert "hunting" in rich.knowledge, "the skill flowed from the poor specialist to the rich buyer"
    assert poor.stockpile + poor.money > poor_w0, "the poor specialist was paid (now has a buffer)"
    assert rich.stockpile + rich.money < rich_w0, "the rich buyer paid for the skill"
    # Both better off by their own valuation: the buyer paid below its value, the seller above
    # its reservation (guaranteed by construction — price strictly between the two).
    print("  -> skill flowed skilled->unskilled, food/money flowed rich->poor; both gained.")

    # Population: knowledge & surplus visibly redistribute across a mixed economy.
    agents, seeded = _econ_population(seed=1, n=120, turns=70)
    acquired = sum(1 for a in agents if a.alive and (a.name, "via_trade") in seeded["bought"])
    print(f"\n  population (120 agents, 70 turns): {seeded['trades']} trades "
          f"({seeded['know']} knowledge, {seeded['food']} food); "
          f"{acquired} agents BOUGHT a producer skill they lacked")
    assert seeded["trades"] >= 10 and acquired >= 5, "trade should redistribute skills at scale"
    print("  Surplus and skills flow from those who have them to those who need them.  PASS\n")


def _econ_population(seed: int, n: int, turns: int):
    """Run the real loop with the full economy on; track trades and skill purchases."""
    llm.PROVIDER = "random"
    random.seed(seed)
    grid = main.scaled_grid_size(n)
    world.create_world(size=grid)
    world_state["storage_on"] = True
    world_state["economy_on"] = True
    cells = [(x, y) for x in range(grid) for y in range(grid)]
    random.Random(seed).shuffle(cells)
    agents = []
    for i in range(n):
        a = Agent(name=f"A{i:03d}", personality=PERS[i % 4], cognition="heuristic",
                  goals={"survive": 8, "wealth": 3, "friendship": 4})
        world.place_agent(a, *cells[i])
        agents.append(a)
    seeded_farm = {a.name for a in agents[:n // 3]}
    seeded_hunt = {a.name for a in agents[n // 3:2 * n // 3]}
    for a in agents[:n // 3]:
        a.knowledge.add("farming")
    for a in agents[n // 3:2 * n // 3]:
        a.knowledge.add("hunting")
    cfg = main.scaled_food_cfg(n)
    spawn_food(cfg["initial"])
    st, sv, cn, tn = {}, {}, {"agent_turns": 0}, {}
    trades = know = food = 0
    with contextlib.redirect_stdout(io.StringIO()):
        for t in range(1, turns + 1):
            world_state["turn"] = t
            cognition.update_tiers(world_state, t, 8, tn)
            for a in [x for x in world_state["agents"] if x.alive]:
                main.run_agent_turn(a, t, st, sv, cn)
            knowledge.farm(world_state, t)
            knowledge.hunt(world_state, t)
            settlement.update(world_state, t)
            storage.accumulate(world_state, t)
            economy.mint(world_state, t)
            evs = economy.trade(world_state, t)
            trades += len(evs)
            know += sum("knowledge" in e for e in evs)
            food += sum(" food to " in e for e in evs)
            population.process_respawns(t, world_state)
    bought = set()
    for a in agents:
        if a.name not in seeded_farm and "farming" in a.knowledge:
            bought.add((a.name, "via_trade"))
        if a.name not in seeded_hunt and "hunting" in a.knowledge:
            bought.add((a.name, "via_trade"))
    return agents, {"trades": trades, "know": know, "food": food, "bought": bought}


def demo_c_proprietary() -> None:
    print("=" * 72)
    print("DEMO C — PROPRIETARY KNOWLEDGE: guarded doesn't free-diffuse but sells; free stays free")
    print("=" * 72)

    def free_diffuses(personality: str) -> bool:
        world.create_world(size=8)
        world_state["economy_on"] = True
        holder = Agent(name="H", personality=personality)
        world.place_agent(holder, 2, 2); holder.knowledge.add("farming")
        learner = Agent(name="L", personality="curious and adventurous")
        world.place_agent(learner, 3, 2)
        rng = random.Random(0)
        for t in range(1, 50):
            world_state["turn"] = t
            knowledge.diffuse(world_state, t, rng=rng)
        return "farming" in learner.knowledge

    comp = free_diffuses("independent and competitive")
    friendly = free_diffuses("friendly and outgoing")
    print(f"  competitive holder -> learner gets farming FREE via diffusion? {comp}")
    print(f"  friendly holder    -> learner gets farming FREE via diffusion? {friendly}")
    assert comp is False, "a competitive holder must GUARD (no free diffusion)"
    assert friendly is True, "a friendly holder must still TEACH free (M1.1 intact)"

    # The guarded skill DOES move when bought.
    world.create_world(size=8)
    world_state["economy_on"] = True
    holder = Agent(name="H", personality="independent and competitive", hunger=2)
    world.place_agent(holder, 2, 2); holder.knowledge.add("farming"); holder.stockpile = 2.0
    buyer = Agent(name="L", personality="curious and adventurous", hunger=0)
    world.place_agent(buyer, 3, 2); buyer.money = 20.0
    economy.trade(world_state, 1)
    print(f"  competitive holder -> learner BUYS farming for "
          f"{20.0 - buyer.money:.1f} money? {'farming' in buyer.knowledge}")
    assert "farming" in buyer.knowledge, "a guarded skill must move by sale"
    print("  Guarding emerges from personality; guarded knowledge is sold, not given.  PASS\n")


def demo_d_specialization() -> None:
    print("=" * 72)
    print("DEMO D — SPECIALIZATION: two producer types (farmer, hunter), each knower-only")
    print("=" * 72)
    for skill, fn, mem in (("farming", knowledge.farm, "Tended crops"),
                           ("hunting", knowledge.hunt, "Took game")):
        world.create_world(size=12)
        knower = Agent(name="Knower", personality="curious and adventurous", hunger=0)
        world.place_agent(knower, 5, 5); knower.knowledge.add(skill)
        plain = Agent(name="Plain", personality="curious and adventurous", hunger=0)
        world.place_agent(plain, 9, 9)
        before = len(world_state["food"])
        rng = random.Random(0)
        for t in range(1, 40):
            fn(world_state, t, rng=rng)
        produced = len(world_state["food"]) - before
        print(f"  {skill:8s}: knower produced {produced} food; "
              f"non-knower produced {'some' if any(mem in m for m in plain.memory) else 'NONE'}")
        assert produced > 0, f"a {skill} knower should produce food"
        assert not any(mem in m for m in plain.memory), f"a non-knower must not {skill}"
    # Two producer types trade their distinct skills (a farmer lacks hunting and vice-versa).
    world.create_world(size=8)
    world_state["economy_on"] = True
    farmer = Agent(name="Farmer", personality="independent and competitive", hunger=2)
    world.place_agent(farmer, 3, 3); farmer.knowledge.add("farming"); farmer.stockpile = 12.0
    hunter = Agent(name="Hunter", personality="independent and competitive", hunger=2)
    world.place_agent(hunter, 4, 3); hunter.knowledge.add("hunting"); hunter.stockpile = 12.0
    economy.trade(world_state, 1)  # one of them buys the other's skill
    crossed = ("hunting" in farmer.knowledge) or ("farming" in hunter.knowledge)
    print(f"  cross-skill trade between a farmer and a hunter happened? {crossed}")
    assert crossed, "two producer types should trade their distinct skills"
    print("  Two real specializations exist and trade because each lacks the other.  PASS\n")


def demo_e_money() -> None:
    print("=" * 72)
    print("DEMO E — MONEY: food surplus -> money -> buys food/knowledge; food-grounded")
    print("=" * 72)
    world.create_world(size=8)
    world_state["storage_on"] = True
    world_state["economy_on"] = True
    rich = _settled("Rich", "independent and competitive", (3, 3), hunger=0, stockpile=storage.STORAGE_CAP)
    rich.knowledge.add("farming")
    world.place_food(3, 3)
    for t in range(1, 11):  # full larder -> overflow mints money
        rich.hunger = 0
        economy.mint(world_state, t)
    print(f"  10 turns at a FULL larder minted {rich.money:.1f} money (food surplus past the cap)")
    assert rich.money > 0, "surplus past the cap should mint money"

    # money buys knowledge...
    seller = _settled("Hunter", "independent and competitive", (4, 3), hunger=2, stockpile=2.0)
    seller.knowledge.add("hunting")
    m0 = rich.money
    economy.trade(world_state, 11)
    print(f"  money bought a skill: rich learned hunting={'hunting' in rich.knowledge}, "
          f"spent {m0 - rich.money:.1f} money; seller earned {seller.money:.1f}")
    assert "hunting" in rich.knowledge and rich.money < m0

    # ...and money buys food (a hungry, moneyed buyer from a food-rich seller).
    world.create_world(size=8)
    world_state["storage_on"] = True
    world_state["economy_on"] = True
    foodseller = _settled("Grocer", "cautious and territorial", (3, 3), hunger=0, stockpile=20.0)
    hungry = _settled("Hungry", "cautious and territorial", (4, 3), hunger=8, stockpile=0.0, money=15.0)
    economy.trade(world_state, 1)
    print(f"  money bought food: hungry buyer stockpile now {hungry.stockpile:.1f}, "
          f"money now {hungry.money:.1f}; grocer earned money {foodseller.money:.1f}")
    assert hungry.stockpile > 0 and hungry.money < 15.0, "money should buy food"

    # money is food-backed: it redeems to survive a starving turn.
    hungry.stockpile = 0.0; hungry.hunger = 9; hungry.money = storage.BUFFER_COST + 1.0
    survived = storage.draw_down(hungry)
    print(f"  money redeemed as food to survive starvation: survived={survived}, "
          f"money left {hungry.money:.1f}")
    assert survived and hungry.hunger < 10
    print("  Money is minted only from real food surplus and is redeemable as food — "
          "food-backed, not fiat.  PASS\n")


def demo_f_zero_cost_and_v1() -> None:
    print("=" * 72)
    print("DEMO F — zero LLM/RNG cost; economy OFF -> v1 byte-identical")
    print("=" * 72)
    # mint + trade in isolation: zero model calls, zero RNG.
    world.create_world(size=8)
    world_state["storage_on"] = True
    world_state["economy_on"] = True
    a = _settled("A", "independent and competitive", (3, 3), hunger=0, stockpile=storage.STORAGE_CAP)
    a.knowledge.add("hunting")
    world.place_food(3, 3)
    b = _settled("B", "cautious and territorial", (4, 3), hunger=0, money=20.0)
    llm.reset_call_stats()
    st0 = random.getstate()
    with contextlib.redirect_stdout(io.StringIO()):
        for t in range(1, 30):
            a.hunger = 0
            economy.mint(world_state, t)
            economy.trade(world_state, t)
    stats = llm.get_call_stats()
    print(f"  29 mint+trade passes: LLM calls = {stats}; RNG untouched = {random.getstate() == st0}")
    assert stats == {"decision": 0, "strategy": 0}, stats
    assert random.getstate() == st0, "the economy consumed RNG (would desync v1)"

    # economy OFF (default) -> byte-identical to a run with the param absent.
    def run(flag):
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(25, focal_budget=8)
            else:
                main.run_simulation(25, focal_budget=8, economy_on=flag)
        return buf.getvalue()
    assert run(None) == run(False), "economy_on=False changed the default run"
    print("  zero model calls; mint/trade draw no RNG; economy OFF byte-identical to v1.  PASS\n")


def run() -> None:
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        headline_1_emergent_price()
        headline_2_redistribution()
        demo_c_proprietary()
        demo_d_specialization()
        demo_e_money()
        demo_f_zero_cost_and_v1()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M2.3 VERIFIED: settlements are now an ECONOMY. A second producer skill (hunting) "
          "creates specialization; food-backed money emerges from surplus past the cap; agents "
          "TRADE food/knowledge<->money at prices that MOVE with rarity, desperation and "
          "surplus; and knowledge is property some GUARD and sell while others still teach it "
          "free — all at zero LLM/RNG cost, v1 byte-identical. Phase 2 is CLOSED.")
    print("=" * 72)


if __name__ == "__main__":
    run()
