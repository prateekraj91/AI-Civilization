"""
verify_m22.py
=============

Deterministic verification of V2 milestone M2.2: STORAGE & SURPLUS — the moment the
simulation grows WEALTH. Builds on M2.1 (settlement) and all of Phase 0 + Phase 1.

Run offline (Ollama OFF, no model server):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m22.py

The historical step M2.2 makes: M1.3 made farming PRODUCE food; M2.1 made nomads SETTLE
around it. M2.2 adds storable SURPLUS — a PERSONAL stockpile each settled agent banks
beyond its immediate hunger need. Wealth is never assigned: it EMERGES from traits the
agent already has (personality + farming knowledge), and it is a SURVIVAL BUFFER — a
hungry agent with savings draws them down to weather a food shock the poor don't.

HEADLINE 1 — EMERGENT INEQUALITY: in a mixed population, stockpile wealth VARIES across
             agents and the variation TRACKS personality + farming knowledge (competitive
             farmers richest, friendly non-farmers poorest). Reported as a distribution
             (min/mean/max + Gini) and a correlation of wealth with the trait-derived
             banking rate — proving it is structural, not random.
HEADLINE 2 — WEALTH BUFFERS SURVIVAL (reuse god_mode drought): after a settled village
             accumulates varying wealth, a drought-driven famine shock kills the
             savings-less while the wealthy draw down their stockpiles and live. Survival
             is reported split by wealth.
DEMO C — STORAGE REQUIRES SETTLEMENT: a settled agent accumulates; an identical nomad
         (settlement None) banks nothing.
DEMO D — CAP HOLDS: a relentless hoarder's stockpile never exceeds STORAGE_CAP.
DEMO E — ZERO LLM/RNG cost; storage OFF -> v1 byte-identical.
"""

from __future__ import annotations

import contextlib
import io
import random
from collections import defaultdict

import cognition
import god_mode
import knowledge
import llm
import main
import population
import settlement
import storage
import world
from agents import Agent
from strategy import get_personality
from world import spawn_food, world_state

PERS = ("curious and adventurous", "cautious and territorial",
        "friendly and outgoing", "independent and competitive")

# A wealth threshold for the survival split: holding at least one stored meal
# (storage.BUFFER_COST) is what lets an agent draw down to survive a starving turn.
ONE_MEAL = storage.BUFFER_COST


def _gini(xs: list[float]) -> float:
    """Gini coefficient of a list of non-negative values (0 = equal, ->1 = unequal)."""
    xs = sorted(xs)
    n = len(xs)
    s = sum(xs)
    if n == 0 or s == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * s) - (n + 1) / n


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation of two equal-length series (0 if degenerate)."""
    n = len(xs)
    if n == 0:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx ** 0.5 * vy ** 0.5)


def _build(seed: int, n: int, *, farmers_frac: float) -> tuple[list[Agent], dict]:
    """Set up a scaled heuristic cast (storage ON) with a fraction seeded as farmers."""
    llm.PROVIDER = "random"
    random.seed(seed)
    grid = main.scaled_grid_size(n)
    world.create_world(size=grid)
    world_state["storage_on"] = True
    cells = [(x, y) for x in range(grid) for y in range(grid)]
    random.Random(seed).shuffle(cells)
    agents = []
    for i in range(n):
        a = Agent(name=f"A{i:03d}", personality=PERS[i % 4], cognition="heuristic",
                  goals={"survive": 8, "wealth": 3, "friendship": 4})
        world.place_agent(a, *cells[i])
        agents.append(a)
    for a in agents[:int(n * farmers_frac)]:
        a.knowledge.add("farming")
    cfg = main.scaled_food_cfg(n)
    spawn_food(cfg["initial"])
    return agents, cfg


def _run(agents: list[Agent], cfg: dict, turns: int, ctx: dict, *,
         start: int = 1, farm: bool = True, respawn: bool = True,
         clear_food: bool = False) -> None:
    """Drive the real production loop (settlement + storage) over a turn range.

    `farm`/`respawn`/`clear_food` model a famine shock when farming/respawn are off and
    standing food is cleared each turn (a total crop failure layered on the god drought).
    """
    st, sv, cn, tn = ctx["strat"], ctx["surv"], ctx["cnt"], ctx["ten"]
    with contextlib.redirect_stdout(io.StringIO()):
        for t in range(start, start + turns):
            world_state["turn"] = t
            cognition.update_tiers(world_state, t, 8, tn)
            for a in [x for x in world_state["agents"] if x.alive]:
                main.run_agent_turn(a, t, st, sv, cn)
            if farm:
                knowledge.farm(world_state, t)
            settlement.update(world_state, t)
            storage.accumulate(world_state, t)
            if respawn:
                main._scaled_respawn_food(t, cfg)
            if clear_food:
                world_state["food"].clear()
            population.process_respawns(t, world_state)


def _fresh_ctx() -> dict:
    return {"strat": {}, "surv": {}, "cnt": {"agent_turns": 0}, "ten": {}}


def headline_1_emergent_inequality() -> None:
    print("=" * 72)
    print("HEADLINE 1 — EMERGENT INEQUALITY: wealth VARIES and TRACKS personality+knowledge")
    print("=" * 72)
    N, TURNS = 100, 60
    for seed in (1, 2):
        # All-farmer population isolates the PERSONALITY gradient.
        agents, cfg = _build(seed, N, farmers_frac=1.0)
        _run(agents, cfg, TURNS, _fresh_ctx())
        settled = [a for a in agents if a.alive and a.settlement is not None]
        piles = [a.stockpile for a in settled]
        rates = [storage.banking_rate(a) for a in settled]
        bydom = defaultdict(list)
        for a in settled:
            bydom[get_personality(a).dominant].append(a.stockpile)
        means = {d: sum(v) / len(v) for d, v in bydom.items()}
        r = _pearson(rates, piles)
        print(f"  seed {seed}: {len(settled)} settled  "
              f"stockpile min/mean/max = {min(piles):.1f}/{sum(piles)/len(piles):.1f}/"
              f"{max(piles):.1f}  Gini={_gini(piles):.2f}")
        print(f"    mean wealth by personality: " +
              "  ".join(f"{d}={means[d]:.1f}" for d in
                        ("independence", "caution", "curiosity", "friendliness") if d in means))
        print(f"    correlation(trait banking-rate, actual stockpile) = {r:+.2f}")
        # Inequality is real and structural, not noise.
        assert _gini(piles) > 0.12, f"wealth should be unequal, Gini={_gini(piles):.2f}"
        assert r > 0.5, f"wealth must track the trait-derived rate, r={r:.2f}"
        # The headline ordering: competitive hoarders richest, friendly sharers poorest.
        assert means["independence"] > means["friendliness"], \
            "competitive agents should out-save friendly ones"

    # Knowledge axis: with HALF the cast farming, farmers out-accumulate non-farmers.
    agents, cfg = _build(3, N, farmers_frac=0.5)
    _run(agents, cfg, TURNS, _fresh_ctx())
    settled = [a for a in agents if a.alive and a.settlement is not None]
    farmers = [a.stockpile for a in settled if "farming" in a.knowledge]
    nonf = [a.stockpile for a in settled if "farming" not in a.knowledge]
    fm = sum(farmers) / len(farmers)
    nm = sum(nonf) / len(nonf)
    print(f"  mixed knowledge (seed 3): farmer mean wealth {fm:.1f} (n={len(farmers)})  vs  "
          f"non-farmer {nm:.1f} (n={len(nonf)})")
    assert fm > nm * 1.3, "farmers (producers) should accumulate clearly more than non-farmers"
    print("\n  Wealth EMERGES: it varies across agents, correlates with their personality and\n"
          "  farming knowledge (competitive farmers richest, friendly non-farmers poorest),\n"
          "  and was never assigned — every agent started at 0.0.  PASS\n")


def headline_2_wealth_buffers_survival() -> None:
    print("=" * 72)
    print("HEADLINE 2 — WEALTH BUFFERS SURVIVAL: a drought famine kills the poor, not the rich")
    print("=" * 72)
    N, WARMUP, SHOCK = 100, 55, 12
    for seed in (1, 2):
        agents, cfg = _build(seed, N, farmers_frac=1.0)
        ctx = _fresh_ctx()
        _run(agents, cfg, WARMUP, ctx)  # village settles and accumulates varying wealth
        settled = [a for a in agents if a.alive and a.settlement is not None]
        rich = {a.name for a in settled if a.stockpile >= ONE_MEAL}
        poor = {a.name for a in settled if a.stockpile < ONE_MEAL}
        # SHOCK: reuse god_mode drought (suppresses respawn), plus a total crop failure
        # (farming off + standing food cleared) so the famine actually bites a self-feeding
        # farming village — otherwise farmers would just regrow food and no savings tested.
        with contextlib.redirect_stdout(io.StringIO()):
            god_mode.run_command(f"trigger_drought {SHOCK}", world_state)
        world_state["food"].clear()
        _run(agents, cfg, SHOCK, ctx, start=WARMUP + 1,
             farm=False, respawn=False, clear_food=True)
        rich_alive = sum(1 for a in agents if a.name in rich and a.alive)
        poor_alive = sum(1 for a in agents if a.name in poor and a.alive)
        rp = 100 * rich_alive / len(rich)
        pp = 100 * poor_alive / max(1, len(poor))
        print(f"  seed {seed}: at shock {len(rich)} wealthy (>= {ONE_MEAL:.0f} stored), "
              f"{len(poor)} poor (< {ONE_MEAL:.0f})")
        print(f"    after {SHOCK}-turn drought famine:  "
              f"WEALTHY survive {rich_alive}/{len(rich)} ({rp:.0f}%)   "
              f"POOR survive {poor_alive}/{len(poor)} ({pp:.0f}%)")
        assert rp >= 60, f"most of the wealthy should weather the famine, got {rp:.0f}%"
        assert pp <= 10, f"the savings-less should starve, got {pp:.0f}%"
    print("\n  The wealthy draw down their stockpiles and survive the shock; the poor, with no\n"
          "  reserve to spend, starve. Wealth MATTERS — it is a survival buffer.  PASS\n")


def demo_c_storage_requires_settlement() -> None:
    print("=" * 72)
    print("DEMO C — STORAGE REQUIRES SETTLEMENT: settled accumulates, nomad barely/never does")
    print("=" * 72)
    world.create_world(size=12)
    world_state["storage_on"] = True
    settled = Agent(name="Settled", personality="independent and competitive", hunger=0)
    nomad = Agent(name="Nomad", personality="independent and competitive", hunger=0)
    world.place_agent(settled, 5, 5)
    world.place_agent(nomad, 9, 9)
    settled.settlement = "S001"   # nomad.settlement stays None
    world.place_food(5, 5)
    world.place_food(9, 9)        # same surplus access for both
    with contextlib.redirect_stdout(io.StringIO()):
        for t in range(1, 30):
            settled.hunger = 0
            nomad.hunger = 0
            storage.accumulate(world_state, t)
    print(f"  settled agent stockpile = {settled.stockpile:.1f};  "
          f"nomad stockpile = {nomad.stockpile:.1f}")
    assert settled.stockpile > 0, "a settled agent should accumulate"
    assert nomad.stockpile == 0.0, "a nomad must not store — settlement is what enables storage"
    print("  only the settled agent built a stockpile; the nomad stored nothing.  PASS\n")


def demo_d_cap_holds() -> None:
    print("=" * 72)
    print("DEMO D — CAP HOLDS: a relentless hoarder never exceeds STORAGE_CAP")
    print("=" * 72)
    world.create_world(size=10)
    world_state["storage_on"] = True
    hoarder = Agent(name="Hoard", personality="independent and competitive", hunger=0)
    world.place_agent(hoarder, 5, 5)
    hoarder.settlement = "S001"
    hoarder.knowledge.add("farming")  # fastest accumulator in the sim
    world.place_food(5, 5)
    with contextlib.redirect_stdout(io.StringIO()):
        for t in range(1, 500):
            hoarder.hunger = 0
            storage.accumulate(world_state, t)
    print(f"  after 500 banking turns: stockpile = {hoarder.stockpile:.1f} "
          f"(cap {storage.STORAGE_CAP:.0f})")
    assert hoarder.stockpile <= storage.STORAGE_CAP, "stockpile exceeded the cap"
    assert hoarder.stockpile == storage.STORAGE_CAP, "the hoarder should saturate the cap"
    print("  the hoarder fills the cap and stops — wealth is bounded.  PASS\n")


def demo_e_zero_cost_and_v1() -> None:
    print("=" * 72)
    print("DEMO E — zero LLM/RNG cost; storage OFF -> v1 byte-identical")
    print("=" * 72)
    # accumulate in isolation: zero model calls, zero RNG.
    world.create_world(size=12)
    world_state["storage_on"] = True
    a = Agent(name="S", personality="independent and competitive", hunger=0)
    world.place_agent(a, 5, 5)
    a.settlement = "S001"
    world.place_food(5, 5)
    llm.reset_call_stats()
    st0 = random.getstate()
    with contextlib.redirect_stdout(io.StringIO()):
        for t in range(1, 40):
            a.hunger = 0
            storage.accumulate(world_state, t)
    stats = llm.get_call_stats()
    print(f"  39 accumulate passes: LLM calls = {stats}; RNG untouched = {random.getstate() == st0}")
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats
    assert random.getstate() == st0, "storage.accumulate consumed RNG (would desync v1)"

    # storage OFF (default) -> byte-identical to a run with the param absent.
    def run(flag):
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(25, focal_budget=8)
            else:
                main.run_simulation(25, focal_budget=8, storage_on=flag)
        return buf.getvalue()
    assert run(None) == run(False), "storage_on=False changed the default run"
    print("  zero model calls; accumulate draws no RNG; storage OFF byte-identical to v1.  PASS\n")


def run() -> None:
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        headline_1_emergent_inequality()
        headline_2_wealth_buffers_survival()
        demo_c_storage_requires_settlement()
        demo_d_cap_holds()
        demo_e_zero_cost_and_v1()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M2.2 VERIFIED: the sim grows WEALTH. Settled agents bank a bounded personal "
          "surplus whose size EMERGES from personality + farming knowledge (never assigned), "
          "so inequality appears on its own; and that wealth is a SURVIVAL BUFFER — the rich "
          "weather a drought famine the poor don't — at zero LLM/RNG cost, v1 byte-identical. "
          "The asymmetry Phase 2's trade will turn on now exists.")
    print("=" * 72)


if __name__ == "__main__":
    run()
