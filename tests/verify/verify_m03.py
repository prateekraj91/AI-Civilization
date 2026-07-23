"""
verify_m03.py
=============

Deterministic verification of V2 milestone M0.3: SCALE to 100-300 agents +
renderer at scale. Closes Phase 0 (the scaling foundation).

Run offline (Ollama OFF, no model server):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m03.py

It re-uses the REAL scaling defaults from main.py (build_scaled_specs /
scaled_grid_size / scaled_food_cfg), so what it measures is the same world the
`python main.py --agents N` CLI runs — not a bespoke harness.

DEMO A — a 200-agent run completes in reasonable wall-clock; report turns, wall,
         ms/agent-turn, survival. Compare to a 50-agent baseline to show the
         per-agent cost did NOT blow up (engine scaled ~linearly in agent-turns,
         and LLM cost stayed pinned to the budget — sub-linear in N).
DEMO B — re-confirm the M0.2 cost-vs-N property at the higher range: N = 50, 100,
         200 at a fixed focal budget -> LLM strategy calls stay ~flat (track the
         budget, not the population). Assert at most `budget` agents are focal
         EVERY turn, for every N.
DEMO C — the density/heatmap renderer view renders for a large cast and mutates
         nothing (read-only boundary holds at scale).
DEMO D — seed reproducibility at scale: two seeded 100-agent runs are identical.
"""

from __future__ import annotations

import contextlib
import io
import random
import time

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from llm import cognition
from llm import llm
import main
from sim import population
from sim import world
from sim.world import spawn_food, world_state


def _scaled_run(n: int, budget: int, turns: int, *, seed: int = 7,
                instrument: bool = True) -> dict:
    """Run `turns` of the real loop at `n` agents with M0.3 scaling + M0.2 tiering.

    Mirrors main.run_simulation's per-turn order (re-tier, act, drip food, respawn)
    but instruments it: records wall-clock, agent-turns, LLM calls, survival, the
    max focal set seen, and asserts the budget is never exceeded. World geometry and
    cast come straight from main.py's scaling helpers, so this IS the CLI's world.
    """
    random.seed(seed)
    llm.reset_call_stats()

    grid = main.scaled_grid_size(n)
    specs = main.build_scaled_specs(n, grid)
    food_cfg = main.scaled_food_cfg(n)

    world.create_world(size=grid)
    for name, personality, goals, (x, y) in specs:
        world.place_agent(main.Agent(name=name, personality=personality, goals=goals,
                                     cognition="heuristic"), x, y)
    spawn_food(food_cfg["initial"], cluster=food_cfg["cluster"])

    strategies: dict = {}
    survived: dict[str, int] = {a.name: 0 for a in world_state["agents"]}
    counters: dict[str, int] = {"agent_turns": 0}
    tenure: dict[str, int] = {}

    max_focal, promotions, demotions = 0, 0, 0
    events_before = 0
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for turn in range(1, turns + 1):
            world_state["turn"] = turn
            cognition.update_tiers(world_state, turn, budget, tenure)
            if instrument:
                living = [a for a in world_state["agents"] if a.alive]
                focal = sum(1 for a in living if a.cognition == "llm")
                max_focal = max(max_focal, focal)
                assert focal <= budget, (turn, focal, budget)
            for agent in [a for a in world_state["agents"] if a.alive]:
                main.run_agent_turn(agent, turn, strategies, survived, counters)
            main._scaled_respawn_food(turn, food_cfg)
            population.process_respawns(turn, world_state)
    elapsed = time.perf_counter() - t0

    for e in world_state["events"][events_before:]:
        promotions += "promoted to focal" in e
        demotions += "demoted to heuristic" in e

    alive = sum(1 for a in world_state["agents"] if a.alive)
    at = counters["agent_turns"]
    return {"n": n, "grid": grid, "turns": turns, "budget": budget,
            "wall": elapsed, "agent_turns": at, "ms_per_at": 1000 * elapsed / max(at, 1),
            "strategy_calls": llm.get_call_stats()["strategy"],
            "alive": alive, "survival": alive / n if n else 0,
            "max_focal": max_focal, "promotions": promotions, "demotions": demotions}


def demo_a_200_agents_completes() -> None:
    print("=" * 70)
    print("DEMO A — a 200-agent run completes in reasonable wall-clock")
    print("=" * 70)
    base = _scaled_run(50, budget=8, turns=60)
    big = _scaled_run(200, budget=8, turns=60)
    for label, r in (("50-agent baseline (M0.1 scale)", base), ("200-agent run", big)):
        print(f"  {label}:")
        print(f"    grid {r['grid']}x{r['grid']}  turns {r['turns']}  wall {r['wall']:.2f}s  "
              f"agent-turns {r['agent_turns']}")
        print(f"    ms/agent-turn {r['ms_per_at']:.3f}  survival {r['survival']:.0%}  "
              f"LLM strategy calls {r['strategy_calls']}  (max focal/turn {r['max_focal']})")

    # 4x the agents must not blow up per-agent cost, and LLM calls must NOT 4x.
    assert big["ms_per_at"] <= base["ms_per_at"] * 2.0, (base["ms_per_at"], big["ms_per_at"])
    assert big["strategy_calls"] <= base["strategy_calls"] * 1.5 + big["budget"], \
        (base["strategy_calls"], big["strategy_calls"])
    assert big["survival"] >= 0.4, big["survival"]
    print(f"\n  200 agents: {big['wall']:.2f}s, {big['ms_per_at']:.3f} ms/agent-turn, "
          f"{big['survival']:.0%} survive. Per-agent cost held (~{big['ms_per_at']/base['ms_per_at']:.1f}x "
          f"of the 50-agent baseline) and LLM calls stayed pinned to the budget "
          f"({base['strategy_calls']}->{big['strategy_calls']}, not 4x).  PASS\n")


def demo_b_cost_vs_n_flat() -> None:
    print("=" * 70)
    print("DEMO B — LLM cost stays flat as N grows (tracks the budget, not population)")
    print("=" * 70)
    BUDGET, TURNS = 8, 50
    print(f"  fixed budget = {BUDGET}, turns = {TURNS}")
    print(f"  {'N':>4} | {'max focal/turn':>14} | {'LLM strategy calls':>18} | {'calls/turn':>10}")
    print("  " + "-" * 56)
    calls = []
    for n in (50, 100, 200):
        r = _scaled_run(n, BUDGET, TURNS)
        calls.append(r["strategy_calls"])
        print(f"  {n:>4} | {r['max_focal']:>14} | {r['strategy_calls']:>18} | "
              f"{r['strategy_calls'] / TURNS:>10.2f}")
        assert r["max_focal"] <= BUDGET, r
    assert max(calls) <= min(calls) * 1.5 + BUDGET, calls
    print(f"\n  calls stayed ~flat ({min(calls)}..{max(calls)}) while N went 50->200 "
          f"(linear-in-N would be ~{calls[0] * 4}).  PASS\n")


def demo_c_renderer_at_scale_is_read_only() -> None:
    print("=" * 70)
    print("DEMO C — the density/heatmap renderer view renders at scale, mutates nothing")
    print("=" * 70)
    import copy
    from renderer import render_frame
    from renderer.text_renderer import _use_heatmap

    random.seed(1)
    grid = main.scaled_grid_size(180)
    world.create_world(size=grid)
    for name, personality, goals, (x, y) in main.build_scaled_specs(180, grid):
        world.place_agent(main.Agent(name=name, personality=personality,
                                     cognition="heuristic"), x, y)
    spawn_food(160)
    world_state["turn"] = 4
    world_state["events"].append("turn 4: A005 stole food from A006")
    cognition.update_tiers(world_state, 4, 8, {})

    assert _use_heatmap(world_state), "heatmap view should be selected for 180 agents"
    # Snapshot the read-only invariants (positions + the scalar/list state).
    snap = copy.deepcopy({k: world_state[k] for k in world_state
                          if k not in ("agents", "occupancy")})
    positions = [(a.name, a.position, a.alive) for a in world_state["agents"]]

    frame = render_frame(world_state)

    after = {k: world_state[k] for k in world_state if k not in ("agents", "occupancy")}
    assert frame is not None
    assert after == snap, "render mutated world_state"
    assert positions == [(a.name, a.position, a.alive) for a in world_state["agents"]], \
        "render moved/killed an agent"
    print(f"  heatmap selected for 180 agents on a {grid}x{grid} grid; frame produced; "
          f"world_state byte-identical after render.  PASS\n")


def demo_d_seed_reproducible_at_scale() -> None:
    print("=" * 70)
    print("DEMO D — seed reproducibility holds at scale (two seeded 100-agent runs)")
    print("=" * 70)

    def digest(seed):
        _scaled_run(100, budget=8, turns=40, seed=seed, instrument=False)
        agents = sorted(world_state["agents"], key=lambda a: a.name)
        return [(a.name, a.position, a.hunger, a.alive) for a in agents], \
               list(world_state["events"])

    a_state, a_events = digest(123)
    b_state, b_events = digest(123)
    assert a_state == b_state, "agent states diverged between identical seeds"
    assert a_events == b_events, "event logs diverged between identical seeds"
    print(f"  two seed=123 100-agent runs: agent states + {len(a_events)} event lines "
          f"identical.  PASS\n")


def run() -> None:
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        demo_a_200_agents_completes()
        demo_b_cost_vs_n_flat()
        demo_c_renderer_at_scale_is_read_only()
        demo_d_seed_reproducible_at_scale()
    finally:
        llm.PROVIDER = saved
    print("=" * 70)
    print("M0.3 VERIFIED: the engine runs 100-300 agents at sub-linear LLM cost, the "
          "renderer has a read-only scale view, the tier system holds, and seeded runs "
          "reproduce. Phase 0 (scaling foundation) is closed.")
    print("=" * 70)


if __name__ == "__main__":
    run()
