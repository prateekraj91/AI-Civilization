"""
main.py
=======

Entry point for AI Civilization — a multi-agent, personality- and goal-driven
simulation.

Three agents (Alex, Bob, Kira) share one world and one food supply, act
SEQUENTIALLY each turn, perceive their neighbours by name, and remember sightings
in bounded memory. What makes them feel *different* is the behaviour layer:

  - Personality (personality.py) → distinct per-turn instincts (Phase 1).
  - Goals (agents.Agent.goals)   → fed into the strategy prompt (Phase 2).
  - Memory (bounded)             → recent entries fed into the prompt (Phase 3).
  - Strategy (strategy.py)       → the LLM is asked for a high-level plan only
                                   every STRATEGY_INTERVAL turns; in between, the
                                   plan is executed in pure Python (Phase 4). This
                                   cuts inference cost ~5x.

The simulation knows NOTHING about model providers — all AI calls go through
llm.get_strategy(). Run offline with AICIV_PROVIDER=random.

OUT OF SCOPE (intentionally): economies, villages, governments, factions,
religion, crafting, combat, trading, conversations, God Mode.
"""

import argparse
import contextlib
import math
import os
import random
import sys
import time

import alliance
import beliefs
import conversation
import culture
import discontent
import economy
import empire
import god_mode
import heuristic
import kingdoms
import knowledge
import labor
import leadership
import lineage
import metallurgy
import monarchy
import population
import religion
import settlement
import storage
import taxation
import uprising
import writing
from cognition import update_tiers
from agents import Agent
from llm import PROVIDER, get_call_stats, get_strategy, reset_call_stats
from strategy import (
    Strategy,
    build_strategy_prompt,
    choose_action,
    format_goals,
    get_personality,
)
from world import (
    create_world,
    execute_action,
    is_dead,
    is_dependent_child,
    observe,
    place_agent,
    record_memory,
    record_social_memories,
    render,
    spawn_food,
    update_hunger,
    world_state,
)

# --- Output modes ---------------------------------------------------------
# Presentation only — never affects the simulation. DEBUG_MODE (default) prints a
# terse per-turn summary; VERBOSE_MODE prints the full detailed report.
# Override with: AICIV_OUTPUT=verbose python main.py
_OUTPUT = os.getenv("AICIV_OUTPUT", "debug").lower()
VERBOSE_MODE = _OUTPUT == "verbose"
DEBUG_MODE = not VERBOSE_MODE  # default

# Maximum turns to simulate (or until every agent has starved). Day 9 lengthened
# runs so social dynamics (talk + trust) had time to emerge; Day 11 keeps them
# long (50) so survival pressure has room to force visible eat/starve outcomes.
NUM_TURNS = 50

# Phase 4: how often (in turns) to refresh an agent's strategy via the LLM.
# Between refreshes the cached strategy is executed in Python — no inference.
STRATEGY_INTERVAL = 5

# V2 M0.2: how many agents may run the expensive LLM ("focal") mind AT ONCE. The
# tiering system (cognition.update_tiers) keeps the most interesting `budget`
# agents focal and the rest on the zero-LLM heuristic mind, so inference cost
# scales with this number, NOT with population. Kept small; with the V1 trio
# (3 agents) it is >= the cast so EVERYONE is focal and the run is byte-identical
# to v1 — the tiering only bites once agents > budget. `--focal-budget` overrides;
# `--focal-budget 0` makes everyone heuristic (the M0.1 zero-LLM run).
DEFAULT_FOCAL_BUDGET = 8

# Day 15 God mode: pause into the interactive God menu every N turns. Default 0
# (OFF) so normal/automated runs never block on input(); set AICIV_GOD_EVERY=10 to
# drop into the menu every 10 turns. The pause happens at a clean turn boundary, so
# resuming continues the loop uncorrupted.
GOD_EVERY = int(os.getenv("AICIV_GOD_EVERY", "0"))

# --- Food economy: scarcity knobs (Day 11) --------------------------------
# These are the ONLY dials for survival pressure — keep them named here, not as
# magic numbers buried in the loop. Day 11 deliberately REVERSES the Day 9
# abundance rebalance: Day 9 flooded the map (INITIAL_FOOD=14, topped back up to
# 12 EVERY turn) so the social layer could emerge with nobody starving. Day 11
# makes food genuinely SCARCE so agents must compete or cooperate — supply is
# tuned BELOW what three hungry agents consume, so not everyone can stay fed.
#
#   knob                  Day 9 (abundant)      Day 11 (scarce, current)
#   INITIAL_FOOD          14                    5      (was 3; raised for contact)
#   respawn rule          top up to 12 / turn   ~1 food every 5 turns
#   placement             scattered             clustered at centre (contention)
#
# Demand vs supply (why someone starves): EAT_RELIEF=7 means each agent needs
# ~1 food per 7 turns; three agents demand ~0.43 food/turn. The drip below
# supplies ~0.20 food/turn — a deliberate deficit, so food runs out under load.
#
# Day 11 contention tune-up: pure scarcity made agents starve alone in separate
# corners (0 talks, 0 trust changes). Two fixes pull them together WITHOUT making
# the world generous: (1) INITIAL_FOOD 3->5 so they live a few turns longer and
# actually meet; (2) FOOD_CLUSTERED — scarce food spawns on the SAME central
# tiles they start near, so they converge and compete instead of dispersing.
INITIAL_FOOD = 5            # food cells on the map at t=0 (was 3; Day 9 was 14)
FOOD_RESPAWN_EVERY = 5      # respawn cadence: add food once every N turns (~0.2/turn)
FOOD_RESPAWN_AMOUNT = 1     # food cells added on each respawn tick
FOOD_RESPAWN_CAP = 5        # never respawn above this many standing food cells; this
                            # only bounds accumulation AFTER agents die — while the
                            # world is contested, demand keeps food well below it.
FOOD_CLUSTERED = True       # spawn food in a central arena (world.FOOD_CLUSTER_RADIUS)
                            # so agents contend over shared tiles, not scatter.

# The starting cast. Each agent has a distinct dominant trait so behaviour and
# logs are easy to tell apart: Alex = friendly, Bob = cautious, Kira = independent.
AGENT_SPECS = [
    ("Alex", "friendly and outgoing", {"survive": 7, "friendship": 8, "wealth": 2}, (4, 4)),
    ("Bob", "cautious and territorial", {"survive": 9, "wealth": 5, "friendship": 2}, (6, 4)),
    ("Kira", "independent and competitive", {"survive": 7, "wealth": 8, "friendship": 1}, (4, 6)),
]

# --- V2 M0.3: large-cast world geometry -----------------------------------
# Scaling to 100-300 agents needs a world sized to match, or 200 agents on a 10x10
# grid are all dead by turn 5 from pure contention. These ratios reproduce the M0.1
# 50-agent economy (which sustained ~60% survival) at any N: keep the agent DENSITY
# fixed (so the absolute grid grows with population) and scale food supply with the
# population's demand. They are defaults a large run can override (--grid-size); the
# default 3-agent run never touches this path, so v1 is byte-for-byte unchanged.
#
# Survival at scale is a GEOMETRY/behaviour lever, not a property of the mind: the
# heuristic forages about as well as the food it can reach allows. Loosen density or
# raise the food ratios to lift survival; tighten them to force lethal competition.
SCALE_DENSITY = 0.125        # target agents-per-cell (grid = sqrt(N / density))
SCALE_INITIAL_FOOD = 0.9     # food cells at t=0, as a multiple of N
SCALE_FOOD_PER_TURN = 0.16   # food added per turn, as a multiple of N (~demand at EAT_RELIEF=7)
SCALE_FOOD_CAP = 1.6         # never respawn above this multiple of N standing food cells

# Personalities cycled across a procedurally generated large cast, so behaviour is
# mixed (foragers, campers, socialisers) the way the named trio is — an all-one-trait
# crowd forages badly and skews survival.
SCALE_PERSONALITIES = (
    "curious and adventurous",
    "cautious and territorial",
    "friendly and outgoing",
    "independent and competitive",
)


# A large cast should never be put on a grid smaller than the v1 default.
GRID_FLOOR = 10


def scaled_grid_size(n: int) -> int:
    """Grid edge length that keeps agent density at SCALE_DENSITY for `n` agents."""
    return max(GRID_FLOOR, math.ceil(math.sqrt(n / SCALE_DENSITY)))


def build_scaled_specs(n: int, grid: int) -> list[tuple]:
    """Procedurally build `n` agent specs on distinct random cells of a `grid` world.

    Returns the same (name, personality, goals, (x, y)) spec tuples AGENT_SPECS uses,
    so run_simulation places them through the identical path. Positions are drawn from
    the seeded global `random` (main() seeds before calling this), so a seeded large
    run is reproducible. Personalities cycle SCALE_PERSONALITIES for a mixed crowd.
    """
    cells = [(x, y) for x in range(grid) for y in range(grid)]
    random.shuffle(cells)
    goals = {"survive": 8, "wealth": 3, "friendship": 4}
    specs: list[tuple] = []
    for i in range(n):
        x, y = cells[i]
        specs.append((f"A{i:03d}", SCALE_PERSONALITIES[i % len(SCALE_PERSONALITIES)],
                      dict(goals), (x, y)))
    return specs


def scaled_food_cfg(n: int) -> dict:
    """Food economy (initial / per-turn / cap, scattered) sized to `n` agents."""
    return {
        "initial": round(SCALE_INITIAL_FOOD * n),
        "per_turn": max(1, round(SCALE_FOOD_PER_TURN * n)),
        "cap": round(SCALE_FOOD_CAP * n),
        "cluster": False,  # scatter so food is reachable everywhere, not a central pile
    }


# Memory entries worth surfacing in the end-of-run summary (Phase 5).
_IMPORTANT_MEMORY_KEYS = ("Observed", "Ate food", "Starved", "New strategy", "Blocked",
                          "stole", "Trust in", "allied", "ALLIANCE", "BETRAYED",
                          "proposed an alliance", "died on turn", "appeared on turn",
                          # M4.1 lineage moments (these strings only ever exist when
                          # lineage is on, so the default summary is unchanged):
                          "was born", "Born to", "Came of age", "Died of old age")


def important_memories(memory: list[str], limit: int = 5) -> list[str]:
    """The most salient recent memories (sightings, meals, strategy changes).

    Falls back to the last few raw memories if nothing notable was recorded, so
    the summary is never empty for an agent that lived.
    """
    notable = [m for m in memory if any(k in m for k in _IMPORTANT_MEMORY_KEYS)]
    chosen = notable or memory
    return chosen[-limit:]


def maybe_respawn_food(turn: int) -> None:
    """Day 11 scarcity: drip food onto the map slowly instead of topping it up.

    Day 9 refilled to FOOD_RESPAWN_TO (12) EVERY turn — effectively unlimited
    food. Day 11 replaces that with a slow trickle of FOOD_RESPAWN_AMOUNT every
    FOOD_RESPAWN_EVERY turns (~1 food / 5 turns), which is intentionally SLOWER
    than three agents eat, so food genuinely runs out and they must compete. The
    CAP only stops unbounded accumulation once agents stop eating; while the
    world is contested it almost never binds.
    """
    if FOOD_RESPAWN_EVERY <= 0:
        return
    # Day 15: a god-triggered drought suppresses ALL respawn while it lasts.
    if turn <= world_state.get("drought_until", 0):
        return
    if turn % FOOD_RESPAWN_EVERY == 0 and len(world_state["food"]) < FOOD_RESPAWN_CAP:
        spawn_food(FOOD_RESPAWN_AMOUNT, cluster=FOOD_CLUSTERED)


def _scaled_respawn_food(turn: int, cfg: dict) -> None:
    """M0.3 large-cast food drip: add cfg['per_turn'] food each turn up to cfg['cap'].

    The scaled analogue of maybe_respawn_food for a big population — it tops the map
    up EVERY turn (not every Nth) at a rate matched to N agents' demand, so a large
    cast isn't starved by the v1 trio's deliberately scarce trickle. Honours the same
    god-drought suppression. Only used when run_simulation is given a food_cfg.
    """
    if turn <= world_state.get("drought_until", 0):
        return
    if len(world_state["food"]) < cfg["cap"]:
        spawn_food(cfg["per_turn"], cluster=cfg["cluster"])


def living_agents() -> list[Agent]:
    """All agents still alive, in turn order."""
    return [a for a in world_state["agents"] if a.alive]


def log_agent_turn(agent: Agent, strat: Strategy, refreshed: bool,
                   observation: str, observed: list[str],
                   action: str, note: str, result: str) -> None:
    """Print one agent's slice of a turn (VERBOSE_MODE)."""
    x, y = agent.position
    source = "refreshed via LLM" if refreshed else "cached"
    print(f"  --- {agent.name} (pos ({x},{y}), hunger {agent.hunger}) ---")
    print(f"    Strategy: {strat.label()} ({source})")
    print("\n".join(f"    {line}" for line in observation.splitlines()))
    if observed:
        print(f"    Detected nearby: {', '.join(observed)}")
    print(f"    Action: {action}  ({note})")
    print(f"    Result: {result}")
    print()


def run_agent_turn(agent: Agent, turn: int, strategies: dict[str, Strategy],
                   survived: dict[str, int], counters: dict[str, int]) -> str:
    """Advance one agent through one turn against the shared world.

    Refreshes the agent's strategy via the LLM only when it is missing or stale
    (every STRATEGY_INTERVAL turns); otherwise the cached strategy is executed in
    pure Python. Returns a short action label for the terse summary.
    """
    # Time passes first: hunger grows. Reaching the limit means starvation.
    update_hunger(agent)

    # M4.1 lineage: a DEPENDENT CHILD takes no actions — it does not forage (no
    # brink-eat: children don't feed themselves), produce, trade, talk or fight.
    # It is fed from its parents' stores at end of turn (lineage.update); if they
    # cannot feed it, it starves like anyone through the same death path. It costs
    # ZERO LLM calls (no strategy is ever built for it). Gated on lineage_on, so a
    # default run never enters this branch and stays byte-identical.
    if is_dependent_child(agent, world_state):
        if is_dead(agent):
            survivors = population.announce_death(agent, turn, world_state, cause="starved")
            if VERBOSE_MODE:
                print(f"  --- {agent.name} ---")
                print(f"    {agent.name} (a child) has starved at {agent.position}; "
                      f"{len(survivors)} survivor(s) recorded the death.")
                print()
            return "starved"
        survived[agent.name] = turn
        counters["agent_turns"] += 1
        if VERBOSE_MODE:
            print(f"  --- {agent.name} (dependent child, age {agent.age}, "
                  f"hunger {agent.hunger}) stays by its family ---\n")
        return "child"

    if is_dead(agent):
        # A meal underfoot saves you at the brink: reaching food costs a turn to
        # step on and another to eat, so an agent that arrived at high hunger
        # would otherwise starve one tick before eating. If it is standing on
        # food, it eats now instead of dying.
        if agent.position in world_state["food"]:
            survived[agent.name] = turn
            counters["agent_turns"] += 1
            result = execute_action(agent, "eat")
            if VERBOSE_MODE:
                print(f"  --- {agent.name} ate at the brink (hunger now {agent.hunger}) ---")
                print(f"    {result}\n")
            return "eat"
        # M2.2 survival buffer: no reachable food, but a SETTLED agent with savings draws
        # its stockpile down to live instead of starving — wealth weathers a food shock
        # (a drought) that kills its savings-less neighbours. Gated on the storage system
        # being on, so a v1/storage-off run never reaches here and stays byte-identical.
        if world_state.get("storage_on") and storage.draw_down(agent):
            survived[agent.name] = turn
            counters["agent_turns"] += 1
            record_memory(agent, f"Survived on stored food (stockpile now {agent.stockpile:.1f})")
            world_state["events"].append(
                f"turn {turn}: {agent.name} drew on savings to survive starvation "
                f"(stockpile now {agent.stockpile:.1f})")
            if VERBOSE_MODE:
                print(f"  --- {agent.name} drew on its stockpile to survive "
                      f"(hunger now {agent.hunger}, stockpile {agent.stockpile:.1f}) ---\n")
            return "buffer"
        # Day 14: death is now an event the society registers — a DEATH line in
        # events[], a memory of it on every survivor, and a queued respawn.
        survivors = population.announce_death(agent, turn, world_state, cause="starved")
        if VERBOSE_MODE:
            print(f"  --- {agent.name} ---")
            print(f"    {agent.name} has died of starvation at {agent.position}.")
            print(f"    {len(survivors)} survivor(s) recorded the death; "
                  f"respawn due turn {turn + population.RESPAWN_DELAY}.")
            print()
        return "starved"

    survived[agent.name] = turn
    counters["agent_turns"] += 1

    # Strategy caching (Phase 4): only hit the LLM when due for a refresh. Any
    # message just received rides into this single call so a reply/reaction needs
    # NO extra inference (Day 8).
    strat = strategies.get(agent.name)
    refresh_due = strat is None or (turn - strat.issued_turn) >= STRATEGY_INTERVAL
    incoming = conversation.pending_incoming(agent, turn)
    # M0.1: the heuristic mind reads structured perception itself (world.scan), so the
    # human-readable observation string is only built when the LLM mind needs it (or
    # for the verbose log) — never for a refreshing heuristic agent.
    is_heuristic = getattr(agent, "cognition", "llm") == "heuristic"
    need_obs = VERBOSE_MODE or (refresh_due and not is_heuristic)
    observation = observe(agent, world_state) if need_obs else ""

    refreshed = False
    if refresh_due:
        # The SINGLE cognition switch (M0.1): an agent flagged "heuristic" derives its
        # strategy from pure Python (zero LLM calls); otherwise the model layer is
        # asked, exactly as in V1. Both return the same strategy dict shape, so the
        # Strategy construction and everything below are mind-agnostic.
        if is_heuristic:
            data = heuristic.decide_strategy(agent, world_state)
        else:
            data = get_strategy(build_strategy_prompt(agent, observation, incoming=incoming,
                                                      state=world_state))
        strat = Strategy(kind=data["strategy"], target=data.get("target", ""),
                         message=data.get("message", ""), reaction=data.get("reaction", ""),
                         issued_turn=turn)
        strategies[agent.name] = strat
        record_memory(agent, f"New strategy: {strat.label()}")
        refreshed = True

    # Consume any delivered messages and react (deterministic off-refresh, the
    # strategy call's reaction on a refresh turn). No new LLM call either way.
    conversation.process_inbox(agent, refreshed, strat.reaction, turn, world_state)

    # Detection + social memory still happen every turn (Days 7-8 preserved).
    observed = record_social_memories(agent, world_state)

    # Execute the cached strategy in Python (no inference). A talk action is
    # delivered via the conversation layer; everything else mutates the world.
    action, note = choose_action(agent, strat, world_state)
    if action.startswith("talk_to_"):
        result = conversation.handle_talk(agent, action, strat, refreshed, turn, world_state)
    elif action.startswith("steal_from_"):
        result = conversation.handle_steal(agent, action, turn, world_state)
    elif action.startswith("ally_with_"):
        result = alliance.handle_ally(agent, action, turn, world_state)
    elif action.startswith("betray_alliance_"):
        result = alliance.handle_betray(agent, action, turn, world_state)
    else:
        result = execute_action(agent, action)

    if VERBOSE_MODE:
        log_agent_turn(agent, strat, refreshed, observation, observed, action, note, result)
    return action


def print_agent_summary(survived: dict[str, int], num_turns: int = NUM_TURNS) -> None:
    """Phase 5: per-agent post-run report for easy analysis."""
    print("=" * 56)
    print("AGENT SUMMARY")
    print("=" * 56)
    for agent in world_state["agents"]:
        pers = get_personality(agent)
        print(f"Agent:           {agent.name}")
        print(f"Personality:     {agent.personality} (dominant: {pers.dominant})")
        print(f"Goals:           {format_goals(agent.goals)}")
        print(f"Status:          {'ALIVE' if agent.alive else 'DEAD'}")
        print(f"Turns survived:  {survived.get(agent.name, 0)} / {num_turns}")
        # M2.2: read-only wealth overlay — printed ONLY when the storage system is on, so a
        # default (storage-off) summary is byte-identical to v1. Pure read of agent state.
        if world_state.get("storage_on"):
            print(f"Stockpile:       {agent.stockpile:.1f} / {storage.STORAGE_CAP:.0f}")
        # M2.3: read-only money/skills overlay — printed ONLY when the economy is on, so a
        # default run's summary is byte-identical to v1. Pure read of agent state.
        if world_state.get("economy_on"):
            skills = ", ".join(sorted(s for s in agent.knowledge
                                      if s in economy.PRODUCER_SKILLS)) or "(none)"
            print(f"Money:           {agent.money:.1f}")
            print(f"Producer skills: {skills}")
        # M3.1: read-only employment overlay — printed ONLY when wage labor is on, so a default
        # run's summary is byte-identical to v1. Pure read of world_state["employments"].
        if world_state.get("labor_on"):
            emps = world_state.get("employments", [])
            as_boss = [l for l in emps if l["employer"] == agent.name]
            as_worker = next((l for l in emps if l["worker"] == agent.name), None)
            if as_boss:
                workers = ", ".join(sorted(l["worker"] for l in as_boss))
                print(f"Employs:         {len(as_boss)} ({workers})")
            elif as_worker is not None:
                print(f"Employed by:     {as_worker['employer']} at wage {as_worker['wage']:.2f}")
        # M3.2: read-only leadership overlay — printed ONLY when the system is on, so a default
        # run's summary is byte-identical to v1. Pure read of world_state["leaders"].
        if world_state.get("leadership_on"):
            leaders = world_state.get("leaders", {})
            leads = next((r for r in leaders.values() if r["leader"] == agent.name), None)
            if leads is not None:
                print(f"Leads:           {leads['leader']}'s following of "
                      f"{len(leads['followers'])} (since turn {leads['since']})")
            else:
                follows = next((r["leader"] for r in leaders.values()
                                if agent.name in r["followers"]), None)
                if follows is not None:
                    print(f"Follows:         {follows}")
        # M3.4: read-only monarchy overlay — printed ONLY when the system is on (default run is
        # byte-identical to v1). Pure read of world_state["monarchs"].
        if world_state.get("monarchy_on"):
            monarchs = world_state.get("monarchs", {})
            reigns = next((f"{sid} (since turn {r['since']}, garrison {len(r['garrison'])})"
                           for sid, r in monarchs.items() if r["monarch"] == agent.name), None)
            if reigns is not None:
                print(f"Reigns over:     {reigns}  [by force]")
            else:
                serves = next((r["monarch"] for r in monarchs.values()
                               if agent.name in r["garrison"]), None)
                if serves is not None:
                    print(f"Soldiers for:    {serves}")
        # M3.5: read-only kingdoms/vassalage overlay — printed ONLY when the system is on (default
        # run is byte-identical to v1). Pure read of world_state["kingdoms"].
        if world_state.get("kingdoms_on"):
            kdoms = world_state.get("kingdoms", {})
            rec = kdoms.get(agent.name)
            if rec is not None:
                print(f"Rules kingdom:   {len(rec['settlements'])} settlements, "
                      f"{len(rec['vassals'])} vassals (since turn {rec['founded']})  [feudal crown]")
            else:
                liege = next((k for k, r in sorted(kdoms.items())
                              if agent.name in r["vassals"].values()), None)
                if liege is not None:
                    print(f"Vassal of:       {liege}  [sworn fealty]")
        # M3.6: read-only empire overlay — printed ONLY when the system is on (default run is
        # byte-identical to v1). Pure read of world_state["empires"] (renderer/summary read-only).
        if world_state.get("empire_on"):
            empires = world_state.get("empires", {})
            emp = empires.get(agent.name)
            if emp is not None:
                print(f"Rules empire:    {len(emp['subject_kings'])} subject-kings "
                      f"(since turn {emp['founded']})  [imperial crown]")
            else:
                overlord = next((e for e, r in sorted(empires.items())
                                 if agent.name in r["subject_kings"]), None)
                if overlord is not None:
                    print(f"Subject-king of: {overlord}  [submitted in war]")
        # M4.1: read-only lineage overlay — printed ONLY when the system is on, so a default
        # run's summary is byte-identical. Pure read of the agent's lineage fields.
        if world_state.get("lineage_on"):
            stage = "dependent child" if agent.dependent else "adult"
            print(f"Age:             {agent.age} of lifespan {agent.lifespan} ({stage})")
            if agent.parents:
                print(f"Parents:         {agent.parents[0]} and {agent.parents[1]}")
        # M4.4: read-only discontent overlay — printed ONLY when the gauge is on, so a default run's
        # summary is byte-identical to v1. Pure read of world_state["discontent"]. A living settled
        # adult over the resentment bar is flagged so the summary reads the resentful faction at a glance.
        if world_state.get("discontent_on"):
            level = discontent.agent_discontent(agent.name, world_state)
            tag = "  [RESENTFUL]" if level >= discontent.RESENTMENT_THRESHOLD else ""
            print(f"Discontent:      {level:.1f} / {discontent.DISCONTENT_CAP:.0f}"
                  f" (resentment at {discontent.RESENTMENT_THRESHOLD:.0f}){tag}")
        # M4.7: read-only beliefs overlay — printed ONLY when the system is on, so a default run's
        # summary is byte-identical. Pure read of world_state["beliefs"].
        if world_state.get("beliefs_on"):
            held = sorted(beliefs.agent_beliefs(agent.name, world_state))
            print(f"Beliefs:         {', '.join(held) if held else '(none yet)'}")
        # M4.8: read-only faith overlay — printed ONLY when religion is on. Pure read of world_state["faiths"].
        if world_state.get("religion_on"):
            faith = next((f for f in world_state.get("faiths", {}).values()
                          if agent.name in f["followers"]), None)
            if faith is not None:
                role = "PROPHET of" if faith["prophet"] == agent.name else "follows"
                print(f"Faith:           {role} {faith['name']}")
        print("Important memories:")
        for mem in important_memories(agent.memory):
            print(f"  - {mem}")
        print()


def print_settlement_pressure() -> None:
    """M4.4: world-level SETTLEMENT PRESSURE report — printed ONLY when the discontent gauge is on
    (a default run never calls it, so the default summary is byte-identical to v1).

    For each settlement (sorted), show the size of its RESENTFUL faction (members over the resentment
    threshold — the number M4.5 will trigger an uprising on) alongside its aggregate discontent, both
    DERIVED on demand from the per-agent gauge. This is the legible world read-out that lets a
    tyrant's settlement be ranked against a fair leader's at a glance — a pure read, no action.
    """
    print("=" * 56)
    print("SETTLEMENT PRESSURE (world_state['discontent'], M4.4)")
    print("=" * 56)
    settlements = world_state.get("settlements", {})
    if not settlements:
        print("(no settlements)")
        print()
        return
    for sid in sorted(settlements):
        pressure = discontent.settlement_pressure(sid, world_state)
        total = discontent.settlement_discontent(sid, world_state)
        members = sum(1 for m in settlements[sid]["members"]
                      if any(a.name == m and a.alive for a in world_state["agents"]))
        flag = "  << PRESSURE" if pressure >= discontent.PRESSURE_UPRISING_HINT else ""
        print(f"{sid}: {pressure}/{members} resentful (aggregate discontent {total:.1f}){flag}")
    print()


def print_belief_cultures() -> None:
    """M4.7: world-level BELIEF CULTURES report — printed ONLY when beliefs are on (a default run
    never calls it, so the default summary is byte-identical to v1).

    For each settlement (sorted), show its DOMINANT beliefs (most-held, with holder counts) — the
    proto-culture that emerged from what its members lived through and spread among themselves. A pure
    read; this is the legible readout M4.8 (religion) and M4.9 (cultural identity) will build on.
    """
    print("=" * 56)
    print("BELIEF CULTURES (world_state['beliefs'], M4.7)")
    print("=" * 56)
    settlements = world_state.get("settlements", {})
    if not settlements:
        print("(no settlements)")
        print()
        return
    for sid in sorted(settlements):
        dom = beliefs.dominant_beliefs(sid, world_state)
        if dom:
            profile = "; ".join(f"'{b}' x{n}" for b, n in dom)
        else:
            profile = "(no shared beliefs)"
        print(f"{sid}: {profile}")
    print()


def print_metallurgy() -> None:
    """M4.11: world-level METALLURGY report — printed ONLY when metallurgy is on (a default run never
    calls it, so the default summary is byte-identical to v1). Per settlement: whether it has a FORGE
    (knows metalworking) and can field ARMED fighters (knows weapons) — the material balance of power."""
    print("=" * 56)
    print("METALLURGY (M4.11 — forges & arms)")
    print("=" * 56)
    settlements = world_state.get("settlements", {})
    if not settlements:
        print("(no settlements)")
        print()
        return
    for sid in sorted(settlements):
        forge = metallurgy.is_metallurgical(world_state, sid)
        armed = metallurgy.is_armed_settlement(world_state, sid)
        tags = []
        if forge:
            tags.append("FORGE (better tools)")
        if armed:
            tags.append("ARMED (weapons)")
        print(f"{sid}: {' + '.join(tags) if tags else 'neolithic (no metallurgy)'}")
    print()


def print_records() -> None:
    """M4.10: world-level WRITTEN RECORDS report — printed ONLY when writing is on (a default run
    never calls it, so the default summary is byte-identical to v1).

    Per settlement: whether it is LITERATE, its written LAW (if any, with who it descends from), its
    archived techs, and its CHRONICLE length — the institutional memory Arc 6 will surface."""
    print("=" * 56)
    print("WRITTEN RECORDS (M4.10 — literacy, law, archive, chronicle)")
    print("=" * 56)
    settlements = world_state.get("settlements", {})
    if not settlements:
        print("(no settlements)")
        print()
        return
    for sid in sorted(settlements):
        if not writing.is_literate(world_state, sid):
            print(f"{sid}: illiterate (no lasting records)")
            continue
        law = writing.written_law(world_state, sid)
        arch = sorted(writing.archive_of(world_state, sid))
        chron = writing.chronicle_of(world_state, sid)
        print(f"{sid}: LITERATE")
        if law is not None:
            desc = f" (from {law['inherited_from']})" if law.get("inherited_from") else ""
            print(f"    law set by {law['set_by']}{desc}: tax {law['tax_rate']}, levy {law['levy_rate']}")
        print(f"    archive: {arch if arch else '(none)'}   chronicle: {len(chron)} entries")
    print()


def print_cultures() -> None:
    """M4.9: world-level CULTURE report — printed ONLY when culture is on (a default run never calls
    it, so the default summary is byte-identical to v1).

    For each settlement: its culture signature (dominant beliefs), whether it is FOREIGN-RULED (an
    imperial fault line), and — if so — its ASSIMILATION progress toward the ruler's culture."""
    print("=" * 56)
    print("CULTURES (M4.9 — identity, foreign rule, assimilation)")
    print("=" * 56)
    settlements = world_state.get("settlements", {})
    if not settlements:
        print("(no settlements)")
        print()
        return
    for sid in sorted(settlements):
        sig = sorted(culture.culture_signature(world_state, sid))
        line = f"{sid}: culture {sig if sig else '(none)'}"
        if culture.is_foreign_ruled(world_state, sid):
            ruler = religion._sovereign(world_state, sid)
            prog = culture.assimilation_progress(world_state, sid)
            line += f"   FOREIGN-RULED by {ruler} (assimilation {prog*100:.0f}%)"
        print(line)
    print()


def print_faiths() -> None:
    """M4.8: world-level FAITHS report — printed ONLY when religion is on (a default run never calls
    it, so the default summary is byte-identical to v1).

    For each faith: its core beliefs, congregation size, prophet (if any), and — per settlement — the
    ruler's ALIGNMENT with it (aligned/defiant), the legibility M4.9 and the renderer build on."""
    print("=" * 56)
    print("FAITHS (world_state['faiths'], M4.8)")
    print("=" * 56)
    faiths = world_state.get("faiths", {})
    if not faiths:
        print("(no faiths have formed)")
        print()
        return
    for fid in sorted(faiths):
        f = faiths[fid]
        core = ", ".join(f"'{b}'" for b in sorted(f["core"]))
        n = sum(1 for m in f["followers"]
                if any(a.name == m and a.alive for a in world_state["agents"]))
        print(f"{f['name']}")
        print(f"    core: {core}")
        print(f"    followers: {n}   prophet: {f['prophet'] or '(none)'}")
        for sid in sorted(f["settlements"]):
            align = religion.ruler_alignment(world_state, sid)
            sovereign = religion._sovereign(world_state, sid)
            if sovereign is not None:
                print(f"    {sid}: ruler {sovereign} is {align.upper()}")
            else:
                print(f"    {sid}: no ruler")
    print()


def print_events_log() -> None:
    """Day 17: dump the full chronological events[] log (deaths, respawns, [GOD]
    interventions). Printed at end-of-run so a captured log shows cause->effect in
    one place — every god intervention is here next to the deaths it caused.
    """
    print("=" * 56)
    print("EVENTS LOG (world_state['events'])")
    print("=" * 56)
    events = world_state["events"]
    if not events:
        print("(no events recorded)")
    for e in events:
        print(e)
    print()


def print_inference_savings(counters: dict[str, int]) -> None:
    """Phase 4 evidence: how much strategy caching reduced LLM calls."""
    stats = get_call_stats()
    agent_turns = counters["agent_turns"]
    strat_calls = stats["strategy"]
    saved = agent_turns - strat_calls
    pct = (100 * saved / agent_turns) if agent_turns else 0.0
    print("=" * 56)
    print("INFERENCE COST (strategy caching)")
    print("=" * 56)
    print(f"Agent-turns executed:      {agent_turns}")
    print(f"LLM strategy calls made:   {strat_calls}")
    print(f"Per-turn design would use: {agent_turns} LLM calls")
    print(f"Saved by caching:          {saved} calls (~{pct:.0f}% fewer)")
    print()


# --- Day 17: reproducibility + run capture --------------------------------
class _Tee:
    """Duplicate writes to several streams at once (used to mirror stdout to a log).

    Presentation only — capturing the run never touches world_state or the loop. It
    just lets `--log` save exactly what the terminal shows, byte for byte.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def parse_god_script(spec: str | None) -> dict[int, list[str]]:
    """Parse a non-interactive god script into {turn: [command, ...]} (Day 17).

    Two accepted forms (same grammar):
      - inline:  "5:trigger_plague Bob;15:drop_treasure 5 5"
      - file:    a path whose lines are "<turn>:<command>" (blank lines and lines
                 starting with '#' are ignored).
    Each entry fires at the END of its turn — the same clean boundary the interactive
    God menu uses — so a scripted run reproduces a hand-played one exactly. Commands
    for the same turn run in listed order. Returns {} for an empty/None spec.
    """
    if not spec:
        return {}
    if os.path.isfile(spec):
        with open(spec) as f:
            raw = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    else:
        raw = [part.strip() for part in spec.split(";") if part.strip()]

    script: dict[int, list[str]] = {}
    for entry in raw:
        if ":" not in entry:
            raise ValueError(f"bad god-script entry {entry!r} (expected '<turn>:<command>')")
        turn_str, command = entry.split(":", 1)
        try:
            turn = int(turn_str.strip())
        except ValueError:
            raise ValueError(f"bad god-script turn in {entry!r} (must be an integer)")
        script.setdefault(turn, []).append(command.strip())
    return script


# Day 19: named pacing presets for --speed, in SECONDS of pause between rendered
# turns. Presentation only — the pause is applied AFTER a turn is fully resolved and
# drawn, so it never touches world_state, the RNG, or what a log captures.
_SPEED_PRESETS = {"slow": 2.0, "normal": 0.5, "fast": 0.1}


def parse_speed(value: str) -> float:
    """Map a --speed value to a per-turn delay in seconds (Day 19).

    Accepts a named preset (slow/normal/fast) or a raw non-negative number for fine
    control (e.g. "0.3"). Raises argparse.ArgumentTypeError on anything else so the
    CLI reports a clean error. The returned delay only ever paces a RENDERED run.
    """
    if value in _SPEED_PRESETS:
        return _SPEED_PRESETS[value]
    try:
        secs = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"--speed must be one of {sorted(_SPEED_PRESETS)} or a number of seconds, "
            f"got {value!r}")
    if secs < 0:
        raise argparse.ArgumentTypeError(f"--speed seconds must be >= 0, got {secs}")
    return secs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI for a reproducible, capturable run (Day 17)."""
    p = argparse.ArgumentParser(
        prog="main.py", description="AI Civilization — multi-agent survival simulation.")
    p.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for a REPRODUCIBLE run. Seeds Python's `random`, which drives "
             "agent/food placement AND the offline 'random' provider, so the same seed "
             "replays an identical run offline. NOTE: the Qwen/Ollama LLM is NOT fully "
             "deterministic even with a seed (sampling temperature), so a seed fixes the "
             "WORLD setup but Qwen-driven turns may still vary slightly. "
             "Falls back to the AICIV_SEED env var.")
    p.add_argument(
        "--turns", type=int, default=None,
        help=f"number of turns to simulate (default {NUM_TURNS}).")
    p.add_argument(
        "--log", metavar="PATH", default=None,
        help="capture the full run (turn-by-turn log + final summary + events[] log, "
             "including god interventions) to PATH as well as stdout.")
    p.add_argument(
        "--god-script", metavar="SPEC", default=None,
        help="run god commands non-interactively. SPEC is either inline "
             "\"5:trigger_plague Bob;15:drop_treasure 5 5\" or a path to a file of "
             "'<turn>:<command>' lines. Each fires at the end of its turn.")
    p.add_argument(
        "--god-every", type=int, default=None,
        help="drop into the interactive God menu every N turns (default from "
             "AICIV_GOD_EVERY, else off). Ignored when --god-script is given.")
    p.add_argument(
        "--cognition", choices=("llm", "heuristic"), default=None,
        help="the BASELINE mind agents start with (V2 M0.1). Default 'llm' for the "
             "trio (exactly as V1), 'heuristic' for a large --agents cast (the focal "
             "budget then promotes the interesting few). 'heuristic' uses a pure-Python "
             "survival policy that makes ZERO model calls. Under M0.2 tiering this is "
             "just the starting tier — the per-turn focal budget governs who actually "
             "runs the LLM mind (see --focal-budget).")
    p.add_argument(
        "--agents", type=int, default=None, metavar="N",
        help="V2 M0.3 scale: run N procedurally-generated agents (mixed personalities) "
             "instead of the default 3-agent trio. The world auto-scales to match "
             "(grid size and food economy sized to N; see --grid-size to override), and "
             "the cast defaults to the heuristic mind with the focal budget on top. "
             "Built for 100-300; small N still uses the named trio.")
    p.add_argument(
        "--grid-size", type=int, default=None, metavar="S",
        help="force the world to an S x S grid (default: 10 for the trio, or "
             f"auto-scaled to keep agent density ~{SCALE_DENSITY} agents/cell for a "
             "large --agents cast).")
    p.add_argument(
        "--seed-knowledge", action="append", default=None, metavar="ITEM[:N]",
        help="V2 M1.1: seed a knowledge ITEM into the first N agents at setup (N "
             "default 1), then watch it SPREAD through contact (knowledge.diffuse, "
             "zero LLM cost). Repeatable. Example: --seed-knowledge fire seeds 'fire' "
             "into one agent. With no --seed-knowledge the run is byte-identical to v1.")
    p.add_argument(
        "--tech-tree", action="store_true",
        help="V2 M1.2: enable unscripted DISCOVERY. Each turn fed/curious agents may "
             "probabilistically invent items whose prerequisites they know "
             "(fire -> tools/cooking -> farming); a discovery then spreads via M1.1. "
             "Zero LLM cost. Off by default, so the run stays byte-identical to v1.")
    p.add_argument(
        "--settlements", action="store_true",
        help="V2 M2.1: enable SETTLEMENTS. Nomads become settlers where reliable "
             "(farmed) food makes staying worthwhile — a persistent settlement EMERGES "
             "when enough agents sustain themselves near the same food, and its members "
             "gain a gentle home-pull (survival still overrides). Zero LLM/RNG cost. Off "
             "by default, so the run stays byte-identical to v1. Pair with --tech-tree or "
             "--seed-knowledge farming so reliable food actually exists.")
    p.add_argument(
        "--storage", action="store_true",
        help="V2 M2.2: enable STORAGE & SURPLUS. Settled, well-fed members bank surplus "
             "food into a PERSONAL stockpile whose size EMERGES from personality (a "
             "competitive agent hoards, a friendly one banks less) and farming knowledge "
             "(a producer accumulates faster) — so wealth inequality emerges. The "
             "stockpile is a SURVIVAL BUFFER: a member that would otherwise starve draws "
             "on it, so the wealthy weather a drought the poor don't. Needs settlements "
             "to do anything (only the settled store); pair with --settlements --tech-tree "
             "(or --seed-knowledge farming). Zero LLM/RNG. Off by default -> v1 identical.")
    p.add_argument(
        "--economy", action="store_true",
        help="V2 M2.3: enable the ECONOMY — TRADE, food-backed MONEY, and proprietary "
             "knowledge (closes Phase 2). Settled agents mint money from food surplus past "
             "the storage cap, then TRADE food/knowledge <-> money with nearby agents at an "
             "EMERGENT price (buyer hunger up, seller surplus down, rarer skills dearer); "
             "competitive agents GUARD skills and sell them while friendly agents still teach "
             "free (M1.1). Pairs with --tech-tree (hunting is a second producer skill); "
             "implies --settlements + --storage. Zero LLM/RNG. Off by default -> v1 identical. "
             "(Wage-labor and minted/fiat money are out of scope, deferred to Phase 3.)")
    p.add_argument(
        "--labor", action="store_true",
        help="V2 M3.1: enable WAGE LABOR — the first institution (opens Phase 3). Rich settled "
             "producers EMPLOY poor, unskilled have-nots at a wage that EMERGES from the labor "
             "market (scarce labor -> high wage; an abundant desperate pool -> near-subsistence "
             "exploitation); the worker's output accrues to the employer, so inequality "
             "COMPOUNDS over time. Roles fall out of existing wealth/skill — nothing is assigned. "
             "Implies --economy (and so --settlements --storage). Zero LLM/RNG. Off by default -> "
             "v1 identical. (Governance/law/revolt and fiat money are out of scope, Phase 3+.)")
    p.add_argument(
        "--leadership", action="store_true",
        help="V2 M3.2: enable LEGITIMATE LEADERSHIP — the first POLITICAL institution and the "
             "first power NOT downstream of wealth. A leader EMERGES as the centre of a coherent "
             "TRUST cluster within a settlement (>= MIN_FOLLOWERS co-settlers trust a common "
             "agent above the trust bar) — never a global-max lookup, and NONE emerges in a "
             "fractured low-trust settlement. Legitimacy is CONTINGENT: the role is lost when the "
             "following erodes (with hysteresis) and a more-trusted centre can displace it. A "
             "leader COORDINATES its followers (a tighter home-pull) — INFLUENCE only, not "
             "tax/law. PURE read of the v1 trust system (writes no trust). Implies --settlements "
             "(needs a settlement to lead). Zero LLM/RNG. Off by default -> v1 identical. "
             "(Taxation, law and revolt are out of scope, later Phase 3.)")
    p.add_argument(
        "--taxation", action="store_true",
        help="V2 M3.3: enable TAXATION & REDISTRIBUTION — the COLLISION of the M3.1 class engine "
             "and the M3.2 legitimacy engine, the first force that BENDS the inequality spiral. A "
             "legitimate leader (M3.2) taxes its wealthy followers' wealth above a threshold and "
             "redistributes to its poor ones, lowering the within-settlement Gini. ONLY a led "
             "settlement can tax (power downstream of legitimacy, not wealth). Over-taxation costs "
             "the leader trust and SELF-LIMITS: it erodes the taxed below M3.2's keep bar, the "
             "following collapses and the leader loses legitimacy (and the power to tax) — consent "
             "of the governed, emergent. Conserves wealth (redistribution, not minting). Implies "
             "--leadership (and so --settlements); pair with --labor for the inequality to bend. "
             "Set the levy with --tax-rate. Zero LLM/RNG. Off by default -> v1 identical. "
             "(Law/legislation, revolt and fiat money are out of scope, later Phase 3.)")
    p.add_argument(
        "--tax-rate", type=float, default=None, metavar="R",
        help=f"V2 M3.3 levy (with --taxation): the fraction of a wealthy follower's wealth ABOVE "
             f"the threshold taxed each turn (default {taxation.DEFAULT_TAX_RATE}). At or below "
             f"the consent band ({taxation.CONSENT_RATE} of the surplus) taxation is tolerated and "
             f"SUSTAINED; well above it the backlash erodes the leader's legitimacy until taxing "
             f"stops.")
    p.add_argument(
        "--monarchy", action="store_true",
        help="V2 M3.4: enable CONQUEST & MONARCHY — the SECOND source of power, DOMINATION by "
             "force (vs M3.2's consent). A wealthy aspirant SPENDS money to muster an army of real "
             "fighters (poor agents who fight for pay) and ATTACKS a settlement to seize it, "
             "becoming MONARCH — a persistent title held by FORCE. A fight is resolved on MUSTERED "
             "force (attacker's bought army vs the defenders: a monarch's garrison, else a trusted "
             "leader's loyal FOLLOWERS, else the militia), so loyalty can repel a richer-but-smaller "
             "attacker while an overwhelming bought force wins — and war KILLS agents. A monarch "
             "levies wealth WITHOUT consent (contrast M3.3); the crown is LOSABLE to a stronger "
             "later army. Implies --settlements (needs a town to seize); pair with --economy/--labor "
             "for the wealth that funds armies and --leadership for the loyalty that defends. Zero "
             "LLM/RNG, deterministic under seed. Off by default -> v1 identical. (Underclass revolt, "
             "inter-kingdom war and multi-settlement kingdoms are out of scope, later Phase 3.)")
    p.add_argument(
        "--kingdoms", action="store_true",
        help="V2 M3.5: enable KINGDOMS & VASSALAGE (feudalism) — the SCALE-UP of M3.4. A MONARCH "
             "conquers NEIGHBOURING settlements into a multi-settlement REALM, a two-level feudal "
             "hierarchy (KING -> VASSAL LORDS -> their settlements). A conquered local ruler keeps "
             "ruling as a VASSAL (local autonomy); a ruler-less town is held directly. TRIBUTE "
             "cascades UP (members -> vassal -> king); vassals owe military SERVICE (the king's host "
             "= its force + loyal vassals' forces); loyalty is CONDITIONAL — heavy tribute erodes a "
             "vassal's trust (M3.3-shape backlash) and a pushed vassal BREAKS AWAY (with hysteresis) "
             "while a fairly-treated one stays. Pooled tribute funds further conquest, so the map "
             "CONSOLIDATES. Implies --monarchy + --settlements; pair with --economy/--labor for "
             "army-funding wealth and --leadership for vassalable trust-leaders. Set the crown's "
             "share with --tribute-rate. Zero LLM/RNG, deterministic under seed. Off by default -> "
             "v1 identical. (Baronial civil war, defection to a rival king and rebellion cascades "
             "are out of scope, later Phase 3.)")
    p.add_argument(
        "--tribute-rate", type=float, default=None, metavar="R",
        help=f"V2 M3.5: the KING's share of a vassal's tribute that cascades up each turn (default "
             f"{kingdoms.DEFAULT_KING_SHARE}). At or below the consent band ({kingdoms.KING_CONSENT}) "
             f"a vassal tolerates the crown and stays loyal; well above it the backlash erodes the "
             f"vassal's trust until it BREAKS AWAY. Only meaningful with --kingdoms.")
    p.add_argument(
        "--empire", action="store_true",
        help="V2 M3.6: enable INTER-KINGDOM WAR & EMPIRE (the CLIMAX of Phase 3) — feudal KINGDOMS "
             "(M3.5) clash. A king opportunistically attacks a NEIGHBOURING kingdom whose defendable "
             "LOYAL host he can beat; each side musters its WHOLE host (king + LOYAL vassals + loyal "
             "subject-kings) and the SAME battle resolves it (war kills agents on both sides). The "
             "KEY coupling: war turns on LOYAL host strength, so a RICHER kingdom with disloyal "
             "vassals fields a smaller host and LOSES to a POORER one whose vassals all muster — "
             "governance beats wealth. The defeated king is SUBJUGATED into the victor's realm as a "
             "high-level vassal (a subject-king), forming a multi-level EMPIRE (emperor -> subject-king "
             "-> vassal-lords -> settlements); tribute cascades through the new level. Empires FRAGMENT: "
             "a subject-king's loyalty is conditional, so an over-taxing or weakening emperor loses him "
             "to a breakaway (with hysteresis) — empires rise AND fall. Implies --kingdoms (+ --monarchy "
             "+ --settlements); pair with --economy/--labor for army-funding wealth and --leadership for "
             "vassalable trust-leaders. Set the emperor's cut with --imperial-share. Zero LLM/RNG, "
             "deterministic under seed. Off by default -> v1 identical. (Vassal defection to the enemy, "
             "diplomacy/alliances and fiat money are out of scope, later phases.)")
    p.add_argument(
        "--imperial-share", type=float, default=None, metavar="R",
        help=f"V2 M3.6: the EMPEROR's share of a subject-king's wealth taken as imperial tribute each "
             f"turn (default {empire.DEFAULT_EMPIRE_SHARE}). At or below the consent band "
             f"({kingdoms.KING_CONSENT}) a subject-king tolerates the emperor and stays loyal; well "
             f"above it the backlash erodes his trust until he BREAKS AWAY and the empire FRAGMENTS. "
             f"Only meaningful with --empire.")
    p.add_argument(
        "--lineage", action="store_true",
        help="V2 M4.1: enable LINEAGE — birth, childhood, aging and family (opens Phase 4). "
             "Settled pairs with MUTUAL high trust, both fed, in a settlement holding a food "
             "SURPLUS, bear CHILDREN who inherit a BLEND of their parents' temperament (small "
             "seeded jitter; knowledge/wealth are NOT inherited — children learn via M1.1 "
             "diffusion at a boosted childhood rate). A child is a DEPENDENT for ~16 turns: it "
             "produces nothing and is fed from its parents' stockpiles (family size gated by "
             "wealth). EVERYONE ages and dies of OLD AGE at a varied seeded lifespan, so "
             "generations TURN. Births become the population engine (bounded by a cap, "
             "food-gated -> Malthusian); the Day 14 respawn becomes extinction insurance only "
             "(its existing below-3 gate). Pair with --settlements --tech-tree/--seed-knowledge "
             "farming (births need a settlement + surplus) and --storage (stockpiles feed "
             "children). Zero LLM; deterministic under seed. Off by default -> byte-identical. "
             "(Wealth inheritance at death is M4.2; dynastic succession of titles is M4.3 — "
             "not built here.)")
    p.add_argument(
        "--discontent", action="store_true",
        help="V2 M4.4: enable DISCONTENT — a legible per-agent CLASS-PRESSURE gauge (opens Arc 2: "
             "Revolt). Each turn, for every settled adult, three grievance drivers ADD to a personal "
             "discontent number, all DERIVED from existing state (no new psychology): DEPRIVATION amid "
             "plenty (hunger felt beside a rich settlement-mate — the strongest), EXPLOITATION (a "
             "subsistence M3.1 wage), and EXTRACTION (being levied by a monarch / tributed up a "
             "kingdom), the last weighted by BURDEN relative to means AND buffered by LEGITIMACY (the "
             "agent's TRUST in its ruler — a consented tax stings far less than a hated levy). The "
             "gauge RISES fast while a grievance persists and DECAYS slowly once relieved (hysteresis "
             "— grievances outlast their causes); floored at 0, soft-capped. Per-agent discontent "
             "shows in the AGENT SUMMARY and settlement PRESSURE (the resentful-faction size) in a "
             "SETTLEMENT PRESSURE summary. MEASURE ONLY — no uprising fires here (that is M4.5, which "
             "will trigger on this pressure). Zero LLM, zero new RNG. Off by default -> byte-identical.")
    p.add_argument(
        "--uprising", action="store_true",
        help="V2 M4.5-M4.6: enable UPRISING + THE REVOLUTIONARY — the relief valve that makes M4.4's "
             "gauge BLOW, and the leader it throws up (implies --discontent and --leadership). When a "
             "settlement's RESENTFUL faction becomes a MAJORITY of its non-ruler "
             "members AND their aggregate discontent clears a floor, the poor RISE against their "
             "FORCE ruler (monarch/vassal-lord — a consent-based trust-leader is never a target). The "
             "MOB's force is its NUMBERS (unpaid, unarmed — the inverse of M3.4, where force is BOUGHT); "
             "the RULER's defence is his garrison/followers PLUS any mercenaries his war chest can still "
             "MUSTER, so a rich tyrant CRUSHES the rising while a drained one FALLS to numbers (wealth is "
             "the counter-revolutionary weapon). The SAME monarchy.resolve_battle decides it (real deaths "
             "via the normal path, so M4.2 inheritance / M4.3 succession compose). On victory the ruler is "
             "DEPOSED (title cleared, kingdom secedes via the existing machinery), his hoard is "
             "EXPROPRIATED to the risers (interrupting inheritance — the heirs get nothing), and the "
             "grievance is answered; on defeat the survivors are cowed (partial reset + fear cooldown, "
             "grievance persists). M4.6 THE REVOLUTIONARY then fills the vacant seat: the rising's LEADER "
             "— derived from the risers (the angriest COMMONER his fellows most trust, not the richest) — "
             "takes power by CONSENT through the existing M3.2 leadership path (a leader, NOT a monarch; "
             "losable like any leader), closing the cycle consent->force->revolt->consent. Pair with "
             "--monarchy/--kingdoms (uprisings need force rulers). Zero LLM, zero new RNG. Off by default "
             "-> byte-identical.")
    p.add_argument(
        "--beliefs", action="store_true",
        help="V2 M4.7: enable BELIEFS — an inner life (opens Arc 3: Belief & Culture). Agents FORM short, "
             "fixed worldview strings from lived EXPERIENCE (e.g. 'the land provides' after sustained "
             "abundance, 'the world is cruel' after starvation, 'the strong take what they want' after "
             "turns under a force ruler, 'the dead watch us' after witnessing many deaths) — earned by a "
             "threshold on real state, never assigned or generated. Beliefs then SPREAD one hop along the "
             "contact network with the SAME trust-weighted, personality-shaped, child-boosted probability "
             "knowledge uses (M1.1) — a trusted neighbour's belief catches, a stranger's rarely; a "
             "CONTRADICTORY belief flips the old one only from a trusted source; children soak up their "
             "parents' beliefs through the childhood window. So settlements grow distinct DOMINANT belief "
             "sets shaped by what they have SUFFERED — proto-cultures, shown per-agent in the AGENT "
             "SUMMARY and per-settlement in a BELIEF CULTURES readout. Inert culture only — beliefs confer "
             "no power yet (religion is M4.8, cultural friction M4.9). Zero LLM (beliefs are STATE, not "
             "generated text). Off by default -> byte-identical.")
    p.add_argument(
        "--religion", action="store_true",
        help="V2 M4.8: enable RELIGION — shared belief becomes POWER (implies --beliefs). When a "
             "settlement's members share a coherent CORE of 2+ beliefs (a majority holding each), those "
             "beliefs crystallise into a named FAITH; towns with the same core share one faith, divergent "
             "cores form different faiths. Each faith may raise a PROPHET — its most DEVOUT-and-TRUSTED "
             "follower (derived from state, not assigned; none if no such figure). Faith then touches "
             "LEGITIMACY by moving TRUST only: believers extend extra trust to a ruler ALIGNED with their "
             "faith (who holds its core) and withdraw it from a DEFIANT one (a conqueror of a different-"
             "faith town), a prophet amplifying the stance — so the SAME extraction breeds LESS discontent "
             "under an aligned crown and MORE under a defiant one, and a defied king's vassals erode toward "
             "BREAKAWAY (M3.5) / his towns toward REVOLT (M4.5) through the UNCHANGED machinery. No new "
             "political mechanic; faith just moves the dial the existing systems read. Faiths/prophets/"
             "alignment show in the summaries. Theocracy (a prophet ruling) and cultural friction (M4.9) "
             "are NOT built here. Pair with --monarchy/--kingdoms/--discontent to see the teeth. Zero LLM, "
             "zero RNG. Off by default -> byte-identical.")
    p.add_argument(
        "--culture", action="store_true",
        help="V2 M4.9: enable CULTURAL IDENTITY & FRICTION — the imperial problem, CLOSING Arc 3 "
             "(implies --religion -> --beliefs). A settlement's CULTURE is its dominant belief set "
             "(M4.7) + faith (M4.8). Conquering a SAME-culture town integrates almost frictionlessly; "
             "conquering a FOREIGN one breeds CHRONIC unrest — every turn under an alien-culture ruler, "
             "its members withdraw loyalty (SUSTAINED), which the UNCHANGED M4.4/M3.5/M4.5 machinery "
             "turns into hotter discontent, likelier BREAKAWAY, and REVOLT (no separate 'independence "
             "movement' — cultural revolt EMERGES from the existing systems). And it only becomes his "
             "over GENERATIONS: dependent CHILDREN raised in the town adopt the RULER's beliefs at the "
             "childhood rate while adults keep theirs, so the town's culture DRIFTS toward the ruler's "
             "as generations turn — a RACE between assimilation (the fault line fades) and revolt (it "
             "fractures the empire), neither scripted; lose the ruler first and the town reverts (drift "
             "persists in the assimilated). Per-settlement culture/foreign-rule/assimilation show in the "
             "summary. Zero LLM (culture is STATE); friction deterministic, assimilation seeded. Off by "
             "default -> byte-identical.")
    p.add_argument(
        "--writing", action="store_true",
        help="V2 M4.10: enable WRITING & RECORDS — institutional memory (opens Arc 4, the road to "
             "modernity; implies --tech-tree). WRITING is a tech: an agent invents it only with the prior "
             "tech TOOLS and from a SETTLEMENT holding a food SURPLUS (scribes need stability), through "
             "the EXISTING M1.2 discovery + M1.1 diffusion (unscripted, seeded). A settlement that becomes "
             "LITERATE gains three powers of the written word: PERSISTENT LAW (a ruler's tax/levy policy is "
             "inscribed and the M4.3 heir INHERITS the written framework instead of a blank slate — an "
             "illiterate town's policy dies with its ruler); KNOWLEDGE PRESERVATION (archived techs are "
             "RE-TAUGHT if their last living master dies, so a literate civilization cannot forget farming "
             "— the cure for the knowledge-extinction collapse an illiterate town still suffers); and "
             "RECORDED HISTORY (major events append to a persistent settlement CHRONICLE — structured "
             "entries, the substrate Arc 6 will read). LAW is minimal here (written policy survives "
             "succession; NO courts/enforcement). Literacy/laws/chronicle lengths show in the summary. "
             "Zero LLM (records are STATE); only discovery draws RNG. Off by default -> byte-identical.")
    p.add_argument(
        "--metallurgy", action="store_true",
        help="V2 M4.11: enable METALLURGY & ARMS — technology transforms war and work (Arc 4; implies "
             "--tech-tree). A small tech chain past TOOLS — metalworking (better tools) then weapons "
             "(arms) — invented from a SETTLEMENT with food SURPLUS through the EXISTING M1.2 discovery "
             "+ M1.1 diffusion (unscripted, seeded). TWO effects: (1) ECONOMY — a farmer who knows "
             "metalworking grows food at double the base yield, so a metallurgical town out-produces a "
             "neolithic one; (2) ARMS — an ARMED combatant (knows weapons) counts ~1.8x an unarmed head "
             "in the shared battle math, so a SMALLER armed host beats a LARGER unarmed one everywhere "
             "battles resolve (conquest, inter-kingdom war, AND uprising). Because arms spread by tech "
             "diffusion, WHO knows weapons is decisive: an armed ruler's garrison CRUSHES an unarmed "
             "peasant mob (steel beats numbers), but a mob that ALSO learned weapons wins on numbers "
             "again (revolt survives in armed societies). Era progression (Bronze/Iron) is M4.12, NOT "
             "here. Zero LLM; only discovery draws RNG. Off by default -> byte-identical.")
    p.add_argument(
        "--stage", choices=("monarchy", "kingdom", "war"), default=None,
        help="DEMO SCENARIO STAGING (default off): set up a starting scene so the verified "
             "M3.4-M3.6 conquest-chain visuals can be WATCHED (organic runs almost never produce "
             "rulers). It STAGES, it does not fake: it positions agents/wealth and runs the EXISTING "
             "code paths — 'monarchy' has a wealthy aspirant SEIZE a town via monarchy.attempt_"
             "conquest (a real MONARCH + castle); 'kingdom' then CONQUERS neighbours via kingdoms."
             "conquer_neighbour (a real king -> vassal-lords kingdom); 'war' sets up TWO adjacent "
             "rival kingdoms (one stronger) so the normal empire.update opportunistic-war logic "
             "CLASHES them and an EMPIRE forms on screen. Implies the matching institutions + "
             "--settlements, owns the cast, and sizes the world. RNG-free staging -> reproducible "
             "under --seed. Pair with --pygame to watch. (Off -> byte-identical to before.)")
    p.add_argument(
        "--focal-budget", type=int, default=None, metavar="N",
        help="V2 M0.2 tiered cognition: the MAX number of agents that may run the "
             f"expensive LLM mind at once (default {DEFAULT_FOCAL_BUDGET}, or 0 when "
             "--cognition heuristic). Each turn the most socially/strategically "
             "interesting N living agents are promoted to 'focal' (LLM) and the rest "
             "run the zero-LLM heuristic mind, so inference cost scales with DRAMA, not "
             "population. N >= the cast => everyone focal => byte-identical to v1; N=0 "
             "=> everyone heuristic (the M0.1 zero-LLM run).")
    p.add_argument(
        "--render", choices=("plain", "rich"), default="plain",
        help="output style. 'plain' (default) is the unchanged turn-by-turn text "
             "print. 'rich' shows a live in-place dashboard (grid + per-agent status "
             "+ event log) via the `rich` library. With --render rich the dashboard "
             "owns the terminal during the run and the plain per-turn text is "
             "suppressed there; under --log that plain text is still captured to the "
             "log file byte-for-byte, and the end-of-run summary prints to both.")
    p.add_argument(
        "--speed", type=parse_speed, default=_SPEED_PRESETS["normal"], metavar="SPEED",
        help="pacing for a RENDERED run: slow (~2.0s/turn), normal (~0.5s/turn, "
             "default), fast (~0.1s/turn), or a raw number of seconds (e.g. 0.3). The "
             "pause is presentation-only — it applies ONLY with a renderer (--render "
             "rich / --pygame), after each turn is drawn, and never affects tests, "
             "plain/logged runs, or the seeded RNG. Demo: --render rich --speed slow.")
    p.add_argument(
        "--pygame", action="store_true",
        help="VISUAL renderer (Pygame, SLICE 1): open a window and watch the world — a "
             "muted terrain grid with food dots and one circle per LIVING agent, COLOURED "
             "by personality and SIZED by wealth (richer = bigger). READ-ONLY: it draws "
             "world_state and writes nothing back, reusing the real per-turn sim loop (it "
             "never advances the world itself). SPACE pauses, ESC / window-close quits; "
             "--speed sets the pace. Honours the world-setup flags (--agents, --grid-size, "
             "--seed, institution flags). Optional dependency: if Pygame is not installed "
             "the run exits with a 'pip install pygame' message. (Settlements, rulers, "
             "kingdoms and war are later slices — this slice draws only terrain + agents.)")
    return p.parse_args(argv)


def run_simulation(num_turns: int, *, god_script: dict[int, list[str]] | None = None,
                   god_every: int = 0, renderer: "Any" = None,
                   turn_delay: float = 0.0,
                   agent_specs: "list | None" = None,
                   cognition: str = "llm",
                   focal_budget: "int | None" = None,
                   grid_size: "int | None" = None,
                   food_cfg: "dict | None" = None,
                   knowledge_seed: "list | None" = None,
                   tech_tree: "dict | None" = None,
                   settlements: bool = False,
                   storage_on: bool = False,
                   economy_on: bool = False,
                   labor_on: bool = False,
                   leadership_on: bool = False,
                   taxation_on: bool = False,
                   tax_rate: "float | None" = None,
                   monarchy_on: bool = False,
                   kingdoms_on: bool = False,
                   tribute_rate: "float | None" = None,
                   empire_on: bool = False,
                   empire_share: "float | None" = None,
                   stage: "str | None" = None,
                   lineage_on: bool = False,
                   discontent_on: bool = False,
                   uprising_on: bool = False,
                   beliefs_on: bool = False,
                   religion_on: bool = False,
                   culture_on: bool = False,
                   writing_on: bool = False,
                   metallurgy_on: bool = False) -> None:
    """The setup + shared survival loop + end-of-run analysis (Day 17 extracted).

    Pulled out of main() so the exact production loop can be driven head-less with an
    explicit turn count and an optional non-interactive god script. The caller is
    responsible for seeding `random` BEFORE calling this (so world setup is part of the
    reproducible sequence) and for any stdout capture.

    Day 18: an optional `renderer` (renderer.RichRenderer) draws a live dashboard from
    world_state after each turn. When given, the dashboard owns the terminal and the
    plain per-turn prints are redirected to `renderer.sink` (the log file under --log,
    else os.devnull) so they never scroll over the dashboard but are still captured.
    The renderer ONLY READS world_state — it cannot affect the simulation, so a run is
    byte-identical with or without it (the plain text is merely routed elsewhere). When
    `renderer is None` the path is exactly the pre-Day-18 plain behaviour.

    V2 M0.2 (tiered cognition): when `focal_budget` is not None, each turn begins by
    re-assigning the focal (LLM) set via cognition.update_tiers — the most interesting
    `focal_budget` living agents run the LLM mind, the rest the heuristic mind, with
    hysteresis so the set doesn't thrash. `focal_budget is None` (the default for direct
    callers/tests) DISABLES tiering, leaving every agent on its setup `cognition` — so
    pure-v1 and pure-M0.1 runs are untouched. When `focal_budget >= len(living)` the
    update promotes everyone and logs nothing, keeping a small cast byte-identical to v1.

    V2 M0.3 (scale): `grid_size` sizes the world (None -> the v1 10x10 default) and
    `food_cfg` ({"initial", "per_turn", "cap", "cluster"}) drives a population-scaled
    food economy (None -> the v1 INITIAL_FOOD + maybe_respawn_food constants). Both
    None is the exact pre-M0.3 path, so the default run is byte-for-byte unchanged.

    V2 M1.1 (knowledge): `knowledge_seed` is a list of (item, count) pairs — each item
    is granted to the first `count` agents at setup, after which it spreads ONLY through
    contact via knowledge.diffuse (called once per turn). None / empty seeds nothing, so
    diffusion self-gates to a no-op and the run is byte-identical to v1.

    V2 M1.2 (discovery): `tech_tree` ({item -> frozenset(prereqs)}) enables unscripted
    invention — each turn knowledge.discover lets fed/curious agents probabilistically
    invent items whose prereqs they know; the new item then spreads via the same
    diffusion. None / empty -> no discovery (zero RNG) -> v1 byte-identical.
    """
    god_script = god_script or {}

    # --- Setup ----------------------------------------------------------
    reset_call_stats()
    # M0.3: a large cast needs a bigger world; grid_size None keeps the v1 10x10.
    create_world(size=grid_size) if grid_size is not None else create_world()
    # M2.2: record the storage opt-in on world_state so the read-side switches (the
    # survival-buffer draw-down in run_agent_turn and the wealth overlay in the summary)
    # see it. Surplus accumulation is gated separately at its call site below. With this
    # False (the default) neither read fires, so the run is byte-identical to v1.
    world_state["storage_on"] = storage_on
    # M2.3: same for the economy (money minting + trade pass + proprietary-knowledge guarding
    # in diffuse + money-redemption in the survival buffer). False (default) -> none fire ->
    # the run is byte-identical to v1.
    world_state["economy_on"] = economy_on
    # M3.1: the wage-labor institution flag. False (default) -> labor.update never called and no
    # employment links form -> byte-identical to v1.
    world_state["labor_on"] = labor_on
    # M3.2: the legitimate-leadership institution flag. False (default) -> leadership.update never
    # called, leaders stays empty, the home-pull is never retargeted -> byte-identical to v1.
    world_state["leadership_on"] = leadership_on
    # M3.3: the taxation institution flag + rate. False (default) -> taxation.update never called
    # -> byte-identical to v1. The rate (when given) is the levy on follower wealth above the
    # threshold; None keeps the documented default already on world_state.
    world_state["taxation_on"] = taxation_on
    if tax_rate is not None:
        world_state["tax_rate"] = tax_rate
    # M3.4: the conquest/monarchy institution flag. False (default) -> monarchy.update never called
    # -> no armies, no crowns, no battle deaths -> byte-identical to v1.
    world_state["monarchy_on"] = monarchy_on
    # M3.5: the kingdoms/vassalage institution flag + tribute rate. False (default) -> kingdoms.update
    # never called, kingdoms stays empty -> byte-identical to v1. The rate (when given) is the king's
    # share of vassal tribute; None keeps the documented default already on world_state.
    world_state["kingdoms_on"] = kingdoms_on
    if tribute_rate is not None:
        world_state["tribute_rate"] = tribute_rate
    # M3.6: the inter-kingdom war / empire institution flag + imperial share. False (default) ->
    # empire.update never called, empires stays empty -> byte-identical to v1. The share (when given)
    # is the emperor's cut of a subject-king's wealth; None keeps the documented default.
    world_state["empire_on"] = empire_on
    if empire_share is not None:
        world_state["empire_share"] = empire_share
    # M0.1: `cognition` ("llm" default, or "heuristic" for a zero-LLM mind) is stamped
    # on every agent at setup, so `--cognition heuristic` runs the whole cast call-free
    # with no other change. `agent_specs` lets a harness (e.g. verify_m01) seed a custom
    # cast; absent it, the V1 trio is unchanged.
    specs = agent_specs if agent_specs is not None else AGENT_SPECS
    for name, personality, goals, (x, y) in specs:
        place_agent(Agent(name=name, personality=personality, goals=goals,
                          cognition=cognition), x, y)
    # M0.3: food_cfg drives a population-scaled economy; None keeps the v1 constants.
    if food_cfg is not None:
        spawn_food(food_cfg["initial"], cluster=food_cfg["cluster"])
    else:
        spawn_food(INITIAL_FOOD, cluster=FOOD_CLUSTERED)

    # M1.1: seed knowledge into the first `count` agents per (item, count). From here it
    # only spreads through contact (knowledge.diffuse). No seed -> diffusion is a no-op
    # and the run stays byte-identical to v1.
    if knowledge_seed:
        for item, count in knowledge_seed:
            for agent in world_state["agents"][:count]:
                knowledge.grant(world_state, agent, item, turn=0)

    # OPTIONAL SCENARIO STAGING (default None -> never imported -> byte-identical to before).
    # Sets up a monarchy/kingdom/war SCENE using the EXISTING verified institution code paths
    # (monarchy.attempt_conquest, kingdoms.conquer_neighbour) so the conquest-chain visuals can
    # be watched; the normal per-turn loop then runs from the staged state. RNG-free, so a staged
    # run stays reproducible under the seed. This only writes records the engine itself produces.
    if stage is not None:
        import scenario
        scenario.apply(world_state, stage, cognition=cognition)

    # M4.1: the lineage flag + founding-cast setup. Runs AFTER staging so a staged cast is
    # covered too. init_cast draws ages/lifespans from the seeded stream (part of the
    # reproducible sequence) and writes the lineage block (pop_cap/birth_seq) onto
    # world_state. False (default) -> init_cast/update never called, zero RNG drawn ->
    # byte-identical to v1 (respawn included — its code is untouched).
    world_state["lineage_on"] = lineage_on
    if lineage_on:
        lineage.init_cast(world_state)

    # M4.4 (opens Arc 2): the DISCONTENT flag — a legible per-agent pressure gauge derived from the
    # existing class signals (deprivation amid plenty, subsistence wages, buffered extraction). False
    # (default) -> discontent.update never called, no "discontent" key ever written to world_state ->
    # byte-identical to v1 (the v1 golden master included). MEASURE ONLY: it never acts on a ruler
    # (uprisings are M4.5). Zero LLM, zero new RNG.
    world_state["discontent_on"] = discontent_on

    # M4.5 (Arc 2): the UPRISING flag — the relief valve that CONSUMES M4.4's gauge (the resentful
    # poor rise against a force ruler). False (default) -> uprising.update never called, no cooldown
    # key ever written -> byte-identical to v1 (the golden master included). Implies discontent (no
    # pressure to blow without the gauge). Zero LLM, zero new RNG.
    world_state["uprising_on"] = uprising_on

    # M4.7 (opens Arc 3): the BELIEFS flag — an inner life of worldviews that FORM from lived
    # experience and SPREAD by trusted contact. False (default) -> beliefs.update never called, no
    # "beliefs" key ever written -> byte-identical to v1. Beliefs are inert culture here (no mechanical
    # power — that is M4.8 religion / M4.9 identity). Zero LLM; formation deterministic, spread reuses
    # the M1.1 diffusion RNG.
    world_state["beliefs_on"] = beliefs_on

    # M4.8 (Arc 3): the RELIGION flag — a coherent shared belief set becomes a FAITH with a PROPHET,
    # and faith touches LEGITIMACY (aligned rulers blessed, defiant ones resented — by MOVING TRUST the
    # existing M4.4/M3.5/M4.5 systems already read). False (default) -> religion.update never called, no
    # "faiths" key written -> byte-identical to v1. Implies beliefs (no faith without belief). Zero LLM/RNG.
    world_state["religion_on"] = religion_on

    # M4.9 (Arc 3 close): the CULTURE flag — cultural identity + conquest FRICTION + generational
    # ASSIMILATION. A settlement's culture is derived from its beliefs/faith; a foreign-culture ruler
    # breeds chronic unrest (by SUSTAINING the trust/discontent dials M4.4/M3.5/M4.5 read) and is
    # assimilated only as its children grow up in his culture (M4.1). False (default) -> culture.update
    # never called -> byte-identical to v1. Implies religion (-> beliefs). Zero LLM; friction
    # deterministic, assimilation draws seeded RNG.
    world_state["culture_on"] = culture_on

    # M4.10 (opens Arc 4): the WRITING flag — institutional memory. A settlement that becomes LITERATE
    # (invents/learns writing) gains persistent LAW (survives succession), knowledge PRESERVATION
    # (archived skills re-taught, curing knowledge-extinction) and recorded HISTORY (a persistent
    # chronicle). False (default) -> writing.update never called, no laws/archives/chronicles written ->
    # byte-identical to v1. Rides the tech tree (implies --tech-tree). Zero LLM; records are STATE.
    world_state["writing_on"] = writing_on

    # M4.11 (Arc 4): the METALLURGY flag — technology transforms war and work. A metallurgical settlement
    # (knows metalworking) out-produces a neolithic one, and ARMED combatants (know weapons) multiply
    # force in the shared battle math (conquest/war/uprising) so knowledge beats numbers. False (default)
    # -> metallurgy.update never called, no one learns metalworking/weapons -> byte-identical to v1 (the
    # farm boost and battle multiplier are both no-ops without the skills). Rides the tech tree. Zero LLM.
    world_state["metallurgy_on"] = metallurgy_on

    strategies: dict[str, Strategy] = {}
    survived: dict[str, int] = {a.name: 0 for a in world_state["agents"]}
    counters: dict[str, int] = {"agent_turns": 0}
    # M0.2: per-run hysteresis memory for the tiering system — {name: consecutive
    # turns spent focal}. Lives here (like `strategies`) so it is naturally fresh per
    # run and never pollutes world_state. Untouched when focal_budget is None.
    focal_tenure: dict[str, int] = {}

    if VERBOSE_MODE:
        print(f"AI Civilization — personality-driven simulation (provider: {PROVIDER})")
        print(f"Strategy refresh every {STRATEGY_INTERVAL} turns.")
        print(f"Agents: {', '.join(a.name for a in world_state['agents'])}")
        print()

    # --- The shared survival loop ---------------------------------------
    # Day 18: in rich mode the Live dashboard owns the terminal for the whole loop,
    # and the plain per-turn prints are redirected to renderer.sink (log or devnull)
    # so they don't scroll over it. Both context managers are no-ops when there is no
    # renderer, so the plain path below is byte-identical to before.
    live_cm = renderer.live() if renderer is not None else contextlib.nullcontext()
    sink_cm = (contextlib.redirect_stdout(renderer.sink)
               if renderer is not None else contextlib.nullcontext())
    with live_cm, sink_cm:
      # For a STAGED run, draw the starting scene ONCE before turn 1 so the viewer sees the
      # staged kingdoms/castles before the loop begins (e.g. before an inter-kingdom war fires).
      # Read-only and gated on `stage`, so non-staged runs are completely unaffected.
      if stage is not None and renderer is not None:
          renderer.update(world_state)
      for turn in range(1, num_turns + 1):
        world_state["turn"] = turn

        # M0.2: re-tier BEFORE anyone acts, so each agent's `cognition` reflects how
        # interesting it is RIGHT NOW (the events from last turn are in the window).
        # Disabled (no-op) when focal_budget is None; a no-transition no-op when the
        # budget covers the whole cast (keeps a small run byte-identical to v1).
        if focal_budget is not None:
            update_tiers(world_state, turn, focal_budget, focal_tenure)

        if VERBOSE_MODE:
            print("=" * 56)
            print(f"TURN {turn}  |  food on map: {len(world_state['food'])}")
            print(render(world_state))
            print()

        # Snapshot order at turn start so mid-turn deaths don't disturb iteration.
        actions: list[tuple[str, str]] = []
        for agent in [a for a in world_state["agents"] if a.alive]:
            action = run_agent_turn(agent, turn, strategies, survived, counters)
            actions.append((agent.name, action))

        if DEBUG_MODE:
            print(f"TURN {turn}")
            for name, action in actions:
                print(f"{name} -> {action}")
            print()
            print(f"Food remaining: {len(world_state['food'])}")
            print()

        # M1.2: agents may INVENT items whose prereqs they know (pure state math, ZERO
        # LLM calls) — runs first so a fresh discovery can begin spreading immediately.
        # No-op drawing no RNG when tech_tree is empty/None.
        knowledge.discover(world_state, turn, tech_tree)

        # M1.1: spread knowledge one hop along this turn's contact network (pure-Python
        # state diffusion, ZERO LLM calls). A no-op drawing no RNG when no agent knows
        # anything, so a v1 run with no seeded knowledge is byte-identical.
        knowledge.diffuse(world_state, turn)

        # M1.3: fed farmers PRODUCE food into world_state (the headline tech effect),
        # changing the food economy survival rides on. No-op drawing no RNG when nobody
        # is a fed farmer, so a v1 / no-farming run is byte-identical.
        knowledge.farm(world_state, turn)

        # M2.3 specialization: fed HUNTERS produce food too, by a different mechanic/location
        # (roaming game on a wider ring). Like farm() it is gated purely on KNOWING 'hunting'
        # and is a no-op drawing no RNG when nobody is a fed hunter — so a v1 / no-hunting run
        # stays byte-identical. Always called (not economy-gated): it is a knowledge effect.
        knowledge.hunt(world_state, turn)

        # M2.1: nomads become SETTLERS where reliable food makes staying worthwhile —
        # settlements EMERGE from the food economy (zero LLM, zero RNG, a deterministic
        # threshold on sustained presence). Runs AFTER farm() so this turn's freshly
        # grown food already counts. Gated on the opt-in `settlements` flag, so a default
        # run never calls it and stays byte-identical to v1.
        if settlements:
            settlement.update(world_state, turn)

        # M2.2: settled, well-fed members beside surplus food BANK it into a personal
        # stockpile — wealth that emerges from personality + farming knowledge and later
        # buffers survival. Runs AFTER settlement.update so this turn's new members can
        # bank, and AFTER farm() so the surplus they store already exists. Zero LLM/RNG;
        # gated on the opt-in flag, so a default run never calls it (byte-identical to v1).
        if storage_on:
            storage.accumulate(world_state, turn)

        # M2.3: the economy. mint() turns food surplus past the storage cap into FOOD-BACKED
        # money; trade() then runs one pass of voluntary, mutually-beneficial exchange
        # (food/knowledge <-> money) at emergent prices across adjacent agents. Both AFTER
        # accumulate so this turn's stockpiles/overflow are current. Zero LLM/RNG; gated on
        # the opt-in flag, so a default run never calls them (byte-identical to v1).
        if economy_on:
            economy.mint(world_state, turn)
            economy.trade(world_state, turn)

        # M3.1: wage labor — the first INSTITUTION. Rich producers EMPLOY poor have-nots at an
        # emergent wage; the worker's output accrues to the employer, who pays the wage, so the
        # rich-poor gap COMPOUNDS (the intended disequilibrium). Runs AFTER trade/mint so this
        # turn's wealth/skill state drives who employs whom. Zero LLM/RNG; gated on the opt-in
        # flag, so a default run never calls it (byte-identical to v1).
        if labor_on:
            labor.update(world_state, turn)

        # M3.2: legitimate leadership — the first POLITICAL institution. A leader EMERGES as the
        # centre of a coherent trust cluster within a settlement (>= MIN_FOLLOWERS co-settlers
        # trust a common agent), persists only while that following holds, and coordinates its
        # followers (a tighter home-pull). PURE read of the v1 trust system — writes no trust.
        # Runs AFTER settlement.update so this turn's membership is current; AFTER labor so a
        # trusted poor agent can lead a settlement its richest member does not (power decoupled
        # from wealth). Zero LLM/RNG; gated on the opt-in flag, so a default run never calls it
        # (byte-identical to v1).
        if leadership_on:
            leadership.update(world_state, turn)

        # M3.3: taxation & redistribution — the COLLISION of the M3.1 class engine and the M3.2
        # legitimacy engine. A legitimate leader taxes its wealthy followers and redistributes to
        # its poor ones, BENDING the inequality spiral; over-taxation costs the leader trust and
        # self-limits via M3.2's contingency. Runs AFTER leadership.update so it taxes with THIS
        # turn's leader/following, and the trust it WRITES feeds NEXT turn's leadership re-eval
        # (the backlash). Zero LLM/RNG, conserves wealth; gated on the opt-in flag, so a default
        # run never calls it (byte-identical to v1).
        if taxation_on:
            taxation.update(world_state, turn)

        # M3.4: conquest & monarchy — the SECOND source of power, DOMINATION by force. A wealthy
        # aspirant spends money to muster an army of real fighters and SEIZES a settlement, becoming
        # MONARCH (rule by force, no consent — contrast M3.3); standing monarchs levy by force, and
        # the crown is LOSABLE to a stronger later army. Runs AFTER leadership/taxation so it reads
        # this turn's followers (a leader's loyal defenders) and current wealth. Conquest can KILL
        # agents deterministically (via population.announce_death). Zero LLM/RNG; gated on the opt-in
        # flag, so a default run never calls it (byte-identical to v1).
        if monarchy_on:
            monarchy.update(world_state, turn)

        # M3.5: kingdoms & vassalage — the SCALE-UP of M3.4. A monarch conquers NEIGHBOURING
        # settlements into a multi-settlement REALM (king -> vassal lords -> their settlements);
        # tribute cascades UP, loyal vassals owe military SERVICE, and a vassal pushed by heavy
        # tribute can BREAK AWAY (conditional loyalty). Runs AFTER monarchy.update so this turn's
        # monarchs are the kings-in-waiting and conquest reuses the SAME fight (resolve_battle). Zero
        # LLM/RNG; gated on the opt-in flag, so a default run never calls it (byte-identical to v1).
        if kingdoms_on:
            kingdoms.update(world_state, turn)

        # M3.6: inter-kingdom war & empire — the CLIMAX of Phase 3. Feudal KINGDOMS (M3.5) clash:
        # each musters its WHOLE LOYAL host (king + loyal vassals + loyal subject-kings), the SAME
        # resolve_battle decides it, and the loser's king is SUBJUGATED into the victor's EMPIRE (a
        # multi-level hierarchy: emperor -> subject-king -> vassal-lords -> settlements). A subject-
        # king's loyalty stays CONDITIONAL, so an over-taxing or weakening emperor FRAGMENTS (empires
        # rise AND fall). Runs AFTER kingdoms.update so this turn's realms/loyalties are the ones that
        # war, and reuses the SAME fight (resolve_battle). War kills agents on BOTH sides. Zero LLM/RNG;
        # gated on the opt-in flag, so a default run never calls it (byte-identical to v1).
        if empire_on:
            empire.update(world_state, turn)

        # M4.1: lineage — aging, coming of age, OLD-AGE deaths, child-feeding, and BIRTHS.
        # Runs AFTER the institutions so births read this turn's settled/trust/food state,
        # and a ruler who died of old age is cleaned up by the SAME institution updates
        # next turn that already handle a battle death (succession is M4.3, not built).
        # Runs BEFORE process_respawns, whose UNCHANGED living < TARGET_POPULATION gate is
        # the extinction floor: births are the engine, respawn the backstop. Zero LLM; all
        # randomness from the seeded stream; gated on the opt-in flag, so a default run
        # never calls it (byte-identical to v1).
        if lineage_on:
            for baby in lineage.update(world_state, turn):
                survived.setdefault(baby.name, turn)
                if DEBUG_MODE:
                    print(f"*** {baby.name} was born to {baby.parents[0]} and "
                          f"{baby.parents[1]} on turn {turn} ***")
                    print()
                elif VERBOSE_MODE:
                    print(f"  *** {baby.name} was born to {baby.parents[0]} and "
                          f"{baby.parents[1]} ***\n")

        # M4.4 (opens Arc 2): DISCONTENT — advance the per-agent class-pressure gauge. Runs LAST,
        # AFTER every institution and lineage have settled this turn, so it reads the FINAL state the
        # oppressed actually experience (levies applied, tribute cascaded, births/aging done). It is a
        # pure MEASURE — it reads existing state and writes only world_state["discontent"], never
        # touching a ruler (uprisings are M4.5). Zero LLM, zero new RNG; gated on the opt-in flag, so a
        # default run never calls it (byte-identical to v1, the golden master included).
        if discontent_on:
            discontent.update(world_state, turn)

        # M4.5 (Arc 2): UPRISING — the relief valve. Runs LAST, AFTER discontent.update so it reads
        # THIS turn's fresh gauge: where a resentful majority carries enough aggregate grievance, the
        # poor RISE against their force ruler and may DEPOSE + EXPROPRIATE him (deaths route through
        # the normal population.announce_death path, so M4.2 inheritance / M4.3 succession compose).
        # Zero LLM, zero new RNG; gated on the opt-in flag, so a default run never calls it
        # (byte-identical to v1). Uprisings need force rulers, so pair with --monarchy/--kingdoms.
        if uprising_on:
            uprising.update(world_state, turn)

        # M4.7 (Arc 3): BELIEFS — tally each agent's lived experience, FORM beliefs it now warrants,
        # and SPREAD beliefs one hop along this turn's contact network (trust-weighted like M1.1
        # knowledge). Runs LAST so it reads the turn's final state (this turn's deaths, rulers, hunger).
        # Zero LLM; formation deterministic, spread draws the seeded RNG like knowledge diffusion; gated
        # on the opt-in flag, so a default run never calls it (byte-identical to v1). Inert culture —
        # beliefs confer no mechanical power yet (that is M4.8/M4.9).
        if beliefs_on:
            beliefs.update(world_state, turn)

        # M4.8 (Arc 3): RELIGION — crystallise coherent shared belief into FAITHS, raise PROPHETS from
        # the devout, and move believers' trust in their rulers by faith-alignment. Runs AFTER
        # beliefs.update so it reads this turn's belief sets; it writes ONLY trust (which next turn's
        # M4.4 discontent / M3.5 breakaway / M4.5 uprising read), touching no other system. Zero LLM,
        # zero RNG; gated on the opt-in flag, so a default run never calls it (byte-identical to v1).
        if religion_on:
            religion.update(world_state, turn)

        # M4.9 (Arc 3 close): CULTURE — sustain the conquest-friction loyalty tax on foreign-ruled
        # towns and assimilate their children toward the ruler's culture. Runs AFTER religion.update so
        # it reads this turn's beliefs/faith; it writes ONLY trust (which next turn's M4.4/M3.5/M4.5
        # read) and children's beliefs (M4.7 state), touching no other system. Friction is
        # deterministic; assimilation draws the seeded RNG; gated on the flag so a default run never
        # calls it (byte-identical to v1).
        if culture_on:
            culture.update(world_state, turn)

        # M4.10 (Arc 4): WRITING & RECORDS — invent writing (M1.2 machinery, seeded), then exercise the
        # three powers of literacy: inscribe/inherit written LAW, ARCHIVE + re-teach techs (curing
        # knowledge-extinction), and append MAJOR events to a persistent CHRONICLE. Runs LATE so it reads
        # this turn's settled rulers and institutional events; writing spreads via the ordinary M1.1
        # diffusion. Zero LLM; only discovery draws RNG; gated on the flag so a default run never calls it
        # (byte-identical to v1).
        if writing_on:
            writing.update(world_state, turn)

        # M4.11 (Arc 4): METALLURGY — invent the metalworking->weapons chain (M1.2 machinery, seeded);
        # it spreads via ordinary M1.1 diffusion, and its effects are read live elsewhere (the farm
        # yield boost in knowledge.farm, the armed force multiplier in monarchy.resolve_battle). Zero
        # LLM; only discovery draws RNG; gated on the flag so a default run never calls it and — with no
        # one knowing metalworking/weapons — the boost and multiplier are no-ops (byte-identical to v1).
        if metallurgy_on:
            metallurgy.update(world_state, turn)

        if food_cfg is not None:
            _scaled_respawn_food(turn, food_cfg)
        else:
            maybe_respawn_food(turn)

        # Day 14: bring in any blank-slate newcomer whose respawn has come due. New
        # agents enter at turn's end and first act NEXT turn, so mid-turn iteration
        # is never disturbed. Track them for the summary like any other agent.
        for newcomer in population.process_respawns(turn, world_state):
            survived[newcomer.name] = turn
            if DEBUG_MODE:
                print(f"*** {newcomer.name} entered the world on turn {turn} (blank slate) ***")
                print()
            elif VERBOSE_MODE:
                print(f"  *** {newcomer.name} entered the world (blank slate) ***\n")

        # Day 17: fire any scripted god commands at this clean turn boundary. Same
        # semantics as the interactive menu — world_state is mutated here, perceived
        # NEXT turn — but driven from a file/flag so a dramatic run reproduces exactly.
        if turn in god_script:
            for command in god_script[turn]:
                print(f"[GOD-SCRIPT turn {turn}] {command}")
                god_mode.run_command(command, world_state)
            print()
        # Day 15: otherwise pause into the interactive God menu at the boundary. A
        # script and the live menu are mutually exclusive so an automated/recorded run
        # never blocks on input().
        elif god_every > 0 and turn % god_every == 0:
            god_mode.god_menu(world_state, turn)

        # Day 18: redraw the live dashboard from the now-resolved turn (READ only).
        # Day 19: then pause `turn_delay`s so a human can watch the rendered run. The
        # sleep is gated on `renderer` so it ONLY ever paces a rendered run — a plain
        # or logged-plain run has no renderer and never sleeps, and the pause touches
        # neither world_state nor the RNG, so reproducibility is unaffected.
        if renderer is not None:
            renderer.update(world_state)
            if turn_delay > 0:
                time.sleep(turn_delay)

        # End only when the world is BOTH empty AND has no respawn pending — a
        # scheduled newcomer can still repopulate an emptied world.
        if not living_agents() and not world_state["pending_respawns"]:
            if VERBOSE_MODE:
                print("All agents have died and no respawn is pending. Ending simulation.")
            break

    # --- End-of-run analysis (both modes) -------------------------------
    print()
    print_agent_summary(survived, num_turns)
    # M4.4: the world-level pressure read-out, printed ONLY when the gauge is on so a default run's
    # end-of-run output is byte-identical to v1.
    if world_state.get("discontent_on"):
        print_settlement_pressure()
    # M4.7: the world-level belief-cultures read-out, printed ONLY when beliefs are on.
    if world_state.get("beliefs_on"):
        print_belief_cultures()
    # M4.8: the world-level faiths read-out, printed ONLY when religion is on.
    if world_state.get("religion_on"):
        print_faiths()
    # M4.9: the world-level cultures read-out, printed ONLY when culture is on.
    if world_state.get("culture_on"):
        print_cultures()
    # M4.10: the world-level written-records read-out, printed ONLY when writing is on.
    if world_state.get("writing_on"):
        print_records()
    # M4.11: the world-level metallurgy read-out, printed ONLY when metallurgy is on.
    if world_state.get("metallurgy_on"):
        print_metallurgy()
    print_inference_savings(counters)
    print_events_log()


def _make_renderer(mode: str, *, sink: "Any" = None, turn_delay: float = 0.0):
    """Build the optional renderer for the chosen mode (None for plain mode).

    Imported lazily so a plain run never imports `rich`/`pygame` (or the renderer
    package) at all — keeping the default path's dependencies and import-time behaviour
    unchanged. `sink` is where the plain per-turn text is redirected during the loop:
    the open log file under --log, else None (the renderer defaults it to os.devnull).
    `turn_delay` paces the visual renderer (it waits this long per turn itself).
    """
    if mode == "rich":
        from renderer import RichRenderer
        return RichRenderer(sink=sink)
    if mode == "pygame":
        # SLICE 1 visual renderer. Same .live()/.update()/.sink interface the sim drives;
        # it paces itself (turn_delay), so the sim's own per-turn sleep stays at 0.
        from renderer.pygame_renderer import PygameRenderer
        return PygameRenderer(sink=sink, turn_delay=turn_delay)
    return None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Day 18: importing `rich` consumes some of the global `random` stream at import
    # time. Since the offline provider AND world/food placement draw from that same
    # stream, importing it AFTER seeding would shift the sequence and make a seeded
    # --render rich run diverge from the plain run. Trigger the import BEFORE seeding
    # (it is cached, so the later RichRenderer construction is free) so the seed
    # governs an identical world whether or not the dashboard is on.
    if args.render == "rich":
        import renderer  # noqa: F401  (import-for-side-effect: warm rich before seed)

    # --pygame selects the VISUAL renderer; it overrides --render. Resolve the mode once
    # and (like rich) warm the optional dependency BEFORE seeding so the import can never
    # shift the RNG stream — keeping a seeded --pygame run's world identical to a plain
    # one. Pygame is OPTIONAL: if it isn't installed, fail gracefully with a clear hint
    # rather than a traceback, and never make the core sim depend on it.
    render_mode = "pygame" if args.pygame else args.render
    if render_mode == "pygame":
        try:
            import pygame  # noqa: F401  (warm + availability check before seed)
        except ImportError:
            print("Pygame is not installed. Install it with:  pip install pygame")
            return

    # Seed BEFORE any world setup so placement + food spawns + provider RNG are all
    # part of the reproducible sequence. --seed wins over AICIV_SEED; absent both, the
    # run stays unseeded (varied), exactly as before Day 17.
    seed = args.seed if args.seed is not None else (
        int(os.environ["AICIV_SEED"]) if os.environ.get("AICIV_SEED") else None)
    if seed is not None:
        random.seed(seed)

    num_turns = args.turns if args.turns is not None else NUM_TURNS
    god_script = parse_god_script(args.god_script)
    god_every = args.god_every if args.god_every is not None else GOD_EVERY

    # M0.3: a large --agents cast switches on the scaled world. `large` gates the new
    # path so the default trio run is byte-for-byte unchanged (agent_specs/grid/food
    # all stay None below). The cast is built AFTER seeding so a seeded scale run is
    # reproducible (placement positions come from the seeded RNG).
    large = args.agents is not None and args.agents > len(AGENT_SPECS)
    if large:
        grid_size = args.grid_size if args.grid_size is not None else scaled_grid_size(args.agents)
        agent_specs = build_scaled_specs(args.agents, grid_size)
        food_cfg = scaled_food_cfg(args.agents)
    else:
        grid_size = args.grid_size   # may still override the trio's grid; else None
        agent_specs = None
        food_cfg = None

    # M1.1: parse --seed-knowledge entries "ITEM" or "ITEM:N" into (item, count) pairs.
    # No flag -> None -> nothing seeded -> diffusion is a no-op -> v1 byte-identical.
    knowledge_seed = None
    if args.seed_knowledge:
        knowledge_seed = []
        for entry in args.seed_knowledge:
            item, _, count = entry.partition(":")
            knowledge_seed.append((item, int(count) if count else 1))

    # M1.2: --tech-tree turns on unscripted discovery using the canonical TECH_TREE.
    # Absent it, tech_tree is None -> discovery is a no-op -> v1 byte-identical.
    # M4.10/M4.11: --writing and --metallurgy ride the tech tree (their prereqs are `tools`), so each
    # implies --tech-tree.
    tech_tree = knowledge.TECH_TREE if (args.tech_tree or args.writing or args.metallurgy) else None

    # M2.3/M3.1: the economy builds ON settlement + storage; wage labor (M3.1) builds ON the
    # economy. So --labor implies --economy, and --economy implies --settlements + --storage (a
    # trader/employer must be a settled agent with a stockpile). Each can still be enabled alone.
    economy_on = args.economy or args.labor
    # M3.2: legitimate leadership needs only a SETTLEMENT to lead (a coherent trust cluster of
    # co-settlers) — it is the first power NOT downstream of wealth, so it does NOT pull in the
    # economy/labor. --leadership therefore implies --settlements only. Each can be enabled alone.
    # M3.3: taxation needs a legitimate LEADER to levy, so --taxation implies --leadership (and so
    # --settlements). It does NOT force --labor: taxation works on whatever wealth exists, but the
    # inequality it BENDS is the M3.1 spiral, so a user demonstrates the collision with --labor too.
    # M4.6: the REVOLUTIONARY rules through existing M3.2 leadership, so --uprising implies
    # --leadership (the deposed seat is filled by consent, not left permanently empty).
    leadership_on = args.leadership or args.taxation or args.uprising
    # M3.6: empire BUILDS ON kingdoms (an emperor is a king who conquered another king), so --empire
    # implies --kingdoms.
    kingdoms_on = args.kingdoms or args.empire
    # M3.5: kingdoms BUILD ON monarchy (a king is a monarch who expanded), so --kingdoms implies
    # --monarchy; both need a settlement to seize/realm.
    monarchy_on = args.monarchy or kingdoms_on
    # M3.4/M3.5: conquest needs only a SETTLEMENT to seize (and wealth, which the run may supply via
    # --economy/--labor, and loyalty to collide with via --leadership). So --monarchy/--kingdoms imply
    # --settlements only — they do NOT force the economy or leadership; each can be enabled alone.
    settlements_on = args.settlements or economy_on or leadership_on or monarchy_on
    storage_on = args.storage or economy_on

    # M0.1 baseline mind: explicit --cognition wins; else 'llm' for the trio (v1) and
    # 'heuristic' for a large cast (the focal budget promotes the interesting few).
    cognition = args.cognition if args.cognition is not None else ("heuristic" if large else "llm")

    # M0.2: resolve the focal budget. An explicit --focal-budget always wins; absent it,
    # default to DEFAULT_FOCAL_BUDGET — except a small `--cognition heuristic` run keeps
    # 0 focal slots (the M0.1 zero-LLM run a user expects). A large heuristic cast still
    # gets the budget, since tiering on top is the whole point of scaling.
    if args.focal_budget is not None:
        focal_budget = args.focal_budget
    elif cognition == "heuristic" and not large:
        focal_budget = 0
    else:
        focal_budget = DEFAULT_FOCAL_BUDGET

    # DEMO SCENARIO STAGING (--stage). Default None -> nothing below fires -> byte-identical. When
    # set, it IMPLIES the matching institutions (so their verified per-turn updates run), OWNS the
    # world (the scenario builds the cast; --agents is ignored), sizes a world big enough for the
    # scene, and defaults to a call-free heuristic cast so the demo is watchable offline.
    stage = args.stage
    empire_on = args.empire
    # M4.5: --uprising implies --discontent (there is no pressure to blow without the gauge).
    uprising_on = args.uprising
    discontent_on = args.discontent or uprising_on
    # M4.9: --culture implies --religion (-> --beliefs): culture is built from faith + belief.
    culture_on = args.culture
    # M4.8: --religion implies --beliefs (no faith without belief).
    religion_on = args.religion or culture_on
    beliefs_on = args.beliefs or religion_on
    if stage is not None:
        monarchy_on = True
        kingdoms_on = kingdoms_on or stage in ("kingdom", "war")
        empire_on = empire_on or stage == "war"
        settlements_on = True
        # A LIVING demo needs storage ON: well-fed producers bank a surplus and DRAW IT DOWN to
        # weather a shock (a levied town, a far-seated king), so the realm survives instead of
        # starving out within ~40 turns. The money economy is deliberately left OFF for staging:
        # minting would steadily enrich commoners into fresh conquerors, whose churn dissolves the
        # very kingdom/empire the demo is meant to SHOW. Storage + the seeded producers keep the
        # staged realm both alive and STRUCTURALLY STABLE for the whole run.
        storage_on = True
        agent_specs = []                       # the scenario constructs the whole cast itself
        food_cfg = None
        if grid_size is None:
            grid_size = 30 if stage == "war" else 24
        if args.cognition is None:
            cognition, focal_budget = "heuristic", 0   # no LLM calls for the visual demo

    # --log mirrors stdout to a file via a Tee for the whole run, then restores it.
    # The visual --pygame renderer is a live watch tool, so it always takes the no-log
    # branch below (which suppresses KeyboardInterrupt for a clean window-close); --log
    # is ignored for it.
    if args.log and render_mode != "pygame":
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        log_file = open(args.log, "w")
        original = sys.stdout
        sys.stdout = _Tee(original, log_file)
        # Day 18: in rich mode the dashboard takes the terminal during the loop, so the
        # plain per-turn text is routed to the log file ONLY (not owned by the renderer
        # — main closes it). The end-of-run summary still prints through the Tee to both.
        renderer = _make_renderer(args.render, sink=log_file)
        try:
            if seed is not None:
                print(f"[run] seed={seed} turns={num_turns} provider={PROVIDER}")
            run_simulation(num_turns, god_script=god_script, god_every=god_every,
                           renderer=renderer, turn_delay=args.speed,
                           cognition=cognition, focal_budget=focal_budget,
                           agent_specs=agent_specs, grid_size=grid_size, food_cfg=food_cfg,
                           knowledge_seed=knowledge_seed, tech_tree=tech_tree,
                           settlements=settlements_on,
                           storage_on=storage_on, economy_on=economy_on,
                           labor_on=args.labor, leadership_on=leadership_on,
                           taxation_on=args.taxation, tax_rate=args.tax_rate,
                           monarchy_on=monarchy_on,
                           kingdoms_on=kingdoms_on, tribute_rate=args.tribute_rate,
                           empire_on=empire_on, empire_share=args.imperial_share,
                           stage=stage, lineage_on=args.lineage,
                           discontent_on=discontent_on, uprising_on=uprising_on,
                           beliefs_on=beliefs_on, religion_on=religion_on, culture_on=culture_on,
                           writing_on=args.writing, metallurgy_on=args.metallurgy)
        finally:
            sys.stdout = original
            log_file.close()
        print(f"[run] captured to {args.log}")
    else:
        # No log: a renderer (rich/pygame) drops the plain per-turn text (devnull) and
        # shows only its view; the summary prints to the terminal after the run. The
        # pygame renderer paces itself (turn_delay below is 0 for it), so closing the
        # window raises KeyboardInterrupt and ends the run cleanly via the suppress.
        renderer = _make_renderer(render_mode, sink=None, turn_delay=args.speed)
        sim_delay = 0.0 if render_mode == "pygame" else args.speed
        with contextlib.suppress(KeyboardInterrupt):
            run_simulation(num_turns, god_script=god_script, god_every=god_every,
                           renderer=renderer, turn_delay=sim_delay,
                           cognition=cognition, focal_budget=focal_budget,
                           agent_specs=agent_specs, grid_size=grid_size, food_cfg=food_cfg,
                           knowledge_seed=knowledge_seed, tech_tree=tech_tree,
                           settlements=settlements_on,
                           storage_on=storage_on, economy_on=economy_on,
                           labor_on=args.labor, leadership_on=leadership_on,
                           taxation_on=args.taxation, tax_rate=args.tax_rate,
                           monarchy_on=monarchy_on,
                           kingdoms_on=kingdoms_on, tribute_rate=args.tribute_rate,
                           empire_on=empire_on, empire_share=args.imperial_share,
                           stage=stage, lineage_on=args.lineage,
                           discontent_on=discontent_on, uprising_on=uprising_on,
                           beliefs_on=beliefs_on, religion_on=religion_on, culture_on=culture_on,
                           writing_on=args.writing, metallurgy_on=args.metallurgy)


if __name__ == "__main__":
    main()
