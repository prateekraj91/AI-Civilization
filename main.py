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
import conversation
import god_mode
import heuristic
import population
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
                          "proposed an alliance", "died on turn", "appeared on turn")


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
        print("Important memories:")
        for mem in important_memories(agent.memory):
            print(f"  - {mem}")
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
             "pause is presentation-only — it applies ONLY with --render rich, after "
             "each turn is drawn, and never affects tests, plain/logged runs, or the "
             "seeded RNG. Demo invocation: --render rich --speed slow --turns 30.")
    return p.parse_args(argv)


def run_simulation(num_turns: int, *, god_script: dict[int, list[str]] | None = None,
                   god_every: int = 0, renderer: "Any" = None,
                   turn_delay: float = 0.0,
                   agent_specs: "list | None" = None,
                   cognition: str = "llm",
                   focal_budget: "int | None" = None,
                   grid_size: "int | None" = None,
                   food_cfg: "dict | None" = None) -> None:
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
    """
    god_script = god_script or {}

    # --- Setup ----------------------------------------------------------
    reset_call_stats()
    # M0.3: a large cast needs a bigger world; grid_size None keeps the v1 10x10.
    create_world(size=grid_size) if grid_size is not None else create_world()
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
    print_inference_savings(counters)
    print_events_log()


def _make_renderer(mode: str, *, sink: "Any" = None):
    """Build the optional Day 18 renderer for --render (None for plain mode).

    Imported lazily so a plain run never imports `rich` (or the renderer package) at
    all — keeping the default path's dependencies and import-time behaviour unchanged.
    `sink` is where the plain per-turn text is redirected during the loop: the open log
    file under --log, else None (the renderer defaults it to os.devnull).
    """
    if mode != "rich":
        return None
    from renderer import RichRenderer
    return RichRenderer(sink=sink)


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

    # --log mirrors stdout to a file via a Tee for the whole run, then restores it.
    if args.log:
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
                           agent_specs=agent_specs, grid_size=grid_size, food_cfg=food_cfg)
        finally:
            sys.stdout = original
            log_file.close()
        print(f"[run] captured to {args.log}")
    else:
        # No log: rich mode drops the plain per-turn text (devnull) and shows only the
        # dashboard; the summary prints to the terminal after the run.
        renderer = _make_renderer(args.render, sink=None)
        with contextlib.suppress(KeyboardInterrupt):
            run_simulation(num_turns, god_script=god_script, god_every=god_every,
                           renderer=renderer, turn_delay=args.speed,
                           cognition=cognition, focal_budget=focal_budget,
                           agent_specs=agent_specs, grid_size=grid_size, food_cfg=food_cfg)


if __name__ == "__main__":
    main()
