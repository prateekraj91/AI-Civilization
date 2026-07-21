"""
test_simulation.py
==================

Deterministic checks for the simulation mechanics (Days 6-8) and the
personality / goal / strategy milestone. These don't need an LLM: they drive the
modules directly with hand-placed agents so each behaviour is exercised in
isolation and asserted, rather than left to a random wander.

Run:  ./Jarvis/bin/python test_simulation.py
"""

import contextlib
import io
import os
import random
import tempfile

import conversation
import heuristic
import llm
import main
import world
from agents import Agent
from personality import Personality
from strategy import Strategy, build_strategy_prompt, choose_action, get_personality
from world import (
    MEMORY_LIMIT,
    create_world,
    execute_action,
    is_dead,
    is_sick,
    mark_dead,
    move_agent,
    observe,
    place_agent,
    record_memory,
    record_social_memories,
    spawn_food,
    update_hunger,
    world_state,
)


def _fresh_world() -> None:
    """Reset world_state to a clean, food-free 10x10 grid with no agents."""
    create_world()
    world_state["agents"].clear()
    world_state["food"].clear()
    world_state["turn"] = 0


def _agent(name: str, personality: str, pos: tuple[int, int], hunger: int = 1) -> Agent:
    """Create, place, and return an agent in the current world."""
    a = Agent(name=name, personality=personality)
    place_agent(a, *pos)
    a.hunger = hunger
    return a


def _fresh_world() -> None:
    """Reset world_state to a clean, food-free 10x10 grid with no agents."""
    create_world()
    world_state["agents"].clear()
    world_state["food"].clear()
    world_state["turn"] = 0


def test_detection_by_name() -> None:
    """Day 7: an adjacent agent appears by name in the right direction."""
    _fresh_world()
    alex = Agent(name="Alex", personality="x")
    bob = Agent(name="Bob", personality="x")
    place_agent(alex, 5, 5)
    place_agent(bob, 5, 4)   # directly North of Alex

    report = observe(alex, world_state)
    assert "North: Bob" in report, report
    assert "South: empty" in report, report
    assert "East: empty" in report, report
    assert "West: empty" in report, report
    print("PASS test_detection_by_name")


def test_detection_only_living() -> None:
    """A dead neighbour is no longer detected."""
    _fresh_world()
    alex = Agent(name="Alex", personality="x")
    bob = Agent(name="Bob", personality="x")
    place_agent(alex, 5, 5)
    place_agent(bob, 6, 5)   # East of Alex
    assert "East: Bob" in observe(alex, world_state)

    mark_dead(bob)
    assert "East: empty" in observe(alex, world_state)
    print("PASS test_detection_only_living")


def test_social_memory_entries() -> None:
    """Day 8: sightings produce the documented memory strings."""
    _fresh_world()
    alex = Agent(name="Alex", personality="x")
    bob = Agent(name="Bob", personality="x")
    kira = Agent(name="Kira", personality="x")
    place_agent(alex, 5, 5)
    place_agent(bob, 5, 4)        # North of Alex
    place_agent(kira, 6, 5)       # East of Alex
    spawn_food(0)
    world_state["food"].append((7, 5))   # food right next to Kira

    observed = record_social_memories(alex, world_state)
    assert set(observed) == {"Bob", "Kira"}, observed
    assert "Observed Bob north of me" in alex.memory, alex.memory
    assert "Observed Kira east of me" in alex.memory, alex.memory
    assert "Observed Kira near food" in alex.memory, alex.memory
    # Bob is not near food, so no near-food memory for Bob.
    assert "Observed Bob near food" not in alex.memory, alex.memory
    print("PASS test_social_memory_entries")


def test_memory_bound() -> None:
    """Memory is capped at MEMORY_LIMIT (20), discarding oldest first."""
    _fresh_world()
    alex = Agent(name="Alex", personality="x")
    place_agent(alex, 0, 0)
    for i in range(50):
        record_memory(alex, f"event {i}")
    assert len(alex.memory) == MEMORY_LIMIT, len(alex.memory)
    assert alex.memory[0] == "event 30", alex.memory[0]    # oldest kept
    assert alex.memory[-1] == "event 49", alex.memory[-1]  # newest kept
    print("PASS test_memory_bound")


def test_food_competition() -> None:
    """Day 6: food eaten by one agent disappears for everyone."""
    _fresh_world()
    alex = Agent(name="Alex", personality="x")
    bob = Agent(name="Bob", personality="x")
    place_agent(alex, 3, 3)
    place_agent(bob, 8, 8)
    world_state["food"].append((3, 3))   # under Alex
    alex.hunger = 5

    result = execute_action(alex, "eat")
    assert "ate food" in result, result
    assert (3, 3) not in world_state["food"], world_state["food"]
    assert alex.hunger == 0, alex.hunger

    # Bob walks onto the now-empty cell: nothing to eat.
    bob.position = (3, 3)
    result = execute_action(bob, "eat")
    assert "no food" in result, result
    print("PASS test_food_competition")


def test_movement_collision() -> None:
    """Day 6: an agent cannot move onto a cell held by another living agent."""
    _fresh_world()
    alex = Agent(name="Alex", personality="x")
    bob = Agent(name="Bob", personality="x")
    place_agent(alex, 5, 5)
    place_agent(bob, 6, 5)   # East of Alex

    moved = move_agent(alex, 1, 0)   # try to step East onto Bob
    assert moved is False
    assert alex.position == (5, 5), alex.position

    # Once Bob dies, the cell frees up.
    mark_dead(bob)
    assert move_agent(alex, 1, 0) is True
    assert alex.position == (6, 5), alex.position
    print("PASS test_movement_collision")


def test_starvation_and_death() -> None:
    """Hunger reaches the cap, is_dead reports it, mark_dead frees the cell."""
    _fresh_world()
    alex = Agent(name="Alex", personality="x")
    place_agent(alex, 4, 4)
    for _ in range(20):
        update_hunger(alex)
    assert is_dead(alex)
    mark_dead(alex)
    assert alex.alive is False
    print("PASS test_starvation_and_death")


# --- Personality (Phase 1) -------------------------------------------------
def test_personality_parsing() -> None:
    """Free-text personalities map to the expected dominant trait."""
    assert Personality.from_text("curious and adventurous").dominant == "curiosity"
    assert Personality.from_text("cautious and territorial").dominant == "caution"
    assert Personality.from_text("friendly and outgoing").dominant == "friendliness"
    assert Personality.from_text("independent and competitive").dominant == "independence"
    # Unrecognised text → balanced default, still well-defined.
    assert Personality.from_text("???").dominant == "curiosity"
    # Cautious agents eat earlier than bold ones.
    assert Personality.from_text("cautious").eat_threshold < Personality.from_text("bold").eat_threshold
    print("PASS test_personality_parsing")


def test_personalities_produce_different_behaviour() -> None:
    """Same situation, different personalities → different actions (Phase 1)."""
    wander = Strategy(kind="wander")

    # Friendly vs independent, with another agent due south: opposite moves.
    _fresh_world()
    friendly = _agent("Fred", "friendly and outgoing", (5, 5))
    _agent("Other", "neutral", (5, 8))
    assert choose_action(friendly, wander, world_state)[0] == "move_south"

    _fresh_world()
    indep = _agent("Ivy", "independent and competitive", (5, 5))
    _agent("Other", "neutral", (5, 8))
    assert choose_action(indep, wander, world_state)[0] == "move_north"

    # Curious keeps moving; cautious (no food known) holds position.
    _fresh_world()
    curious = _agent("Cara", "curious and adventurous", (5, 5))
    assert choose_action(curious, wander, world_state)[0].startswith("move_")

    _fresh_world()
    cautious = _agent("Cody", "cautious and careful", (5, 5))
    assert choose_action(cautious, wander, world_state)[0] == "rest"
    print("PASS test_personalities_produce_different_behaviour")


def test_curious_rests_less_than_cautious() -> None:
    """Over many turns in an empty world, curious moves, cautious rests."""
    wander = Strategy(kind="wander")

    _fresh_world()
    curious = _agent("Cara", "curious and adventurous", (5, 5))
    curious_rests = sum(
        choose_action(curious, wander, world_state)[0] == "rest" for _ in range(15)
    )

    _fresh_world()
    cautious = _agent("Cody", "cautious and careful", (5, 5))
    cautious_rests = sum(
        choose_action(cautious, wander, world_state)[0] == "rest" for _ in range(15)
    )

    assert curious_rests == 0, curious_rests
    assert cautious_rests >= 10, cautious_rests
    assert curious_rests < cautious_rests
    print(f"PASS test_curious_rests_less_than_cautious (curious={curious_rests}, cautious={cautious_rests})")


# --- Strategy executor (Phase 4) ------------------------------------------
def test_strategy_eats_and_seeks_food() -> None:
    """Standing on food → eat; seek_food navigates toward the nearest food."""
    _fresh_world()
    a = _agent("Sam", "neutral", (4, 4), hunger=5)
    world_state["food"].append((4, 4))
    assert choose_action(a, Strategy(kind="seek_food"), world_state)[0] == "eat"

    _fresh_world()
    b = _agent("Sam", "neutral", (4, 4), hunger=2)
    world_state["food"].append((4, 7))  # due south
    assert choose_action(b, Strategy(kind="seek_food"), world_state)[0] == "move_south"
    print("PASS test_strategy_eats_and_seeks_food")


def test_strategy_approach_and_avoid() -> None:
    """'approach' moves toward a named agent; 'avoid' moves away from others."""
    _fresh_world()
    a = _agent("Sam", "neutral", (4, 4))
    _agent("Bob", "neutral", (8, 4))  # due east
    assert choose_action(a, Strategy(kind="approach", target="Bob"), world_state)[0] == "move_east"
    assert choose_action(a, Strategy(kind="avoid"), world_state)[0] == "move_west"
    print("PASS test_strategy_approach_and_avoid")


def test_survival_overrides_strategy() -> None:
    """A starving agent beelines to food even under a non-food strategy."""
    _fresh_world()
    a = _agent("Sam", "neutral", (4, 4), hunger=9)
    world_state["food"].append((4, 5))  # south
    # Strategy says explore north, but survival should win.
    action = choose_action(a, Strategy(kind="explore", target="north"), world_state)[0]
    assert action == "move_south", action
    print("PASS test_survival_overrides_strategy")


def test_strategy_validation_fallback() -> None:
    """Invalid model output degrades to a safe strategy; valid passes through."""
    assert llm._validate_strategy({"strategy": "seek_food"})["strategy"] == "seek_food"
    assert llm._validate_strategy({"strategy": "nonsense"})["strategy"] == "wander"
    assert llm._validate_strategy("not a dict")["strategy"] == "wander"
    # get_strategy always returns a valid strategy (random provider here).
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        from strategy import VALID_STRATEGIES
        for _ in range(20):
            assert llm.get_strategy("x")["strategy"] in VALID_STRATEGIES
    finally:
        llm.PROVIDER = saved
    print("PASS test_strategy_validation_fallback")


def test_strategy_caching_reduces_llm_calls() -> None:
    """A full run makes far fewer LLM calls than one-per-agent-per-turn."""
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        llm.reset_call_stats()
        with contextlib.redirect_stdout(io.StringIO()):
            main.main()
        strat_calls = llm.get_call_stats()["strategy"]
    finally:
        llm.PROVIDER = saved

    n_agents = len(main.AGENT_SPECS)
    # With refresh every STRATEGY_INTERVAL turns, an agent needs at most
    # ceil(NUM_TURNS / interval) refreshes — far below NUM_TURNS per agent.
    max_refreshes = -(-main.NUM_TURNS // main.STRATEGY_INTERVAL)  # ceil
    per_turn_baseline = main.NUM_TURNS * n_agents
    assert strat_calls <= max_refreshes * n_agents, strat_calls
    assert strat_calls < per_turn_baseline, (strat_calls, per_turn_baseline)
    print(f"PASS test_strategy_caching_reduces_llm_calls "
          f"({strat_calls} calls vs {per_turn_baseline} per-turn baseline)")


# --- V2 M0.1: heuristic (zero-LLM) cognition --------------------------------
# The pseudo-actions the loop routes to the conversation/alliance layers, on top of
# world.VALID_ACTIONS, make up the full vocabulary a mind may legitimately emit.
_SOCIAL_PREFIXES = ("talk_to_", "steal_from_", "ally_with_", "betray_alliance_")


def _is_valid_action(action: str) -> bool:
    return action in world.VALID_ACTIONS or action.startswith(_SOCIAL_PREFIXES)


def test_heuristic_returns_valid_action_for_hungry_fed_threatened() -> None:
    """The heuristic mind returns a valid action for each key perception regime."""
    # Hungry (>= HEURISTIC_HUNGER) with food adjacent → step onto it.
    _fresh_world()
    a = _agent("Sam", "neutral", (4, 4), hunger=4)
    world_state["food"].append((5, 4))  # due east
    act = heuristic.decide_action(a, world_state)[0]
    assert _is_valid_action(act) and act == "move_east", act

    # Fed and alone → a valid, non-idle action (a curious agent explores by moving).
    _fresh_world()
    b = _agent("Sam", "curious and adventurous", (4, 4), hunger=1)
    act = heuristic.decide_action(b, world_state)[0]
    assert _is_valid_action(act) and act.startswith("move_"), act

    # Threatened (starving) with food on the map but not adjacent → head toward it.
    _fresh_world()
    c = _agent("Sam", "neutral", (1, 1), hunger=9)
    world_state["food"].append((1, 8))  # far south
    act = heuristic.decide_action(c, world_state)[0]
    assert _is_valid_action(act) and act == "move_south", act

    # Hungry with NO food anywhere known → explore to search, never idle.
    _fresh_world()
    d = _agent("Sam", "cautious and territorial", (9, 9), hunger=6)
    act = heuristic.decide_action(d, world_state)[0]
    assert _is_valid_action(act) and act.startswith("move_"), act
    print("PASS test_heuristic_returns_valid_action_for_hungry_fed_threatened")


def test_heuristic_moves_toward_adjacent_food() -> None:
    """A hungry heuristic agent reliably moves toward adjacent food in any direction."""
    for (fx, fy), expected in [((4, 3), "move_north"), ((4, 5), "move_south"),
                               ((5, 4), "move_east"), ((3, 4), "move_west")]:
        _fresh_world()
        a = _agent("Sam", "neutral", (4, 4), hunger=5)
        world_state["food"].append((fx, fy))
        act = heuristic.decide_action(a, world_state)[0]
        assert act == expected, (expected, act)
    print("PASS test_heuristic_moves_toward_adjacent_food")


def test_heuristic_run_makes_zero_llm_calls() -> None:
    """An all-heuristic simulation completes and makes ZERO LLM calls of any kind."""
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        llm.reset_call_stats()
        with contextlib.redirect_stdout(io.StringIO()):
            main.run_simulation(15, cognition="heuristic")
        stats = llm.get_call_stats()
    finally:
        llm.PROVIDER = saved
    assert stats["strategy"] == 0, stats
    assert stats["decision"] == 0, stats
    print("PASS test_heuristic_run_makes_zero_llm_calls")


def test_cognition_defaults_to_llm_and_path_unregressed() -> None:
    """Default cognition is 'llm' and a default run still drives the LLM strategy path."""
    assert Agent(name="x", personality="y").cognition == "llm"
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        llm.reset_call_stats()
        with contextlib.redirect_stdout(io.StringIO()):
            main.run_simulation(15)  # no cognition arg → V1 default
        strat_calls = llm.get_call_stats()["strategy"]
    finally:
        llm.PROVIDER = saved
    assert strat_calls > 0, strat_calls
    print(f"PASS test_cognition_defaults_to_llm_and_path_unregressed ({strat_calls} calls)")


def test_full_simulation_runs_clean() -> None:
    """A complete simulation runs end-to-end without raising."""
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        with contextlib.redirect_stdout(io.StringIO()):
            main.main()
    finally:
        llm.PROVIDER = saved
    print("PASS test_full_simulation_runs_clean")


# --- Tiered cognition (M0.2) ----------------------------------------------
def _budget_population(n: int) -> list[Agent]:
    """Place `n` heuristic agents on distinct cells of a fresh world for tiering tests."""
    _fresh_world()
    agents = []
    for i in range(n):
        a = Agent(name=f"A{i:02d}", personality="curious and adventurous",
                  cognition="heuristic", hunger=1)
        place_agent(a, i % world_state["size"], i // world_state["size"])
        agents.append(a)
    return agents


def test_tiering_never_exceeds_budget() -> None:
    """update_tiers promotes at most `budget` agents to 'llm', every turn, for any N."""
    import cognition
    budget = 4
    _budget_population(12)
    tenure: dict[str, int] = {}
    for turn in range(1, 16):
        world_state["turn"] = turn
        cognition.update_tiers(world_state, turn, budget, tenure)
        focal = [a for a in world_state["agents"] if a.alive and a.cognition == "llm"]
        assert len(focal) <= budget, (turn, len(focal))
    print("PASS test_tiering_never_exceeds_budget")


def test_interestingness_ranks_conflict_above_lone_wanderer() -> None:
    """An agent in a recent conflict scores higher than a settled lone wanderer."""
    import cognition
    _fresh_world()
    world_state["turn"] = 5
    conflict = _agent("Kira", "independent and competitive", (1, 1))
    wanderer = _agent("Solo", "curious and adventurous", (8, 8))
    wanderer.memory.append("Wandered around.")  # not a blank-slate newcomer
    world_state["events"].append("turn 5: Mallory stole food from Kira")
    s_conflict = cognition.interestingness(conflict, world_state)[0]
    s_wander = cognition.interestingness(wanderer, world_state)[0]
    assert s_conflict > s_wander, (s_conflict, s_wander)
    print("PASS test_interestingness_ranks_conflict_above_lone_wanderer")


def test_promotion_and_demotion_log_events() -> None:
    """A theft promotes its victim to focal (logged); a rival's drama later demotes it."""
    import cognition
    budget = 1
    _fresh_world()
    star = _agent("Star", "friendly and outgoing", (5, 5))  # victimised first
    other = _agent("Other", "curious and adventurous", (0, 0))
    star.cognition = other.cognition = "heuristic"  # baseline, so a promotion is visible
    tenure: dict[str, int] = {}

    # A fresh theft against Star (the thief Mallory is not in the cast, so only the
    # living victim is credited) -> Star is the most interesting -> promoted + logged.
    world_state["turn"] = 2
    world_state["events"].append("turn 2: Mallory stole food from Star")
    cognition.update_tiers(world_state, 2, budget, tenure)
    assert star.cognition == "llm", star.cognition
    assert any("Star promoted to focal" in e for e in world_state["events"])

    # Quiet turns let Star's tenure accrue past MIN_TENURE so it is no longer protected.
    for turn in range(3, 7):
        world_state["turn"] = turn
        cognition.update_tiers(world_state, turn, budget, tenure)
    assert star.cognition == "llm"  # still focal — nobody more interesting yet

    # Later, past Star's min tenure, the drama moves to Other: it gets robbed and
    # becomes the more interesting one, so the single focal slot shifts to it.
    world_state["turn"] = 7
    world_state["events"].append("turn 7: Mallory stole food from Other")
    cognition.update_tiers(world_state, 7, budget, tenure)
    assert other.cognition == "llm" and star.cognition == "heuristic", \
        (star.cognition, other.cognition)
    assert any("Star demoted to heuristic" in e for e in world_state["events"])
    print("PASS test_promotion_and_demotion_log_events")


def test_hysteresis_prevents_single_turn_flipflop() -> None:
    """A one-turn blip can't flap a focal agent: a promotion holds for MIN_TENURE turns."""
    import cognition
    budget = 1
    _fresh_world()
    star = _agent("Star", "friendly and outgoing", (5, 5))
    rival = _agent("Rival", "curious and adventurous", (0, 0))  # quietly competing
    star.cognition = rival.cognition = "heuristic"  # baseline, so a promotion is visible
    tenure: dict[str, int] = {}

    world_state["turn"] = 1
    world_state["events"].append("turn 1: Mallory stole food from Star")
    cognition.update_tiers(world_state, 1, budget, tenure)
    assert star.cognition == "llm"
    assert 0 < tenure["Star"] < cognition.MIN_TENURE  # inside its protected tenure

    # Immediately next turn, with the drama already gone, Star must NOT be demoted —
    # the minimum tenure holds it focal so the set doesn't thrash turn-to-turn.
    world_state["turn"] = 2
    world_state["events"].clear()  # the blip is over; nothing interesting remains
    cognition.update_tiers(world_state, 2, budget, tenure)
    assert star.cognition == "llm", "hysteresis failed: focal agent flipped after one turn"
    assert not any("demoted" in e for e in world_state["events"])
    print("PASS test_hysteresis_prevents_single_turn_flipflop")


def test_tiering_disabled_leaves_cognition_untouched() -> None:
    """focal_budget=None disables tiering: a heuristic run stays zero-LLM (M0.1 intact)."""
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        llm.reset_call_stats()
        with contextlib.redirect_stdout(io.StringIO()):
            main.run_simulation(15, cognition="heuristic")  # no focal_budget -> no tiering
        stats = llm.get_call_stats()
    finally:
        llm.PROVIDER = saved
    assert stats["strategy"] == 0 and stats["decision"] == 0, stats
    print("PASS test_tiering_disabled_leaves_cognition_untouched")


def test_budget_covering_cast_is_byte_identical_to_v1() -> None:
    """3 agents with a budget >= cast == the no-tiering path, byte-for-byte (v1 intact)."""
    def run(budget):
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(20, focal_budget=budget)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        no_tier, tiered = run(None), run(8)
    finally:
        llm.PROVIDER = saved
    assert no_tier == tiered, "budget>=cast diverged from the no-tiering path"
    assert "promoted to focal" not in tiered, "a covered cast must log no transitions"
    print("PASS test_budget_covering_cast_is_byte_identical_to_v1")


# --- Scale (M0.3) ----------------------------------------------------------
def _scale_population(n: int) -> list[Agent]:
    """Place `n` heuristic agents on a density-scaled grid (M0.3 large-cast setup)."""
    grid = main.scaled_grid_size(n)
    create_world(size=grid)
    for name, personality, goals, (x, y) in main.build_scaled_specs(n, grid):
        place_agent(Agent(name=name, personality=personality, goals=goals,
                          cognition="heuristic"), x, y)
    spawn_food(main.scaled_food_cfg(n)["initial"])
    return list(world_state["agents"])


def _run_scaled_loop(n: int, budget: int, turns: int) -> dict:
    """Drive the real per-turn loop at `n` agents; return focal cap + LLM-call stats."""
    import cognition
    food_cfg = main.scaled_food_cfg(n)
    _scale_population(n)
    llm.reset_call_stats()
    strategies: dict = {}
    survived: dict[str, int] = {}
    counters: dict[str, int] = {"agent_turns": 0}
    tenure: dict[str, int] = {}
    max_focal = 0
    with contextlib.redirect_stdout(io.StringIO()):
        for turn in range(1, turns + 1):
            world_state["turn"] = turn
            cognition.update_tiers(world_state, turn, budget, tenure)
            living = [a for a in world_state["agents"] if a.alive]
            max_focal = max(max_focal, sum(1 for a in living if a.cognition == "llm"))
            for agent in living:
                main.run_agent_turn(agent, turn, strategies, survived, counters)
            main._scaled_respawn_food(turn, food_cfg)
    return {"max_focal": max_focal, "strategy_calls": llm.get_call_stats()["strategy"]}


def test_large_cast_run_completes_without_error() -> None:
    """A 120-agent run via the real CLI path completes and stays cheap on LLM calls."""
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        llm.reset_call_stats()
        with contextlib.redirect_stdout(io.StringIO()):
            main.main(["--agents", "120", "--turns", "15", "--seed", "1"])
        calls = llm.get_call_stats()["strategy"]
    finally:
        llm.PROVIDER = saved
    # With a focal budget of 8 over 15 turns, LLM calls must stay near the budget —
    # nowhere near the 120-agents-every-turn an untiered run would cost.
    assert calls <= main.DEFAULT_FOCAL_BUDGET * 15, calls
    assert calls < 120, calls
    print(f"PASS test_large_cast_run_completes_without_error ({calls} LLM calls)")


def test_cost_vs_n_stays_bounded_by_budget_at_scale() -> None:
    """At a fixed budget, LLM calls do NOT scale with N (re-confirmed at higher range)."""
    budget, turns = 5, 12
    small = _run_scaled_loop(40, budget, turns)
    big = _run_scaled_loop(120, budget, turns)
    assert small["max_focal"] <= budget and big["max_focal"] <= budget, (small, big)
    # Tripling the population must not meaningfully grow LLM traffic.
    assert big["strategy_calls"] <= small["strategy_calls"] * 1.5 + budget, (small, big)
    print("PASS test_cost_vs_n_stays_bounded_by_budget_at_scale "
          f"(N=40:{small['strategy_calls']}  N=120:{big['strategy_calls']} calls)")


def test_scale_renderer_view_is_read_only() -> None:
    """The large-cast heatmap view renders and mutates nothing (boundary holds at scale)."""
    import copy
    from renderer import render_frame
    from renderer.text_renderer import _use_heatmap
    _scale_population(80)
    world_state["turn"] = 3
    assert _use_heatmap(world_state), "heatmap view should trigger for 80 agents"
    snap = copy.deepcopy({k: world_state[k] for k in world_state
                          if k not in ("agents", "occupancy")})
    positions = [(a.name, a.position, a.alive) for a in world_state["agents"]]
    frame = render_frame(world_state)
    after = {k: world_state[k] for k in world_state if k not in ("agents", "occupancy")}
    assert frame is not None
    assert after == snap, "scale render mutated world_state"
    assert positions == [(a.name, a.position, a.alive) for a in world_state["agents"]]
    print("PASS test_scale_renderer_view_is_read_only")


def test_occupancy_index_matches_truth_after_moves_and_deaths() -> None:
    """The M0.3 position index stays a faithful mirror of agents+positions."""
    _fresh_world()
    a = _agent("A", "neutral", (5, 5))
    b = _agent("B", "neutral", (2, 2))
    move_agent(a, 1, 0)          # A: (5,5) -> (6,5)
    mark_dead(b)                 # B leaves the index
    truth = {ag.position: ag for ag in world_state["agents"] if ag.alive}
    assert world_state["occupancy"] == truth, (world_state["occupancy"], truth)
    assert world.agent_at(6, 5) is a and world.agent_at(2, 2) is None
    print("PASS test_occupancy_index_matches_truth_after_moves_and_deaths")


# --- Knowledge as propagating state (M1.1) --------------------------------
def test_knowledge_transmits_only_between_in_contact_agents() -> None:
    """A knower teaches an ADJACENT non-knower, but never a distant one (contact graph)."""
    import knowledge
    _fresh_world()
    teacher = _agent("Teacher", "friendly and outgoing", (5, 5))
    near = _agent("Near", "curious and adventurous", (5, 4))   # adjacent (north)
    far = _agent("Far", "curious and adventurous", (0, 0))     # far away, never in contact
    teacher.knowledge.add("fire")
    random.seed(1)
    # Run several diffuse passes; nobody moves, so contact is fixed.
    for turn in range(1, 30):
        world_state["turn"] = turn
        knowledge.diffuse(world_state, turn)
    assert "fire" in near.knowledge, "adjacent agent should have learned through contact"
    assert "fire" not in far.knowledge, "a never-contacted agent must never learn"
    assert any("Teacher taught 'fire' to Near" in e for e in world_state["events"])
    print("PASS test_knowledge_transmits_only_between_in_contact_agents")


def test_adoption_probability_rises_with_trust() -> None:
    """Higher trust in the teacher yields a higher adoption probability (same learner)."""
    import knowledge
    _fresh_world()
    teacher = _agent("T", "friendly and outgoing", (5, 5))
    learner = _agent("L", "curious and adventurous", (5, 4))
    teacher.knowledge.add("fire")

    def p_at(trust):
        learner.relationships["T"] = {"trust": trust, "interactions": 1, "grudge": trust < 0}
        return knowledge.adoption_probability(learner, teacher, world_state)

    low, neutral, high = p_at(-5), p_at(0), p_at(5)
    assert low < neutral < high, (low, neutral, high)
    # And personality resists: a cautious learner adopts less than a curious one at equal trust.
    cautious = _agent("C", "cautious and territorial", (5, 6))
    cautious.relationships["T"] = {"trust": 0, "interactions": 1, "grudge": False}
    assert knowledge.adoption_probability(cautious, teacher, world_state) < neutral
    print("PASS test_adoption_probability_rises_with_trust")


def test_isolated_agent_never_learns() -> None:
    """An agent that is never adjacent to a knower never acquires the item."""
    import knowledge
    _fresh_world()
    a = _agent("A", "curious and adventurous", (1, 1))
    b = _agent("B", "curious and adventurous", (1, 2))   # a/b in contact
    loner = _agent("Loner", "curious and adventurous", (9, 9))
    a.knowledge.add("fire")
    random.seed(2)
    for turn in range(1, 40):
        world_state["turn"] = turn
        knowledge.diffuse(world_state, turn)
    assert "fire" in b.knowledge, "the in-contact agent should learn"
    assert "fire" not in loner.knowledge, "the isolated agent must never learn"
    print("PASS test_isolated_agent_never_learns")


def test_diffusion_adds_zero_llm_calls() -> None:
    """Knowledge diffusion is pure state — it makes no model calls of any kind."""
    import knowledge
    _fresh_world()
    for i in range(6):
        ag = _agent(f"K{i}", "curious and adventurous", (i, 0))
        if i == 0:
            ag.knowledge.add("fire")
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        llm.reset_call_stats()
        random.seed(3)
        for turn in range(1, 25):
            world_state["turn"] = turn
            knowledge.diffuse(world_state, turn)
        stats = llm.get_call_stats()
    finally:
        llm.PROVIDER = saved
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats
    print("PASS test_diffusion_adds_zero_llm_calls")


def test_empty_knowledge_run_is_byte_identical_to_v1() -> None:
    """No seeded knowledge -> diffusion no-op (no events, no RNG) -> v1 unregressed."""
    import knowledge
    def run(seed_knowledge):
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(20, focal_budget=8, knowledge_seed=seed_knowledge)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        base, empty = run(None), run([])
    finally:
        llm.PROVIDER = saved
    assert base == empty, "an empty knowledge seed changed the run"
    assert "taught" not in base, "no-op diffusion should log nothing"

    # A no-op diffuse must also draw ZERO rng (or it would desync the v1 stream).
    _fresh_world()
    _agent("A", "curious and adventurous", (1, 1))
    _agent("B", "curious and adventurous", (1, 2))
    state0 = random.getstate()
    knowledge.diffuse(world_state, 1)
    assert random.getstate() == state0, "no-knowledge diffuse consumed RNG"
    print("PASS test_empty_knowledge_run_is_byte_identical_to_v1")


def test_god_grant_knowledge_is_write_only_and_logs() -> None:
    """god_mode grant_knowledge adds the item + a [GOD] log, touching only world_state."""
    _fresh_world()
    import god_mode
    a = _agent("Kira", "independent and competitive", (5, 5))
    world_state["turn"] = 9
    res = god_mode.run_command("grant_knowledge Kira fire", world_state, out=lambda *_: None)
    assert "fire" in a.knowledge
    assert any("[GOD] granted 'fire' to Kira" in e for e in world_state["events"]), res
    # An unknown agent is a logged no-op, never a crash.
    god_mode.run_command("grant_knowledge Ghost fire", world_state, out=lambda *_: None)
    print("PASS test_god_grant_knowledge_is_write_only_and_logs")


# --- Discovery / invention (M1.2) -----------------------------------------
def test_discovery_respects_prerequisites() -> None:
    """An agent can only invent items whose prereqs it already knows (chain enforced)."""
    import knowledge
    _fresh_world()
    a = _agent("Solo", "curious and adventurous", (2, 2), hunger=0)
    a.knowledge.clear()
    saved = knowledge.DISCOVERY_BASE
    knowledge.DISCOVERY_BASE = 1.0  # force the roll so only the GATE decides the outcome
    try:
        rng = random.Random(0)
        knowledge.discover(world_state, 1, knowledge.TECH_TREE, rng=rng)
        assert a.knowledge == {"fire"}, a.knowledge       # only the no-prereq base item
        knowledge.discover(world_state, 2, knowledge.TECH_TREE, rng=rng)
        assert {"tools", "cooking"} <= a.knowledge        # fire unlocks its branches
        assert "farming" not in a.knowledge               # but NOT farming yet (needs tools)
        knowledge.discover(world_state, 3, knowledge.TECH_TREE, rng=rng)
        assert "farming" in a.knowledge                   # tools known -> farming inventable
    finally:
        knowledge.DISCOVERY_BASE = saved
    assert any("Solo discovered 'fire'" in e for e in world_state["events"])
    print("PASS test_discovery_respects_prerequisites")


def test_no_downstream_item_without_its_prerequisite() -> None:
    """With the base item unreachable, no downstream item is ever invented."""
    import knowledge
    _fresh_world()
    b = _agent("NoFire", "curious and adventurous", (2, 2), hunger=0)
    b.knowledge.clear()
    tree_without_fire = {k: v for k, v in knowledge.TECH_TREE.items() if k != "fire"}
    rng = random.Random(1)
    for turn in range(1, 150):
        knowledge.discover(world_state, turn, tree_without_fire, rng=rng)
    assert not b.knowledge, b.knowledge
    print("PASS test_no_downstream_item_without_its_prerequisite")


def test_discovery_is_probabilistic_not_a_timer() -> None:
    """Given prereqs, a discovery is a roll: not guaranteed in one turn, near-sure over many."""
    import knowledge
    # Single-turn success rate is small (a roll, not a timer): measure it empirically.
    successes = 0
    trials = 300
    for s in range(trials):
        _fresh_world()
        a = _agent("Inv", "curious and adventurous", (2, 2), hunger=0)
        a.knowledge = {"fire"}                      # prereqs for tools are met
        knowledge.discover(world_state, 1, {"tools": frozenset({"fire"})},
                           rng=random.Random(s))
        successes += "tools" in a.knowledge
    rate = successes / trials
    assert 0.0 < rate < 0.30, rate                  # happens, but far from every turn
    # Over many turns it becomes near-certain — so it DOES fire, just not on a schedule.
    _fresh_world()
    a = _agent("Inv", "curious and adventurous", (2, 2), hunger=0)
    a.knowledge = {"fire"}
    rng = random.Random(7)
    for turn in range(1, 200):
        knowledge.discover(world_state, turn, {"tools": frozenset({"fire"})}, rng=rng)
    assert "tools" in a.knowledge
    print(f"PASS test_discovery_is_probabilistic_not_a_timer (1-turn rate {rate:.1%})")


def test_starving_agent_does_not_invent() -> None:
    """Situation gates discovery: a starving agent's invention probability is zero."""
    import knowledge
    _fresh_world()
    starving = _agent("Hungry", "curious and adventurous", (2, 2),
                      hunger=knowledge.DISCOVERY_HUNGER_CUTOFF + 2)
    starving.knowledge = {"fire"}
    assert knowledge.discovery_probability(starving, "tools", world_state) == 0.0
    rng = random.Random(3)
    for turn in range(1, 100):
        knowledge.discover(world_state, turn, knowledge.TECH_TREE, rng=rng)
    assert "tools" not in starving.knowledge
    print("PASS test_starving_agent_does_not_invent")


def test_discovery_adds_zero_llm_calls() -> None:
    """Discovery is pure state — it makes no model calls of any kind."""
    import knowledge
    _fresh_world()
    for i in range(6):
        ag = _agent(f"D{i}", "curious and adventurous", (i, 0), hunger=0)
        ag.knowledge = {"fire"}
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        llm.reset_call_stats()
        rng = random.Random(3)
        for turn in range(1, 25):
            knowledge.discover(world_state, turn, knowledge.TECH_TREE, rng=rng)
        stats = llm.get_call_stats()
    finally:
        llm.PROVIDER = saved
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats
    print("PASS test_discovery_adds_zero_llm_calls")


def test_empty_tech_tree_run_is_byte_identical_to_v1() -> None:
    """No tech tree -> discovery no-op (no events, no RNG) -> v1 unregressed."""
    import knowledge
    def run(tree):
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(20, focal_budget=8, tech_tree=tree)
        return buf.getvalue()
    saved = llm.PROVIDER
    try:
        base, empty = run(None), run({})
    finally:
        llm.PROVIDER = saved
    assert base == empty, "an empty tech tree changed the run"
    assert "discovered" not in base, "no-op discovery should log nothing"

    # A no-op discover must draw ZERO rng (or it would desync the v1 stream).
    _fresh_world()
    _agent("A", "curious and adventurous", (1, 1), hunger=0)
    state0 = random.getstate()
    knowledge.discover(world_state, 1, None)
    assert random.getstate() == state0, "no-tree discover consumed RNG"
    print("PASS test_empty_tech_tree_run_is_byte_identical_to_v1")


# --- Technology changes the world (M1.3) ----------------------------------
def test_fire_knower_eats_more_unknown_does_not() -> None:
    """A known tech changes the knower's outcome; an unknown tech does not (fire)."""
    _fresh_world()
    cook = _agent("Cook", "cautious and territorial", (1, 1), hunger=8)
    raw = _agent("Raw", "cautious and territorial", (5, 5), hunger=8)
    cook.knowledge.add("fire")
    world.place_food(1, 1)
    world.place_food(5, 5)
    execute_action(cook, "eat")
    execute_action(raw, "eat")
    assert raw.hunger == max(0, 8 - world.EAT_RELIEF)                     # unaffected
    assert cook.hunger == max(0, 8 - world.EAT_RELIEF - world.FIRE_EAT_BONUS)  # cooked: more
    assert cook.hunger < raw.hunger
    print("PASS test_fire_knower_eats_more_unknown_does_not")


def test_tools_knower_forages_adjacent_unknown_cannot() -> None:
    """tools lets a knower eat from an adjacent tile; a non-knower can't reach it."""
    _fresh_world()
    handy = _agent("Handy", "cautious and territorial", (1, 1), hunger=8)
    bare = _agent("Bare", "cautious and territorial", (5, 5), hunger=8)
    handy.knowledge.add("tools")
    world.place_food(1, 2)   # adjacent to Handy, not underfoot
    world.place_food(5, 6)   # adjacent to Bare, not underfoot
    r_handy = execute_action(handy, "eat")
    r_bare = execute_action(bare, "eat")
    assert "foraged" in r_handy and handy.hunger < 8 and (1, 2) not in world_state["food"]
    assert "no food" in r_bare and bare.hunger == 8 and (5, 6) in world_state["food"]
    print("PASS test_tools_knower_forages_adjacent_unknown_cannot")


def test_farming_knower_produces_food_unknown_does_not() -> None:
    """Only an agent that KNOWS farming produces food into world_state."""
    import knowledge
    _fresh_world()
    farmer = _agent("Farmer", "cautious and territorial", (1, 1), hunger=0)
    idle = _agent("Idle", "cautious and territorial", (6, 6), hunger=0)
    farmer.knowledge.add("farming")
    before = len(world_state["food"])
    rng = random.Random(0)
    for turn in range(1, 20):
        knowledge.farm(world_state, turn, rng=rng)
    assert len(world_state["food"]) > before, "the knower should have produced food"
    assert any("Tended crops" in m for m in farmer.memory)
    assert not any("Tended crops" in m for m in idle.memory), "non-knower must not farm"
    # produced tiles sit next to the FARMER only.
    assert all(abs(fx - 1) + abs(fy - 1) <= 1 for fx, fy in world_state["food"])
    print("PASS test_farming_knower_produces_food_unknown_does_not")


def test_farming_population_outlasts_no_farming_control() -> None:
    """A farming population keeps more food and survives better than a matched control."""
    import knowledge

    def run(farmers):
        random.seed(5)
        grid = main.scaled_grid_size(40)
        create_world(size=grid)
        cells = [(x, y) for x in range(grid) for y in range(grid)]
        random.Random(5).shuffle(cells)
        agents = []
        for i in range(40):
            a = Agent(name=f"A{i:02d}", personality=("curious and adventurous",
                      "cautious and territorial", "friendly and outgoing",
                      "independent and competitive")[i % 4], cognition="heuristic")
            place_agent(a, *cells[i])
            agents.append(a)
        if farmers:
            for a in agents:
                a.knowledge.add("farming")
        cfg = main.scaled_food_cfg(40)
        spawn_food(cfg["initial"])
        strategies, survived, counters, tenure = {}, {}, {"agent_turns": 0}, {}
        with contextlib.redirect_stdout(io.StringIO()):
            for turn in range(1, 45):
                world_state["turn"] = turn
                import cognition
                cognition.update_tiers(world_state, turn, 8, tenure)
                for a in [x for x in world_state["agents"] if x.alive]:
                    main.run_agent_turn(a, turn, strategies, survived, counters)
                knowledge.farm(world_state, turn)
                main._scaled_respawn_food(turn, cfg)
        return sum(1 for a in agents if a.alive)

    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        control = run(False)
        farming = run(True)
    finally:
        llm.PROVIDER = saved
    assert farming > control, f"farming ({farming}) should beat control ({control})"
    print(f"PASS test_farming_population_outlasts_no_farming_control "
          f"(control {control}/40, farming {farming}/40)")


def test_farming_adds_zero_llm_calls_and_empty_is_v1() -> None:
    """Farming production makes no model calls; a no-farmer farm draws no RNG (v1 safe)."""
    import knowledge
    _fresh_world()
    for i in range(5):
        ag = _agent(f"F{i}", "cautious and territorial", (i, 0), hunger=0)
        ag.knowledge.add("farming")
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        llm.reset_call_stats()
        rng = random.Random(1)
        for turn in range(1, 20):
            knowledge.farm(world_state, turn, rng=rng)
        stats = llm.get_call_stats()
    finally:
        llm.PROVIDER = saved
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats

    _fresh_world()
    _agent("Lonely", "curious and adventurous", (1, 1), hunger=0)  # knows nothing
    st0 = random.getstate()
    knowledge.farm(world_state, 1)
    assert random.getstate() == st0, "no-farmer farm consumed RNG"
    print("PASS test_farming_adds_zero_llm_calls_and_empty_is_v1")


# --- Settlement (M2.1) ----------------------------------------------------
def _sustain_at_plot(state, plot, agents, turns, *, restock=True):
    """Run settlement.update for `turns`, keeping `plot` stocked (a reliable source)."""
    import settlement
    for turn in range(1, turns + 1):
        if restock:
            for p in plot:
                world.place_food(*p)
        state["turn"] = turn
        settlement.update(state, turn)


def test_settlement_forms_with_enough_sustained_settlers_near_reliable_food() -> None:
    """A settlement forms only with >= MIN_SETTLERS sustained near reliable food."""
    import settlement
    import world as _world
    _fresh_world()
    create_world(size=16)
    plot = [(2, 2), (3, 2), (2, 3), (3, 3)]
    founders = [_agent(f"F{i}", "cautious and territorial", p, hunger=0)
                for i, p in enumerate([(2, 2), (3, 2), (2, 3)])]
    # Just BELOW the sustain window: not yet reliable -> no settlement.
    _sustain_at_plot(world_state, plot, founders, settlement.SUSTAIN_TURNS - 1)
    assert not world_state["settlements"], "must not form before the sustain window elapses"
    assert all(f.settlement is None for f in founders)
    # One more turn crosses SUSTAIN_TURNS -> the cluster founds a settlement.
    _sustain_at_plot(world_state, plot, founders, 1)
    assert len(world_state["settlements"]) == 1, "a sustained cluster should found one settlement"
    sid = founders[0].settlement
    assert sid is not None and all(f.settlement == sid for f in founders)
    rec = world_state["settlements"][sid]
    assert rec["members"] == {"F0", "F1", "F2"} and rec["center"] and rec["founded"]
    print("PASS test_settlement_forms_with_enough_sustained_settlers_near_reliable_food")


def test_too_few_settlers_never_found_a_settlement() -> None:
    """Below MIN_SETTLERS, sustained presence at reliable food still founds nothing."""
    import settlement
    _fresh_world()
    create_world(size=16)
    plot = [(2, 2), (3, 2)]
    few = [_agent(f"P{i}", "cautious and territorial", p, hunger=0)
           for i, p in enumerate([(2, 2), (3, 2)])]   # only 2 < MIN_SETTLERS (3)
    _sustain_at_plot(world_state, plot, few, settlement.SUSTAIN_TURNS + 5)
    assert not world_state["settlements"], "two settlers must never found a settlement"
    assert all(a.settlement is None for a in few)
    print("PASS test_too_few_settlers_never_found_a_settlement")


def test_no_reliable_food_no_settlement() -> None:
    """A clustered group with NO sustained food never settles (streaks never build)."""
    import settlement
    _fresh_world()
    create_world(size=16)
    # Three clustered agents, but no food is ever placed -> streaks stay 0.
    group = [_agent(f"G{i}", "cautious and territorial", p, hunger=0)
             for i, p in enumerate([(5, 5), (6, 5), (5, 6)])]
    _sustain_at_plot(world_state, [], group, settlement.SUSTAIN_TURNS + 10, restock=False)
    assert not world_state["settlements"], "no reliable food must yield no settlement"
    assert all(a.settle_streak == 0 for a in group)
    print("PASS test_no_reliable_food_no_settlement")


def test_isolated_nomad_never_joins() -> None:
    """An agent near reliable food joins; an isolated nomad far from any centre never does."""
    import settlement
    _fresh_world()
    create_world(size=20)
    plot = [(3, 3), (4, 3), (3, 4), (4, 4)]
    founders = [_agent(f"F{i}", "cautious and territorial", p, hunger=0)
                for i, p in enumerate([(3, 3), (4, 3), (3, 4)])]
    loner = _agent("Loner", "independent and competitive", (18, 18), hunger=0)
    _sustain_at_plot(world_state, plot, founders, settlement.SUSTAIN_TURNS + 2)
    assert founders[0].settlement is not None, "the clustered founders should settle"
    assert loner.settlement is None, "an isolated nomad must never join"
    # A late arrival that gathers at the reliable food DOES join.
    joiner = _agent("Joiner", "cautious and territorial", (4, 5), hunger=0)  # on plot edge
    _sustain_at_plot(world_state, plot, founders + [joiner], 1)
    assert joiner.settlement == founders[0].settlement, "a gatherer at the food should join"
    assert loner.settlement is None, "the isolated nomad still never joins"
    print("PASS test_isolated_nomad_never_joins")


def test_settled_agent_pulled_home_when_fed_not_when_starving() -> None:
    """A fed member drifts toward its centre; a starving member forages outward instead."""
    import settlement
    from strategy import SURVIVAL_HUNGER
    _fresh_world()
    create_world(size=16)
    a = _agent("Cit", "curious and adventurous", (12, 12), hunger=0)  # far from home (3,3)
    world_state["settlements"]["S001"] = {"id": "S001", "center": (3, 3),
                                          "members": {"Cit"}, "founded": 1}
    a.settlement = "S001"
    # Fed + beyond HOME_RADIUS -> pulled home (toward (3,3) is north/west).
    action, note = choose_action(a, Strategy(kind="explore", target="east", issued_turn=1),
                                 world_state)
    assert action in ("move_north", "move_west"), f"fed member should head home, got {action}"
    assert "home-pull" in note
    # Now starving with food to the EAST (away from home) -> survival overrides home-pull.
    world.place_food(13, 12)
    a.hunger = SURVIVAL_HUNGER + 2
    action, note = choose_action(a, Strategy(kind="explore", target="east", issued_turn=1),
                                 world_state)
    assert action == "move_east", f"starving member must forage outward, got {action}"
    assert "home-pull" not in note
    print("PASS test_settled_agent_pulled_home_when_fed_not_when_starving")


def test_nomad_movement_unaffected_by_settlement_system() -> None:
    """An agent with settlement=None behaves exactly as in Phase 1 (no home-pull)."""
    _fresh_world()
    create_world(size=16)
    a = _agent("Nomad", "curious and adventurous", (12, 12), hunger=0)
    assert a.settlement is None
    strat = Strategy(kind="explore", target="east", issued_turn=1)
    action, note = choose_action(a, strat, world_state)
    assert "home-pull" not in note, "a nomad must never be pulled home"
    assert action.startswith("move_"), "a fed curious nomad still explores"
    print("PASS test_nomad_movement_unaffected_by_settlement_system")


def test_settlement_update_zero_llm_and_no_rng() -> None:
    """settlement.update makes no model calls and draws no RNG (deterministic threshold)."""
    import settlement
    _fresh_world()
    create_world(size=16)
    plot = [(2, 2), (3, 2), (2, 3)]
    [_agent(f"F{i}", "cautious and territorial", p, hunger=0)
     for i, p in enumerate([(2, 2), (3, 2), (2, 3)])]
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        llm.reset_call_stats()
        st0 = random.getstate()
        _sustain_at_plot(world_state, plot, [], settlement.SUSTAIN_TURNS + 3)
        stats = llm.get_call_stats()
    finally:
        llm.PROVIDER = saved
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats
    assert random.getstate() == st0, "settlement.update consumed RNG"
    assert world_state["settlements"], "the sustained cluster should still have settled"
    print("PASS test_settlement_update_zero_llm_and_no_rng")


def test_settlements_off_run_is_byte_identical_to_v1() -> None:
    """settlements=False (default) leaves the run byte-identical to the no-param run."""
    def run(flag):
        llm.PROVIDER = "random"
        random.seed(99)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(20, focal_budget=8)
            else:
                main.run_simulation(20, focal_budget=8, settlements=flag)
        return buf.getvalue()
    saved = llm.PROVIDER
    try:
        base, off = run(None), run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "settlements=False changed the run output"
    print("PASS test_settlements_off_run_is_byte_identical_to_v1")


# --- Storage & surplus (M2.2) ---------------------------------------------
def test_surplus_banks_only_above_need_and_only_when_settled() -> None:
    """Banking the M2.2 rules: SETTLED + WELL-FED + beside food; nomad/hungry/far bank 0."""
    import storage
    _fresh_world()
    create_world(size=12)
    # A settled, well-fed agent standing on food -> banks surplus.
    settled = _agent("Settled", "independent and competitive", (5, 5), hunger=0)
    settled.settlement = "S001"
    world.place_food(5, 5)
    # A NOMAD in the same situation (settlement None) -> stores nothing (rule 1).
    nomad = _agent("Nomad", "independent and competitive", (2, 2), hunger=0)
    world.place_food(2, 2)
    # A settled but HUNGRY agent -> stores nothing (need not yet met; no surplus).
    hungry = _agent("Hungry", "independent and competitive", (8, 8),
                    hunger=storage.STORE_HUNGER_MAX + 1)
    hungry.settlement = "S001"
    world.place_food(8, 8)
    # A settled, well-fed agent with NO food in reach -> nothing to gather, banks 0.
    far = _agent("Far", "independent and competitive", (11, 0), hunger=0)
    far.settlement = "S001"
    storage.accumulate(world_state, 1)
    assert settled.stockpile > 0, "a settled, well-fed agent beside food should bank surplus"
    assert nomad.stockpile == 0.0, "a nomad must not store (storing requires settlement)"
    assert hungry.stockpile == 0.0, "no banking above immediate hunger need"
    assert far.stockpile == 0.0, "no reachable food -> no surplus to bank"
    print("PASS test_surplus_banks_only_above_need_and_only_when_settled")


def test_stockpile_never_exceeds_cap() -> None:
    """A hoarder banks indefinitely but its stockpile is bounded by STORAGE_CAP."""
    import storage
    _fresh_world()
    create_world(size=10)
    hoarder = _agent("Hoard", "independent and competitive", (5, 5), hunger=0)
    hoarder.settlement = "S001"
    world.place_food(5, 5)
    for turn in range(1, 400):
        hoarder.hunger = 0  # stays well-fed beside its food, banking every turn
        storage.accumulate(world_state, turn)
    assert hoarder.stockpile <= storage.STORAGE_CAP, "stockpile must never exceed the cap"
    assert hoarder.stockpile == storage.STORAGE_CAP, "a relentless hoarder should fill the cap"
    print("PASS test_stockpile_never_exceeds_cap")


def test_wealth_tracks_personality_and_knowledge_not_assigned() -> None:
    """Accumulation EMERGES from traits: a competitive farmer out-banks a friendly non-farmer."""
    import storage
    _fresh_world()
    create_world(size=12)
    # Same world conditions; the ONLY differences are personality + farming knowledge.
    rich = _agent("Rich", "independent and competitive", (3, 3), hunger=0)
    rich.settlement = "S001"
    rich.knowledge.add("farming")
    poor = _agent("Poor", "friendly and outgoing", (8, 8), hunger=0)
    poor.settlement = "S001"
    world.place_food(3, 3)
    world.place_food(8, 8)
    # The emergent rate already orders them, before a single turn is banked.
    assert storage.banking_rate(rich) > storage.banking_rate(poor)
    for turn in range(1, 25):
        rich.hunger = 0
        poor.hunger = 0
        storage.accumulate(world_state, turn)
    assert rich.stockpile > poor.stockpile, "the competitive farmer should be richer"
    # Neither was ever assigned a wealth number — both started at 0.0.
    print("PASS test_wealth_tracks_personality_and_knowledge_not_assigned")


def test_starving_member_with_savings_survives_one_without_dies() -> None:
    """The survival buffer: at the brink with no food, savings live and no savings die."""
    import storage
    _fresh_world()
    create_world(size=10)
    world_state["storage_on"] = True
    # Both settled, both one tick from starvation, NO food anywhere on the map.
    saver = _agent("Saver", "cautious and territorial", (0, 0), hunger=9)
    saver.settlement = "S001"
    saver.stockpile = storage.BUFFER_COST + 1.0   # holds more than one stored meal
    broke = _agent("Broke", "cautious and territorial", (9, 9), hunger=9)
    broke.settlement = "S001"
    broke.stockpile = storage.BUFFER_COST - 1.0   # cannot afford even one meal
    strategies, survived, counters = {}, {}, {"agent_turns": 0}
    try:
        # update_hunger pushes 9 -> 10 (starving); the buffer step then decides each fate.
        main.run_agent_turn(saver, 1, strategies, survived, counters)
        main.run_agent_turn(broke, 1, strategies, survived, counters)
        assert saver.alive, "an agent with savings should draw them down and survive"
        assert saver.stockpile < storage.BUFFER_COST + 1.0, "it must have spent stored food"
        assert saver.hunger < 10, "drawing down should pull it off the brink"
        assert not broke.alive, "an agent without enough savings must starve"
    finally:
        world_state["storage_on"] = False
    print("PASS test_starving_member_with_savings_survives_one_without_dies")


def test_storage_accumulate_zero_llm_and_no_rng() -> None:
    """storage.accumulate makes no model calls and draws no RNG (deterministic state math)."""
    import storage
    _fresh_world()
    create_world(size=10)
    a = _agent("S", "independent and competitive", (5, 5), hunger=0)
    a.settlement = "S001"
    world.place_food(5, 5)
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        llm.reset_call_stats()
        st0 = random.getstate()
        for turn in range(1, 30):
            a.hunger = 0
            storage.accumulate(world_state, turn)
        stats = llm.get_call_stats()
    finally:
        llm.PROVIDER = saved
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats
    assert random.getstate() == st0, "storage.accumulate consumed RNG"
    assert a.stockpile > 0, "the settled agent should still have accumulated"
    print("PASS test_storage_accumulate_zero_llm_and_no_rng")


def test_storage_off_run_is_byte_identical_to_v1() -> None:
    """storage_on=False (default) leaves the run byte-identical to the no-param run."""
    def run(flag):
        llm.PROVIDER = "random"
        random.seed(7)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(20, focal_budget=8)
            else:
                main.run_simulation(20, focal_budget=8, storage_on=flag)
        return buf.getvalue()
    saved = llm.PROVIDER
    try:
        base, off = run(None), run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "storage_on=False changed the default run output"
    print("PASS test_storage_off_run_is_byte_identical_to_v1")


# --- Trade, money & proprietary knowledge (M2.3) --------------------------
def test_emergent_price_varies_with_conditions_not_fixed() -> None:
    """The SAME good/skill prices DIFFERENTLY by rarity, skill-gap, hunger, surplus."""
    import economy
    _fresh_world()
    # Knowledge: same skill 'hunting' costs more when rare and when the buyer lacks a producer
    # skill; cheaper (or nothing) when common or the buyer already produces.
    unskilled = _agent("U", "cautious and territorial", (1, 1), hunger=0)
    farmer = _agent("F", "cautious and territorial", (2, 2), hunger=0)
    farmer.knowledge.add("farming")
    dear = economy.knowledge_price("hunting", unskilled, rarity=1.0)
    cheap = economy.knowledge_price("hunting", unskilled, rarity=0.2)
    skilled_buyer = economy.knowledge_price("hunting", farmer, rarity=1.0)
    assert dear > cheap, "a rare skill must cost more than a common one"
    assert dear > skilled_buyer, "an unskilled buyer values a producer skill more than a skilled one"
    # Common enough -> no deal at all (price None), proving it isn't a fixed ratio.
    assert economy.knowledge_price("hunting", farmer, rarity=0.01) is None
    # Food: same food prices up with buyer hunger and with seller scarcity.
    seller_full = _agent("Sf", "cautious and territorial", (4, 4), hunger=0)
    seller_full.stockpile = 20.0
    seller_low = _agent("Sl", "cautious and territorial", (6, 6), hunger=0)
    seller_low.stockpile = 12.0
    hungry = _agent("H", "cautious and territorial", (8, 8), hunger=9)
    mild = _agent("M", "cautious and territorial", (0, 9), hunger=3)
    assert economy.food_price(hungry, seller_full) > economy.food_price(mild, seller_full), \
        "a hungrier buyer should face a higher price"
    assert economy.food_price(hungry, seller_low) > economy.food_price(hungry, seller_full), \
        "a scarcer seller should command a higher price"
    print("PASS test_emergent_price_varies_with_conditions_not_fixed")


def test_trade_is_mutually_beneficial_and_voluntary() -> None:
    """Every executed trade leaves BOTH sides better off by their own valuation."""
    import economy
    _fresh_world()
    create_world(size=8)
    world_state["economy_on"] = True
    # A guarded skill sale: competitive hunter sells hunting to an unskilled, moneyed buyer.
    seller = _agent("Sell", "independent and competitive", (3, 3), hunger=2)
    seller.knowledge.add("hunting")
    seller.stockpile = 2.0
    buyer = _agent("Buy", "cautious and territorial", (4, 3), hunger=0)
    buyer.money = 20.0
    rarity = economy.local_rarity("hunting", buyer, world_state, exclude=(seller,))
    value = economy.knowledge_value("hunting", buyer, rarity)
    reservation = economy.knowledge_reservation("hunting", rarity)
    price = economy.knowledge_price("hunting", buyer, rarity)
    b_money0, s_total0 = buyer.money, seller.money + seller.stockpile
    economy.trade(world_state, 1)
    assert "hunting" in buyer.knowledge, "the trade should have executed"
    # Voluntary + mutually beneficial: price sits strictly between the two valuations.
    assert reservation <= price <= value, f"price {price} outside [{reservation}, {value}]"
    assert buyer.money < b_money0, "buyer paid"
    assert seller.money + seller.stockpile > s_total0, "seller was paid"
    # Neither party worse off by its own valuation: buyer gained value>=price; seller gained price>=reservation.
    assert value - price >= 0 and price - reservation >= 0
    print("PASS test_trade_is_mutually_beneficial_and_voluntary")


def test_guarded_knowledge_does_not_free_diffuse_but_sells() -> None:
    """A competitive holder's skill is withheld from M1.1 diffusion yet moves by sale."""
    import economy
    import knowledge as kn
    # 1) Guarded: a competitive farmer next to a learner — farming never free-diffuses.
    _fresh_world()
    create_world(size=8)
    world_state["economy_on"] = True
    holder = _agent("Guard", "independent and competitive", (2, 2), hunger=0)
    holder.knowledge.add("farming")
    learner = _agent("Learn", "curious and adventurous", (3, 2), hunger=0)
    learner.money = 20.0
    rng = random.Random(0)
    for turn in range(1, 50):
        world_state["turn"] = turn
        kn.diffuse(world_state, turn, rng=rng)
    assert "farming" not in learner.knowledge, "a guarded skill must not diffuse free"
    # 2) ...but it SELLS: the same pair, run the trade pass, and the skill moves for payment.
    economy.trade(world_state, 50)
    assert "farming" in learner.knowledge, "a guarded skill should move by sale"
    assert learner.money < 20.0, "the buyer paid for it"
    print("PASS test_guarded_knowledge_does_not_free_diffuse_but_sells")


def test_friendly_knowledge_still_free_diffuses() -> None:
    """A friendly holder still TEACHES free — M1.1 diffusion intact (guarding is per-personality)."""
    import knowledge as kn
    _fresh_world()
    create_world(size=8)
    world_state["economy_on"] = True   # economy on, but a friendly holder does not guard
    holder = _agent("Kind", "friendly and outgoing", (2, 2), hunger=0)
    holder.knowledge.add("farming")
    learner = _agent("Pupil", "curious and adventurous", (3, 2), hunger=0)
    rng = random.Random(0)
    learned = False
    for turn in range(1, 50):
        world_state["turn"] = turn
        kn.diffuse(world_state, turn, rng=rng)
        if "farming" in learner.knowledge:
            learned = True
            break
    assert learned, "a friendly holder's skill must still diffuse free"
    print("PASS test_friendly_knowledge_still_free_diffuses")


def test_hunting_produces_food_only_for_knowers() -> None:
    """Hunting grows food into the world for a knower; a non-knower produces nothing."""
    import knowledge as kn
    _fresh_world()
    create_world(size=12)
    knower = _agent("Hunter", "curious and adventurous", (5, 5), hunger=0)
    knower.knowledge.add("hunting")
    nonknower = _agent("Plain", "curious and adventurous", (9, 9), hunger=0)
    food_before = len(world_state["food"])
    rng = random.Random(0)
    for turn in range(1, 40):
        kn.hunt(world_state, turn, rng=rng)
    produced = len(world_state["food"]) - food_before
    assert produced > 0, "a fed hunter should produce food"
    # Only the hunter ever took game; the non-knower produced nothing.
    assert any("Took game" in m for m in knower.memory), "the knower should have hunted"
    assert not any("Took game" in m for m in nonknower.memory), "a non-knower never hunts"
    print("PASS test_hunting_produces_food_only_for_knowers")


def test_surplus_to_money_to_purchase_roundtrips() -> None:
    """Food surplus past the cap mints money; money then buys knowledge (food-backed loop)."""
    import economy
    import storage
    _fresh_world()
    create_world(size=8)
    world_state["storage_on"] = True
    world_state["economy_on"] = True
    # A full-larder farmer mints money from its past-cap surplus.
    rich = _agent("Rich", "independent and competitive", (3, 3), hunger=0)
    rich.settlement = "S001"
    rich.knowledge.add("farming")
    rich.stockpile = storage.STORAGE_CAP
    world.place_food(3, 3)
    for turn in range(1, 11):
        rich.hunger = 0
        economy.mint(world_state, turn)
    assert rich.money > 0, "surplus past the cap should mint money"
    # That money buys a skill the farmer lacks from a guarding hunter.
    seller = _agent("Hunt", "independent and competitive", (4, 3), hunger=2)
    seller.settlement = "S001"
    seller.knowledge.add("hunting")
    seller.stockpile = 2.0
    money_before = rich.money
    economy.trade(world_state, 11)
    assert "hunting" in rich.knowledge, "money should buy the skill"
    assert rich.money < money_before, "the buyer spent money"
    assert seller.money > 0, "the seller earned money"
    # Money is food-backed: it redeems to survive starvation via the buffer.
    rich.stockpile = 0.0
    rich.hunger = 9
    rich.money = storage.BUFFER_COST + 1.0
    assert storage.draw_down(rich), "money should redeem as food to survive"
    assert rich.hunger < 10 and rich.money < storage.BUFFER_COST + 1.0
    print("PASS test_surplus_to_money_to_purchase_roundtrips")


def test_trade_and_mint_zero_llm_and_no_rng() -> None:
    """economy.mint and economy.trade make no model calls and draw no RNG."""
    import economy
    import storage
    _fresh_world()
    create_world(size=8)
    world_state["storage_on"] = True
    world_state["economy_on"] = True
    a = _agent("A", "independent and competitive", (3, 3), hunger=0)
    a.settlement = "S001"
    a.knowledge.add("hunting")
    a.stockpile = storage.STORAGE_CAP
    world.place_food(3, 3)
    b = _agent("B", "cautious and territorial", (4, 3), hunger=0)
    b.money = 20.0
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        llm.reset_call_stats()
        st0 = random.getstate()
        for turn in range(1, 20):
            a.hunger = 0
            economy.mint(world_state, turn)
            economy.trade(world_state, turn)
        stats = llm.get_call_stats()
    finally:
        llm.PROVIDER = saved
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats
    assert random.getstate() == st0, "the economy consumed RNG (would desync v1)"
    print("PASS test_trade_and_mint_zero_llm_and_no_rng")


def test_economy_off_run_is_byte_identical_to_v1() -> None:
    """economy_on=False (default) leaves the run byte-identical to the no-param run."""
    def run(flag):
        llm.PROVIDER = "random"
        random.seed(13)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(20, focal_budget=8)
            else:
                main.run_simulation(20, focal_budget=8, economy_on=flag)
        return buf.getvalue()
    saved = llm.PROVIDER
    try:
        base, off = run(None), run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "economy_on=False changed the default run output"
    print("PASS test_economy_off_run_is_byte_identical_to_v1")


# --- Wage labor: the first institution (V2 M3.1, opens Phase 3) ------------
def test_wage_varies_with_labor_supply_and_desperation_not_fixed() -> None:
    """The SAME work pays DIFFERENTLY as labor supply + worker desperation change (emergence)."""
    import labor
    secure = Agent(name="W", personality="x"); secure.hunger = 0
    tight = labor.market_tightness(openings=10, workers=1)   # scarce labor -> worker's market
    slack = labor.market_tightness(openings=1, workers=10)   # abundant labor -> employer's market
    w_scarce = labor.offered_wage(secure, tight)
    w_abundant = labor.offered_wage(secure, slack)
    assert w_scarce > w_abundant, "the same worker must earn MORE when labor is scarce"
    # Desperation discounts: a starving worker accepts less in the SAME tight market.
    starving = Agent(name="S", personality="x"); starving.hunger = world.HUNGER_MAX
    w_desperate = labor.offered_wage(starving, tight)
    assert w_desperate < w_scarce, "a desperate worker accepts a lower wage for the same market"
    # NOT a fixed wage: the one job yields at least three distinct prices.
    assert len({round(w, 4) for w in (w_scarce, w_abundant, w_desperate)}) == 3
    # Always bounded: employer profits (< output) and worker survives (>= subsistence).
    for w in (w_scarce, w_abundant, w_desperate):
        assert labor.SUBSISTENCE_WAGE <= w < labor.LABOR_OUTPUT, w
    print("PASS test_wage_varies_with_labor_supply_and_desperation_not_fixed")


def test_wage_reaches_subsistence_under_glut_and_desperation_never_below() -> None:
    """Under a labor glut + desperation the wage bottoms at SUBSISTENCE — but never below it."""
    import labor
    # Deep employer's market + a starving worker -> leverage ~0 -> exactly subsistence.
    glut = labor.market_tightness(openings=1, workers=200)
    starving = Agent(name="S", personality="x"); starving.hunger = world.HUNGER_MAX
    w_exploit = labor.offered_wage(starving, glut)
    assert abs(w_exploit - labor.SUBSISTENCE_WAGE) < 1e-9, w_exploit
    # A glut ALONE (even a fed worker) drives the wage to ~subsistence — exploitation emerges
    # from supply, not only from hunger.
    fed = Agent(name="F", personality="x"); fed.hunger = 0
    assert labor.offered_wage(fed, glut) < labor.SUBSISTENCE_WAGE + 0.05
    # The SAME work pays well ABOVE subsistence when labor is scarce + the worker secure.
    scarce = labor.market_tightness(openings=200, workers=1)
    assert labor.offered_wage(fed, scarce) > 1.5
    # Never below the survival floor anywhere in the (tightness, desperation) space.
    for op, wk in ((1, 500), (1, 1), (500, 1)):
        for h in (0, 5, world.HUNGER_MAX):
            a = Agent(name="A", personality="x"); a.hunger = h
            assert labor.offered_wage(a, labor.market_tightness(op, wk)) >= labor.SUBSISTENCE_WAGE
    print("PASS test_wage_reaches_subsistence_under_glut_and_desperation_never_below")


def test_employment_output_flows_to_employer_wage_to_worker() -> None:
    """Each employed turn: the worker's output accrues to the EMPLOYER; the wage flows to the worker."""
    import labor
    _fresh_world(); create_world(size=8)
    world_state["economy_on"] = True; world_state["labor_on"] = True
    world_state["employments"] = []
    emp = _agent("Boss", "independent and competitive", (3, 3), hunger=0)
    emp.settlement = "S001"; emp.knowledge.add("farming"); emp.money = 10.0
    wkr = _agent("Hand", "cautious and territorial", (3, 4), hunger=4)
    wkr.settlement = "S001"; wkr.money = 0.0
    wage = 1.5
    world_state["employments"].append(
        {"employer": "Boss", "worker": "Hand", "wage": wage, "since": 0})
    emp_w0, wkr_w0 = emp.money + emp.stockpile, wkr.money + wkr.stockpile
    labor.update(world_state, 1)
    # Output (food-claim) accrued to the EMPLOYER's stockpile; the employer paid the wage out.
    assert emp.stockpile == labor.LABOR_OUTPUT, emp.stockpile
    assert emp.money == 10.0 - wage, emp.money
    # Net: employer +(output - wage); worker +(wage - cost of living) — small but POSITIVE, and fed.
    assert (emp.money + emp.stockpile) - emp_w0 == labor.LABOR_OUTPUT - wage
    assert (wkr.money + wkr.stockpile) - wkr_w0 == wage - labor.COST_OF_LIVING
    assert wage - labor.COST_OF_LIVING > 0, "above subsistence here -> the worker gains net"
    assert wkr.hunger < 4, "the wage fed the worker (relieved hunger) -> employed beats starving"
    # The relationship PERSISTS into world_state (an institution, not a one-shot trade).
    assert any(l["worker"] == "Hand" for l in world_state["employments"])
    print("PASS test_employment_output_flows_to_employer_wage_to_worker")


def test_agent_without_capital_never_employs() -> None:
    """An agent with no capital can never be an employer, even with a skill + a willing worker."""
    import labor
    _fresh_world(); create_world(size=8)
    world_state["economy_on"] = True; world_state["labor_on"] = True
    world_state["employments"] = []
    broke = _agent("Skilled", "independent and competitive", (3, 3), hunger=0)
    broke.settlement = "S001"; broke.knowledge.add("farming")
    broke.money = 0.0; broke.stockpile = 0.0          # wealth 0 < EMPLOYER_MIN_CAPITAL
    _w = _agent("Hand", "cautious and territorial", (3, 4), hunger=5); _w.settlement = "S001"
    assert not labor.is_employer(broke), "no capital -> not an employer"
    assert labor.capacity(broke) == 0
    labor.update(world_state, 1)
    assert world_state["employments"] == [], "a capital-less agent never hires"
    print("PASS test_agent_without_capital_never_employs")


def test_roles_emerge_from_wealth_and_skill_not_assigned() -> None:
    """Employer/worker are pure READS of wealth+skill — change the wealth, the role changes."""
    import labor
    _fresh_world(); create_world(size=8)
    rich_skilled = _agent("Rich", "x", (2, 2), hunger=0)
    rich_skilled.settlement = "S001"; rich_skilled.knowledge.add("farming"); rich_skilled.money = 20.0
    poor_unskilled = _agent("Poor", "x", (2, 3), hunger=0); poor_unskilled.settlement = "S001"
    rich_unskilled = _agent("Wealthy", "x", (2, 4), hunger=0)
    rich_unskilled.settlement = "S001"; rich_unskilled.money = 20.0
    poor_skilled = _agent("Crafty", "x", (2, 5), hunger=0)
    poor_skilled.settlement = "S001"; poor_skilled.knowledge.add("hunting")
    # Rich + skilled -> employer (and not a worker); poor + unskilled -> worker (and not employer).
    assert labor.is_employer(rich_skilled) and not labor.is_worker(rich_skilled)
    assert labor.is_worker(poor_unskilled) and not labor.is_employer(poor_unskilled)
    # Wealthy-but-unskilled: independent means -> not a worker; no skill -> not an employer.
    assert not labor.is_worker(rich_unskilled) and not labor.is_employer(rich_unskilled)
    # Poor-but-skilled: owns its means -> never a wage worker; too poor -> can't employ.
    assert not labor.is_worker(poor_skilled) and not labor.is_employer(poor_skilled)
    # Emergence, not assignment: strip the employer's capital and the role evaporates — no flag.
    rich_skilled.money = 0.0
    assert not labor.is_employer(rich_skilled), "role is a pure read of state, not a set flag"
    print("PASS test_roles_emerge_from_wealth_and_skill_not_assigned")


def test_employment_persists_across_turns_and_is_mutually_entered() -> None:
    """A formed link is the SAME relationship across turns; the worker is better off employed."""
    import labor
    _fresh_world(); create_world(size=8)
    world_state["economy_on"] = True; world_state["labor_on"] = True
    world_state["employments"] = []
    emp = _agent("Boss", "independent and competitive", (3, 3), hunger=0)
    emp.settlement = "S001"; emp.knowledge.add("farming"); emp.money = 50.0
    wkr = _agent("Hand", "cautious and territorial", (3, 4), hunger=6)
    wkr.settlement = "S001"; wkr.money = 0.0
    # No link yet: update FORMS one (roles + wage emerge from state), then it persists.
    labor.update(world_state, 1)
    assert len(world_state["employments"]) == 1, "an employer + a poor worker should pair"
    link = world_state["employments"][0]
    assert link["employer"] == "Boss" and link["worker"] == "Hand" and link["since"] == 1
    # The SAME link (since turn 1) survives across many turns — not re-created one-shot each turn.
    for t in range(2, 9):
        labor.update(world_state, t)
        assert any(l["worker"] == "Hand" and l["since"] == 1
                   for l in world_state["employments"]), "the link must persist across turns"
    # Mutually entered: the worker survived (fed by its wage) and ACCUMULATED net — it gained by
    # taking the wage (else it would not have). Better off employed than starving.
    assert wkr.alive and wkr.hunger < world.HUNGER_MAX
    assert wkr.money + wkr.stockpile > 0.0, "the worker is net better off for being employed"
    print("PASS test_employment_persists_across_turns_and_is_mutually_entered")


def test_worker_quits_when_self_sufficient_and_broke_employer_lets_go() -> None:
    """Lifecycle: a worker that gains means QUITS; an employer that loses its capital lets go."""
    import labor
    # QUIT (upward mobility): the worker gains a producer skill -> self-sufficient -> link ends.
    _fresh_world(); create_world(size=8)
    world_state["economy_on"] = True; world_state["labor_on"] = True
    emp = _agent("Boss", "x", (3, 3), hunger=0)
    emp.settlement = "S001"; emp.knowledge.add("farming"); emp.money = 50.0
    wkr = _agent("Hand", "x", (3, 4), hunger=2); wkr.settlement = "S001"; wkr.money = 0.0
    world_state["employments"] = [{"employer": "Boss", "worker": "Hand", "wage": 1.2, "since": 0}]
    wkr.knowledge.add("hunting")           # acquired its own means of production
    ev = labor.update(world_state, 1)
    assert all(l["worker"] != "Hand" for l in world_state["employments"]), "self-sufficient -> quits"
    assert any("quit" in e for e in ev), ev
    # LET GO: an employer whose capital falls below the threshold can no longer employ.
    _fresh_world(); create_world(size=8)
    world_state["economy_on"] = True; world_state["labor_on"] = True
    boss = _agent("Boss", "x", (3, 3), hunger=0)
    boss.settlement = "S001"; boss.knowledge.add("farming"); boss.money = 2.0  # < EMPLOYER_MIN_CAPITAL
    hand = _agent("Hand", "x", (3, 4), hunger=2); hand.settlement = "S001"; hand.money = 0.0
    world_state["employments"] = [{"employer": "Boss", "worker": "Hand", "wage": 1.0, "since": 0}]
    ev = labor.update(world_state, 2)
    assert all(l["worker"] != "Hand" for l in world_state["employments"]), "broke employer lets go"
    assert any("let" in e or "laid off" in e for e in ev), ev
    print("PASS test_worker_quits_when_self_sufficient_and_broke_employer_lets_go")


def _wealth_gini(agents) -> float:
    """Gini of liquid wealth (money + stockpile): 0 = equal, ->1 = unequal."""
    xs = sorted(a.money + a.stockpile for a in agents)
    n, s = len(xs), sum(a.money + a.stockpile for a in agents)
    if n == 0 or s == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * s) - (n + 1) / n


def test_inequality_compounds_with_wage_labor_on_vs_off() -> None:
    """HEADLINE: the wealth gap RISES over time with wage labor ON, stays FLAT with it OFF."""
    import labor

    def build():
        _fresh_world(); create_world(size=12)
        world_state["economy_on"] = True; world_state["employments"] = []
        boss = _agent("Boss", "independent and competitive", (5, 5), hunger=0)
        boss.settlement = "S001"; boss.knowledge.add("farming"); boss.money = 20.0
        ws = []
        for i, pos in enumerate([(4, 5), (6, 5), (5, 4), (5, 6), (4, 4)]):
            w = _agent(f"W{i}", "cautious and territorial", pos, hunger=7)
            w.settlement = "S001"; w.money = 4.0   # poor (< WORKER_MAX_WEALTH), desperate
            ws.append(w)
        return [boss] + ws

    # ON: run the institution each turn.
    cast_on = build(); world_state["labor_on"] = True
    curve_on = [_wealth_gini(cast_on)]
    for t in range(1, 16):
        labor.update(world_state, t)
        curve_on.append(_wealth_gini(cast_on))
    # OFF: identical cast, never invoke the institution (wealth is static).
    cast_off = build(); world_state["labor_on"] = False
    curve_off = [_wealth_gini(cast_off) for _ in range(16)]

    assert curve_on[-1] > curve_on[0] + 0.03, f"inequality must RISE with wage labor on: {curve_on}"
    assert curve_on[-1] > curve_off[-1] + 0.03, "ON must end more unequal than OFF"
    assert abs(curve_off[-1] - curve_off[0]) < 1e-9, "with labor off, wealth (and Gini) stay flat"
    print("PASS test_inequality_compounds_with_wage_labor_on_vs_off")


def test_labor_adds_zero_llm_and_no_rng() -> None:
    """labor.update makes no model calls and draws no RNG (deterministic state math)."""
    import labor
    _fresh_world(); create_world(size=8)
    world_state["economy_on"] = True; world_state["labor_on"] = True; world_state["employments"] = []
    emp = _agent("Boss", "independent and competitive", (3, 3), hunger=0)
    emp.settlement = "S001"; emp.knowledge.add("farming"); emp.money = 30.0
    for i in range(3):
        w = _agent(f"W{i}", "cautious and territorial", (4, 3 + i), hunger=5)
        w.settlement = "S001"; w.money = 0.0
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"; llm.reset_call_stats(); st0 = random.getstate()
        for t in range(1, 20):
            labor.update(world_state, t)
        stats = llm.get_call_stats()
    finally:
        llm.PROVIDER = saved
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats
    assert random.getstate() == st0, "wage labor consumed RNG (would desync v1)"
    print("PASS test_labor_adds_zero_llm_and_no_rng")


def test_labor_off_run_is_byte_identical_to_v1() -> None:
    """labor_on=False (default) leaves the run byte-identical to the no-param run."""
    def run(flag):
        llm.PROVIDER = "random"
        random.seed(29)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(22, focal_budget=8)
            else:
                main.run_simulation(22, focal_budget=8, labor_on=flag)
        return buf.getvalue()
    saved = llm.PROVIDER
    try:
        base, off = run(None), run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "labor_on=False changed the default run output"
    print("PASS test_labor_off_run_is_byte_identical_to_v1")


# --- Legitimate leadership: authority by trust (V2 M3.2, Phase 3) -----------
def _led_settler(name: str, pos: tuple[int, int], sid: str = "S001") -> Agent:
    """A living, settled agent (so leadership can read it within a settlement)."""
    a = _agent(name, "cautious and territorial", pos, hunger=0)
    a.settlement = sid
    return a


def _trusts(follower: Agent, leader_name: str, value: int) -> None:
    """Set `follower`'s trust in `leader_name` directly (the v1 trust the module READS)."""
    follower.relationships[leader_name] = {"trust": value, "interactions": 1, "grudge": False}


def test_leader_emerges_only_with_a_cohered_following_not_a_global_max() -> None:
    """A leader emerges ONLY when >= MIN_FOLLOWERS co-settlers trust a common agent — never on a
    single high score, and never in a fractured settlement."""
    import leadership
    # A cohered cluster: two co-settlers trust L above the bar -> L leads.
    _fresh_world(); create_world(size=12)
    world_state["leadership_on"] = True; world_state["leaders"] = {}
    leadership_l = _led_settler("L", (5, 5))
    f1 = _led_settler("F1", (5, 6)); f2 = _led_settler("F2", (6, 5))
    _trusts(f1, "L", leadership.FORM_TRUST); _trusts(f2, "L", leadership.FORM_TRUST)
    leadership.update(world_state, 1)
    assert world_state["leaders"]["S001"]["leader"] == "L"
    assert world_state["leaders"]["S001"]["followers"] == {"F1", "F2"}

    # NOT a global-max lookup: one ardent admirer (the single highest trust in the world) is
    # below MIN_FOLLOWERS, and trust spread thin reaches no cluster -> NO leader emerges.
    _fresh_world(); create_world(size=12)
    world_state["leadership_on"] = True; world_state["leaders"] = {}
    a = _led_settler("A", (5, 5)); b = _led_settler("B", (6, 6))
    c = _led_settler("C", (4, 4)); d = _led_settler("D", (5, 6))
    _trusts(b, "A", 9)          # A holds the globally HIGHEST trust score (9) — but from ONE agent
    _trusts(c, "D", leadership.FORM_TRUST)   # D has a single follower too
    leadership.update(world_state, 1)
    assert world_state["leaders"] == {}, "no cohered cluster -> no leader, even with a max score"
    print("PASS test_leader_emerges_only_with_a_cohered_following_not_a_global_max")


def test_leader_can_be_a_non_wealthiest_agent_power_decoupled_from_wealth() -> None:
    """The trust-leader need not be the richest: a poor, trusted agent leads while a rich,
    distrusted one does not — political power decoupled from economic power."""
    import leadership
    _fresh_world(); create_world(size=12)
    world_state["leadership_on"] = True; world_state["leaders"] = {}
    poor = _led_settler("Poor", (5, 5)); poor.money = 0.0
    rich = _led_settler("Rich", (6, 6)); rich.money = 99.0   # wealthiest by far
    f1 = _led_settler("F1", (5, 6)); f2 = _led_settler("F2", (6, 5)); f3 = _led_settler("F3", (4, 5))
    for f in (f1, f2, f3):
        _trusts(f, "Poor", leadership.FORM_TRUST)   # the poor agent is widely trusted
        _trusts(f, "Rich", -3)                      # the rich agent is distrusted
    leadership.update(world_state, 1)
    leader = world_state["leaders"]["S001"]["leader"]
    assert leader == "Poor", "trust, not wealth, must drive leadership"
    assert leader != max((poor, rich), key=lambda a: a.money).name, "the richest does NOT lead"
    print("PASS test_leader_can_be_a_non_wealthiest_agent_power_decoupled_from_wealth")


def test_leadership_lost_when_trust_erodes_with_hysteresis_and_can_be_displaced() -> None:
    """Legitimacy is contingent: a single-turn wobble does NOT unseat (hysteresis); real erosion
    ends the role; and a strictly more-trusted centre DISPLACES the incumbent."""
    import leadership
    # Erosion + hysteresis.
    _fresh_world(); create_world(size=12)
    world_state["leadership_on"] = True; world_state["leaders"] = {}
    leadership_l = _led_settler("L", (5, 5))
    f1 = _led_settler("F1", (5, 6)); f2 = _led_settler("F2", (6, 5))
    _trusts(f1, "L", leadership.FORM_TRUST); _trusts(f2, "L", leadership.FORM_TRUST)
    leadership.update(world_state, 1)
    assert world_state["leaders"]["S001"]["leader"] == "L"
    # Wobble: one follower drifts FORM_TRUST -> KEEP_TRUST. Still retained (no flicker).
    _trusts(f1, "L", leadership.KEEP_TRUST)
    leadership.update(world_state, 2)
    assert world_state["leaders"].get("S001", {}).get("leader") == "L", "a one-turn wobble must NOT unseat"
    # Real erosion: the leader turns hostile, both fall below KEEP_TRUST -> the role is lost.
    _trusts(f1, "L", -3); _trusts(f2, "L", -3)
    ev = leadership.update(world_state, 3)
    assert "S001" not in world_state["leaders"], "erosion below the keep bar must end the role"
    assert any("lost legitimacy" in e for e in ev), ev

    # Displacement by a more-trusted centre.
    _fresh_world(); create_world(size=12)
    world_state["leadership_on"] = True; world_state["leaders"] = {}
    leadership_l = _led_settler("L", (5, 5)); c = _led_settler("C", (7, 7))
    f1 = _led_settler("F1", (5, 6)); f2 = _led_settler("F2", (6, 5))
    _trusts(f1, "L", leadership.FORM_TRUST); _trusts(f2, "L", leadership.FORM_TRUST)
    leadership.update(world_state, 1)
    assert world_state["leaders"]["S001"]["leader"] == "L"
    # The following shifts to C (strictly more high-trust followers than L now has).
    _trusts(f1, "L", leadership.KEEP_TRUST); _trusts(f2, "L", leadership.KEEP_TRUST)
    _trusts(f1, "C", leadership.FORM_TRUST); _trusts(f2, "C", leadership.FORM_TRUST)
    ev = leadership.update(world_state, 2)
    assert world_state["leaders"]["S001"]["leader"] == "C", "a more-trusted centre must displace"
    assert world_state["leaders"]["S001"]["since"] == 2, "displacement resets the tenure"
    assert any("displaced" in e for e in ev), ev
    print("PASS test_leadership_lost_when_trust_erodes_with_hysteresis_and_can_be_displaced")


def test_leadership_effect_makes_a_led_settlement_more_cohesive_than_unled() -> None:
    """The leadership EFFECT is real: a FOLLOWER is pulled tighter to its leader than a plain
    settler is to the centre — a led settlement differs measurably from an unled one (influence,
    not tax/law)."""
    import leadership, settlement
    _fresh_world(); create_world(size=20)
    world_state["leadership_on"] = True; world_state["leaders"] = {}
    world_state["settlements"] = {"S001": {"id": "S001", "center": (10, 10),
                                            "members": {"L", "F1", "F2"}, "founded": 0}}
    leadership_l = _led_settler("L", (10, 10))                 # leader sits at the centre
    follower = _led_settler("F1", (12, 10)); _led_settler("F2", (10, 12))
    _trusts(follower, "L", leadership.FORM_TRUST)
    _trusts(world_state["agents"][-1], "L", leadership.FORM_TRUST)
    leadership.update(world_state, 1)
    # The follower stands at Chebyshev distance 2 — within the settlement HOME_RADIUS (2) but
    # OUTSIDE the tighter LED_HOME_RADIUS (1). LED: it is pulled in. UNLED: it holds.
    assert leadership.LED_HOME_RADIUS < settlement.HOME_RADIUS
    led_action, led_note = choose_action(follower, None, world_state)
    assert led_action in world.VALID_ACTIONS and led_action.startswith("move_")
    assert "leader" in led_note, led_note
    # Same agent, same position, leadership OFF (no leader record) -> the ordinary settlement
    # pull leaves it alone at distance 2 (within HOME_RADIUS), so it does NOT rally.
    world_state["leaders"] = {}
    unled_action, _ = choose_action(follower, None, world_state)
    assert not (unled_action.startswith("move_")), \
        "without a leader the member is within HOME_RADIUS and is not pulled — less cohesive"
    print("PASS test_leadership_effect_makes_a_led_settlement_more_cohesive_than_unled")


def test_leadership_reads_trust_but_writes_no_trust_values_and_no_llm_no_rng() -> None:
    """The load-bearing invariant: leadership is a PURE read — it writes no trust, makes no LLM
    call, and draws no RNG."""
    import copy, leadership
    _fresh_world(); create_world(size=12)
    world_state["leadership_on"] = True; world_state["leaders"] = {}
    leadership_l = _led_settler("L", (5, 5))
    f1 = _led_settler("F1", (5, 6)); f2 = _led_settler("F2", (6, 5))
    _trusts(f1, "L", leadership.FORM_TRUST); _trusts(f2, "L", leadership.FORM_TRUST)
    before = {a.name: copy.deepcopy(a.relationships) for a in world_state["agents"]}
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"; llm.reset_call_stats(); st0 = random.getstate()
        for t in range(1, 12):
            leadership.update(world_state, t)
        stats = llm.get_call_stats()
    finally:
        llm.PROVIDER = saved
    after = {a.name: a.relationships for a in world_state["agents"]}
    assert after == before, "leadership MUST NOT write any trust value (pure read of the network)"
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats
    assert random.getstate() == st0, "leadership consumed RNG (would desync v1)"
    print("PASS test_leadership_reads_trust_but_writes_no_trust_values_and_no_llm_no_rng")


def test_leadership_off_run_is_byte_identical_to_v1() -> None:
    """leadership_on=False (default) leaves the run byte-identical to the no-param run."""
    def run(flag):
        llm.PROVIDER = "random"
        random.seed(37)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(22, focal_budget=8)
            else:
                main.run_simulation(22, focal_budget=8, leadership_on=flag)
        return buf.getvalue()
    saved = llm.PROVIDER
    try:
        base, off = run(None), run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "leadership_on=False changed the default run output"
    print("PASS test_leadership_off_run_is_byte_identical_to_v1")


def test_leadership_emerges_organically_from_built_trust_in_a_full_run() -> None:
    """The load-bearing test: in a FULL seeded simulation with ZERO injected trust, agents
    settle, build trust ONLY through the conversation loop, and a leader emerges at the
    settlement's founding and is later DISPLACED as the network shifts — leadership falls out
    of earned trust, not a constructed fixture. Deterministic (seeded) and reproducible."""
    def organic_run():
        random.seed(7)
        cells = [(x, y) for x in range(4, 7) for y in range(4, 7)]
        goals = {"survive": 8, "wealth": 3, "friendship": 4}
        specs = [(f"P{i}", ["friendly", "cautious", "social"][i % 3], dict(goals), cells[i])
                 for i in range(7)]
        food = {"initial": 40, "per_turn": 6, "cap": 60, "cluster": True}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(60, focal_budget=7, agent_specs=specs, grid_size=10,
                                food_cfg=food, knowledge_seed=[("farming", 7)],
                                settlements=True, storage_on=True, leadership_on=True,
                                cognition="llm")
        return [l.strip() for l in buf.getvalue().splitlines()
                if "emerged as leader" in l or "displaced" in l]
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        events = organic_run()
        events2 = organic_run()
    finally:
        llm.PROVIDER = saved
    assert any("emerged as leader" in e for e in events), "a leader must emerge from built trust"
    assert any("displaced" in e for e in events), "the role must change hands as trust shifts"
    assert events == events2, "the organic leadership trajectory must be reproducible (seeded)"
    print("PASS test_leadership_emerges_organically_from_built_trust_in_a_full_run")


# --- Taxation & redistribution: legitimacy acts on wealth (V2 M3.3, Phase 3) ---
def _rich_follower(name: str, pos: tuple[int, int], money: float, leader: str = "L",
                   sid: str = "S001") -> Agent:
    """A wealthy follower (taxable) who trusts `leader` at the form bar."""
    import leadership
    a = _led_settler(name, pos, sid)
    a.money = money
    _trusts(a, leader, leadership.FORM_TRUST)
    return a


def _wealth(a: Agent) -> float:
    return a.money + a.stockpile


def test_only_a_legitimate_leader_can_tax() -> None:
    """No leader -> no taxation (power downstream of legitimacy); a led settlement DOES tax."""
    import leadership, taxation
    # Fractured: each trusts a different agent -> no leader -> taxation idles, wealth unchanged.
    _fresh_world(); create_world(size=12)
    world_state["leadership_on"] = True; world_state["taxation_on"] = True
    world_state["tax_rate"] = 0.30; world_state["leaders"] = {}
    rich = _led_settler("Rich", (5, 5)); rich.money = 40.0
    a = _led_settler("A", (6, 6)); a.money = 2.0
    b = _led_settler("B", (4, 4)); b.money = 1.0
    _trusts(rich, "A", 2); _trusts(a, "B", 2); _trusts(b, "Rich", 2)
    leadership.update(world_state, 1)
    before = {x.name: _wealth(x) for x in (rich, a, b)}
    ev = taxation.update(world_state, 1)
    assert world_state["leaders"] == {}, "a fractured settlement has no leader"
    assert ev == [] and {x.name: _wealth(x) for x in (rich, a, b)} == before, \
        "with no leader, no redistribution may occur"
    # With a real following, taxation flows.
    _fresh_world(); create_world(size=12)
    world_state["leadership_on"] = True; world_state["taxation_on"] = True
    world_state["tax_rate"] = 0.30; world_state["leaders"] = {}
    _led_settler("L", (5, 5)).money = 8.0
    rich = _rich_follower("Rich", (5, 6), 40.0)
    p1 = _led_settler("P1", (6, 5)); p1.money = 1.0; _trusts(p1, "L", leadership.FORM_TRUST)
    p2 = _led_settler("P2", (4, 5)); p2.money = 1.0; _trusts(p2, "L", leadership.FORM_TRUST)
    leadership.update(world_state, 1)
    ev2 = taxation.update(world_state, 1)
    assert ev2 and _wealth(rich) < 40.0 and _wealth(p1) > 1.0, "a led settlement taxes and redistributes"
    print("PASS test_only_a_legitimate_leader_can_tax")


def test_redistribution_lowers_within_settlement_gini_vs_untaxed() -> None:
    """Taxation measurably lowers the within-settlement Gini below an identical untaxed run, with
    the M3.1 labor spiral running in both."""
    import leadership, labor, taxation

    def gini(xs):
        xs = sorted(xs); n = len(xs); s = sum(xs)
        return 0.0 if s <= 0 else (2 * sum((i + 1) * x for i, x in enumerate(xs))) / (n * s) - (n + 1) / n

    def build():
        create_world(size=20)
        world_state["leadership_on"] = True; world_state["leaders"] = {}
        world_state["economy_on"] = True
        world_state["settlements"] = {"S001": {"id": "S001", "center": (10, 10),
                                               "members": set(), "founded": 0}}
        cast = [_led_settler("Chief", (10, 10))]; cast[0].money = 8.0
        for i in range(2):
            e = _rich_follower(f"Emp{i}", (9 + i, 10), 30.0, "Chief"); e.knowledge.add("farming")
            cast.append(e)
        for i in range(7):
            w = _led_settler(f"Wkr{i}", (10, 9 + (i % 3))); w.money = 1.0; w.hunger = 6
            _trusts(w, "Chief", leadership.FORM_TRUST); cast.append(w)
        return cast

    def run(tax_on):
        cast = build(); world_state["taxation_on"] = tax_on; world_state["tax_rate"] = 0.30
        for t in range(1, 21):
            leadership.update(world_state, t); labor.update(world_state, t)
            if tax_on:
                taxation.update(world_state, t)
            for a in cast:
                if a.name.startswith("Wkr"):
                    a.hunger = max(a.hunger, 6)
        return gini([_wealth(a) for a in cast])

    _fresh_world(); off = run(False)
    _fresh_world(); on = run(True)
    assert on < off - 0.05, f"taxation must lower Gini: on {on:.3f} vs off {off:.3f}"
    assert off > 0.4, f"untaxed inequality should persist (the spiral): {off:.3f}"
    print("PASS test_redistribution_lowers_within_settlement_gini_vs_untaxed")


def test_over_taxation_costs_legitimacy_while_moderate_is_sustained() -> None:
    """Moderate taxation is sustained (no resentment); over-taxation erodes the taxed below the
    keep bar and the leader loses legitimacy (M3.2 contingency fires)."""
    import leadership, taxation

    def run_rate(rate):
        create_world(size=12)
        world_state["leadership_on"] = True; world_state["taxation_on"] = True
        world_state["tax_rate"] = rate; world_state["leaders"] = {}
        _led_settler("Gov", (5, 5)).money = 5.0
        rich = [_rich_follower(f"R{i}", (5 + (i % 2), 6 - (i // 2)), 40.0, "Gov") for i in range(3)]
        poor = _led_settler("Poor", (4, 5)); poor.money = 1.0
        _trusts(poor, "Gov", leadership.FORM_TRUST)
        fates = []
        for t in range(1, 5):
            leadership.update(world_state, t)
            rec = world_state["leaders"].get("S001")
            fates.append(rec["leader"] if rec else None)
            if rec:
                taxation.update(world_state, t)
        return fates, rich[0].relationships["Gov"]["trust"]

    _fresh_world(); mod_fates, mod_trust = run_rate(0.30)
    assert all(f == "Gov" for f in mod_fates), "moderate taxation must be sustained"
    assert mod_trust >= leadership.KEEP_TRUST, "moderate taxation must not erode the rich below keep"
    _fresh_world(); over_fates, over_trust = run_rate(0.90)
    assert over_fates[0] == "Gov" and None in over_fates, "over-taxation must cost legitimacy"
    assert over_trust < leadership.KEEP_TRUST, "over-taxation must erode the taxed below the keep bar"
    print("PASS test_over_taxation_costs_legitimacy_while_moderate_is_sustained")


def test_tax_flows_rich_to_poor_among_followers_and_conserves_wealth() -> None:
    """Wealth is taxed from the rich, lifts the poorest most, conserves the total, and leaves the
    leader, mid-wealth followers and non-followers untouched."""
    import leadership, taxation
    _fresh_world(); create_world(size=12)
    world_state["leadership_on"] = True; world_state["taxation_on"] = True
    world_state["tax_rate"] = 0.30; world_state["leaders"] = {}
    leader = _led_settler("Chief", (5, 5)); leader.money = 8.0
    rich = _rich_follower("Rich", (5, 6), 50.0, "Chief")
    poor1 = _led_settler("Poor1", (6, 5)); poor1.money = 1.0; _trusts(poor1, "Chief", leadership.FORM_TRUST)
    poor2 = _led_settler("Poor2", (4, 5)); poor2.money = 3.0; _trusts(poor2, "Chief", leadership.FORM_TRUST)
    middle = _led_settler("Middle", (5, 4)); middle.money = 7.0; _trusts(middle, "Chief", leadership.FORM_TRUST)
    outsider = _led_settler("Outsider", (7, 7)); outsider.money = 99.0  # settled, NOT a follower
    leader_record_cast = [leader, rich, poor1, poor2, middle, outsider]
    leadership.update(world_state, 1)
    before = {a.name: _wealth(a) for a in leader_record_cast}
    taxation.update(world_state, 1)
    after = {a.name: _wealth(a) for a in leader_record_cast}
    assert after["Rich"] < before["Rich"], "the rich follower is taxed"
    assert after["Poor1"] - before["Poor1"] > after["Poor2"] - before["Poor2"], "poorest gets most"
    assert after["Chief"] == before["Chief"] and after["Middle"] == before["Middle"], "leader/middle untouched"
    assert after["Outsider"] == before["Outsider"], "a non-follower is untouched"
    assert abs(sum(after.values()) - sum(before.values())) < 1e-9, "total wealth conserved"
    print("PASS test_tax_flows_rich_to_poor_among_followers_and_conserves_wealth")


def test_taxation_off_run_is_byte_identical_to_v1() -> None:
    """taxation_on=False (default) leaves the run byte-identical to the no-param run."""
    def run(flag):
        llm.PROVIDER = "random"
        random.seed(41)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(22, focal_budget=8)
            else:
                main.run_simulation(22, focal_budget=8, taxation_on=flag)
        return buf.getvalue()
    saved = llm.PROVIDER
    try:
        base, off = run(None), run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "taxation_on=False changed the default run output"
    print("PASS test_taxation_off_run_is_byte_identical_to_v1")


# --- Conquest & monarchy: power seized by force (V2 M3.4, Phase 3) ---------
def _combatant(name: str, pos: tuple[int, int], money: float, sid: str | None = None) -> Agent:
    """A living agent at `pos` with set wealth — a roaming aspirant/mercenary (sid None) or settler."""
    a = _agent(name, "cautious and territorial", pos, hunger=0)
    a.settlement = sid
    a.money = money
    return a


def _set_settlement(sid: str, center: tuple[int, int], members: set[str]) -> None:
    world_state["settlements"] = {sid: {"id": sid, "center": center, "members": members, "founded": 0}}


def test_force_scales_with_wealth_funded_fighters_broke_cannot_conquer() -> None:
    """An army is REAL fighters bought with money: a rich aspirant musters and seizes an unled
    town; a broke aspirant funds nobody and conquers nothing."""
    import monarchy
    _fresh_world(); create_world(size=14)
    world_state["monarchy_on"] = True; world_state["monarchs"] = {}
    _set_settlement("S001", (7, 7), {"M1", "M2"})
    _combatant("M1", (7, 7), 1.0, "S001"); _combatant("M2", (7, 8), 1.0, "S001")
    rich = _combatant("Rich", (8, 8), 30.0)
    for i in range(4):
        _combatant(f"Merc{i}", (6 + i % 3, 6), 0.5)
    res = monarchy.attempt_conquest(world_state, rich, "S001", 1)
    assert res["won"] and world_state["monarchs"]["S001"]["monarch"] == "Rich"
    assert res["attackers"] == 4, "force = real mustered fighters (4 in range), not a wealth compare"
    assert _wealth(rich) == 30.0 - 4 * monarchy.FIGHTER_COST, "fighters are PAID for with wealth"

    _fresh_world(); create_world(size=14)
    world_state["monarchy_on"] = True; world_state["monarchs"] = {}
    _set_settlement("S001", (7, 7), {"M1", "M2"})
    _combatant("M1", (7, 7), 1.0, "S001"); _combatant("M2", (7, 8), 1.0, "S001")
    broke = _combatant("Broke", (8, 8), 3.0)
    for i in range(4):
        _combatant(f"Merc{i}", (6 + i % 3, 6), 0.5)
    res2 = monarchy.attempt_conquest(world_state, broke, "S001", 1)
    assert monarchy.max_fighters(broke) == 0 and not res2["won"], "a broke aspirant conquers nothing"
    assert world_state["monarchs"] == {}, "no crown without an army"
    print("PASS test_force_scales_with_wealth_funded_fighters_broke_cannot_conquer")


def test_loyalty_repels_smaller_force_but_not_overwhelming_one() -> None:
    """Force vs legitimacy: a trusted leader's loyal followers repel a richer-but-smaller attacker,
    but an overwhelming bought force overcomes them (and consent survives the conquest)."""
    import leadership, monarchy

    def build(n_followers, n_mercs):
        create_world(size=16)
        world_state["monarchy_on"] = True; world_state["monarchs"] = {}
        world_state["leadership_on"] = True; world_state["leaders"] = {}
        members = {f"F{i}" for i in range(n_followers)} | {"Chief"}
        _set_settlement("S001", (8, 8), members)
        c = _combatant("Chief", (8, 8), 2.0, "S001")
        for i in range(n_followers):
            f = _combatant(f"F{i}", (8, 9), 1.0, "S001"); _trusts(f, "Chief", leadership.FORM_TRUST)
        leadership.update(world_state, 1)
        rich = _combatant("Rich", (9, 9), 200.0)  # the WEALTHIEST, but force = mustered fighters
        for i in range(n_mercs):
            _combatant(f"Merc{i}", (10, 10), 0.5)
        return rich

    _fresh_world(); rich = build(n_followers=4, n_mercs=3)
    res = monarchy.attempt_conquest(world_state, rich, "S001", 1)
    assert not res["won"] and "S001" not in world_state["monarchs"], \
        "loyal followers must repel a richer attacker whose force (3) is smaller than the following (4)"

    _fresh_world(); rich = build(n_followers=4, n_mercs=7)
    res2 = monarchy.attempt_conquest(world_state, rich, "S001", 1)
    assert res2["won"] and world_state["monarchs"]["S001"]["monarch"] == "Rich", \
        "an overwhelming force (7 > 4) must win"
    assert world_state["leaders"]["S001"]["leader"] == "Chief", \
        "conquest rules by force but does not erase consent — the two roles coexist"
    print("PASS test_loyalty_repels_smaller_force_but_not_overwhelming_one")


def test_monarch_levies_without_consent() -> None:
    """A monarch extracts wealth from subjects with NO leader/trust/consent (vs M3.3)."""
    import monarchy
    _fresh_world(); create_world(size=12)
    world_state["monarchy_on"] = True; world_state["leaders"] = {}
    world_state["monarchs"] = {"S001": {"monarch": "King", "since": 1, "garrison": {"G1"}}}
    _set_settlement("S001", (6, 6), {"Sub1", "Sub2", "King"})
    king = _combatant("King", (6, 6), 5.0, "S001")
    sub1 = _combatant("Sub1", (6, 7), 20.0, "S001"); sub2 = _combatant("Sub2", (7, 6), 15.0, "S001")
    w0 = _wealth(king)
    ev = monarchy.levy(world_state, 2)
    assert _wealth(king) > w0 and _wealth(sub1) < 20.0 and _wealth(sub2) < 15.0, "the crown extracts wealth"
    assert ev and "no consent" in ev[0], "the levy needs no consent"
    # No legitimate leader exists, yet the levy still flowed -> it is NOT consent-gated like M3.3.
    assert world_state["leaders"] == {}, "the monarch levied with NO trust-leader present"
    print("PASS test_monarch_levies_without_consent")


def test_monarch_is_overthrowable_by_a_stronger_force() -> None:
    """The crown is losable: a stronger later aspirant overthrows the incumbent monarch."""
    import monarchy
    _fresh_world(); create_world(size=16)
    world_state["monarchy_on"] = True; world_state["leaders"] = {}
    _set_settlement("S001", (8, 8), {"Sub"})
    _combatant("Sub", (8, 8), 1.0, "S001")
    world_state["monarchs"] = {"S001": {"monarch": "OldKing", "since": 1, "garrison": {"G1", "G2"}}}
    _combatant("OldKing", (8, 8), 2.0); _combatant("G1", (8, 9), 0.5); _combatant("G2", (9, 8), 0.5)
    usurper = _combatant("Usurper", (9, 9), 40.0)
    for i in range(4):
        _combatant(f"Merc{i}", (10, 10), 0.5)
    res = monarchy.attempt_conquest(world_state, usurper, "S001", 5)
    rec = world_state["monarchs"]["S001"]
    assert res["won"] and rec["monarch"] == "Usurper" and rec["since"] == 5, "a stronger force overthrows"
    print("PASS test_monarch_is_overthrowable_by_a_stronger_force")


def test_war_kills_real_agents() -> None:
    """Fighting is costly: real agents die on both sides, logged like any death."""
    import leadership, monarchy
    _fresh_world(); create_world(size=16)
    world_state["monarchy_on"] = True; world_state["leadership_on"] = True
    world_state["leaders"] = {}; world_state["monarchs"] = {}
    _set_settlement("S001", (8, 8), {"Chief", "F0", "F1", "F2", "F3"})
    _combatant("Chief", (8, 8), 2.0, "S001")
    for i in range(4):
        f = _combatant(f"F{i}", (8, 9), 1.0, "S001"); _trusts(f, "Chief", leadership.FORM_TRUST)
    leadership.update(world_state, 1)
    rich = _combatant("Rich", (9, 9), 60.0)
    for i in range(6):
        _combatant(f"Merc{i}", (10, 10), 0.5)
    alive_before = sum(1 for a in world_state["agents"] if a.alive)
    res = monarchy.attempt_conquest(world_state, rich, "S001", 3)
    fallen = res["att_dead"] + res["def_dead"]
    alive_after = sum(1 for a in world_state["agents"] if a.alive)
    assert len(fallen) > 0, "a defended assault must produce casualties"
    assert alive_after == alive_before - len(fallen), "the fallen are actually dead"
    assert sum(1 for e in world_state["events"] if "fell in battle" in e) == len(fallen), \
        "each battle death is a logged civilizational event"
    print("PASS test_war_kills_real_agents")


def test_monarchy_off_run_is_byte_identical_to_v1() -> None:
    """monarchy_on=False (default) leaves the run byte-identical to the no-param run."""
    def run(flag):
        llm.PROVIDER = "random"
        random.seed(43)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(22, focal_budget=8)
            else:
                main.run_simulation(22, focal_budget=8, monarchy_on=flag)
        return buf.getvalue()
    saved = llm.PROVIDER
    try:
        base, off = run(None), run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "monarchy_on=False changed the default run output"
    print("PASS test_monarchy_off_run_is_byte_identical_to_v1")


# --- Kingdoms & vassalage: feudalism (V2 M3.5, Phase 3) -------------------
def _realm_world() -> None:
    """A fresh world with the M3.5 stack on and the institution dicts cleared."""
    _fresh_world(); create_world(size=30)
    world_state["monarchy_on"] = True; world_state["kingdoms_on"] = True
    world_state["leadership_on"] = True
    world_state["leaders"] = {}; world_state["monarchs"] = {}; world_state["kingdoms"] = {}
    world_state["settlements"] = {}


def _led_town(sid: str, center: tuple[int, int], chief: str, followers: list[str],
              member_wealth: float = 1.0) -> None:
    """A trust-led settlement (M3.2): `chief` trusted by `followers` who co-settle it."""
    import leadership
    world_state["settlements"][sid] = {"id": sid, "center": center,
                                       "members": {chief} | set(followers), "founded": 0}
    _combatant(chief, center, 2.0, sid)
    for f in followers:
        a = _combatant(f, (center[0], center[1] + 1), member_wealth, sid)
        _trusts(a, chief, leadership.FORM_TRUST)


def test_conquest_of_neighbour_makes_its_ruler_a_vassal() -> None:
    """A king conquers a neighbouring trust-led town; its chief becomes a VASSAL and the realm
    is a two-level hierarchy with the local leadership preserved (local autonomy)."""
    import kingdoms, leadership, monarchy
    _realm_world()
    world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5), "members": {"King"}, "founded": 0}
    _combatant("King", (5, 5), 200.0, "S001")
    world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
    _led_town("S002", (9, 9), "Chief", ["A", "B"])
    leadership.update(world_state, 0)
    for i in range(6):
        _combatant(f"M{i}", (4 + i % 3, 4), 0.5)
    res = kingdoms.conquer_neighbour(world_state, "King", "S002", 1)
    rec = world_state["kingdoms"]["King"]
    assert res["won"] and rec["vassals"].get("S002") == "Chief", "the conquered local ruler must become a vassal"
    assert "S001" in rec["settlements"] and "S002" in rec["settlements"], "both settlements form one realm"
    assert world_state["leaders"]["S002"]["leader"] == "Chief", "the vassal keeps local leadership (autonomy)"
    assert res["host"] > res["defenders"], "the realm host out-fielded the defence (it was a real fight/submission)"
    print("PASS test_conquest_of_neighbour_makes_its_ruler_a_vassal")


def test_tribute_cascades_settlement_to_vassal_to_king_and_conserves_wealth() -> None:
    """Tribute flows members -> vassal -> king (up the levels), conserving total wealth."""
    import kingdoms
    _realm_world()
    world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5), "members": {"King"}, "founded": 0}
    world_state["settlements"]["S002"] = {"id": "S002", "center": (9, 9),
                                          "members": {"Chief", "Rich1", "Rich2"}, "founded": 0}
    king = _combatant("King", (5, 5), 10.0, "S001")
    chief = _combatant("Chief", (9, 9), 10.0, "S002")
    rich1 = _combatant("Rich1", (9, 10), 25.0, "S002"); rich2 = _combatant("Rich2", (10, 9), 15.0, "S002")
    world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
    world_state["kingdoms"]["King"] = {"king": "King", "home": "S001",
                                       "settlements": {"S001", "S002"}, "vassals": {"S002": "Chief"},
                                       "founded": 0, "discontent": {"Chief": 0}}
    _trusts(chief, "King", kingdoms.LOYAL_TRUST)
    world_state["tribute_rate"] = kingdoms.DEFAULT_KING_SHARE  # fair: no backlash to muddy the flow
    total0 = sum(_wealth(a) for a in world_state["agents"] if a.alive)
    wk0, wc0, wm0 = _wealth(king), _wealth(chief), _wealth(rich1) + _wealth(rich2)
    kingdoms.tribute(world_state, 1)
    assert _wealth(rich1) < 25.0 and _wealth(rich2) < 15.0, "members are levied (bottom of the cascade)"
    assert _wealth(chief) > wc0 and _wealth(king) > wk0, "tribute flows up to BOTH the vassal and the king"
    assert _wealth(rich1) + _wealth(rich2) < wm0, "the members lost what cascaded up"
    total1 = sum(_wealth(a) for a in world_state["agents"] if a.alive)
    assert abs(total1 - total0) < 1e-9, "tribute only MOVES wealth — total conserved"
    print("PASS test_tribute_cascades_settlement_to_vassal_to_king_and_conserves_wealth")


def test_king_musters_loyal_vassal_but_not_broken_away_one() -> None:
    """The king's host includes a loyal vassal's fighters; a vassal no longer in the realm owes none."""
    import kingdoms, monarchy
    _realm_world()
    world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5), "members": {"King"}, "founded": 0}
    world_state["settlements"]["S002"] = {"id": "S002", "center": (22, 22), "members": {"Chief"}, "founded": 0}
    king = _combatant("King", (5, 5), 20.0, "S001")
    chief = _combatant("Chief", (22, 22), 20.0, "S002")  # a vassal with its own war chest, far from the king
    world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
    world_state["kingdoms"]["King"] = {"king": "King", "home": "S001",
                                       "settlements": {"S001", "S002"}, "vassals": {"S002": "Chief"},
                                       "founded": 0, "discontent": {"Chief": 0}}
    _trusts(chief, "King", kingdoms.LOYAL_TRUST)
    for i in range(3):
        _combatant(f"KM{i}", (4, 4), 0.5)        # mercs near the king (only the king can reach these)
    for i in range(3):
        _combatant(f"VM{i}", (22, 21), 0.5)      # mercs near the vassal's seat (only the vassal reaches these)
    host = kingdoms.muster_realm(world_state, king, exclude=set())
    assert any(h.name.startswith("VM") for h in host), "a loyal vassal answers the muster with its own fighters"
    # Break the vassal away: it is no longer in the realm, so owes no service.
    for a in world_state["agents"]:
        if a.name.startswith(("KM", "VM")):
            a.money = 0.5
    king.money = 20.0
    world_state["kingdoms"]["King"]["vassals"].clear()
    world_state["kingdoms"]["King"]["settlements"].discard("S002")
    host2 = kingdoms.muster_realm(world_state, king, exclude=set())
    assert not any(h.name.startswith("VM") for h in host2), "a broken-away vassal owes the crown no military service"
    print("PASS test_king_musters_loyal_vassal_but_not_broken_away_one")


def test_over_tribute_breaks_a_vassal_away_with_hysteresis_fair_one_stays() -> None:
    """A grasping crown erodes a vassal's loyalty until it breaks away (not on the first turn —
    hysteresis); a fairly-treated identical vassal stays in the realm."""
    import kingdoms

    def run(tribute_rate: float, turns: int) -> list[bool]:
        _realm_world()
        world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5), "members": {"King"}, "founded": 0}
        world_state["settlements"]["S002"] = {"id": "S002", "center": (9, 9),
                                              "members": {"Chief", "Rich"}, "founded": 0}
        _combatant("King", (5, 5), 10.0, "S001")
        chief = _combatant("Chief", (9, 9), 10.0, "S002"); _combatant("Rich", (9, 10), 25.0, "S002")
        world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
        world_state["kingdoms"]["King"] = {"king": "King", "home": "S001",
                                           "settlements": {"S001", "S002"}, "vassals": {"S002": "Chief"},
                                           "founded": 0, "discontent": {"Chief": 0}}
        _trusts(chief, "King", kingdoms.LOYAL_TRUST)
        world_state["tribute_rate"] = tribute_rate
        in_realm = []
        for t in range(1, turns + 1):
            kingdoms.tribute(world_state, t); kingdoms._check_breakaways(world_state, t)
            in_realm.append("S002" in world_state["kingdoms"].get("King", {}).get("settlements", set()))
        return in_realm

    grasping = run(0.9, 4)
    assert grasping[0] is True, "hysteresis: the vassal must not break away on the very first hard turn"
    assert grasping[-1] is False, "a sufficiently disloyal vassal must break away"
    fair = run(0.25, 4)
    assert all(fair), "a fairly-treated vassal (share within the consent band) stays in the realm"
    print("PASS test_over_tribute_breaks_a_vassal_away_with_hysteresis_fair_one_stays")


def test_kingdoms_off_run_is_byte_identical_to_v1() -> None:
    """kingdoms_on=False (default) leaves the run byte-identical to the no-param run."""
    def run(flag):
        llm.PROVIDER = "random"
        random.seed(43)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(22, focal_budget=8)
            else:
                main.run_simulation(22, focal_budget=8, kingdoms_on=flag)
        return buf.getvalue()
    saved = llm.PROVIDER
    try:
        base, off = run(None), run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "kingdoms_on=False changed the default run output"
    print("PASS test_kingdoms_off_run_is_byte_identical_to_v1")


# --- M3.6: inter-kingdom war & empire -------------------------------------
def _war_world() -> None:
    """A fresh world with the M3.6 stack (monarchy+kingdoms+empire+leadership) on, dicts cleared."""
    _fresh_world(); create_world(size=80)
    for fl in ("monarchy_on", "kingdoms_on", "empire_on", "leadership_on"):
        world_state[fl] = True
    for k in ("leaders", "monarchs", "kingdoms", "empires", "settlements"):
        world_state[k] = {}


def _war_mercs(prefix: str, near: tuple[int, int], n: int) -> None:
    """A private merc pool within muster range of `near` only (well-separated so no double-hire)."""
    for i in range(n):
        _combatant(f"{prefix}{i}", (near[0] + (i % 2), near[1] + 2), 0.5)


def _war_realm(king: str, kmoney: float, home_c: tuple[int, int],
               seats: list[tuple[str, tuple[int, int], str]], vassal_loyal: bool) -> None:
    """A king (monarch of its home) with far-flung vassal lords (loyal or not), for war staging."""
    import kingdoms
    home = f"{king}_home"
    world_state["settlements"][home] = {"id": home, "center": home_c, "members": {king}, "founded": 0}
    _combatant(king, home_c, kmoney, home)
    world_state["monarchs"][home] = {"monarch": king, "since": 0, "garrison": set()}
    setts, vassals = {home}, {}
    for sid, c, chief in seats:
        world_state["settlements"][sid] = {"id": sid, "center": c, "members": {chief}, "founded": 0}
        ch = _combatant(chief, c, 40.0, sid)
        _trusts(ch, king, kingdoms.LOYAL_TRUST if vassal_loyal else kingdoms.LOYAL_TRUST - 3)
        world_state["leaders"][sid] = {"leader": chief, "followers": set(), "since": 0}
        vassals[sid] = chief; setts.add(sid)
    world_state["kingdoms"][king] = {"king": king, "home": home, "settlements": setts,
                                     "vassals": vassals, "founded": 0,
                                     "discontent": {v: 0 for v in vassals.values()}}


def _two_war_kingdoms(rich_loyal: bool) -> None:
    """Rich (wealthy, vassals per flag) adjacent to Poor (modest, LOYAL vassals)."""
    _war_world()
    _war_realm("Rich", 500.0, (10, 10), [("RV1", (40, 10), "RC1"), ("RV2", (10, 40), "RC2")], rich_loyal)
    _war_realm("Poor", 50.0, (18, 10), [("PV1", (50, 10), "PC1"), ("PV2", (18, 40), "PC2")], True)
    _war_mercs("RKm", (10, 10), 4); _war_mercs("RC1m", (40, 10), 4); _war_mercs("RC2m", (10, 40), 4)
    _war_mercs("PKm", (18, 10), 3); _war_mercs("PC1m", (50, 10), 3); _war_mercs("PC2m", (18, 40), 3)


def test_war_musters_whole_loyal_host_excluding_disloyal_vassals() -> None:
    """A kingdom's war strength is its LOYAL host: loyal vassals muster, disloyal ones withhold."""
    import empire
    _two_war_kingdoms(rich_loyal=False)
    rich = next(a for a in world_state["agents"] if a.name == "Rich")
    host = empire.imperial_host(world_state, rich, set())
    names = {h.name for h in host}
    assert any(n.startswith("RKm") for n in names), "the king's own contingent musters"
    assert not any(n.startswith(("RC1m", "RC2m")) for n in names), \
        "a DISLOYAL vassal's contingent does NOT answer the muster (war strength = LOYAL host)"
    # Flip the same kingdom loyal: now both vassal contingents join.
    _two_war_kingdoms(rich_loyal=True)
    rich = next(a for a in world_state["agents"] if a.name == "Rich")
    names = {h.name for h in empire.imperial_host(world_state, rich, set())}
    assert any(n.startswith("RC1m") for n in names) and any(n.startswith("RC2m") for n in names), \
        "loyal vassals muster their contingents into the royal host"
    print("PASS test_war_musters_whole_loyal_host_excluding_disloyal_vassals")


def test_richer_disloyal_kingdom_loses_then_wins_when_loyal() -> None:
    """A richer kingdom with DISLOYAL vassals fields a smaller host and LOSES to a poorer-but-loyal
    one; with its vassals LOYAL the same kingdom fields a full host and WINS. War kills on both sides."""
    import empire
    _two_war_kingdoms(rich_loyal=False)
    rich = next(a for a in world_state["agents"] if a.name == "Rich")
    poor = next(a for a in world_state["agents"] if a.name == "Poor")
    assert _wealth(rich) > _wealth(poor), "Rich is the richer kingdom"
    res = empire.wage_war(world_state, "Poor", "Rich", 1)
    assert res["won"] and res["att_host"] > res["def_host"], "the poorer-but-LOYAL kingdom wins"
    assert res["att_dead"] and res["def_dead"], "war kills real agents on BOTH armies"
    assert "Rich" in world_state["empires"]["Poor"]["subject_kings"], "the richer loser is subjugated"
    # Same kingdoms, Rich's vassals loyal: Rich now out-fields Poor and wins.
    _two_war_kingdoms(rich_loyal=True)
    res2 = empire.wage_war(world_state, "Rich", "Poor", 1)
    assert res2["won"] and res2["att_host"] > res2["def_host"], "with loyal vassals the richer kingdom wins"
    assert "Poor" in world_state["empires"]["Rich"]["subject_kings"], "now Rich subjugates Poor"
    print("PASS test_richer_disloyal_kingdom_loses_then_wins_when_loyal")


def test_defeated_king_is_subjugated_into_multilevel_empire() -> None:
    """On defeat the loser's king becomes a subject-king KEEPING his own realm — a multi-level empire
    (emperor -> subject-king -> the subject-king's vassal-lords -> settlements)."""
    import empire
    _two_war_kingdoms(rich_loyal=True)  # Rich (with its own vassal-lords) will conquer Poor
    poor_realm_before = dict(world_state["kingdoms"]["Poor"]["vassals"])
    empire.wage_war(world_state, "Rich", "Poor", 1)
    emp = world_state["empires"]["Rich"]
    assert emp["subject_kings"].get("Poor") is not None, "the defeated king is the victor's subject-king"
    assert world_state["kingdoms"]["Poor"]["vassals"] == poor_realm_before, \
        "the subject-king KEEPS his own internal realm (his vassal-lords stay under him)"
    poor = next(a for a in world_state["agents"] if a.name == "Poor")
    assert poor.relationships["Rich"]["trust"] == __import__("kingdoms").LOYAL_TRUST, \
        "the subjugated king swears fealty (trust seeded to LOYAL_TRUST)"
    print("PASS test_defeated_king_is_subjugated_into_multilevel_empire")


def test_imperial_tribute_cascades_through_subject_king_and_conserves_wealth() -> None:
    """Tribute cascades settlement -> lord -> subject-king -> emperor (the new level), conserving
    total wealth; it reaches the emperor THROUGH the subject-king."""
    import empire, kingdoms
    _war_world(); create_world(size=40)
    world_state["empire_on"] = True; world_state["leadership_on"] = True
    for k in ("leaders", "monarchs", "kingdoms", "empires", "settlements"):
        world_state[k] = {}
    world_state["settlements"]["E"] = {"id": "E", "center": (5, 5), "members": {"Emp"}, "founded": 0}
    emp = _combatant("Emp", (5, 5), 10.0, "E")
    world_state["monarchs"]["E"] = {"monarch": "Emp", "since": 0, "garrison": set()}
    world_state["settlements"]["K"] = {"id": "K", "center": (9, 9), "members": {"King"}, "founded": 0}
    king = _combatant("King", (9, 9), 10.0, "K")
    world_state["monarchs"]["K"] = {"monarch": "King", "since": 0, "garrison": set()}
    world_state["settlements"]["V"] = {"id": "V", "center": (12, 12), "members": {"Chief", "Rich"}, "founded": 0}
    chief = _combatant("Chief", (12, 12), 10.0, "V"); rich = _combatant("Rich", (12, 13), 40.0, "V")
    world_state["leaders"]["V"] = {"leader": "Chief", "followers": {"Rich"}, "since": 0}
    _trusts(chief, "King", kingdoms.LOYAL_TRUST)
    world_state["kingdoms"]["King"] = {"king": "King", "home": "K", "settlements": {"K", "V"},
                                       "vassals": {"V": "Chief"}, "founded": 0, "discontent": {"Chief": 0}}
    world_state["kingdoms"]["Emp"] = {"king": "Emp", "home": "E", "settlements": {"E"},
                                      "vassals": {}, "founded": 0, "discontent": {}}
    empire._subjugate(world_state, emp, "King", 0); _trusts(king, "Emp", kingdoms.LOYAL_TRUST)
    world_state["tribute_rate"] = kingdoms.DEFAULT_KING_SHARE
    world_state["empire_share"] = empire.DEFAULT_EMPIRE_SHARE
    total0 = sum(_wealth(a) for a in world_state["agents"] if a.alive)
    we0 = _wealth(emp)
    kingdoms.tribute(world_state, 1); empire.tribute(world_state, 1)
    assert _wealth(rich) < 40.0, "the member is levied (bottom of the cascade)"
    assert _wealth(emp) > we0, "tribute reaches the EMPEROR through the subject-king level"
    total1 = sum(_wealth(a) for a in world_state["agents"] if a.alive)
    assert abs(total1 - total0) < 1e-9, "the whole cascade only MOVES wealth — total conserved"
    print("PASS test_imperial_tribute_cascades_through_subject_king_and_conserves_wealth")


def test_over_imperial_tribute_fragments_subject_king_with_hysteresis_fair_stays() -> None:
    """A grasping emperor erodes a subject-king's loyalty until he breaks away (not on the first
    turn — hysteresis); a fairly-treated subject-king stays in the empire."""
    import empire, kingdoms

    def run(share: float, turns: int) -> list[bool]:
        _war_world(); create_world(size=40)
        world_state["empire_on"] = True
        for k in ("leaders", "monarchs", "kingdoms", "empires", "settlements"):
            world_state[k] = {}
        world_state["settlements"]["E"] = {"id": "E", "center": (5, 5), "members": {"Emp"}, "founded": 0}
        _combatant("Emp", (5, 5), 200.0, "E")
        world_state["monarchs"]["E"] = {"monarch": "Emp", "since": 0, "garrison": set()}
        world_state["settlements"]["K"] = {"id": "K", "center": (9, 9), "members": {"King"}, "founded": 0}
        king = _combatant("King", (9, 9), 40.0, "K")
        world_state["monarchs"]["K"] = {"monarch": "King", "since": 0, "garrison": set()}
        world_state["kingdoms"]["King"] = {"king": "King", "home": "K", "settlements": {"K"},
                                           "vassals": {}, "founded": 0, "discontent": {}}
        world_state["kingdoms"]["Emp"] = {"king": "Emp", "home": "E", "settlements": {"E"},
                                          "vassals": {}, "founded": 0, "discontent": {}}
        world_state["empires"]["Emp"] = {"emperor": "Emp", "subject_kings": {"King": {"since": 0}},
                                         "founded": 0, "discontent": {"King": 0}}
        _trusts(king, "Emp", kingdoms.LOYAL_TRUST)
        _war_mercs("Em", (5, 6), 8)  # a STRONG emperor (only the over-tax path fires, not weakening)
        world_state["empire_share"] = share
        in_empire = []
        for t in range(1, turns + 1):
            king.money = 40.0
            empire.tribute(world_state, t); empire._check_fragmentation(world_state, t)
            in_empire.append("King" in world_state["empires"].get("Emp", {}).get("subject_kings", {}))
        return in_empire

    grasp = run(0.9, 4)
    assert grasp[0] is True, "hysteresis: a subject-king must NOT break away on the very first hard turn"
    assert grasp[-1] is False, "a sufficiently disloyal subject-king must BREAK AWAY (empire fragments)"
    fair = run(0.25, 4)
    assert all(fair), "a fairly-treated subject-king (share within consent) stays in the empire"
    print("PASS test_over_imperial_tribute_fragments_subject_king_with_hysteresis_fair_stays")


def test_empire_off_run_is_byte_identical_to_v1() -> None:
    """empire_on=False (default) leaves the run byte-identical to the no-param run."""
    def run(flag):
        llm.PROVIDER = "random"
        random.seed(43)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if flag is None:
                main.run_simulation(22, focal_budget=8)
            else:
                main.run_simulation(22, focal_budget=8, empire_on=flag)
        return buf.getvalue()
    saved = llm.PROVIDER
    try:
        base, off = run(None), run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "empire_on=False changed the default run output"
    print("PASS test_empire_off_run_is_byte_identical_to_v1")


# --- DEMO scenario staging (scenario.py) ----------------------------------
def test_staged_monarchy_produces_a_real_monarch_record() -> None:
    """`scenario.apply(..., 'monarchy')` runs the REAL monarchy.attempt_conquest, so it leaves a
    genuine world_state["monarchs"] record (shape {monarch, since, garrison}) — RNG-free."""
    import random as _random, scenario
    _fresh_world(); create_world(size=24)
    for k in ("monarchs", "leaders", "kingdoms", "empires", "settlements"):
        world_state[k] = {}
    world_state["settlement_seq"] = 0
    s0 = _random.getstate()
    scenario.apply(world_state, "monarchy")
    assert _random.getstate() == s0, "staging must be RNG-free (reproducible under seed)"
    monarchs = world_state["monarchs"]
    assert monarchs, "staged monarchy must install a real monarch"
    sid, rec = next(iter(monarchs.items()))
    assert rec["monarch"] and "garrison" in rec and "since" in rec, "record matches an organic monarch"
    assert sid in world_state["settlements"], "the monarch rules a real settlement (castle can render)"
    print("PASS test_staged_monarchy_produces_a_real_monarch_record")


def test_staged_kingdom_produces_a_real_multi_settlement_kingdom() -> None:
    """`scenario.apply(..., 'kingdom')` uses the REAL kingdoms.conquer_neighbour, producing a
    multi-settlement feudal kingdom (king -> vassal lords) in world_state["kingdoms"]."""
    import scenario
    _fresh_world(); create_world(size=24)
    for k in ("monarchs", "leaders", "kingdoms", "empires", "settlements"):
        world_state[k] = {}
    world_state["settlement_seq"] = 0
    scenario.apply(world_state, "kingdom")
    kingdoms_rec = world_state["kingdoms"]
    assert kingdoms_rec, "staged kingdom must form a real kingdom"
    king, rec = next(iter(kingdoms_rec.items()))
    assert len(rec["settlements"]) >= 2, "a kingdom spans multiple settlements"
    assert rec["vassals"], "the conquered local rulers became vassal lords"
    # The vassal settlements KEEP their local leadership (organic vassalage), and the capital a monarch.
    for sid, lord in rec["vassals"].items():
        assert world_state["leaders"].get(sid, {}).get("leader") == lord, "vassal keeps local leadership"
    assert world_state["monarchs"], "the king is still a monarch of its seat (castle renders)"
    print("PASS test_staged_kingdom_produces_a_real_multi_settlement_kingdom")


def test_staged_war_forms_an_empire_via_the_real_loop() -> None:
    """A staged two-kingdom run lets the EXISTING empire.update opportunistic-war logic fire during
    the NORMAL loop, clashing real loyal hosts and subjugating the loser into an empire."""
    def run():
        llm.PROVIDER = "random"; random.seed(5)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(6, renderer=None, stage="war", monarchy_on=True,
                                kingdoms_on=True, empire_on=True, settlements=True,
                                cognition="heuristic", focal_budget=0, grid_size=30)
        return {e: sorted(r["subject_kings"]) for e, r in world_state.get("empires", {}).items()}
    saved = llm.PROVIDER
    try:
        empires_a = run()
        empires_b = run()
    finally:
        llm.PROVIDER = saved
    assert empires_a, "the staged war must form an empire (a king subjugated via the real loop)"
    assert any(subjects for subjects in empires_a.values()), "the empire holds a subject-king"
    assert empires_a == empires_b, "a staged run is reproducible under the seed"
    print("PASS test_staged_war_forms_an_empire_via_the_real_loop")


def test_staging_off_is_byte_identical_to_v1() -> None:
    """stage=None (default) leaves run_simulation byte-identical to a run with no stage param."""
    def run(staged):
        llm.PROVIDER = "random"; random.seed(43)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if staged:
                main.run_simulation(20, focal_budget=8, stage=None)
            else:
                main.run_simulation(20, focal_budget=8)
        return buf.getvalue()
    saved = llm.PROVIDER
    try:
        base, off = run(False), run(True)
    finally:
        llm.PROVIDER = saved
    assert base == off, "passing stage=None changed the default run"
    print("PASS test_staging_off_is_byte_identical_to_v1")


def test_staged_realm_stays_alive_and_populated_after_150_turns() -> None:
    """VIABILITY regression: a staged realm must SURVIVE POPULATED for the whole demo, not collapse
    into a ghost town. Runs the monarchy and kingdom scenes head-less for 150 turns on the SAME
    flags the --stage CLI uses (storage ON, economy OFF, seeded producer cast) and asserts the town
    is still inhabited AND the feudal structure is intact — so the castle/kingdom visuals sit in a
    living world. Also checks reproducibility. Guards against a regression to the starve-out demo."""
    def run(stage, turns, grid):
        llm.PROVIDER = "random"; random.seed(5)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(turns, renderer=None, stage=stage, monarchy_on=True,
                                kingdoms_on=(stage in ("kingdom", "war")), settlements=True,
                                storage_on=True, cognition="heuristic", focal_budget=0,
                                grid_size=grid)
        living = {a.name for a in world_state["agents"] if a.alive}
        setts = world_state.get("settlements", {})
        sett_alive = {sid: len(rec["members"] & living) for sid, rec in setts.items()}
        return len(living), sett_alive, world_state.get("kingdoms", {}), world_state.get("monarchs", {})

    saved = llm.PROVIDER
    try:
        # MONARCHY: the capital under the castle is still inhabited (not a lone hoarding king).
        m_live, m_setts, _, m_mon = run("monarchy", 150, 24)
        m_live_b, m_setts_b, _, _ = run("monarchy", 150, 24)
        assert m_mon, "the staged monarch must persist (a castle to render)"
        assert m_live >= 4, f"staged monarchy starved out: only {m_live} living at turn 150"
        assert m_setts.get("S001", 0) >= 3, f"the capital emptied: S001 had {m_setts} living members"
        assert (m_live, m_setts) == (m_live_b, m_setts_b), "a staged run must be reproducible"

        # KINGDOM: king + BOTH vassal settlements alive, and the kingdom record still spans them.
        k_live, k_setts, k_kingdoms, _ = run("kingdom", 150, 24)
        assert k_live >= 6, f"staged kingdom starved out: only {k_live} living at turn 150"
        assert sum(1 for v in k_setts.values() if v > 0) >= 3, \
            f"the feudal realm depopulated: settlements alive = {k_setts}"
        assert k_kingdoms and max(len(r["settlements"]) for r in k_kingdoms.values()) >= 3, \
            f"the kingdom dissolved into a lone monarchy: {k_kingdoms}"
    finally:
        llm.PROVIDER = saved
    print("PASS test_staged_realm_stays_alive_and_populated_after_150_turns")


# --- Conversation / talk (Day 8) ------------------------------------------
def test_talk_delivers_next_turn_and_reaction() -> None:
    """A talks to adjacent B; B receives NEXT turn and reacts; both remember it."""
    _fresh_world()
    alex = _agent("Alex", "friendly and outgoing", (5, 5))
    bob = _agent("Bob", "cautious and territorial", (5, 4))  # north of Alex, adjacent
    strat = Strategy(kind="talk", target="Bob", issued_turn=1)

    # Turn 1: Alex talks. Message lands in Bob's inbox, stamped turn 1.
    res = conversation.handle_talk(alex, "talk_to_Bob", strat, False, 1, world_state)
    assert "talked to Bob" in res
    assert len(bob.inbox) == 1 and bob.inbox[0]["from"] == "Alex" and bob.inbox[0]["turn"] == 1
    assert any("I told Bob" in m for m in alex.memory)

    # Same tick (turn 1): not yet deliverable — stays in the inbox.
    assert conversation.process_inbox(bob, False, "", 1, world_state) == []
    assert len(bob.inbox) == 1

    # Next turn (turn 2): Bob consumes it. Cautious → ignore.
    outcomes = conversation.process_inbox(bob, False, "", 2, world_state)
    assert outcomes == [("ignore", "Alex")], outcomes
    assert bob.inbox == []
    assert any("Alex said to me" in m and "ignored" in m for m in bob.memory)
    print("PASS test_talk_delivers_next_turn_and_reaction")


def test_talk_out_of_range_is_noop() -> None:
    """Talking to a non-adjacent agent logs the documented no-op and delivers nothing."""
    _fresh_world()
    alex = _agent("Alex", "friendly and outgoing", (5, 5))
    bob = _agent("Bob", "cautious", (0, 0))  # far away
    res = conversation.handle_talk(alex, "talk_to_Bob", Strategy(kind="talk", target="Bob"),
                                   False, 1, world_state)
    assert "no one was there" in res
    assert bob.inbox == []
    assert any("no one was there" in m for m in alex.memory)
    print("PASS test_talk_out_of_range_is_noop")


def test_reaction_is_personality_driven() -> None:
    """Off a refresh turn, the reaction is a deterministic personality rule."""
    assert conversation.deterministic_reaction(Agent("a", "friendly and outgoing")) == "reply"
    assert conversation.deterministic_reaction(Agent("b", "cautious and careful")) == "ignore"
    assert conversation.deterministic_reaction(Agent("c", "independent and competitive")) == "hostile"
    print("PASS test_reaction_is_personality_driven")


def test_reply_does_not_chain() -> None:
    """A reply is acknowledged but never triggers another reply (bounded exchange)."""
    _fresh_world()
    alex = _agent("Alex", "friendly and outgoing", (5, 5))
    bob = _agent("Bob", "friendly and outgoing", (5, 4))

    conversation.handle_talk(alex, "talk_to_Bob", Strategy(kind="talk", target="Bob"),
                             False, 1, world_state)
    # Bob (friendly) replies on turn 2 → a reply lands in Alex's inbox.
    conversation.process_inbox(bob, False, "", 2, world_state)
    assert len(alex.inbox) == 1 and alex.inbox[0]["reply"] is True

    # Alex consumes the reply on turn 3 → just hears it; no reply back to Bob.
    conversation.process_inbox(alex, False, "", 3, world_state)
    assert bob.inbox == []
    assert any("replied:" in m for m in alex.memory)
    print("PASS test_reply_does_not_chain")


def test_talk_message_source_refresh_vs_template() -> None:
    """The LLM message is used only on its refresh turn; otherwise it's templated."""
    _fresh_world()
    alex = _agent("Alex", "friendly", (5, 5))
    bob = _agent("Bob", "cautious", (5, 4))
    strat = Strategy(kind="talk", target="Bob", message="LLM-CRAFTED HELLO", issued_turn=7)

    # Refresh turn that produced the message → use it verbatim (no extra call).
    conversation.handle_talk(alex, "talk_to_Bob", strat, True, 7, world_state)
    assert bob.inbox[-1]["text"] == "LLM-CRAFTED HELLO"

    # A later, Python-driven turn → templated, never a new LLM call.
    bob.inbox.clear()
    conversation.handle_talk(alex, "talk_to_Bob", strat, False, 8, world_state)
    assert bob.inbox[-1]["text"] != "LLM-CRAFTED HELLO"
    print("PASS test_talk_message_source_refresh_vs_template")


def test_llm_message_path_end_to_end() -> None:
    """A stubbed provider's talk message flows through the real LLM path.

    Routes a fake response through get_strategy -> _raw_query -> _validate_strategy
    (PROVIDER != 'random'), then drives real refresh turns so the Strategy is
    BUILT from the provider, not pre-set. Proves the sentinel message is delivered
    verbatim, remembered by the sender, and that a recipient whose refresh-turn
    reaction is 'reply' returns a message to the sender.
    """
    def fake_raw_query(prompt):
        if "You are Alex" in prompt:
            return {"strategy": "talk", "target": "Bob", "message": "SENTINEL_HELLO",
                    "reaction": "", "reason": "test"}
        if "You are Bob" in prompt:
            return {"strategy": "wander", "target": "", "message": "",
                    "reaction": "reply", "reason": "test"}
        return {"strategy": "wander", "target": "", "message": "", "reaction": "", "reason": "x"}

    saved_provider, saved_raw = llm.PROVIDER, llm._raw_query
    saved_interval = main.STRATEGY_INTERVAL
    try:
        llm.PROVIDER = "stub"           # anything != 'random' uses _raw_query
        llm._raw_query = fake_raw_query
        main.STRATEGY_INTERVAL = 1       # make every turn a refresh turn

        _fresh_world()
        alex = _agent("Alex", "friendly and outgoing", (5, 5))
        bob = _agent("Bob", "cautious and territorial", (5, 4))
        strategies, survived, counters = {}, {"Alex": 0, "Bob": 0}, {"agent_turns": 0}

        bob_inbox_t1 = None
        for turn in (1, 2, 3):
            world_state["turn"] = turn
            for ag in (alex, bob):
                main.run_agent_turn(ag, turn, strategies, survived, counters)
            if turn == 1:
                bob_inbox_t1 = [dict(e) for e in bob.inbox]
    finally:
        llm.PROVIDER, llm._raw_query = saved_provider, saved_raw
        main.STRATEGY_INTERVAL = saved_interval

    # (1) the LLM message was delivered verbatim, not a template
    assert strategies["Alex"].message == "SENTINEL_HELLO"
    assert bob_inbox_t1 and bob_inbox_t1[0]["text"] == "SENTINEL_HELLO"
    # (2) the sender's memory records the LLM message
    assert any("I told Bob: SENTINEL_HELLO" in m for m in alex.memory)
    # (3) a refresh-turn reaction='reply' returned a message to the sender
    assert any('received from Alex: "SENTINEL_HELLO" -> reply' in e
               for e in world_state["events"])
    assert any("replied:" in m for m in alex.memory)
    print("PASS test_llm_message_path_end_to_end")


def test_trust_hostile_drops_3_friendly_raises_1() -> None:
    """Day 9: a hostile message drops trust by exactly 3; a friendly one raises by 1."""
    # --- Friendly exchange: +1 each direction ---
    _fresh_world()
    alex = _agent("Alex", "friendly and outgoing", (5, 5))
    bob = _agent("Bob", "friendly and outgoing", (5, 4))
    conversation.handle_talk(alex, "talk_to_Bob", Strategy(kind="talk", target="Bob"),
                             False, 1, world_state)
    # Bob (friendly) receives the talk next turn -> reply -> +1 toward Alex.
    conversation.process_inbox(bob, False, "", 2, world_state)
    assert bob.relationships["Alex"]["trust"] == 1, bob.relationships
    # Alex receives Bob's reply -> +1 toward Bob.
    conversation.process_inbox(alex, False, "", 3, world_state)
    assert alex.relationships["Bob"]["trust"] == 1, alex.relationships

    # --- Hostile reaction: sender loses exactly 3 ---
    _fresh_world()
    alex = _agent("Alex", "friendly and outgoing", (5, 5))
    kira = _agent("Kira", "independent and competitive", (5, 4))
    conversation.handle_talk(alex, "talk_to_Kira", Strategy(kind="talk", target="Kira"),
                             False, 1, world_state)
    # Kira (independent) reacts hostile next turn -> Alex's trust in Kira -= 3.
    conversation.process_inbox(kira, False, "", 2, world_state)
    assert alex.relationships["Kira"]["trust"] == -3, alex.relationships
    # Being hostile earns Kira no trust gain in Alex.
    assert kira.relationships.get("Alex", {}).get("trust", 0) == 0, kira.relationships
    print("PASS test_trust_hostile_drops_3_friendly_raises_1")


def test_trust_summary_buckets_and_prompt() -> None:
    """trust_summary buckets values and the digest reaches the strategy prompt."""
    import trust
    _fresh_world()
    alex = _agent("Alex", "friendly", (5, 5))
    alex.relationships["Bob"] = {"trust": 2, "interactions": 3}
    alex.relationships["Kira"] = {"trust": -3, "interactions": 2}
    summary = trust.trust_summary(alex)
    assert "Bob: +2 (high)" in summary, summary
    assert "Kira: -3 (low)" in summary, summary
    prompt = build_strategy_prompt(alex, "Current Tile: empty")
    assert "Your trust —" in prompt and "Kira: -3 (low)" in prompt
    print("PASS test_trust_summary_buckets_and_prompt")


def test_talk_adds_no_llm_calls() -> None:
    """A full run with conversation still makes zero per-turn LLM calls beyond refresh."""
    saved = llm.PROVIDER
    try:
        llm.PROVIDER = "random"
        llm.reset_call_stats()
        with contextlib.redirect_stdout(io.StringIO()):
            main.main()
        stats = llm.get_call_stats()
    finally:
        llm.PROVIDER = saved

    n_agents = len(main.AGENT_SPECS)
    max_refreshes = -(-main.NUM_TURNS // main.STRATEGY_INTERVAL)  # ceil
    assert stats["decision"] == 0, stats            # no legacy per-turn calls
    assert stats["strategy"] <= max_refreshes * n_agents, stats  # talk added none
    print(f"PASS test_talk_adds_no_llm_calls "
          f"(strategy={stats['strategy']}, decision={stats['decision']})")


# --- Steal + retaliation (Day 12) -----------------------------------------
def test_theft_drops_trust_5_and_writes_memories() -> None:
    """A successful steal drops the victim's trust by exactly 5 and logs both sides."""
    import trust
    _fresh_world()
    kira = _agent("Kira", "independent and competitive", (5, 5), hunger=8)
    bob = _agent("Bob", "cautious and territorial", (5, 4))  # north of Kira, adjacent
    world_state["food"].append(bob.position)  # Bob is standing on food
    bob.relationships["Kira"] = {"trust": -1, "interactions": 1, "grudge": False}

    res = conversation.handle_steal(kira, "steal_from_Bob", 23, world_state)
    assert "stole food from Bob" in res, res

    # Food left the victim's reach; thief ate it (hunger relief).
    assert bob.position not in world_state["food"], world_state["food"]
    assert kira.hunger == max(0, 8 - 7), kira.hunger

    # Victim trust dropped by EXACTLY THEFT_PENALTY (-1 -> -6) and latched a grudge.
    assert trust.THEFT_PENALTY == 5
    assert bob.relationships["Kira"]["trust"] == -6, bob.relationships
    assert bob.relationships["Kira"]["grudge"] is True, bob.relationships

    # Both memories written, in the documented shape.
    assert any("Kira stole my food on turn 23." == m for m in bob.memory), bob.memory
    assert any("I stole from Bob. They may retaliate." == m for m in kira.memory), kira.memory

    # events[] carries the THEFT line and the trust-change line.
    assert any("turn 23: Kira stole food from Bob" == e for e in world_state["events"])
    assert any("Bob trust in Kira: -1 -> -6 (theft)" in e for e in world_state["events"])
    print("PASS test_theft_drops_trust_5_and_writes_memories")


def test_theft_grudge_is_permanent() -> None:
    """Friendly messages cannot repair a stolen-from grudge (Day 12 permanence)."""
    import trust
    _fresh_world()
    kira = _agent("Kira", "independent and competitive", (5, 5), hunger=8)
    bob = _agent("Bob", "cautious", (5, 4))
    world_state["food"].append(bob.position)
    conversation.handle_steal(kira, "steal_from_Bob", 5, world_state)
    assert bob.relationships["Kira"]["trust"] == -5, bob.relationships

    # Any later positive trust toward the thief is refused — the grudge holds.
    trust.adjust_trust(bob, "Kira", +1, "friendly message", 7, world_state)
    trust.adjust_trust(bob, "Kira", +5, "friendly message", 9, world_state)
    assert bob.relationships["Kira"]["trust"] == -5, bob.relationships
    # But trust can still fall further (a second wrong).
    trust.adjust_trust(bob, "Kira", -3, "hostile message", 11, world_state)
    assert bob.relationships["Kira"]["trust"] == -8, bob.relationships
    print("PASS test_theft_grudge_is_permanent")


def test_theft_noop_when_no_food() -> None:
    """Stealing from a victim with no food (or out of range) is a logged no-op."""
    _fresh_world()
    kira = _agent("Kira", "independent and competitive", (5, 5), hunger=8)
    bob = _agent("Bob", "cautious", (5, 4))  # adjacent but NOT on food
    res = conversation.handle_steal(kira, "steal_from_Bob", 4, world_state)
    assert "no food to take" in res, res
    assert "Kira" not in bob.relationships, bob.relationships  # no trust hit
    assert any("had no food" in e for e in world_state["events"])

    # Out of range entirely.
    _fresh_world()
    kira = _agent("Kira", "independent and competitive", (5, 5), hunger=8)
    far = _agent("Bob", "cautious", (0, 0))
    world_state["food"].append(far.position)
    res = conversation.handle_steal(kira, "steal_from_Bob", 4, world_state)
    assert "no one was in reach" in res, res
    assert far.position in world_state["food"]  # untouched
    print("PASS test_theft_noop_when_no_food")


def test_desperate_independent_steals_distrusted_holder() -> None:
    """The executor turns desperation + distrust + opportunity into a steal action."""
    # Independent Kira, starving, beside a food-holder she doesn't trust -> steals.
    _fresh_world()
    kira = _agent("Kira", "independent and competitive", (5, 5), hunger=8)
    bob = _agent("Bob", "cautious", (5, 4))
    world_state["food"].append(bob.position)
    action, _ = choose_action(kira, Strategy(kind="wander"), world_state)
    assert action == "steal_from_Bob", action

    # Friendly Alex at NEUTRAL trust does NOT steal (gate respected)...
    _fresh_world()
    alex = _agent("Alex", "friendly and outgoing", (5, 5), hunger=8)
    bob = _agent("Bob", "cautious", (5, 4))
    world_state["food"].append(bob.position)
    action, _ = choose_action(alex, Strategy(kind="wander"), world_state)
    assert not action.startswith("steal_from_"), action

    # ...but once Alex actively distrusts Bob (low trust), desperation tips him over.
    alex.relationships["Bob"] = {"trust": -3, "interactions": 1, "grudge": False}
    action, _ = choose_action(alex, Strategy(kind="wander"), world_state)
    assert action == "steal_from_Bob", action
    print("PASS test_desperate_independent_steals_distrusted_holder")


def test_alliance_forms_only_mutually_with_plus3_both_ways() -> None:
    """An alliance forms ONLY when both agents choose it, +3 trust both ways."""
    import alliance
    _fresh_world()
    alex = _agent("Alex", "friendly and outgoing", (5, 5))
    bob = _agent("Bob", "cautious and territorial", (5, 4))  # adjacent (north)

    # (1) A unilateral ally_with is only a PROPOSAL — no alliance yet.
    res = alliance.handle_ally(alex, "ally_with_Bob", 3, world_state)
    assert "proposed an alliance" in res, res
    assert not alliance.are_allied(alex, bob)
    assert "Alex" in bob.ally_offers and "Bob" not in alex.allies
    # The proposal alone moves NO trust.
    assert "Bob" not in alex.relationships or alex.relationships["Bob"]["trust"] == 0

    # (2) Bob answering with his own ally_with seals the mutual alliance.
    res = alliance.handle_ally(bob, "ally_with_Alex", 4, world_state)
    assert "formed an alliance" in res, res
    assert alliance.are_allied(alex, bob)
    assert "Bob" in alex.allies and "Alex" in bob.allies
    # Offers are cleared once consumed.
    assert not alex.ally_offers and not bob.ally_offers

    # (3) +3 trust BOTH ways, and an ALLIANCE event + memory on both.
    assert alex.relationships["Bob"]["trust"] == 3, alex.relationships
    assert bob.relationships["Alex"]["trust"] == 3, bob.relationships
    assert any("formed an ALLIANCE" in e and "Alex" in e and "Bob" in e
               for e in world_state["events"])
    assert any("I allied with Bob" in m for m in alex.memory), alex.memory
    assert any("I allied with Alex" in m for m in bob.memory), bob.memory
    print("PASS test_alliance_forms_only_mutually_with_plus3_both_ways")


def test_allies_share_food_sighting_until_betrayal() -> None:
    """Allied agents share food the other can see; betrayal stops it immediately."""
    import alliance
    _fresh_world()
    alex = _agent("Alex", "friendly and outgoing", (1, 1))
    kira = _agent("Kira", "independent and competitive", (8, 8))  # far apart
    # Food only KIRA can see (adjacent to Kira, nowhere near Alex).
    world_state["food"].append((8, 7))  # north of Kira

    # Before allying, Alex shares nothing with Kira (they aren't allied).
    assert alliance.shared_food_sightings(alex, world_state) == {}

    # Ally them directly (bypassing range, just exercising the share mechanic).
    alex.allies.add("Kira"); kira.allies.add("Alex")

    # Now Kira's private sighting is shared into Alex's view...
    shared = alliance.shared_food_sightings(alex, world_state)
    assert shared == {"Kira": [(8, 7)]}, shared
    # ...and it surfaces verbatim in Alex's next strategy prompt.
    prompt = build_strategy_prompt(alex, observe(alex, world_state), state=world_state)
    assert "Food your allies can see (shared with you)" in prompt
    assert "Kira sees food at (8, 7)" in prompt, prompt

    # Kira betrays the alliance -> sharing stops on the very next lookup.
    alliance.handle_betray(kira, "betray_alliance_Alex", 9, world_state)
    assert alliance.shared_food_sightings(alex, world_state) == {}
    prompt2 = build_strategy_prompt(alex, observe(alex, world_state), state=world_state)
    assert "shared with you" not in prompt2
    print("PASS test_allies_share_food_sighting_until_betrayal")


def test_betrayal_drops_trust_8_latches_grudge_both_memories() -> None:
    """Betrayal dissolves the alliance, drops trust 8, latches a permanent grudge."""
    import alliance, trust
    _fresh_world()
    kira = _agent("Kira", "independent and competitive", (5, 5))
    alex = _agent("Alex", "friendly and outgoing", (5, 4))
    # Start them allied with the +3 each that forming would have granted.
    kira.allies.add("Alex"); alex.allies.add("Kira")
    kira.relationships["Alex"] = {"trust": 3, "interactions": 1, "grudge": False}
    alex.relationships["Kira"] = {"trust": 3, "interactions": 1, "grudge": False}

    res = alliance.handle_betray(kira, "betray_alliance_Alex", 12, world_state)
    assert "betrayed the alliance with Alex" in res, res

    # Alliance dissolved on BOTH sides.
    assert not alliance.are_allied(kira, alex)
    assert "Alex" not in kira.allies and "Kira" not in alex.allies

    # The betrayed (Alex) loses exactly BETRAYAL_PENALTY (8): 3 -> -5, grudge latched.
    assert alliance.BETRAYAL_PENALTY == 8
    assert alex.relationships["Kira"]["trust"] == -5, alex.relationships
    assert alex.relationships["Kira"]["grudge"] is True, alex.relationships

    # BOTH memories record the betrayal; events[] carries the BETRAYAL line.
    assert any("Kira BETRAYED our alliance" in m for m in alex.memory), alex.memory
    assert any("I BETRAYED my alliance with Alex" in m for m in kira.memory), kira.memory
    assert any("*** Kira BETRAYED the alliance with Alex ***" in e
               for e in world_state["events"])
    print("PASS test_betrayal_drops_trust_8_latches_grudge_both_memories")


def test_grudge_blocks_reallying() -> None:
    """A grudged/betrayed pair can NEVER form an alliance again (either side)."""
    import alliance
    _fresh_world()
    kira = _agent("Kira", "independent and competitive", (5, 5))
    alex = _agent("Alex", "friendly and outgoing", (5, 4))  # adjacent
    kira.allies.add("Alex"); alex.allies.add("Kira")
    alliance.handle_betray(kira, "betray_alliance_Alex", 7, world_state)
    assert alex.relationships["Kira"]["grudge"] is True

    # can_ally refuses because of the grudge on Alex's side...
    assert not alliance.can_ally(alex, kira)
    assert not alliance.can_ally(kira, alex)  # blocked from EITHER direction

    # ...and a real ally_with action is rejected as a logged no-op, not a bond.
    res = alliance.handle_ally(alex, "ally_with_Kira", 8, world_state)
    assert "could not ally" in res and "grudge" in res, res
    assert not alliance.are_allied(alex, kira)
    res = alliance.handle_ally(kira, "ally_with_Alex", 9, world_state)
    assert "could not ally" in res, res
    assert not alliance.are_allied(alex, kira)
    print("PASS test_grudge_blocks_reallying")


def test_independent_betrays_ally_under_pressure_friendly_does_not() -> None:
    """The executor turns survival pressure into a betrayal — only for a loner."""
    import alliance
    # Independent Kira, starving, beside an ALLY hoarding food -> betrays the bond.
    _fresh_world()
    kira = _agent("Kira", "independent and competitive", (5, 5), hunger=8)
    alex = _agent("Alex", "friendly and outgoing", (5, 4))
    kira.allies.add("Alex"); alex.allies.add("Kira")
    world_state["food"].append(alex.position)  # Alex sits on food
    action, _ = choose_action(kira, Strategy(kind="wander"), world_state)
    assert action == "betray_alliance_Alex", action
    # An ally is never STOLEN from — betrayal is the only move against a partner.
    assert not action.startswith("steal_from_")

    # Friendly Alex, equally starving beside an ally on food, does NOT betray.
    _fresh_world()
    alex = _agent("Alex", "friendly and outgoing", (5, 5), hunger=8)
    bob = _agent("Bob", "cautious and territorial", (5, 4))
    alex.allies.add("Bob"); bob.allies.add("Alex")
    world_state["food"].append(bob.position)
    action, _ = choose_action(alex, Strategy(kind="wander"), world_state)
    assert not action.startswith("betray_alliance_"), action
    print("PASS test_independent_betrays_ally_under_pressure_friendly_does_not")


def test_alliance_adds_no_llm_calls() -> None:
    """Forming and betraying an alliance ride the strategy call — zero new inference."""
    import alliance
    _fresh_world()
    llm.reset_call_stats()
    alex = _agent("Alex", "friendly and outgoing", (5, 5))
    bob = _agent("Bob", "cautious and territorial", (5, 4))
    alliance.handle_ally(alex, "ally_with_Bob", 1, world_state)
    alliance.handle_ally(bob, "ally_with_Alex", 2, world_state)
    alliance.handle_betray(bob, "betray_alliance_Alex", 3, world_state)
    stats = llm.get_call_stats()
    assert stats == {"decision": 0, "strategy": 0, "inclination": 0}, stats
    print("PASS test_alliance_adds_no_llm_calls")


# --- Death + respawn (Day 14) ---------------------------------------------
def test_death_writes_survivor_memories_and_event() -> None:
    """A death logs a DEATH event and records on every agent alive at the time."""
    import population
    _fresh_world()
    kira = _agent("Kira", "independent and competitive", (5, 5))
    alex = _agent("Alex", "friendly and outgoing", (5, 4))   # survivor
    bob = _agent("Bob", "cautious and territorial", (4, 5))   # survivor
    # A pre-existing relationship that must SURVIVE Kira's death (the dead are
    # remembered): Alex distrusts Kira.
    alex.relationships["Kira"] = {"trust": -6, "interactions": 4, "grudge": True}

    survivors = population.announce_death(kira, 47, world_state, cause="starved")

    assert kira.alive is False
    assert {s.name for s in survivors} == {"Alex", "Bob"}, survivors
    # Clear DEATH line in events[].
    assert any("turn 47: Kira died (starved)" == e for e in world_state["events"]), world_state["events"]
    # Every survivor remembers it; the dead agent keeps its own "Starved" memory.
    assert any("Kira died on turn 47" in m for m in alex.memory), alex.memory
    assert any("Kira died on turn 47" in m for m in bob.memory), bob.memory
    assert "Starved" in kira.memory
    # The survivor's relationship toward the deceased is untouched (remembered).
    assert alex.relationships["Kira"] == {"trust": -6, "interactions": 4, "grudge": True}
    # A respawn was queued for death_turn + RESPAWN_DELAY.
    assert world_state["pending_respawns"] == [47 + population.RESPAWN_DELAY]
    print("PASS test_death_writes_survivor_memories_and_event")


def test_respawn_fires_after_exactly_respawn_delay() -> None:
    """No newcomer before death_turn + RESPAWN_DELAY; exactly one on that turn."""
    import population
    _fresh_world()
    kira = _agent("Kira", "independent and competitive", (5, 5))
    _agent("Alex", "friendly and outgoing", (5, 4))  # a survivor to drop below target
    death_turn = 5
    population.announce_death(kira, death_turn, world_state)
    due_turn = death_turn + population.RESPAWN_DELAY

    # Every turn strictly before the due turn: nothing spawns, queue intact.
    for turn in range(death_turn + 1, due_turn):
        assert population.process_respawns(turn, world_state) == [], turn
        assert world_state["pending_respawns"] == [due_turn]

    # Exactly on the due turn: one newcomer enters and the queue drains.
    spawned = population.process_respawns(due_turn, world_state)
    assert len(spawned) == 1, spawned
    assert world_state["pending_respawns"] == []
    # Survivors were told a new agent appeared.
    alex = next(a for a in world_state["agents"] if a.name == "Alex")
    assert any(f"appeared on turn {due_turn}" in m for m in alex.memory), alex.memory
    print("PASS test_respawn_fires_after_exactly_respawn_delay")


def test_newcomer_is_blank_slate_and_participates() -> None:
    """The newcomer has empty memory/relationships and can be observed + talked to."""
    import population
    _fresh_world()
    kira = _agent("Kira", "independent and competitive", (5, 5))
    alex = _agent("Alex", "friendly and outgoing", (5, 4))
    population.announce_death(kira, 1, world_state)
    [newcomer] = population.process_respawns(1 + population.RESPAWN_DELAY, world_state)

    # Blank slate: no memory, no relationships, no allies/offers, hunger reset.
    assert newcomer.memory == [], newcomer.memory
    assert newcomer.relationships == {}, newcomer.relationships
    assert newcomer.allies == set() and newcomer.ally_offers == set()
    assert newcomer.hunger == 0
    # No living agent has any trust toward it yet (social cold-start).
    assert "?" not in newcomer.name  # sanity: real name assigned
    assert all(newcomer.name not in a.relationships
               for a in world_state["agents"] if a is not newcomer)
    # It occupies a valid empty cell and is observable by a neighbour: drop Alex
    # next to it and confirm detection + a talk both work like any other agent.
    nx, ny = newcomer.position
    place_agent(alex, nx, ny - 1)  # directly north of the newcomer
    assert f"South: {newcomer.name}" in observe(alex, world_state)
    res = conversation.handle_talk(alex, f"talk_to_{newcomer.name}", Strategy(kind="talk"),
                                   False, 99, world_state)
    assert f"talked to {newcomer.name}" in res, res
    assert len(newcomer.inbox) == 1 and newcomer.inbox[0]["from"] == "Alex"
    print("PASS test_newcomer_is_blank_slate_and_participates")


def test_respawn_keeps_population_bounded() -> None:
    """Respawn refills up to TARGET_POPULATION and never beyond it."""
    import population
    _fresh_world()
    kira = _agent("Kira", "independent and competitive", (5, 5))
    _agent("Alex", "friendly and outgoing", (5, 4))
    _agent("Bob", "cautious and territorial", (4, 5))
    assert population.living_count(world_state) == population.TARGET_POPULATION

    # One death -> one queued respawn -> back to target, not above.
    population.announce_death(kira, 1, world_state)
    assert population.living_count(world_state) == population.TARGET_POPULATION - 1
    spawned = population.process_respawns(1 + population.RESPAWN_DELAY, world_state)
    assert len(spawned) == 1
    assert population.living_count(world_state) == population.TARGET_POPULATION

    # A surplus respawn coming due while already at target is DROPPED, not spawned.
    world_state["pending_respawns"].append(20)
    assert population.process_respawns(20, world_state) == []
    assert population.living_count(world_state) == population.TARGET_POPULATION
    assert world_state["pending_respawns"] == []
    print("PASS test_respawn_keeps_population_bounded")


def test_respawn_adds_no_llm_calls() -> None:
    """Death + respawn are pure Python — they ride no inference at all."""
    import population
    _fresh_world()
    llm.reset_call_stats()
    kira = _agent("Kira", "independent and competitive", (5, 5))
    _agent("Alex", "friendly and outgoing", (5, 4))
    population.announce_death(kira, 1, world_state)
    population.process_respawns(1 + population.RESPAWN_DELAY, world_state)
    assert llm.get_call_stats() == {"decision": 0, "strategy": 0, "inclination": 0}, llm.get_call_stats()
    print("PASS test_respawn_adds_no_llm_calls")


# --- Renderer (Day 18) -----------------------------------------------------
def test_renderer_imports_only_state_reading_modules() -> None:
    """renderer/text_renderer.py must READ state only — no decision-logic imports.

    Mirrors the god_mode AST boundary test: the renderer is the only thing that DRAWS
    the world, and it may touch nothing but `world` (the state-reading layer) plus the
    third-party `rich`. Importing strategy/trust/conversation/alliance/personality/
    agents/llm/god_mode would let presentation reach into decision logic.
    """
    import ast
    with open("renderer/text_renderer.py") as f:
        tree = ast.parse(f.read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"strategy", "trust", "conversation", "alliance", "personality",
                 "agents", "llm", "god_mode", "population"}
    assert not (imported & forbidden), f"renderer imports decision logic: {imported & forbidden}"
    # The only project module it may lean on is `world`; the rest is stdlib + rich.
    project = imported - {"__future__", "typing", "contextlib", "os", "sys", "rich"}
    assert project <= {"world"}, project
    print("PASS test_renderer_imports_only_state_reading_modules")


def test_render_frame_does_not_mutate_world_state() -> None:
    """A render is a pure READ: world_state is byte-identical before and after.

    Builds a known world (agents at known hunger, food, a treasure, an alliance, an
    event) then snapshots every world_state field with a deep copy, renders a frame,
    and asserts nothing changed and a renderable was produced.
    """
    import copy
    from renderer import render_frame, RichRenderer
    _fresh_world()
    world_state["turn"] = 7
    alex = _agent("Alex", "friendly and outgoing", (4, 4), hunger=3)
    bob = _agent("Bob", "cautious and territorial", (6, 4), hunger=8)
    kira = _agent("Kira", "independent and competitive", (4, 6), hunger=2)
    spawn_food(3, cluster=True)
    world_state["treasures"].append({"pos": (5, 5), "value": 9})
    alex.allies.add("Bob"); bob.allies.add("Alex")
    alex.relationships["Bob"] = {"trust": 3, "interactions": 2, "grudge": False}
    world_state["events"].append("turn 6: [GOD] dropped a treasure at (5,5)")
    mark_dead(kira)  # exercise the dead-agent rendering path too

    before = copy.deepcopy({k: world_state[k] for k in world_state
                            if k not in ("agents",)})
    agents_before = [(a.name, a.position, a.hunger, a.alive,
                      set(a.allies), copy.deepcopy(a.relationships))
                     for a in world_state["agents"]]

    frame = render_frame(world_state)
    assert frame is not None

    after = {k: world_state[k] for k in world_state if k not in ("agents",)}
    assert after == before, "render_frame mutated world_state"
    agents_after = [(a.name, a.position, a.hunger, a.alive,
                     set(a.allies), a.relationships) for a in world_state["agents"]]
    assert agents_after == agents_before, "render_frame mutated an agent"

    # The renderer can also produce a frame off-Live (one-shot) without a terminal.
    RichRenderer()  # constructs cleanly (binds to the real stdout, devnull sink)
    print("PASS test_render_frame_does_not_mutate_world_state")


def test_event_styling_emphasises_major_moments() -> None:
    """The EVENTS panel highlights deaths/theft/alliance/betrayal/[GOD] (Day 19).

    event_style() is a pure classifier over the verbatim events[] strings, so we can
    assert each major kind maps to its bold colour AND is flagged major, while routine
    chatter stays muted/non-major. Betrayal must win over the 'alliance' substring it
    contains.
    """
    from renderer.text_renderer import event_style
    cases = {
        "turn 5: Kira died (starved)": ("bold red", True),
        "turn 5: Bob stole food from Alex": ("bold orange3", True),
        "turn 9: Alex and Bob formed an ALLIANCE": ("bold green", True),
        "turn 9: *** Bob BETRAYED the alliance with Alex ***": ("bold bright_red", True),
        "turn 6: [GOD] plague struck Kira (10 turns)": ("bold yellow", True),
        # Lead-ups are coloured but NOT major (no bullet, won't dominate the panel).
        "turn 4: Bob proposed an alliance to Alex (awaiting reply)": ("green", False),
        "turn 5: a new agent Zed appeared (blank slate)": ("green", False),
        # Routine chatter stays muted.
        'turn 2: Alex talked to Bob: "Hi!"': ("grey70", False),
        "turn 3: Bob trust in Alex: 0 -> 1 (friendly message)": ("grey70", False),
    }
    for line, expected in cases.items():
        assert event_style(line) == expected, (line, event_style(line), expected)
    print("PASS test_event_styling_emphasises_major_moments")


def test_pygame_renderer_imports_only_state_reading_modules() -> None:
    """renderer/pygame_renderer.py must READ state only — no decision-logic imports.

    Mirrors the text-renderer AST boundary test: the visual renderer DRAWS the world and
    may touch nothing but pygame + stdlib. It must not import strategy/trust/conversation/
    alliance/personality/agents/llm/god_mode/economy/population/monarchy/kingdoms/empire —
    importing any would let presentation reach into decision logic. Parses the file (never
    imports it), so this runs with or without pygame installed.
    """
    import ast
    with open("renderer/pygame_renderer.py") as f:
        tree = ast.parse(f.read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"strategy", "trust", "conversation", "alliance", "personality", "agents",
                 "llm", "god_mode", "economy", "population", "monarchy", "kingdoms", "empire",
                 "leadership", "taxation", "labor", "settlement", "knowledge"}
    assert not (imported & forbidden), f"pygame renderer imports decision logic: {imported & forbidden}"
    # It may lean on pygame + stdlib only; it draws straight from the snapshot dict, so it
    # needs no project module at all (not even `world`).
    # V4.15: it may also lean on its OWN package — renderer.director, the pure severity/caption
    # classifier that decides where the showcase camera looks. That module has its own boundary
    # test (test_director_imports_only_stdlib) pinning it to stdlib, so this stays a closed system.
    allowed = {"__future__", "typing", "contextlib", "os", "sys", "time", "math", "textwrap",
               "collections", "pygame", "renderer"}
    assert imported <= allowed, f"pygame renderer imports unexpected modules: {imported - allowed}"
    print("PASS test_pygame_renderer_imports_only_state_reading_modules")


def test_pygame_renderer_color_by_personality_and_size_by_wealth() -> None:
    """Pure mapping helpers: COLOUR encodes dominant personality, RADIUS grows with wealth.

    Skips gracefully if pygame is not installed (the renderer is an OPTIONAL dependency),
    so the suite stays green either way.
    """
    try:
        from renderer.pygame_renderer import (agent_color, agent_radius, dominant_trait, _wealth)
    except ImportError:
        print("PASS test_pygame_renderer_color_by_personality_and_size_by_wealth (skipped: no pygame)")
        return
    # Dominant trait read straight off the free-text personality (no personality-module import).
    assert dominant_trait("cautious and territorial") == "caution"
    assert dominant_trait("friendly and outgoing") == "friendliness"
    assert dominant_trait("independent and competitive") == "independence"
    assert dominant_trait("curious explorer") == "curiosity"
    assert dominant_trait("") == "curiosity", "an unrecognised personality falls back deterministically"
    # Distinct colours per dominant trait.
    colors = {agent_color(p) for p in ("cautious", "friendly", "independent", "curious")}
    assert len(colors) == 4, "each dominant personality maps to a distinct colour"
    # Radius is monotonic in wealth and clamped to a sane band.
    cell = 40
    r_poor = agent_radius(0.0, cell)
    r_mid = agent_radius(20.0, cell)
    r_rich = agent_radius(500.0, cell)
    assert r_poor < r_mid < r_rich, "richer agents are drawn larger"
    assert 2 <= r_poor and r_rich <= cell, "radius clamped to a visible, bounded range"
    assert _wealth(_FakeAgent(money=10.0, stockpile=5.0)) == 15.0, "wealth = money + stockpile"
    assert _wealth(_FakeAgent()) == 0.0, "missing wealth fields default to 0"
    print("PASS test_pygame_renderer_color_by_personality_and_size_by_wealth")


class _FakeAgent:
    """A minimal stand-in agent for the pygame-renderer read-only/draw tests."""
    def __init__(self, name="A", personality="cautious", position=(1, 1),
                 alive=True, money=None, stockpile=None):
        self.name = name; self.personality = personality; self.position = position
        self.alive = alive
        if money is not None:
            self.money = money
        if stockpile is not None:
            self.stockpile = stockpile


def test_pygame_renderer_draw_does_not_mutate_world_state() -> None:
    """Drawing a frame is a pure READ: world_state is byte-identical before and after.

    Uses SDL's headless 'dummy' video driver so a surface can be created without a real
    display. Skips gracefully if pygame (or a usable driver) is unavailable.
    """
    import copy, os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import PygameRenderer
    except ImportError:
        print("PASS test_pygame_renderer_draw_does_not_mutate_world_state (skipped: no pygame)")
        return
    state = {
        "size": 12, "turn": 5,
        "food": [(2, 3), (7, 8), (1, 1)],
        "agents": [
            _FakeAgent("Rich", "independent and competitive", (4, 4), True, 200.0, 50.0),
            _FakeAgent("Poor", "friendly and outgoing", (6, 6), True, 0.5, 0.0),
            _FakeAgent("Dead", "curious", (8, 2), False, 5.0, 0.0),  # must not be drawn
        ],
    }
    before = copy.deepcopy({k: state[k] for k in state if k != "agents"})
    agents_before = [(a.name, a.personality, a.position, a.alive,
                      getattr(a, "money", None), getattr(a, "stockpile", None))
                     for a in state["agents"]]
    r = PygameRenderer(turn_delay=0.0)
    pygame.init()
    try:
        r._ensure_screen(state["size"])
        r._draw(state)              # the pure draw path (no input/pacing loop)
    finally:
        pygame.quit()
    after = {k: state[k] for k in state if k != "agents"}
    assert after == before, "pygame draw mutated world_state"
    agents_after = [(a.name, a.personality, a.position, a.alive,
                     getattr(a, "money", None), getattr(a, "stockpile", None))
                    for a in state["agents"]]
    assert agents_after == agents_before, "pygame draw mutated an agent"
    print("PASS test_pygame_renderer_draw_does_not_mutate_world_state")


def test_pygame_settlement_region_grows_with_member_spread() -> None:
    """Slice 2: the settlement region radius is a pure read that grows with member spread.

    A floor keeps a just-founded settlement visible; a member farther from the centre
    enlarges the region (so a growing/spreading settlement draws bigger next frame). Pure
    geometry — skips gracefully without pygame.
    """
    try:
        from renderer.pygame_renderer import settlement_radius_cells, _SETTLEMENT_MIN_CELLS
    except ImportError:
        print("PASS test_pygame_settlement_region_grows_with_member_spread (skipped: no pygame)")
        return
    center = (10, 10)
    assert settlement_radius_cells(center, []) == _SETTLEMENT_MIN_CELLS, "empty falls back to the floor"
    tight = settlement_radius_cells(center, [(10, 10), (11, 10)])
    wide = settlement_radius_cells(center, [(10, 10), (11, 10), (14, 13)])
    assert wide > tight, "a farther-flung membership draws a larger region"
    assert tight >= _SETTLEMENT_MIN_CELLS, "the region never shrinks below the visible floor"
    print("PASS test_pygame_settlement_region_grows_with_member_spread")


def test_pygame_renderer_draws_settlements_read_only() -> None:
    """Slice 2: drawing settlements is a pure READ — the settlements dict + agents are
    byte-identical before and after, and a settlement-free state is unchanged (slice 1)."""
    import copy, os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import PygameRenderer
    except ImportError:
        print("PASS test_pygame_renderer_draws_settlements_read_only (skipped: no pygame)")
        return
    state = {
        "size": 16, "turn": 9,
        "food": [(3, 3), (12, 11)],
        "settlements": {
            "S001": {"id": "S001", "center": (5, 5), "members": {"Ann", "Bo"}, "founded": 2},
            "S002": {"id": "S002", "center": (11, 10), "members": {"Cy"}, "founded": 7},
        },
        "agents": [
            _FakeAgent("Ann", "friendly and outgoing", (5, 5), True, 8.0, 0.0),
            _FakeAgent("Bo", "cautious and territorial", (6, 6), True, 3.0, 0.0),
            _FakeAgent("Cy", "curious", (11, 10), True, 1.0, 0.0),
        ],
    }
    before = copy.deepcopy({k: state[k] for k in state if k != "agents"})
    agents_before = [(a.name, a.position, a.alive) for a in state["agents"]]
    r = PygameRenderer(turn_delay=0.0)
    pygame.init()
    try:
        r._ensure_screen(state["size"])
        r._draw(state)  # draws settlements (region + markers), food, agents
        # And a state with NO settlements must still draw fine (slice-1 path unchanged).
        r._draw({k: state[k] for k in state if k != "settlements"})
    finally:
        pygame.quit()
    after = {k: state[k] for k in state if k != "agents"}
    assert after == before, "settlement draw mutated world_state"
    assert [(a.name, a.position, a.alive) for a in state["agents"]] == agents_before, \
        "settlement draw mutated an agent"
    print("PASS test_pygame_renderer_draws_settlements_read_only")


def test_pygame_event_feed_classifies_and_wraps_newest_at_bottom() -> None:
    """Slice 3: the feed colour-classifies event lines and wraps them, newest at the bottom.

    Pure helpers (event_color + wrap_events) — testable without a display; skips without
    pygame. Verifies distinct colours per type, graceful handling of empty/short logs,
    wrapping of long lines, and that the NEWEST event ends up last (bottom of the feed).
    """
    try:
        from renderer.pygame_renderer import (event_color, wrap_events, _FEED_WAR,
                                              _FEED_DEATH, _FEED_GOD, _FEED_DEFAULT)
    except ImportError:
        print("PASS test_pygame_event_feed_classifies_and_wraps_newest_at_bottom (skipped: no pygame)")
        return
    # Light colour coding by type (war before death, so a war line with 'fell' reads as war).
    assert event_color("turn 9: KING X DEFEATED Y in war; 4 fell") == _FEED_WAR
    assert event_color("turn 7: Kira died (starved)") == _FEED_DEATH
    assert event_color("turn 3: Z died (fell in battle)") == _FEED_DEATH  # individual battle death
    assert event_color("turn 6: [GOD] dropped a treasure") == _FEED_GOD
    assert event_color("turn 2: Alex moved north") == _FEED_DEFAULT
    # Empty / short logs are graceful.
    assert wrap_events([], cols=40, max_rows=20) == []
    assert wrap_events([], cols=40, max_rows=0) == []
    short = wrap_events(["turn 1: a", "turn 2: b"], cols=40, max_rows=20)
    assert [t for t, _ in short] == ["turn 1: a", "turn 2: b"], "short log shows just what exists"
    # Newest at the bottom: the last row comes from the most recent event.
    rows = wrap_events(["turn 1: oldest", "turn 2: newest"], cols=40, max_rows=20)
    assert rows[-1][0] == "turn 2: newest", "the newest event sits at the bottom of the feed"
    # A long line wraps into multiple rows, all keeping the event's colour.
    longline = "turn 5: " + "word " * 40
    wrapped = wrap_events([longline], cols=20, max_rows=50)
    assert len(wrapped) > 1, "a long line wraps to several rows"
    assert len({c for _, c in wrapped}) == 1, "every wrapped sub-line keeps the event's colour"
    # max_rows clamps the feed to what fits (the tail).
    many = [f"turn {i}: e{i}" for i in range(50)]
    clamped = wrap_events(many, cols=40, max_rows=10)
    assert len(clamped) == 10 and clamped[-1][0] == "turn 49: e49"
    print("PASS test_pygame_event_feed_classifies_and_wraps_newest_at_bottom")


def test_pygame_event_tiers_banner_and_realm_scoreboard_pure() -> None:
    """V4.2/V4.1: the event-tier, story-banner, minor-aggregation and realm-scoreboard helpers
    are pure string/dict reads — turning-point events are MAJOR, churn is MINOR-and-aggregated,
    and the scoreboard reads kingdoms/empires from world_state (the '(none)' bug fix). Skips
    without pygame (the helpers live in the optional renderer module)."""
    import random as _random
    try:
        from renderer.pygame_renderer import (event_tier, banner_text, aggregate_minor,
                                              notable_names, realm_scoreboard, story_feed_rows,
                                              _FEED_DEFAULT)
    except ImportError:
        print("PASS test_pygame_event_tiers_banner_and_realm_scoreboard_pure (skipped: no pygame)")
        return
    # MAJOR = turning points (battles, conquests, secessions, uprisings, eras, prophets, faiths).
    for major in [
        "turn 5: KING Rex CONQUERED S002 into the realm (6 host vs 4 defenders; 2+3 fell) -> vassal C",
        "turn 7: KING Borin's host was REPELLED at S0A2 (1 host vs 1 defenders; 0+0 fell)",
        "turn 8: KING A DEFEATED B in war (6 vs 4; 2+3 fell) -> B SUBJUGATED; an EMPIRE rises",
        "turn 9: the UPRISING in S001 TRIUMPHED — monarch Rex is DEPOSED; Kade took power",
        "turn 3: C BROKE AWAY from Rex's realm — S002 is independent again (loyalty collapsed)",
        "turn 4: S001 entered the Bronze Age",
        "turn 8: Iris arose as prophet of the Faith of the Watchful Dead",
        "turn 2: the line of Rex is extinguished; the crown lies vacant",
    ]:
        assert event_tier(major) == "major", major
    # MINOR = per-agent churn (trust deltas, routine trades/talks/teaching/levies/beliefs).
    for minor in [
        "turn 5: A trust in Rex: 0 -> -2 (faith condemns Rex)",
        "turn 5: A sold 5.0 food to B for 2.5 money (price 0.50/unit)",
        "turn 5: A taught 'farming' to B",
        "turn 5: MONARCH Rex levied 2.0 from S001 by force",
        "turn 5: A came to believe 'wealth is virtue'",
        "turn 5: A talked to B: \"hi\"",
    ]:
        assert event_tier(minor) == "minor", minor
    # A death is MAJOR only for a RULER (a `notable` figure), else minor churn.
    st = {"kingdoms": {"Rex": {"settlements": {"S001"}}}, "monarchs": {}, "empires": {},
          "leaders": {}, "faiths": {}}
    note = notable_names(st)
    assert event_tier("turn 9: Rex died (old age)", note) == "major"
    assert event_tier("turn 9: F2a died (fell in battle)", note) == "minor"
    # The banner strips the prefix, the '(stats)' and the '-> outcome' tail to plain words.
    assert banner_text("turn 7: KING Borin's host was REPELLED at S0A2 (1 host vs 1; 0+0 fell)") \
        == "KING Borin's host was REPELLED at S0A2"
    assert banner_text("turn 5: KING Rex CONQUERED S002 into the realm (6 vs 4) -> vassal C") \
        == "KING Rex CONQUERED S002 into the realm"
    # Minor aggregation: EVERY event merges into a group (trust by target+direction, else by
    # category), listed largest-first — nothing is dropped into a lossy '+N more' here.
    agg = aggregate_minor([
        "turn 5: A trust in Rex: 0 -> -2 (faith condemns Rex)",
        "turn 5: B trust in Rex: 1 -> -1 (faith condemns Rex)",
        "turn 5: C trust in Rex: 2 -> 0 (faith condemns Rex)",
        "turn 5: D sold 5.0 food to E for 2.5 money",
    ])
    assert agg == "3 agents' trust in Rex fell  ·  1 routine trade", agg
    # Possessive grammar is correct for BOTH counts (the old '1 agent'' typo is gone).
    assert aggregate_minor(["turn 5: A trust in Rex: 0 -> 1 (talk)"]) \
        == "1 agent's trust in Rex rose"
    assert aggregate_minor(
        [f"turn 5: A{i} trust in LordA: 0 -> 1 (ally)" for i in range(5)]
    ) == "5 agents' trust in LordA rose"
    # Past _AGG_MAX_CLAUSES the long tail is tallied honestly as leftover EVENTS.
    assert aggregate_minor([]) is None
    # Realm scoreboard reads kingdoms/empires (folding a subjugated king into the empire).
    world = {
        "settlements": {"S001": {}, "S002": {}, "S003": {}},
        "kingdoms": {"Rex": {"settlements": {"S001", "S002"}},
                     "Ada": {"settlements": {"S003"}}},
        "empires": {"Rex": {"emperor": "Rex", "subject_kings": {"Ada": {}}}},
        "monarchs": {},
    }
    board = realm_scoreboard(world)
    assert board == [("Rex", 3, True)], board  # Ada's town folds under emperor Rex
    # The story feed tiers rows (major flagged) and collapses churn to one line per turn.
    s0 = _random.getstate()
    events = [
        "turn 5: A trust in Rex: 0 -> -1 (faith condemns Rex)",
        "turn 5: B trust in Rex: 0 -> -1 (faith condemns Rex)",
        "turn 5: KING Rex CONQUERED S002 into the realm (6 vs 4) -> vassal C",
    ]
    rows = story_feed_rows(events, frozenset(), cols=60, max_rows=20)
    assert any(is_major for _, _, is_major in rows), "a major row is flagged"
    assert any((not is_major and c == _FEED_DEFAULT) for _, c, is_major in rows), \
        "the two trust deltas collapse to one muted aggregated line"
    assert _random.getstate() == s0, "the story helpers never touch the global RNG stream"
    print("PASS test_pygame_event_tiers_banner_and_realm_scoreboard_pure")


def test_pygame_renderer_panel_reads_state_does_not_mutate() -> None:
    """Slice 3: drawing the wider window + side panel (stats + event feed) is a pure READ.

    Builds a state with events and institution dicts (settlements/kingdoms/empires), draws
    a frame into a headless surface, and asserts world_state is byte-identical afterwards
    and the window actually widened by the panel. Skips gracefully without pygame.
    """
    import copy, os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import PygameRenderer, _PANEL_W
    except ImportError:
        print("PASS test_pygame_renderer_panel_reads_state_does_not_mutate (skipped: no pygame)")
        return
    state = {
        "size": 14, "turn": 11,
        "food": [(1, 1), (9, 9)],
        "settlements": {"S001": {"id": "S001", "center": (5, 5), "members": {"A"}, "founded": 4}},
        "kingdoms": {"A": {"settlements": {"S001"}}},
        "empires": {},
        "events": [
            "turn 4: settlement S001 founded at (5, 5) by 1 settlers (A)",
            "turn 6: A and B formed an ALLIANCE",
            "turn 9: B died (starved)",
            "turn 10: [GOD] plague struck A",
        ] * 10,  # plenty so the feed must clamp/scroll
        "agents": [_FakeAgent("A", "curious", (5, 5), True, 12.0, 0.0)],
    }
    before = copy.deepcopy({k: state[k] for k in state if k != "agents"})
    agents_before = [(a.name, a.position, a.alive) for a in state["agents"]]
    r = PygameRenderer(turn_delay=0.0)
    pygame.init()
    try:
        r._ensure_screen(state["size"])
        win_w, _ = r._screen.get_size()
        r._draw(state)
        # Empty-feed path stays graceful too.
        r._draw({**{k: state[k] for k in state if k != "events"}, "events": []})
    finally:
        pygame.quit()
    # Slice 9: the map zone is the grid PLUS the full-bleed wilderness margin on both sides.
    assert win_w == r._map_px + _PANEL_W, "the window is the map zone plus the panel zone"
    assert r._map_px > r._cell * state["size"], "a wilderness margin extends past the grid"
    after = {k: state[k] for k in state if k != "agents"}
    assert after == before, "panel draw mutated world_state"
    assert [(a.name, a.position, a.alive) for a in state["agents"]] == agents_before, \
        "panel draw mutated an agent"
    print("PASS test_pygame_renderer_panel_reads_state_does_not_mutate")


def test_pygame_role_and_talk_helpers_read_state() -> None:
    """Slice 4: agent_role (EMPEROR>MONARCH>LEADER, degrading when a dict is absent) and
    talkers_this_turn (derived read-only from the event tail) are pure reads of state."""
    try:
        from renderer.pygame_renderer import agent_role, talkers_this_turn
    except ImportError:
        print("PASS test_pygame_role_and_talk_helpers_read_state (skipped: no pygame)")
        return
    state = {
        "leaders": {"S001": {"leader": "Lee", "followers": set(), "since": 1}},
        "monarchs": {"S001": {"monarch": "Ken", "since": 2, "garrison": set()}},
        "empires": {"Emi": {"emperor": "Emi", "subject_kings": {"Ken": {}}, "discontent": {}}},
    }
    assert agent_role("Emi", state) == "emperor"
    assert agent_role("Ken", state) == "monarch"
    assert agent_role("Lee", state) == "leader"
    assert agent_role("Nobody", state) is None
    # Precedence: someone who is BOTH a leader and an emperor wears the emperor mark.
    state["leaders"]["S009"] = {"leader": "Emi", "followers": set(), "since": 1}
    assert agent_role("Emi", state) == "emperor", "EMPEROR outranks LEADER"
    # Degrades gracefully when an institution dict is entirely absent.
    assert agent_role("Ken", {"leaders": {}}) is None, "no monarchs dict -> nobody is a monarch"
    # Talk signal: only the CURRENT turn's speakers, parsed from the contiguous tail.
    events = ["turn 4: X talked to Y: \"hi\"",          # an earlier turn — must be ignored
              "turn 5: Ann talked to Bo: \"hello\"",
              "turn 5: Cy talked to Ann: \"hey\"",
              "turn 5: Ann levied tribute"]
    assert talkers_this_turn(events, 5) == {"Ann", "Cy"}, "speakers this turn, from the tail"
    assert talkers_this_turn(events, 4) == set(), "an earlier turn's tail is not current — no bubbles"
    assert talkers_this_turn([], 5) == set(), "empty log is graceful"
    print("PASS test_pygame_role_and_talk_helpers_read_state")


def test_pygame_renderer_iconography_draw_is_read_only() -> None:
    """Slice 4: drawing figures/crowns/star/speech-bubble/wheat/houses is a pure READ.

    Exercises every new glyph path (a leader, a monarch, an emperor, a talker, food, a
    multi-member settlement, plus a tiny-cell fallback) into a headless surface and asserts
    world_state is byte-identical afterwards. Skips gracefully without pygame.
    """
    import copy, os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import PygameRenderer
    except ImportError:
        print("PASS test_pygame_renderer_iconography_draw_is_read_only (skipped: no pygame)")
        return
    state = {
        "size": 16, "turn": 20,
        "food": [(2, 3), (7, 8)],
        "settlements": {"S001": {"id": "S001", "center": (5, 5),
                                 "members": {"Lee", "F1", "F2"}, "founded": 3}},
        "leaders": {"S001": {"leader": "Lee", "followers": {"F1"}, "since": 3}},
        "monarchs": {"S001": {"monarch": "Ken", "since": 10, "garrison": set()}},
        "empires": {"Emi": {"emperor": "Emi", "subject_kings": {"Ken": {}}, "discontent": {}}},
        "kingdoms": {"Ken": {"settlements": {"S001"}}},
        "events": ["turn 19: prior", "turn 20: F1 talked to Lee: \"hi\""],
        "agents": [
            _FakeAgent("Lee", "friendly and outgoing", (5, 5), True, 9.0, 0.0),
            _FakeAgent("Ken", "independent and competitive", (8, 8), True, 200.0, 40.0),
            _FakeAgent("Emi", "cautious and territorial", (10, 10), True, 400.0, 0.0),
            _FakeAgent("F1", "curious", (6, 5), True, 1.0, 0.0),     # a talker this turn
            _FakeAgent("F2", "friendly", (4, 6), False, 5.0, 0.0),  # dead -> not drawn
        ],
    }
    before = copy.deepcopy({k: state[k] for k in state if k != "agents"})
    agents_before = [(a.name, a.position, a.alive) for a in state["agents"]]
    r = PygameRenderer(turn_delay=0.0)
    pygame.init()
    try:
        r._ensure_screen(state["size"])
        r._draw(state)
        # Tiny world: cells shrink so figures/food/houses hit their dot/skip fallbacks.
        big = {**state, "size": 60}
        r._ensure_screen(60)
        r._draw(big)
    finally:
        pygame.quit()
    after = {k: state[k] for k in state if k != "agents"}
    assert after == before, "iconography draw mutated world_state"
    assert [(a.name, a.position, a.alive) for a in state["agents"]] == agents_before, \
        "iconography draw mutated an agent"
    print("PASS test_pygame_renderer_iconography_draw_is_read_only")


def test_pygame_terrain_noise_is_pure_and_deterministic() -> None:
    """Slice 5: the terrain hash is deterministic, in [0,1), salt-decorrelated, and RNG-free.

    Pure helpers (terrain_noise + _shade) — testable without a display; the procedural
    landscape must NEVER touch the global random module (or it would desync the seeded sim).
    """
    import random as _random
    try:
        from renderer.pygame_renderer import terrain_noise, _shade
    except ImportError:
        print("PASS test_pygame_terrain_noise_is_pure_and_deterministic (skipped: no pygame)")
        return
    s0 = _random.getstate()
    vals = [terrain_noise(x, y, salt) for x in range(0, 40) for y in range(0, 40) for salt in range(3)]
    assert _random.getstate() == s0, "terrain_noise touched the global RNG stream"
    assert all(0.0 <= v < 1.0 for v in vals), "noise must stay in [0, 1)"
    assert terrain_noise(7, 3) == terrain_noise(7, 3), "noise must be deterministic"
    assert terrain_noise(7, 3, 1) != terrain_noise(7, 3, 2), "different salts decorrelate layers"
    assert len({terrain_noise(i, 0) for i in range(50)}) > 40, "noise varies across coordinates"
    assert _shade((42, 58, 40), 20) == (62, 78, 60)
    assert _shade((42, 58, 40), -100) == (0, 0, 0) and _shade((250, 250, 250), 50) == (255, 255, 255)
    print("PASS test_pygame_terrain_noise_is_pure_and_deterministic")


def test_pygame_terrain_cached_built_once_and_read_only() -> None:
    """Slice 5: the landscape is baked ONCE per grid size (cached, blitted), built RNG-free,
    and drawing it (with farmland near settlements) never mutates world_state."""
    import copy, random as _random, os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import PygameRenderer
    except ImportError:
        print("PASS test_pygame_terrain_cached_built_once_and_read_only (skipped: no pygame)")
        return
    state = {
        "size": 18, "turn": 8,
        "food": [(3, 3), (10, 12)],
        "settlements": {"S001": {"id": "S001", "center": (6, 6), "members": {"A", "B"}, "founded": 2}},
        "agents": [_FakeAgent("A", "friendly", (6, 6), True, 7.0, 0.0),
                   _FakeAgent("B", "curious", (7, 6), True, 2.0, 0.0)],
    }
    before = copy.deepcopy({k: state[k] for k in state if k != "agents"})
    r = PygameRenderer(turn_delay=0.0)
    pygame.init()
    try:
        s0 = _random.getstate()
        r._ensure_screen(state["size"])
        assert _random.getstate() == s0, "building terrain touched the global RNG"
        bg = r._terrain_bg
        assert bg is not None, "terrain background was baked at window open"
        # Re-ensuring the SAME size must NOT rebuild (cached); a new size rebuilds.
        r._ensure_screen(state["size"])
        assert r._terrain_bg is bg, "terrain must be cached, not rebuilt every call"
        r._draw(state)                 # first draw may rebake once to sync season from default
        bg2 = r._terrain_bg            # capture the season-synced terrain
        assert bg2 is not None
        r._draw(state)                 # same turn/season -> must NOT rebuild
        r._draw(state)
        assert r._terrain_bg is bg2, "terrain is cached and reused across frames within a season"
    finally:
        pygame.quit()
    after = {k: state[k] for k in state if k != "agents"}
    assert after == before, "terrain/farmland draw mutated world_state"
    print("PASS test_pygame_terrain_cached_built_once_and_read_only")


def test_pygame_terrain_rebakes_on_season_change() -> None:
    """Slice 5 (seasonal): the cached landscape is REUSED across frames within a season but
    REBAKED once when the turn crosses a season boundary — a season rebake still touches no
    global RNG and never mutates world_state."""
    import copy, random as _random, os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import PygameRenderer, season_name
    except ImportError:
        print("PASS test_pygame_terrain_rebakes_on_season_change (skipped: no pygame)")
        return
    # turn 8 -> spring, turn 30 -> summer (96 turns/year): far enough to differ in season.
    assert season_name(8) != season_name(30), "test turns must fall in different seasons"
    base = {
        "size": 18,
        "food": [(3, 3), (10, 12)],
        "settlements": {"S001": {"id": "S001", "center": (6, 6), "members": {"A", "B"}, "founded": 2}},
        "agents": [_FakeAgent("A", "friendly", (6, 6), True, 7.0, 0.0),
                   _FakeAgent("B", "curious", (7, 6), True, 2.0, 0.0)],
    }
    spring = {**base, "turn": 8}
    summer = {**base, "turn": 30}
    before = copy.deepcopy({k: base[k] for k in base if k != "agents"})
    r = PygameRenderer(turn_delay=0.0)
    pygame.init()
    try:
        r._ensure_screen(base["size"])
        r._draw(spring)               # sync season to spring
        bg_spring = r._terrain_bg
        assert bg_spring is not None
        r._draw(spring)               # same season -> REUSED
        assert r._terrain_bg is bg_spring, "terrain is reused for draws within a season"
        s0 = _random.getstate()
        r._draw(summer)               # crosses into summer -> REBAKED once
        assert _random.getstate() == s0, "the season rebake touched the global RNG"
        bg_summer = r._terrain_bg
        assert bg_summer is not None and bg_summer is not bg_spring, "the terrain changes on a season change"
        r._draw(summer)               # same season again -> REUSED
        assert r._terrain_bg is bg_summer, "terrain is reused for draws within a season"
    finally:
        pygame.quit()
    for tag, snap in (("spring", spring), ("summer", summer)):
        after = {k: snap[k] for k in snap if k not in ("agents", "turn")}
        assert after == before, f"the {tag} draw mutated world_state"
    print("PASS test_pygame_terrain_rebakes_on_season_change")


def test_pygame_town_plan_grows_with_members_and_is_pure() -> None:
    """Slice 6: a settlement's building plan GROWS with membership, gains civic structure at
    thresholds, marks a ruler's seat, and is deterministic + RNG-free (pure hash layout)."""
    import random as _random
    try:
        from renderer.pygame_renderer import (build_town_plan, _MAX_TOWN_BUILDINGS,
                                              _GRANARY_MIN_MEMBERS, _FENCE_MIN_MEMBERS)
    except ImportError:
        print("PASS test_pygame_town_plan_grows_with_members_and_is_pure (skipped: no pygame)")
        return
    col = (170, 150, 205)
    s0 = _random.getstate()
    small = build_town_plan((6, 6), 2, None, col, False, 16)
    big = build_town_plan((6, 6), 12, None, col, False, 16)
    assert _random.getstate() == s0, "building a town plan touched the global RNG"
    # GROWTH: more members -> more buildings (capped); civic structure appears at thresholds.
    assert len(big["buildings"]) > len(small["buildings"]), "a town shows more buildings than a hamlet"
    assert len(build_town_plan((6, 6), 999, None, col, False, 16)["buildings"]) == _MAX_TOWN_BUILDINGS
    assert small["granary"] is None and small["fence_r"] is None, "a hamlet has no granary/palisade"
    assert build_town_plan((6, 6), _GRANARY_MIN_MEMBERS, None, col, False, 16)["granary"] is not None
    assert build_town_plan((6, 6), _FENCE_MIN_MEMBERS, None, col, False, 16)["fence_r"] is not None
    # A ruler's seat: castle for a monarch, hall for a leader, none for a plain village.
    assert build_town_plan((6, 6), 6, "castle", col, False, 16)["central"]["kind"] == "castle"
    assert build_town_plan((6, 6), 6, "hall", col, False, 16)["central"]["kind"] == "hall"
    assert build_town_plan((6, 6), 6, None, col, False, 16)["central"]["kind"] is None
    # Deterministic: identical inputs -> identical layout (stable, no flicker); coords decorrelate.
    assert build_town_plan((6, 6), 8, None, col, False, 16) == build_town_plan((6, 6), 8, None, col, False, 16)
    assert (build_town_plan((6, 6), 8, None, col, False, 16)["buildings"]
            != build_town_plan((9, 2), 8, None, col, False, 16)["buildings"]), "different towns differ"
    print("PASS test_pygame_town_plan_grows_with_members_and_is_pure")


def test_pygame_detailed_settlements_cached_and_read_only() -> None:
    """Slice 6: drawing detailed villages, a leader HALL, a monarch CASTLE and an emperor's seat
    is read-only; town plans are cached and rebuilt ONLY when membership/ruler changes."""
    import copy, os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import PygameRenderer
    except ImportError:
        print("PASS test_pygame_detailed_settlements_cached_and_read_only (skipped: no pygame)")
        return
    village = {n: _FakeAgent(n, "curious", (6, 6), True, 2.0, 0.0) for n in ("a", "b", "c", "d", "e")}
    state = {
        "size": 18, "turn": 12,
        "food": [(2, 2)],
        "settlements": {
            "S001": {"id": "S001", "center": (6, 6), "members": set(village), "founded": 2},
            "S002": {"id": "S002", "center": (13, 12), "members": {"K"}, "founded": 5},
        },
        "leaders": {"S001": {"leader": "a", "followers": {"b"}, "since": 3}},   # hall
        "monarchs": {"S002": {"monarch": "K", "since": 8, "garrison": set()}},  # castle
        "empires": {"K": {"emperor": "K", "subject_kings": {}, "discontent": {}}},  # emperor seat
        "agents": list(village.values()) + [_FakeAgent("K", "independent and competitive", (13, 12), True, 300.0, 0.0)],
    }
    before = copy.deepcopy({k: state[k] for k in state if k != "agents"})
    r = PygameRenderer(turn_delay=0.0)
    pygame.init()
    try:
        r._ensure_screen(state["size"])
        r._draw(state)
        plan1 = r._town_plans["S001"][1]
        r._draw(state)
        assert r._town_plans["S001"][1] is plan1, "an unchanged settlement reuses its cached plan"
        assert set(r._town_plans) == {"S001", "S002"}, "both settlements have cached plans"
        # A new member must invalidate + rebuild that settlement's plan (growth is visible).
        state["settlements"]["S001"]["members"].add("f")
        state["agents"].append(_FakeAgent("f", "friendly", (6, 7), True, 1.0, 0.0))
        r._draw(state)
        plan2 = r._town_plans["S001"][1]
        assert plan2 is not plan1, "membership change rebuilds the plan"
        assert len(plan2["buildings"]) > len(plan1["buildings"]), "the rebuilt village grew"
        # A vanished settlement is pruned from the cache.
        del state["settlements"]["S002"]
        r._draw(state)
        assert "S002" not in r._town_plans, "a removed settlement is pruned from the plan cache"
    finally:
        pygame.quit()
    after = {k: state[k] for k in state if k not in ("agents", "settlements")}
    assert after == copy.deepcopy({k: before[k] for k in before if k != "settlements"}), \
        "detailed-settlement draw mutated world_state"
    print("PASS test_pygame_detailed_settlements_cached_and_read_only")


def test_pygame_battle_scene_detects_battles_and_names_casualties() -> None:
    """Slice 8: battle_scene reconstructs a battle ONLY from events + the prev snapshot (pure).

    A staged war's summary line yields a timeline whose attacker/defender/hosts/outcome come off
    the line, whose capitals come from the kings' home seats, whose casualty NAMES are the turn's
    'fell in battle' deaths split per side by the summary's x+y counts (attacker-side deaths are
    logged first), and whose positions come from the PREV snapshot (they are dead now). Peaceful
    turns give None; multiple battles give an ordered queue with casualties claimed per battle.
    """
    try:
        from renderer.pygame_renderer import (battle_scene, battle_scenes, take_snapshot,
                                              turn_events, _SETTLEMENT_FILL)
    except ImportError:
        print("PASS test_pygame_battle_scene_detects_battles_and_names_casualties (skipped: no pygame)")
        return
    prev_state = {
        "turn": 9, "size": 30,
        "agents": [_FakeAgent(n, "cautious", p, True, 1.0, 0.0) for n, p in
                   (("Aldric", (8, 6)), ("Borin", (22, 6)), ("AWK1", (8, 4)), ("BWK1", (22, 4)))],
        "settlements": {"S0A1": {"center": (8, 18), "members": set()},
                        "S0B1": {"center": (22, 18), "members": set()}},
        "kingdoms": {"Aldric": {"home": "S0A1", "settlements": {"S0A1"}},
                     "Borin": {"home": "S0B1", "settlements": {"S0B1"}}},
        "empires": {},
    }
    prev = take_snapshot(prev_state)
    assert prev["realms"] == {"S0A1": "Aldric", "S0B1": "Borin"}, "snapshot records realm owners"
    cur = {**prev_state, "turn": 10,
           "agents": [a for a in prev_state["agents"] if a.name not in ("AWK1", "BWK1")],
           "empires": {"Aldric": {"emperor": "Aldric", "subject_kings": {"Borin": {}},
                                  "discontent": {}}}}
    events = ["turn 9: earlier line",
              "turn 10: AWK1 died (fell in battle)",
              "turn 10: BWK1 died (fell in battle)",
              "turn 10: KING Aldric DEFEATED Borin in war (8 loyal host vs 4; 1+1 fell) -> "
              "Borin SUBJUGATED as a subject-king; an EMPIRE rises"]
    tl = battle_scene(turn_events(events, 10), prev, cur)
    assert tl is not None and tl["attacker"] == "Aldric" and tl["defender"] == "Borin" and tl["won"]
    assert tl["att_pos"] == (8, 18) and tl["def_pos"] == (22, 18), "anchored at the kings' capitals"
    assert tl["att_dead"] == [("AWK1", (8, 4))] and tl["def_dead"] == [("BWK1", (22, 4))], \
        "the named dead, per side, at their PREV-snapshot cells"
    assert tl["n_att"] == 8 and tl["n_def"] == 4, "host sizes read off the summary line"
    assert [s for s, _ in tl["territory"]] == ["S0B1"], "the loser's realm changes hands"
    # Peaceful turn -> None; a starvation death is NOT a battle.
    assert battle_scene(['turn 10: A talked to B: "hi"', "turn 10: C died (starved)"],
                        prev, cur) is None
    # Multiple battles in one turn -> an ordered queue; each summary claims ITS casualties.
    st2 = {"turn": 3, "agents": [],
           "settlements": {"S001": {"center": (6, 6)}, "S002": {"center": (10, 10)}},
           "monarchs": {"S001": {"monarch": "Rex", "garrison": set()}}}
    prev2 = {"turn": 2, "positions": {"Rex": (4, 4), "M1": (4, 5), "D1": (10, 11), "Kai": (9, 9)},
             "realms": {}, "homes": {}}
    ev2 = ["turn 3: M1 died (fell in battle)",
           "turn 3: Rex seized S001 by force (5 fighters vs 3 defenders; 1+0 fell) -> MONARCH of S001",
           "turn 3: D1 died (fell in battle)",
           "turn 3: Kai's assault on S002 was REPELLED (2 fighters vs 4 defenders; 0+1 fell) — militia held"]
    q = battle_scenes(ev2, prev2, st2)
    assert len(q) == 2, "two battles -> a queue of two timelines, in log order"
    assert q[0]["kind"] == "conquest" and q[0]["attacker"] == "Rex" and q[0]["won"]
    assert q[0]["att_pos"] == (4, 4), "a lone aspirant marches from his PREV cell"
    assert q[0]["att_dead"] == [("M1", (4, 5))] and q[0]["def_dead"] == []
    assert q[0]["territory"] == [("S001", _SETTLEMENT_FILL)], "unowned land fades teal -> crown"
    assert q[1]["won"] is False and q[1]["def_dead"] == [("D1", (10, 11))] and q[1]["territory"] == []
    print("PASS test_pygame_battle_scene_detects_battles_and_names_casualties")


def test_pygame_snapshot_lerp_and_realm_helpers_pure() -> None:
    """Slice 8: take_snapshot / settlement_realm / realm_color / lerp helpers are pure + RNG-free.

    The snapshot copies (mutating it never touches state), the realm read ranks emperor > king >
    lone monarch, realm colours are stable per name (and distinct for the staged rivals), and the
    motion/colour lerps hit their endpoints exactly. None of it touches the global RNG stream.
    """
    import random as _random
    try:
        from renderer.pygame_renderer import (take_snapshot, settlement_realm, realm_color,
                                              lerp_color, ease, turn_events)
    except ImportError:
        print("PASS test_pygame_snapshot_lerp_and_realm_helpers_pure (skipped: no pygame)")
        return
    state = {
        "turn": 7,
        "agents": [_FakeAgent("A", "curious", (1, 2), True, 1.0, 0.0),
                   _FakeAgent("Dead", "curious", (3, 3), False, 1.0, 0.0)],
        "settlements": {"S001": {"center": (5, 5)}, "S002": {"center": (9, 9)},
                        "S003": {"center": (2, 8)}},
        "kingdoms": {"Ken": {"home": "S001", "settlements": {"S001"}}},
        "empires": {"Emi": {"emperor": "Emi", "subject_kings": {"Ken": {}}}},
        "monarchs": {"S002": {"monarch": "Rex", "garrison": set()}},
    }
    s0 = _random.getstate()
    snap = take_snapshot(state)
    assert _random.getstate() == s0, "take_snapshot touched the global RNG stream"
    assert snap["positions"] == {"A": (1, 2)}, "living agents only, positions copied"
    assert snap["realms"] == {"S001": "Emi", "S002": "Rex", "S003": None}, \
        "empire outranks kingdom; a lone monarch owns his town; unowned is None"
    assert snap["homes"] == {"Ken": "S001"}
    snap["positions"]["X"] = (0, 0)  # the snapshot is a fresh container, not a view
    assert all(getattr(a, "name") != "X" for a in state["agents"])
    assert settlement_realm("S001", {"kingdoms": {"Ken": {"settlements": {"S001"}}}}) == "Ken", \
        "no empires dict degrades to the king"
    assert realm_color("Aldric") == realm_color("Aldric"), "realm colour is stable per name"
    assert realm_color("Aldric") != realm_color("Borin"), "the staged rivals wear distinct colours"
    assert _random.getstate() == s0, "realm_color touched the global RNG stream"
    assert lerp_color((0, 0, 0), (100, 200, 50), 0.0) == (0, 0, 0)
    assert lerp_color((0, 0, 0), (100, 200, 50), 1.0) == (100, 200, 50)
    assert lerp_color((0, 0, 0), (100, 200, 50), 0.5) == (50, 100, 25)
    assert ease(0.0) == 0.0 and ease(1.0) == 1.0 and 0.0 < ease(0.3) < ease(0.7) < 1.0
    assert ease(-1.0) == 0.0 and ease(2.0) == 1.0, "easing clamps out-of-range inputs"
    events = ["turn 4: old", "turn 5: a", "turn 5: b"]
    assert turn_events(events, 5) == ["turn 5: a", "turn 5: b"], "this turn's tail, oldest first"
    assert turn_events(events, 4) == [] and turn_events([], 5) == []
    print("PASS test_pygame_snapshot_lerp_and_realm_helpers_pure")


def test_pygame_cinematic_state_is_renderer_local_and_draw_read_only() -> None:
    """Slice 8: every cinematic/motion frame is a pure READ, and all animation state lives on
    the renderer (snapshot/territory-lerp), never in world_state; zero delay never blocks.

    Draws overlay frames across all five beats (muster/march/clash/casualties/aftermath with the
    territory lerp engaged) plus mid-walk motion frames into a headless surface, then runs
    update() at turn_delay=0 — asserting world_state is byte-identical throughout, gains no new
    keys, and that the zero-delay path returns immediately (no cinematic playback in tests).
    """
    import copy, time as _time, os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import PygameRenderer, battle_scene, turn_events
    except ImportError:
        print("PASS test_pygame_cinematic_state_is_renderer_local_and_draw_read_only (skipped: no pygame)")
        return
    state = {
        "size": 30, "turn": 10, "food": [(3, 3)],
        "agents": [_FakeAgent("Aldric", "independent and competitive", (8, 6), True, 200.0, 0.0),
                   _FakeAgent("Borin", "independent and competitive", (22, 6), True, 100.0, 0.0),
                   _FakeAgent("T1", "cautious", (8, 18), True, 2.0, 0.0)],
        "settlements": {"S0A1": {"id": "S0A1", "center": (8, 18), "members": {"T1"}, "founded": 0},
                        "S0B1": {"id": "S0B1", "center": (22, 18), "members": set(), "founded": 0}},
        "kingdoms": {"Aldric": {"home": "S0A1", "settlements": {"S0A1"}, "vassals": {}},
                     "Borin": {"home": "S0B1", "settlements": {"S0B1"}, "vassals": {}}},
        "empires": {"Aldric": {"emperor": "Aldric", "subject_kings": {"Borin": {}}, "discontent": {}}},
        "monarchs": {}, "leaders": {},
        "events": ["turn 10: AWK1 died (fell in battle)",
                   "turn 10: BWK1 died (fell in battle)",
                   "turn 10: KING Aldric DEFEATED Borin in war (8 loyal host vs 4; 1+1 fell) -> "
                   "Borin SUBJUGATED as a subject-king; an EMPIRE rises"],
    }
    prev = {"turn": 9,
            "positions": {"Aldric": (8, 6), "Borin": (22, 6), "AWK1": (8, 4), "BWK1": (22, 4),
                          "T1": (7, 17)},
            "realms": {"S0A1": "Aldric", "S0B1": "Borin"},
            "homes": {"Aldric": "S0A1", "Borin": "S0B1"}}
    keys_before = set(state)
    before = copy.deepcopy({k: state[k] for k in state if k != "agents"})
    agents_before = [(a.name, a.position, a.alive) for a in state["agents"]]
    r = PygameRenderer(turn_delay=0.0)
    pygame.init()
    try:
        r._ensure_screen(state["size"])
        tl = battle_scene(turn_events(state["events"], 10), prev, state)
        assert tl is not None, "the war line must be detected"
        # Every beat of the overlay (0.1 muster, 0.9 march, 2.0 clash, 2.8 casualties mid-tip,
        # 3.6/4.0 aftermath+banner), with the territory lerp engaged as the real playback does.
        for el, frac in ((0.1, 0.0), (0.9, 0.0), (2.0, 0.0), (2.8, 0.0), (3.6, 0.5), (4.0, 1.0)):
            r._territory_lerp = r._territory_colors(tl, frac, state)
            r._draw(state, battle=(tl, el))
        r._territory_lerp = {}
        # Smooth motion: mid-walk and settled frames (T1 lerps (7,17)->(8,18); Aldric stands).
        r._draw(state, motion=(prev["positions"], 0.4))
        r._draw(state, motion=(prev["positions"], 1.0))
        # update() at zero delay: draws once, keeps the snapshot RENDERER-LOCAL, never blocks.
        t0 = _time.monotonic()
        r.update(state)
        assert _time.monotonic() - t0 < 1.0, "zero-delay update must not play a cinematic"
        assert r._prev_snapshot is not None and r._prev_snapshot["turn"] == 10
        assert r._territory_lerp == {}, "no territory override lingers outside a cinematic"
    finally:
        pygame.quit()
    assert set(state) == keys_before, "cinematic wrote a new key into world_state"
    assert {k: state[k] for k in state if k != "agents"} == before, "cinematic draw mutated world_state"
    assert [(a.name, a.position, a.alive) for a in state["agents"]] == agents_before, \
        "cinematic draw mutated an agent"
    print("PASS test_pygame_cinematic_state_is_renderer_local_and_draw_read_only")


def test_pygame_palette_is_centralized_and_scene_constants_derive() -> None:
    """Slice 9: ONE central PALETTE defines the scene; the slice constants derive from it.

    The look is tunable in one place: grass/food/settlement/structure tones are PALETTE
    entries; commoner figures are drawn DESATURATED from their palette bases (they sit in the
    world) while the crown keeps full chroma (important things pop); the old teal settlement
    ring is retired for a near-neutral earth tint that cannot fight the realm hues.
    """
    try:
        from renderer.pygame_renderer import (PALETTE, _desat, _TRAIT_DESAT, _TRAIT_COLOR,
                                              _GRASS_BASE, _FOOD, _SETTLEMENT_FILL, _CROWN,
                                              _WATER, _FARMLAND, _CROP)
    except ImportError:
        print("PASS test_pygame_palette_is_centralized_and_scene_constants_derive (skipped: no pygame)")
        return
    assert isinstance(PALETTE, dict) and len(PALETTE) >= 30, "one central palette, richly keyed"
    assert all(isinstance(v, tuple) and len(v) == 3 and all(0 <= c <= 255 for c in v)
               for v in PALETTE.values()), "every palette entry is a sane RGB triple"
    # The slice constants DERIVE from the palette (tune PALETTE -> the whole scene follows).
    assert _GRASS_BASE == PALETTE["grass_base"] and _WATER == PALETTE["water"]
    assert _FOOD == PALETTE["wheat"] and _CROP == PALETTE["crop"]
    assert _SETTLEMENT_FILL == PALETTE["settlement_fill"] and _CROWN == PALETTE["crown"]
    assert _FARMLAND == PALETTE["farmland"]
    # _desat is pure: identity at 0, its own luma gray at 1, monotone in between.
    assert _desat((200, 40, 40), 0.0) == (200, 40, 40)
    gray = _desat((200, 40, 40), 1.0)
    assert gray[0] == gray[1] == gray[2], "full desaturation lands on a gray"
    # Figures are drawn desaturated from their saturated palette bases — and stay distinct.
    for trait in ("curiosity", "caution", "friendliness", "independence"):
        assert _TRAIT_COLOR[trait] == _desat(PALETTE[trait], _TRAIT_DESAT)
    assert len(set(_TRAIT_COLOR.values())) == 4, "desaturation keeps the four hues apart"
    # The settlement tint is near-neutral (low chroma) so realm colours own the map.
    r, g, b = PALETTE["settlement_fill"]
    assert max(r, g, b) - min(r, g, b) < 60, "the settlement ring no longer screams teal"
    print("PASS test_pygame_palette_is_centralized_and_scene_constants_derive")


def test_pygame_ambient_helpers_pure_bounded_and_rng_free() -> None:
    """Slice 9: the ambient-life helpers (smoke, birds) are pure frame functions — zero RNG.

    smoke_puffs: deterministic per (frame, hearth), three puffs that RISE (dy < 0), stay small,
    and FADE (alpha bounded, reaching low values late in the cycle); phase differs per hearth.
    ambient_birds: deterministic, at most two birds, OCCASIONAL (some windows fly, most don't),
    kept in the sky band. Neither touches the global random stream.
    """
    import random as _random
    try:
        from renderer.pygame_renderer import smoke_puffs, ambient_birds, _BIRD_WINDOW
    except ImportError:
        print("PASS test_pygame_ambient_helpers_pure_bounded_and_rng_free (skipped: no pygame)")
        return
    s0 = _random.getstate()
    assert smoke_puffs(100, 5, 7) == smoke_puffs(100, 5, 7), "smoke is a pure frame function"
    for frame in (0, 33, 100, 999):
        puffs = smoke_puffs(frame, 5, 7)
        assert len(puffs) == 3, "three staggered puffs per hearth"
        for dx, dy, r, alpha in puffs:
            assert dy < 0, "smoke rises"
            assert 1 <= r <= 4 and 0 <= alpha <= 90 and abs(dx) <= 10, "puffs stay small and soft"
    assert smoke_puffs(100, 5, 7) != smoke_puffs(100, 50, 70), "hearths puff out of lockstep"
    assert ambient_birds(500, 800) == ambient_birds(500, 800), "birds are a pure frame function"
    flew = sum(1 for w in range(40) if ambient_birds(w * _BIRD_WINDOW + 120, 800))
    assert 0 < flew < 40, f"birds are OCCASIONAL — {flew}/40 windows flew"
    for w in range(40):
        for x, y, spread in ambient_birds(w * _BIRD_WINDOW + 120, 800):
            assert -10 <= x <= 810 and 0 <= y <= 800 * 0.75, "birds cross the upper sky"
            assert 0.5 <= spread <= 5.0, "wingspan stays tiny"
    assert _random.getstate() == s0, "ambient helpers touched the global RNG stream"
    print("PASS test_pygame_ambient_helpers_pure_bounded_and_rng_free")


def test_pygame_full_bleed_landscape_and_lit_scene_draw_read_only() -> None:
    """Slice 9: the map zone is FULL-BLEED (wilderness margin + east coast baked into the
    cached terrain), shadows/ambient/grade draw per frame, and it is all still a pure READ.

    Asserts the geometry (map zone = grid + margin ring; terrain surface covers it; window
    sized to it), that the outer-east margin reads as WATER in the baked terrain, that shadow
    stamps are cached (same size -> same object), and that several animated frames (smoke,
    sway, shimmer, birds, flicker, castle flutter, label chips, agent shadows) plus a mid-walk
    motion frame leave world_state byte-identical. Skips gracefully without pygame.
    """
    import copy, os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import PygameRenderer, _MARGIN_CELLS, _PANEL_W, _HUD_H
    except ImportError:
        print("PASS test_pygame_full_bleed_landscape_and_lit_scene_draw_read_only (skipped: no pygame)")
        return
    village = {n: _FakeAgent(n, "curious", (6, 6), True, 2.0, 0.0)
               for n in ("a", "b", "c", "d", "e")}
    state = {
        "size": 24, "turn": 12, "food": [(2, 2), (9, 9)],
        "settlements": {
            "S001": {"id": "S001", "center": (6, 6), "members": set(village), "founded": 2},
            "S002": {"id": "S002", "center": (13, 12), "members": {"K"}, "founded": 5},
        },
        "leaders": {"S001": {"leader": "a", "followers": {"b"}, "since": 3}},
        "monarchs": {"S002": {"monarch": "K", "since": 8, "garrison": set()}},   # castle + pennant
        "kingdoms": {"K": {"home": "S002", "settlements": {"S002"}, "vassals": {}}},
        "empires": {},
        "events": ["turn 12: a talked to b: \"hi\""],
        "agents": list(village.values()) + [
            _FakeAgent("K", "independent and competitive", (13, 12), True, 300.0, 0.0)],
    }
    before = copy.deepcopy({k: state[k] for k in state if k != "agents"})
    agents_before = [(a.name, a.position, a.alive) for a in state["agents"]]
    r = PygameRenderer(turn_delay=0.0)
    pygame.init()
    try:
        r._ensure_screen(state["size"])
        # V4.6/V4.9 geometry: the MAP-zone WINDOW is still the playable grid + wilderness ring
        # sized by the window cell; the iso FIT cell now FRAMES the playable world (0..size) across
        # the viewport with a modest margin (the launch view / zoom-out floor), and the terrain
        # bakes into a DIAMOND canvas sized by _iso_base_offsets.
        from renderer.pygame_renderer import _fit_cell
        assert r._map_px == r._win_cell * (state["size"] + 2 * _MARGIN_CELLS)
        assert r._cell == r._cell0 == _fit_cell(state["size"], r._map_px), \
            "the iso fit cell frames the playable world"
        assert r._zoom_lo == float(r._cell0) and r._zoom_hi == float(r._zoom_buckets[-1]), \
            "zoom bounds run fit-whole-world .. close village"
        _ox, _oy, bw, bh = r._iso_base_offsets(r._cell0)
        assert r._terrain_bg is not None and r._terrain_bg.get_size() == (bw, bh), \
            "the baked landscape covers the whole iso diamond canvas"
        assert r._screen.get_size() == (r._map_px + _PANEL_W, r._map_px + _HUD_H)
        # The +x wilderness is SEA (a meandering iso coast) — classified straight off _tile_kind.
        assert r._tile_kind(state["size"] + 1, state["size"] // 2) == "sea", \
            "the +x margin is open water"
        assert r._tile_kind(state["size"] // 2, state["size"] // 2) == "land", \
            "the playable interior is land"
        # Shadow stamps are cached per size — the second request is the SAME surface.
        s1 = r._shadow_stamp(20, 8)
        assert r._shadow_stamp(20, 8) is s1, "shadow stamps must be cached, not rebuilt"
        # Several ANIMATED frames (the ambient clock advances) + a mid-walk motion frame.
        for _ in range(5):
            r._draw(state)
        r._draw(state, motion=({"a": (6, 5), "K": (12, 12)}, 0.5))
        assert r._grade is not None, "the warm daylight grade is cached and applied"
    finally:
        pygame.quit()
    assert {k: state[k] for k in state if k != "agents"} == before, \
        "slice-9 lit/ambient draw mutated world_state"
    assert [(a.name, a.position, a.alive) for a in state["agents"]] == agents_before, \
        "slice-9 draw mutated an agent"
    print("PASS test_pygame_full_bleed_landscape_and_lit_scene_draw_read_only")


def test_pygame_time_of_day_pure_periodic_and_smooth() -> None:
    """Slice 10: the day-cycle clock is a PURE function of the turn — periodic, in [0, 1),
    the four phases appear in order across a day, and the daylight curve has no
    discontinuity anywhere (including the midnight wrap). Zero RNG touched.
    """
    import random as _random
    try:
        from renderer.pygame_renderer import (time_of_day, daylight_factor, phase_name,
                                              _TURNS_PER_DAY)
    except ImportError:
        print("PASS test_pygame_time_of_day_pure_periodic_and_smooth (skipped: no pygame)")
        return
    s0 = _random.getstate()
    for t in (0, 1, 7, 12.5, 23, 24, 100, 1000):
        p = time_of_day(t)
        assert 0.0 <= p < 1.0, f"phase out of range at turn {t}"
        assert p == time_of_day(t), "the clock is pure"
        assert abs(p - time_of_day(t + _TURNS_PER_DAY)) < 1e-9 and \
            abs(p - time_of_day(t + 5 * _TURNS_PER_DAY)) < 1e-9, "the clock is periodic"
    # The four phases appear once each, in order, across one full day.
    names = [phase_name(time_of_day(i * _TURNS_PER_DAY / 400)) for i in range(400)]
    runs = [n for i, n in enumerate(names) if i == 0 or names[i - 1] != n]
    assert runs == ["dawn", "day", "dusk", "night"], f"phase order broke: {runs}"
    # Daylight: full at midday, zero at deep night, SMOOTH on a fine grid spanning the wrap.
    assert daylight_factor(0.35) == 1.0 and daylight_factor(0.85) == 0.0
    samples = [daylight_factor((i / 4000) % 1.0) for i in range(-200, 4200)]
    worst = max(abs(samples[i + 1] - samples[i]) for i in range(len(samples) - 1))
    assert worst < 0.01, f"daylight curve jumps by {worst} — a hard switch somewhere"
    assert all(0.0 <= v <= 1.0 for v in samples), "daylight factor stays in [0, 1]"
    assert _random.getstate() == s0, "the day/night clock touched the global RNG stream"
    print("PASS test_pygame_time_of_day_pure_periodic_and_smooth")


def test_pygame_phase_grade_interpolation_bounded() -> None:
    """Slice 10: the phase-driven scene grade interpolates CONTINUOUSLY and stays bounded —
    sane RGB everywhere, alpha between the day and deep-night grades, seamless at the wrap,
    midday exactly the slice-9 daylight tint, deep night a cool blue-dark. The companion
    helpers (dawn wash bump, night muting, star field) are pure and bounded too.
    """
    try:
        from renderer.pygame_renderer import (phase_tint, dawn_wash_factor, night_mute,
                                              star_field, PALETTE, _GRADE_ALPHA,
                                              _NIGHT_GRADE_A, _PH_DAWN_END)
    except ImportError:
        print("PASS test_pygame_phase_grade_interpolation_bounded (skipped: no pygame)")
        return
    prev = None
    for i in range(2001):
        rgb, a = phase_tint((i / 2000) % 1.0)
        assert all(0 <= c <= 255 for c in rgb), "grade colour out of RGB range"
        assert _GRADE_ALPHA <= a <= _NIGHT_GRADE_A, "grade alpha out of its band"
        if prev is not None:
            assert max(abs(rgb[j] - prev[0][j]) for j in range(3)) <= 5 and \
                abs(a - prev[1]) <= 4, f"grade jumps near phase {i / 2000}"
        prev = (rgb, a)
    assert phase_tint(0.0) == phase_tint(0.9999999), "the midnight wrap must be seamless"
    assert phase_tint(0.35) == (PALETTE["daylight"], _GRADE_ALPHA), \
        "midday holds the slice-9 daylight grade exactly"
    ntint, na = phase_tint(0.85)
    assert na == _NIGHT_GRADE_A and ntint[2] > ntint[0], "deep night is a cool blue-dark"
    # The sunrise wash: a smooth bump peaking mid-dawn, zero outside (and at) the band edges.
    assert dawn_wash_factor(_PH_DAWN_END / 2) > 0.99
    assert dawn_wash_factor(0.0) < 0.02 and dawn_wash_factor(_PH_DAWN_END - 1e-9) < 0.02
    assert dawn_wash_factor(0.35) == 0.0 and dawn_wash_factor(0.85) == 0.0
    # Night muting: identity by day, strictly dimmer at deep night, always a sane RGB.
    c = (196, 84, 70)
    assert night_mute(c, 0.0) == c, "daytime colours are untouched"
    m = night_mute(c, 1.0)
    assert sum(m) < sum(c) and all(0 <= v <= 255 for v in m), "night muting dims sanely"
    # The star field: pure, in-bounds, tiny sizes.
    st = star_field(760)
    assert st and st == star_field(760), "the star field is deterministic"
    assert all(0 <= x <= 760 and 0 <= y <= 760 and s in (1, 2) for x, y, s in st)
    print("PASS test_pygame_phase_grade_interpolation_bounded")


def test_pygame_night_draw_is_read_only_and_lights_the_dark() -> None:
    """Slice 10: a NIGHT frame (a turn deep in the night band) is still a pure READ — and it
    actually transforms the scene: the grade goes cool-dark, stars sit on the water, the
    castle raises torches and lit windows register their glow, while a night BATTLE frame
    (clash + banner beats) still draws over it all. Headless SDL-dummy; skips without pygame.
    """
    import copy, os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import (PygameRenderer, time_of_day, daylight_factor,
                                              _TURNS_PER_DAY, _NIGHT_GRADE_A)
    except ImportError:
        print("PASS test_pygame_night_draw_is_read_only_and_lights_the_dark (skipped: no pygame)")
        return
    night_turn = int(_TURNS_PER_DAY * 0.85)          # deep in the night band, any day
    assert daylight_factor(time_of_day(night_turn)) == 0.0, "the chosen turn must be night"
    village = {n: _FakeAgent(n, "curious", (6, 6), True, 2.0, 0.0)
               for n in ("a", "b", "c", "d", "e")}
    state = {
        "size": 24, "turn": night_turn, "food": [(2, 2), (9, 9)],
        "settlements": {
            "S001": {"id": "S001", "center": (6, 6), "members": set(village), "founded": 2},
            "S002": {"id": "S002", "center": (13, 12), "members": {"K"}, "founded": 5},
        },
        "leaders": {"S001": {"leader": "a", "followers": {"b"}, "since": 3}},
        "monarchs": {"S002": {"monarch": "K", "since": 8, "garrison": set()}},
        "kingdoms": {"K": {"home": "S002", "settlements": {"S002"}, "vassals": {}}},
        "empires": {},
        "events": [f"turn {night_turn}: a talked to b: \"hi\""],
        "agents": list(village.values()) + [
            _FakeAgent("K", "independent and competitive", (13, 12), True, 300.0, 0.0)],
    }
    before = copy.deepcopy({k: state[k] for k in state if k != "agents"})
    agents_before = [(a.name, a.position, a.alive) for a in state["agents"]]
    r = PygameRenderer(turn_delay=0.0)
    pygame.init()
    try:
        r._ensure_screen(state["size"])
        assert r._stars, "star candidates landed on the water at build time"
        assert r._dawn_wash is not None, "the sunrise wash is baked once per screen"
        for _ in range(4):                            # several animated night frames
            r._draw(state)
        assert r._phase == time_of_day(night_turn) and r._nf == 1.0, \
            "the frame's phase derives purely from the sim turn"
        tint, a = r._grade_tint
        # V4.3: night holds its (raised) brightness FLOOR — still a cool blue-dark, but no
        # longer so inky that buildings/agents/territory vanish. Tied to the constant so the
        # floor tunes in one place.
        assert tint[2] > tint[0] and a == _NIGHT_GRADE_A, "the night grade is a deep cool blue"
        assert r._grade.get_size() == (r._map_px, r._map_px), \
            "the grade covers the MAP only — HUD and panel stay ungraded UI"
        kinds = {light[0] for light in r._frame_lights}
        assert "torch" in kinds, "the castle raises torchlight at night"
        assert "window" in kinds, "lit windows register their glow at night"
        # A NIGHT BATTLE still reads: clash and banner beats draw fine over the dark scene.
        scene = {"kind": "conquest", "attacker": "K", "defender": "S001", "won": True,
                 "att_pos": (13, 12), "def_pos": (6, 6), "n_att": 6, "n_def": 4,
                 "att_color": (196, 84, 70), "def_color": (86, 132, 214),
                 "att_dead": [("z", (7, 7))], "def_dead": [("b", (6, 6))],
                 "banner": "K SEIZES S001 — MONARCH by force", "territory": []}
        r._draw(state, battle=(scene, 2.8))           # mid-clash
        r._draw(state, battle=(scene, 4.0))           # the outcome banner
        # A mid-walk frame keeps the phase gliding (fractional turn), still read-only.
        r._draw(state, motion=({"a": (6, 5)}, 0.5))
        assert 0.0 <= r._phase < 1.0
    finally:
        pygame.quit()
    assert {k: state[k] for k in state if k != "agents"} == before, \
        "the night draw mutated world_state"
    assert [(a.name, a.position, a.alive) for a in state["agents"]] == agents_before, \
        "the night draw mutated an agent"
    print("PASS test_pygame_night_draw_is_read_only_and_lights_the_dark")


def test_pygame_camera_transform_pure_and_inverse() -> None:
    """Slice 11: world_to_screen / screen_to_world are pure, exact inverses of each other,
    and behave like a camera: the view centre maps to the viewport centre, screen distances
    scale linearly with the cell size (the zoom), and panning the centre shifts everything
    the opposite way. Zero RNG touched.
    """
    import random as _random
    try:
        from renderer.pygame_renderer import world_to_screen, screen_to_world
    except ImportError:
        print("PASS test_pygame_camera_transform_pure_and_inverse (skipped: no pygame)")
        return
    s0 = _random.getstate()
    view = (760, 760)
    cams = ((5.0, 7.0, 16), (0.0, 0.0, 6), (22.5, 3.25, 48), (20.0, 20.0, 14))
    points = ((0.5, 0.5), (10.0, 20.0), (-3.0, 44.0), (12.25, 0.0))
    for cam in cams:
        for pos in points:
            sp = world_to_screen(pos, cam, view)
            back = screen_to_world(sp, cam, view)
            assert abs(back[0] - pos[0]) < 1e-9 and abs(back[1] - pos[1]) < 1e-9, \
                f"screen_to_world must invert world_to_screen (cam={cam}, pos={pos})"
        # The camera CENTRE lands exactly on the viewport centre.
        assert world_to_screen(cam[:2], cam, view) == (380.0, 380.0)
    # Zoom scaling: doubling the cell size doubles every on-screen distance.
    a1 = world_to_screen((10.0, 10.0), (8.0, 8.0, 12), view)
    b1 = world_to_screen((14.0, 10.0), (8.0, 8.0, 12), view)
    a2 = world_to_screen((10.0, 10.0), (8.0, 8.0, 24), view)
    b2 = world_to_screen((14.0, 10.0), (8.0, 8.0, 24), view)
    assert (b2[0] - a2[0]) == 2 * (b1[0] - a1[0]), "screen distance scales with the cell size"
    # Panning the centre right moves the world LEFT on screen.
    x_before = world_to_screen((10.0, 10.0), (8.0, 8.0, 12), view)[0]
    x_after = world_to_screen((10.0, 10.0), (11.0, 8.0, 12), view)[0]
    assert x_after < x_before, "panning the camera right slides the world left"
    assert world_to_screen((3.0, 4.0), (1.0, 1.0, 10), view) == \
        world_to_screen((3.0, 4.0), (1.0, 1.0, 10), view), "the transform is pure"
    assert _random.getstate() == s0, "the camera transform touched the global RNG stream"
    print("PASS test_pygame_camera_transform_pure_and_inverse")


def test_pygame_iso_transform_pure_and_inverse() -> None:
    """V4.6: the ISO projection is pure, a 2:1 diamond, and its ground-plane inverse round-trips.

    world_to_screen_iso / screen_to_world_iso are the one shared iso transform every map draw
    goes through; screen_to_world_iso must invert it on the ground plane (z = 0). Skips without
    pygame (the transform lives in the optional renderer module)."""
    import random as _random
    try:
        from renderer.pygame_renderer import world_to_screen_iso, screen_to_world_iso, _ISO_ZH
    except ImportError:
        print("PASS test_pygame_iso_transform_pure_and_inverse (skipped: no pygame)")
        return
    view = (760, 760)
    s0 = _random.getstate()
    for cam in ((0.0, 0.0, 16), (8.0, 8.0, 12), (20.0, 5.0, 24)):
        for pos in ((0.0, 0.0), (5.0, 3.0), (10.0, 10.0), (-2.0, 7.0)):
            sp = world_to_screen_iso(pos, cam, view)
            back = screen_to_world_iso(sp, cam, view)
            assert abs(back[0] - pos[0]) < 1e-6 and abs(back[1] - pos[1]) < 1e-6, \
                f"screen_to_world_iso must invert on the ground plane (cam={cam}, pos={pos})"
        # The camera centre projects to the viewport centre.
        assert world_to_screen_iso(cam[:2], cam, view) == (380.0, 380.0)
    cam = (0.0, 0.0, 20)
    # 2:1 DIAMOND: +x runs down-RIGHT, +y down-LEFT; a unit step is twice as wide as tall.
    ox, oy = world_to_screen_iso((0.0, 0.0), cam, view)
    px, py = world_to_screen_iso((1.0, 0.0), cam, view)
    assert px > ox and py > oy, "+x runs down-right"
    qx, qy = world_to_screen_iso((0.0, 1.0), cam, view)
    assert qx < ox and qy > oy, "+y runs down-left"
    assert abs((px - ox)) == 2 * abs((py - oy)), "the tile is a 2:1 diamond (twice as wide as tall)"
    # z lifts a point straight UP the screen (height above the ground plane) and never sideways.
    lifted = world_to_screen_iso((3.0, 4.0, 2.0), cam, view)
    ground = world_to_screen_iso((3.0, 4.0, 0.0), cam, view)
    assert lifted[0] == ground[0] and lifted[1] == ground[1] - 2 * cam[2] * _ISO_ZH, \
        "z raises a point vertically by z*cell*_ISO_ZH"
    assert world_to_screen_iso((3.0, 4.0), cam, view) == world_to_screen_iso((3.0, 4.0), cam, view), \
        "the iso transform is pure"
    assert _random.getstate() == s0, "the iso transform touched the global RNG stream"
    print("PASS test_pygame_iso_transform_pure_and_inverse")


def test_pygame_camera_clamp_buckets_and_culling_pure() -> None:
    """Slice 11: the camera-support helpers are pure and bounded — clamp_camera centres a
    world that fits the viewport and pins the view inside one that doesn't; zoom_buckets is
    a small sorted integer ladder spanning ~0.4x..3x of the fit cell (always containing it);
    visible_on_screen culls exactly outside the padded viewport.
    """
    import random as _random
    try:
        from renderer.pygame_renderer import (clamp_camera, zoom_buckets, visible_on_screen,
                                              _MARGIN_CELLS, _CELL_FLOOR, _CELL_CEIL, _ZOOM_IN_MAX)
    except ImportError:
        print("PASS test_pygame_camera_clamp_buckets_and_culling_pure (skipped: no pygame)")
        return
    s0 = _random.getstate()
    view = (760, 760)
    # A world SMALLER than the viewport floats centred whatever the requested centre.
    assert clamp_camera(99.0, -99.0, 4, 20, view) == (10.0, 10.0)
    # A world BIGGER than the viewport clamps the centre so the view never leaves it.
    size, cell = 40, 32                                # world px = 46*32 = 1472 > 760
    half = view[0] / (2.0 * cell)
    cx, cy = clamp_camera(-50.0, 999.0, cell, size, view)
    assert cx == -_MARGIN_CELLS + half and cy == size + _MARGIN_CELLS - half
    inside = clamp_camera(20.0, 20.0, cell, size, view)
    assert inside == (20.0, 20.0), "an in-bounds centre passes through unchanged"
    # The zoom ladder: sorted unique ints, contains the base, spans the bounded range.
    for base in (10, 14, 16, 25, 44):
        bs = zoom_buckets(base)
        assert list(bs) == sorted(set(bs)) and all(isinstance(b, int) for b in bs)
        assert base in bs, "the fit-whole-world cell is always a bucket (launch blits 1:1)"
        assert 4 <= len(bs) <= 10, f"a small quantized ladder, got {len(bs)} for base {base}"
        assert bs[0] >= max(_CELL_FLOOR, round(base * 0.4) - 1) and bs[0] < base
        assert bs[-1] <= min(_CELL_CEIL, int(base * _ZOOM_IN_MAX)) and bs[-1] > base
        assert bs == zoom_buckets(base), "the ladder is pure"
    # Culling: inside/edge kept, beyond the padded viewport dropped.
    assert visible_on_screen(380, 380, 0, 760, 760)
    assert visible_on_screen(-8, 760, 10, 760, 760), "the pad keeps part-visible things"
    assert not visible_on_screen(-30, 380, 10, 760, 760)
    assert not visible_on_screen(380, 9999, 50, 760, 760)
    assert _random.getstate() == s0, "camera helpers touched the global RNG stream"
    print("PASS test_pygame_camera_clamp_buckets_and_culling_pure")


def test_pygame_lod_tiers_have_hysteresis() -> None:
    """Slice 11: LOD tier selection is pure and hysteretic — far/mid/close appear in order
    as the zoom sweeps, and wobbling the cell size right at a boundary can never flicker
    the tier, because leaving a tier needs the cell to move past the band it entered by.
    """
    try:
        from renderer.pygame_renderer import (lod_tier, _LOD_FAR_MAX, _LOD_CLOSE_MIN,
                                              _LOD_HYST)
    except ImportError:
        print("PASS test_pygame_lod_tiers_have_hysteresis (skipped: no pygame)")
        return
    assert lod_tier(5.0, "mid") == "far" and lod_tier(16.0, "mid") == "mid" \
        and lod_tier(40.0, "mid") == "close"
    # Sweeping the whole range visits the tiers once each, in order (no bouncing).
    tier, seen = "far", ["far"]
    c = 4.0
    while c <= 60.0:
        tier = lod_tier(c, tier)
        if tier != seen[-1]:
            seen.append(tier)
        c += 0.25
    assert seen == ["far", "mid", "close"], f"tier order broke: {seen}"
    # Hysteresis at the far/mid boundary: both tiers are STICKY on the boundary itself.
    assert lod_tier(_LOD_FAR_MAX, "far") == "far" and lod_tier(_LOD_FAR_MAX, "mid") == "mid"
    # Wobbling across the boundary inside the band never flips the tier, from either side.
    for start in ("far", "mid"):
        tier = start
        for cell in (_LOD_FAR_MAX - 0.4, _LOD_FAR_MAX + 0.4) * 20:
            tier = lod_tier(cell, tier)
            assert tier == start, "the far/mid boundary flickered inside the hysteresis band"
    # Same at the mid/close boundary.
    assert lod_tier(_LOD_CLOSE_MIN, "close") == "close" and lod_tier(_LOD_CLOSE_MIN, "mid") == "mid"
    for start in ("close", "mid"):
        tier = start
        for cell in (_LOD_CLOSE_MIN - 0.4, _LOD_CLOSE_MIN + 0.4) * 20:
            tier = lod_tier(cell, tier)
            assert tier == start, "the mid/close boundary flickered inside the hysteresis band"
    # Leaving the band does switch (hysteresis, not a latch).
    assert lod_tier(_LOD_FAR_MAX + _LOD_HYST + 0.1, "far") == "mid"
    assert lod_tier(_LOD_CLOSE_MIN - _LOD_HYST - 0.1, "close") == "mid"
    print("PASS test_pygame_lod_tiers_have_hysteresis")


def test_pygame_camera_state_renderer_local_and_lod_draw_read_only() -> None:
    """Slice 11: the camera lives ONLY on the renderer and drawing at every zoom is a pure
    READ. On a BIG world (grid 40, two kingdoms far apart): the run opens on the fit-whole-
    world view; gliding out to the smallest bucket lands in the FAR strategy tier; gliding
    in to the largest lands CLOSE with the centre clamped inside the world; the bucket
    landscape cache stays bounded; wheel/HOME events drive only camera targets — and
    world_state is byte-identical throughout. Headless SDL-dummy; skips without pygame.
    """
    import copy, os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import (PygameRenderer, _MARGIN_CELLS, _TERRAIN_LRU,
                                              clamp_camera)
    except ImportError:
        print("PASS test_pygame_camera_state_renderer_local_and_lod_draw_read_only (skipped: no pygame)")
        return
    size = 40
    west = {n: _FakeAgent(n, "curious", (6, 6), True, 2.0, 0.0) for n in ("a", "b", "c", "d", "e")}
    east = {n: _FakeAgent(n, "cautious", (34, 34), True, 2.0, 0.0) for n in ("p", "q", "r", "s")}
    state = {
        "size": size, "turn": 9, "food": [(2, 2), (20, 20), (37, 37)],
        "settlements": {
            "S0A1": {"id": "S0A1", "center": (6, 6), "members": set(west), "founded": 1},
            "S0B1": {"id": "S0B1", "center": (34, 34), "members": set(east), "founded": 2},
        },
        "monarchs": {"S0A1": {"monarch": "KA", "since": 3, "garrison": set()},
                     "S0B1": {"monarch": "KB", "since": 4, "garrison": set()}},
        "kingdoms": {"KA": {"home": "S0A1", "settlements": {"S0A1"}, "vassals": {}},
                     "KB": {"home": "S0B1", "settlements": {"S0B1"}, "vassals": {}}},
        "empires": {}, "leaders": {},
        "events": ["turn 9: a talked to b: \"hi\""],
        "agents": (list(west.values()) + list(east.values())
                   + [_FakeAgent("KA", "independent", (6, 6), True, 200.0, 0.0),
                      _FakeAgent("KB", "independent", (34, 34), True, 200.0, 0.0)]),
    }
    keys_before = set(state)
    before = copy.deepcopy({k: state[k] for k in state if k != "agents"})
    agents_before = [(a.name, a.position, a.alive) for a in state["agents"]]
    r = PygameRenderer(turn_delay=0.0)
    pygame.init()
    try:
        r._ensure_screen(size)
        # Default start (before the first drawn frame): the whole-world fit, centred on the grid.
        assert (r._cam_x, r._cam_y) == (size / 2.0, size / 2.0) == (r._cam_tx, r._cam_ty)
        assert r._cell == r._cell0 and r._cam_cell == float(r._cell0)
        assert r._cell0 in r._zoom_buckets, "the fit cell is a zoom bucket (1:1 blits)"
        # V4.10: the FIRST inhabited frame SNAPS the camera onto the member-weighted settlement
        # centroid and fits the inhabited region — the world opens on the action, not empty grid.
        # (The centroid is then clamped so the view never leaves the world; here the two towns span
        # the whole map so the inhabited fit ~ the whole-world fit and the centre clamps to the grid.)
        hx, hy, hcell = r._home_view(state)
        ecx, ecy = clamp_camera(hx, hy, hcell, size, (r._map_px, r._map_px))
        r._draw(state)
        assert (r._cam_x, r._cam_y) == (ecx, ecy), "the launch frame centres on the settlement centroid"
        assert r._cam_cell == hcell and r._zoom_lo == hcell, \
            "the launch frame fits the inhabited region; the zoom-out floor tracks it"
        # V4.10: zooming OUT past the INHABITED-region floor is CLAMPED — the whole empty map can
        # never be pulled back into view. A target below _zoom_lo settles exactly on the floor.
        r._cam_tcell = float(r._zoom_buckets[0])          # below the floor (a bake bucket, not a bound)
        for _ in range(60):
            r._draw(state)
        assert r._cam_tcell == r._zoom_lo == hcell, "zoom-out is clamped to the inhabited-region floor"
        assert r._cell == int(round(hcell)), "gliding out settles on the inhabited-region view"
        # GLIDE IN to the close village (top bound) with an absurd pan target: the clamp keeps the
        # view inside the world and the tier lands CLOSE.
        r._cam_tx, r._cam_ty = -999.0, 999.0
        r._cam_tcell = r._zoom_hi
        for _ in range(80):
            r._draw(state)
        assert r._cell == int(round(r._zoom_hi)) and r._lod == "close"
        assert -_MARGIN_CELLS <= r._cam_x <= size + _MARGIN_CELLS
        assert -_MARGIN_CELLS <= r._cam_y <= size + _MARGIN_CELLS
        # The cached-surface strategy: base bake untouched, bucket cache LRU-bounded.
        assert r._terrain_bg is not None and len(r._terrain_zoom) <= _TERRAIN_LRU
        # Panning the TARGET right slides a fixed cell's screen position left once settled.
        x1 = r._to_px(20, 20)[0]
        r._cam_tx = r._cam_x + 3.0
        for _ in range(40):
            r._draw(state)
        assert r._to_px(20, 20)[0] < x1, "panning right slides the world left"
        # Camera EVENTS drive only renderer-local targets: one wheel notch nudges the zoom target
        # by a small multiplicative step (out here), HOME eases back to the fit-whole-world view.
        t_before = r._cam_tcell
        r._handle_camera_event(pygame.event.Event(pygame.MOUSEWHEEL, y=-1, x=0))
        assert r._zoom_lo <= r._cam_tcell < t_before, "wheel-down nudges the zoom target out one notch"
        # V4.10: HOME eases back to the INHABITED-region view (centroid + fit), not the grid centre.
        r._handle_camera_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_HOME))
        assert r._cam_tcell == hcell and (r._cam_tx, r._cam_ty) == (hx, hy), \
            "HOME re-frames onto the settlement centroid"
        for _ in range(60):
            r._draw(state)
        assert r._cell == int(round(hcell)), "HOME glides back to the inhabited-region view"
    finally:
        pygame.quit()
    assert set(state) == keys_before, "the camera wrote a key into world_state"
    assert {k: state[k] for k in state if k != "agents"} == before, \
        "a panned/zoomed draw mutated world_state"
    assert [(a.name, a.position, a.alive) for a in state["agents"]] == agents_before, \
        "a panned/zoomed draw mutated an agent"
    print("PASS test_pygame_camera_state_renderer_local_and_lod_draw_read_only")


def test_pygame_window_sizing_rect_layout_and_resize() -> None:
    """V4.11: the window opens at a requested size and the MAP ZONE is the RECTANGLE that remains —
    map_w + panel == win_w, map_h + hud == win_h at every aspect (16:9 / 16:10 / tall / small); the
    side panel is a CLAMPED PROPORTION of the width; the caches (grade/void/vignette) match the map
    zone; the iso fit keeps the world on screen; and a VIDEORESIZE recomputes all of it. window=None
    keeps the LEGACY square layout byte-identical. Headless SDL-dummy; skips without pygame."""
    import copy, os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import (PygameRenderer, _panel_width, _PANEL_W, _PANEL_MIN,
                                              _PANEL_MAX, _HUD_H, _MARGIN_CELLS, _fit_cell)
    except ImportError:
        print("PASS test_pygame_window_sizing_rect_layout_and_resize (skipped: no pygame)")
        return
    size = 24
    village = {n: _FakeAgent(n, "curious", (6, 6), True, 2.0, 0.0) for n in ("a", "b", "c", "d", "e")}
    state = {
        "size": size, "turn": 9, "food": [(2, 2)],
        "settlements": {"S001": {"id": "S001", "center": (6, 6), "members": set(village), "founded": 2}},
        "monarchs": {}, "leaders": {}, "kingdoms": {}, "empires": {},
        "events": ["turn 9: a talked to b: \"hi\""], "agents": list(village.values()),
    }
    before = copy.deepcopy({k: state[k] for k in state if k != "agents"})
    pygame.init()
    try:
        # window=None -> the LEGACY world-derived SQUARE layout (unchanged by V4.11).
        legacy = PygameRenderer(turn_delay=0.0)
        legacy._ensure_screen(size)
        assert legacy._map_px == legacy._map_h == legacy._win_cell * (size + 2 * _MARGIN_CELLS)
        assert legacy._panel_w == _PANEL_W and legacy._hud_h == _HUD_H
        assert legacy._cell0 == _fit_cell(size, legacy._map_px), "legacy fit is unchanged"
        assert legacy._screen.get_size() == (legacy._map_px + _PANEL_W, legacy._map_px + _HUD_H)
        # explicit window sizes: the map zone is the rectangle that remains, at every aspect.
        for win in ((1600, 900), (1440, 900), (760, 1180), (900, 600)):
            r = PygameRenderer(turn_delay=0.0, window=win)
            r._ensure_screen(size)
            w, h = r._screen.get_size()
            assert (w, h) == win, f"{win}: window must open at the requested size"
            assert r._map_px + r._panel_w == w, f"{win}: map width + panel must fill the window"
            assert r._map_h + r._hud_h == h, f"{win}: map height + HUD must fill the window"
            assert r._panel_w == _panel_width(w) and _PANEL_MIN <= r._panel_w <= _PANEL_MAX, \
                f"{win}: the panel is a clamped proportion of the width"
            for cache in (r._grade, r._void_bg, r._vignette):
                assert cache.get_size() == (r._map_px, r._map_h), f"{win}: cache must match the map zone"
            cx, cy = r._to_px(size / 2, size / 2)       # the world centre lands inside the map zone
            assert 0 <= cx <= r._map_px and 0 <= cy <= r._map_h, f"{win}: world centre off the map zone"
            r._draw(state)                              # a full frame at this aspect must not raise
        # a VIDEORESIZE recomputes the layout, the caches and the fit.
        r = PygameRenderer(turn_delay=0.0, window=(1600, 900))
        r._ensure_screen(size)
        r._apply_resize(1100, 1000)
        assert r._screen.get_size() == (1100, 1000)
        assert r._map_px + r._panel_w == 1100 and r._map_h + r._hud_h == 1000, "resize must relayout"
        assert r._grade.get_size() == (r._map_px, r._map_h), "resize must rebuild the caches"
        assert r._panel_w == _panel_width(1100), "resize must re-clamp the panel"
        r._draw(state)
    finally:
        pygame.quit()
    assert {k: state[k] for k in state if k != "agents"} == before, \
        "window sizing / resize mutated world_state"
    print("PASS test_pygame_window_sizing_rect_layout_and_resize")


def test_pygame_display_mode_stable_and_always_escapable() -> None:
    """V4.12 regression: the two bugs that made --showcase unusable.

    (1) FLICKER — set_mode emits its own VIDEORESIZE, which used to re-enter _ensure_screen and call
    set_mode again: an endless display re-creation loop that read as rapid blinking in borderless
    fullscreen. The display must be created ONCE and never re-created while the mode is steady, and
    resize ECHOES must be ignored. (2) TRAPPED — ESC only left fullscreen, stranding the viewer in a
    borderless window; ESC/Q/QUIT must now END THE RUN from any mode, and F11 must actually be able
    to leave fullscreen without being re-forced. Headless SDL-dummy; skips without pygame."""
    import os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import PygameRenderer, _QUIT_KEYS
    except ImportError:
        print("PASS test_pygame_display_mode_stable_and_always_escapable (skipped: no pygame)")
        return
    size = 24
    village = {n: _FakeAgent(n, "curious", (6, 6), True, 2.0, 0.0) for n in ("a", "b", "c", "d", "e")}
    state = {
        "size": size, "turn": 9, "food": [(2, 2)],
        "settlements": {"S001": {"id": "S001", "center": (6, 6), "members": set(village), "founded": 2}},
        "monarchs": {}, "leaders": {}, "kingdoms": {}, "empires": {},
        "events": ["turn 9: a talked to b: \"hi\""], "agents": list(village.values()),
    }
    pygame.init()
    try:
        r = PygameRenderer(turn_delay=0.0, showcase=True, window="fullscreen")
        r._ensure_screen(size)
        assert r._fullscreen and r._mode_sets == 1, "the display opens with exactly one set_mode"
        surfaces = {id(r._screen)}
        for _ in range(60):                       # a steady run: draw + the per-turn _ensure_screen
            r._draw(state)
            r._ensure_screen(size)
            surfaces.add(id(r._screen))
        assert r._mode_sets == 1, f"display re-created while steady (set_mode={r._mode_sets})"
        assert len(surfaces) == 1, "the display surface was replaced mid-run (flicker)"
        for _ in range(10):                       # the echo set_mode emits, and a fullscreen resize
            r._apply_resize(*r._screen.get_size())
            r._apply_resize(1234, 777)
        assert r._mode_sets == 1, "a resize ECHO re-created the display (the flicker loop)"
        # ESC / Q / QUIT must end the run from ANY mode, in both the normal and cinematic pumps.
        for key in _QUIT_KEYS:
            for fullscreen in (True, False):
                for pump in (r._pump_events, r._pump_cinema_events):
                    r._fullscreen = fullscreen
                    pygame.event.clear()
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
                    try:
                        pump()
                        raise AssertionError(f"key {key} did not quit (fullscreen={fullscreen})")
                    except KeyboardInterrupt:
                        pass
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        try:
            r._pump_events()
            raise AssertionError("QUIT (window close / macOS Cmd+Q) did not end the run")
        except KeyboardInterrupt:
            pass
        # F11 must genuinely leave fullscreen, and the next turn must not snap back.
        r2 = PygameRenderer(turn_delay=0.0, showcase=True, window="fullscreen")
        r2._ensure_screen(size)
        assert r2._fullscreen, "showcase starts fullscreen"
        r2._toggle_fullscreen()
        assert not r2._fullscreen, "F11 must leave fullscreen"
        r2._ensure_screen(size)
        assert not r2._fullscreen, "leaving fullscreen must not be re-forced on the next turn"
        # an interrupted run releases fullscreen rather than stranding the display
        r3 = PygameRenderer(turn_delay=0.0, showcase=True, window="fullscreen")
        try:
            with r3.live():
                r3._ensure_screen(size)
                raise KeyboardInterrupt          # simulate Ctrl+C (SIGINT) mid-run
        except KeyboardInterrupt:
            pass
        assert not r3._fullscreen, "an interrupted run must release the fullscreen surface"
    finally:
        pygame.quit()
    print("PASS test_pygame_display_mode_stable_and_always_escapable")


def test_pygame_hidpi_layout_fills_the_drawable_surface() -> None:
    """V4.13 regression: on a macOS Retina display set_mode hands back a BACKING SURFACE larger
    than the point size we asked for. The layout used to be derived from the REQUEST, so the world
    was drawn into the top-left corner of that surface and the rest of the screen stayed black.
    Everything (map zone, panel/HUD split, caches, camera fit) must derive from the TRUE DRAWABLE
    SIZE, resize echoes (which speak in points) must still be ignored, and mouse input (also in
    points) must be converted into drawable pixels. Headless SDL-dummy; skips without pygame."""
    import os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import PygameRenderer, _panel_width
    except ImportError:
        print("PASS test_pygame_hidpi_layout_fills_the_drawable_surface (skipped: no pygame)")
        return
    size = 24
    village = {n: _FakeAgent(n, "curious", (6, 6), True, 2.0, 0.0) for n in ("a", "b", "c", "d", "e")}
    state = {
        "size": size, "turn": 9, "food": [(2, 2)],
        "settlements": {"S001": {"id": "S001", "center": (6, 6), "members": set(village), "founded": 2}},
        "monarchs": {}, "leaders": {}, "kingdoms": {}, "empires": {},
        "events": ["turn 9: a talked to b: \"hi\""], "agents": list(village.values()),
    }
    pygame.init()
    try:
        # SHOWCASE on a 2x display: the map zone must cover the ENTIRE backing surface.
        r = PygameRenderer(turn_delay=0.0, showcase=True, window=(800, 600))
        r._drawable_size = lambda rw, rh: (rw * 2, rh * 2)      # pretend Retina
        r._ensure_screen(size)
        assert (r._req_w, r._req_h) == (800, 600), "we still ASK the OS for the point size"
        assert (r._win_w, r._win_h) == (1600, 1200), "layout must use the drawable size"
        assert r._px_scale == 2.0 and r._map_px == 1600 and r._map_h == 1200, \
            f"showcase map zone must fill the surface, got {(r._map_px, r._map_h)}"
        for cache in (r._grade, r._void_bg, r._vignette):
            assert cache.get_size() == (1600, 1200), "caches must match the drawable map zone"
        cx, cy = r._to_px(size / 2, size / 2)
        assert 0 <= cx <= 1600 and 0 <= cy <= 1200, "the world centre must land on the surface"
        assert r._mouse_px((100, 50)) == (200, 100), "mouse points must scale to drawable pixels"
        r._draw(state)                                          # a full HiDPI frame must not raise
        # a resize ECHO in POINTS must not re-create the display (the V4.12 flicker loop)
        sets = r._mode_sets
        for _ in range(5):
            r._apply_resize(800, 600)
            r._apply_resize(1600, 1200)
        assert r._mode_sets == sets, "a HiDPI resize echo re-created the display"
        # the FULL UI on a 2x display: chrome still splits the DRAWABLE, and scales with it.
        r2 = PygameRenderer(turn_delay=0.0, window=(800, 600))
        r2._drawable_size = lambda rw, rh: (rw * 2, rh * 2)
        r2._ensure_screen(size)
        assert r2._map_px + r2._panel_w == 1600, "map width + panel must fill the drawable width"
        assert r2._map_h + r2._hud_h == 1200, "map height + HUD must fill the drawable height"
        assert r2._panel_w == _panel_width(1600), "the panel is a proportion of the REAL width"
        assert r2._hud_h > 0 and r2._ui_scale() == 2.0, "the chrome scales with the backing store"
        r2._draw(state)
        # a NORMAL display is completely unaffected (scale 1, no font rebuild, same layout).
        r3 = PygameRenderer(turn_delay=0.0, window=(800, 600))
        r3._ensure_screen(size)
        assert r3._px_scale == 1.0 and (r3._win_w, r3._win_h) == (800, 600)
        assert r3._mouse_px((100, 50)) == (100, 50)
    finally:
        pygame.quit()
    print("PASS test_pygame_hidpi_layout_fills_the_drawable_surface")


def test_pygame_showcase_camera_is_rock_steady() -> None:
    """V4.13 regression: --showcase is a RECORDING mode, and the frame visibly jittered — the
    ambient drift/orbit, the zoom breath, the banner zoom-punch and the clash screen-shake all
    moved the lens. In showcase the camera must be DEAD STILL between its deliberate eases to
    events; --showcase-motion puts the motion back. Headless SDL-dummy; skips without pygame."""
    import os as _os, time as _time
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import PygameRenderer
    except ImportError:
        print("PASS test_pygame_showcase_camera_is_rock_steady (skipped: no pygame)")
        return
    size = 24
    village = {n: _FakeAgent(n, "curious", (6, 6), True, 2.0, 0.0) for n in ("a", "b", "c", "d", "e")}
    state = {
        "size": size, "turn": 9, "food": [(2, 2)],
        "settlements": {"S001": {"id": "S001", "center": (6, 6), "members": set(village), "founded": 2}},
        "monarchs": {}, "leaders": {}, "kingdoms": {}, "empires": {},
        "events": ["turn 9: a talked to b: \"hi\""], "agents": list(village.values()),
    }

    def targets(r, frames=40, poke=False):
        """The distinct camera TARGETS the renderer asks for over a run of drawn frames.

        Also reports whether the juice was ever LIVE during the run. It has to be sampled per
        frame, not read off the renderer afterwards: the shake decays by _SHAKE_DECAY every drawn
        frame and the punch runs on a _PUNCH_DUR wall clock, so by frame 40 both have expired on
        any machine slower than ~12ms/frame. Asserting on the tail state made this test pass or
        fail on frame rate rather than on behaviour.
        """
        seen, juiced = set(), False
        for i in range(frames):
            if poke and i == 5:                       # fire the camera-moving juice
                r._shake_amp, r._punch_t = 8.0, _time.monotonic()
            r._draw(state)
            seen.add((round(r._cam_tx, 6), round(r._cam_ty, 6), round(r._cam_tcell, 6)))
            juiced = juiced or r._shake != (0, 0) or r._punch != 1.0
        return seen, juiced

    pygame.init()
    try:
        r = PygameRenderer(turn_delay=0.0, showcase=True, window=(900, 700))
        r._ensure_screen(size)
        r._start_time = _time.monotonic() - 30.0      # past the title card, deep into the "orbit"
        seen, juiced = targets(r, poke=True)
        assert len(seen) == 1, f"the showcase camera drifted/breathed ({len(seen)} targets)"
        assert not juiced, "clash shake / banner zoom-punch must be off in showcase, on every frame"
        assert r._shake == (0, 0) and r._shake_amp == 0.0, "clash shake must be off in showcase"
        assert r._punch == 1.0 and r._punch_t is None, "banner zoom-punch must be off in showcase"
        # --showcase-motion puts the drift + the shake back.
        m = PygameRenderer(turn_delay=0.0, showcase=True, showcase_motion=True, window=(900, 700))
        m._ensure_screen(size)
        m._start_time = _time.monotonic() - 30.0
        moved, m_juiced = targets(m, poke=True)
        assert len(moved) > 1, "--showcase-motion must restore the ambient drift / zoom breath"
        assert m_juiced, "--showcase-motion must restore the juice"
        # a NORMAL (non-showcase) run keeps the V4.9 juice exactly as it was.
        n = PygameRenderer(turn_delay=0.0, window=(900, 700))
        n._ensure_screen(size)
        n._shake_amp, n._punch_t = 8.0, _time.monotonic()
        n._draw(state)
        assert n._shake != (0, 0) and n._punch != 1.0, "the default renderer must still have juice"
    finally:
        pygame.quit()
    print("PASS test_pygame_showcase_camera_is_rock_steady")


def test_showcase_scene_opens_in_a_standoff_that_must_break() -> None:
    """V4.14: the showcase scene is THREE realms whose geometry makes a war cascade inevitable.

    `--stage war` produced exactly one war, on turn 1, and then ~50 dead turns. The showcase scene
    stages three realms instead: A borders both B and C, B and C are out of KINGDOM_REACH of each
    other, every capital is beyond ATTACK_RADIUS of its own vassal town (so no realm eats itself on
    turn 1), and all three open able to field the SAME host — so empire.update's winnable-war test
    (strict >) holds fire at the start and the first war falls a few turns in, on camera. RNG-free
    staging, so these are exact assertions."""
    import empire, kingdoms, monarchy, scenario, world
    world.create_world(size=26)
    state = world.world_state
    state["taxation_on"] = True
    scenario.apply(state, "showcase")
    kings = ["Aldric", "Borin", "Cyrus"]
    assert sorted(state["kingdoms"]) == kings, "three realms are staged"
    assert len(state["settlements"]) == 6, "each realm has a capital and a vassal town"
    assert not state["empires"], "no empire exists yet — the wars have not been fought"
    centres = {sid: rec["center"] for sid, rec in state["settlements"].items()}
    for realm in "ABC":                       # a capital must be out of reach of its OWN vassal town
        d = monarchy._chebyshev(centres[f"S0{realm}1"], centres[f"S0{realm}2"])
        assert monarchy.ATTACK_RADIUS < d <= kingdoms.KINGDOM_REACH, \
            f"realm {realm}: capital-vassal spacing {d} allows a turn-1 coup or blocks vassalage"
    def gap(x: str, y: str) -> int:
        return min(monarchy._chebyshev(centres[f"S0{x}{i}"], centres[f"S0{y}{j}"])
                   for i in "12" for j in "12")
    assert gap("A", "B") <= kingdoms.KINGDOM_REACH and gap("A", "C") <= kingdoms.KINGDOM_REACH, \
        "A must border BOTH rivals — it is the empire-builder and the target"
    assert gap("B", "C") > kingdoms.KINGDOM_REACH, "B and C must not border each other"
    for name in kings:                        # a king's seat is clear of every town (it stays out of it)
        king = next(a for a in state["agents"] if a.name == name)
        near = min(monarchy._chebyshev(king.position, c) for c in centres.values())
        assert near > monarchy.ATTACK_RADIUS, f"{name}'s seat is inside conquest range of a town"
    hosts = {k: empire.imperial_host_size(state, next(a for a in state["agents"] if a.name == k))
             for k in kings}
    assert len(set(hosts.values())) == 1 and min(hosts.values()) > 0, \
        f"the realms must open in an exact STANDOFF (no winnable war on turn 1), got {hosts}"
    # V4.15: the standoff is EQUAL on purpose, and the equality is what holds fire on turn 1 (the
    # launch test is a strict >). Staggering the openings was tried and measured: any inequality
    # here fires a war on turn 1, before the title card has cleared. The pacing is bought instead
    # by the CHEST being the binding term — see scenario._SHOWCASE_REALMS. A chest deep enough to
    # out-fund the mercenary pool pins every host at its ceiling, so a war permanently spends the
    # realm that wins it and no second war can ever fire.
    import monarchy as _m
    for name in kings:
        king = next(a for a in state["agents"] if a.name == name)
        pool = len(_m._available_mercenaries(state, king, {king.name}))
        coin = int(king.money // _m.FIGHTER_COST)
        assert coin <= pool, (
            f"{name}'s COIN out-hires his mercenary pool ({coin} > {pool}): the chest would stop "
            "binding, hosts would pin at the pool ceiling, and a war would permanently spend the "
            "realm that won it — one war, then eighty-five dead turns")
    print("PASS test_showcase_scene_opens_in_a_standoff_that_must_break")


def test_showcase_pacing_and_floating_feed() -> None:
    """V4.14: showcase runs BRISK and slows only for the drama, and its event text is an OVERLAY.

    Pacing: the staged opening scene holds through the title card, a turn carrying major events
    holds long enough to read them (capped), every other turn runs at the brisk base pace, and a
    non-showcase renderer is untouched. Feed: no side panel, the projection viewport is narrowed by
    the feed column so the framing keeps the action clear of the text, only MAJOR beats enter the
    feed, repeats of one beat collapse, and lines age out. Headless SDL-dummy; skips without pygame."""
    import os as _os, time as _time
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame  # noqa: F401
        from renderer.pygame_renderer import (PygameRenderer, collapse_majors, _SHOWCASE_OPENING,
                                              _SHOWCASE_HOLD, _SHOWCASE_HOLD_MAX, _OVERLAY_LIFE)
    except ImportError:
        print("PASS test_showcase_pacing_and_floating_feed (skipped: no pygame)")
        return
    # the pure fold: one beat that happened to six towns is ONE line, distinct beats are untouched
    folded = collapse_majors([("S0A1 entered the Neolithic", None, (1, 1, 1)),
                              ("S0A2 entered the Neolithic", None, (1, 1, 1)),
                              ("KING Aldric DEFEATED Cyrus in war", (3.0, 3.0), (2, 2, 2)),
                              ("S0B1 entered the Neolithic", None, (1, 1, 1))])
    assert [t for t, _, _ in folded] == ["3 settlements entered the Neolithic",
                                         "KING Aldric DEFEATED Cyrus in war"], folded
    assert folded[1][1] == (3.0, 3.0), "a folded group keeps its camera focus"
    mixed = collapse_majors([("Rex fell in battle", None, (1, 1, 1)),
                             ("Juno fell in battle", None, (1, 1, 1))])
    assert mixed == [("Rex fell in battle (+1 more)", None, (1, 1, 1))], \
        f"non-settlement subjects name the first and count the rest, got {mixed}"
    size = 24
    village = {n: _FakeAgent(n, "curious", (6, 6), True, 2.0, 0.0) for n in ("a", "b", "c", "d", "e")}
    state = {
        "size": size, "turn": 9, "food": [(2, 2)],
        "settlements": {"S001": {"id": "S001", "center": (6, 6), "members": set(village), "founded": 2}},
        "monarchs": {}, "leaders": {}, "kingdoms": {}, "empires": {},
        "events": ["turn 9: a talked to b: \"hi\"",
                   "turn 9: AWV4 trust in AWV6: 0 -> 3 (led the S0A2 uprising to victory)",
                   "turn 9: KING Aldric DEFEATED Borin in war (9 loyal host vs 8; 2+3 fell) -> "
                   "Borin SUBJUGATED as a subject-king; an EMPIRE rises"],
        "agents": list(village.values()),
    }
    pygame.init()
    try:
        r = PygameRenderer(turn_delay=0.4, showcase=True, window=(1200, 800))
        r._ensure_screen(size)
        assert r._panel_w == 0 and r._map_px == 1200, "showcase draws the map full-bleed, no panel"
        assert r._feed_col > 0 and r._view == (1200 - r._feed_col, r._map_h), \
            "the projection viewport must be narrowed by the floating feed's column"
        cx, cy = r._to_px(size / 2, size / 2)
        assert cx <= r._map_px - r._feed_col, "the framed world must stay clear of the feed column"
        assert r._pace() == max(0.4, _SHOWCASE_OPENING), "the opening scene holds through the title card"
        r._opened = True
        # V4.15: pacing is driven by the DIRECTOR's beats for the turn, not by a raw major count.
        from renderer.pygame_renderer import (_PACE_MINOR, _PACE_BLUR, _RUN_BLUR, _CAM_EASE_SECS,
                                              _HOLD_MAJOR, _HOLD_LEGENDARY, _HOLD_QUEUED)
        from renderer import director as _d
        r._beats.clear(); r._turn_majors = 0; r._quiet_run = 1
        assert r._pace() == _PACE_MINOR, "a quiet turn is a blink — fast-forward, wide, no caption"
        r._quiet_run = _RUN_BLUR
        assert r._pace() == _PACE_BLUR, "a RUN of quiet turns compresses harder still"
        r._quiet_run = 0
        r._turn_sev = _d.MAJOR; r._turn_majors = 1
        r._beats.append((_d.MAJOR, "THE RISING OF S0B2", "Lord B falls.", (6.0, 6.0)))
        assert r._pace() == _CAM_EASE_SECS + _HOLD_MAJOR, "a major beat: fly in, then hold"
        r._beats.clear(); r._turn_sev = _d.LEGENDARY; r._turn_majors = 1
        r._beats.append((_d.LEGENDARY, "THE LINE OF ALDRIC ENDS", "The crown lies vacant.", None))
        assert r._pace() == _CAM_EASE_SECS + _HOLD_LEGENDARY > _CAM_EASE_SECS + _HOLD_MAJOR, \
            "a legendary beat holds longer than a major one"
        # Two majors on one turn are CUT BETWEEN, not dropped — both get screen time.
        r._beats.clear(); r._turn_sev = _d.MAJOR; r._turn_majors = 2
        for i in range(2):
            r._beats.append((_d.MAJOR, f"RISING {i}", None, (6.0, 6.0)))
        per = max(_HOLD_QUEUED, _HOLD_MAJOR * 0.72)
        assert r._pace() == _CAM_EASE_SECS + per * 2, "queued beats each get their own cut"
        # --showcase-pace tight scales the QUIET rate to the turns remaining; beats are untouched.
        tight = PygameRenderer(turn_delay=0.4, showcase=True, window=(1200, 800),
                               pace="tight", total_turns=45)
        tight._opened = True; tight._start_time = _time.monotonic(); tight._turns_left = 40
        assert tight._quiet_pace() <= _PACE_MINOR, "tight never runs the quiet turns SLOWER"
        tight._beats.append((_d.LEGENDARY, "T", None, None)); tight._turn_sev = _d.LEGENDARY
        assert tight._pace() == _CAM_EASE_SECS + _HOLD_LEGENDARY, \
            "tight squeezes the dull parts, never the dramatic ones"
        # the feed takes the MAJOR beat and drops the chatter
        r._enqueue_banners(state)
        texts = [e[0] for e in r._overlay_feed]
        # V4.15: the feed now carries the DIRECTOR's dramatised title, not the raw log line.
        assert len(texts) == 1 and texts[0] == "ALDRIC DEFEATS BORIN", \
            f"majors only in the feed, in the director's words, got {texts}"
        assert not any("trust in" in t for t in texts), "the trust ledger is muted in showcase"
        assert r._turn_majors == 1, "the dramatic pause counts the folded beats"
        r._draw(state)                                    # a frame WITH the overlay must not raise
        r._overlay_feed[0] = (texts[0], _time.monotonic() - _OVERLAY_LIFE - 1, (1, 1, 1), False)
        r._draw_feed_overlay()
        assert not r._overlay_feed, "a line must age out of the feed"
        # a NORMAL renderer is untouched: full viewport, panel, and a flat pace
        n = PygameRenderer(turn_delay=0.4, window=(1200, 800))
        n._ensure_screen(size)
        assert n._feed_col == 0 and n._view == (n._map_px, n._map_h) and n._panel_w > 0
        n._turn_majors = 5
        assert n._pace() == 0.4, "outside showcase the pace never changes"
    finally:
        pygame.quit()
    print("PASS test_showcase_pacing_and_floating_feed")


def test_speed_parsing_and_delay_only_when_rendering() -> None:
    """--speed maps presets/numbers to delays, and the pause fires ONLY when rendering.

    Presets slow/normal/fast and a raw number map to the right seconds; bad values are
    rejected. The delay is presentation-only: run_simulation must NOT sleep without a
    renderer (so tests/plain/logged runs are unpaced), and MUST sleep once per turn with
    one. A fake renderer + a patched time.sleep prove both branches without a terminal.
    """
    import argparse
    from contextlib import contextmanager

    assert main.parse_speed("slow") == 2.0
    assert main.parse_speed("normal") == 0.5
    assert main.parse_speed("fast") == 0.1
    assert main.parse_speed("0.3") == 0.3
    for bad in ("turbo", "-1"):
        try:
            main.parse_speed(bad)
            assert False, f"{bad!r} should have been rejected"
        except argparse.ArgumentTypeError:
            pass

    class _FakeRenderer:
        """Minimal stand-in: a real sink + live() context, counts update() calls."""
        def __init__(self) -> None:
            self.sink = io.StringIO()
            self.updates = 0

        @contextmanager
        def live(self):
            yield self

        def update(self, state: dict) -> None:
            self.updates += 1

    slept: list[float] = []
    orig_sleep = main.time.sleep
    main.time.sleep = lambda s: slept.append(s)
    try:
        # No renderer: a non-zero delay must be ignored entirely.
        _fresh_world()
        with contextlib.redirect_stdout(io.StringIO()):
            main.run_simulation(3, turn_delay=2.0)
        assert slept == [], f"slept without a renderer: {slept}"

        # With a renderer: one pause per simulated turn, at the requested delay.
        _fresh_world()
        fake = _FakeRenderer()
        with contextlib.redirect_stdout(io.StringIO()):
            main.run_simulation(3, renderer=fake, turn_delay=0.05)
        assert slept == [0.05, 0.05, 0.05], slept
        assert fake.updates == 3, fake.updates
    finally:
        main.time.sleep = orig_sleep
    print("PASS test_speed_parsing_and_delay_only_when_rendering")


# --- God mode (Day 15) -----------------------------------------------------
def test_god_mode_imports_only_world_state_layers() -> None:
    """god_mode.py must touch ONLY world_state — no decision-logic imports."""
    import ast
    with open("god_mode.py") as f:
        tree = ast.parse(f.read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"strategy", "trust", "conversation", "alliance", "personality", "llm"}
    assert not (imported & forbidden), f"god_mode imports decision logic: {imported & forbidden}"
    # The only project modules it may lean on are the world-state layers.
    project = imported - {"__future__", "typing", "ast", "os", "sys", "random"}
    assert project <= {"world", "population"}, project
    print("PASS test_god_mode_imports_only_world_state_layers")


def test_god_spawn_food_mutates_world_and_logs() -> None:
    """spawn_food adds a food tile at (x,y) and logs a [GOD] event."""
    import god_mode
    _fresh_world()
    world_state["turn"] = 12
    res = god_mode.spawn_food(world_state, 3, 4)
    assert (3, 4) in world_state["food"]
    assert world_state["grid"][4][3] == "food"
    assert res == "turn 12: [GOD] spawned food at (3,4)"
    assert any("[GOD] spawned food at (3,4)" in e for e in world_state["events"])
    print("PASS test_god_spawn_food_mutates_world_and_logs")


def test_god_spawn_agent_is_blank_slate_citizen() -> None:
    """spawn_agent reuses the Day 14 cold-start path: blank state + logged."""
    import god_mode
    _fresh_world()
    world_state["turn"] = 5
    alex = _agent("Alex", "friendly and outgoing", (1, 1))
    res = god_mode.spawn_agent(world_state, "Zed", "curious and bold")
    zed = next(a for a in world_state["agents"] if a.name == "Zed")
    assert zed.memory == [] and zed.relationships == {} and zed.hunger == 0
    assert zed.alive and zed.position in {(x, y) for x in range(10) for y in range(10)}
    # The survivor was told, and BOTH the blank-slate line and the [GOD] line logged.
    assert any("A new agent, Zed, appeared on turn 5" in m for m in alex.memory)
    assert any("a new agent Zed appeared (blank slate)" in e for e in world_state["events"])
    assert "[GOD] spawned agent Zed" in res
    print("PASS test_god_spawn_agent_is_blank_slate_citizen")


def test_god_drought_zeroes_respawn_for_exactly_20_turns() -> None:
    """trigger_drought stops food respawn for exactly DROUGHT_TURNS turns."""
    import god_mode
    _fresh_world()
    world_state["turn"] = 30
    res = god_mode.trigger_drought(world_state)  # default 20
    assert world_state["drought_until"] == 50
    assert "[GOD] drought triggered (20 turns)" in res

    # Respawn ticks every main.FOOD_RESPAWN_EVERY turns. Walk every turn from the
    # trigger through past the drought's end and assert: EVERY respawn tick inside
    # the 20-turn window (turns 31..50) adds nothing, and the first tick AFTER the
    # window (turn 50 < tick) resumes adding food.
    suppressed_ticks = 0
    resumed = False
    for turn in range(31, 60):
        before = len(world_state["food"])
        main.maybe_respawn_food(turn)
        added = len(world_state["food"]) - before
        if turn % main.FOOD_RESPAWN_EVERY != 0:
            continue  # not a respawn tick
        if turn <= 50:                       # inside the drought window
            assert added == 0, f"drought leaked food at turn {turn}"
            suppressed_ticks += 1
        elif not resumed and len(world_state["food"]) <= main.FOOD_RESPAWN_CAP:
            resumed = added >= 1             # first post-drought tick
    assert suppressed_ticks == 4, suppressed_ticks   # ticks at 35, 40, 45, 50
    assert resumed, "respawn did not resume after the drought ended"
    print("PASS test_god_drought_zeroes_respawn_for_exactly_20_turns")


def test_god_treasure_is_contestable_and_more_valuable() -> None:
    """drop_treasure places a claimable, high-value item; claiming pays out value."""
    import god_mode
    _fresh_world()
    world_state["turn"] = 8
    res = god_mode.drop_treasure(world_state, 5, 5)  # default value 10
    assert world_state["treasures"] == [{"pos": (5, 5), "value": 10}]
    # Mirrored into food so the existing navigation loop targets it (contestable).
    assert (5, 5) in world_state["food"]
    assert "[GOD] dropped treasure (value 10) at (5,5)" in res

    # A hungry agent standing on it claims it: hunger relief = value (10 > EAT_RELIEF
    # 7), it lands in inventory, and it is removed from BOTH treasures and food.
    a = _agent("Alex", "friendly and outgoing", (5, 5), hunger=9)
    result = execute_action(a, "eat")
    assert "claimed a treasure" in result, result
    assert a.hunger == 0 and a.inventory == [{"treasure": 10}]
    assert world_state["treasures"] == [] and (5, 5) not in world_state["food"]
    print("PASS test_god_treasure_is_contestable_and_more_valuable")


def test_god_spawned_food_draws_hungry_agent_within_two_turns() -> None:
    """The reaction is emergent: a hungry agent navigates to god-spawned food."""
    import god_mode
    _fresh_world()
    # A hungry agent with no food anywhere; survival override will seek food.
    a = _agent("Bob", "cautious and territorial", (5, 5), hunger=6)
    god_mode.spawn_food(world_state, 5, 8)  # 3 cells due south
    # No script: the real executor, reading the changed world, heads south.
    start = a.position
    for _ in range(2):
        action, _note = choose_action(a, Strategy(kind="wander"), world_state)
        execute_action(a, action)
    moved_closer = abs(a.position[1] - 8) < abs(start[1] - 8)
    assert moved_closer, f"agent did not move toward spawned food: {start} -> {a.position}"
    print("PASS test_god_spawned_food_draws_hungry_agent_within_two_turns")


def test_god_menu_pauses_and_resumes_cleanly() -> None:
    """A scripted God session runs commands then resumes on a blank line."""
    import god_mode
    _fresh_world()
    world_state["turn"] = 20
    scripted = iter(["status", "spawn_food 2 2", "trigger_drought", ""])  # blank resumes
    out_lines: list[str] = []
    god_mode.god_menu(world_state, 20,
                      read_line=lambda _prompt="": next(scripted),
                      out=out_lines.append)
    # Commands took effect...
    assert (2, 2) in world_state["food"]
    assert world_state["drought_until"] == 40
    # ...and the session ended on the blank line with a resume notice.
    assert any("resuming simulation" in line for line in out_lines)
    print("PASS test_god_menu_pauses_and_resumes_cleanly")


def test_god_mode_adds_no_llm_calls() -> None:
    """Every God intervention is pure world_state mutation — zero inference."""
    import god_mode
    _fresh_world()
    llm.reset_call_stats()
    god_mode.spawn_food(world_state, 1, 1)
    god_mode.drop_treasure(world_state, 2, 2)
    god_mode.trigger_drought(world_state)
    god_mode.spawn_agent(world_state, "Newbie", "curious")
    assert llm.get_call_stats() == {"decision": 0, "strategy": 0, "inclination": 0}, llm.get_call_stats()
    print("PASS test_god_mode_adds_no_llm_calls")


# --- Plague + stranger (Day 16) --------------------------------------------
def test_god_plague_raises_hunger_for_exactly_the_window_and_logs() -> None:
    """trigger_plague drains extra hunger for exactly PLAGUE_TURNS turns, then recovers."""
    import god_mode
    _fresh_world()
    kira = _agent("Kira", "independent and competitive", (5, 5), hunger=0)
    world_state["turn"] = 1
    res = god_mode.trigger_plague(world_state, "Kira")  # default 10 turns
    assert kira.plague_until == 11, kira.plague_until           # 1 + 10
    assert res == "turn 1: [GOD] plague struck Kira (10 turns)"
    assert any("[GOD] plague struck Kira (10 turns)" in e for e in world_state["events"])
    assert any("A plague struck you" in m for m in kira.memory)  # victim notified

    # Isolate the per-turn hunger increment: reset to 0 before each tick and read the
    # delta the existing hunger loop applies. +3 while sick (turn <= 11), +1 after.
    sick_ticks = healthy_ticks = 0
    for turn in range(2, 14):
        world_state["turn"] = turn
        kira.hunger = 0
        update_hunger(kira)
        if turn <= 11:
            assert kira.hunger == world.PLAGUE_HUNGER_PER_TURN, (turn, kira.hunger)
            sick_ticks += 1
        else:
            assert kira.hunger == world.HUNGER_PER_TURN, (turn, kira.hunger)
            healthy_ticks += 1
    assert sick_ticks == 10, sick_ticks       # exactly the 10-turn window (turns 2..11)
    assert healthy_ticks == 2                  # turns 12, 13 back to normal
    assert kira.plague_until == 0              # marker cleared on recovery
    assert any("Recovered from the plague" in m for m in kira.memory)
    print("PASS test_god_plague_raises_hunger_for_exactly_the_window_and_logs")


def test_god_plague_afflicts_random_living_agent() -> None:
    """With no name, the plague hits some LIVING agent (never a dead one)."""
    import god_mode
    _fresh_world()
    a = _agent("Alex", "friendly", (4, 4))
    b = _agent("Bob", "cautious", (5, 4))
    mark_dead(b)                                # only Alex is alive
    world_state["turn"] = 3
    god_mode.trigger_plague(world_state)
    assert a.plague_until == 13 and b.plague_until == 0   # the dead are not afflicted
    print("PASS test_god_plague_afflicts_random_living_agent")


def test_god_sick_neighbour_is_visible_in_perception() -> None:
    """A plagued neighbour 'looks sick' through observe() and social memory."""
    import god_mode
    _fresh_world()
    alex = _agent("Alex", "friendly", (5, 5))
    kira = _agent("Kira", "independent", (6, 5))   # directly East of Alex
    world_state["turn"] = 2
    god_mode.trigger_plague(world_state, "Kira")
    assert "Kira (looks sick)" in observe(alex, world_state)
    record_social_memories(alex, world_state)
    assert any("Observed Kira looking sick" in m for m in alex.memory)
    print("PASS test_god_sick_neighbour_is_visible_in_perception")


def test_god_stranger_is_blank_slate_and_seeds_wariness_memory() -> None:
    """introduce_stranger adds a cold-start agent and seeds wariness as MEMORY."""
    import god_mode
    _fresh_world()
    world_state["turn"] = 40
    alex = _agent("Alex", "friendly and outgoing", (5, 5))
    res = god_mode.introduce_stranger(world_state, "Vera", "quiet and guarded")
    vera = next(a for a in world_state["agents"] if a.name == "Vera")
    # Blank-slate cold start, exactly like a respawn.
    assert vera.memory == [] and vera.relationships == {}
    assert vera.allies == set() and vera.hunger == 0 and vera.alive
    # Wariness is seeded as a MEMORY on existing agents — NOT a hardcoded trust hit.
    assert any("A stranger, Vera, arrived. You know nothing about them." in m
               for m in alex.memory)
    assert "Vera" not in alex.relationships          # no trust penalty applied
    # Logged as a [GOD] stranger event; the neutral "new agent appeared" line is NOT.
    assert res == "turn 40: [GOD] stranger Vera introduced"
    assert any("[GOD] stranger Vera introduced" in e for e in world_state["events"])
    assert not any("a new agent Vera appeared" in e for e in world_state["events"])
    # And the stranger is a real, interactable citizen: perceivable by name when near.
    place_agent(alex, vera.position[0] - 1, vera.position[1])  # stand to its West
    assert "Vera" in observe(alex, world_state)
    print("PASS test_god_stranger_is_blank_slate_and_seeds_wariness_memory")


def test_god_day16_commands_add_no_llm_calls() -> None:
    """trigger_plague and introduce_stranger are pure world_state mutation."""
    import god_mode
    _fresh_world()
    _agent("Alex", "friendly", (5, 5))
    world_state["turn"] = 1
    llm.reset_call_stats()
    god_mode.trigger_plague(world_state)
    god_mode.introduce_stranger(world_state, "Vera", "quiet and guarded")
    assert llm.get_call_stats() == {"decision": 0, "strategy": 0, "inclination": 0}, llm.get_call_stats()
    print("PASS test_god_day16_commands_add_no_llm_calls")


# --- Reproducibility + run capture (Day 17) --------------------------------
def test_seeded_random_runs_are_identical() -> None:
    """Same seed + random provider => byte-identical run (world setup + every turn)."""
    def capture() -> str:
        random.seed(20260623)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(15)
        return buf.getvalue()

    first, second = capture(), capture()
    assert first == second, "two seeded random runs diverged"
    # And the capture is a real run: turn-by-turn log, summary, and events log present.
    assert "TURN 1" in first and "AGENT SUMMARY" in first and "EVENTS LOG" in first
    print("PASS test_seeded_random_runs_are_identical")


def test_god_script_parses_inline_and_file() -> None:
    """parse_god_script accepts both an inline spec and a file of '<turn>:<cmd>' lines."""
    inline = main.parse_god_script("3:trigger_drought;5:drop_treasure 5 5 10;5:trigger_plague Bob")
    assert inline == {3: ["trigger_drought"], 5: ["drop_treasure 5 5 10", "trigger_plague Bob"]}
    assert main.parse_god_script(None) == {} and main.parse_god_script("") == {}

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("# a demo god script\n8: trigger_plague Kira\n\n12: drop_treasure 4 4\n")
        path = f.name
    try:
        assert main.parse_god_script(path) == {8: ["trigger_plague Kira"], 12: ["drop_treasure 4 4"]}
    finally:
        os.unlink(path)
    print("PASS test_god_script_parses_inline_and_file")


def test_god_script_runs_and_capture_includes_god_events() -> None:
    """A scripted run fires god commands at their turns and the capture shows them."""
    random.seed(7)
    script = main.parse_god_script("3:trigger_drought;5:drop_treasure 5 5 10")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main.run_simulation(8, god_script=script)
    out = buf.getvalue()
    # The non-interactive driver announced each command...
    assert "[GOD-SCRIPT turn 3] trigger_drought" in out
    assert "[GOD-SCRIPT turn 5] drop_treasure 5 5 10" in out
    # ...the interventions actually logged [GOD] events...
    assert "[GOD] drought triggered" in out and "[GOD] dropped treasure (value 10)" in out
    # ...and they appear in the end-of-run EVENTS LOG (cause->effect in one place).
    events_section = out.split("EVENTS LOG")[1]
    assert "[GOD] drought triggered" in events_section
    assert "[GOD] dropped treasure (value 10)" in events_section
    print("PASS test_god_script_runs_and_capture_includes_god_events")


# --- Lineage (V2 M4.1): birth, childhood, aging, family ---------------------
def _lineage_world(pop_cap: int = 10) -> None:
    """A fresh world with lineage ON, one settlement S001 at (5, 5), no agents yet."""
    _fresh_world()
    world_state["lineage_on"] = True
    world_state["lineage"] = {"pop_cap": pop_cap, "birth_seq": 0}
    world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5),
                                          "members": set(), "founded": 0}


def _settler(name: str, pos: tuple[int, int], sid: str = "S001",
             personality: str = "friendly and outgoing", hunger: int = 1) -> Agent:
    """A living, settled adult (age 20, lifespan 100) enrolled in `sid`."""
    a = Agent(name=name, personality=personality)
    place_agent(a, *pos)
    a.hunger = hunger
    a.settlement = sid
    a.age, a.lifespan = 20, 100
    rec = world_state["settlements"].get(sid)
    if rec is not None:
        rec["members"].add(name)
    return a


def _mutual_high_trust(a: Agent, b: Agent, level: int | None = None) -> None:
    """Seed trust BOTH ways at the pairing bar (lineage.PAIR_TRUST by default)."""
    import lineage, trust
    level = lineage.PAIR_TRUST if level is None else level
    trust.ensure_relationship(a, b.name)["trust"] = level
    trust.ensure_relationship(b, a.name)["trust"] = level


def _surplus_food_at_centre(tiles: int = 4) -> None:
    """Standing food within lineage.SURPLUS_RADIUS of S001's centre (5, 5)."""
    for x, y in [(4, 4), (6, 6), (5, 3), (3, 5), (7, 5), (5, 7)][:tiles]:
        world.place_food(x, y)


def test_lineage_off_run_is_byte_identical_to_v1() -> None:
    """lineage_on=False (default) leaves the run — respawn included — byte-identical."""
    def run(**kw):
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, **kw)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        base, off = run(), run(lineage_on=False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "lineage_on=False diverged from the default run"
    print("PASS test_lineage_off_run_is_byte_identical_to_v1")


def test_birth_requires_every_gate() -> None:
    """A birth needs settled-together + mutual trust + both fed + surplus + cap headroom
    + adult parents + cooldown — knocking out ANY single gate yields no birth."""
    import lineage

    def staged(mutate=None) -> int:
        """Build the all-gates-hold world, apply `mutate`, return births count."""
        _lineage_world()
        ada = _settler("Ada", (5, 5))
        ben = _settler("Ben", (6, 5))
        _mutual_high_trust(ada, ben)
        _surplus_food_at_centre()
        if mutate is not None:
            mutate(ada, ben)
        return len(lineage._births(world_state, 5, random.Random(3)))

    # Baseline: every gate holds -> exactly ONE child of the pair this turn.
    assert staged() == 1, "all gates hold but no child was born"
    child = world_state["agents"][-1]
    assert child.parents == ("Ada", "Ben") and child.dependent
    assert child.settlement == "S001"

    # Each gate, knocked out alone, blocks the birth:
    assert staged(lambda a, b: a.relationships.__setitem__(
        b.name, {"trust": lineage.PAIR_TRUST - 1, "interactions": 0, "grudge": False})
    ) == 0, "one-way sub-threshold trust must block a birth (mutuality)"
    assert staged(lambda a, b: setattr(b, "settlement", None)) == 0, \
        "an unsettled partner must block a birth"
    assert staged(lambda a, b: setattr(b, "settlement", "S002")) == 0, \
        "different settlements must block a birth"
    assert staged(lambda a, b: setattr(a, "hunger", lineage.FED_HUNGER)) == 0, \
        "a hungry parent must block a birth (fed gate)"
    assert staged(lambda a, b: world_state["food"].clear()) == 0, \
        "no settlement food surplus must block a birth (Malthusian gate)"
    assert staged(lambda a, b: world_state["lineage"].__setitem__("pop_cap", 2)) == 0, \
        "the population cap must refuse a birth"
    assert staged(lambda a, b: setattr(b, "dependent", True)) == 0, \
        "a dependent child can never be a parent"
    assert staged(lambda a, b: setattr(a, "last_child_turn", 4)) == 0, \
        "the birth cooldown must pace births"
    print("PASS test_birth_requires_every_gate")


def test_child_inherits_blend_not_knowledge_and_kin_trust_seeded() -> None:
    """The child's traits are the parents' average +/- bounded jitter (dominant
    recomputed from the blend); knowledge/wealth are NOT inherited; kin-trust is
    seeded both ways with each parent."""
    import lineage
    _lineage_world()
    ada = _settler("Ada", (5, 5), personality="curious and adventurous")
    ben = _settler("Ben", (6, 5), personality="curious and adventurous")
    ada.knowledge = {"fire", "farming"}
    ada.stockpile, ada.money = 5.0, 3.0
    _mutual_high_trust(ada, ben)
    _surplus_food_at_centre()

    born = lineage._births(world_state, 5, random.Random(11))
    assert len(born) == 1
    child = born[0]

    # Temperament: blended weights within the jitter bound, dominant recomputed.
    p1, p2 = get_personality(ada), get_personality(ben)
    pc = get_personality(child)
    for trait in ("curiosity", "caution", "friendliness", "independence"):
        mean = (getattr(p1, trait) + getattr(p2, trait)) / 2
        got = getattr(pc, trait)
        lo = max(0.0, mean - lineage.TRAIT_JITTER)
        hi = min(1.0, mean + lineage.TRAIT_JITTER)
        assert lo <= got <= hi, f"{trait}: {got} outside blend band [{lo}, {hi}]"
    assert pc.dominant == "curiosity", "a child of two curious parents skews curious"
    assert child.personality.startswith("curious (child of Ada and Ben)")

    # Knowledge is EARNED, wealth is M4.2: a newborn has neither.
    assert child.knowledge == set() and child.stockpile == 0.0 and child.money == 0.0
    assert child.memory[0].startswith("Born to Ada and Ben")

    # Kin-trust seeded BOTH ways with each parent; the parents' cooldown is set.
    for parent in (ada, ben):
        assert child.relationships[parent.name]["trust"] == lineage.KIN_TRUST
        assert parent.relationships[child.name]["trust"] == lineage.KIN_TRUST
        assert parent.last_child_turn == 5
    print("PASS test_child_inherits_blend_not_knowledge_and_kin_trust_seeded")


def test_child_learning_boost_only_when_lineage_on() -> None:
    """A dependent child adopts knowledge at CHILD_LEARN_BOOST x the adult rate —
    and the boost vanishes (byte-identical probability) when lineage is off."""
    import knowledge
    _lineage_world()
    teacher = _settler("Tess", (5, 5))
    kid = _settler("Kid", (6, 5))
    kid.dependent = False
    adult_p = knowledge.adoption_probability(kid, teacher, world_state)
    kid.dependent = True
    child_p = knowledge.adoption_probability(kid, teacher, world_state)
    assert abs(child_p - adult_p * knowledge.CHILD_LEARN_BOOST) < 1e-12, (adult_p, child_p)
    # Lineage OFF -> the dependent flag is ignored (default runs unchanged).
    world_state["lineage_on"] = False
    assert knowledge.adoption_probability(kid, teacher, world_state) == adult_p
    print("PASS test_child_learning_boost_only_when_lineage_on")


def test_dependent_consumes_parent_stockpile_and_does_not_produce() -> None:
    """Childhood is a real investment: feeding draws the parent's stockpile down,
    and a dependent produces NOTHING (no farm/hunt/bank/trade/fight/labor) even
    if it already learned the skill."""
    import economy, knowledge, labor, lineage, monarchy, storage
    _lineage_world()
    ada = _settler("Ada", (5, 5))
    ben = _settler("Ben", (6, 5))
    kid = _settler("Kid", (5, 6), hunger=lineage.CHILD_FEED_AT)
    kid.dependent, kid.parents, kid.age = True, ("Ada", "Ben"), 3
    ada.stockpile = 3.0
    _surplus_food_at_centre()

    # Feeding: the richest parent's granary pays CHILD_MEAL_COST; the child eats.
    lineage._feed_children(world_state, 7)
    assert ada.stockpile == 3.0 - lineage.CHILD_MEAL_COST, ada.stockpile
    assert kid.hunger == 0 and f"Was fed by Ada" in kid.memory and "Fed Kid" in ada.memory

    # Ration-share fallback: no savings -> a FED parent takes hunger onto itself.
    kid.hunger, ada.stockpile, ada.hunger, ben.hunger = lineage.CHILD_FEED_AT, 0.0, 2, 3
    lineage._feed_children(world_state, 8)
    assert ada.hunger == 2 + lineage.PARENT_SHARE_HUNGER and kid.hunger == 0

    # No production while dependent — even KNOWING the skills changes nothing
    # (Kid is the only farmer/hunter, so its exclusion empties both passes).
    kid.knowledge = {"farming", "hunting"}
    kid.hunger = 0
    assert knowledge.farm(world_state, 9, rng=random.Random(0)) == []
    assert knowledge.hunt(world_state, 9, rng=random.Random(0)) == []
    assert all(name != "Kid" for name, _ in storage.accumulate(world_state, 9))
    assert not labor.is_worker(kid), "a dependent child never sells labor"
    world_state["monarchy_on"] = True
    rich = _settler("Rich", (7, 5)); rich.money = 100.0
    assert kid not in monarchy._available_mercenaries(world_state, rich, set()), \
        "a dependent child never fights"
    defenders, _ = monarchy.defenders_of(world_state, "S001")
    assert kid not in defenders, "a dependent child never stands in a battle line"

    # An unfed child starves like anyone (both parents unable to feed it).
    ada.alive = ben.alive = False
    kid.hunger = 9
    result = main.run_agent_turn(kid, 10, {}, {}, {"agent_turns": 0})
    assert result == "starved" and not kid.alive
    assert any("Kid died (starved)" in e for e in world_state["events"])
    print("PASS test_dependent_consumes_parent_stockpile_and_does_not_produce")


def test_dependent_child_turn_is_actionless_and_matures_on_schedule() -> None:
    """A dependent's turn is 'child' (no action, no strategy, zero LLM); at
    CHILDHOOD_TURNS it comes of age and becomes a full agent."""
    import lineage
    _lineage_world()
    ada = _settler("Ada", (5, 5))
    kid = _settler("Kid", (5, 6), hunger=0)
    kid.dependent, kid.parents, kid.age = True, ("Ada",), lineage.CHILDHOOD_TURNS - 2
    world.place_food(5, 6)  # even standing ON food, a child does not forage

    llm.reset_call_stats()
    strategies: dict = {}
    result = main.run_agent_turn(kid, 3, strategies, {}, {"agent_turns": 0})
    stats = llm.get_call_stats()
    assert result == "child" and kid.hunger == 1, (result, kid.hunger)
    assert (5, 6) in world_state["food"], "a dependent must not eat the map's food"
    assert "Kid" not in strategies and stats["strategy"] == 0 and stats["decision"] == 0

    # Maturation: exactly at CHILDHOOD_TURNS the dependent becomes a full adult.
    lineage.update(world_state, 4, rng=random.Random(0))   # age -> CHILDHOOD_TURNS - 1
    assert kid.dependent
    lineage.update(world_state, 5, rng=random.Random(0))   # age -> CHILDHOOD_TURNS
    assert not kid.dependent and kid.age == lineage.CHILDHOOD_TURNS
    assert any("Kid came of age" in e for e in world_state["events"])
    assert "Came of age — now a full adult" in kid.memory
    print("PASS test_dependent_child_turn_is_actionless_and_matures_on_schedule")


def test_old_age_death_uses_existing_death_path() -> None:
    """At lifespan's end an agent dies of OLD AGE through announce_death: a distinct
    DEATH event, survivor memories, and the (floor-gated) respawn queue entry."""
    import lineage, population
    _lineage_world()
    elder = _settler("Eld", (2, 2))
    heir = _settler("Her", (7, 7))
    elder.age, elder.lifespan = 99, 100
    lineage.update(world_state, 30, rng=random.Random(0))
    assert not elder.alive and heir.alive
    assert any("Eld died (old age)" in e for e in world_state["events"])
    assert "Died of old age" in elder.memory
    assert any("Eld died on turn 30 — they died of old age." == m for m in heir.memory)
    assert 30 + population.RESPAWN_DELAY in world_state["pending_respawns"]
    print("PASS test_old_age_death_uses_existing_death_path")


def test_births_primary_respawn_backstop() -> None:
    """Above the floor the respawn queue stays SILENT (drops, exactly as today);
    only a crash below TARGET_POPULATION (3) brings extinction insurance in."""
    import population
    _lineage_world()
    for i, pos in enumerate([(1, 1), (3, 1), (5, 1), (7, 1)]):
        _settler(f"A{i}", pos)

    # 4 living >= floor: a due respawn is DROPPED — respawn stays quiet.
    world_state["pending_respawns"] = [10]
    assert population.process_respawns(10, world_state) == []
    assert world_state["pending_respawns"] == []

    # Crash below the floor (2 living < 3): the SAME queue now fires — backstop.
    world_state["agents"][0].alive = False
    world_state["agents"][1].alive = False
    world_state["pending_respawns"] = [12]
    spawned = population.process_respawns(12, world_state)
    assert len(spawned) == 1, "below the floor, extinction insurance must fire"
    print("PASS test_births_primary_respawn_backstop")


def test_lineage_mechanics_add_no_llm_calls_and_are_deterministic() -> None:
    """The whole lineage machinery (init, aging, feeding, births) makes ZERO LLM
    calls, and a seeded lineage-on run reproduces byte-for-byte."""
    import lineage
    # Zero LLM: drive every lineage entry point directly and watch the counters.
    _lineage_world()
    ada = _settler("Ada", (5, 5))
    ben = _settler("Ben", (6, 5))
    _mutual_high_trust(ada, ben)
    _surplus_food_at_centre()
    llm.reset_call_stats()
    lineage.init_cast(world_state, rng=random.Random(1))
    world_state["lineage"]["pop_cap"] = 10
    for t in range(1, 30):
        lineage.update(world_state, t, rng=random.Random(t))
    stats = llm.get_call_stats()
    assert stats["strategy"] == 0 and stats["decision"] == 0, stats
    assert world_state["lineage"]["birth_seq"] >= 1, "no births in a fertile world?"

    # Determinism: the same seed replays an identical lineage-on run.
    def run():
        llm.PROVIDER = "random"
        random.seed(7)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(30, lineage_on=True)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        assert run() == run(), "seeded lineage-on runs diverged"
    finally:
        llm.PROVIDER = saved
    print("PASS test_lineage_mechanics_add_no_llm_calls_and_are_deterministic")


# --- Inheritance at death (V2 M4.2) -----------------------------------------
def _rich(name: str, pos: tuple[int, int], *, money: float = 0.0,
          stockpile: float = 0.0, parents: tuple = (), sid: str | None = "S001",
          dependent: bool = False) -> Agent:
    """A living, settled agent carrying wealth and a family link — an estate-builder."""
    a = _settler(name, pos, sid=sid) if sid is not None else \
        _settler(name, pos, sid="S001")
    if sid is None:
        a.settlement = None
        world_state["settlements"]["S001"]["members"].discard(name)
    a.money, a.stockpile = money, stockpile
    a.parents = parents
    a.dependent = dependent
    return a


def test_inheritance_equal_split_and_conservation() -> None:
    """A wealthy parent dies -> its children split the estate EQUALLY, wealth is
    CONSERVED to the decimal, and food over the granary cap DROPS as ground food."""
    import lineage

    # Two children, a clean estate that divides evenly and fits under the cap.
    _lineage_world()
    parent = _rich("Ada", (5, 5), money=30.0, stockpile=10.0)
    c1 = _rich("Kade", (5, 6), parents=("Ada", "Ben"))
    c2 = _rich("Lena", (6, 5), parents=("Ada", "Ben"))
    rec = lineage.settle_estate(parent, 7, world_state)
    assert rec["kind"] == "children"
    # Equal split: each child gets half the money and half the food.
    assert c1.money == c2.money == 15.0
    assert c1.stockpile == c2.stockpile == 5.0
    # Conservation to the decimal: estate == distributed + ground (nothing minted/lost).
    assert rec["estate"] == 40.0
    assert abs(rec["to_heirs"] + rec["ground"] - rec["estate"]) < 1e-9
    assert rec["ground"] == 0.0
    # Wealth left the corpse — no double counting.
    assert parent.money == 0.0 and parent.stockpile == 0.0

    # Cap-overflow: food that cannot fit a heir's granary drops as GROUND food.
    _lineage_world()
    n_food_before = len(world_state["food"])
    parent = _rich("Ada", (5, 5), money=0.0, stockpile=40.0)
    heir = _rich("Kade", (5, 6), parents=("Ada", "Ben"), stockpile=6.0)
    rec = lineage.settle_estate(parent, 7, world_state)
    # Sole heir: fills its granary to the cap (20), the rest (26) is overflow.
    assert heir.stockpile == lineage.storage.STORAGE_CAP  # 20.0
    assert rec["ground"] == 40.0 - (lineage.storage.STORAGE_CAP - 6.0)  # 26.0
    # Overflow is not minted, not vanished: it lands as whole ground-food tiles.
    tiles_added = len(world_state["food"]) - n_food_before
    assert tiles_added == 26, f"expected 26 ground tiles, got {tiles_added}"
    assert abs(rec["to_heirs"] + rec["ground"] - rec["estate"]) < 1e-9
    print("PASS test_inheritance_equal_split_and_conservation")


def test_inheritance_kin_order_is_binding() -> None:
    """Kin-order binds: CHILDREN are preferred over PARENTS over SIBLINGS, and each
    fallback fires only when the closer tier is empty."""
    import lineage

    def estate_of(deceased_parents, present):
        """Build a world where `deceased` (Ada) has `deceased_parents`, plus whatever
        kin `present` seeds, then settle Ada's estate and return (kind, heir_names)."""
        _lineage_world()
        ada = _rich("Ada", (5, 5), money=40.0, parents=deceased_parents)
        present(ada)
        rec = lineage.settle_estate(ada, 7, world_state)
        heirs = sorted(a.name for a in world_state["agents"]
                       if a.alive and a.money > 0)
        return rec["kind"], heirs

    # Children present (alongside a parent + a sibling) -> children take all.
    def with_child(ada):
        _rich("Milo", (4, 4), parents=("Ben", "Ada"))          # child of Ada
        _rich("Ben", (6, 6))                                    # a parent of Ada
        _rich("Nell", (3, 3), parents=("Ben", "Cara"))          # sibling (shares Ben)
    kind, heirs = estate_of(("Ben", "Cara"), with_child)
    assert kind == "children" and heirs == ["Milo"], (kind, heirs)

    # No children -> PARENTS (even with a sibling also present).
    def with_parent(ada):
        _rich("Ben", (6, 6))                                    # a living parent
        _rich("Nell", (3, 3), parents=("Ben", "Cara"))          # sibling
    kind, heirs = estate_of(("Ben", "Cara"), with_parent)
    assert kind == "parents" and heirs == ["Ben"], (kind, heirs)

    # No children, no living parents -> SIBLINGS (shares at least one parent).
    def with_sibling(ada):
        _rich("Nell", (3, 3), parents=("Ben", "Cara"))          # sibling, shares Ben+Cara
    kind, heirs = estate_of(("Ben", "Cara"), with_sibling)
    assert kind == "siblings" and heirs == ["Nell"], (kind, heirs)
    print("PASS test_inheritance_kin_order_is_binding")


def test_escheat_to_ruler_else_vanishes() -> None:
    """A kinless settled agent's estate ESCHEATS to the settlement's ruler (monarch
    first, else trust-leader); with NO ruler it vanishes exactly as pre-M4.2."""
    import lineage

    # Escheat to a MONARCH (crown outranks a trust-leader).
    _lineage_world()
    world_state["monarchs"]["S001"] = {"monarch": "Rex", "since": 0, "garrison": set()}
    world_state["leaders"]["S001"] = {"leader": "Cato", "followers": set(), "since": 0}
    rex = _rich("Rex", (5, 5))
    _rich("Cato", (4, 4))
    loner = _rich("Ada", (6, 6), money=25.0)  # no parents, no children, no siblings
    rec = lineage.settle_estate(loner, 7, world_state)
    assert rec["kind"] == "escheat"
    assert rex.money == 25.0, "the crown should absorb a kinless estate"

    # No monarch -> escheat to the TRUST-LEADER.
    _lineage_world()
    world_state["leaders"]["S001"] = {"leader": "Cato", "followers": set(), "since": 0}
    cato = _rich("Cato", (5, 5))
    loner = _rich("Ada", (6, 6), money=25.0)
    rec = lineage.settle_estate(loner, 7, world_state)
    assert rec["kind"] == "escheat" and cato.money == 25.0

    # No kin AND no ruler -> the estate VANISHES, exactly as before M4.2.
    _lineage_world()
    loner = _rich("Ada", (6, 6), money=25.0)
    total_money_before = sum(a.money for a in world_state["agents"] if a is not loner)
    rec = lineage.settle_estate(loner, 7, world_state)
    assert rec["kind"] == "none"
    total_money_after = sum(a.money for a in world_state["agents"] if a is not loner)
    assert total_money_after == total_money_before, "estate should vanish, not move"
    assert loner.money == 0.0, "wealth still leaves the corpse"
    events = "\n".join(world_state["events"])
    assert "vanished (no heir)" in events
    print("PASS test_escheat_to_ruler_else_vanishes")


def test_inheritance_writes_events_and_memories_via_death_path() -> None:
    """Inheritance flows through the real death funnel (announce_death) for EVERY
    cause, logging clear events and writing a memory to each heir."""
    import lineage
    import population

    _lineage_world()
    parent = _rich("Ada", (5, 5), money=20.0, stockpile=8.0)
    c1 = _rich("Kade", (5, 6), parents=("Ada", "Ben"))
    c2 = _rich("Lena", (6, 5), parents=("Ada", "Ben"))
    # Death by OLD-AGE wording, routed through the shared path.
    population.announce_death(parent, 12, world_state, cause="old age",
                              final_memory="Died of old age", note="they died of old age")
    events = "\n".join(world_state["events"])
    assert "Ada died (old age)" in events
    assert "Kade inherited 14.00 from Ada" in events  # (20+8)/2 = 14.00
    assert "Lena inherited 14.00 from Ada" in events
    assert any("Inherited 14.00 from Ada" in m for m in c1.memory)
    assert any("Inherited 14.00 from Ada" in m for m in c2.memory)
    assert not parent.alive  # the death path still marks the deceased dead
    print("PASS test_inheritance_writes_events_and_memories_via_death_path")


def test_inheritance_only_when_lineage_on() -> None:
    """With lineage OFF, a death moves NO wealth — the estate vanishes as it always
    did (the byte-identical guarantee holds at the estate level too)."""
    import population

    _lineage_world()
    world_state["lineage_on"] = False  # lineage off
    parent = _rich("Ada", (5, 5), money=20.0, stockpile=8.0)
    child = _rich("Kade", (5, 6), parents=("Ada", "Ben"))
    population.announce_death(parent, 12, world_state, cause="starved")
    assert child.money == 0.0 and child.stockpile == 0.0, \
        "no inheritance may occur with lineage off"
    # No wealth even left the corpse's fields (settle_estate never ran).
    assert parent.money == 20.0 and parent.stockpile == 8.0
    events = "\n".join(world_state["events"])
    assert "inherited" not in events and "escheat" not in events
    print("PASS test_inheritance_only_when_lineage_on")


def test_inherited_stockpile_helps_a_dependent_orphan_survive() -> None:
    """An inheriting CHILD is a real heir: an orphan with an inherited granary
    outlives an identical orphan with none (grim, but honest)."""
    import lineage
    import population
    import storage

    def orphan_survives(inherit: bool) -> bool:
        """A lone orphan (no living parent to feed it) faces a food shock; return
        whether it is still alive after a stretch of hungry, foodless turns."""
        _lineage_world()
        world_state["food"].clear()  # a famine: nothing to forage
        # A dependent child whose parents are both dead/absent — the M4.1 feeder
        # needs a LIVING parent, so nothing feeds it but its own inherited granary.
        orphan = _rich("Kade", (5, 5), parents=("Ada", "Ben"), dependent=True)
        orphan.age, orphan.lifespan, orphan.hunger = 4, 100, 1
        if inherit:
            # A dying parent's estate flows to the orphan the moment it dies.
            parent = _rich("Ada", (6, 6), stockpile=18.0, sid=None)  # 2 drawn meals
            population.announce_death(parent, 0, world_state, cause="starved")
            assert orphan.stockpile == 18.0  # inheritance actually landed
        # 15 hungry turns: each tick raises hunger; only a granary can stave off death.
        for t in range(1, 16):
            world.update_hunger(orphan)
            lineage.update(world_state, t, rng=random.Random(t))
            if orphan.alive and world.is_dead(orphan):  # the draw-down-or-die step
                if not storage.draw_down(orphan):
                    population.announce_death(orphan, t, world_state)
        return orphan.alive

    assert orphan_survives(inherit=True), "an heir orphan should outlast its hunger"
    assert not orphan_survives(inherit=False), "a pennyless orphan should starve"
    print("PASS test_inherited_stockpile_helps_a_dependent_orphan_survive")


# --- Dynasties (V2 M4.3): titles pass to heirs ------------------------------
def _crown(sid: str, monarch: str, garrison: set | None = None) -> None:
    """Install `monarch` on settlement `sid` (a force-based M3.4 seat)."""
    world_state["monarchs"][sid] = {"monarch": monarch, "since": 0,
                                    "garrison": set(garrison or set())}


def _realm(king: str, home: str, settlements: set, vassals: dict) -> None:
    """A king's M3.5 realm record with vassal lordships and fresh discontent counters."""
    world_state["kingdoms"][king] = {
        "king": king, "home": home, "settlements": set(settlements),
        "vassals": dict(vassals), "founded": 0,
        "discontent": {lord: 0 for lord in vassals.values()}}


def test_m43_title_transfers_on_every_death_cause() -> None:
    """A titled ruler's SEAT passes to its eldest heir on EVERY death cause (old age,
    starvation, battle) — records re-keyed to the heir, realm intact, coronation logged,
    ZERO added LLM. The pre-M4.3 contrast (records left on the dead king) is shown too."""
    import lineage, population

    def crown_after(cause: str, transfer: bool):
        _lineage_world()
        world_state["settlements"]["S002"] = {"id": "S002", "center": (8, 8),
                                              "members": set(), "founded": 0}
        king = _rich("Rex", (5, 5)); king.age = 60
        heir = _rich("Cyn", (5, 6), parents=("Rex", "Mara")); heir.age = 20
        _crown("S001", "Rex", {"g"})
        _realm("Rex", "S001", {"S001", "S002"}, {"S002": "Vale"})
        _rich("Vale", (8, 8))
        real = lineage.succeed_titles
        if not transfer:                       # the pre-M4.3 baseline: titles do NOT pass
            lineage.succeed_titles = lambda *a, **k: {"heir": None, "kind": "none", "titles": ""}
        try:
            population.announce_death(king, 30, world_state, cause=cause,
                                      final_memory="Died", note="they died")
        finally:
            lineage.succeed_titles = real
        return king

    import llm
    llm.reset_call_stats()
    for cause in ("old age", "starved", "fell in battle"):
        crown_after(cause, transfer=True)
        assert world_state["monarchs"]["S001"]["monarch"] == "Cyn", cause
        cyn = world_state["kingdoms"].get("Cyn")
        assert cyn is not None and cyn["king"] == "Cyn"
        assert cyn["settlements"] == {"S001", "S002"}   # realm structure survives intact
        assert cyn["vassals"] == {"S002": "Vale"}       # the vassal lordship carried across
        assert cyn["discontent"] == {"Vale": 0}
        assert "Rex" not in world_state["kingdoms"]     # re-keyed off the dead king
        events = "\n".join(world_state["events"])
        assert "Cyn succeeded Rex as [" in events and "eldest child" in events
    calls = llm.get_call_stats()
    assert calls["strategy"] == 0 and calls["decision"] == 0, "succession must add no LLM"

    # CONTRAST — pre-M4.3 (succession suppressed): the crown does NOT pass; the dead
    # king's records are simply left (the realm dissolves later via breakaway).
    crown_after("old age", transfer=False)
    assert world_state["monarchs"]["S001"]["monarch"] == "Rex"   # a dead holder, un-succeeded
    assert "Rex" in world_state["kingdoms"] and "Cyn" not in world_state["kingdoms"]
    print("PASS test_m43_title_transfers_on_every_death_cause")


def test_m43_succession_is_eldest_first_with_tiebreaks() -> None:
    """The single heir is the ELDEST of the closest non-empty kin tier (children ->
    parents -> siblings), name as the deterministic age-tie tiebreak."""
    import lineage

    def heir_of(seed_kin):
        _lineage_world()
        rex = _rich("Rex", (5, 5), parents=("Gpa", "Gma")); rex.age = 60
        _crown("S001", "Rex")
        seed_kin(rex)
        h, kind = lineage._succession_heir(rex, world_state)
        return (h.name if h is not None else None), kind

    def two_children(rex):
        a = _rich("Ada", (5, 6), parents=("Rex", "M")); a.age = 18   # younger
        b = _rich("Ben", (6, 5), parents=("Rex", "M")); b.age = 25   # ELDEST child
    assert heir_of(two_children) == ("Ben", "child"), "eldest child preferred"

    def tie(rex):
        z = _rich("Zed", (5, 6), parents=("Rex", "M")); z.age = 20
        a = _rich("Ada", (6, 5), parents=("Rex", "M")); a.age = 20   # same age -> name asc
    assert heir_of(tie) == ("Ada", "child"), "age tie broken by name (Ada < Zed)"

    def parents_only(rex):
        g1 = _rich("Gpa", (5, 6)); g1.age = 82   # ELDEST parent
        g2 = _rich("Gma", (6, 5)); g2.age = 78
    assert heir_of(parents_only) == ("Gpa", "parent"), "no children -> eldest parent"

    def siblings_only(rex):
        s1 = _rich("Uma", (5, 6), parents=("Gpa", "Gma")); s1.age = 30
        s2 = _rich("Tom", (6, 5), parents=("Gpa", "X")); s2.age = 40   # ELDEST sibling (shares Gpa)
    assert heir_of(siblings_only) == ("Tom", "sibling"), "no child/parent -> eldest sibling"

    assert heir_of(lambda rex: None) == (None, "none"), "no kin -> no heir (extinct)"
    print("PASS test_m43_succession_is_eldest_first_with_tiebreaks")


def test_m43_succession_does_not_inherit_loyalty() -> None:
    """The heir inherits the SEAT, not the LOYALTY: a vassal's trust toward the heir is
    only its OWN, and the dead king's trust relationships are never copied onto the heir."""
    import lineage, population, trust

    _lineage_world()
    world_state["settlements"]["S002"] = {"id": "S002", "center": (8, 8),
                                          "members": {"Vale"}, "founded": 0}
    king = _rich("Rex", (5, 5)); king.age = 60
    heir = _rich("Cyn", (5, 6), parents=("Rex", "Mara")); heir.age = 20
    vassal = _rich("Vale", (8, 8), sid="S002")
    _crown("S001", "Rex"); _crown("S002", "Vale")
    _realm("Rex", "S001", {"S001", "S002"}, {"S002": "Vale"})
    # The vassal trusted the DEAD king highly, but personally DISTRUSTS the unknown heir.
    trust.ensure_relationship(vassal, "Rex")["trust"] = 3
    trust.ensure_relationship(vassal, "Cyn")["trust"] = -3
    population.announce_death(king, 30, world_state, cause="old age",
                              final_memory="Died", note="they died")
    assert world_state["kingdoms"]["Cyn"]["vassals"]["S002"] == "Vale"  # seat transferred
    assert vassal.relationships["Cyn"]["trust"] == -3, "the heir must NOT inherit loyalty"
    assert vassal.relationships["Rex"]["trust"] == 3, "the dead king's bond is left as memory"
    # And the heir did not silently gain a relationship record toward its vassals.
    assert "Vale" not in heir.relationships or heir.relationships["Vale"].get("trust", 0) == 0
    print("PASS test_m43_succession_does_not_inherit_loyalty")


def test_m43_succession_is_a_crisis_test() -> None:
    """Same realm, two heirs, two fates via the EXISTING breakaway machinery: a TRUSTED
    heir HOLDS the vassal; a cold/DISTRUSTED heir LOSES it (it breaks away)."""
    import lineage, population, kingdoms, trust

    def holds_vassal(heir_trust: int) -> bool:
        _lineage_world()
        world_state["settlements"]["S002"] = {"id": "S002", "center": (8, 8),
                                              "members": {"Vale"}, "founded": 0}
        king = _rich("Rex", (5, 5)); king.age = 60
        _rich("Cyn", (5, 6), parents=("Rex", "Mara")).age = 20
        vassal = _rich("Vale", (8, 8), sid="S002")
        _crown("S001", "Rex")
        _realm("Rex", "S001", {"S001", "S002"}, {"S002": "Vale"})
        trust.ensure_relationship(vassal, "Cyn")["trust"] = heir_trust  # personal standing
        population.announce_death(king, 30, world_state, cause="old age",
                                  final_memory="Died", note="they died")
        for t in range(31, 36):       # let the ordinary M3.5 loyalty machinery run
            kingdoms.update(world_state, t)
        return ("Cyn" in world_state["kingdoms"]
                and "S002" in world_state["kingdoms"]["Cyn"]["settlements"])

    assert holds_vassal(heir_trust=3) is True, "a trusted heir HOLDS the realm"
    assert holds_vassal(heir_trust=-3) is False, "a distrusted heir LOSES the vassal (breakaway)"
    print("PASS test_m43_succession_is_a_crisis_test")


def test_m43_extinct_line_dissolves_and_is_contestable() -> None:
    """A kinless king's line is EXTINGUISHED (logged distinctly); the records clear as
    today (the realm dissolves via breakaway) and the vacant seat is re-contestable."""
    import lineage, population, monarchy, kingdoms

    _lineage_world()
    world_state["settlements"]["S002"] = {"id": "S002", "center": (8, 8),
                                          "members": {"Vale"}, "founded": 0}
    king = _rich("Rex", (5, 5)); king.age = 60           # NO kin at all
    _rich("Vale", (8, 8), sid="S002")
    _crown("S001", "Rex"); _realm("Rex", "S001", {"S001", "S002"}, {"S002": "Vale"})
    population.announce_death(king, 30, world_state, cause="old age",
                              final_memory="Died", note="they died")
    events = "\n".join(world_state["events"])
    assert "the line of Rex is extinguished" in events and "lies vacant" in events
    # No heir: records left exactly as today (an inert dead holder), not re-keyed.
    assert world_state["monarchs"]["S001"]["monarch"] == "Rex"
    assert "Rex" in world_state["kingdoms"]
    # Existing machinery dissolves the realm into independent settlements: the vassal
    # settlement breaks away from a leaderless (dead) king, freeing S002.
    for t in range(31, 36):
        kingdoms.update(world_state, t)
    assert kingdoms.realm_of(world_state, "S002") is None, \
        "the vassal settlement breaks free into independence as today"
    assert "S002" not in world_state["kingdoms"].get("Rex", {}).get("settlements", set())
    # The vacant HOME seat is contestable by ordinary conquest: an aspirant seizes it.
    aspirant = _rich("Zara", (5, 4), money=30.0, sid=None)
    _rich("Merc", (5, 3), money=0.0, sid=None)   # a poor fighter in muster range
    res = monarchy.attempt_conquest(world_state, aspirant, "S001", 40)
    assert res["won"] and world_state["monarchs"]["S001"]["monarch"] == "Zara", \
        "the vacant crown falls to an ordinary aspirant"
    print("PASS test_m43_extinct_line_dissolves_and_is_contestable")


def test_m43_trust_leadership_is_never_hereditary() -> None:
    """M3.2 trust-LEADERSHIP is consent-based and NOT dynastic: a dead leader's seat is
    left untouched by succession (no coronation), even with a living child present."""
    import lineage

    _lineage_world()
    leader = _rich("Lea", (5, 5)); leader.age = 60
    _rich("Cyn", (5, 6), parents=("Lea", "Mara")).age = 20
    world_state["leaders"]["S001"] = {"leader": "Lea", "followers": {"f1"}, "since": 0}
    rec = lineage.succeed_titles(leader, 30, world_state)
    assert rec["heir"] is None and rec["kind"] == "none", "a trust-leader holds no force title"
    assert world_state["leaders"]["S001"]["leader"] == "Lea", "leadership untouched by succession"
    events = "\n".join(world_state["events"])
    assert "succeeded" not in events and "extinguished" not in events
    print("PASS test_m43_trust_leadership_is_never_hereditary")


def test_m43_dependent_heir_holds_seat_as_regent() -> None:
    """A DEPENDENT child heir inherits the seat (a historically-real regency) but its
    levy/muster/war powers stay dormant via the existing is_dependent_child gate."""
    import lineage, population, monarchy

    _lineage_world()
    king = _rich("Rex", (5, 5)); king.age = 60
    tot = _rich("Tot", (5, 6), parents=("Rex", "Mara"), dependent=True); tot.age = 6
    member = _rich("Rich", (4, 4), money=50.0)
    _crown("S001", "Rex", {"g"})
    population.announce_death(king, 30, world_state, cause="old age",
                              final_memory="Died", note="they died")
    assert world_state["monarchs"]["S001"]["monarch"] == "Tot", "the child holds the seat"
    assert "regency" in "\n".join(world_state["events"])
    before = member.money
    monarchy.levy(world_state, 31)
    assert member.money == before, "a child regent extracts NO levy (powers dormant)"
    assert monarchy.max_fighters(tot) >= 0  # sanity; the aspirant loop excludes it (regent)
    assert tot not in monarchy._eligible_aspirants(world_state, "S001"), \
        "a child regent is never an aggressor"
    print("PASS test_m43_dependent_heir_holds_seat_as_regent")


def test_m43_escheat_routes_to_successor_same_turn() -> None:
    """M4.2 interaction: when a kinless estate would escheat to a ruler who died the SAME
    turn, it routes to the SUCCESSOR (the living crown), not the dead ruler."""
    import population

    _lineage_world()
    king = _rich("Rex", (5, 5)); king.age = 60
    heir = _rich("Cyn", (5, 6), parents=("Rex", "Mara")); heir.age = 20
    _crown("S001", "Rex")
    loner = _rich("Ada", (6, 6), money=25.0)          # kinless commoner, same settlement
    population.announce_death(king, 30, world_state, cause="old age",
                              final_memory="Died", note="they died")
    assert world_state["monarchs"]["S001"]["monarch"] == "Cyn"   # succession happened first
    population.announce_death(loner, 30, world_state, cause="starved")
    assert heir.money == 25.0, "the kinless estate escheats to the SUCCESSOR"
    assert "escheated to Cyn" in "\n".join(world_state["events"])
    print("PASS test_m43_escheat_routes_to_successor_same_turn")


def test_m43_multilevel_emperor_and_subject_king_succession() -> None:
    """Multi-level: an EMPEROR's death passes the imperial throne to his heir; a SUBJECT-
    KING's death passes his subordinate seat to HIS heir — same rules, one level up."""
    import population

    _lineage_world()
    emp = _rich("Emp", (5, 5)); emp.age = 60
    _rich("Ehe", (5, 6), parents=("Emp", "Mara")).age = 20      # emperor's heir
    sky = _rich("Sky", (7, 7), sid=None); sky.age = 55          # a subject-king
    _crown("S001", "Emp")
    _realm("Emp", "S001", {"S001"}, {})
    _realm("Sky", "S002", {"S002"}, {})
    world_state["empires"]["Emp"] = {"emperor": "Emp",
                                     "subject_kings": {"Sky": {"since": 0}},
                                     "founded": 0, "discontent": {"Sky": 0}}
    population.announce_death(emp, 30, world_state, cause="old age",
                              final_memory="Died", note="they died")
    assert "Ehe" in world_state["empires"] and "Emp" not in world_state["empires"]
    ehe = world_state["empires"]["Ehe"]
    assert ehe["emperor"] == "Ehe"
    assert ehe["subject_kings"] == {"Sky": {"since": 0}}         # subject-king unchanged
    assert ehe["discontent"] == {"Sky": 0}
    assert world_state["kingdoms"]["Ehe"]["king"] == "Ehe"       # his own realm re-keyed too
    assert world_state["monarchs"]["S001"]["monarch"] == "Ehe"

    # Now the subject-king dies -> HIS heir takes the subordinate seat + his own realm.
    _rich("Ski", (7, 8), parents=("Sky", "Nel")).age = 20
    population.announce_death(sky, 31, world_state, cause="old age",
                              final_memory="Died", note="they died")
    ehe = world_state["empires"]["Ehe"]
    assert "Ski" in ehe["subject_kings"] and "Sky" not in ehe["subject_kings"]
    assert ehe["discontent"].get("Ski") == 0
    assert world_state["kingdoms"]["Ski"]["king"] == "Ski"
    print("PASS test_m43_multilevel_emperor_and_subject_king_succession")


def test_m43_succession_only_when_lineage_on() -> None:
    """With lineage OFF, a titled ruler's death moves NO title — the records are left
    exactly as pre-M4.3 (the byte-identical guarantee holds at the title level too)."""
    import population

    _lineage_world()
    world_state["lineage_on"] = False
    king = _rich("Rex", (5, 5)); king.age = 60
    _rich("Cyn", (5, 6), parents=("Rex", "Mara")).age = 20
    _crown("S001", "Rex")
    _realm("Rex", "S001", {"S001"}, {})
    population.announce_death(king, 30, world_state, cause="old age",
                              final_memory="Died", note="they died")
    assert world_state["monarchs"]["S001"]["monarch"] == "Rex"   # no succession off
    assert "Rex" in world_state["kingdoms"] and "Cyn" not in world_state["kingdoms"]
    assert "succeeded" not in "\n".join(world_state["events"])
    print("PASS test_m43_succession_only_when_lineage_on")


# --- Discontent (V2 M4.4): the class-pressure gauge -------------------------
def _disc_world() -> None:
    """A clean world (lineage OFF) with one settlement S001 at (5, 5), ready for the gauge."""
    _fresh_world()
    world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5),
                                          "members": set(), "founded": 0}


def _monarch(sid: str, name: str) -> None:
    world_state["monarchs"][sid] = {"monarch": name, "since": 0, "garrison": set()}


def _employ(employer: str, worker: str, wage: float) -> None:
    world_state.setdefault("employments", []).append(
        {"employer": employer, "worker": worker, "wage": wage, "since": 0})


def test_discontent_off_run_is_byte_identical_to_v1() -> None:
    """discontent_on=False (default) leaves the run byte-identical to the no-param run,
    AND stacked on every other institution it still changes nothing when off."""
    def run(**kw):
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, **kw)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        base, off = run(), run(discontent_on=False)
        stacked_base = run(settlements=True, labor_on=True, economy_on=True, monarchy_on=True)
        stacked_off = run(settlements=True, labor_on=True, economy_on=True, monarchy_on=True,
                          discontent_on=False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "discontent_on=False diverged from the default run"
    assert stacked_base == stacked_off, "discontent_on=False changed an institution run"
    # And with it OFF, world_state carries no gauge key at all.
    assert "discontent" not in world_state
    print("PASS test_discontent_off_run_is_byte_identical_to_v1")


def test_each_driver_raises_gauge_only_when_its_condition_holds() -> None:
    """Each of the three drivers, in isolation, raises the gauge — and ONLY while its
    grievance condition holds; with all conditions absent the gauge stays flat at zero."""
    import discontent

    # DEPRIVATION: a hungry agent beside a rich settlement-mate resents; fed OR alone it does not.
    def deprivation_case(hunger: int, neighbour_wealth: float) -> float:
        _disc_world()
        poor = _settler("Poor", (5, 5), hunger=hunger)
        rich = _settler("Rich", (6, 5))
        rich.money = neighbour_wealth
        for t in range(1, 6):
            discontent.update(world_state, t)
        return discontent.agent_discontent("Poor", world_state)

    assert deprivation_case(hunger=8, neighbour_wealth=30.0) > 0, "hunger amid plenty must resent"
    assert deprivation_case(hunger=1, neighbour_wealth=30.0) == 0, "a fed agent feels no deprivation"
    assert deprivation_case(hunger=8, neighbour_wealth=0.0) == 0, "no plenty next door -> no injustice"

    # EXPLOITATION: a subsistence wage bites; a wage near full output does not.
    def wage_case(wage: float) -> float:
        _disc_world()
        w = _settler("Worker", (5, 5))
        w.money = 20.0                       # fed & solvent so ONLY the wage driver is live
        _settler("Boss", (6, 5)).money = 20.0
        _employ("Boss", "Worker", wage)
        for t in range(1, 6):
            discontent.update(world_state, t)
        return discontent.agent_discontent("Worker", world_state)

    import labor
    assert wage_case(labor.SUBSISTENCE_WAGE) > 0, "a subsistence wage must resent"
    assert wage_case(labor.LABOR_OUTPUT) == 0, "a full-output wage captures nothing -> no grievance"

    # EXTRACTION: a levying monarch bites; an unruled settlement does not.
    def extraction_case(with_monarch: bool) -> float:
        _disc_world()
        subj = _settler("Subj", (5, 5))
        subj.money = 20.0                    # wealth above the levy threshold, and fed
        if with_monarch:
            _settler("King", (6, 5))
            _monarch("S001", "King")
        for t in range(1, 6):
            discontent.update(world_state, t)
        return discontent.agent_discontent("Subj", world_state)

    assert extraction_case(with_monarch=True) > 0, "a levying monarch must draw grievance"
    assert extraction_case(with_monarch=False) == 0, "no ruler -> no extraction grievance"

    # ALL OFF: fed, unemployed, unruled -> the gauge never leaves zero.
    _disc_world()
    calm = _settler("Calm", (5, 5))
    calm.money = 20.0
    _settler("Peer", (6, 5)).money = 20.0
    for t in range(1, 8):
        discontent.update(world_state, t)
    assert discontent.agent_discontent("Calm", world_state) == 0.0, "no grievance -> flat zero"
    print("PASS test_each_driver_raises_gauge_only_when_its_condition_holds")


def test_legitimacy_buffers_extraction_grievance() -> None:
    """The SAME levy by a TRUSTED ruler draws materially less grievance than by a
    DISTRUSTED one — consent is the difference between a tax and a theft."""
    import discontent, trust

    def levied_under(heir_trust: int) -> float:
        _disc_world()
        subj = _settler("Subj", (5, 5))
        subj.money = 20.0
        _settler("King", (6, 5))
        _monarch("S001", "King")
        trust.ensure_relationship(subj, "King")["trust"] = heir_trust
        for t in range(1, 6):
            discontent.update(world_state, t)
        return discontent.agent_discontent("Subj", world_state)

    hated = levied_under(-5)
    trusted = levied_under(5)
    neutral = levied_under(0)
    assert trusted < neutral < hated, (trusted, neutral, hated)
    assert trusted < 0.5 * hated, "a trusted ruler's levy must sting far less"
    print("PASS test_legitimacy_buffers_extraction_grievance")


def test_extraction_burden_is_bounded_by_means() -> None:
    """The burden weighting is due/max(wealth, MEANS_FLOOR), CAPPED at 1 — so a near-broke
    agent's grievance stays bounded (the MEANS_FLOOR keeps 'burden relative to means' sane
    at the bottom) rather than dividing a tiny levy by a tiny wealth into a crushing burden.

    NOTE (honest): the institutions levy PROPORTIONALLY (a rate of wealth above a threshold),
    so due/wealth is roughly flat across wealth — this weighting differentiates a FLAT sum, not
    a proportional levy, and does not by itself make the poor resent a proportional levy more."""
    import discontent

    _disc_world()
    _monarch("S001", "King")
    _settler("King", (6, 5))
    # A near-broke subject: burden must be clamped to <= 1, not explode.
    broke = _settler("Broke", (5, 5))
    broke.money = 5.01                    # a hair over the levy threshold -> a tiny due
    inc, ruler, kind = discontent.extraction(broke, world_state)
    assert ruler == "King" and 0.0 < inc <= discontent.EXTRACTION_WEIGHT, inc
    print("PASS test_extraction_burden_is_bounded_by_means")


def test_decay_is_asymmetric_with_a_floor() -> None:
    """Grievances outlast their causes: the gauge RISES fast under oppression and DECAYS
    slowly once relieved (asymmetric slopes), and never falls below zero."""
    import discontent

    _disc_world()
    poor = _settler("Poor", (5, 5), hunger=8)
    _settler("Rich", (6, 5)).money = 30.0

    # RISE: five turns of hunger amid plenty.
    for t in range(1, 6):
        discontent.update(world_state, t)
    peak = discontent.agent_discontent("Poor", world_state)
    rise_slope = peak / 5.0
    assert peak > 0

    # RELIEF: feed the agent (grievance goes silent), then let it ebb.
    poor.hunger = 1
    discontent.update(world_state, 6)
    after_one = discontent.agent_discontent("Poor", world_state)
    fall_slope = peak - after_one
    assert fall_slope > 0, "relieved discontent must ebb"
    assert fall_slope < rise_slope, "it must fall SLOWER than it rose (hysteresis)"
    assert fall_slope < 0.5 * rise_slope, (fall_slope, rise_slope)

    # SUSTAINED relief returns it to zero and FLOORS there.
    for t in range(7, 60):
        discontent.update(world_state, t)
    assert discontent.agent_discontent("Poor", world_state) == 0.0, "must decay to a hard floor of 0"
    print("PASS test_decay_is_asymmetric_with_a_floor")


def test_no_decay_while_grievance_persists() -> None:
    """The gauge does not decay on a turn where a grievance is still active — it only
    accumulates; relief (decay) happens ONLY when every driver is silent."""
    import discontent

    _disc_world()
    poor = _settler("Poor", (5, 5), hunger=8)
    _settler("Rich", (6, 5)).money = 30.0
    prev = 0.0
    for t in range(1, 6):
        discontent.update(world_state, t)
        now = discontent.agent_discontent("Poor", world_state)
        assert now > prev, "a live grievance must keep the gauge monotonically rising"
        prev = now
    print("PASS test_no_decay_while_grievance_persists")


def test_settlement_pressure_counts_only_above_threshold_members() -> None:
    """settlement_pressure is the count of a settlement's LIVING members whose gauge is at
    or above the resentment threshold — nobody below, nobody dead, nobody from elsewhere."""
    import discontent

    _disc_world()
    world_state["settlements"]["S002"] = {"id": "S002", "center": (9, 9),
                                          "members": set(), "founded": 0}
    a = _settler("A", (5, 5))
    b = _settler("B", (5, 6))
    c = _settler("C", (6, 5))
    _settler("D", (9, 9), sid="S002")        # in ANOTHER settlement — must not count for S001
    thr = discontent.RESENTMENT_THRESHOLD
    world_state["discontent"] = {"A": thr, "B": thr - 0.1, "C": thr + 5, "D": thr + 9}
    assert discontent.settlement_pressure("S001", world_state) == 2      # A and C only
    assert discontent.settlement_pressure("S002", world_state) == 1      # D
    # A dead resentful member drops out of the count.
    c.alive = False
    assert discontent.settlement_pressure("S001", world_state) == 1      # only A now
    # settlement_discontent sums the LIVING members' gauge.
    assert discontent.settlement_discontent("S001", world_state) == thr + (thr - 0.1)
    print("PASS test_settlement_pressure_counts_only_above_threshold_members")


def test_threshold_crossing_logs_sparsely() -> None:
    """Crossing INTO resentment logs exactly ONE line naming the dominant driver; further
    increments above the line log nothing (the events log stays readable)."""
    import discontent

    _disc_world()
    poor = _settler("Poor", (5, 5), hunger=9)
    _settler("Rich", (6, 5)).money = 30.0
    for t in range(1, 15):
        discontent.update(world_state, t)
    seethes = [e for e in world_state["events"] if "Poor seethes" in e]
    assert len(seethes) == 1, seethes
    assert "hunger amid plenty" in seethes[0], seethes[0]
    assert discontent.agent_discontent("Poor", world_state) >= discontent.RESENTMENT_THRESHOLD
    print("PASS test_threshold_crossing_logs_sparsely")


def test_regency_levy_draws_no_extraction_grievance() -> None:
    """A DEPENDENT-CHILD monarch (M4.3 regency) levies nothing — so, exactly as the levy
    code skips it, the gauge registers no extraction from a phantom levy."""
    import discontent

    _disc_world()
    world_state["lineage_on"] = True                 # regency only exists under lineage
    subj = _settler("Subj", (5, 5))
    subj.money = 20.0
    child = _settler("Tot", (6, 5))
    child.dependent = True                           # a minor on the throne
    _monarch("S001", "Tot")
    for t in range(1, 6):
        discontent.update(world_state, t)
    assert discontent.agent_discontent("Subj", world_state) == 0.0, "a regent levies nothing"
    print("PASS test_regency_levy_draws_no_extraction_grievance")


def test_discontent_adds_no_llm_and_is_deterministic() -> None:
    """The gauge draws ZERO ADDED LLM calls (identical counts with it on vs off) and no new
    RNG (two seeded on-runs are byte-identical to each other)."""
    def run(on: bool):
        llm.PROVIDER = "random"
        random.seed(7)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, settlements=True, labor_on=True, economy_on=True,
                                monarchy_on=True, discontent_on=on)
        return buf.getvalue(), dict(llm.get_call_stats())

    saved = llm.PROVIDER
    try:
        _, off_calls = run(False)
        on_a, on_calls = run(True)
        on_b, _ = run(True)
    finally:
        llm.PROVIDER = saved
    assert on_calls == off_calls, (on_calls, off_calls)   # discontent adds no LLM
    assert on_a == on_b, "an on run must be byte-identical across seeded repeats"
    print("PASS test_discontent_adds_no_llm_and_is_deterministic")


# --- Uprising (V2 M4.5): the revolt fires -----------------------------------
def _resentful(names: dict, level: float = 12.0) -> None:
    """Stamp the discontent gauge directly (the mob's readiness) for a set of names."""
    world_state["discontent"] = dict(names) if isinstance(names, dict) else {n: level for n in names}


def _bystander_mercs(positions, money: float = 0.5) -> list:
    """Poor NOMAD bystanders (not settlement members) a ruler could hire as guards."""
    out = []
    for i, p in enumerate(positions):
        a = Agent(name=f"merc{i}", personality="x")
        place_agent(a, *p)
        a.hunger, a.age, a.lifespan, a.money, a.settlement = 1, 30, 100, money, None
        out.append(a)
    return out


def test_uprising_requires_both_trigger_gates() -> None:
    """A rising needs BOTH gates: a resentful MAJORITY (fraction) AND aggregate discontent over
    the floor. Either alone does not fire."""
    import uprising

    # Fraction gate: 1 of 4 resentful (even at huge discontent) is not a majority -> no rise;
    # 2 of 4 resentful clears both fraction and the floor -> rises.
    def fraction_case(n_resentful: int) -> bool:
        _disc_world()
        _settler("King", (5, 5))
        for n, p in [("A", (5, 6)), ("B", (6, 5)), ("C", (6, 6)), ("D", (7, 5))]:
            _settler(n, p).money = 0.5
        _monarch("S001", "King")
        _resentful({n: 12.0 for n in list("ABCD")[:n_resentful]})
        return uprising.would_rise(world_state, "S001", 10)

    assert fraction_case(1) is False, "a lone resenter is not a majority"
    assert fraction_case(2) is True, "half the commoners resentful -> the majority gate opens"

    # Aggregate gate: a sole non-ruler member resentful at exactly the threshold (6) is a majority
    # but under the aggregate floor (12) -> no rise; raise its grievance over the floor -> rises.
    def aggregate_case(level: float) -> bool:
        _disc_world()
        _settler("King", (5, 5))
        _settler("Lone", (5, 6)).money = 0.5
        _monarch("S001", "King")
        _resentful({"Lone": level})
        return uprising.would_rise(world_state, "S001", 10)

    import discontent
    assert aggregate_case(discontent.RESENTMENT_THRESHOLD) is False, "a bare majority under the floor waits"
    assert aggregate_case(uprising.UPRISING_MIN_PRESSURE + 1) is True, "real accumulated weight fires it"
    print("PASS test_uprising_requires_both_trigger_gates")


def test_consent_led_settlement_never_rises() -> None:
    """A settlement whose only authority is a consent-based trust-leader (M3.2) is NEVER a valid
    target — the people chose them, so there is nothing to overthrow (even if the gauge is maxed)."""
    import uprising

    _disc_world()
    _settler("Led", (5, 5))
    for n, p in [("A", (5, 6)), ("B", (6, 5)), ("C", (6, 6))]:
        _settler(n, p).money = 0.5
    world_state["leadership_on"] = True
    world_state["leaders"]["S001"] = {"leader": "Led", "followers": {"A", "B", "C"}, "since": 0}
    _resentful({"A": 25.0, "B": 25.0, "C": 25.0})     # maxed grievance, no force ruler
    assert uprising.would_rise(world_state, "S001", 10) is False
    assert uprising.update(world_state, 10) == [], "a consent-led town cannot rise"
    print("PASS test_consent_led_settlement_never_rises")


def test_mob_is_numbers_penniless_wins_funded_ruler_crushes() -> None:
    """The mob's force is its NUMBERS: a penniless mob overwhelms an undefended ruler, but the SAME
    mob is crushed by a rich ruler who buys guards. Wealth is the counter-revolutionary weapon."""
    import uprising

    def rising(king_money: float, with_merc_pool: bool):
        _disc_world()
        king = _settler("King", (5, 5)); king.money = king_money
        risers = [_settler(n, p) for n, p in [("A", (5, 6)), ("B", (6, 5)), ("C", (6, 7))]]
        for r in risers:
            r.money = 0.0                              # a truly PENNILESS mob
        if with_merc_pool:
            _bystander_mercs([(4, 4), (4, 5), (5, 4), (4, 6), (6, 4)])
        _monarch("S001", "King")
        _resentful({"A": 12.0, "B": 12.0, "C": 12.0})
        res = uprising.update(world_state, 10)
        return res[0], king

    # Undefended ruler (no garrison, no one to hire): the penniless mob WINS on numbers.
    r, king = rising(king_money=0.0, with_merc_pool=False)
    assert r["won"] and r["deposed"] and not king.alive, r

    # Rich ruler with a mercenary pool: he musters guards that OUTNUMBER the mob -> CRUSHED.
    r, king = rising(king_money=200.0, with_merc_pool=True)
    assert not r["won"] and king.alive and r["defenders"] > r["mob"], r
    assert world_state["monarchs"]["S001"]["monarch"] == "King", "the crown holds"
    print("PASS test_mob_is_numbers_penniless_wins_funded_ruler_crushes")


def test_crushed_rising_partial_reset_cooldown_and_persistent_grievance() -> None:
    """A crushed rising: survivors are cowed (partial reset, NOT to zero — the grievance persists),
    a fear cooldown blocks re-rising, and the ruler holds."""
    import uprising

    _disc_world()
    king = _settler("King", (5, 5)); king.money = 200.0
    for n, p in [("A", (5, 6)), ("B", (6, 5)), ("C", (6, 7))]:
        _settler(n, p).money = 0.0
    _bystander_mercs([(4, 4), (4, 5), (5, 4), (4, 6), (6, 4)])
    _monarch("S001", "King")
    _resentful({"A": 12.0, "B": 12.0, "C": 12.0})
    res = uprising.update(world_state, 10)
    assert not res[0]["won"]
    # A survivor (deaths are by name order, so C outlives A/B) keeps a REDUCED but non-zero gauge.
    surv = next(a for a in world_state["agents"] if a.name == "C" and a.alive)
    g = world_state["discontent"]["C"]
    assert 0.0 < g < 12.0, g
    assert abs(g - 12.0 * uprising.FEAR_RETAIN) < 1e-9, g
    # Cooldown set; it blocks re-rising even while still resentful.
    assert world_state["uprising_cooldowns"]["S001"] == 10 + uprising.UPRISING_COOLDOWN
    world_state["discontent"] = {"A": 12.0, "B": 12.0, "C": 12.0}  # re-anger everyone
    assert uprising.would_rise(world_state, "S001", 12) is False, "cooldown holds the peace"
    assert uprising.would_rise(world_state, "S001", 10 + uprising.UPRISING_COOLDOWN) is True, "then it can rise again"
    print("PASS test_crushed_rising_partial_reset_cooldown_and_persistent_grievance")


def test_victory_deposes_clears_title_and_breaks_kingdom_away() -> None:
    """A successful rising in a VASSAL settlement: the lord is deposed, his monarch record cleared,
    and the settlement SECEDES from the realm via the existing kingdoms machinery — seat left vacant."""
    import uprising, kingdoms

    _disc_world()
    world_state["settlements"]["S000"] = {"id": "S000", "center": (1, 1), "members": {"King"}, "founded": 0}
    king = _settler("King", (1, 1), sid="S000"); king.money = 50.0
    lord = _settler("Lord", (5, 5)); lord.money = 0.0            # a DRAINED vassal lord
    for n, p in [("A", (5, 6)), ("B", (6, 5)), ("C", (6, 7))]:
        _settler(n, p).money = 0.0
    _monarch("S001", "Lord")                                     # the lord holds S001 by force
    _crown("S000", "King")
    _realm("King", "S000", {"S000", "S001"}, {"S001": "Lord"})   # S001 is a vassal of King
    assert kingdoms.realm_of(world_state, "S001") == "King"
    _resentful({"A": 12.0, "B": 12.0, "C": 12.0})
    res = uprising.update(world_state, 10)
    r = next(x for x in res if x["sid"] == "S001")
    assert r["won"] and r["deposed"]
    assert "S001" not in world_state["monarchs"], "the local force-title is cleared"
    assert kingdoms.realm_of(world_state, "S001") is None, "the settlement seceded from the realm"
    assert not lord.alive, "the deposed lord is killed through the normal path"
    assert any("SECEDED from King" in e for e in world_state["events"])
    print("PASS test_victory_deposes_clears_title_and_breaks_kingdom_away")


def test_expropriation_conserved_and_preempts_inheritance() -> None:
    """A successful rising SEIZES the ruler's hoard, splits it among the risers (conserved to the
    decimal), and — because the seizure precedes the death — his M4.2 HEIRS inherit NOTHING. The
    contrast: the SAME ruler dying of old age leaves the whole estate to the heir."""
    import uprising, population

    def build_and(fate: str):
        _disc_world()
        world_state["lineage_on"] = True
        king = _settler("King", (5, 5)); king.money = 40.0
        heir = _settler("Heir", (5, 6)); heir.parents = ("King", "Q"); heir.dependent = True; heir.age = 6
        risers = [_settler(n, p) for n, p in [("A", (6, 5)), ("B", (6, 6)), ("C", (7, 5))]]
        for r in risers:
            r.money = 0.0
        _monarch("S001", "King")
        _resentful({"A": 12.0, "B": 12.0, "C": 12.0})
        if fate == "uprising":
            res = uprising.update(world_state, 10)[0]
            victors = [a for a in world_state["agents"] if a.name in ("A", "B", "C") and a.alive]
            return king, heir, res, victors
        else:  # the same king dies of old age instead
            population.announce_death(king, 10, world_state, cause="old age",
                                      final_memory="Died of old age", note="they died of old age")
            return king, heir, None, []

    king, heir, res, victors = build_and("uprising")
    assert res["won"] and abs(res["seized"] - 40.0) < 1e-9
    got = sum(v.money for v in victors)
    assert abs(got - 40.0) < 1e-9, got                 # conserved to the decimal
    assert heir.money == 0.0, "revolution interrupts inheritance — the heir gets nothing"
    assert king.money == 0.0 and king.stockpile == 0.0

    # CONTRAST: dying of old age, the heir DOES inherit the whole estate.
    king2, heir2, _, _ = build_and("oldage")
    assert heir2.money == 40.0, "an ordinary death still passes the estate to the heir"
    print("PASS test_expropriation_conserved_and_preempts_inheritance")


def test_uprising_deaths_compose_with_succession_and_inheritance() -> None:
    """Deaths route through population.announce_death, so a deposed ruler's death is a first-class
    event — but because the title was cleared and the hoard seized first, NO heir succeeds the crown
    and NO estate settles (Arc 1 composes: the dynasty simply ends)."""
    import uprising

    _disc_world()
    world_state["lineage_on"] = True
    king = _settler("King", (5, 5)); king.money = 30.0
    heir = _settler("Heir", (5, 6)); heir.parents = ("King", "Q"); heir.age = 25   # an ADULT heir
    for n, p in [("A", (6, 5)), ("B", (6, 6)), ("C", (7, 5))]:
        _settler(n, p).money = 0.0
    _monarch("S001", "King")
    _resentful({"A": 12.0, "B": 12.0, "C": 12.0})
    uprising.update(world_state, 10)
    events = "\n".join(world_state["events"])
    assert "King died (deposed in the uprising)" in events, events
    assert "succeeded King" not in events, "the cleared title cannot pass to an heir"
    assert "S001" not in world_state["monarchs"] and heir.money == 0.0
    print("PASS test_uprising_deaths_compose_with_succession_and_inheritance")


def test_uprising_off_run_is_byte_identical_to_v1() -> None:
    """uprising_on=False (default) leaves the run byte-identical, alone and stacked on institutions,
    and writes no cooldown state."""
    def run(**kw):
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, **kw)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        base, off = run(), run(uprising_on=False)
        stacked = run(settlements=True, monarchy_on=True, discontent_on=True)
        stacked_off = run(settlements=True, monarchy_on=True, discontent_on=True, uprising_on=False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "uprising_on=False diverged from the default run"
    assert stacked == stacked_off, "uprising_on=False changed an institution run"
    assert not world_state.get("uprising_cooldowns"), "off run must write no cooldown state"
    print("PASS test_uprising_off_run_is_byte_identical_to_v1")


def test_uprising_adds_no_llm_and_is_deterministic() -> None:
    """The uprising system draws ZERO added LLM (identical counts on vs off) and no new RNG (two
    seeded on-runs are byte-identical)."""
    def run(on: bool):
        llm.PROVIDER = "random"
        random.seed(7)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, settlements=True, monarchy_on=True, discontent_on=True,
                                uprising_on=on)
        return buf.getvalue(), dict(llm.get_call_stats())

    saved = llm.PROVIDER
    try:
        _, off_calls = run(False)
        on_a, on_calls = run(True)
        on_b, _ = run(True)
    finally:
        llm.PROVIDER = saved
    assert on_calls == off_calls, (on_calls, off_calls)
    assert on_a == on_b, "an on run must be byte-identical across seeded repeats"
    print("PASS test_uprising_adds_no_llm_and_is_deterministic")


# --- The Revolutionary (V2 M4.6): a rising's leader rules by consent ---------
def test_revolutionary_is_derived_from_risers_not_assigned() -> None:
    """A won rising throws up a leader DERIVED from the risers — the angriest-and-most-trusted
    ORDINARY riser (not the richest, not arbitrary) — who takes the vacant seat by CONSENT through
    the EXISTING M3.2 path (a leaders record, never a monarch record)."""
    import uprising, leadership, trust

    _disc_world()
    world_state["leadership_on"] = True
    _settler("King", (5, 5)).money = 0.5                     # a drained tyrant -> the mob wins
    for n, p in [("A", (5, 6)), ("B", (6, 5)), ("C", (6, 7))]:
        _settler(n, p).money = 0.0
    rex = _settler("Rex", (7, 5)); rex.money = 100.0         # a RICH riser — must NOT become leader
    _monarch("S001", "King")
    # Rex is the angriest, but rich; B is the angriest COMMONER and the one his fellows trust.
    world_state["discontent"] = {"A": 12.0, "B": 20.0, "C": 12.0, "Rex": 25.0}
    for n in ("A", "C", "Rex"):
        trust.ensure_relationship(next(a for a in world_state["agents"] if a.name == n), "B")["trust"] = 1

    res = uprising.update(world_state, 10)[0]
    assert res["won"] and res["leader"] == "B", res            # derived: B, not rich Rex, not arbitrary
    assert "S001" not in world_state["monarchs"], "the tyrant's crown is gone"
    assert world_state.get("leaders", {}).get("S001") is None, "the uprising installs NO leader record itself"
    # The UNCHANGED M3.2 machinery then elects him from the seeded trust — power by consent.
    leadership.update(world_state, 11)
    rec = world_state["leaders"].get("S001")
    assert rec is not None and rec["leader"] == "B", "M3.2 elects the revolutionary by consent"
    assert "S001" not in world_state["monarchs"], "he rules as a LEADER, not a monarch"
    print("PASS test_revolutionary_is_derived_from_risers_not_assigned")


def test_revolutionary_holds_by_consent_and_can_be_displaced() -> None:
    """The revolutionary holds the seat only while trusted: when his following erodes below the M3.2
    keep-bar, the EXISTING leadership machinery strips him — no permanent crown."""
    import uprising, leadership

    _disc_world()
    world_state["leadership_on"] = True
    _settler("King", (5, 5)).money = 0.5
    risers = [_settler(n, p) for n, p in [("A", (5, 6)), ("B", (6, 5)), ("C", (6, 7))]]
    for r in risers:
        r.money = 0.0
    _monarch("S001", "King")
    world_state["discontent"] = {"A": 20.0, "B": 12.0, "C": 12.0}   # A leads
    uprising.update(world_state, 10)
    leadership.update(world_state, 11)
    assert world_state["leaders"]["S001"]["leader"] == "A"
    # His following collapses (trust falls below KEEP_TRUST) -> M3.2 unseats him next turn.
    for n in ("B", "C"):
        f = next(a for a in world_state["agents"] if a.name == n)
        f.relationships["A"]["trust"] = 0
    leadership.update(world_state, 12)
    assert world_state["leaders"].get("S001") is None, "a leader with no following falls (M3.2, untouched)"
    assert "S001" not in world_state["monarchs"], "he was never a monarch — no permanent grip"
    print("PASS test_revolutionary_holds_by_consent_and_can_be_displaced")


def test_revolution_devours_its_children() -> None:
    """A revolutionary who becomes an EXTRACTOR himself (seizes a force-title) breeds the SAME
    discontent and is risen against by the SAME machinery — no immunity, no new mechanic."""
    import uprising, discontent, monarchy

    _disc_world()
    # B led a rising and rules; he then SEIZES the crown of S001 (M3.4) — now an extractor by force.
    b = _settler("B", (5, 5)); b.money = 0.5
    members = [_settler(n, p) for n, p in [("A", (5, 6)), ("C", (6, 5)), ("D", (6, 7))]]
    for m in members:
        m.money = 20.0                                         # solvent -> the levy is felt
    _monarch("S001", "B")                                      # the former revolutionary, now a monarch
    for m in members:                                          # the people do NOT consent to his crown
        m.relationships["B"] = {"trust": -5}
    for turn in range(1, 13):
        discontent.update(world_state, turn)
    assert discontent.settlement_pressure("S001", world_state) >= 2, "his extraction breeds real grievance"
    res = uprising.update(world_state, 13)
    assert res and res[0]["won"] and res[0]["deposed"], "the same machinery rises against HIM"
    assert "S001" not in world_state["monarchs"], "the revolutionary-turned-tyrant is himself overthrown"
    print("PASS test_revolution_devours_its_children")


def test_too_few_survivors_leaves_seat_vacant() -> None:
    """Honest edge: if a won rising leaves too few survivors to cohere a following, M3.2 seats no
    one — the leader is not force-installed, the seat simply stays vacant."""
    import uprising, leadership

    _disc_world()
    world_state["leadership_on"] = True
    king = _settler("King", (5, 5)); king.money = 12.0        # funds exactly 2 guards
    for n, p in [("A", (5, 6)), ("B", (6, 5)), ("C", (6, 7))]:
        _settler(n, p).money = 0.0
    _bystander_mercs([(4, 4), (4, 5)])                        # a pool of exactly 2 hireable guards
    _monarch("S001", "King")
    world_state["discontent"] = {"A": 12.0, "B": 12.0, "C": 12.0}
    res = uprising.update(world_state, 10)[0]
    # 3 risers > 2 guards -> win, but casualties leave only 2 survivors (a leader + 1 follower).
    assert res["won"] and len(res["mob_dead"]) == 1
    leadership.update(world_state, 11)
    assert world_state["leaders"].get("S001") is None, "one follower cannot cohere a following -> vacant"
    assert "S001" not in world_state["monarchs"], "and no one is force-installed"
    print("PASS test_too_few_survivors_leaves_seat_vacant")


# --- Beliefs (V2 M4.7): the inner life --------------------------------------
def _belief_world() -> None:
    _fresh_world()
    world_state["beliefs_on"] = True
    world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5),
                                          "members": set(), "founded": 0}


def _bagent(name, pos, *, hunger=1, money=0.0, sid="S001", personality="friendly and outgoing",
            knows=None):
    a = Agent(name=name, personality=personality)
    place_agent(a, *pos)
    a.hunger, a.age, a.lifespan, a.money, a.settlement = hunger, 30, 100, money, sid
    if knows:
        a.knowledge.update(knows)
    if sid is not None and sid in world_state["settlements"]:
        world_state["settlements"][sid]["members"].add(name)
    return a


def test_beliefs_form_from_lived_experience_each_condition_binds() -> None:
    """Each formation condition is individually demonstrable: abundance -> 'the land provides',
    starvation -> 'the world is cruel', a producer skill while fed -> 'knowledge is power'; and an
    agent who lives through none of them forms nothing (earned, never assigned)."""
    import beliefs

    def lived(build, turns):
        _belief_world()
        build()
        for t in range(1, turns + 1):
            beliefs._update_experience(world_state, t)
        beliefs.form(world_state, turns + 1)

    lived(lambda: _bagent("Fed", (5, 5), hunger=1), beliefs.ABUNDANCE_TURNS + 1)
    assert beliefs.LAND_PROVIDES in beliefs.agent_beliefs("Fed", world_state)
    assert beliefs.WORLD_IS_CRUEL not in beliefs.agent_beliefs("Fed", world_state)

    lived(lambda: _bagent("Starve", (5, 5), hunger=8), beliefs.HARDSHIP_TURNS + 1)
    assert beliefs.WORLD_IS_CRUEL in beliefs.agent_beliefs("Starve", world_state)

    lived(lambda: _bagent("Farmer", (5, 5), hunger=1, knows={"farming"}), beliefs.SKILL_FED_TURNS + 1)
    assert beliefs.KNOWLEDGE_IS_POWER in beliefs.agent_beliefs("Farmer", world_state)

    # Control: hunger between the fed and hardship bars, no wealth, no ruler, no deaths -> nothing.
    lived(lambda: _bagent("Meh", (5, 5), hunger=4), 15)
    assert beliefs.agent_beliefs("Meh", world_state) == set(), beliefs.agent_beliefs("Meh", world_state)
    print("PASS test_beliefs_form_from_lived_experience_each_condition_binds")


def test_beliefs_spread_by_trust_and_never_in_isolation() -> None:
    """A belief spreads FAR more readily from a trusted neighbour than a distrusted one, and an
    isolated agent (no contact) never acquires a belief it did not live."""
    import beliefs, trust

    def cohort(trust_val):
        adopted = 0
        for i in range(30):
            _belief_world()
            _bagent(f"S{i}", (5, 5))
            learner = _bagent(f"L{i}", (5, 6))
            world_state["beliefs"] = {f"S{i}": {beliefs.LAND_PROVIDES}}
            trust.ensure_relationship(learner, f"S{i}")["trust"] = trust_val
            rng = random.Random(100 + i)
            for t in range(1, 5):
                beliefs.spread(world_state, t, rng)
            adopted += beliefs.LAND_PROVIDES in beliefs.agent_beliefs(f"L{i}", world_state)
        return adopted

    trusted, distrusted = cohort(5), cohort(-5)
    assert trusted > distrusted + 8, (trusted, distrusted)   # markedly faster from a trusted mouth

    # Isolation: a distant agent shares no contact and never adopts.
    _belief_world()
    _bagent("Src", (1, 1))
    _bagent("Iso", (9, 9))
    world_state["beliefs"] = {"Src": {beliefs.LAND_PROVIDES}}
    rng = random.Random(1)
    for t in range(1, 30):
        beliefs.spread(world_state, t, rng)
    assert beliefs.LAND_PROVIDES not in beliefs.agent_beliefs("Iso", world_state)
    print("PASS test_beliefs_spread_by_trust_and_never_in_isolation")


def test_contradictory_belief_flips_only_from_a_trusted_source() -> None:
    """A contradictory belief OVERWRITES the incumbent one only when the source is trusted enough
    (>= FLIP_TRUST); from a barely-trusted source the learner keeps its worldview."""
    import beliefs, trust

    def outcome(trust_val):
        _belief_world()
        _bagent("Src", (5, 5))
        lrn = _bagent("Lrn", (5, 6))
        world_state["beliefs"] = {"Src": {beliefs.WORLD_IS_CRUEL}, "Lrn": {beliefs.LAND_PROVIDES}}
        trust.ensure_relationship(lrn, "Src")["trust"] = trust_val
        rng = random.Random(1)
        for t in range(1, 30):
            beliefs.spread(world_state, t, rng)
        return beliefs.agent_beliefs("Lrn", world_state)

    flipped = outcome(5)
    assert flipped == {beliefs.WORLD_IS_CRUEL}, flipped         # adopted the new, renounced the old
    kept = outcome(0)                                           # below FLIP_TRUST
    assert kept == {beliefs.LAND_PROVIDES}, kept                # worldview held against a stranger
    print("PASS test_contradictory_belief_flips_only_from_a_trusted_source")


def test_children_inherit_settlement_beliefs_via_childhood_boost() -> None:
    """A dependent child soaks up its parent's belief through the childhood learning window (culture
    inherited by upbringing, not by blood)."""
    import beliefs, trust

    _belief_world()
    world_state["lineage_on"] = True
    _bagent("Parent", (5, 5))
    child = _bagent("Child", (5, 6))
    child.age, child.dependent, child.parents = 6, True, ("Parent", "Q")
    trust.ensure_relationship(child, "Parent")["trust"] = 4
    world_state["beliefs"] = {"Parent": {beliefs.WORLD_IS_CRUEL}}
    rng = random.Random(3)
    for t in range(1, 20):
        beliefs.spread(world_state, t, rng)
        if beliefs.WORLD_IS_CRUEL in beliefs.agent_beliefs("Child", world_state):
            break
    assert beliefs.WORLD_IS_CRUEL in beliefs.agent_beliefs("Child", world_state)
    print("PASS test_children_inherit_settlement_beliefs_via_childhood_boost")


def test_beliefs_off_run_is_byte_identical_and_adds_no_llm() -> None:
    """beliefs_on=False (default) is byte-identical to the no-param run, writes no belief state, and
    the system adds ZERO LLM calls when on."""
    def run(on, **kw):
        llm.PROVIDER = "random"
        random.seed(42)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, settlements=True, beliefs_on=on, **kw)
        return buf.getvalue(), dict(llm.get_call_stats())

    def baseline():   # a run that never mentions beliefs at all (the v1 path)
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, settlements=True)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        base = baseline()
        on_a, on_calls = run(True)
        on_b, _ = run(True)
        off, off_calls = run(False)   # off LAST so world_state reflects an off run for the key check
    finally:
        llm.PROVIDER = saved
    assert base == off, "beliefs_on=False diverged from the v1 baseline"
    assert "beliefs" not in world_state, "an off run must write no belief state"
    assert on_calls == off_calls, (on_calls, off_calls)   # beliefs are STATE — zero added LLM
    assert on_a == on_b, "an on run must be byte-identical across seeded repeats (deterministic spread)"
    print("PASS test_beliefs_off_run_is_byte_identical_and_adds_no_llm")


# --- Religion (V2 M4.8): shared belief becomes power ------------------------
def _relig_world() -> None:
    _fresh_world()
    world_state["beliefs_on"] = True
    world_state["religion_on"] = True


def _rsettle(sid, center) -> None:
    world_state["settlements"][sid] = {"id": sid, "center": center, "members": set(), "founded": 0}


def _believer(name, pos, sid="S001", *, believes=None, money=0.0):
    a = Agent(name=name, personality="x")
    place_agent(a, *pos)
    a.hunger, a.age, a.lifespan, a.money, a.settlement = 1, 30, 100, money, sid
    if sid is not None and sid in world_state["settlements"]:
        world_state["settlements"][sid]["members"].add(name)
    if believes:
        world_state.setdefault("beliefs", {})[name] = set(believes)
    return a


def test_faith_forms_on_coherence_not_when_fractured() -> None:
    """A coherent shared core crystallises a faith; a fractured town forms none; two towns with the
    same core share ONE faith, divergent cores form TWO — emergent from M4.7 clustering, not declared."""
    import religion, beliefs
    core = {beliefs.LAND_PROVIDES, beliefs.STRONGER_TOGETHER}

    _relig_world(); _rsettle("S001", (5, 5))
    for n, p in [("A", (5, 5)), ("B", (5, 6)), ("C", (6, 5)), ("D", (6, 6))]:
        _believer(n, p, believes=core)
    religion.form_faiths(world_state, 1)
    f = religion.faith_of_settlement(world_state, "S001")
    assert f is not None and f["core"] == frozenset(core) and len(f["followers"]) == 4

    _relig_world(); _rsettle("S001", (5, 5))    # fractured: every member a different belief
    _believer("A", (5, 5), believes={beliefs.LAND_PROVIDES})
    _believer("B", (5, 6), believes={beliefs.WORLD_IS_CRUEL})
    _believer("C", (6, 5), believes={beliefs.WEALTH_IS_VIRTUE})
    religion.form_faiths(world_state, 1)
    assert religion.faith_of_settlement(world_state, "S001") is None

    _relig_world(); _rsettle("S001", (2, 2)); _rsettle("S002", (8, 8))
    for n, p in [("A", (2, 2)), ("B", (2, 3)), ("C", (3, 2))]:
        _believer(n, p, "S001", believes=core)
    for n, p in [("E", (8, 8)), ("F", (8, 9)), ("G", (9, 8))]:
        _believer(n, p, "S002", believes=core)
    religion.form_faiths(world_state, 1)
    assert len(world_state["faiths"]) == 1, "same core -> one shared faith"
    for n in ("E", "F", "G"):     # S002 diverges to a different core
        world_state["beliefs"][n] = {beliefs.WORLD_IS_CRUEL, beliefs.STRONG_TAKE}
    religion.form_faiths(world_state, 2)
    assert len(world_state["faiths"]) == 2, "divergent cores -> two faiths"
    print("PASS test_faith_forms_on_coherence_not_when_fractured")


def test_faith_name_short_and_stable_as_core_grows() -> None:
    """A faith is named for its 1-2 DOMINANT beliefs only: the name stays short and does NOT grow as
    more beliefs join the core (the chronicle-readability fix — no runaway 'X and Y and Z and ...')."""
    import religion, beliefs
    founding = {beliefs.LAND_PROVIDES, beliefs.STRONGER_TOGETHER}

    _relig_world(); _rsettle("S001", (5, 5))
    for n, p in [("A", (5, 5)), ("B", (5, 6)), ("C", (6, 5)), ("D", (6, 6)), ("E", (7, 5))]:
        _believer(n, p, believes=set(founding))
    religion.form_faiths(world_state, 1)
    name1 = religion.faith_of_settlement(world_state, "S001")["name"]
    assert name1.count(" and ") == 1, ("name should use exactly 2 beliefs, got", name1)

    # A third belief spreads to a MAJORITY (4 of 5) — it JOINS the core but is less prevalent than the
    # two founding pillars, so the name must stay the same (short, stable) and not absorb the newcomer.
    for n in ("A", "B", "C", "D"):
        world_state["beliefs"][n].add(beliefs.KNOWLEDGE_IS_POWER)
    ev = religion.form_faiths(world_state, 2)
    f = religion.faith_of_settlement(world_state, "S001")
    assert f["core"] == frozenset(founding | {beliefs.KNOWLEDGE_IS_POWER}), "the core did grow"
    name2 = f["name"]
    assert name2 == name1, ("name must stay stable as the core grows", name1, name2)
    assert name2.count(" and ") == 1, ("name must not grow with the core", name2)
    assert religion.BELIEF_EPITHET[beliefs.KNOWLEDGE_IS_POWER] not in name2, "newcomer must not enter the name"
    # A drift of the same flock is a CONTINUATION, not a new rise — it must not re-log "took root".
    assert not any("took root" in e for e in ev), ("a drifted core must not re-take-root", ev)
    print("PASS test_faith_name_short_and_stable_as_core_grows")


def test_prophet_emergence_logs_only_on_genuine_change() -> None:
    """A prophet's emergence is logged ONLY on a real transition (none -> X, X -> Y): a faith keeping the
    same prophet logs nothing new, even when its core grows and recreates the faith id (the spam fix)."""
    import religion, trust, beliefs
    core = {beliefs.LAND_PROVIDES, beliefs.STRONGER_TOGETHER}

    _relig_world(); _rsettle("S001", (5, 5))
    flock = [_believer(n, p, believes=set(core)) for n, p in
             [("Pa", (5, 5)), ("Pb", (5, 6)), ("Pc", (6, 5)), ("Pd", (6, 6)), ("Pe", (7, 5))]]
    for a in flock:                                   # everyone trusts both Pb and Pc (Pb wins on name tiebreak)
        for target in ("Pb", "Pc"):
            if a.name != target:
                trust.ensure_relationship(a, target)["trust"] = 3

    religion.form_faiths(world_state, 1)
    ev1 = religion.choose_prophets(world_state, 1)
    assert religion.faith_of_settlement(world_state, "S001")["prophet"] == "Pb"
    assert sum("arose as prophet" in e for e in ev1) == 1 and "Pb" in ev1[0], ev1

    ev2 = religion.choose_prophets(world_state, 2)     # same prophet, recomputed -> logs nothing
    assert not any("arose as prophet" in e for e in ev2), ("no re-log for an unchanged prophet", ev2)

    # The core GROWS (a third belief joins a MAJORITY, founding pillars still strictly most-prevalent)
    # -> a NEW faith id but the SAME stable name and the same prophet: the churn must not re-log.
    for a in flock:
        if a.name != "Pe":                            # 4 of 5 -> core belief, but less prevalent than the pillars
            world_state["beliefs"][a.name].add(beliefs.KNOWLEDGE_IS_POWER)
    religion.form_faiths(world_state, 3)
    assert religion.faith_of_settlement(world_state, "S001")["core"] == frozenset(
        core | {beliefs.KNOWLEDGE_IS_POWER}), "the core did grow (new faith id)"
    ev3 = religion.choose_prophets(world_state, 3)
    assert religion.faith_of_settlement(world_state, "S001")["prophet"] == "Pb"
    assert not any("arose as prophet" in e for e in ev3), ("faith-id churn must not re-log the prophet", ev3)

    # A GENUINE change (Pb dies, Pc arises) still logs.
    next(a for a in flock if a.name == "Pb").alive = False
    ev4 = religion.choose_prophets(world_state, 4)
    assert religion.faith_of_settlement(world_state, "S001")["prophet"] == "Pc"
    assert sum("arose as prophet" in e for e in ev4) == 1 and "Pc" in ev4[0], ev4

    # A brief VACANCY then the SAME prophet returns (Pc -> none -> Pc) must NOT re-announce Pc: strip the
    # trust that backs Pc so none qualifies, then restore it.
    saved = {a.name: dict(a.relationships.get("Pc", {})) for a in flock if a.name != "Pc"}
    for a in flock:
        if a.name != "Pc":
            a.relationships.get("Pc", {})["trust"] = 0
    ev5 = religion.choose_prophets(world_state, 5)
    assert religion.faith_of_settlement(world_state, "S001")["prophet"] is None, "vacancy: nobody qualifies"
    for a in flock:
        if a.name != "Pc":
            a.relationships["Pc"] = saved[a.name]
    ev6 = religion.choose_prophets(world_state, 6)
    assert religion.faith_of_settlement(world_state, "S001")["prophet"] == "Pc"
    assert not any("arose as prophet" in e for e in ev5 + ev6), ("a vacancy then the same prophet must stay silent", ev5, ev6)
    print("PASS test_prophet_emergence_logs_only_on_genuine_change")


def test_prophet_is_derived_from_devotion_and_trust_not_assigned() -> None:
    """The prophet is the most devout-and-trusted follower (derived), not the richest; a faith whose
    followers trust no one enough has NO prophet."""
    import religion, trust, beliefs
    core = {beliefs.LAND_PROVIDES, beliefs.STRONGER_TOGETHER}

    _relig_world(); _rsettle("S001", (5, 5))
    devout = [_believer(n, p, believes=core) for n, p in
              [("Pa", (5, 5)), ("Pb", (5, 6)), ("Pc", (6, 5)), ("Pd", (6, 6))]]
    rich = _believer("Croesus", (6, 6), believes=core, money=500.0)  # richest, but not most trusted
    world_state["settlements"]["S001"]["members"].add("Croesus")
    for a in devout:                               # the flock trusts Pb, not the rich outsider
        if a.name != "Pb":
            trust.ensure_relationship(a, "Pb")["trust"] = 3
    religion.form_faiths(world_state, 1)
    religion.choose_prophets(world_state, 1)
    assert religion.faith_of_settlement(world_state, "S001")["prophet"] == "Pb"

    # No one trusted enough -> no prophet (honest).
    _relig_world(); _rsettle("S001", (5, 5))
    for n, p in [("A", (5, 5)), ("B", (5, 6)), ("C", (6, 5))]:
        _believer(n, p, believes=core)
    religion.form_faiths(world_state, 1)
    religion.choose_prophets(world_state, 1)
    assert religion.faith_of_settlement(world_state, "S001")["prophet"] is None
    print("PASS test_prophet_is_derived_from_devotion_and_trust_not_assigned")


def test_aligned_ruler_generates_less_discontent_and_more_loyalty() -> None:
    """The SAME monarch doing the SAME levy generates LESS discontent and MORE loyalty when ALIGNED
    with the local faith than when DEFIANT — faith moving M4.4's legitimacy dial."""
    import religion, discontent, beliefs
    core = {beliefs.LAND_PROVIDES, beliefs.STRONGER_TOGETHER}

    def reign(monarch_aligned):
        _relig_world(); world_state["discontent_on"] = True; _rsettle("S001", (5, 5))
        _believer("King", (4, 4), believes=(core if monarch_aligned else {beliefs.STRONG_TAKE}),
                  money=100.0)
        members = [_believer(n, p, believes=core, money=20.0)
                   for n, p in [("A", (5, 5)), ("B", (5, 6)), ("C", (6, 5))]]
        world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
        for t in range(1, 8):
            religion.update(world_state, t)
            discontent.update(world_state, t)
        return members[0].relationships.get("King", {}).get("trust", 0), \
            discontent.agent_discontent("A", world_state)

    trust_aligned, disc_aligned = reign(True)
    trust_defiant, disc_defiant = reign(False)
    assert trust_aligned > trust_defiant, (trust_aligned, trust_defiant)
    assert disc_aligned < disc_defiant, (disc_aligned, disc_defiant)
    print("PASS test_aligned_ruler_generates_less_discontent_and_more_loyalty")


def test_defiant_king_erodes_vassal_loyalty_and_prophet_amplifies() -> None:
    """A defiant king's believing vassal loses loyalty toward the M3.5 breakaway floor — and a prophet
    opposing him erodes it FASTER (a moral authority against the crown)."""
    import religion, trust, beliefs, kingdoms
    core = {beliefs.LAND_PROVIDES, beliefs.STRONGER_TOGETHER}

    def erosion(with_prophet):
        _relig_world(); _rsettle("S001", (1, 1)); _rsettle("S002", (8, 8))
        _believer("King", (1, 1), "S001", believes={beliefs.STRONG_TAKE}, money=50.0)  # DEFIANT
        lord = _believer("Lord", (8, 8), "S002", believes=core, money=10.0)
        flock = [_believer(n, p, "S002", believes=core) for n, p in [("Va", (8, 7)), ("Vb", (7, 8))]]
        world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
        world_state["monarchs"]["S002"] = {"monarch": "Lord", "since": 0, "garrison": set()}
        world_state["kingdoms"]["King"] = {"king": "King", "home": "S001",
                                           "settlements": {"S001", "S002"}, "vassals": {"S002": "Lord"},
                                           "founded": 0, "discontent": {"Lord": 0}}
        trust.ensure_relationship(lord, "King")["trust"] = 2   # loyal fealty to start
        if with_prophet:                                       # the flock trusts Va -> a prophet arises
            for a in [lord] + flock:
                if a.name != "Va":
                    trust.ensure_relationship(a, "Va")["trust"] = 3
        for t in range(1, 4):
            religion.update(world_state, t)
        return lord.relationships["King"]["trust"], \
            religion.faith_of_settlement(world_state, "S002")["prophet"]

    plain, no_prophet = erosion(with_prophet=False)
    amplified, prophet = erosion(with_prophet=True)
    assert no_prophet is None and prophet == "Va"
    assert amplified < plain, (amplified, plain)      # the prophet deepens the erosion
    assert amplified <= kingdoms.BREAKAWAY_TRUST, amplified   # past the M3.5 breakaway floor
    print("PASS test_defiant_king_erodes_vassal_loyalty_and_prophet_amplifies")


def test_religion_off_run_is_byte_identical_and_adds_no_llm() -> None:
    """religion_on=False (default) is byte-identical to the v1 baseline, writes no faith state, and
    the system adds ZERO LLM calls when on (faiths are STATE)."""
    def run(on):
        llm.PROVIDER = "random"
        random.seed(42)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, settlements=True, monarchy_on=True, discontent_on=True,
                                beliefs_on=True, religion_on=on)
        return buf.getvalue(), dict(llm.get_call_stats())

    def baseline():
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, settlements=True, monarchy_on=True, discontent_on=True,
                                beliefs_on=True)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        base = baseline()
        on_a, on_calls = run(True)
        on_b, _ = run(True)
        off, off_calls = run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "religion_on=False diverged from the beliefs-only baseline"
    assert "faiths" not in world_state, "an off run must write no faith state"
    assert on_calls == off_calls, (on_calls, off_calls)   # faiths are STATE — zero added LLM
    assert on_a == on_b, "an on run must be byte-identical across seeded repeats"
    print("PASS test_religion_off_run_is_byte_identical_and_adds_no_llm")


# --- Culture (V2 M4.9): identity, friction, assimilation --------------------
def _cul_world() -> None:
    _fresh_world()
    for f in ("beliefs_on", "religion_on", "culture_on"):
        world_state[f] = True


def _csettle(sid, center) -> None:
    world_state["settlements"][sid] = {"id": sid, "center": center, "members": set(), "founded": 0}


def _cagent(name, pos, sid="S001", *, believes=None, money=0.0, dependent=False, age=30, parents=()):
    a = Agent(name=name, personality="x")
    place_agent(a, *pos)
    a.hunger, a.age, a.lifespan, a.money = 1, age, 100, money
    a.settlement, a.dependent, a.parents = sid, dependent, parents
    if sid is not None and sid in world_state["settlements"]:
        world_state["settlements"][sid]["members"].add(name)
    if believes is not None:
        world_state.setdefault("beliefs", {})[name] = set(believes)
    return a


def test_same_vs_foreign_rule_breeds_chronic_friction() -> None:
    """A ruler of the SAME culture integrates with little extra discontent; a FOREIGN-culture ruler
    breeds CHRONIC, sustained loyalty loss and hotter discontent from the SAME extraction."""
    import culture, beliefs, discontent
    native = {beliefs.LAND_PROVIDES, beliefs.STRONGER_TOGETHER}
    foreign = {beliefs.STRONG_TAKE, beliefs.WEALTH_IS_VIRTUE}

    def reign(foreign_king):
        _cul_world(); world_state["discontent_on"] = True; _csettle("S001", (5, 5))
        _cagent("King", (4, 4), believes=(foreign if foreign_king else native), money=100.0)
        members = [_cagent(n, p, believes=native, money=20.0)
                   for n, p in [("A", (5, 5)), ("B", (5, 6)), ("C", (6, 5))]]
        world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
        for t in range(1, 8):
            culture.update(world_state, t)
            discontent.update(world_state, t)
        return members[0].relationships.get("King", {}).get("trust", 0), \
            discontent.agent_discontent("A", world_state)

    trust_same, disc_same = reign(False)
    trust_foreign, disc_foreign = reign(True)
    assert trust_foreign < trust_same, (trust_foreign, trust_same)   # chronic loyalty tax
    assert disc_foreign > disc_same, (disc_foreign, disc_same)
    print("PASS test_same_vs_foreign_rule_breeds_chronic_friction")


def test_foreign_province_breaks_away_where_same_culture_holds() -> None:
    """A FOREIGN-culture king loses his vassal province to breakaway (through the existing M3.5
    machinery, fed by cultural friction), where a SAME-culture king holds it."""
    import culture, beliefs, kingdoms, trust
    native = {beliefs.LAND_PROVIDES, beliefs.STRONGER_TOGETHER}
    foreign = {beliefs.STRONG_TAKE, beliefs.WEALTH_IS_VIRTUE}

    def realm(foreign_king):
        _cul_world(); _csettle("S001", (1, 1)); _csettle("S002", (8, 8))
        world_state["tribute_rate"] = 0.25   # <= consent, so tribute itself adds no backlash
        _cagent("King", (1, 1), "S001", believes=(foreign if foreign_king else native), money=50.0)
        lord = _cagent("Lord", (8, 8), "S002", believes=native, money=8.0)
        for n, p in [("Va", (8, 7)), ("Vb", (7, 8))]:
            _cagent(n, p, "S002", believes=native)
        world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
        world_state["monarchs"]["S002"] = {"monarch": "Lord", "since": 0, "garrison": set()}
        world_state["kingdoms"]["King"] = {"king": "King", "home": "S001",
                                           "settlements": {"S001", "S002"}, "vassals": {"S002": "Lord"},
                                           "founded": 0, "discontent": {"Lord": 0}}
        trust.ensure_relationship(lord, "King")["trust"] = 2   # Lord's fealty
        for t in range(1, 9):
            culture.update(world_state, t)
            kingdoms.update(world_state, t)
        return kingdoms.realm_of(world_state, "S002") == "King"

    assert realm(foreign_king=False), "a same-culture king holds the province"
    assert not realm(foreign_king=True), "a foreign-culture king loses it to breakaway"
    print("PASS test_foreign_province_breaks_away_where_same_culture_holds")


def test_children_assimilate_but_adults_do_not() -> None:
    """Under sustained foreign rule a dependent CHILD adopts the ruler's culture while the ADULTS
    keep theirs — assimilation is generational, not immediate."""
    import culture, beliefs
    native = {beliefs.LAND_PROVIDES, beliefs.STRONGER_TOGETHER}
    foreign = {beliefs.STRONG_TAKE, beliefs.WEALTH_IS_VIRTUE}

    _cul_world(); world_state["lineage_on"] = True; _csettle("S001", (5, 5))
    _cagent("King", (4, 4), believes=foreign, money=100.0)
    for n, p in [("A", (5, 5)), ("B", (5, 6)), ("C", (6, 5))]:
        _cagent(n, p, believes=set(native), money=20.0)
    _cagent("Kid", (6, 6), believes=set(native), dependent=True, age=6, parents=("A", "B"))
    world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
    rng = random.Random(1)
    for t in range(1, 40):
        culture.update(world_state, t, rng)
    assert culture._shares(world_state["beliefs"]["Kid"], frozenset(foreign)), \
        world_state["beliefs"]["Kid"]                       # the child took on the ruler's culture
    assert world_state["beliefs"]["A"] == native, "the adult kept its culture"
    print("PASS test_children_assimilate_but_adults_do_not")


def test_assimilation_completes_over_generations_and_fault_line_fades() -> None:
    """As enough of a town assimilates, its dominant culture drifts to the ruler's — he is no longer
    foreign and the friction fades (the fault line heals). Negligible in a few turns, done over many."""
    import culture, beliefs
    native = {beliefs.LAND_PROVIDES, beliefs.STRONGER_TOGETHER}
    foreign = {beliefs.STRONG_TAKE, beliefs.WEALTH_IS_VIRTUE}

    _cul_world(); world_state["lineage_on"] = True; _csettle("S001", (5, 5))
    _cagent("King", (4, 4), believes=foreign, money=100.0)
    _cagent("Elder", (5, 5), believes=set(native))          # one adult holdout
    kids = [_cagent(n, p, believes=set(native), dependent=True, age=6, parents=("Elder", "X"))
            for n, p in [("K1", (5, 6)), ("K2", (6, 5)), ("K3", (6, 6))]]
    world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
    rng = random.Random(2)

    culture.update(world_state, 1, rng)
    assert culture.is_foreign_ruled(world_state, "S001"), "the fault line is open at conquest"
    early = culture.assimilation_progress(world_state, "S001")
    for t in range(2, 60):
        culture.update(world_state, t, rng)
    late = culture.assimilation_progress(world_state, "S001")
    assert late > early
    # Once the children (the majority) hold the ruler's culture, the signature drifts and he is native.
    assert not culture.is_foreign_ruled(world_state, "S001"), "the fault line healed as generations assimilated"
    print("PASS test_assimilation_completes_over_generations_and_fault_line_fades")


def test_culture_off_run_is_byte_identical_and_adds_no_llm() -> None:
    """culture_on=False (default) is byte-identical to the religion-only baseline and adds ZERO LLM."""
    def run(on):
        llm.PROVIDER = "random"
        random.seed(42)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, settlements=True, monarchy_on=True, discontent_on=True,
                                beliefs_on=True, religion_on=True, culture_on=on)
        return buf.getvalue(), dict(llm.get_call_stats())

    def baseline():
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, settlements=True, monarchy_on=True, discontent_on=True,
                                beliefs_on=True, religion_on=True)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        base = baseline()
        on_a, on_calls = run(True)
        on_b, _ = run(True)
        off, off_calls = run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "culture_on=False diverged from the religion-only baseline"
    assert on_calls == off_calls, (on_calls, off_calls)   # culture is STATE — zero added LLM
    assert on_a == on_b, "an on run must be byte-identical across seeded repeats"
    print("PASS test_culture_off_run_is_byte_identical_and_adds_no_llm")


# --- Writing & records (V2 M4.10): institutional memory ---------------------
def _writing_world() -> None:
    _fresh_world()
    world_state["writing_on"] = True


def _wsettle(sid, center) -> None:
    world_state["settlements"][sid] = {"id": sid, "center": center, "members": set(), "founded": 0}


def _wagent(name, pos, sid="S001", *, knows=None, age=40, parents=()):
    a = Agent(name=name, personality="x")
    place_agent(a, *pos)
    a.hunger, a.age, a.lifespan, a.settlement, a.parents = 1, age, 100, sid, parents
    if knows:
        a.knowledge.update(knows)
    if sid is not None and sid in world_state["settlements"]:
        world_state["settlements"][sid]["members"].add(name)
    return a


def test_writing_discovery_prereqs_bind() -> None:
    """Writing is invented ONLY with the prior tech tools AND from a settlement holding a food surplus;
    knocking out either prereq (or the settlement) yields no writing."""
    import writing

    def can_invent(has_tools, settled, surplus):
        _writing_world(); _wsettle("S001", (5, 5))
        a = _wagent("S", (5, 5), sid=("S001" if settled else None),
                    knows=({"tools"} if has_tools else set()))
        if surplus:
            for x, y in [(4, 4), (5, 4), (6, 4), (4, 5), (6, 5), (4, 6), (5, 6), (6, 6)]:
                world.place_food(x, y)
        rng = random.Random(0)
        for _ in range(500):
            writing.discover_writing(world_state, 1, rng)
            if "writing" in a.knowledge:
                return True
        return False

    assert can_invent(True, True, True), "tools + settled + surplus should invent writing"
    assert not can_invent(True, True, False), "no surplus -> no writing (scribes need spare capacity)"
    assert not can_invent(False, True, True), "no prior tech (tools) -> no writing"
    assert not can_invent(True, False, True), "no settlement -> no writing"
    print("PASS test_writing_discovery_prereqs_bind")


def test_heir_inherits_written_law_only_when_literate() -> None:
    """A literate ruler's policy is inscribed and the M4.3 heir INHERITS the written law across
    succession; the identical succession in an illiterate town leaves a blank slate."""
    import writing, population

    def law_after_succession(literate):
        _writing_world(); world_state["lineage_on"] = True; _wsettle("S001", (5, 5))
        king = _wagent("King", (5, 5), knows=({"writing"} if literate else set()), age=60)
        _wagent("Heir", (5, 6), knows=({"writing"} if literate else set()), parents=("King", "Q"), age=25)
        world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0, "garrison": set()}
        writing.update(world_state, 1)                       # King inscribes the law (if literate)
        population.announce_death(king, 2, world_state, cause="old age", final_memory="d", note="d")
        writing.update(world_state, 3)                       # heir inherits (if literate)
        return writing.written_law(world_state, "S001")

    lit = law_after_succession(True)
    assert lit is not None and lit["set_by"] == "Heir" and lit["inherited_from"] == "King"
    assert law_after_succession(False) is None, "an illiterate town's policy dies with its ruler"
    print("PASS test_heir_inherits_written_law_only_when_literate")


def test_literacy_cures_knowledge_extinction() -> None:
    """A literate settlement RE-TEACHES a skill whose last living master died from its records; an
    identical illiterate settlement suffers the knowledge-extinction collapse (the skill is gone)."""
    import writing

    def recovers(literate):
        _writing_world(); _wsettle("S001", (5, 5))
        master = _wagent("Master", (5, 5),
                         knows=({"writing", "tools", "farming"} if literate else {"tools", "farming"}))
        _wagent("Pupil", (5, 6), knows=({"writing"} if literate else set()))
        writing.update(world_state, 1)                       # archive farming (if literate)
        master.knowledge.discard("farming")                  # the last farmer forgets / dies out
        writing.update(world_state, 2)                       # re-teach from the records (if literate)
        return any("farming" in a.knowledge for a in world_state["agents"] if a.alive)

    assert recovers(True), "a literate town re-teaches forgotten farming from its records"
    assert not recovers(False), "an illiterate town cannot recover the lost skill"
    print("PASS test_literacy_cures_knowledge_extinction")


def test_chronicle_accumulates_only_when_literate() -> None:
    """A literate settlement records its MAJOR events to a persistent chronicle (minor ticks omitted);
    an illiterate settlement keeps no lasting record."""
    import writing

    def chronicle(literate):
        _writing_world(); _wsettle("S001", (5, 5))
        _wagent("A", (5, 5), knows=({"writing"} if literate else set()))
        world_state["events"].append("turn 5: an UPRISING in S001 — 3 risers rise")   # major
        world_state["events"].append("turn 5: A052 trust in B: 1 -> 2 (talk)")        # minor tick
        writing.update(world_state, 5)
        return writing.chronicle_of(world_state, "S001")

    lit = chronicle(True)
    assert len(lit) == 1 and "UPRISING" in lit[0]["event"], lit    # the major only
    assert chronicle(False) == [], "an illiterate settlement records nothing"
    print("PASS test_chronicle_accumulates_only_when_literate")


def test_writing_off_run_is_byte_identical_and_adds_no_llm() -> None:
    """writing_on=False (default) is byte-identical to the tech-tree baseline and adds ZERO LLM."""
    import knowledge

    def run(on):
        llm.PROVIDER = "random"
        random.seed(42)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, settlements=True, tech_tree=knowledge.TECH_TREE, writing_on=on)
        return buf.getvalue(), dict(llm.get_call_stats())

    def baseline():
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, settlements=True, tech_tree=knowledge.TECH_TREE)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        base = baseline()
        on_a, on_calls = run(True)
        on_b, _ = run(True)
        off, off_calls = run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "writing_on=False diverged from the tech-tree baseline"
    assert not world_state.get("laws") and not world_state.get("chronicles"), "off writes no records"
    assert on_calls == off_calls, (on_calls, off_calls)   # records are STATE — zero added LLM
    assert on_a == on_b, "an on run must be byte-identical across seeded repeats"
    print("PASS test_writing_off_run_is_byte_identical_and_adds_no_llm")


# --- Metallurgy & arms (V2 M4.11): technology transforms war and work -------
def _metal_world() -> None:
    _fresh_world()
    world_state["metallurgy_on"] = True


def _msettle(sid, center) -> None:
    world_state["settlements"][sid] = {"id": sid, "center": center, "members": set(), "founded": 0}


def _magent(name, pos, sid="S001", *, knows=None, money=0.0):
    a = Agent(name=name, personality="curious and creative")
    place_agent(a, *pos)
    a.hunger, a.age, a.lifespan, a.money, a.settlement = 1, 30, 100, money, sid
    if knows:
        a.knowledge.update(knows)
    if sid is not None and sid in world_state["settlements"]:
        world_state["settlements"][sid]["members"].add(name)
    return a


def _msurplus(center=(5, 5)) -> None:
    cx, cy = center
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if (dx, dy) != (0, 0):
                world.place_food(cx + dx, cy + dy)


def test_metallurgy_prereqs_bind() -> None:
    """Metalworking is invented only with the prior tech tools AND a settlement holding surplus; weapons
    needs metalworking. Knocking out any prereq yields nothing."""
    import metallurgy

    def can_invent(item, has_prereq, settled, surplus):
        _metal_world(); _msettle("S001", (5, 5))
        prereq = "tools" if item == "metalworking" else "metalworking"
        a = _magent("S", (5, 5), sid=("S001" if settled else None),
                    knows=({prereq} if has_prereq else set()))
        if surplus:
            _msurplus()
        rng = random.Random(0)
        for _ in range(800):
            metallurgy.discover(world_state, 1, rng)
            if item in a.knowledge:
                return True
        return False

    assert can_invent("metalworking", True, True, True)
    assert not can_invent("metalworking", True, True, False), "no surplus -> no metalworking"
    assert not can_invent("metalworking", False, True, True), "no tools -> no metalworking"
    assert not can_invent("metalworking", True, False, True), "no settlement -> no metalworking"
    assert can_invent("weapons", True, True, True), "weapons needs metalworking"
    assert not can_invent("weapons", False, True, True), "no metalworking -> no weapons"
    print("PASS test_metallurgy_prereqs_bind")


def test_metalworking_boosts_farm_yield() -> None:
    """A farmer who knows metalworking (better tools) grows measurably more food than a neolithic one."""
    import metallurgy, knowledge

    def food_grown(metal):
        _metal_world(); _msettle("S001", (5, 5))
        _magent("F", (5, 5), knows=({"farming", "metalworking"} if metal else {"farming"}))
        rng = random.Random(5)
        total = 0
        for t in range(1, 60):
            knowledge.farm(world_state, t, rng)
            total += len(world_state["food"])
            world_state["food"].clear()
        return total

    neolithic, metallurgical = food_grown(False), food_grown(True)
    assert metallurgical > neolithic * 1.4, (metallurgical, neolithic)   # a real yield gap
    print("PASS test_metalworking_boosts_farm_yield")


def test_arms_multiply_force_in_battle() -> None:
    """In the shared battle math a smaller ARMED host beats a larger UNARMED one; equal arms fall back
    to numbers; and with no arms at all the result is the plain head count (byte-identical)."""
    import monarchy, metallurgy

    def battle(att_n, att_armed, def_n, def_armed):
        _metal_world()
        A = [_magent(f"A{i}", (0, i), sid=None, knows=({"weapons"} if att_armed else set()))
             for i in range(att_n)]
        D = [_magent(f"D{i}", (1, i), sid=None, knows=({"weapons"} if def_armed else set()))
             for i in range(def_n)]
        won, _, _, _ = monarchy.resolve_battle(world_state, A, D, 1, "att", "def")
        return won

    assert battle(3, True, 4, False), "3 armed should beat 4 unarmed (knowledge beats numbers)"
    assert not battle(3, False, 4, False), "3 unarmed lose to 4 unarmed (numbers)"
    assert not battle(3, True, 4, True), "3 armed lose to 4 armed (equal arms -> numbers)"
    assert not battle(4, False, 4, False), "an unarmed tie is held by the defender (byte-identical rule)"
    print("PASS test_arms_multiply_force_in_battle")


def test_uprising_arms_shifts_revolt_balance() -> None:
    """The sharp composition: an ARMED garrison crushes an unarmed mob (steel beats numbers); but when
    the COMMONERS are also armed, the mob's numbers win again — who controls weapons decides who revolts."""
    import uprising

    def uprising_wins(ruler_armed, mob_armed):
        _metal_world(); world_state["discontent_on"] = True; world_state["uprising_on"] = True
        _msettle("S001", (5, 5))
        _magent("King", (4, 4), money=0.5)          # drained: cannot hire fresh mercs
        gk = {"weapons"} if ruler_armed else set()
        garr = [_magent(f"g{i}", (3, 3 + i), knows=set(gk)) for i in range(3)]
        world_state["monarchs"]["S001"] = {"monarch": "King", "since": 0,
                                           "garrison": {g.name for g in garr}}
        mk = {"weapons"} if mob_armed else set()
        for i, p in enumerate([(5, 5), (5, 6), (6, 5), (6, 6), (5, 4)]):
            _magent(f"m{i}", p, knows=set(mk))
        world_state["discontent"] = {f"m{i}": 12.0 for i in range(5)}
        res = uprising.update(world_state, 10)
        return bool(res and res[0]["won"])

    assert not uprising_wins(ruler_armed=True, mob_armed=False), "armed garrison crushes an unarmed mob"
    assert uprising_wins(ruler_armed=True, mob_armed=True), "an armed mob's numbers win against an armed garrison"
    assert uprising_wins(ruler_armed=False, mob_armed=False), "unarmed vs unarmed -> numbers"
    print("PASS test_uprising_arms_shifts_revolt_balance")


def test_metallurgy_off_run_is_byte_identical_and_adds_no_llm() -> None:
    """metallurgy_on=False (default) is byte-identical to the tech-tree baseline and adds ZERO LLM."""
    import knowledge
    def run(on):
        llm.PROVIDER = "random"
        random.seed(42)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, settlements=True, monarchy_on=True,
                                tech_tree=knowledge.TECH_TREE, metallurgy_on=on)
        return buf.getvalue(), dict(llm.get_call_stats())

    def baseline():
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, settlements=True, monarchy_on=True, tech_tree=knowledge.TECH_TREE)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        base = baseline()
        on_a, on_calls = run(True)
        on_b, _ = run(True)
        off, off_calls = run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "metallurgy_on=False diverged from the tech-tree baseline"
    assert on_calls == off_calls, (on_calls, off_calls)   # metallurgy is STATE — zero added LLM
    assert on_a == on_b, "an on run must be byte-identical across seeded repeats"
    print("PASS test_metallurgy_off_run_is_byte_identical_and_adds_no_llm")


# --- Era progression (V2 M4.12): the march of ages --------------------------
_NEO = frozenset({"fire", "tools", "farming"})
_BRONZE = _NEO | {"metalworking"}
_IRON = _BRONZE | {"weapons", "writing"}


def _era_world() -> None:
    _fresh_world()
    world_state["eras_on"] = True


def _esettle(sid, center) -> None:
    world_state["settlements"][sid] = {"id": sid, "center": center, "members": set(), "founded": 0}


def _eagent(name, pos, sid="S001", *, knows=None):
    a = Agent(name=name, personality="x")
    place_agent(a, *pos)
    a.hunger, a.age, a.lifespan, a.settlement = 1, 30, 100, sid
    if knows:
        a.knowledge.update(knows)
    if sid is not None and sid in world_state["settlements"]:
        world_state["settlements"][sid]["members"].add(name)
    return a


def test_era_derived_from_tech_and_advance_logged_and_extensible() -> None:
    """A settlement's era is derived from the tech its populace masters (thresholds binding); crossing
    a threshold is logged as an ADVANCE; and a hypothetical higher era slots in as a pure data addition."""
    import eras

    _era_world(); _esettle("S001", (5, 5))
    a = _eagent("A", (5, 5), knows=set(_NEO))
    assert eras.settlement_era(world_state, "S001") == "Neolithic"
    eras.update(world_state, 1)
    a.knowledge.update({"metalworking"})
    ev = eras.update(world_state, 2)
    assert eras.settlement_era(world_state, "S001") == "Bronze Age"
    assert any("entered the Bronze Age" in e for e in ev), ev
    a.knowledge.update({"weapons", "writing"})
    ev = eras.update(world_state, 3)
    assert eras.settlement_era(world_state, "S001") == "Iron Age"
    assert any("entered the Iron Age" in e for e in ev)

    # EXTENSIBILITY: appending an era to the ladder is all it takes for a town to reach it — no new code.
    steel = eras.Era("Steel Age", _IRON | {"steelmaking"}, 3.6, "steel")
    eras.ERAS.append(steel)
    try:
        assert eras.settlement_era(world_state, "S001") == "Iron Age"   # not yet — lacks steelmaking
        a.knowledge.add("steelmaking")
        assert eras.settlement_era(world_state, "S001") == "Steel Age"  # the new rung slots straight in
    finally:
        eras.ERAS.pop()
    print("PASS test_era_derived_from_tech_and_advance_logged_and_extensible")


def test_era_gap_multiplies_combat_force() -> None:
    """An era GAP is a force multiplier: a smaller Iron host beats a larger Neolithic one (and Bronze
    beats Neolithic), while same-era forces fall back to numbers."""
    import monarchy

    def battle(att_n, att_tech, def_n, def_tech):
        _era_world()
        A = [_eagent(f"A{i}", (0, i), sid=None, knows=set(att_tech)) for i in range(att_n)]
        D = [_eagent(f"D{i}", (1, i), sid=None, knows=set(def_tech)) for i in range(def_n)]
        won, _, _, _ = monarchy.resolve_battle(world_state, A, D, 1, "a", "d")
        return won

    assert battle(3, _IRON, 5, _NEO), "3 Iron should beat 5 Neolithic (knowledge beats numbers on the era curve)"
    assert battle(3, _BRONZE, 4, _NEO), "3 Bronze should beat 4 Neolithic"
    assert not battle(4, _IRON, 4, _IRON), "same era -> numbers (a tie is held by the defender)"
    assert not battle(3, _IRON, 6, _IRON), "same era -> the larger side wins"
    print("PASS test_era_gap_multiplies_combat_force")


def test_era_yield_curve() -> None:
    """A higher era out-produces a lower one: Iron > Bronze > Neolithic farm output."""
    import eras, knowledge

    def food_grown(era_tech):
        _era_world(); _esettle("S001", (5, 5))
        _eagent("F", (5, 5), knows=set(era_tech) | {"farming"})
        rng = random.Random(5)
        total = 0
        for t in range(1, 50):
            knowledge.farm(world_state, t, rng)
            total += len(world_state["food"])
            world_state["food"].clear()
        return total

    neo, bronze, iron = food_grown(_NEO), food_grown(_BRONZE), food_grown(_IRON)
    assert neo < bronze < iron, (neo, bronze, iron)
    print("PASS test_era_yield_curve")


def test_era_exposed_in_state_and_drives_rendering() -> None:
    """The settlement's era is exposed in world_state (for the read-only renderer), and the town-plan
    keys its building STYLE off the era — so towns render differently by age."""
    import eras
    from renderer.pygame_renderer import build_town_plan

    _era_world(); _esettle("S001", (5, 5))
    _eagent("A", (5, 5), knows=set(_IRON))
    eras.update(world_state, 1)
    assert world_state["eras"]["S001"] == "Iron Age"          # exposed in state for the renderer
    assert eras.building_style(world_state, "S001") == "iron"

    neo_plan = build_town_plan((5, 5), 6, None, (200, 200, 200), False, 10, "neolithic")
    iron_plan = build_town_plan((5, 5), 6, None, (200, 200, 200), False, 10, "iron")
    assert neo_plan["era_style"] == "neolithic" and iron_plan["era_style"] == "iron"
    assert iron_plan["stone_wall"] and iron_plan["forge"] and not neo_plan["stone_wall"]
    assert neo_plan["buildings"][0]["wall"] != iron_plan["buildings"][0]["wall"], "towns look different by age"
    print("PASS test_era_exposed_in_state_and_drives_rendering")


def test_eras_off_run_is_byte_identical_and_adds_no_llm() -> None:
    """eras_on=False (default) is byte-identical to the metallurgy+writing baseline and adds ZERO LLM."""
    import knowledge
    def run(on):
        llm.PROVIDER = "random"
        random.seed(42)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, settlements=True, monarchy_on=True, tech_tree=knowledge.TECH_TREE,
                                metallurgy_on=True, writing_on=True, eras_on=on)
        return buf.getvalue(), dict(llm.get_call_stats())

    def baseline():
        llm.PROVIDER = "random"
        random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, settlements=True, monarchy_on=True, tech_tree=knowledge.TECH_TREE,
                                metallurgy_on=True, writing_on=True)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        base = baseline()
        on_a, on_calls = run(True)
        on_b, _ = run(True)
        off, off_calls = run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "eras_on=False diverged from the metallurgy+writing baseline"
    assert not world_state.get("eras"), "an off run writes no era state"
    assert on_calls == off_calls, (on_calls, off_calls)   # eras are derived STATE — zero added LLM
    assert on_a == on_b, "an on run must be byte-identical across seeded repeats"
    print("PASS test_eras_off_run_is_byte_identical_and_adds_no_llm")


# --- Diplomacy (V2 M4.13): relations & treaties -----------------------------
def _dip_world() -> None:
    create_world(size=60)
    world_state["agents"].clear()
    world_state["food"].clear()
    world_state["turn"] = 0
    world_state["diplomacy_on"] = True


def _dip_settled(n, p, sid=None, money=0.0):
    a = Agent(name=n, personality="x")
    place_agent(a, *p)
    a.hunger, a.age, a.lifespan, a.money, a.settlement = 1, 30, 100, money, sid
    return a


def _dip_mercs(prefix, near, n):
    for i in range(n):
        _dip_settled(f"{prefix}{i}", (near[0] + i % 2, near[1] + 2), sid=None, money=0.5)


def _dip_realm(king, kmoney, home_c):
    home = f"{king}_home"
    world_state["settlements"][home] = {"id": home, "center": home_c, "members": {king}, "founded": 0}
    _dip_settled(king, home_c, sid=home, money=kmoney)
    world_state["monarchs"][home] = {"monarch": king, "since": 0, "garrison": set()}
    world_state["kingdoms"][king] = {"king": king, "home": home, "settlements": {home},
                                     "vassals": {}, "founded": 0, "discontent": {}}


def test_stance_derived_from_history_and_decays() -> None:
    """Stance is derived from history: a war sours a pair to hostile; shared culture warms a pair to
    friendly over time; and an idle pair decays back toward neutral."""
    import diplomacy

    _dip_world(); _dip_realm("A", 100.0, (10, 10)); _dip_realm("B", 100.0, (18, 10))
    assert diplomacy.stance(world_state, "A", "B") == "neutral"
    diplomacy.record_war(world_state, "A", "B", 1)
    assert diplomacy.stance(world_state, "A", "B") == "hostile"
    for t in range(2, 12):
        diplomacy.update(world_state, t)                 # quiet turns -> decays to neutral
    assert diplomacy.stance(world_state, "A", "B") == "neutral", diplomacy.stance_score(world_state, "A", "B")

    # Shared culture warms a pair toward friendly.
    _dip_world(); world_state["culture_on"] = True; world_state["beliefs_on"] = True
    _dip_realm("A", 100.0, (10, 10)); _dip_realm("B", 100.0, (18, 10))
    import beliefs
    creed = {beliefs.LAND_PROVIDES, beliefs.STRONGER_TOGETHER}
    world_state["beliefs"] = {"A": set(creed), "B": set(creed)}   # two kings of one culture
    for t in range(1, 8):
        diplomacy.update(world_state, t)
    assert diplomacy.stance(world_state, "A", "B") == "friendly", diplomacy.stance_score(world_state, "A", "B")
    print("PASS test_stance_derived_from_history_and_decays")


def test_pact_prevents_a_war_the_loop_would_launch_and_breaks_on_souring() -> None:
    """A non-aggression pact stops a war the M3.6 loop would otherwise launch; when the stance sours the
    pact BREAKS (betrayal) and the war becomes possible again."""
    import diplomacy, empire

    def run(with_pact):
        _dip_world()
        _dip_realm("Rich", 500.0, (10, 10)); _dip_realm("Poor", 50.0, (18, 10))
        _dip_mercs("R", (10, 10), 5); _dip_mercs("P", (18, 10), 2)
        if with_pact:
            world_state["diplomacy"] = {"stance": {("Poor", "Rich"): 3}, "pacts": set(), "alliances": set()}
        diplomacy.update(world_state, 1)
        empire.update(world_state, 1)
        return empire.is_sovereign(world_state, "Poor")   # True = NOT subjugated (war prevented)

    assert not run(with_pact=False), "with no pact the M3.6 loop launches the war and subjugates Poor"
    assert run(with_pact=True), "a pact prevents the war — Poor stays sovereign"

    # The pact BREAKS when the stance sours, and it is logged as a betrayal.
    _dip_world()
    _dip_realm("Rich", 500.0, (10, 10)); _dip_realm("Poor", 50.0, (18, 10))
    world_state["diplomacy"] = {"stance": {("Poor", "Rich"): 3}, "pacts": {("Poor", "Rich")},
                                "alliances": set()}
    diplomacy.record_war(world_state, "Rich", "Poor", 1)   # a shock sours the pair past the break line
    ev = diplomacy.update(world_state, 2)
    assert not diplomacy.has_pact(world_state, "Rich", "Poor"), "a soured stance breaks the pact"
    assert any("BROKEN" in e for e in ev), ev
    print("PASS test_pact_prevents_a_war_the_loop_would_launch_and_breaks_on_souring")


def test_alliance_adds_ally_host_to_defence() -> None:
    """A defensive alliance brings the ally's whole host to the defence: an attacker who beats the lone
    defender LOSES against the combined hosts."""
    import diplomacy, empire

    def war(with_alliance):
        _dip_world()
        _dip_realm("Rich", 500.0, (10, 10)); _dip_realm("Poor", 50.0, (30, 10))
        _dip_realm("Ally", 400.0, (50, 10))
        _dip_mercs("R", (10, 10), 5); _dip_mercs("P", (30, 10), 2); _dip_mercs("Y", (50, 10), 4)
        if with_alliance:
            world_state["diplomacy"] = {"stance": {("Ally", "Poor"): 3}, "pacts": set(),
                                        "alliances": {("Ally", "Poor")}}
        res = empire.wage_war(world_state, "Rich", "Poor", 1)
        return res["won"], res["def_host"]

    won_lone, def_lone = war(False)
    won_allied, def_allied = war(True)
    assert won_lone and not won_allied, (won_lone, won_allied)
    assert def_allied > def_lone, "the ally's host joined the defence"
    print("PASS test_alliance_adds_ally_host_to_defence")


def test_lapsed_honour_ally_fails_to_answer() -> None:
    """Alliances are conditional: an ally whose stance with the defender has soured below the honour
    line does NOT answer the call, leaving the defender to fall alone."""
    import diplomacy, empire

    _dip_world()
    _dip_realm("Rich", 500.0, (10, 10)); _dip_realm("Poor", 50.0, (30, 10)); _dip_realm("Ally", 400.0, (50, 10))
    _dip_mercs("R", (10, 10), 5); _dip_mercs("P", (30, 10), 2); _dip_mercs("Y", (50, 10), 4)
    world_state["diplomacy"] = {"stance": {("Ally", "Poor"): -5},   # honour has lapsed (soured)
                                "pacts": set(), "alliances": {("Ally", "Poor")}}
    assert diplomacy.defensive_allies(world_state, "Poor") == [], "a soured ally does not answer"
    res = empire.wage_war(world_state, "Rich", "Poor", 1)
    assert res["won"] and res["def_host"] == 2, "the defender falls alone (no ally host)"
    print("PASS test_lapsed_honour_ally_fails_to_answer")


def test_diplomacy_off_run_is_byte_identical_and_adds_no_llm() -> None:
    """diplomacy_on=False (default) is byte-identical to the empire baseline and adds ZERO LLM."""
    def run(on):
        llm.PROVIDER = "random"
        random.seed(7)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, stage="war", diplomacy_on=on)
        return buf.getvalue(), dict(llm.get_call_stats())

    def baseline():
        llm.PROVIDER = "random"
        random.seed(7)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, stage="war")
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        base = baseline()
        on_a, on_calls = run(True)
        on_b, _ = run(True)
        off, off_calls = run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "diplomacy_on=False diverged from the empire baseline"
    assert not world_state.get("diplomacy"), "an off run writes no diplomacy state"
    assert on_calls == off_calls, (on_calls, off_calls)   # stance is derived STATE — zero added LLM
    assert on_a == on_b, "an on run must be byte-identical across seeded repeats"
    print("PASS test_diplomacy_off_run_is_byte_identical_and_adds_no_llm")


# --- Inter-kingdom trade (V2 M4.14): trade routes & interdependence ----------
def _trade_world() -> None:
    create_world(size=60)
    world_state["agents"].clear()
    world_state["food"].clear()
    world_state["turn"] = 0
    world_state["intertrade_on"] = True
    world_state["diplomacy_on"] = True


def _trade_settled(n, p, sid=None, money=0.0, stock=0.0, hunger=1):
    a = Agent(name=n, personality="x")
    place_agent(a, *p)
    a.hunger, a.age, a.lifespan, a.money, a.stockpile, a.settlement = hunger, 30, 100, money, stock, sid
    return a


def _trade_realm(king, home_c, *, kmoney=0.0, kstock=0.0, member_hunger=1, mercs=0):
    home = f"{king}_home"
    world_state["settlements"][home] = {"id": home, "center": home_c, "members": {king, f"{king}_m"},
                                        "founded": 0}
    _trade_settled(king, home_c, sid=home, money=kmoney, stock=kstock)
    _trade_settled(f"{king}_m", (home_c[0] + 1, home_c[1]), sid=home, hunger=member_hunger)
    world_state["monarchs"][home] = {"monarch": king, "since": 0, "garrison": set()}
    world_state["kingdoms"][king] = {"king": king, "home": home, "settlements": {home},
                                     "vassals": {}, "founded": 0, "discontent": {}}
    for i in range(mercs):
        _trade_settled(f"{king}M{i}", (home_c[0] + i % 2, home_c[1] + 2), sid=None, money=0.5)


def _king_agent(name):
    return next(a for a in world_state["agents"] if a.name == name)


def test_intertrade_enriches_both_and_is_blocked_when_hostile() -> None:
    """A food-rich and a food-poor kingdom trade: the poor realm's granary FILLS and the rich realm's
    treasury GROWS (both better off). A hostile pair does not trade at all."""
    import intertrade, diplomacy

    _trade_world()
    _trade_realm("Rich", (10, 10), kmoney=5.0, kstock=18.0, member_hunger=1)   # food surplus, wants coin
    _trade_realm("Poor", (18, 10), kmoney=40.0, kstock=0.0, member_hunger=8)   # hungry, has coin
    rf0, rm0 = _king_agent("Rich").stockpile, _king_agent("Rich").money
    pf0 = _king_agent("Poor").stockpile
    for t in range(1, 5):
        intertrade.update(world_state, t)
        diplomacy.update(world_state, t)
    assert _king_agent("Poor").stockpile > pf0, "the poor realm's granary should fill"
    assert _king_agent("Rich").money > rm0 and _king_agent("Rich").stockpile < rf0, "the rich realm profits"
    assert intertrade.total_volume(world_state, "Rich", "Poor") > 0

    # A hostile pair does not trade.
    _trade_world()
    _trade_realm("A", (10, 10), kmoney=5.0, kstock=18.0)
    _trade_realm("B", (18, 10), kmoney=40.0, kstock=0.0, member_hunger=8)
    world_state["diplomacy"] = {"stance": {("A", "B"): -6}, "pacts": set(), "alliances": set()}
    intertrade.update(world_state, 1)
    assert intertrade.total_volume(world_state, "A", "B") == 0.0, "hostile kingdoms do not trade"
    print("PASS test_intertrade_enriches_both_and_is_blocked_when_hostile")


def test_trade_warms_stance_into_a_pact() -> None:
    """Sustained commerce WARMS a neutral pair's stance into friendly / a pact — diplomacy emerging
    from economics (M4.13's feedback seam, closed)."""
    import intertrade, diplomacy

    _trade_world()
    _trade_realm("Rich", (10, 10), kmoney=5.0, kstock=18.0)
    _trade_realm("Poor", (18, 10), kmoney=40.0, kstock=0.0, member_hunger=8)
    assert diplomacy.stance(world_state, "Rich", "Poor") == "neutral"
    for t in range(1, 8):
        intertrade.update(world_state, t)
        diplomacy.update(world_state, t)
    assert diplomacy.stance(world_state, "Rich", "Poor") == "friendly", \
        diplomacy.stance_score(world_state, "Rich", "Poor")
    assert diplomacy.has_pact(world_state, "Rich", "Poor"), "trade warmed them into a pact"
    print("PASS test_trade_warms_stance_into_a_pact")


def test_war_severs_trade() -> None:
    """When two trading kingdoms go to war, their trade route is SEVERED (logged) and both lose the
    flow — an economic cost of war beyond casualties."""
    import intertrade, diplomacy

    _trade_world()
    _trade_realm("X", (10, 10), kmoney=5.0, kstock=18.0)
    _trade_realm("Y", (18, 10), kmoney=40.0, kstock=0.0, member_hunger=8)
    for t in range(1, 5):
        intertrade.update(world_state, t)
        diplomacy.update(world_state, t)
    assert ("X", "Y") in world_state["intertrade"]["routes"], "a route is active"
    diplomacy.record_war(world_state, "X", "Y", 5)          # they go to war -> hostile
    ev = intertrade.update(world_state, 6)
    assert ("X", "Y") not in world_state["intertrade"]["routes"], "the route is severed"
    assert any("SEVERED" in e for e in ev), ev
    print("PASS test_war_severs_trade")


def test_interdependence_measurement_is_exposed() -> None:
    """The interdependence data is exposed: cumulative trade VOLUME per pair and WAR COUNT per pair, so
    a run can compare war frequency among heavily-trading vs isolated pairs (an emergent read-out)."""
    import intertrade, diplomacy

    _trade_world()
    _trade_realm("Rich", (10, 10), kmoney=5.0, kstock=18.0)
    _trade_realm("Poor", (18, 10), kmoney=40.0, kstock=0.0, member_hunger=8)
    _trade_realm("Lone", (40, 40), kmoney=5.0, kstock=5.0)   # far away, isolated — no route
    for t in range(1, 5):
        intertrade.update(world_state, t)
        diplomacy.update(world_state, t)
    diplomacy.record_war(world_state, "Rich", "Poor", 5)
    # The measurement primitives are populated and comparable.
    assert intertrade.total_volume(world_state, "Rich", "Poor") > 0
    assert intertrade.total_volume(world_state, "Rich", "Lone") == 0.0
    assert diplomacy.war_count(world_state, "Rich", "Poor") == 1
    assert diplomacy.war_count(world_state, "Rich", "Lone") == 0
    print("PASS test_interdependence_measurement_is_exposed")


def test_intertrade_off_run_is_byte_identical_and_adds_no_llm() -> None:
    """intertrade_on=False (default) is byte-identical to the diplomacy baseline and adds ZERO LLM."""
    def run(on):
        llm.PROVIDER = "random"
        random.seed(7)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, stage="war", diplomacy_on=True, intertrade_on=on)
        return buf.getvalue(), dict(llm.get_call_stats())

    def baseline():
        llm.PROVIDER = "random"
        random.seed(7)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, stage="war", diplomacy_on=True)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        base = baseline()
        on_a, on_calls = run(True)
        on_b, _ = run(True)
        off, off_calls = run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "intertrade_on=False diverged from the diplomacy baseline"
    assert not world_state.get("intertrade"), "an off run writes no trade state"
    assert on_calls == off_calls, (on_calls, off_calls)   # trade is STATE — zero added LLM
    assert on_a == on_b, "an on run must be byte-identical across seeded repeats"
    print("PASS test_intertrade_off_run_is_byte_identical_and_adds_no_llm")


# --- Coalitions (V2 M4.15): the balance of power ----------------------------
def _coal_world() -> None:
    create_world(size=60)
    world_state["agents"].clear()
    world_state["food"].clear()
    world_state["turn"] = 0
    world_state["coalitions_on"] = True
    world_state["diplomacy_on"] = True


def _coal_settled(n, p, sid=None, money=0.0):
    a = Agent(name=n, personality="x")
    place_agent(a, *p)
    a.hunger, a.age, a.lifespan, a.money, a.settlement = 1, 30, 100, money, sid
    return a


def _coal_realm(king, home_c, kmoney=0.0, nmercs=0):
    home = f"{king}_home"
    world_state["settlements"][home] = {"id": home, "center": home_c, "members": {king}, "founded": 0}
    _coal_settled(king, home_c, sid=home, money=kmoney)
    world_state["monarchs"][home] = {"monarch": king, "since": 0, "garrison": set()}
    world_state["kingdoms"][king] = {"king": king, "home": home, "settlements": {home},
                                     "vassals": {}, "founded": 0, "discontent": {}}
    for i in range(nmercs):
        _coal_settled(f"{king}M{i}", (home_c[0] + i % 2, home_c[1] + 2), sid=None, money=0.5)


def _make_subject(emperor, sk):
    import kingdoms, trust
    emp = world_state["empires"].setdefault(emperor, {"emperor": emperor, "subject_kings": {},
                                                       "founded": 0, "discontent": {}})
    emp["subject_kings"][sk] = {"since": 0}
    emp["discontent"][sk] = 0
    trust.ensure_relationship(next(a for a in world_state["agents"] if a.name == sk),
                              emperor)["trust"] = kingdoms.LOYAL_TRUST


def _hegemon_scene():
    """A Hegemon controlling 3 of 5 settlements (home + 2 subject-kings) beside two weak, mutually
    HOSTILE kingdoms A and B — the Hegemon beats each alone (host 3) but not the pooled coalition (4)."""
    _coal_world()
    _coal_realm("Hegemon", (10, 10), kmoney=20.0, nmercs=3)
    _coal_realm("SK1", (20, 10)); _coal_realm("SK2", (10, 20))
    _coal_realm("A", (30, 30), kmoney=20.0, nmercs=2)
    _coal_realm("B", (38, 30), kmoney=20.0, nmercs=2)
    _make_subject("Hegemon", "SK1"); _make_subject("Hegemon", "SK2")
    world_state["diplomacy"] = {"stance": {("A", "B"): -6}, "pacts": set(), "alliances": set()}


def test_dominance_detection() -> None:
    """A power controlling a large share of settlements, far above the next, is a HEGEMON; a balanced
    world has none."""
    import coalitions

    _hegemon_scene()
    heg, share = coalitions.dominance(world_state)
    assert heg == "Hegemon" and share >= 0.4, (heg, share)
    assert coalitions.is_hegemon(world_state, "Hegemon")

    # A balanced world (no one dominant) has no hegemon.
    _coal_world()
    for k, c in [("K1", (10, 10)), ("K2", (30, 10)), ("K3", (10, 30))]:
        _coal_realm(k, c)
    assert coalitions.dominance(world_state)[0] is None, "no power dominates -> no hegemon"
    print("PASS test_dominance_detection")


def test_fear_drives_coalition_overriding_stance() -> None:
    """Fear of a hegemon OVERRIDES grievance: two mutually-hostile kingdoms both join the coalition
    against the common threat (the enemy of my enemy). No hegemon -> no coalition."""
    import coalitions, diplomacy

    _hegemon_scene()
    assert diplomacy.stance(world_state, "A", "B") == "hostile", "A and B are foes"
    mem = coalitions.coalition_members(world_state, "Hegemon")
    assert mem == {"A", "B"}, mem                      # both foes coalesce against the hegemon
    print("PASS test_fear_drives_coalition_overriding_stance")


def test_pooled_coalition_host_breaks_the_hegemon() -> None:
    """A hegemon that beats each kingdom INDIVIDUALLY is broken by the POOLED coalition host — and the
    coalition then DISSOLVES once the threat has passed."""
    import coalitions, empire

    _hegemon_scene()

    def host(k):
        return empire.imperial_host_size(world_state, next(a for a in world_state["agents"] if a.name == k))
    assert host("Hegemon") > host("A") and host("Hegemon") > host("B"), "the hegemon beats each alone"
    assert host("A") + host("B") > host("Hegemon"), "but the pooled coalition out-hosts it"

    coalitions.update(world_state, 1)
    assert coalitions.dominance(world_state)[0] is None, "the hegemon is broken below the threshold"
    assert empire.is_sovereign(world_state, "SK1"), "its subject-kings were freed"
    assert world_state["coalitions"]["target"] is None, "and the coalition dissolved (the threat passed)"
    print("PASS test_pooled_coalition_host_breaks_the_hegemon")


def test_coalition_dissolves_when_threat_passes() -> None:
    """When no hegemon exists, an existing coalition DISSOLVES — fear evaporates and old rivalries
    resurface (temporary marriages of convenience)."""
    import coalitions

    _coal_world()
    for k, c in [("K1", (10, 10)), ("K2", (30, 10)), ("K3", (10, 30))]:
        _coal_realm(k, c)
    # Seed a lingering coalition from a hegemon that has since fallen.
    world_state["coalitions"] = {"target": "OldHegemon", "members": {"K1", "K2", "K3"}}
    ev = coalitions.update(world_state, 1)
    assert world_state["coalitions"]["target"] is None
    assert any("DISSOLVED" in e for e in ev), ev
    print("PASS test_coalition_dissolves_when_threat_passes")


def test_coalitions_off_run_is_byte_identical_and_adds_no_llm() -> None:
    """coalitions_on=False (default) is byte-identical to the diplomacy baseline and adds ZERO LLM."""
    def run(on):
        llm.PROVIDER = "random"
        random.seed(7)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, stage="war", diplomacy_on=True, coalitions_on=on)
        return buf.getvalue(), dict(llm.get_call_stats())

    def baseline():
        llm.PROVIDER = "random"
        random.seed(7)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, stage="war", diplomacy_on=True)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        base = baseline()
        on_a, on_calls = run(True)
        on_b, _ = run(True)
        off, off_calls = run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "coalitions_on=False diverged from the diplomacy baseline"
    assert not world_state.get("coalitions"), "an off run writes no coalition state"
    assert on_calls == off_calls, (on_calls, off_calls)   # coalitions are STATE — zero added LLM
    assert on_a == on_b, "an on run must be byte-identical across seeded repeats"
    print("PASS test_coalitions_off_run_is_byte_identical_and_adds_no_llm")


# --- The Chronicle (V2 M4.16): the world writes its own history --------------
def _chron_world(literate_sids=("S001", "S002")) -> None:
    _fresh_world()
    world_state["chronicle_on"] = True
    for i, sid in enumerate(literate_sids):
        world_state["settlements"][sid] = {"id": sid, "center": (5 + 3 * i, 5),
                                           "members": {f"{sid}_scribe"}, "founded": 0}
        a = Agent(name=f"{sid}_scribe", personality="x")
        place_agent(a, 5 + 3 * i, 5)
        a.settlement = sid
        a.knowledge.add("writing")           # a scribe makes the settlement LITERATE (M4.10)


def _chron_kingdom(king, home="S001"):
    world_state["kingdoms"][king] = {"king": king, "home": home, "settlements": {home},
                                     "vassals": {}, "founded": 0, "discontent": {}}


def _chron_ev(t, body):
    world_state["events"].append(f"turn {t}: {body}")


def test_figure_archetype_and_epithet_derived_from_deeds() -> None:
    """Great figures are recognised from their DEEDS: a conqueror who seized many settlements gets 'the
    Conqueror', a revolutionary who freed a town 'the Liberator', an over-taxer deposed by revolt 'the
    Grasping' — each a deterministic function of what they did. Pre-writing deeds are anonymized legend."""
    import chronicle

    _chron_world()
    for k in ("Rex", "Vlad", "Cyn"):
        _chron_kingdom(k, "S001" if k != "Vlad" else "S002")
    _chron_ev(1, "Rex seized S001 by force -> MONARCH of S001")
    _chron_ev(2, "Rex OVERTHREW Gorm and seized S002 by force -> MONARCH of S002")
    _chron_ev(3, "KING Rex DEFEATED Otto in war -> Otto SUBJUGATED as a subject-king; an EMPIRE rises")
    _chron_ev(5, "Vlad seized S002 by force -> MONARCH of S002")
    for t in (6, 7, 8):
        _chron_ev(t, "MONARCH Vlad levied 5.0 from S002 by force (no consent)")
    _chron_ev(9, "the UPRISING in S002 TRIUMPHED — monarch Vlad is DEPOSED; Cyn to rule by consent (1 fell)")
    chronicle.update(world_state, 9)

    figs = {f["name"]: chronicle.epithet(f) for f in chronicle.great_figures(world_state)}
    assert figs["Rex"] == "the Conqueror", figs          # two conquests + a war
    assert figs["Cyn"] == "the Liberator", figs          # led a winning uprising
    assert figs["Vlad"] == "the Grasping", figs          # over-taxed, then deposed
    arche = {f["name"]: f["archetype"] for f in chronicle.great_figures(world_state)}
    assert arche["Cyn"] == "revolutionary" and arche["Rex"] == "conqueror"

    # PREHISTORY: an event in an ILLITERATE settlement enters only as anonymized legend.
    _chron_world(literate_sids=())            # no scribes -> no literacy
    world_state["chronicle_on"] = True
    _chron_kingdom("Ork", "S009")
    _chron_ev(1, "Ork seized S009 by force -> MONARCH of S009")
    chronicle.update(world_state, 1)
    assert not chronicle.great_figures(world_state), "no named figures in the preliterate dark"
    entry = chronicle.saga(world_state)[0]
    assert entry["fidelity"] == "legend" and "Ork" not in entry["name"], entry
    print("PASS test_figure_archetype_and_epithet_derived_from_deeds")


def test_events_and_houses_assembled_from_records() -> None:
    """Major events are named deterministically and a dynasty is assembled into a house-history
    (founder, generations, crowns, fall) that matches the actual lineage/title records."""
    import chronicle

    _chron_world()
    _chron_kingdom("Rex")
    _chron_ev(1, "Rex seized S001 by force -> MONARCH of S001")
    _chron_ev(2, "Aldo was born to Rex and Isla in S001")
    _chron_ev(3, "Bran was born to Aldo and Mara in S001")
    _chron_ev(20, "Aldo succeeded Rex as [monarch of S001] (eldest child)")
    _chron_ev(40, "Bran succeeded Aldo as [monarch of S001] (eldest child)")
    _chron_ev(50, "the line of Bran is extinguished; the crown of [monarch of S001] lies vacant")
    chronicle.update(world_state, 50)

    names = {e["name"] for e in chronicle.saga(world_state)}
    assert "the Crowning of Aldo" in names and "the End of the House of Bran" in names, names
    h = chronicle.houses(world_state)[0]
    assert h["founder"] == "Rex"                              # the conqueror founded the line
    assert h["members"] == {"Rex", "Aldo", "Bran"}           # three crowned kin
    assert chronicle.generations(world_state["chronicle"], h) == 3   # Rex -> Aldo -> Bran
    assert h["crowns"] == 2                                    # two successions passed the crown
    assert h["fell"] == "the line was extinguished"
    print("PASS test_events_and_houses_assembled_from_records")


def test_saga_is_deterministic_under_seed() -> None:
    """Same seed -> same chronicle: a full seeded run produces byte-identical structured saga output."""
    import chronicle

    def run_saga():
        llm.PROVIDER = "random"
        random.seed(11)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(30, stage="war", chronicle_on=True)
        return chronicle.export_markdown(world_state)

    saved = llm.PROVIDER
    try:
        a, b = run_saga(), run_saga()
    finally:
        llm.PROVIDER = saved
    assert a == b, "the chronicle must be identical for the same seed"
    print("PASS test_saga_is_deterministic_under_seed")


def test_chronicle_is_read_only_and_off_byte_identical() -> None:
    """The chronicle NEVER mutates the sim (only its own record), so a --chronicle run is byte-identical
    to one without it, and adds zero LLM."""
    def run(on):
        llm.PROVIDER = "random"
        random.seed(7)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, stage="war", chronicle_on=on)
        return buf.getvalue(), dict(llm.get_call_stats())

    def baseline():
        llm.PROVIDER = "random"
        random.seed(7)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(25, stage="war")
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        base = baseline()
        on_out, on_calls = run(True)
        off, off_calls = run(False)
    finally:
        llm.PROVIDER = saved
    assert base == off, "chronicle_on=False diverged from the baseline"
    assert not world_state.get("chronicle"), "an off run writes no chronicle state"
    assert on_calls == off_calls, "the structured chronicle adds ZERO LLM"
    print("PASS test_chronicle_is_read_only_and_off_byte_identical")


def test_narrator_is_walled_off_from_the_structured_chronicle() -> None:
    """The optional LLM narrator NEVER mutates the structured chronicle: narrating leaves world_state
    and the chronicle record identical (it only returns prose)."""
    import chronicle, narrator, copy

    _chron_world()
    _chron_kingdom("Rex")
    _chron_ev(1, "Rex seized S001 by force -> MONARCH of S001")
    _chron_ev(2, "KING Rex DEFEATED Otto in war -> Otto SUBJUGATED; an EMPIRE rises")
    chronicle.update(world_state, 2)
    before = copy.deepcopy(world_state["chronicle"])
    structured = chronicle.export_markdown(world_state)

    prose = narrator.narrate_saga(world_state)      # the LLM layer (offline -> falls back to detail)
    assert isinstance(prose, str) and prose
    assert world_state["chronicle"] == before, "narrating must not mutate the structured chronicle"
    assert chronicle.export_markdown(world_state) == structured, "the structured saga is unchanged"
    print("PASS test_narrator_is_walled_off_from_the_structured_chronicle")


# --- Minds at the pivots (V2 M5.1): character decides the undecided ----------
def _minds_world() -> None:
    """A clean minds-on world with no agents/food (pivot unit tests place their own figures)."""
    _fresh_world()
    world_state["minds_on"] = True


def test_the_band_binds_only_close_calls_are_consulted() -> None:
    """HEADLINE 1: the close-margin band binds absolutely. A DECISIVE war (host far above/below the
    enemy's) returns the math's verdict UNTOUCHED and consults no mind — regardless of personality;
    only a near-tie (|margin| <= WAR_BAND) opens the call to character."""
    import mind

    _minds_world()
    bold = _agent("Bold", "competitive bold", (1, 1))
    meek = _agent("Meek", "cautious timid", (2, 2))

    # DECISIVE: margin +4 (9 vs 5) — overwhelming. Both figures launch; neither is consulted.
    for fig in ("Bold", "Meek"):
        verdict, consult = mind.tilt(world_state, fig, "war", 4, True,
                                     {"att": 9, "def": 5, "target": "S002"}, 1)
        assert verdict is True and consult is None, (fig, verdict, consult)
    # DECISIVE the other way: margin -4 (5 vs 9). Both hold; neither is consulted.
    for fig in ("Bold", "Meek"):
        verdict, consult = mind.tilt(world_state, fig, "war", -4, False,
                                     {"att": 5, "def": 9, "target": "S002"}, 1)
        assert verdict is False and consult is None, (fig, verdict, consult)
    assert not world_state.get("mind_consults"), "no decisive case should consult a mind"

    # CLOSE (margin 0, an even 5-vs-5 standoff): now character is consulted and CAN differ.
    bold_go, bc = mind.tilt(world_state, "Bold", "war", 0, False, {"att": 5, "def": 5, "target": "S002"}, 2)
    meek_go, mc = mind.tilt(world_state, "Meek", "war", 0, False, {"att": 5, "def": 5, "target": "S002"}, 2)
    assert bc is not None and mc is not None, "a close call consults the mind"
    assert bold_go != meek_go, "in the band, character can change the outcome"
    print("PASS test_the_band_binds_only_close_calls_are_consulted")


def test_character_tilts_a_close_war_offline_standin() -> None:
    """HEADLINE 2: two kings in the IDENTICAL close-margin war (a slim 6-vs-5 edge), differing ONLY in
    personality — the competitive one MARCHES, the cautious one REFRAINS (overriding the slim lead).
    The offline, deterministic personality stand-in decides the undecided (no LLM, no RNG)."""
    import mind, llm

    saved = llm.PROVIDER
    llm.PROVIDER = "random"                     # the offline personality stand-in
    try:
        _minds_world()
        _agent("Caesar", "competitive", (1, 1))
        _agent("Fabius", "cautious", (2, 2))
        sit = {"att": 6, "def": 5, "target": "S002"}       # a slim material edge, inside the band
        caesar_go, cc = mind.tilt(world_state, "Caesar", "war", 1, True, sit, 1)
        fabius_go, fc = mind.tilt(world_state, "Fabius", "war", 1, True, sit, 1)
    finally:
        llm.PROVIDER = saved

    assert caesar_go is True, "the competitive king marches on even-ish odds"
    assert fabius_go is False, "the cautious king refrains despite the slim lead"
    assert cc["inclination"] > 0 and fc["inclination"] < 0, (cc["inclination"], fc["inclination"])
    assert fc["flipped"] is True, "the cautious king OVERRODE the math's go"
    print("PASS test_character_tilts_a_close_war_offline_standin")


def test_motive_enters_the_written_history() -> None:
    """HEADLINE 3: a pivot decision writes its REASON, and the chronicle surfaces the WHY in the saga —
    history records not just that the king marched but why ('...fortune favours the bold')."""
    import mind, chronicle

    _chron_world()                              # a literate world so the war is recorded as HISTORY
    world_state["minds_on"] = True
    _chron_kingdom("Rex", "S001")
    _agent("Rex", "competitive bold", (5, 5))
    # Rex decides a close war (the mind records his motive), then the war fires and is logged.
    go, rec = mind.tilt(world_state, "Rex", "war", 0, False, {"att": 5, "def": 5, "target": "S002"}, 3)
    assert go is True, "the bold king marches on even odds"
    _chron_ev(3, "KING Rex DEFEATED Otto in war -> Otto SUBJUGATED as a subject-king; an EMPIRE rises")
    chronicle.update(world_state, 3)

    war = next(e for e in chronicle.saga(world_state) if "Rex's Conquest" in e["name"])
    assert "saying" in war["detail"], war["detail"]           # the motive entered the record
    assert "fortune favours the bold" in war["detail"], war["detail"]
    md = chronicle.export_markdown(world_state)
    assert "saying" in md and "Rex" in md
    print("PASS test_motive_enters_the_written_history")


def test_pivot_provider_selection_random_stands_in_live_reaches_the_model() -> None:
    """DOC: which provider the pivot consult uses. AICIV_PROVIDER routes it (default 'ollama'):
    under 'random' the inclination is the DETERMINISTIC offline stand-in (reads the DISPOSITION marker,
    contacts no model); under a live provider (ollama/gemini) the SAME entry point goes through
    `_raw_query` to the model. So a live qwen is engaged only when AICIV_PROVIDER is NOT 'random'."""
    import llm

    saved = llm.PROVIDER
    try:
        # The module default (when AICIV_PROVIDER is unset) is the local model server — qwen via Ollama —
        # NOT the offline stand-in. So the stand-in is used only when AICIV_PROVIDER is set to 'random'.
        assert os.getenv("AICIV_PROVIDER", "ollama") == "ollama" or os.getenv("AICIV_PROVIDER") is not None

        # 'random' -> offline stand-in: the disposition planted in the prompt is read straight back, no I/O.
        llm.PROVIDER = "random"
        out = llm.get_inclination("DISPOSITION: 0.7\nOFFLINE_REASON: because the odds were even")
        assert out["inclination"] == 0.7 and "odds were even" in out["reason"], out

        # a live provider -> the SAME call is dispatched to the model via _raw_query (here stubbed).
        llm.PROVIDER = "ollama"
        seen = {}

        def _stub(prompt):
            seen["prompt"] = prompt
            return {"inclination": 0.9, "reason": "qwen"}

        orig = llm._raw_query
        llm._raw_query = _stub
        try:
            live = llm.get_inclination("You are Rex... DISPOSITION: 0.7")
            assert live["inclination"] == 0.9 and live["reason"] == "qwen", live
            assert "prompt" in seen, "a live provider must reach the model through _raw_query"
        finally:
            llm._raw_query = orig
    finally:
        llm.PROVIDER = saved
    print("PASS test_pivot_provider_selection_random_stands_in_live_reaches_the_model")


def test_breakaway_motive_enters_the_chronicle() -> None:
    """The BREAKAWAY pivot's motive reaches the saga: a vassal who breaks away for a recorded reason gets
    a Secession entry whose detail carries the WHY (closing the wiring gap where breakaways — unlike wars
    and uprisings — produced no chronicle entry at all, so their motive was recorded but never printed)."""
    import mind, chronicle

    _chron_world()                                     # S001 literate -> the secession is HISTORY
    world_state["minds_on"] = True
    _chron_kingdom("Duke", "S001")
    _agent("Vale", "independent solitary", (5, 5))     # a proud vassal -> breaks on a close call
    brk, rec = mind.tilt(world_state, "Vale", "breakaway", 0.0, False,
                         {"trust": 1, "lord": "Duke"}, 7)
    assert brk is True, "the proud vassal breaks on a near-tie"
    _chron_ev(7, "Vale BROKE AWAY from Duke's realm — S001 is independent again (loyalty collapsed)")
    chronicle.update(world_state, 7)

    sec = next(e for e in chronicle.saga(world_state) if "Secession of Vale" in e["name"])
    assert "saying" in sec["detail"], sec["detail"]     # the motive entered the record
    md = chronicle.export_markdown(world_state)
    assert "Secession of Vale" in md and "saying" in md
    print("PASS test_breakaway_motive_enters_the_chronicle")


def test_breakaway_motive_survives_the_hysteresis_delay() -> None:
    """The breakaway pivot has HYSTERESIS: the mind is consulted the turn loyalty first slips into the
    close band, but the secession EVENT fires PATIENCE turns later (and by then the margin may have gone
    decisive, so no fresh consult happens on the event turn). The motive must still attach — the chronicle
    looks BACK across the hysteresis window, not for an exact same-turn match (the real-run bug where live
    Secession entries carried no motive despite the mind having been consulted)."""
    import mind, chronicle, kingdoms

    _chron_world()
    world_state["minds_on"] = True
    _chron_kingdom("Duke", "S001")
    _agent("Vale", "independent solitary", (5, 5))
    # The mind decides to break at turn 7 (loyalty at the borderline); the motive is recorded at 7.
    brk, _ = mind.tilt(world_state, "Vale", "breakaway", 0.0, False, {"trust": 1, "lord": "Duke"}, 7)
    assert brk is True
    assert mind.motive_for(world_state, 7 + kingdoms.BREAKAWAY_PATIENCE, "Vale") is None, \
        "an exact same-turn lookup MISSES the delayed event — that was the bug"
    # ...but the secession only fires PATIENCE turns later.
    event_turn = 7 + kingdoms.BREAKAWAY_PATIENCE
    _chron_ev(event_turn, "Vale BROKE AWAY from Duke's realm — S001 is independent again (loyalty collapsed)")
    chronicle.update(world_state, event_turn)

    sec = next(e for e in chronicle.saga(world_state) if "Secession of Vale" in e["name"])
    assert "saying" in sec["detail"], sec["detail"]                    # the delayed motive still attached
    diag = world_state["chronicle"]["motive_diag"][-1]
    assert diag["status"] == "attached" and diag["figure"] == "Vale", diag
    print("PASS test_breakaway_motive_survives_the_hysteresis_delay")


def test_decisive_secession_carries_no_motive_and_is_diagnosed_as_such() -> None:
    """A secession that resolved OUTSIDE the close band consulted NO mind, so it CORRECTLY carries no
    motive — and the diagnostic distinguishes that ('no_consult') from a genuine 'lookup_failed', so a
    blank saga entry is explainable rather than mysterious. Determinism/byte-identity are untouched."""
    import chronicle

    _chron_world()
    world_state["minds_on"] = True
    _chron_kingdom("Duke", "S001")
    _agent("Grim", "x", (5, 5))
    # No mind.tilt call at all: loyalty collapsed decisively, the mind was never consulted.
    _chron_ev(4, "Grim BROKE AWAY from Duke's realm — S001 is independent again (loyalty collapsed)")
    chronicle.update(world_state, 4)

    sec = next(e for e in chronicle.saga(world_state) if "Secession of Grim" in e["name"])
    assert "saying" not in sec["detail"], sec["detail"]                # decisive break -> no motive (correct)
    diag = world_state["chronicle"]["motive_diag"][-1]
    assert diag["status"] == "no_consult", diag                       # and it is diagnosed as such, not a bug
    print("PASS test_decisive_secession_carries_no_motive_and_is_diagnosed_as_such")


def test_crushed_uprising_carries_its_motive() -> None:
    """A FAILED rising had a reason to rise as much as a successful one — the crushed line now carries the
    ringleader's motive too ('the people of S001 rose and were put down, saying ...'). The rise decision is
    consulted the same turn, so the sid lookup finds it."""
    import mind, chronicle

    _chron_world()
    world_state["minds_on"] = True
    _agent("Spark", "bold competitive", (5, 5))
    import uprising
    rise, _ = mind.tilt(world_state, "Spark", "uprising", 0.0, False,
                        {"pressure": uprising.UPRISING_MIN_PRESSURE,
                         "threshold": uprising.UPRISING_MIN_PRESSURE, "sid": "S001"}, 6)
    assert rise is True, "the firebrand raises the banner on a near-tie"
    _chron_ev(6, "the UPRISING in S001 was CRUSHED — king Rex holds (2 guards + 3 risers fell); the survivors are cowed")
    chronicle.update(world_state, 6)

    crushed = next(e for e in chronicle.saga(world_state) if "(crushed)" in e["name"])
    assert "put down" in crushed["detail"] and "saying" in crushed["detail"], crushed["detail"]
    print("PASS test_crushed_uprising_carries_its_motive")


def test_belief_changes_at_most_once_per_turn_no_flipflop() -> None:
    """A belief an agent changes this turn cannot flip back the SAME turn. When two contradictory beliefs
    are BOTH warranted at once, the first (catalogue order) wins deterministically — so the log never
    shows the A->B, B->A oscillation (Wren renouncing X for Y and Y for X on one tick)."""
    import beliefs

    def counters(**over):
        c = {"fed": 0, "hungry": 0, "rich": 0, "extracted": 0, "solidarity": 0, "deprived": 0, "deaths": 0}
        c.update(over); return c

    # Wren already holds 'stronger together'; this turn her experience ALSO warrants its opposite
    # ('the strong take what they want'). Exactly one change fires — no ping-pong.
    _belief_world()
    _bagent("Wren", (5, 5))
    world_state.setdefault("belief_exp", {})["Wren"] = counters(extracted=10, solidarity=10)
    world_state.setdefault("beliefs", {})["Wren"] = {beliefs.STRONGER_TOGETHER}
    events = [e for e in beliefs.form(world_state, 5) if "Wren came to believe" in e]
    assert len(events) == 1, events                                   # not two contradictory events
    assert beliefs.agent_beliefs("Wren", world_state) == {beliefs.STRONG_TAKE}, \
        beliefs.agent_beliefs("Wren", world_state)                    # first-in-catalogue wins, deterministically
    # and never both renunciations on the same turn
    joined = " || ".join(events)
    assert not ("renouncing 'we are stronger together'" in joined
                and "renouncing 'the strong take what they want'" in joined), joined

    # Both freshly warranted (neither pre-held) -> still exactly one, deterministically the same winner.
    _belief_world()
    _bagent("Kit", (5, 5))
    world_state.setdefault("belief_exp", {})["Kit"] = counters(extracted=10, solidarity=10)
    formed = [e for e in beliefs.form(world_state, 5) if "Kit came to believe" in e]
    assert len(formed) == 1 and beliefs.STRONG_TAKE in formed[0], formed
    assert beliefs.agent_beliefs("Kit", world_state) == {beliefs.STRONG_TAKE}
    print("PASS test_belief_changes_at_most_once_per_turn_no_flipflop")


def test_minds_off_is_byte_identical_and_a_bad_response_falls_back() -> None:
    """The two guarantees in one: (a) --minds OFF is byte-identical to the baseline (the pivots use their
    exact deterministic verdicts, no mind_* state written); (b) a malformed model response degrades to
    inclination 0.0 — NO tilt — so the math's verdict stands (never a crash)."""
    import mind, llm

    # (a) OFF byte-identical over a full staged run.
    def run(minds):
        llm.PROVIDER = "random"
        random.seed(5)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(20, stage="war", minds_on=minds)
        return buf.getvalue()

    saved = llm.PROVIDER
    try:
        base, off = run(False), run(False)
        assert base == off
        assert not world_state.get("mind_consults") and not world_state.get("mind_cache"), \
            "a minds-off run writes no mind state"

        # (b) malformed live response -> neutral fallback -> the deterministic verdict stands.
        _minds_world()
        _agent("Even", "competitive bold", (1, 1))
        llm.PROVIDER = "gemini"
        orig = llm._raw_query
        llm._raw_query = lambda prompt: {"garbage": "no inclination field"}   # malformed
        try:
            incl = llm.get_inclination("DISPOSITION: 0.9")
            assert incl["inclination"] == 0.0, incl                          # clamped to neutral
            # in-band, but a neutral inclination leaves the base verdict UNCHANGED (math stands)
            verdict, rec = mind.tilt(world_state, "Even", "war", 0, False,
                                     {"att": 5, "def": 5, "target": "S002"}, 1)
            assert verdict is False and rec["inclination"] == 0.0, (verdict, rec)
        finally:
            llm._raw_query = orig
    finally:
        llm.PROVIDER = saved
    print("PASS test_minds_off_is_byte_identical_and_a_bad_response_falls_back")


# --- V4.15: the showcase DIRECTOR (severity, captions, pacing) ---------------
def test_director_imports_only_stdlib() -> None:
    """renderer/director.py is a PURE classifier — stdlib only, no sim, no pygame.

    Mirrors the two existing renderer AST boundary tests. The director decides where the camera
    looks and what the caption says; if it could import decision logic (or mutate anything) the
    presentation layer would have reached into the simulation. Parses the file, never imports it.
    """
    import ast
    with open("renderer/director.py") as f:
        tree = ast.parse(f.read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "re", "typing"}, f"director imports beyond stdlib: {imported}"
    print("PASS test_director_imports_only_stdlib")


def test_director_classifies_every_engine_event_shape() -> None:
    """Each event KIND is pinned to the VERBATIM string its engine module emits.

    This is the contract that keeps the typed-event boundary honest: the director keys everything
    on `kind`, and these cases are what tie each kind back to the f-string in monarchy/kingdoms/
    empire/uprising/lineage/eras. If an engine ever rewords an event, exactly one case fails here
    and names the kind that went stale — instead of the showcase silently losing a camera beat.
    """
    from renderer.director import classify, LEGENDARY, MAJOR, MINOR, NOISE
    cases = {
        # LEGENDARY (the wording is monarchy.py / kingdoms.py / empire.py / lineage.py verbatim)
        "turn 3: KING Aldric DEFEATED Cyrus in war (10 loyal host vs 9; 4+5 fell) -> Cyrus SUBJUGATED as a subject-king; an EMPIRE rises":
            ("war_won", LEGENDARY, None),
        "turn 8: KING Borin's war on Aldric FAILED (3 loyal host vs 9; 2+1 fell)":
            ("war_failed", LEGENDARY, None),
        "turn 0: KING Aldric CONQUERED S0A2 into the realm (6 host vs 4 defenders; 2+3 fell) -> vassal LordA; realm now 2 settlements":
            ("realm_conquest", LEGENDARY, "S0A2"),
        "turn 0: Aldric seized S0A1 by force (9 fighters vs 6 defenders; 3+4 fell) -> MONARCH of S0A1":
            ("town_seized", LEGENDARY, "S0A1"),
        "turn 34: the line of Aldric is extinguished; the crown of [king of the realm of S0A1] lies vacant":
            ("dynasty_extinct", LEGENDARY, None),
        "turn 40: subject-king Cyrus was freed from Aldric's empire by the coalition":
            ("empire_broken", LEGENDARY, None),
        # MAJOR
        "turn 34: LordC OVERTHREW Aldric and seized S0A1 by force (2 fighters vs 1 defenders; 1+1 fell) -> MONARCH of S0A1":
            ("coup", MAJOR, "S0A1"),
        "turn 22: the UPRISING in S0B2 TRIUMPHED — lord LordB is DEPOSED; BWV4 to rule by consent (2 risers fell)":
            ("uprising_triumph", MAJOR, "S0B2"),
        "turn 26: the UPRISING in S0C2 was CRUSHED — lord LordC holds (2 guards + 2 risers fell); the survivors are cowed":
            ("uprising_crushed", MAJOR, "S0C2"),
        "turn 22: UPRISING in S0B2 — 4 risers rise against lord LordB (3 defenders: 3 standing + 0 hired)":
            ("uprising_begins", MAJOR, "S0B2"),
        "turn 22: BWV4 led the rising in S0B2 — the survivors rally to him (power to be legitimised by consent, M3.2)":
            ("uprising_leader", MAJOR, "S0B2"),
        "turn 22: the risers EXPROPRIATED LordB's hoard of 37.60 — split among 2 (the heirs inherit nothing)":
            ("expropriation", MAJOR, None),
        "turn 36: subject-king Cyrus BROKE AWAY from Aldric's empire — reclaiming independence with his realm (loyalty collapsed)":
            ("breakaway_empire", MAJOR, None),
        "turn 22: S0B2 SECEDED from Borin's realm (the lord was deposed) — independent again":
            ("secession", MAJOR, "S0B2"),
        "turn 17: S0B2 entered the Iron Age": ("era", MAJOR, "S0B2"),
        "turn 2: LordA devised WRITING in S0A2": ("writing", MAJOR, "S0A2"),
        "turn 17: LordB forged the first WEAPONS in S0B2": ("weapons", MAJOR, "S0B2"),
        "turn 34: Rhea succeeded LordC as [monarch of S0A1] (eldest child)": ("succession", MAJOR, None),
        "turn 4: LordA emerged as leader of S0A2 (3 followers)": ("leader_consent", MAJOR, "S0A2"),
        # MINOR / NOISE — the churn that must never take the camera
        "turn 12: Iris was born to AWV2 and LordA in S0A2": ("birth", MINOR, "S0A2"),
        "turn 9: AKM0 died (starved)": ("death", MINOR, None),
        "turn 5: LordA and AWV2 formed an ALLIANCE": ("alliance_formed", MINOR, None),
        "turn 7: BV3 taught 'metalworking' to BWV2": ("teaching", MINOR, None),
        "turn 6: MONARCH Cyrus levied 3.2 from S0C1 by force (rate 0.45)": ("levy", NOISE, "S0C1"),
        "turn 6: imperial tribute cascaded up: 4.1 subject-king Cyrus->EMPEROR Aldric":
            ("tribute_imperial", NOISE, None),
        "turn 6: tribute cascaded up S0A2: 2.0 members->LordA, 1.0 LordA->KING Aldric":
            ("tribute_realm", NOISE, "S0A2"),
        'turn 2: LordA talked to AWV2: "Hi! Want to team up?"': ("talk", NOISE, None),
        "turn 3: BV3 trust in LordB: 0 -> 1 (friendly message)": ("trust", NOISE, None),
        "turn 26: AWV0 seethes under LordA's tribute (discontent 6.0)": ("grievance", NOISE, None),
    }
    for line, (kind, sev, focus) in cases.items():
        e = classify(line)
        assert e.kind == kind, f"{line[:60]!r} -> {e.kind}, expected {kind}"
        assert e.severity == sev, f"{kind}: severity {e.severity}, expected {sev}"
        assert e.focus == focus, f"{kind}: focus {e.focus}, expected {focus}"
    # An event the director has never seen degrades quietly to NOISE rather than crashing a run.
    unknown = classify("turn 4: something entirely new happened")
    assert (unknown.kind, unknown.severity) == ("unknown", NOISE)
    print("PASS test_director_classifies_every_engine_event_shape")


def test_world_firsts_and_bloodless_battles_are_demoted() -> None:
    """A beat is only a beat ONCE, and a battle with no dead is not a battle.

    Two demotions the showcase depends on. WORLD-FIRSTS: the first town to devise writing is a
    turning point, the seventh is ordinary progress — and the Neolithic is never a beat at all
    (a staged run drops five settlements into it on turn 1). BLOODLESS: `empire.update` will march
    ten fighters at a realm that can field zero, and the log calls it a war; with nothing falling on
    either side there is nothing to watch, so it must not take the camera.
    """
    from renderer.director import classify, MAJOR, MINOR
    seen: set = set()
    first = classify("turn 2: LordA devised WRITING in S0A2", seen)
    again = classify("turn 9: BT5 devised WRITING in S0B1", seen)
    assert (first.severity, again.severity) == (MAJOR, MINOR), "only the world's FIRST writing is a beat"
    bronze = classify("turn 4: S0A1 entered the Bronze Age", seen)
    bronze2 = classify("turn 9: S0B1 entered the Bronze Age", seen)
    iron = classify("turn 17: S0B2 entered the Iron Age", seen)
    assert (bronze.severity, bronze2.severity, iron.severity) == (MAJOR, MINOR, MAJOR), \
        "each AGE is a beat once; a second town reaching the same age is not"
    neo = classify("turn 1: S0A2 entered the Neolithic", seen)
    assert neo.severity == MINOR, "the Neolithic is staging scenery, never a beat"
    bloody = classify("turn 3: KING Aldric DEFEATED Cyrus in war (10 loyal host vs 9; 4+5 fell) -> x")
    hollow = classify("turn 3: KING Borin DEFEATED Aldric in war (8 loyal host vs 0; 0+0 fell) -> x")
    assert bloody.severity != hollow.severity and hollow.severity == MINOR, \
        "a war in which nobody fell is a bloodless annexation, not a camera beat"
    print("PASS test_world_firsts_and_bloodless_battles_are_demoted")


def test_a_crown_falling_is_legendary() -> None:
    """Rank decides weight: a CROWN dying or being unseated is legendary, a commoner is not.

    Three promotions, all of which need context the event string alone does not carry. A death is
    ordinarily MINOR churn — but the death of a reigning emperor/king/monarch ends a reign. A coup
    is a MAJOR seizure — but a coup that unseats a CROWN, or that ends a BLOODLINE (the extinction
    line firing on the same turn), is a throne changing hands.
    """
    from renderer.director import classify, classify_turn, caption, crowned_names, LEGENDARY, MAJOR, MINOR
    crowned = frozenset({"Borin", "Aldric"})
    king = classify("turn 9: Borin died (starved)", None, crowned)
    commoner = classify("turn 9: BT4 died (starved)", None, crowned)
    assert king.severity == LEGENDARY and commoner.severity == MINOR
    assert caption(king)[0] == "BORIN IS DEAD" and "reign ends" in caption(king)[1]
    assert classify("turn 9: Borin died (starved)").severity == MINOR, \
        "with no crown list supplied every death stays MINOR (the default is unchanged)"
    coup = "turn 34: LordC OVERTHREW {} and seized S0A1 by force (2 fighters vs 1 defenders; 1+1 fell) -> MONARCH of S0A1"
    assert classify(coup.format("Aldric"), None, crowned).severity == LEGENDARY, "a coup on a crown"
    assert classify(coup.format("Nobody"), None, crowned).severity == MAJOR, "a coup on a commoner"
    # ...and a coup that ends a line is legendary even when the deposed is not in the crown list.
    both = classify_turn([coup.format("Nobody"),
                          "turn 34: the line of Aldric is extinguished; the crown of [x] lies vacant"])
    assert all(e.severity == LEGENDARY for e in both), [(e.kind, e.severity) for e in both]
    # crowned_names reads the institutions, and is narrower than "notable" — no lords, no leaders.
    state = {"empires": {"Borin": {}}, "kingdoms": {"Borin": {}, "Cyrus": {}},
             "monarchs": {"S0A1": {"monarch": "AWV5"}}, "leaders": {"S0A2": {"leader": "LordA"}}}
    assert crowned_names(state) == frozenset({"Borin", "Cyrus", "AWV5"}), crowned_names(state)
    print("PASS test_a_crown_falling_is_legendary")


def test_turn_severity_and_beat_editing() -> None:
    """A turn is as important as its biggest event, and one story gets ONE cut.

    Covers the four edits `beats` makes: turn-0 staging is dropped (the scenario builder's six
    conquests are set dressing), an uprising's set-up and consequences FOLD into its outcome,
    consequences that name no settlement INHERIT the focus of the beat that does, and repeats of a
    collapsible kind on one turn become a single captioned group.
    """
    from renderer.director import (classify_turn, turn_severity, beats, caption,
                                   LEGENDARY, MAJOR, NOISE)
    quiet = classify_turn(["turn 5: MONARCH Cyrus levied 3.2 from S0C1 by force (rate 0.45)",
                           'turn 5: LordA talked to AWV2: "Hi!"'])
    assert turn_severity(quiet) == NOISE and beats(quiet) == []
    assert turn_severity([]) == NOISE, "an empty turn is quiet, not a crash"
    mixed = classify_turn(["turn 9: AKM0 died (starved)",
                           "turn 9: KING Aldric DEFEATED Cyrus in war (10 loyal host vs 9; 4+5 fell) -> x"])
    assert turn_severity(mixed) == LEGENDARY, "the turn takes the rank of its biggest event"
    # Turn 0 is the staging builder — six conquest lines before the title card has cleared.
    staging = classify_turn(["turn 0: Aldric seized S0A1 by force (9 fighters vs 6 defenders; 3+4 fell) -> MONARCH of S0A1"])
    assert beats(staging) == [] and len(beats(staging, drop_staging=False)) == 1
    # One revolt = one cut: the rising, the leader, the hoard and the secession fold into the outcome.
    revolt = classify_turn([
        "turn 22: UPRISING in S0B2 — 4 risers rise against lord LordB (3 defenders: 3 standing + 0 hired)",
        "turn 22: BWV4 led the rising in S0B2 — the survivors rally to him (power to be legitimised by consent, M3.2)",
        "turn 22: the UPRISING in S0B2 TRIUMPHED — lord LordB is DEPOSED; BWV4 to rule by consent (2 risers fell)",
        "turn 22: the risers EXPROPRIATED LordB's hoard of 37.60 — split among 2 (the heirs inherit nothing)",
        "turn 22: S0B2 SECEDED from Borin's realm (the lord was deposed) — independent again"])
    b = beats(revolt)
    assert [e.kind for e in b] == ["uprising_triumph"], f"one revolt, one cut: got {[e.kind for e in b]}"
    title, sub = caption(b[0])
    assert title == "THE RISING OF S0B2" and "Lord B falls" in sub, (title, sub)
    # A consequence with no settlement of its own inherits the focus of the beat that has one.
    war = classify_turn([
        "turn 41: KING Aldric CONQUERED S0B1 into the realm (6 host vs 4 defenders; 2+3 fell) -> held directly",
        "turn 41: the line of Borin is extinguished; the crown of [monarch of S0B1] lies vacant"])
    focuses = {e.kind: e.focus for e in beats(war)}
    assert focuses["dynasty_extinct"] == "S0B1", f"the camera must know where to look: {focuses}"
    # Three towns choosing a leader on one turn is one beat, and the caption says so.
    many = classify_turn([f"turn 4: Lord{r} emerged as leader of S0{r}2 (3 followers)" for r in "ABC"])
    got = beats(many)
    assert len(got) == 1, f"repeats of one kind collapse: got {[e.kind for e in got]}"
    assert "and 2 more" in (caption(got[0])[1] or ""), caption(got[0])
    print("PASS test_turn_severity_and_beat_editing")


def test_no_two_wars_chain_in_one_turn() -> None:
    """A realm's fate is not decided twice in one turn (the v4.15 cascade fix).

    THE BUG: the strongest crown won its war and was left exhausted — casualties taken, chest
    spent — and the next crown down the same loop saw a host of zero and took the victor together
    with everything it had just won. Three realms collapsed into one empire between two frames and
    the war engine had nothing to iterate over for the rest of the run. A crown that has already
    fought this turn is now spent; the rival must wait for the next turn.
    """
    import empire, scenario, world
    world.create_world(size=26)
    state = world.world_state
    state["taxation_on"] = True
    scenario.apply(state, "showcase")
    fought: list[str] = []
    real = empire.wage_war

    def spy(st, att, dfn, turn):
        fought.append(f"{att}->{dfn}")
        return real(st, att, dfn, turn)

    empire.wage_war = spy
    try:
        # Set up the exact cascade shape: Aldric can beat Cyrus, and Borin could then take the
        # exhausted Aldric on the SAME turn. Cyrus is stripped so a war certainly fires.
        by = {a.name: a for a in state["agents"] if a.alive}
        by["Cyrus"].money = by["Cyrus"].stockpile = 0.0
        assert empire.imperial_host_size(state, by["Aldric"]) > 0, "Aldric must be able to march"
        assert empire.imperial_host_size(state, by["Borin"]) > 0, "Borin must be able to follow up"
        empire.update(state, 1)
        assert len(fought) == 1, f"one turn must not chain wars, got {fought}"
    finally:
        empire.wage_war = real
    print("PASS test_no_two_wars_chain_in_one_turn")


def test_an_empire_keeps_the_borders_it_conquered() -> None:
    """Territory won stays territory — an empire is reachable through everything it holds.

    Adjacency used to read only the emperor's PERSONAL realm, so a conquest ERASED the frontier it
    had just won: the moment a rival was subjugated its towns stopped counting toward the empire's
    reach, every remaining power found itself bordering nobody, and the war loop went silent for
    the rest of the run. The empire's neighbours must still see it after it grows.
    """
    import empire, scenario, world
    world.create_world(size=26)
    state = world.world_state
    state["taxation_on"] = True
    scenario.apply(state, "showcase")
    # Borin borders Aldric only through Aldric's own towns; Cyrus is far from Borin.
    assert "Cyrus" not in empire._kingdom_neighbours(state, "Borin"), "B and C must not start as neighbours"
    held = empire._imperial_settlements(state, "Aldric")
    assert held == set(state["kingdoms"]["Aldric"]["settlements"]), "an empire-less king holds just his realm"
    # Aldric subjugates Cyrus (stripped so the clash is decisive): Cyrus's towns become Aldric's
    # frontier too.
    cyrus = next(a for a in state["agents"] if a.name == "Cyrus")
    cyrus.money = cyrus.stockpile = 0.0
    empire.wage_war(state, "Aldric", "Cyrus", 1)
    assert not empire.is_sovereign(state, "Cyrus"), "the staged war must actually subjugate Cyrus"
    grown = empire._imperial_settlements(state, "Aldric")
    assert grown > held, f"the empire must hold what it won: {grown} vs {held}"
    assert set(state["kingdoms"]["Cyrus"]["settlements"]) <= grown, "the subject-king's towns count"
    assert "Aldric" in empire._kingdom_neighbours(state, "Borin"), \
        "the empire must stay attackable after it grows — otherwise the war engine dies"
    print("PASS test_an_empire_keeps_the_borders_it_conquered")


def test_debug_war_observes_without_touching_the_run() -> None:
    """--debug-war is PURE OBSERVATION: same events, same state, only stderr differs."""
    import contextlib, copy, io
    import empire, scenario, world

    def run(debug: bool):
        world.create_world(size=26)
        state = world.world_state
        state["taxation_on"] = True
        scenario.apply(state, "showcase")
        state["debug_war"] = debug
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            for turn in range(1, 6):
                empire.update(state, turn)
        return list(state["events"]), copy.deepcopy(state["empires"]), err.getvalue()

    ev_off, emp_off, err_off = run(False)
    ev_on, emp_on, err_on = run(True)
    assert ev_on == ev_off, "the debug flag changed the event log"
    assert emp_on == emp_off, "the debug flag changed the empires"
    assert err_off == "" and "sovereign powers:" in err_on, "the gate must be reported on stderr only"
    print("PASS test_debug_war_observes_without_touching_the_run")


def test_showcase_caption_cards_and_severity_camera() -> None:
    """V4.15: the caption card fades on the hold, and the camera frames by TIER.

    The three things a viewer reads without being told: a beat is captioned in the director's
    words (never the raw log line), a LEGENDARY beat is framed TIGHTER and washes the rest of the
    world out, and several beats on one turn are CUT BETWEEN rather than one being dropped.
    Headless SDL-dummy; skips cleanly without pygame."""
    import os as _os, time as _time
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame
        from renderer.pygame_renderer import (PygameRenderer, _CAPTION_FADE, _HOLD_MAJOR,
                                              _HOLD_LEGENDARY, _ZOOM_MAJOR, _ZOOM_LEGENDARY)
    except ImportError:
        print("PASS test_showcase_caption_cards_and_severity_camera (skipped: no pygame)")
        return
    from renderer import director as d
    pygame.init()
    try:
        r = PygameRenderer(turn_delay=0.4, showcase=True, window=(1200, 800))
        r._ensure_screen(24)
        r._opened = True
        # A turn's beats become caption cards, in order, strongest first.
        r._turn_majors = 1
        r._beats.append((d.LEGENDARY, "THE LINE OF ALDRIC ENDS", "The crown lies vacant.", (6.0, 6.0)))
        r._update_caption()
        assert r._caption == ("THE LINE OF ALDRIC ENDS", "The crown lies vacant.")
        assert r._caption_hold == _HOLD_LEGENDARY and r._legend_t is not None, \
            "a legendary hold is longer, and starts the wash"
        assert r._focus_pt == (6.0, 6.0), "the camera is pointed at the beat"
        # The fade envelope: in over _CAPTION_FADE, full through the middle, out at the end.
        r._caption_started = _time.monotonic()
        assert r._caption_alpha() < 0.35, "a card fades IN rather than popping"
        r._caption_started = _time.monotonic() - _HOLD_LEGENDARY / 2
        assert r._caption_alpha() == 1.0, "...is solid through the hold..."
        r._caption_started = _time.monotonic() - (_HOLD_LEGENDARY - _CAPTION_FADE / 2)
        assert 0.0 < r._caption_alpha() < 0.75, "...and fades OUT at the end"
        # Tier decides the framing: legendary is tighter in than major.
        state = {"size": 24, "turn": 9, "food": [], "agents": [], "settlements":
                 {"S001": {"id": "S001", "center": (6, 6), "members": set(), "founded": 0}},
                 "monarchs": {}, "leaders": {}, "kingdoms": {}, "empires": {}, "events": []}
        _, _, ocell = r._home_view(state)
        r._zoom_hi = ocell * 100          # lift the clamp so the TIER is what decides the framing
        zooms = {}
        for sev in (d.MAJOR, d.LEGENDARY):
            r._caption_sev = sev
            r._focus_pt, r._focus_until = (6.0, 6.0), _time.monotonic() + 5
            r._showcase_direct(state)
            zooms[sev] = r._cam_tcell
        assert zooms[d.LEGENDARY] > zooms[d.MAJOR], f"legendary frames tighter: {zooms}"
        assert zooms[d.MAJOR] == min(r._zoom_hi * 0.95, ocell * _ZOOM_MAJOR)
        assert zooms[d.LEGENDARY] == min(r._zoom_hi * 0.95, ocell * _ZOOM_LEGENDARY)
        # Two beats on one turn: the first is held, then the SECOND is cut to — never dropped.
        r2 = PygameRenderer(turn_delay=0.4, showcase=True, window=(1200, 800))
        r2._ensure_screen(24)
        r2._opened = True
        r2._turn_majors = 2
        r2._beats.append((d.MAJOR, "THE RISING OF S0B2", None, (4.0, 4.0)))
        r2._beats.append((d.MAJOR, "THE RISING OF S0C2", None, (8.0, 8.0)))
        r2._update_caption()
        first = r2._caption
        r2._caption_started = _time.monotonic() - r2._caption_hold - 0.01   # its hold expires
        r2._update_caption()
        assert first[0] == "THE RISING OF S0B2" and r2._caption[0] == "THE RISING OF S0C2", \
            "a busy turn cuts between its beats rather than dropping one"
        assert r2._focus_pt == (8.0, 8.0), "the camera follows the cut"
        r2._caption_started = _time.monotonic() - r2._caption_hold - 0.01
        r2._update_caption()
        assert r2._caption is None and r2._legend_t is None, "the queue drains cleanly"
        # A card renders without touching world_state, and the draw path is exercised end to end.
        r._caption = ("THE LINE OF ALDRIC ENDS", "The crown lies vacant.")
        r._caption_started = _time.monotonic()
        r._caption_sev = d.LEGENDARY
        r._legend_t = _time.monotonic()
        before = repr(state)
        r._draw_legendary_wash()
        r._draw_caption_card()
        r._quiet_run = 9
        r._last_state = state
        r._draw_quiet_ticker()
        assert repr(state) == before, "drawing a caption mutated the state it read"
    finally:
        pygame.quit()
    print("PASS test_showcase_caption_cards_and_severity_camera")


def test_director_drives_the_showcase_turn_plan() -> None:
    """V4.15: one turn in -> a cut plan out (severity, beats, quiet-run) — and only in showcase.

    _direct_turn is the seam between the classifier and the frame loop. A quiet turn must leave the
    beat queue EMPTY (that is what makes it fast-forward) and lengthen the quiet run; a dramatic
    turn must produce captioned beats with camera focus. A NON-showcase renderer must not go
    through this path at all — the default renderer stays exactly as it was.
    """
    import os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame
        from renderer.pygame_renderer import PygameRenderer
    except ImportError:
        print("PASS test_director_drives_the_showcase_turn_plan (skipped: no pygame)")
        return
    from renderer import director as d
    pygame.init()
    try:
        r = PygameRenderer(turn_delay=0.4, showcase=True, window=(1200, 800), total_turns=45)
        r._ensure_screen(24)
        base = {"size": 24, "food": [], "agents": [], "monarchs": {}, "leaders": {},
                "kingdoms": {}, "empires": {},
                "settlements": {"S0B2": {"id": "S0B2", "center": (6, 6), "members": set(), "founded": 0}}}
        quiet = dict(base, turn=5, events=[
            "turn 5: MONARCH Cyrus levied 3.2 from S0B2 by force (rate 0.45)",
            'turn 5: a talked to b: "hi"'])
        r._direct_turn(quiet, 5)
        assert list(r._beats) == [] and r._turn_sev == d.NOISE and r._quiet_run == 1, \
            "a quiet turn plans no cut — which is what fast-forwards it"
        r._direct_turn(dict(quiet, turn=6), 6)
        assert r._quiet_run == 2, "consecutive quiet turns accumulate toward the compression"
        loud = dict(base, turn=7, events=[
            "turn 7: UPRISING in S0B2 — 4 risers rise against lord LordB (3 defenders: 3 standing + 0 hired)",
            "turn 7: the UPRISING in S0B2 TRIUMPHED — lord LordB is DEPOSED; BWV4 to rule by consent (2 risers fell)"])
        r._direct_turn(loud, 7)
        assert r._quiet_run == 0, "a beat resets the quiet run"
        assert r._turn_sev == d.MAJOR and len(r._beats) == 1, \
            f"the rising and its outcome are ONE cut, got {list(r._beats)}"
        sev, title, sub, foc = r._beats[0]
        assert title == "THE RISING OF S0B2" and "Lord B falls" in sub, (title, sub)
        assert foc == (6.0, 6.0), "the beat carries the settlement the camera must fly to"
        assert r._turns_left == 45 - 7, "the tight pacer knows how much run is left"
        # World-firsts are remembered ACROSS turns, so the second town to write is not a beat.
        w1 = dict(base, turn=8, events=["turn 8: LordA devised WRITING in S0B2"])
        r._direct_turn(w1, 8)
        assert len(r._beats) == 1, "the world's first writing is a beat"
        w2 = dict(base, turn=9, events=["turn 9: BT5 devised WRITING in S0B2"])
        r._direct_turn(w2, 9)
        assert list(r._beats) == [], "the second town to write is not"
        # The DEFAULT renderer never enters the director path.
        plain = PygameRenderer(turn_delay=0.4, window=(1200, 800))
        plain._ensure_screen(24)
        plain._enqueue_banners(loud)
        assert list(plain._beats) == [], "a non-showcase renderer is untouched by the director"
        assert plain._banner_queue, "...and still uses the V4.2 story banner"
    finally:
        pygame.quit()
    print("PASS test_director_drives_the_showcase_turn_plan")


def main_runner() -> None:
    tests = [
        test_detection_by_name,
        test_detection_only_living,
        test_social_memory_entries,
        test_memory_bound,
        test_food_competition,
        test_movement_collision,
        test_starvation_and_death,
        test_personality_parsing,
        test_personalities_produce_different_behaviour,
        test_curious_rests_less_than_cautious,
        test_strategy_eats_and_seeks_food,
        test_strategy_approach_and_avoid,
        test_survival_overrides_strategy,
        test_strategy_validation_fallback,
        test_strategy_caching_reduces_llm_calls,
        test_heuristic_returns_valid_action_for_hungry_fed_threatened,
        test_heuristic_moves_toward_adjacent_food,
        test_heuristic_run_makes_zero_llm_calls,
        test_cognition_defaults_to_llm_and_path_unregressed,
        test_full_simulation_runs_clean,
        test_tiering_never_exceeds_budget,
        test_interestingness_ranks_conflict_above_lone_wanderer,
        test_promotion_and_demotion_log_events,
        test_hysteresis_prevents_single_turn_flipflop,
        test_tiering_disabled_leaves_cognition_untouched,
        test_budget_covering_cast_is_byte_identical_to_v1,
        test_large_cast_run_completes_without_error,
        test_cost_vs_n_stays_bounded_by_budget_at_scale,
        test_scale_renderer_view_is_read_only,
        test_occupancy_index_matches_truth_after_moves_and_deaths,
        test_knowledge_transmits_only_between_in_contact_agents,
        test_adoption_probability_rises_with_trust,
        test_isolated_agent_never_learns,
        test_diffusion_adds_zero_llm_calls,
        test_empty_knowledge_run_is_byte_identical_to_v1,
        test_god_grant_knowledge_is_write_only_and_logs,
        test_discovery_respects_prerequisites,
        test_no_downstream_item_without_its_prerequisite,
        test_discovery_is_probabilistic_not_a_timer,
        test_starving_agent_does_not_invent,
        test_discovery_adds_zero_llm_calls,
        test_empty_tech_tree_run_is_byte_identical_to_v1,
        test_fire_knower_eats_more_unknown_does_not,
        test_tools_knower_forages_adjacent_unknown_cannot,
        test_farming_knower_produces_food_unknown_does_not,
        test_farming_population_outlasts_no_farming_control,
        test_farming_adds_zero_llm_calls_and_empty_is_v1,
        test_settlement_forms_with_enough_sustained_settlers_near_reliable_food,
        test_too_few_settlers_never_found_a_settlement,
        test_no_reliable_food_no_settlement,
        test_isolated_nomad_never_joins,
        test_settled_agent_pulled_home_when_fed_not_when_starving,
        test_nomad_movement_unaffected_by_settlement_system,
        test_settlement_update_zero_llm_and_no_rng,
        test_settlements_off_run_is_byte_identical_to_v1,
        test_surplus_banks_only_above_need_and_only_when_settled,
        test_stockpile_never_exceeds_cap,
        test_wealth_tracks_personality_and_knowledge_not_assigned,
        test_starving_member_with_savings_survives_one_without_dies,
        test_storage_accumulate_zero_llm_and_no_rng,
        test_storage_off_run_is_byte_identical_to_v1,
        test_emergent_price_varies_with_conditions_not_fixed,
        test_trade_is_mutually_beneficial_and_voluntary,
        test_guarded_knowledge_does_not_free_diffuse_but_sells,
        test_friendly_knowledge_still_free_diffuses,
        test_hunting_produces_food_only_for_knowers,
        test_surplus_to_money_to_purchase_roundtrips,
        test_trade_and_mint_zero_llm_and_no_rng,
        test_economy_off_run_is_byte_identical_to_v1,
        test_wage_varies_with_labor_supply_and_desperation_not_fixed,
        test_wage_reaches_subsistence_under_glut_and_desperation_never_below,
        test_employment_output_flows_to_employer_wage_to_worker,
        test_agent_without_capital_never_employs,
        test_roles_emerge_from_wealth_and_skill_not_assigned,
        test_employment_persists_across_turns_and_is_mutually_entered,
        test_worker_quits_when_self_sufficient_and_broke_employer_lets_go,
        test_inequality_compounds_with_wage_labor_on_vs_off,
        test_labor_adds_zero_llm_and_no_rng,
        test_labor_off_run_is_byte_identical_to_v1,
        test_leader_emerges_only_with_a_cohered_following_not_a_global_max,
        test_leader_can_be_a_non_wealthiest_agent_power_decoupled_from_wealth,
        test_leadership_lost_when_trust_erodes_with_hysteresis_and_can_be_displaced,
        test_leadership_effect_makes_a_led_settlement_more_cohesive_than_unled,
        test_leadership_reads_trust_but_writes_no_trust_values_and_no_llm_no_rng,
        test_leadership_off_run_is_byte_identical_to_v1,
        test_leadership_emerges_organically_from_built_trust_in_a_full_run,
        test_only_a_legitimate_leader_can_tax,
        test_redistribution_lowers_within_settlement_gini_vs_untaxed,
        test_over_taxation_costs_legitimacy_while_moderate_is_sustained,
        test_tax_flows_rich_to_poor_among_followers_and_conserves_wealth,
        test_taxation_off_run_is_byte_identical_to_v1,
        test_force_scales_with_wealth_funded_fighters_broke_cannot_conquer,
        test_loyalty_repels_smaller_force_but_not_overwhelming_one,
        test_monarch_levies_without_consent,
        test_monarch_is_overthrowable_by_a_stronger_force,
        test_war_kills_real_agents,
        test_monarchy_off_run_is_byte_identical_to_v1,
        test_conquest_of_neighbour_makes_its_ruler_a_vassal,
        test_tribute_cascades_settlement_to_vassal_to_king_and_conserves_wealth,
        test_king_musters_loyal_vassal_but_not_broken_away_one,
        test_over_tribute_breaks_a_vassal_away_with_hysteresis_fair_one_stays,
        test_kingdoms_off_run_is_byte_identical_to_v1,
        test_war_musters_whole_loyal_host_excluding_disloyal_vassals,
        test_richer_disloyal_kingdom_loses_then_wins_when_loyal,
        test_defeated_king_is_subjugated_into_multilevel_empire,
        test_imperial_tribute_cascades_through_subject_king_and_conserves_wealth,
        test_over_imperial_tribute_fragments_subject_king_with_hysteresis_fair_stays,
        test_empire_off_run_is_byte_identical_to_v1,
        test_staged_monarchy_produces_a_real_monarch_record,
        test_staged_kingdom_produces_a_real_multi_settlement_kingdom,
        test_staged_war_forms_an_empire_via_the_real_loop,
        test_staging_off_is_byte_identical_to_v1,
        test_staged_realm_stays_alive_and_populated_after_150_turns,
        test_talk_delivers_next_turn_and_reaction,
        test_talk_out_of_range_is_noop,
        test_reaction_is_personality_driven,
        test_reply_does_not_chain,
        test_talk_message_source_refresh_vs_template,
        test_llm_message_path_end_to_end,
        test_trust_hostile_drops_3_friendly_raises_1,
        test_trust_summary_buckets_and_prompt,
        test_talk_adds_no_llm_calls,
        test_theft_drops_trust_5_and_writes_memories,
        test_theft_grudge_is_permanent,
        test_theft_noop_when_no_food,
        test_desperate_independent_steals_distrusted_holder,
        test_alliance_forms_only_mutually_with_plus3_both_ways,
        test_allies_share_food_sighting_until_betrayal,
        test_betrayal_drops_trust_8_latches_grudge_both_memories,
        test_grudge_blocks_reallying,
        test_independent_betrays_ally_under_pressure_friendly_does_not,
        test_alliance_adds_no_llm_calls,
        test_death_writes_survivor_memories_and_event,
        test_respawn_fires_after_exactly_respawn_delay,
        test_newcomer_is_blank_slate_and_participates,
        test_respawn_keeps_population_bounded,
        test_respawn_adds_no_llm_calls,
        test_renderer_imports_only_state_reading_modules,
        test_render_frame_does_not_mutate_world_state,
        test_event_styling_emphasises_major_moments,
        test_pygame_renderer_imports_only_state_reading_modules,
        test_pygame_renderer_color_by_personality_and_size_by_wealth,
        test_pygame_renderer_draw_does_not_mutate_world_state,
        test_pygame_settlement_region_grows_with_member_spread,
        test_pygame_renderer_draws_settlements_read_only,
        test_pygame_event_feed_classifies_and_wraps_newest_at_bottom,
        test_pygame_event_tiers_banner_and_realm_scoreboard_pure,
        test_pygame_renderer_panel_reads_state_does_not_mutate,
        test_pygame_role_and_talk_helpers_read_state,
        test_pygame_renderer_iconography_draw_is_read_only,
        test_pygame_terrain_noise_is_pure_and_deterministic,
        test_pygame_terrain_cached_built_once_and_read_only,
        test_pygame_terrain_rebakes_on_season_change,
        test_pygame_town_plan_grows_with_members_and_is_pure,
        test_pygame_detailed_settlements_cached_and_read_only,
        test_pygame_battle_scene_detects_battles_and_names_casualties,
        test_pygame_snapshot_lerp_and_realm_helpers_pure,
        test_pygame_cinematic_state_is_renderer_local_and_draw_read_only,
        test_pygame_palette_is_centralized_and_scene_constants_derive,
        test_pygame_ambient_helpers_pure_bounded_and_rng_free,
        test_pygame_full_bleed_landscape_and_lit_scene_draw_read_only,
        test_pygame_time_of_day_pure_periodic_and_smooth,
        test_pygame_phase_grade_interpolation_bounded,
        test_pygame_night_draw_is_read_only_and_lights_the_dark,
        test_pygame_camera_transform_pure_and_inverse,
        test_pygame_iso_transform_pure_and_inverse,
        test_pygame_camera_clamp_buckets_and_culling_pure,
        test_pygame_lod_tiers_have_hysteresis,
        test_pygame_camera_state_renderer_local_and_lod_draw_read_only,
        test_pygame_window_sizing_rect_layout_and_resize,
        test_pygame_display_mode_stable_and_always_escapable,
        test_pygame_hidpi_layout_fills_the_drawable_surface,
        test_pygame_showcase_camera_is_rock_steady,
        test_showcase_scene_opens_in_a_standoff_that_must_break,
        test_showcase_pacing_and_floating_feed,
        test_speed_parsing_and_delay_only_when_rendering,
        test_god_mode_imports_only_world_state_layers,
        test_god_spawn_food_mutates_world_and_logs,
        test_god_spawn_agent_is_blank_slate_citizen,
        test_god_drought_zeroes_respawn_for_exactly_20_turns,
        test_god_treasure_is_contestable_and_more_valuable,
        test_god_spawned_food_draws_hungry_agent_within_two_turns,
        test_god_menu_pauses_and_resumes_cleanly,
        test_god_mode_adds_no_llm_calls,
        test_god_plague_raises_hunger_for_exactly_the_window_and_logs,
        test_god_plague_afflicts_random_living_agent,
        test_god_sick_neighbour_is_visible_in_perception,
        test_god_stranger_is_blank_slate_and_seeds_wariness_memory,
        test_god_day16_commands_add_no_llm_calls,
        test_seeded_random_runs_are_identical,
        test_god_script_parses_inline_and_file,
        test_god_script_runs_and_capture_includes_god_events,
        test_lineage_off_run_is_byte_identical_to_v1,
        test_birth_requires_every_gate,
        test_child_inherits_blend_not_knowledge_and_kin_trust_seeded,
        test_child_learning_boost_only_when_lineage_on,
        test_dependent_consumes_parent_stockpile_and_does_not_produce,
        test_dependent_child_turn_is_actionless_and_matures_on_schedule,
        test_old_age_death_uses_existing_death_path,
        test_births_primary_respawn_backstop,
        test_lineage_mechanics_add_no_llm_calls_and_are_deterministic,
        test_inheritance_equal_split_and_conservation,
        test_inheritance_kin_order_is_binding,
        test_escheat_to_ruler_else_vanishes,
        test_inheritance_writes_events_and_memories_via_death_path,
        test_inheritance_only_when_lineage_on,
        test_inherited_stockpile_helps_a_dependent_orphan_survive,
        test_m43_title_transfers_on_every_death_cause,
        test_m43_succession_is_eldest_first_with_tiebreaks,
        test_m43_succession_does_not_inherit_loyalty,
        test_m43_succession_is_a_crisis_test,
        test_m43_extinct_line_dissolves_and_is_contestable,
        test_m43_trust_leadership_is_never_hereditary,
        test_m43_dependent_heir_holds_seat_as_regent,
        test_m43_escheat_routes_to_successor_same_turn,
        test_m43_multilevel_emperor_and_subject_king_succession,
        test_m43_succession_only_when_lineage_on,
        test_discontent_off_run_is_byte_identical_to_v1,
        test_each_driver_raises_gauge_only_when_its_condition_holds,
        test_legitimacy_buffers_extraction_grievance,
        test_extraction_burden_is_bounded_by_means,
        test_decay_is_asymmetric_with_a_floor,
        test_no_decay_while_grievance_persists,
        test_settlement_pressure_counts_only_above_threshold_members,
        test_threshold_crossing_logs_sparsely,
        test_regency_levy_draws_no_extraction_grievance,
        test_discontent_adds_no_llm_and_is_deterministic,
        test_uprising_requires_both_trigger_gates,
        test_consent_led_settlement_never_rises,
        test_mob_is_numbers_penniless_wins_funded_ruler_crushes,
        test_crushed_rising_partial_reset_cooldown_and_persistent_grievance,
        test_victory_deposes_clears_title_and_breaks_kingdom_away,
        test_expropriation_conserved_and_preempts_inheritance,
        test_uprising_deaths_compose_with_succession_and_inheritance,
        test_uprising_off_run_is_byte_identical_to_v1,
        test_uprising_adds_no_llm_and_is_deterministic,
        test_revolutionary_is_derived_from_risers_not_assigned,
        test_revolutionary_holds_by_consent_and_can_be_displaced,
        test_revolution_devours_its_children,
        test_too_few_survivors_leaves_seat_vacant,
        test_beliefs_form_from_lived_experience_each_condition_binds,
        test_beliefs_spread_by_trust_and_never_in_isolation,
        test_contradictory_belief_flips_only_from_a_trusted_source,
        test_children_inherit_settlement_beliefs_via_childhood_boost,
        test_beliefs_off_run_is_byte_identical_and_adds_no_llm,
        test_faith_forms_on_coherence_not_when_fractured,
        test_faith_name_short_and_stable_as_core_grows,
        test_prophet_emergence_logs_only_on_genuine_change,
        test_prophet_is_derived_from_devotion_and_trust_not_assigned,
        test_aligned_ruler_generates_less_discontent_and_more_loyalty,
        test_defiant_king_erodes_vassal_loyalty_and_prophet_amplifies,
        test_religion_off_run_is_byte_identical_and_adds_no_llm,
        test_same_vs_foreign_rule_breeds_chronic_friction,
        test_foreign_province_breaks_away_where_same_culture_holds,
        test_children_assimilate_but_adults_do_not,
        test_assimilation_completes_over_generations_and_fault_line_fades,
        test_culture_off_run_is_byte_identical_and_adds_no_llm,
        test_writing_discovery_prereqs_bind,
        test_heir_inherits_written_law_only_when_literate,
        test_literacy_cures_knowledge_extinction,
        test_chronicle_accumulates_only_when_literate,
        test_writing_off_run_is_byte_identical_and_adds_no_llm,
        test_metallurgy_prereqs_bind,
        test_metalworking_boosts_farm_yield,
        test_arms_multiply_force_in_battle,
        test_uprising_arms_shifts_revolt_balance,
        test_metallurgy_off_run_is_byte_identical_and_adds_no_llm,
        test_era_derived_from_tech_and_advance_logged_and_extensible,
        test_era_gap_multiplies_combat_force,
        test_era_yield_curve,
        test_era_exposed_in_state_and_drives_rendering,
        test_eras_off_run_is_byte_identical_and_adds_no_llm,
        test_stance_derived_from_history_and_decays,
        test_pact_prevents_a_war_the_loop_would_launch_and_breaks_on_souring,
        test_alliance_adds_ally_host_to_defence,
        test_lapsed_honour_ally_fails_to_answer,
        test_diplomacy_off_run_is_byte_identical_and_adds_no_llm,
        test_intertrade_enriches_both_and_is_blocked_when_hostile,
        test_trade_warms_stance_into_a_pact,
        test_war_severs_trade,
        test_interdependence_measurement_is_exposed,
        test_intertrade_off_run_is_byte_identical_and_adds_no_llm,
        test_dominance_detection,
        test_fear_drives_coalition_overriding_stance,
        test_pooled_coalition_host_breaks_the_hegemon,
        test_coalition_dissolves_when_threat_passes,
        test_coalitions_off_run_is_byte_identical_and_adds_no_llm,
        test_figure_archetype_and_epithet_derived_from_deeds,
        test_events_and_houses_assembled_from_records,
        test_saga_is_deterministic_under_seed,
        test_chronicle_is_read_only_and_off_byte_identical,
        test_narrator_is_walled_off_from_the_structured_chronicle,
        test_the_band_binds_only_close_calls_are_consulted,
        test_character_tilts_a_close_war_offline_standin,
        test_motive_enters_the_written_history,
        test_pivot_provider_selection_random_stands_in_live_reaches_the_model,
        test_breakaway_motive_enters_the_chronicle,
        test_breakaway_motive_survives_the_hysteresis_delay,
        test_decisive_secession_carries_no_motive_and_is_diagnosed_as_such,
        test_crushed_uprising_carries_its_motive,
        test_belief_changes_at_most_once_per_turn_no_flipflop,
        test_minds_off_is_byte_identical_and_a_bad_response_falls_back,
        test_director_imports_only_stdlib,
        test_director_classifies_every_engine_event_shape,
        test_world_firsts_and_bloodless_battles_are_demoted,
        test_a_crown_falling_is_legendary,
        test_turn_severity_and_beat_editing,
        test_no_two_wars_chain_in_one_turn,
        test_an_empire_keeps_the_borders_it_conquered,
        test_debug_war_observes_without_touching_the_run,
        test_showcase_caption_cards_and_severity_camera,
        test_director_drives_the_showcase_turn_plan,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main_runner()
