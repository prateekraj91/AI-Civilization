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
    assert stats == {"decision": 0, "strategy": 0}, stats
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
    assert stats == {"decision": 0, "strategy": 0}, stats
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
    assert stats == {"decision": 0, "strategy": 0}, stats

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
    assert stats == {"decision": 0, "strategy": 0}, stats
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
    assert stats == {"decision": 0, "strategy": 0}, stats
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
    assert stats == {"decision": 0, "strategy": 0}, stats
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
    assert stats == {"decision": 0, "strategy": 0}, stats
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
    assert stats == {"decision": 0, "strategy": 0}, stats
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
    assert llm.get_call_stats() == {"decision": 0, "strategy": 0}, llm.get_call_stats()
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
    assert llm.get_call_stats() == {"decision": 0, "strategy": 0}, llm.get_call_stats()
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
    assert llm.get_call_stats() == {"decision": 0, "strategy": 0}, llm.get_call_stats()
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
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main_runner()
