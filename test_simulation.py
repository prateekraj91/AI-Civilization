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
    assert stats == {"decision": 0, "strategy": 0}, stats
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
    allowed = {"__future__", "typing", "contextlib", "os", "sys", "time", "math", "textwrap", "pygame"}
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
    assert win_w == r._cell * state["size"] + _PANEL_W, "the window widened by the panel zone"
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
        r._draw(state)
        r._draw(state)
        assert r._terrain_bg is bg, "the cached terrain is reused across frames (not regenerated)"
    finally:
        pygame.quit()
    after = {k: state[k] for k in state if k != "agents"}
    assert after == before, "terrain/farmland draw mutated world_state"
    print("PASS test_pygame_terrain_cached_built_once_and_read_only")


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
        test_pygame_renderer_panel_reads_state_does_not_mutate,
        test_pygame_role_and_talk_helpers_read_state,
        test_pygame_renderer_iconography_draw_is_read_only,
        test_pygame_terrain_noise_is_pure_and_deterministic,
        test_pygame_terrain_cached_built_once_and_read_only,
        test_pygame_town_plan_grows_with_members_and_is_pure,
        test_pygame_detailed_settlements_cached_and_read_only,
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
