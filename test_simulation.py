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

import llm
import main
from agents import Agent
from personality import Personality
from strategy import Strategy, choose_action, get_personality
from world import (
    MEMORY_LIMIT,
    create_world,
    execute_action,
    is_dead,
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
        test_full_simulation_runs_clean,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main_runner()
