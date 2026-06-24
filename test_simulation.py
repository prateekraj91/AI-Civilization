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
