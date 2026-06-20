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

import conversation
import llm
import main
from agents import Agent
from personality import Personality
from strategy import Strategy, build_strategy_prompt, choose_action, get_personality
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
        test_talk_delivers_next_turn_and_reaction,
        test_talk_out_of_range_is_noop,
        test_reaction_is_personality_driven,
        test_reply_does_not_chain,
        test_talk_message_source_refresh_vs_template,
        test_llm_message_path_end_to_end,
        test_trust_hostile_drops_3_friendly_raises_1,
        test_trust_summary_buckets_and_prompt,
        test_talk_adds_no_llm_calls,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main_runner()
