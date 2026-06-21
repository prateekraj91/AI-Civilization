"""
verify_day12.py
===============

Day 12 verification harness (READ-ONLY observer — changes NOTHING in the
simulation to make a check pass). Drives the real production machinery
(main.run_agent_turn + the real world/strategy/conversation/trust code) for a
50-turn clustered-scarcity run and reports:

  1. A 50-turn log of who EATS / STARVES / TALKS / STEALS.
  2. A theft under scarcity: victim loses food, the THEFT is logged, and the
     victim's trust in the thief drops by exactly 5 and STAYS down (grudge).
  3. The victim's memory entry and the thief's memory entry.
  4. The victim references the grudge afterward — the trust line that would
     appear in its NEXT strategy prompt (built with the real prompt builder).
  5. The inference benchmark: calls-per-agent-turn (must be unchanged — Day 12
     rides the existing strategy call, zero new per-turn inference).

Reproducible: random is seeded so food placement / the offline provider replay
identically. Run with the real model via:

    AICIV_PROVIDER=ollama Jarvis/bin/python verify_day12.py     # Qwen, think-off
    AICIV_PROVIDER=random Jarvis/bin/python verify_day12.py     # offline fallback
"""

import random

import llm
import main
import trust
from agents import Agent
from strategy import Strategy, build_strategy_prompt
from world import create_world, observe, place_agent, spawn_food, world_state

SEED = 21


def run():
    random.seed(SEED)
    llm.reset_call_stats()
    create_world()
    for name, personality, goals, (x, y) in main.AGENT_SPECS:
        place_agent(Agent(name=name, personality=personality, goals=goals), x, y)
    spawn_food(main.INITIAL_FOOD, cluster=main.FOOD_CLUSTERED)

    strategies: dict[str, Strategy] = {}
    survived = {a.name: 0 for a in world_state["agents"]}
    counters = {"agent_turns": 0}
    thefts = []  # (turn, thief, victim)

    print("=" * 66)
    print(f"DAY 12 — STEAL + RETALIATION  (provider={llm.PROVIDER}, seed={SEED})")
    print(f"clustered scarce food: INITIAL_FOOD={main.INITIAL_FOOD}, "
          f"+{main.FOOD_RESPAWN_AMOUNT}/{main.FOOD_RESPAWN_EVERY} turns")
    print("=" * 66)
    print(" turn | food | events")
    print("-" * 66)

    for turn in range(1, main.NUM_TURNS + 1):
        world_state["turn"] = turn
        food_before = len(world_state["food"])
        evs = []
        for agent in [a for a in world_state["agents"] if a.alive]:
            action = main.run_agent_turn(agent, turn, strategies, survived, counters)
            if action == "eat":
                evs.append(f"{agent.name} ate")
            elif action == "starved":
                evs.append(f"{agent.name} STARVED")
            elif action.startswith("talk_to_"):
                evs.append(f"{agent.name} talked->{action[len('talk_to_'):]}")
            elif action.startswith("steal_from_"):
                victim_name = action[len("steal_from_"):]
                victim_obj = next(a for a in world_state["agents"]
                                  if a.name == victim_name)
                # Snapshot the memories AT theft time — by end of run the thief's
                # bounded memory may have evicted them under later events.
                thief_mem = next((m for m in reversed(agent.memory)
                                  if m.startswith("I stole from")), None)
                victim_mem = next((m for m in reversed(victim_obj.memory)
                                   if "stole my food" in m), None)
                thefts.append((turn, agent.name, victim_name, thief_mem, victim_mem))
                evs.append(f"*** {agent.name} STOLE from {victim_name} ***")
        main.maybe_respawn_food(turn)
        if evs:
            print(f" {turn:>4} | {food_before:>4} | " + ", ".join(evs))
        if not [a for a in world_state["agents"] if a.alive]:
            print(f"      |      | (all agents dead — run ends at turn {turn})")
            break

    return {"survived": survived, "counters": counters, "thefts": thefts}


def find(name):
    return next(a for a in world_state["agents"] if a.name == name)


def report_theft(r):
    print()
    print("=" * 66)
    print("THEFT UNDER SCARCITY")
    print("=" * 66)
    if not r["thefts"]:
        print("  No theft this seed — re-run with a different SEED.")
        return None
    turn, thief_name, victim_name, thief_mem, victim_mem = r["thefts"][0]
    thief, victim = find(thief_name), find(victim_name)
    print(f"  First theft: turn {turn} — {thief_name} stole from {victim_name}.")
    print(f"  Total thefts in run: {len(r['thefts'])}  "
          f"{[(t, th, v) for t, th, v, _, _ in r['thefts']]}")

    print()
    print("  events[] lines for this theft:")
    for e in world_state["events"]:
        if (f"stole food from {victim_name}" in e and f"turn {turn}:" in e) \
           or (f"{victim_name} trust in {thief_name}" in e and "theft" in e):
            print(f"    - {e}")

    rel = victim.relationships.get(thief_name, {})
    print()
    print(f"  Victim trust: {victim_name} -> {thief_name} = {rel.get('trust')}  "
          f"(penalty {trust.THEFT_PENALTY}, grudge={rel.get('grudge')})")

    print()
    print("  Memory entries (captured at theft time):")
    for t, th, v, tm, vm in r["thefts"]:
        print(f"    turn {t}: victim ({v}) remembers: {vm!r}")
        print(f"             thief  ({th}) remembers: {tm!r}")
    return (thief_name, victim_name)


def report_grudge_persists(names):
    print()
    print("=" * 66)
    print("VICTIM REFERENCES THE GRUDGE AFTERWARD")
    print("=" * 66)
    if names is None:
        return
    thief_name, victim_name = names
    victim = find(victim_name)

    # The trust line the victim WOULD carry into its next strategy prompt — built
    # with the real prompt builder, no LLM call. Proves the grudge is surfaced.
    summary = trust.trust_summary(victim)
    print(f"  trust_summary({victim_name}): {summary!r}")
    prompt = build_strategy_prompt(victim, observe(victim, world_state)
                                   if victim.alive else "Current Tile: empty")
    grudge_line = next((ln for ln in prompt.splitlines() if "Your trust" in ln), "")
    print(f"  -> appears verbatim in {victim_name}'s next strategy prompt:")
    print(f"     {grudge_line!r}")
    holds = thief_name in summary and "grudge" in summary
    print(f"  grudge still present and surfaced to the model: {holds}")

    # Demonstrate permanence directly: a friendly +1 cannot lift it.
    before = victim.relationships[thief_name]["trust"]
    trust.adjust_trust(victim, thief_name, +1, "friendly message", 999, world_state)
    after = victim.relationships[thief_name]["trust"]
    print(f"  forgiveness attempt (+1 friendly): {before} -> {after}  "
          f"({'REFUSED — permanent' if before == after else 'changed!'})")


def report_benchmark(r):
    print()
    print("=" * 66)
    print("INFERENCE BENCHMARK (Day 12 adds no per-turn calls)")
    print("=" * 66)
    stats = llm.get_call_stats()
    at = r["counters"]["agent_turns"]
    per = stats["strategy"] / at if at else 0.0
    print(f"  agent-turns executed:      {at}")
    print(f"  strategy LLM calls:        {stats['strategy']}")
    print(f"  legacy per-turn decisions: {stats['decision']}  (must be 0)")
    print(f"  calls per agent-turn:      {per:.3f}  "
          f"(per-turn design = 1.000; steal/talk ride the same call)")


if __name__ == "__main__":
    r = run()
    names = report_theft(r)
    report_grudge_persists(names)
    report_benchmark(r)
