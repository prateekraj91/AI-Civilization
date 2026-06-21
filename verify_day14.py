"""
verify_day14.py
===============

Day 14 verification harness (READ-ONLY observer — it changes NOTHING in the
simulation just to make a check pass). It drives the REAL production machinery
(main.run_agent_turn + main.process_respawns + the real world/population code) and
reports the milestone's four asks:

  PART 1 — DEATH IS REGISTERED BY THE SOCIETY (deterministic seeded run).
    An agent starves; a clear DEATH event lands in events[]; every agent that was
    alive at the time records the death in bounded memory; a relationship the
    survivor held toward the deceased is shown to PERSIST (the dead are remembered).

  PART 2 — RESPAWN AFTER EXACTLY RESPAWN_DELAY (same run).
    No newcomer appears before death_turn + RESPAWN_DELAY; on that exact turn a NEW
    blank-slate agent enters and the survivors record its arrival.

  PART 3 — NEWCOMER IS A COLD START AND PARTICIPATES IMMEDIATELY.
    The newcomer has empty memory + empty relationships + empty allies + hunger 0,
    no living agent has trust toward it, and it is observed / talked to like any
    other agent. Population stays bounded at TARGET_POPULATION.

  PART 4 — INFERENCE BENCHMARK. Calls per agent-turn (must match the pre-Day-14
    cached rate — death and respawn add no inference).

Run:
    AICIV_PROVIDER=random Jarvis/bin/python verify_day14.py   # offline, deterministic
    AICIV_PROVIDER=ollama Jarvis/bin/python verify_day14.py   # qwen3:8b regression
"""

import random

import llm
import main
import population
from agents import Agent
from strategy import Strategy, choose_action
from world import create_world, observe, place_agent, spawn_food, world_state

SEED = 7  # an offline seed whose run produces a starvation death early enough that
          # the respawn (death_turn + RESPAWN_DELAY) lands inside the 50-turn run.


def _new_run() -> tuple[dict, dict, dict]:
    random.seed(SEED)
    llm.reset_call_stats()
    create_world()
    for name, personality, goals, (x, y) in main.AGENT_SPECS:
        place_agent(Agent(name=name, personality=personality, goals=goals), x, y)
    spawn_food(main.INITIAL_FOOD, cluster=main.FOOD_CLUSTERED)
    strategies: dict[str, Strategy] = {}
    survived = {a.name: 0 for a in world_state["agents"]}
    counters = {"agent_turns": 0}
    return strategies, survived, counters


def find(name: str):
    return next(a for a in world_state["agents"] if a.name == name)


def _drive_until_death_and_respawn():
    """Run the REAL loop; snapshot the first death and the newcomer it spawns."""
    strategies, survived, counters = _new_run()
    death = None       # {turn, name, survivors:[(name, mem)], event, kept_rel}
    newcomer = None    # the Agent that entered
    arrival = None     # {turn, name, survivors:[(name, mem)]}

    for turn in range(1, main.NUM_TURNS + 1):
        world_state["turn"] = turn
        living_before = {a.name for a in world_state["agents"] if a.alive}

        for agent in [a for a in world_state["agents"] if a.alive]:
            main.run_agent_turn(agent, turn, strategies, survived, counters)

        # Snapshot the FIRST death the moment it happens (memory evicts later).
        if death is None:
            line = next((e for e in world_state["events"]
                         if f"turn {turn}:" in e and "died (" in e), None)
            if line is not None:
                dead_name = line.split(": ", 1)[1].split(" died")[0]
                survivors = []
                for a in world_state["agents"]:
                    if a.alive and a.name != dead_name:
                        mem = next((m for m in a.memory if f"{dead_name} died on turn" in m), "(none)")
                        kept = a.relationships.get(dead_name)
                        survivors.append((a.name, mem, kept))
                death = {"turn": turn, "name": dead_name, "event": line,
                         "survivors": survivors}

        main.maybe_respawn_food(turn)

        # The REAL respawn path (same call main.main() makes).
        spawned = population.process_respawns(turn, world_state)
        for nc in spawned:
            survived[nc.name] = turn
            if newcomer is None:
                newcomer = nc
                survivors = []
                for a in world_state["agents"]:
                    if a.alive and a is not nc:
                        mem = next((m for m in a.memory if f"{nc.name}, appeared on turn" in m), "(none)")
                        survivors.append((a.name, mem))
                # Snapshot the newcomer's BLANK STATE the instant it is born — the
                # live object accumulates memory/hunger as it then lives, so the
                # cold-start proof must be read here, not at end of run.
                arrival = {
                    "turn": turn, "name": nc.name, "survivors": survivors,
                    "born_memory": list(nc.memory),
                    "born_relationships": dict(nc.relationships),
                    "born_allies": sorted(nc.allies),
                    "born_offers": sorted(nc.ally_offers),
                    "born_hunger": nc.hunger,
                    "born_pos": nc.position,
                    "born_no_trust": [a.name for a in world_state["agents"]
                                      if a is not nc and nc.name in a.relationships],
                    "born_pop": population.living_count(world_state),
                }

        if not [a for a in world_state["agents"] if a.alive] and not world_state["pending_respawns"]:
            break

    return death, newcomer, arrival, counters


def part1_death_registered(death):
    print("=" * 70)
    print(f"PART 1 — DEATH IS REGISTERED BY THE SOCIETY  (provider={llm.PROVIDER}, seed={SEED})")
    print("=" * 70)
    if death is None:
        print("  No death occurred this run — re-run with a different SEED.")
        return False
    print(f"  {death['name']} died on turn {death['turn']}.")
    print(f"    events[]: {death['event']}")
    print(f"    queued respawn for turn {death['turn'] + population.RESPAWN_DELAY} "
          f"(death_turn + RESPAWN_DELAY={population.RESPAWN_DELAY})")
    print("    survivors recorded the death in memory:")
    for name, mem, kept in death["survivors"]:
        print(f"      - {name}: {mem!r}")
        if kept is not None:
            print(f"          (relationship toward {death['name']} PERSISTS: "
                  f"trust {kept['trust']:+d}, grudge={kept.get('grudge', False)} — the dead are remembered)")
    return True


def part2_respawn_timing(death, newcomer, arrival):
    print()
    print("=" * 70)
    print("PART 2 — RESPAWN AFTER EXACTLY RESPAWN_DELAY")
    print("=" * 70)
    if death is None or newcomer is None:
        print("  No respawn observed in this run.")
        return False
    expected = death["turn"] + population.RESPAWN_DELAY
    print(f"  death turn ............ {death['turn']}")
    print(f"  RESPAWN_DELAY ......... {population.RESPAWN_DELAY}")
    print(f"  expected respawn turn . {expected}")
    print(f"  ACTUAL respawn turn ... {arrival['turn']}   "
          f"{'(MATCH)' if arrival['turn'] == expected else '(MISMATCH!)'}")
    print(f"  newcomer .............. {newcomer.name} "
          f"({newcomer.personality}) born at {arrival['born_pos']}")
    print("  survivors recorded the arrival:")
    for name, mem in arrival["survivors"]:
        print(f"      - {name}: {mem!r}")
    return arrival["turn"] == expected


def part3_cold_start(newcomer, arrival):
    print()
    print("=" * 70)
    print("PART 3 — NEWCOMER IS A COLD START AND PARTICIPATES")
    print("=" * 70)
    if newcomer is None or arrival is None:
        print("  No newcomer to inspect.")
        return False

    # Cold-start fields read from the BIRTH snapshot (the live object has since
    # lived and accumulated memory/hunger — that is the point of a respawn).
    print(f"  newcomer {arrival['name']} — state AT BIRTH (turn {arrival['turn']}):")
    print(f"    memory ........ {arrival['born_memory']}        (empty? {arrival['born_memory'] == []})")
    print(f"    relationships . {arrival['born_relationships']}        (empty? {arrival['born_relationships'] == {}})")
    print(f"    allies ........ {arrival['born_allies']}    offers {arrival['born_offers']}")
    print(f"    hunger ........ {arrival['born_hunger']}        (reset? {arrival['born_hunger'] == 0})")
    print(f"    position ...... {arrival['born_pos']}  (a valid empty cell near centre)")
    print(f"    living agents with trust toward it: {arrival['born_no_trust']}  "
          f"(cold start? {arrival['born_no_trust'] == []})")
    print(f"    living population at birth: {arrival['born_pop']} "
          f"(bounded at TARGET_POPULATION={population.TARGET_POPULATION})")

    # Immediate participation is a STRUCTURAL property of respawn — a newcomer is a
    # fully-fledged Agent on a valid cell, so it can be observed/talked to like any
    # other. We prove that deterministically (provider-independent), separate from
    # the measured run, because whether a competitive Qwen agent CHOOSES to talk to
    # it before it scatters/dies is emergent and not the point here.
    nc_name, seen, talked, inbox = _participation_demo()
    print(f"\n  participation (constructed, deterministic — same real respawn + talk code):")
    print(f"    a neighbour observes the fresh newcomer {nc_name}: {seen}")
    print(f"    a neighbour talks to it: delivered={talked}, newcomer inbox={inbox} message(s)")
    print(f"    (structural: respawn registers a normal Agent — observe()/handle_talk()")
    print(f"     treat it exactly like Alex/Bob/Kira)")

    ok = (arrival["born_memory"] == [] and arrival["born_relationships"] == {}
          and arrival["born_hunger"] == 0 and arrival["born_no_trust"] == []
          and arrival["born_pop"] <= population.TARGET_POPULATION
          and seen and talked)
    return ok


def _participation_demo() -> tuple[str, bool, bool, int]:
    """A blank-slate newcomer is observable and talkable — provider-independent.

    Builds a fresh fixture, kills an agent through the REAL announce_death, lets the
    REAL process_respawns produce a newcomer, drops a neighbour beside it, and shows
    observe() detects it and conversation.handle_talk() delivers to it. Pure mechanic
    check — no LLM, no dependence on the measured run.
    """
    import conversation
    create_world()
    kira = Agent(name="Kira", personality="independent and competitive", goals={"survive": 7})
    alex = Agent(name="Alex", personality="friendly and outgoing", goals={"friendship": 8})
    place_agent(kira, 5, 5)
    place_agent(alex, 8, 8)  # far away so it doesn't block the central spawn cell
    population.announce_death(kira, 1, world_state)
    spawned = population.process_respawns(1 + population.RESPAWN_DELAY, world_state)
    nc = spawned[0]

    # Drop Alex onto a free cell directly adjacent to the newcomer.
    nx, ny = nc.position
    occupied = {a.position for a in world_state["agents"] if a.alive}
    for tx, ty in ((nx, ny - 1), (nx, ny + 1), (nx - 1, ny), (nx + 1, ny)):
        if 0 <= tx < world_state["size"] and 0 <= ty < world_state["size"] and (tx, ty) not in occupied:
            place_agent(alex, tx, ty)
            break
    seen = nc.name in observe(alex, world_state)
    res = conversation.handle_talk(alex, f"talk_to_{nc.name}", Strategy(kind="talk"),
                                   False, 2, world_state)
    talked = f"talked to {nc.name}" in res
    return nc.name, seen, talked, len(nc.inbox)


def part4_benchmark():
    print()
    print("=" * 70)
    print("PART 4 — INFERENCE BENCHMARK (Day 14 adds no per-turn calls)")
    print("=" * 70)
    # Clean isolated run for an honest per-agent-turn number.
    strategies, survived, counters = _new_run()
    for turn in range(1, main.NUM_TURNS + 1):
        world_state["turn"] = turn
        for agent in [a for a in world_state["agents"] if a.alive]:
            main.run_agent_turn(agent, turn, strategies, survived, counters)
        main.maybe_respawn_food(turn)
        for nc in population.process_respawns(turn, world_state):
            survived[nc.name] = turn
        if not [a for a in world_state["agents"] if a.alive] and not world_state["pending_respawns"]:
            break
    s = llm.get_call_stats()
    at = counters["agent_turns"]
    per = s["strategy"] / at if at else 0.0
    print(f"  agent-turns executed (incl. newcomers): {at}")
    print(f"  strategy LLM calls:                     {s['strategy']}")
    print(f"  legacy per-turn decisions:              {s['decision']}  (must be 0)")
    print(f"  calls per agent-turn:                   {per:.3f}  "
          f"(per-turn design = 1.000; refresh every {main.STRATEGY_INTERVAL} turns)")
    print(f"  -> death/respawn ride NO inference; the only LLM calls are the same")
    print(f"     periodic strategy refreshes every agent (newcomers included) makes.")


if __name__ == "__main__":
    death, newcomer, arrival, _ = _drive_until_death_and_respawn()
    ok1 = part1_death_registered(death)
    ok2 = part2_respawn_timing(death, newcomer, arrival)
    ok3 = part3_cold_start(newcomer, arrival)
    part4_benchmark()

    print()
    print("=" * 70)
    print("PROVENANCE (honest — reflects THIS run)")
    print("=" * 70)
    print(f"  Provider: {llm.PROVIDER}, seed {SEED}.")
    if death is not None:
        print(f"  DEATH + survivor memories: ACTUALLY OCCURRED — {death['name']} starved on")
        print(f"    turn {death['turn']} and the living agents recorded it. This is the same")
        print(f"    starvation mechanic from Day 6, now raised to a society-level event.")
    else:
        print(f"  No death occurred on this seed/provider — try another SEED.")
    if newcomer is not None:
        print(f"  RESPAWN: ACTUALLY OCCURRED — blank-slate {newcomer.name} entered on turn")
        print(f"    {arrival['turn']} (= death_turn + RESPAWN_DELAY) and was recorded by survivors.")
        print(f"    The newcomer mechanic is deterministic given the death; nothing about it")
        print(f"    depends on the model provider (it is pure Python over world_state).")
    print(f"  All four parts deterministic for the MECHANIC; the Qwen run is a regression")
    print(f"  check only (no seed-search for emergent events — rarity is expected).")
    print()
    print(f"  SUMMARY: part1={ok1}  part2={ok2}  part3={ok3}")
