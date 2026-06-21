"""
verify_day13.py
===============

Day 13 verification harness (READ-ONLY observer — it changes NOTHING in the
simulation just to make a check pass). It drives the REAL production machinery
(main.run_agent_turn + the real world/strategy/alliance/conversation/trust code)
and reports four things the milestone asks for:

  PART 1 — EMERGENT ALLIANCE (seeded production run, offline provider).
    A clustered-scarcity run in which an alliance FORMS on its own: trust rises
    +3 BOTH ways, an ALLIANCE event and a memory land on both agents, and the two
    allies then demonstrably SHARE a food sighting (one sees food the other does
    not, surfaced verbatim in the partner's real strategy prompt).

  PART 2 — BETRAYAL (constructed deterministic scenario).
    Why constructed: only an independent/competitive agent (Kira) betrays, and
    only while ALLIED — but Kira allies just reluctantly (high trust only), so in
    the offline seeds she essentially never enters an alliance to betray. So we
    build the allied state through the REAL handlers, then let the REAL executor
    (strategy.choose_action) decide: starving beside an ally on food, Kira emits
    betray_alliance. The alliance dissolves, trust drops 8, a PERMANENT grudge
    latches, and both memories record it.

  PART 3 — GRUDGE BLOCKS RE-ALLYING. The betrayed pair tries to ally again; the
    real can_ally / handle_ally refuse it from either side.

  PART 4 — INFERENCE BENCHMARK. Calls per agent-turn from Part 1 (must be the
    same as before Day 13 — alliance/betray ride the existing strategy call).

HONESTY NOTE (printed at the end too): Part 1 is a seeded run on the OFFLINE
'random' provider (it never itself emits 'ally' — the alliance is produced by the
deterministic strategy EXECUTOR once talk has built trust, which is exactly the
emergent path). Part 2's betrayal is a DETERMINISTIC constructed scenario, not an
organically-occurring Qwen event. Run under Qwen with:

    AICIV_PROVIDER=ollama Jarvis/bin/python verify_day13.py   # qwen3:8b, think-off
    AICIV_PROVIDER=random Jarvis/bin/python verify_day13.py   # offline (default here)
"""

import random

import alliance
import llm
import main
import trust
from agents import Agent
from strategy import Strategy, build_strategy_prompt, choose_action
from world import create_world, observe, place_agent, spawn_food, world_state

SEED = 48  # a seed whose offline run forms an Alex+Bob alliance early (turn 7)


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


def part1_emergent_alliance():
    print("=" * 70)
    print(f"PART 1 — EMERGENT ALLIANCE  (provider={llm.PROVIDER}, seed={SEED})")
    print("=" * 70)
    import world
    strategies, survived, counters = _new_run()

    formed = None          # snapshot dict, taken AT formation (memory evicts later)
    share_evidence = None   # snapshot dict, taken AT the first shared sighting
    print(" turn | events")
    print("-" * 70)
    for turn in range(1, main.NUM_TURNS + 1):
        world_state["turn"] = turn
        evs = []
        for agent in [a for a in world_state["agents"] if a.alive]:
            action = main.run_agent_turn(agent, turn, strategies, survived, counters)
            if action.startswith("ally_with_"):
                evs.append(f"{agent.name}->ally {action[len('ally_with_'):]}")
            elif action.startswith("betray_alliance_"):
                evs.append(f"{agent.name}->BETRAY {action[len('betray_alliance_'):]}")
            elif action == "eat":
                evs.append(f"{agent.name} ate")
            elif action == "starved":
                evs.append(f"{agent.name} STARVED")

        # Snapshot the first ALLIANCE the MOMENT it forms — trust and the memory
        # lines are captured now because bounded memory evicts them by end of run.
        if formed is None:
            line = next((e for e in world_state["events"]
                         if f"turn {turn}:" in e and "formed an ALLIANCE" in e), None)
            if line is not None:
                parts = line.split(": ", 1)[1].replace(" formed an ALLIANCE", "")
                a, b = [p.strip() for p in parts.split(" and ")]
                ra, rb = find(a), find(b)
                formed = {
                    "turn": turn, "a": a, "b": b, "event": line,
                    "trust_a": ra.relationships[b]["trust"],
                    "trust_b": rb.relationships[a]["trust"],
                    "allies_a": sorted(ra.allies), "allies_b": sorted(rb.allies),
                    "mem_a": next((m for m in ra.memory if "allied with" in m), "(none)"),
                    "mem_b": next((m for m in rb.memory if "allied with" in m), "(none)"),
                }

        # Snapshot the first turn an ally shares a sighting the viewer cannot see —
        # including the verbatim prompt line, built from the live state THIS turn.
        if formed is not None and share_evidence is None:
            for agent in [a for a in world_state["agents"] if a.alive]:
                shared = alliance.shared_food_sightings(agent, world_state)
                if shared:
                    prompt = build_strategy_prompt(agent, observe(agent, world_state),
                                                   state=world_state)
                    line = next((ln for ln in prompt.splitlines()
                                 if "shared with you" in ln), "")
                    share_evidence = {
                        "turn": turn, "viewer": agent.name, "shared": shared,
                        "own": sorted(world.visible_food(agent, world_state)),
                        "prompt_line": line,
                    }
                    break

        main.maybe_respawn_food(turn)
        if evs:
            print(f" {turn:>4} | " + ", ".join(evs))
        if not [a for a in world_state["agents"] if a.alive]:
            break

    if formed is None:
        print("\n  No alliance formed this seed — re-run with a different SEED.")
        return None, counters

    f = formed
    print()
    print(f"  ALLIANCE formed turn {f['turn']}: {f['a']} <-> {f['b']}")
    print(f"    events[]: {f['event']}")
    print(f"    trust both ways:  {f['a']}->{f['b']} = {f['trust_a']:+d}   "
          f"{f['b']}->{f['a']} = {f['trust_b']:+d}   "
          f"(formation grants +{alliance.ALLY_TRUST_BONUS} each)")
    print(f"    {f['a']} allies: {f['allies_a']}   {f['b']} allies: {f['allies_b']}")
    print(f"    memory ({f['a']}): {f['mem_a']!r}")
    print(f"    memory ({f['b']}): {f['mem_b']!r}")

    print()
    print("  SHARED FOOD SIGHTING (the concrete benefit):")
    if share_evidence is None:
        print("    (allies never split far enough to see different food this run)")
    else:
        s = share_evidence
        print(f"    turn {s['turn']}: {s['viewer']} learns of food only its ally sees:")
        for ally_name, coords in s["shared"].items():
            print(f"      - ally {ally_name} sees food at "
                  f"{', '.join(str(c) for c in coords)}; "
                  f"{s['viewer']}'s own view is {s['own']}")
        print(f"    -> appears verbatim in {s['viewer']}'s strategy prompt that turn:")
        print(f"       {s['prompt_line']!r}")
    return formed, counters


def part2_betrayal():
    print()
    print("=" * 70)
    print("PART 2 — BETRAYAL  (constructed deterministic scenario)")
    print("=" * 70)
    # Build the allied state through the REAL handlers: Kira (independent) and Alex
    # are adjacent with trust already high enough for a reluctant loner to ally.
    create_world()
    kira = Agent(name="Kira", personality="independent and competitive",
                 goals={"survive": 7, "wealth": 8, "friendship": 1})
    alex = Agent(name="Alex", personality="friendly and outgoing",
                 goals={"survive": 7, "friendship": 8, "wealth": 2})
    place_agent(kira, 5, 5)
    place_agent(alex, 5, 4)  # adjacent (north of Kira)
    kira.relationships["Alex"] = {"trust": 4, "interactions": 3, "grudge": False}
    alex.relationships["Kira"] = {"trust": 4, "interactions": 3, "grudge": False}

    # Real mutual handshake: Alex proposes, Kira accepts -> alliance forms.
    alliance.handle_ally(alex, "ally_with_Kira", 1, world_state)
    alliance.handle_ally(kira, "ally_with_Alex", 2, world_state)
    print(f"  allied via real handlers: Kira.allies={sorted(kira.allies)}, "
          f"Alex.allies={sorted(alex.allies)}")
    print(f"  trust after forming: Kira->Alex={kira.relationships['Alex']['trust']:+d}, "
          f"Alex->Kira={alex.relationships['Kira']['trust']:+d}")

    # Now survival pressure: Kira is starving, Alex is sitting on the only food in
    # reach. The REAL executor decides — no scripted action.
    kira.hunger = 8
    world_state["food"].append(alex.position)  # Alex hoards the food
    pre_trust = alex.relationships["Kira"]["trust"]
    action, note = choose_action(kira, Strategy(kind="wander"), world_state)
    print(f"\n  executor decision for starving Kira beside ally Alex-on-food:")
    print(f"    action = {action!r}   ({note})")
    assert action == "betray_alliance_Alex", action

    result = alliance.handle_betray(kira, action, 3, world_state)
    print(f"    -> {result}")
    print()
    print("  AFTER BETRAYAL:")
    print(f"    alliance dissolved both sides: "
          f"Kira.allies={sorted(kira.allies)}, Alex.allies={sorted(alex.allies)}")
    rel = alex.relationships["Kira"]
    print(f"    betrayed Alex trust in Kira: {pre_trust:+d} -> {rel['trust']:+d}  "
          f"(penalty {alliance.BETRAYAL_PENALTY}, grudge={rel['grudge']})")
    print(f"    events[]: " + next(e for e in world_state["events"] if "BETRAYED" in e))
    print(f"    memory (betrayed Alex): "
          f"{next(m for m in alex.memory if 'BETRAYED' in m)!r}")
    print(f"    memory (betrayer Kira): "
          f"{next(m for m in kira.memory if 'BETRAYED' in m)!r}")
    return kira, alex


def part3_grudge_blocks_really(kira, alex):
    print()
    print("=" * 70)
    print("PART 3 — GRUDGE BLOCKS RE-ALLYING")
    print("=" * 70)
    print(f"  can_ally(Alex, Kira) = {alliance.can_ally(alex, kira)}  "
          f"(grudge on betrayed side)")
    print(f"  can_ally(Kira, Alex) = {alliance.can_ally(kira, alex)}  "
          f"(blocked from EITHER direction)")
    # Real ally_with actions are refused, not silently formed.
    r1 = alliance.handle_ally(alex, "ally_with_Kira", 4, world_state)
    r2 = alliance.handle_ally(kira, "ally_with_Alex", 5, world_state)
    print(f"  Alex tries ally_with_Kira -> {r1}")
    print(f"  Kira tries ally_with_Alex -> {r2}")
    print(f"  still not allied: {not alliance.are_allied(alex, kira)}")

    # And the grudge rides into Alex's next prompt — he references the betrayal.
    prompt = build_strategy_prompt(alex, "Current Tile: empty", state=world_state)
    grudge_line = next((ln for ln in prompt.splitlines() if "Your trust" in ln), "")
    print(f"  grudge surfaced to the model: {grudge_line!r}")


def part4_benchmark(counters):
    print()
    print("=" * 70)
    print("PART 4 — INFERENCE BENCHMARK (Day 13 adds no per-turn calls)")
    print("=" * 70)
    stats = llm.get_call_stats()  # NOTE: reflects Part 2/3 too; report Part 1's count
    print("  (counters below are from the full process; Part 1's run is the")
    print("   representative production loop — alliance/betray add 0 inference)")
    at = counters["agent_turns"]
    # Re-run Part 1 cleanly to isolate its call count for an honest per-turn number.
    strategies, survived, counters2 = _new_run()
    for turn in range(1, main.NUM_TURNS + 1):
        world_state["turn"] = turn
        for agent in [a for a in world_state["agents"] if a.alive]:
            main.run_agent_turn(agent, turn, strategies, survived, counters2)
        main.maybe_respawn_food(turn)
        if not [a for a in world_state["agents"] if a.alive]:
            break
    s = llm.get_call_stats()
    at2 = counters2["agent_turns"]
    per = s["strategy"] / at2 if at2 else 0.0
    print(f"  agent-turns executed:      {at2}")
    print(f"  strategy LLM calls:        {s['strategy']}")
    print(f"  legacy per-turn decisions: {s['decision']}  (must be 0)")
    print(f"  calls per agent-turn:      {per:.3f}  "
          f"(per-turn design = 1.000; ally/betray ride the same call)")


if __name__ == "__main__":
    formed, counters = part1_emergent_alliance()
    kira, alex = part2_betrayal()
    part3_grudge_blocks_really(kira, alex)
    part4_benchmark(counters)
    print()
    print("=" * 70)
    print("PROVENANCE (honest — reflects THIS run)")
    print("=" * 70)
    if formed is not None:
        print(f"  PART 1 alliance + food-sharing: ACTUALLY OCCURRED on provider")
        print(f"    '{llm.PROVIDER}', seed {SEED}, turn {formed['turn']}. Emergent — the")
        print(f"    provider never itself emits 'ally'; talk built the trust that the")
        print(f"    deterministic strategy executor turned into a mutual alliance.")
    else:
        print(f"  PART 1 alliance: NO alliance formed on provider '{llm.PROVIDER}', "
              f"seed {SEED}.")
        print(f"    On this run the agents did not build enough trust to ally (under")
        print(f"    Qwen, competitive play tends to let one agent out-compete the rest).")
        print(f"    The alliance + food-sharing demonstration is reproducible on the")
        print(f"    offline 'random' provider at this seed — run with AICIV_PROVIDER=random.")
    print(f"  PART 2 betrayal: a CONSTRUCTED deterministic scenario (no LLM on that")
    print(f"    path). Kira is rarely allied in practice, so betrayal does not occur")
    print(f"    organically. The mechanic (dissolve, -8, permanent grudge, both")
    print(f"    memories) is real production code; the SETUP is hand-built, not an")
    print(f"    observed {llm.PROVIDER} event.")
