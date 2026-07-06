"""
knowledge.py
============

Knowledge as PROPAGATING STATE — V2 milestone M1.1, the first piece of culture.

The idea
--------
Phase 0 built a scalable social-contact network: at 200+ agents they meet, talk,
and build trust through actual proximity. M1.1 puts the first cultural artefact
ONTO that network — a `knowledge` set on each agent (named facts/skills like
"fire" or "food_location_north") that exists beyond any single agent and SPREADS
between them through contact.

Crucially this is CHEAP STATE DIFFUSION, not an LLM call per learner: when a
knower and a non-knower are adjacent (the same N/S/E/W contact channel talk/steal/
ally already use), the item may be adopted with a probability shaped by how much
the LEARNER trusts the teacher and by the learner's personality. Pure Python, zero
inference, so it scales to hundreds of agents at no added model cost.

What it is NOT (kept strictly in scope)
---------------------------------------
No discovery/invention (that is M1.2) — items are SEEDED into agents (at setup or
via a god grant) and this module only moves EXISTING items along the contact graph.
No tech-changes-the-world effects (that is M1.3) — knowing "fire" does nothing to
an agent's behaviour yet; M1.1 is solely about the item spreading correctly and
cheaply. Knowing an item has no behavioural effect, so a run with NO seeded
knowledge is byte-identical to v1 (and `diffuse` consumes zero RNG then — see the
guard in `diffuse`).

Determinism
-----------
Every iteration over a knowledge set is `sorted(...)` so the order of `rng` draws
is independent of Python's per-process string-hash randomisation — two seeded runs
diffuse identically. Transmissions are computed from a SNAPSHOT of who-knows-what
at the start of the turn and applied after, so an item travels at most one hop per
turn (no within-turn chain reactions) and the result never depends on agent order.
"""

from __future__ import annotations

import random
from typing import Any

import world
from strategy import SURVIVAL_HUNGER, get_personality

# --- Tech tree (M1.2: discovery) -------------------------------------------
# A tiny prerequisite tree, declared as DATA (not branching code): {item -> the
# items that must be known first}. fire is a base discovery (no prereq); it branches
# into tools and cooking; tools unlocks farming. Discovery walks this map, so adding
# an item is a one-line data change, never new control flow. Kept deliberately small
# and generic — M1.2 is about the INVENTION mechanic, not a real tech tree.
#
# This is the canonical tree. It is only ACTIVE in a run that opts in (run_simulation's
# `tech_tree` arg / the --tech-tree CLI flag); the default trio run gets no tree, so
# `discover` is a no-op that draws zero RNG and v1 stays byte-identical.
TECH_TREE: dict[str, frozenset[str]] = {
    "fire": frozenset(),                 # base: discoverable by anyone
    "tools": frozenset({"fire"}),        # needs fire
    "cooking": frozenset({"fire"}),      # needs fire (a sibling branch of tools)
    "farming": frozenset({"tools"}),     # needs tools (and so, transitively, fire)
    # M2.3 specialization: hunting is a SECOND producer skill, a SIBLING of farming off the
    # same `tools` prereq. It produces food into world_state too (see `hunt`) but by a
    # different mechanic/location (roaming game on a wider ring, not a tended adjacent plot),
    # so a population can hold two DISTINCT producer skills — the specialization that gives
    # knowledge-trade something real to exchange (a farmer lacks hunting and vice-versa).
    "hunting": frozenset({"tools"}),     # needs tools — the farming-sibling producer skill
}

# --- Proprietary knowledge (M2.3) ------------------------------------------
# PROPRIETARY: the skills that carry trade value and so CAN be guarded — i.e. withheld from
# free M1.1 diffusion and released only for payment (sold via economy.trade). It is the whole
# tech vocabulary; WHETHER a holder actually guards a given item is decided per-agent by
# `guards`, purely from personality — never assigned. (Free diffusion is unchanged for every
# non-guarding holder; trade is an ADDITIONAL path, only for what a guard won't give away.)
PROPRIETARY: frozenset[str] = frozenset(TECH_TREE)


def guards(agent: Any, item: str) -> bool:
    """Whether `agent` WITHHOLDS `item` from free diffusion to sell it instead (M2.3).

    EMERGES from personality, not an assignment: an independent/competitive agent GUARDS its
    valuable skills (treats know-how as property to be sold); a friendly/curious/cautious
    agent does not — it still TEACHES the item free through the unchanged M1.1 diffusion. So
    the SAME skill leaks free from a generous holder while a competitive holder tries to sell
    it. Only PROPRIETARY items can be guarded (a non-skill fact has no trade value to guard).
    Pure read of personality + the item set; no LLM, no mutation.
    """
    if item not in PROPRIETARY:
        return False
    return get_personality(agent).dominant == "independence"

# --- Discovery model (M1.2) ------------------------------------------------
# Base per-agent-per-turn chance of inventing an item whose prereqs are all known,
# BEFORE personality/situation shaping. Small on purpose: discovery must be a rare
# situational roll, never a timer, so the turn it first fires VARIES by run.
DISCOVERY_BASE = 0.02

# Personality shapes the inventor: the curious tinker most; the cautious least. (The
# independent tinker a fair bit — a loner experimenting alone — the friendly less, as
# they spend their spare capacity socialising.) Multiplies the base chance.
DISCOVERY_PERSONALITY: dict[str, float] = {
    "curiosity": 2.0,
    "independence": 1.0,
    "friendliness": 0.8,
    "caution": 0.6,
}

# Situation: only an agent with spare capacity tinkers — a fed agent invents, a hungry
# one is busy surviving and a starving one not at all. Hunger scales the chance down
# to zero at this cutoff, so discovery rides the SAME scarcity pressure the rest of the
# sim runs on (idle/fed -> invent; starving -> never). Tied to SURVIVAL_HUNGER so it
# lines up with where the executor's survival override already takes over.
DISCOVERY_HUNGER_CUTOFF = SURVIVAL_HUNGER

# --- Farming production (M1.2 item -> M1.3 world effect) --------------------
# Farming is the headline M1.3 effect: a KNOWER produces food INTO world_state, so a
# population that knows it can manufacture supply instead of only racing for a fixed,
# scarce amount — the moment what agents know changes what the world IS. It rides the
# existing food economy (it just adds food tiles the normal perception/eat loop then
# uses), never a scripted "farmers win".
#
# Gated THREE ways so it is a consequence, not a cheat: (1) the agent must KNOW
# 'farming'; (2) it must be fed enough to work the land (a starving farmer forages, it
# does not farm) — so a population still has to survive the early scramble before
# farming can stabilise it; and (3) farmers STABILISE rather than hoard — once the
# world already holds FARM_FOOD_PER_CAPITA food per living agent, they rest, so the
# supply plateaus at a sustainable abundance instead of growing without bound. The net
# effect: a knowing population maintains a steady, reachable food supply (each tile
# grown right next to the farmer, where scattered respawn never reaches) and stops
# starving — survival improves as a CONSEQUENCE of the economy, never a scripted win.
FARM_YIELD = 0.5                # per fed-farmer-per-turn chance of growing one food tile
FARM_HUNGER_CUTOFF = SURVIVAL_HUNGER  # a farmer hungrier than this is busy surviving
FARM_FOOD_PER_CAPITA = 2.0      # farmers rest once the world holds this much food/agent

# --- Adoption model --------------------------------------------------------
# Base per-contact-per-turn chance a non-knower adopts an item from an adjacent
# knower, BEFORE trust/personality shaping. Tuned so a single seed knower spreads
# as a believable S-curve through a crowd over tens of turns rather than saturating
# in one or two (see verify_m11).
ADOPTION_BASE = 0.25

# Personality of the LEARNER shapes how readily it takes on a new idea: the curious
# soak it up, the cautious and the independent resist. Multiplies the base chance.
PERSONALITY_ADOPT: dict[str, float] = {
    "curiosity": 1.6,
    "friendliness": 1.2,
    "caution": 0.6,
    "independence": 0.5,
}

# Trust shaping: each point of the learner's trust IN THE TEACHER nudges the chance
# up (and a grudge/low trust nudges it down). A linear bump, then clamped — so a
# trusted friend's idea catches far more readily than a distrusted stranger's, and a
# hated rival's barely at all, without ever hitting 0 or 1.
TRUST_COEFF = 0.12
ADOPTION_MIN = 0.02
ADOPTION_MAX = 0.95

# M4.1 lineage: childhood is a LEARNING WINDOW. A dependent child adopts knowledge
# through the SAME diffusion channel at this multiple of the adult rate — children
# inherit no knowledge at birth, they EARN it, just faster while young (and their
# kin-trust in their parents raises it further through the ordinary trust term).
# Only applied when the run's lineage system is on, so a default run's adoption
# probabilities are byte-identical to before.
CHILD_LEARN_BOOST = 2.0


def adoption_probability(learner: Any, teacher: Any, state: dict[str, Any]) -> float:
    """Chance `learner` adopts a known item from adjacent `teacher` this turn.

    Shaped by (a) the learner's personality — curious adopts readily, cautious/
    independent resist — and (b) how much the learner TRUSTS the teacher (the same
    relationships[...]['trust'] Phase 0 maintains). Pure read; no LLM, no mutation.
    A brand-new pair (no relationship yet) sits at neutral trust 0, so strangers
    still share at the base rate — just less than trusted friends do.
    """
    pers = get_personality(learner)
    mult = PERSONALITY_ADOPT.get(pers.dominant, 1.0)
    trust = learner.relationships.get(teacher.name, {}).get("trust", 0)
    p = ADOPTION_BASE * mult * (1.0 + TRUST_COEFF * trust)
    # M4.1: a dependent child soaks up knowledge at a boosted rate (the childhood
    # learning window). False for every agent when lineage is off -> byte-identical.
    if world.is_dependent_child(learner, state):
        p *= CHILD_LEARN_BOOST
    return max(ADOPTION_MIN, min(ADOPTION_MAX, p))


def has_any_knowledge(state: dict[str, Any]) -> bool:
    """True if ANY living agent knows anything — the gate that keeps a no-knowledge
    run byte-identical to v1 (and draws ZERO rng): when this is False, `diffuse`
    returns immediately without touching the contact graph or the RNG stream."""
    return any(a.knowledge for a in state["agents"] if a.alive)


def diffuse(state: dict[str, Any], turn: int,
            rng: "random.Random | None" = None) -> list[tuple[str, str, str]]:
    """Spread knowledge one hop along the contact network for this turn.

    For every adjacent (knower, non-knower) pair, the non-knower may ADOPT a held
    item with `adoption_probability`. Transmissions are decided against a snapshot of
    who-knows-what at turn start and applied afterwards, so an item moves at most one
    hop/turn and the outcome is independent of agent iteration order. Each adoption is
    logged to events[] ("turn 12: A052 taught 'fire' to A101") and recorded as a
    memory on both sides. Returns the list of (teacher, item, learner) for callers.

    Cost: O(agents x neighbours) with O(1) neighbour lookup (the M0.3 occupancy
    index) and ZERO LLM calls. Returns [] and draws no RNG when nobody knows anything
    — so a v1 run with no seeded knowledge is unaffected, byte for byte.
    """
    if not has_any_knowledge(state):
        return []
    draw = (rng or random).random
    # M2.3: when the economy is on, a teacher's GUARDED items (see `guards`) are withheld from
    # free diffusion — they move only by sale (economy.trade). With the economy off nobody
    # guards, so this is a no-op and M1.1 diffusion is byte-identical to before.
    economy_on = state.get("economy_on", False)

    living = [a for a in state["agents"] if a.alive]
    by_name = {a.name: a for a in living}
    # Snapshot at turn start: decisions never see this turn's own adoptions.
    snapshot = {a.name: frozenset(a.knowledge) for a in living}
    # learner_name -> {item: teacher_name} (the FIRST teacher to land it wins/logs).
    pending: dict[str, dict[str, str]] = {}

    for teacher in living:  # world_state["agents"] order is stable
        t_known = snapshot[teacher.name]
        if not t_known:
            continue
        neighbours = world.adjacent_agents(teacher, state)
        for lname in sorted(neighbours):  # sorted -> hash-seed-independent rng order
            learner = neighbours[lname]
            l_known = snapshot[lname]
            already = pending.setdefault(lname, {})
            for item in sorted(t_known):
                if item in l_known or item in already:
                    continue
                if economy_on and guards(teacher, item):
                    continue  # M2.3: a guarded skill is sold, not taught free
                if draw() < adoption_probability(learner, teacher, state):
                    already[item] = teacher.name

    transmissions: list[tuple[str, str, str]] = []
    for lname in sorted(pending):
        learner = by_name[lname]
        for item in sorted(pending[lname]):
            teacher_name = pending[lname][item]
            learner.knowledge.add(item)
            world.record_memory(learner, f"Learned '{item}' from {teacher_name}")
            world.record_memory(by_name[teacher_name], f"Taught '{item}' to {lname}")
            state["events"].append(f"turn {turn}: {teacher_name} taught '{item}' to {lname}")
            transmissions.append((teacher_name, item, lname))
    return transmissions


def discovery_probability(agent: Any, item: str, state: dict[str, Any]) -> float:
    """Chance `agent` INVENTS `item` this turn, assuming its prereqs are already met.

    Shaped by (a) personality — the curious tinker most, the cautious least — and
    (b) situation: a fed agent has spare capacity to experiment, a starving one does
    not, so the chance scales linearly down to zero at DISCOVERY_HUNGER_CUTOFF. Pure
    read; no LLM, no mutation. NOT a function of the turn number — there is no timer.
    """
    pers = get_personality(agent)
    mult = DISCOVERY_PERSONALITY.get(pers.dominant, 1.0)
    situation = max(0.0, 1.0 - agent.hunger / DISCOVERY_HUNGER_CUTOFF)
    return DISCOVERY_BASE * mult * situation


def discover(state: dict[str, Any], turn: int,
             tree: "dict[str, frozenset[str]] | None" = None,
             rng: "random.Random | None" = None) -> list[tuple[str, str]]:
    """Let agents INVENT items they don't know whose prerequisites they DO know (M1.2).

    For every living agent and every undiscovered item in `tree` whose prereqs the
    agent already knows, roll `discovery_probability`. A success adds the item to that
    agent's knowledge (from where M1.1 `diffuse` spreads it — no new spread code here)
    and logs "turn 34: A052 discovered 'fire'". Prereq checks read a turn-start
    SNAPSHOT, so an agent can't chain fire->tools->farming in a single turn; each item
    is at most one fresh invention per agent per turn.

    Discovery is probabilistic and situational, NEVER a timer: the turn it first fires
    depends on who is fed, curious, and lucky, so it varies by run. Returns the list of
    (agent, item) discovered. A no-op drawing ZERO rng when `tree` is empty/None — so a
    run with no tech tree is byte-identical to v1.

    Cost: O(agents x tree size), ZERO LLM calls.
    """
    if not tree:
        return []
    draw = (rng or random).random

    living = [a for a in state["agents"] if a.alive]
    # Snapshot knowledge so this turn's inventions don't unlock downstream items until
    # the next turn (no within-turn fire->tools->farming cascade) and order can't matter.
    snapshot = {a.name: frozenset(a.knowledge) for a in living}
    discoveries: list[tuple[str, str]] = []

    for agent in living:  # world_state["agents"] order is stable
        known = snapshot[agent.name]
        for item in sorted(tree):  # sorted -> hash-seed-independent rng order
            if item in known or not tree[item] <= known:
                continue  # already known, or prereqs not met -> cannot invent (gated)
            p = discovery_probability(agent, item, state)
            if p > 0.0 and draw() < p:  # starving agents (p == 0) don't even roll
                agent.knowledge.add(item)
                world.record_memory(agent, f"Discovered '{item}'")
                state["events"].append(f"turn {turn}: {agent.name} discovered '{item}'")
                discoveries.append((agent.name, item))
    return discoveries


def _empty_adjacent_cell(agent: Any, state: dict[str, Any]) -> "tuple[int, int] | None":
    """First adjacent N/S/E/W cell that is empty ground (no wall/agent/food), or None.

    Where a farmer can put a new food tile: in bounds, unoccupied, not already food.
    Fixed offset order -> deterministic placement -> reproducible.
    """
    x, y = agent.position
    size = state["size"]
    food = state["food"]
    occ = state.get("occupancy", {})
    for dx, dy in world._ADJ_OFFSETS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in occ and (nx, ny) not in food:
            return (nx, ny)
    return None


def farm(state: dict[str, Any], turn: int,
         rng: "random.Random | None" = None) -> list[tuple[str, tuple[int, int]]]:
    """Let agents who KNOW 'farming' PRODUCE food into world_state (M1.3 headline).

    Each fed farmer (knows 'farming' AND hunger < FARM_HUNGER_CUTOFF) has a FARM_YIELD
    chance of growing one food tile on an empty neighbouring cell — adding to the food
    supply the existing perception/eat loop already uses, so survival improves as a
    measured CONSEQUENCE of the food economy shifting, not a scripted win. Returns the
    (farmer, cell) productions. A no-op drawing ZERO rng when nobody is a fed farmer —
    so a run with no farming knowledge (incl. every v1 run) is byte-identical.

    Cost: O(agents) to find farmers; ZERO LLM calls.
    """
    # M4.1: a dependent child does not produce, even if it has already learned the
    # skill — production waits for maturity (always False when lineage is off).
    farmers = [a for a in state["agents"]
               if a.alive and "farming" in a.knowledge and a.hunger < FARM_HUNGER_CUTOFF
               and not world.is_dependent_child(a, state)]
    if not farmers:
        return []  # v1 / no fed farmers -> no-op, zero rng
    # Stabiliser, not hoarder: once the world already holds enough food per living
    # agent, farmers rest. Bounds the supply at a sustainable abundance (no runaway).
    living = sum(1 for a in state["agents"] if a.alive)
    if len(state["food"]) >= FARM_FOOD_PER_CAPITA * living:
        return []
    draw = (rng or random).random
    produced: list[tuple[str, tuple[int, int]]] = []
    for agent in farmers:  # world_state["agents"] order is stable
        if draw() < FARM_YIELD:
            cell = _empty_adjacent_cell(agent, state)
            if cell is not None:
                world.place_food(cell[0], cell[1], state)
                world.record_memory(agent, "Tended crops")
                produced.append((agent.name, cell))
    return produced


# --- Hunting production (M2.3 specialization) ------------------------------
# Hunting is the SECOND producer skill — a sibling to farming that also grows the food
# supply, but by a DISTINCT mechanic so the two specializations are genuinely different (and
# so knowledge of each is worth trading to someone who only has the other). Where a farmer
# tends a plot on the cell right beside it (radius 1), a hunter takes roaming GAME from a
# wider ring around it (HUNT_RADIUS_MIN..HUNT_RADIUS_MAX) — food appears further out, not
# underfoot. Same three-way gating as farming so it is a consequence, never a cheat: must
# KNOW hunting, must be fed enough to hunt (a starving hunter forages), and hunters STABILISE
# (rest once the world already holds enough food per agent) rather than flooding the map.
HUNT_YIELD = 0.5                  # per fed-hunter-per-turn chance of taking one food tile
HUNT_HUNGER_CUTOFF = SURVIVAL_HUNGER   # a hunter hungrier than this is busy surviving
HUNT_FOOD_PER_CAPITA = 2.0        # hunters rest once the world holds this much food/agent
HUNT_RADIUS_MIN = 2               # game appears no closer than this (beyond the farm plot)
HUNT_RADIUS_MAX = 3               # ...and no further than this (the hunter's range)


def _empty_cell_in_ring(agent: Any, state: dict[str, Any],
                        rmin: int, rmax: int) -> "tuple[int, int] | None":
    """First empty ground cell at Chebyshev distance rmin..rmax from `agent`, or None (M2.3).

    Where a hunter drops game: in bounds, unoccupied, not already food, and OUT in the field
    (distance >= rmin) rather than adjacent like a farm plot. Deterministic scan order (rings
    out, then row-major within a ring) -> reproducible placement, no RNG of its own.
    """
    x, y = agent.position
    size = state["size"]
    food = state["food"]
    occ = state.get("occupancy", {})
    for r in range(rmin, rmax + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue  # only the ring at exactly distance r
                nx, ny = x + dx, y + dy
                if (0 <= nx < size and 0 <= ny < size
                        and (nx, ny) not in occ and (nx, ny) not in food):
                    return (nx, ny)
    return None


def hunt(state: dict[str, Any], turn: int,
         rng: "random.Random | None" = None) -> list[tuple[str, tuple[int, int]]]:
    """Let agents who KNOW 'hunting' PRODUCE food into world_state (M2.3, farming's sibling).

    Each fed hunter (knows 'hunting' AND hunger < HUNT_HUNGER_CUTOFF) has a HUNT_YIELD chance
    of taking one food tile from the ring of roaming game around it — a second, independent
    supply channel alongside farming. Returns the (hunter, cell) takes. A no-op drawing ZERO
    rng when nobody is a fed hunter — so a run with no hunting knowledge (incl. every v1 run)
    is byte-identical, exactly like `farm`. Cost: O(agents); ZERO LLM calls.
    """
    # M4.1: dependents don't hunt either — see the matching exclusion in farm().
    hunters = [a for a in state["agents"]
               if a.alive and "hunting" in a.knowledge and a.hunger < HUNT_HUNGER_CUTOFF
               and not world.is_dependent_child(a, state)]
    if not hunters:
        return []  # v1 / no fed hunters -> no-op, zero rng
    living = sum(1 for a in state["agents"] if a.alive)
    if len(state["food"]) >= HUNT_FOOD_PER_CAPITA * living:
        return []  # stabiliser: enough food already, hunters rest (no runaway supply)
    draw = (rng or random).random
    taken: list[tuple[str, tuple[int, int]]] = []
    for agent in hunters:  # world_state["agents"] order is stable
        if draw() < HUNT_YIELD:
            cell = _empty_cell_in_ring(agent, state, HUNT_RADIUS_MIN, HUNT_RADIUS_MAX)
            if cell is not None:
                world.place_food(cell[0], cell[1], state)
                world.record_memory(agent, "Took game while hunting")
                taken.append((agent.name, cell))
    return taken


def grant(state: dict[str, Any], agent: Any, item: str, turn: int) -> None:
    """Seed `item` into one agent's knowledge and log it (used at setup / by god mode).

    The single place an item ENTERS the world from outside the contact graph. From
    here it can only spread by `diffuse` through actual meetings.
    """
    agent.knowledge.add(item)
    world.record_memory(agent, f"Knows '{item}'")
    state["events"].append(f"turn {turn}: {agent.name} now knows '{item}' (seeded)")
