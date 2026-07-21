"""
scenario.py
===========

DEMO / SCENARIO STAGING — set up a starting scene so the (already-verified) M3.4–M3.6
conquest-chain institutions can be WATCHED on the map. Optional, default OFF.

Why this exists (and what it is NOT)
------------------------------------
A documented finding of this project is that ORGANIC runs almost never produce a monarch,
kingdom, or empire — an egalitarian world never concentrates enough wealth for one agent to
field an army that conquers a settlement (let alone a rival realm). So the verified M3.4
(monarchy), M3.5 (kingdoms) and M3.6 (empire) mechanics, though tested, are almost never SEEN.

This module STAGES a scene — it does NOT fake behaviour. It only sets up positions, wealth and
trust, then calls the EXISTING, VERIFIED code paths to produce the institution records:

    * monarchy.attempt_conquest   — a wealthy aspirant musters a real army and seizes a town,
                                     populating world_state["monarchs"][sid] EXACTLY as an
                                     organic conquest would (monarchy.resolve_battle decides it).
    * kingdoms.conquer_neighbour  — the new monarch marches its realm host on a neighbouring
                                     trust-led town, vassalising its lord into world_state
                                     ["kingdoms"][king] (the same fight, the same submission rule).
    * empire.update / empire.wage_war — left to fire in the NORMAL per-turn loop: two rival
                                     kingdoms are positioned adjacent with one stronger, so the
                                     existing opportunistic-war logic clashes their whole loyal
                                     hosts and SUBJUGATES the loser into an empire on screen.

NO new battle/conquest maths is written here. The records produced are indistinguishable from
organic ones because they ARE produced by the organic code. After staging, the normal simulation
runs from the staged state — the per-turn rules are untouched.

Determinism & cost
------------------
Staging is RNG-FREE (fixed positions/wealth; place_agent, attempt_conquest, conquer_neighbour are
all zero-RNG), so a staged run is fully reproducible under a seed — the only RNG is the normal
loop. This module is invoked ONCE at world setup (run_simulation), gated on an explicit `--stage`
flag; when not staged it is never imported, so default runs are byte-identical to before. It lives
OUTSIDE the per-turn decision logic and outside god_mode/the renderer (their boundaries are
untouched); it is a setup helper that writes world_state through the same world/institution layers
the engine itself uses.
"""

from __future__ import annotations

from typing import Any

import kingdoms
import leadership
import monarchy
import world
from agents import Agent

# Personality strings chosen so the renderer's personality-colour + role glyphs read clearly:
# a ruler is red (independence), a vassal lord pink (friendliness), commoners blue (caution).
_RULER = "ambitious, independent and competitive"
_LORD = "friendly and outgoing"
_FOLK = "cautious and territorial"

# VIABILITY (the whole point of the staging fix): a demo realm must FEED ITSELF for the whole
# run, not collapse into a ghost town. Every staged inhabitant is seeded as a self-sufficient
# PRODUCER — it KNOWS how to farm and hunt (and cooks with fire, for a bigger meal), so the
# per-turn knowledge.farm/knowledge.hunt loops keep regrowing the food the town eats. This is
# NOT a mechanics cheat: farming/hunting are ordinary seeded knowledge (the same items
# --seed-knowledge grants), and food is only ever grown by the VERIFIED knowledge loops from a
# FED producer — a starving one forages instead. Seeding the skill just guarantees the demo
# starts with the producers an organic world would have had to evolve.
_PRODUCER = frozenset({"fire", "tools", "farming", "hunting"})


def _place(state: dict[str, Any], name: str, pos: tuple[int, int], personality: str,
           cognition: str, money: float = 0.0, stockpile: float = 0.0,
           sid: str | None = None, knowledge: "frozenset[str] | set[str]" = _PRODUCER) -> Agent:
    """Create + place a living agent through the SAME world layer the engine uses (no RNG).

    `knowledge` defaults to the self-feeding _PRODUCER skill set so a staged inhabitant sustains
    itself (and the town) across the whole run; pass an empty set for a genuinely skill-less body.
    """
    a = Agent(name=name, personality=personality, goals={"survive": 8, "wealth": 4},
              cognition=cognition, knowledge=set(knowledge))
    size = state.get("size", 0) or 1
    x = max(0, min(size - 1, pos[0]))
    y = max(0, min(size - 1, pos[1]))
    world.place_agent(a, x, y)
    a.money = money
    a.stockpile = stockpile
    a.settlement = sid
    a.hunger = 0
    return a


def _settlement(state: dict[str, Any], sid: str, center: tuple[int, int],
                member_names: list[str]) -> None:
    """Register a settlement record (the same shape settlement.update writes)."""
    state.setdefault("settlements", {})[sid] = {
        "id": sid, "center": tuple(center), "members": set(member_names), "founded": 0}
    digits = "".join(c for c in sid if c.isdigit())
    state["settlement_seq"] = max(state.get("settlement_seq", 0), int(digits or 0))


def _food_around(state: dict[str, Any], center: tuple[int, int], radius: int) -> None:
    """Fill a solid food FIELD around a settlement so its inhabitants can always reach a meal.

    A filled square (not a thin diamond) gives the town a dense, walkable larder to bootstrap on;
    from there the per-turn knowledge.farm/hunt loops keep it topped up (they cap the world's food
    at a sustainable per-capita abundance, so this seed does not runaway). This is what lets a
    staged capital stay POPULATED — residents eat here rather than wandering off to starve.
    """
    cx, cy = center
    size = state.get("size", 0)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < size and 0 <= y < size and (x, y) not in state["food"]:
                state["food"].append((x, y))
                state["grid"][y][x] = world.FOOD


def _mercs(state: dict[str, Any], prefix: str, near: tuple[int, int], n: int, cognition: str) -> None:
    """Scatter `n` POOR agents within muster range of `near` — the labour pool an army hires from.

    Spread across distinct cells (within MUSTER_RADIUS), wealth below MERC_MAX_WEALTH so the
    EXISTING muster (monarchy.muster) will hire them. These are real agents who fight and die.

    VIABILITY: a mercenary is a self-feeding _PRODUCER (default) and carries a small food buffer,
    so the SURVIVING garrison it forms after the conquest can feed itself standing watch instead of
    starving by turn 30. The buffer keeps wealth < monarchy.MERC_MAX_WEALTH so muster still hires it.
    """
    for i in range(n):
        x = near[0] + (i % 4) - 2
        y = near[1] + (i // 4) - 1
        _place(state, f"{prefix}{i}", (x, y), _FOLK, cognition, money=0.5, stockpile=3.5)


# --- Level 1: a real MONARCHY (a CASTLE appears) ---------------------------
def stage_monarchy(state: dict[str, Any], cognition: str, center: tuple[int, int]) -> Agent:
    """A wealthy aspirant SEIZES a town by force (real attempt_conquest) -> a monarch + castle.

    Builds a township with a militia, a rich aspirant, and a pool of mercenaries, then runs the
    VERIFIED monarchy.attempt_conquest: the aspirant musters an army that out-numbers the militia
    and becomes MONARCH of the town — world_state["monarchs"]["S001"] is populated exactly as an
    organic conquest would. Returns the new monarch agent.
    """
    cx, cy = center
    sid = "S001"
    members = []
    for i, (dx, dy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)]):
        members.append(_place(state, f"Town{i}", (cx + dx, cy + dy), _FOLK, cognition,
                              money=8.0, stockpile=14.0, sid=sid).name)
    _settlement(state, sid, (cx, cy), members)
    _food_around(state, (cx, cy), 4)
    # Aspirant + mercs sit INSIDE the town's food field (a few cells north) so the garrison the
    # conquest leaves behind stands watch ON reliable food and feeds itself, not on a barren ridge.
    aspirant = _place(state, "Rex", (cx, cy - 2), _RULER, cognition, money=200.0, stockpile=60.0)
    _mercs(state, "RexM", (cx, cy - 3), 9, cognition)
    monarchy.attempt_conquest(state, aspirant, sid, 0)      # THE REAL CONQUEST -> monarchs[S001]=Rex
    return aspirant


# --- Level 2: a real feudal KINGDOM (king -> vassal lords -> settlements) ---
def stage_kingdom(state: dict[str, Any], cognition: str, center: tuple[int, int]) -> Agent:
    """The monarch CONQUERS neighbouring trust-led towns (real conquer_neighbour) -> a kingdom.

    On top of stage_monarchy: two adjacent towns each have a trust-leader and followers. The king
    marches its realm host on each (the VERIFIED kingdoms.conquer_neighbour); an out-matched town
    SUBMITS and its lord becomes a VASSAL. Result: world_state["kingdoms"]["Rex"] with three
    settlements and two vassal lords — a real multi-settlement feudal hierarchy. Returns the king.
    """
    king = stage_monarchy(state, cognition, center)
    cx, cy = center
    # Each vassal town is a VIABLE settlement in its own right — a lord plus several producer
    # followers on a food field of its own — so the whole feudal realm stays alive and populated
    # (not a capital ringed by two collapsing hamlets). Followers trust the chief (FORM_TRUST) so
    # the conquered town keeps its local leadership as an organic vassalage.
    for sid, (tx, ty), chief_name, fol in (
        ("S002", (cx + 7, cy - 1), "Chief2", ["F2a", "F2b", "F2c", "F2d"]),
        ("S003", (cx - 1, cy + 7), "Chief3", ["F3a", "F3b", "F3c", "F3d"]),
    ):
        chief = _place(state, chief_name, (tx, ty), _LORD, cognition, money=10.0, stockpile=30.0, sid=sid)
        members = [chief.name]
        for j, fn in enumerate(fol):
            f = _place(state, fn, (tx + (j % 2), ty + 1 + j // 2), _FOLK, cognition,
                       money=8.0, stockpile=20.0, sid=sid)
            f.relationships[chief_name] = {"trust": leadership.FORM_TRUST, "interactions": 1, "grudge": False}
            members.append(fn)
        _settlement(state, sid, (tx, ty), members)
        state.setdefault("leaders", {})[sid] = {"leader": chief_name, "followers": set(fol), "since": 0}
        _food_around(state, (tx, ty), 3)
        king.money = 30.0                                   # a war chest sized to muster ~6 fighters
        _mercs(state, sid + "M", (king.position[0], king.position[1] - 3), 6, cognition)
        kingdoms.conquer_neighbour(state, king.name, sid, 0)  # THE REAL CONQUEST -> vassalage
    king.money, king.stockpile = 60.0, 90.0
    return king


# --- Level 3: TWO rival kingdoms -> the loop fires a WAR, an EMPIRE forms ---
def _stage_realm(state: dict[str, Any], cognition: str, prefix: str, king_name: str,
                 king_pos: tuple[int, int], home_sid: str, home_center: tuple[int, int],
                 vassal_sid: str, vassal_center: tuple[int, int], vassal_chief: str) -> tuple[Agent, Agent]:
    """Build ONE kingdom (king monarch of a seized home + one vassalised neighbour). Returns (king, lord).

    NB: the king sits well CLEAR of every town (so monarchy.update's per-turn conquest loop finds
    it ineligible and never drains its war chest before the empire war), and the commoners are kept
    POOR (below MIN_WAR_CHEST) so they are never aspirants either — leaving the kings + vassal lords
    as the only armed parties, which is what makes the staged inter-kingdom war clean.
    """
    members = []
    for i, (dx, dy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)]):
        # Commoners are self-feeding _PRODUCERS but kept POOR (wealth < MIN_WAR_CHEST) so they are
        # never war-chest aspirants — the kings + vassal lords stay the only armed parties, which is
        # what keeps the staged inter-kingdom war clean. Farming (not a cash buffer) is what feeds them.
        members.append(_place(state, f"{prefix}T{i}", (home_center[0] + dx, home_center[1] + dy),
                              _FOLK, cognition, money=4.0, stockpile=5.0, sid=home_sid).name)
    _settlement(state, home_sid, home_center, members)
    _food_around(state, home_center, 3)
    king = _place(state, king_name, king_pos, _RULER, cognition, money=120.0, stockpile=90.0)
    _food_around(state, king_pos, 2)                        # the king's seat has its own larder
    _mercs(state, f"{prefix}KM", (king_pos[0], king_pos[1] - 2), 9, cognition)
    monarchy.attempt_conquest(state, king, home_sid, 0)     # REAL: king seizes its capital

    chief = _place(state, vassal_chief, vassal_center, _LORD, cognition, money=10.0, stockpile=30.0, sid=vassal_sid)
    vmem = [chief.name]
    for j in range(4):
        f = _place(state, f"{prefix}V{j}", (vassal_center[0] + (j % 2), vassal_center[1] + 1 + j // 2),
                   _FOLK, cognition, money=3.0, stockpile=5.0, sid=vassal_sid)
        f.relationships[vassal_chief] = {"trust": leadership.FORM_TRUST, "interactions": 1, "grudge": False}
        vmem.append(f.name)
    _settlement(state, vassal_sid, vassal_center, vmem)
    state.setdefault("leaders", {})[vassal_sid] = {"leader": vassal_chief, "followers": set(vmem[1:]), "since": 0}
    _food_around(state, vassal_center, 3)
    king.money = 30.0
    _mercs(state, f"{prefix}VM", (king_pos[0], king_pos[1] - 2), 6, cognition)
    kingdoms.conquer_neighbour(state, king_name, vassal_sid, 0)  # REAL: vassalise the neighbour
    return king, chief


def _arm_for_war(state: dict[str, Any], cognition: str, prefix: str, king: Agent, lord: Agent,
                 n_king: int, n_lord: int, chest: float) -> None:
    """Refill the king + vassal lord's war chests and place FRESH mercenaries near each seat.

    The staging conquests spent the kings' gold and consumed their merc pools; to let the EXISTING
    opportunistic-war loop muster real hosts, restore the war chests and lay down a fresh pool of
    the poor near each funder (the king AND its loyal vassal both contribute to the realm host).
    Sizing one realm larger than the other is what makes the stronger one attack — governance and
    force are still decided by the verified empire.imperial_host / wage_war, not by this setup.
    """
    king.money, king.stockpile = chest, max(king.stockpile, 90.0)
    lord.money, lord.stockpile = chest, max(lord.stockpile, 60.0)
    _mercs(state, f"{prefix}WK", (king.position[0], king.position[1] - 3), n_king, cognition)
    _mercs(state, f"{prefix}WV", (lord.position[0], lord.position[1] + 2), n_lord, cognition)


def stage_war(state: dict[str, Any], cognition: str) -> None:
    """Two adjacent rival kingdoms — A stronger than B — so the loop's empire war fires on screen.

    Builds kingdom A (left) and kingdom B (right) via the real monarchy+kingdom paths, positioned
    so A's frontier town is within KINGDOM_REACH of B's capital. A is armed with a larger loyal
    host than B. With --stage war the caller turns the empire system ON, so empire.update — the
    VERIFIED opportunistic-war logic — clashes A's whole loyal host against B's, A wins, and B's
    king is SUBJUGATED into A's EMPIRE (world_state["empires"]["Aldric"]) during the normal run.
    A god/manual `empire.wage_war` could force it too, but here the normal loop fires it.
    """
    # Positions assume a >= 30 grid (the caller sizes a staged war world accordingly). Kings sit to
    # the NORTH, well clear (> ATTACK_RADIUS) of every town; the realms' frontier towns (S0A2/S0B1)
    # lie within KINGDOM_REACH (8) of each other so empire.update sees them as neighbours.
    a_king, a_lord = _stage_realm(state, cognition, "A", "Aldric", (8, 6),
                                  "S0A1", (8, 18), "S0A2", (14, 18), "LordA")
    b_king, b_lord = _stage_realm(state, cognition, "B", "Borin", (22, 6),
                                  "S0B1", (22, 18), "S0B2", (28, 18), "LordB")
    # A fields a clearly larger LOYAL host than B, so the stronger realm opens the war.
    _arm_for_war(state, cognition, "A", a_king, a_lord, n_king=8, n_lord=8, chest=300.0)
    _arm_for_war(state, cognition, "B", b_king, b_lord, n_king=4, n_lord=3, chest=120.0)


# --- Level 4: THREE realms in a standoff -> a WATCHABLE cascade of wars -----
# V4.14 SHOWCASE STAGING. `--stage war` gives exactly ONE war, and it fires on turn 1 — before a
# viewer has even read the title card — after which nothing happens for the rest of the run. The
# showcase scene fixes the PACING at the source, still without faking anything:
#
#   * THREE realms, not two, arranged so the middle realm (A) borders BOTH rivals and they do not
#     border each other — A is the natural empire-builder and the natural target.
#   * They open in a STANDOFF: every realm can field about the same host, and empire.update only
#     launches a war it can WIN (strict >), so NO war fires on turn 1. Tribute then fattens the
#     war chests at different rates (A's realm is the richest) and the mercenary pools drift, so
#     within a few turns one crown pulls ahead and marches — the opening war lands around turn
#     3-6, i.e. AFTER the title card, with the camera already settled.
#   * The winner pays for it: casualties and spent coin shrink its host, which is what flips the
#     THIRD realm's own winnable-war test — the rival strikes the weakened empire a few turns
#     later. The cascade is emergent (empire.update decides every step); the staging only sets up
#     the standoff that makes it happen on camera.
#
# Every record is still produced by the verified monarchy/kingdoms/empire paths, and the staging
# itself stays RNG-free — the timing variance comes from the normal loop, as it should.
# A host is `min(wealth // FIGHTER_COST, mercenaries in muster range)` per funder, so with a deep
# chest the MERCENARY POOL is what a realm's strength actually equals. Every realm therefore opens
# with the SAME pool — an exact standoff — and the tie is broken by the sim itself: the poor wander
# in and out of muster range, tribute shifts the chests, and a war fires the turn one crown can
# genuinely out-field a neighbour. Nothing here decides who wins, or when.
_SHOW_CHEST_KING = 45.0        # a royal chest deep enough that the merc pool is the binding limit
_SHOW_LARDER_KING = 60.0       # the king's own food store (he sits away from his towns)
_SHOW_CHEST_LORD = 0.0         # a vassal lord opens BELOW monarchy.MIN_WAR_CHEST (money + stockpile
_SHOW_LARDER_LORD = 8.0        # < 10): he can neither muster nor stage a turn-1 coup, so all three
                               # hosts are exactly the king's own. Tribute later lifts him past the
                               # floor — which is where the feudal betrayals come from.
_SHOW_MERCS_KING = 9           # the poor within muster range of each king's seat (= the realm's host)
_SHOW_MERCS_LORD = 6           # ...and of each vassal lord, for the hosts he can raise later


def stage_showcase(state: dict[str, Any], cognition: str) -> None:
    """Three rival realms in a border STANDOFF, on a 36-cell world — the showcase scene.

    A (west) borders B (north-east) and C (south-east); B and C are out of KINGDOM_REACH of each
    other, so A is both the natural empire-builder and the natural target. Each realm is built by
    the REAL monarchy + kingdoms paths (a king who seized his capital and vassalised a neighbour)
    and armed to EQUAL strength. Every capital sits more than monarchy.ATTACK_RADIUS from its own
    vassal town and every king's seat well clear of both, so no realm can eat itself on turn 1.
    """
    # The scene is deliberately COMPACT (a 26-cell world): B and C must be more than
    # kingdoms.KINGDOM_REACH apart while both border A, which sets the shape, and everything else
    # is pulled in as tight as the radii allow so the towns read LARGE on screen rather than as
    # specks in an empty map. Each capital is 6 cells from its own vassal town (over
    # monarchy.ATTACK_RADIUS, under KINGDOM_REACH) and every king's seat is clear of all six.
    realms = (
        # prefix, king,     seat,     capital, capital centre, vassal sid, vassal centre, lord
        ("A", "Aldric", (2, 19), "S0A1", (12, 12), "S0A2", (18, 12), "LordA"),
        ("B", "Borin", (23, 3), "S0B1", (8, 5), "S0B2", (14, 3), "LordB"),
        ("C", "Cyrus", (23, 22), "S0C1", (16, 19), "S0C2", (10, 21), "LordC"),
    )
    for prefix, king_name, seat, home_sid, home_c, vassal_sid, vassal_c, lord_name in realms:
        king, lord = _stage_realm(state, cognition, prefix, king_name, seat,
                                  home_sid, home_c, vassal_sid, vassal_c, lord_name)
        king.money, king.stockpile = _SHOW_CHEST_KING, _SHOW_LARDER_KING
        lord.money, lord.stockpile = _SHOW_CHEST_LORD, _SHOW_LARDER_LORD
        # Fresh mercenary pools: the staging conquests above spent the ones they were built with.
        # Placed BELOW each seat (the northern kings' seats sit near the top edge).
        _mercs(state, f"{prefix}WK", (king.position[0], king.position[1] + 2), _SHOW_MERCS_KING, cognition)
        _mercs(state, f"{prefix}WV", (lord.position[0], lord.position[1] + 2), _SHOW_MERCS_LORD, cognition)


# --- Dispatch --------------------------------------------------------------
STAGES = ("monarchy", "kingdom", "war", "showcase")


def apply(state: dict[str, Any], kind: str, *, cognition: str = "heuristic") -> None:
    """Stage scenario `kind` into `state` using the verified institution code paths (RNG-free).

    Ensures the institution dicts exist, then builds the requested scene. Called once at world
    setup; the normal per-turn loop runs from here. `war` and `empire` are synonyms.
    """
    for k in ("monarchs", "leaders", "kingdoms", "empires", "settlements"):
        state.setdefault(k, {})
    cx = cy = state.get("size", 30) // 2
    if kind == "monarchy":
        stage_monarchy(state, cognition, (cx, cy))
    elif kind == "kingdom":
        stage_kingdom(state, cognition, (cx, cy))
    elif kind in ("war", "empire"):
        stage_war(state, cognition)
    elif kind == "showcase":
        stage_showcase(state, cognition)
    else:
        raise ValueError(f"unknown scenario stage: {kind!r} (expected one of {STAGES})")
