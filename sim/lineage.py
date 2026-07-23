"""
lineage.py
==========

LINEAGE — V2 milestone M4.1: birth, childhood, aging, and family. Opens Phase 4
(Generations & Dynasties) on top of Phases 0-3.

The idea (time becomes generational)
------------------------------------
Through Phase 3 agents only die of starvation or battle, and the population is
maintained by RESPAWN — blank slates appear when headcount drops. M4.1 replaces
that with GENERATIONAL life: agents PAIR, bear CHILDREN who inherit temperament,
raise them at real cost, AGE, and die of OLD AGE — so the cast turns over and
time itself starts to matter. Everything later in Phase 4 (inheritance,
dynasties) stands on this.

Like every system since Phase 0, nothing here is installed — it EMERGES from
asymmetries the sim already has:

  * PAIRING rides the existing TRUST system (mutual trust at the same "high"
    bar alliances/loyalty/leadership already use — trust.HIGH_THRESHOLD).
  * Births require the existing SETTLEMENT system (M2.1) and its local FOOD
    surplus — a nomadic, hungry, or fractured world bears no children.
  * Child-rearing draws down the parents' M2.2 STOCKPILES (or their own meals),
    so family size is gated by wealth — Malthus, not a scripted growth curve.
  * Children LEARN through the existing M1.1 diffusion (boosted while young);
    knowledge is EARNED, never copied at birth.

SCOPE BOUNDARIES (stated, deliberate — do not blur):
  * INHERITANCE OF WEALTH at death IS built here (M4.2 — see settle_estate and the
    "Inheritance at death" section below). A newborn still starts with nothing
    (stockpile 0, money 0 — no wealth is inherited AT BIRTH); what M4.2 adds is
    that a dead agent's wealth no longer vanishes — it PASSES TO KIN.
  * DYNASTIC SUCCESSION of titles IS built here (M4.3 — see succeed_titles and the
    "Dynastic succession of titles" section below). Unlike M4.2's PARTIBLE wealth, a
    title is IMPARTIBLE: on a holder's death the SEAT (monarch/king/vassal/emperor/
    subject-king) passes by PRIMOGENITURE to the SINGLE eldest heir, while ALL children
    still split the gold. Both fire on the same death. The heir inherits the SEAT, not
    the LOYALTY — trust is personal and is never copied, so a weak heir can lose what its
    father built through the existing breakaway machinery. Trust-LEADERSHIP (M3.2, by
    consent) is NOT hereditary. A kinless holder's line is EXTINGUISHED (records clear as
    today). Escheat (kinless WEALTH) still pays the estate to the settlement's ruler.

The design, mechanically
------------------------
1. PAIRING & BIRTH — two agents may produce a child only when ALL hold: both
   LIVING and SETTLED in the SAME settlement; MUTUAL trust >= PAIR_TRUST (the
   existing alliance/loyalty bar); both currently FED (hunger below the same
   SURVIVAL_HUNGER bar discovery/farming use); the settlement holds a FOOD
   SURPLUS (standing local food at/above one tile per living member — several
   turns of headroom over the ~1/7 food-per-turn each member actually consumes);
   neither is a dependent child; each parent is past its BIRTH_COOLDOWN; and the
   living population is below the cap. Every gate binds — knock out any one and
   no child is born.
2. INHERITANCE AT BIRTH (temperament only) — the child's Personality trait
   weights are the AVERAGE of its parents' plus a small jitter drawn from the
   SIM's seeded RNG (sim mechanics, so the sim stream is correct here — unlike
   the renderer, which must never touch it); the dominant trait is recomputed
   from the blend. Goals blend the same way (drives are temperament). Knowledge
   is NOT inherited (children learn via M1.1 diffusion, boosted — see
   knowledge.CHILD_LEARN_BOOST); wealth is NOT inherited (M4.2); trust/memories
   start blank apart from the kin-trust seed.
3. CHILDHOOD — a child is a DEPENDENT for CHILDHOOD_TURNS: it takes no actions
   (no foraging/farming/labor/trade/war — see the exclusions in main/knowledge/
   storage/labor/economy/monarchy), it is FED from its parents' stockpiles (or a
   fed parent's own ration, at real hunger cost), and it learns at a boosted
   adoption rate. If no parent can feed it, it starves like anyone — harsh but
   honest. On reaching maturity it becomes a full agent.
4. AGING & NATURAL DEATH — every agent carries an age and a natural LIFESPAN
   (deterministic from the seeded stream at creation; founders get varied adult
   ages at world setup). At lifespan's end the agent dies of OLD AGE through the
   EXISTING death path (population.announce_death, distinct wording), so events,
   survivor memories and the post-mortem all work like any death.
5. POPULATION — births are the engine; the Day 14 respawn becomes extinction
   insurance ONLY. No respawn code changes at all: process_respawns already
   fires only while living < TARGET_POPULATION (3) — that IS the hard floor —
   and silently drops respawns above it. Births are refused at the cap
   (pop_cap, sized from the founding cast) and food-gated, so growth is
   Malthusian: abundance -> growth toward the cap, scarcity -> stagnation.
6. INHERITANCE AT DEATH (M4.2 — settle_estate) — death stops ERASING wealth and
   starts PASSING it on, so history accumulates in families. On ANY death (old
   age, starvation, battle — every cause funnels through the SAME single hook,
   population.announce_death), the deceased's money AND stockpile form the ESTATE
   and are split PARTIBLY and EQUALLY down a fixed kin-order:
     a. surviving CHILDREN (dependents included — an heir's inherited stockpile
        helps feed it); else
     b. surviving PARENTS; else
     c. surviving SIBLINGS (agents sharing a parent);
     d. no kin at all -> ESCHEAT to the settlement's current RULER (monarch first,
        else trust-leader) if one lives — the crown profits from a kinless death;
        no settlement/ruler -> the estate vanishes exactly as it did before M4.2.
   Money has no cap, so it all flows to heirs; inherited FOOD respects the M2.2
   storage cap and any overflow DROPS at the deceased's tile as ground food rather
   than vanishing. Wealth is strictly CONSERVED: estate == sum distributed to
   heirs + ground-drop overflow (no minting, no leakage). Every transfer logs an
   event ("X inherited N from Y", escheat logged distinctly) and writes a memory
   to each heir. This moves only MOVABLE WEALTH; the deceased's TITLE passes
   separately by M4.3 succession (item 7).
7. DYNASTIC SUCCESSION OF TITLES (M4.3 — succeed_titles) — death stops evaporating
   CROWNS and starts passing them on, so realms outlive their founders. On ANY death
   (same single hook, population.announce_death, run just before settle_estate), the
   deceased's FORCE titles (monarch seats, kingdom crowns, vassal lordships, imperial
   thrones, subject-king seats) pass IMPARTIBLY to the SINGLE eldest surviving heir down
   the SAME kin-order M4.2 uses (children -> parents -> siblings; eldest-first, name as
   tiebreak). The realm STRUCTURE survives intact under the heir (records re-keyed to the
   heir's name, tribute/vassal/discontent bookkeeping carried across), a coronation is
   logged, and — crucially — NO trust is copied: the heir holds the realm only on its OWN
   standing, so an unknown/distrusted heir's vassals erode and BREAK AWAY through the
   EXISTING M3.5/M3.6 loyalty machinery (succession is a CRISIS TEST, no new mechanic). A
   DEPENDENT child heir takes the seat as a REGENT (its levy/muster/war powers stay dormant
   via the existing is_dependent_child gate). A holder with NO living kin has its LINE
   EXTINGUISHED — the records clear as today (the existing breakaway/re-conquest machinery
   dissolves the leaderless realm) and the extinction is logged distinctly. Trust-
   LEADERSHIP (M3.2, consent) is NOT hereditary and is never touched.

Cost & determinism
------------------
ZERO LLM calls — pure Python over world_state. All randomness (trait jitter,
lifespans) comes from the seeded sim stream, drawn in stable sorted order, so a
seeded run reproduces exactly. Inheritance (M4.2) and title succession (M4.3) draw
NO RNG at all — an equal split down a sorted kin-order, a deterministic ground-drop
placement, and an eldest-first single-heir choice over sorted records — so both are
reproducible under seed. Everything is gated on world_state["lineage_on"]
(the --lineage flag): with it OFF (default) no function here is ever called, no
RNG is drawn, and the run — including respawn — is byte-identical to today.
"""

from __future__ import annotations

import math
import random
from typing import Any

from sim import population
from sim import storage
from sim import trust
from sim import world
from sim.agents import Agent
from sim.personality import TRAIT_NAMES, Personality
from llm.strategy import SURVIVAL_HUNGER, get_personality

# --- Pairing & birth gates (documented) -------------------------------------
# PAIR_TRUST: the MUTUAL trust two settled agents must hold in each other to pair.
# Tied to trust.HIGH_THRESHOLD — the SAME "high" bar an independent agent needs to
# ally (strategy._will_ally), a follower needs to follow (leadership.FORM_TRUST)
# and a vassal needs to stay loyal (kingdoms.LOYAL_TRUST) — so family rides the
# existing trust economy, never a new courtship system.
PAIR_TRUST = trust.HIGH_THRESHOLD  # 2

# FED_HUNGER: a parent must be below this hunger to bear/feed a child — the same
# SURVIVAL_HUNGER bar that gates discovery ("a starving agent doesn't tinker") and
# farming ("a starving farmer forages"). A starving pair bears no children.
FED_HUNGER = SURVIVAL_HUNGER  # 5

# BIRTH_COOLDOWN: minimum turns between a parent's consecutive children — births
# are PACED, so even a rich, trusting pair raises a family over generations of
# turns rather than in a burst.
BIRTH_COOLDOWN = 10

# SURPLUS_RADIUS / SURPLUS_FOOD_PER_MEMBER: the settlement food-surplus gate.
# The settlement holds a surplus when the standing food within SURPLUS_RADIUS
# (Chebyshev) of its centre is at least SURPLUS_FOOD_PER_MEMBER per living member.
# Each member consumes ~HUNGER_PER_TURN/EAT_RELIEF ≈ 0.14 food/turn, so one whole
# standing tile per member is several turns of RELIABLE headroom above current
# consumption — food beyond need, which is what makes a child affordable. Radius is
# the settlement footprint plus one (members range over CLUSTER_RADIUS = 2).
SURPLUS_RADIUS = 3
SURPLUS_FOOD_PER_MEMBER = 1.0

# POP_CAP_FACTOR: births are refused once the living population reaches the cap.
# The Day 14 TARGET_POPULATION (3) is a trio TOP-UP target, not a world bound, so
# the lineage cap rides it as a floor and gives births headroom above the founding
# cast: pop_cap = max(TARGET_POPULATION + 1, ceil(POP_CAP_FACTOR * founders)).
# Growth is therefore bounded (no runaway) but real (a fed trio can double).
POP_CAP_FACTOR = 2.0

# --- Inheritance (temperament only — wealth is M4.2, titles M4.3) -----------
# TRAIT_JITTER: the +/- bound on the per-trait deterministic jitter added to the
# parents' averaged Personality weights, drawn from the seeded sim RNG — children
# RESEMBLE their parents but never clone them.
TRAIT_JITTER = 0.1

# KIN_TRUST: the starting trust seeded BOTH WAYS between child and each parent —
# above PAIR_TRUST, so a family is born already inside the high-trust band the
# social systems (alliances, leadership, diffusion shaping) all read. Rides the
# existing relationships records; no new bond type.
KIN_TRUST = 3

# --- Childhood (a real investment, kept minimal) -----------------------------
# CHILDHOOD_TURNS: how long a child stays a DEPENDENT (no production, fed by its
# parents, boosted learning) before it comes of age as a full agent.
CHILDHOOD_TURNS = 16

# Feeding: a dependent whose hunger has reached CHILD_FEED_AT is fed one meal by a
# parent — CHILD_MEAL_COST units off the parent's stockpile (the wealth gate on
# family size: ~0.25 units/turn/child against a banking rate of ~0.2-0.5), or, if
# no parent has the savings, a FED parent (hunger <= FED_HUNGER) SHARES its own
# ration and takes PARENT_SHARE_HUNGER hunger onto itself — a real cost either
# way. Relief mirrors a normal meal (world.EAT_RELIEF). No parent able to pay
# either price -> the child goes unfed and starves like anyone.
CHILD_FEED_AT = 4
CHILD_MEAL_COST = 1.0
CHILD_MEAL_RELIEF = world.EAT_RELIEF  # 7
PARENT_SHARE_HUNGER = 2

# --- Aging & natural death ---------------------------------------------------
# LIFESPAN_MIN/MAX: every agent's natural lifespan in turns, drawn once from the
# seeded stream at creation — varied, so generations STAGGER instead of dying in
# lockstep. INIT_AGE_MIN/MAX: the founding cast enters mid-life (already adult,
# ages varied) so founders age out across a long run rather than all at once.
LIFESPAN_MIN = 80
LIFESPAN_MAX = 120
INIT_AGE_MIN = CHILDHOOD_TURNS  # founders are at least adults
INIT_AGE_MAX = 40

# Names the birth sequence cycles through (uniqued via population._unique_name so
# a name is never reused while any holder — living or remembered — exists).
# Deliberately distinct from AGENT_SPECS and NEWCOMER_SPECS.
NAME_POOL: tuple[str, ...] = (
    "Iris", "Juno", "Kade", "Lena", "Milo", "Nell", "Orin", "Pia",
    "Quin", "Rhea", "Sol", "Tara", "Umi", "Vera", "Wren", "Yale",
)

# The human-readable adjective for each dominant trait, used in the child's
# personality STRING. The string is presentation; the child's ACTUAL traits are
# the blended Personality stamped into the get_personality cache (see
# blend_personality) — but the adjective keeps a fallback re-parse (cache lost,
# e.g. after deserialization) at least dominant-correct.
_TRAIT_ADJECTIVE = {
    "curiosity": "curious",
    "caution": "cautious",
    "friendliness": "friendly",
    "independence": "independent",
}


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    """King-move distance — the same radius metric the settlement layer uses."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


# --- Setup -------------------------------------------------------------------
def init_cast(state: dict[str, Any], rng: "random.Random | None" = None) -> None:
    """Seed ages/lifespans onto the founding cast and size the population cap (M4.1).

    Called once at world setup when lineage is ON (run_simulation), AFTER any
    staging so it covers the whole starting cast. Each founder gets a varied adult
    age and a natural lifespan, drawn from the seeded sim stream in stable agent
    order — deterministic under seed. Also writes the lineage block onto
    world_state: {"pop_cap", "birth_seq"} (documented single source of truth for
    the birth machinery). Never called with lineage off -> zero RNG drawn -> the
    default run is byte-identical.
    """
    r = rng or random
    living = [a for a in state["agents"] if a.alive]
    for a in living:  # stable world_state order -> reproducible draws
        a.age = r.randint(INIT_AGE_MIN, INIT_AGE_MAX)
        a.lifespan = r.randint(LIFESPAN_MIN, LIFESPAN_MAX)
    state["lineage"] = {
        # Births are refused at this cap (the existing top-up target is the floor
        # respawn already enforces; this gives births bounded headroom above it).
        "pop_cap": max(population.TARGET_POPULATION + 1,
                       math.ceil(POP_CAP_FACTOR * len(living))),
        # Monotonic birth counter — cycles NAME_POOL, counts total births.
        "birth_seq": 0,
    }


# --- Inheritance at birth (temperament only) ---------------------------------
def blend_personality(pa: Any, pb: Any, rng: "random.Random") -> Personality:
    """The child's Personality: parents' trait weights averaged + bounded jitter.

    Each of the four trait weights is the parents' mean plus a jitter drawn
    uniformly from [-TRAIT_JITTER, +TRAIT_JITTER] (the seeded sim stream —
    deterministic under seed), clamped to [0, 1]. Traits are visited in the fixed
    TRAIT_NAMES declaration order so the RNG draw order is stable. The dominant
    trait then falls out of Personality.dominant recomputed on the BLEND — a child
    of two curious parents skews curious, but never clones either. Pure function
    of the parents + rng; exposed for tests/verification.
    """
    p1, p2 = get_personality(pa), get_personality(pb)
    weights = {}
    for name in TRAIT_NAMES:  # fixed order -> stable rng draw sequence
        mean = (getattr(p1, name) + getattr(p2, name)) / 2.0
        weights[name] = min(1.0, max(0.0, mean + rng.uniform(-TRAIT_JITTER, TRAIT_JITTER)))
    return Personality(**weights)


def _blend_goals(pa: Any, pb: Any) -> dict[str, int]:
    """The child's goal weights: the rounded average of its parents' (temperament).

    Deterministic (no jitter — the personality jitter is variation enough) and
    covers the union of both parents' drives, so a child of a wealth-driven and a
    friendship-driven parent carries a genuine mix.
    """
    ga, gb = pa.goals or {}, pb.goals or {}
    return {k: round((ga.get(k, 0) + gb.get(k, 0)) / 2)
            for k in sorted(set(ga) | set(gb))}


# --- Birth gates --------------------------------------------------------------
def settlement_surplus(state: dict[str, Any], sid: str) -> bool:
    """Whether settlement `sid` holds a FOOD SURPLUS — reliable food above current
    consumption (the Malthusian valve on births). True when the standing food
    within SURPLUS_RADIUS of the centre is >= SURPLUS_FOOD_PER_MEMBER per living
    member (see the constants for why that is genuine headroom). Pure read.
    """
    rec = state.get("settlements", {}).get(sid)
    if rec is None:
        return False
    members = sum(1 for a in state["agents"]
                  if a.alive and a.settlement == sid)
    if members == 0:
        return False
    center = rec["center"]
    local_food = sum(1 for f in state["food"] if _chebyshev(f, center) <= SURPLUS_RADIUS)
    return local_food >= SURPLUS_FOOD_PER_MEMBER * members


def _eligible_parent(a: Any, turn: int) -> bool:
    """The per-agent birth gates: living, settled, adult, FED, past the cooldown.

    (The pair gates — same settlement, mutual trust, settlement surplus, cap —
    are checked by the pairing loop; this is only what disqualifies one agent.)
    """
    if not a.alive or a.settlement is None:
        return False
    if getattr(a, "dependent", False):
        return False  # a dependent child cannot be a parent
    if a.hunger >= FED_HUNGER:
        return False  # not currently fed -> no child
    return (turn - a.last_child_turn) >= BIRTH_COOLDOWN


def _mutual_trust(pa: Any, pb: Any) -> bool:
    """Both directions of the pair's trust at/above the alliance bar (pure read)."""
    return (pa.relationships.get(pb.name, {}).get("trust", 0) >= PAIR_TRUST
            and pb.relationships.get(pa.name, {}).get("trust", 0) >= PAIR_TRUST)


def _empty_cell_near(pos: tuple[int, int], state: dict[str, Any],
                     radius: int = 2) -> "tuple[int, int] | None":
    """Nearest empty ground cell (no agent, no food) within `radius` of `pos`.

    Where a newborn is placed: beside its parent. Deterministic order (distance,
    then coordinates) so placement is reproducible without an RNG draw.
    """
    x, y = pos
    size = state["size"]
    occ = state.get("occupancy", {})
    food = set(state["food"])
    candidates = [
        (nx, ny)
        for nx in range(max(0, x - radius), min(size, x + radius + 1))
        for ny in range(max(0, y - radius), min(size, y + radius + 1))
        if (nx, ny) != pos and (nx, ny) not in occ and (nx, ny) not in food
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda c: (_chebyshev(c, pos), c))
    return candidates[0]


# --- Birth --------------------------------------------------------------------
def _spawn_child(state: dict[str, Any], turn: int, pa: Any, pb: Any,
                 rng: "random.Random") -> "Any | None":
    """Create ONE child of `pa` and `pb`: the single place a birth enters the world.

    The child is a genuine newborn: temperament BLENDED from the parents (see
    blend_personality), goals blended, knowledge EMPTY (it will learn via M1.1
    diffusion), wealth ZERO (inheritance at death is M4.2 — deliberately not
    built), trust blank apart from the KIN_TRUST seed both ways with each parent,
    a fresh lifespan from the seeded stream, DEPENDENT for CHILDHOOD_TURNS, and a
    member of its parents' settlement (the family home). Placed on an empty cell
    beside a parent; returns None (no birth) if the neighbourhood is full.
    """
    cell = _empty_cell_near(pa.position, state) or _empty_cell_near(pb.position, state)
    if cell is None:
        return None

    lin = state["lineage"]
    lin["birth_seq"] += 1
    base = NAME_POOL[(lin["birth_seq"] - 1) % len(NAME_POOL)]
    name = population._unique_name(base, state)

    pers = blend_personality(pa, pb, rng)
    child = Agent(
        name=name,
        personality=f"{_TRAIT_ADJECTIVE[pers.dominant]} (child of {pa.name} and {pb.name})",
        goals=_blend_goals(pa, pb),
        cognition=getattr(pa, "cognition", "llm"),  # raised in the parents' world
        parents=(pa.name, pb.name),
        dependent=True,
        lifespan=rng.randint(LIFESPAN_MIN, LIFESPAN_MAX),
    )
    # Stamp the BLENDED traits into the personality cache (the string above is
    # presentation; get_personality returns this blend everywhere it is read).
    child._personality_cache = (child.personality, pers)

    world.place_agent(child, *cell)
    sid = pa.settlement
    child.settlement = sid
    rec = state["settlements"].get(sid)
    if rec is not None:
        rec["members"].add(name)

    # Kin-trust both ways — family bonds ride the existing trust records.
    for parent in (pa, pb):
        trust.ensure_relationship(child, parent.name)["trust"] = KIN_TRUST
        rel = trust.ensure_relationship(parent, name)
        rel["trust"] = max(rel["trust"], KIN_TRUST)
        parent.last_child_turn = turn

    world.record_memory(child, f"Born to {pa.name} and {pb.name} in {sid}")
    world.record_memory(pa, f"{name} was born — my child with {pb.name}")
    world.record_memory(pb, f"{name} was born — my child with {pa.name}")
    state["events"].append(
        f"turn {turn}: {name} was born to {pa.name} and {pb.name} in {sid}")
    return child


def _births(state: dict[str, Any], turn: int, rng: "random.Random") -> list[Any]:
    """One deterministic pairing pass: settled, trusting, fed pairs bear children.

    Settlements are visited in sorted id order; within one, eligible members in
    sorted name order, each pairing with the first later-named eligible member it
    MUTUALLY trusts at the alliance bar. Every gate binds here: no settlement or
    no surplus -> the settlement is skipped; an unfed/cooling-down/dependent agent
    never enters the pool; the population cap stops the whole pass. An agent
    parents at most one child per turn.
    """
    lin = state.get("lineage", {})
    cap = lin.get("pop_cap", 0)
    used: set[str] = set()
    born: list[Any] = []
    for sid in sorted(state.get("settlements", {})):
        if not settlement_surplus(state, sid):
            continue  # the Malthusian gate: no local surplus -> no births here
        elig = sorted((a for a in state["agents"]
                       if a.settlement == sid and _eligible_parent(a, turn)),
                      key=lambda a: a.name)
        for i, pa in enumerate(elig):
            if pa.name in used:
                continue
            for pb in elig[i + 1:]:
                if pb.name in used or not _mutual_trust(pa, pb):
                    continue
                if population.living_count(state) >= cap:
                    return born  # population bound: births refused at the cap
                child = _spawn_child(state, turn, pa, pb, rng)
                if child is not None:
                    used.update((pa.name, pb.name))
                    born.append(child)
                break  # pa pairs at most once this turn (found its match or not)
    return born


# --- Childhood upkeep -----------------------------------------------------------
def _feed_children(state: dict[str, Any], turn: int) -> None:
    """Parents feed their dependent children — the REAL cost of childhood.

    A hungry dependent (hunger >= CHILD_FEED_AT) is fed by the first parent that
    can pay: stockpile first (richest parent's granary, CHILD_MEAL_COST drawn
    down — the visible wealth cost that gates family size), else a FED parent
    shares its own ration (takes PARENT_SHARE_HUNGER onto itself). Both parents
    dead, broke and hungry -> the child goes unfed and starves like anyone
    (harsh but honest). Deterministic; no RNG.
    """
    by_name = {a.name: a for a in state["agents"]}
    for child in state["agents"]:  # stable order
        if not child.alive or not child.dependent or child.hunger < CHILD_FEED_AT:
            continue
        parents = [by_name[n] for n in child.parents
                   if n in by_name and by_name[n].alive]
        feeder = None
        # Stockpile first: richest granary pays (ties by name -> deterministic).
        for p in sorted(parents, key=lambda p: (-p.stockpile, p.name)):
            if p.stockpile >= CHILD_MEAL_COST:
                p.stockpile -= CHILD_MEAL_COST
                feeder = p
                break
        if feeder is None:
            # No savings: a fed parent shares its own ration at real hunger cost.
            for p in sorted(parents, key=lambda p: (p.hunger, p.name)):
                if p.hunger <= FED_HUNGER:
                    p.hunger = min(world.HUNGER_MAX, p.hunger + PARENT_SHARE_HUNGER)
                    feeder = p
                    break
        if feeder is None:
            continue  # nobody can feed it — hunger keeps climbing
        child.hunger = max(0, child.hunger - CHILD_MEAL_RELIEF)
        world.record_memory(child, f"Was fed by {feeder.name}")
        world.record_memory(feeder, f"Fed {child.name}")


# --- Inheritance at death (M4.2) ---------------------------------------------
# The estate is the deceased's MOVABLE WEALTH: money + stockpile. Money is a
# food-claim with no store cap (economy.mint) so it flows to heirs in full;
# inherited FOOD is bounded by the M2.2 granary cap (storage.STORAGE_CAP) and any
# overflow drops as ground food rather than vanishing. Titles are NOT part of the
# estate — a crown/vassal seat passes by conquest/trust, and dynastic succession
# of the seat is M4.3; a dead ruler's title record clears exactly as it does today.
def _living_heirs(deceased: Any, state: dict[str, Any]) -> "tuple[list[Any], str]":
    """The heirs of `deceased`, in the fixed partible kin-order (pure read).

    Returns (heirs, kind) where kind is "children" | "parents" | "siblings" |
    "escheat" | "none". Only LIVING agents other than the deceased are ever heirs.
      * children — living agents that name the deceased as a parent;
      * else parents — living agents whose name is in deceased.parents;
      * else siblings — living agents sharing at least one parent with the deceased;
      * else escheat — the settlement's ruler (see _settlement_ruler), as a
        single heir; kind "escheat" so the caller logs it distinctly;
      * else none — no kin and no ruler; the estate vanishes as it did pre-M4.2.
    Each heir list is sorted by name so the equal split is deterministic.
    """
    living = [a for a in state["agents"] if a.alive and a is not deceased]
    children = sorted((a for a in living if deceased.name in (a.parents or ())),
                      key=lambda a: a.name)
    if children:
        return children, "children"

    by_name = {a.name: a for a in living}
    parents = sorted((by_name[n] for n in (deceased.parents or ()) if n in by_name),
                     key=lambda a: a.name)
    if parents:
        return parents, "parents"

    dparents = set(deceased.parents or ())
    if dparents:
        siblings = sorted((a for a in living if dparents & set(a.parents or ())),
                          key=lambda a: a.name)
        if siblings:
            return siblings, "siblings"

    ruler = _settlement_ruler(deceased, state)
    if ruler is not None:
        return [ruler], "escheat"
    return [], "none"


def _settlement_ruler(deceased: Any, state: dict[str, Any]) -> "Any | None":
    """The LIVING ruler of the deceased's settlement — monarch first, else trust-leader.

    Mirrors monarchy._holder_name (crown outranks a trust-leader) but resolves to the
    living Agent, and never returns the deceased itself (a ruler's own kinless estate
    cannot escheat to a corpse). None if the deceased was a nomad, the seat is vacant,
    or the titled agent is not currently alive. This is the ONLY title record M4.2
    reads, and it reads it purely to route WEALTH — the seat itself is untouched (M4.3).
    """
    sid = deceased.settlement
    if sid is None:
        return None
    mon = state.get("monarchs", {}).get(sid)
    holder = mon["monarch"] if mon is not None else \
        (state.get("leaders", {}).get(sid) or {}).get("leader")
    if holder is None or holder == deceased.name:
        return None
    by_name = {a.name: a for a in state["agents"] if a.alive}
    return by_name.get(holder)


def _drop_ground_food(state: dict[str, Any], pos: tuple[int, int], units: float) -> int:
    """Drop `int(units)` whole ground-food tiles near `pos` (deterministic). Returns
    the number placed.

    Inherited food beyond a heir's granary cap does not vanish — it falls at the
    deceased's tile as standing food (one tile per WHOLE food-unit of overflow),
    onto the nearest empty cells (no agent, no existing food) in the same distance-
    then-coordinate order the newborn placement uses. No RNG. The estate LEDGER is
    conserved to the exact float (see settle_estate); the map is a whole-tile
    rendering of that overflow, so a sub-unit remainder is not painted as a tile.
    """
    whole = int(units + 1e-9)
    if whole <= 0:
        return 0
    x, y = pos
    size = state["size"]
    occ = state.get("occupancy", {})
    food = set(state["food"])
    radius = 1
    dropped = 0
    while dropped < whole and radius <= size:
        cells = sorted(
            ((nx, ny)
             for nx in range(max(0, x - radius), min(size, x + radius + 1))
             for ny in range(max(0, y - radius), min(size, y + radius + 1))
             if (nx, ny) not in occ and (nx, ny) not in food),
            key=lambda c: (_chebyshev(c, pos), c))
        for cell in cells:
            if dropped >= whole:
                break
            if world.place_food(cell[0], cell[1], state):
                food.add(cell)
                dropped += 1
        radius += 1  # widen the ring if the neighbourhood filled up
    return dropped


def settle_estate(deceased: Any, turn: int, state: dict[str, Any]) -> dict[str, Any]:
    """Distribute the deceased's estate to kin (M4.2). The single inheritance hook.

    Called from population.announce_death (the one funnel for EVERY death — old age,
    starvation, battle) only when lineage is on and there is wealth to move, AFTER the
    cell is freed so the ground-drop lands on an unobstructed tile. Zeroes the estate
    off the deceased (wealth leaves the corpse — no double counting), splits it EQUALLY
    down _living_heirs' kin-order, and logs + memorises every transfer. Returns an
    accounting record {estate, kind, per_heir, to_heirs, ground} for tests/verification.

    Conservation (to the exact float): estate == to_heirs + ground. Money (no cap) is
    always fully distributed; inherited food fills each heir's granary to STORAGE_CAP
    and the remainder becomes `ground`. ZERO RNG, ZERO LLM.
    """
    estate_money = float(deceased.money)
    estate_food = float(deceased.stockpile)
    estate = estate_money + estate_food
    record: dict[str, Any] = {
        "estate": estate, "kind": "none", "per_heir": 0.0, "to_heirs": 0.0, "ground": 0.0}
    if estate <= 0.0:
        return record  # nothing to inherit — no event, no memory (silent as before)

    heirs, kind = _living_heirs(deceased, state)
    record["kind"] = kind
    # Wealth leaves the deceased regardless of whether an heir exists.
    deceased.money = 0.0
    deceased.stockpile = 0.0
    if not heirs:
        # Kinless AND no ruler: the estate vanishes exactly as it did pre-M4.2.
        state["events"].append(
            f"turn {turn}: {deceased.name}'s estate of {estate:.2f} vanished (no heir)")
        return record

    n = len(heirs)
    money_each = estate_money / n
    food_each = estate_food / n
    ground = 0.0
    to_heirs = 0.0
    for heir in heirs:  # sorted -> deterministic
        heir.money += money_each
        room = max(0.0, storage.STORAGE_CAP - heir.stockpile)
        into_store = min(food_each, room)
        heir.stockpile += into_store
        ground += food_each - into_store  # overflow past the granary cap
        received = money_each + into_store
        to_heirs += received
        if kind == "escheat":
            world.record_memory(
                heir, f"The estate of {deceased.name} ({estate:.2f}) escheated to me")
            state["events"].append(
                f"turn {turn}: {deceased.name}'s estate of {estate:.2f} "
                f"escheated to {heir.name} (no kin)")
        else:
            world.record_memory(
                heir, f"Inherited {received:.2f} from {deceased.name}")
            state["events"].append(
                f"turn {turn}: {heir.name} inherited {received:.2f} from {deceased.name}")

    dropped = _drop_ground_food(state, deceased.position, ground)
    if ground > 0.0:
        state["events"].append(
            f"turn {turn}: {ground:.2f} of {deceased.name}'s estate dropped as "
            f"ground food ({dropped} tiles, over the granary cap)")

    record.update(per_heir=money_each + food_each, to_heirs=to_heirs, ground=ground)
    return record


# --- Dynastic succession of titles (M4.3) ------------------------------------
# WEALTH is PARTIBLE (M4.2 — split equally among all children); a TITLE is
# IMPARTIBLE — a crown cannot be divided. On a title-holder's death the SEAT passes
# by PRIMOGENITURE-style SINGLE succession to the ELDEST surviving kin (the same M4.2
# kin tiers — children -> parents -> siblings — but eldest-first, not an equal split).
# Both fire on the SAME death: the eldest gets the crown; ALL children still split the
# gold (settle_estate is untouched).
#
# SCOPE (stated, deliberate): only FORCE-based titles are dynastic — MONARCH seats
# (M3.4), KINGDOMS and their VASSAL LORDSHIPS (M3.5), and EMPIRES / SUBJECT-KING seats
# (M3.6). Trust-LEADERSHIP (M3.2) is CONSENT-based and is NOT hereditary: a dead
# trust-leader's `leaders[sid]` record is left to clear exactly as today (leadership
# re-emerges from trust next turn), and succeed_titles never touches it.
#
# THE HEIR INHERITS THE SEAT, NOT THE LOYALTY. Title RECORDS transfer to the heir's
# name (the realm structure — kingdom, vassals, tribute flow — survives intact under
# the heir), but TRUST is PERSONAL and is NEVER copied: a vassal's/subject-king's
# loyalty toward the heir is only what it PERSONALLY holds (plus any M4.1 kin-trust if
# family). So a trusted heir holds the realm while an unknown/distrusted one inherits a
# realm whose vassals erode and BREAK AWAY through the EXISTING M3.5/M3.6 machinery —
# succession is a CRISIS TEST, with NO new loyalty mechanic added here.
#
# EXTINCT LINES: a holder who dies with NO living kin leaves the title records to clear
# exactly as they do today (a dead king's realm dissolves via the existing breakaway
# logic; a vacant monarch seat is re-contested by the ordinary conquest machinery) —
# succeed_titles only logs the line's extinction distinctly and moves nothing.
def _eldest(cands: list[Any]) -> "Any | None":
    """The single heir among `cands`: ELDEST first, NAME as the deterministic tiebreak."""
    return sorted(cands, key=lambda a: (-a.age, a.name))[0] if cands else None


def _succession_heir(deceased: Any, state: dict[str, Any]) -> "tuple[Any | None, str]":
    """The SINGLE title-heir of `deceased`, by primogeniture down the M4.2 kin-order.

    Returns (heir, kind) with kind "child" | "parent" | "sibling" | "none". The tiers
    are M4.2's (children -> parents -> siblings), but a title is impartible so exactly
    ONE heir is chosen — the ELDEST of the closest non-empty tier (age desc, then name
    as a deterministic tiebreak). A DEPENDENT child is a valid heir (a child monarch —
    see succeed_titles). Only LIVING agents other than the deceased are ever heirs; pure
    read, ZERO RNG. Mirrors M4.2's _living_heirs so wealth and titles follow the SAME
    bloodline, differing only in partible-all vs impartible-eldest.
    """
    living = [a for a in state["agents"] if a.alive and a is not deceased]
    child = _eldest([a for a in living if deceased.name in (a.parents or ())])
    if child is not None:
        return child, "child"
    by_name = {a.name: a for a in living}
    parent = _eldest([by_name[n] for n in (deceased.parents or ()) if n in by_name])
    if parent is not None:
        return parent, "parent"
    dparents = set(deceased.parents or ())
    if dparents:
        sibling = _eldest([a for a in living if dparents & set(a.parents or ())])
        if sibling is not None:
            return sibling, "sibling"
    return None, "none"


def _holds_force_title(name: str, state: dict[str, Any]) -> bool:
    """True iff `name` holds ANY dynastic FORCE title (monarch/king/vassal/emperor/
    subject-king). A pure trust-leader (M3.2, consent) holds none — leadership is not
    hereditary. Pure read."""
    if any(m["monarch"] == name for m in state.get("monarchs", {}).values()):
        return True
    if name in state.get("kingdoms", {}):
        return True
    if any(name in k["vassals"].values() for k in state.get("kingdoms", {}).values()):
        return True
    if name in state.get("empires", {}):
        return True
    if any(name in e["subject_kings"] for e in state.get("empires", {}).values()):
        return True
    return False


def _title_summary(name: str, state: dict[str, Any]) -> str:
    """A deterministic human string of every FORCE title `name` currently holds (for the
    coronation / extinction log). Sorted throughout; reads only, mutates nothing."""
    parts: list[str] = []
    for sid in sorted(s for s, m in state.get("monarchs", {}).items() if m["monarch"] == name):
        parts.append(f"monarch of {sid}")
    if name in state.get("kingdoms", {}):
        sids = ", ".join(sorted(state["kingdoms"][name]["settlements"])) or "no lands"
        parts.append(f"king of the realm of {sids}")
    for sid in sorted(sid for k in state.get("kingdoms", {}).values()
                      for sid, lord in k["vassals"].items() if lord == name):
        parts.append(f"vassal lord of {sid}")
    if name in state.get("empires", {}):
        parts.append("emperor")
    if any(name in e["subject_kings"] for e in state.get("empires", {}).values()):
        parts.append("subject-king")
    return "; ".join(parts) if parts else "no title"


def _transfer_titles(state: dict[str, Any], old: str, new: str) -> None:
    """Rewrite EVERY force-title record from name `old` to `new` (the seat passes intact).

    Re-keys the name-keyed realm/empire dicts and rewrites the sid-keyed monarch/vassal
    holder fields, carrying the discontent (breakaway-hysteresis) counters across so the
    realm STRUCTURE survives whole. It copies NO trust — the heir stands on its OWN
    loyalty. Garrison/follower rosters (real fighters) are left untouched: the heir is a
    figurehead, not a soldier, and the dead are filtered out of them elsewhere. Leadership
    (leaders[sid], M3.2 consent) is never touched. Deterministic; ZERO RNG.
    """
    kingdoms = state.get("kingdoms", {})
    empires = state.get("empires", {})

    # 1. MONARCH seats (sid-keyed) — the heir holds the seat the deceased held.
    for m in state.get("monarchs", {}).values():
        if m["monarch"] == old:
            m["monarch"] = new

    # 2. KINGDOM (name-keyed) — re-key the realm and rename its king field.
    if old in kingdoms:
        rec = kingdoms.pop(old)
        rec["king"] = new
        if new in kingdoms:  # rare personal union — the heir already wore a crown
            keep = kingdoms[new]
            keep["settlements"] |= rec["settlements"]
            keep["vassals"].update(rec["vassals"])
            keep["discontent"].update(rec["discontent"])
        else:
            kingdoms[new] = rec

    # 3. VASSAL LORDSHIPS + their discontent counters (across every realm).
    for krec in kingdoms.values():
        for sid, lord in list(krec["vassals"].items()):
            if lord == old:
                krec["vassals"][sid] = new
        if old in krec["discontent"]:
            krec["discontent"][new] = krec["discontent"].pop(old)

    # 4. EMPIRE (name-keyed) — re-key the empire and rename its emperor field.
    if old in empires:
        erec = empires.pop(old)
        erec["emperor"] = new
        if new in empires:  # rare personal union of empires
            keep = empires[new]
            keep["subject_kings"].update(erec["subject_kings"])
            keep["discontent"].update(erec["discontent"])
        else:
            empires[new] = erec

    # 5. SUBJECT-KING seats + their discontent counters (across every empire).
    for erec in empires.values():
        if old in erec["subject_kings"]:
            erec["subject_kings"][new] = erec["subject_kings"].pop(old)
        if old in erec["discontent"]:
            erec["discontent"][new] = erec["discontent"].pop(old)

    # 6. Cleanup — a crown cannot be its own vassal/subject (if the heir already held a
    #    subordinate seat inside the realm it now leads, that seat folds into the crown).
    for king, krec in kingdoms.items():
        for sid in [s for s, lord in krec["vassals"].items() if lord == king]:
            krec["vassals"].pop(sid)
            krec["discontent"].pop(king, None)
    for emperor, erec in empires.items():
        if emperor in erec["subject_kings"]:
            erec["subject_kings"].pop(emperor)
            erec["discontent"].pop(emperor, None)


def succeed_titles(deceased: Any, turn: int, state: dict[str, Any]) -> dict[str, Any]:
    """Pass the deceased's FORCE titles to a single heir by primogeniture (M4.3).

    The second hook (alongside settle_estate) inside population.announce_death — the ONE
    funnel EVERY death cause runs through, so old-age, starvation and battle deaths all
    succeed identically. Returns an accounting dict {heir, kind, titles} for tests/verify.

      * The deceased holds NO force title (a commoner, or a consent-only trust-leader) ->
        nothing happens (leadership is not hereditary).
      * A living heir exists -> ALL title records transfer to the heir (_transfer_titles);
        the realm structure survives intact under the heir; a coronation is logged; NO
        trust is copied (the heir must hold the realm on its own standing). A DEPENDENT
        child heir still takes the seat but its levy/muster/war powers stay dormant via the
        existing is_dependent_child gate until it comes of age — a historically-real child
        REGENCY (the seat is defended by its inherited garrison; it just wages nothing).
      * NO living kin -> the LINE IS EXTINGUISHED: the records clear exactly as today (the
        existing breakaway/re-conquest machinery dissolves the leaderless realm), logged
        distinctly. succeed_titles moves nothing in this case.

    Deterministic; ZERO RNG; ZERO LLM. Runs only when lineage is on (the caller gates it),
    so a lineage-off run never calls this and stays byte-identical.
    """
    result: dict[str, Any] = {"heir": None, "kind": "none", "titles": ""}
    if not _holds_force_title(deceased.name, state):
        return result  # no dynastic title to pass (commoner / trust-leader only)

    titles = _title_summary(deceased.name, state)
    result["titles"] = titles
    heir, kind = _succession_heir(deceased, state)

    if heir is None:
        # Extinct line — leave the records to the existing dissolution machinery.
        result["kind"] = "extinct"
        state["events"].append(
            f"turn {turn}: the line of {deceased.name} is extinguished; "
            f"the crown of [{titles}] lies vacant")
        world.record_memory(deceased, f"Died with no heir — the line of {deceased.name} is extinguished")
        return result

    _transfer_titles(state, deceased.name, heir.name)
    result.update(heir=heir.name, kind=kind)
    regency = " (a minor — the crown is held in regency)" if getattr(heir, "dependent", False) else ""
    state["events"].append(
        f"turn {turn}: {heir.name} succeeded {deceased.name} as [{titles}] "
        f"(eldest {kind}){regency}")
    world.record_memory(heir, f"Succeeded {deceased.name} as [{titles}]{regency}")
    return result


# --- The per-turn update ---------------------------------------------------------
def update(state: dict[str, Any], turn: int,
           rng: "random.Random | None" = None) -> list[Any]:
    """Advance lineage one turn: age, mature, die of old age, feed, bear (M4.1).
    Returns the children born this turn (for the caller's logging).

    Called once per turn by the main loop ONLY when lineage_on — a default run
    never reaches here and stays byte-identical (respawn untouched: the existing
    process_respawns gate, living < TARGET_POPULATION, IS the extinction floor;
    above it a queued respawn is silently dropped exactly as today, so with
    births active the respawn system stays quiet on its own).

    Order within the turn: (0) any agent without a lifespan — a backstop respawn
    newcomer — is assigned one from the seeded stream; (1) everyone ages, and
    dependents whose age reaches CHILDHOOD_TURNS come of age (full agents from
    next turn); (2) agents at their lifespan die of OLD AGE via the existing
    death path (announce_death, distinct wording — survivor memories, events,
    post-mortem all standard; a dead ruler's titles now pass to an heir via M4.3
    succeed_titles, which announce_death runs on every death); (3) parents feed
    dependent children; (4) births.
    """
    r = rng or random

    living = [a for a in state["agents"] if a.alive]

    # 0. A backstop newcomer (Day 14 respawn) enters without a lifespan — mortal
    #    like everyone once lineage governs time. Drawn in stable order.
    for a in living:
        if a.lifespan <= 0:
            a.lifespan = r.randint(LIFESPAN_MIN, LIFESPAN_MAX)

    # 1. Aging is universal — kings, leaders, children, everyone.
    for a in living:
        a.age += 1
        if a.dependent and a.age >= CHILDHOOD_TURNS:
            a.dependent = False
            world.record_memory(a, "Came of age — now a full adult")
            state["events"].append(f"turn {turn}: {a.name} came of age")

    # 2. Natural death at lifespan's end, through the EXISTING death path.
    for a in living:
        if a.alive and a.age >= a.lifespan:
            population.announce_death(
                a, turn, state, cause="old age",
                final_memory="Died of old age",
                note="they died of old age")

    # 3. Childhood upkeep: the real, visible cost of raising the next generation.
    _feed_children(state, turn)

    # 4. New life, if every gate holds.
    return _births(state, turn, r)
