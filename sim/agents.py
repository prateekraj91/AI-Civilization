"""
agents.py
=========

Defines the Agent data model for AI Civilization.

ARCHITECTURE NOTE
-----------------
An Agent is a *plain data container*. It holds no game logic and never mutates
the world directly. Agents only ever READ from `world_state` (the single source
of truth) when deciding what to do. Any change to the world flows back through
the world layer — never by an agent writing to global state.

Keeping Agent as a dataclass (pure data) makes future features trivial to add:
  - serialization / save-load
  - sending agent state to Gemini as context
  - God Mode mutating agents via world_state

DAY 6 ADDITION
--------------
The world now hosts MULTIPLE agents (Alex, Bob, Kira) that share one grid and
compete for the same food. Each agent carries an explicit `alive` flag so the
simulation loop can skip the dead without removing them from the world (their
final state stays inspectable, which matters for logging and future features
like reputation or post-mortems).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Agent:
    """A single inhabitant of the civilization.

    Fields are intentionally simple/primitive so the whole agent can be
    serialized and embedded inside `world_state` without special handling.
    """

    name: str
    personality: str

    # goals: a weighted map of drives, e.g. {"survive": 8, "wealth": 3}.
    # Higher number = stronger motivation. Used later for decision-making.
    goals: dict[str, int] = field(default_factory=dict)

    # hunger: 0 = full, higher = hungrier. Simple survival stat for now.
    hunger: int = 0

    # alive: False once the agent has starved (Day 6). The simulation keeps dead
    # agents in the world for inspection but skips their turns.
    alive: bool = True

    # position: (x, y). No world grid yet (Day 1), but the field exists so the
    # data model is ready for movement without a future schema migration.
    position: tuple[int, int] = (0, 0)

    # inventory: list of items the agent is carrying.
    inventory: list[Any] = field(default_factory=list)

    # memory: append-only log of what the agent has experienced/observed.
    # No summarization yet (explicitly out of scope for Day 1).
    memory: list[Any] = field(default_factory=list)

    # inbox: pending messages from other agents (Day 8 "talk"). Each entry is a
    # dict {"from", "text", "turn", "reply"}. A message sent on turn T is only
    # consumed on a LATER turn, so it lands in the recipient's NEXT decision
    # context (never the same tick). Cleared once consumed.
    inbox: list[Any] = field(default_factory=list)

    # relationships: per-other-agent opinion (Day 9 trust). Lazily created on
    # first contact as {other_name: {"trust": int, "interactions": int}}. Pure
    # Python bookkeeping over conversation events — never an LLM call.
    relationships: dict[str, Any] = field(default_factory=dict)

    # allies: names of agents this one is currently ALLIED with (Day 13). Always
    # mutual — if Bob is in Alex.allies then Alex is in Bob.allies. Allies share
    # food sightings; betrayal removes the name from both sides permanently.
    allies: set = field(default_factory=set)

    # ally_offers: names of agents who have PROPOSED an alliance to this one and
    # are awaiting its answer (Day 13). An alliance forms only when the offered
    # agent answers with its own ally_with action — never unilaterally.
    ally_offers: set = field(default_factory=set)

    # plague_until: the last turn this agent is sick (Day 16 God mode). While the
    # current turn is <= this value the hunger-update step drains extra hunger per
    # turn (simulating a plague); 0 means healthy. God mode sets it via world_state;
    # the existing hunger loop reads it and applies the effect — no scripted reaction.
    plague_until: int = 0

    # knowledge: the named facts/skills this agent KNOWS (V2 milestone M1.1) — e.g.
    # {"fire", "food_location_north"}. This is the first piece of CULTURE: state that
    # exists beyond any single agent and spreads between them. It propagates purely
    # through the existing contact network (knowledge.diffuse): when a knower and a
    # non-knower are adjacent, the item may be adopted with a probability shaped by
    # trust + personality — NO LLM call per learner. Empty by default, so an agent with
    # no seeded knowledge behaves exactly as in v1 (diffusion self-gates to a no-op when
    # nobody knows anything). Discovery/invention is OUT of scope here (that is M1.2);
    # M1.1 is purely about correct, cheap SPREADING of already-known items.
    knowledge: set = field(default_factory=set)

    # cognition: which MIND drives this agent's strategy choice (V2 milestone M0.1).
    #   "llm"       -> ask the model layer for a high-level strategy (the V1 default,
    #                  so existing behaviour is unchanged).
    #   "heuristic" -> derive the strategy from pure-Python rules (heuristic.py),
    #                  making ZERO model calls.
    # Either way the strategy is the SAME shape the executor (strategy.choose_action)
    # runs, so the rest of the loop is identical and cannot tell the minds apart.
    # This is a clean per-agent switch only — there is NO tier/promotion system here
    # (that is M0.2); an agent's cognition does not change on its own.
    cognition: str = "llm"

    # settlement: the id of the SETTLEMENT this agent belongs to, or None for a nomad
    # (V2 milestone M2.1 — the first durable civilizational artifact). None by default,
    # so a nomad behaves exactly as in Phase 1. Set by settlement.update when the agent
    # joins a forming/existing settlement that grew around reliable (farmed) food; once
    # set it gives the agent a gentle "home-pull" toward its settlement centre in
    # strategy.choose_action (survival still overrides — a starving member forages out).
    settlement: "str | None" = None

    # settle_streak: consecutive turns this agent has been NEAR reliable food (M2.1).
    # Pure bookkeeping read only by settlement.update: a settlement forms when enough
    # agents have SUSTAINED a streak together near the same food, which is what makes
    # settling a CONSEQUENCE of the food economy (transient scattered food never keeps a
    # cluster's streak high for long; a maintained farm plot does). 0 = not near food now.
    settle_streak: int = 0

    # stockpile: this agent's PERSONAL stored-food reserve (V2 milestone M2.2 — storable
    # SURPLUS, the moment the sim grows WEALTH). 0.0 by default, so a nomad/v1 agent is
    # unchanged. Only a SETTLED agent banks into it (settlement is what makes real storage
    # possible — see storage.accumulate); the amount it accumulates EMERGES from traits it
    # already has — its personality (a hoarding independent/competitive agent banks more, a
    # sharing friendly one less) and its farming KNOWLEDGE (a producer accumulates faster) —
    # so wealth ends up VARYING across agents as a consequence of WHO they are, never an
    # assigned "richness" stat. Bounded by storage.STORAGE_CAP. It is a SURVIVAL BUFFER: a
    # member that would otherwise starve draws this down to weather a food shock (a drought
    # that kills its savings-less neighbours), wiring wealth straight into survival.
    stockpile: float = 0.0

    # money: this agent's emergent, FOOD-BACKED currency (V2 milestone M2.3 — trade closes
    # the economy). 0.0 by default, so a nomad/v1 agent is unchanged. Money is a CLAIM ON
    # FOOD, never minted by an authority and never fiat (decreed/minted currency is deferred
    # to Phase 3): it is created ONLY when a settled agent's food surplus runs PAST the M2.2
    # storage cap (economy.mint) — food it produced but cannot store — and it has value only
    # because it is redeemable as food (storage.draw_down spends it to survive) and because
    # other agents accept it in trade. It is the unit of account that lets agents TRADE
    # food/knowledge across their differences without a double coincidence of wants
    # (economy.trade); a buyer pays money, a seller banks it. Wealth in money, like stockpile,
    # EMERGES from who an agent is (a hoarding producer mints the most) — never assigned.
    money: float = 0.0

    # --- V2 M4.1 LINEAGE (birth, childhood, aging, family) -------------------
    # All five fields are inert defaults unless the run opts in (--lineage /
    # world_state["lineage_on"]): nothing reads them when the system is off, so a
    # default run is byte-identical. Wealth inheritance at death is M4.2 and
    # dynastic succession of titles is M4.3 — NEITHER is built here.

    # age: turns since birth (or since world creation for the founding cast, which
    # lineage.init_cast seeds with varied adult ages from the seeded stream).
    age: int = 0

    # lifespan: the age at which this agent dies of OLD AGE. Drawn once from the
    # seeded sim stream at creation (founders in init_cast, newborns at birth,
    # backstop respawns on their first lineage tick). 0 = unset/immortal — the
    # lineage-off world where only starvation and battle kill.
    lifespan: int = 0

    # parents: the two parent names, or () for a founder/respawned blank slate.
    # The permanent family link a birth records (read by M4.2/M4.3 later).
    parents: tuple = ()

    # dependent: True while this agent is a CHILD — it takes no actions (no
    # foraging/production/trade/war), is fed from its parents' stores, and learns
    # at a boosted rate. Flips False when age reaches lineage.CHILDHOOD_TURNS.
    dependent: bool = False

    # last_child_turn: when this agent last parented a child — births are paced
    # by lineage.BIRTH_COOLDOWN. Large-negative default = never.
    last_child_turn: int = -10**9
