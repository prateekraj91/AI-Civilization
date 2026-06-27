"""
strategy.py
===========

The lightweight STRATEGY layer (Phase 4) and the PERSONALITY-driven action
executor (Phase 1). This is where "what the agent wants" becomes "what the agent
does this turn" — almost always WITHOUT calling the LLM.

The cost problem
----------------
Asking the LLM for an action every single turn is expensive and slow. Instead:

    every N turns:   ask the LLM for a high-level STRATEGY (cheap, occasional)
    every turn:      EXECUTE that strategy in pure Python (free, instant)

A `Strategy` is a tiny cached intent ("seek_food", "explore north", "approach
Bob"). Between refreshes, `choose_action()` turns the cached strategy — coloured
by the agent's personality, hunger, and surroundings — into one concrete action
from world.VALID_ACTIONS. The result is the same closed action vocabulary every
other layer already speaks, so nothing downstream changes.

How personality shows up (Phase 1)
----------------------------------
Personality affects execution at three points:
  - eat/rest cadence: cautious agents eat early and rest near food; others push on.
  - the strategy default (when the strategy is vague): friendly → toward agents,
    independent → away from agents, curious → keep exploring, cautious → toward food.
  - exploration wander pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import alliance
import leadership
import settlement
import trust
import world
from personality import Personality

# The closed set of high-level strategies the LLM may pick (mirrors the
# milestone's examples: "Search for food", "Explore north", "Stay near Bob").
VALID_STRATEGIES: tuple[str, ...] = (
    "seek_food",   # head toward the nearest food
    "explore",     # roam, optionally in a named direction (target = direction)
    "approach",    # move toward a named agent (target = agent name)
    "avoid",       # keep away from other agents
    "talk",        # talk to a named agent (target = agent name); message optional
    "steal",       # take food from a named agent (target = agent name) — Day 12
    "ally",        # propose/accept an alliance with a named agent (target) — Day 13
    "betray",      # betray a current ally (target = agent name) — Day 13
    "rest",        # conserve / hold position
    "wander",      # no strong plan — defer to personality (safe default)
)

# Compass directions usable as an `explore` target, and their (dx, dy) deltas.
DIRECTIONS: tuple[str, ...] = ("north", "south", "east", "west")
_DELTA: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}
_OPPOSITE: dict[str, str] = {
    "north": "south", "south": "north", "east": "west", "west": "east",
}

# Hunger at/above which survival overrides the current strategy and the agent
# makes a beeline for food regardless of personality. Day 9 rebalance: lowered
# from 7 to 5 so an agent starts seeking with enough buffer to reach food AND
# spend the extra turn it costs to actually eat it before starving.
SURVIVAL_HUNGER: int = 5

# A friendly agent only goes out of its way to socialise when WELL-fed (below
# this hunger). Above it, food comes first — otherwise a friendly agent can chat
# (or chase a fleeing loner) until it starves. Leaves a comfortable buffer before
# SURVIVAL_HUNGER takes over.
SOCIAL_MAX_HUNGER: int = 3

# How near (Manhattan distance) counts as "near food" for cautious resting.
NEAR_FOOD_RADIUS: int = 2

# Hunger at/above which an agent is desperate enough to STEAL from a neighbour
# (Day 12). Higher than SURVIVAL_HUNGER: an agent first tries to reach its own
# food and only robs someone when genuinely close to starving. Stealing is also
# gated on DISTRUST and personality (see _will_steal) so it is a rational choice
# under scarcity, never indiscriminate.
STEAL_DESPERATION: int = 6


@dataclass
class Strategy:
    """A cached high-level intent plus when it was issued.

    `kind` is one of VALID_STRATEGIES. `target` is a direction for "explore" or an
    agent name for "approach"/"talk" (empty otherwise). `issued_turn` lets the
    caller decide when the strategy is stale and a refresh is due.

    Two optional fields ride along from the SAME strategy LLM call so that talking
    never needs an extra inference (Day 8):
      - `message`:  what to say if this strategy is "talk" (used only on the
                    refresh turn that produced it; non-refresh turns template it).
      - `reaction`: how to react to an incoming message this turn — one of
                    reply/ignore/hostile, or "" to fall back to a personality rule.
    """

    kind: str = "wander"
    target: str = ""
    message: str = ""
    reaction: str = ""
    issued_turn: int = -10_000

    def label(self) -> str:
        """Compact human label, e.g. 'explore north' or 'talk Bob'."""
        return f"{self.kind} {self.target}".strip()


# --- Personality caching ---------------------------------------------------
def get_personality(agent: Any) -> Personality:
    """Return the agent's parsed Personality, caching it on the agent.

    Parsing is cheap, but caching keeps trait lookups trivial and avoids
    re-parsing the same string every turn. The cache invalidates itself if the
    personality text ever changes.
    """
    cached = getattr(agent, "_personality_cache", None)
    if cached is None or cached[0] != agent.personality:
        cached = (agent.personality, Personality.from_text(agent.personality))
        agent._personality_cache = cached
    return cached[1]


# --- Geometry / navigation helpers ----------------------------------------
def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    """King-move distance — the radius the M2.1 home-pull measures against."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _nearest(pos: tuple[int, int],
             targets: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Closest target position to `pos` by Manhattan distance, or None.

    M0.3: an inlined first-minimum-wins loop rather than `min(key=lambda ...)`. This
    is called once per hungry/social agent per turn over the whole food (or agent)
    list, so at 200-300 agents it was a top runtime cost; folding the distance inline
    drops ~1M lambda + helper calls per scaled run. Tie-break is unchanged — the FIRST
    target achieving the minimum wins (strict `<`), so output is byte-identical."""
    if not targets:
        return None
    px, py = pos
    best = None
    best_d = -1
    for tx, ty in targets:
        d = abs(px - tx) + abs(py - ty)
        if best is None or d < best_d:
            best, best_d = (tx, ty), d
    return best


def _other_agent_positions(agent: Any, state: dict[str, Any]) -> list[tuple[int, int]]:
    return [a.position for a in state["agents"] if a.alive and a is not agent]


def _agent_position(state: dict[str, Any], name: str) -> tuple[int, int] | None:
    for a in state["agents"]:
        if a.alive and a.name == name:
            return a.position
    return None


def _dirs_toward(src: tuple[int, int], dst: tuple[int, int]) -> list[str]:
    """Directions that reduce distance to `dst`, strongest axis first."""
    sx, sy = src
    dx, dy = dst
    candidates: list[tuple[str, int]] = []
    if dx > sx:
        candidates.append(("east", dx - sx))
    elif dx < sx:
        candidates.append(("west", sx - dx))
    if dy > sy:
        candidates.append(("south", dy - sy))
    elif dy < sy:
        candidates.append(("north", sy - dy))
    candidates.sort(key=lambda c: -c[1])
    return [d for d, _ in candidates]


def _dirs_away(src: tuple[int, int], frm: tuple[int, int]) -> list[str]:
    """Directions that increase distance from `frm`."""
    return [_OPPOSITE[d] for d in _dirs_toward(src, frm)]


def _open_dirs(scan: dict[str, Any]) -> list[str]:
    """Directions the agent could actually move into (no wall, no other agent)."""
    return [d for d, cell in scan["cells"].items()
            if not cell["wall"] and not cell["blocked"]]


def _adjacent_food_dir(scan: dict[str, Any]) -> str | None:
    """A direction holding reachable food, if any."""
    for d, cell in scan["cells"].items():
        if cell["food"] and not cell["blocked"]:
            return d
    return None


def _adjacent_agent_names(scan: dict[str, Any]) -> list[str]:
    """Names of living agents in adjacent cells (sorted for determinism)."""
    return sorted(
        cell["agent"].name for cell in scan["cells"].values()
        if cell["agent"] is not None
    )


def _adjacent_food_holders(scan: dict[str, Any]) -> list[Any]:
    """Adjacent living agents that are standing on a food tile (theft targets).

    A neighbour on food shows up as cell["agent"] set AND cell["food"] True (the
    cell is a food coordinate that the neighbour occupies). Sorted by name so the
    choice is deterministic.
    """
    holders = [
        cell["agent"] for cell in scan["cells"].values()
        if cell["agent"] is not None and cell["food"]
    ]
    return sorted(holders, key=lambda a: a.name)


def _will_steal(thief: Any, victim: Any, pers: Personality) -> bool:
    """Whether `thief` would rob `victim` of food when desperate (Day 12).

    Encodes requirement: steal on LOW trust, and let personality tilt it —
    independent/competitive agents steal more readily, friendly/cautious less so.

      - Never steal from someone you actively trust (bucket 'high').
      - Independent (competitive) agents steal from anyone they don't trust
        (neutral OR low) — they put survival over the relationship.
      - Everyone else steals only from an agent they actively DISTRUST (low) —
        a real grudge, e.g. after a prior theft or hostility.
    """
    raw = thief.relationships.get(victim.name, {}).get("trust", 0)
    bucket = trust.trust_bucket(raw)
    if bucket == "high":
        return False
    if pers.dominant == "independence":
        return True
    return bucket == "low"


def _will_ally(agent: Any, other: Any, pers: Personality) -> bool:
    """Whether `agent` would form an alliance with `other` (Day 13).

    Eligibility (alive, not already allied, no grudge either way) is checked by
    alliance.can_ally; this adds the PERSONALITY-driven willingness on top:

      - Friendly / cautious agents ally READILY — with any neighbour they do not
        actively distrust (neutral or high trust is enough).
      - Independent / competitive agents (Kira) ally RELUCTANTLY — only with
        someone they ACTIVELY trust (high), because a loner joins forces only when
        the relationship has already proven itself.

    A latent grudge always wins (can_ally returns False) regardless of trust.
    """
    if not alliance.can_ally(agent, other):
        return False
    bucket = trust.trust_bucket(agent.relationships.get(other.name, {}).get("trust", 0))
    if pers.dominant == "independence":
        return bucket == "high"
    return bucket != "low"


def _will_betray(agent: Any, ally_name: str, pers: Personality) -> bool:
    """Whether `agent` would betray its ally `ally_name` this turn (Day 13).

    Betrayal is personality-driven and rational, never random: only an
    independent/competitive agent betrays, and only under genuine survival
    pressure (hunger >= STEAL_DESPERATION), when the alliance no longer pays its
    way against the permanent-grudge cost. Friendly and cautious agents keep the
    alliances they form — they never betray. The agent must actually be allied.
    """
    if ally_name not in getattr(agent, "allies", set()):
        return False
    if pers.dominant != "independence":
        return False
    return agent.hunger >= STEAL_DESPERATION


def _navigate(scan: dict[str, Any], preferred: list[str]) -> str:
    """Pick the best open move from `preferred`, falling back to any open dir.

    Returns a `move_<dir>` action, or "rest" if completely boxed in.
    """
    open_dirs = _open_dirs(scan)
    for d in preferred:
        if d in open_dirs:
            return f"move_{d}"
    if open_dirs:
        return f"move_{open_dirs[0]}"
    return "rest"


def _explore(agent: Any, scan: dict[str, Any]) -> str:
    """Wander to an open cell, rotating direction so the agent actually roams.

    Never rests while a move exists — this is what makes curious/exploratory
    agents move noticeably more than cautious ones. The rotation seed shifts with
    hunger so paths meander instead of running dead straight into a wall.
    """
    open_dirs = _open_dirs(scan)
    if not open_dirs:
        return "rest"
    order = ("north", "east", "south", "west")
    start = (agent.hunger + len(agent.name)) % 4
    for i in range(4):
        d = order[(start + i) % 4]
        if d in open_dirs:
            return f"move_{d}"
    return f"move_{open_dirs[0]}"


def _near_food(pos: tuple[int, int], state: dict[str, Any]) -> bool:
    return any(_manhattan(pos, f) <= NEAR_FOOD_RADIUS for f in state["food"])


# --- The executor ----------------------------------------------------------
def choose_action(agent: Any, strat: Strategy | None,
                  state: dict[str, Any]) -> tuple[str, str]:
    """Decide this turn's concrete action from strategy + personality + senses.

    Returns (action, note) where `action` is a member of world.VALID_ACTIONS and
    `note` is a short human explanation for logging. Pure Python — no LLM call.

    Priority cascade:
      1. Eat if standing on worthwhile food.
      2. Survival: if starving, grab adjacent food; else (Day 12) if desperate
         and an adjacent distrusted neighbour sits on food, STEAL it; else
         beeline to the nearest food.
      3. Alliance (Day 13): when not starving, ACCEPT a pending ally offer from a
         willing neighbour, or (friendly/cautious) PROPOSE one to a trusted
         neighbour. Allying is a social investment, never an emergency act.
      4. Cautious rest: a cautious, well-fed agent near food holds position.
      5. Execute the cached strategy if it yields a concrete move (incl. an
         LLM-chosen 'steal'/'ally'/'betray').
      6. Otherwise fall back to the personality default.
    """
    pers = get_personality(agent)
    s = world.scan(agent, state)
    pos = s["pos"]

    # 1. Eat what's underfoot when it's worth a turn.
    if s["on_food"] and agent.hunger >= pers.eat_threshold:
        return "eat", "standing on food"

    # 2. Survival override — ignore strategy when close to starving.
    if agent.hunger >= SURVIVAL_HUNGER:
        # M1.3 tools: a tool-user FORAGES adjacent food in place (reach) rather than
        # spending a turn stepping onto it — a real edge under survival pressure. Gated
        # on knowing 'tools', so a v1 agent (empty knowledge) behaves exactly as before.
        if "tools" in getattr(agent, "knowledge", ()) and _adjacent_food_dir(s) is not None:
            return "eat", "survival: forage adjacent food (tools)"
        # 2a. Prefer free, unowned food adjacent to us.
        d = _adjacent_food_dir(s)
        if d:
            return f"move_{d}", "survival: grab adjacent food"
        # 2b. Desperation turns on neighbours sitting on food. Day 13: an ally
        # hoarding food while we starve is the moment an independent agent betrays
        # the alliance (it no longer pays) — checked BEFORE theft, since you can't
        # steal from an ally, you renounce them first. Day 12: anyone else we
        # distrust, we steal from outright.
        if agent.hunger >= STEAL_DESPERATION:
            for holder in _adjacent_food_holders(s):
                if holder.name in agent.allies and _will_betray(agent, holder.name, pers):
                    return f"betray_alliance_{holder.name}", f"desperate: betray ally {holder.name}"
            for holder in _adjacent_food_holders(s):
                if holder.name not in agent.allies and _will_steal(agent, holder, pers):
                    return f"steal_from_{holder.name}", f"desperate: steal from {holder.name}"
        # 2c. Otherwise head for the nearest free food on the map.
        nearest = _nearest(pos, state["food"])
        if nearest:
            return _navigate(s, _dirs_toward(pos, nearest)), "survival: head to food"

    # 3. Alliance (Day 13). Allying is a social investment, so it is gated on NOT
    # starving (survival above always wins).
    #   3a. ACCEPT first: if a willing neighbour has already offered, seal it —
    #       this is the second half of the mutual handshake, and responding to a
    #       proposal takes precedence over resting or another greeting.
    #   3b. Otherwise friendly/cautious agents PROPOSE — but only to a neighbour
    #       they ALREADY TRUST (high), so a bond is built by talking first and an
    #       explicit talk/seek strategy is never overridden by an unprompted
    #       proposal. Independent agents never court; they only ever accept.
    if agent.hunger < SURVIVAL_HUNGER:
        adj = world.adjacent_agents(agent, state)
        for name in sorted(agent.ally_offers):
            other = adj.get(name)
            if other is not None and _will_ally(agent, other, pers):
                return f"ally_with_{name}", f"accept alliance with {name}"
        if pers.dominant in ("friendliness", "caution"):
            for name in sorted(adj):
                other = adj[name]
                raw = agent.relationships.get(name, {}).get("trust", 0)
                if (trust.trust_bucket(raw) == "high"
                        and agent.name not in other.ally_offers
                        and not alliance.are_allied(agent, other)
                        and _will_ally(agent, other, pers)):
                    return f"ally_with_{name}", f"propose alliance to {name}"

    # 3b. Home-pull (M2.1): a SETTLED, fed agent drifts back toward its settlement
    # centre once it has wandered beyond settlement.HOME_RADIUS, instead of roaming
    # freely — this is what makes a settlement a PLACE its members stay near. It sits
    # AFTER the survival override and the eat-underfoot check above, so a starving
    # member still forages outward (survival always wins) and an agent on food still
    # eats; and after alliance accept/propose, so a member already beside a willing
    # partner can still seal a bond. It is INERT for a nomad (settlement is None) and
    # when the system is off (settlements never form), so a v1 run is byte-identical.
    sid = getattr(agent, "settlement", None)
    if sid is not None:
        record = state.get("settlements", {}).get(sid)
        if record is not None:
            target, radius = record["center"], settlement.HOME_RADIUS
            note = "home-pull: drift toward settlement"
            # M3.2: a FOLLOWER coordinates more tightly around its LEADER (the minimal
            # leadership effect — influence, not tax/law). following_target retargets the
            # pull to the leader's tile at a tighter radius; it returns None (so the plain
            # settlement pull above stands) for the leader itself, a non-follower, or when
            # leadership is off (leaders empty), keeping a no-leadership run byte-identical.
            led = leadership.following_target(state, agent)
            if led is not None:
                target, radius = led
                note = "home-pull: rally to leader"
            if _chebyshev(pos, target) > radius:
                return (_navigate(s, _dirs_toward(pos, target)), note)

    # 4. Cautious agents conserve near a food cache when not yet hungry.
    if pers.dominant == "caution" and agent.hunger < pers.comfort and _near_food(pos, state):
        return "rest", "cautious: resting near food"

    # 4b. A well-fed FRIENDLY agent actively seeks company so social dynamics
    # (talk + trust) actually happen — otherwise a food/explore strategy keeps it
    # near the abundant food and it never meets anyone. Only when well-fed, so it
    # never socialises itself into starvation. (This is also what builds the trust
    # an alliance later needs — see step 3.)
    if pers.dominant == "friendliness" and agent.hunger < SOCIAL_MAX_HUNGER:
        adjacent = _adjacent_agent_names(s)
        if adjacent:
            return f"talk_to_{adjacent[0]}", f"friendly: greet {adjacent[0]}"
        nearest_agent = _nearest(pos, _other_agent_positions(agent, state))
        if nearest_agent:
            return _navigate(s, _dirs_toward(pos, nearest_agent)), "friendly: seek company"

    # 4. Execute the cached strategy.
    acted = _strategy_action(agent, strat, s, state)
    if acted is not None:
        return acted

    # 5. Personality-driven default.
    return _personality_default(agent, s, state, pers)


def _strategy_action(agent: Any, strat: Strategy | None, scan: dict[str, Any],
                     state: dict[str, Any]) -> tuple[str, str] | None:
    """Translate a concrete strategy into an action, or None to defer.

    Returns None for vague strategies ("wander", or "explore" with no usable
    direction) so the personality default takes over — this is deliberately how
    personality keeps shining through even while a strategy is cached.
    """
    if strat is None:
        return None
    pos = scan["pos"]

    if strat.kind == "seek_food":
        d = _adjacent_food_dir(scan)
        if d:
            return f"move_{d}", "seek_food: step onto food"
        nearest = _nearest(pos, state["food"])
        if nearest:
            return _navigate(scan, _dirs_toward(pos, nearest)), "seek_food: toward nearest food"
        return None

    if strat.kind == "explore":
        if strat.target in _DELTA and strat.target in _open_dirs(scan):
            return f"move_{strat.target}", f"explore {strat.target}"
        return None  # no/blocked direction -> personality default

    if strat.kind == "approach":
        target_pos = _agent_position(state, strat.target)
        if target_pos:
            return _navigate(scan, _dirs_toward(pos, target_pos)), f"approach {strat.target}"
        return None

    if strat.kind == "avoid":
        nearest = _nearest(pos, _other_agent_positions(agent, state))
        if nearest:
            return _navigate(scan, _dirs_away(pos, nearest)), "avoid: move away from others"
        return None

    if strat.kind == "talk":
        target = strat.target
        if target and target in _adjacent_agent_names(scan):
            return f"talk_to_{target}", f"talk to {target}"
        target_pos = _agent_position(state, target)
        if target_pos:  # alive but out of range — close the distance first
            return _navigate(scan, _dirs_toward(pos, target_pos)), f"approach {target} to talk"
        # target missing/dead — still emit talk so it logs the documented no-op
        if target:
            return f"talk_to_{target}", f"talk to {target} (no one there)"
        return None

    if strat.kind == "steal":
        # Day 12: the LLM chose to rob a named agent. Steal only if that agent is
        # adjacent AND on food; if alive but out of reach, close the distance.
        target = strat.target
        target_pos = _agent_position(state, target)
        if target and target_pos:
            on_food = target_pos in state["food"]
            if target in _adjacent_agent_names(scan) and on_food:
                return f"steal_from_{target}", f"steal from {target}"
            return _navigate(scan, _dirs_toward(pos, target_pos)), f"approach {target} to steal"
        return None

    if strat.kind == "ally":
        # Day 13: the LLM chose to ally with a named agent. Forming is mutual and
        # range-gated (adjacent), so only emit ally_with when the target is in
        # reach; if alive but out of range, close the distance first.
        target = strat.target
        if target and target in _adjacent_agent_names(scan):
            return f"ally_with_{target}", f"ally with {target}"
        target_pos = _agent_position(state, target)
        if target and target_pos:
            return _navigate(scan, _dirs_toward(pos, target_pos)), f"approach {target} to ally"
        return None

    if strat.kind == "betray":
        # Day 13: the LLM chose to betray an ally. Valid only if currently allied;
        # needs no adjacency (renouncing a bond is one-sided). Otherwise defer.
        target = strat.target
        if target and target in getattr(agent, "allies", set()):
            return f"betray_alliance_{target}", f"betray ally {target}"
        return None

    if strat.kind == "rest":
        return "rest", "strategy: rest"

    # "wander" (and anything else) defers to personality.
    return None


def _personality_default(agent: Any, scan: dict[str, Any], state: dict[str, Any],
                         pers: Personality) -> tuple[str, str]:
    """Default behaviour when the strategy gives no concrete move (Phase 1)."""
    pos = scan["pos"]
    dom = pers.dominant

    if dom == "friendliness":
        adjacent = _adjacent_agent_names(scan)
        if adjacent:  # someone in reach — say hello (templated, no LLM call)
            return f"talk_to_{adjacent[0]}", f"friendly: greet {adjacent[0]}"
        nearest = _nearest(pos, _other_agent_positions(agent, state))
        if nearest:
            return _navigate(scan, _dirs_toward(pos, nearest)), "friendly: toward nearest agent"
        return _explore(agent, scan), "friendly: explore (no one around)"

    if dom == "independence":
        nearest = _nearest(pos, _other_agent_positions(agent, state))
        if nearest:
            return _navigate(scan, _dirs_away(pos, nearest)), "independent: away from others"
        return _explore(agent, scan), "independent: explore alone"

    if dom == "caution":
        nearest = _nearest(pos, state["food"])
        if nearest:
            return _navigate(scan, _dirs_toward(pos, nearest)), "cautious: toward known food"
        return "rest", "cautious: hold position"

    # curiosity (and the balanced default)
    return _explore(agent, scan), "curious: explore"


# --- Strategy prompt (Phases 2 & 3) ---------------------------------------
def format_goals(goals: dict[str, int]) -> str:
    """Goals rendered strongest-first, e.g. 'survive=8, friendship=5, wealth=3'."""
    if not goals:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in sorted(goals.items(), key=lambda kv: -kv[1]))


def recent_memories(memory: list[str], limit: int) -> list[str]:
    """The most recent `limit` memories (keeps prompts compact, Phase 3)."""
    return memory[-limit:]


def hunger_line(hunger: int) -> str:
    """An escalating, unambiguous hunger status line for the prompt (Day 9).

    Vague hunger lets the model wander while starving; spelling out the urgency
    reliably pushes seek_food over explore when it matters.
    """
    if hunger >= 7:
        return (f"Hunger: {hunger}/10 — CRITICAL: you will DIE within a few turns. "
                f"Finding and eating food is the ONLY priority.")
    if hunger >= 4:
        return f"Hunger: {hunger}/10 — getting hungry; head toward food soon."
    return f"Hunger: {hunger}/10 — well fed."


def build_strategy_prompt(agent: Any, observation: str, *, memory_limit: int = 6,
                          incoming: list[str] | None = None,
                          state: dict[str, Any] | None = None) -> str:
    """Build the (occasional) strategy prompt: identity + goals + memory + senses.

    Compact by design — it is sent only every N turns. It tells the model who the
    agent is (personality), WHAT IT WANTS (goals, Phase 2), what it has recently
    seen (memories, Phase 3), any messages just received (Day 8), its current
    alliances and the food its allies can see (Day 13), and its surroundings, then
    asks for ONE strategy.

    The schema carries `message` (what to say if talking) and `reaction` (how to
    answer a received message) so a "talk", a reply, an ally/betray decision all
    cost NO extra inference — they ride along with this single strategy call. The
    allies + shared-food blocks appear only when `state` is supplied.
    """
    pers = get_personality(agent)
    mems = recent_memories(agent.memory, memory_limit)
    mem_block = "\n".join(f"- {m}" for m in mems) if mems else "- (none yet)"

    trust_line = trust.trust_summary(agent)
    trust_block = f"{trust_line}\n" if trust_line else ""

    inbox_block = ""
    if incoming:
        lines = "\n".join(f"- You received from {m}" for m in incoming)
        inbox_block = (
            f"\nMessages you just received (decide a reaction):\n{lines}\n"
        )

    # Day 13: current alliances, pending offers, and the concrete benefit —
    # food only your allies can see, folded straight in so it informs the plan.
    ally_block = ""
    allies = sorted(getattr(agent, "allies", set()))
    offers = sorted(getattr(agent, "ally_offers", set()))
    if allies:
        ally_block += f"Your allies: {', '.join(allies)} (you share food sightings).\n"
    if offers:
        ally_block += f"Alliance proposals awaiting your answer from: {', '.join(offers)}.\n"
    if state is not None:
        shared = alliance.shared_food_sightings(agent, state)
        if shared:
            sights = "; ".join(
                f"{name} sees food at {', '.join(str(c) for c in coords)}"
                for name, coords in shared.items()
            )
            ally_block += f"Food your allies can see (shared with you): {sights}.\n"

    # Day 15: surface any God-dropped treasure so the LLM path values it ABOVE
    # ordinary food (the deterministic executor already targets it, since treasure
    # is mirrored into the food list). Reading world_state only — no new mechanic.
    treasure_block = ""
    if state is not None and state.get("treasures"):
        spots = "; ".join(
            f"({t['pos'][0]}, {t['pos'][1]}) worth {t['value']}"
            for t in state["treasures"]
        )
        treasure_block = (
            f"TREASURE on the map (worth more than food — go claim it): {spots}.\n"
        )

    return (
        f"You are {agent.name}, a {agent.personality} agent on a shared 10x10 grid.\n"
        f"Dominant trait: {pers.dominant}.\n"
        f"{hunger_line(agent.hunger)}\n"
        f"Your goals (higher = more important): {format_goals(agent.goals)}\n"
        f"{trust_block}"
        f"{ally_block}"
        f"{treasure_block}\n"
        f"Recent memories:\n{mem_block}\n"
        f"{inbox_block}\n"
        f"Surroundings:\n{observation}\n\n"
        f"Pick ONE high-level strategy to pursue for the next few turns, consistent "
        f"with your personality and goals, and informed by your memories.\n"
        f"Valid strategies: {', '.join(VALID_STRATEGIES)}.\n"
        f"- If hunger is 6 or more, choose 'seek_food' (survival comes first)...\n"
        f"- ...UNLESS you are starving and a nearby agent you DISTRUST is sitting on "
        f"food: then 'steal' (target = their name) is a rational last resort. Food "
        f"is scarce. Stealing makes a lasting enemy, so weigh it against friendship — "
        f"if you are independent/competitive you steal readily; if friendly/cautious, "
        f"only when truly desperate or already wronged.\n"
        f"- 'ally' (target = a nearby agent's name) when you TRUST a neighbour and "
        f"want to team up: allies share food sightings, so two scouts beat one under "
        f"scarcity. An alliance forms only if BOTH of you choose it, and a grudge on "
        f"either side blocks it. Friendly/cautious agents ally readily; "
        f"independent/competitive agents only with someone they already trust.\n"
        f"- 'betray' (target = an ally's name) tears up an alliance: a desperate, "
        f"opportunistic last resort when the alliance no longer pays. It drops their "
        f"trust hard and earns a PERMANENT grudge (you can never ally again), so weigh "
        f"it heavily — only an independent/competitive agent under real survival "
        f"pressure should consider it.\n"
        f"- 'explore' may set target to one of: {', '.join(DIRECTIONS)}.\n"
        f"- 'approach'/'talk'/'steal'/'ally'/'betray' must set target to an agent's name.\n"
        f"- If 'talk', also set message to what you say.\n"
        f"- If you received a message, set reaction to one of: reply, ignore, hostile.\n\n"
        f"Respond with ONLY a JSON object, no extra text, shaped exactly:\n"
        f'{{"strategy": "<one valid strategy>", "target": "<direction/name or empty>", '
        f'"message": "<what to say if talking, else empty>", '
        f'"reaction": "<reply|ignore|hostile if you got a message, else empty>", '
        f'"reason": "<short reason>"}}'
    )
