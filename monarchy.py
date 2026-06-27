"""
monarchy.py
===========

CONQUEST & MONARCHY — power seized by FORCE (V2 milestone M3.4, Phase 3: Institutions). On top
of M3.1 (wage labor), M3.2 (legitimate leadership), M3.3 (taxation) and all of Phase 0/1/2.

The historical step M3.4 makes — the SECOND source of power
-----------------------------------------------------------
M3.2 built authority by CONSENT: a trust-legitimated leader, power GRANTED from below. M3.4
builds the OTHER source of power: DOMINATION — power SEIZED from above by FORCE. A wealthy agent
converts money into an ARMY of real fighters and attacks a settlement to seize it, becoming its
MONARCH — a persistent ruling title that confers authority by FORCE, not consent. This completes
the class engine's arc: M3.1 wealth COMPOUNDS -> M3.2/M3.3 legitimacy RESISTS and REDISTRIBUTES ->
M3.4 wealth ESCALATES to violence and takes the crown anyway. Scope is held to: conflict (force
vs force), conquest of ONE settlement, the MONARCH institution (rule by force, losable), and the
force-vs-legitimacy collision. NO underclass revolt, NO inter-kingdom war/diplomacy, NO
multi-settlement kingdoms (all later).

Military power EMERGES from WEALTH via real FIGHTERS (never a raw wealth compare)
--------------------------------------------------------------------------------
An ASPIRANT (a living agent with a war chest >= MIN_WAR_CHEST) raises a FORCE by SPENDING money
to hire FIGHTERS — real, poor/desperate agents (wealth < MERC_MAX_WEALTH) who fight for pay,
echoing M3.1's labor pool. Each fighter costs FIGHTER_COST (paid to the mercenary, money then
food), and the aspirant can muster at most `war_chest // FIGHTER_COST` of the nearest available
poor agents within MUSTER_RADIUS. Attacker force = the NUMBER of fighters actually mustered. So
power runs wealth -> soldiers -> force, with soldiers being REAL agents — a broke aspirant
musters nobody and conquers nothing; it is never "wealth = combat power" directly.

A FIGHT for a settlement — the outcome turns on MUSTERED FORCE, not on wealth
----------------------------------------------------------------------------
Attacker force (the bought army) is resolved against the settlement's DEFENSE:
  * a MONARCH holds it -> its standing GARRISON defends;
  * else a legitimate M3.2 LEADER holds it -> the leader's FOLLOWERS fight for it (loyalty
    becomes defensive force — this is where force and legitimacy COLLIDE);
  * else (unorganised) -> the settlement's living MEMBERS resist.
The attacker WINS iff attacker_force > defender_force (strict — a tie is held by the defender,
the incumbent's advantage; the attacker must OVERCOME the defence). The result is therefore NOT
a foregone wealth-max: a richer attacker who musters FEWER fighters than a trusted leader has
loyal followers is REPELLED, while an overwhelming bought force wins. Deterministic and seeded —
ZERO RNG (resolution and casualties are state-math over sorted names), so a fight can never
desync the seeded stream. War is COSTLY: each side loses `round(CASUALTY_RATE * opposing_force)`
fighters (capped at its own size) — real agents who DIE (via population.announce_death, the same
death path starvation uses: a battle event, survivor memories, a queued respawn). Figureheads
(the aspirant, the defending leader/monarch) are not slain — they win or lose the TITLE; the
rank-and-file pay in blood, so a defended conquest thins the defender's following (feeding back
into M3.2 next turn) and a failed assault burns the aspirant's army.

MONARCH — the institution (persistent, rule-by-FORCE, LOSABLE)
-------------------------------------------------------------
The victor becomes MONARCH: world_state["monarchs"][sid] = {monarch, since, garrison}. The
garrison is the surviving mustered army — a STANDING force that defends the crown. A monarch
rules by FORCE: it LEVIES wealth from the settlement's members WITHOUT consent — extracting
MONARCH_LEVY_RATE of each member's wealth above MONARCH_LEVY_THRESHOLD into its OWN coffers.
Contrast M3.3 sharply: M3.3 taxation needs a legitimate leader and the CONSENT of the governed,
is REDISTRIBUTIVE (rich followers -> poor followers), conserves wealth, and SELF-LIMITS through a
trust backlash; a monarch's levy needs NEITHER a leader NOR consent, is EXTRACTIVE (members ->
the crown), and has no trust check on it — domination, not government. The crown is LOSABLE: a
later, stronger aspirant attacks the garrison with the SAME mechanic and OVERTHROWS the monarch
(succession is contested, never permanent).

Monarch vs trust-leader — two roles, two sources, force takes PRECEDENCE
-----------------------------------------------------------------------
A monarch (force) and a trust-leader (consent) are DIFFERENT roles from DIFFERENT sources and can
coexist in one settlement: conquest does NOT erase the leaders[] record (the leader may still be
trusted — consent survives occupation), but the MONARCH overrides governance — it is the monarch,
not the consent-leader, who levies and rules the seized settlement. FORCE > CONSENT for control:
the dark point of M3.4 is precisely that a trusted, legitimate leader can be conquered by a bought
army and reduced to a powerless figurehead under a crown it never granted.

Cost & determinism
------------------
ZERO LLM calls and ZERO RNG. Deterministic iteration (sorted settlement ids, aspirants by wealth
then name, mercenaries by distance then name, casualties by name) so outcomes are reproducible
under seed and cannot desync the stream. Records live only in world_state["monarchs"] (no new
Agent field). A run with the institution OFF never calls `update`, so it is byte-identical to v1.
Imports world + population (one-directional), keeping the world layer dependency-free.
"""

from __future__ import annotations

from typing import Any

import economy
import population
import world

# --- Tunable constants (documented) ----------------------------------------
# FIGHTER_COST: money (food-claim) to hire and feed one fighter for the campaign. The aspirant
# pays this to each mercenary — wealth literally becomes soldiers. It is the exchange rate of the
# whole milestone: war chest / FIGHTER_COST is the ceiling on an army, so force scales with wealth.
FIGHTER_COST = 5.0

# MIN_WAR_CHEST: the wealth an aspirant needs before it can wage war at all. Set to two fighters'
# worth, so a genuinely broke agent can never muster a force or conquer anything — the floor that
# makes conquest a tool of the RICH (the M3.1 winners), not of anyone.
MIN_WAR_CHEST = 10.0

# MERC_MAX_WEALTH: only the POOR sell their swords. An agent below this wealth is desperate enough
# to fight for pay (echoing M3.1's worker pool); the comfortable do not enlist. This is what makes
# an army a transfer FROM the rich aspirant TO the poor fighters — paid violence, not conscription.
MERC_MAX_WEALTH = 5.0

# MUSTER_RADIUS / ATTACK_RADIUS: war is LOCAL. An aspirant hires mercenaries within MUSTER_RADIUS
# of itself and can only assault a settlement whose centre lies within ATTACK_RADIUS — you raise a
# force from those near you and march on a town you can reach, not across the whole map.
MUSTER_RADIUS = 5
ATTACK_RADIUS = 5

# CASUALTY_RATE: the share of the OPPOSING force each side kills, so war is destructive on both
# sides (capped at a side's own size). 0.5 -> a clash is bloody but not annihilating: a defender
# repelling a 4-strong assault loses ~2, the attacker's army is half-spent — enough that war
# COSTS real agents and a failed campaign cripples an aspirant, without wiping whole populations.
CASUALTY_RATE = 0.5

# MONARCH_LEVY_RATE / MONARCH_LEVY_THRESHOLD: the crown's EXTRACTIVE levy — MONARCH_LEVY_RATE of
# each member's wealth above the threshold, taken to the MONARCH's own coffers WITHOUT consent
# (contrast M3.3's consensual, redistributive, self-limiting tax). Rule by force needs no consent.
MONARCH_LEVY_RATE = 0.20
MONARCH_LEVY_THRESHOLD = 5.0


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _wealth(a: Any) -> float:
    """An agent's liquid wealth = money + stored food (both food-claims) — the M3.1 class metric."""
    return a.money + a.stockpile


def max_fighters(aspirant: Any) -> int:
    """How many fighters `aspirant`'s war chest could fund (0 if below the war-chest floor).

    The hard link wealth -> force: a broke aspirant (wealth < MIN_WAR_CHEST) can fund NONE and so
    can never conquer; a rich one can fund war_chest // FIGHTER_COST. Pure read (musters nothing).
    """
    if _wealth(aspirant) < MIN_WAR_CHEST:
        return 0
    return int(_wealth(aspirant) // FIGHTER_COST)


def _available_mercenaries(state: dict[str, Any], aspirant: Any, exclude: set[str]) -> list[Any]:
    """Poor living agents near `aspirant` who would fight for pay (sorted: nearest, then name).

    Mercenaries are the desperate (wealth < MERC_MAX_WEALTH) within MUSTER_RADIUS, excluding the
    aspirant itself and any name in `exclude` (e.g. the settlement's own defenders — you do not
    hire the people you are attacking). Deterministic order so the muster is reproducible.
    """
    cands = [a for a in state["agents"]
             if a.alive and a.name != aspirant.name and a.name not in exclude
             and _wealth(a) < MERC_MAX_WEALTH
             and _chebyshev(a.position, aspirant.position) <= MUSTER_RADIUS]
    return sorted(cands, key=lambda a: (_chebyshev(a.position, aspirant.position), a.name))


def muster(state: dict[str, Any], aspirant: Any, exclude: set[str]) -> list[Any]:
    """Spend the aspirant's war chest to hire fighters; return the mustered force (the soldiers).

    The aspirant pays FIGHTER_COST (money first, then stored food) to each of up to
    `max_fighters` nearest available mercenaries — wealth becomes a standing army of REAL agents.
    Each payment is a transfer (rich aspirant -> poor fighter), echoing M3.1's wage. Returns the
    list of hired agents (the attacker force). ZERO RNG.
    """
    room = max_fighters(aspirant)
    if room <= 0:
        return []
    hired: list[Any] = []
    for merc in _available_mercenaries(state, aspirant, exclude):
        if len(hired) >= room or _wealth(aspirant) < FIGHTER_COST:
            break
        economy._settle(aspirant, merc, FIGHTER_COST)  # pay the mercenary (money then food)
        world.record_memory(merc, f"Took {aspirant.name}'s coin to fight")
        hired.append(merc)
    return hired


def defenders_of(state: dict[str, Any], sid: str) -> tuple[list[Any], str]:
    """The agents who will DEFEND settlement `sid`, and the KIND of defence (force vs legitimacy).

      * a MONARCH holds it      -> its living GARRISON defends            (kind "garrison");
      * else a legitimate LEADER -> the leader's living FOLLOWERS defend   (kind "loyalty");
      * else (unorganised)       -> the settlement's living MEMBERS resist (kind "militia").
    Pure read. The figurehead (monarch/leader) is NOT counted as a combat unit — it holds or loses
    the TITLE; the rank-and-file are the force that fights and dies.
    """
    living = {a.name: a for a in state["agents"] if a.alive}
    mon = state.get("monarchs", {}).get(sid)
    if mon is not None:
        return [living[n] for n in sorted(mon["garrison"]) if n in living], "garrison"
    lead = state.get("leaders", {}).get(sid)
    if lead is not None:
        return [living[n] for n in sorted(lead["followers"]) if n in living], "loyalty"
    rec = state.get("settlements", {}).get(sid)
    if rec is not None:
        members = [living[n] for n in sorted(rec["members"])
                   if n in living and living[n].name != _holder_name(state, sid)]
        return members, "militia"
    return [], "none"


def _holder_name(state: dict[str, Any], sid: str) -> str | None:
    """The current title-holder of `sid` (monarch first, else trust-leader) — excluded from militia."""
    mon = state.get("monarchs", {}).get(sid)
    if mon is not None:
        return mon["monarch"]
    lead = state.get("leaders", {}).get(sid)
    return lead["leader"] if lead is not None else None


def _fell_in_battle(state: dict[str, Any], units: list[Any], n: int, turn: int, side: str) -> list[str]:
    """Kill the first `n` of `units` (sorted by name -> deterministic) as battle casualties.

    Routes each death through population.announce_death (battle wording), so a war death is a
    first-class civilizational event — a DEATH line, survivor memories, and a queued respawn —
    exactly like starvation, never a silent flag flip. Returns the names of the fallen.
    """
    fallen: list[str] = []
    for unit in sorted(units, key=lambda a: a.name)[:max(0, n)]:
        if not unit.alive:
            continue
        population.announce_death(unit, turn, state, cause="fell in battle",
                                  final_memory="Fell in battle",
                                  note=f"they fell in battle ({side})")
        fallen.append(unit.name)
    return fallen


def resolve_battle(state: dict[str, Any], attackers: list[Any], defenders: list[Any],
                   turn: int, att_side: str, def_side: str) -> tuple[bool, list[str], list[str], list[Any]]:
    """Resolve a clash between two mustered forces. Deterministic, ZERO RNG (M3.4, reused by M3.5).

    The attacker WINS iff its force strictly OUTNUMBERS the defenders (`att > def_`); a tie is held
    by the defender (the incumbent's advantage). Each side then loses `round(CASUALTY_RATE *
    opposing_force)` fighters (capped at its own size), slain by name order via `_fell_in_battle`
    (a real, logged death + queued respawn). Returns (won, att_dead, def_dead, survivors). Shared by
    M3.4 single-settlement conquest and M3.5 realm conquest so BOTH use the identical fight maths.
    """
    att, def_ = len(attackers), len(defenders)
    won = att > def_  # strict: the attacker must OVERCOME the defence; a tie is held by the seat
    # Casualties: each side loses round(CASUALTY_RATE * opposing_force), capped at its own size.
    att_loss = min(att, round(CASUALTY_RATE * def_))
    def_loss = min(def_, round(CASUALTY_RATE * att))
    att_dead = _fell_in_battle(state, attackers, att_loss, turn, att_side)
    def_dead = _fell_in_battle(state, defenders, def_loss, turn, def_side)
    survivors = [f for f in attackers if f.alive]
    return won, att_dead, def_dead, survivors


def attempt_conquest(state: dict[str, Any], aspirant: Any, sid: str, turn: int) -> dict[str, Any]:
    """Resolve ONE assault by `aspirant` on settlement `sid`. Deterministic (ZERO RNG), M3.4.

    Musters the aspirant's bought army, resolves it against the settlement's defenders (garrison /
    loyal followers / militia), applies casualties to BOTH sides, and — on victory — installs (or
    replaces) the monarch with the surviving army as the new garrison. Returns a result dict
    {won, attackers, defenders, kind, att_dead, def_dead, monarch} and logs the battle to events.
    """
    defenders, kind = defenders_of(state, sid)
    exclude = {d.name for d in defenders} | {aspirant.name}
    holder = _holder_name(state, sid)
    if holder is not None:
        exclude.add(holder)
    army = muster(state, aspirant, exclude)
    att, def_ = len(army), len(defenders)
    won, att_dead, def_dead, survivors = resolve_battle(
        state, army, defenders, turn, f"{aspirant.name}'s army", f"defending {sid}")

    monarchs = state.setdefault("monarchs", {})
    if won:
        deposed = monarchs.get(sid, {}).get("monarch")
        monarchs[sid] = {"monarch": aspirant.name, "since": turn,
                         "garrison": {f.name for f in survivors}}
        if deposed is not None and deposed != aspirant.name:
            verb = f"OVERTHREW {deposed} and seized"
        elif kind == "loyalty":
            verb = f"CONQUERED (defeating {holder}'s followers in)"
        else:
            verb = "seized"
        state["events"].append(
            f"turn {turn}: {aspirant.name} {verb} {sid} by force "
            f"({att} fighters vs {def_} defenders; {len(att_dead)}+{len(def_dead)} fell) "
            f"-> MONARCH of {sid}")
        world.record_memory(aspirant, f"Became MONARCH of {sid} by force ({att} vs {def_})")
    else:
        # Repelled. If a monarch was defending, its garrison shrinks to the survivors.
        if kind == "garrison":
            monarchs[sid]["garrison"] = {d.name for d in defenders if d.alive}
        state["events"].append(
            f"turn {turn}: {aspirant.name}'s assault on {sid} was REPELLED "
            f"({att} fighters vs {def_} defenders; {len(att_dead)}+{len(def_dead)} fell)"
            f"{f' — {kind} held' if kind != 'none' else ''}")
        world.record_memory(aspirant, f"Failed to take {sid} ({att} vs {def_})")

    return {"won": won, "attackers": att, "defenders": def_, "kind": kind,
            "att_dead": att_dead, "def_dead": def_dead,
            "monarch": monarchs.get(sid, {}).get("monarch")}


def levy(state: dict[str, Any], turn: int) -> list[str]:
    """Each monarch EXTRACTS wealth from its settlement's members — rule by FORCE, no consent.

    For every settlement with a monarch, take MONARCH_LEVY_RATE of each member's wealth ABOVE
    MONARCH_LEVY_THRESHOLD (money then food) into the MONARCH's own coffers. Unlike M3.3 this needs
    no legitimate leader and no consent, and is EXTRACTIVE (to the crown) rather than redistributive
    — domination. Returns the events logged. ZERO RNG.
    """
    events: list[str] = []
    living = {a.name: a for a in state["agents"] if a.alive}
    for sid in sorted(state.get("monarchs", {})):
        mon = state["monarchs"][sid]
        crown = living.get(mon["monarch"])
        rec = state.get("settlements", {}).get(sid)
        if crown is None or rec is None:
            continue
        taken = 0.0
        for name in sorted(rec["members"]):
            subject = living.get(name)
            if subject is None or subject.name == crown.name:
                continue
            due = MONARCH_LEVY_RATE * max(0.0, _wealth(subject) - MONARCH_LEVY_THRESHOLD)
            if due <= 0:
                continue
            economy._settle(subject, crown, due)  # subject -> crown (money then food)
            taken += due
        if taken > 0:
            events.append(
                f"turn {turn}: MONARCH {crown.name} levied {taken:.1f} from {sid} by force "
                f"(no consent)")
            world.record_memory(crown, f"Levied {taken:.1f} from {sid} by force")
    state["events"].extend(events)
    return events


def fieldable_force(state: dict[str, Any], aspirant: Any, sid: str) -> int:
    """How many fighters `aspirant` could ACTUALLY muster against `sid` right now (a dry run).

    The lesser of what its war chest funds (`max_fighters`) and how many mercenaries are in range
    (excluding the settlement's own defenders/holder). Lets the loop launch only WINNABLE assaults
    — a rational commander raises an army to win, not to bleed — without paying or moving anyone.
    """
    defenders, _ = defenders_of(state, sid)
    exclude = {d.name for d in defenders} | {aspirant.name}
    holder = _holder_name(state, sid)
    if holder is not None:
        exclude.add(holder)
    return min(max_fighters(aspirant), len(_available_mercenaries(state, aspirant, exclude)))


def _eligible_aspirants(state: dict[str, Any], sid: str) -> list[Any]:
    """Living agents who could assault `sid` this turn: a war chest, in range, not its holder.

    Sorted by wealth (desc) then name so the STRONGEST contender attacks first and the loop is
    deterministic. Pure read.
    """
    rec = state["settlements"][sid]
    center = rec["center"]
    holder = _holder_name(state, sid)
    members = rec["members"]
    aspirants = [a for a in state["agents"]
                 if a.alive and a.name != holder and max_fighters(a) > 0
                 and a.name not in members  # an outsider marches on the town (no internal coup yet)
                 and _chebyshev(a.position, center) <= ATTACK_RADIUS]
    return sorted(aspirants, key=lambda a: (-_wealth(a), a.name))


def update(state: dict[str, Any], turn: int) -> list[str]:
    """Advance the conquest/monarchy institution one turn (ZERO LLM, ZERO RNG, M3.4).

    Two deterministic stages over sorted settlement ids: (1) every standing MONARCH levies its
    settlement by force; (2) the single strongest eligible ASPIRANT near each settlement attempts
    ONE conquest — mustering a bought army and fighting the defenders (garrison / loyal followers /
    militia), which may install, overthrow, or fail to take the crown, at the cost of real lives.
    Returns the turn's events (levy + battles). Caller gates on `monarchy_on`, so an off run never
    calls this and stays byte-identical to v1.
    """
    before = len(state["events"])              # both levy() and attempt_conquest() append here
    levy(state, turn)
    for sid in sorted(state.get("settlements", {})):
        defenders, _ = defenders_of(state, sid)
        # The strongest aspirant who could ACTUALLY field a winning force assaults (one fight per
        # settlement per turn). A rational commander does not march an army it knows is too small,
        # so an aspirant who cannot out-muster the defence bides its time — no suicidal spam. This
        # is a LOOP policy only; attempt_conquest itself happily resolves any matchup (so the
        # verify/tests can stage a doomed assault directly and watch loyalty repel it).
        for aspirant in _eligible_aspirants(state, sid):
            if fieldable_force(state, aspirant, sid) > len(defenders):
                attempt_conquest(state, aspirant, sid, turn)
                break
    return state["events"][before:]
