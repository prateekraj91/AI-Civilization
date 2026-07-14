"""
kingdoms.py
===========

KINGDOMS & VASSALAGE — feudalism (V2 milestone M3.5, Phase 3: Institutions). On top of M3.4
(conquest & monarchy), M3.3 (taxation), M3.2 (leadership), M3.1 (wage labor) and all of Phase 0/1/2.

The historical step M3.5 makes — single settlements become FEUDAL KINGDOMS
--------------------------------------------------------------------------
M3.4 built a MONARCH who seizes ONE settlement by force. M3.5 is the SCALE-UP: a monarch conquers
NEIGHBOURING settlements to build a multi-settlement KINGDOM, structured as a FEUDAL HIERARCHY — a
KING atop VASSAL LORDS who rule their own settlements on the king's behalf. The model is faithful
feudalism: the vassal bargain is TRIBUTE + military SERVICE flowing UP, PROTECTION + local
AUTONOMY flowing DOWN, and loyalty is CONDITIONAL — it can erode under a grasping overlord and a
pushed vassal can BREAK AWAY. That conditional loyalty is what makes royal power CONTINGENT (on
keeping vassals loyal) rather than absolute, and it is what makes the institution generative.

Scope (held strictly; boundaries stated so later milestones own the rest)
-------------------------------------------------------------------------
IN scope: kingdom FORMATION by conquest of a neighbour, the two-level vassal HIERARCHY
(king -> vassal lords -> their settlements), the tribute/service BARGAIN, CONDITIONAL loyalty,
and BREAKAWAY (a sufficiently disloyal vassal LEAVING the realm). OUT of scope (deferred): baronial
CIVIL WAR between vassals, vassal DEFECTION to a RIVAL king (needs M3.6 inter-kingdom war), and
full organised REBELLION cascades. Breakaway is the only disloyalty payoff M3.5 builds; the richer
ones come later.

The feudal model (emerges from M3.4 conquest + M3.2 trust; zero LLM, zero RNG, state math)
-----------------------------------------------------------------------------------------
1. FORMATION BY CONQUEST (reuses M3.4). A monarch (M3.4) can conquer a NEIGHBOURING settlement
   (`conquer_neighbour`) — the M3.4 fight aimed OUTWARD at an adjacent settlement, resolved with
   the SAME `monarchy.resolve_battle`. Winning brings that settlement into the conqueror's REALM. A
   much-weaker settlement that cannot possibly win MAY SUBMIT without a fight (a host outnumbering
   the defence by >= SUBMIT_RATIO is accepted peacefully — no needless bloodshed), a special case
   of conquest the defender can't win. State: world_state["kingdoms"][king] = {king, home,
   settlements, vassals, founded, discontent}.

2. THE VASSAL HIERARCHY (what makes it a kingdom, not a big settlement). A conquered settlement
   KEEPS its local ruler — its M3.2 trust-leader or M3.4 local monarch — who becomes a VASSAL LORD
   ruling that settlement ON THE KING'S BEHALF (a two-level structure: KING -> VASSAL LORDS ->
   their settlements; the local `monarchs`/`leaders` record is NOT erased, preserving local
   autonomy). If the conquered settlement had NO local ruler, the king holds it DIRECTLY (installed
   as its monarch with the surviving host as garrison) — a directly-held settlement, not a vassalage.

3. THE FEUDAL BARGAIN (up: tribute + service; down: protection + autonomy).
   * TRIBUTE cascades UP the levels (`tribute`): a vassal's settlement members are levied
     TRIBUTE_RATE of their wealth above a threshold by the vassal (members -> vassal), and a
     KING_SHARE of THAT take then flows up to the king (vassal -> king). So a large realm
     concentrates wealth at the crown THROUGH its vassals, conserving total wealth (it only moves).
   * MILITARY SERVICE (`muster_realm`): a king's host = its own bought fighters PLUS every LOYAL
     vassal's mustered fighters (funded by the wealth tribute fattened). So realm military strength
     = the SUM of loyal vassals' forces — this is what makes kingdoms strong (and sets up M3.6). A
     disloyal or broken-away vassal answers no call.
   * DOWNWARD the vassal keeps LOCAL AUTONOMY (still runs its settlement) and PROTECTION (it is
     part of the realm, defended by the king's host).

4. CONDITIONAL LOYALTY (the generative feudal tension — ties to M3.2/M3.3 trust). A vassal's
   loyalty IS its trust in the king (reusing the v1 trust system). On conquest the defeated lord
   swears FEALTY (trust seeded to LOYAL_TRUST). HEAVY royal demands then ERODE it: when the king's
   share exceeds KING_CONSENT each vassal withdraws round((rate - KING_CONSENT) * RESENT_SCALE)
   trust per turn (exactly M3.3's tax-backlash shape, now aimed at the crown). A vassal whose trust
   falls to/below BREAKAWAY_TRUST for BREAKAWAY_PATIENCE consecutive turns (HYSTERESIS — no
   single-turn flip; recovering loyalty resets the counter) BREAKS AWAY: it leaves the realm with
   its settlement, which becomes independent again. A fairly-treated vassal (a king whose share
   stays within the consent band draws no resentment) keeps its fealty and stays. Royal power is
   thus CONTINGENT on not over-grasping — tyranny over vassals is self-limiting, exactly as M3.3's
   tyranny over subjects was.

5. TERRITORIAL COMPOUNDING. Pooled tribute funds the king's further conquest -> realms grow ->
   the map CONSOLIDATES (many small settlements -> fewer, larger kingdoms). Emergent, seed-varying,
   never scripted (the loop in `update` only launches WINNABLE conquests, like M3.4's guard).

Monarch / leader / king / vassal — how the roles compose
--------------------------------------------------------
`monarchs[sid]` / `leaders[sid]` still record who holds each settlement LOCALLY (M3.4 force / M3.2
consent). `kingdoms[king]` is the REALM overlay on top: which settlements form one realm under a
king and who their vassal lords are. A vassal lord is whatever it already was locally (a trust-leader
or a local monarch) PLUS now sworn to a king. The king is itself a monarch of its home seat.

Cost & determinism
------------------
ZERO LLM calls and ZERO RNG. Deterministic iteration (sorted king names, sorted settlement ids,
sorted member/vassal names; the fight reuses `monarchy.resolve_battle`, also RNG-free). Trust writes
go through the existing logged `trust.adjust_trust` (the M3.3 path). Records live only in
world_state["kingdoms"] (no new Agent field). A run with the institution OFF never calls `update`
(kingdoms stays empty), so it is byte-identical to v1. Imports world + monarchy + economy + trust
(one-directional), keeping the world layer dependency-free.
"""

from __future__ import annotations

from typing import Any

import economy
import monarchy
import trust
import world

# --- Tunable constants (documented) ----------------------------------------
# TRIBUTE_RATE: the share of a settlement member's wealth ABOVE TRIBUTE_THRESHOLD a VASSAL LORD
# levies from its settlement each turn (members -> vassal). The bottom level of the cascade; mirrors
# M3.4's monarch levy (same rate/threshold) — a vassal taxes its own town exactly as a local monarch
# would, the difference being it now owes a share upward.
TRIBUTE_RATE = monarchy.MONARCH_LEVY_RATE          # 0.20
TRIBUTE_THRESHOLD = monarchy.MONARCH_LEVY_THRESHOLD  # 5.0

# DEFAULT_KING_SHARE: the share of a vassal's fresh tribute that cascades UP to the KING (vassal ->
# king) when a run does not name one. 0.25 sits under KING_CONSENT, so the default crown is a
# MODERATE overlord — sustainable, its vassals stay loyal. A run sets world_state["tribute_rate"]
# to push it (e.g. 0.9) and watch loyalty erode. This is the M3.5 analogue of M3.3's tax_rate lever.
DEFAULT_KING_SHARE = 0.25

# KING_CONSENT / RESENT_SCALE: the consent band on ROYAL demands and the erosion past it — the same
# shape as M3.3's tax backlash, now between vassal and king. A king's share within KING_CONSENT
# draws NO resentment (a fair overlord is tolerated); above it each vassal withdraws
# round((rate - KING_CONSENT) * RESENT_SCALE) trust per turn. Tied to M3.3's values so feudal and
# civic over-extraction punish identically.
KING_CONSENT = 0.35
RESENT_SCALE = 4.0

# LOYAL_TRUST: the trust a vassal must hold in its king to be LOYAL — to answer the military muster
# and to count as content. Tied to trust.HIGH_THRESHOLD so "loyal" reads exactly as the v1 trust
# system's "high". Fealty on conquest seeds trust to exactly this.
LOYAL_TRUST = trust.HIGH_THRESHOLD                 # 2

# BREAKAWAY_TRUST / BREAKAWAY_PATIENCE: a vassal whose trust in the king falls to/below
# BREAKAWAY_TRUST (trust.LOW_THRESHOLD — the v1 "low" bar) for BREAKAWAY_PATIENCE CONSECUTIVE turns
# BREAKS AWAY. The patience counter is the HYSTERESIS: a single bad turn never flips a vassal out of
# the realm, and a turn of recovered loyalty resets it — so breakaway is sustained disaffection, not
# a wobble. 2 turns below the floor is enough to read as a settled refusal of the crown.
BREAKAWAY_TRUST = trust.LOW_THRESHOLD              # -2
BREAKAWAY_PATIENCE = 2

# KINGDOM_REACH / SUBMIT_RATIO: war is LOCAL (as in M3.4). A king can only march on a settlement
# whose centre lies within KINGDOM_REACH of its home seat — realms grow into the NEIGHBOURHOOD, not
# across the map. A defence outnumbered by the royal host by >= SUBMIT_RATIO submits WITHOUT a fight
# (a hopeless stand yields rather than bleed) — the bloodless special case of conquest.
KINGDOM_REACH = 8
SUBMIT_RATIO = 3


def _wealth(a: Any) -> float:
    """An agent's liquid wealth = money + stored food (both food-claims) — the M3.1 class metric."""
    return a.money + a.stockpile


def _find(state: dict[str, Any], name: str | None) -> Any | None:
    """The living agent called `name`, or None (small linear scan; mirrors monarchy/leadership)."""
    if name is None:
        return None
    for a in state["agents"]:
        if a.alive and a.name == name:
            return a
    return None


def _trust_in(vassal: Any, king_name: str) -> int:
    """A vassal's LOYALTY = its trust in the king — a pure read of the v1 trust network."""
    return vassal.relationships.get(king_name, {}).get("trust", 0)


def realm_of(state: dict[str, Any], sid: str) -> str | None:
    """The king whose realm contains settlement `sid`, or None if it is independent. Pure read."""
    for king in sorted(state.get("kingdoms", {})):
        if sid in state["kingdoms"][king]["settlements"]:
            return king
    return None


def _king_home(state: dict[str, Any], king_name: str) -> str | None:
    """The king's directly-held seat: the settlement it is MONARCH of (M3.4), else its own town."""
    for sid in sorted(state.get("monarchs", {})):
        if state["monarchs"][sid]["monarch"] == king_name:
            return sid
    king = _find(state, king_name)
    return getattr(king, "settlement", None) if king is not None else None


def _ensure_kingdom(state: dict[str, Any], king_name: str, turn: int) -> dict[str, Any]:
    """Return king's realm record, creating it (seeded with the home seat) on first conquest."""
    kingdoms = state.setdefault("kingdoms", {})
    rec = kingdoms.get(king_name)
    if rec is None:
        home = _king_home(state, king_name)
        rec = {"king": king_name, "home": home,
               "settlements": {home} if home is not None else set(),
               "vassals": {}, "founded": turn, "discontent": {}}
        kingdoms[king_name] = rec
    return rec


# --- Military service: the realm host = king's force + loyal vassals' forces ---
def muster_realm(state: dict[str, Any], king: Any, exclude: set[str]) -> list[Any]:
    """Raise the KING's host: its own bought fighters PLUS every LOYAL vassal's mustered fighters.

    The king musters from its own war chest (monarchy.muster), then CALLS its vassals: each vassal
    lord still in the realm AND loyal (trust in king >= LOYAL_TRUST) answers by mustering fighters
    from its own (tribute-fattened) wealth, which join the royal host. A disloyal vassal withholds
    service; a broken-away one is no longer in the realm at all. So realm strength = the king's force
    + the SUM of loyal vassals' forces. Returns the combined host (real agents). ZERO RNG.
    """
    host = monarchy.muster(state, king, exclude)
    taken = exclude | {f.name for f in host}
    rec = state.get("kingdoms", {}).get(king.name)
    if rec is not None:
        for sid in sorted(rec["vassals"]):
            vassal = _find(state, rec["vassals"][sid])
            if vassal is None or _trust_in(vassal, king.name) < LOYAL_TRUST:
                continue  # a disloyal/absent vassal answers no muster (military SERVICE is conditional)
            contingent = monarchy.muster(state, vassal, taken | {vassal.name})
            taken |= {f.name for f in contingent}
            host.extend(contingent)
    return host


def realm_host_size(state: dict[str, Any], king: Any) -> int:
    """A dry-run count of the host the king COULD field now (king + loyal vassals), paying no one.

    The lesser, per contributor, of what its war chest funds and how many free mercenaries are in
    range — so the loop launches only WINNABLE realm conquests (a rational crown does not march a
    host it knows is too small). Pure read; mirrors monarchy.fieldable_force at the realm level.
    """
    size = min(monarchy.max_fighters(king), len(monarchy._available_mercenaries(state, king, {king.name})))
    rec = state.get("kingdoms", {}).get(king.name)
    if rec is not None:
        for sid in sorted(rec["vassals"]):
            vassal = _find(state, rec["vassals"][sid])
            if vassal is None or _trust_in(vassal, king.name) < LOYAL_TRUST:
                continue
            size += min(monarchy.max_fighters(vassal),
                        len(monarchy._available_mercenaries(state, vassal, {vassal.name})))
    return size


# --- Tribute: members -> vassal -> king (cascades up, conserves wealth) -----
def tribute(state: dict[str, Any], turn: int) -> list[str]:
    """Run the feudal tribute cascade for every realm: members -> vassal -> king. ZERO RNG, M3.5.

    For each kingdom (sorted), and each vassal settlement (sorted): the VASSAL LORD levies
    TRIBUTE_RATE of every member's wealth above TRIBUTE_THRESHOLD (members -> vassal), then a
    KING_SHARE of THAT fresh take cascades up to the king (vassal -> king). Wealth is only moved
    (economy._settle), so the total is CONSERVED across the cascade. Heavy royal demands then write
    the loyalty BACKLASH through the trust system (the M3.3 path): a king whose share exceeds
    KING_CONSENT loses each vassal's trust by round((rate - KING_CONSENT) * RESENT_SCALE) per turn —
    the erosion that makes loyalty conditional. Returns the events logged.
    """
    rate = state.get("tribute_rate", DEFAULT_KING_SHARE)
    living = {a.name: a for a in state["agents"] if a.alive}
    resent = -round(max(0.0, rate - KING_CONSENT) * RESENT_SCALE)
    events: list[str] = []

    for king_name in sorted(state.get("kingdoms", {})):
        rec = state["kingdoms"][king_name]
        king = living.get(king_name)
        if king is None:
            continue
        for sid in sorted(rec["vassals"]):
            vassal = living.get(rec["vassals"][sid])
            srec = state.get("settlements", {}).get(sid)
            if vassal is None or srec is None:
                continue
            # Level 1 — members -> vassal (the vassal levies its own settlement).
            gross = 0.0
            for name in sorted(srec["members"]):
                m = living.get(name)
                if m is None or m.name in (vassal.name, king_name):
                    continue
                due = TRIBUTE_RATE * max(0.0, _wealth(m) - TRIBUTE_THRESHOLD)
                if due <= 0:
                    continue
                economy._settle(m, vassal, due)
                gross += due
            # Level 2 — vassal -> king (a share of THAT take cascades up the hierarchy).
            up = rate * gross
            if up > 0:
                economy._settle(vassal, king, up)
                events.append(
                    f"turn {turn}: tribute cascaded up {sid}: {gross:.1f} members->{vassal.name}, "
                    f"{up:.1f} {vassal.name}->KING {king_name}")
            # Backlash — a grasping crown erodes its vassals' loyalty (writes trust, M3.3 shape).
            if resent < 0:
                trust.adjust_trust(vassal, king_name, resent, "heavy royal tribute", turn, state)

    state["events"].extend(events)
    return events


def secede_settlement(state: dict[str, Any], sid: str, turn: int, reason: str) -> str | None:
    """Detach settlement `sid` from whatever realm contains it (the realm loses it). Returns the
    king it left, or None if `sid` was independent. ZERO RNG.

    Factored out of the M3.5 breakaway path (`_check_breakaways`) so any cause that frees a
    settlement — a vassal's loyalty collapsing (M3.5) OR the people rising against their lord
    (M4.5 uprising) — cuts the realm tie through the SAME machinery: the settlement leaves the
    king's `settlements`/`vassals`, its breakaway-hysteresis counter is dropped, a realm left with
    no settlements is dissolved, and the event is logged. The LOCAL ruler record (monarch/leader)
    is NOT touched here — this cuts only the realm tie; the caller owns the local seat.
    """
    king_name = realm_of(state, sid)
    if king_name is None:
        return None
    rec = state["kingdoms"][king_name]
    lord = rec["vassals"].pop(sid, None)
    rec["settlements"].discard(sid)
    if lord is not None:
        rec["discontent"].pop(lord, None)
    state["events"].append(
        f"turn {turn}: {sid} SECEDED from {king_name}'s realm ({reason}) — independent again")
    if not rec["settlements"]:
        state["kingdoms"].pop(king_name)
    return king_name


# --- Conditional loyalty: a pushed vassal breaks away (with hysteresis) -----
def _check_breakaways(state: dict[str, Any], turn: int) -> list[str]:
    """Drop vassals whose loyalty has collapsed for BREAKAWAY_PATIENCE consecutive turns. M3.5.

    A vassal whose trust in the king is <= BREAKAWAY_TRUST raises a discontent counter; once it
    reaches BREAKAWAY_PATIENCE the vassal BREAKS AWAY — its settlement leaves the realm and becomes
    independent again (the local ruler record stays; only the realm tie is cut). Loyalty at/above the
    floor RESETS the counter (hysteresis both ways). Returns the events logged.
    """
    events: list[str] = []
    for king_name in sorted(state.get("kingdoms", {})):
        rec = state["kingdoms"][king_name]
        king = _find(state, king_name)
        for sid in sorted(rec["vassals"]):
            vassal = _find(state, rec["vassals"][sid])
            disc = rec["discontent"]
            if vassal is not None and king is not None and _trust_in(vassal, king_name) > BREAKAWAY_TRUST:
                disc[vassal.name] = 0  # loyalty holds (or recovered) — reset the hysteresis counter
                continue
            # A vassal whose lord has died/vanished cannot hold the seat for the crown -> also breaks.
            name = rec["vassals"][sid]
            disc[name] = disc.get(name, 0) + 1
            if disc[name] < BREAKAWAY_PATIENCE:
                continue
            rec["vassals"].pop(sid)
            rec["settlements"].discard(sid)
            disc.pop(name, None)
            events.append(
                f"turn {turn}: {name} BROKE AWAY from {king_name}'s realm — {sid} is independent again "
                f"(loyalty collapsed)")
            v = _find(state, name)
            if v is not None:
                world.record_memory(v, f"Broke away from {king_name}'s realm, freeing {sid}")
        if not rec["settlements"]:
            state["kingdoms"].pop(king_name)
    state["events"].extend(events)
    return events


# --- Formation / growth: a king conquers a neighbouring settlement ----------
def _incorporate(state: dict[str, Any], king: Any, sid: str, holder: str | None,
                 survivors: list[Any], turn: int) -> None:
    """Bring conquered settlement `sid` into king's realm: vassalise its lord, or hold it directly."""
    rec = _ensure_kingdom(state, king.name, turn)
    rec["settlements"].add(sid)
    if holder is not None and holder != king.name:
        # The defeated local ruler swears FEALTY and rules on as a VASSAL LORD (local autonomy kept).
        rec["vassals"][sid] = holder
        rec["discontent"][holder] = 0
        vassal = _find(state, holder)
        if vassal is not None:
            cur = _trust_in(vassal, king.name)
            trust.adjust_trust(vassal, king.name, LOYAL_TRUST - cur,
                               "swore fealty after conquest", turn, state)
    else:
        # No local ruler -> the king holds it DIRECTLY (installed as its monarch, host as garrison).
        state.setdefault("monarchs", {})[sid] = {
            "monarch": king.name, "since": turn, "garrison": {f.name for f in survivors}}


def conquer_neighbour(state: dict[str, Any], king_name: str, sid: str, turn: int) -> dict[str, Any]:
    """King `king_name` attempts to conquer NEIGHBOURING settlement `sid` into its realm. M3.5.

    Raises the realm host (own force + loyal vassals — `muster_realm`) and resolves it against the
    target's defenders (M3.4 `defenders_of`: garrison / loyal followers / militia). A hopeless
    defence (outnumbered by >= SUBMIT_RATIO) SUBMITS without a fight; otherwise the SAME
    `monarchy.resolve_battle` decides it, with real casualties. On victory the settlement is
    incorporated (`_incorporate`: its local ruler becomes a vassal, or the king holds it directly).
    Returns {won, submitted, host, defenders, kind, att_dead, def_dead, king, vassal}. ZERO RNG.
    """
    king = _find(state, king_name)
    if king is None:
        return {"won": False, "submitted": False, "host": 0, "defenders": 0, "kind": "none"}
    defenders, kind = monarchy.defenders_of(state, sid)
    exclude = {d.name for d in defenders} | {king.name}
    holder = monarchy._holder_name(state, sid)
    if holder is not None:
        exclude.add(holder)
    host = muster_realm(state, king, exclude)
    n_host, n_def = len(host), len(defenders)

    submitted = n_host > n_def and n_host >= SUBMIT_RATIO * max(1, n_def)
    if submitted:
        won, att_dead, def_dead, survivors = True, [], [], host  # a hopeless defence yields, no blood
    else:
        won, att_dead, def_dead, survivors = monarchy.resolve_battle(
            state, host, defenders, turn, f"{king_name}'s royal host", f"defending {sid}")

    if won:
        _incorporate(state, king, sid, holder, survivors, turn)
        rec = state["kingdoms"][king_name]
        verb = "accepted the submission of" if submitted else "CONQUERED"
        vassal = rec["vassals"].get(sid)
        held = f"vassal {vassal}" if vassal is not None else "held directly"
        state["events"].append(
            f"turn {turn}: KING {king_name} {verb} {sid} into the realm "
            f"({n_host} host vs {n_def} defenders; {len(att_dead)}+{len(def_dead)} fell) -> {held}; "
            f"realm now {len(rec['settlements'])} settlements")
        world.record_memory(king, f"Brought {sid} into the realm ({held})")
    else:
        state["events"].append(
            f"turn {turn}: KING {king_name}'s host was REPELLED at {sid} "
            f"({n_host} host vs {n_def} defenders; {len(att_dead)}+{len(def_dead)} fell)")

    return {"won": won, "submitted": submitted, "host": n_host, "defenders": n_def, "kind": kind,
            "att_dead": att_dead, "def_dead": def_dead, "king": king_name,
            "vassal": state.get("kingdoms", {}).get(king_name, {}).get("vassals", {}).get(sid)}


def _neighbours(state: dict[str, Any], home_sid: str) -> list[str]:
    """Independent settlements within KINGDOM_REACH of `home_sid`'s centre (sorted ids). Pure read."""
    sets = state.get("settlements", {})
    home = sets.get(home_sid)
    if home is None:
        return []
    out = []
    for sid in sorted(sets):
        if sid == home_sid or realm_of(state, sid) is not None:
            continue
        if monarchy._chebyshev(home["center"], sets[sid]["center"]) <= KINGDOM_REACH:
            out.append(sid)
    return out


def update(state: dict[str, Any], turn: int) -> list[str]:
    """Advance the kingdoms/vassalage institution one turn (ZERO LLM, ZERO RNG, M3.5).

    Three deterministic stages: (1) the tribute cascade (members -> vassals -> king) plus the
    loyalty backlash from a grasping crown; (2) BREAKAWAYS — vassals whose loyalty has collapsed for
    BREAKAWAY_PATIENCE turns leave the realm; (3) FORMATION/GROWTH — every monarch (a king-in-waiting)
    marches its realm host on the nearest INDEPENDENT neighbour it can actually out-field, bringing
    it in as a vassalage (or directly). Caller gates on `kingdoms_on`, so an off run never calls this
    and stays byte-identical to v1.
    """
    before = len(state["events"])
    tribute(state, turn)
    _check_breakaways(state, turn)

    # Formation/growth: a monarch (M3.4 seat) is a king-in-waiting. Living monarchs only; strongest
    # realms expand first (host size desc, then name) so growth is deterministic. One conquest per
    # king/turn. NOTE: like M3.4's loop, only WINNABLE conquests launch (no suicidal spam); the
    # verify/tests can still drive conquer_neighbour directly for a staged matchup.
    kings = [m["monarch"] for s in sorted(state.get("monarchs", {}))
             for m in [state["monarchs"][s]] if _find(state, m["monarch"]) is not None]
    for king_name in sorted(set(kings), key=lambda k: (-realm_host_size(state, _find(state, k)), k)):
        king = _find(state, king_name)
        home = _king_home(state, king_name)
        if home is None:
            continue
        # M4.3 REGENCY: a DEPENDENT child-king holds the realm but launches no conquest —
        # the existing is_dependent_child gate keeps its war powers dormant until it comes
        # of age (a no-op when lineage is off, so byte-identical there).
        if world.is_dependent_child(king, state):
            continue
        for sid in _neighbours(state, home):
            defenders, _ = monarchy.defenders_of(state, sid)
            if realm_host_size(state, king) > len(defenders):
                conquer_neighbour(state, king_name, sid, turn)
                break
    return state["events"][before:]
