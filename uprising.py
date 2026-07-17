"""
uprising.py
===========

UPRISING — the revolt FIRES (V2 milestone M4.5), and THE REVOLUTIONARY takes power by consent
(V2 milestone M4.6, which CLOSES Arc 2: Revolt & Class Conflict). On top of M4.4 (the discontent
GAUGE), all of Arc 1 (M4.1 lineage, M4.2 inheritance, M4.3 dynasties) and all of Phases 0-3 (labor,
leadership, taxation, monarchy, kingdoms, empire).

M4.6 — THE REVOLUTIONARY (the section at `_pick_revolutionary`/`_seed_leadership` below): M4.5 leaves a
won rising's seat VACANT; M4.6 fills it with the mob's own leader, DERIVED from the risers (angriest +
most trusted by his fellow risers) and legitimised by CONSENT through the UNCHANGED M3.2 leadership path —
power seized by force, then held only while trusted. This closes the cycle consent -> force -> revolt ->
consent, and produces the project's second emergent great-figure: the REVOLUTIONARY, born of grievance
(where M3.4's conqueror was born of wealth). See that section's block comment for the full design.

The historical step M4.5 makes — the pressure engine gets its RELIEF VALVE
-------------------------------------------------------------------------
Phase 3 built a pressure engine and M4.4 gave it a legible GAUGE, but the gauge only MEASURED —
the oppressed still just starved and seethed. M4.5 makes the gauge BLOW: when a settlement's
resentful poor become a MAJORITY carrying enough aggregate grievance, they RISE against their
force-based ruler. The mob's weapon is NUMBERS — unpaid, unarmed risers — the exact inverse of
M3.4 conquest, where force had to be BOUGHT. A rich tyrant BUYS guards and crushes the rising; a
tyrant whose treasury the mob (or his own wars) has drained falls to sheer numbers. Wealth is the
counter-revolutionary weapon.

SCOPE — M4.5 IS THE UPRISING EVENT, and ONLY that. It ends with the ruler DEPOSED and the seat
VACANT. WHO leads the freed settlement afterward — a rising's leader becoming a legitimate
trust-based ruler (the REVOLUTIONARY FIGURE) — is M4.6, NOT built here (stated as a boundary in
`_depose`). For now the force-title simply clears and the settlement is left free.

Only a FORCE ruler is a valid target
------------------------------------
A rising fires ONLY against a MONARCH (M3.4) or a VASSAL LORD (M3.5) — a ruler who extracts by
FORCE (levy/tribute) without consent. A settlement whose only authority is a consent-based
TRUST-LEADER (M3.2) is NEVER a target: the people CHOSE them, so there is nothing to overthrow.
The felt oppressor is resolved exactly as M4.4's `discontent._ruler_and_due` sees it — the mob
rises against precisely whoever the gauge says is extracting from them — so a consent-led town is
revolt-immune BY CONSTRUCTION (and M4.4 already shows fair rule generates ~zero resentment anyway
— belt and braces).

The trigger — a resentful MAJORITY carrying real weight (both gates documented, tunable)
----------------------------------------------------------------------------------------
An uprising fires in a settlement when BOTH hold: (1) the RESENTFUL faction (M4.4's
above-threshold members, ex-ruler, ex-children) is at least `UPRISING_FRACTION` of the non-ruler
members — a majority, not an absolute count, so scale doesn't distort it; AND (2) their AGGREGATE
discontent clears `UPRISING_MIN_PRESSURE` — so a tiny village of two mildly-annoyed people cannot
"revolt". A settlement that just failed a rising is under a FEAR cooldown (`UPRISING_COOLDOWN`
turns) and cannot rise again until it lapses — a crushed people do not rise every turn.

The battle — reuse `monarchy.resolve_battle`, INVERTED
------------------------------------------------------
THE MOB's force = the COUNT of risers (the resentful members themselves) — numbers, no coin.
THE RULER's defence = his funded force: the existing GARRISON / loyal FOLLOWERS (`defenders_of`)
PLUS any mercenaries he can still MUSTER from his war chest (`monarchy.muster`). The SAME shared
battle maths decides it (strict `>`, casualties BOTH sides, every death through
`population.announce_death` — which now also fires M4.2 inheritance and M4.3 succession, so the
composition is handled explicitly below).

Outcomes
--------
CRUSHED (mob loses): deaths on both sides (already dealt by the battle); the SURVIVING risers'
discontent is partially reset by FEAR (`FEAR_RETAIN`) but NOT to zero — M4.4's slow decay means
the grievance persists and can rise again; a fear cooldown is set; the failed rising is logged.
VICTORIOUS (mob wins): (a) EXPROPRIATION FIRST — the deposed ruler's hoard is SEIZED and split
equally among the surviving risers (conserved to the decimal). This is what makes it a CLASS
action, not a murder, and it INTERRUPTS INHERITANCE: because the seizure ZEROES the ruler's wealth
BEFORE `announce_death` runs, M4.2's `settle_estate` finds an empty estate and the ruler's heirs
get NOTHING — the wealth went to the peasants. (b) DEPOSE — the ruler's force-title records for
this settlement are CLEARED and the settlement SECEDES from any realm (the existing M3.5 machinery,
`kingdoms.secede_settlement`); with the title gone, M4.3's `succeed_titles` finds nothing to pass,
so no heir inherits the crown — the seat is left VACANT. The ruler is then killed through the
normal death path. (c) The victorious risers' discontent drops SHARPLY (the grievance is answered).

Cost & determinism
------------------
ZERO LLM and ZERO new RNG — deterministic over a stable (sorted-by-name) iteration, reusing the
RNG-free shared battle maths. Cooldowns live in `world_state["uprising_cooldowns"]` (no new Agent
field). A run with the system OFF never calls `update`, so it is byte-identical to v1 (the v1
golden master included). Imports discontent + monarchy + kingdoms + population + world
(one-directional — this is the highest institutional layer, consuming the gauge M4.4 exposes).
"""

from __future__ import annotations

from typing import Any

import discontent
import kingdoms
import leadership
import monarchy
import population
import trust
import world

# --- Tunable constants (documented) ----------------------------------------
# UPRISING_FRACTION: the share of a settlement's NON-RULER members who must be RESENTFUL (M4.4
# above-threshold) for a rising to be possible — a resentful MAJORITY, not an absolute count, so a
# big town and a small one need the same PROPORTION aggrieved. At >= 0.5 at least half the commoners
# must have crossed into resentment.
UPRISING_FRACTION = 0.5

# UPRISING_MIN_PRESSURE: the floor the resentful faction's AGGREGATE discontent must clear for the
# rising to fire — the second gate. Set to two fully-resentful members' worth
# (~2 * RESENTMENT_THRESHOLD), so a tiny village of a couple of mildly-annoyed people does not
# "revolt": real risings need real, accumulated grievance, not just a bare majority barely over the line.
UPRISING_MIN_PRESSURE = 2.0 * discontent.RESENTMENT_THRESHOLD  # = 12.0

# UPRISING_COOLDOWN: after a CRUSHED rising, the terror of defeat buys the ruler this many turns
# before the settlement can rise again — a beaten people do not muster the courage to rise every
# single turn. The grievance itself is untouched (it persists and keeps climbing via M4.4); only the
# ACT of rising is suppressed for the cooldown.
UPRISING_COOLDOWN = 8

# FEAR_RETAIN: the fraction of its discontent a SURVIVING riser keeps after a crushed rising. Below 1
# so fear DOES take the edge off (a partial reset), but well above 0 so the grievance is NOT erased —
# M4.4's hysteresis then lets it climb back. A crushed revolt cows the people; it does not content them.
FEAR_RETAIN = 0.6

# VICTOR_DISCONTENT: the discontent a VICTORIOUS riser is left with — the grievance is ANSWERED (the
# oppressor is gone and his hoard is theirs), so it drops sharply toward zero.
VICTOR_DISCONTENT = 0.0

# --- M4.6 THE REVOLUTIONARY (closes Arc 2): the rising's leader legitimised by consent ------------
# When a rising WINS, M4.5 leaves the seat VACANT. M4.6 fills it: the agent who LED the mob emerges as
# the settlement's new LEADER — but NOT as a monarch. Power is seized by FORCE, then LEGITIMISED by
# CONSENT: the victory SEEDS his standing (surviving risers gain trust in him — a real, earned bump
# through the v1 trust system), and the UNCHANGED M3.2 leadership machinery then elects him from that
# trust exactly as it elects any leader — so he holds the seat only while trusted, and falls like any
# leader if his following erodes (M3.2 hysteresis/displacement, untouched). No new institution, no new
# title, no special power: the revolutionary rules through existing M3.2 leadership. If the victory bump
# is too thin (too few survivors to clear MIN_FOLLOWERS), NO leader emerges and the seat stays vacant —
# honest, not forced. This closes the political cycle: consent (M3.2) -> force (M3.4) -> revolt (M4.5) ->
# consent again (M4.6); and if the revolutionary later becomes an EXTRACTOR himself (seizes a force-title
# / over-taxes), the SAME discontent+uprising machinery rises against HIM (the revolution devours its
# children — verified, needs no new mechanic).

# REV_GRIEVANCE_WEIGHT / REV_TRUST_WEIGHT: the revolutionary is DERIVED from the risers, not assigned —
# the mob's natural focal point is the riser scoring highest on a WEIGHTED SUM of (a) his own discontent
# (the angriest — most grievance) and (b) the TRUST his fellow risers place in him (the M3.2 trust-cluster
# logic: the riser the other risers most trust). Both matter; equal default weights, tunable; name breaks ties.
REV_GRIEVANCE_WEIGHT = 1.0
REV_TRUST_WEIGHT = 1.0

# REVOLUTIONARY_WEALTH_MARGIN: the revolutionary must be an ORDINARY riser — a commoner thrown up BY the
# mob, never a rich agent RIDING it. "Rich" is judged RELATIVE to the mob, not on an absolute bar: a
# candidate is excluded from LEADING only if his (pre-expropriation) wealth exceeds the MEDIAN rioter's by
# more than this margin — i.e. a clear wealth OUTLIER. A flat absolute bar would wrongly exclude the very
# people who feel EXTRACTION grievance (they must hold wealth above the levy threshold to be levied at all),
# so it is relative: a leader may be modestly better off than the median rioter, but not a class apart.
# (An excluded outlier still rises and fights — it just does not get to lead the poor it stands above.)
REVOLUTIONARY_WEALTH_MARGIN = 3 * monarchy.MERC_MAX_WEALTH  # = 15.0 above the median rioter

# VICTORY_TRUST_BUMP: the trust each OTHER surviving riser gains in the revolutionary for leading them to
# victory — a real, earned bump written through the v1 trust system (logged). Sized to clear the M3.2
# FORM_TRUST bar from a neutral start, so a genuine following can cohere; a riser who already DISTRUSTS
# him (a grudge / deep negative) is not dragged over the bar by it — not everyone rallies (honest).
VICTORY_TRUST_BUMP = leadership.FORM_TRUST + 1  # = 3


def _find(state: dict[str, Any], name: str) -> "Any | None":
    return next((a for a in state["agents"] if a.name == name), None)


def _wealth(a: Any) -> float:
    """Liquid wealth = money + stored food (the M3.1 class metric) — used to gate out a rich leader."""
    return a.money + a.stockpile


def _settlement_ruler(state: dict[str, Any], sid: str) -> "tuple[str | None, str]":
    """The FORCE ruler the members of `sid` would rise against, and its KIND — a pure read.

    Mirrors M4.4's `discontent._ruler_and_due`: the felt oppressor is the VASSAL LORD if `sid` is a
    vassal settlement in a realm (tribute, M3.5), else the MONARCH holding it (levy, M3.4), else
    None — a consent-only trust-leader or an unruled settlement is NOT a valid target.
    """
    king = kingdoms.realm_of(state, sid)
    if king is not None:
        vassal = state.get("kingdoms", {}).get(king, {}).get("vassals", {}).get(sid)
        if vassal is not None:
            return vassal, "lord"
    mon = state.get("monarchs", {}).get(sid)
    if mon is not None:
        return mon["monarch"], "monarch"
    return None, "none"


def _risers(state: dict[str, Any], sid: str, ruler: str,
            defenders: set[str]) -> list[Any]:
    """The resentful members of `sid` who would take up the rising (sorted by name → deterministic).

    A riser is a LIVING settlement member who (a) is not the ruler, (b) is not a dependent child
    (the M4.1 gate — children do not fight), (c) is not already defending the crown (a paid
    garrison soldier / loyal follower has taken the ruler's side), and (d) has crossed M4.4's
    RESENTMENT_THRESHOLD. These numbers ARE the mob's whole force.
    """
    rec = state.get("settlements", {}).get(sid)
    if rec is None:
        return []
    gauge = state.get("discontent", {})
    living = {a.name: a for a in state["agents"] if a.alive}
    out = []
    for name in sorted(rec["members"]):
        if name == ruler or name in defenders:
            continue
        agent = living.get(name)
        if agent is None or world.is_dependent_child(agent, state):
            continue
        if gauge.get(name, 0.0) >= discontent.RESENTMENT_THRESHOLD:
            out.append(agent)
    return out


def _rise_gates(state: dict[str, Any], sid: str, risers: list[Any],
                defenders: set[str], turn: int) -> "tuple[bool, float]":
    """The HARD trigger gates as (gates_ok, aggregate). Pure read.

    gates_ok = off the fear cooldown AND a resentful MAJORITY of the non-ruler pool is rising AND that
    faction is non-empty. `aggregate` is their summed grievance — the second, SOFT gate a rising clears
    at UPRISING_MIN_PRESSURE. Splitting the hard gates (structural: cooldown + majority) from the soft
    pressure gate lets M5.1 open ONLY the pressure closeness to the ringleader's character (a majority
    is a structural fact; whether borderline weight is "enough" is the close call a firebrand decides).
    """
    if turn < state.get("uprising_cooldowns", {}).get(sid, 0):
        return False, 0.0  # still cowed by a recent crushing
    rec = state["settlements"][sid]
    living = {a.name for a in state["agents"] if a.alive}
    # Non-ruler members are the pool a majority is measured against (children/ruler/defenders excluded
    # — a child cannot rise, a soldier has already sided with the crown, the ruler cannot rise against itself).
    ruler, _ = _settlement_ruler(state, sid)
    pool = [n for n in rec["members"] if n in living and n != ruler and n not in defenders
            and not world.is_dependent_child(_find(state, n), state)]
    if not pool or not risers:
        return False, 0.0
    if len(risers) < UPRISING_FRACTION * len(pool):
        return False, 0.0  # gate 1: a resentful MAJORITY is required
    aggregate = sum(discontent.agent_discontent(r.name, state) for r in risers)
    return True, aggregate


def _should_rise(state: dict[str, Any], sid: str, risers: list[Any],
                 defenders: set[str], turn: int) -> bool:
    """Both trigger gates (fraction + aggregate floor), plus the fear cooldown. Pure read (deterministic)."""
    gates_ok, aggregate = _rise_gates(state, sid, risers, defenders, turn)
    return gates_ok and aggregate >= UPRISING_MIN_PRESSURE  # gate 2: real accumulated weight


def _expropriate(state: dict[str, Any], ruler_agent: Any, victors: list[Any],
                 turn: int) -> float:
    """Seize the deposed ruler's whole hoard and split it EQUALLY among the surviving risers.

    Runs BEFORE the ruler's death, so the seizure ZEROES the estate and M4.2 inheritance finds
    nothing — the revolution disinherits the heirs. Conserved to the exact float: the sum handed to
    victors equals the seized hoard (any float remainder goes to the first riser by name). Splitting
    as MONEY (a food-claim with no granary cap) keeps it clean and cap-free. Returns the seized total.
    """
    seized = float(ruler_agent.money) + float(ruler_agent.stockpile)
    ruler_agent.money = 0.0
    ruler_agent.stockpile = 0.0
    if seized <= 0.0 or not victors:
        return seized
    n = len(victors)
    shares = [seized / n for _ in range(n)]
    shares[0] += seized - sum(shares)  # absorb the float residual → exact conservation
    for victor, share in zip(sorted(victors, key=lambda a: a.name), shares):
        victor.money += share
        world.record_memory(victor, f"Seized {share:.2f} from the deposed {ruler_agent.name}")
    state["events"].append(
        f"turn {turn}: the risers EXPROPRIATED {ruler_agent.name}'s hoard of {seized:.2f} "
        f"— split among {n} (the heirs inherit nothing)")
    return seized


def _depose(state: dict[str, Any], ruler_agent: Any, sid: str, kind: str, turn: int) -> None:
    """Strip the deposed ruler's force-title for `sid` and free the settlement — then kill the ruler.

    Ordering is deliberate so the composition with Arc 1 is exact: (1) the monarch record for `sid`
    is cleared and the settlement SECEDES from any realm via the existing M3.5 machinery
    (`kingdoms.secede_settlement`); (2) ONLY THEN is the ruler killed through the normal death path.
    Because the force-title for this seat is already gone (and the hoard already seized), M4.3's
    `succeed_titles` finds nothing to pass and M4.2's `settle_estate` finds an empty estate — no heir
    takes the crown, no heir takes the gold. The seat is left VACANT (who fills it is M4.6, not here).
    """
    # 1. Cut the realm tie first (existing machinery), then clear the local monarch seat.
    kingdoms.secede_settlement(state, sid, turn, f"uprising deposed {ruler_agent.name}")
    mon = state.get("monarchs", {}).get(sid)
    if mon is not None and mon["monarch"] == ruler_agent.name:
        del state["monarchs"][sid]
    # 2. Kill the deposed ruler through the SAME death funnel every cause uses (M4.2/M4.3 now no-ops
    #    for this seat — title cleared, hoard seized). A queued respawn + survivor memories, as ever.
    population.announce_death(ruler_agent, turn, state, cause="deposed in the uprising",
                              final_memory="Deposed and killed in the uprising",
                              note="they were deposed in an uprising")


def _reset_discontent(state: dict[str, Any], agents: list[Any], to_value: "float | None",
                      retain: "float | None") -> None:
    """Set (victory) or scale-down (crushed fear) the gauge of the given risers. Pure gauge write."""
    gauge = state.setdefault("discontent", {})
    for a in agents:
        if not a.alive:
            continue
        if to_value is not None:
            gauge[a.name] = to_value
        elif retain is not None:
            gauge[a.name] = max(0.0, gauge.get(a.name, 0.0) * retain)


def _pick_revolutionary(state: dict[str, Any], risers: list[Any]) -> "Any | None":
    """The rising's leader, DERIVED from the risers (M4.6) — not assigned. Returns an agent or None.

    Among the ORDINARY risers (not a wealth OUTLIER riding the mob — wealth within
    REVOLUTIONARY_WEALTH_MARGIN of the median rioter's), the natural focal point is the one scoring
    highest on the weighted sum of his own DISCONTENT (the angriest) and the TRUST his FELLOW risers
    place in him (the M3.2 trust-cluster — the riser the mob most trusts). Deterministic: ties break on
    name. None if no ordinary candidate survives — then the seat honestly stays vacant.
    """
    alive = [r for r in risers if r.alive]
    if not alive:
        return None
    wealths = sorted(_wealth(r) for r in alive)
    median = wealths[len(wealths) // 2]
    ceiling = median + REVOLUTIONARY_WEALTH_MARGIN
    pool = [r for r in alive if _wealth(r) <= ceiling]
    if not pool:
        return None

    def score(cand: Any) -> float:
        grievance = discontent.agent_discontent(cand.name, state)
        backing = sum(o.relationships.get(cand.name, {}).get("trust", 0)
                      for o in risers if o.name != cand.name)
        return REV_GRIEVANCE_WEIGHT * grievance + REV_TRUST_WEIGHT * backing

    return sorted(pool, key=lambda c: (-score(c), c.name))[0]


def _seed_leadership(state: dict[str, Any], leader: Any, victors: list[Any],
                     sid: str, turn: int) -> None:
    """SEED the revolutionary's legitimacy: the surviving risers gain trust in him (M4.6).

    Writes a real, earned trust bump through the v1 trust system for every OTHER surviving riser
    (a follower who already carries a grudge against him is refused by adjust_trust — not everyone
    rallies). It installs NO leader record itself: the UNCHANGED M3.2 `leadership.update` then elects
    him from this trust exactly as it elects any leader — so he holds the seat purely by CONSENT and
    only while it lasts. If too few followers clear the bar, M3.2 simply seats no one (vacant, honest).
    """
    for v in victors:
        if v.name == leader.name:
            continue
        trust.adjust_trust(v, leader.name, VICTORY_TRUST_BUMP,
                           f"led the {sid} uprising to victory", turn, state)
    state["events"].append(
        f"turn {turn}: {leader.name} led the rising in {sid} — the survivors rally to him "
        f"(power to be legitimised by consent, M3.2)")
    world.record_memory(leader, f"Led the {sid} uprising and rallied its survivors")


def _rise(state: dict[str, Any], sid: str, turn: int) -> "dict[str, Any] | None":
    """Resolve ONE settlement's potential rising this turn. Returns a result dict if it fired, else None.

    The whole M4.5 event for `sid`: resolve the target ruler, assemble the mob (numbers) and the
    ruler's funded defence (garrison/followers + freshly mustered mercenaries), check both trigger
    gates, and — if they hold — fight it with the shared battle maths and apply the outcome
    (crushed → fear + cooldown; victorious → expropriate → depose → answer the grievance).
    """
    ruler, kind = _settlement_ruler(state, sid)
    if ruler is None:
        return None  # no force ruler → nothing to overthrow (consent-led / unruled)
    ruler_agent = _find(state, ruler)
    if ruler_agent is None or not ruler_agent.alive:
        return None

    base_def, _def_kind = monarchy.defenders_of(state, sid)
    def_names = {d.name for d in base_def}
    risers = _risers(state, sid, ruler, def_names)
    # The HARD gates (cooldown + resentful majority) are structural — a rising is impossible without
    # them. The SOFT pressure gate is the close call.
    gates_ok, aggregate = _rise_gates(state, sid, risers, def_names, turn)
    if not gates_ok:
        return None
    rise = aggregate >= UPRISING_MIN_PRESSURE
    if state.get("minds_on"):
        # M5.1 PIVOT: when the accumulated grievance sits within UPRISING_BAND of the trigger, the
        # RINGLEADER (the angriest riser — the natural firebrand) decides whether the hour has come: a
        # daring one raises the banner on borderline weight, a cautious one waits. Off / out-of-band ->
        # exactly `aggregate >= UPRISING_MIN_PRESSURE` (byte-identical to v1).
        import mind
        ringleader = max(risers, key=lambda r: (discontent.agent_discontent(r.name, state), r.name))
        rise, _ = mind.tilt(state, ringleader.name, "uprising", aggregate - UPRISING_MIN_PRESSURE, rise,
                            {"pressure": round(aggregate, 1), "threshold": UPRISING_MIN_PRESSURE,
                             "sid": sid}, turn)
    if not rise:
        return None

    # THE RULER's DEFENCE: his standing garrison/loyal followers PLUS any mercenaries his war chest
    # can still buy (you cannot hire your own attackers or your own defenders). A drained treasury
    # buys none → the mob's numbers decide.
    exclude = {ruler} | def_names | {r.name for r in risers}
    mercs = monarchy.muster(state, ruler_agent, exclude)
    defence = base_def + mercs

    n_mob, n_def = len(risers), len(defence)
    state["events"].append(
        f"turn {turn}: UPRISING in {sid} — {n_mob} risers rise against {kind} {ruler} "
        f"({n_def} defenders: {len(base_def)} standing + {len(mercs)} hired)")

    won, mob_dead, def_dead, survivors = monarchy.resolve_battle(
        state, risers, defence, turn, f"risers of {sid}", f"{ruler}'s guard")

    result = {"sid": sid, "ruler": ruler, "kind": kind, "won": won,
              "mob": n_mob, "defenders": n_def, "mob_dead": mob_dead, "def_dead": def_dead,
              "seized": 0.0, "deposed": False, "leader": None}

    if won:
        living_victors = [r for r in survivors if r.alive]
        # M4.6: identify the revolutionary from the surviving risers BEFORE the grievance is answered
        # (the pick reads discontent) — the angriest-and-most-trusted ordinary riser, derived not assigned.
        revolutionary = _pick_revolutionary(state, living_victors)
        result["seized"] = _expropriate(state, ruler_agent, living_victors, turn)
        _depose(state, ruler_agent, sid, kind, turn)
        # M4.6: SEED his legitimacy through the trust system; M3.2 leadership.update then elects him by
        # CONSENT (or seats no one if the following is too thin). No monarch record, no special power.
        if revolutionary is not None:
            _seed_leadership(state, revolutionary, living_victors, sid, turn)
            result["leader"] = revolutionary.name
        _reset_discontent(state, living_victors, VICTOR_DISCONTENT, None)
        result["deposed"] = True
        vacancy = (f"{revolutionary.name} to rule by consent"
                   if revolutionary is not None else "the seat lies vacant (no leader emerged)")
        state["events"].append(
            f"turn {turn}: the UPRISING in {sid} TRIUMPHED — {kind} {ruler} is DEPOSED; "
            f"{vacancy} ({len(mob_dead)} risers fell)")
    else:
        # Crushed. The surviving risers are cowed (partial reset, grievance persists); a fear
        # cooldown falls over the settlement; the ruler's garrison shrinks to its survivors.
        _reset_discontent(state, [r for r in risers if r.alive], None, FEAR_RETAIN)
        state.setdefault("uprising_cooldowns", {})[sid] = turn + UPRISING_COOLDOWN
        mon = state.get("monarchs", {}).get(sid)
        if mon is not None and mon["monarch"] == ruler:
            mon["garrison"] = {d.name for d in base_def if d.alive}
        state["events"].append(
            f"turn {turn}: the UPRISING in {sid} was CRUSHED — {kind} {ruler} holds "
            f"({len(def_dead)} guards + {len(mob_dead)} risers fell); the survivors are cowed")
    return result


def update(state: dict[str, Any], turn: int) -> list[dict[str, Any]]:
    """Advance the uprising system one turn (ZERO LLM, ZERO RNG, M4.5). Returns the risings that fired.

    For each settlement (sorted → deterministic) that still exists, resolve one potential rising
    (`_rise`). Runs AFTER `discontent.update` so it reads THIS turn's fresh gauge, and it is the last
    institutional act of the turn: it CONSUMES the pressure M4.4 exposes and, on victory, DEPOSES a
    ruler through the same death path every institution already uses. Caller gates on `uprising_on`,
    so an off run never calls this — no cooldown key is written — and stays byte-identical to v1.
    """
    results: list[dict[str, Any]] = []
    # Snapshot the settlement ids up front: a rising can dissolve a realm / free a settlement, but
    # never adds one, so a stable sorted snapshot keeps the iteration deterministic and safe.
    for sid in sorted(state.get("settlements", {})):
        if sid not in state.get("settlements", {}):
            continue
        res = _rise(state, sid, turn)
        if res is not None:
            results.append(res)
    return results


# --- Derived read-outs (pure reads, for the summary / tests) -----------------
def would_rise(state: dict[str, Any], sid: str, turn: int) -> bool:
    """True iff `sid` currently meets BOTH trigger gates (and is off cooldown) — a pure read.

    Lets the summary / tests ask "is this settlement about to blow?" without resolving a battle or
    mutating anything. Mirrors the exact trigger `_rise` uses.
    """
    ruler, _ = _settlement_ruler(state, sid)
    if ruler is None:
        return False
    base_def, _ = monarchy.defenders_of(state, sid)
    def_names = {d.name for d in base_def}
    risers = _risers(state, sid, ruler, def_names)
    return _should_rise(state, sid, risers, def_names, turn)
