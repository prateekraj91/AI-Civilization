"""
empire.py
=========

INTER-KINGDOM WAR & EMPIRE — the CLIMAX of Phase 3 (V2 milestone M3.6, Phase 3: Institutions). On
top of M3.5 (kingdoms & vassalage), M3.4 (conquest & monarchy), M3.3 (taxation), M3.2 (leadership),
M3.1 (wage labor) and all of Phase 0/1/2.

The historical step M3.6 makes — feudal KINGDOMS clash, and EMPIRES rise (and fall)
------------------------------------------------------------------------------------
M3.5 built feudal KINGDOMS (king -> vassal lords -> settlements), with tribute cascading up and
CONDITIONAL loyalty (an over-taxed vassal breaks away; a king's strength is the sum of his LOYAL
vassals' forces). M3.6 sets KINGDOMS against EACH OTHER in WAR, and the loser is absorbed into an
EMPIRE — a THIRD level of the same feudal hierarchy: EMPEROR -> subject-KING -> vassal-lords ->
settlements. And empires are no more permanent than kingdoms: a subjugated subject-king is a
high-level vassal whose loyalty is just as CONDITIONAL, so an emperor who over-taxes or weakens
FRAGMENTS. Power is contingent at EVERY level — that is the thematic capstone of Phase 3.

The deep coupling — the whole point of the milestone — is that M3.5's LOYALTY DECIDES WARS. Each
kingdom musters its WHOLE host (king's force + every LOYAL vassal's contingent — `kingdoms.muster_
realm`), so a kingdom's war strength is its LOYAL realm strength, NOT its nominal size or wealth. A
RICHER kingdom whose vassals are DISLOYAL fields a SMALLER host and LOSES to a POORER kingdom whose
vassals all muster. Wealth FUNDS fighters, but loyalty FIELDS them: good governance beats raw wealth.

Scope (held strictly; boundaries stated so later milestones own the rest)
-------------------------------------------------------------------------
IN scope: inter-kingdom WAR (whole feudal hosts clash, resolved by the SAME `monarchy.resolve_
battle`), SUBJUGATION of the defeated king into the victor's realm (a multi-level EMPIRE), and
FRAGMENTATION (a subjugated subject-king can later break away — empires rise AND fall). OUT of scope
(deferred): active vassal DEFECTION-TO-THE-ENEMY mid-battle (a vassal switching sides during a war),
DIPLOMACY / ALLIANCES between kingdoms, and fiat money. The war is force-on-force between whole
realms; no one changes sides mid-clash and no one negotiates instead of fighting.

The model (emerges from M3.5 muster + loyalty; zero LLM, zero RNG, deterministic state math)
--------------------------------------------------------------------------------------------
1. WAR TRIGGER — OPPORTUNISTIC (emergent, not scripted). A king may attack a NEIGHBOURING kingdom
   (an adjacent realm — the realm-vs-realm analogue of M3.5's `conquer_neighbour`) when his
   assessable LOYAL host can BEAT the target's defendable LOYAL host. A king does NOT start a war he
   would lose (the SAME winnable-assault guard as M3.4/M3.5 — `imperial_host_size`, a dry-run count).
   Gain is implicit: any kingdom won yields territory + tribute + a tributary subject-king, so the
   only question a rational crown asks is "can I win?". Greed + assessed strength drive it.

2. THE CLASH — whole feudal HOSTS, loyalty decides (the headline coupling). Each side musters its
   WHOLE host via `imperial_host` (the king's realm host — `kingdoms.muster_realm` — PLUS every LOYAL
   subject-king's whole host, recursively, for a multi-level empire). A DISLOYAL or broken-away
   vassal/subject-king does NOT answer the call. The clash is resolved by the EXISTING shared
   `monarchy.resolve_battle`: strict `>` decides it, casualties fall on BOTH sides (war kills real
   agents in both armies), deterministic + seeded. So war strength = LOYAL host strength, and a rich
   tyrant's brittle realm (disloyal vassals withhold their swords) can lose to a poorer, loyal one.

3. SUBJUGATION -> multi-level EMPIRE (what happens to the loser). On defeat the loser's KING is
   SUBJUGATED — he becomes a high-level VASSAL (a "subject-king") of the victor, KEEPING his own
   internal realm (his vassals/settlements stay under him) but now owing TRIBUTE and military SERVICE
   UPWARD. This reuses M3.5's vassal machinery ONE LEVEL UP: a subject-king is to the emperor what a
   vassal-lord is to a king. Tribute now cascades through MORE levels (settlement -> lord ->
   subject-king -> emperor); the emperor can muster a LOYAL subject-king's WHOLE host for further
   war. State: world_state["empires"][emperor] = {emperor, subject_kings, founded, discontent}.

4. FRAGMENTATION — empires rise AND fall (the rise-and-fall cycle). A subject-king's loyalty is
   CONDITIONAL exactly as a vassal-lord's is (M3.5's logic, nested): an emperor whose imperial share
   exceeds the consent band erodes the subject-king's trust (the M3.3-shape backlash), and a
   WEAKENING emperor (one who can no longer field a host as large as the subject-king's own — the
   protection bargain failing) erodes it too. A subject-king whose trust collapses for
   BREAKAWAY_PATIENCE consecutive turns BREAKS AWAY (HYSTERESIS — no single-turn flip), reclaiming
   INDEPENDENCE and taking his whole realm with him. A FAIR, strong emperor holds the empire
   together. So empires that overextend or over-tax FRAGMENT — there is no permanent empire.

Reuse (no duplication) — M3.6 RIDES M3.5 and M3.4
-------------------------------------------------
The fight is `monarchy.resolve_battle` (shared by M3.4 single-settlement, M3.5 realm, M3.6 empire).
The muster is `kingdoms.muster_realm` (king + loyal vassal-lords); `imperial_host` wraps it to add
loyal subject-kings, recursively. Loyalty, the consent band, the backlash, the breakaway hysteresis
and the trust thresholds are M3.5's constants (kingdoms.KING_CONSENT / RESENT_SCALE / LOYAL_TRUST /
BREAKAWAY_TRUST / BREAKAWAY_PATIENCE), applied one level up. Tribute moves wealth via economy._settle
and writes loyalty through the logged trust.adjust_trust (the M3.3 path). The empire is a pure
overlay on world_state["empires"] (no new Agent field); kingdoms[subject_king] is left intact.

Cost & determinism
------------------
ZERO LLM calls and ZERO RNG. Deterministic iteration (sorted emperor names, sorted subject-king
names, sorted king names; the clash reuses RNG-free `monarchy.resolve_battle`; casualties by name).
A run with the institution OFF never calls `update` (empires stays empty), so it is byte-identical
to v1. Imports kingdoms + monarchy + economy + trust + world (one-directional), keeping the world
layer dependency-free.
"""

from __future__ import annotations

from typing import Any

from sim import economy
from sim import kingdoms
from sim import monarchy
from sim import trust
from sim import world

# --- Tunable constants (documented) ----------------------------------------
# DEFAULT_EMPIRE_SHARE: the emperor's share of a subject-king's wealth (above EMPIRE_THRESHOLD) taken
# as imperial tribute each turn (subject-king -> emperor) when a run does not name one. 0.25 sits
# UNDER kingdoms.KING_CONSENT (0.35), so the default emperor is a MODERATE overlord — sustainable, his
# subject-kings stay loyal. A run sets world_state["empire_share"] to push it (e.g. 0.9) and watch a
# subject-king's loyalty erode toward fragmentation. This is the imperial analogue of M3.5's KING_SHARE
# / M3.3's tax_rate lever, governed by the SAME consent band so over-extraction punishes identically.
DEFAULT_EMPIRE_SHARE = 0.25

# EMPIRE_THRESHOLD: the wealth floor a subject-king keeps untaxed by his emperor (mirrors M3.5's
# TRIBUTE_THRESHOLD / M3.4's MONARCH_LEVY_THRESHOLD one level up) — the emperor levies only the
# subject-king's wealth ABOVE this, so a poor subject-king owes nothing.
EMPIRE_THRESHOLD = monarchy.MONARCH_LEVY_THRESHOLD  # 5.0

# All loyalty mechanics are M3.5's, applied one level up (no new constants — a subject-king is a
# vassal of the emperor): kingdoms.LOYAL_TRUST to muster/stay, kingdoms.KING_CONSENT / RESENT_SCALE
# for the over-tax backlash, kingdoms.BREAKAWAY_TRUST / BREAKAWAY_PATIENCE for the fragmentation
# hysteresis, and kingdoms.KINGDOM_REACH for realm-vs-realm adjacency.


def _find(state: dict[str, Any], name: str | None) -> Any | None:
    """The living agent called `name`, or None (mirrors kingdoms/monarchy/leadership)."""
    if name is None:
        return None
    for a in state["agents"]:
        if a.alive and a.name == name:
            return a
    return None


def _trust_in(subject: Any, lord_name: str) -> int:
    """A subject-king's LOYALTY = its trust in the emperor — a pure read of the v1 trust network."""
    return subject.relationships.get(lord_name, {}).get("trust", 0)


def emperor_of(state: dict[str, Any], king_name: str) -> str | None:
    """The emperor whose empire holds subject-king `king_name`, or None if independent. Pure read."""
    for emperor in sorted(state.get("empires", {})):
        if king_name in state["empires"][emperor]["subject_kings"]:
            return emperor
    return None


def is_sovereign(state: dict[str, Any], king_name: str) -> bool:
    """True iff `king_name` is an INDEPENDENT king (not a subject-king of any emperor). Pure read."""
    return emperor_of(state, king_name) is None


# --- Military service: the imperial host = realm host + loyal subject-kings' whole hosts ---
def imperial_host(state: dict[str, Any], sovereign: Any, exclude: set[str]) -> list[Any]:
    """Raise `sovereign`'s WHOLE host: his realm host PLUS every LOYAL subject-king's whole host.

    The sovereign first musters his own realm (`kingdoms.muster_realm`: own force + loyal vassal-
    lords). Then he CALLS his subject-kings: each subject-king still in the empire AND loyal (trust in
    the sovereign >= LOYAL_TRUST) answers by mustering his OWN whole host (recursively — so a
    subject-king brings his loyal vassal-lords, and a sub-subject-king his own, for a multi-level
    empire). A disloyal subject-king withholds service; a broken-away one is no longer in the empire.
    So the imperial host = the SUM of LOYAL hosts at every level. Returns the combined host (real
    agents). ZERO RNG. (Boundary: no vassal switches to the ENEMY here — defection is out of scope.)
    """
    host = kingdoms.muster_realm(state, sovereign, exclude)
    taken = exclude | {f.name for f in host}
    emp = state.get("empires", {}).get(sovereign.name)
    if emp is not None:
        for sk_name in sorted(emp["subject_kings"]):
            sk = _find(state, sk_name)
            if sk is None or _trust_in(sk, sovereign.name) < kingdoms.LOYAL_TRUST:
                continue  # a disloyal/absent subject-king answers no imperial muster (service is conditional)
            contingent = imperial_host(state, sk, taken | {sk.name})
            taken |= {f.name for f in contingent}
            host.extend(contingent)
    return host


def imperial_host_size(state: dict[str, Any], sovereign: Any) -> int:
    """A dry-run count of the host `sovereign` COULD field now (his realm + loyal subject-kings).

    Mirrors `kingdoms.realm_host_size` one level up: the realm host the sovereign can raise plus, for
    each LOYAL subject-king, that subject-king's own imperial host (recursively). Pure read (pays no
    one) — lets the war loop launch only WINNABLE wars (a rational crown does not march a host it
    knows is too small). ZERO RNG.
    """
    size = kingdoms.realm_host_size(state, sovereign)
    emp = state.get("empires", {}).get(sovereign.name)
    if emp is not None:
        for sk_name in sorted(emp["subject_kings"]):
            sk = _find(state, sk_name)
            if sk is None or _trust_in(sk, sovereign.name) < kingdoms.LOYAL_TRUST:
                continue
            size += imperial_host_size(state, sk)
    return size


# --- Imperial tribute: subject-king -> emperor (cascades up the new level) ---
def tribute(state: dict[str, Any], turn: int) -> list[str]:
    """Run the IMPERIAL tribute level for every empire: subject-king -> emperor. ZERO RNG, M3.6.

    This is the TOP level of the feudal cascade, run AFTER `kingdoms.tribute` has already filled each
    subject-king's coffers from his own realm (settlement -> lord -> subject-king). For each empire
    (sorted) and each subject-king (sorted): a share (`empire_share`, default DEFAULT_EMPIRE_SHARE) of
    the subject-king's wealth ABOVE EMPIRE_THRESHOLD cascades up to the emperor. Wealth is only MOVED
    (economy._settle), so the whole chain (settlement -> lord -> subject-king -> emperor) CONSERVES
    total wealth. Heavy imperial demands then write the loyalty BACKLASH through the trust system (the
    M3.3 path): an emperor whose share exceeds kingdoms.KING_CONSENT loses each subject-king's trust by
    round((rate - KING_CONSENT) * RESENT_SCALE) per turn. A WEAKENING emperor (one who can no longer
    field a host as large as the subject-king's own) also loses 1 trust/turn (the protection bargain
    failing). Both erosions feed the fragmentation hysteresis. Returns the events logged.
    """
    rate = state.get("empire_share", DEFAULT_EMPIRE_SHARE)
    living = {a.name: a for a in state["agents"] if a.alive}
    resent = -round(max(0.0, rate - kingdoms.KING_CONSENT) * kingdoms.RESENT_SCALE)
    events: list[str] = []

    for emperor_name in sorted(state.get("empires", {})):
        emp = state["empires"][emperor_name]
        emperor = living.get(emperor_name)
        if emperor is None:
            continue
        for sk_name in sorted(emp["subject_kings"]):
            sk = living.get(sk_name)
            if sk is None:
                continue
            # The imperial level — subject-king -> emperor (a share of the subject-king's wealth,
            # which already holds what cascaded up from his realm this turn).
            due = rate * max(0.0, (sk.money + sk.stockpile) - EMPIRE_THRESHOLD)
            if due > 0:
                economy._settle(sk, emperor, due)
                events.append(
                    f"turn {turn}: imperial tribute cascaded up: {due:.1f} subject-king {sk_name}"
                    f"->EMPEROR {emperor_name}")
            # Backlash 1 — a grasping emperor erodes his subject-king's loyalty (writes trust, M3.3 shape).
            if resent < 0:
                trust.adjust_trust(sk, emperor_name, resent, "heavy imperial tribute", turn, state)
            # Backlash 2 — a WEAKENING emperor (cannot out-field the subject-king's own host) loses
            # the bargain's credibility: the strong subject-king no longer needs the weak overlord.
            elif imperial_host_size(state, emperor) < kingdoms.realm_host_size(state, sk):
                trust.adjust_trust(sk, emperor_name, -1, "the emperor weakened", turn, state)

    state["events"].extend(events)
    return events


# --- Conditional loyalty: a subject-king fragments away (with hysteresis) ----
def _check_fragmentation(state: dict[str, Any], turn: int) -> list[str]:
    """Drop subject-kings whose loyalty has collapsed for BREAKAWAY_PATIENCE consecutive turns. M3.6.

    The M3.5 breakaway logic, nested one level up: a subject-king whose trust in the emperor is <=
    BREAKAWAY_TRUST raises a discontent counter; once it reaches BREAKAWAY_PATIENCE he BREAKS AWAY —
    reclaiming INDEPENDENCE and taking his whole realm with him (kingdoms[subject_king] is untouched;
    only the imperial tie is cut). Loyalty above the floor RESETS the counter (hysteresis both ways).
    So an empire FRAGMENTS exactly as a kingdom does. Returns the events logged.
    """
    events: list[str] = []
    for emperor_name in sorted(state.get("empires", {})):
        emp = state["empires"][emperor_name]
        emperor = _find(state, emperor_name)
        for sk_name in sorted(emp["subject_kings"]):
            sk = _find(state, sk_name)
            disc = emp["discontent"]
            if sk is not None and emperor is not None:
                # M5.1 PIVOT (same breakaway pivot, one level up): a subject-king whose loyalty sits
                # within BREAKAWAY_BAND of the floor decides by CHARACTER whether to fragment away; far
                # above it he decisively stays. Off / out-of-band -> exactly the M3.6 rule (byte-identical).
                margin = kingdoms.BREAKAWAY_TRUST - _trust_in(sk, emperor_name)   # >0 => leans to BREAK
                break_now = _trust_in(sk, emperor_name) <= kingdoms.BREAKAWAY_TRUST
                if state.get("minds_on"):
                    from llm import mind
                    break_now, _ = mind.tilt(state, sk_name, "breakaway", margin, break_now,
                                             {"trust": _trust_in(sk, emperor_name), "lord": emperor_name}, turn)
                if not break_now:
                    disc[sk_name] = 0  # loyalty holds (or recovered) — reset the hysteresis counter
                    continue
            # A subject-king whose emperor has died/vanished also reclaims independence.
            disc[sk_name] = disc.get(sk_name, 0) + 1
            if disc[sk_name] < kingdoms.BREAKAWAY_PATIENCE:
                continue
            emp["subject_kings"].pop(sk_name)
            disc.pop(sk_name, None)
            events.append(
                f"turn {turn}: subject-king {sk_name} BROKE AWAY from {emperor_name}'s empire — "
                f"reclaiming independence with his realm (loyalty collapsed)")
            if sk is not None:
                world.record_memory(sk, f"Broke away from {emperor_name}'s empire, reclaiming my crown")
        if not emp["subject_kings"]:
            state["empires"].pop(emperor_name)  # an emperor with no subject-kings is just a king again
    state["events"].extend(events)
    return events


# --- War & subjugation: kingdoms clash, the loser is absorbed ----------------
def _subjugate(state: dict[str, Any], victor: Any, loser_name: str, turn: int) -> None:
    """Absorb the defeated king `loser_name` into `victor`'s empire as a high-level VASSAL (M3.6).

    The loser KEEPS his own realm (kingdoms[loser] is untouched — his vassals/settlements stay under
    him) but is added to the victor's empire as a subject-king, swearing FEALTY (trust seeded to
    LOYAL_TRUST). If the loser was already a subject-king of ANOTHER emperor he is taken from them
    (conquered away). The empire record is created on the victor's first conquest. A multi-level
    hierarchy results: EMPEROR -> subject-king -> the subject-king's vassal-lords -> settlements.
    """
    # Detach from any prior emperor (he has been conquered away).
    for other in sorted(state.get("empires", {})):
        if other != victor.name and loser_name in state["empires"][other]["subject_kings"]:
            state["empires"][other]["subject_kings"].pop(loser_name)
            state["empires"][other]["discontent"].pop(loser_name, None)
            if not state["empires"][other]["subject_kings"]:
                state["empires"].pop(other)
    emp = state.setdefault("empires", {}).setdefault(
        victor.name, {"emperor": victor.name, "subject_kings": {}, "founded": turn, "discontent": {}})
    emp["subject_kings"][loser_name] = {"since": turn}
    emp["discontent"][loser_name] = 0
    loser = _find(state, loser_name)
    if loser is not None:
        cur = _trust_in(loser, victor.name)
        trust.adjust_trust(loser, victor.name, kingdoms.LOYAL_TRUST - cur,
                           "submitted to the emperor after defeat", turn, state)


def wage_war(state: dict[str, Any], attacker_name: str, defender_name: str, turn: int) -> dict[str, Any]:
    """King `attacker_name` wages WAR on neighbouring kingdom `defender_name`. Deterministic, M3.6.

    Both kings muster their WHOLE LOYAL hosts (`imperial_host`: king + loyal vassals + loyal subject-
    kings, recursively) — the attacker first, then the defender from the REMAINING mercenary pool (no
    fighter serves both armies). The clash is resolved by the SAME `monarchy.resolve_battle` (strict
    `>`, casualties on BOTH sides — war kills real agents in both armies). On victory the defeated king
    is SUBJUGATED (`_subjugate`) into the attacker's empire as a subject-king. Returns {won, att_host,
    def_host, att_dead, def_dead, attacker, defender}. ZERO RNG. NOTE: like M3.4/M3.5 this resolves
    ANY matchup the caller stages (so a verify/test can drive a doomed war); the `update` loop only
    LAUNCHES winnable wars.
    """
    attacker, defender = _find(state, attacker_name), _find(state, defender_name)
    if attacker is None or defender is None:
        return {"won": False, "att_host": 0, "def_host": 0, "att_dead": [], "def_dead": [],
                "attacker": attacker_name, "defender": defender_name}
    # Each side's whole loyal host; exclude the rival's leadership and don't double-hire mercenaries.
    att_host = imperial_host(state, attacker, {attacker_name, defender_name})
    taken = {f.name for f in att_host} | {attacker_name, defender_name}
    def_host = imperial_host(state, defender, taken)
    # M4.13 DIPLOMACY: this war leaves a lasting grievance, and the defender's honouring ALLIES bring
    # their whole loyal hosts to the defence — so an attacker faces the COMBINED hosts of all who answer.
    # Gated on diplomacy_on (lazily imported), so a non-diplomacy run is byte-identical.
    if state.get("diplomacy_on"):
        from sim import diplomacy
        diplomacy.record_war(state, attacker_name, defender_name, turn)
        for ally_name in diplomacy.defensive_allies(state, defender_name):
            ally = _find(state, ally_name)
            if ally is None:
                continue
            contingent = imperial_host(state, ally, taken | {ally_name})
            taken |= {f.name for f in contingent}
            def_host.extend(contingent)
    # M4.15 COALITION: if the hegemon attacks a coalition member, the OTHER members muster to its defence
    # (fear-driven mutual defence, so the hegemon cannot pick members off one by one). Gated (byte-identical off).
    if state.get("coalitions_on"):
        from sim import coalitions
        for backer_name in coalitions.coalition_backers(state, defender_name, attacker_name):
            if backer_name in taken:
                continue
            backer = _find(state, backer_name)
            if backer is None:
                continue
            contingent = imperial_host(state, backer, taken | {backer_name})
            taken |= {f.name for f in contingent}
            def_host.extend(contingent)
    n_att, n_def = len(att_host), len(def_host)

    won, att_dead, def_dead, _ = monarchy.resolve_battle(
        state, att_host, def_host, turn, f"{attacker_name}'s imperial host", f"{defender_name}'s host")

    if won:
        _subjugate(state, attacker, defender_name, turn)
        state["events"].append(
            f"turn {turn}: KING {attacker_name} DEFEATED {defender_name} in war "
            f"({n_att} loyal host vs {n_def}; {len(att_dead)}+{len(def_dead)} fell) -> "
            f"{defender_name} SUBJUGATED as a subject-king; an EMPIRE rises")
        world.record_memory(attacker, f"Conquered {defender_name}'s kingdom, making him my subject-king")
    else:
        state["events"].append(
            f"turn {turn}: KING {attacker_name}'s war on {defender_name} FAILED "
            f"({n_att} loyal host vs {n_def}; {len(att_dead)}+{len(def_dead)} fell) — the kingdom held")
        world.record_memory(attacker, f"Failed to conquer {defender_name}'s kingdom ({n_att} vs {n_def})")

    return {"won": won, "att_host": n_att, "def_host": n_def, "att_dead": att_dead,
            "def_dead": def_dead, "attacker": attacker_name, "defender": defender_name}


def _realm_settlements(state: dict[str, Any], king_name: str) -> set[str]:
    """All settlements under `king_name`'s realm (his own; subject-kings keep their OWN realms)."""
    return set(state.get("kingdoms", {}).get(king_name, {}).get("settlements", set()))


def _imperial_settlements(state: dict[str, Any], king_name: str) -> set[str]:
    """Every settlement a sovereign CONTROLS: his own realm plus his subject-kings', recursively.

    An empire's borders are the borders of everything it holds, not just of the emperor's personal
    realm. Reading only the personal realm made a conquest ERASE the frontier it had just won: the
    moment a rival realm was subjugated, its towns stopped counting toward the empire's reach, the
    empire's remaining neighbours found themselves bordering nobody, and the war engine went quiet
    for the rest of the run — the map consolidated and then froze. Territory won stays territory.
    """
    out = set(_realm_settlements(state, king_name))
    emp = state.get("empires", {}).get(king_name)
    if emp is not None:
        for sk in sorted(emp["subject_kings"]):
            if sk != king_name:
                out |= _imperial_settlements(state, sk)
    return out


def _kingdom_neighbours(state: dict[str, Any], king_name: str) -> list[str]:
    """Other SOVEREIGN kings whose territory lies within KINGDOM_REACH of `king_name`'s (sorted).

    Realm-vs-realm adjacency (the analogue of M3.5's settlement reach): two powers are neighbours if
    any settlement either CONTROLS (see `_imperial_settlements`) is within kingdoms.KINGDOM_REACH of
    any settlement the other controls. Only INDEPENDENT kingdoms are candidates (a subject-king is
    already inside an empire and is reached through its emperor). Pure read.
    """
    sets = state.get("settlements", {})
    mine = [sets[s]["center"] for s in _imperial_settlements(state, king_name) if s in sets]
    out = []
    for other in sorted(state.get("kingdoms", {})):
        # Skip self, subject-kings (already inside an empire), and any king whose agent is no longer
        # living — a dead king's realm is not a war target (mirrors the attacker aliveness guard in
        # `update`), and admitting one would later NoneType-crash the host-size dry runs.
        if other == king_name or not is_sovereign(state, other) or _find(state, other) is None:
            continue
        theirs = [sets[s]["center"] for s in _imperial_settlements(state, other) if s in sets]
        if any(monarchy._chebyshev(a, b) <= kingdoms.KINGDOM_REACH for a in mine for b in theirs):
            out.append(other)
    return out


def _war_debug(state: dict[str, Any], turn: int) -> Any:
    """Return a per-turn war-gate LOGGER, or a no-op when `--debug-war` is off (pure observation).

    The showcase lives or dies on whether the war engine actually fires, so this makes the gate
    legible: for every sovereign crown it reports the host it could field, each neighbour it
    weighs, and the verdict. It writes to STDERR only — never to world_state and never to the
    event log — so a debugged run is byte-identical to an undebugged one. It also prints the
    SOVEREIGN COUNT, which is what silently drops to one when an empire swallows the map (a run
    with fewer than two sovereign powers has no war engine left to fire).
    """
    if not state.get("debug_war"):
        return lambda *a, **k: None
    import sys
    kings = sorted(k for k in state.get("kingdoms", {}) if is_sovereign(state, k))
    sizes = {k: imperial_host_size(state, _find(state, k)) for k in kings if _find(state, k)}
    print(f"[war] turn {turn:3d} | sovereign powers: {len(sizes)} "
          f"({', '.join(f'{k}={v}' for k, v in sizes.items()) or 'none'})", file=sys.stderr)

    def log(att: str, tgt: "str | None", a_size: "int | None", d_size: "int | None", why: str) -> None:
        edge = "" if a_size is None else f" host={a_size}"
        edge += "" if d_size is None else f" vs {d_size}"
        against = f" -> {tgt}" if tgt else ""
        print(f"[war]          {att}{against}{edge}: {why}", file=sys.stderr)
    return log


def update(state: dict[str, Any], turn: int) -> list[str]:
    """Advance the inter-kingdom war / empire institution one turn (ZERO LLM, ZERO RNG, M3.6).

    Three deterministic stages: (1) the IMPERIAL tribute level (subject-king -> emperor) plus the
    loyalty backlash from a grasping or weakening emperor; (2) FRAGMENTATION — subject-kings whose
    loyalty has collapsed for BREAKAWAY_PATIENCE turns reclaim independence; (3) WAR — each SOVEREIGN
    king opportunistically attacks the strongest neighbouring kingdom whose defendable LOYAL host he
    can out-field, subjugating it into his empire. One war per attacker per turn; strongest realms act
    first (host size desc, then name) so the order is deterministic. Only WINNABLE wars launch (the
    M3.4/M3.5 guard — no suicidal wars). Caller gates on `empire_on`, so an off run never calls this
    and stays byte-identical to v1.
    """
    before = len(state["events"])
    tribute(state, turn)
    _check_fragmentation(state, turn)
    # --debug-war: a pure OBSERVATION of the gate below (why a war does or does not fire this
    # turn). Writes to stderr only — never to world_state, never to the event log — so a run with
    # it on is byte-identical to one with it off. Off by default; `dbg` is a no-op then.
    dbg = _war_debug(state, turn)

    # War: a SOVEREIGN king (independent realm) marches on the strongest neighbour it can beat. The
    # attacker's assessable host and the defender's defendable host are both dry-run LOYAL counts, so
    # a kingdom with disloyal vassals assesses (and fields) a SMALLER host. Strongest attacker first.
    kings = [k for k in sorted(state.get("kingdoms", {})) if is_sovereign(state, k)
             and _find(state, k) is not None]
    # A realm's fate is not decided twice in one turn. Without this, a single turn CASCADES: the
    # strongest crown wins its war and is left exhausted (casualties taken, chest spent), the next
    # crown down the list sees a host of zero and takes the victor AND everything it just won, and
    # a three-realm world collapses into one empire between two frames. A crown that has just
    # fought — as attacker or defender — is spent for the turn; the rival must wait until next
    # turn, by which point the defender's host is a real number again. Hosts recover, so nothing
    # is made permanently immune; the conquest simply takes the turn it should always have taken.
    fought: set[str] = set()
    for attacker_name in sorted(kings, key=lambda k: (-imperial_host_size(state, _find(state, k)), k)):
        if attacker_name in fought:
            continue
        attacker = _find(state, attacker_name)
        # M4.3 REGENCY: a DEPENDENT child-king (or child-emperor) holds its realm but wages
        # no war — the existing is_dependent_child gate keeps its powers dormant until it
        # comes of age (a no-op when lineage is off, so byte-identical there).
        if world.is_dependent_child(attacker, state):
            dbg(attacker_name, None, None, None, "REGENCY (a dependent child-king wages no war)")
            continue
        att_strength = imperial_host_size(state, attacker)
        # Target the strongest neighbour the attacker can still out-field (deterministic, winnable).
        # M4.13 DIPLOMACY (gated, lazily imported -> byte-identical off): a NON-AGGRESSION pact bars the
        # war outright, and an alliance is DETERRENCE — the attacker weighs the defender's host PLUS every
        # honouring ally's, so it refrains from a war it would lose against the combined defence.
        diplo = state.get("diplomacy_on")
        coal = state.get("coalitions_on")
        minds = state.get("minds_on")
        if diplo:
            from sim import diplomacy
        if coal:
            from sim import coalitions
        if minds:
            from llm import mind
        targets = []
        neighbours = _kingdom_neighbours(state, attacker_name)
        if not neighbours:
            dbg(attacker_name, None, att_strength, None, "NO NEIGHBOUR within KINGDOM_REACH")
        for t in neighbours:
            if t in fought:
                dbg(attacker_name, t, att_strength, None, "SPENT — it already fought this turn")
                continue
            if diplo and diplomacy.war_forbidden(state, attacker_name, t):
                dbg(attacker_name, t, att_strength, None, "BARRED by a non-aggression pact")
                continue
            # M4.15: two coalition members do not fight each other while the common hegemon threatens
            # them (fear suspends the feud). Gated -> byte-identical off.
            if coal and coalitions.allied_against_hegemon(state, attacker_name, t):
                dbg(attacker_name, t, att_strength, None, "SUSPENDED — both fear the hegemon")
                continue
            def_size = (diplomacy.defensive_host_size(state, t) if diplo
                        else imperial_host_size(state, _find(state, t)))
            # M5.1 PIVOT: the launch is decisive when the assessed edge is large (the math stands); when
            # the two hosts are within a hair (|margin| <= WAR_BAND) the KING's own character tilts the
            # call — a bold, competitive crown marches on even odds; a cautious one holds despite a
            # slim lead. Off / out-of-band -> exactly `att_strength > def_size` (byte-identical to v1).
            go = att_strength > def_size
            if minds:
                go, _ = mind.tilt(state, attacker_name, "war", att_strength - def_size, go,
                                  {"att": att_strength, "def": def_size, "target": t}, turn)
            dbg(attacker_name, t, att_strength, def_size,
                "GO — can out-field the defence" if go else "HOLD — cannot out-field the defence")
            if go:
                targets.append(t)
        if not targets:
            continue
        target = max(targets, key=lambda t: (imperial_host_size(state, _find(state, t)), t))
        dbg(attacker_name, target, att_strength, None, "WAR LAUNCHED")
        wage_war(state, attacker_name, target, turn)
        fought.update((attacker_name, target))
    return state["events"][before:]
