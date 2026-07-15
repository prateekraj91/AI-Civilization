"""
diplomacy.py
============

RELATIONS & TREATIES — kingdoms gain a second verb (V2 milestone M4.13, OPENS Arc 5: Diplomacy & the
Interstate System). On top of Arc 4 (eras/metallurgy/writing), Arc 3 (culture), Arc 2 (revolt), Arc 1
(dynasties) and all of Phases 0-3.

The historical step M4.13 makes — war stops being a kingdom's only language
--------------------------------------------------------------------------
Kingdoms had exactly ONE interaction: opportunistic WAR (M3.6). M4.13 gives every kingdom-PAIR a
STANCE (hostile/neutral/friendly) DERIVED from their shared history, and TREATIES — non-aggression
pacts and defensive alliances — that CONSTRAIN the existing war machinery. A pact stops a war the
M3.6 loop would otherwise launch; an alliance brings a friend's host to a defender's aid. This is the
substrate the later milestones build on (trade M4.14, coalitions M4.15).

SCOPE — M4.13 is pairwise STANCE + NON-AGGRESSION pacts + DEFENSIVE alliances, and ONLY that. It does
NOT build inter-kingdom TRADE (M4.14) or anti-hegemon COALITIONS (M4.15) — stated as a boundary (a
`trade volume -> warmer` seam is marked in `_recompute_stance` for M4.14). Treaties EMERGE from stance
thresholds + conditions — NO agent deliberation, NO LLM — exactly as every institution in this sim does.

How it works (emergent; zero LLM; deterministic stance math)
------------------------------------------------------------
1. STANCE IS DERIVED FROM HISTORY. Each unordered pair of sovereign kings carries an integer score in
   `world_state["diplomacy"]["stance"]`. Each turn: a pair sharing CULTURE (M4.7-9) or FAITH (M4.8) WARMS
   toward FRIENDLY (IDENTITY_WARMTH/turn, capped); an unrelated pair DECAYS toward NEUTRAL (old feeling
   fades). One-off shocks are applied as events: a WAR between them (record_war, from empire.wage_war)
   drops it toward HOSTILE; a BROKEN treaty (betrayal) drops it further. `stance(k1,k2)` maps the score
   to hostile/neutral/friendly.
2. NON-AGGRESSION PACT. A pair whose stance is warm enough (>= PACT_FORM) and not mid-conflict FORMS a
   pact; while it holds, the M3.6 war loop reads `has_pact` and REFRAINS from attacking. It BREAKS when
   the stance sours past PACT_BREAK — a distinct BETRAYAL event that sours the pair further. Treaties are
   PERSONAL to the ruler: on a king's death/subjugation they LAPSE (a new reign must re-earn them), so a
   COLD heir does not renew the pact and the war his father prevented reopens (composes with M4.3).
3. DEFENSIVE ALLIANCE. A warmer pair (>= ALLIANCE_FORM) forms an alliance; when one ally is attacked, the
   other CONTRIBUTES its whole loyal host to the DEFENCE (`empire.imperial_host`) — so the attacker faces
   the COMBINED hosts. But honour is CONDITIONAL: an ally whose stance with the defender has soured below
   HONOUR_AT does NOT answer, and alliances are never ironclad.

Cost & determinism
------------------
ZERO LLM and ZERO RNG — stance is deterministic integer math over sorted pairs. A run with the system OFF
never calls `update` (no "diplomacy" key), the war loop never checks a pact, and wage_war never adds an
ally, so it is byte-identical to v1. Imports world; lazily imports empire/culture/religion (the systems it
reads) so there is no load-time cycle.
"""

from __future__ import annotations

from typing import Any

import world

# --- Tunable constants -------------------------------------------------------
FRIENDLY_AT = 3           # stance score at/above which a pair is FRIENDLY
HOSTILE_AT = -3           # stance score at/below which a pair is HOSTILE (else NEUTRAL)

IDENTITY_WARMTH = 1       # per-turn warming for a pair sharing culture/faith...
IDENTITY_CAP = 6          # ...capped here (a shared identity trends friendly, not infinitely warm)
STANCE_DECAY = 1          # per-turn drift toward neutral for an unrelated pair (old feeling fades slowly)

WAR_PENALTY = 6           # stance hit when the two kingdoms go to war (a lasting grievance)
BETRAYAL_PENALTY = 4      # extra stance hit when a treaty is broken (betrayal outlasts the pact)

PACT_FORM_AT = 2          # stance at/above which a NON-AGGRESSION pact forms...
PACT_BREAK_AT = -2        # ...and at/below which it BREAKS (betrayal)
ALLIANCE_FORM_AT = 4      # stance at/above which a DEFENSIVE alliance forms (warmer than a pact)...
ALLIANCE_BREAK_AT = 0     # ...and at/below which it dissolves
HONOUR_AT = 0             # an ally answers a call to defence only while its stance is at/above this


def _pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))  # canonical unordered-pair key


def _find(state: dict[str, Any], name: str) -> "Any | None":
    return next((x for x in state["agents"] if x.name == name), None)


def _dip(state: dict[str, Any]) -> dict[str, Any]:
    return state.setdefault("diplomacy", {"stance": {}, "pacts": set(), "alliances": set()})


# --- Stance read-outs --------------------------------------------------------
def stance_score(state: dict[str, Any], k1: str, k2: str) -> int:
    return _dip(state)["stance"].get(_pair(k1, k2), 0)


def stance(state: dict[str, Any], k1: str, k2: str) -> str:
    s = stance_score(state, k1, k2)
    if s <= HOSTILE_AT:
        return "hostile"
    if s >= FRIENDLY_AT:
        return "friendly"
    return "neutral"


def has_pact(state: dict[str, Any], k1: str, k2: str) -> bool:
    return _pair(k1, k2) in _dip(state)["pacts"]


def has_alliance(state: dict[str, Any], k1: str, k2: str) -> bool:
    return _pair(k1, k2) in _dip(state)["alliances"]


# --- History shocks (called from the systems that make history) --------------
def record_war(state: dict[str, Any], k1: str, k2: str, turn: int) -> None:
    """A war between two kingdoms leaves a lasting grievance — the stance drops toward hostile. Called
    from `empire.wage_war` (lazily, gated on diplomacy_on), so it fires exactly when a war actually does."""
    dip = _dip(state)
    key = _pair(k1, k2)
    dip["stance"][key] = dip["stance"].get(key, 0) - WAR_PENALTY


# --- The per-turn engine -----------------------------------------------------
def _sovereign_kings(state: dict[str, Any]) -> list[str]:
    import empire
    return [k for k in sorted(state.get("kingdoms", {}))
            if empire.is_sovereign(state, k) and _find(state, k) is not None]


def _shared_identity(state: dict[str, Any], k1: str, k2: str) -> bool:
    """True if the two kings share a CULTURE (M4.7-9) or a FAITH (M4.8) — the warmth of kinship between
    crowns. Pure read; only consults culture/religion when those systems are on (else no identity signal)."""
    a1, a2 = _find(state, k1), _find(state, k2)
    if a1 is None or a2 is None:
        return False
    if state.get("culture_on"):
        import culture
        b1, b2 = state.get("beliefs", {}).get(k1), state.get("beliefs", {}).get(k2)
        if b1 and b2 and culture.same_culture(a1, a2, state):
            return True
    if state.get("religion_on"):
        import religion
        homes = state.get("kingdoms", {})
        h1 = homes.get(k1, {}).get("home")
        h2 = homes.get(k2, {}).get("home")
        f1 = religion.faith_of_settlement(state, h1) if h1 else None
        f2 = religion.faith_of_settlement(state, h2) if h2 else None
        if f1 is not None and f1 is f2:
            return True
    # M4.14 SEAM: trade volume between the pair -> extra warmth, wired in when inter-kingdom trade lands.
    return False


def _recompute_stance(state: dict[str, Any], kings: list[str], turn: int) -> None:
    """Warm pairs that share identity toward friendly; decay unrelated pairs toward neutral. ZERO RNG."""
    dip = _dip(state)
    stances = dip["stance"]
    for i, k1 in enumerate(kings):
        for k2 in kings[i + 1:]:
            key = _pair(k1, k2)
            s = stances.get(key, 0)
            if _shared_identity(state, k1, k2):
                s = min(IDENTITY_CAP, s + IDENTITY_WARMTH)
            elif s > 0:
                s = max(0, s - STANCE_DECAY)
            elif s < 0:
                s = min(0, s + STANCE_DECAY)
            if s == 0:
                stances.pop(key, None)
            else:
                stances[key] = s


def _reconcile(state: dict[str, Any], kings: "set[str]") -> list[str]:
    """Drop stance/treaty entries that reference a king who is no longer a living sovereign — treaties
    are PERSONAL, so a dead/subjugated king's pacts LAPSE (his heir must re-earn them). Returns lapse events."""
    dip = _dip(state)
    events: list[str] = []
    for key in [k for k in dip["stance"] if k[0] not in kings or k[1] not in kings]:
        del dip["stance"][key]
    for kind in ("pacts", "alliances"):
        for key in [k for k in dip[kind] if k[0] not in kings or k[1] not in kings]:
            dip[kind].discard(key)
            label = "non-aggression pact" if kind == "pacts" else "defensive alliance"
            events.append(f"the {label} between {key[0]} and {key[1]} LAPSED (a crown changed hands)")
    return events


def update(state: dict[str, Any], turn: int) -> list[str]:
    """Advance diplomacy one turn (M4.13): reconcile, recompute stance, then form/break treaties by
    threshold. ZERO LLM, ZERO RNG. Caller gates on `diplomacy_on`, so an off run never calls this (no
    "diplomacy" key) and stays byte-identical. Returns the events logged."""
    dip = _dip(state)
    kings = _sovereign_kings(state)
    kingset = set(kings)
    events = _reconcile(state, kingset)
    _recompute_stance(state, kings, turn)

    # Form / break treaties from the current stance (deterministic thresholds; sorted pairs).
    for i, k1 in enumerate(kings):
        for k2 in kings[i + 1:]:
            key = _pair(k1, k2)
            s = stance_score(state, k1, k2)
            # Non-aggression pact.
            if key not in dip["pacts"] and s >= PACT_FORM_AT:
                dip["pacts"].add(key)
                events.append(f"{k1} and {k2} signed a NON-AGGRESSION PACT (stance {s})")
            elif key in dip["pacts"] and s <= PACT_BREAK_AT:
                dip["pacts"].discard(key)
                dip["alliances"].discard(key)
                dip["stance"][key] = s - BETRAYAL_PENALTY   # betrayal outlasts the pact
                events.append(f"{k1} and {k2}'s pact was BROKEN — a betrayal (stance fell to {s})")
            # Defensive alliance (a warmer tier; requires the pact-level trust too).
            if key not in dip["alliances"] and s >= ALLIANCE_FORM_AT:
                dip["alliances"].add(key)
                dip["pacts"].add(key)                       # an alliance implies non-aggression
                events.append(f"{k1} and {k2} formed a DEFENSIVE ALLIANCE (stance {s})")
            elif key in dip["alliances"] and s <= ALLIANCE_BREAK_AT:
                dip["alliances"].discard(key)
                events.append(f"{k1} and {k2}'s alliance dissolved (stance {s})")

    state.setdefault("events", []).extend(f"turn {turn}: {e}" for e in events)
    return events


# --- Hooks the war machinery reads (M3.6) ------------------------------------
def war_forbidden(state: dict[str, Any], attacker: str, defender: str) -> bool:
    """True if a non-aggression pact bars the M3.6 loop from launching this war. Pure read."""
    return has_pact(state, attacker, defender)


def defensive_allies(state: dict[str, Any], defender: str) -> list[str]:
    """The sovereign allies who will ANSWER a call to defend `defender` — allied AND still honouring
    (stance >= HONOUR_AT). A soured ally does not answer (honour is conditional). Sorted, pure read."""
    import empire
    dip = _dip(state)
    out = []
    for key in sorted(dip["alliances"]):
        ally = key[0] if key[1] == defender else key[1] if key[0] == defender else None
        if ally is None:
            continue
        if (empire.is_sovereign(state, ally) and _find(state, ally) is not None
                and stance_score(state, ally, defender) >= HONOUR_AT):
            out.append(ally)
    return out


def defensive_host_size(state: dict[str, Any], defender: str) -> int:
    """The defender's own dry-run host PLUS every honouring ally's — the size the M3.6 loop should weigh
    (an alliance DETERS an attack it would otherwise launch on the lone defender). Pure read."""
    import empire
    d = _find(state, defender)
    total = empire.imperial_host_size(state, d) if d is not None else 0
    for ally in defensive_allies(state, defender):
        a = _find(state, ally)
        if a is not None:
            total += empire.imperial_host_size(state, a)
    return total
