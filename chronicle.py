"""
chronicle.py
============

THE CHRONICLE — the world writes its own history (V2 milestone M4.16, CLOSES Arc 6 and COMPLETES the
v3 plan). On top of ALL prior milestones (Arcs 1-5, Phases 0-3).

The historical step M4.16 makes — the civilization becomes LEGIBLE AS HISTORY
----------------------------------------------------------------------------
Thirty milestones have generated rich history — conquerors (M3.4), revolutionaries (M4.6), prophets
(M4.8), dynasts (M4.3), wars, revolts, coronations, dynasties, beliefs, eras, coalitions — all in
world_state["events"]. M4.16 READS that record and composes it into a readable SAGA: named GREAT
FIGURES with archetypes and epithets derived from their DEEDS, named EVENTS, and dynastic HOUSE
histories. It turns the sim from something WATCHED into something that TELLS ITS STORY.

This milestone SURFACES existing history — it adds NO simulation mechanics. The structured chronicle is
READ-ONLY on the sim: it reads world_state["events"] and current state, and writes ONLY its own record
in world_state["chronicle"] (never mutating any sim field), so a --chronicle run is byte-identical.

HISTORY IS WRITTEN BY THE LITERATE (composes with M4.10)
-------------------------------------------------------
The chronicle ACCUMULATES INCREMENTALLY each turn (a cursor over the event log — never a retrospective
full-log scan). Fidelity depends on LITERACY: an event whose settlement is LITERATE at the time (M4.10
writing) is recorded fully — names, details, a named figure; an event in an ILLITERATE age enters only
as thin LEGEND (anonymized: "a warlord seized a town; the name is lost"). So a run's saga begins as
murky PREHISTORY and SHARPENS into detailed HISTORY at the moment writing emerges — the M4.10
recorded-history power paying off. (With writing never enabled, the whole saga is oral legend.)

The optional --narrate LLM layer is walled off (see `narrator.py`): it renders the SAME structured
entries as prose and NEVER touches the sim, determinism, or this structured record. Everything here is
ZERO LLM and deterministic.
"""

from __future__ import annotations

import re
from typing import Any

# --- Epithet / archetype tuning (deterministic thresholds) -------------------
CONQUEROR_MANY = 2       # this many conquests -> "the Conqueror"
GRASPING_LEVIES = 3      # this many recorded heavy levies -> "the Grasping"
LONG_REIGN = 15          # a reign this long (turns), untainted by heavy levies -> "the Just"

_BLANK_DEEDS = {"conquests": 0, "wars": 0, "uprisings": 0, "prophecies": 0, "crowns": 0, "levies": 0}


def _chron(state: dict[str, Any]) -> dict[str, Any]:
    return state.setdefault("chronicle", {
        "cursor": 0, "figures": {}, "events": [], "houses": {}, "house_of": {},
        "births": {}, "literacy_dawn": None, "motive_diag": []})


def _figure(ch: dict[str, Any], name: str, turn: int) -> dict[str, Any]:
    fig = ch["figures"].get(name)
    if fig is None:
        fig = {"name": name, "archetype": "figure", "deeds": dict(_BLANK_DEEDS),
               "first": turn, "last": turn, "house": None, "fell": None}
        ch["figures"][name] = fig
    fig["last"] = turn
    return fig


def _set_archetype(fig: dict[str, Any]) -> None:
    """The figure's defining role, by precedence over what they DID (military deeds define most)."""
    d = fig["deeds"]
    if d["conquests"] + d["wars"] >= 1:
        fig["archetype"] = "conqueror"
    elif d["uprisings"] >= 1:
        fig["archetype"] = "revolutionary"
    elif d["prophecies"] >= 1:
        fig["archetype"] = "prophet"
    elif d["crowns"] >= 1:
        fig["archetype"] = "dynast"


def epithet(fig: dict[str, Any]) -> str:
    """A DETERMINISTIC epithet from the figure's deed pattern (no LLM). Documented rule order."""
    d = fig["deeds"]
    reign = fig["last"] - fig["first"]
    fell = fig["fell"] or ""
    if d["prophecies"] >= 1:
        return "the Prophet"
    if d["uprisings"] >= 1:
        return "the Liberator"          # led a winning uprising / freed a town
    if "deposed" in fell and d["levies"] >= 1:
        return "the Grasping"           # over-taxed into revolt
    if d["conquests"] >= CONQUEROR_MANY or d["wars"] >= 1:
        return "the Conqueror"
    if d["levies"] >= GRASPING_LEVIES:
        return "the Grasping"
    if "extinguished" in fell:
        return "the Last"
    if d["crowns"] >= 1 and reign >= LONG_REIGN and d["levies"] == 0:
        return "the Just"               # a long, untainted reign
    if d["conquests"] >= 1:
        return "the Conqueror"
    if d["crowns"] >= 1:
        return "the Crowned"
    return "the Notable"


def titled(fig: dict[str, Any]) -> str:
    return f"{fig['name']} {epithet(fig)}"


# --- Fidelity: literacy decides history vs legend ----------------------------
def _fidelity(state: dict[str, Any], sid: "str | None") -> str:
    """'history' if the record can be written, else 'legend' (M4.10 literacy). A NAMED settlement must be
    LITERATE; a world-spanning event with no single settlement (`sid` None) is history once writing exists
    ANYWHERE (a literate world keeps records of great events). Pre-writing everything is legend."""
    try:
        import writing
        if sid is not None:
            return "history" if writing.is_literate(state, sid) else "legend"
        if any(writing.is_literate(state, s) for s in state.get("settlements", {})):
            return "history"
    except Exception:
        pass
    return "legend"


_ANON = {"conqueror": "a warlord", "war": "a great war", "uprising": "a revolt",
         "coronation": "a king", "prophet": "a holy figure", "fall": "a fallen power"}


def _record_event(ch: dict[str, Any], turn: int, kind: str, name: str, detail: str, fidelity: str,
                  actors: "tuple" = ()) -> None:
    if fidelity == "legend":
        name = {"war": "a forgotten war", "uprising": "a forgotten revolt",
                "coronation": "an unremembered crowning", "conquest": "a lost conquest",
                "extinction": "a vanished line", "hegemon_fall": "the fall of a great power",
                "breakaway": "a forgotten secession", "faith": "the rise of a creed"}.get(kind, "an event lost to time")
        detail = "its names are lost to the preliterate dark"
    if fidelity == "history" and ch["literacy_dawn"] is None:
        ch["literacy_dawn"] = turn      # the record turns to true history the moment a literate age can write it
    ch["events"].append({"turn": turn, "kind": kind, "name": name, "detail": detail,
                         "fidelity": fidelity, "actors": list(actors)})


# --- House assembly ----------------------------------------------------------
def _house_for(ch: dict[str, Any], predecessor: str) -> str:
    """The house a crown belongs to: the predecessor's house, or the predecessor as its FOUNDER."""
    return ch["house_of"].get(predecessor, predecessor)


def _ensure_house(ch: dict[str, Any], founder: str, turn: int) -> dict[str, Any]:
    h = ch["houses"].get(founder)
    if h is None:
        h = {"founder": founder, "members": {founder}, "crowns": 0, "fell": None, "founded": turn}
        ch["houses"][founder] = h
        ch["house_of"][founder] = founder
    return h


# --- The per-turn incremental pass -------------------------------------------
# Compiled patterns for the events that make HISTORY (figures / named events / houses).
_P_MONARCH = re.compile(r"^(\S+) .*-> MONARCH of (\S+)$")
_P_WAR = re.compile(r"^KING (\S+) DEFEATED (\S+) in war")
_P_SUCCEED = re.compile(r"^(\S+) succeeded (\S+) as \[(.*?)\] \(eldest (\w+)\)")
_P_EXTINCT = re.compile(r"^the line of (\S+) is extinguished")
_P_UPRISING_WIN = re.compile(r"^the UPRISING in (\S+) TRIUMPHED — (\w+) (\S+) is DEPOSED; (\S+) ")
_P_UPRISING_LED = re.compile(r"^(\S+) led the rising in (\S+)")
_P_UPRISING_CRUSH = re.compile(r"^the UPRISING in (\S+) was CRUSHED — (\w+) (\S+) holds")
_P_PROPHET = re.compile(r"^(\S+) arose as prophet of (.+)$")
_P_LEVY = re.compile(r"^MONARCH (\S+) levied [\d.]+ from (\S+)")
_P_BIRTH = re.compile(r"^(\S+) was born to (\S+) and (\S+) in (\S+)")
_P_WRITING = re.compile(r"^(\S+) devised WRITING in (\S+)")
_P_ERA = re.compile(r"^(\S+) entered the (.+)$")
_P_FAITH = re.compile(r"^(.+?) took root in (\S+)")
_P_HEGEMON_FALL = re.compile(r"DEFEATED hegemon (\S+) ")
# A vassal (M3.5 realm) or subject-king (M3.6 empire) reclaiming independence — the BREAKAWAY pivot's
# event. Both event strings share "<name> BROKE AWAY from <lord>'s realm|empire".
_P_BREAKAWAY = re.compile(r"^(?:subject-king )?(\S+) BROKE AWAY from (\S+)'s (?:realm|empire)")
_P_TURN = re.compile(r"^turn (\d+): (.*)$")


def _process(state: dict[str, Any], ch: dict[str, Any], turn: int, body: str) -> None:
    """Classify ONE event body (the text after 'turn N: ') into the chronicle. Pure over ch + reads."""
    m = _P_MONARCH.match(body)
    if m and "levied" not in body:
        actor, sid = m.group(1), m.group(2)
        fid = _fidelity(state, sid)
        if fid == "history":
            fig = _figure(ch, actor, turn); fig["deeds"]["conquests"] += 1; _set_archetype(fig)
            if actor not in ch["house_of"]:
                _ensure_house(ch, actor, turn)          # a conqueror founds a line by taking a crown
        _record_event(ch, turn, "conquest", f"{actor}'s Seizure of {sid}",
                      f"{actor} took {sid} by force", fid, (actor,))
        return

    m = _P_WAR.match(body)
    if m:
        victor, loser = m.group(1), m.group(2)
        fid = _fidelity(state, _home_of(state, victor))
        if fid == "history":
            fig = _figure(ch, victor, turn); fig["deeds"]["wars"] += 1; _set_archetype(fig)
        detail = f"KING {victor} defeated {loser} and forged an empire"
        _record_event(ch, turn, "war", f"{victor}'s Conquest of {loser}'s Kingdom",
                      detail + _motive(state, turn, figure=victor, pivot="war"), fid, (victor, loser))
        return

    m = _P_SUCCEED.match(body)
    if m:
        heir, pred, titles = m.group(1), m.group(2), m.group(3)
        fid = _fidelity(state, _first_sid(titles) or _home_of(state, heir))
        if fid == "history":
            fig = _figure(ch, heir, turn); fig["deeds"]["crowns"] += 1; _set_archetype(fig)
            house = _house_for(ch, pred)
            h = _ensure_house(ch, house, turn)
            h["members"].add(heir); h["crowns"] += 1
            ch["house_of"][heir] = house
        _record_event(ch, turn, "coronation", f"the Crowning of {heir}",
                      f"{heir} succeeded {pred} as [{titles}]", fid, (heir, pred))
        return

    m = _P_UPRISING_WIN.match(body)
    if m:
        sid, _kind, deposed, liberator = m.group(1), m.group(2), m.group(3), m.group(4)
        fid = _fidelity(state, sid)
        if fid == "history":
            fig = _figure(ch, liberator, turn); fig["deeds"]["uprisings"] += 1; _set_archetype(fig)
            if deposed in ch["figures"]:
                ch["figures"][deposed]["fell"] = "deposed in an uprising"
            hz = ch["house_of"].get(deposed)
            if hz in ch["houses"]:
                ch["houses"][hz]["fell"] = "ended by revolt"
        detail = f"the people of {sid} rose, deposed {deposed}, and {liberator} took power"
        _record_event(ch, turn, "uprising", f"the {sid} Uprising",
                      detail + _motive(state, turn, sid=sid, pivot="uprising"), fid, (liberator, deposed))
        return

    m = _P_UPRISING_LED.match(body)
    if m:
        leader, sid = m.group(1), m.group(2)
        if _fidelity(state, sid) == "history":
            fig = _figure(ch, leader, turn); fig["deeds"]["uprisings"] += 1; _set_archetype(fig)
        return

    m = _P_UPRISING_CRUSH.match(body)
    if m:
        sid = m.group(1)
        # A failed rising had a reason to rise as much as a successful one — wire the motive here too
        # (the ringleader's rise decision is recorded the same turn, so the same sid lookup finds it).
        _record_event(ch, turn, "uprising", f"the {sid} Uprising (crushed)",
                      f"the people of {sid} rose and were put down"
                      + _motive(state, turn, sid=sid, pivot="uprising"), _fidelity(state, sid))
        return

    m = _P_PROPHET.match(body)
    if m:
        prophet, faith = m.group(1), m.group(2)
        fid = _fidelity(state, _home_of(state, prophet))
        if fid == "history":
            fig = _figure(ch, prophet, turn); fig["deeds"]["prophecies"] += 1; _set_archetype(fig)
        _record_event(ch, turn, "prophet", f"the Advent of {faith}",
                      f"{prophet} arose as prophet of {faith}", fid, (prophet,))
        return

    m = _P_LEVY.match(body)
    if m:
        monarch, sid = m.group(1), m.group(2)
        if _fidelity(state, sid) == "history":
            fig = _figure(ch, monarch, turn)      # a levying crown is a figure of record
            fig["deeds"]["levies"] += 1
            if fig["archetype"] == "figure":
                fig["archetype"] = "dynast"        # a ruler, even one known only for its exactions
        return

    m = _P_EXTINCT.match(body)
    if m:
        line = m.group(1)
        fid = _fidelity(state, _first_sid(body) or _home_of(state, line))
        if fid == "history":
            if line in ch["figures"]:
                ch["figures"][line]["fell"] = "the line was extinguished"
            h = ch["house_of"].get(line)
            if h in ch["houses"]:
                ch["houses"][h]["fell"] = "the line was extinguished"
        _record_event(ch, turn, "extinction", f"the End of the House of {line}",
                      f"the line of {line} died out", fid, (line,))
        return

    m = _P_HEGEMON_FALL.search(body)
    if m and "COALITION" in body:
        heg = m.group(1)
        if heg in ch["figures"]:
            ch["figures"][heg]["fell"] = "broken by a coalition"
        _record_event(ch, turn, "hegemon_fall", f"the Fall of the Hegemon {heg}",
                      f"a coalition of the many broke the empire of {heg}",
                      _fidelity(state, _home_of(state, heg)), (heg,))
        return

    m = _P_BREAKAWAY.match(body)
    if m:
        breaker, lord = m.group(1), m.group(2)
        # Fidelity from the seceding domain (the sid named in a realm break), else the breaker's home.
        fid = _fidelity(state, _first_sid(body) or _home_of(state, breaker))
        _record_event(ch, turn, "breakaway", f"the Secession of {breaker}",
                      f"{breaker} broke away from {lord} and reclaimed independence"
                      + _motive(state, turn, figure=breaker, pivot="breakaway"), fid, (breaker, lord))
        return

    m = _P_BIRTH.match(body)
    if m:
        ch["births"][m.group(1)] = (m.group(2), m.group(3), turn, m.group(4))
        return

    m = _P_WRITING.match(body)
    if m:
        if ch["literacy_dawn"] is None:
            ch["literacy_dawn"] = turn
            _record_event(ch, turn, "writing", "the Invention of Writing",
                          f"{m.group(1)} of {m.group(2)} devised writing — history begins", "history")
        return

    m = _P_ERA.match(body)
    if m and "Age" in m.group(2):
        sid, era = m.group(1), m.group(2)
        _record_event(ch, turn, "era", f"{sid} enters {era}", f"{sid} reached {era}",
                      _fidelity(state, sid))
        return

    m = _P_FAITH.match(body)
    if m:
        faith, sid = m.group(1), m.group(2)
        _record_event(ch, turn, "faith", f"the Rise of {faith}",
                      f"{faith} took root in {sid}", _fidelity(state, sid), ())
        return


def _motive(state: dict[str, Any], turn: int, figure: "str | None" = None,
            sid: "str | None" = None, pivot: "str | None" = None) -> str:
    """The WHY behind a pivot decision (M5.1), as a trailing clause — or "" when minds are off/absent.

    If a figure consulted its mind at the pivot that produced this event, its recorded reason is
    surfaced so the saga records not just WHAT happened but WHY ("...saying 'the odds were even and
    fortune favours the bold'"). Read-only on the SIM (pulls the reason mind.py already logged); empty
    for a minds-off run, so its saga text is unchanged.

    The lookup honours each pivot's TIMING. War and uprising events fire the very turn the mind is
    consulted, so an exact same-turn match is right. A BREAKAWAY, though, has hysteresis: the mind is
    consulted when loyalty first slips into the close band, but the secession only fires PATIENCE turns
    later — so it needs a lookback (mind.breakaway_motive), or the reason is silently dropped. Whichever
    lookup is used, `_diagnose_motive` records whether the (missing) motive was 'no mind consulted' or a
    genuine 'lookup failed', so a blank saga entry is diagnosable rather than mysterious."""
    if not state.get("minds_on"):
        return ""
    reason = None
    try:
        import mind
        if pivot == "breakaway" and figure:
            reason = mind.breakaway_motive(state, turn, figure)   # hysteresis-tolerant lookback
        elif figure:
            reason = mind.motive_for(state, turn, figure)
        else:
            reason = mind.motive_at(state, turn, sid)
    except Exception:
        reason = None
    _diagnose_motive(state, turn, pivot, figure, sid, reason)
    return f", saying “{reason}”" if reason else ""


def _diagnose_motive(state: dict[str, Any], turn: int, pivot: "str | None",
                     figure: "str | None", sid: "str | None", reason: "str | None") -> None:
    """Record WHY a pivot event did/didn't carry a motive, so a blank one is diagnosable (M5.1).

    Three outcomes: 'attached' (a motive was found and written), 'no_consult' (no mind was ever consulted
    for this pivot — it resolved decisively outside the close band, so the empty motive is CORRECT), and
    'lookup_failed' (a mind WAS consulted here yet no motive attached — a wiring bug worth surfacing).
    Writes ONLY the chronicle's own diag list (never a sim field), so determinism/byte-identity hold; a
    minds-off run never reaches here. Read by verify/tests (and printable in the cost readout)."""
    if pivot is None:
        return
    consulted = False
    try:
        import mind
        # Match the motive-lookup window per pivot (breakaway lags by its hysteresis; war/uprising fire
        # the same turn). acted_only: a consult counts only if it DECIDED TO ACT — the very case that
        # records a motive — so a decisive break preceded by a DECLINED consult reads 'no_consult', not
        # a false 'lookup_failed'. A true 'lookup_failed' (an acted consult whose motive never attaches)
        # stays a tripwire for a future key/window regression.
        window = mind._breakaway_window() if pivot == "breakaway" else 0
        consulted = mind.pivot_consulted(state, turn, pivot, figure=figure, sid=sid,
                                         window=window, acted_only=True)
    except Exception:
        consulted = False
    status = "attached" if reason else ("lookup_failed" if consulted else "no_consult")
    _chron(state).setdefault("motive_diag", []).append(
        {"turn": turn, "pivot": pivot, "figure": figure, "sid": sid, "status": status})


def _home_of(state: dict[str, Any], king: str) -> "str | None":
    return state.get("kingdoms", {}).get(king, {}).get("home")


def _first_sid(titles: str) -> "str | None":
    m = re.search(r"\b(S\d{3})\b", titles)
    return m.group(1) if m else None


def update(state: dict[str, Any], turn: int) -> None:
    """Advance the chronicle one turn (M4.16): fold THIS turn's new events into the structured record.

    Reads world_state["events"] from a cursor (never a full retrospective scan) and writes ONLY
    world_state["chronicle"] — never any sim field, so the run is byte-identical. Fidelity (history vs
    legend) is decided by literacy AT THIS MOMENT, so pre-writing events enter as legend and later ones
    as full history. ZERO LLM, ZERO RNG. Caller gates on `chronicle_on`."""
    ch = _chron(state)
    events = state.get("events", [])
    for e in events[ch["cursor"]:]:
        m = _P_TURN.match(e)                       # each event carries its OWN turn (robust to batching)
        if m:
            _process(state, ch, int(m.group(1)), m.group(2))
    ch["cursor"] = len(events)


# --- Derived read-outs (pure; the saga / export / renderer panel read these) --
def generations(ch: dict[str, Any], house: dict[str, Any]) -> int:
    """Depth of the house's bloodline among its crowned members (from the recorded parent links)."""
    births = ch["births"]
    members = house["members"]

    def depth(name: str, seen: frozenset) -> int:
        if name in seen:
            return 0
        b = births.get(name)
        parents = [p for p in (b[0], b[1]) if b and p in members] if b else []
        return 1 + max((depth(p, seen | {name}) for p in parents), default=0)
    return max((depth(m, frozenset()) for m in members), default=1)


def great_figures(state: dict[str, Any]) -> list:
    """The chronicled figures, most-deeds first (then name) — each with its derived epithet/archetype."""
    figs = list(_chron(state)["figures"].values())
    return sorted(figs, key=lambda f: (-sum(f["deeds"].values()), f["name"]))


def houses(state: dict[str, Any]) -> list:
    return sorted(_chron(state)["houses"].values(), key=lambda h: (-h["crowns"], h["founder"]))


def saga(state: dict[str, Any]) -> list:
    """The chronological list of named historical entries (the timeline of the age)."""
    return sorted(_chron(state)["events"], key=lambda e: (e["turn"], e["name"]))


# --- Presentation: markdown export + renderer-panel lines ---------------------
_ARCH_TITLE = {"conqueror": "Conqueror", "revolutionary": "Revolutionary", "prophet": "Prophet",
               "dynast": "Dynast", "figure": "Figure"}


def export_markdown(state: dict[str, Any]) -> str:
    """The full structured saga of the run as readable markdown — the history this world wrote (ZERO LLM)."""
    ch = _chron(state)
    dawn = ch["literacy_dawn"]
    out: list[str] = ["# The Chronicle of the Age", ""]
    out.append(f"*Writing was invented in turn {dawn}; before it, the record is legend.*"
               if dawn is not None else "*This age never learned to write; its whole story is legend.*")
    out.append("")

    out.append("## The Great Figures")
    figs = great_figures(state)
    if not figs:
        out.append("*(none recorded in living memory)*")
    for f in figs:
        d = f["deeds"]
        deeds = ", ".join(f"{v} {k}" for k, v in d.items() if v) or "no recorded deeds"
        fell = f["fell"] or "passed from the record"
        out.append(f"- **{titled(f)}** — {_ARCH_TITLE.get(f['archetype'], 'Figure')}, "
                   f"active turns {f['first']}–{f['last']}. Deeds: {deeds}. Fate: {fell}.")
    out.append("")

    out.append("## The Houses")
    hs = houses(state)
    if not hs:
        out.append("*(no dynasty took root)*")
    for h in hs:
        fell = h["fell"] or "endures"
        out.append(f"- **the House of {h['founder']}** — founded turn {h['founded']}, "
                   f"{generations(ch, h)} generation(s), {h['crowns']} crown(s) passed, "
                   f"{len(h['members'])} titled kin ({', '.join(sorted(h['members']))}). Fate: {fell}.")
    out.append("")

    out.append("## The Saga")
    for e in saga(state):
        tag = "" if e["fidelity"] == "history" else "  _(legend)_"
        out.append(f"- **Turn {e['turn']} — {e['name']}**: {e['detail']}.{tag}")
    out.append("")
    return "\n".join(out)


def saga_lines(state: dict[str, Any], limit: int = 14) -> list[str]:
    """Compact recent saga lines for a read-only renderer panel (most recent last). Pure read."""
    lines = [f"T{e['turn']} {e['name']}" + ("" if e["fidelity"] == "history" else " (legend)")
             for e in saga(state)]
    return lines[-limit:]
