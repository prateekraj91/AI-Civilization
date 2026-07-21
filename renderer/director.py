"""
renderer/director.py
====================

THE SHOWCASE DIRECTOR — severity, captions and camera cues for `--showcase`.

Why this module exists
----------------------
The simulation is dramatic; the renderer is not. A 90-turn run treats the turn a dynasty
ends exactly like the turn six villagers taught each other to cook. This module is the
missing EDITOR: it reads the turn's events, decides how important the turn is, and hands
the renderer a pacing/camera/caption plan for it.

BOUNDARY (same rule as the rest of `renderer/`)
-----------------------------------------------
Pure, READ-ONLY, stdlib-only. It never mutates world_state, never imports a decision-logic
module, and never touches pygame. Everything here is a deterministic function of the event
strings plus a small amount of per-run memory (which world-firsts have already happened),
so a seeded run cuts the same way every time.

The typed-event boundary
------------------------
The engine emits events as plain strings (`"turn 3: KING Aldric DEFEATED Cyrus in war (...)"`).
Rather than sprinkle substring tests through the renderer, this module parses each line ONCE
at the boundary into a typed `Event` carrying a `kind` — and everything downstream (severity,
captions, camera focus) keys on `kind`, never on the text. The parse table below is the single
place where log wording matters; each pattern is ANCHORED to the start of the event body, and
each has a test pinning it to the exact string the engine emits.

This mirrors `chronicle.py`, which already turns the same string stream into typed records —
the established way this project types events without reaching into the sim.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

# --- Severity tiers ---------------------------------------------------------
# Ordered weakest -> strongest; `max()` over a turn's events gives the turn severity.
NOISE = "noise"
MINOR = "minor"
MAJOR = "major"
LEGENDARY = "legendary"

TIERS = (NOISE, MINOR, MAJOR, LEGENDARY)
_RANK = {t: i for i, t in enumerate(TIERS)}


def rank(tier: str) -> int:
    """The ordinal of a tier (NOISE=0 .. LEGENDARY=3); unknown tiers read as NOISE."""
    return _RANK.get(tier, 0)


class Event(NamedTuple):
    """One classified event. `focus` is the settlement the camera should fly to (or None);
    `actors` are the named figures involved, in the order the log names them."""
    turn: int
    kind: str
    severity: str
    focus: str | None
    actors: tuple[str, ...]
    fields: dict[str, str]
    raw: str


# --- The parse table --------------------------------------------------------
# (kind, anchored pattern). ORDER MATTERS: the first match wins, so specific forms come
# before the general ones they are a special case of (a coup before a plain seizure, an
# uprising outcome before the uprising itself).
#
# Named groups the rest of the module reads:
#   sid    -> the settlement to frame          a  / b   -> actors (attacker / defender)
#   who    -> a single actor                   x        -> a free field (era, amount, titles)
_NAME = r"[A-Za-z][\w'-]*"
_SID = r"S\w+"

_PATTERNS: tuple[tuple[str, str], ...] = (
    # -- LEGENDARY -----------------------------------------------------------
    ("war_won", rf"^KING (?P<a>{_NAME}) DEFEATED (?P<b>{_NAME}) in war"),
    ("war_failed", rf"^KING (?P<a>{_NAME})'s war on (?P<b>{_NAME}) FAILED"),
    ("dynasty_extinct", rf"^the line of (?P<who>{_NAME}) is extinguished; the crown of \[(?P<x>.*?)\] lies vacant"),
    ("empire_broken", rf"^subject-king (?P<who>{_NAME}) was freed from (?P<b>{_NAME})'s empire by the coalition"),
    # A crown seizing a town INTO a realm, and an aspirant seizing an unheld town, are both
    # conquests. The coup form (below, MAJOR) is matched first so it never lands here.
    ("realm_conquest", rf"^KING (?P<a>{_NAME}) (?:CONQUERED|accepted the submission of) (?P<sid>{_SID}) into the realm"),
    ("town_seized", rf"^(?P<a>{_NAME}) (?:seized|CONQUERED \(defeating (?P<b>{_NAME})'s followers in\)) (?P<sid>{_SID}) by force"),

    # -- MAJOR ---------------------------------------------------------------
    ("coup", rf"^(?P<a>{_NAME}) OVERTHREW (?P<b>{_NAME}) and seized (?P<sid>{_SID}) by force"),
    ("uprising_triumph", rf"^the UPRISING in (?P<sid>{_SID}) TRIUMPHED — (?P<x>\w+) (?P<b>{_NAME}) is DEPOSED; (?P<a>.+?) \((?P<n>\d+) risers fell\)"),
    ("uprising_crushed", rf"^the UPRISING in (?P<sid>{_SID}) was CRUSHED — (?P<x>\w+) (?P<b>{_NAME}) holds \((?P<n>\d+) guards \+ (?P<m>\d+) risers fell\)"),
    ("uprising_begins", rf"^UPRISING in (?P<sid>{_SID}) — (?P<n>\d+) risers rise against (?P<x>\w+) (?P<b>{_NAME}) \((?P<m>\d+) defenders"),
    ("uprising_leader", rf"^(?P<who>{_NAME}) led the rising in (?P<sid>{_SID})"),
    ("expropriation", rf"^the risers EXPROPRIATED (?P<who>{_NAME})'s hoard of (?P<n>[\d.]+) — split among (?P<m>\d+)"),
    ("breakaway_empire", rf"^subject-king (?P<who>{_NAME}) BROKE AWAY from (?P<b>{_NAME})'s empire"),
    ("breakaway_realm", rf"^(?P<who>{_NAME}) BROKE AWAY from (?P<b>{_NAME})'s realm — (?P<sid>{_SID}) is independent"),
    ("secession", rf"^(?P<sid>{_SID}) SECEDED from (?P<b>{_NAME})'s realm"),
    ("assault_repelled", rf"^(?P<a>{_NAME})'s assault on (?P<sid>{_SID}) was REPELLED"),
    ("host_repelled", rf"^KING (?P<a>{_NAME})'s host was REPELLED at (?P<sid>{_SID})"),
    ("era", rf"^(?P<sid>{_SID}) entered the (?P<x>.+?)$"),
    # World-FIRSTS are major; the same feat in the ninth town is ordinary progress. The
    # demotion is applied in `classify` (it needs per-run memory), not here.
    ("writing", rf"^(?P<who>{_NAME}) devised WRITING in (?P<sid>{_SID})"),
    ("weapons", rf"^(?P<who>{_NAME}) forged the first WEAPONS in (?P<sid>{_SID})"),
    ("succession", rf"^(?P<a>{_NAME}) succeeded (?P<b>{_NAME}) as \[(?P<x>.*?)\] \(eldest"),
    ("leader_consent", rf"^(?P<who>{_NAME}) emerged as leader of (?P<sid>{_SID})"),
    ("leader_displaced", rf"^(?P<a>{_NAME}) displaced (?P<b>{_NAME}) as leader of (?P<sid>{_SID})"),
    ("prophet", rf"^(?P<who>{_NAME}) arose as prophet of (?P<x>.+)$"),
    ("coalition_dissolved", rf"^the coalition against (?P<b>{_NAME}) DISSOLVED"),

    # -- MINOR ---------------------------------------------------------------
    ("metalworking", rf"^(?P<who>{_NAME}) mastered METALWORKING in (?P<sid>{_SID})"),
    ("written_law", rf"^(?P<who>{_NAME}) inscribed the written law of (?P<sid>{_SID})"),
    ("legitimacy_lost", rf"^(?P<who>{_NAME}) lost legitimacy as leader of (?P<sid>{_SID})"),
    ("faith_root", r"^(?P<x>.+?) took root in (?P<sid>" + _SID + r")"),
    ("birth", rf"^(?P<who>{_NAME}) was born to (?P<a>{_NAME}) and (?P<b>{_NAME}) in (?P<sid>{_SID})"),
    ("death", rf"^(?P<who>{_NAME}) died \((?P<x>[^)]*)\)"),
    ("alliance_formed", rf"^(?P<a>{_NAME}) and (?P<b>{_NAME}) formed an ALLIANCE"),
    ("betrayal", rf"^\*+ (?P<a>{_NAME}) BETRAYED the alliance with (?P<b>{_NAME})"),
    ("inheritance", rf"^(?P<a>{_NAME}) inherited (?P<n>[\d.]+) from (?P<b>{_NAME})"),
    ("teaching", rf"^(?P<a>{_NAME}) taught '(?P<x>[^']*)' to (?P<b>{_NAME})"),
    ("discovery", rf"^(?P<who>{_NAME}) discovered '(?P<x>[^']*)'"),
    ("came_of_age", rf"^(?P<who>{_NAME}) came of age"),
    ("theft", rf"^(?P<a>{_NAME}) stole food from (?P<b>{_NAME})"),
    ("law_inherited", rf"^(?P<a>{_NAME}) inherited the written law of (?P<sid>{_SID}) from (?P<b>{_NAME})"),
    ("law_retaught", rf"^(?P<sid>{_SID}) RE-TAUGHT '(?P<x>[^']*)' from its records"),

    # -- NOISE ---------------------------------------------------------------
    ("levy", rf"^MONARCH (?P<who>{_NAME}) levied (?P<n>[\d.]+) from (?P<sid>{_SID})"),
    ("tribute_imperial", r"^imperial tribute cascaded up:"),
    ("tribute_realm", rf"^tribute cascaded up (?P<sid>{_SID}):"),
    ("redistribution", rf"^(?P<who>{_NAME}) taxed (?P<n>\d+) wealthy followers"),
    ("estate_vanished", rf"^(?P<who>{_NAME})'s estate of (?P<n>[\d.]+) vanished"),
    ("estate_dropped", rf"^(?P<n>[\d.]+) of (?P<who>{_NAME})'s estate dropped"),
    ("talk", rf"^(?P<a>{_NAME}) talked to (?P<b>{_NAME}):"),
    ("reply", rf"^(?P<a>{_NAME}) (?:received from|heard) (?P<b>{_NAME})"),
    ("alliance_proposed", rf"^(?P<a>{_NAME}) proposed an alliance to (?P<b>{_NAME})"),
    ("trust", rf"^(?P<a>{_NAME}) trust in (?P<b>{_NAME}):"),
    ("seethes", rf"^(?P<who>{_NAME}) seethes with hunger"),
    # The discontent gauge ticking up under a lord's exactions. NOISE to the caption track — but
    # it is the PRESSURE the uprising eventually releases, so it carries its value for 5.4's
    # red-pulse overlay rather than being discarded.
    ("grievance", rf"^(?P<who>{_NAME}) seethes under (?P<b>{_NAME})'s (?:tribute|levies|taxes) \(discontent (?P<n>[\d.]+)\)"),
    ("escheat", rf"^(?P<who>{_NAME})'s estate of (?P<n>[\d.]+) escheated to (?P<b>{_NAME})"),
    ("savings", rf"^(?P<who>{_NAME}) drew on savings"),
    ("joined", rf"^(?P<who>{_NAME}) joined settlement (?P<sid>{_SID})"),
    ("god", r"^\[GOD\]"),
)

_COMPILED = tuple((kind, re.compile(pat)) for kind, pat in _PATTERNS)

# --- kind -> severity -------------------------------------------------------
SEVERITY: dict[str, str] = {
    # LEGENDARY — the run's headline turns.
    "war_won": LEGENDARY,
    "war_failed": LEGENDARY,
    "dynasty_extinct": LEGENDARY,
    "empire_broken": LEGENDARY,
    "realm_conquest": LEGENDARY,
    "town_seized": LEGENDARY,
    # MAJOR — the turns worth stopping the camera for.
    "coup": MAJOR,
    "uprising_triumph": MAJOR,
    "uprising_crushed": MAJOR,
    "uprising_begins": MAJOR,
    "uprising_leader": MAJOR,
    "expropriation": MAJOR,
    "breakaway_empire": MAJOR,
    "breakaway_realm": MAJOR,
    "secession": MAJOR,
    "assault_repelled": MAJOR,
    "host_repelled": MAJOR,
    "era": MAJOR,
    "writing": MAJOR,          # demoted to MINOR after the world's first (see classify)
    "weapons": MAJOR,          # ditto
    "succession": MAJOR,
    "leader_consent": MAJOR,
    "leader_displaced": MAJOR,
    "prophet": MAJOR,
    "coalition_dissolved": MAJOR,
    # MINOR — real, but not worth a cut.
    "metalworking": MINOR,
    "written_law": MINOR,
    "legitimacy_lost": MINOR,
    "faith_root": MINOR,
    "birth": MINOR,
    "death": MINOR,
    "alliance_formed": MINOR,
    "betrayal": MINOR,
    "inheritance": MINOR,
    "teaching": MINOR,
    "discovery": MINOR,
    "came_of_age": MINOR,
    "theft": MINOR,
    "law_inherited": MINOR,
    "law_retaught": MINOR,
    # NOISE — book-keeping. Never captioned, never framed.
    "levy": NOISE,
    "tribute_imperial": NOISE,
    "tribute_realm": NOISE,
    "redistribution": NOISE,
    "estate_vanished": NOISE,
    "estate_dropped": NOISE,
    "talk": NOISE,
    "reply": NOISE,
    "alliance_proposed": NOISE,
    "trust": NOISE,
    "seethes": NOISE,
    "grievance": NOISE,
    "escheat": NOISE,
    "savings": NOISE,
    "joined": NOISE,
    "god": NOISE,
    "unknown": NOISE,
}

# Kinds that are a world-FIRST beat: major the first time the world sees them, ordinary after.
# `era` is keyed by the era NAME, so the world's first Bronze Age is a beat and the fifth town to
# reach it is not.
_FIRST_ONLY: frozenset[str] = frozenset({"writing", "weapons", "era"})

# Only these ages are a story beat at all. A staged run drops five settlements into the Neolithic
# on turn 1 — that is the scenario builder finishing its work, not a civilisation crossing an age.
_BEAT_ERAS = ("bronze", "iron")

# Battles whose outcome is only a beat if BLOOD was actually spilled. `empire.update` will happily
# march ten fighters against a realm that can field zero: it wins, the log calls it a war, and
# nothing whatsoever happens on screen. A battle with no fallen on either side is a bloodless
# annexation — real in the ledger, worthless as drama — so it is demoted out of the camera's reach.
_BATTLE_KINDS = frozenset({"war_won", "war_failed", "realm_conquest", "town_seized", "coup"})
_FELL = re.compile(r"(\d+)\+(\d+) fell")


def _first_key(kind: str, g: dict[str, str]) -> str | None:
    """The world-first ledger key for an event, or None if this kind is not first-gated."""
    if kind not in _FIRST_ONLY:
        return None
    if kind == "era":
        era = g.get("x", "").lower()
        if not any(name in era for name in _BEAT_ERAS):
            return "era:ignored"        # a non-headline age: never a beat, first or not
        return f"era:{era.strip()}"
    return kind


def _bloodless(raw: str) -> bool:
    """True if a battle line reports no fallen on either side (pure read of its '(N+M fell)' tail)."""
    m = _FELL.search(raw)
    return m is not None and m.group(1) == "0" and m.group(2) == "0"


def strip_prefix(line: str) -> tuple[int | None, str]:
    """Split a raw log line into (turn, body). Returns (None, line) if it carries no prefix."""
    if line.startswith("turn "):
        head, sep, body = line.partition(": ")
        if sep:
            num = head[5:].strip()
            if num.isdigit():
                return int(num), body
    return None, line


def _actors(kind: str, g: dict[str, str]) -> tuple[str, ...]:
    """The named figures in an event, in log order (deduped, blanks dropped)."""
    out: list[str] = []
    for key in ("who", "a", "b"):
        v = g.get(key)
        if v and v not in out:
            out.append(v)
    return tuple(out)


def classify(line: str, seen_firsts: "set[str] | None" = None) -> Event:
    """Parse ONE verbatim event line into a typed `Event` (pure apart from `seen_firsts`).

    `seen_firsts` is the per-run memory of which world-first feats have already happened; pass
    the SAME set across a run and the second town to devise writing is MINOR while the first is
    MAJOR. Pass None to treat every line in isolation (every first-only kind stays MAJOR).
    An unrecognised line lands as kind 'unknown' at NOISE — the director degrades quietly rather
    than framing a line it does not understand.
    """
    turn, body = strip_prefix(line)
    for kind, pat in _COMPILED:
        m = pat.match(body)
        if not m:
            continue
        g = {k: v for k, v in m.groupdict().items() if v is not None}
        sev = SEVERITY.get(kind, NOISE)
        key = _first_key(kind, g)
        if key == "era:ignored":
            sev = MINOR                          # the Neolithic is scenery, not a turning point
        elif key is not None and seen_firsts is not None:
            if key in seen_firsts:
                sev = MINOR                      # the world already has writing; this town caught up
            else:
                seen_firsts.add(key)
        if kind in _BATTLE_KINDS and _bloodless(line):
            sev = MINOR                          # a bloodless annexation is a ledger entry, not a war
        return Event(turn=turn if turn is not None else -1, kind=kind, severity=sev,
                     focus=g.get("sid"), actors=_actors(kind, g), fields=g, raw=line)
    return Event(turn=turn if turn is not None else -1, kind="unknown", severity=NOISE,
                 focus=None, actors=(), fields={}, raw=line)


def classify_turn(lines: "list[str]", seen_firsts: "set[str] | None" = None) -> list[Event]:
    """Classify every line of one turn, in log order (see `classify` for `seen_firsts`)."""
    return [classify(ln, seen_firsts) for ln in lines]


def turn_severity(events: "list[Event]") -> str:
    """The severity of a TURN = the max severity of its events (NOISE for an empty turn)."""
    return max((e.severity for e in events), key=rank, default=NOISE)


# The events an uprising's resolution already tells: they fire on the same turn, in the same
# place, and captioning each one separately would spend three cuts on one story.
_FOLDS_INTO_UPRISING = ("uprising_begins", "uprising_leader", "expropriation", "secession")

# The consequences of a beat that carry no settlement of their own in the log line. They inherit
# the focus of the strongest beat on the same turn, so the camera still knows where to look.
_INHERITS_FOCUS = frozenset({"expropriation", "succession", "dynasty_extinct", "war_won",
                             "war_failed", "breakaway_empire", "empire_broken"})


# Kinds where several firing on the SAME turn is one story ("three towns chose leaders"), not
# several cuts. The first carries the group; `fields['also']` records how many others there were.
_COLLAPSIBLE = frozenset({"leader_consent", "era", "writing", "weapons", "birth", "secession",
                          "breakaway_empire", "breakaway_realm", "metalworking"})


def _collapse_same_kind(ordered: "list[Event]") -> list[Event]:
    """Fold repeats of a collapsible kind on one turn into their first occurrence (pure)."""
    out: list[Event] = []
    seen: dict[str, int] = {}
    for e in ordered:
        if e.kind in _COLLAPSIBLE and e.kind in seen:
            i = seen[e.kind]
            grouped = dict(out[i].fields)
            grouped["also"] = str(int(grouped.get("also", "0")) + 1)
            out[i] = out[i]._replace(fields=grouped)
            continue
        seen.setdefault(e.kind, len(out))
        out.append(e)
    return out


def beats(events: "list[Event]", floor: str = MAJOR, drop_staging: bool = True) -> list[Event]:
    """The events worth cutting to this turn, strongest first then log order (stable).

    Three edits, in order:

    * STAGING. Turn 0 is the scenario builder laying down three realms through the real conquest
      paths — six legendary-looking lines before the viewer has read the title card. It is set
      dressing, not story, so it is dropped (`drop_staging=False` keeps it).
    * FOLDING. An uprising is ONE story: the rising, the leader rallying the survivors, the hoard
      changing hands and the town seceding all fold into the TRIUMPHED/CRUSHED line for the same
      settlement. Otherwise turn 34 spends five cuts telling one revolt.
    * FOCUS INHERITANCE. A war, a succession or an expropriation names no settlement in its log
      line. Rather than leave the camera with nowhere to go, these inherit the focus of the
      strongest beat that DOES name one this turn.
    """
    live = [e for e in events if rank(e.severity) >= rank(floor)]
    if drop_staging:
        live = [e for e in live if e.turn != 0]
    resolved = {e.focus for e in live if e.kind in ("uprising_triumph", "uprising_crushed")}
    live = [e for e in live if not (e.kind in _FOLDS_INTO_UPRISING
                                    and (e.focus in resolved or e.focus is None and resolved))]
    ordered = sorted(live, key=lambda e: (-rank(e.severity), events.index(e)))
    ordered = _collapse_same_kind(ordered)
    anchor = next((e.focus for e in ordered if e.focus), None)
    if anchor is None:
        return ordered
    return [e._replace(focus=anchor) if e.focus is None and e.kind in _INHERITS_FOCUS else e
            for e in ordered]


# --- Captions ---------------------------------------------------------------
# A caption is a TITLE line and at most one SUBTITLE line. Terse, declarative, dramatised —
# never the raw log line. Names are spaced out of their engine form (LordB -> "Lord B") so a
# viewer reads a person, not an identifier; settlement ids are left as they are (they ARE the
# names of the places on the map).
_TITLED = re.compile(r"^(Lord|King|Queen|Chief|Prince|Princess)([A-Z0-9].*)$")


def person(name: str) -> str:
    """A log name rendered for a caption: 'LordB' -> 'Lord B', 'Aldric' -> 'Aldric' (pure)."""
    m = _TITLED.match(name)
    return f"{m.group(1)} {m.group(2)}" if m else name


def _n(fields: dict[str, str], key: str, default: str = "0") -> str:
    """A numeric field as a clean string ('4.00' -> '4')."""
    v = fields.get(key, default)
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else f"{f:.1f}"
    except ValueError:
        return v


def _plural(n: str, one: str, many: str) -> str:
    return one if n == "1" else many


# kind -> (title, subtitle) builders over an Event. Every MAJOR/LEGENDARY kind has one; a kind
# with no template falls back to a stripped version of the log line, so nothing is ever blank.
_CAPTIONS: dict[str, Any] = {
    "war_won": lambda e, f: (
        f"{person(f['a']).upper()} DEFEATS {person(f['b']).upper()}",
        f"{person(f['b'])} kneels. An empire rises."),
    "war_failed": lambda e, f: (
        f"{person(f['a']).upper()}'S WAR FAILS",
        f"{person(f['b'])}'s kingdom holds."),
    "realm_conquest": lambda e, f: (
        f"{f['sid']} FALLS TO {person(f['a']).upper()}",
        "The town is taken into the realm."),
    "town_seized": lambda e, f: (
        f"{person(f['a']).upper()} TAKES {f['sid']}",
        "Crowned by force."),
    "dynasty_extinct": lambda e, f: (
        f"THE LINE OF {person(f['who']).upper()} ENDS",
        "The crown lies vacant."),
    "empire_broken": lambda e, f: (
        f"{person(f['who']).upper()} IS FREED",
        f"The coalition breaks {person(f['b'])}'s empire."),
    "coup": lambda e, f: (
        f"{person(f['a']).upper()} OVERTHROWS {person(f['b']).upper()}",
        f"{f['sid']} changes hands by the sword."),
    "uprising_triumph": lambda e, f: (
        f"THE RISING OF {f['sid']}",
        f"{person(f['b'])} falls. {_n(f, 'n')} "
        f"{_plural(_n(f, 'n'), 'riser', 'risers')} fell. The hoard is theirs."),
    "uprising_crushed": lambda e, f: (
        f"THE RISING OF {f['sid']} IS CRUSHED",
        f"{person(f['b'])} holds. {_n(f, 'm')} "
        f"{_plural(_n(f, 'm'), 'riser', 'risers')} fell. The rest are cowed."),
    "uprising_begins": lambda e, f: (
        f"{f['sid']} RISES",
        f"{_n(f, 'n')} against {person(f['b'])}'s {_n(f, 'm')}."),
    "uprising_leader": lambda e, f: (
        f"{person(f['who']).upper()} LEADS THE RISING",
        f"The survivors of {f['sid']} rally to him."),
    "expropriation": lambda e, f: (
        "THE HOARD IS TAKEN",
        f"{_n(f, 'n')} stripped from {person(f['who'])}, split {_n(f, 'm')} ways."),
    "breakaway_empire": lambda e, f: (
        f"{person(f['who']).upper()} BREAKS AWAY",
        f"The realm leaves {person(f['b'])}'s empire."),
    "breakaway_realm": lambda e, f: (
        f"{person(f['who']).upper()} BREAKS AWAY",
        f"{f.get('sid', 'The town')} is independent again."),
    "secession": lambda e, f: (
        f"{f['sid']} SECEDES",
        f"No longer {person(f['b'])}'s."),
    "assault_repelled": lambda e, f: (
        f"{f['sid']} HOLDS",
        f"{person(f['a'])}'s assault is thrown back."),
    "host_repelled": lambda e, f: (
        f"{f['sid']} HOLDS",
        f"{person(f['a'])}'s host is thrown back."),
    "era": lambda e, f: (
        f"{f['sid']} ENTERS THE {f['x'].replace('the ', '').upper()}",
        None),
    "writing": lambda e, f: (
        "WRITING IS DEVISED",
        f"{person(f['who'])} sets down the first words in {f['sid']}."),
    "weapons": lambda e, f: (
        f"{f['sid']} FORGES THE FIRST WEAPONS",
        f"{person(f['who'])} works the iron."),
    "succession": lambda e, f: (
        f"{person(f['a']).upper()} SUCCEEDS {person(f['b']).upper()}",
        "The crown passes."),
    "leader_consent": lambda e, f: (
        f"{f['sid']} CHOOSES {person(f['who']).upper()}",
        "Raised by consent, not by force."),
    "prophet": lambda e, f: (
        f"{person(f['who']).upper()} SPEAKS",
        f"A prophet of {f['x']}."),
    "coalition_dissolved": lambda e, f: (
        "THE COALITION DISSOLVES",
        f"{person(f['b'])} is no longer feared."),
}


def caption(e: Event) -> tuple[str, str | None]:
    """A dramatised (title, subtitle) card for one classified event (pure).

    Terse and declarative: a title line and at most one subtitle. An event with no template
    (or one whose log line lacked a field) degrades to the plain event body as a title, so the
    director can always show SOMETHING rather than crash mid-run.
    """
    builder = _CAPTIONS.get(e.kind)
    if builder is not None:
        try:
            title, sub = builder(e, e.fields)
            also = int(e.fields.get("also", "0"))
            if also:
                extra = f"and {also} more the same turn."
                sub = f"{sub} {extra}" if sub else extra.capitalize()
            return title, sub
        except KeyError:
            pass
    _, body = strip_prefix(e.raw)
    return body.split(" (")[0].split(" -> ")[0].strip(" ;—-").upper(), None
