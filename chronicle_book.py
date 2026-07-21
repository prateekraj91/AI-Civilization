"""
chronicle_book.py
=================

THE CHRONICLE AS A HISTORY BOOK — the structured record (M4.16, `chronicle.py`) composed into
narrative prose, as if written by a single anonymous chronicler of the civilization looking back on
its age. This is a PRESENTATION layer over `world_state["chronicle"]` + the event stream: it adds NO
simulation mechanics, mutates NO sim field, and is fully DETERMINISTIC (same seed -> same book). Zero
LLM is required; with `--minds` the pivot consults' recorded MOTIVES (already logged by `mind.py`)
give figures their stated reasons in the prose, and without minds the book still writes cleanly.

Difference from `chronicle.export_markdown` (the DIGEST, still available)
------------------------------------------------------------------------
The digest is a structured listing — Great Figures / Houses / a turn-stamped Saga. The BOOK is
continuous past-tense prose in chapters, with NO turn numbers, NO agent codes, NO settlement ids and
NO mechanic names: settlements and coded figures are given real names (deterministic from the seed),
time is told in years and reigns, and each chapter follows the causal chain that the digest only
lists — surplus made lords, lords made tribute, tribute made rage, rage made revolt.

Everything here READS `world_state` and is pure.
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Any

import chronicle

# --- Deterministic naming ----------------------------------------------------
# A coded agent is a scenario token (AT4, AWV4, AKM3, BWK8), a staged lord (LordA), or a monarchy-demo
# token (F2a, Chief3) — anything not already a human name. Real names (Aldric, Iris, born children)
# never match, so they pass through unchanged.
_CODE_RE = re.compile(r"^(?:[A-C][A-Z]{1,2}\d+|Lord[A-C]|Chief\d+|F\d+[a-z])$")

# Evocative place-names and given-names, assigned to ids/codes by a seeded shuffle. Both pools are far
# larger than any run needs, so assignment never runs dry and never collides.
_PLACE_POOL = [
    "Ashfall", "Vharun", "Duncarrow", "Emberhold", "Greymoor", "Karth", "Thornevale", "Ravensmark",
    "Blackmere", "Highfen", "Stonewatch", "Oldharrow", "Windmere", "Caldern", "Fenwick", "Ryehold",
    "Marrowford", "Dunhollow", "Storncrag", "Ashmere", "Colwyn", "Brammel", "Harrowgate", "Nethermill",
    "Whitlow", "Corham", "Draymoor", "Eldwick", "Faircrest", "Gallowfen", "Holt", "Ironvale",
]
_NAME_POOL = [
    "Corvin", "Halden", "Maren", "Torvald", "Edric", "Sable", "Bram", "Rowan", "Alda", "Neris",
    "Garrick", "Wystan", "Cael", "Doran", "Merrick", "Ysolde", "Fenn", "Osric", "Ballard", "Renna",
    "Thane", "Gveld", "Aldous", "Bryn", "Cordis", "Hesper", "Lorne", "Perrin", "Sorrel", "Talia",
    "Verrin", "Wynn", "Dagmar", "Emeric", "Godric", "Isolde", "Katrin", "Loris", "Mabon", "Orla",
]


def _stable_int(*parts: Any) -> int:
    """A process-stable integer from the given parts (hashlib, so it never varies with PYTHONHASHSEED)."""
    h = hashlib.md5("::".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:12], 16)


def _assign(pool: list[str], keys: list[str], salt: Any) -> dict[str, str]:
    """Deterministically map sorted `keys` onto distinct names from `pool` (seeded shuffle, then zip)."""
    rng = random.Random(_stable_int(salt, "|".join(sorted(keys))))
    names = pool[:]
    rng.shuffle(names)
    return {k: names[i % len(names)] for i, k in enumerate(sorted(keys))}


class Names:
    """The run's deterministic naming: settlement ids -> place-names, coded agents -> given-names.

    Real human names (kings, born children) pass through unchanged; only codes are renamed. Built from
    every id/code that appears anywhere in the chronicle, so a name is stable wherever it is used.
    """

    def __init__(self, state: dict[str, Any], seed: Any) -> None:
        salt = seed if seed is not None else _stable_int("world", *sorted(state.get("settlements", {})))
        self._place = _assign(_PLACE_POOL, self._collect_sids(state), (salt, "places"))
        self._person = _assign(_NAME_POOL, self._collect_codes(state), (salt, "people"))

    @staticmethod
    def _collect_sids(state: dict[str, Any]) -> list[str]:
        sids = set(state.get("settlements", {}))
        for e in chronicle._chron(state)["events"]:
            sids.update(re.findall(r"S[0-9A-Z]{3}", e.get("detail", "") + " " + e.get("name", "")))
        return list(sids)

    @staticmethod
    def _collect_codes(state: dict[str, Any]) -> list[str]:
        ch = chronicle._chron(state)
        toks: set[str] = set(ch["figures"])
        for e in ch["events"]:
            toks.update(e.get("actors", []))
            toks.update(re.findall(r"[A-Za-z]\w*", e.get("detail", "")))
        for child, b in ch["births"].items():
            toks.update([child, b[0], b[1]])
        return [t for t in toks if _CODE_RE.match(t)]

    def place(self, sid: "str | None") -> str:
        if not sid:
            return "a nameless place"
        return self._place.get(sid, sid)

    def person(self, name: "str | None") -> str:
        if not name:
            return "one whose name is lost"
        return self._person.get(name, name)


def place_name_map(sids: "list[str] | set[str]", seed: Any) -> dict[str, str]:
    """The settlement id -> place-name map alone (no chronicle needed), for the RENDERER to label the
    world with the SAME names the book uses. Deterministic from `seed` + the given set of ids, so the
    footage and the chronicle name one world. `seed=None` -> a world-fingerprint salt (still stable)."""
    sids = list(sids)
    salt = seed if seed is not None else _stable_int("world", *sorted(sids))
    return _assign(_PLACE_POOL, sids, (salt, "places"))


# --- Time as the world would tell it -----------------------------------------
_ORD_ONES = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
             "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
             "sixteenth", "seventeenth", "eighteenth", "nineteenth"]
_TENS_ORD = {20: "twentieth", 30: "thirtieth", 40: "fortieth", 50: "fiftieth", 60: "sixtieth",
             70: "seventieth", 80: "eightieth", 90: "ninetieth"}
_TENS_CARD = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
              70: "seventy", 80: "eighty", 90: "ninety"}


def ordinal(n: int) -> str:
    """A spelled-out ordinal ('thirty-ninth') — a chronicle tells years in words, never as '39th'."""
    if 0 <= n < 20:
        return _ORD_ONES[n]
    if n < 100:
        tens, ones = (n // 10) * 10, n % 10
        return _TENS_ORD[tens] if ones == 0 else f"{_TENS_CARD[tens]}-{_ORD_ONES[ones]}"
    return f"{n}th"


def year_phrase(turn: int) -> str:
    """A turn told as a year of the age (turn 1 == the first year). No raw numbers leak out."""
    return f"the {ordinal(max(1, turn))} year"


# --- Epithet-bearing figure helpers ------------------------------------------
def titled_person(names: Names, fig: dict[str, Any]) -> str:
    """A figure named and titled: 'Borin the Grasping' (real or renamed subject + its earned epithet)."""
    return f"{names.person(fig['name'])} {chronicle.epithet(fig)}"


# --- Reading the structured record -------------------------------------------
def _events(state: dict[str, Any], kind: str, *, history_only: bool = False) -> list[dict[str, Any]]:
    evs = [e for e in chronicle._chron(state)["events"] if e["kind"] == kind]
    if history_only:
        evs = [e for e in evs if e["fidelity"] == "history"]
    return sorted(evs, key=lambda e: e["turn"])


# A settlement id is S + three alphanumerics (S0A2 in the staged realms, S001 in the monarchy demo).
_SID = r"S[0-9A-Z]{3}"
_P_WRITING_DETAIL = re.compile(rf"^(\S+) of ({_SID}) devised writing")
_P_WAR_DETAIL = re.compile(r"^KING (\S+) defeated (\S+)")
_P_ERA_DETAIL = re.compile(rf"^({_SID}) reached (.+)$")


def _kings(state: dict[str, Any]) -> list[str]:
    """The named kings of the age, in the order the record first speaks of each (victors and the
    kings they broke). Derived from the war entries — no scenario knowledge."""
    order: list[str] = []
    for e in _events(state, "war", history_only=True):
        m = _P_WAR_DETAIL.match(e["detail"])
        if m:
            for k in (m.group(1), m.group(2)):
                if k not in order:
                    order.append(k)
    return order


def _writing(state: dict[str, Any]) -> "tuple[str, str, int] | None":
    """(scribe, sid, turn) of the first writing, or None if the age never learned to write."""
    for e in _events(state, "writing"):
        m = _P_WRITING_DETAIL.match(e["detail"])
        if m:
            return m.group(1), m.group(2), e["turn"]
    return None


def _unique_wars(state: dict[str, Any]) -> list[tuple[int, str, str]]:
    """The wars as (turn, victor, loser), ONE entry per (victor, loser) pair — a king who beats the
    same rival again and again is one story, not a refrain. First occurrence wins; chronological."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[int, str, str]] = []
    for e in _events(state, "war", history_only=True):
        m = _P_WAR_DETAIL.match(e["detail"])
        if m and (m.group(1), m.group(2)) not in seen:
            seen.add((m.group(1), m.group(2)))
            out.append((e["turn"], m.group(1), m.group(2)))
    return out


def _unique_breakers(state: dict[str, Any]) -> list[tuple[int, str, str]]:
    """Secessions as (turn, breaker, lord), ONE per breaker — a realm that secedes, is retaken, and
    secedes again is the same strain, told once (its first break)."""
    seen: set[str] = set()
    out: list[tuple[int, str, str]] = []
    for e in _events(state, "breakaway", history_only=True):
        m = re.match(r"^(\S+) broke away from (\S+)", e["detail"])
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append((e["turn"], m.group(1), m.group(2)))
    return out


# --- Motives (M5.1): varied per figure, deduped, omitted when they would only repeat --------------
# The offline mind gives ONE canned reason per (pivot, acted), so three war-victors would all "say"
# the same thing. A chronicle in which every ruler sounds identical is a failed chronicle, so a
# recorded reason that is one of those generic stand-ins is RE-VOICED per figure — flavoured by what
# the record shows the figure to be (a grasping levier, a lifelong warrior, a firebrand) — and NEVER
# repeated: once a phrasing is spent, the next figure that would reach for it is given silence
# instead. A chronicler who cannot say why a king acted says nothing. A genuinely distinct reason (a
# live model's own words) is used as given, still deduped.
_MOTIVE_VARIANTS: dict[tuple, dict[str, list[str]]] = {
    ("war", True): {
        "grasping": ["reckoning that the tribute his towns had bled for would buy the victory his birth had not"],
        "warrior": ["trusting to the sword, as he had trusted it in every year of his rule"],
        "_any": ["judging the odds even and the bolder course the surer",
                 "seeing his rival no stronger than himself, and himself in want of a wider realm",
                 "unwilling to let an even chance go by"],
    },
    ("breakaway", True): {
        "grasping": ["loath to send another season's tribute to a throne he had never chosen"],
        "_any": ["holding that a crown he did not love had no claim upon him",
                 "reasoning that a master too distant to command his heart was no master at all",
                 "judging himself better standing alone than kneeling far from home"],
    },
    ("uprising", True): {
        "firebrand": ["certain that the town would rise behind him if only a hand raised the banner"],
        "_any": ["judging that the hour had come at last", "reading in the hungry faces that the moment was ripe"],
    },
    ("uprising", False): {
        "_any": ["believing the hour had come, though it had not"],
    },
}


class Motives:
    """Re-voices the M5.1 pivot motives so no two figures sound alike, deduping across the whole book.

    Off with minds off (every clause empty). Holds the run-wide set of phrasings already spent, so a
    generic stand-in reason surfaces at most once and thereafter yields to silence. Pure over `state`.
    """

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state
        self._on = bool(state.get("minds_on"))
        self._used: set[str] = set()
        try:
            import mind
            self._generic = set(mind._REASON.values())
        except Exception:
            self._generic = set()

    def _raw(self, turn: int, figure: "str | None", sid: "str | None", pivot: str) -> "str | None":
        try:
            import mind
            if pivot == "breakaway" and figure:
                return mind.breakaway_motive(self._state, turn, figure)
            if figure:
                return mind.motive_for(self._state, turn, figure)
            if sid:
                return mind.motive_at(self._state, turn, sid)
        except Exception:
            return None
        return None

    def clause(self, turn: int, pivot: str, *, figure: "str | None" = None, sid: "str | None" = None,
               acted: bool = True, fig: "dict | None" = None) -> str:
        """A trailing motive clause (', reckoning that…') for this pivot, or '' — never a repeat."""
        if not self._on:
            return ""
        reason = self._raw(turn, figure, sid, pivot)
        if not reason:
            return ""                                 # no mind was consulted — say nothing
        if reason not in self._generic:               # a genuine, distinct reason: use it, once
            r = reason[0].lower() + reason[1:]
            if r in self._used:
                return ""
            self._used.add(r)
            return f", by his own account because {r}"
        # A generic stand-in: re-voice it per figure, and never twice.
        pool = _MOTIVE_VARIANTS.get((pivot, acted), {})
        buckets = _profile_buckets(fig)
        cands: list[str] = []
        for b in buckets:
            cands.extend(pool.get(b, []))
        cands.extend(pool.get("_any", []))
        rng = random.Random(_stable_int(figure or sid or "", pivot, acted))
        rng.shuffle(cands)
        for c in cands:
            if c not in self._used:
                self._used.add(c)
                return f", {c}"
        return ""                                     # nothing distinct left — silence over repetition


def _profile_buckets(fig: "dict | None") -> list[str]:
    """Which flavour buckets a figure draws its re-voiced motive from, by what the record shows it did."""
    if not fig:
        return []
    d = fig["deeds"]
    out: list[str] = []
    if d.get("levies", 0) >= chronicle.GRASPING_LEVIES:
        out.append("grasping")
    if d.get("wars", 0) + d.get("conquests", 0) >= 2:
        out.append("warrior")
    if d.get("uprisings", 0) >= 1:
        out.append("firebrand")
    return out


# --- Chapter prose -----------------------------------------------------------
# Each chapter/section is continuous past-tense prose: a line of scene-setting, what happened and why
# it followed from what came before, and a closing consequence. Paragraphs join with blank lines.
def _para(*sentences: str) -> str:
    return " ".join(s for s in sentences if s)


def _and_list(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# --- Prologue: the age before literacy (its names are lost) -------------------
def prologue_before_the_written_word(state: dict[str, Any], N: Names) -> "tuple[str, list[str]]":
    """The preliterate age, gathered into ONE opening section rather than interleaved as lost-name
    refrains among the named history. Ends at the dawn of writing, where the chronicle proper begins."""
    places = [N.place(s) for s in sorted(state.get("settlements", {}))]
    w = _writing(state)
    lost_conquests = [e for e in _events(state, "conquest") if e["fidelity"] == "legend"]
    paras: list[str] = []

    if places:
        paras.append(_para(
            f"In the beginning the people were few and scattered, and it was the land itself that "
            f"gathered them: where the soil gave more grain than a household could eat, households "
            f"stayed, and stored, and became neighbours. Out of that first surplus grew the towns whose "
            f"names have come down to us — {_and_list(places)} — each a cluster of hearths around a "
            f"common granary. But none of this was written while it happened, for as yet no one could "
            f"write, and what those first years held we know only as the old people later told it."))
    else:
        paras.append(_para(
            "In the beginning the people were few and scattered, and gathered where the land fed them. "
            "But none of it was written while it happened, and what those first years held is only "
            "what the old people later told."))

    if lost_conquests:
        paras.append(_para(
            "It is said the towns did not stay their own for long. Strong men came — some from among "
            "the people, some out of the hungry country beyond — and took the granaries by force; and a "
            "man who could hold a town and feed a following from it was a king, whatever he had been "
            "before. The record knows only that it happened, and more than once. It does not know their "
            "names. Their wars and their conquests reach us as rumour, lost to the dark before writing, "
            "and here they must stay — a murmur of violence under everything the chronicle can properly "
            "tell."))

    if w:
        scribe, sid, turn = w
        paras.append(_para(
            f"Then, in {year_phrase(turn)} of the age, the dark lifted. In {N.place(sid)} a lettered "
            f"man the record remembers as {N.person(scribe)} pressed the first true words into clay — "
            f"grain-tallies, at first, and then the names of men and the things they had done. It is a "
            f"small thing, a few marks in mud, and it divided all time in two: everything before it is "
            f"legend, and everything after it is history. From this year the chronicle can keep faith "
            f"with names, and the deeds of the kings who came next would not be allowed to fade. It is "
            f"with the first of them that the true history begins."))
    else:
        paras.append(_para(
            "And the dark never lifted. This age never learned to write; every king and every war of it "
            "has reached us only as legend, and legend is all this chronicle can be."))

    return "Before the Written Word", paras


# --- Chapter I: kings and the tribute that made them ------------------------
def chapter_age_of_kings(state: dict[str, Any], N: Names, M: Motives) -> "tuple[str, list[str]]":
    ch = chronicle._chron(state)
    kings = _kings(state)
    king_figs = [ch["figures"][k] for k in kings if k in ch["figures"]]
    wars = _unique_wars(state)
    paras: list[str] = []

    if kings:
        paras.append(_para(
            f"With writing came kings worth writing of. The age that opened was an age of crowns: "
            f"{_and_list([N.person(k) for k in kings])}, and lesser men who wore the name, each seated "
            f"above a handful of towns that were his to guard and, more to the purpose, his to tax. A "
            f"king of those years did not live among his people. He kept his seat apart, with his own "
            f"larder and his own hired spears, and the towns below sent up to him whatever he asked."))

    grasping = [f for f in king_figs if f["deeds"].get("levies", 0) >= chronicle.GRASPING_LEVIES]
    if grasping:
        worst = max(grasping, key=lambda f: f["deeds"]["levies"])
        others = [f for f in grasping if f is not worst]
        line = (f"And what they asked was tribute, some of them without limit. {titled_person(N, worst)} "
                f"was the hungriest")
        if others:
            line += f", though {_and_list([N.person(f['name']) for f in others])} was scarcely gentler"
        line += (". Season upon season the levies climbed from the granaries to the royal seats — grain "
                 "to feed soldiers, coin to hire more — and season upon season the towns that grew the "
                 "grain kept a little less of it. It bought the kings their wars. It was also, though no "
                 "one yet dared say so, the first cause of everything that would in time break them.")
        paras.append(_para(line))

    if wars:
        turn, victor, loser = wars[0]
        vfig = ch["figures"].get(victor)
        vname = titled_person(N, vfig) if vfig else N.person(victor)
        motive = M.clause(turn, "war", figure=victor, acted=True, fig=vfig)
        paras.append(_para(
            f"It was {vname} who struck first. In {year_phrase(turn)}, with a war chest his "
            f"towns had filled, he fell upon the kingdom of {N.person(loser)} and broke it{motive}. "
            f"{N.person(loser)}'s crown passed to him, and for the first time one man held more than "
            f"his own country. This was the beginning of empire in the land — and, though it wore the "
            f"face of a triumph, the beginning of the end of kings."))

    paras.append(_para(
        "For a crown taken by force is a crown that must be held by force, and holding it cost more "
        "tribute, and more tribute cost the towns more of what little the levies had left them. The "
        "wider a king reached, the harder he had to squeeze — and the towns were counting every "
        "season of it. What the kings had built to make themselves great was already, quietly, "
        "teaching their people to hate them."))

    return "The Age of Kings", paras


# --- Chapter II: the empire forms and cannot hold ---------------------------
def chapter_empire_and_breaking(state: dict[str, Any], N: Names, M: Motives) -> "tuple[str, list[str]]":
    ch = chronicle._chron(state)
    wars = _unique_wars(state)
    breakers = _unique_breakers(state)
    paras: list[str] = []

    paras.append(_para(
        "The years that followed were years of one crown swallowing another, and then failing to keep "
        "it down. The land was too wide and the loyalties too thin for any single throne to hold what "
        "its armies could take."))

    # The chain of conquests AFTER the first (Chapter I told that one) — each victor named, one motive.
    later = wars[1:]
    if later:
        told = []
        for turn, victor, loser in later:
            vfig = ch["figures"].get(victor)
            vname = titled_person(N, vfig) if vfig else N.person(victor)
            motive = M.clause(turn, "war", figure=victor, acted=True, fig=vfig)
            told.append(f"in {year_phrase(turn)} {vname} brought {N.person(loser)} down in his turn{motive}")
        chain = "; and ".join(told)
        paras.append(_para(
            f"The strongest of the kings did not stay strongest: {chain}. Each conqueror became, in his "
            f"hour, the next man's prize, and the crown of empire never sat long on any one head."))

    # The secessions — what was conquered would not stay conquered. Motive on the first only, to keep
    # the roll of names clean; the rest are named plainly.
    if breakers:
        turn0, who0, _lord0 = breakers[0]
        bfig0 = ch["figures"].get(who0)
        motive0 = M.clause(turn0, "breakaway", figure=who0, acted=True, fig=bfig0)
        line = (f"But it was not war that undid the empires so much as sheer distance. A realm seized is "
                f"a realm that strains to be free, and one after another the subject kings pulled loose. "
                f"{N.person(who0)} was the first to go, in {year_phrase(turn0)}{motive0}")
        rest = [N.person(w) for _t, w, _l in breakers[1:]]
        if rest:
            line += (f"; and {_and_list(rest)} took the same road after him, each preferring his own "
                     f"small crown to a seat beneath a greater one")
        line += (". No sooner was an empire forged than it began to shed its provinces, and the grand "
                 "conquests of the age dissolved almost as fast as they were won.")
        paras.append(_para(line))

    paras.append(_para(
        "So the age of great crowns spent itself: the kings made war to grow, and taxed to make war, "
        "and the growing would not hold and the taxing would not stop. The empires broke along their "
        "own seams. But the deeper break was not between king and king. It was building, all this "
        "while, between the kings and the towns that fed them — and it had not yet been paid."))

    return "The Empire and Its Breaking", paras


# --- Chapter III: the risings — tribute repaid in kind, and the last king standing ------------------
_RISING_FORMS = [
    "In {place} the people rose against {lord}{motive} and pulled him from his seat, and {lib} was "
    "lifted up in his place to hold it in the people's name.",
    "{place} was next. Its people cast {lord} out{motive}, and {lib} came up out of the crowd to take "
    "the vacant seat.",
    "In {place}, too, the grievance boiled over: {lord} was thrown down{motive}, and it was {lib} the "
    "people raised to the seat he had held.",
    "Then {place} turned on {lord} and unseated him{motive}, and gave the seat to {lib}, one of their own.",
]


def chapter_the_risings(state: dict[str, Any], N: Names, M: Motives) -> "tuple[str, list[str]]":
    ch = chronicle._chron(state)
    ups = _events(state, "uprising", history_only=True)
    kings = set(_kings(state))
    w = _writing(state)
    scribe = w[0] if w else None
    paras: list[str] = []

    paras.append(_para(
        "And in the end it was paid. The towns had grown the grain and sent it up and kept less of it "
        "each year, and there is a limit to what a people will surrender to a seat they never chose. "
        "When the breaking came at last it did not come from a rival king. It came from below."))

    won, king_falls = [], []
    for e in ups:
        if "crushed" in e["name"]:
            continue
        m = re.match(r"^the people of (\S+) rose, deposed (\S+), and (\S+) took power", e["detail"])
        if not m:
            continue
        rec = (e["turn"], m.group(1), m.group(2), m.group(3))       # turn, sid, deposed, leader
        (king_falls if m.group(2) in kings else won).append(rec)

    # The wave against the lords — varied, and with the scribe's own fall noted where it happens.
    for i, (turn, sid, deposed, leader) in enumerate(won):
        lfig = ch["figures"].get(leader)
        lead = titled_person(N, lfig) if lfig else N.person(leader)
        motive = M.clause(turn, "uprising", sid=sid, acted=True, fig=lfig)
        sentence = _RISING_FORMS[i % len(_RISING_FORMS)].format(
            place=N.place(sid), lord=N.person(deposed), lib=lead, motive=motive)
        if deposed == scribe:
            sentence += (f" There was a bitter justice in it: {N.person(deposed)} had given his people "
                         f"their first letters, and they used the memory those letters kept to remember "
                         f"every grain he had taken from them.")
        paras.append(_para(sentence))

    # A king's fall is not a lord's — the grasping crown gets its paragraph, brought down by its own towns.
    for turn, sid, deposed, leader in king_falls:
        dfig = ch["figures"].get(deposed)
        dname = titled_person(N, dfig) if dfig else N.person(deposed)
        lfig = ch["figures"].get(leader)
        lead = titled_person(N, lfig) if lfig else N.person(leader)
        deeds = dfig["deeds"] if dfig else {}
        made = []
        if deeds.get("wars") or deeds.get("conquests"):
            made.append("had won his crown in war")
        if deeds.get("levies", 0) >= chronicle.GRASPING_LEVIES:
            made.append("had taxed his towns to the bone to pay for the winning")
        record = _and_list(made) or "had ruled as the others ruled"
        paras.append(_para(
            f"But the hardest fall was a king's. {dname}, who {record}, came in {year_phrase(turn)} to "
            f"the end every grasping crown earns. It was not a rival army that unseated him. It was "
            f"{N.place(sid)}, the very town whose granaries had filled his war chest: it rose beneath him "
            f"and would give no more, and {lead} led it. The hungriest crown of the age ended as it had "
            f"fed — torn from him by the same hands that had fed it."))

    crushed = [re.match(r"^the people of (\S+)", e["detail"]).group(1)
               for e in ups if "crushed" in e["name"] and re.match(r"^the people of (\S+)", e["detail"])]
    if crushed:
        paras.append(_para(
            f"Not every rising won. In {_and_list([N.place(s) for s in crushed])} the people rose and "
            f"were put down, and the seat held a while longer over the bodies of those who had dared. But "
            f"a throne that must be defended against its own town is a throne already lost, whatever the "
            f"tally of one season's dead."))

    # Who was left standing — one last king, or a field of them the risings never reached.
    survivors = list(state.get("kingdoms", {}))
    named = [titled_person(N, ch["figures"][k]) if k in ch["figures"] else N.person(k) for k in survivors]
    if len(survivors) == 1:
        paras.append(_para(
            f"When the dust settled, one crown alone was still upright. {named[0]} had outlived every "
            f"rival and outlasted every revolt, and sat his throne at the close of the age as though the "
            f"whole convulsion had been arranged to leave him there. Whether that was strength or only "
            f"luck, the chronicle does not presume to say."))
    elif len(survivors) > 1:
        paras.append(_para(
            f"And yet, when the dust settled, the crowns themselves still stood — {_and_list(named)} all "
            f"kept their thrones. This is the strange verdict of the age: the risings cast down the lords "
            f"and spent the rebels, but the kings it began with were the kings it ended with. The fury "
            f"broke "
            f"upward from the granaries and stopped at the foot of the throne, rearranging every seat "
            f"beneath the crowns and leaving the crowns untouched. What the towns had truly won, and what "
            f"they had only bled for, the chronicle leaves for a wiser hand to weigh."))

    paras.append(_para(
        "So closes the age this chronicle can tell. It opened with a surplus of grain and it ended with "
        "a surplus of grief, and the road between the two ran straight: the surplus made lords, and the "
        "lords made tribute, and the tribute made rage, and the rage made revolt. What the risings will "
        "make in their turn, another hand must set down."))
    return "The Risings", paras


# --- Assembly ----------------------------------------------------------------
_ALL_CHAPTERS = [chapter_age_of_kings, chapter_empire_and_breaking, chapter_the_risings]


def export_book(state: dict[str, Any], seed: Any = None, *, chapters: "int | None" = None) -> str:
    """The full history book: a prologue for the lost age, then numbered chapters from first literacy.

    `chapters` caps how many NUMBERED chapters to render (the prologue always shows) — used to preview
    the opening for a voice check. Deterministic given seed; pure over `state`. Zero LLM required.
    """
    N = Names(state, seed)
    M = Motives(state)
    out = ["# The Chronicle of the Age", "",
           "*As set down by one hand, looking back on all of it.*", ""]

    ptitle, ppar = prologue_before_the_written_word(state, N)
    out.append(f"### {ptitle}")
    out.append("")
    out.extend(_interleave(ppar))
    out.append("")

    builders = _ALL_CHAPTERS if chapters is None else _ALL_CHAPTERS[:chapters]
    for i, build in enumerate(builders, start=1):
        title, paras = build(state, N, M)
        out.append(f"## {_roman(i)}. {title}")
        out.append("")
        out.extend(_interleave(paras))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _interleave(paras: list[str]) -> list[str]:
    lines: list[str] = []
    for p in paras:
        lines.append(p)
        lines.append("")
    return lines[:-1] if lines else lines


_ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def _roman(n: int) -> str:
    return _ROMAN[n] if 0 <= n < len(_ROMAN) else str(n)
