"""
writing.py
==========

WRITING & RECORDS — institutional memory (V2 milestone M4.10, OPENS Arc 4: the Deep Tech Tree, the
road to modernity). On top of Arc 3 (beliefs/religion/culture), Arc 2 (revolt), Arc 1 (dynasties) and
all of Phases 0-3.

The historical step M4.10 makes — memory escapes the individual
--------------------------------------------------------------
Until now everything dies with its holder: a ruler's policy vanishes at death, a skill is lost when
its last knower dies (the "knowledge extinction" collapse — a town that forgets farming starves), and
history is logged but never persists in-world. WRITING is the keystone of the modernity arc: recorded
things OUTLIVE their makers. A settlement that becomes LITERATE gains three powers of the written word —
persistent LAW, knowledge PRESERVATION, and recorded HISTORY — so its institutions, its skills and its
memory accumulate across generations instead of resetting at every death. This is the substrate the
Chronicle (Arc 6) will read.

SCOPE — M4.10 is writing-as-a-tech + its three memory powers, and ONLY that. LAW is kept MINIMAL:
written policies (tax/levy/redistribution rates) SURVIVE succession as a persistent record — there are
NO courts, enforcement, or disputes (possible later). Stated as a boundary. Writing changes no other
module's code: it reads state, reuses the M1.2 discovery machinery for its own invention, rides the
existing M1.1 diffusion to spread, and records STATE (never generated prose — ZERO LLM).

How it works (emergent; zero LLM; seeded discovery like every tech)
------------------------------------------------------------------
1. WRITING IS A TECH: an agent invents `writing` only with the prior tech `tools` AND from a SETTLEMENT
   holding a food SURPLUS (scribes need stability + spare capacity) — the invention reuses M1.2's
   `knowledge.discovery_probability` (personality- and hunger-shaped, probabilistic, NOT a timer), and
   once known it SPREADS through the ordinary M1.1 `knowledge.diffuse` like any skill. A settlement is
   LITERATE once at least LITERACY_MIN of its living members can write (`is_literate`).
2. THREE POWERS, each composing with an existing system:
   a. PERSISTENT LAW: in a literate settlement, the ruler's policy is INSCRIBED (laws[sid]); on the
      ruler's death the HEIR (M4.3) inherits the written framework instead of a blank slate. An
      illiterate settlement's policy still dies with its ruler.
   b. KNOWLEDGE PRESERVATION: a literate settlement ARCHIVES its known techs (archives[sid]); a skill
      whose last living knower dies is RE-TAUGHT from the records — so a literate civilization cannot
      forget farming (the cure for the knowledge-extinction collapse). An illiterate one still loses it.
   c. RECORDED HISTORY: a literate settlement appends its MAJOR events (foundings, coronations, wars,
      uprisings, prophets, breakaways) to a persistent CHRONICLE (chronicles[sid]) — plain structured
      entries, NOT prose. An illiterate settlement keeps no lasting record.
"""

from __future__ import annotations

import random
from typing import Any

from sim import knowledge
from sim import lineage
from sim import world

# --- Constants (tunable) -----------------------------------------------------
WRITING = "writing"              # the tech string
WRITING_PREREQ = "tools"         # the prior tech a scribe must already hold
LITERACY_MIN = 1                 # living members who must know writing for a settlement to be LITERATE

# Which knowledge is worth ARCHIVING (and re-teaching): the real techs, plus writing itself. A town's
# archive is its recorded curriculum — what it can always re-teach even if every practitioner dies.
ARCHIVABLE: frozenset[str] = frozenset(knowledge.TECH_TREE) | {WRITING}

# Event substrings a literate settlement chronicles as MAJOR history (the rest — trust ticks, levies,
# teaching — is noise the chronicle omits). Matched against this turn's world events that name the town.
MAJOR_KEYWORDS = ("founded", "took root", "succeeded", "extinguished", "UPRISING", "TRIUMPHED",
                  "CRUSHED", "BROKE AWAY", "SECEDED", "OVERTHREW", "CONQUERED", "seized",
                  "prophet", "famine", "drought")


def _living_members(state: dict[str, Any], sid: str) -> list[Any]:
    rec = state.get("settlements", {}).get(sid)
    if rec is None:
        return []
    return [a for a in state["agents"] if a.alive and a.name in rec["members"]]


def is_literate(state: dict[str, Any], sid: str) -> bool:
    """True if settlement `sid` has at least LITERACY_MIN living members who know writing (a scribe)."""
    return sum(1 for a in _living_members(state, sid) if WRITING in a.knowledge) >= LITERACY_MIN


def _ruler(state: dict[str, Any], sid: str) -> "str | None":
    """The settlement's ruler whose policy becomes written law — monarch (M3.4) else trust-leader (M3.2)."""
    mon = state.get("monarchs", {}).get(sid)
    if mon is not None:
        return mon["monarch"]
    lead = state.get("leaders", {}).get(sid)
    return lead["leader"] if lead is not None else None


# --- 1. Writing as a tech: gated invention (reuses M1.2), spread via M1.1 -----
def discover_writing(state: dict[str, Any], turn: int, rng: "random.Random | None" = None) -> list[str]:
    """Let eligible agents INVENT writing (M1.2 machinery, ZERO LLM). Returns the inventors' names.

    Eligible = knows the prior tech (`tools`), does NOT yet know writing, and is in a SETTLEMENT holding
    a food SURPLUS (scribes need stability + spare capacity — the binding environmental prereq). The
    chance reuses `knowledge.discovery_probability` (personality/hunger shaped, probabilistic). Draws RNG
    from the seeded stream exactly as tech discovery does; gated on writing_on by the caller, so an off
    run never invents writing and stays byte-identical. Once known it spreads via ordinary M1.1 diffusion."""
    draw = (rng or random).random
    invented: list[str] = []
    for agent in [a for a in state["agents"] if a.alive]:  # stable order
        sid = getattr(agent, "settlement", None)
        if (WRITING in agent.knowledge or WRITING_PREREQ not in agent.knowledge
                or sid is None or not lineage.settlement_surplus(state, sid)):
            continue
        p = knowledge.discovery_probability(agent, WRITING, state)
        if p > 0.0 and draw() < p:
            agent.knowledge.add(WRITING)
            world.record_memory(agent, "Devised WRITING — marks that outlast the voice")
            state["events"].append(f"turn {turn}: {agent.name} devised WRITING in {sid}")
            invented.append(agent.name)
    return invented


# --- 2a. Persistent law: policy inscribed, inherited across succession --------
def _current_policy(state: dict[str, Any]) -> dict[str, Any]:
    """The ruler's policy captured as written law — the tax/levy/tribute framework of the moment."""
    from sim import monarchy
    return {"tax_rate": state.get("tax_rate", 0.25),
            "levy_rate": monarchy.MONARCH_LEVY_RATE,
            "tribute_rate": state.get("tribute_rate", 0.2)}


def enact_laws(state: dict[str, Any], turn: int) -> list[str]:
    """Inscribe / carry forward each literate settlement's written law (M4.10a). ZERO RNG.

    A literate ruled settlement with no law yet has its ruler INSCRIBE the current policy (laws[sid]);
    when the ruler has since CHANGED (a succession, M4.3), the heir INHERITS the standing written law
    (its policy persists; only `set_by` updates) — the institution outlives the individual. An
    ILLITERATE settlement never gets a written law, so its policy dies with its ruler (blank slate).
    Returns the events logged."""
    laws = state.setdefault("laws", {})
    events: list[str] = []
    for sid in sorted(state.get("settlements", {})):
        if not is_literate(state, sid):
            continue
        ruler = _ruler(state, sid)
        if ruler is None:
            continue
        law = laws.get(sid)
        if law is None:
            laws[sid] = {**_current_policy(state), "set_by": ruler, "turn": turn}
            ev = f"turn {turn}: {ruler} inscribed the written law of {sid}"
            events.append(ev)
        elif law["set_by"] != ruler:
            law["inherited_from"] = law["set_by"]
            law["set_by"] = ruler
            ev = (f"turn {turn}: {ruler} inherited the written law of {sid} "
                  f"from {law['inherited_from']} (institutional continuity)")
            events.append(ev)
    state.setdefault("events", []).extend(events)
    return events


# --- 2b. Knowledge preservation: archive + re-teach a forgotten skill ---------
def preserve_knowledge(state: dict[str, Any], turn: int) -> list[str]:
    """Archive a literate settlement's techs and RE-TEACH any that its last knower let die (M4.10b).

    Records every ARCHIVABLE tech a living member knows into archives[sid]; then, for any archived tech
    NO living member currently knows, RE-TEACHES it to one living adult member from the records — so a
    literate town cannot suffer the knowledge-extinction collapse (it can always recover farming). An
    illiterate town has no archive and loses a skill with its last knower. Deterministic, ZERO RNG."""
    archives = state.setdefault("archives", {})
    events: list[str] = []
    for sid in sorted(state.get("settlements", {})):
        if not is_literate(state, sid):
            continue
        members = _living_members(state, sid)
        arch = archives.setdefault(sid, set())
        known_now: set[str] = set()
        for m in members:
            arch |= (m.knowledge & ARCHIVABLE)
            known_now |= m.knowledge
        # Re-teach any recorded skill the living have forgotten, to a stable adult member.
        adults = sorted((m for m in members if not world.is_dependent_child(m, state)),
                        key=lambda a: a.name)
        for skill in sorted(arch - known_now):
            if not adults:
                break
            scholar = adults[0]
            scholar.knowledge.add(skill)
            world.record_memory(scholar, f"Re-learned '{skill}' from the records of {sid}")
            ev = f"turn {turn}: {sid} RE-TAUGHT '{skill}' from its records (no living master remained)"
            events.append(ev)
    state.setdefault("events", []).extend(events)
    return events


# --- 2c. Recorded history: a persistent settlement chronicle -----------------
def record_history(state: dict[str, Any], turn: int) -> list[str]:
    """Append this turn's MAJOR events that name a literate settlement to its persistent CHRONICLE
    (M4.10c). Structured entries {turn, event} — NOT prose, ZERO LLM. Illiterate settlements record
    nothing. Returns the chronicle entries added (as strings, for logging/tests)."""
    chronicles = state.setdefault("chronicles", {})
    this_turn = [e for e in state.get("events", []) if e.startswith(f"turn {turn}: ")]
    added: list[str] = []
    for sid in sorted(state.get("settlements", {})):
        if not is_literate(state, sid):
            continue
        chron = chronicles.setdefault(sid, [])
        for e in this_turn:
            text = e.split(": ", 1)[1]
            if sid in e and any(kw in e for kw in MAJOR_KEYWORDS):
                chron.append({"turn": turn, "event": text})
                added.append(f"{sid}: {text}")
    return added


def update(state: dict[str, Any], turn: int, rng: "random.Random | None" = None) -> list[str]:
    """Advance writing one turn (M4.10): invent writing, then exercise the three powers of literacy.

    Order: discover writing (seeded) -> record this turn's history (before writing's own bookkeeping
    events) -> inscribe/inherit law -> archive + re-teach knowledge. ZERO LLM; only discovery draws RNG.
    Runs LATE in the turn so it reads settled rulers and this turn's institutional events. Caller gates
    on writing_on, so an off run never calls this (no laws/archives/chronicles written) and stays
    byte-identical. Returns events."""
    events = []
    invented = discover_writing(state, turn, rng)
    record_history(state, turn)                     # chronicle institutional majors (pre-bookkeeping)
    events += enact_laws(state, turn)
    events += preserve_knowledge(state, turn)
    return events


# --- Derived read-outs (pure reads, for the summary / Arc 6 / tests) ---------
def written_law(state: dict[str, Any], sid: str) -> "dict[str, Any] | None":
    return state.get("laws", {}).get(sid)


def chronicle_of(state: dict[str, Any], sid: str) -> list:
    return state.get("chronicles", {}).get(sid, [])


def archive_of(state: dict[str, Any], sid: str) -> set:
    return state.get("archives", {}).get(sid, set())
