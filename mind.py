"""
mind.py
=======

MINDS AT THE PIVOTS — great figures gain minds at the moments history turns
(V2 milestone M5.1, OPENS Phase 5). On top of the complete v3 plan (Arcs 1-6, tagged v4.0).

The step M5.1 makes — CHARACTER at the margins, without breaking the physics
------------------------------------------------------------------------------
For thirty-five milestones the institutional layer (war, breakaway, revolt) has been pure,
deterministic STATE MATH — and that determinism is exactly why every A/B finding is credible.
M5.1 adds a soul to the great figures WITHOUT breaking that: at a handful of PIVOT POINTS where
the math's verdict is CLOSE — where material conditions genuinely leave the outcome undecided —
the figure's MIND is consulted and TILTS the result. Decisive situations stay pure math; a mind
is NEVER consulted when the odds are overwhelming. Character decides only the undecided.

The three pivots (a clean seam invites more — treaty-breaking, succession disputes — later):
  - WAR       — a king deciding an opportunistic war        (empire.update launch check).
  - BREAKAWAY — a vassal/subject-king deciding to break     (kingdoms/empire loyalty check).
  - UPRISING  — a resentful settlement deciding to rise      (uprising trigger check).

The close-margin band (the verifiability spine)
-----------------------------------------------
At each pivot the caller computes the deterministic verdict AND a signed MARGIN (distance from the
threshold; positive = the math leans toward ACTING). If |margin| > BAND the math's verdict stands,
UNTOUCHED — the mind is never even consulted (no mind talks a 3-host king into attacking a 12-host
coalition; no contented town is talked into revolt). If |margin| <= BAND the mind's INCLINATION
(-1..+1) tilts the effective margin and MAY flip the outcome. BAND sizes are documented and tunable
per pivot. The mind can only decide what the world left undecided.

The mind (the consult) — mirrors the v1 strategy layer's provider abstraction
-----------------------------------------------------------------------------
`llm.get_inclination(prompt)` is the provider-agnostic primitive (like get_strategy). The prompt is
the figure's PERSONALITY + BELIEFS (M4.7) + the situation + the question; the answer is an
inclination in [-1, 1] plus a short REASON. The reason is fed to the CHRONICLE (M4.16) so history
records not just WHAT figures did but WHY. Consults are CACHED per (figure, situation-signature),
and a malformed/absent response falls back to inclination 0.0 (the math's verdict stands) — never a
crash, never a stall.

Determinism (the ABSOLUTE constraint)
-------------------------------------
Under AICIV_PROVIDER=random (every test and verify) the inclination is the OFFLINE STAND-IN: a
DETERMINISTIC, personality-weighted disposition computed here and carried through llm.get_inclination
(which reads it from a marker in the prompt, exactly as _random_decision reads "food is North"). It
consumes NO global RNG, so the institutional layer stays RNG-free and two minds-on runs are
byte-identical. The stand-in is itself the feature: personality-at-the-pivots works offline; a live
LLM upgrades it to genuine reasoning online (walled off like --narrate — prose only, no new physics).
With --minds OFF, `tilt` returns the caller's deterministic verdict and writes nothing, so an off run
is byte-identical to v1 everywhere.
"""

from __future__ import annotations

from typing import Any

import llm

# --- The close-margin BAND per pivot (documented, tunable) -------------------
# The band is the half-width, in each pivot's own units, of the "too close to call" zone around the
# deterministic threshold. Outside it the math is decisive and the mind is never consulted; inside it
# the figure's character tilts the call. Chosen small so only genuinely marginal junctures are opened
# to character — the physics stays in command of everything else.
WAR_BAND = 2.0        # host-count units: the two hosts are within 2 fighters of a tie.
BREAKAWAY_BAND = 1.0  # trust units: loyalty sits within 1 of the breakaway floor.
UPRISING_BAND = 3.0   # pressure units: aggregate grievance sits within 3 of the trigger.

_BAND = {"war": WAR_BAND, "breakaway": BREAKAWAY_BAND, "uprising": UPRISING_BAND}

# Tilt strength per pivot: a full |inclination| = 1 moves the effective margin across the whole band
# (+0.5 so a wholly decisive character can flip even a case sitting at the band's edge). A moderate
# character moves it proportionally less, so a slim material lead usually survives a mild temperament.
_TILT = {k: v + 0.5 for k, v in _BAND.items()}


# --- Agent / personality / belief reads (all pure) ---------------------------
def _find(state: dict[str, Any], name: str | None) -> Any | None:
    if name is None:
        return None
    for a in state["agents"]:
        if a.alive and a.name == name:
            return a
    return None


def _personality(agent: Any) -> Any:
    """The agent's typed Personality (reuses strategy.py's per-agent cache)."""
    import strategy
    return strategy.get_personality(agent)


def _disposition(agent: Any, state: dict[str, Any], pivot: str) -> float:
    """The figure's base inclination to ACT at this pivot, in [-1, 1], from typed traits (+ belief nudge).

    A single axis of BOLDNESS-vs-RESTRAINT, weighted per pivot from the four traits (personality.py):
    curiosity (daring) and independence (competitive/proud) push toward the dramatic act; caution and
    friendliness (loyalty/peaceableness) pull back. This is what makes a competitive king MARCH and a
    cautious one REFRAIN in the identical close situation — character deciding the undecided, offline.
    """
    p = _personality(agent)
    if pivot == "war":
        # A bold, competitive crown marches; a cautious, peaceable one holds.
        v = (p.curiosity + p.independence) - (p.caution + p.friendliness)
    elif pivot == "breakaway":
        # Pride (independence) drives a break; caution and loyalty (friendliness) counsel endurance.
        v = (1.5 * p.independence + 0.5 * p.curiosity) - (p.caution + p.friendliness)
    else:  # uprising
        # A daring firebrand rises; a cautious, contented temperament simmers.
        v = (p.curiosity + p.independence) - (1.5 * p.caution + 0.5 * p.friendliness)
    v += _belief_nudge(agent.name, state, pivot)
    return max(-1.0, min(1.0, v))


# Beliefs (M4.7) that colour a figure's disposition at the pivots — a small nudge so personality
# dominates (the headline stand-in is personality-only) while conviction still leaves a fingerprint.
_BELIEF_NUDGE = {
    "the strong take what they want": {"war": 0.20, "breakaway": 0.15, "uprising": 0.20},
    "wealth is virtue": {"war": 0.10},
    "we are stronger together": {"war": -0.15, "breakaway": -0.25},
    "the land provides": {"war": -0.10, "uprising": -0.20},
    "greed is a poison": {"war": -0.20},
    "the world is cruel": {"uprising": 0.15, "breakaway": 0.10},
}


def _belief_nudge(name: str, state: dict[str, Any], pivot: str) -> float:
    if not state.get("beliefs_on"):
        return 0.0
    try:
        import beliefs
        held = beliefs.agent_beliefs(name, state)
    except Exception:
        return 0.0
    return sum(_BELIEF_NUDGE.get(b, {}).get(pivot, 0.0) for b in held)


# --- Offline reason templates (the WHY that enters history) ------------------
# Keyed (pivot, acted?) — a template string derived from the figure's leaning. Online these are the
# model's own words; offline they are these deterministic templates, so a motive always reaches the
# chronicle. Phrased as the figure's stated motive so the saga reads "X marched, saying '...'".
_REASON = {
    ("war", True): "the odds were even and fortune favours the bold",
    ("war", False): "the risk was too great to march on so slim an edge",
    ("breakaway", True): "no distant crown deserves my loyalty when it cannot command my heart",
    ("breakaway", False): "better a place within the realm than a lonely, defenceless independence",
    ("uprising", True): "the hour had come, and the people would follow",
    ("uprising", False): "the time was not yet ripe to raise the banner",
}


# --- The consult -------------------------------------------------------------
def _signature(pivot: str, figure: str, situation: dict[str, Any]) -> str:
    """A stable cache key: the pivot, the figure, and the situation's decisive numbers (sorted)."""
    parts = [pivot, figure]
    for k in sorted(situation):
        if k == "turn":
            continue  # the same standoff on different turns is the same question — cache across turns
        parts.append(f"{k}={situation[k]}")
    return "|".join(parts)


def _prompt(agent: Any, state: dict[str, Any], pivot: str, situation: dict[str, Any],
            disposition: float) -> str:
    """Build the consult prompt: personality + beliefs + situation + question, plus offline markers.

    The narrative half is what a live model reads and reasons over. The trailing marker lines
    (DISPOSITION / OFFLINE_REASON) are what the offline stand-in reads — mirroring how llm._random_*
    reads cheap signals from the prompt — so the SAME entry point serves both providers.
    """
    p = _personality(agent)
    held = ""
    if state.get("beliefs_on"):
        try:
            import beliefs
            bs = sorted(beliefs.agent_beliefs(agent.name, state))
            held = "; ".join(bs) if bs else "no firm convictions"
        except Exception:
            held = "no firm convictions"
    acted_reason = _REASON[(pivot, disposition >= 0)]
    lines = [
        f"You are {agent.name}, a leader whose nature is {p.describe()}.",
        f"Your convictions: {held}." if held else "",
        _question(pivot, situation),
        "Answer as JSON: {\"inclination\": <-1.0 restraint .. +1.0 action>, \"reason\": <one short clause>}.",
        # --- offline stand-in markers (ignored by a live model's reasoning) ---
        f"DISPOSITION: {disposition:.4f}",
        f"OFFLINE_REASON: {acted_reason}",
    ]
    return "\n".join(ln for ln in lines if ln)


def _question(pivot: str, s: dict[str, Any]) -> str:
    if pivot == "war":
        return (f"Your host numbers {s.get('att')}; the enemy's {s.get('def')}. "
                f"Do you march to war, or hold?")
    if pivot == "breakaway":
        return (f"Your loyalty to {s.get('lord')} has worn to {s.get('trust')}. "
                f"Do you break away and reclaim your independence, or endure?")
    return (f"Your people's grievance has risen near the breaking point "
            f"({s.get('pressure')} against a threshold of {s.get('threshold')}). "
            f"Do you raise the banner of revolt, or wait?")


def _consult(state: dict[str, Any], agent: Any, pivot: str, situation: dict[str, Any]) -> dict[str, Any]:
    """Ask the figure's mind for an inclination in [-1, 1] + reason. Cached; robust; personality-driven.

    Cache hits cost nothing. On a miss the (offline or live) llm.get_inclination is consulted exactly
    once and the result is cached per (figure, situation-signature). The offline stand-in returns the
    personality disposition carried in the prompt; a live model returns genuine reasoning; a malformed
    response degrades to inclination 0.0 (the math stands).
    """
    cache = state.setdefault("mind_cache", {})
    sig = _signature(pivot, agent.name, situation)
    if sig in cache:
        return cache[sig]
    disposition = _disposition(agent, state, pivot)
    result = llm.get_inclination(_prompt(agent, state, pivot, situation, disposition))
    cache[sig] = result
    return result


# --- The public pivot entry point --------------------------------------------
def tilt(state: dict[str, Any], figure_name: str, pivot: str, margin: float, base_act: bool,
         situation: dict[str, Any], turn: int) -> "tuple[bool, dict | None]":
    """Decide a pivot: return (verdict, consult_record | None). The mind may only tilt the undecided.

    `margin` is the signed distance from the threshold (positive = the math leans to ACT); `base_act`
    is the caller's exact deterministic verdict (so OFF / out-of-band is byte-identical). Behaviour:
      - minds OFF, or |margin| > band  -> return (base_act, None): the math stands, nothing consulted.
      - in-band, inclination == 0       -> return (base_act, None-equivalent record): neutral mind,
                                           the math STILL stands (this is also the malformed fallback).
      - in-band, inclination != 0       -> tilt the effective margin; the verdict MAY flip.
    On an ACTED, flipped-or-not in-band decision the figure's REASON is recorded for the chronicle so
    the motive enters history.
    """
    if not state.get("minds_on") or abs(margin) > _BAND[pivot]:
        return base_act, None
    agent = _find(state, figure_name)
    if agent is None:
        return base_act, None

    consult = _consult(state, agent, pivot, situation)
    incl = float(consult.get("inclination", 0.0))
    if incl == 0.0:
        verdict = base_act                      # a characterless mind leaves the physics untouched
    else:
        verdict = (margin + incl * _TILT[pivot]) > 0

    record = {"turn": turn, "figure": figure_name, "pivot": pivot, "margin": margin,
              "sid": situation.get("sid"), "inclination": incl,
              "reason": str(consult.get("reason", "")),
              "acted": verdict, "flipped": verdict != base_act,
              "reason_text": _REASON[(pivot, verdict)]}
    state.setdefault("mind_consults", []).append(record)
    # Feed the motive to the chronicle ONLY when the figure ACTS (an act produces the event the saga
    # will attach the WHY to; a refusal produces no historical event to annotate).
    if verdict:
        motive = record["reason"] or record["reason_text"]
        state.setdefault("mind_motives", []).append(
            {"turn": turn, "figure": figure_name, "pivot": pivot,
             "sid": situation.get("sid"), "reason": motive})
    return verdict, record


def motive_for(state: dict[str, Any], turn: int, figure: str) -> "str | None":
    """The recorded motive for `figure`'s acted pivot on `turn`, if any — read by the chronicle (M4.16)."""
    for m in state.get("mind_motives", []):
        if m["turn"] == turn and m["figure"] == figure:
            return m["reason"]
    return None


def motive_at(state: dict[str, Any], turn: int, sid: str) -> "str | None":
    """The recorded motive for a settlement-scoped acted pivot (an uprising) on `turn` — for the chronicle."""
    for m in state.get("mind_motives", []):
        if m["turn"] == turn and m.get("sid") == sid:
            return m["reason"]
    return None


def _breakaway_window() -> int:
    """The hysteresis span (turns) between a break DECISION and the secession EVENT — kingdoms.BREAKAWAY_PATIENCE.

    Lazy import (kingdoms imports mind, so a module-level import would cycle); falls back to a safe 2 if
    kingdoms cannot be read, so the lookback never crashes the read-only chronicle pass."""
    try:
        import kingdoms
        return kingdoms.BREAKAWAY_PATIENCE
    except Exception:
        return 2


def breakaway_motive(state: dict[str, Any], event_turn: int, figure: str) -> "str | None":
    """The motive behind `figure`'s breakaway EVENT at `event_turn`, tolerant of the breakaway HYSTERESIS.

    Unlike war (the event fires the very turn the mind is consulted, so an exact turn match suffices), a
    breakaway has a PATIENCE delay: the mind is consulted the turn loyalty first slips into the close
    band, but the secession EVENT only fires BREAKAWAY_PATIENCE turns later — and by then the margin may
    have gone decisive, so NO fresh consult happens on the event turn. A same-turn lookup (motive_for)
    therefore misses it. So look BACK across the hysteresis window for the most recent breakaway decision
    this figure took that drove the secession. Returns the reason, or None if the break was never a
    mind-driven decision (loyalty decisive throughout — no consult, correctly no motive)."""
    window = _breakaway_window()
    best = None
    for m in state.get("mind_motives", []):
        if (m.get("pivot") == "breakaway" and m["figure"] == figure
                and event_turn - window <= m["turn"] <= event_turn):
            if best is None or m["turn"] > best["turn"]:
                best = m
    return best["reason"] if best else None


def pivot_consulted(state: dict[str, Any], event_turn: int, pivot: str, figure: "str | None" = None,
                    sid: "str | None" = None, window: int = 0, acted_only: bool = False) -> bool:
    """Whether a `pivot` mind was CONSULTED (by figure or by settlement) within `window` turns up to `event_turn`.

    With `acted_only`, count ONLY consults whose verdict was to ACT — i.e. exactly the consults that
    RECORD a motive (a consult that DECLINES writes none). That is what the chronicle wants when telling
    'no mind drove this' (out-of-band decisive, OR the mind declined and the act came from later decisive
    math — correctly blank) apart from 'a mind DID act here but its motive never attached' (a real lookup
    bug). Counting declined consults would false-flag the blank case as a bug."""
    for c in state.get("mind_consults", []):
        if c.get("pivot") != pivot or not (event_turn - window <= c["turn"] <= event_turn):
            continue
        if acted_only and not c.get("acted"):
            continue
        if figure is not None and c.get("figure") == figure:
            return True
        if sid is not None and c.get("sid") == sid:
            return True
    return False


def consult_count(state: dict[str, Any]) -> int:
    """How many times a mind was actually consulted this run (for the cost readout / verify)."""
    return len(state.get("mind_consults", []))
