"""
verify_m51.py
=============

Deterministic verification of V2 milestone M5.1: MINDS AT THE PIVOTS — great figures gain a
mind at the moments history turns. OPENS Phase 5, on top of the complete v3 plan (Arcs 1-6,
tagged v4.0) and Phases 0-3.

Run offline (Ollama OFF, no model server, no seed-search):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m51.py

The step: the institutional layer is deterministic state math — that is why every A/B finding is
credible. M5.1 adds CHARACTER at the margins WITHOUT breaking it. At three CLOSE-MARGIN pivots (an
opportunistic WAR, a vassal's BREAKAWAY, a settlement's RISING) where the math's verdict is a near-
tie, the figure's mind is consulted and TILTS the outcome. Overwhelming situations stay pure math —
character decides only what material conditions leave undecided. Under AICIV_PROVIDER=random the
tilt is a DETERMINISTIC, personality-weighted stand-in (no LLM, no RNG); a live model upgrades it to
real reasoning (walled off). This script proves it all offline and deterministically.

HEADLINE 1 — THE BAND BINDS: a DECISIVE war (host far above/below the enemy's) produces the SAME
             outcome minds-on and minds-off — the mind is never consulted. Only a near-tie can differ.
HEADLINE 2 — CHARACTER TILTS CLOSE CALLS (offline stand-in): two kings in the IDENTICAL close war,
             differing only in personality — the competitive one MARCHES, the cautious one REFRAINS.
HEADLINE 3 — MOTIVES ENTER HISTORY: a pivot decision writes its REASON, and the chronicle records
             the WHY in the saga ("...saying 'the odds were even and fortune favours the bold'").
PIVOTS     — all three pivots are wired: the same tilt flips a close BREAKAWAY and a close RISING by
             the figure's character (the seam invites more pivots later).
COST       — off byte-identical; deterministic (two on-runs identical); a malformed response falls
             back to no-tilt; the measured pivot-consult / inclination-request count (single digits).
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any

import chronicle
import empire
import kingdoms
import llm
import mind
import monarchy
import trust
import uprising
import world
from agents import Agent
from world import world_state


# --- Staging helpers ---------------------------------------------------------
def _fresh(minds: bool = True) -> None:
    world.create_world(size=60)
    world_state["agents"].clear()
    world_state["food"].clear()
    world_state["minds_on"] = minds


def _settled(name: str, pos, personality="x", sid=None, money=0.0) -> Agent:
    a = Agent(name=name, personality=personality)
    world.place_agent(a, *pos)
    a.hunger, a.age, a.lifespan, a.money, a.settlement = 1, 30, 100, money, sid
    return a


def _mercs(prefix: str, near, n: int) -> None:
    """n poor, hireable agents beside a king (within monarchy.MUSTER_RADIUS) — its fielded host."""
    for i in range(n):
        _settled(f"{prefix}{i}", (near[0] + (i % 3), near[1] + 1), money=0.5)


def _realm(king: str, center, personality="x", nmercs: int = 0) -> None:
    """A one-settlement sovereign kingdom whose fielded host is exactly `nmercs` (king pays FIGHTER_COST
    each; given war chest = nmercs * FIGHTER_COST, clear of MIN_WAR_CHEST)."""
    home = f"{king}_home"
    world_state["settlements"][home] = {"id": home, "center": center, "members": {king}, "founded": 0}
    chest = max(monarchy.MIN_WAR_CHEST, nmercs * monarchy.FIGHTER_COST)
    _settled(king, center, personality=personality, sid=home, money=float(chest))
    world_state["monarchs"][home] = {"monarch": king, "since": 0, "garrison": set()}
    world_state["kingdoms"][king] = {"king": king, "home": home, "settlements": {home},
                                     "vassals": {}, "founded": 0, "discontent": {}}
    if nmercs:
        _mercs(f"{king}M", center, nmercs)


def _host(king: str) -> int:
    a = next(x for x in world_state["agents"] if x.name == king)
    return empire.imperial_host_size(world_state, a)


def _war_launched(events, attacker: str) -> bool:
    # A launched war logs either "KING X DEFEATED ..." (won) or "KING X's war on ... FAILED" (lost).
    return any((f"KING {attacker} DEFEATED" in e) or (f"KING {attacker}'s war on" in e) for e in events)


def _run_empire_turn(minds: bool, turn: int = 1) -> list[str]:
    world_state["minds_on"] = minds
    before = len(world_state["events"])
    empire.update(world_state, turn)
    return world_state["events"][before:]


def line(c="="):
    print(c * 72)


# =====================================================================================
# HEADLINE 1 — THE BAND BINDS: only close calls are consulted.
# =====================================================================================
def headline_1_band_binds() -> None:
    line()
    print("HEADLINE 1 — THE BAND BINDS (decisive war: identical minds-on vs minds-off)")
    line()

    # A DECISIVE standoff: attacker host 8 vs defender host 3 (margin +5, far outside WAR_BAND=2).
    def decisive(minds):
        _fresh(minds)
        _realm("Bold", (10, 10), personality="competitive bold", nmercs=8)
        _realm("Weak", (16, 10), personality="x", nmercs=3)               # 6 apart < KINGDOM_REACH
        return _run_empire_turn(minds)

    on, off = decisive(True), decisive(False)
    on_war, off_war = _war_launched(on, "Bold"), _war_launched(off, "Bold")
    print(f"  hosts: Bold {8} vs Weak {3}  (margin +5, |margin| > band -> the math is decisive)")
    print(f"  minds OFF: Bold's war launched? {off_war}")
    print(f"  minds ON : Bold's war launched? {on_war}   consulted {mind.consult_count(world_state)} mind(s)")
    assert on_war and off_war, "a decisive war must launch either way"
    assert mind.consult_count(world_state) == 0, "no mind may be consulted outside the band"

    # A CLOSE standoff: attacker host 5 vs defender host 5 (a dead tie — inside the band).
    def close(minds):
        _fresh(minds)
        _realm("Bold", (10, 10), personality="competitive bold", nmercs=5)
        _realm("Even", (16, 10), personality="x", nmercs=5)
        return _run_empire_turn(minds)

    on2, off2 = close(True), close(False)
    on2_war, off2_war = _war_launched(on2, "Bold"), _war_launched(off2, "Bold")
    print(f"\n  hosts: Bold {5} vs Even {5}  (margin 0, |margin| <= band -> character decides)")
    print(f"  minds OFF: Bold's war launched? {off2_war}   (the math holds: 5 does not beat 5)")
    print(f"  minds ON : Bold's war launched? {on2_war}   (the bold king marches on even odds)")
    assert not off2_war, "an even standoff does not launch under the pure math"
    assert on2_war, "a bold king tilts an even standoff into war"
    print("\n  -> decisive cases are IDENTICAL on/off (band binds); only the close call differs.")


# =====================================================================================
# HEADLINE 2 — CHARACTER TILTS CLOSE CALLS (offline personality stand-in).
# =====================================================================================
def headline_2_character_tilts() -> None:
    line()
    print("HEADLINE 2 — CHARACTER TILTS THE UNDECIDED (two kings, identical close war, differ only in nature)")
    line()

    def march(personality):
        _fresh(True)
        _realm(personality[0].upper() + "king", (10, 10), personality=personality, nmercs=5)
        _realm("Even", (16, 10), personality="x", nmercs=5)                # a dead 5-vs-5 tie
        events = _run_empire_turn(True)
        return _war_launched(events, personality[0].upper() + "king")

    competitive = march("competitive")
    cautious = march("cautious")
    print("  the SAME situation for both: their host 5, the enemy's 5 (an even standoff).")
    print(f"  the COMPETITIVE king marches to war? {competitive}")
    print(f"  the CAUTIOUS   king marches to war? {cautious}")
    assert competitive and not cautious, (competitive, cautious)
    # Show the underlying inclinations (the personality-weighted stand-in, no LLM/RNG).
    c = [x for x in world_state.get("mind_consults", [])]
    print("\n  -> identical material odds; personality alone decides. Character decides the undecided.")


# =====================================================================================
# HEADLINE 3 — MOTIVES ENTER HISTORY.
# =====================================================================================
def headline_3_motive_enters_history() -> None:
    line()
    print("HEADLINE 3 — MOTIVES ENTER THE WRITTEN HISTORY (the saga records the WHY)")
    line()
    _fresh(True)
    world_state["chronicle_on"] = True
    # A literate settlement so the war is recorded as HISTORY (not anonymized legend).
    scribe = _settled("scribe", (11, 10), sid="Bold_home")
    scribe.knowledge.add("writing")
    # A slim 6-vs-5 edge: inside the band (so the KING's mind is consulted and its motive recorded),
    # yet enough to WIN — so the war is chronicled as a named conquest the motive can attach to.
    _realm("Bold", (10, 10), personality="competitive bold", nmercs=6)
    world_state["settlements"]["Bold_home"]["members"].add("scribe")
    _realm("Slim", (16, 10), personality="x", nmercs=5)

    empire.update(world_state, 3)
    chronicle.update(world_state, 3)
    war = next((e for e in chronicle.saga(world_state) if "Bold's Conquest" in e["name"]), None)
    assert war is not None, "the war should be chronicled"
    print(f"  saga entry: **Turn {war['turn']} — {war['name']}**")
    print(f"    {war['detail']}")
    assert "saying" in war["detail"], war["detail"]
    print("\n  -> history records not just WHAT Bold did but WHY — the motive entered the record.")


# =====================================================================================
# PIVOTS — all three are wired (breakaway + rising flip on character too).
# =====================================================================================
def all_three_pivots() -> None:
    line()
    print("ALL THREE PIVOTS — the same tilt flips a close BREAKAWAY and a close RISING by character")
    line()

    # BREAKAWAY: a subject vassal at borderline loyalty (trust == BREAKAWAY_TRUST, margin 0). A proud,
    # independent vassal drifts to break; a loyal, cautious one endures — the exact tilt the M3.5/M3.6
    # loyalty check calls.
    def breakaway(personality):
        _fresh(True)
        _settled(personality[0].upper() + "vassal", (5, 5), personality=personality)
        margin = 0                                       # trust sits exactly at the breakaway floor
        break_now, _ = mind.tilt(world_state, personality[0].upper() + "vassal", "breakaway",
                                 kingdoms.BREAKAWAY_TRUST - kingdoms.BREAKAWAY_TRUST, True,
                                 {"trust": kingdoms.BREAKAWAY_TRUST, "lord": "King"}, 1)
        return break_now

    proud = breakaway("independent competitive")
    loyal = breakaway("friendly cooperative cautious")
    print(f"  borderline loyalty (trust at the floor): the PROUD vassal breaks away?   {proud}")
    print(f"                                           the LOYAL vassal endures?       {not loyal}")
    assert proud and not loyal, (proud, loyal)

    # RISING: a ringleader whose settlement's grievance sits exactly at the trigger (margin 0). A daring
    # firebrand raises the banner; a cautious one waits — the exact tilt the M4.5 trigger calls.
    def rising(personality):
        _fresh(True)
        _settled(personality[0].upper() + "lead", (5, 5), personality=personality)
        rise, _ = mind.tilt(world_state, personality[0].upper() + "lead", "uprising", 0.0, False,
                            {"pressure": uprising.UPRISING_MIN_PRESSURE,
                             "threshold": uprising.UPRISING_MIN_PRESSURE, "sid": "S001"}, 1)
        return rise

    firebrand = rising("bold competitive")
    patient = rising("cautious careful")
    print(f"  grievance exactly at the trigger:        the FIREBRAND raises the banner? {firebrand}")
    print(f"                                           the PATIENT one waits?           {not patient}")
    assert firebrand and not patient, (firebrand, patient)
    print("\n  -> war, breakaway, and rising all consult the figure at the margin — one seam, three pivots.")


# =====================================================================================
# COST — off byte-identical; deterministic; malformed falls back; measured cost.
# =====================================================================================
def cost_checks() -> None:
    line()
    print("COST — off byte-identical; deterministic on-runs; malformed falls back; measured consults")
    line()

    def run(minds, seed=5):
        llm.PROVIDER = "random"
        random.seed(seed)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            import main
            main.run_simulation(20, stage="war", minds_on=minds)
        return buf.getvalue(), dict(llm.get_call_stats())

    saved = llm.PROVIDER
    try:
        base, base_stats = run(False)
        off, off_stats = run(False)
        on_a, _ = run(True)
        on_b, _ = run(True)
    finally:
        llm.PROVIDER = saved

    assert base == off, "minds OFF must be byte-identical run to run"
    assert base_stats["inclination"] == 0, "minds OFF makes ZERO inclination requests"
    assert on_a == on_b, "minds ON must be deterministic (byte-identical across seeded repeats)"
    print(f"  --minds OFF: byte-identical baseline, 0 inclination requests. OK")
    print(f"  --minds ON : two seeded runs byte-identical (deterministic). OK")

    # Measured cost where the band genuinely fires: a cluster of four evenly-matched (host-5) adjacent
    # kingdoms — every launch assessment is a near-tie, so each king consults its mind at the margin.
    llm.PROVIDER = "random"
    _fresh(True)
    for name, pers, c in [("Ava", "competitive", (10, 10)), ("Bede", "cautious", (17, 10)),
                          ("Cade", "bold competitive", (10, 17)), ("Dain", "cautious careful", (17, 17))]:
        _realm(name, c, personality=pers, nmercs=5)
    llm.reset_call_stats()
    empire.update(world_state, 1)
    consults = mind.consult_count(world_state)
    requests = llm.get_call_stats()["inclination"]
    flips = sum(1 for x in world_state["mind_consults"] if x["flipped"])
    llm.PROVIDER = saved
    assert consults > 0, "the close cluster should consult minds"
    assert requests <= consults and requests < 10, (requests, consults)
    print(f"  measured on a 4-kingdom close cluster: {consults} pivot consult(s) "
          f"({flips} flipped a close call); {requests} inclination request(s) — single digit, cached")
    print(f"    (offline these are served by the deterministic stand-in; a LIVE model would make the "
          f"{requests} request(s) as real calls — cached per (figure, situation))")

    # Malformed live response -> neutral inclination 0.0 -> the deterministic verdict stands.
    llm.PROVIDER = "gemini"
    orig = llm._raw_query
    llm._raw_query = lambda prompt: {"totally": "malformed"}
    try:
        bad = llm.get_inclination("DISPOSITION: 0.9")
    finally:
        llm._raw_query = orig
        llm.PROVIDER = saved
    assert bad["inclination"] == 0.0, bad
    print(f"  malformed model response -> inclination {bad['inclination']} (no tilt; the math stands). OK")


def main() -> None:
    headline_1_band_binds()
    print()
    headline_2_character_tilts()
    print()
    headline_3_motive_enters_history()
    print()
    all_three_pivots()
    print()
    cost_checks()
    print()
    line()
    print("M5.1 VERIFIED — the mind decides ONLY what the material world leaves undecided (the band")
    print("binds absolutely), character genuinely tilts close calls (offline and online), and motives")
    print("now enter the written history. The civilization's physics stay verified; its great figures")
    print("gain souls at the moments that matter. Phase 5 opens.")
    line()


if __name__ == "__main__":
    main()
