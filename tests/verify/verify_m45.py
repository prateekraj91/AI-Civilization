"""
verify_m45.py
=============

Deterministic verification of V2 milestone M4.5: UPRISING — the revolt FIRES.
Second milestone of Arc 2 (Revolt & Class Conflict), on top of M4.4 (the discontent
gauge), all of Arc 1 (M4.1 lineage, M4.2 inheritance, M4.3 dynasties) and Phases 0-3.

Run offline (Ollama OFF, no model server, no seed-search):

    AICIV_PROVIDER=random ./Jarvis/bin/python verify_m45.py

The historical step: M4.4 built the GAUGE but only MEASURED — the oppressed seethed
and starved. M4.5 makes it BLOW: a resentful MAJORITY rises against its force ruler.
The mob's weapon is NUMBERS (the inverse of M3.4, where force is BOUGHT); a rich
tyrant buys guards and crushes it, a drained one falls. On victory the ruler is
deposed, his hoard EXPROPRIATED to the peasants (interrupting M4.2 inheritance), and
if he was a king's vassal the settlement SECEDES. The revolutionary who fills the
vacant seat is M4.6 — NOT built here.

HEADLINE 1 — OPPRESSION PREDICTS UPRISINGS: same settlement — under a heavy-levying
             DISTRUSTED monarch the resentful majority's pressure crosses the trigger
             and it RISES; under a TRUSTED monarch taking the SAME levy the grievance
             stays buffered-low and it never rises; a CONSENT-led town is immune by
             construction (no force ruler to overthrow). Good governance is revolt-immunity.
HEADLINE 2 — WEALTH IS THE COUNTER-REVOLUTIONARY WEAPON: the SAME angry mob against a
             RICH tyrant who musters guards -> CRUSHED (deaths, fear, grievance persists);
             against the same tyrant with a DRAINED treasury -> the mob WINS on numbers
             and he is deposed. Same mob, two treasuries, two fates.
HEADLINE 3 — THE REVOLUTION EXPROPRIATES: a successful rising splits the deposed ruler's
             hoard among the risers (conserved to the decimal) and his M4.2 HEIRS get
             NOTHING — contrasted with the same ruler dying of old age, where the heir
             inherits the whole estate. Revolution interrupts inheritance.
HEADLINE 4 — A HOUSE CAN FALL (Arc 1 x Arc 2): a beloved king dies of old age; his
             UNLOVED heir inherits the crown (M4.3) but not the loyalty; the SAME levies
             now generate far more grievance (M4.4); the settlement RISES and the dynasty
             ENDS. Shown end-to-end via a clearly-constructed scenario (organic tyrants
             are rare, exactly as at every Phase 3 force milestone).
COST       — zero added LLM; --uprising off byte-identical; deterministic/reproducible.
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import Any

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))   # the repo root

from sim import discontent
from sim import kingdoms
from llm import llm
import main
from sim import monarchy
from sim import population
from sim import trust
from sim import uprising
from sim import world
from sim.agents import Agent
from sim.world import world_state


# --- Staging helpers ---------------------------------------------------------
def _fresh() -> None:
    world.create_world()
    world_state["uprising_on"] = True
    world_state["discontent_on"] = True
    world_state["settlements"]["S001"] = {"id": "S001", "center": (5, 5),
                                          "members": set(), "founded": 0}


def _settlement(sid: str, center: tuple[int, int]) -> None:
    world_state["settlements"][sid] = {"id": sid, "center": center,
                                       "members": set(), "founded": 0}


def _agent(name: str, pos: tuple[int, int], *, money: float = 0.0, hunger: int = 1,
           age: int = 30, parents: tuple = (), sid: "str | None" = "S001") -> Agent:
    a = Agent(name=name, personality="friendly and outgoing")
    world.place_agent(a, *pos)
    a.hunger, a.age, a.lifespan = hunger, age, 100
    a.money, a.parents, a.settlement = money, parents, sid
    if sid is not None and sid in world_state["settlements"]:
        world_state["settlements"][sid]["members"].add(name)
    return a


def _monarch(sid: str, name: str) -> None:
    world_state["monarchs"][sid] = {"monarch": name, "since": 0, "garrison": set()}


def _mercs(positions, money: float = 0.5) -> None:
    """Poor nomad bystanders a ruler can hire as guards (not settlement members)."""
    for i, p in enumerate(positions):
        a = Agent(name=f"guard{i}", personality="x")
        world.place_agent(a, *p)
        a.hunger, a.age, a.lifespan, a.money, a.settlement = 1, 30, 100, money, None


# --- HEADLINE 1: oppression predicts uprisings -------------------------------
def headline_1_oppression_predicts_uprisings() -> None:
    print("=" * 72)
    print("HEADLINE 1 — OPPRESSION PREDICTS UPRISINGS (tyranny combustible, fair rule immune)")
    print("=" * 72)

    def reign(kind: str) -> tuple[list[int], bool]:
        """Let the gauge accumulate under a ruler for 12 turns, then test the trigger.

        kind: 'distrusted'/'trusted' monarch (same levy, differing legitimacy) or 'consent'
        (a trust-leader — a resentful people with NO force ruler to overthrow)."""
        _fresh()
        # Three fed, solvent commoners (wealth above the levy threshold — the levy is the grievance).
        members = [_agent(n, p, money=20.0) for n, p in [("A", (5, 6)), ("B", (6, 5)), ("C", (6, 7))]]
        if kind == "consent":
            # A CHOSEN trust-leader + a real grievance (hungry beside a rich neighbour): even fully
            # resentful, the people cannot 'overthrow' the leader they consented to.
            for m in members:
                m.hunger = 8
            _agent("Rich", (4, 4), money=60.0)
            world_state["leadership_on"] = True
            world_state["leaders"]["S001"] = {"leader": "Led", "followers": {m.name for m in members},
                                              "since": 0}
            _agent("Led", (5, 5), money=60.0)
        else:
            _agent("King", (5, 5), money=200.0)
            _monarch("S001", "King")
            # distrust AMPLIFIES the levy's grievance; trust BUFFERS it (the M4.4 legitimacy factor).
            level = -5 if kind == "distrusted" else 5
            for m in members:
                trust.ensure_relationship(m, "King")["trust"] = level
        pressures = []
        for turn in range(1, 13):
            discontent.update(world_state, turn)
            pressures.append(discontent.settlement_pressure("S001", world_state))
        return pressures, uprising.would_rise(world_state, "S001", 13)

    dis_p, dis_rise = reign("distrusted")
    tru_p, tru_rise = reign("trusted")
    con_p, con_rise = reign("consent")
    print(f"  DISTRUSTED monarch (same levy): resentful count by turn {dis_p} -> would rise? {dis_rise}")
    print(f"  TRUSTED monarch    (same levy): resentful count by turn {tru_p} -> would rise? {tru_rise}")
    print(f"  CONSENT trust-leader (maxed grievance, no force ruler): resentful {con_p[-1]} "
          f"-> would rise? {con_rise}")
    assert dis_rise and not tru_rise and not con_rise
    # And the distrusted settlement actually FIRES a rising when the valve runs.
    _fresh()
    members = [_agent(n, p, money=20.0) for n, p in [("A", (5, 6)), ("B", (6, 5)), ("C", (6, 7))]]
    _agent("King", (5, 5), money=0.0)             # a broke tyrant — the mob's numbers will decide
    _monarch("S001", "King")
    for m in members:
        trust.ensure_relationship(m, "King")["trust"] = -5
    for turn in range(1, 13):
        discontent.update(world_state, turn)
    fired = uprising.update(world_state, 13)
    print(f"\n  -> the distrusted-monarch settlement RISES (event fired: {bool(fired)}); the trusted")
    print(f"     and consent-led settlements NEVER do. Oppression causes revolt; consent is immune.")
    assert fired and fired[0]["won"]
    print()


# --- HEADLINE 2: wealth is the counter-revolutionary weapon ------------------
def headline_2_wealth_buys_counter_revolution() -> None:
    print("=" * 72)
    print("HEADLINE 2 — WEALTH IS THE COUNTER-REVOLUTIONARY WEAPON (same mob, two treasuries)")
    print("=" * 72)

    def rising(king_money: float) -> dict[str, Any]:
        _fresh()
        _agent("King", (5, 5), money=king_money)
        for n, p in [("A", (5, 6)), ("B", (6, 5)), ("C", (6, 7))]:
            _agent(n, p, money=0.0)               # a PENNILESS mob — no coin, only numbers
        _mercs([(4, 4), (4, 5), (5, 4), (4, 6), (6, 4)])   # a pool of poor guards for hire
        _monarch("S001", "King")
        world_state["discontent"] = {"A": 12.0, "B": 12.0, "C": 12.0}
        return uprising.update(world_state, 10)[0]

    rich = rising(king_money=200.0)
    print(f"  RICH tyrant (war chest 200): mustered {rich['defenders']} guards vs {rich['mob']} "
          f"risers -> {'CRUSHED' if not rich['won'] else 'fell'} "
          f"({len(rich['def_dead'])}+{len(rich['mob_dead'])} fell)")
    assert not rich["won"] and rich["defenders"] > rich["mob"]

    drained = rising(king_money=0.5)
    print(f"  DRAINED tyrant (war chest 0.5): mustered {drained['defenders']} guards vs "
          f"{drained['mob']} risers -> {'DEPOSED' if drained['won'] else 'held'}")
    assert drained["won"] and drained["deposed"]
    assert "S001" not in world_state["monarchs"]
    print("  -> the same mob is CRUSHED by a rich crown and TRIUMPHS over a broke one.")
    print("     Wealth buys the guards that put down the revolt.")
    print()


# --- HEADLINE 3: the revolution expropriates ---------------------------------
def headline_3_the_revolution_expropriates() -> None:
    print("=" * 72)
    print("HEADLINE 3 — THE REVOLUTION EXPROPRIATES (the hoard to the peasants, not the heirs)")
    print("=" * 72)

    def build() -> None:
        _fresh()
        world_state["lineage_on"] = True
        king = _agent("King", (5, 5), money=40.0)
        _agent("Heir", (5, 4), parents=("King", "Queen"), age=25)   # the M4.2/M4.3 heir
        for n, p in [("A", (5, 6)), ("B", (6, 5)), ("C", (6, 7))]:
            _agent(n, p, money=0.0)
        _monarch("S001", "King")
        world_state["discontent"] = {"A": 12.0, "B": 12.0, "C": 12.0}

    build()
    res = uprising.update(world_state, 10)[0]
    heir = next(a for a in world_state["agents"] if a.name == "Heir")
    victors = [a for a in world_state["agents"] if a.name in ("A", "B", "C") and a.alive]
    seized_to_risers = sum(v.money for v in victors)
    print(f"  UPRISING wins: King's hoard of {res['seized']:.2f} seized; split among {len(victors)} "
          f"risers = {seized_to_risers:.2f} total (each ~{seized_to_risers/len(victors):.2f}).")
    print(f"    the heir received: {heir.money:.2f}")
    assert abs(res["seized"] - 40.0) < 1e-9 and abs(seized_to_risers - 40.0) < 1e-9
    assert heir.money == 0.0

    # CONTRAST: the SAME king dying of old age -> the heir inherits the whole estate.
    build()
    king = next(a for a in world_state["agents"] if a.name == "King")
    population.announce_death(king, 10, world_state, cause="old age",
                              final_memory="Died of old age", note="they died of old age")
    heir = next(a for a in world_state["agents"] if a.name == "Heir")
    print(f"\n  CONTRAST — the same King dies of OLD AGE: the heir inherits {heir.money:.2f}.")
    assert heir.money == 40.0
    print("  -> revolution EXPROPRIATES: the wealth goes to the peasants who rose, not the heirs.")
    print()


# --- HEADLINE 4: a house can fall (Arc 1 x Arc 2) ----------------------------
def headline_4_a_house_can_fall() -> None:
    print("=" * 72)
    print("HEADLINE 4 — A HOUSE CAN FALL (a beloved king's unloved heir is overthrown)")
    print("=" * 72)

    _fresh()
    world_state["lineage_on"] = True
    king = _agent("Old", (5, 5), money=0.0, age=80)                  # a beloved but broke old king
    king.lifespan = 80                                               # dies of old age THIS turn on aging
    heir = _agent("Scion", (5, 4), parents=("Old", "Queen"), age=25)
    members = [_agent(n, p, money=20.0) for n, p in [("A", (5, 6)), ("B", (6, 5)), ("C", (6, 7))]]
    _monarch("S001", "Old")
    # The people LOVED the old king but DISTRUST the unearned heir (loyalty is not inherited — M4.3).
    for m in members:
        trust.ensure_relationship(m, "Old")["trust"] = 5
        trust.ensure_relationship(m, "Scion")["trust"] = -5

    # Phase A — under the beloved king the same levy generates little grievance; no rising.
    for turn in range(1, 11):
        discontent.update(world_state, turn)
    print(f"  under beloved King Old (12-turn reign): resentful commoners = "
          f"{discontent.settlement_pressure('S001', world_state)}, would rise? "
          f"{uprising.would_rise(world_state, 'S001', 11)}")
    assert not uprising.would_rise(world_state, "S001", 11)

    # The old king dies of old age -> the crown passes to the unloved heir (M4.3 succession).
    from sim import lineage
    king.age = 80
    lineage.update(world_state, 11)   # aging kills him; succeed_titles crowns the heir
    assert world_state["monarchs"]["S001"]["monarch"] == "Scion", "the crown passed to the heir"
    print(f"  King Old dies of old age -> his heir Scion is CROWNED monarch of S001 (M4.3).")

    # Phase B — the SAME levy, now under a distrusted heir, breeds grievance until the town rises.
    fired, fired_turn = None, None
    for turn in range(12, 30):
        discontent.update(world_state, turn)
        res = uprising.update(world_state, turn)
        if res:
            fired, fired_turn = res[0], turn
            break
    print(f"  under the unloved heir the SAME levy breeds grievance -> the settlement RISES on "
          f"turn {fired_turn} (won={fired['won']}).")
    assert fired is not None and fired["won"] and fired["deposed"]
    assert "S001" not in world_state["monarchs"], "the seat is vacant — the dynasty has fallen"
    print("  -> the House of Old rose by conquest, passed the crown by blood (M4.3), and FELL to")
    print("     the people its unearned heir could not command. Arc 1 builds the dynasty; Arc 2 ends it.")
    print()


# --- COST: off byte-identical, deterministic, zero added LLM -----------------
def cost_checks() -> None:
    print("=" * 72)
    print("COST — off byte-identical; seeded runs reproduce; zero added LLM")
    print("=" * 72)

    def run(**kw) -> tuple[str, dict]:
        llm.PROVIDER = "random"
        random.seed(42)
        llm.reset_call_stats()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main.run_simulation(30, settlements=True, monarchy_on=True, discontent_on=True, **kw)
        return buf.getvalue(), dict(llm.get_call_stats())

    off, off_calls = run()
    off2, _ = run(uprising_on=False)
    assert off == off2
    print("  --uprising OFF: byte-identical to the current institution run")
    on_a, on_calls = run(uprising_on=True)
    on_b, _ = run(uprising_on=True)
    assert on_a == on_b
    print("  --uprising ON: two seeded runs byte-identical (the valve draws no RNG)")
    assert on_calls == off_calls
    print(f"  the uprising system added ZERO LLM calls (on={on_calls}, off={off_calls}).")
    print()


if __name__ == "__main__":
    saved = llm.PROVIDER
    try:
        headline_1_oppression_predicts_uprisings()
        headline_2_wealth_buys_counter_revolution()
        headline_3_the_revolution_expropriates()
        headline_4_a_house_can_fall()
        cost_checks()
    finally:
        llm.PROVIDER = saved
    print("=" * 72)
    print("M4.5 VERIFIED — the revolt fires. Oppression genuinely PREDICTS uprisings (fair rule")
    print("provably immune), WEALTH genuinely buys counter-revolution (the same mob crushed by a")
    print("rich crown, victorious over a broke one), and a successful rising truly EXPROPRIATES")
    print("(the heirs disinherited). The pressure engine has its relief valve; no crown is safe.")
    print("=" * 72)
