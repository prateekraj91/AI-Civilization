"""
labor.py
========

WAGE LABOR — the first INSTITUTION (V2 milestone M3.1, opens Phase 3: Institutions). On top
of all of Phase 0 + Phase 1 + Phase 2 (M2.1 settlement, M2.2 storage/wealth, M2.3 trade/money).

The historical step M3.1 makes
------------------------------
Phases 1-2 built EMERGENT MATERIAL MECHANICS (knowledge, food, wealth, prices) that mostly
EQUILIBRATE — left alone they settle toward a steady state. Phase 3 builds INSTITUTIONS:
persistent structures that coordinate and constrain many agents. The governing rule for all
of Phase 3: an institution must EMERGE from existing asymmetries, never be installed/scripted.

Wage labor is the first, and the first DISEQUILIBRATING mechanic: a rich agent EMPLOYS a poor
agent to produce for it, paying a wage. Because the employer captures the difference between
what the worker produces and what it is paid, the rich-poor gap COMPOUNDS instead of settling.
That compounding is the intended result — it manufactures the class tension that later Phase 3
milestones (law, conflict, governance) will respond to. This module builds ONLY: employment
relationships, emergent wages, and the measurable compounding. NO governance/leaders/law/tax/
revolt (later Phase 3); NO fiat/minted money (still deferred) — wages are paid in the existing
food-backed money/stockpile.

Roles EMERGE from wealth + skill state (never assigned)
-------------------------------------------------------
- EMPLOYER: a settled agent with CAPITAL (money + stored food to pay wages, >= EMPLOYER_MIN_
  CAPITAL) AND a production opportunity it cannot fully exploit alone — it KNOWS a producer
  skill (farming/hunting), the "means of production", but its own labor yields only so much, so
  hiring hands multiplies its output. Your rich M2.3 producers fall into this with no flag set.
- WORKER: a settled POOR agent lacking independent means — it has NO producer skill (no means
  of its own) and little wealth (< WORKER_MAX_WEALTH), so a wage beats its alternative
  (foraging/starving). Your poor M2.2 have-nots fall into this.
An agent with no capital can never be an employer (capacity 0); a self-sufficient agent (owns a
producer skill or is wealthy) never takes a wage. The roles are pure reads of existing state.

The relationship is PERSISTENT (this is what makes it an INSTITUTION)
--------------------------------------------------------------------
A link {employer, worker, wage, since} lives in world_state["employments"] and SURVIVES across
turns until it ends. Lifecycle, each turn (`update`):
  * SETTLE every active link: the worker produces LABOR_OUTPUT for the employer (the employer's
    means + the worker's labor -> output to the EMPLOYER's stockpile), and the employer PAYS the
    worker the wage (money first, then food). Net: employer +(output - wage), worker +(wage).
  * It ENDS when: a party dies; the employer can't pay / loses its means (FIRES, insolvent);
    or the worker becomes self-sufficient (gains a producer skill or climbs past
    SELF_SUFFICIENT_WEALTH -> QUITS, upward mobility).
  * New links FORM by matching available employers (spare capacity, nearby) to available poor
    workers at the emergent wage.

The wage EMERGES from the labor market (NOT a fixed wage)
--------------------------------------------------------
wage = reservation + leverage * (LABOR_OUTPUT - reservation), bounded to [subsistence, output):
  * reservation = SUBSISTENCE_WAGE — the worker's survival floor (its alternative is starving),
    so the worker always at least survives (this is what separates exploitation from slavery:
    the worker still gains NET, just little).
  * leverage = market_tightness * (1 - desperation), in [0, 1):
      - market_tightness = openings / (openings + workers): a WORKER'S MARKET (scarce labor,
        many openings) -> ~1 -> wage near output (worker captures the value); an EMPLOYER'S
        MARKET (abundant desperate labor, few openings) -> ~0 -> wage near subsistence.
      - desperation = hunger / HUNGER_MAX: a starving worker accepts less, pushing the wage down.
  * Both an ABUNDANT-labor glut AND worker DESPERATION compress the band toward SUBSISTENCE —
    emergent EXPLOITATION (employer captures most of the value). Subsistence is POSSIBLE under
    those conditions but never automatic: the SAME work pays DIFFERENTLY as supply/desperation
    change. wage < LABOR_OUTPUT always (employer profits); wage >= reservation always (worker
    survives) — so the relationship is voluntary and mutually beneficial, just lopsided.

Cost & determinism
------------------
ZERO LLM calls and ZERO RNG — the labor market is deterministic state-math over a stable
iteration (world_state["agents"] order; employers matched to the NEAREST available workers with
sorted tie-breaks). A run with the institution OFF never calls `update`, so it is byte-identical
to v1. Imports world + storage + economy (one-directional), keeping the world layer dependency-
free; employment links live only in world_state (no new Agent field), so an off run is unchanged.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from sim import economy
from sim import storage
from sim import world

# --- Tunable constants (documented) ----------------------------------------
# LABOR_OUTPUT: the food-claim value one worker produces for its employer each turn (the
# employer's means of production + the worker's labor). It is the CEILING on the wage — pay must
# stay below it for the employer to profit — and the whole value the two sides bargain to split.
LABOR_OUTPUT = 2.0

# SUBSISTENCE_WAGE: the wage floor — the worker's survival alternative. Tuned to ~1 food-claim
# per turn, which is roughly the per-turn hunger drain (HUNGER_PER_TURN), so a worker paid this
# barely stays alive (its wage is consumed surviving and it accumulates ~nothing — the treadmill
# that keeps the working class poor while employers compound). A wage can fall TO this under a
# labor glut + desperation, but never below it (below subsistence the worker would rather forage).
SUBSISTENCE_WAGE = 1.0

# EMPLOYER_MIN_CAPITAL: liquid wealth (money + stockpile) an agent needs before it can employ at
# all — without capital to advance wages you cannot be a boss. The capital gate is what makes the
# employer role fall out of M2.2/M2.3 wealth rather than being assigned.
EMPLOYER_MIN_CAPITAL = 5.0

# CAPITAL_PER_WORKER: how much capital backs each hire. Capacity = capital // this (capped below),
# so a richer employer hires MORE hands — and since it profits from each, it can hire still more
# next turn. That reinvestment is the engine of compounding inequality.
CAPITAL_PER_WORKER = 5.0

# MAX_WORKERS_PER_EMPLOYER: a ceiling on one employer's workforce, so a single tycoon can't hire
# literally everyone in one turn (which would be degenerate). Several employers still form.
MAX_WORKERS_PER_EMPLOYER = 8

# WORKER_MAX_WEALTH: above this liquid wealth an agent is no longer poor enough to sell its labor
# (it has independent means), so it isn't a worker. The poverty gate for the worker role.
WORKER_MAX_WEALTH = 5.0

# SELF_SUFFICIENT_WEALTH: a worker whose wealth climbs past this QUITS — it has accumulated enough
# to stop selling its labor (upward mobility out of the working class). Higher than WORKER_MAX_
# WEALTH so there's hysteresis (a worker isn't hired and fired on the same threshold).
SELF_SUFFICIENT_WEALTH = 15.0

# HIRE_RADIUS: an employer can only hire workers within this Chebyshev distance — labor is local
# (you employ hands near your settlement, not across the map). Equal to a small village reach.
HIRE_RADIUS = 3

# COST_OF_LIVING: the subsistence an employed worker CONSUMES each turn. This is the crux of the
# treadmill (and of why inequality compounds rather than equalises): an employed worker works for
# its employer instead of provisioning itself, so the wage IS its food — each turn it spends
# COST_OF_LIVING of its earnings on subsistence (consumed: deducted from wealth, and it relieves
# that much hunger so the wage keeps it ALIVE). Net worker wealth change = wage - COST_OF_LIVING,
# which under a subsistence wage is ~0: the worker survives but never gets ahead, while the
# employer banks the surplus and reinvests it. Set equal to SUBSISTENCE_WAGE, so a subsistence
# wage exactly covers living and leaves nothing — emergent class stasis at the bottom. (A worker
# in a tight labor market earns ABOVE subsistence and so DOES accumulate — upward mobility when
# labor is scarce.) Being fed by the wage is precisely why employed beats unemployed (which
# starves) even at subsistence — the worker still gains NET (survival), distinguishing this from
# slavery; it just gains little.
COST_OF_LIVING = SUBSISTENCE_WAGE


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _wealth(a: Any) -> float:
    """An agent's liquid wealth = money + stored food (both food-claims). The class metric."""
    return a.money + a.stockpile


def is_employer(a: Any) -> bool:
    """Whether `a` qualifies as an EMPLOYER — emerges from capital + a producer skill + settlement.

    Capital to advance wages, a producer skill (the means of production it can't fully work
    alone), and a settlement to host the work. No flag is set anywhere; this is a pure read.
    """
    if a.settlement is None:
        return False
    if not any(s in a.knowledge for s in economy.PRODUCER_SKILLS):
        return False  # no means of production -> nothing to employ labor on
    return _wealth(a) >= EMPLOYER_MIN_CAPITAL


def is_worker(a: Any) -> bool:
    """Whether `a` qualifies as a WORKER — emerges from poverty + lacking any means.

    Poor (little wealth) and UNskilled (no producer skill of its own), so selling labor for a
    wage beats its alternative. Pure read of existing state — never assigned.
    """
    if a.settlement is None:
        return False
    if world.is_dependent_child(a):
        return False  # M4.1: a dependent child does not sell labor — work waits for maturity
    if any(s in a.knowledge for s in economy.PRODUCER_SKILLS):
        return False  # has its own means -> self-sufficient, not a wage worker
    return _wealth(a) < WORKER_MAX_WEALTH


def _self_sufficient(a: Any) -> bool:
    """A worker that has gained a producer skill OR grown wealthy enough to stop selling labor."""
    return (any(s in a.knowledge for s in economy.PRODUCER_SKILLS)
            or _wealth(a) >= SELF_SUFFICIENT_WEALTH)


def capacity(a: Any) -> int:
    """How many workers `a` can back with its capital (0 if it has none -> never an employer)."""
    return min(MAX_WORKERS_PER_EMPLOYER, int(_wealth(a) // CAPITAL_PER_WORKER))


def market_tightness(openings: int, workers: int) -> float:
    """Labor-market tightness in [0, 1): openings / (openings + workers).

    ->1 when openings far exceed workers (a WORKER'S market, scarce labor -> high wages); ->0
    when workers far exceed openings (an EMPLOYER'S market, abundant labor -> subsistence wages).
    """
    total = openings + workers
    if total == 0:
        return 0.5
    return openings / total


def offered_wage(worker: Any, tightness: float) -> float:
    """The emergent wage offered to `worker` given current market `tightness` (pure; no RNG).

    wage = subsistence + leverage * (LABOR_OUTPUT - subsistence), where the worker's leverage is
    market tightness DISCOUNTED by its own desperation (hunger). Scarce labor + a secure worker
    -> high wage; abundant labor + a desperate worker -> near subsistence. The SAME work thus
    pays differently as conditions change — the emergence/exploitation proof. Bounded into
    [SUBSISTENCE_WAGE, LABOR_OUTPUT) since tightness < 1 whenever a worker is being hired.
    """
    desperation = min(1.0, max(0.0, worker.hunger / world.HUNGER_MAX))
    leverage = tightness * (1.0 - desperation)
    return SUBSISTENCE_WAGE + leverage * (LABOR_OUTPUT - SUBSISTENCE_WAGE)


def _credit_output(employer: Any, amount: float, economy_on: bool) -> None:
    """Credit `amount` of produced food-claim to the employer: stockpile first, overflow->money.

    The worker's product accrues to the EMPLOYER. Stored as food up to the M2.2 cap; anything past
    it becomes money under the same surplus-past-cap->money rule M2.3 uses (food-backed). With the
    economy off, overflow is simply dropped (uncapturable surplus) — keeping money food-grounded.
    """
    room = max(0.0, storage.STORAGE_CAP - employer.stockpile)
    to_stock = min(amount, room)
    employer.stockpile += to_stock
    overflow = amount - to_stock
    if overflow > 0 and economy_on:
        employer.money += overflow


def _consume_subsistence(worker: Any) -> None:
    """The worker spends COST_OF_LIVING on subsistence this turn (the treadmill).

    Deducted from its just-earned wealth (money first, then stored food) and converted into
    hunger relief — the wage IS the employed worker's food, which is why it survives at all. The
    consumed wealth is gone (eaten), so net worker wealth gain = wage - COST_OF_LIVING: ~0 at a
    subsistence wage. This is what keeps the working class poor while employers compound.
    """
    cost = min(COST_OF_LIVING, worker.money + worker.stockpile)
    from_money = min(cost, worker.money)
    worker.money -= from_money
    worker.stockpile -= (cost - from_money)
    worker.hunger = max(0, worker.hunger - int(round(cost)))


def update(state: dict[str, Any], turn: int) -> list[str]:
    """Advance the wage-labor institution one turn (ZERO LLM, ZERO RNG, M3.1).

    Three deterministic stages: (1) SETTLE active links — produce for the employer and pay the
    worker, or end the link if it has broken (death / insolvency / the worker became self-
    sufficient); (2) the surviving links persist into world_state; (3) FORM new links by matching
    available employers to the nearest available poor workers at the market wage. Returns the
    event strings logged. Caller gates invocation on the `labor` flag, so an off run never calls
    this and stays byte-identical to v1.
    """
    economy_on = state.get("economy_on", False)
    living = {a.name: a for a in state["agents"] if a.alive}
    events: list[str] = []

    # 1+2. Settle each active link, keeping only those that remain valid (persistence).
    survivors: list[dict[str, Any]] = []
    for link in sorted(state["employments"], key=lambda l: (l["employer"], l["worker"])):
        emp = living.get(link["employer"])
        wkr = living.get(link["worker"])
        if emp is None or wkr is None:
            continue  # a party died -> the relationship ends
        if not is_employer(emp):
            events.append(f"turn {turn}: {emp.name} lost its means and let {wkr.name} go")
            continue  # employer lost capital/skill -> can no longer employ
        if _self_sufficient(wkr):
            events.append(f"turn {turn}: {wkr.name} quit {emp.name} (now self-sufficient)")
            continue  # worker climbed out of the working class -> quits
        if _wealth(emp) < link["wage"]:
            events.append(f"turn {turn}: {emp.name} laid off {wkr.name} (insolvent)")
            continue  # cannot make payroll -> fires
        # Produce for the employer (its means + the worker's labor), pay the wage, and have the
        # worker consume its subsistence (the treadmill): net employer +(output - wage), net
        # worker +(wage - cost-of-living) ~ 0 at a subsistence wage, but it stays ALIVE (fed).
        _credit_output(emp, LABOR_OUTPUT, economy_on)
        economy._settle(emp, wkr, link["wage"])  # employer pays worker (money then food)
        _consume_subsistence(wkr)
        world.record_memory(wkr, f"Worked for {emp.name} (wage {link['wage']:.2f})")
        world.record_memory(emp, f"Employed {wkr.name} (paid {link['wage']:.2f})")
        survivors.append(link)
    state["employments"] = survivors

    # 3. Form new links: match available employers to nearby available poor workers.
    counts = Counter(l["employer"] for l in survivors)
    employed = {l["worker"] for l in survivors}
    employers = [a for a in state["agents"]
                 if a.alive and is_employer(a) and counts[a.name] < capacity(a)]
    workers = [a for a in state["agents"]
               if a.alive and is_worker(a) and a.name not in employed]
    if not employers or not workers:
        return events
    openings = sum(capacity(a) - counts[a.name] for a in employers)
    tightness = market_tightness(openings, len(workers))

    used: set[str] = set()
    for emp in employers:  # world_state order is stable
        room = capacity(emp) - counts[emp.name]
        if room <= 0:
            continue
        # Nearest available workers within reach (sorted: distance then name -> deterministic).
        cands = sorted(
            (w for w in workers
             if w.name not in used and _chebyshev(emp.position, w.position) <= HIRE_RADIUS),
            key=lambda w: (_chebyshev(emp.position, w.position), w.name))
        for w in cands[:room]:
            wage = offered_wage(w, tightness)
            if wage >= LABOR_OUTPUT:
                continue  # no profit for the employer -> no offer
            if _wealth(emp) < wage:
                continue  # can't advance the first wage
            if wage < SUBSISTENCE_WAGE:
                continue  # below the worker's survival floor -> it would refuse (never happens)
            state["employments"].append(
                {"employer": emp.name, "worker": w.name, "wage": wage, "since": turn})
            used.add(w.name)
            counts[emp.name] += 1
            world.record_memory(w, f"Hired by {emp.name} at wage {wage:.2f}")
            events.append(f"turn {turn}: {emp.name} hired {w.name} at wage {wage:.2f}")
    return events
