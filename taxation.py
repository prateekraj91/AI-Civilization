"""
taxation.py
===========

TAXATION & REDISTRIBUTION — legitimacy acts on wealth (V2 milestone M3.3, Phase 3:
Institutions). On top of M3.1 (wage labor) and M3.2 (legitimate leadership), and all of
Phase 0 + Phase 1 + Phase 2.

The historical step M3.3 makes — the COLLISION of the two Phase 3 engines
------------------------------------------------------------------------
M3.1 built the CLASS ENGINE: wage labor, where the rich employer captures the worker's
surplus and inequality COMPOUNDS (a disequilibrating spiral). M3.2 built the LEGITIMACY
ENGINE: a leader legitimated by TRUST, a political power DECOUPLED from wealth. M3.3 is their
collision — the first force that BENDS the M3.1 spiral: a legitimate leader TAXES its wealthy
followers and REDISTRIBUTES to its poor ones. Political legitimacy acquires the power to act on
economic inequality. Scope is held to taxation + redistribution + a legitimacy backlash that
self-limits over-taxation; NO law/legislation, NO revolt (later Phase 3), NO fiat money (taxes
move the existing food-backed money/stockpile, conserving total wealth — redistribution, not
minting).

Taxation requires LEGITIMACY (this is what ties the two engines together)
------------------------------------------------------------------------
ONLY a settlement with a legitimate M3.2 leader can tax. No leader -> no taxation (the power to
tax is downstream of legitimacy, not of wealth or of a flag). The leader taxes its FOLLOWERS
(the M3.2 trust cluster) — never non-followers, never itself. A follower whose wealth exceeds
RICH_THRESHOLD is taxed TAX_RATE of the EXCESS above that threshold (progressive: only wealth
over the line is taxed), into a common pool; the pool is redistributed to followers below
POOR_THRESHOLD, weighted by NEED (the poorest receive the most). Taxation only happens when
there is BOTH a tax base (a rich follower) AND a redistribution target (a poor follower) — it
exists to redistribute, so with no one to lift it does not levy. Total wealth is conserved
(pool taken == pool given): this redistributes claims, it does not create money.

Redistribution BENDS inequality (the headline)
----------------------------------------------
The pool flows from the richest followers to the poorest, compressing the wealth spread. Over
time the within-settlement Gini in a led+taxed settlement falls measurably below an identical
led-but-untaxed one — taxation dampens the M3.1 compounding spiral. That bending is the point.

Legitimacy BACKLASH self-limits taxation (emergent consent of the governed)
---------------------------------------------------------------------------
Taxation COSTS the leader trust — but only OVERREACH does. Up to CONSENT_RATE the governed
accept taxation as legitimate (consent), so a moderate levy draws NO resentment; above it each
taxed follower's trust in the leader falls by round((rate - CONSENT_RATE) * RESENT_SCALE) — the
harder the overreach, the steeper the fall. Redistribution meanwhile EARNS the leader GRATITUDE
(+trust) from every poor follower it lifts. So under MODERATE taxation the leader is sustained
(no resentment; the poor's gratitude even GROWS its support); under OVER-taxation resentment
erodes the wealthy followers' trust below M3.2's KEEP_TRUST faster than gratitude can offset in
the cluster's count, the following falls below MIN_FOLLOWERS, and next turn M3.2's contingency
STRIPS the leadership — and with it the power to tax. Taxation is self-limiting through the
trust system: tyranny is punished by withdrawal of consent, emergent, not scripted.

This is the FIRST Phase 3 institution that WRITES trust (M3.2 only read it). The writes go
through the existing `trust.adjust_trust` — the same logged, grudge-aware path conversations
use — so the backlash is a first-class part of the trust network the leadership engine reads.

Cost & determinism
------------------
ZERO LLM calls and ZERO RNG — deterministic state-math over sorted settlement ids and sorted
follower names. Conserves total wealth. A run with the institution OFF never calls `update`
(and `tax_rate` is inert), so it is byte-identical to v1. Imports world + storage + trust
(one-directional); no new Agent field — taxation rides world_state + the M3.2 leader record.
"""

from __future__ import annotations

from typing import Any

import storage
import trust
import world

# --- Tunable constants (documented) ----------------------------------------
# DEFAULT_TAX_RATE: the fraction of a wealthy follower's wealth-ABOVE-threshold taken each turn
# when a run enables taxation without naming a rate. 0.25 sits comfortably under CONSENT_RATE, so
# the default is a MODERATE, consented levy — sustainable, visibly redistributive, not tyranny.
DEFAULT_TAX_RATE = 0.25

# RICH_THRESHOLD: wealth (money + stockpile) above which a FOLLOWER is taxed, and only on the
# EXCESS above it (progressive). Set well above the M3.1 worker/employer gates (WORKER_MAX_WEALTH
# = EMPLOYER_MIN_CAPITAL = 5) so the taxed are the genuinely rich — the employers who compounded
# wealth in M3.1 — not the merely-getting-by. A follower at or below it pays nothing.
RICH_THRESHOLD = 10.0

# POOR_THRESHOLD: wealth below which a FOLLOWER receives redistribution. Tied to the M3.1
# subsistence band (~SUBSISTENCE/WORKER scale) so the recipients are the have-nots the wage
# treadmill keeps poor. The [POOR_THRESHOLD, RICH_THRESHOLD) middle is neither taxed nor paid.
POOR_THRESHOLD = 5.0

# CONSENT_RATE: the share of surplus the governed ACCEPT as a legitimate levy — the consent band.
# Taxation at or below this draws NO resentment (a fair tax is tolerated); only the overreach
# beyond it costs the leader trust. This is what makes moderate taxation SUSTAINED and tyranny
# self-limiting, and it emerges through the trust system rather than a hard "max tax" rule.
CONSENT_RATE = 0.35

# RESENT_SCALE: converts overreach (rate - CONSENT_RATE, in [0, 1)) into the trust each
# OVER-taxed follower withdraws from the leader per turn. Sized so a steep overreach (e.g. rate
# 0.9 -> ~0.55 over the band) costs ~2 trust/turn — enough to drive a FORM_TRUST follower below
# KEEP_TRUST within a couple of turns, firing M3.2's contingency. A mild overreach costs ~1.
RESENT_SCALE = 4.0

# GRATITUDE: the trust a poor follower GAINS in the leader each turn it receives redistribution.
# +1 (one friendly-interaction's worth) — small, but it means fair redistribution actively GROWS
# a leader's support among the poor, the counterweight that lets moderate taxation be sustained.
GRATITUDE = 1


def _wealth(a: Any) -> float:
    """An agent's liquid wealth = money + stored food (both food-claims) — the M3.1 class metric."""
    return a.money + a.stockpile


def _take(a: Any, amount: float) -> float:
    """Remove up to `amount` of wealth from `a` (money first, then stored food). Returns taken.

    Mirrors economy._settle's payer side: money and stockpile are both food-claims, so tax is
    drawn from whichever the follower holds. A rich follower's due is always < its wealth (we tax
    only a fraction of the excess), so the full amount is taken; the return guards conservation.
    """
    from_money = min(amount, a.money)
    a.money -= from_money
    from_food = min(amount - from_money, a.stockpile)
    a.stockpile -= from_food
    return from_money + from_food


def _give(a: Any, amount: float) -> None:
    """Credit `amount` of wealth to `a`: stored food up to the M2.2 cap, overflow to money.

    The poor are credited in FOOD first (it relieves hunger / is directly usable), with anything
    past the storage cap held as food-backed money — the same surplus-past-cap rule M3.1 uses.
    """
    room = max(0.0, storage.STORAGE_CAP - a.stockpile)
    to_stock = min(amount, room)
    a.stockpile += to_stock
    a.money += amount - to_stock


def update(state: dict[str, Any], turn: int) -> list[str]:
    """Advance the taxation institution one turn (ZERO LLM, ZERO RNG, conserves wealth, M3.3).

    For each settlement WITH a legitimate M3.2 leader (sorted ids -> deterministic): tax follower
    wealth above RICH_THRESHOLD at the run's rate into a pool, redistribute the pool to followers
    below POOR_THRESHOLD weighted by need, then write the legitimacy BACKLASH through the trust
    system — over-taxed followers withdraw trust, lifted poor grant gratitude. The backlash feeds
    M3.2's next-turn re-evaluation, so over-taxation self-limits (the leader loses legitimacy and
    the power to tax). Returns the events logged. Caller gates on the `taxation_on` flag, so an
    off run never calls this and stays byte-identical to v1.
    """
    leaders = state.get("leaders", {})
    rate = state.get("tax_rate", DEFAULT_TAX_RATE)
    living = {a.name: a for a in state["agents"] if a.alive}
    events: list[str] = []

    for sid in sorted(leaders):
        rec = leaders[sid]
        leader = living.get(rec["leader"])
        if leader is None:
            continue  # the leader is gone; M3.2 will clear the stale record — nothing to tax with
        followers = [living[n] for n in sorted(rec["followers"]) if n in living]
        rich = [f for f in followers if _wealth(f) > RICH_THRESHOLD]
        poor = [f for f in followers if _wealth(f) < POOR_THRESHOLD]
        if not rich or not poor:
            continue  # taxation needs BOTH a tax base and a redistribution target, else it idles

        # 1. TAX: take TAX_RATE of each rich follower's wealth ABOVE the threshold into the pool.
        pool = 0.0
        for f in rich:
            pool += _take(f, rate * (_wealth(f) - RICH_THRESHOLD))

        # 2. REDISTRIBUTE: hand the pool to the poor weighted by NEED (poorest first). The last
        # recipient takes the exact remainder so the pool is conserved to the cent (taken == given).
        needs = [POOR_THRESHOLD - _wealth(p) for p in poor]
        total_need = sum(needs)
        given = 0.0
        for i, p in enumerate(poor):
            share = (pool - given) if i == len(poor) - 1 else pool * needs[i] / total_need
            _give(p, share)
            given += share

        # 3. BACKLASH (writes trust through the normal logged path): overreach beyond the consent
        # band costs the taxed their trust in the leader; redistribution earns the poor's gratitude.
        resent = -round(max(0.0, rate - CONSENT_RATE) * RESENT_SCALE)
        if resent < 0:
            for f in rich:
                trust.adjust_trust(f, leader.name, resent, "taxed by leader", turn, state)
        for p in poor:
            trust.adjust_trust(p, leader.name, GRATITUDE, "redistribution from leader", turn, state)

        tag = "resented overreach" if resent < 0 else "consented"
        events.append(
            f"turn {turn}: {leader.name} taxed {len(rich)} wealthy followers {pool:.1f} and "
            f"redistributed to {len(poor)} poor in {sid} (rate {rate:.0%}, {tag})")
        world.record_memory(leader, f"Taxed {len(rich)} rich, redistributed {pool:.1f} to {len(poor)} poor")

    state["events"].extend(events)
    return events
