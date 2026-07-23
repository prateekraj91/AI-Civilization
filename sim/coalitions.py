"""
coalitions.py
=============

COALITIONS & THE BALANCE OF POWER — the many band against the one (V2 milestone M4.15, CLOSES Arc 5:
Diplomacy & the Interstate System). On top of M4.14 (trade), M4.13 (treaties), Arc 4 (eras), Arc 3
(culture), Arc 2 (revolt), Arc 1 (dynasties) and all of Phases 0-3.

The historical step M4.15 makes — the interstate system becomes SELF-BALANCING
------------------------------------------------------------------------------
Kingdoms had three verbs: war (M3.6), treaty (M4.13), trade (M4.14). M4.15 adds the fourth — COALITION.
When one power grows dominant, the weaker powers band together against it out of FEAR — even across old
hostilities and cultural lines (the enemy of my enemy). This anti-hegemon dynamic prevents any single
empire from swallowing the world: a hegemon that beats every kingdom INDIVIDUALLY can still be torn down
by the coalition of all it threatens. When the threat passes the coalition DISSOLVES, old grievances
resurface, and the field clears for the next would-be conqueror — so history churns rather than settling.
This is the capstone of the interstate system and CLOSES Arc 5.

SCOPE — M4.15 is dominance DETECTION + fear-driven coalition FORMATION + coalition ACTION (joint
defence/attack vs the hegemon) + DISSOLUTION + the balance-of-power MEASUREMENT. It reuses M4.13's
mutual-defence and the M3.6 host/battle machinery. Zero LLM.

How it works (emergent; zero LLM; deterministic state math)
-----------------------------------------------------------
1. DOMINANCE. A sovereign power's strength is the number of settlements it CONTROLS (its realm plus its
   subject-kings' realms, recursively). It is a HEGEMON when its share of all settlements clears
   HEGEMON_SHARE AND it towers HEGEMON_MARGIN× over the next-largest power — dominance derived from state.
2. FEAR-DRIVEN COALITION. When a hegemon exists, EVERY other sovereign power joins a coalition against it
   — membership is driven by FEAR of the hegemon and OVERRIDES ordinary stance: mutually hostile or
   culturally foreign kingdoms STILL coalesce against the common threat. Recorded in world_state.
3. ACTION. DEFENCE — if the hegemon attacks a member, the others muster to the defence (M4.13 mutual
   defence, extended to the coalition), so it cannot pick members off one by one. JOINT ATTACK — the
   coalition pools every member's loyal host and, when that out-hosts the hegemon, assaults it; on
   victory the hegemon's empire is BROKEN (its subject-kings freed and its vassals seceded), dropping it
   below the dominance threshold. The many bring down the one none could face alone.
4. DISSOLUTION. Once the hegemon falls below the threshold the fear evaporates: the coalition dissolves,
   the untouched M4.13 stances resurface, and former allies drift apart (and may then fight). Temporary
   marriages of convenience; knocking down one hegemon clears the field for the next.

Cost & determinism
------------------
ZERO LLM and ZERO RNG — dominance and membership are deterministic reads; the joint attack reuses the
RNG-free `monarchy.resolve_battle`. A run with the system OFF never calls `update` (no "coalitions" key),
the war loop never suppresses an internal fight or adds a coalition defender, so it is byte-identical to
v1. Imports world; lazily imports empire/kingdoms/monarchy/diplomacy (the systems it reads) — no cycle.
"""

from __future__ import annotations

from typing import Any

from sim import world

# --- Tunable constants -------------------------------------------------------
HEGEMON_SHARE = 0.4      # a power controlling at least this share of all settlements may be a hegemon...
HEGEMON_MARGIN = 1.5     # ...but ONLY if it towers this many times over the next-largest power (a true peer-less dominant)


def _coal(state: dict[str, Any]) -> dict[str, Any]:
    return state.setdefault("coalitions", {"target": None, "members": set()})


def _sovereign_kings(state: dict[str, Any]) -> list[str]:
    from sim import empire
    return [k for k in sorted(state.get("kingdoms", {}))
            if empire.is_sovereign(state, k) and empire._find(state, k) is not None]


def controlled_settlements(state: dict[str, Any], king: str) -> set:
    """All settlements `king` controls: his own realm's PLUS every subject-king's (recursively). Pure read."""
    sids = set(state.get("kingdoms", {}).get(king, {}).get("settlements", set()))
    emp = state.get("empires", {}).get(king)
    if emp is not None:
        for sk in emp["subject_kings"]:
            sids |= controlled_settlements(state, sk)
    return sids


def _power(state: dict[str, Any], king: str) -> int:
    return len(controlled_settlements(state, king))


# --- 1. Dominance detection --------------------------------------------------
def dominance(state: dict[str, Any]) -> "tuple[str | None, float]":
    """The world's HEGEMON and its settlement share, or (None, top_share) if no power dominates.

    The strongest sovereign is a hegemon iff it controls >= HEGEMON_SHARE of all settlements AND at least
    HEGEMON_MARGIN× the next-largest power (peer-less dominance). Pure read; deterministic (name tiebreak)."""
    sovereigns = _sovereign_kings(state)
    total = len(state.get("settlements", {}))
    if not sovereigns or total == 0:
        return None, 0.0
    powers = {k: _power(state, k) for k in sovereigns}
    top = max(sovereigns, key=lambda k: (powers[k], k))
    share = powers[top] / total
    second = max((powers[k] for k in sovereigns if k != top), default=0)
    is_heg = share >= HEGEMON_SHARE and powers[top] >= HEGEMON_MARGIN * max(second, 1)
    return (top if is_heg else None), share


def is_hegemon(state: dict[str, Any], king: str) -> bool:
    heg, _ = dominance(state)
    return heg == king


def dominance_share(state: dict[str, Any]) -> float:
    """The strongest power's share of all settlements — the balance-of-power metric (1.0 = one power owns all)."""
    return dominance(state)[1]


# --- 2. Fear-driven coalition membership -------------------------------------
def coalition_members(state: dict[str, Any], hegemon: str) -> set:
    """The powers that FEAR `hegemon`: every OTHER sovereign king. Fear overrides stance — mutually
    hostile or culturally foreign kingdoms are included alike (the enemy of my enemy). Pure read."""
    return {k for k in _sovereign_kings(state) if k != hegemon}


def active_target(state: dict[str, Any]) -> "str | None":
    return _coal(state)["target"]


def members(state: dict[str, Any]) -> set:
    return set(_coal(state)["members"])


def allied_against_hegemon(state: dict[str, Any], k1: str, k2: str) -> bool:
    """True if k1 and k2 are BOTH current coalition members — so the war loop suspends their feud while
    they face the common hegemon (fear trumps grievance). Pure read."""
    m = _coal(state)["members"]
    return _coal(state)["target"] is not None and k1 in m and k2 in m


def coalition_backers(state: dict[str, Any], defender: str, attacker: str) -> list[str]:
    """The coalition members who muster to DEFEND `defender` when the hegemon `attacker` assaults it —
    the others in the coalition (so the hegemon cannot pick members off one by one). Pure read."""
    coal = _coal(state)
    if coal["target"] == attacker and defender in coal["members"]:
        return sorted(m for m in coal["members"] if m != defender)
    return []


# --- 3. Coalition action: the pooled joint attack ----------------------------
def pooled_host_size(state: dict[str, Any], member_names: set) -> int:
    from sim import empire
    return sum(empire.imperial_host_size(state, empire._find(state, m))
               for m in member_names if empire._find(state, m) is not None)


def _fragment(state: dict[str, Any], hegemon: str, turn: int) -> list[str]:
    """BREAK the defeated hegemon: free every subject-king (empire dissolves) and secede every vassal
    (realm shrinks to its home), dropping it below the dominance threshold. Reuses the M3.6/M4.5 machinery."""
    from sim import kingdoms
    events: list[str] = []
    emp = state.get("empires", {}).get(hegemon)
    if emp is not None:
        for sk in sorted(emp["subject_kings"]):
            events.append(f"turn {turn}: subject-king {sk} was freed from {hegemon}'s empire by the coalition")
        state["empires"].pop(hegemon)
    rec = state.get("kingdoms", {}).get(hegemon)
    if rec is not None:
        for sid in sorted(rec["vassals"]):
            kingdoms.secede_settlement(state, sid, turn, f"the coalition broke {hegemon}")
    state["events"].extend(events)
    return events


def _joint_attack(state: dict[str, Any], hegemon: str, member_names: set, turn: int) -> bool:
    """The coalition pools every member's loyal host and assaults the hegemon (shared M3.6 battle math).
    On victory the hegemon's empire is BROKEN (`_fragment`). Returns whether the coalition won."""
    from sim import empire
    from sim import monarchy
    heg = empire._find(state, hegemon)
    if heg is None:
        return False
    taken = {hegemon}
    host: list[Any] = []
    for m in sorted(member_names):
        a = empire._find(state, m)
        if a is None:
            continue
        contingent = empire.imperial_host(state, a, taken | {m})
        taken |= {f.name for f in contingent}
        host.extend(contingent)
    heg_host = empire.imperial_host(state, heg, taken)
    n_c, n_h = len(host), len(heg_host)
    won, cd, hd, _ = monarchy.resolve_battle(
        state, host, heg_host, turn, "the coalition host", f"hegemon {hegemon}'s host")
    if won:
        state["events"].append(
            f"turn {turn}: the COALITION of {len(member_names)} powers ({n_c} pooled host) DEFEATED "
            f"hegemon {hegemon} ({n_h} host; {len(cd)}+{len(hd)} fell) — his empire is BROKEN")
        _fragment(state, hegemon, turn)
    else:
        state["events"].append(
            f"turn {turn}: the coalition's assault on hegemon {hegemon} was REPELLED "
            f"({n_c} pooled vs {n_h}; {len(cd)}+{len(hd)} fell)")
    return won


# --- The per-turn engine -----------------------------------------------------
def update(state: dict[str, Any], turn: int) -> list[str]:
    """Advance the coalition system one turn (M4.15): detect a hegemon, form/maintain the fear-driven
    coalition against it, launch a WINNABLE pooled attack, and DISSOLVE when the threat passes. ZERO LLM,
    ZERO RNG. Runs BEFORE empire.update so the coalition exists for this turn's mutual defence. Caller
    gates on `coalitions_on`, so an off run never calls this (no "coalitions" key) and stays byte-identical.
    Returns events."""
    from sim import empire
    coal = _coal(state)
    events: list[str] = []
    heg, share = dominance(state)

    if heg is None:
        if coal["target"] is not None:
            events.append(f"turn {turn}: the coalition against {coal['target']} DISSOLVED — the threat has "
                          f"passed; old rivalries resurface")
            coal["target"], coal["members"] = None, set()
        state.setdefault("events", []).extend(events)
        return events

    mem = coalition_members(state, heg)
    if coal["target"] != heg:
        # Note when fear is overriding grievance: any two members who are mutually hostile still coalesce.
        events.append(f"turn {turn}: hegemon {heg} controls {share:.0%} of the world — the {len(mem)} "
                      f"weaker powers COALESCE against it (fear over grievance)")
    coal["target"], coal["members"] = heg, mem

    # JOINT ATTACK: pool the hosts and strike if the coalition out-fields the hegemon (winnable only).
    heg_agent = empire._find(state, heg)
    if mem and heg_agent is not None and pooled_host_size(state, mem) > empire.imperial_host_size(state, heg_agent):
        _joint_attack(state, heg, mem, turn)
        # Re-detect: a broken hegemon dissolves the coalition (the fear that formed it is gone).
        heg2, _ = dominance(state)
        if heg2 is None:
            events.append(f"turn {turn}: with {heg} broken, the coalition DISSOLVED — the balance restored")
            coal["target"], coal["members"] = None, set()

    state.setdefault("events", []).extend(events)
    return events
