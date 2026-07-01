"""
renderer/pygame_renderer.py
===========================

A Pygame VISUAL renderer for AI Civilization — SLICE 1 (the smallest watchable thing).

WHAT IT DOES (this slice)
-------------------------
Opens a window and draws ONE FRAME from a `world_state` snapshot:

  * the grid as a muted flat terrain background (faint cell lines when cells are big),
  * food as small green dots at their world positions,
  * each LIVING agent as a coloured circle at its (x, y), where
        - COLOUR encodes the agent's dominant PERSONALITY trait, and
        - RADIUS encodes WEALTH (money + stockpile) — richer agents are bigger
          (clamped to a sane min/max so nobody vanishes or fills the screen).
  Dead agents are not drawn.

A tiny HUD line shows turn / living / food. Later slices added settlements (2), the
event-feed panel (3), iconography (4), detailed terrain (5), detailed towns & castles
(6), and — slice 8 — WAR & MOTION: a realm-colour territory layer, battles replayed
as short read-only cinematics, and smooth inter-turn agent movement.

ARCHITECTURE RULE (same boundary as the text renderer, obeyed strictly)
-----------------------------------------------------------------------
This module ONLY READS `world_state`. It mutates NOTHING and never imports decision
logic (strategy / trust / conversation / alliance / personality / agents / llm /
god_mode / economy / population / monarchy / kingdoms / empire). It does not even
import `world`: every datum it draws is read straight off the snapshot dict
(`state["size"]`, `state["food"]`, `state["agents"]`, agent `.position` /
`.personality` / `.money` / `.stockpile` / `.alive`). The personality→colour map is
inlined here (the keyword sets that personality.py uses) precisely so we never import
the personality decision module. A boundary test asserts this file imports no
decision logic, mirroring the text-renderer test.

HOW IT PLUGS IN (reuses the real sim — no turn logic here)
----------------------------------------------------------
It exposes the SAME tiny interface `run_simulation()` already drives for the text
dashboard — `.live()` (a context manager owning the window), `.update(state)` (draw
one frame, called by the sim AFTER each fully-resolved turn), and `.sink` (where the
plain per-turn text is swallowed). The simulation advances itself through its own
loop and merely calls `.update()` to be drawn; the renderer NEVER advances the world.
Pacing and pause/quit are handled inside `.update()` so the window stays responsive.

Pygame is an OPTIONAL dependency: it is imported lazily (only when this module is
loaded, which only happens when `--pygame` is requested), so the core sim never
depends on it. If it is missing the launcher prints a clear `pip install pygame`.
"""

from __future__ import annotations

import contextlib
import math
import os
import textwrap
import time
from typing import Any

try:
    import pygame
except ImportError as exc:  # pragma: no cover - exercised only without pygame installed
    raise ImportError(
        "Pygame is required for the visual renderer. Install it with:  pip install pygame"
    ) from exc


# --- Palette (RGB) ---------------------------------------------------------
# Muted flat terrain and a green food dot; agent colours below are keyed to the
# dominant PERSONALITY trait so a viewer can read temperament at a glance.
_TERRAIN = (38, 42, 36)        # muted dark olive — a calm flat ground
_GRID_LINE = (48, 53, 46)      # barely-there cell lines (drawn only when cells are large)
_FOOD = (96, 200, 96)          # food dots
_HUD_BG = (24, 26, 22)
_HUD_FG = (210, 214, 200)
_OUTLINE = (16, 18, 14)        # thin dark ring around each agent for contrast

# Dominant trait -> colour. Distinct hues, none of them food-green.
_TRAIT_COLOR = {
    "curiosity": (240, 198, 70),      # amber — the explorer
    "caution": (92, 146, 230),        # blue — the careful/territorial
    "friendliness": (236, 112, 178),  # pink — the social
    "independence": (220, 84, 84),    # red — the competitive/aloof
}
_DEFAULT_COLOR = (180, 180, 180)      # grey — unrecognised personality

# Slice 2: SETTLEMENTS. A teal tint, chosen distinct from every personality colour
# (amber/blue/pink/red) and from food-green, so a settlement reads as its own kind of
# thing. The region fill is TRANSLUCENT (low alpha) so it stays background context
# under the agents; a slightly stronger edge ring gives it a soft boundary, and a small
# centre marker makes the centre identifiable.
_SETTLEMENT_FILL = (80, 180, 175)     # teal — settlement region tint (drawn translucent)
_SETTLEMENT_EDGE = (120, 220, 215)    # brighter teal — soft boundary ring + centre marker
_SETTLEMENT_LABEL = (150, 210, 205)   # muted teal — subtle label
_SETTLEMENT_FILL_ALPHA = 46           # how opaque the region tint is (0..255; low = subtle)
_SETTLEMENT_EDGE_ALPHA = 130          # the boundary ring, a touch stronger than the fill
_SETTLEMENT_MIN_CELLS = 1.6           # smallest region radius, in grid cells
_SETTLEMENT_LABEL_MIN_CELL = 10       # only draw labels when cells are big enough to stay legible

# Slice 4: ICONOGRAPHY. Procedural glyphs (no asset files) so the MAP is self-explanatory:
# agents are little FIGURES, rulers wear CROWNS / a leader STAR, talkers get a SPEECH BUBBLE,
# food is a WHEAT stalk, and settlements show HOUSE buildings. A handful of primitives each.
_CROWN = (245, 205, 70)       # gold crown — monarch (king) / emperor
_STAR = (242, 228, 138)       # pale-gold star — a trust-leader
_BUBBLE = (236, 239, 231)     # near-white speech bubble (someone is talking this turn)
_BUBBLE_DOT = (92, 98, 88)    # the "..." inside the bubble
_HOUSE_WALL = (156, 124, 94)  # warm clay walls of a settlement house
_HOUSE_ROOF = (120, 84, 68)   # darker roof
_FIGURE_MIN_R = 3             # below this radius a cell is too small for a figure -> plain dot
_FOOD_GLYPH_MIN_CELL = 9      # below this cell size food stays a simple dot (wheat won't read)
_HOUSE_MIN_CELL = 8           # below this cell size houses stay implied by the region only
_MAX_HOUSES = 6               # cap the house glyphs drawn per settlement (keeps a village tidy)

# Slice 5: DETAILED TERRAIN & ATMOSPHERE. A living landscape, drawn FIRST under everything.
# The grass texture + scattered features (trees/rocks/pond) + vignette + frame are baked ONCE
# into a cached background surface (built per window/grid size, blitted each frame) — so it
# costs nothing per turn and never desyncs. ALL procedural variation comes from a pure
# coordinate HASH (`terrain_noise`), never the global `random` module, so seeded sim runs stay
# byte-identical. Kept muted so the slice-1..4 foreground (agents/food/buildings) stays legible.
_GRASS_BASE = (42, 58, 40)    # base grassland — a touch greener than the old flat fill
_GRASS_VAR = 9                # fine per-tile tonal swing (+/-), the cheap value-noise texture
_GRASS_PATCH = 12             # low-frequency swing -> broad patches of darker/lighter grass
_GRASS_SPECK_HI = (60, 86, 54)   # occasional lighter stipple speck
_GRASS_SPECK_LO = (32, 48, 32)   # occasional darker stipple speck
_TREE_TRUNK = (74, 52, 34)
_TREE_CANOPY = (44, 76, 44)
_TREE_CANOPY_HI = (56, 96, 54)
_ROCK = (92, 96, 90)
_ROCK_HI = (122, 126, 118)
_WATER = (46, 84, 116)
_WATER_HI = (74, 118, 150)
_FARMLAND = (110, 82, 50)     # tilled-dirt tint near a settlement (drawn translucent each frame)
_FARMLAND_FURROW = (84, 60, 38)
_FARMLAND_ALPHA = 52
_VIGNETTE_MAX = 92            # edge darkening strength (alpha) for atmospheric depth
_FRAME_OUTER = (22, 26, 20)   # dark outer frame around the map zone
_FRAME_INNER = (78, 90, 66)   # a thin lighter inner line, for a framed-map look
# Feature density thresholds on terrain_noise (sparse, so the map stays readable).
_TREE_THRESHOLD = 0.93        # ~7% of cells get a tree
_ROCK_THRESHOLD = 0.965       # ~3.5% of cells get a rock
_STIPPLE_STEP = 4             # stipple sampling stride in pixels (coarser = cheaper)

# Slice 6: DETAILED SETTLEMENTS & CASTLES. Villages become clusters of detailed houses that
# GROW with membership, with civic structure (well/plaza, granary, paths, a wall once big), and
# a ruler's seat becomes a HALL (leader) or a CASTLE (monarch/king/emperor). Every layout is
# derived from `terrain_noise` (the pure coordinate hash) — NEVER the sim RNG — and CACHED per
# settlement, rebuilt only when its membership/ruler/cell changes, so per-frame cost stays low.
_WALL_TONES = ((156, 128, 96), (172, 150, 118), (140, 116, 90), (178, 160, 128), (150, 132, 104))
_ROOF_TONES = ((122, 84, 66), (150, 98, 70), (98, 74, 56), (112, 100, 68), (132, 90, 72))
_DOOR = (66, 46, 32)
_WINDOW_LIT = (226, 198, 120)     # a warm lit window
_WINDOW_DARK = (74, 78, 84)       # an unlit window
_CHIMNEY = (96, 84, 76)
_PATH = (124, 106, 80)            # dirt road between buildings
_PLAZA = (140, 120, 88)           # packed-earth market square at the centre
_WELL_STONE = (148, 148, 146)
_WELL_WATER = (58, 96, 128)
_GRANARY_WALL = (180, 148, 102)
_FENCE = (112, 92, 66)            # the perimeter palisade of a large settlement
_CROP = (104, 162, 72)            # green crop rows in the farmland
_CASTLE_STONE = (150, 152, 158)
_CASTLE_STONE_DK = (114, 116, 124)
_GATE = (52, 44, 38)
_DEFAULT_RULER = (170, 150, 205)  # fallback royal tone if the ruler agent can't be found
_TOWN_MIN_CELL = 8                # below this cell size, fall back to the slice-4 simple houses
_MIN_TOWN_BUILDINGS = 2
_MAX_TOWN_BUILDINGS = 16          # cap so a metropolis stays tidy/cheap
_GRANARY_MIN_MEMBERS = 5          # a granary appears once a settlement is established
_FENCE_MIN_MEMBERS = 8            # a palisade ring appears once a settlement is large

# Slice 8: WAR & MOTION. The sim resolves a battle INSTANTLY inside one turn (wage_war /
# conquer_neighbour / attempt_conquest are pure state maths) — so the renderer REPLAYS each battle
# as a short five-beat CINEMATIC (muster -> march -> clash -> casualties -> aftermath) rebuilt from
# data the sim already emitted: the event log names who fought, who fell BY NAME, who won and what
# changed hands, and a renderer-local SNAPSHOT of last turn supplies where everyone stood before.
# Presentation (dust, jitter, flashes) is invented; outcomes never are. While a scene plays the sim
# is simply not stepped (the renderer already owns pacing), and all visual randomness comes from
# `terrain_noise` keyed on frame+position — the seeded sim RNG is never touched. A REALM layer
# colours owned settlements by their TOP ruler (emperor > king > lone monarch) so conquest is
# readable as territory: on a won battle the loser's tint LERPS to the victor's colour instead of
# snapping. Smooth motion: every living agent LERPs from last turn's cell to this turn's across
# the inter-turn delay (with a subtle walk bob) — nobody teleports.
_REALM_PALETTE = (                # distinct banner hues, hashed per ruler name; none food/teal
    (196, 84, 70),                # rust red
    (86, 132, 214),               # royal blue
    (206, 168, 66),               # gold
    (146, 96, 198),               # purple
    (88, 172, 108),               # green
    (214, 116, 162),              # rose
    (150, 190, 90),               # olive-lime
    (222, 138, 72),               # ember orange
)
_REALM_FILL_ALPHA = 62            # realm territory tint (a touch stronger than plain teal)
_REALM_EDGE_ALPHA = 150
_SPEAR = (208, 206, 198)          # a soldier's spear shaft
_DUST = (150, 138, 112)           # dust puffs behind a marching host
_FLASH = (255, 250, 235)          # the white clash-flash bursts
_FALLEN = (148, 148, 142)         # a fallen soldier's gray marker
_BANNER_BG = (14, 13, 10)         # the aftermath banner band
_BANNER_FG = (242, 232, 204)
_BATTLE_CHIP = (236, 138, 74)     # the small HUD "BATTLE" indicator (matches the feed's war orange)
_MAX_SOLDIER_GLYPHS = 12          # cap the figures PER SIDE (the true counts live in the banner/log)
# The five beats, in seconds (~4s total; ANY key skips a scene to its end-state).
_CIN_MUSTER, _CIN_MARCH, _CIN_CLASH, _CIN_FALL, _CIN_AFTER = 0.5, 1.0, 1.0, 0.6, 1.0
_CIN_TOTAL = _CIN_MUSTER + _CIN_MARCH + _CIN_CLASH + _CIN_FALL + _CIN_AFTER
_WALK_BOB = 0.09                  # walk-bob amplitude as a fraction of a cell

# Trait -> keywords (a verbatim inline of personality.TRAIT_KEYWORDS, so we classify
# the agent's free-text personality WITHOUT importing the personality decision module).
# Tie-break order matches personality.DOMINANCE_ORDER (curiosity first).
_TRAIT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("curiosity", ("curious", "adventurous", "explorer", "exploring", "inquisitive", "bold")),
    ("caution", ("cautious", "careful", "timid", "territorial", "defensive", "wary")),
    ("friendliness", ("friendly", "outgoing", "social", "sociable", "kind", "cooperative")),
    ("independence", ("independent", "competitive", "aloof", "solitary", "loner", "lone")),
)

# --- Layout / sizing tunables ----------------------------------------------
_TARGET_PX = 760               # aim the grid area near this many pixels on its long edge
_MIN_CELL = 5                  # floor so a large world still fits a window
_MAX_CELL = 44                 # ceiling so a tiny world isn't comically huge
_HUD_H = 26                    # status strip height (under the MAP zone)
_WEALTH_CEIL = 60.0            # wealth mapped to the largest radius (sqrt ramp below it)

# Slice 3: LEGIBILITY. The window widens into a MAP zone (left, the square grid) and a
# PANEL zone (right sidebar) holding a state summary above a scrolling EVENT FEED, so the
# viewer can READ what is happening while watching the map.
_PANEL_W = 320                 # sidebar width in pixels
_PANEL_BG = (20, 22, 18)       # panel background — a shade off the terrain so the zones read apart
_PANEL_DIV = (58, 64, 54)      # thin divider lines inside the panel
_PANEL_PAD = 10                # inner margin
_PANEL_TITLE = (150, 210, 205) # section headers (teal, echoing settlements)
_STAT_LABEL = (138, 144, 130)  # muted stat captions
_STAT_VALUE = (226, 230, 216)  # bright stat values
# Light per-type colour coding for the feed (kept readable, never garish).
_FEED_GOD = (236, 205, 90)     # [GOD] interventions — yellow
_FEED_WAR = (236, 138, 74)     # conquest / war / breakaway — strong orange
_FEED_DEATH = (226, 100, 100)  # a death — reddish
_FEED_TOWN = _SETTLEMENT_EDGE  # settlements forming/growing — teal (matches the map region)
_FEED_SOCIAL = (122, 200, 132) # alliances / trust / trade / tribute — green
_FEED_DEFAULT = (170, 176, 164)# routine chatter — muted grey
_FEED_SCAN = 80                # how many tail events to consider before wrapping to fit


def dominant_trait(personality: str | None) -> str:
    """Classify a free-text personality into its dominant trait (pure string read).

    Counts keyword hits per trait and returns the strongest; ties (and an empty /
    unrecognised description) fall back to the first trait in keyword order
    (curiosity), exactly as personality.py's dominance tie-break does. No import of
    the personality module — this is a self-contained read of the `.personality` text.
    """
    text = (personality or "").lower()
    best_trait, best_hits = "curiosity", 0
    for trait, words in _TRAIT_KEYWORDS:
        hits = sum(text.count(w) for w in words)
        if hits > best_hits:
            best_trait, best_hits = trait, hits
    return best_trait


def agent_color(personality: str | None) -> tuple[int, int, int]:
    """The RGB colour for an agent's dominant personality trait (pure read)."""
    return _TRAIT_COLOR.get(dominant_trait(personality), _DEFAULT_COLOR)


def _wealth(agent: Any) -> float:
    """Liquid wealth = money + stockpile, each defaulting to 0 if the field is absent."""
    return float(getattr(agent, "money", 0.0) or 0.0) + float(getattr(agent, "stockpile", 0.0) or 0.0)


def agent_radius(wealth: float, cell: int) -> int:
    """Map wealth -> a pixel radius, clamped so every agent is visible but bounded.

    A sqrt ramp (so differences read at the low/common end of wealth) between a
    cell-relative minimum and maximum. The richest agents are visibly larger without
    overflowing their cell into illegibility.
    """
    r_min = max(2.0, cell * 0.26)
    r_max = max(r_min + 1.0, cell * 0.58)
    frac = max(0.0, min(1.0, math.sqrt(max(0.0, wealth) / _WEALTH_CEIL)))
    return int(round(r_min + frac * (r_max - r_min)))


def _cell_size(size: int) -> int:
    """Pixels per grid cell so a `size`x`size` world fits near _TARGET_PX (clamped)."""
    if size <= 0:
        return _MAX_CELL
    return max(_MIN_CELL, min(_MAX_CELL, _TARGET_PX // size))


def settlement_radius_cells(center: tuple[int, int],
                            member_positions: list[tuple[int, int]]) -> float:
    """The settlement region radius, in grid cells, from its CURRENT members' spread.

    A pure geometry read: the region reaches the farthest living member from the centre
    (plus a one-cell margin), floored at _SETTLEMENT_MIN_CELLS so a tiny/just-founded
    settlement still reads as a place. With no locatable members it falls back to the
    floor. Because it is recomputed every frame from the members handed in, a settlement
    that GROWS simply draws a larger region next frame — no animation/state needed.
    """
    cx, cy = center
    spread = 0.0
    for px, py in member_positions:
        spread = max(spread, math.hypot(px - cx, py - cy))
    return max(_SETTLEMENT_MIN_CELLS, spread + 1.0)


def event_color(line: str) -> tuple[int, int, int]:
    """Light per-type colour for one verbatim event-log line (pure string read).

    Classifies the plain-English event string the engine already wrote — never reads or
    changes events[] content. Order matters: a war/conquest line is coloured before a
    death (so 'KING X DEFEATED Y ... fell' reads as war), while an individual battle/
    starvation death ('Z died (fell in battle)') has no war keyword and reads as a death.
    """
    low = line.lower()
    if "[god]" in low or "god-script" in low:
        return _FEED_GOD
    if any(k in low for k in ("betrayed", "conquered", "seized", "overthrew", "overthrown",
                              "subjugated", "defeated", "repelled", "broke away", "war",
                              "empire", "crown")):
        return _FEED_WAR
    if any(k in low for k in ("died", "fell in battle", "starved")):
        return _FEED_DEATH
    if any(k in low for k in ("settlement", "settled", "joined", "founded")):
        return _FEED_TOWN
    if any(k in low for k in ("alliance", "trust", "ally", "friend", "trade", "tribute",
                              "wage", "employ", "levied", "redistribut")):
        return _FEED_SOCIAL
    return _FEED_DEFAULT


def wrap_events(events: list[str], cols: int, max_rows: int) -> list[tuple[str, tuple[int, int, int]]]:
    """Turn the tail of events[] into (text, colour) rows for the feed (pure, no pygame).

    Each event line is colour-classified (`event_color`) then wrapped to `cols` characters
    (monospace, so a character budget maps cleanly to pixel width); every wrapped sub-line
    inherits its event's colour. The last `max_rows` rows are returned, so the NEWEST line
    sits at the bottom of the feed. Handles empty/short logs (returns [] / what fits). The
    colour-keyed wrapping is split out here so it can be unit-tested without a display.
    """
    rows: list[tuple[str, tuple[int, int, int]]] = []
    for line in events[-_FEED_SCAN:]:
        color = event_color(line)
        for sub in (textwrap.wrap(line, width=max(1, cols)) or [""]):
            rows.append((sub, color))
    return rows[-max_rows:] if max_rows > 0 else []


def talkers_this_turn(events: list[str], turn: int) -> set[str]:
    """Names that SPOKE this turn, derived read-only from the event tail (Slice 4).

    The engine logs a talk as `turn {turn}: {speaker} talked to {target}: "..."`. A turn's
    lines sit contiguously at the tail of events[], so we scan backwards over the current
    turn's block (stopping at the first earlier-turn line) and collect each speaker. No new
    state is added — this is a pure read of the existing log, used to pop a speech bubble
    over talkers for that frame only.
    """
    prefix = f"turn {turn}: "
    marker = " talked to "
    out: set[str] = set()
    for line in reversed(events):
        if not line.startswith(prefix):
            break  # reached an earlier turn — the current turn's lines are the contiguous tail
        rest = line[len(prefix):]
        idx = rest.find(marker)
        if idx > 0:
            out.add(rest[:idx])
    return out


def agent_role(name: str, state: dict[str, Any]) -> str | None:
    """The highest ruling role `name` holds, or None — a pure read of the institution dicts.

    Precedence EMPEROR > MONARCH (king) > LEADER, so an agent who is several at once wears
    only its top insignia. Each lookup degrades gracefully when its dict is absent (no
    empires -> nobody is an emperor, etc.), so the map simply shows fewer markers.
    """
    if name in state.get("empires", {}):                      # empires are keyed by emperor name
        return "emperor"
    if any(r.get("monarch") == name for r in state.get("monarchs", {}).values()):
        return "monarch"
    if any(r.get("leader") == name for r in state.get("leaders", {}).values()):
        return "leader"
    return None


def terrain_noise(x: int, y: int, salt: int = 0) -> float:
    """A deterministic pseudo-random value in [0, 1) from integer coords (Slice 5).

    A pure integer hash (xorshift-style mixing) of (x, y, salt) — NO global `random`, no
    import, no state. Same input -> same output forever, so the procedural landscape is
    stable across frames and seeds and can NEVER touch the simulation's seeded RNG stream.
    `salt` lets independent feature layers (grass tone / trees / rocks / stipple) decorrelate.
    """
    h = (x * 374761393 + y * 668265263 + salt * 2147483647) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    h ^= (h >> 16)
    return (h & 0xFFFFFF) / 0x1000000


def _shade(color: tuple[int, int, int], delta: int) -> tuple[int, int, int]:
    """Lighten (delta>0) or darken (delta<0) an RGB colour, clamped to [0, 255]. Pure."""
    return tuple(max(0, min(255, c + delta)) for c in color)


def _pick(seq: tuple, t: float):
    """Pick `seq[i]` from a fraction t in [0,1) (deterministic; for hash-driven variety)."""
    return seq[min(len(seq) - 1, int(t * len(seq)))]


def build_town_plan(center: tuple[int, int], n_members: int, central_kind: str | None,
                    ruler_color: tuple[int, int, int], emperor: bool, cell: int) -> dict[str, Any]:
    """Lay out a settlement's buildings + civic structure deterministically (pure, no pygame).

    GROWTH: the number of detailed houses scales with `n_members` (a hamlet shows a couple of
    huts, a town a dense ring of many). Each building's offset, size, roof style and wall/roof
    tone come from `terrain_noise` keyed to the settlement centre — so the village looks organic
    yet is STABLE frame to frame (no flicker) and never touches the sim RNG. STRUCTURE: a central
    well/plaza, a granary once established, a palisade ring once large, and a central `central_kind`
    seat ('castle' for a monarch, 'hall' for a leader, None for a plain village). Returns a plan of
    pixel offsets relative to the centre; the renderer caches it and rebuilds only on change. The
    layout maths is split out here so growth/castle behaviour is unit-testable without a display.
    """
    cxh = (int(center[0]) * 7 + 3) & 0xFFFF
    cyh = (int(center[1]) * 13 + 5) & 0xFFFF

    def nz(i: int, s: int) -> float:                 # deterministic [0,1) per (building i, channel s)
        return terrain_noise(cxh, cyh, i * 131 + s * 17 + 1)

    n_buildings = min(_MAX_TOWN_BUILDINGS, max(_MIN_TOWN_BUILDINGS, int(n_members)))
    base_r = cell * 1.15
    ring_gap = cell * 1.2
    buildings: list[dict[str, Any]] = []
    placed, ring = 0, 0
    while placed < n_buildings:
        per_ring = 5 + ring * 3                       # outer rings hold more houses
        for k in range(per_ring):
            if placed >= n_buildings:
                break
            ang = (k / per_ring) * 2 * math.pi + (nz(placed, 1) - 0.5) * 0.7
            rad = base_r + ring * ring_gap + (nz(placed, 2) - 0.5) * cell * 0.35
            w = max(5, int(cell * (0.65 + nz(placed, 3) * 0.5)))
            h = max(5, int(w * (0.7 + nz(placed, 4) * 0.45)))
            buildings.append({
                "dx": int(rad * math.cos(ang)), "dy": int(rad * math.sin(ang)),
                "w": w, "h": h,
                "wall": _pick(_WALL_TONES, nz(placed, 5)),
                "roof": _pick(_ROOF_TONES, nz(placed, 6)),
                "hip": nz(placed, 7) > 0.5,           # hip vs gable roof
                "lit": nz(placed, 8) > 0.45,          # lit windows
            })
            placed += 1
        ring += 1
    cluster_r = int(base_r + ring * ring_gap)

    granary = None
    if n_members >= _GRANARY_MIN_MEMBERS:
        gang = nz(99, 1) * 2 * math.pi
        granary = {"dx": int(cluster_r * 0.72 * math.cos(gang)),
                   "dy": int(cluster_r * 0.72 * math.sin(gang)), "scale": cell}
    fence_r = cluster_r + int(cell * 0.5) if n_members >= _FENCE_MIN_MEMBERS else None
    off = int(cell * 0.9) if central_kind else 0     # nudge the well aside when a seat owns the centre
    return {
        "buildings": buildings,
        "central": {"kind": central_kind, "color": ruler_color, "emperor": emperor, "scale": cell},
        "granary": granary,
        "fence_r": fence_r,
        "well": {"dx": off, "dy": off, "scale": max(3, int(cell * 0.7))},
        "cluster_r": cluster_r,
        "plaza_r": max(cell, int(base_r * 0.95)),
        "path_w": max(1, cell // 6),
    }


# --- Slice 8: pure helpers (snapshot / motion / realms / battle detection) --------------
def ease(t: float) -> float:
    """Smoothstep easing on [0,1] — gentle in/out for motion and colour lerps (pure)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def lerp_color(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Linear RGB interpolation, clamped to [0,1] — endpoints are exact (pure)."""
    t = max(0.0, min(1.0, t))
    return tuple(int(round(c1[i] + (c2[i] - c1[i]) * t)) for i in range(3))


def realm_color(name: Any) -> tuple[int, int, int]:
    """A stable banner colour for ruler `name`: a pure string hash into _REALM_PALETTE.

    Same name -> same colour forever (across frames, seeds and runs); no RNG. Distinct realms
    usually land on distinct hues (8 slots), and the staged rivals (Aldric/Borin) provably do.
    """
    h = 5381
    for ch in str(name):
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return _REALM_PALETTE[h % len(_REALM_PALETTE)]


def settlement_realm(sid: str, state: dict[str, Any]) -> str | None:
    """The TOP ruler whose realm settlement `sid` belongs to, or None — a pure institution read.

    A settlement inside a kingdom is coloured by its king — unless that king is (or serves) an
    EMPEROR, in which case the whole realm wears the empire's colour (so a subjugated kingdom
    visibly changes hands). A monarch-held town outside any kingdom wears its monarch's colour.
    Degrades gracefully when any institution dict is absent (an unowned town keeps the teal).
    """
    kingdoms = state.get("kingdoms", {})
    empires = state.get("empires", {})
    for king in sorted(kingdoms):
        if sid in (kingdoms[king].get("settlements") or ()):
            for emp in sorted(empires):
                if king == emp or king in (empires[emp].get("subject_kings") or {}):
                    return emp
            return king
    mon = state.get("monarchs", {}).get(sid)
    return mon.get("monarch") if mon else None


def take_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """A renderer-local snapshot of the just-drawn turn, for diffing when the NEXT turn arrives.

    Pure READ, fresh containers (mutating the snapshot can never touch world_state): living
    agents' positions (motion lerp + where the soon-to-be-dead stood), each settlement's TOP
    realm ruler (territory colour before a conquest), and each king's home seat (a war's anchor).
    """
    return {
        "turn": int(state.get("turn", 0)),
        "positions": {a.name: (a.position[0], a.position[1])
                      for a in state.get("agents", [])
                      if getattr(a, "alive", True) and getattr(a, "position", None)},
        "realms": {sid: settlement_realm(sid, state) for sid in state.get("settlements", {})},
        "homes": {k: (state["kingdoms"][k] or {}).get("home")
                  for k in state.get("kingdoms", {})},
    }


def turn_events(events: list[str], turn: int) -> list[str]:
    """THIS turn's event lines, oldest first — the same contiguous-tail read as talkers_this_turn."""
    prefix = f"turn {turn}: "
    out: list[str] = []
    for line in reversed(events):
        if not line.startswith(prefix):
            break
        out.append(line)
    out.reverse()
    return out


def _fell_counts(rest: str) -> tuple[int, int]:
    """Parse '; {x}+{y} fell' from a battle summary -> (attacker_dead, defender_dead) or (0, 0)."""
    idx = rest.rfind(" fell)")
    if idx < 0:
        return (0, 0)
    pair = rest[:idx].rpartition("; ")[2]
    a, plus, b = pair.partition("+")
    if plus and a.strip().isdigit() and b.strip().isdigit():
        return int(a), int(b)
    return (0, 0)


def _host_counts(rest: str) -> tuple[int, int]:
    """Parse '({n} ... vs {m} ...' from a battle summary -> (attacker_host, defender_host)."""
    i, j = rest.find("("), rest.find(";")
    seg = rest[i + 1:j] if 0 <= i < j else ""
    nums = [int(tok) for tok in seg.split() if tok.isdigit()]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)


def _agent_pos(name: str, state: dict[str, Any]) -> tuple[int, int] | None:
    """A living agent's current cell by name, or None (pure read)."""
    for a in state.get("agents", []):
        if getattr(a, "name", None) == name and getattr(a, "position", None):
            return (a.position[0], a.position[1])
    return None


def _center_of(sid: str, state: dict[str, Any]) -> tuple[int, int] | None:
    """A settlement's centre cell, or None (pure read)."""
    center = state.get("settlements", {}).get(sid, {}).get("center")
    return (int(center[0]), int(center[1])) if center else None


def _seat_of(name: str, prev: dict[str, Any], state: dict[str, Any]) -> tuple[int, int] | None:
    """A king's capital cell: his kingdom's home settlement, else where he stood/stands (pure)."""
    home = (state.get("kingdoms", {}).get(name, {}) or {}).get("home") \
        or (prev.get("homes") or {}).get(name)
    center = _center_of(home, state) if home else None
    if center is not None:
        return center
    return (prev.get("positions") or {}).get(name) or _agent_pos(name, state)


def _battle_summary(rest: str, prev: dict[str, Any],
                    state: dict[str, Any]) -> dict[str, Any] | None:
    """Classify ONE prefix-stripped event line as a battle summary -> a partial timeline, or None.

    Recognises exactly the summary strings the engine writes: an inter-kingdom war won/FAILED
    (empire.wage_war), a realm conquest CONQUERED/REPELLED (kingdoms.conquer_neighbour), and a
    settlement seizure '-> MONARCH of'/REPELLED assault (monarchy.attempt_conquest). Everything in
    the returned dict is read off the line + snapshots — attacker, defender, anchors, outcome,
    banner wording, and (for a won battle) the territory that changes hands with its OLD colour.
    """
    prev_realms = prev.get("realms") or {}

    def old_tint(sid: str) -> tuple[int, int, int]:
        top = prev_realms.get(sid)
        return realm_color(top) if top else _SETTLEMENT_FILL

    if rest.startswith("KING ") and " DEFEATED " in rest and " in war " in rest:
        a, _, tail = rest[len("KING "):].partition(" DEFEATED ")
        b = tail.split(" in war ")[0]
        sids = sorted(s for s, top in prev_realms.items() if top == b)
        return {"kind": "war", "attacker": a, "defender": b, "won": "SUBJUGATED" in rest,
                "att_pos": _seat_of(a, prev, state), "def_pos": _seat_of(b, prev, state),
                "banner": f"{a} DEFEATS {b} — {b} subjugated as vassal",
                "att_color": realm_color(a), "def_color": realm_color(b),
                "territory": [(s, old_tint(s)) for s in sids]}
    if rest.startswith("KING ") and "'s war on " in rest and " FAILED" in rest:
        a = rest[len("KING "):rest.index("'s war on ")]
        b = rest[rest.index("'s war on ") + len("'s war on "):rest.index(" FAILED")]
        return {"kind": "war", "attacker": a, "defender": b, "won": False,
                "att_pos": _seat_of(a, prev, state), "def_pos": _seat_of(b, prev, state),
                "banner": f"{a}'s war on {b} FAILS — the kingdom holds",
                "att_color": realm_color(a), "def_color": realm_color(b), "territory": []}
    if rest.startswith("KING ") and " CONQUERED " in rest and " into the realm" in rest:
        a = rest[len("KING "):rest.index(" CONQUERED ")]
        sid = rest[rest.index(" CONQUERED ") + len(" CONQUERED "):rest.index(" into the realm")]
        return {"kind": "realm", "attacker": a, "defender": sid, "won": True,
                "att_pos": _seat_of(a, prev, state), "def_pos": _center_of(sid, state),
                "banner": f"KING {a} CONQUERS {sid} into the realm",
                "att_color": realm_color(a), "def_color": old_tint(sid),
                "territory": [(sid, old_tint(sid))]}
    if rest.startswith("KING ") and "'s host was REPELLED at " in rest:
        cut = rest.index("'s host was REPELLED at ")
        a = rest[len("KING "):cut]
        sid = rest[cut + len("'s host was REPELLED at "):].split(" ")[0]
        return {"kind": "realm", "attacker": a, "defender": sid, "won": False,
                "att_pos": _seat_of(a, prev, state), "def_pos": _center_of(sid, state),
                "banner": f"KING {a} REPELLED at {sid}",
                "att_color": realm_color(a), "def_color": old_tint(sid), "territory": []}
    if " by force (" in rest and "-> MONARCH of " in rest:
        a = rest.split(" ", 1)[0]
        sid = rest[rest.index("-> MONARCH of ") + len("-> MONARCH of "):].strip()
        return {"kind": "conquest", "attacker": a, "defender": sid, "won": True,
                "att_pos": (prev.get("positions") or {}).get(a) or _agent_pos(a, state),
                "def_pos": _center_of(sid, state),
                "banner": f"{a} SEIZES {sid} — MONARCH by force",
                "att_color": realm_color(a), "def_color": old_tint(sid),
                "territory": [(sid, old_tint(sid))]}
    if "'s assault on " in rest and " was REPELLED " in rest:
        a = rest[:rest.index("'s assault on ")]
        sid = rest[rest.index("'s assault on ") + len("'s assault on "):rest.index(" was REPELLED")]
        return {"kind": "conquest", "attacker": a, "defender": sid, "won": False,
                "att_pos": (prev.get("positions") or {}).get(a) or _agent_pos(a, state),
                "def_pos": _center_of(sid, state),
                "banner": f"{a}'s assault on {sid} REPELLED",
                "att_color": realm_color(a), "def_color": old_tint(sid), "territory": []}
    return None


def battle_scenes(events_this_turn: list[str], prev_snapshot: dict[str, Any] | None,
                  state: dict[str, Any]) -> list[dict[str, Any]]:
    """Detect this turn's battles and build a cinematic TIMELINE for each (pure, unit-testable).

    Walks the turn's lines in order. Individual '{name} died (fell in battle)' lines accumulate;
    when a battle SUMMARY arrives, its '{x}+{y} fell' counts claim the x+y most recent fallen —
    the first x fell attacking, the next y defending (the engine kills attacker-side casualties
    first, then defender-side, THEN writes the summary, so the split is exact). Fallen positions
    come from the PREV snapshot (they are dead in current state). Nothing is invented: attacker,
    defender, host counts, the fallen's names, the outcome line and the territory that changed
    hands are all read from what the sim wrote. Multiple battles in a turn -> a queue, in order.
    """
    prev = prev_snapshot or {}
    scenes: list[dict[str, Any]] = []
    pending: list[str] = []                       # battle deaths not yet claimed by a summary
    for line in events_this_turn:
        rest = line.split(": ", 1)[1] if ": " in line else line
        if rest.endswith(" died (fell in battle)"):
            pending.append(rest[:-len(" died (fell in battle)")])
            continue
        scene = _battle_summary(rest, prev, state)
        if scene is None:
            continue
        x, y = _fell_counts(rest)
        claimed = pending[-(x + y):] if (x + y) else []
        del pending[len(pending) - len(claimed):]
        pos = prev.get("positions") or {}
        scene["att_dead"] = [(n, pos.get(n)) for n in claimed[:x]]
        scene["def_dead"] = [(n, pos.get(n)) for n in claimed[x:x + y]]
        scene["n_att"], scene["n_def"] = _host_counts(rest)
        if scene["def_pos"] is None:
            scene["def_pos"] = scene["att_pos"]
        if scene["att_pos"] is None:
            scene["att_pos"] = scene["def_pos"]
        if scene["att_pos"] is None:
            continue                              # nowhere on the map to stage it — skip honestly
        scenes.append(scene)
    return scenes


def battle_scene(events_this_turn: list[str], prev_snapshot: dict[str, Any] | None,
                 state: dict[str, Any]) -> dict[str, Any] | None:
    """The FIRST battle timeline this turn, or None on a peaceful turn (pure)."""
    scenes = battle_scenes(events_this_turn, prev_snapshot, state)
    return scenes[0] if scenes else None


class PygameRenderer:
    """Draws world_state to a Pygame window each turn (READ only); paces + handles input.

    Implements the same interface `run_simulation()` drives for the text dashboard:
    `.live()` owns the window for the run, `.update(state)` draws one frame after each
    resolved turn, and `.sink` swallows the plain per-turn text. The renderer NEVER
    advances the simulation — it only reads and draws what the sim produced.

    Controls: SPACE pauses/resumes, ESC or closing the window ends the run (raised as
    KeyboardInterrupt, which the launcher suppresses for a clean exit).
    """

    def __init__(self, *, sink: Any | None = None, turn_delay: float = 0.4) -> None:
        # `turn_delay` (seconds/turn) paces the watch; the renderer waits this long
        # between turns itself (responsively), so the sim's own sleep is left at 0.
        self.turn_delay = max(0.0, float(turn_delay))
        self._owns_sink = sink is None
        self.sink = sink if sink is not None else open(os.devnull, "w")
        self._screen: Any = None
        self._font: Any = None
        self._cell = _MAX_CELL
        self._size = 0
        self.paused = False
        self._last_state: dict[str, Any] | None = None
        self._terrain_bg: Any = None      # Slice 5: cached landscape, built once per grid size
        self._town_plans: dict[str, tuple] = {}  # Slice 6: cached (key, plan) per settlement id
        # Slice 8: ALL animation state lives HERE, renderer-local — never in world_state.
        self._prev_snapshot: dict[str, Any] | None = None  # last turn: positions/realms/homes
        self._territory_lerp: dict[str, tuple] = {}        # sid -> mid-lerp realm tint (aftermath)
        self._big_font: Any = None                         # the aftermath banner face

    # -- lifecycle ---------------------------------------------------------
    @contextlib.contextmanager
    def live(self):
        """Open the window for the duration of a run; quit Pygame cleanly on exit."""
        pygame.init()
        pygame.display.set_caption("AI Civilization — live")
        with contextlib.suppress(Exception):
            self._font = pygame.font.SysFont("menlo,monospace", 14)
        with contextlib.suppress(Exception):
            self._big_font = pygame.font.SysFont("menlo,monospace", 22, bold=True)
        try:
            yield self
        finally:
            pygame.quit()
            if self._owns_sink:
                self.sink.close()

    def _ensure_screen(self, size: int) -> None:
        """Create (or resize) the window: a square MAP zone on the left + a PANEL on the right.

        Slice 3 widens the window by _PANEL_W for the event-feed sidebar. The MAP keeps its
        square aspect in the top-left (the slice-1/2 coordinate mapping is unchanged); the
        HUD strip sits under the map, and the panel spans the full window height on the right.
        """
        if self._screen is not None and size == self._size:
            return
        self._size = size
        self._cell = _cell_size(size)
        grid_px = self._cell * max(1, size)
        self._screen = pygame.display.set_mode((grid_px + _PANEL_W, grid_px + _HUD_H))
        # Slice 5: bake the procedural landscape ONCE for this grid size (cached, blitted each
        # frame). Pure-hash texture/features — no RNG, so it never desyncs a seeded sim.
        self._terrain_bg = self._build_terrain(grid_px)
        # Slice 6: town plans hold pixel offsets, so a resize (new cell size) invalidates them.
        self._town_plans = {}

    # -- the per-turn hook the sim calls -----------------------------------
    def update(self, state: dict[str, Any]) -> None:
        """Draw the just-resolved turn: replay its battles as cinematics, then animate motion.

        Slice 8. The sim calls this AFTER the turn is fully resolved and advances itself only
        when we return — so a playing cinematic merely delays the next step call; the sim's
        sequence is byte-identical whether or not scenes play. Battle detection diffs this
        turn's events against the renderer-local snapshot of LAST turn; the snapshot is
        retaken last, ready for the next diff. READ only throughout.
        """
        self._last_state = state
        self._ensure_screen(int(state.get("size", 0)) or 1)
        self._pump_events()
        if self._prev_snapshot is not None and self.turn_delay > 0:
            lines = turn_events(state.get("events") or [], int(state.get("turn", 0)))
            for scene in battle_scenes(lines, self._prev_snapshot, state):
                self._play_cinematic(scene, state)
        self._animate_turn(state)
        self._prev_snapshot = take_snapshot(state)

    # -- input -------------------------------------------------------------
    def _pump_events(self) -> None:
        """Drain the OS event queue; toggle pause on SPACE, end the run on quit/ESC."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused

    def _pump_cinema_events(self) -> bool:
        """Input during a cinematic: quit/ESC still ends the run; ANY other key SKIPS the scene."""
        skip = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt
                skip = True
        return skip

    def _animate_turn(self, state: dict[str, Any]) -> None:
        """Pace the per-turn delay while LERPing every mover from last turn's cell to its new one.

        Slice 8's replacement for the old static wait: the same responsive loop (SPACE pauses,
        resume restarts the delay), but each frame is drawn with a motion fraction so agents WALK
        to their new cells instead of teleporting. Zero delay (tests / --speed 0) draws exactly
        one settled frame and returns without blocking — the old slice-1 behaviour.
        """
        prev_pos = (self._prev_snapshot or {}).get("positions") or {}
        if self.turn_delay <= 0:
            self._draw(state)
            return
        start = time.monotonic()
        while True:
            self._pump_events()
            if self.paused:
                self._draw(state, paused=True)
                start = time.monotonic()      # resume restarts the walk (the old deadline reset)
                time.sleep(0.01)
                continue
            t = min(1.0, (time.monotonic() - start) / self.turn_delay)
            self._draw(state, motion=(prev_pos, t))
            if t >= 1.0:
                return
            time.sleep(1 / 60)

    # -- drawing (pure reads of `state`) -----------------------------------
    def _to_px(self, x: int, y: int) -> tuple[int, int]:
        """Centre of grid cell (x, y) in pixels."""
        c = self._cell
        return (x * c + c // 2, y * c + c // 2)

    def _draw(self, state: dict[str, Any], *, paused: bool = False,
              motion: tuple[dict[str, tuple], float] | None = None,
              battle: tuple[dict[str, Any], float] | None = None) -> None:
        """One frame. `motion`=(prev_positions, t) lerps agents mid-walk; `battle`=(scene,
        elapsed) overlays a cinematic beat. Both default off -> the slice-1..7 static frame."""
        screen = self._screen
        if screen is None:
            return
        size = self._size
        cell = self._cell
        grid_px = cell * size

        # Slice 5: the cached LANDSCAPE (textured grass, trees/rocks/pond, vignette + frame)
        # under everything, blitted not rebuilt. Fallback flat fill if it isn't built yet.
        screen.fill(_FRAME_OUTER)  # base for the HUD/panel gutters; map zone is overdrawn below
        if self._terrain_bg is not None:
            screen.blit(self._terrain_bg, (0, 0))
        else:
            screen.fill(_GRASS_BASE, (0, 0, grid_px, grid_px))

        # Slice 5: settled land looks CULTIVATED — a translucent tilled-dirt tint (with furrows)
        # under the slice-2 region. Dynamic (settlements come and go), but cheap. No-op if none.
        self._draw_settlement_ground(state)

        # Slice 2: SETTLEMENTS as soft translucent regions UNDER everything else, so a
        # settlement reads as a background "place" with food and agents sitting on top.
        # No-op (slice-1 behaviour) when there are no settlements in world_state.
        self._draw_settlements(state)

        # Slice 4: FOOD as a little wheat stalk (a stalk + a few grain strokes), still green
        # and at the same positions; a plain dot when cells are too small for wheat to read.
        for fx, fy in state.get("food", []):
            self._draw_wheat(*self._to_px(fx, fy), cell)

        # Slice 4: AGENTS as little FIGURES (head + body) in their personality colour, scaled
        # by wealth; rulers wear a CROWN/STAR and anyone talking this turn gets a SPEECH
        # BUBBLE — so a stranger watching ONLY the map can read role and conversation.
        talkers = talkers_this_turn(state.get("events", []) or [], state.get("turn", 0))
        for agent in state.get("agents", []):
            if not getattr(agent, "alive", True):
                continue
            pos = getattr(agent, "position", None)
            if not pos:
                continue
            cx, cy = self._agent_px(agent, motion)  # slice 8: mid-walk lerp when motion plays
            r = agent_radius(_wealth(agent), cell)
            color = agent_color(getattr(agent, "personality", ""))
            figure_top = self._draw_agent_figure(cx, cy, r, color)
            self._draw_role_marker(cx, figure_top, r, agent_role(agent.name, state))
            if agent.name in talkers:
                self._draw_speech_bubble(cx + r + 1, figure_top, r)

        # Slice 8: the battle cinematic overlay (soldiers/dust/clash/fallen/banner), drawn over
        # the map but under the HUD/panel so the feed stays readable while a scene plays.
        if battle is not None:
            self._draw_battle_overlay(*battle)

        self._draw_hud(state, grid_px, paused, in_battle=battle is not None)
        # Slice 3: the right sidebar — a state summary above a scrolling event feed. Drawn
        # last so it owns the right zone cleanly; a pure read of state, like everything else.
        self._draw_panel(state, grid_px)
        pygame.display.flip()

    def _agent_px(self, agent: Any, motion: tuple[dict[str, tuple], float] | None) -> tuple[int, int]:
        """The agent's on-screen pixel centre — lerped from LAST turn's cell while motion plays.

        Slice 8. A mover glides (smoothstep) from its previous cell to the current one with a
        subtle walk bob; a newly spawned agent (no previous cell) appears at its cell with no
        lerp-from-nowhere; a stationary agent (or a settled frame, t>=1) sits exactly on-cell.
        """
        pos = agent.position
        cx, cy = self._to_px(pos[0], pos[1])
        if motion is None:
            return cx, cy
        prev_pos, t = motion
        pp = prev_pos.get(agent.name)
        if pp is None or (pp[0], pp[1]) == (pos[0], pos[1]) or t >= 1.0:
            return cx, cy
        px, py = self._to_px(int(pp[0]), int(pp[1]))
        e = ease(t)
        bob = abs(math.sin(t * math.pi * 3.0)) * max(1.0, self._cell * _WALK_BOB)
        return int(round(px + (cx - px) * e)), int(round(py + (cy - py) * e - bob))

    # -- Slice 5: cached procedural landscape (built ONCE; pure hash, no sim RNG) --
    def _build_terrain(self, grid_px: int) -> Any:
        """Bake the landscape into a Surface ONCE: textured grass, features, vignette, frame.

        Everything here is deterministic from `terrain_noise` (a pure coordinate hash) — it
        never calls `random`, so it cannot perturb the seeded sim. Built per grid size and
        cached, so it is free to blit each frame. Returns the finished background Surface.
        """
        if grid_px <= 0:
            return None
        cell, size = self._cell, self._size
        surf = pygame.Surface((grid_px, grid_px))
        surf.fill(_GRASS_BASE)

        # 1) GROUND texture: per-tile tonal value-noise + a low-frequency patch swing, so the
        #    grass reads as ground with broad lighter/darker patches rather than a flat colour.
        tile = max(3, cell // 2)
        for ty in range(0, grid_px, tile):
            for tx in range(0, grid_px, tile):
                fine = terrain_noise(tx // tile, ty // tile, 1) - 0.5
                patch = terrain_noise(tx // (tile * 5 + 1), ty // (tile * 5 + 1), 2) - 0.5
                shade = int(fine * 2 * _GRASS_VAR + patch * 2 * _GRASS_PATCH)
                surf.fill(_shade(_GRASS_BASE, shade), (tx, ty, tile, tile))

        # 2) STIPPLE: sparse light/dark specks for grain (cheap; most samples place nothing).
        for sy in range(0, grid_px, _STIPPLE_STEP):
            for sx in range(0, grid_px, _STIPPLE_STEP):
                h = terrain_noise(sx, sy, 3)
                if h > 0.90:
                    surf.set_at((sx, sy), _GRASS_SPECK_HI)
                elif h < 0.07:
                    surf.set_at((sx, sy), _GRASS_SPECK_LO)

        # 3) A POND in one deterministic off-centre region (kept clear of the central food arena).
        self._build_pond(surf, grid_px, cell)

        # 4) Scattered TREES and ROCKS, one chance per world cell (sparse thresholds).
        for cy in range(size):
            for cx in range(size):
                px, py = cx * cell + cell // 2, cy * cell + cell // 2
                if terrain_noise(cx, cy, 4) > _TREE_THRESHOLD:
                    self._build_tree(surf, px, py, cell)
                elif terrain_noise(cx, cy, 5) > _ROCK_THRESHOLD:
                    self._build_rock(surf, px, py, cell)

        # 5) ATMOSPHERE: a soft edge vignette for depth, and a clean framed border.
        self._build_vignette(surf, grid_px)
        pygame.draw.rect(surf, _FRAME_OUTER, (0, 0, grid_px, grid_px), 3)
        pygame.draw.rect(surf, _FRAME_INNER, (3, 3, grid_px - 6, grid_px - 6), 1)
        return surf

    def _build_pond(self, surf: Any, grid_px: int, cell: int) -> None:
        """A still pond in a fixed off-centre spot (deterministic; never the central arena)."""
        pcx = int(grid_px * 0.22)
        pcy = int(grid_px * 0.74)
        rx = max(cell, int(grid_px * 0.10))
        ry = max(cell, int(grid_px * 0.07))
        pygame.draw.ellipse(surf, _WATER, (pcx - rx, pcy - ry, 2 * rx, 2 * ry))
        pygame.draw.ellipse(surf, _WATER_HI, (pcx - rx, pcy - ry, 2 * rx, 2 * ry), 1)
        pygame.draw.ellipse(surf, _WATER_HI,
                            (pcx - rx // 2, pcy - ry // 2, rx, ry // 2), 1)  # a faint highlight

    def _build_tree(self, surf: Any, px: int, py: int, cell: int) -> None:
        """A simple tree: a brown trunk + a rounded green canopy with a lighter top."""
        r = max(2, int(cell * 0.42))
        trunk_w = max(1, r // 3)
        pygame.draw.rect(surf, _TREE_TRUNK, (px - trunk_w // 2, py, trunk_w, r))
        pygame.draw.circle(surf, _TREE_CANOPY, (px, py), r)
        pygame.draw.circle(surf, _TREE_CANOPY_HI, (px - r // 4, py - r // 4), max(1, r // 2))
        pygame.draw.circle(surf, _shade(_TREE_CANOPY, -14), (px, py), r, 1)

    def _build_rock(self, surf: Any, px: int, py: int, cell: int) -> None:
        """A small boulder: a grey blob with a light top facet."""
        r = max(2, int(cell * 0.3))
        pygame.draw.circle(surf, _ROCK, (px, py), r)
        pygame.draw.circle(surf, _ROCK_HI, (px - r // 4, py - r // 4), max(1, r // 2))
        pygame.draw.circle(surf, _shade(_ROCK, -18), (px, py), r, 1)

    def _build_vignette(self, surf: Any, grid_px: int) -> None:
        """Darken the map edges with nested translucent rings for atmospheric depth."""
        vign = pygame.Surface((grid_px, grid_px), pygame.SRCALPHA)
        rings = 26
        band = max(1, grid_px // (rings * 2))
        for i in range(rings):
            a = int(_VIGNETTE_MAX * (1 - i / rings) ** 2)
            inset = i * band
            if a > 0 and grid_px - 2 * inset > 0:
                pygame.draw.rect(vign, (0, 0, 0, a),
                                 (inset, inset, grid_px - 2 * inset, grid_px - 2 * inset),
                                 max(1, band))
        surf.blit(vign, (0, 0))

    def _draw_settlement_ground(self, state: dict[str, Any]) -> None:
        """Tint settled land toward tilled DIRT (with clipped furrows) — cultivated cue (READ).

        A translucent brown disc per settlement, sized just inside its region, with a few
        horizontal furrow lines clipped to the disc so it reads as a ploughed field. Drawn
        under the slice-2 teal region so the two blend. No-op when there are no settlements.
        """
        settlements = state.get("settlements")
        if not settlements:
            return
        cell = self._cell
        grid_px = cell * self._size
        pos_by_name = {
            a.name: a.position
            for a in state.get("agents", [])
            if getattr(a, "alive", True) and getattr(a, "position", None) is not None
        }
        overlay = pygame.Surface((grid_px, grid_px), pygame.SRCALPHA)
        for sid in sorted(settlements):
            center = settlements[sid].get("center")
            if center is None:
                continue
            members = settlements[sid].get("members") or ()
            mpos = [pos_by_name[n] for n in members if n in pos_by_name]
            rad = max(cell, int(round(settlement_radius_cells(center, mpos) * cell * 0.85)))
            cx, cy = self._to_px(int(center[0]), int(center[1]))
            pygame.draw.circle(overlay, (*_FARMLAND, _FARMLAND_ALPHA), (cx, cy), rad)
            step = max(3, cell // 2)
            crop_dx = max(4, cell // 2)
            for fy in range(cy - rad + step, cy + rad, step):    # furrows, clipped to the disc
                half = int((rad * rad - (fy - cy) ** 2) ** 0.5)
                if half <= 1:
                    continue
                pygame.draw.line(overlay, (*_FARMLAND_FURROW, _FARMLAND_ALPHA),
                                 (cx - half, fy), (cx + half, fy), 1)
                # Slice 6: CROP ROWS — little green tufts standing along each furrow.
                for x in range(cx - half + 2, cx + half - 1, crop_dx):
                    pygame.draw.line(overlay, (*_CROP, _FARMLAND_ALPHA + 40),
                                     (x, fy), (x, fy - max(2, cell // 6)), 1)
        self._screen.blit(overlay, (0, 0))

    # -- Slice 4: procedural map glyphs (all primitive shapes; pure drawing) -----
    def _draw_agent_figure(self, cx: int, cy: int, r: int, color: tuple[int, int, int]) -> int:
        """Draw a little person (head circle + trapezoid body) centred on (cx, cy).

        Colour is the personality colour and the whole figure scales with `r` (wealth), so
        slice-1's two encodings survive the upgrade. Returns the y of the figure's TOP (where
        a crown/star/bubble is stacked). A tiny cell (r below _FIGURE_MIN_R) falls back to the
        slice-1 dot so it never collapses into noise.
        """
        if r < _FIGURE_MIN_R:
            pygame.draw.circle(self._screen, _OUTLINE, (cx, cy), r + 1)
            pygame.draw.circle(self._screen, color, (cx, cy), r)
            return cy - r
        screen = self._screen
        head_r = max(2, round(r * 0.6))
        hx, hy = cx, cy - head_r                     # head sits in the upper half of the cell
        bw = max(2, round(r * 0.95))                 # body half-width at the base
        top, bot = hy + head_r - 1, cy + r           # body spans from under the head to the base
        body = [(cx - round(bw * 0.5), top), (cx + round(bw * 0.5), top),
                (cx + bw, bot), (cx - bw, bot)]
        pygame.draw.polygon(screen, color, body)
        pygame.draw.polygon(screen, _OUTLINE, body, 1)
        pygame.draw.circle(screen, color, (hx, hy), head_r)
        pygame.draw.circle(screen, _OUTLINE, (hx, hy), head_r, 1)
        return hy - head_r

    def _draw_role_marker(self, cx: int, top_y: int, r: int, role: str | None) -> None:
        """Stamp a ruler's insignia just above a figure: leader STAR, monarch / emperor CROWN."""
        if role is None:
            return
        gap = max(2, r // 3)
        base = top_y - gap
        if role == "leader":
            self._draw_star(cx, base - max(3, r // 2), max(3, r * 0.7))
        elif role == "monarch":
            self._draw_crown(cx, base, max(4, r), double=False)
        elif role == "emperor":
            self._draw_crown(cx, base, max(5, int(r * 1.2)), double=True)

    def _draw_crown(self, cx: int, base_y: int, w: int, *, double: bool) -> None:
        """A gold zig-zag crown sitting on baseline `base_y`; `double` stacks a second (emperor)."""
        h = max(3, w)
        pts = [(cx - w, base_y), (cx - w, base_y - h),
               (cx - w // 2, base_y - h // 3), (cx, base_y - h),
               (cx + w // 2, base_y - h // 3), (cx + w, base_y - h), (cx + w, base_y)]
        pygame.draw.polygon(self._screen, _CROWN, pts)
        pygame.draw.polygon(self._screen, _OUTLINE, pts, 1)
        if double:                                   # emperor: a smaller crown above the first
            self._draw_crown(cx, base_y - h - 1, max(2, (w * 2) // 3), double=False)

    def _draw_star(self, cx: int, cy: int, r: float) -> None:
        """A small five-point star (a trust-leader's mark) centred on (cx, cy)."""
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rad = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
        pygame.draw.polygon(self._screen, _STAR, pts)
        pygame.draw.polygon(self._screen, _OUTLINE, pts, 1)

    def _draw_speech_bubble(self, cx: int, top_y: int, r: int) -> None:
        """A small rounded speech bubble (with a downward tail + '...') marking a talker."""
        screen = self._screen
        w = max(7, r + 4)
        h = max(5, (r * 3) // 4 + 3)
        rect = pygame.Rect(cx - w // 2, top_y - h - 2, w, h)
        rad = max(2, h // 3)
        pygame.draw.rect(screen, _BUBBLE, rect, border_radius=rad)
        pygame.draw.rect(screen, _OUTLINE, rect, 1, border_radius=rad)
        pygame.draw.polygon(screen, _BUBBLE, [(rect.centerx - 2, rect.bottom - 1),
                                              (rect.centerx + 2, rect.bottom - 1),
                                              (rect.centerx, rect.bottom + 3)])
        if w >= 11:                                  # three dots only when the bubble is roomy
            dy = rect.centery
            for dx in (-3, 0, 3):
                pygame.draw.circle(screen, _BUBBLE_DOT, (rect.centerx + dx, dy), 1)

    def _draw_wheat(self, cx: int, cy: int, cell: int) -> None:
        """Food as a wheat stalk: a vertical stem + a few angled grain strokes (green)."""
        if cell < _FOOD_GLYPH_MIN_CELL:
            pygame.draw.circle(self._screen, _FOOD, (cx, cy), max(1, cell // 6))
            return
        screen = self._screen
        s = max(3, cell // 3)
        base, top = cy + s // 2, cy - s
        pygame.draw.line(screen, _FOOD, (cx, base), (cx, top), 1)        # the stalk
        for off in range(0, s + 1, max(2, s // 3)):                      # grain strokes up the stem
            yy = top + off
            pygame.draw.line(screen, _FOOD, (cx, yy), (cx - s // 2, yy - s // 3), 1)
            pygame.draw.line(screen, _FOOD, (cx, yy), (cx + s // 2, yy - s // 3), 1)

    def _draw_house(self, cx: int, cy: int, s: int) -> None:
        """A simple building: a square wall + a triangular roof, centred on (cx, cy)."""
        screen = self._screen
        half = max(2, s // 2)
        wall = pygame.Rect(cx - half, cy - half + half // 2, 2 * half, half + half // 2)
        pygame.draw.rect(screen, _HOUSE_WALL, wall)
        pygame.draw.rect(screen, _OUTLINE, wall, 1)
        roof = [(cx - half - 1, wall.top), (cx + half + 1, wall.top), (cx, cy - half - half // 2)]
        pygame.draw.polygon(screen, _HOUSE_ROOF, roof)
        pygame.draw.polygon(screen, _OUTLINE, roof, 1)

    def _draw_settlements(self, state: dict[str, Any]) -> None:
        """Draw each settlement as a translucent teal region + centre marker (READ only).

        For every record in world_state["settlements"] (the M2.1 settlements): size a soft
        region to the spread of its CURRENT living members, fill it translucently onto a
        per-frame alpha overlay (so it never paints over the agents opaquely), ring it with
        a slightly stronger edge, and stamp a small diamond at the centre. A short member-
        count label is added only when cells are large enough to stay legible. Pure read:
        agent positions are looked up by name from state["agents"]; nothing is written back.
        """
        settlements = state.get("settlements")
        if not settlements:
            return  # slice-1 behaviour: nothing extra drawn when no settlements exist
        screen = self._screen
        cell = self._cell
        grid_px = cell * self._size
        # Map member NAME -> current position (living agents only) for the spread read.
        pos_by_name = {
            a.name: a.position
            for a in state.get("agents", [])
            if getattr(a, "alive", True) and getattr(a, "position", None) is not None
        }
        # Personality + ruler reads (for a seat's CASTLE/HALL and its colour). Pure dict reads.
        personality_by_name = {a.name: getattr(a, "personality", "")
                               for a in state.get("agents", []) if getattr(a, "alive", True)}
        monarchs, leaders = state.get("monarchs", {}), state.get("leaders", {})
        empires = state.get("empires", {})
        # One translucent overlay per frame for the TERRITORY tint; circles drawn here blend over
        # the terrain WITHOUT darkening the buildings/agents (drawn afterwards, straight on screen).
        overlay = pygame.Surface((grid_px, grid_px), pygame.SRCALPHA)
        towns: list[tuple] = []
        for sid in sorted(settlements):
            rec = settlements[sid]
            center = rec.get("center")
            if center is None:
                continue
            members = rec.get("members") or ()
            member_positions = [pos_by_name[n] for n in members if n in pos_by_name]
            radius_px = int(round(settlement_radius_cells(center, member_positions) * cell))
            cx, cy = self._to_px(int(center[0]), int(center[1]))
            # Slice 8: the REALM layer — an owned settlement wears its TOP ruler's banner colour
            # (emperor > king > lone monarch); during a battle's aftermath, _territory_lerp holds
            # the mid-fade tint instead, so conquered land BLEEDS to the victor's colour rather
            # than snapping. Unowned settlements keep the slice-2 teal exactly as before.
            owner = settlement_realm(sid, state)
            tint = self._territory_lerp.get(sid) or (realm_color(owner) if owner is not None else None)
            if tint is not None:
                fill, fill_a, edge, edge_a = tint, _REALM_FILL_ALPHA, _shade(tint, 45), _REALM_EDGE_ALPHA
            else:
                fill, fill_a, edge, edge_a = (_SETTLEMENT_FILL, _SETTLEMENT_FILL_ALPHA,
                                              _SETTLEMENT_EDGE, _SETTLEMENT_EDGE_ALPHA)
            pygame.draw.circle(overlay, (*fill, fill_a), (cx, cy), radius_px)
            pygame.draw.circle(overlay, (*edge, edge_a), (cx, cy), radius_px, width=2)
            towns.append((sid, center, cx, cy, len(members), radius_px))
        screen.blit(overlay, (0, 0))

        # Slice 6: each settlement is now a detailed, GROWING built place — a cached plan of
        # houses + civic structure + a ruler's HALL/CASTLE. Drawn under food/agents (which the
        # caller draws afterwards). Tiny cells fall back to the slice-4 simple-house glyphs.
        for sid, center, cx, cy, count, radius_px in towns:
            top_y = cy
            if cell >= _TOWN_MIN_CELL:
                mon = monarchs.get(sid, {}).get("monarch")
                led = leaders.get(sid, {}).get("leader")
                if mon is not None:
                    kind, ruler, is_emp = "castle", mon, (mon in empires)
                elif led is not None:
                    kind, ruler, is_emp = "hall", led, False
                else:
                    kind, ruler, is_emp = None, None, False
                color = agent_color(personality_by_name.get(ruler, "")) if ruler else _DEFAULT_RULER
                key = (count, kind, ruler, is_emp, cell)
                cached = self._town_plans.get(sid)
                if cached is None or cached[0] != key:          # rebuild ONLY on membership/ruler change
                    cached = (key, build_town_plan(center, count, kind, color, is_emp, cell))
                    self._town_plans[sid] = cached
                self._draw_town(cx, cy, cached[1])
                top_y = cy - cached[1]["cluster_r"]
            else:
                self._draw_settlement_houses(cx, cy, radius_px, count, cell)
            if self._font is not None and cell >= _SETTLEMENT_LABEL_MIN_CELL:
                label = self._font.render(f"{sid}·{count}", True, _SETTLEMENT_LABEL)
                screen.blit(label, (cx - label.get_width() // 2, top_y - label.get_height() - 2))
        # Prune plans for settlements that no longer exist (keeps the cache bounded).
        self._town_plans = {s: v for s, v in self._town_plans.items() if s in settlements}

    def _draw_settlement_houses(self, cx: int, cy: int, radius_px: int, count: int, cell: int) -> None:
        """Ring a settlement's centre with a few HOUSE glyphs (count ~ membership, capped).

        Houses are placed on a deterministic ring (fixed angles, so the picture is stable and
        RNG-free) inside the region; a single-member hamlet gets one house at the centre. Below
        _HOUSE_MIN_CELL the region tint alone implies the place (houses won't read that small).
        """
        if cell < _HOUSE_MIN_CELL:
            return
        n = min(_MAX_HOUSES, max(1, count))
        house_s = max(4, int(cell * 0.9))
        if n == 1:
            self._draw_house(cx, cy, house_s)
            return
        ring = max(cell, int(radius_px * 0.5))
        for i in range(n):
            ang = -math.pi / 2 + (i / n) * 2 * math.pi
            self._draw_house(int(cx + ring * math.cos(ang)), int(cy + ring * math.sin(ang)), house_s)

    # -- Slice 6: detailed settlement rendering from a cached plan (pure drawing) ---
    def _draw_town(self, cx: int, cy: int, plan: dict[str, Any]) -> None:
        """Render a settlement from its cached `plan`: plaza, fence, paths, buildings, seat, well.

        Drawables are painted back-to-front (a packed-earth plaza and palisade behind, dirt paths,
        then every building/granary/seat sorted by ground-y so southern structures overlap northern
        ones), giving a clustered village real depth. All offsets are pixel deltas from (cx, cy).
        """
        screen = self._screen
        pygame.draw.circle(screen, _PLAZA, (cx, cy), plan["plaza_r"])          # market square ground
        if plan["fence_r"]:
            self._draw_fence_ring(cx, cy, plan["fence_r"])
        for b in plan["buildings"]:                                            # dirt roads to each house
            pygame.draw.line(screen, _PATH, (cx, cy), (cx + b["dx"], cy + b["dy"]), plan["path_w"])

        ops: list[tuple[int, str, dict]] = [(b["dy"], "house", b) for b in plan["buildings"]]
        if plan["granary"]:
            ops.append((plan["granary"]["dy"], "granary", plan["granary"]))
        if plan["central"]["kind"]:
            ops.append((0, "central", plan["central"]))
        for _dy, kind, d in sorted(ops, key=lambda o: o[0]):                   # painter's order by ground-y
            if kind == "house":
                self._draw_building(cx + d["dx"], cy + d["dy"], d["w"], d["h"],
                                    d["wall"], d["roof"], d["hip"], d["lit"])
            elif kind == "granary":
                self._draw_granary(cx + d["dx"], cy + d["dy"], d["scale"])
            elif d["kind"] == "castle":
                self._draw_castle(cx, cy, d["scale"], d["color"], d["emperor"])
            else:
                self._draw_hall(cx, cy, d["scale"], d["color"])
        w = plan["well"]
        self._draw_well(cx + w["dx"], cy + w["dy"], w["scale"])

    def _draw_building(self, gx: int, gy: int, w: int, h: int, wall: tuple, roof: tuple,
                       hip: bool, lit: bool) -> None:
        """A detailed house at ground-centre (gx, gy): walls, gabled/hip roof + shading, door,
        windows and a chimney — a recognisable dwelling rather than a plain square."""
        s = self._screen
        half = max(2, w // 2)
        wall_rect = pygame.Rect(gx - half, gy - h, 2 * half, h)
        pygame.draw.rect(s, wall, wall_rect)
        pygame.draw.rect(s, _OUTLINE, wall_rect, 1)
        roof_h = max(3, int(h * 0.7))
        rtop = wall_rect.top
        if hip:                                                               # hip roof (trapezoid)
            pk = max(1, half // 2)
            roof_pts = [(gx - half - 1, rtop), (gx + half + 1, rtop),
                        (gx + pk, rtop - roof_h), (gx - pk, rtop - roof_h)]
            ridge = [(gx - pk, rtop - roof_h), (gx + pk, rtop - roof_h), (gx + half + 1, rtop)]
        else:                                                                 # gabled roof (peak)
            roof_pts = [(gx - half - 1, rtop), (gx + half + 1, rtop), (gx, rtop - roof_h)]
            ridge = [(gx, rtop - roof_h), (gx + half + 1, rtop), (gx, rtop)]
        pygame.draw.polygon(s, roof, roof_pts)
        pygame.draw.polygon(s, _shade(roof, -24), ridge)                      # shaded sunless slope
        pygame.draw.polygon(s, _OUTLINE, roof_pts, 1)
        dw, dh = max(2, w // 4), max(3, h // 2)                               # door
        pygame.draw.rect(s, _DOOR, (gx - dw // 2, gy - dh, dw, dh))
        if w >= 9:                                                            # windows
            win = _WINDOW_LIT if lit else _WINDOW_DARK
            wsz = max(2, w // 5)
            pygame.draw.rect(s, win, (gx - half + 2, gy - h + 2, wsz, wsz))
            pygame.draw.rect(s, win, (gx + half - 2 - wsz, gy - h + 2, wsz, wsz))
        if h >= 7:                                                            # chimney
            cw = max(1, w // 6)
            pygame.draw.rect(s, _CHIMNEY, (gx + half - cw - 1, rtop - roof_h // 2, cw, roof_h // 2 + 2))

    def _crenellate(self, x: int, top_y: int, width: int, color: tuple) -> None:
        """A row of merlons (battlement notches) along a tower/keep top edge."""
        m = max(2, width // 7)
        n = max(2, width // (2 * m))
        for i in range(n):
            mx = x + i * 2 * m
            if mx + m <= x + width:
                pygame.draw.rect(self._screen, color, (mx, top_y - m, m, m))

    def _draw_castle(self, cx: int, cy: int, scale: int, color: tuple, emperor: bool) -> None:
        """A monarch's CASTLE: a tall stone keep with battlements + flanking towers (with conical
        roofs in the RULER's colour), a gate, and a banner — unmistakably grander than a village.
        An emperor's seat is taller with a second banner."""
        s = self._screen
        kw = max(8, int(scale * 1.7))
        kh = max(12, int(scale * (2.8 if emperor else 2.3)))
        tw = max(5, int(scale * 0.95))
        th = int(kh * 0.82)
        for sx in (cx - kw // 2 - tw // 2 + 1, cx + kw // 2 + tw // 2 - 1):   # two flanking towers
            trect = pygame.Rect(sx - tw // 2, cy - th, tw, th)
            pygame.draw.rect(s, _CASTLE_STONE_DK, trect)
            pygame.draw.rect(s, _OUTLINE, trect, 1)
            self._crenellate(trect.left, trect.top, tw, _CASTLE_STONE)
            pygame.draw.polygon(s, color, [(trect.left - 1, trect.top - 2),  # conical roof in ruler colour
                                           (trect.right + 1, trect.top - 2),
                                           (sx, trect.top - tw)])
        keep = pygame.Rect(cx - kw // 2, cy - kh, kw, kh)                     # the central keep
        pygame.draw.rect(s, _CASTLE_STONE, keep)
        pygame.draw.rect(s, _OUTLINE, keep, 1)
        self._crenellate(keep.left, keep.top, kw, _CASTLE_STONE_DK)
        gw, gh = max(3, kw // 3), max(4, kh // 3)                             # gate
        pygame.draw.rect(s, _GATE, (cx - gw // 2, cy - gh, gw, gh))
        pygame.draw.arc(s, _GATE, (cx - gw // 2, cy - gh - gw // 2, gw, gw), 0, math.pi, 2)
        for wy in (cy - kh + kh // 3, cy - kh + 2 * kh // 3):                 # arrow-slit windows
            pygame.draw.rect(s, _WINDOW_DARK, (cx - 1, wy, 2, max(2, kh // 6)))
        pole_top = keep.top - max(4, scale)                                   # banner pole + pennant
        pygame.draw.line(s, _OUTLINE, (cx, keep.top), (cx, pole_top), 1)
        pygame.draw.polygon(s, color, [(cx, pole_top), (cx + scale, pole_top + 2), (cx, pole_top + 5)])
        if emperor:
            pygame.draw.polygon(s, _shade(color, 30),
                                [(cx, pole_top + 5), (cx + scale - 2, pole_top + 7), (cx, pole_top + 10)])

    def _draw_hall(self, cx: int, cy: int, scale: int, color: tuple) -> None:
        """A trust-leader's HALL: a longhouse larger than a hut, with a big gabled roof, a double
        door and a small pennant in the leader's colour — between a common house and a castle."""
        s = self._screen
        w, h = max(10, int(scale * 1.9)), max(8, int(scale * 1.4))
        self._draw_building(cx, cy, w, h, _WALL_TONES[1], _ROOF_TONES[3], hip=False, lit=True)
        dw = max(3, w // 4)                                                   # a grander double door
        pygame.draw.rect(s, _DOOR, (cx - dw // 2, cy - max(4, h // 2), dw, max(4, h // 2)))
        pygame.draw.line(s, _shade(_DOOR, 30), (cx, cy - max(4, h // 2)), (cx, cy), 1)
        peak = cy - h - max(3, int(h * 0.7))
        pygame.draw.line(s, _OUTLINE, (cx, peak), (cx, peak - max(4, scale)), 1)  # pennant pole
        pygame.draw.polygon(s, color, [(cx, peak - max(4, scale)),
                                       (cx + max(4, scale - 1), peak - max(4, scale) + 2),
                                       (cx, peak - max(4, scale) + 5)])

    def _draw_granary(self, gx: int, gy: int, scale: int) -> None:
        """A granary: a stout light-walled store with a tall conical roof, set near the fields."""
        s = self._screen
        w, h = max(6, int(scale * 0.95)), max(8, int(scale * 1.5))
        rect = pygame.Rect(gx - w // 2, gy - h, w, h)
        pygame.draw.rect(s, _GRANARY_WALL, rect)
        pygame.draw.rect(s, _OUTLINE, rect, 1)
        for ly in range(rect.top + 2, rect.bottom, max(2, h // 4)):           # plank lines
            pygame.draw.line(s, _shade(_GRANARY_WALL, -22), (rect.left, ly), (rect.right, ly), 1)
        pygame.draw.polygon(s, _ROOF_TONES[2], [(rect.left - 2, rect.top), (rect.right + 2, rect.top),
                                                (gx, rect.top - int(scale * 1.1))])
        pygame.draw.polygon(s, _OUTLINE, [(rect.left - 2, rect.top), (rect.right + 2, rect.top),
                                          (gx, rect.top - int(scale * 1.1))], 1)

    def _draw_well(self, gx: int, gy: int, scale: int) -> None:
        """A stone well with water and a little gabled roof on two posts (the village centre)."""
        s = self._screen
        r = max(3, scale // 2)
        pygame.draw.circle(s, _WELL_STONE, (gx, gy), r)
        pygame.draw.circle(s, _WELL_WATER, (gx, gy), max(1, r - 2))
        pygame.draw.circle(s, _OUTLINE, (gx, gy), r, 1)
        ph = max(4, scale)
        pygame.draw.line(s, _TREE_TRUNK, (gx - r, gy), (gx - r, gy - ph), 1)
        pygame.draw.line(s, _TREE_TRUNK, (gx + r, gy), (gx + r, gy - ph), 1)
        pygame.draw.polygon(s, _ROOF_TONES[0], [(gx - r - 1, gy - ph), (gx + r + 1, gy - ph),
                                                (gx, gy - ph - r)])

    def _draw_fence_ring(self, cx: int, cy: int, radius: int) -> None:
        """A palisade: posts joined by rails ringing a large settlement (deterministic spacing)."""
        s = self._screen
        n = max(10, int(radius / max(3, self._cell * 0.6)))
        prev = None
        for i in range(n + 1):
            ang = (i / n) * 2 * math.pi
            x, y = int(cx + radius * math.cos(ang)), int(cy + radius * math.sin(ang))
            if prev is not None:
                pygame.draw.line(s, _FENCE, prev, (x, y), 1)
            pygame.draw.circle(s, _shade(_FENCE, 18), (x, y), 1)
            prev = (x, y)

    def _draw_hud(self, state: dict[str, Any], grid_px: int, paused: bool,
                  in_battle: bool = False) -> None:
        """A one-line status strip under the grid (turn / living / food / pause / battle)."""
        screen = self._screen
        pygame.draw.rect(screen, _HUD_BG, (0, grid_px, grid_px, _HUD_H))
        if self._font is None:
            return
        turn = state.get("turn", 0)
        living = sum(1 for a in state.get("agents", []) if getattr(a, "alive", True))
        food = len(state.get("food", []))
        text = f"turn {turn}   living {living}   food {food}"
        towns = len(state.get("settlements", {}))
        if towns:  # only shown once settlements exist, so the slice-1 HUD is unchanged
            text += f"   towns {towns}"
        text += "   [space] pause  [esc] quit"
        if paused:
            text = "PAUSED — [space] resume   " + text
        label = self._font.render(text, True, _HUD_FG)
        screen.blit(label, (8, grid_px + (_HUD_H - label.get_height()) // 2))
        if in_battle:  # slice 8: a small indicator while a battle cinematic plays
            chip = self._font.render("BATTLE — any key skips", True, _BATTLE_CHIP)
            screen.blit(chip, (grid_px - chip.get_width() - 8,
                               grid_px + (_HUD_H - chip.get_height()) // 2))

    # -- Slice 3: the right sidebar (state summary + event feed) ------------
    def _stat_lines(self, state: dict[str, Any]) -> list[tuple[str, str]]:
        """(label, value) rows for the panel's state summary (pure read of `state`).

        Always turn / living / food; then a count for each institution layer PRESENT in
        world_state ("whatever exists"), so the summary reflects which systems are on.
        """
        lines = [
            ("turn", str(state.get("turn", 0))),
            ("living", str(sum(1 for a in state.get("agents", []) if getattr(a, "alive", True)))),
            ("food", str(len(state.get("food", [])))),
        ]
        for key, label in (("settlements", "settlements"), ("kingdoms", "kingdoms"),
                           ("empires", "empires")):
            if key in state:
                lines.append((label, str(len(state.get(key) or {}))))
        return lines

    def _feed_rows(self, state: dict[str, Any], inner_w: int,
                   max_rows: int) -> list[tuple[str, tuple[int, int, int]]]:
        """Colour-coded, wrapped event rows sized to the panel (a read of state["events"])."""
        char_w = max(1, self._font.size("M")[0])
        cols = max(8, inner_w // char_w)
        return wrap_events(state.get("events") or [], cols, max_rows)

    def _draw_panel(self, state: dict[str, Any], grid_px: int) -> None:
        """Draw the right sidebar: a STATE summary above a scrolling EVENT feed (READ only).

        Top block = current-state counts (turn/living/food + settlements/kingdoms/empires
        where present); below a divider, the EVENTS feed shows the most recent log lines
        that fit, wrapped to the panel and lightly colour-coded by type, newest at the
        bottom. Pure read — it never touches world_state.
        """
        screen = self._screen
        font = self._font
        win_h = grid_px + _HUD_H
        x0, pad = grid_px, _PANEL_PAD
        inner_w = _PANEL_W - 2 * pad
        pygame.draw.rect(screen, _PANEL_BG, (x0, 0, _PANEL_W, win_h))
        if font is None:
            return
        line_h = font.get_height() + 3
        y = pad

        # STATE summary.
        screen.blit(font.render("STATE", True, _PANEL_TITLE), (x0 + pad, y))
        y += line_h + 2
        for label, value in self._stat_lines(state):
            screen.blit(font.render(label, True, _STAT_LABEL), (x0 + pad, y))
            val = font.render(value, True, _STAT_VALUE)
            screen.blit(val, (x0 + _PANEL_W - pad - val.get_width(), y))
            y += line_h

        # Divider + EVENTS header.
        y += 5
        pygame.draw.line(screen, _PANEL_DIV, (x0 + pad, y), (x0 + _PANEL_W - pad, y))
        y += 7
        screen.blit(font.render("EVENTS", True, _PANEL_TITLE), (x0 + pad, y))
        y += line_h + 2

        # Feed: fill the remaining height, newest at the bottom; graceful when empty.
        feed_top, feed_bottom = y, win_h - pad
        max_rows = max(1, (feed_bottom - feed_top) // line_h)
        rows = self._feed_rows(state, inner_w, max_rows)
        if not rows:
            screen.blit(font.render("(no events yet)", True, _FEED_DEFAULT), (x0 + pad, feed_top))
            return
        for text, color in rows:
            screen.blit(font.render(text, True, color), (x0 + pad, feed_top))
            feed_top += line_h

    # -- Slice 8: the battle cinematic (playback + overlay drawing) ---------
    def _play_cinematic(self, scene: dict[str, Any], state: dict[str, Any]) -> None:
        """REPLAY one already-resolved battle as a five-beat scene; ANY key skips to the end.

        The sim is NOT stepped while this plays — the renderer owns pacing, and this loop simply
        does not return control to the caller's turn loop until the scene ends. Everything shown
        comes from the scene timeline (parsed from the event log + last turn's snapshot); the
        only inventions are presentation (dust/jitter/flashes), hashed off frame+position via
        terrain_noise — never the sim RNG. During the AFTERMATH beat, _territory_lerp fades the
        conquered settlements from their old realm colour to the victor's; it is cleared on exit
        so the map settles onto the true current colours (also the skip-key end-state).
        """
        start = time.monotonic()
        after_start = _CIN_TOTAL - _CIN_AFTER
        try:
            while True:
                if self._pump_cinema_events():
                    return                        # skipped -> next frame is the settled end-state
                el = time.monotonic() - start
                if el >= _CIN_TOTAL:
                    return
                frac = 0.0 if el < after_start else (el - after_start) / _CIN_AFTER
                self._territory_lerp = self._territory_colors(scene, frac, state)
                self._draw(state, battle=(scene, el))
                time.sleep(1 / 60)
        finally:
            self._territory_lerp = {}

    def _territory_colors(self, scene: dict[str, Any], frac: float,
                          state: dict[str, Any]) -> dict[str, tuple]:
        """sid -> the mid-lerp realm tint: the loser's OLD colour easing to the current owner's.

        The 'to' colour is read live from the CURRENT state (the sim already applied the
        conquest), so at frac=1 the lerp lands exactly on what the realm layer will draw anyway
        — clearing the override afterwards is seamless. Empty when no territory changed hands.
        """
        out: dict[str, tuple] = {}
        for sid, from_c in scene.get("territory") or ():
            owner = settlement_realm(sid, state)
            to_c = realm_color(owner) if owner is not None else _SETTLEMENT_FILL
            out[sid] = lerp_color(from_c, to_c, ease(frac))
        return out

    def _formation(self, n: int, ux: float, uy: float, salt: int) -> list[tuple[float, float]]:
        """Rank-and-file pixel offsets for `n` soldiers facing (ux, uy) — deterministic ranks of
        four with a small per-soldier hash jitter (terrain_noise), so a host reads as a host."""
        px_, py_ = -uy, ux                        # the across-the-line direction
        cell = self._cell
        out = []
        for i in range(n):
            row, col = divmod(i, 4)
            across = (col - 1.5) * cell * 0.62
            back = row * cell * 0.55
            jx = (terrain_noise(i, salt, 11) - 0.5) * cell * 0.25
            jy = (terrain_noise(i, salt, 12) - 0.5) * cell * 0.25
            out.append((across * px_ - back * ux + jx, across * py_ - back * uy + jy))
        return out

    def _draw_soldier(self, x: float, y: float, color: tuple, facing_right: bool) -> None:
        """An armed soldier: the slice-4 figure in its side's realm colour, carrying a spear."""
        r = max(3, int(self._cell * 0.30))
        top = self._draw_agent_figure(int(x), int(y), r, color)
        sx = int(x) + (r + 1 if facing_right else -(r + 1))
        tip_y = top - max(2, r // 2)
        pygame.draw.line(self._screen, _SPEAR, (sx, int(y) + r), (sx, tip_y), 1)
        pygame.draw.polygon(self._screen, _SPEAR,
                            [(sx - 2, tip_y), (sx + 2, tip_y), (sx, tip_y - max(3, r // 2))])

    def _draw_fallen(self, x: float, y: float, name: str, tip: float) -> None:
        """A casualty going down: the figure TIPS over (tip 0->1), then lies as a gray marker
        with its NAME above — the named dead are real agents the battle killed, not extras."""
        cell = self._cell
        r = max(3, int(cell * 0.30))
        if tip < 1.0:                             # mid-fall: a gray body rotating to the ground
            ang = ease(tip) * (math.pi / 2)
            hx = x + math.sin(ang) * 2 * r
            hy = y + r - math.cos(ang) * 2 * r
            pygame.draw.line(self._screen, _FALLEN, (int(x), int(y + r)), (int(hx), int(hy)),
                             max(2, r // 2))
            pygame.draw.circle(self._screen, _FALLEN, (int(hx), int(hy)), max(2, r // 2))
        else:                                     # down: a lying marker
            w, h = max(6, int(cell * 0.62)), max(3, int(cell * 0.24))
            rect = pygame.Rect(int(x - w / 2), int(y + r - h), w, h)
            pygame.draw.ellipse(self._screen, _FALLEN, rect)
            pygame.draw.ellipse(self._screen, _OUTLINE, rect, 1)
        if self._font is not None and self._cell >= _SETTLEMENT_LABEL_MIN_CELL:
            lab = self._font.render(name, True, _FALLEN)
            self._screen.blit(lab, (int(x) - lab.get_width() // 2,
                                    int(y) - r * 3 - lab.get_height()))

    def _draw_banner(self, text: str, fade: float) -> None:
        """The aftermath outcome banner: a translucent band across the map with the verdict."""
        grid_px = self._cell * self._size
        band_h = max(36, self._cell * 2)
        y0 = (grid_px - band_h) // 2
        band = pygame.Surface((grid_px, band_h), pygame.SRCALPHA)
        band.fill((*_BANNER_BG, int(210 * max(0.0, min(1.0, fade)))))
        self._screen.blit(band, (0, y0))
        pygame.draw.line(self._screen, _CROWN, (6, y0), (grid_px - 6, y0), 1)
        pygame.draw.line(self._screen, _CROWN, (6, y0 + band_h), (grid_px - 6, y0 + band_h), 1)
        font = self._big_font or self._font
        if font is None:
            return
        label = font.render(text, True, _BANNER_FG)
        if label.get_width() > grid_px - 16 and self._font is not None:
            label = self._font.render(text, True, _BANNER_FG)   # long verdicts drop to the small face
        self._screen.blit(label, ((grid_px - label.get_width()) // 2,
                                  y0 + (band_h - label.get_height()) // 2))

    def _draw_battle_overlay(self, scene: dict[str, Any], el: float) -> None:
        """One frame of the cinematic at elapsed `el`: muster -> march -> clash -> fall -> banner.

        Pure drawing from the scene timeline. Beat boundaries are the _CIN_* constants; all
        scatter (dust, melee jitter, flash placement) comes from terrain_noise keyed on the
        30Hz frame index + element index, so playback is deterministic and RNG-free.
        """
        cell = self._cell
        screen = self._screen
        ax, ay = self._to_px(*scene["att_pos"])
        bx, by = self._to_px(*scene["def_pos"])
        dist = math.hypot(bx - ax, by - ay) or 1.0
        ux, uy = (bx - ax) / dist, (by - ay) / dist
        # The clash line sits at the DEFENDER settlement's edge, pulled toward the attacker.
        m = (bx - ux * min(dist * 0.5, cell * 2.4), by - uy * min(dist * 0.5, cell * 2.4))
        meet_a = (m[0] - ux * cell * 0.55, m[1] - uy * cell * 0.55)
        meet_d = (m[0] + ux * cell * 0.55, m[1] + uy * cell * 0.55)
        n_a = max(1, min(_MAX_SOLDIER_GLYPHS, int(scene.get("n_att") or 1)))
        n_d = max(0, min(_MAX_SOLDIER_GLYPHS, int(scene.get("n_def") or 0)))
        form_a = self._formation(n_a, ux, uy, salt=1)
        form_d = self._formation(n_d, -ux, -uy, salt=2)
        frame = int(el * 30)
        t1, t2, t3 = _CIN_MUSTER, _CIN_MUSTER + _CIN_MARCH, _CIN_MUSTER + _CIN_MARCH + _CIN_CLASH
        t4 = t3 + _CIN_FALL

        # Beat state: where each host stands, how many have mustered, and the melee jitter.
        if el < t1:                                       # MUSTER at the attacker's capital
            a_c, d_c = (float(ax), float(ay)), meet_d
            vis_a = max(1, int(math.ceil(n_a * (el / t1))))
            vis_d = int(math.ceil(n_d * (el / t1)))
            jit = 0.0
        elif el < t2:                                     # MARCH on the defender's settlement
            p = ease((el - t1) / _CIN_MARCH)
            a_c = (ax + (meet_a[0] - ax) * p, ay + (meet_a[1] - ay) * p)
            d_c, vis_a, vis_d, jit = meet_d, n_a, n_d, 0.0
            for k in range(5):                            # dust puffs behind the moving host
                if terrain_noise(frame, k, 31) > 0.35:
                    back = cell * (0.8 + terrain_noise(frame, k, 32) * 1.6)
                    side = (terrain_noise(frame, k, 33) - 0.5) * cell * 1.5
                    pygame.draw.circle(
                        screen, _DUST,
                        (int(a_c[0] - ux * back - uy * side), int(a_c[1] - uy * back + ux * side)),
                        max(1, int(terrain_noise(frame, k, 34) * cell * 0.22)))
        else:                                             # CLASH / FALL / AFTERMATH at the line
            a_c, d_c, vis_a, vis_d = meet_a, meet_d, n_a, n_d
            jit = cell * 0.30 if el < t3 else 0.0

        # The named dead hold the FIRST slots of their side; each falls at its own moment.
        dead = [(nm, True) for nm, _p in scene.get("att_dead") or ()] + \
               [(nm, False) for nm, _p in scene.get("def_dead") or ()]
        fall_at = {}
        for j, (nm, att_side) in enumerate(dead):
            fall_at[(att_side, nm)] = t3 + _CIN_FALL * (j + 0.4) / max(1, len(dead))

        for side_dead, center, form, vis, color, facing in (
                ([nm for nm, s in dead if s], a_c, form_a, vis_a, scene["att_color"], ux >= 0),
                ([nm for nm, s in dead if not s], d_c, form_d, vis_d, scene["def_color"], ux < 0)):
            att_side = form is form_a
            for i in range(vis):
                ox, oy = form[i]
                x, y = center[0] + ox, center[1] + oy
                if jit > 0:                       # melee: position shake + brief lunges
                    x += (terrain_noise(frame, i, 41 if att_side else 43) - 0.5) * jit * 2
                    y += (terrain_noise(frame, i, 42 if att_side else 44) - 0.5) * jit * 2
                if i < len(side_dead):
                    nm = side_dead[i]
                    at = fall_at[(att_side, nm)]
                    if el >= at:
                        self._draw_fallen(x, y, nm, tip=(el - at) / 0.25)
                        continue
                self._draw_soldier(x, y, color, facing)

        if t2 <= el < t3:                                 # white clash-flash bursts at the line
            for k in range(3):
                if terrain_noise(frame, k, 21) > 0.45:
                    fx = m[0] + (terrain_noise(frame, k, 22) - 0.5) * cell * 1.8
                    fy = m[1] + (terrain_noise(frame, k, 23) - 0.5) * cell * 1.2
                    fr = 2 + int(terrain_noise(frame, k, 24) * cell * 0.4)
                    pygame.draw.circle(screen, _FLASH, (int(fx), int(fy)), fr)
                    pygame.draw.circle(screen, _shade(_FLASH, -70), (int(fx), int(fy)), fr, 1)

        if el >= t4:                                      # AFTERMATH: the verdict, prominently
            self._draw_banner(scene["banner"], fade=(el - t4) / 0.25)
