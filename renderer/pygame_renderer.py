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
(6), slice 8 — WAR & MOTION: a realm-colour territory layer, battles replayed as short
read-only cinematics, and smooth inter-turn agent movement — and slice 9 — VISUAL
POLISH: a full-bleed landscape (wilderness margin + coast, no black void), one
top-left sun with ground shadows and lit/shaded building faces, a single central
PALETTE (calm earthy base; rulers/war stay saturated and pop), ambient life (chimney
smoke, wheat sway, water shimmer, birds, banner flutter, window flicker) on a
renderer-local frame clock, and a warm daylight grade tying the scene together.
Slice 10 — DAY/NIGHT CYCLE: time of day derives PURELY from the sim turn
(dawn -> day -> dusk -> night, ~24 turns per day), a continuously interpolated
colour grade sweeps the whole map through golden dawns, neutral days, burning
dusks and deep blue nights; at night the lit windows GLOW, castles raise
torchlight, stars mirror on the water, shadows fade, and realm/figure colours
mute so the lights carry the scene — while battles stay bright and readable.

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
# Slice 9: ONE central PALETTE for every scene-defining colour, so the whole look is tunable
# in one place. Design intent: a calm, earthy base (grass/soil/wood/water), figures slightly
# DESATURATED to sit *in* the world, and the IMPORTANT things — rulers, realm banners, battle
# effects — kept saturated so they pop against it. The module-level constant names the slices
# already use are kept, but each is now DERIVED from PALETTE (UI chrome — panel/HUD/feed —
# stays beside its slice: it frames the scene rather than being part of it).
def _desat(color: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    """Desaturate by fraction f in [0,1]: lerp toward the colour's own luma gray (pure)."""
    g = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
    return tuple(int(round(c + (g - c) * max(0.0, min(1.0, f)))) for c in color)


PALETTE: dict[str, tuple[int, int, int]] = {
    # ground & water
    "grass_base": (44, 58, 40),        # the grassland base tone
    "grass_speck_hi": (60, 86, 54),
    "grass_speck_lo": (32, 48, 32),
    "sand": (146, 132, 98),            # the shoreline strip
    "water": (46, 84, 116),
    "water_shallow": (58, 104, 132),
    "water_hi": (92, 136, 166),        # ripple/shimmer highlights
    "farmland": (110, 82, 50),
    "farmland_furrow": (84, 60, 38),
    "crop": (118, 156, 80),
    "wheat": (176, 184, 106),          # food — a soft gold-green that belongs to the grass
    # nature features
    "tree_trunk": (74, 52, 34),
    "tree_canopy": (44, 76, 44),
    "tree_canopy_hi": (58, 96, 56),
    "rock": (92, 96, 90),
    "rock_hi": (122, 126, 118),
    # settlements & structures (the old teal ring is retired for a warm, desaturated earth
    # tint that never fights the realm hues)
    "settlement_fill": (164, 152, 118),
    "settlement_edge": (186, 174, 138),
    "settlement_label": (206, 198, 170),
    "path": (124, 106, 80),
    "plaza": (140, 120, 88),
    "door": (66, 46, 32),
    "window_lit": (226, 198, 120),
    "window_dark": (74, 78, 84),
    "chimney": (96, 84, 76),
    "granary": (180, 148, 102),
    "fence": (112, 92, 66),
    "well_stone": (148, 148, 146),
    "well_water": (58, 96, 128),
    "castle_stone": (154, 156, 160),
    "castle_stone_dk": (112, 114, 122),
    "gate": (52, 44, 38),
    # figures: SATURATED bases; what is drawn is _desat(base, _TRAIT_DESAT) so commoners sit
    # in the landscape while rulers/war (crown/realm/flash) keep full chroma and pop
    "curiosity": (240, 198, 70),       # amber — the explorer
    "caution": (92, 146, 230),         # blue — the careful/territorial
    "friendliness": (236, 112, 178),   # pink — the social
    "independence": (220, 84, 84),     # red — the competitive/aloof
    "crown": (245, 205, 70),           # stays saturated: royalty must pop
    # atmosphere & ambient life
    "shadow": (10, 12, 8),             # the ground-shadow tone (drawn translucent)
    "smoke": (206, 202, 194),
    "bird": (52, 56, 50),
    "daylight": (255, 214, 156),       # the warm full-scene grade (very low alpha)
    # slice 10: the day/night cycle
    "night": (13, 22, 54),             # the deep cool blue-dark night grade
    "dawn_gold": (255, 196, 112),      # the dawn grade + the directional sunrise wash
    "dusk_ember": (255, 134, 88),      # the burning orange/pink dusk grade
    "starlight": (222, 230, 250),      # stars mirrored on the night water
    "window_glow": (255, 186, 92),     # the halo around a lit window at night
    "torch_flame": (255, 168, 64),     # torch flame + its warm halo
    "torch_core": (255, 232, 158),     # the white-hot torch heart
    "moon_smoke": (222, 230, 244),     # chimney smoke catching the moonlight
}
_TRAIT_DESAT = 0.22                    # how far commoner figures step toward the earth tones

_TERRAIN = (38, 42, 36)        # muted dark olive — a calm flat ground (pre-slice-5 fallback)
_GRID_LINE = (48, 53, 46)      # barely-there cell lines (drawn only when cells are large)
_FOOD = PALETTE["wheat"]       # food glyphs — toned to the grass, no longer neon
_HUD_BG = (24, 26, 22)
_HUD_FG = (210, 214, 200)
_OUTLINE = (16, 18, 14)        # thin dark ring around each agent for contrast

# Dominant trait -> DRAWN colour (desaturated from the palette base; still four distinct hues).
_TRAIT_COLOR = {t: _desat(PALETTE[t], _TRAIT_DESAT)
                for t in ("curiosity", "caution", "friendliness", "independence")}
_DEFAULT_COLOR = (180, 180, 180)      # grey — unrecognised personality

# Slice 2 (re-toned in slice 9): SETTLEMENTS. The region tint is now a warm desaturated
# earth (see PALETTE) — a settled place reads as worked ground rather than a teal stamp,
# and the subtle ring no longer fights the slice-8 realm colours. Still TRANSLUCENT (low
# alpha) so it stays background context under the agents.
_SETTLEMENT_FILL = PALETTE["settlement_fill"]
_SETTLEMENT_EDGE = PALETTE["settlement_edge"]
_SETTLEMENT_LABEL = PALETTE["settlement_label"]
_SETTLEMENT_FILL_ALPHA = 38           # how opaque the region tint is (0..255; low = subtle)
_SETTLEMENT_EDGE_ALPHA = 110          # the boundary ring, a touch stronger than the fill
_SETTLEMENT_MIN_CELLS = 1.6           # smallest region radius, in grid cells
_SETTLEMENT_LABEL_MIN_CELL = 10       # only draw labels when cells are big enough to stay legible

# Slice 4: ICONOGRAPHY. Procedural glyphs (no asset files) so the MAP is self-explanatory:
# agents are little FIGURES, rulers wear CROWNS / a leader STAR, talkers get a SPEECH BUBBLE,
# food is a WHEAT stalk, and settlements show HOUSE buildings. A handful of primitives each.
_CROWN = PALETTE["crown"]     # gold crown — monarch (king) / emperor (kept saturated)
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
_GRASS_BASE = PALETTE["grass_base"]
_GRASS_VAR = 9                # fine per-tile tonal swing (+/-), the cheap value-noise texture
_GRASS_PATCH = 12             # low-frequency swing -> broad patches of darker/lighter grass
_GRASS_SPECK_HI = PALETTE["grass_speck_hi"]   # occasional lighter stipple speck
_GRASS_SPECK_LO = PALETTE["grass_speck_lo"]   # occasional darker stipple speck
_TREE_TRUNK = PALETTE["tree_trunk"]
_TREE_CANOPY = PALETTE["tree_canopy"]
_TREE_CANOPY_HI = PALETTE["tree_canopy_hi"]
_ROCK = PALETTE["rock"]
_ROCK_HI = PALETTE["rock_hi"]
_WATER = PALETTE["water"]
_WATER_SHALLOW = PALETTE["water_shallow"]
_WATER_HI = PALETTE["water_hi"]
_SAND = PALETTE["sand"]
_FARMLAND = PALETTE["farmland"]   # tilled-dirt tint near a settlement (translucent each frame)
_FARMLAND_FURROW = PALETTE["farmland_furrow"]
_FARMLAND_ALPHA = 52
_VIGNETTE_MAX = 48            # slice 9: softened — depth without the old black-edged stage
_FRAME_OUTER = (22, 26, 20)   # dark outer frame around the map zone
_FRAME_INNER = (78, 90, 66)   # a thin lighter inner line, for a framed-map look
# Feature density thresholds on terrain_noise (sparse, so the map stays readable).
_TREE_THRESHOLD = 0.93        # ~7% of PLAYABLE cells get a tree
_TREE_THRESHOLD_WILD = 0.80   # the wilderness fringe is denser (~20%) — the forest closes in
_ROCK_THRESHOLD = 0.965       # ~3.5% of cells get a rock
_STIPPLE_STEP = 4             # stipple sampling stride in pixels (coarser = cheaper)

# Slice 6: DETAILED SETTLEMENTS & CASTLES. Villages become clusters of detailed houses that
# GROW with membership, with civic structure (well/plaza, granary, paths, a wall once big), and
# a ruler's seat becomes a HALL (leader) or a CASTLE (monarch/king/emperor). Every layout is
# derived from `terrain_noise` (the pure coordinate hash) — NEVER the sim RNG — and CACHED per
# settlement, rebuilt only when its membership/ruler/cell changes, so per-frame cost stays low.
_WALL_TONES = ((156, 128, 96), (172, 150, 118), (140, 116, 90), (178, 160, 128), (150, 132, 104))
_ROOF_TONES = ((122, 84, 66), (150, 98, 70), (98, 74, 56), (112, 100, 68), (132, 90, 72))
_DOOR = PALETTE["door"]
_WINDOW_LIT = PALETTE["window_lit"]   # a warm lit window (flickers gently, slice 9)
_WINDOW_DARK = PALETTE["window_dark"]
_CHIMNEY = PALETTE["chimney"]
_PATH = PALETTE["path"]           # dirt road between buildings
_PLAZA = PALETTE["plaza"]         # packed-earth market square at the centre
_WELL_STONE = PALETTE["well_stone"]
_WELL_WATER = PALETTE["well_water"]
_GRANARY_WALL = PALETTE["granary"]
_FENCE = PALETTE["fence"]         # the perimeter palisade of a large settlement
_CROP = PALETTE["crop"]           # crop rows in the farmland (muted to sit with the grass)
_CASTLE_STONE = PALETTE["castle_stone"]
_CASTLE_STONE_DK = PALETTE["castle_stone_dk"]
_GATE = PALETTE["gate"]
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

# Slice 9: VISUAL POLISH — light, palette, ambient life. The map zone becomes FULL-BLEED: a
# wilderness MARGIN ring (rougher tones, denser trees) around the playable grid and an east
# COAST across the margin, all baked into the cached terrain like slice 5 (coordinate hash,
# never sim RNG). One consistent SUN from the TOP-LEFT: ground shadows (translucent ellipse
# stamps, cached per size; static ones baked, only agents'/soldiers' move) fall to the
# bottom-right, and buildings get a lit left face / shaded right face. AMBIENT LIFE breathes
# on a renderer-local frame counter + terrain_noise: chimney smoke, wheat/crop sway, water
# shimmer, occasional birds, banner flutter, window flicker — each deliberately subtle. A
# very light warm DAYLIGHT grade (cached full-map tint) ties the scene together.
_MARGIN_CELLS = 3                 # wilderness ring width, in cells, around the playable grid
_SHADOW = PALETTE["shadow"]
_SHADOW_ALPHA = 54                # ground-shadow opacity (soft, not inky)
_SMOKE = PALETTE["smoke"]
_BIRD = PALETTE["bird"]
_GRADE_ALPHA = 12                 # the warm daylight grade — barely-there
_SMOKE_CYCLE = 48                 # frames for one chimney puff to rise and fade
_BIRD_WINDOW = 240                # frames per bird-flight window (~4s at 60fps)
_BIRD_CHANCE = 0.72               # noise threshold: most windows have NO birds ("every so often")
_FACE_SHADE = -14                 # how much a building's sun-away face darkens
_FACE_LIGHT = 16                  # how much its sun-side edge lightens

# Slice 10: DAY/NIGHT CYCLE. Time of day derives PURELY from the sim TURN (one day =
# _TURNS_PER_DAY turns): `time_of_day(turn)` -> a phase in [0,1) that drives EVERYTHING —
# the full-scene grade (keyframe-interpolated, never a hard switch), the directional dawn
# wash, star/torch/window-glow intensity, shadow fading and colour muting. No new state,
# no RNG: the same seeded run always has the same nights. The slice-9 static daylight
# grade becomes one keyframe of this cycle; the grade SURFACE stays cached and is only
# REFILLED when the interpolated tint changes (never rebuilt per frame).
_TURNS_PER_DAY = 24               # sim turns per full day/night cycle (tunable)
_PH_DAWN_END = 0.15               # dawn  [0.00, 0.15) — night brightens into gold
_PH_DAY_END = 0.55                # day   [0.15, 0.55) — the slice-9 neutral daylight
_PH_DUSK_END = 0.70               # dusk  [0.55, 0.70) — gold burns down into night
_NIGHT_GRADE_A = 118              # the deep-night grade alpha (day keeps _GRADE_ALPHA)
_DAWN_GRADE_A = 44                # the mid-dawn gold grade alpha
_DUSK_GRADE_A = 58                # the mid-dusk ember grade alpha
_DAWN_WASH_MAX_A = 46             # the directional sunrise wash at its mid-dawn peak
_STAR_COUNT = 320                 # hashed star candidates; only those on WATER are kept
_NIGHT_EPS = 0.02                 # below this night factor the lights pass is skipped
_SHADOW_NIGHT_KEEP = 0.30         # fraction of shadow alpha kept at deep night (sun-cast)
_NIGHT_MUTE_MAX = 0.75            # how far commoner/realm colours step into the dark

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
    overflowing their cell into illegibility. Slice 9 trims both bounds so PEOPLE read
    smaller than BUILDINGS — a town should dwarf its townsfolk.
    """
    r_min = max(2.0, cell * 0.21)
    r_max = max(r_min + 1.0, cell * 0.46)
    frac = max(0.0, min(1.0, math.sqrt(max(0.0, wealth) / _WEALTH_CEIL)))
    return int(round(r_min + frac * (r_max - r_min)))


def _cell_size(size: int) -> int:
    """Pixels per grid cell so grid + wilderness margin fit near _TARGET_PX (clamped).

    Slice 9: the map zone is the playable grid PLUS a _MARGIN_CELLS wilderness ring on every
    side (the full-bleed landscape), so the budget divides by size + 2*margin — the whole
    scene, not just the grid, stays near the target and the window height stays laptop-sane.
    """
    if size <= 0:
        return _MAX_CELL
    return max(_MIN_CELL, min(_MAX_CELL, _TARGET_PX // (size + 2 * _MARGIN_CELLS)))


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


# --- Slice 9: pure ambient-life helpers (frame-counter + hash driven, zero sim RNG) -------
def smoke_puffs(frame: int, sx: int, sy: int) -> list[tuple[int, int, int, int]]:
    """Chimney smoke for the hearth at (sx, sy): three staggered puffs -> (dx, dy, r, alpha).

    Each puff cycles over _SMOKE_CYCLE frames: it RISES (dy grows more negative), drifts
    gently with a sine wind, swells (r grows) and FADES (alpha -> 0). Phase is offset by the
    hearth's own coordinates so a street of chimneys never puffs in lockstep. Pure function of
    (frame, sx, sy) — deterministic, RNG-free, unit-testable.
    """
    out = []
    for i in range(3):
        ph = (frame * 0.9 + i * (_SMOKE_CYCLE / 3) + (sx * 13 + sy * 7) % _SMOKE_CYCLE) % _SMOKE_CYCLE
        drift = math.sin(frame * 0.05 + i * 2.1 + sx * 0.3) * 1.5 + ph * 0.12
        r = 1 + int(ph / 16)
        alpha = max(0, int(80 * (1.0 - ph / _SMOKE_CYCLE)))
        out.append((int(drift), -int(2 + ph * 0.55), r, alpha))
    return out


def ambient_birds(frame: int, map_px: int) -> list[tuple[float, float, float]]:
    """The birds crossing the sky this frame -> (x, y, wing_spread); usually an empty list.

    Time is cut into _BIRD_WINDOW-frame windows; a window hosts a bird only when its hash
    clears _BIRD_CHANCE (so flights are OCCASIONAL, not constant). Within its window the bird
    crosses the whole map (direction hashed), bobbing slightly, wings flapping via the spread
    oscillation. Pure function of (frame, map_px) — deterministic, RNG-free.
    """
    out = []
    window = frame // _BIRD_WINDOW
    p = (frame % _BIRD_WINDOW) / _BIRD_WINDOW
    for k in range(2):
        if terrain_noise(window, k, 77) <= _BIRD_CHANCE:
            continue
        ltr = terrain_noise(window, k, 78) > 0.5
        x = (p if ltr else 1.0 - p) * map_px
        y = map_px * (0.10 + 0.55 * terrain_noise(window, k, 79)) + math.sin(frame * 0.2 + k * 3.1) * 3
        spread = 2.5 + 1.5 * math.sin(frame * 0.45 + k)
        out.append((x, y, spread))
    return out


# --- Slice 10: pure day/night helpers (phase from the TURN; zero state, zero sim RNG) -----
def time_of_day(turn: float) -> float:
    """The day-cycle phase in [0, 1) for a (possibly fractional) turn number (pure).

    One full day = _TURNS_PER_DAY turns; the renderer passes a fractional turn while the
    inter-turn walk animation plays so the light GLIDES rather than stepping per turn.
    Periodic by construction: time_of_day(t) == time_of_day(t + _TURNS_PER_DAY), forever.
    """
    return (float(turn) / _TURNS_PER_DAY) % 1.0


def daylight_factor(phase: float) -> float:
    """How much SUN is on the scene at `phase`: 1.0 at midday, 0.0 at deep night (pure).

    Smoothstep ramps span the whole dawn and dusk bands, so the curve is continuous
    everywhere — including the midnight wrap (0 at phase->1⁻ and 0 at phase 0). Drives
    shadow strength, colour muting (via its complement, the night factor) and gates the
    daytime-only ambience (birds).
    """
    p = phase % 1.0
    if p < _PH_DAWN_END:
        return ease(p / _PH_DAWN_END)
    if p < _PH_DAY_END:
        return 1.0
    if p < _PH_DUSK_END:
        return 1.0 - ease((p - _PH_DAY_END) / (_PH_DUSK_END - _PH_DAY_END))
    return 0.0


def phase_name(phase: float) -> str:
    """The human name of the band `phase` falls in — dawn / day / dusk / night (pure)."""
    p = phase % 1.0
    if p < _PH_DAWN_END:
        return "dawn"
    if p < _PH_DAY_END:
        return "day"
    if p < _PH_DUSK_END:
        return "dusk"
    return "night"


# The grade keyframes: (phase, PALETTE key, alpha). Interpolation eases between neighbours,
# and the first/last keys are identical so the midnight wrap is seamless. Midday holds the
# slice-9 daylight tint exactly, so a noon frame looks the way slice 9 always did.
_TINT_KEYS: tuple[tuple[float, str, int], ...] = (
    (0.000, "night", _NIGHT_GRADE_A),
    (0.075, "dawn_gold", _DAWN_GRADE_A),      # mid-dawn: the golden hour
    (0.150, "daylight", _GRADE_ALPHA),
    (0.550, "daylight", _GRADE_ALPHA),
    (0.625, "dusk_ember", _DUSK_GRADE_A),     # mid-dusk: the sky burns
    (0.700, "night", _NIGHT_GRADE_A),
    (1.000, "night", _NIGHT_GRADE_A),
)


def phase_tint(phase: float) -> tuple[tuple[int, int, int], int]:
    """The full-scene grade at `phase` -> ((r, g, b), alpha), keyframe-interpolated (pure).

    Eased lerp between the _TINT_KEYS neighbours bracketing `phase`: warm gold at dawn,
    the neutral slice-9 daylight through the day, orange/pink at dusk, a deep cool
    blue-dark through the night — never a hard switch, seamless at the wrap.
    """
    p = phase % 1.0
    for i in range(len(_TINT_KEYS) - 1):
        p0, key0, a0 = _TINT_KEYS[i]
        p1, key1, a1 = _TINT_KEYS[i + 1]
        if p0 <= p <= p1:
            t = ease((p - p0) / (p1 - p0)) if p1 > p0 else 0.0
            return (lerp_color(PALETTE[key0], PALETTE[key1], t),
                    int(round(a0 + (a1 - a0) * t)))
    return PALETTE[_TINT_KEYS[-1][1]], _TINT_KEYS[-1][2]  # pragma: no cover - p is always bracketed


def dawn_wash_factor(phase: float) -> float:
    """The strength of the directional SUNRISE wash: a smooth bump peaking mid-dawn (pure).

    Zero everywhere outside the dawn band (and zero AT both band edges, so the wash fades
    in from night and fully dissolves into plain day — no pop).
    """
    p = phase % 1.0
    if p >= _PH_DAWN_END:
        return 0.0
    return math.sin(math.pi * (p / _PH_DAWN_END))


def night_mute(color: tuple[int, int, int], nf: float) -> tuple[int, int, int]:
    """Dim + gently desaturate a colour into the dark by night factor `nf` in [0,1] (pure).

    Identity at nf=0 (daytime colours untouched); at deep night, commoner figures and realm
    territory step back toward shadow so the lit windows and torches carry the scene.
    """
    t = _NIGHT_MUTE_MAX * max(0.0, min(1.0, nf))
    return lerp_color(color, _desat(_shade(color, -70), 0.45), t)


def star_field(map_px: int, n: int = _STAR_COUNT) -> list[tuple[int, int, int]]:
    """`n` hash-placed star candidates over the map zone -> (x, y, size) (pure, RNG-free).

    Candidates cover the WHOLE zone; the renderer keeps only those landing on WATER (the
    sea margin and the pond), so the top-down night sky appears as reflections and the
    land stays readable. Same map size -> the same stars, forever.
    """
    out = []
    for k in range(n):
        x = int(terrain_noise(k, 1, 57) * max(1, map_px))
        y = int(terrain_noise(k, 2, 57) * max(1, map_px))
        out.append((x, y, 1 if terrain_noise(k, 3, 57) < 0.85 else 2))
    return out


def _q8(a: float) -> int:
    """Quantize an alpha to a multiple of 8, clamped to [0, 255] (pure).

    The night-light halos are cached surfaces keyed by (radius, colour, alpha); quantizing
    the continuously-fading alpha keeps that cache small and bounded.
    """
    return max(0, min(255, int(a))) & ~7


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
        # Slice 9: full-bleed geometry + light/ambient caches (all renderer-local).
        self._margin_px = 0                                # the wilderness ring, in pixels
        self._map_px = 0                                   # full map zone = margin + grid + margin
        self._frame = 0                                    # ambient-life clock (per drawn frame)
        self._stamps: dict[tuple, Any] = {}                # cached shadow/smoke alpha stamps
        self._pond: tuple | None = None                    # baked pond geometry, for the shimmer
        self._grade: Any = None                            # cached full-scene grade overlay
        # Slice 10: DAY/NIGHT — the current frame's light, DERIVED inside _draw from the sim
        # turn (never stored in world_state). Defaults are midday so pre-frame calls (terrain
        # baking, tests poking single methods) behave exactly like slice 9.
        self._phase = 0.35                                 # day-cycle phase in [0,1)
        self._dl = 1.0                                     # daylight factor (1 midday, 0 night)
        self._nf = 0.0                                     # night factor = 1 - daylight
        self._grade_tint: tuple | None = None              # the tint the grade is filled with
        self._dawn_wash: Any = None                        # cached directional sunrise gradient
        self._stars: list[tuple[int, int, int]] = []       # star candidates that landed on water
        self._frame_lights: list[tuple] = []               # this frame's (kind, x, y, size) lights

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
        # Slice 9: the MAP zone is full-bleed — a wilderness margin rings the playable grid.
        self._margin_px = _MARGIN_CELLS * self._cell
        self._map_px = grid_px + 2 * self._margin_px
        self._screen = pygame.display.set_mode((self._map_px + _PANEL_W, self._map_px + _HUD_H))
        # Slice 5/9: bake the procedural landscape ONCE for this grid size (cached, blitted each
        # frame). Pure-hash texture/features — no RNG, so it never desyncs a seeded sim.
        self._terrain_bg = self._build_terrain()
        self._grade = self._build_grade()
        # Slice 10: the sunrise wash and the water-borne starfield are geometry-dependent,
        # so they are (re)built here with the terrain — pure hash, cached, never per frame.
        self._dawn_wash = self._build_dawn_wash()
        self._stars = self._build_stars()
        self._stamps = {}
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
        """Centre of grid cell (x, y) in pixels (offset past the slice-9 wilderness margin)."""
        c = self._cell
        m = self._margin_px
        return (m + x * c + c // 2, m + y * c + c // 2)

    def _draw(self, state: dict[str, Any], *, paused: bool = False,
              motion: tuple[dict[str, tuple], float] | None = None,
              battle: tuple[dict[str, Any], float] | None = None) -> None:
        """One frame. `motion`=(prev_positions, t) lerps agents mid-walk; `battle`=(scene,
        elapsed) overlays a cinematic beat. Both default off -> the slice-1..7 static frame."""
        screen = self._screen
        if screen is None:
            return
        cell = self._cell
        map_px = self._map_px
        # Slice 9: the ambient-life clock — one tick per DRAWN frame (renderer-local; the sim
        # never sees it), driving smoke/sway/shimmer/birds/flutter/flicker phases.
        self._frame = (self._frame + 1) % (1 << 20)
        # Slice 10: the DAY/NIGHT clock — the phase derives PURELY from the sim turn (made
        # fractional mid-walk so the light glides through the inter-turn animation rather
        # than stepping). Everything below reads these three derived values; nothing writes.
        turn_f = float(state.get("turn", 0))
        if motion is not None:
            turn_f += -1.0 + max(0.0, min(1.0, motion[1]))
        self._phase = time_of_day(turn_f)
        self._dl = daylight_factor(self._phase)
        self._nf = 1.0 - self._dl
        self._frame_lights = []           # towns register window/torch lights as they draw

        # Slice 5/9: the cached FULL-BLEED landscape (textured grass + wilderness fringe +
        # coast + pond + vignette) under everything, blitted not rebuilt. Fallback flat fill.
        screen.fill(_FRAME_OUTER)  # base for the HUD/panel gutters; map zone is overdrawn below
        if self._terrain_bg is not None:
            screen.blit(self._terrain_bg, (0, 0))
        else:
            screen.fill(_GRASS_BASE, (0, 0, map_px, map_px))
        self._draw_water_shimmer()  # slice 9: ripple glints on the baked pond + coast

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
            # Slice 10: figures step back into the dark at night (identity by day).
            color = night_mute(agent_color(getattr(agent, "personality", "")), self._nf)
            self._blit_shadow(cx, cy + r, r * 2.1, max(2, int(r * 0.8)))  # slice 9: grounded
            figure_top = self._draw_agent_figure(cx, cy, r, color)
            self._draw_role_marker(cx, figure_top, r, agent_role(agent.name, state))
            if agent.name in talkers:
                self._draw_speech_bubble(cx + r + 1, figure_top, r)

        # Slice 9/10: occasional birds (a daytime ambience — they roost as dusk falls), then
        # the full-scene grade. Slice 10 turns the static daylight tint into the day/night
        # cycle: the cached grade surface is REFILLED (never rebuilt) whenever the phase
        # tint moves, and blitted over the whole map zone (the HUD/panel stay ungraded).
        if self._dl > 0.35:
            self._draw_birds()
        if self._grade is not None:
            tint = phase_tint(self._phase)
            if tint != self._grade_tint:
                self._grade.fill((*tint[0], tint[1]))
                self._grade_tint = tint
            screen.blit(self._grade, (0, 0))
        # Slice 10: the directional sunrise wash — gold pouring in from the sun side.
        wash = dawn_wash_factor(self._phase)
        if wash > 0.01 and self._dawn_wash is not None:
            self._dawn_wash.set_alpha(int(255 * wash))
            screen.blit(self._dawn_wash, (0, 0))
        # Slice 10: the NIGHT LIGHTS pierce the grade — stars on the water, window glow,
        # torchlight — so towns twinkle in the dark instead of drowning in it.
        self._draw_night_lights()

        # Slice 8: the battle cinematic overlay (soldiers/dust/clash/fallen/banner) — drawn
        # ABOVE the grade since slice 10, so a night battle stays vivid and readable (the
        # clash flashes and the outcome banner never dim with the scene).
        if battle is not None:
            self._draw_battle_overlay(*battle)

        self._draw_hud(state, map_px, paused, in_battle=battle is not None)
        # Slice 3: the right sidebar — a state summary above a scrolling event feed. Drawn
        # last so it owns the right zone cleanly; a pure read of state, like everything else.
        self._draw_panel(state, map_px)
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

    # -- Slice 5/9: cached procedural landscape (built ONCE; pure hash, no sim RNG) --
    def _coast_x(self, y: int) -> int:
        """The shoreline x for pixel row y: the sea fills the outer EAST margin, meandering.

        A per-cell-row hash, linearly smoothed between rows, keeps the water strictly inside
        the wilderness margin (width 0.37..0.72 of it) — the playable grid never gets wet.
        Deterministic, so the baked coast and the per-frame shimmer agree forever.
        """
        m, cell, map_px = self._margin_px, self._cell, self._map_px
        if m <= 0:
            return map_px
        row, t = divmod(max(0, y), max(1, cell))
        a = terrain_noise(row, 9, 91)
        b = terrain_noise(row + 1, 9, 91)
        wiggle = (a + (b - a) * (t / max(1, cell))) - 0.5
        return int(map_px - m * (0.55 + 0.35 * wiggle))

    def _build_terrain(self) -> Any:
        """Bake the FULL-BLEED landscape ONCE: grass, wilderness fringe, coast, features, light.

        Slice 9 extends slice 5 edge to edge: the ground texture covers the whole map zone; the
        margin ring beyond the playable grid darkens/roughens into WILDERNESS with denser trees;
        an EAST COAST (sand -> shallow -> open water) meanders across the margin; every tree and
        rock casts a soft baked shadow from the top-left sun. Still 100% terrain_noise — it never
        calls `random`, so it cannot perturb the seeded sim. Cached per grid size, free to blit.
        """
        map_px, cell, size, m = self._map_px, self._cell, self._size, self._margin_px
        if map_px <= 0:
            return None
        grid_px = cell * size
        surf = pygame.Surface((map_px, map_px))
        surf.fill(_GRASS_BASE)

        # 1) GROUND texture over the WHOLE zone: per-tile value-noise + broad patches; tiles in
        #    the margin darken and roughen smoothly with distance past the playable edge.
        tile = max(3, cell // 2)
        for ty in range(0, map_px, tile):
            for tx in range(0, map_px, tile):
                fine = terrain_noise(tx // tile, ty // tile, 1) - 0.5
                patch = terrain_noise(tx // (tile * 5 + 1), ty // (tile * 5 + 1), 2) - 0.5
                shade = int(fine * 2 * _GRASS_VAR + patch * 2 * _GRASS_PATCH)
                d = max(m - tx, tx + tile - (m + grid_px), m - ty, ty + tile - (m + grid_px), 0)
                if d > 0:                          # the wilderness fringe: darker, rougher
                    w = min(1.0, d / max(1.0, 1.5 * cell))
                    shade += int((-10 + fine * 14) * w)
                surf.fill(_shade(_GRASS_BASE, shade), (tx, ty, tile, tile))

        # 2) STIPPLE grain across the whole zone (cheap; most samples place nothing).
        for sy in range(0, map_px, _STIPPLE_STEP):
            for sx in range(0, map_px, _STIPPLE_STEP):
                h = terrain_noise(sx, sy, 3)
                if h > 0.90:
                    surf.set_at((sx, sy), _GRASS_SPECK_HI)
                elif h < 0.07:
                    surf.set_at((sx, sy), _GRASS_SPECK_LO)

        # 3) The EAST COAST: for each pixel row, sand strip -> sunlit shallow -> open water.
        for y in range(map_px):
            wx = self._coast_x(y)
            if wx < map_px - 1:
                pygame.draw.line(surf, _WATER, (wx, y), (map_px - 1, y), 1)
                pygame.draw.line(surf, _WATER_SHALLOW, (wx, y), (min(wx + 3, map_px - 1), y), 1)
                if wx >= 4:
                    pygame.draw.line(surf, _SAND, (wx - 4, y), (wx - 1, y), 1)

        # 4) A POND in one deterministic off-centre spot inside the playable land.
        self._build_pond(surf, grid_px, cell)

        # 5) TREES and ROCKS over the EXTENDED cell range (margin included) — denser at the
        #    fringe so the forest visibly closes in; nothing planted in the sea.
        for cy in range(-_MARGIN_CELLS, size + _MARGIN_CELLS):
            for cx in range(-_MARGIN_CELLS, size + _MARGIN_CELLS):
                px = m + cx * cell + cell // 2
                py = m + cy * cell + cell // 2
                if not (0 <= px < map_px and 0 <= py < map_px) or px > self._coast_x(py) - cell:
                    continue
                fringe = not (0 <= cx < size and 0 <= cy < size)
                if terrain_noise(cx, cy, 4) > (_TREE_THRESHOLD_WILD if fringe else _TREE_THRESHOLD):
                    self._build_tree(surf, px, py, cell)
                elif terrain_noise(cx, cy, 5) > _ROCK_THRESHOLD:
                    self._build_rock(surf, px, py, cell)

        # 6) ATMOSPHERE: the (slice-9 softened) vignette and a thin border — a landscape that
        #    continues past the frame, not a stage floating in blackness.
        self._build_vignette(surf, map_px)
        pygame.draw.rect(surf, _FRAME_OUTER, (0, 0, map_px, map_px), 2)
        return surf

    def _build_grade(self) -> Any:
        """The cached full-scene GRADE surface (slice 10: refilled — never rebuilt — whenever
        the interpolated phase tint changes; it starts on the current phase's tint)."""
        if self._map_px <= 0:
            return None
        grade = pygame.Surface((self._map_px, self._map_px), pygame.SRCALPHA)
        tint = phase_tint(self._phase)
        grade.fill((*tint[0], tint[1]))
        self._grade_tint = tint
        return grade

    def _build_dawn_wash(self) -> Any:
        """The cached directional SUNRISE wash: a soft gold gradient strongest at the sun-side
        (west) edge, baked once; per frame only its surface alpha scales with dawn_wash_factor,
        so dawn pours in gradually and dissolves into plain day with zero rebuild cost."""
        if self._map_px <= 0:
            return None
        wash = pygame.Surface((self._map_px, self._map_px), pygame.SRCALPHA)
        strips = 28
        sw = max(1, self._map_px // strips + 1)
        gold = PALETTE["dawn_gold"]
        for i in range(strips):
            a = int(_DAWN_WASH_MAX_A * (1.0 - i / strips) ** 1.6)
            if a > 0:
                wash.fill((*gold, a), (i * sw, 0, sw, self._map_px))
        return wash

    def _build_stars(self) -> list[tuple[int, int, int]]:
        """Keep only the star_field candidates that land on WATER (the sea past the coast, or
        inside the pond): the top-down night sky appears as REFLECTIONS, so the land stays
        readable. Deterministic per map size — the renderer twinkles them per frame."""
        out: list[tuple[int, int, int]] = []
        for x, y, s in star_field(self._map_px):
            if self._margin_px > 0 and x > self._coast_x(y) + 5 and x < self._map_px - 2:
                out.append((x, y, s))
            elif self._pond is not None:
                pcx, pcy, rx, ry = self._pond
                if rx > 3 and ry > 2 and (((x - pcx) / (rx * 0.85)) ** 2 +
                                          ((y - pcy) / (ry * 0.85)) ** 2) < 1.0:
                    out.append((x, y, s))
        return out

    def _draw_night_lights(self) -> None:
        """Slice 10: the lights that pierce the dark — star reflections, window glow, torches.

        Drawn OVER the phase grade so at night they carry the scene. Every intensity scales
        with the night factor (they fade in through dusk and out through dawn — never pop),
        twinkle/flicker rides the frame clock through terrain_noise (zero sim RNG), and every
        halo is a cached soft stamp with a quantized alpha (a bounded cache, cheap blits).
        """
        nf = self._nf
        if nf <= _NIGHT_EPS:
            return
        screen, f = self._screen, self._frame
        for k, (x, y, s) in enumerate(self._stars):        # stars mirrored on the water
            tw = terrain_noise(f // 14, k, 66)
            a = _q8(nf * (95 + 140 * tw))
            if a <= 0:
                continue
            stamp = self._soft_stamp(s, PALETTE["starlight"], a)
            screen.blit(stamp, (x - stamp.get_width() // 2, y - stamp.get_height() // 2))
        for kind, x, y, s in self._frame_lights:
            if kind == "window":                           # the towns twinkle
                fl = 0.82 + 0.18 * terrain_noise(f // 3, x * 7 + y * 3, 67)
                halo = self._soft_stamp(max(3, s * 2), PALETTE["window_glow"],
                                        _q8(34 * nf * fl))
                screen.blit(halo, (x - halo.get_width() // 2, y - halo.get_height() // 2))
                core = self._soft_stamp(max(1, (s + 1) // 2), PALETTE["window_lit"],
                                        _q8(150 * nf * fl))
                screen.blit(core, (x - core.get_width() // 2, y - core.get_height() // 2))
            else:                                          # torchlight at the seats of power
                fl = 0.70 + 0.30 * terrain_noise(f // 2, x * 5 + y, 68)
                wob = int(round(terrain_noise(f // 2, x, 69) * 2 - 1))
                halo = self._soft_stamp(max(4, int(s * 1.6)), PALETTE["torch_flame"],
                                        _q8(46 * nf * fl))
                screen.blit(halo, (x - halo.get_width() // 2, y + wob - halo.get_height() // 2))
                pygame.draw.circle(screen, PALETTE["torch_core"], (x, y + wob), max(1, s // 4))
                pygame.draw.circle(screen, PALETTE["torch_flame"], (x, y + wob),
                                   max(2, s // 3), 1)

    def _build_pond(self, surf: Any, grid_px: int, cell: int) -> None:
        """A still pond in a fixed off-centre spot (deterministic; never the central arena).

        Slice 9: offset past the margin, with a sun-side highlight rim; its geometry is kept
        (renderer-local) so the per-frame water shimmer knows where to glint.
        """
        pcx = self._margin_px + int(grid_px * 0.22)
        pcy = self._margin_px + int(grid_px * 0.74)
        rx = max(cell, int(grid_px * 0.10))
        ry = max(cell, int(grid_px * 0.07))
        self._pond = (pcx, pcy, rx, ry)
        pygame.draw.ellipse(surf, _WATER, (pcx - rx, pcy - ry, 2 * rx, 2 * ry))
        pygame.draw.ellipse(surf, _WATER_HI, (pcx - rx, pcy - ry, 2 * rx, 2 * ry), 1)
        pygame.draw.ellipse(surf, _WATER_HI,
                            (pcx - rx // 2, pcy - ry // 2, rx, ry // 2), 1)  # sun-side rim

    def _build_tree(self, surf: Any, px: int, py: int, cell: int) -> None:
        """A simple tree: baked ground shadow, brown trunk, rounded canopy with a lit top-left."""
        r = max(2, int(cell * 0.42))
        trunk_w = max(1, r // 3)
        self._blit_shadow(px, py + r, r * 2.4, max(2, int(r * 0.7)), target=surf)  # slice 9
        pygame.draw.rect(surf, _TREE_TRUNK, (px - trunk_w // 2, py, trunk_w, r))
        pygame.draw.circle(surf, _TREE_CANOPY, (px, py), r)
        pygame.draw.circle(surf, _TREE_CANOPY_HI, (px - r // 4, py - r // 4), max(1, r // 2))
        pygame.draw.circle(surf, _shade(_TREE_CANOPY, -14), (px, py), r, 1)

    def _build_rock(self, surf: Any, px: int, py: int, cell: int) -> None:
        """A small boulder: baked ground shadow, grey blob, light top-left facet."""
        r = max(2, int(cell * 0.3))
        self._blit_shadow(px, py + r // 2 + 1, r * 2.2, max(2, int(r * 0.7)), target=surf)  # slice 9
        pygame.draw.circle(surf, _ROCK, (px, py), r)
        pygame.draw.circle(surf, _ROCK_HI, (px - r // 4, py - r // 4), max(1, r // 2))
        pygame.draw.circle(surf, _shade(_ROCK, -18), (px, py), r, 1)

    # -- Slice 9: light & shadow + ambient-life machinery -------------------
    def _shadow_stamp(self, w: int, h: int, alpha: int = _SHADOW_ALPHA) -> Any:
        """A cached translucent shadow ellipse (SRCALPHA) of size (w, h) — built once per
        (size, alpha); slice 10 passes quantized night-faded alphas so the cache stays small."""
        key = ("shadow", w, h, alpha)
        stamp = self._stamps.get(key)
        if stamp is None:
            stamp = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.ellipse(stamp, (*_SHADOW, alpha), (0, 0, w, h))
            self._stamps[key] = stamp
        return stamp

    def _blit_shadow(self, cx: float, cy: float, w: float, h: float, target: Any = None) -> None:
        """Ground a thing: a soft shadow ellipse under it, nudged toward the bottom-right
        (one consistent top-left sun for the whole scene). Cheap — one cached-stamp blit.

        Slice 10: shadows are SUN-cast, so the per-frame ones fade with the daylight factor
        (quantized alpha -> the stamp cache stays bounded). The terrain BAKE (target given)
        always stamps full daylight shadows — the baked landscape is phase-neutral and the
        night grade darkens it wholesale instead.
        """
        w, h = max(3, int(w)), max(2, int(h))
        alpha = _SHADOW_ALPHA if target is not None or self._dl >= 1.0 else \
            _q8(_SHADOW_ALPHA * (_SHADOW_NIGHT_KEEP + (1.0 - _SHADOW_NIGHT_KEEP) * self._dl))
        if alpha <= 0:
            return
        (target if target is not None else self._screen).blit(
            self._shadow_stamp(w, h, alpha),
            (int(cx - w / 2 + max(1, w // 10)), int(cy - h / 2 + 1)))

    def _soft_stamp(self, r: int, color: tuple, alpha: int) -> Any:
        """A cached translucent soft disc (smoke puffs etc.), keyed by radius/colour/alpha."""
        key = ("soft", r, color, alpha)
        stamp = self._stamps.get(key)
        if stamp is None:
            stamp = pygame.Surface((2 * r + 2, 2 * r + 2), pygame.SRCALPHA)
            pygame.draw.circle(stamp, (*color, alpha), (r + 1, r + 1), r)
            self._stamps[key] = stamp
        return stamp

    def _draw_water_shimmer(self) -> None:
        """Ambient ripple glints on the pond and along the coast — a few short light dashes
        whose positions re-hash every ~⅓s (terrain_noise on a coarse frame index; zero RNG)."""
        f = self._frame
        if self._pond is not None:
            pcx, pcy, rx, ry = self._pond
            for k in range(3):
                u = terrain_noise(f // 18, k, 71) - 0.5
                v = terrain_noise(f // 18, k, 72) - 0.5
                if (u * 1.2) ** 2 + (v * 1.1) ** 2 < 0.18:      # keep the glint inside the water
                    x, y = pcx + u * rx * 1.2, pcy + v * ry * 1.1
                    w = 3 + int(3 * terrain_noise(f // 18, k, 73))
                    pygame.draw.line(self._screen, _WATER_HI,
                                     (int(x - w), int(y)), (int(x + w), int(y)), 1)
        if self._margin_px > 0:
            for k in range(4):
                y = int(terrain_noise(f // 22, k, 74) * (self._map_px - 3)) + 1
                x0 = self._coast_x(y)
                x = x0 + 3 + terrain_noise(f // 22, k, 75) * max(0, self._map_px - x0 - 8)
                if x < self._map_px - 3:
                    w = 3 + int(3 * terrain_noise(f // 22, k, 76))
                    pygame.draw.line(self._screen, _WATER_HI,
                                     (int(x), y), (int(min(x + w, self._map_px - 3)), y), 1)

    def _draw_birds(self) -> None:
        """The occasional bird crossing the sky: tiny v-shapes, flapping — pure frame+hash."""
        for x, y, spread in ambient_birds(self._frame, self._map_px):
            xi, yi, s = int(x), int(y), max(1.5, spread)
            pygame.draw.line(self._screen, _BIRD, (int(xi - s), int(yi - s * 0.6)), (xi, yi), 1)
            pygame.draw.line(self._screen, _BIRD, (xi, yi), (int(xi + s), int(yi - s * 0.6)), 1)

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
        map_px = self._map_px
        pos_by_name = {
            a.name: a.position
            for a in state.get("agents", [])
            if getattr(a, "alive", True) and getattr(a, "position", None) is not None
        }
        overlay = pygame.Surface((map_px, map_px), pygame.SRCALPHA)
        # Slice 9: the crops SWAY — a gentle 1px lean whose phase drifts along the row (frame
        # counter + position; zero sim RNG). Recomputed per frame, it costs a sin per tuft.
        f = self._frame
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
                    sway = int(round(math.sin(f * 0.10 + x * 0.4 + fy * 0.15)))
                    pygame.draw.line(overlay, (*_CROP, _FARMLAND_ALPHA + 40),
                                     (x, fy), (x + sway, fy - max(2, cell // 6)), 1)
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
        # Slice 9: a highlight rim on the sun-side (left) points — gold catches the light.
        pygame.draw.line(self._screen, _shade(_CROWN, 55),
                         (cx - w, base_y - h), (cx - w // 2, base_y - h // 3), 1)
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
        """Food as a wheat stalk: a stem + grain strokes, SWAYING ±1px on the ambient clock."""
        if cell < _FOOD_GLYPH_MIN_CELL:
            pygame.draw.circle(self._screen, _FOOD, (cx, cy), max(1, cell // 6))
            return
        screen = self._screen
        s = max(3, cell // 3)
        # Slice 9: a gentle lean, phase-shifted by position so a field ripples, not marches.
        sway = int(round(math.sin(self._frame * 0.12 + cx * 0.35 + cy * 0.2)))
        base, top = cy + s // 2, cy - s
        pygame.draw.line(screen, _FOOD, (cx, base), (cx + sway, top), 1)  # the stalk
        for off in range(0, s + 1, max(2, s // 3)):                      # grain strokes up the stem
            yy = top + off
            pygame.draw.line(screen, _FOOD, (cx + sway, yy), (cx + sway - s // 2, yy - s // 3), 1)
            pygame.draw.line(screen, _FOOD, (cx + sway, yy), (cx + sway + s // 2, yy - s // 3), 1)

    def _draw_house(self, cx: int, cy: int, s: int) -> None:
        """A simple building: a square wall + a triangular roof, centred on (cx, cy)."""
        screen = self._screen
        half = max(2, s // 2)
        self._blit_shadow(cx, cy + half, s * 1.3, max(2, s // 3))  # slice 9: grounded
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
        map_px = self._map_px
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
        overlay = pygame.Surface((map_px, map_px), pygame.SRCALPHA)
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
                # Slice 10: realm banners MUTE at night (dimmer, fainter) so the window/torch
                # light carries the scene; by day this is exactly the slice-8 tint.
                tint = night_mute(tint, self._nf)
                fill, fill_a = tint, int(round(_REALM_FILL_ALPHA * (1.0 - 0.45 * self._nf)))
                edge, edge_a = _shade(tint, 45), int(round(_REALM_EDGE_ALPHA * (1.0 - 0.35 * self._nf)))
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
        pending_labels: list[tuple[str, int, int, int]] = []
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
            pending_labels.append((sid, cx, top_y, count))
        # Slice 9: labels drawn LAST on a translucent chip, clear of the buildings (above the
        # cluster), clamped on-map, and nudged upward when two settlements' labels collide.
        if self._font is not None and cell >= _SETTLEMENT_LABEL_MIN_CELL:
            placed: list[Any] = []
            for sid, cx, top_y, count in pending_labels:
                label = self._font.render(f"{sid}·{count}", True, _SETTLEMENT_LABEL)
                rect = label.get_rect(midbottom=(cx, top_y - 4))
                rect.left = max(3, min(rect.left, self._map_px - rect.width - 3))
                rect.top = max(3, rect.top)
                while any(rect.colliderect(p) for p in placed) and rect.top > 3:
                    rect.move_ip(0, -(rect.height + 2))         # the later label steps up and away
                chip = pygame.Surface((rect.width + 6, rect.height + 2), pygame.SRCALPHA)
                chip.fill((*_HUD_BG, 110))
                screen.blit(chip, (rect.left - 3, rect.top - 1))
                screen.blit(label, rect)
                placed.append(rect)
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

        # Slice 9: GROUND SHADOWS first (one top-left sun) so every structure sits on the earth
        # rather than floating; cached alpha-stamp blits, drawn under all the buildings.
        for b in plan["buildings"]:
            self._blit_shadow(cx + b["dx"], cy + b["dy"] + 1, b["w"] * 1.35, max(3, int(b["h"] * 0.36)))
        if plan["granary"]:
            g = plan["granary"]
            self._blit_shadow(cx + g["dx"], cy + g["dy"] + 1, g["scale"] * 1.4, max(3, int(g["scale"] * 0.4)))
        if plan["central"]["kind"]:
            sc = plan["central"]["scale"]
            self._blit_shadow(cx, cy + 2, sc * (3.4 if plan["central"]["kind"] == "castle" else 2.4),
                              max(4, int(sc * 0.5)))
        wl = plan["well"]
        self._blit_shadow(cx + wl["dx"], cy + wl["dy"] + 1, wl["scale"] * 1.3, max(2, wl["scale"] // 2))

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

        # Slice 9: CHIMNEY SMOKE — the lit (occupied) dwellings breathe; a hash picks which
        # hearths are burning, and smoke_puffs (pure) drives the rise/drift/fade per frame.
        # Slice 10: at night the smoke catches the MOONLIGHT — a paler, cooler tint (the
        # night factor is quantized so the soft-stamp cache stays bounded).
        smoke_c = lerp_color(_SMOKE, PALETTE["moon_smoke"], 0.7 * (round(self._nf * 8) / 8.0))
        for b in plan["buildings"]:
            if not b["lit"] or b["h"] < 7 or b["w"] < 9:
                continue
            if terrain_noise(b["dx"], b["dy"], 61) < 0.45:
                continue
            gx, gy = cx + b["dx"], cy + b["dy"]
            half = max(2, b["w"] // 2)
            roof_h = max(3, int(b["h"] * 0.7))
            cw = max(1, b["w"] // 6)
            chx = gx + half - cw - 1 + cw // 2                  # matches the drawn chimney
            chy = gy - b["h"] - roof_h // 2 - 1
            for dx, dy, r, alpha in smoke_puffs(self._frame, gx, gy):
                if alpha > 0:
                    stamp = self._soft_stamp(r, smoke_c, alpha)
                    screen.blit(stamp, (chx + dx - stamp.get_width() // 2,
                                        chy + dy - stamp.get_height() // 2))

        # Slice 10: TORCHLIGHT — a castle raises two gate torches, a large (palisaded) town
        # one by its well. Positions are plan-derived (stable per settlement); the flame and
        # halo are drawn later, over the grade, so they burn through the night.
        if self._nf > _NIGHT_EPS:
            central = plan["central"]
            if central["kind"] == "castle":
                off = max(4, int(central["scale"] * 1.15))
                self._frame_lights.append(("torch", cx - off, cy - 3, central["scale"]))
                self._frame_lights.append(("torch", cx + off, cy - 3, central["scale"]))
            elif plan["fence_r"]:
                self._frame_lights.append(("torch", cx + wl["dx"],
                                           cy + wl["dy"] - wl["scale"] - 2,
                                           max(4, wl["scale"])))

    def _draw_building(self, gx: int, gy: int, w: int, h: int, wall: tuple, roof: tuple,
                       hip: bool, lit: bool) -> None:
        """A detailed house at ground-centre (gx, gy): walls, gabled/hip roof + shading, door,
        windows and a chimney — a recognisable dwelling rather than a plain square."""
        s = self._screen
        half = max(2, w // 2)
        wall_rect = pygame.Rect(gx - half, gy - h, 2 * half, h)
        pygame.draw.rect(s, wall, wall_rect)
        # Slice 9: one consistent top-left sun — the right face falls into shade, the left
        # edge catches light (the roof's sun-away slope was already darkened; now unified).
        face_w = max(1, wall_rect.width // 3)
        pygame.draw.rect(s, _shade(wall, _FACE_SHADE),
                         (wall_rect.right - face_w, wall_rect.top, face_w, wall_rect.height))
        pygame.draw.line(s, _shade(wall, _FACE_LIGHT),
                         (wall_rect.left + 1, wall_rect.top + 1),
                         (wall_rect.left + 1, wall_rect.bottom - 2), 1)
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
            # Slice 9: a lit window FLICKERS — a subtle hearth-light brightness oscillation.
            win = (_shade(_WINDOW_LIT, int(5 * math.sin(self._frame * 0.35 + gx * 0.7)))
                   if lit else _WINDOW_DARK)
            wsz = max(2, w // 5)
            if lit and self._nf > _NIGHT_EPS:
                # Slice 10: at night the pane itself burns brighter, and each lit window
                # registers a GLOW light drawn later over the grade — towns twinkle.
                win = lerp_color(win, (255, 238, 168), 0.8 * self._nf)
                wy = gy - h + 2 + wsz // 2
                self._frame_lights.append(("window", gx - half + 2 + wsz // 2, wy, wsz))
                self._frame_lights.append(("window", gx + half - 2 - wsz + wsz // 2, wy, wsz))
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
        color = night_mute(color, self._nf)  # slice 10: banners mute in the dark
        kw = max(8, int(scale * 1.7))
        kh = max(12, int(scale * (2.8 if emperor else 2.3)))
        tw = max(5, int(scale * 0.95))
        th = int(kh * 0.82)
        # Slice 9: the top-left sun — the WEST tower stands lit, the EAST tower in shade.
        towers = ((cx - kw // 2 - tw // 2 + 1, _CASTLE_STONE),
                  (cx + kw // 2 + tw // 2 - 1, _CASTLE_STONE_DK))
        for sx, tone in towers:
            trect = pygame.Rect(sx - tw // 2, cy - th, tw, th)
            pygame.draw.rect(s, tone, trect)
            pygame.draw.rect(s, _OUTLINE, trect, 1)
            self._crenellate(trect.left, trect.top, tw, _shade(tone, 20))
            pygame.draw.polygon(s, color, [(trect.left - 1, trect.top - 2),  # conical roof in ruler colour
                                           (trect.right + 1, trect.top - 2),
                                           (sx, trect.top - tw)])
        keep = pygame.Rect(cx - kw // 2, cy - kh, kw, kh)                     # the central keep
        pygame.draw.rect(s, _CASTLE_STONE, keep)
        face_w = max(1, kw // 3)                                              # sun-away facet
        pygame.draw.rect(s, _shade(_CASTLE_STONE, _FACE_SHADE), (keep.right - face_w, keep.top, face_w, kh))
        pygame.draw.line(s, _shade(_CASTLE_STONE, _FACE_LIGHT),
                         (keep.left + 1, keep.top + 1), (keep.left + 1, keep.bottom - 2), 1)
        pygame.draw.rect(s, _OUTLINE, keep, 1)
        self._crenellate(keep.left, keep.top, kw, _CASTLE_STONE_DK)
        gw, gh = max(3, kw // 3), max(4, kh // 3)                             # gate
        pygame.draw.rect(s, _GATE, (cx - gw // 2, cy - gh, gw, gh))
        pygame.draw.arc(s, _GATE, (cx - gw // 2, cy - gh - gw // 2, gw, gw), 0, math.pi, 2)
        for wy in (cy - kh + kh // 3, cy - kh + 2 * kh // 3):                 # arrow-slit windows
            pygame.draw.rect(s, _WINDOW_DARK, (cx - 1, wy, 2, max(2, kh // 6)))
        pole_top = keep.top - max(4, scale)                                   # banner pole + pennant
        pygame.draw.line(s, _OUTLINE, (cx, keep.top), (cx, pole_top), 1)
        # Slice 9: the pennant FLUTTERS (tip riding a gentle frame-clock sine) and its upper
        # edge carries a highlight rim — royalty catches the light.
        fl = int(round(math.sin(self._frame * 0.3 + cx * 0.2) * 1.5))
        tip = (cx + scale, pole_top + 2 + fl)
        pygame.draw.polygon(s, color, [(cx, pole_top), tip, (cx, pole_top + 5)])
        pygame.draw.line(s, _shade(color, 55), (cx, pole_top), tip, 1)
        if emperor:
            tip2 = (cx + scale - 2, pole_top + 7 - fl)
            pygame.draw.polygon(s, _shade(color, 30), [(cx, pole_top + 5), tip2, (cx, pole_top + 10)])

    def _draw_hall(self, cx: int, cy: int, scale: int, color: tuple) -> None:
        """A trust-leader's HALL: a longhouse larger than a hut, with a big gabled roof, a double
        door and a small pennant in the leader's colour — between a common house and a castle."""
        s = self._screen
        color = night_mute(color, self._nf)  # slice 10: pennants mute in the dark
        w, h = max(10, int(scale * 1.9)), max(8, int(scale * 1.4))
        self._draw_building(cx, cy, w, h, _WALL_TONES[1], _ROOF_TONES[3], hip=False, lit=True)
        dw = max(3, w // 4)                                                   # a grander double door
        pygame.draw.rect(s, _DOOR, (cx - dw // 2, cy - max(4, h // 2), dw, max(4, h // 2)))
        pygame.draw.line(s, _shade(_DOOR, 30), (cx, cy - max(4, h // 2)), (cx, cy), 1)
        peak = cy - h - max(3, int(h * 0.7))
        pygame.draw.line(s, _OUTLINE, (cx, peak), (cx, peak - max(4, scale)), 1)  # pennant pole
        # Slice 9: the leader's pennant flutters gently too, with a lit upper edge.
        fl = int(round(math.sin(self._frame * 0.3 + cx * 0.2) * 1.5))
        top = peak - max(4, scale)
        tip = (cx + max(4, scale - 1), top + 2 + fl)
        pygame.draw.polygon(s, color, [(cx, top), tip, (cx, top + 5)])
        pygame.draw.line(s, _shade(color, 55), (cx, top), tip, 1)

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

    def _draw_hud(self, state: dict[str, Any], map_px: int, paused: bool,
                  in_battle: bool = False) -> None:
        """A one-line status strip under the map zone (turn / living / food / pause / battle)."""
        screen = self._screen
        pygame.draw.rect(screen, _HUD_BG, (0, map_px, map_px, _HUD_H))
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
        screen.blit(label, (8, map_px + (_HUD_H - label.get_height()) // 2))
        # Slice 10: the time-of-day readout — a tiny sun/moon dial at the far right of the
        # HUD (the phase hand rides its ring) with the phase name beside it. UI chrome: the
        # HUD/panel are never night-graded, so this stays readable at every phase.
        dial_x = map_px - 16
        tag = self._font.render(phase_name(self._phase), True, _STAT_LABEL)
        screen.blit(tag, (dial_x - 14 - tag.get_width(),
                          map_px + (_HUD_H - tag.get_height()) // 2))
        self._draw_phase_dial(dial_x, map_px + _HUD_H // 2)
        if in_battle:  # slice 8: a small indicator while a battle cinematic plays
            chip = self._font.render("BATTLE — any key skips", True, _BATTLE_CHIP)
            screen.blit(chip, (dial_x - 14 - tag.get_width() - 12 - chip.get_width(),
                               map_px + (_HUD_H - chip.get_height()) // 2))

    def _draw_phase_dial(self, cx: int, cy: int) -> None:
        """Slice 10: the sun/moon dial — a ring whose hand rides the full day cycle, around
        an icon that is a rayed SUN by day and cools into a crescent MOON by night."""
        s = self._screen
        r = 9
        pygame.draw.circle(s, _shade(_HUD_BG, 14), (cx, cy), r + 2)
        pygame.draw.circle(s, _PANEL_DIV, (cx, cy), r + 2, 1)
        body = lerp_color(PALETTE["crown"], PALETTE["starlight"], self._nf)
        pygame.draw.circle(s, body, (cx, cy), 4)
        if self._dl >= 0.5:                              # sun rays
            for i in range(8):
                ang = i * math.pi / 4
                s0, s1 = 5.5, 7.5
                pygame.draw.line(s, body,
                                 (cx + int(math.cos(ang) * s0), cy + int(math.sin(ang) * s0)),
                                 (cx + int(math.cos(ang) * s1), cy + int(math.sin(ang) * s1)), 1)
        else:                                            # crescent: bite the disc
            pygame.draw.circle(s, _shade(_HUD_BG, 14), (cx + 2, cy - 1), 3)
        hand = self._phase * 2 * math.pi - math.pi / 2   # midnight at the top, noon below
        pygame.draw.circle(s, _HUD_FG,
                           (cx + int(round(math.cos(hand) * (r + 2))),
                            cy + int(round(math.sin(hand) * (r + 2)))), 2)

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

    def _draw_panel(self, state: dict[str, Any], map_px: int) -> None:
        """Draw the right sidebar: a STATE summary above a scrolling EVENT feed (READ only).

        Top block = current-state counts (turn/living/food + settlements/kingdoms/empires
        where present); below a divider, the EVENTS feed shows the most recent log lines
        that fit, wrapped to the panel and lightly colour-coded by type, newest at the
        bottom. Pure read — it never touches world_state.
        """
        screen = self._screen
        font = self._font
        win_h = map_px + _HUD_H
        x0, pad = map_px, _PANEL_PAD
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
        self._blit_shadow(x, y + r, r * 2.0, max(2, int(r * 0.8)))  # slice 9: grounded like all figures
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
        map_px = self._map_px
        band_h = max(36, self._cell * 2)
        y0 = (map_px - band_h) // 2
        band = pygame.Surface((map_px, band_h), pygame.SRCALPHA)
        band.fill((*_BANNER_BG, int(210 * max(0.0, min(1.0, fade)))))
        self._screen.blit(band, (0, y0))
        pygame.draw.line(self._screen, _CROWN, (6, y0), (map_px - 6, y0), 1)
        pygame.draw.line(self._screen, _CROWN, (6, y0 + band_h), (map_px - 6, y0 + band_h), 1)
        font = self._big_font or self._font
        if font is None:
            return
        label = font.render(text, True, _BANNER_FG)
        if label.get_width() > map_px - 16 and self._font is not None:
            label = self._font.render(text, True, _BANNER_FG)   # long verdicts drop to the small face
        self._screen.blit(label, ((map_px - label.get_width()) // 2,
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
