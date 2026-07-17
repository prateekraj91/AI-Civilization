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
Slice 11 — CAMERA: PAN, ZOOM & LEVEL-OF-DETAIL — the world becomes EXPLORABLE.
ONE shared pure transform (world_to_screen / screen_to_world) maps grid coords to
screen pixels for EVERY map draw; the camera (a world-cell centre + an effective
cell size as the zoom, all renderer-local) pans with arrows/WASD/mouse-drag,
zooms on the cursor with the wheel (+/- on the centre), HOME refits the whole
world, and every move GLIDES. Cached layers (the baked landscape) exist at a
small ladder of QUANTIZED integer-cell zoom buckets, and per frame only the
visible sub-rect is blitted; dynamic layers simply redraw through the transform,
so a close-up re-renders MORE detail instead of upscaling pixels. Three detail
tiers with hysteresis: FAR reads as a grand-strategy map (dots, block towns,
realm banners dominant, ambience off), MID is the slice 1-10 look, CLOSE adds
villager name tags with everything larger. Off-screen work is culled, and a
battle that starts off-screen glides the camera to itself. The camera never
touches the sim: a panned/zoomed seeded run logs byte-identical events.

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

import collections
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
    # visual enhancements (slice 12)
    "rain": (140, 155, 170),           # rain streak
    "snow": (240, 245, 255),           # snowflake
    "fog": (200, 210, 220),            # dawn fog
    "cloud": (255, 255, 255),          # drifting clouds
    "cloud_shadow": (10, 12, 14),      # cloud shadows
    "trail": (110, 100, 80),           # footprint paths
    "foam": (230, 240, 255),           # wave foam
    "spring_grass": (70, 90, 50),      # seasonal grass tints
    "summer_grass": (44, 58, 40),
    "autumn_grass": (85, 75, 40),
    "winter_grass": (150, 155, 160),
    "heart": (220, 60, 90),            # emotions
    "sword": (170, 180, 190),
    "coin": (240, 190, 40),
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

# M4.12 ERAS: a settlement's building STYLE keys off its era (written to world_state["eras"] by the sim,
# read-only here). Neolithic = the earthy mud-and-thatch default; Bronze = richer timber + a forge; Iron
# = grey dressed STONE with a wall ring. Each style has its own wall/roof tone set so towns VISIBLY evolve
# by age. Unknown/absent era falls back to 'neolithic', so a non-era run renders exactly as before.
_ERA_STYLE = {"Bronze Age": "bronze", "Iron Age": "iron"}   # era NAME -> style key (else 'neolithic')
_ERA_WALL_TONES = {
    "neolithic": _WALL_TONES,
    "bronze": ((150, 120, 84), (168, 138, 96), (134, 106, 74), (176, 146, 104), (144, 116, 82)),
    "iron": ((150, 150, 156), (168, 168, 174), (132, 132, 140), (178, 178, 184), (142, 142, 150)),
}
_ERA_ROOF_TONES = {
    "neolithic": _ROOF_TONES,
    "bronze": ((120, 92, 60), (146, 110, 68), (100, 80, 52), (128, 100, 60), (134, 104, 66)),
    "iron": ((96, 100, 110), (118, 122, 132), (84, 88, 98), (126, 130, 140), (104, 108, 118)),
}
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

# Slice 11: CAMERA — PAN, ZOOM & LEVEL-OF-DETAIL. The window/viewport never changes size;
# the WORLD slides and scales behind it. The camera is a world-cell CENTRE plus an effective
# integer CELL size (the zoom), all renderer-local like _town_plans; one shared pure
# transform (world_to_screen/screen_to_world) maps world -> screen for every map draw, so
# no draw call does its own camera maths (the side PANEL and HUD are UI — never
# transformed). CACHED-SURFACE STRATEGY (the key engineering decision, stated explicitly):
# the baked landscape is rendered at a small geometric ladder of QUANTIZED integer cell
# sizes ("zoom buckets", ~6-8 across the range, LRU-capped); zoom steps TARGET buckets
# exactly, so a settled camera blits its bucket's visible sub-rect 1:1 (~free), and only
# while a zoom glide is mid-flight is the nearest bucket's sub-rect scaled by the small
# residual ratio (pygame.transform.scale of one viewport-sized region — cheap, and motion
# masks it). The bake is NEVER rebuilt per frame. Dynamic layers (towns, agents, food, war)
# redraw through the transform at the live cell size, so close-ups stay CRISP — a bigger
# cell re-renders more detail via the slice-4/6 cell thresholds, never a blurry upscale
# (town plans already cache per cell size and rebuilding one is cheap pure maths).
_ZOOM_OUT_MAX = 0.4               # lower zoom bound, as a fraction of the fit-whole-world cell
_ZOOM_IN_MAX = 3.0                # upper zoom bound (relative to fit), before the absolute cap
_CELL_FLOOR = 4                   # never draw cells smaller than this (strategy view stays readable)
_CELL_CEIL = 72                   # ...or bigger than this (close-up cap for tiny worlds)
_ZOOM_STEP_RATIO = 1.35           # the geometric ladder between zoom buckets (~6-8 buckets)
_TERRAIN_LRU = 3                  # non-base bucket landscapes kept baked at once (bounded memory)
_CAM_GLIDE = 0.22                 # per-frame ease toward the camera target — pan/zoom glides
_PAN_FRAC = 0.02                  # held-key pan speed: fraction of the viewport per frame
_LOD_FAR_MAX = 11.0               # at/below this cell size the map is the FAR strategy view
                                  # (11 keeps FAR reachable: every ladder's low bucket dips in)
_LOD_CLOSE_MIN = 26.0             # at/above this cell size the map is the CLOSE village view
_LOD_HYST = 1.0                   # hysteresis band so a tier never flickers at its boundary
_CAM_HOLD_KEYS = (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN,
                  pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_s)   # held-pan keys (polled)

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
                    ruler_color: tuple[int, int, int], emperor: bool, cell: int,
                    era_style: str = "neolithic") -> dict[str, Any]:
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

    # M4.12: era-specific building palette (earthy Neolithic -> timber Bronze -> stone Iron).
    wall_tones = _ERA_WALL_TONES.get(era_style, _WALL_TONES)
    roof_tones = _ERA_ROOF_TONES.get(era_style, _ROOF_TONES)

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
                "wall": _pick(wall_tones, nz(placed, 5)),
                "roof": _pick(roof_tones, nz(placed, 6)),
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
        # M4.12 ERAS: the visible age of the town — its build style, a Bronze+ forge, and Iron stone walls.
        "era_style": era_style,
        "forge": era_style in ("bronze", "iron"),     # a smithy appears once a town works metal
        "stone_wall": era_style == "iron",             # the palisade hardens into dressed stone
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


def weather_type(phase: float) -> str:
    """The current weather type based on the day cycle (pure)."""
    p = phase % 1.0
    if p < _PH_DAWN_END:
        return "fog"
    elif p > _PH_DUSK_END:
        return "snow"
    elif p > _PH_DAY_END:
        return "rain"
    return "clear"


def season_name(turn: float) -> str:
    """The current season based on turn count (pure). Assumes 96 turns per year."""
    year_progression = (float(turn) / 96.0) % 1.0
    if year_progression < 0.25:
        return "spring"
    elif year_progression < 0.50:
        return "summer"
    elif year_progression < 0.75:
        return "autumn"
    return "winter"


def ambient_clouds(frame: int, map_px: int) -> list[tuple[float, float, float, float]]:
    """Drifting clouds (x, y, w, h) based on the frame clock (pure)."""
    out = []
    # 3 cloud layers moving at different speeds
    for k in range(3):
        w = max(40, int(map_px * (0.15 + 0.1 * terrain_noise(k, 1, 81))))
        h = max(20, int(w * 0.6))
        speed = 0.1 + 0.05 * k
        x = (frame * speed + map_px * terrain_noise(k, 2, 82)) % (map_px + w) - w
        y = map_px * terrain_noise(k, 3, 83)
        out.append((x, y, w, h))
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


# --- Slice 11: pure camera / culling / LOD helpers (unit-testable, zero pygame state) -----
def world_to_screen(pos: tuple[float, float], cam: tuple[float, float, float],
                    view: tuple[int, int]) -> tuple[float, float]:
    """Map a WORLD point (grid-cell coords; cell (x, y)'s centre is (x+0.5, y+0.5)) to
    screen pixels under camera `cam` = (centre_x, centre_y, cell_px), viewport `view`.

    THE one shared transform: every map draw goes through this (via _to_px /
    _base_to_screen), so no draw call does its own camera maths. Pure.
    """
    cx, cy, cell = cam
    return ((pos[0] - cx) * cell + view[0] * 0.5,
            (pos[1] - cy) * cell + view[1] * 0.5)


def screen_to_world(px: tuple[float, float], cam: tuple[float, float, float],
                    view: tuple[int, int]) -> tuple[float, float]:
    """The exact inverse of world_to_screen (pure): screen pixels -> world grid coords."""
    cx, cy, cell = cam
    return ((px[0] - view[0] * 0.5) / cell + cx,
            (px[1] - view[1] * 0.5) / cell + cy)


def clamp_camera(cx: float, cy: float, cell: float, size: int,
                 view: tuple[int, int]) -> tuple[float, float]:
    """Clamp a camera CENTRE so the view never leaves the world (margin included). Pure.

    The world spans cells [-_MARGIN_CELLS, size + _MARGIN_CELLS] on each axis. An axis
    whose whole span FITS inside the viewport is simply centred (a zoomed-out world floats
    centred in the window rather than pinning to a corner).
    """
    lo, hi = -_MARGIN_CELLS, size + _MARGIN_CELLS
    out = []
    for c, v in ((cx, view[0]), (cy, view[1])):
        half = v / (2.0 * max(1e-9, cell))            # half the viewport, in world cells
        if hi - lo <= 2 * half:
            out.append(size / 2.0)
        else:
            out.append(max(lo + half, min(hi - half, c)))
    return out[0], out[1]


def zoom_buckets(base_cell: int) -> tuple[int, ...]:
    """The quantized zoom ladder for cached surfaces: a small geometric set of INTEGER
    cell sizes spanning ~0.4x..3x of the fit-whole-world cell (clamped to sane px). Pure.

    Integer cells keep every bucket's bake pixel-consistent with the live transform, and
    the ladder always contains the base cell itself, so the launch (fit) view blits 1:1.
    """
    base = max(_CELL_FLOOR, int(base_cell))
    lo = max(_CELL_FLOOR, int(round(base * _ZOOM_OUT_MAX)))
    hi = max(base, min(_CELL_CEIL, int(round(base * _ZOOM_IN_MAX))))
    out = {base}
    c = float(base)
    while int(round(c)) > lo:
        c /= _ZOOM_STEP_RATIO
        out.add(max(lo, int(round(c))))
    c = float(base)
    while int(round(c)) < hi:
        c *= _ZOOM_STEP_RATIO
        out.add(min(hi, int(round(c))))
    return tuple(sorted(out))


def visible_on_screen(x: float, y: float, margin: float, view_w: int, view_h: int) -> bool:
    """Is a screen-space point inside the viewport, padded by `margin` px? (pure culling —
    everything positional skips its draw when this is False, essential on big worlds)."""
    return -margin <= x <= view_w + margin and -margin <= y <= view_h + margin


def lod_tier(cell_px: float, prev: str = "mid") -> str:
    """The detail tier for an effective cell size — 'far' / 'mid' / 'close' (pure).

    Discrete tiers keyed off the zoom with a hysteresis band: a tier is only LEFT once the
    cell moves _LOD_HYST past the boundary it entered through, so wobbling the zoom right
    at a threshold can never flicker detail on and off frame to frame.
    """
    if prev == "far":
        if cell_px < _LOD_FAR_MAX + _LOD_HYST:
            return "far"
        prev = "mid"
    if prev == "close":
        if cell_px > _LOD_CLOSE_MIN - _LOD_HYST:
            return "close"
        prev = "mid"
    if cell_px <= _LOD_FAR_MAX - _LOD_HYST:
        return "far"
    if cell_px >= _LOD_CLOSE_MIN + _LOD_HYST:
        return "close"
    return "mid"


def _pond_geom(grid_px: int, cell: int, margin: int) -> tuple[int, int, int, int]:
    """The pond's (cx, cy, rx, ry) in the pixel space of `cell`/`margin` (pure) — one
    formula shared by every zoom bucket's bake and the base-space shimmer/star reads,
    so the glints always land on the same water at every zoom."""
    return (margin + int(grid_px * 0.22), margin + int(grid_px * 0.74),
            max(cell, int(grid_px * 0.10)), max(cell, int(grid_px * 0.07)))


class PygameRenderer:
    """Draws world_state to a Pygame window each turn (READ only); paces + handles input.

    Implements the same interface `run_simulation()` drives for the text dashboard:
    `.live()` owns the window for the run, `.update(state)` draws one frame after each
    resolved turn, and `.sink` swallows the plain per-turn text. The renderer NEVER
    advances the simulation — it only reads and draws what the sim produced.

    Controls: SPACE pauses/resumes, ESC or closing the window ends the run (raised as
    KeyboardInterrupt, which the launcher suppresses for a clean exit). Slice 11 camera:
    arrows/WASD (or left-mouse drag) pan, the wheel zooms on the cursor (+/- on the view
    centre), HOME refits the whole world; every move glides and none of it can touch the
    sim — a panned/zoomed seeded run logs byte-identical events.
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
        self._trails: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=6))
        self._current_season = "summer"                    # tracks season for terrain rebakes
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
        # Slice 11: CAMERA — all state renderer-local, like _town_plans. The camera is a
        # world-cell CENTRE plus an effective cell size (the zoom); the *_t fields are the
        # glide TARGETS that input moves, eased toward each drawn frame. `_cell` (the live
        # integer cell every size read uses) and `_cam_draw` (this frame's frozen shared
        # transform) derive from it in _update_camera. Nothing here can reach the sim.
        self._cell0 = _MAX_CELL            # the fit-whole-world cell — the zoom-1.0 reference
        self._cam_x = 0.0                  # view centre, world grid coords
        self._cam_y = 0.0
        self._cam_cell = float(_MAX_CELL)  # live zoom, as a float cell size (glides smoothly)
        self._cam_tx = 0.0                 # glide targets (pan / zoom land here)
        self._cam_ty = 0.0
        self._cam_tcell = float(_MAX_CELL)
        self._cam_draw = (0.0, 0.0, _MAX_CELL)     # this frame's (centre_x, centre_y, cell)
        self._zoom_buckets: tuple[int, ...] = ()   # the quantized integer-cell zoom ladder
        self._terrain_zoom: dict[int, Any] = {}    # bucket cell -> baked landscape (LRU-capped)
        self._lod = "mid"                  # current detail tier (far/mid/close, hysteresis)
        self._drag: tuple | None = None    # mouse-drag pan anchor (screen px, camera centre)

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
        self._cell = self._cell0 = _cell_size(size)
        grid_px = self._cell * max(1, size)
        # Slice 9: the MAP zone is full-bleed — a wilderness margin rings the playable grid.
        # Slice 11: this is also the fixed VIEWPORT — the window never grows with zoom.
        self._margin_px = _MARGIN_CELLS * self._cell
        self._map_px = grid_px + 2 * self._margin_px
        self._screen = pygame.display.set_mode((self._map_px + _PANEL_W, self._map_px + _HUD_H))
        # Slice 11: the camera opens on the FIT-WHOLE-WORLD view and owns this grid size —
        # the base-space pond geometry is fixed here so shimmer/stars agree at every zoom.
        self._pond = _pond_geom(grid_px, self._cell, self._margin_px)
        self._zoom_buckets = zoom_buckets(self._cell0)
        self._cam_x = self._cam_tx = size / 2.0
        self._cam_y = self._cam_ty = size / 2.0
        self._cam_cell = self._cam_tcell = float(self._cell0)
        self._cam_draw = (self._cam_x, self._cam_y, self._cell)
        self._lod = lod_tier(self._cell0, "mid")
        self._drag = None
        self._terrain_zoom = {}
        # Slice 5/9: bake the procedural landscape ONCE for this grid size (cached, blitted each
        # frame). Pure-hash texture/features — no RNG, so it never desyncs a seeded sim.
        self._terrain_bg = self._build_terrain(self._cell0)
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
        """Drain the OS event queue; camera events first, pause on SPACE, quit/ESC ends."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if self._handle_camera_event(event):
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused

    def _pump_cinema_events(self) -> bool:
        """Input during a cinematic: quit/ESC still ends the run; any NON-CAMERA key SKIPS
        the scene (slice 11: exploring — pan/zoom/home — must never swallow a battle)."""
        skip = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if self._handle_camera_event(event):
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt
                skip = True
        return skip

    # -- Slice 11: the camera (renderer-local; the sim is never consulted or touched) ------
    def _handle_camera_event(self, event: Any) -> bool:
        """Route one OS event to the camera; True when the camera consumed it.

        Wheel zoom is CURSOR-anchored (the world point under the mouse stays put); +/-
        zooms on the view centre; HOME refits the whole world; left-drag pans. Held pan
        keys (arrows/WASD) return True here so they never skip a cinematic — the actual
        panning is polled per frame in _update_camera for smooth held-key movement.
        """
        centre = (self._map_px // 2, self._map_px // 2)
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self._zoom_step(1, centre)
                return True
            if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self._zoom_step(-1, centre)
                return True
            if event.key == pygame.K_HOME:                 # refit the whole world
                self._cam_tx = self._cam_ty = self._size / 2.0
                self._cam_tcell = float(self._cell0)
                return True
            return event.key in _CAM_HOLD_KEYS
        if event.type == pygame.MOUSEWHEEL and event.y:
            anchor = centre
            with contextlib.suppress(Exception):
                mx, my = pygame.mouse.get_pos()
                if mx < self._map_px and my < self._map_px:
                    anchor = (mx, my)
            self._zoom_step(1 if event.y > 0 else -1, anchor)
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 \
                and event.pos[0] < self._map_px and event.pos[1] < self._map_px:
            self._drag = (event.pos, (self._cam_tx, self._cam_ty))
            return True
        if event.type == pygame.MOUSEMOTION and self._drag is not None:
            (px, py), (ox, oy) = self._drag
            c = max(1.0, self._cam_tcell)
            self._cam_tx = ox - (event.pos[0] - px) / c
            self._cam_ty = oy - (event.pos[1] - py) / c
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            had_drag, self._drag = self._drag is not None, None
            return had_drag
        return False

    def _zoom_step(self, direction: int, anchor: tuple[int, int]) -> None:
        """Step the zoom TARGET one bucket up/down, keeping the world point under `anchor`
        fixed on screen. Targets land exactly ON buckets, so a settled camera always blits
        its cached landscape 1:1 (the cheap path of the cached-surface strategy)."""
        buckets = self._zoom_buckets or (self._cell0,)
        t = self._cam_tcell
        if direction > 0:
            new = next((float(b) for b in buckets if b > t + 0.5), float(buckets[-1]))
        else:
            new = next((float(b) for b in reversed(buckets) if b < t - 0.5), float(buckets[0]))
        if new == t:
            return
        view = (self._map_px, self._map_px)
        wx, wy = screen_to_world(anchor, (self._cam_tx, self._cam_ty, t), view)
        self._cam_tcell = new
        self._cam_tx = wx - (anchor[0] - view[0] * 0.5) / new
        self._cam_ty = wy - (anchor[1] - view[1] * 0.5) / new
        self._cam_tx, self._cam_ty = clamp_camera(self._cam_tx, self._cam_ty, new,
                                                  self._size, view)

    def _update_camera(self) -> None:
        """Advance the camera one drawn frame: poll held pan keys, glide toward the targets,
        clamp to the world, and freeze this frame's shared transform + LOD tier.

        Runs at the top of every _draw — turn walks, pauses and cinematics all glide. It
        reads input and writes ONLY renderer-local camera fields; panning/zooming during a
        seeded run can never change the event log.
        """
        size, view = self._size, (self._map_px, self._map_px)
        if pygame.display.get_init():                 # held-key panning at constant screen speed
            keys = None
            with contextlib.suppress(Exception):
                keys = pygame.key.get_pressed()
            if keys:
                step = self._map_px * _PAN_FRAC / max(1.0, self._cam_tcell)
                dx = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - \
                     (keys[pygame.K_LEFT] or keys[pygame.K_a])
                dy = (keys[pygame.K_DOWN] or keys[pygame.K_s]) - \
                     (keys[pygame.K_UP] or keys[pygame.K_w])
                if dx or dy:
                    self._cam_tx += dx * step
                    self._cam_ty += dy * step
        self._cam_tx, self._cam_ty = clamp_camera(self._cam_tx, self._cam_ty,
                                                  self._cam_tcell, size, view)
        # Glide: ease a fraction of the remaining distance per frame; SNAP when close, so a
        # settled zoom sits exactly on its integer bucket (terrain then blits 1:1).
        for attr, target in (("_cam_x", self._cam_tx), ("_cam_y", self._cam_ty),
                             ("_cam_cell", self._cam_tcell)):
            cur = getattr(self, attr)
            nxt = cur + (target - cur) * _CAM_GLIDE
            setattr(self, attr, target if abs(target - nxt) < 0.02 else nxt)
        self._cell = max(_CELL_FLOOR, int(round(self._cam_cell)))
        self._cam_x, self._cam_y = clamp_camera(self._cam_x, self._cam_y,
                                                self._cell, size, view)
        self._cam_draw = (self._cam_x, self._cam_y, self._cell)
        self._lod = lod_tier(self._cell, self._lod)

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
        """Centre of grid cell (x, y) in SCREEN pixels — through the ONE shared camera
        transform (slice 11). At the fit-whole-world default this is exactly the slice-9
        mapping (margin offset + cell centre); panned/zoomed it slides and scales."""
        sx, sy = world_to_screen((x + 0.5, y + 0.5), self._cam_draw,
                                 (self._map_px, self._map_px))
        return int(round(sx)), int(round(sy))

    def _base_to_screen(self, bx: float, by: float) -> tuple[int, int]:
        """A BASE-space pixel (the fit-view pixel space the pond/coast/stars are stored in)
        through the same shared transform — so baked-geometry reads track the camera."""
        c0 = max(1, self._cell0)
        sx, sy = world_to_screen((bx / c0 - _MARGIN_CELLS, by / c0 - _MARGIN_CELLS),
                                 self._cam_draw, (self._map_px, self._map_px))
        return int(round(sx)), int(round(sy))

    def _draw(self, state: dict[str, Any], *, paused: bool = False,
              motion: tuple[dict[str, tuple], float] | None = None,
              battle: tuple[dict[str, Any], float] | None = None) -> None:
        """One frame. `motion`=(prev_positions, t) lerps agents mid-walk; `battle`=(scene,
        elapsed) overlays a cinematic beat. Both default off -> the slice-1..7 static frame."""
        screen = self._screen
        if screen is None:
            return
        map_px = self._map_px
        # Slice 9: the ambient-life clock — one tick per DRAWN frame (renderer-local; the sim
        # never sees it), driving smoke/sway/shimmer/birds/flutter/flicker phases.
        self._frame = (self._frame + 1) % (1 << 20)
        # Slice 11: advance the CAMERA first — glide toward its targets, clamp to the world,
        # and freeze this frame's shared transform, effective cell and LOD tier. Everything
        # below draws through them; nothing below does its own camera maths.
        self._update_camera()
        cell = self._cell
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

        # Update season and rebake terrain if needed
        new_season = season_name(turn_f)
        if new_season != self._current_season:
            self._current_season = new_season
            self._terrain_bg = self._build_terrain(self._cell0)
            self._terrain_zoom.clear()

        # Slice 5/9/11: the cached FULL-BLEED landscape (textured grass + wilderness fringe +
        # coast + pond + vignette) under everything — the camera blits only the VISIBLE
        # sub-rect of the nearest quantized zoom bucket's bake, never rebuilding per frame.
        screen.fill(_FRAME_OUTER)  # base for the HUD/panel gutters + beyond-the-world void
        if self._terrain_bg is not None:
            self._blit_terrain()
        else:
            grass_color = PALETTE.get(f"{self._current_season}_grass", _GRASS_BASE)
            screen.fill(grass_color, (0, 0, map_px, map_px))
        if self._lod != "far":       # slice 11: micro-shimmer is off in the strategy view
            self._draw_water_shimmer()  # slice 9: ripple glints on the baked pond + coast
            self._draw_coast_waves()

        # Slice 5: settled land looks CULTIVATED — a translucent tilled-dirt tint (with furrows)
        # under the slice-2 region. Dynamic (settlements come and go), but cheap. No-op if none.
        self._draw_settlement_ground(state)
        if self._dl > 0.35:
            self._draw_cloud_shadows()

        # Slice 2: SETTLEMENTS as soft translucent regions UNDER everything else, so a
        # settlement reads as a background "place" with food and agents sitting on top.
        # No-op (slice-1 behaviour) when there are no settlements in world_state.
        self._draw_settlements(state)

        # Slice 4: FOOD as a little wheat stalk (a stalk + a few grain strokes), still green
        # and at the same positions; a plain dot when cells are too small for wheat to read.
        # Slice 11: off-screen food is culled; the FAR strategy view keeps simple dots.
        for fx, fy in state.get("food", []):
            px, py = self._to_px(fx, fy)
            if not visible_on_screen(px, py, cell, map_px, map_px):
                continue
            if self._lod == "far":
                pygame.draw.circle(screen, _FOOD, (px, py), max(1, cell // 5))
            else:
                self._draw_wheat(px, py, cell)

        # Slice 4: AGENTS as little FIGURES (head + body) in their personality colour, scaled
        # by wealth; rulers wear a CROWN/STAR and anyone talking this turn gets a SPEECH
        # BUBBLE — so a stranger watching ONLY the map can read role and conversation.
        # Slice 11: off-screen agents are culled; at FAR zoom each agent collapses to a small
        # personality-colour dot (no figure/shadow/insignia/bubble — a strategy-map read);
        # at CLOSE zoom villagers additionally wear their NAME under the figure.
        talkers = talkers_this_turn(state.get("events", []) or [], state.get("turn", 0))
        
        self._update_trails(state, motion)
        self._draw_trails()
        
        for agent in state.get("agents", []):
            if not getattr(agent, "alive", True):
                continue
            pos = getattr(agent, "position", None)
            if not pos:
                continue
            cx, cy = self._agent_px(agent, motion)  # slice 8: mid-walk lerp when motion plays
            r = agent_radius(_wealth(agent), cell)
            if not visible_on_screen(cx, cy, r * 4 + cell, map_px, map_px):
                continue
            # Slice 10: figures step back into the dark at night (identity by day).
            color = night_mute(agent_color(getattr(agent, "personality", "")), self._nf)
            if self._lod == "far":
                dot = max(2, r)
                pygame.draw.circle(screen, _OUTLINE, (cx, cy), dot + 1)
                pygame.draw.circle(screen, color, (cx, cy), dot)
                continue
            self._blit_shadow(cx, cy + r, r * 2.1, max(2, int(r * 0.8)))  # slice 9: grounded
            figure_top = self._draw_agent_figure(cx, cy, r, color)
            self._draw_role_marker(cx, figure_top, r, agent_role(agent.name, state))
            if agent.name in talkers:
                if getattr(agent, "personality", "") == "friendliness":
                    self._draw_emotion_icon(cx, figure_top, r, "heart")
                elif getattr(agent, "personality", "") == "independence":
                    self._draw_emotion_icon(cx, figure_top, r, "sword")
                else:
                    self._draw_emotion_icon(cx, figure_top, r, "coin")
                self._draw_speech_bubble(cx + r + 1, figure_top, r)
            if self._lod == "close" and self._font is not None:
                self._draw_name_tag(agent.name, cx, cy + r + 2)

        # Slice 9/10: occasional birds (a daytime ambience — they roost as dusk falls), then
        # the full-scene grade. Slice 10 turns the static daylight tint into the day/night
        # cycle: the cached grade surface is REFILLED (never rebuilt) whenever the phase
        # tint moves, and blitted over the whole map zone (the HUD/panel stay ungraded).
        if self._dl > 0.35 and self._lod != "far":   # slice 11: no ambience on the strategy map
            self._draw_birds()
            self._draw_clouds()
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
            
        self._draw_weather()

        self._draw_hud(state, map_px, paused, in_battle=battle is not None)
        # Slice 3: the right sidebar — a state summary above a scrolling event feed. Drawn
        # last so it owns the right zone cleanly; a pure read of state, like everything else.
        self._draw_panel(state, map_px)
        
        self._draw_minimap(state)
        
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
    def _terrain_surface(self, cell_q: int) -> Any:
        """The baked landscape for zoom bucket `cell_q` (slice 11): the base bake for the
        fit cell, else baked LAZILY on first entry into that bucket and kept in a tiny LRU
        — rebuilt only when the camera crosses into an evicted bucket, NEVER per frame."""
        if cell_q == self._cell0:
            return self._terrain_bg
        surf = self._terrain_zoom.pop(cell_q, None)
        if surf is None:
            surf = self._build_terrain(cell_q)
        self._terrain_zoom[cell_q] = surf              # (re)insert newest-last: LRU order
        while len(self._terrain_zoom) > _TERRAIN_LRU:
            self._terrain_zoom.pop(next(iter(self._terrain_zoom)))
        return surf

    def _blit_terrain(self) -> None:
        """Blit the VISIBLE slice of the cached landscape through the camera (slice 11).

        The stated cached-surface strategy: bakes exist only at quantized integer-cell zoom
        buckets. When the live cell IS a bucket (any settled camera — zoom steps target
        buckets exactly) the visible sub-rect blits 1:1, pixel-perfect and ~free. While a
        zoom glide is between buckets, the nearest bucket's sub-rect is scaled by the small
        residual ratio with pygame.transform.scale — one viewport-sized cheap interim that
        the motion masks. The bake itself is never touched per frame.
        """
        screen, c, view = self._screen, self._cell, self._map_px
        buckets = self._zoom_buckets or (self._cell0,)
        surf = self._terrain_surface(min(buckets, key=lambda b: abs(b - c)))
        if surf is None:
            return
        x0, y0 = world_to_screen((-_MARGIN_CELLS, -_MARGIN_CELLS), self._cam_draw,
                                 (view, view))
        world_px = (self._size + 2 * _MARGIN_CELLS) * c
        vx0, vy0 = max(0, int(x0)), max(0, int(y0))
        vx1, vy1 = min(view, int(x0 + world_px)), min(view, int(y0 + world_px))
        if vx1 <= vx0 or vy1 <= vy0:
            return
        cq = surf.get_width() // (self._size + 2 * _MARGIN_CELLS)
        if c == cq:                                    # resting ON the bucket: plain blit
            screen.blit(surf, (vx0, vy0), (vx0 - int(x0), vy0 - int(y0),
                                           vx1 - vx0, vy1 - vy0))
            return
        s = cq / c                                     # live px -> bucket px
        sw, sh = surf.get_size()
        sx0 = max(0, min(sw - 1, int((vx0 - x0) * s)))
        sy0 = max(0, min(sh - 1, int((vy0 - y0) * s)))
        sx1 = max(sx0 + 1, min(sw, int(math.ceil((vx1 - x0) * s))))
        sy1 = max(sy0 + 1, min(sh, int(math.ceil((vy1 - y0) * s))))
        sub = surf.subsurface((sx0, sy0, sx1 - sx0, sy1 - sy0))
        screen.blit(pygame.transform.scale(sub, (vx1 - vx0, vy1 - vy0)), (vx0, vy0))

    def _coast_x(self, y: int, cell: int | None = None, m: int | None = None,
                 map_px: int | None = None) -> int:
        """The shoreline x for pixel row y: the sea fills the outer EAST margin, meandering.

        A per-cell-row hash, linearly smoothed between rows, keeps the water strictly inside
        the wilderness margin (width 0.37..0.72 of it) — the playable grid never gets wet.
        Deterministic, so the baked coast and the per-frame shimmer agree forever. Slice 11:
        the defaults describe the BASE (fit) geometry; a bucket bake passes its own — the
        row hash is keyed on the WORLD row, so every zoom draws the SAME shoreline.
        """
        cell = cell if cell is not None else self._cell0
        m = m if m is not None else self._margin_px
        map_px = map_px if map_px is not None else self._map_px
        if m <= 0:
            return map_px
        row, t = divmod(max(0, y), max(1, cell))
        a = terrain_noise(row, 9, 91)
        b = terrain_noise(row + 1, 9, 91)
        wiggle = (a + (b - a) * (t / max(1, cell))) - 0.5
        return int(map_px - m * (0.55 + 0.35 * wiggle))

    def _build_terrain(self, cell: int) -> Any:
        """Bake the FULL-BLEED landscape for one cell size: grass, fringe, coast, features.

        Slice 9 extends slice 5 edge to edge: the ground texture covers the whole map zone; the
        margin ring beyond the playable grid darkens/roughens into WILDERNESS with denser trees;
        an EAST COAST (sand -> shallow -> open water) meanders across the margin; every tree and
        rock casts a soft baked shadow from the top-left sun. Still 100% terrain_noise — it never
        calls `random`, so it cannot perturb the seeded sim. Slice 11 parameterizes the bake by
        `cell` so each quantized zoom bucket gets its own CRISP bake (cached; blitted per frame).
        """
        size = self._size
        m = _MARGIN_CELLS * cell
        map_px = cell * size + 2 * m
        if map_px <= 0:
            return None
        grid_px = cell * size
        surf = pygame.Surface((map_px, map_px))
        grass_color = PALETTE.get(f"{self._current_season}_grass", _GRASS_BASE)
        surf.fill(grass_color)

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
                surf.fill(_shade(grass_color, shade), (tx, ty, tile, tile))

        # 2) STIPPLE grain across the whole zone (cheap; most samples place nothing). The
        #    stride scales with the bucket's cell so a close-up bake stays affordable and
        #    keeps roughly the base view's speck density per world area.
        step = max(_STIPPLE_STEP, (_STIPPLE_STEP * cell) // max(1, self._cell0))
        for sy in range(0, map_px, step):
            for sx in range(0, map_px, step):
                h = terrain_noise(sx, sy, 3)
                if h > 0.90:
                    surf.set_at((sx, sy), _GRASS_SPECK_HI)
                elif h < 0.07:
                    surf.set_at((sx, sy), _GRASS_SPECK_LO)

        # 3) The EAST COAST: for each pixel row, sand strip -> sunlit shallow -> open water.
        for y in range(map_px):
            wx = self._coast_x(y, cell, m, map_px)
            if wx < map_px - 1:
                pygame.draw.line(surf, _WATER, (wx, y), (map_px - 1, y), 1)
                pygame.draw.line(surf, _WATER_SHALLOW, (wx, y), (min(wx + 3, map_px - 1), y), 1)
                if wx >= 4:
                    pygame.draw.line(surf, _SAND, (wx - 4, y), (wx - 1, y), 1)

        # 4) A POND in one deterministic off-centre spot inside the playable land.
        self._build_pond(surf, grid_px, cell, m)

        # 5) TREES and ROCKS over the EXTENDED cell range (margin included) — denser at the
        #    fringe so the forest visibly closes in; nothing planted in the sea.
        for cy in range(-_MARGIN_CELLS, size + _MARGIN_CELLS):
            for cx in range(-_MARGIN_CELLS, size + _MARGIN_CELLS):
                px = m + cx * cell + cell // 2
                py = m + cy * cell + cell // 2
                if not (0 <= px < map_px and 0 <= py < map_px) \
                        or px > self._coast_x(py, cell, m, map_px) - cell:
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
        screen, f, view = self._screen, self._frame, self._map_px
        k_px = self._cell / max(1, self._cell0)            # slice 11: base px -> live px
        for k, (x, y, s) in enumerate(self._stars):        # stars mirrored on the water
            sx, sy = self._base_to_screen(x, y)            # slice 11: ride the camera
            if not visible_on_screen(sx, sy, 6, view, view):
                continue
            tw = terrain_noise(f // 14, k, 66)
            a = _q8(nf * (95 + 140 * tw))
            if a <= 0:
                continue
            stamp = self._soft_stamp(max(1, int(round(s * k_px))), PALETTE["starlight"], a)
            screen.blit(stamp, (sx - stamp.get_width() // 2, sy - stamp.get_height() // 2))
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

    def _build_pond(self, surf: Any, grid_px: int, cell: int, margin: int) -> None:
        """A still pond in a fixed off-centre spot (deterministic; never the central arena).

        Slice 9: offset past the margin, with a sun-side highlight rim. Slice 11: the
        geometry comes from the shared pure _pond_geom formula (the base-space copy lives
        in self._pond, set once in _ensure_screen), so the per-frame shimmer and the star
        filter glint on the same water at every zoom bucket.
        """
        pcx, pcy, rx, ry = _pond_geom(grid_px, cell, margin)
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
        whose positions re-hash every ~⅓s (terrain_noise on a coarse frame index; zero RNG).

        Slice 11: glints are computed in the BASE pixel space the pond/coast are stored in,
        then ride the shared camera transform to screen; off-screen glints are culled and
        the dash length scales with the zoom.
        """
        f, view = self._frame, self._map_px
        k_px = self._cell / max(1, self._cell0)
        if self._pond is not None:
            pcx, pcy, rx, ry = self._pond
            for k in range(3):
                u = terrain_noise(f // 18, k, 71) - 0.5
                v = terrain_noise(f // 18, k, 72) - 0.5
                if (u * 1.2) ** 2 + (v * 1.1) ** 2 < 0.18:      # keep the glint inside the water
                    x, y = self._base_to_screen(pcx + u * rx * 1.2, pcy + v * ry * 1.1)
                    if not visible_on_screen(x, y, 12, view, view):
                        continue
                    w = max(2, int((3 + 3 * terrain_noise(f // 18, k, 73)) * k_px))
                    pygame.draw.line(self._screen, _WATER_HI, (x - w, y), (x + w, y), 1)
        if self._margin_px > 0:
            for k in range(4):
                by = int(terrain_noise(f // 22, k, 74) * (self._map_px - 3)) + 1
                x0 = self._coast_x(by)
                bx = x0 + 3 + terrain_noise(f // 22, k, 75) * max(0, self._map_px - x0 - 8)
                if bx >= self._map_px - 3:
                    continue
                bw = 3 + int(3 * terrain_noise(f // 22, k, 76))
                x1, y1 = self._base_to_screen(bx, by)
                x2, _y2 = self._base_to_screen(min(bx + bw, self._map_px - 3), by)
                if not visible_on_screen(x1, y1, 12, view, view):
                    continue
                pygame.draw.line(self._screen, _WATER_HI, (x1, y1), (max(x1 + 1, x2), y1), 1)

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
            if not visible_on_screen(cx, cy, rad + cell, map_px, map_px):
                continue                                         # slice 11: cull off-screen fields
            pygame.draw.circle(overlay, (*_FARMLAND, _FARMLAND_ALPHA), (cx, cy), rad)
            # Slice 11: furrow/tuft spacing also scales with the DISC so a sprawling (or
            # zoomed-in) settlement keeps a bounded tuft count per frame — the ploughed-
            # field read survives at any radius without O(radius²) per-frame cost.
            step = max(3, cell // 2, rad // 18)
            crop_dx = max(4, cell // 2, rad // 18)
            for fy in range(cy - rad + step, cy + rad, step):    # furrows, clipped to the disc
                half = int((rad * rad - (fy - cy) ** 2) ** 0.5)
                if half <= 1:
                    continue
                pygame.draw.line(overlay, (*_FARMLAND_FURROW, _FARMLAND_ALPHA),
                                 (cx - half, fy), (cx + half, fy), 1)
                if self._lod == "far":
                    continue                # slice 11: the strategy view drops the swaying tufts
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
            if not visible_on_screen(cx, cy, radius_px + cell * 3, map_px, map_px):
                continue                          # slice 11: whole town off-screen -> culled
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
                if self._lod == "far":            # slice 11: territory is THE strategic read
                    fill_a = min(150, int(fill_a * 1.8))
                    edge_a = min(220, int(edge_a * 1.3))
            else:
                fill, fill_a, edge, edge_a = (_SETTLEMENT_FILL, _SETTLEMENT_FILL_ALPHA,
                                              _SETTLEMENT_EDGE, _SETTLEMENT_EDGE_ALPHA)
            pygame.draw.circle(overlay, (*fill, fill_a), (cx, cy), radius_px)
            pygame.draw.circle(overlay, (*edge, edge_a), (cx, cy), radius_px, width=2)
            towns.append((sid, center, cx, cy, len(members), radius_px, owner))
        screen.blit(overlay, (0, 0))

        # Slice 6: each settlement is now a detailed, GROWING built place — a cached plan of
        # houses + civic structure + a ruler's HALL/CASTLE. Drawn under food/agents (which the
        # caller draws afterwards). Tiny cells fall back to the slice-4 simple-house glyphs.
        pending_labels: list[tuple[str, int, int, int]] = []
        for sid, center, cx, cy, count, radius_px, owner in towns:
            top_y = cy
            if self._lod == "far":
                # Slice 11 FAR: a settlement is a deliberate strategy-map MARK — a tiny block
                # cluster under a realm banner — not a shrunken village.
                self._draw_far_settlement(sid, cx, cy, count, owner, state)
                top_y = cy - cell * 3
            elif cell >= _TOWN_MIN_CELL:
                mon = monarchs.get(sid, {}).get("monarch")
                led = leaders.get(sid, {}).get("leader")
                if mon is not None:
                    kind, ruler, is_emp = "castle", mon, (mon in empires)
                elif led is not None:
                    kind, ruler, is_emp = "hall", led, False
                else:
                    kind, ruler, is_emp = None, None, False
                color = agent_color(personality_by_name.get(ruler, "")) if ruler else _DEFAULT_RULER
                # M4.12: the town's build style follows its ERA (read-only from world_state["eras"]).
                era_style = _ERA_STYLE.get(state.get("eras", {}).get(sid), "neolithic")
                key = (count, kind, ruler, is_emp, cell, era_style)
                cached = self._town_plans.get(sid)
                if cached is None or cached[0] != key:          # rebuild on membership/ruler/ERA change
                    cached = (key, build_town_plan(center, count, kind, color, is_emp, cell, era_style))
                    self._town_plans[sid] = cached
                self._draw_town(cx, cy, cached[1])
                top_y = cy - cached[1]["cluster_r"]
            else:
                self._draw_settlement_houses(cx, cy, radius_px, count, cell)
            pending_labels.append((sid, cx, top_y, count))
        # Slice 9: labels drawn LAST on a translucent chip, clear of the buildings (above the
        # cluster), clamped on-map, and nudged upward when two settlements' labels collide.
        # Slice 11: the FAR strategy view FORCES name+size labels (they are the map's text).
        if self._font is not None and (cell >= _SETTLEMENT_LABEL_MIN_CELL or self._lod == "far"):
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

    # -- Slice 11: the FAR (strategy-map) settlement mark + CLOSE name tags ---------
    def _draw_far_settlement(self, sid: str, cx: int, cy: int, count: int, owner: Any,
                             state: dict[str, Any]) -> None:
        """FAR zoom: a settlement drawn as a deliberate MAP MARK — a tiny block cluster
        (one to three blocks by population band) under a realm BANNER; a realm's capital
        (a kingdom's home, or a lone monarch's seat) flies a taller flag. The realm colour
        is the dominant strategic read; detail (houses, smoke, figures) is deliberately off.
        """
        s, c = self._screen, self._cell
        blk = max(3, int(c * 0.9))
        offs = ((0, 0), (-blk, blk // 3), (blk, blk // 2))[:1 + min(2, count // 4)]
        for i, (dx, dy) in enumerate(offs):
            rect = pygame.Rect(cx + dx - blk // 2, cy + dy - blk // 2, blk, max(2, (blk * 3) // 4))
            pygame.draw.rect(s, _pick(_WALL_TONES, 0.15 + 0.3 * i), rect)
            pygame.draw.rect(s, _OUTLINE, rect, 1)
        if owner is None:
            return
        color = night_mute(realm_color(owner), self._nf)
        capital = ((state.get("kingdoms", {}).get(owner, {}) or {}).get("home") == sid
                   or (state.get("monarchs", {}).get(sid, {}) or {}).get("monarch") == owner)
        pole_h = c * (3 if capital else 2)
        top = cy - blk - pole_h
        pygame.draw.line(s, _OUTLINE, (cx, cy - blk), (cx, top), 1)
        fl = int(round(math.sin(self._frame * 0.3 + cx * 0.2) * 1.5))   # the pennant flutters
        tip = (cx + max(4, int(c * 1.2)), top + 2 + fl)
        pygame.draw.polygon(s, color, [(cx, top), tip, (cx, top + max(4, c // 2) + 2)])
        pygame.draw.line(s, _shade(color, 55), (cx, top), tip, 1)

    def _draw_name_tag(self, name: str, cx: int, top_y: int) -> None:
        """CLOSE zoom only: the villager's name under the figure (a cached muted render) —
        the village-square read, where you can see WHO is talking to whom."""
        key = ("name", name)
        lab = self._stamps.get(key)
        if lab is None:
            lab = self._font.render(name, True, _STAT_LABEL)
            self._stamps[key] = lab
        self._screen.blit(lab, (cx - lab.get_width() // 2, top_y))

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
        # Slice 11: the zoom readout + the camera keys (the HUD is UI — never transformed).
        text += f"   zoom {self._cell / max(1, self._cell0):.1f}x"
        text += "   [spc]pause [wasd]pan [whl]zoom [home]fit"
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

        # REALMS Scoreboard
        y += 5
        pygame.draw.line(screen, _PANEL_DIV, (x0 + pad, y), (x0 + _PANEL_W - pad, y))
        y += 7
        screen.blit(font.render("REALMS", True, _PANEL_TITLE), (x0 + pad, y))
        y += line_h + 2
        
        realms = state.get("realms", {})
        if not realms:
            screen.blit(font.render("(none)", True, _FEED_DEFAULT), (x0 + pad, y))
            y += line_h
        else:
            sorted_realms = sorted(realms.keys(), key=lambda r: len(realms[r].get("settlements", [])), reverse=True)[:6]
            for r in sorted_realms:
                color = realm_color(r)
                screen.blit(font.render(r[:12], True, color), (x0 + pad, y))
                count = str(len(realms[r].get("settlements", [])))
                val = font.render(count, True, _STAT_VALUE)
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
        # Slice 11: a battle may start OFF-SCREEN on a big world — glide the camera to the
        # midpoint of the two hosts (pan only; the zoom is the viewer's) so it is watched,
        # not missed. Camera state only; the sim and its RNG are untouched.
        (ax, ay), (bx, by) = scene["att_pos"], scene["def_pos"]
        self._cam_tx, self._cam_ty = clamp_camera((ax + bx) / 2.0, (ay + by) / 2.0,
                                                  self._cam_tcell, self._size,
                                                  (self._map_px, self._map_px))
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

    # -- Slice 12: Visual Enhancements --------------------------------------
    def _draw_coast_waves(self) -> None:
        """Draw animated waves and foam along the eastern coast."""
        if self._pond is None:
            return
        grid_px = self._cell * max(1, self._size)
        coast_x = self._margin_px + int(grid_px * 0.96)
        y_start, y_end = self._margin_px, self._margin_px + grid_px
        
        # Draw a sine wave line for the wave crest
        points = []
        for y in range(y_start, y_end, max(2, self._cell // 2)):
            offset = math.sin(self._frame * 0.05 + y * 0.05) * (self._cell * 0.4)
            points.append((coast_x + int(offset), y))
        
        if len(points) > 1:
            pygame.draw.lines(self._screen, PALETTE["water_hi"], False, points, max(1, self._cell // 8))
        
        # Occasional foam bubbles
        for k in range(10):
            if terrain_noise(self._frame // 30, k, 91) > 0.6:
                fy = y_start + (terrain_noise(0, k, 92) * grid_px)
                fx = coast_x + math.sin(self._frame * 0.05 + fy * 0.05) * (self._cell * 0.4)
                fx += (terrain_noise(self._frame // 15, k, 93) - 0.5) * self._cell
                r = max(1, int(terrain_noise(0, k, 94) * (self._cell * 0.2)))
                pygame.draw.circle(self._screen, PALETTE["foam"], (int(fx), int(fy)), r)

    def _draw_cloud_shadows(self) -> None:
        """Draw dark translucent ellipses on the ground representing cloud shadows."""
        clouds = ambient_clouds(self._frame, self._map_px)
        for cx, cy, cw, ch in clouds:
            # Offset shadow slightly down and right
            rect = pygame.Rect(int(cx + cw * 0.2), int(cy + ch * 0.5), cw, ch)
            if rect.right > 0 and rect.bottom > 0 and rect.left < self._map_px and rect.top < self._map_px:
                stamp = pygame.Surface((cw, ch), pygame.SRCALPHA)
                pygame.draw.ellipse(stamp, (*PALETTE["cloud_shadow"], 30), (0, 0, cw, ch))
                self._screen.blit(stamp, rect.topleft)

    def _draw_clouds(self) -> None:
        """Draw translucent white ellipses moving across the map."""
        clouds = ambient_clouds(self._frame, self._map_px)
        for cx, cy, cw, ch in clouds:
            rect = pygame.Rect(int(cx), int(cy), cw, ch)
            if rect.right > 0 and rect.bottom > 0 and rect.left < self._map_px and rect.top < self._map_px:
                stamp = pygame.Surface((cw, ch), pygame.SRCALPHA)
                alpha = int(90 * self._dl) # fade out at night
                pygame.draw.ellipse(stamp, (*PALETTE["cloud"], alpha), (0, 0, cw, ch))
                self._screen.blit(stamp, rect.topleft)

    def _draw_weather(self) -> None:
        """Draw particles for rain, snow, or fog based on the current weather type."""
        weather = weather_type(self._phase)
        if weather == "clear":
            return
            
        screen = self._screen
        map_px = self._map_px
        
        if weather == "rain":
            # Rain streaks
            for k in range(50):
                x = (terrain_noise(0, k, 101) * map_px + self._frame * 10) % map_px
                y = (terrain_noise(0, k, 102) * map_px + self._frame * 20) % map_px
                pygame.draw.line(screen, (*PALETTE["rain"], 150), (int(x), int(y)), (int(x - 3), int(y + 8)), 1)
                
        elif weather == "snow":
            # Snowflakes
            for k in range(40):
                x = (terrain_noise(0, k, 103) * map_px + math.sin(self._frame * 0.05 + k) * 10) % map_px
                y = (terrain_noise(0, k, 104) * map_px + self._frame * 3) % map_px
                r = 1 if terrain_noise(0, k, 105) > 0.5 else 2
                pygame.draw.circle(screen, (*PALETTE["snow"], 180), (int(x), int(y)), r)
                
        elif weather == "fog":
            # Horizontal fog bands
            for k in range(3):
                y = map_px * 0.2 + (k * map_px * 0.3) + math.sin(self._frame * 0.02 + k) * 20
                h = max(20, int(map_px * 0.15))
                alpha = int(40 + 20 * math.sin(self._frame * 0.03 + k * 2))
                band = pygame.Surface((map_px, h), pygame.SRCALPHA)
                # Gradient-like effect using multiple lines
                for i in range(h):
                    a = int(alpha * math.sin(math.pi * (i / h)))
                    pygame.draw.line(band, (*PALETTE["fog"], a), (0, i), (map_px, i))
                screen.blit(band, (0, int(y)))

    def _update_trails(self, state: dict[str, Any], motion: tuple[dict[str, tuple], float] | None) -> None:
        """Record the current position of moving agents to form a trail."""
        if motion is None or self._lod == "far":
            return
        
        for agent in state.get("agents", []):
            if not getattr(agent, "alive", True) or not getattr(agent, "position", None):
                continue
            cx, cy = self._agent_px(agent, motion)
            # Only record if moved significantly
            if not self._trails[agent.name] or math.hypot(self._trails[agent.name][-1][0] - cx, self._trails[agent.name][-1][1] - cy) > self._cell * 0.5:
                self._trails[agent.name].append((cx, cy))

    def _draw_trails(self) -> None:
        """Draw fading footprint paths behind agents."""
        if self._lod == "far":
            return
        for name, path in self._trails.items():
            for i, (tx, ty) in enumerate(path):
                alpha = int(120 * (i + 1) / len(path))
                if alpha > 0 and visible_on_screen(tx, ty, self._cell, self._map_px, self._map_px):
                    stamp = pygame.Surface((4, 4), pygame.SRCALPHA)
                    pygame.draw.circle(stamp, (*PALETTE["trail"], alpha), (2, 2), max(1, self._cell // 8))
                    self._screen.blit(stamp, (int(tx - 2), int(ty - 2)))

    def _draw_emotion_icon(self, cx: int, top_y: int, r: int, state: str) -> None:
        """A small emotion icon floating above the agent's head (heart, sword, or coin)."""
        screen = self._screen
        y = top_y - r - 4
        # Add a gentle float bounce
        y += int(math.sin(self._frame * 0.1 + cx * 0.5) * 2)
        
        if state == "heart":
            color = PALETTE["heart"]
            pygame.draw.circle(screen, color, (cx - 2, y), 2)
            pygame.draw.circle(screen, color, (cx + 2, y), 2)
            pygame.draw.polygon(screen, color, [(cx - 4, y + 1), (cx + 4, y + 1), (cx, y + 5)])
        elif state == "sword":
            color = PALETTE["sword"]
            pygame.draw.line(screen, color, (cx - 3, y + 3), (cx + 3, y - 3), 2)
            pygame.draw.line(screen, color, (cx - 4, y + 2), (cx - 2, y + 4), 1)
        elif state == "coin":
            color = PALETTE["coin"]
            pygame.draw.circle(screen, color, (cx, y), 3)
            pygame.draw.circle(screen, _shade(color, -30), (cx, y), 3, 1)
            pygame.draw.line(screen, _shade(color, -30), (cx, y - 1), (cx, y + 1), 1)

    def _draw_minimap(self, state: dict[str, Any]) -> None:
        """A 120x120 minimap overlay in the bottom left corner."""
        mm_size = 120
        margin = 16
        mm_x = margin
        mm_y = self._map_px - mm_size - margin
        
        # Don't draw if the map is tiny anyway
        if self._size * self._cell < mm_size * 2:
            return
            
        mm = pygame.Surface((mm_size, mm_size), pygame.SRCALPHA)
        mm.fill((*_HUD_BG, 200))
        
        # Calculate scale
        grid_cells = self._size + 2 * _MARGIN_CELLS
        scale = mm_size / float(grid_cells)
        
        # 1. Base terrain
        pygame.draw.rect(mm, _desat(PALETTE["grass_base"], 0.2), (0, 0, mm_size, mm_size))
        # Draw water strip on the right
        water_x = int((_MARGIN_CELLS + self._size * 0.96) * scale)
        if water_x < mm_size:
            pygame.draw.rect(mm, PALETTE["water"], (water_x, 0, mm_size - water_x, mm_size))
            
        # 2. Territories
        for sid, rec in state.get("settlements", {}).items():
            if "center" not in rec: continue
            cx = int((rec["center"][0] + _MARGIN_CELLS) * scale)
            cy = int((rec["center"][1] + _MARGIN_CELLS) * scale)
            owner = settlement_realm(sid, state)
            if owner:
                c = realm_color(owner)
                r = max(3, int(settlement_radius_cells(rec["center"], []) * scale * 4)) # slightly exaggerated
                pygame.draw.circle(mm, (*c, 100), (cx, cy), r)
                
        # 3. Agents
        for agent in state.get("agents", []):
            if not getattr(agent, "alive", True) or not getattr(agent, "position", None):
                continue
            ax, ay = agent.position
            cx = int((ax + _MARGIN_CELLS) * scale)
            cy = int((ay + _MARGIN_CELLS) * scale)
            color = agent_color(getattr(agent, "personality", ""))
            mm.set_at((cx, cy), color)
            
        # 4. Viewport rectangle
        # Screen bounds in world coords
        view = (self._map_px, self._map_px)
        tl_wx, tl_wy = screen_to_world((0, 0), self._cam_draw, view)
        br_wx, br_wy = screen_to_world((self._map_px, self._map_px), self._cam_draw, view)
        
        vx = int((tl_wx + _MARGIN_CELLS) * scale)
        vy = int((tl_wy + _MARGIN_CELLS) * scale)
        vw = int((br_wx - tl_wx) * scale)
        vh = int((br_wy - tl_wy) * scale)
        
        pygame.draw.rect(mm, (255, 255, 255), (vx, vy, vw, vh), 1)
        pygame.draw.rect(mm, _OUTLINE, (0, 0, mm_size, mm_size), 2)
        
        self._screen.blit(mm, (mm_x, mm_y))
