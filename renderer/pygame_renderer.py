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

from renderer import director as _director

try:
    import pygame
except ImportError as exc:  # pragma: no cover - exercised only without pygame installed
    raise ImportError(
        "Pygame is required for the visual renderer. Install it with:  pip install pygame"
    ) from exc


# --- Palette (RGB) ---------------------------------------------------------
# Slice 9 / V4.4: ONE central PALETTE for every scene-defining colour, so the whole look is
# tunable in one place. V4.4 imposes PALETTE DISCIPLINE — a designed frame, not a swatch book.
#
#   BASE HUES (the whole world is toned toward these five families; vary VALUE, not hue):
#     1. GROUND  — warm olive-green   (~ 96° hue)   grass, farmland-crop, wheat
#     2. FOLIAGE — deeper forest-green (~110° hue)   trees, canopy
#     3. STONE/TIMBER — warm desaturated tan→grey    settlements, houses, castles, paths
#     4. WATER   — muted steel-teal    (~200° hue)   sea, pond, wells
#     5. SKY/NIGHT — cool blue         (~220° hue)   the day/night grade, stars, shadow
#
# Design intent (V4.4): the COMMONS are desaturated so they sit BACK in the frame (ordinary
# agents via _TRAIT_DESAT, wheat/crop/trees toned into the ground family, routine UI text kept
# grey); SATURATION IS RESERVED for what carries the story — rulers (crown + vivid robe),
# realm banners + territory edges, battle effects, the story banner and major-event feed lines
# — so they POP against the calm base. Adjacent layers keep VALUE separation (dark ground <
# mid buildings < bright-outlined agents) so silhouettes read even at far zoom. The module-
# level constant names the slices already use are kept, but each is DERIVED from PALETTE (UI
# chrome — panel/HUD/feed — frames the scene rather than being part of it).
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
    "farmland": (108, 84, 54),
    "farmland_furrow": (84, 62, 40),
    "crop": (104, 132, 78),            # V4.4: crop toned INTO the ground family (was brighter)
    "wheat": (156, 158, 104),          # V4.4: food sits into the grass — a muted gold-green
    # nature features (FOLIAGE base — desaturated so the forest reads as calm backdrop)
    "tree_trunk": (70, 52, 36),
    "tree_canopy": (42, 68, 44),       # V4.4: deeper, lower-chroma canopy
    "tree_canopy_hi": (54, 84, 54),
    "rock": (94, 96, 92),
    "rock_hi": (120, 122, 116),
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
    # slice 10 / V4.5: the day/night cycle. The night is a REAL dark again (V4.5 undoes V4.3's
    # flat brightness floor) — point lights, not a raised floor, now make the dark readable.
    "night": (12, 18, 46),             # V4.5: deep cool blue-dark — a town becomes warm pools in it
    "dawn_gold": (255, 196, 112),      # the dawn grade + the directional sunrise wash
    "dusk_ember": (255, 134, 88),      # the burning orange/pink dusk grade
    "starlight": (222, 230, 250),      # stars mirrored on the night water
    # V4.5 POINT-LIGHT palette — warm emitters that cast additive radial pools onto the scene.
    "window_glow": (255, 190, 104),    # lit-window pool (warm amber)
    "torch_flame": (255, 158, 60),     # gate-torch pool (deeper orange)
    "torch_core": (255, 236, 170),     # the white-hot torch heart
    "hearth_glow": (255, 168, 84),     # a town hearth's warm pool over the plaza at night
    "forge_glow": (255, 116, 44),      # a metallurgy town's forge — a hotter, redder pool
    "forge_core": (255, 226, 150),     # the forge's white-hot heart (pulsing)
    "clash_light": (255, 240, 210),    # a battle clash-flash cast as a bright cold-white pool
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
_TRAIT_DESAT = 0.40                    # V4.4: commoner figures step further toward the earth tones
                                       # (saturation is reserved for rulers — see agent_color vivid)

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
# V4.17 RANK SILHOUETTE — rank as SHAPE, so it survives being small, dim or half-occluded:
# rank -> (height scale, shoulder scale, robe kind). The ladder is `agent_role`'s, and the steps
# are deliberately coarse: an emperor is a third taller than a commoner, which reads instantly,
# where a 5% difference would only read side by side. `None` (a commoner) is the unscaled figure.
_RANK_SILHOUETTE: dict[str, tuple[float, float, str | None]] = {
    "emperor": (1.38, 1.34, "cloak"),    # tallest, broadest, full cloak past the feet
    "king":    (1.26, 1.22, "cloak"),
    "monarch": (1.16, 1.12, "mantle"),   # a sovereign crown, but a local one — a short cape
    "lord":    (1.09, 1.06, "mantle"),
    "leader":  (1.05, 1.00, None),       # raised by consent, not by title: no cloth, just a star
}
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
# V4.9: the OUT-OF-BOUNDS void — the world reads as CONTINUING into open water rather than sitting
# as a diamond island on flat background. A radial ocean (deep sea at centre -> near-black at the
# corners) fills behind the transparent-void terrain bake; a soft vignette on top kills the hard
# canvas edge. Both are cached per window size and cost two blits a frame.
_VOID_OCEAN = (30, 66, 92)    # deep open sea just beyond the coast — clearly WATER, blue-teal
_VOID_EDGE = (18, 42, 60)     # the far deep water at the frame corners (still reads as sea, not black)
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
# V4.7: ISO BUILDING VOLUMES. Structures stop being flat forms on the tilt and become 3D VOLUMES
# hung off the iso ground plane — two visible wall faces (a sun-lit one + a shaded one per the same
# top-left sun the whole scene uses) and roof planes, standing at a HEIGHT so a settlement's RANK
# reads by silhouette alone from across the map. Heights (z) and footprints are in WORLD CELLS (z =
# cells straight up the height axis, projected by _ISO_ZH); ERA lifts the whole town's style from low
# Neolithic huts -> Bronze timber -> tall Iron stone. Every quantity is deterministic (nz/terrain_noise,
# never the sim RNG) and baked into the cached plan, so the volume set rebuilds only on the same
# membership/ruler/zoom-bucket change the slice-6 plan already keyed on.
_ERA_HUT_Z = {"neolithic": 0.62, "bronze": 0.82, "iron": 1.04}   # a common dwelling's wall height
_HUT_FOOT = 0.80                 # a hut's ground footprint edge, in world cells (square-ish)
_HALL_Z = 1.75                   # a trust-leader's HALL stands well above the huts
_HALL_FOOT = 1.55
_KEEP_Z = 3.1                    # a monarch's KEEP genuinely TOWERS — a capital reads as a capital
_KEEP_Z_EMPEROR = 3.9            # an emperor's seat rises higher still
_KEEP_FOOT = 1.75
_TOWER_Z = 2.4                   # flanking corner towers — taller than any dwelling
_TOWER_FOOT = 0.72
_GRANARY_Z = 1.55               # a tall narrow store (distinct volume, conical roof)
_GRANARY_FOOT = 0.82
_WELL_Z = 0.5                   # a low stone ring with its little roof on posts
_POST_Z = 0.9                   # palisade post height — a ring of real posts, not a flat line
# V4-fix: PALISADE reads as a defensive WALL — a ring of thicker posts joined by rails, in an
# era-appropriate wood/stone tone (never near-white), not a scatter of pale shards across the town.
_PAL_WALL_Z = 0.72              # the connecting rail/wall panel height (world cells)
_PAL_POST_Z = 1.0              # the posts rise a little above the rail
_PAL_POST_FOOT = 0.34          # a post's footprint — thicker than the old 0.24 shards
_PAL_WOOD = (108, 84, 58)      # a timber palisade (Neolithic/Bronze) — warm brown
_PAL_STONE = (120, 122, 128)   # an Iron-age dressed-stone wall — muted grey, NOT near-white
_FACE_LIT = 18                  # V4.7: how much a volume's SUN-facing (south-west) wall lightens
_FACE_DARK = -30                # ...and how much its shaded (south-east) wall darkens (deeper than 2D)
_ROOF_LIT = 16                  # a sun-facing roof plane lightens
_ROOF_DARK = -26                # a sun-away roof plane darkens
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
# V4.17 (5.2): how far a villager's personality colour is pulled toward their sovereign's banner.
# Tuned as the point where a cluster reads as one realm at a glance while the personality palette
# is still legible within it. A RULER is pulled harder — he does not merely belong to the realm,
# he IS it, and should read as its banner made flesh.
_ALLEGIANCE_MIX = 0.5
_ALLEGIANCE_MIX_RULER = 0.72
# V4.17 (5.3): a crown is an OBJECT, not a badge. When a crowned head dies its crown falls to the
# ground where he stood and LIES there — visibly vacant — until someone takes the seat, at which
# point it is claimed and fades. A vacant crown on the grass is the clearest possible statement
# that a throne is empty and the succession is unresolved.
_CROWN_LIE_SECS = 26.0        # how long a fallen crown lies before the world forgets it
_CROWN_FADE = 1.2             # fade out over this long, whether claimed or merely forgotten
_CROWN_FLAT = 0.42            # vertical squash — a crown lying on its side, not standing up
_CROWN_GLINT_PERIOD = 62      # frames per glint cycle: a slow catch of light, not a blink
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
_NIGHT_GRADE_A = 150              # V4.5: the night is DARK again (undoes V4.3's flat floor of 82)
                                  # — point lights, not a raised floor, now make the dark readable
_DAWN_GRADE_A = 44                # the mid-dawn gold grade alpha
_DUSK_GRADE_A = 58                # the mid-dusk ember grade alpha
_DAWN_WASH_MAX_A = 46             # the directional sunrise wash at its mid-dawn peak
_STAR_COUNT = 320                 # hashed star candidates; only those on WATER are kept
_NIGHT_EPS = 0.02                 # below this night factor the lights pass is skipped
_SHADOW_NIGHT_KEEP = 0.30         # fraction of shadow alpha kept at deep night (sun-cast)
_NIGHT_MUTE_MAX = 0.75            # how far commoner/realm colours step into the dark

# V4.5: POINT LIGHTS. Each emitter (lit window, gate torch, town hearth, metallurgy forge,
# battle clash flash) casts a cached RADIAL-GRADIENT pool blitted ADDITIVELY, so overlapping
# pools bloom and the cost stays flat (one cached surface per radius/colour/quantized-intensity,
# reused every frame). Intensity ties to the night factor (invisible midday, blooming dusk,
# dominant night, dissolving dawn); a per-emitter flicker rides the frame clock via terrain_noise.
_LIGHT_INTENSITY_STEPS = 12       # how finely per-frame light intensity is quantized (cache bound)
_LIGHT_RADIUS_Q = 3               # radii are snapped to this multiple (bounds the stamp cache)
_LIGHT_FALLOFF = 2.0              # radial brightness falloff exponent (higher = tighter pool)
_LIGHT_WINDOW = 3.4               # pool radius as a multiple of the emitter's glyph size...
_LIGHT_TORCH = 4.2               # ...per light kind (torches/hearths/forges throw farther)
_LIGHT_HEARTH = 5.0
_LIGHT_FORGE = 4.6
# V4.17 (5.3): the fallen crown's GLEAM — a halo only, with no bright core and no flame ring. A
# torch drew a hot white core that became the dominant shape, so a vacant crown read as "a gleam
# with a crown in it" rather than "a crown, gleaming". The halo exists solely to keep the gold off
# the night grass; when in doubt it errs DIM, because the silhouette is the thing being read.
_LIGHT_GLEAM = 2.6
_LIGHT_GLEAM_STRENGTH = 0.34
_LIGHT_MAX_A = 200                # cap a single pool's centre additive brightness
# V4-fix: LIGHT BLOWOUT. All the town's additive pools accumulate on ONE offscreen layer, then that
# layer is CLAMPED (per-channel MIN) to a warm cap before compositing — so however many windows/hearth/
# forge pools overlap on a wall face, the added light rises toward WARM AMBER and never saturates to
# white. Blue is capped lowest so the accumulation stays warm; the glow halo below the cap is untouched.
_LIGHT_ACC_CAP = (150, 108, 60)   # max additive light per pixel (R, G, B) — warm, B kept well under R
_LIGHT_DS = 2                     # downscale factor for the light-accumulation layer (perf; pools are smooth)

# V4.9 JUICE — subtle motion that reads as production value COLLECTIVELY, never a carnival. Every
# effect is frame-clock / hash driven (zero sim RNG) and NEUTRAL (byte-identical) when not firing:
# a screen SHAKE on clashes, DUST on a building rising / a COLLAPSE puff when one falls, a gentle
# zoom-PUNCH as the story banner fires, agent squash-&-stretch + arrival hop, rippling flag cloth,
# an impact FLASH + brief slow-motion on the decisive blow, and coins/embers over a market/forge.
_SHAKE_DECAY = 0.80               # clash screen-shake amplitude falloff per drawn frame (short, sharp)
_SHAKE_MAX = 8.0                  # cap the clash shake amplitude (screen px)
_PUNCH_DUR = 0.42                 # story-banner zoom-punch duration (s), ease in -> ease back
_PUNCH_MAX = 0.028                # peak extra zoom of the punch (a gentle few percent)
_FLASH_DECAY = 0.78               # decisive-blow impact-flash falloff per frame
_PUFF_LIFE = 30                   # frames a building rise/collapse puff lives

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
_ZOOM_OUT_MAX = 0.4               # widest terrain-bake bucket, as a fraction of the fit cell (the
                                  # interactive zoom-OUT floor is the fit cell itself, see _zoom_lo)
_ZOOM_IN_MAX = 6.0                # upper zoom bound (relative to fit), before the absolute cap.
                                  # V4.6: the iso FIT cell is half the window cell, so the range
                                  # doubles to keep the same absolute close-up (still _CELL_CEIL-capped)
_CELL_FLOOR = 4                   # never draw cells smaller than this (strategy view stays readable)
_CELL_CEIL = 72                   # ...or bigger than this (close-up cap for tiny worlds)
_ZOOM_STEP_RATIO = 1.35           # the geometric ladder between terrain-bake buckets (~6-8 buckets)
_ZOOM_STEP = 1.1                  # V4.9: small, consistent MULTIPLICATIVE zoom per wheel/±notch —
                                  # the target is a free float (glided), so zooming feels continuous
_FIT_MARGIN = 0.05                # V4.9: modest frame margin the launch/floor fit leaves each side
# V4.10: FRAMING & COMPOSITION. The launch/HOME view and the zoom-OUT floor frame the INHABITED
# region (a square world box around the member-weighted settlement centroid) rather than the whole
# empty grid — the action fills the frame, the wilderness no longer dwarfs the towns. Falls back to
# the grid centre + whole-world fit when no settlement exists yet.
_INHABITED_PAD = 4.5             # world-cell breathing room added around the inhabited region
_INHABITED_MIN_HALF = 6.0       # smallest half-span of the framed box (a lone town isn't over-zoomed)
_FRAME_FILL = 1.5               # zoom past pure width-contain so LAND fills the frame (a 2:1 diamond
                                # in a square viewport would otherwise sit half-height in ocean); the
                                # far wilderness/ocean tips crop, never the padded inhabited region
_FRAME_FILL_SHOWCASE = 1.28     # V4.14: showcase fills the frame LESS hard, so the framed region stays
                                # inside the (already narrowed) viewport and clear of the floating feed
_TERRAIN_LRU = 3                  # non-base bucket landscapes kept baked at once (bounded memory)
_CAM_GLIDE = 0.22                 # per-frame ease toward the camera target — pan/zoom glides
# V4.10 SHOWCASE MODE — a hands-off, trailer-grade recording mode (opt-in via --showcase; every
# addition is gated on self._showcase, so the DEFAULT renderer stays byte-identical). The camera
# auto-DIRECTS: it eases to each major event, holds through its banner, then drifts back over a slow
# ambient orbit; a title card opens; the UI strips to the story banner + a small turn/phase readout.
_SHOWCASE_GLIDE = 0.045           # slow cinematic camera ease (vs _CAM_GLIDE) — never a snap cut
_SHOWCASE_FOCUS_ZOOM = 1.7        # how much CLOSER than the realm overview an event is framed
_SHOWCASE_FOCUS_LEAD = 1.2        # extra seconds the camera holds an event past its banner
_SHOWCASE_ORBIT_R = 2.6           # ambient orbit radius around the overview centre (world cells)
_SHOWCASE_ORBIT_SPEED = 0.17      # ambient orbit angular speed (rad/s) — a slow idle drift
_SHOWCASE_DAY_SECS = 80.0         # wall-clock seconds per full day/night cycle (a dusk within ~2 min)
_TITLE_DUR = 5.0                  # title-card total on-screen time (fade in / hold / fade out)
# V4.14 SHOWCASE PACING: turns run BRISK so a viewer sees several beats in a couple of minutes, and
# the run slows down only where the drama is — the opening scene holds through the title card, and
# any turn carrying a MAJOR event holds long enough for its banner(s) to be read.
_SHOWCASE_OPENING = 6.0           # the staged opening scene holds this long (covers the title card)
_SHOWCASE_HOLD = 2.2              # a turn with ONE major event holds this long — the dramatic pause
_SHOWCASE_HOLD_EACH = 1.2         # ...plus this per extra major event on the same turn
_SHOWCASE_HOLD_MAX = 6.5          # ...capped, so one very busy turn cannot stall the run
# V4.14 FLOATING FEED: in showcase the side panel is gone and the event text is an OVERLAY drawn
# straight over the world, down the outer RIGHT margin — newest at the bottom, older lines fading
# out with age. The map viewport is narrowed by the same column (see _ensure_screen), so the
# framing keeps the action clear of the text instead of drawing it over the battle.
_OVERLAY_W_FRAC = 0.27            # the feed column, as a fraction of the map width
_OVERLAY_PAD_FRAC = 0.022         # its inset from the right / bottom edges, as a fraction of width
_OVERLAY_LIFE = 15.0              # seconds a line lives before it has faded out completely
_OVERLAY_FULL = 0.45              # the fraction of that life spent at full strength before fading
_OVERLAY_MAX = 7                  # how many lines the column ever shows at once
_OVERLAY_SHADOW = (8, 7, 6)       # the outline colour that keeps text legible over bright terrain
# --- V4.15: the DIRECTOR's pacing, camera and caption cards -------------------------------
# The showcase used to give every turn the same screen time and the same wide frame, so a dynasty
# ending looked exactly like six villagers teaching each other to cook. The run is now cut by
# SEVERITY (renderer/director.py classifies the turn): quiet turns are fast-forwarded, a major turn
# is flown to and held under a caption card, and a legendary turn gets the full treatment — a
# tighter frame, a longer hold, and the rest of the world washed out around it.
_PACE_MINOR = 0.08            # a NOISE/MINOR turn: fast-forward, wide, no caption
_PACE_BLUR = 0.03             # ...and harder still once the quiet turns run on (see _RUN_BLUR)
_RUN_BLUR = 5                 # consecutive quiet turns before the run compresses to _PACE_BLUR
_CAM_EASE_SECS = 0.6          # how long the camera takes to fly to a beat before its hold begins
_HOLD_MAJOR = 2.5             # a MAJOR beat holds this long under its caption
_HOLD_LEGENDARY = 4.0         # a LEGENDARY beat holds longer, and framed tighter
_HOLD_QUEUED = 1.8            # ...but when several fire on one turn the camera CUTS between them
_ZOOM_MAJOR = 1.7             # how much closer than the overview a major beat is framed
# A legendary beat is framed CLOSE — near the interactive zoom ceiling, a few streets rather than a
# region. The wash alone was carrying the tier; at 2.3 the frame barely read as tighter than a
# major one. It also has to earn its keep for the rank/allegiance/crown detail, which is drawn at
# sprite scale and is simply invisible from the overview.
_ZOOM_LEGENDARY = 3.6
_CAPTION_FADE = 0.3           # caption cards fade in and out over this long
_CAPTION_BAND = 0.72          # the caption card sits at this fraction down the map (bottom third)
_CAPTION_BG = (10, 10, 14)    # the card's backing plate
_CAPTION_FG = (247, 240, 224) # title text — warm off-white
_CAPTION_SUB = (198, 188, 170)  # subtitle text — a step quieter than the title
_CAPTION_RULE = (196, 150, 78)  # the thin accent rule above the title
_LEGEND_WASH = (16, 14, 20)   # the desaturating wash laid over the world on a legendary hold
_LEGEND_WASH_A = 110          # ...at this alpha, eased in and out over _LEGEND_WASH_EASE
_LEGEND_WASH_EASE = 0.5       # the wash takes this long to arrive, and the same to leave
_TICKER_FG = (150, 146, 158)  # the '…years pass…' ticker during a long quiet stretch
# --showcase-pace tight: the whole run is squeezed toward this many seconds by scaling the
# fast-forward rate to the quiet turns REMAINING, so the beats keep their full holds either way.
_TIGHT_TARGET_SECS = 150.0
_PACE_MODES = ("normal", "tight")

_SHOWCASE_MUTE = (" trust in ",)  # book-keeping that rides in on a major event (the per-follower
                                  # trust ledger a victorious uprising writes) — the beat is the
                                  # uprising, not its paperwork, so showcase drops these outright
_TITLE_NAME = "AI  CIVILIZATION"
_TITLE_SUB = "an emergent history — a world makes itself"
_PAN_FRAC = 0.02                  # held-key pan speed: fraction of the viewport per frame
_LOD_FAR_MAX = 11.0               # at/below this cell size the map is the FAR strategy view
                                  # (11 keeps FAR reachable: every ladder's low bucket dips in)
_LOD_CLOSE_MIN = 26.0             # at/above this cell size the map is the CLOSE village view
_LOD_HYST = 1.0                   # hysteresis band so a tier never flickers at its boundary
_CAM_HOLD_KEYS = (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN,
                  pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_s)   # held-pan keys (polled)
# V4.12: keys that ALWAYS end the run, from any mode (fullscreen included) and before any other
# handler sees them — the viewer must never be trapped in a borderless window. Window-close and
# macOS Cmd+Q arrive as pygame.QUIT, which is handled alongside these.
_QUIT_KEYS = (pygame.K_ESCAPE, pygame.K_q)

# V4.6: ISOMETRIC PROJECTION. The world tilts into a 2:1 diamond isometric view — every map
# draw goes through world_to_screen_iso (and its ground-plane inverse). A world unit maps to a
# tile of half-width `cell` and half-height `cell/2` (the 2:1 diamond); `z` is height ABOVE the
# ground plane, lifting a point by z*cell*_ISO_ZH screen px (0 for ground things this slice —
# V4.7 hangs 3D building forms off it). The camera (centre_x, centre_y in world cells, cell px)
# pans/zooms/HOME through the SAME transform; terrain is baked into a diamond surface per zoom
# bucket (never reprojected per frame). The minimap deliberately stays TOP-DOWN (a plan read).
_ISO_ZH = 0.85                    # screen-rise per unit z, as a fraction of the cell (V4.7 height)
_ISO_ZPAD_CELLS = 2              # bake rows of head-room above the diamond (future building height)
_ISO_ELEV = 0.9                  # elevation-shading strength (lighter on sun-facing ground slopes)
_ISO_ELEV_CELLS = 3.0           # world-cell wavelength of the height field (broad rolling ground)
_ISO_RX = 1.41421356            # a ground circle of world-radius R -> a 2:1 ellipse of rx = R*cell*√2
_CLOUD_Z = 9.0                  # V4.8: cloud-puff height above the ground, in world cells (sky layer)

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
_PANEL_W = 320                 # sidebar width — the LEGACY fixed layout (window=None; the tests' path)
# V4.11 WINDOW SIZING: the window opens at the display resolution (or --window/--fullscreen), is
# RESIZABLE, and the map zone becomes a RECTANGLE that fills it. The side panel is a PROPORTION of the
# window width (clamped), so it never dominates a small screen nor looks lost on a large one.
_PANEL_FRAC = 0.24             # panel width as a fraction of the window width...
_PANEL_MIN = 220               # ...clamped to a sane minimum...
_PANEL_MAX = 380               # ...and maximum (px)


def _panel_width(win_w: int) -> int:
    """The proportional side-panel width for a window `win_w` px wide, clamped (pure)."""
    return int(max(_PANEL_MIN, min(_PANEL_MAX, win_w * _PANEL_FRAC)))


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

# V4.2: EVENT TIERS + STORY BANNER. The feed reads as a STORY, not a log: MAJOR events
# (battles, conquests, coronations/successions, uprisings, secessions, empires rising or
# fragmenting, era advances, faiths founded, prophets, ruler deaths, lines extinguished)
# show in full and bold; the MINOR churn (per-agent trust deltas, routine trades/talks/
# levies) is AGGREGATED to at most one line per turn. A prominent BANNER across the top of
# the map announces each major event in plain words so the map is self-explanatory alone.
_MAJOR_MARKERS = (
    "conquered", "repelled", "defeated", "subjugated", " in war ", "war on ",
    "broke away", "seceded", "overthrew", "-> monarch of", "an empire rises",
    "uprising", "risers rise", "triumphed", "deposed", "freed from",
    "entered the ", "arose as prophet", "took root", "is extinguished", "succeeded ",
)
_STORY_H = 42                  # story-banner band height, in pixels
_STORY_BG = (12, 12, 16)       # the banner band (drawn translucent over the top of the map)
_STORY_FG = (245, 236, 214)    # banner text — warm off-white
_STORY_ACCENT = _FEED_WAR      # the left accent bar (matches the feed's war orange)
_BANNER_SECS = 2.6             # how long one story banner holds on screen
_BANNER_SECS_FAST = 1.3        # ...shortened when several are queued, so it catches up
_BANNER_MAX_QUEUE = 8          # cap the pending banners (a busy turn can never flood)
# Feed-row highlight tiers (drawn behind a MAJOR row so it reads above the aggregated churn).
_FEED_MAJOR_BG_A = 30          # subtle per-row background tint alpha for a major line
_CAT_PHRASE = {                # aggregate wording for a turn's MINOR churn, by category
    "trade": "{n} routine trade{s}",
    "belief": "{n} belief shift{s}",
    "teaching": "{n} teaching{s}",
    "talk": "{n} exchange{s} of words",
    "levy": "{n} routine levy",           # plural ('levies') handled in aggregate_minor (irregular)
    "kin": "{n} birth{s}/inheritance{s}",
    "faith": "{n} faith stirring{s}",
    "other": "{n} minor event{s}",
}
_AGG_MAX_CLAUSES = 3           # how many merged groups a turn's minor summary lists before '+N more'


# --- V4.2: pure event-tier / story helpers (unit-testable, RNG-free, no pygame) -----------
def _strip_prefix(line: str) -> str:
    """Drop the 'turn N: ' prefix the engine writes, leaving the plain event body (pure)."""
    if line.startswith("turn "):
        i = line.find(": ")
        if i != -1:
            return line[i + 2:]
    return line


def _event_turn(line: str) -> int | None:
    """The turn number a log line belongs to, or None if it carries no 'turn N:' prefix."""
    if line.startswith("turn "):
        num = line[5:].split(":", 1)[0].strip()
        if num.isdigit():
            return int(num)
    return None


def _drop_parens(s: str) -> str:
    """Remove every '(...)' span (handling nesting) — strips combat stats from a banner (pure)."""
    out, depth = [], 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def collapse_majors(majors: list[tuple]) -> list[tuple]:
    """Fold this turn's major beats that differ ONLY in their leading subject into one (pure, V4.14).

    Six settlements crossing into the Neolithic on the same turn is one story beat, not six
    banners: they collapse to '6 settlements entered the Neolithic'. Anything with a distinct tail
    is left exactly as it is, and the first member's camera focus + colour carry the group, so the
    showcase still flies to where it happened. Order of first appearance is preserved.
    """
    groups: dict[str, list[tuple]] = {}
    order: list[str] = []
    for text, foc, color in majors:
        head, _, tail = text.partition(" ")
        key = tail or text
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append((head, foc, color))
    out = []
    for key in order:
        members = groups[key]
        head, foc, color = members[0]
        n = len(members)
        if n == 1 or key == head:
            text = f"{head} {key}".strip()
        elif all(h[:1] == "S" and any(c.isdigit() for c in h) for h, _, _ in members):
            text = f"{n} settlements {key}"          # the common case: a whole world crossing an era
        else:
            text = f"{head} {key} (+{n - 1} more)"   # mixed subjects: name the first, count the rest
        out.append((text, foc, color))
    return out


def notable_names(state: dict[str, Any]) -> frozenset[str]:
    """Every ruling/prophetic figure currently in world_state — so a RULER'S death reads as
    MAJOR while a villager's or soldier's death stays MINOR churn (pure institution read)."""
    names: set[str] = set()
    names.update(state.get("empires", {}))
    names.update(state.get("kingdoms", {}))
    for rec in state.get("monarchs", {}).values():
        if isinstance(rec, dict) and rec.get("monarch"):
            names.add(rec["monarch"])
    for rec in state.get("leaders", {}).values():
        if isinstance(rec, dict) and rec.get("leader"):
            names.add(rec["leader"])
    for rec in state.get("faiths", {}).values():
        if isinstance(rec, dict) and rec.get("prophet"):
            names.add(rec["prophet"])
    return frozenset(names)


def event_tier(line: str, notable: "frozenset[str]" = frozenset()) -> str:
    """Classify ONE verbatim event line as 'major' or 'minor' (pure string read).

    MAJOR = the turning points a viewer must never miss (see _MAJOR_MARKERS); a death is
    major only for a `notable` figure (a ruler/prophet), else it is minor churn. Everything
    else — trust deltas, routine trades/talks/levies, births — is minor.
    """
    low = line.lower()
    if " died (" in low or low.rstrip().endswith(" died"):
        return "major" if _died_name(line) in notable else "minor"
    return "major" if any(m in low for m in _MAJOR_MARKERS) else "minor"


def _died_name(line: str) -> str:
    """The subject of a 'X died (cause)' line, or '' (pure)."""
    body = _strip_prefix(line)
    i = body.find(" died")
    return body[:i].strip() if i > 0 else ""


def banner_text(line: str) -> str:
    """A MAJOR event line reduced to a plain-words STORY banner (pure).

    Strips the 'turn N:' prefix, the '(combat stats)' parentheticals and the '-> outcome'
    tail, then collapses whitespace — e.g. 'KING Borin's host was REPELLED at S0A2'.
    """
    s = _drop_parens(_strip_prefix(line))
    if " -> " in s:
        s = s.split(" -> ")[0]
    return " ".join(s.split()).strip(" ;—-")


def _minor_category(low: str) -> str:
    """Bucket a MINOR event (lowercased body) into an aggregation category (pure)."""
    if " trust in " in low:
        return "trust"
    if " sold " in low or " bought " in low or "traded" in low:
        return "trade"
    if "came to believe" in low or "took up '" in low or "renouncing" in low:
        return "belief"
    if " taught " in low:
        return "teaching"
    if ("talked to" in low or "replied to" in low or " heard " in low
            or "received from" in low or "proposed an alliance" in low or "alliance" in low):
        return "talk"
    if "levied" in low:
        return "levy"
    if "inherited" in low or "was born" in low or "came of age" in low or "estate" in low:
        return "kin"
    if "faith" in low or "believe" in low:
        return "faith"
    return "other"


def _trust_key(body: str) -> tuple[str | None, int]:
    """From 'X trust in T: a -> b (reason)' -> (target T, direction +1/-1/0) (pure)."""
    i = body.find(" trust in ")
    if i < 0:
        return (None, 0)
    rest = body[i + len(" trust in "):]
    tgt = rest.split(":", 1)[0].strip()
    nums = rest.split("(")[0]
    if "->" in nums:
        a, _, b = nums.partition("->")
        try:
            av = float(a.split(":")[-1].strip())
            bv = float(b.strip())
            return (tgt, 1 if bv > av else (-1 if bv < av else 0))
        except ValueError:
            return (tgt, 0)
    return (tgt, 0)


def aggregate_minor(lines: list[str]) -> str | None:
    """Collapse a turn's MINOR events into ONE story-feed summary line, or None (pure).

    EVERY event is merged into a group so nothing is silently dropped: trust deltas by
    (target, direction) — so all of a turn's movement on one figure reads as a single honest
    count ('5 agents' trust in LordA rose', grammatical for one agent too) — and everything
    else by its category. The groups are listed largest-first joined by ' · '; past
    _AGG_MAX_CLAUSES the long tail is tallied as '· +N more' (a count of the leftover EVENTS)
    so the line stays roughly one panel wide without ever double-counting.
    """
    if not lines:
        return None
    bodies = [_strip_prefix(l) for l in lines]
    trust: dict[tuple[str, int], int] = {}
    cats: dict[str, int] = {}
    for b in bodies:
        cat = _minor_category(b.lower())
        if cat == "trust":
            tgt, d = _trust_key(b)
            if tgt:
                trust[(tgt, d)] = trust.get((tgt, d), 0) + 1
                continue
            cat = "other"                      # an unparseable trust line -> the generic tally
        cats[cat] = cats.get(cat, 0) + 1
    clauses: list[tuple[int, str]] = []
    for (tgt, d), n in trust.items():
        verb = "rose" if d > 0 else ("fell" if d < 0 else "shifted")
        noun = "agent's" if n == 1 else "agents'"     # possessive: singular 's, plural s'
        clauses.append((n, f"{n} {noun} trust in {tgt} {verb}"))
    for cat, n in cats.items():
        if cat == "levy":                                  # irregular plural: levy -> levies
            clauses.append((n, f"{n} routine {'levy' if n == 1 else 'levies'}"))
        else:
            clauses.append((n, _CAT_PHRASE[cat].format(n=n, s="s" if n != 1 else "")))
    clauses.sort(key=lambda c: (-c[0], c[1]))         # largest group first, then stable by text
    head = "  ·  ".join(text for _, text in clauses[:_AGG_MAX_CLAUSES])
    rest = sum(n for n, _ in clauses[_AGG_MAX_CLAUSES:])
    if rest > 0:
        head += f"  · +{rest} more"
    return head


def _truncate(text: str, cols: int) -> str:
    """One-row fit: `text` clipped to `cols` chars with a trailing ellipsis when it overflows
    (pure). Minor-churn summaries are kept to a SINGLE compact row so they never wrap into a
    stack of grey lines that pushes MAJOR events out of the panel."""
    cols = max(1, cols)
    return text if len(text) <= cols else text[:cols - 1].rstrip() + "…"


def story_feed_rows(events: list[str], notable: "frozenset[str]", cols: int,
                    max_rows: int) -> list[tuple[str, tuple[int, int, int], bool]]:
    """Turn the event tail into STORY rows: MAJOR lines in full (flagged bold), with the MINOR
    churn kept scarce so majors are ALWAYS visible. Newest at the bottom. Pure. Each row is
    (text, colour, is_major); majors keep their event colour, the aggregated churn is muted grey.

    Minor discipline (V4.7 feed fix): a minor turn NEVER gets its own stack of lines. Consecutive
    all-minor turns COLLAPSE into ONE aggregated line, and a major turn's own churn folds into a
    SINGLE compact (ellipsised) line under its majors — so a quiet run can't bury the next battle.
    """
    scan = events[-_FEED_SCAN:]
    order: list[int] = []
    buckets: dict[int, list[str]] = {}
    cur = -1
    for line in scan:
        t = _event_turn(line)
        if t is not None:
            cur = t
        if cur not in buckets:
            buckets[cur] = []
            order.append(cur)
        buckets[cur].append(line)
    rows: list[tuple[str, tuple[int, int, int], bool]] = []
    pending: list[str] = []                    # minors banked across consecutive major-less turns

    def flush() -> None:
        summary = aggregate_minor(pending)
        if summary:
            rows.append((_truncate(summary, cols), _FEED_DEFAULT, False))
        pending.clear()

    for key in order:
        majors, minors = [], []
        for line in buckets[key]:
            (majors if event_tier(line, notable) == "major" else minors).append(line)
        if not majors:
            pending.extend(minors)             # a quiet turn just banks its churn — no row yet
            continue
        flush()                                # close the quiet run as ONE line before the beat
        for m in majors:
            color = event_color(m)
            for sub in (textwrap.wrap(_strip_prefix(m), width=max(1, cols)) or [""]):
                rows.append((sub, color, True))
        summary = aggregate_minor(minors)      # this beat's own churn -> one compact line
        if summary:
            rows.append((_truncate(summary, cols), _FEED_DEFAULT, False))
    flush()                                     # trailing quiet run
    return rows[-max_rows:] if max_rows > 0 else []


def realm_scoreboard(state: dict[str, Any]) -> list[tuple[str, int, bool]]:
    """(name, settlement_count, is_empire) per realm, biggest first — a pure institution read.

    Aggregates every settlement by its TOP ruler (settlement_realm — emperor > king > lone
    monarch), so subjugated kingdoms fold into their empire and the counts MATCH the map's
    territory colours. This is what the REALMS scoreboard reads (fixing the '(none)' bug).
    """
    counts: dict[str, int] = {}
    for sid in state.get("settlements", {}):
        owner = settlement_realm(sid, state)
        if owner is not None:
            counts[owner] = counts.get(owner, 0) + 1
    empires = state.get("empires", {})
    out = [(name, n, name in empires) for name, n in counts.items()]
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


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


def agent_color(personality: str | None, vivid: bool = False) -> tuple[int, int, int]:
    """The RGB colour for an agent's dominant personality trait (pure read).

    V4.4: `vivid` returns the FULL-CHROMA palette base (a ruler's saturated robe), while the
    default returns the desaturated commoner tone (_TRAIT_COLOR) so the commons sit back and
    only rulers pop. Both keep the four traits distinct.
    """
    trait = dominant_trait(personality)
    if vivid:
        return PALETTE.get(trait, _DEFAULT_COLOR)
    return _TRAIT_COLOR.get(trait, _DEFAULT_COLOR)


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


def _fit_cell(size: int, map_px: int) -> int:
    """V4.9: the iso cell that frames the PLAYABLE world (cells 0..size) across the map viewport
    with a modest margin — the launch view and the zoom-OUT floor (pure).

    The 2:1 diamond's on-screen WIDTH is 2*size*cell, so the cell that fills the frame width
    (minus _FIT_MARGIN each side) makes the world itself fill the frame — the wilderness margin
    is allowed to spill past the edges into the void rather than shrinking the world to fit it.
    Clamped to sane pixels; the whole world stays visible (its vertical slack is the filled void).
    """
    if size <= 0 or map_px <= 0:
        return _MAX_CELL
    usable = map_px * (1.0 - 2.0 * _FIT_MARGIN)
    return int(max(_CELL_FLOOR, min(_MAX_CELL, usable / (2.0 * max(1, size)))))


def _fit_cell_rect(size: int, map_w: int, map_h: int) -> int:
    """V4.11: the whole-world fit cell for a RECTANGULAR map viewport (map_w x map_h) — CONTAIN the
    playable world's 2:1 diamond (width 2*size*cell, height size*cell) so whichever dimension binds
    keeps the world visible (pure). For a square viewport this equals _fit_cell (width binds)."""
    if size <= 0 or map_w <= 0 or map_h <= 0:
        return _MAX_CELL
    uw = map_w * (1.0 - 2.0 * _FIT_MARGIN)
    uh = map_h * (1.0 - 2.0 * _FIT_MARGIN)
    cell = min(uw / (2.0 * max(1, size)), uh / max(1, size))   # width-fit vs height-fit -> contain
    return int(max(_CELL_FLOOR, min(_MAX_CELL, cell)))


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

    V4.17: the FULL feudal ladder, the one the sim actually runs (empire.py's header states it:
    EMPEROR -> subject-KING -> vassal-LORDS -> settlements), because rank is now drawn into the
    silhouette and not just stamped above the head:

        emperor > king > monarch > lord > leader > None (a commoner)

    A KING is keyed by name in `kingdoms`; a LORD is a vassal inside some king's realm. Neither
    was visible here before, so a king read as a plain settlement monarch and a lord read as a
    commoner. A sovereign MONARCH outranks a vassal LORD: a crown that answers to nobody sits
    above a lord who has sworn to one, even though the lord belongs to the larger institution.

    An agent who is several at once wears only its top rank. Each lookup degrades gracefully when
    its dict is absent (no empires -> nobody is an emperor), so the map simply shows fewer markers.
    """
    if name in state.get("empires", {}):                      # empires are keyed by emperor name
        return "emperor"
    if name in state.get("kingdoms", {}):                     # kingdoms are keyed by king name
        return "king"
    if any(r.get("monarch") == name for r in state.get("monarchs", {}).values()):
        return "monarch"
    for rec in state.get("kingdoms", {}).values():
        if isinstance(rec, dict) and name in (rec.get("vassals") or {}):
            return "lord"
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

    hut_z = _ERA_HUT_Z.get(era_style, _ERA_HUT_Z["neolithic"])
    n_buildings = min(_MAX_TOWN_BUILDINGS, max(_MIN_TOWN_BUILDINGS, int(n_members)))
    base_rc = 1.25                                    # innermost ring radius, in WORLD CELLS
    ring_gap_c = 1.3                                  # world-cell gap between concentric rings
    buildings: list[dict[str, Any]] = []
    placed, ring = 0, 0
    while placed < n_buildings:
        per_ring = 5 + ring * 3                       # outer rings hold more houses
        for k in range(per_ring):
            if placed >= n_buildings:
                break
            ang = (k / per_ring) * 2 * math.pi + (nz(placed, 1) - 0.5) * 0.7
            radc = base_rc + ring * ring_gap_c + (nz(placed, 2) - 0.5) * 0.35
            foot = _HUT_FOOT * (0.82 + nz(placed, 3) * 0.5)
            # V4.7: offsets & footprint are WORLD-CELL floats (projected each frame through the iso
            # transform), and every hut carries its own HEIGHT z — so the cluster tilts with the world
            # and the town has a low, uneven rooftop line instead of flat stamps on the ground.
            buildings.append({
                "wdx": radc * math.cos(ang), "wdy": radc * math.sin(ang),
                "fw": foot, "fd": foot * (0.82 + nz(placed, 4) * 0.4),
                "z": hut_z * (0.82 + nz(placed, 9) * 0.55),
                "wall": _pick(wall_tones, nz(placed, 5)),
                "roof": _pick(roof_tones, nz(placed, 6)),
                "hip": nz(placed, 7) > 0.5,           # hip (taller pyramid) vs shallow roof
                "lit": nz(placed, 8) > 0.45,          # lit windows
            })
            placed += 1
        ring += 1
    cluster_rc = base_rc + ring * ring_gap_c
    cluster_r = int(cluster_rc * cell)                # pixel radius (plaza ground / forge glow / label)

    granary = None
    if n_members >= _GRANARY_MIN_MEMBERS:
        gang = nz(99, 1) * 2 * math.pi
        granary = {"wdx": cluster_rc * 0.72 * math.cos(gang),
                   "wdy": cluster_rc * 0.72 * math.sin(gang),
                   "fw": _GRANARY_FOOT, "fd": _GRANARY_FOOT, "z": _GRANARY_Z}
    fence_rc = cluster_rc + 0.5 if n_members >= _FENCE_MIN_MEMBERS else None
    fence_r = int(fence_rc * cell) if fence_rc else None
    off = 0.9 if central_kind else 0.0               # nudge the well aside when a seat owns the centre
    seat_z = ((_KEEP_Z_EMPEROR if emperor else _KEEP_Z) if central_kind == "castle"
              else _HALL_Z if central_kind == "hall" else 0.0)
    seat_foot = (_KEEP_FOOT if central_kind == "castle"
                 else _HALL_FOOT if central_kind == "hall" else 0.0)
    return {
        "buildings": buildings,
        "central": {"kind": central_kind, "color": ruler_color, "emperor": emperor,
                    "z": seat_z, "foot": seat_foot},
        "granary": granary,
        "fence_rc": fence_rc,
        "fence_r": fence_r,
        "well": {"wdx": off, "wdy": off, "z": _WELL_Z, "scale": max(3, int(cell * 0.7))},
        "cluster_rc": cluster_rc,
        "cluster_r": cluster_r,
        "plaza_r": max(cell, int(base_rc * 0.95 * cell)),
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


def allegiance_color(base: tuple[int, int, int], sid: "str | None", state: dict[str, Any],
                     lerp: "dict[str, tuple] | None" = None,
                     mix: float = _ALLEGIANCE_MIX) -> tuple[int, int, int]:
    """V4.17 (5.2): pull a personality colour toward the REALM its owner belongs to (pure).

    The people wore only their personality, so when a town changed hands the ground under it
    recoloured and its inhabitants did not — a conquest looked like a paint job rather than a
    change of allegiance. Every villager now carries their sovereign's banner hue mixed into their
    own colour: enough that a cluster reads as one realm at a glance, not so much that the
    personality palette collapses into six identical crowds.

    `lerp` is the renderer's in-flight territory fade (`_territory_lerp`), so the people recolour
    WITH the ground on the same easing rather than snapping a frame apart from it. An unowned town
    (`settlement_realm` -> None) is left entirely alone, which keeps a free village looking free.
    """
    if not sid:
        return base
    tint = (lerp or {}).get(sid)
    if tint is None:
        owner = settlement_realm(sid, state)
        if owner is None:
            return base
        tint = realm_color(owner)
    return lerp_color(base, tint, mix)


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


def ambient_clouds(frame: int, size: int) -> list[tuple[float, float, float]]:
    """Drifting clouds in WORLD space: (world_x, world_y, world_radius) per cloud (pure, V4.8).

    Each cloud has a WORLD position that drifts on the wind (+x world) and wraps across the world
    extent, so the renderer projects both its ground shadow (z=0) and its sky puff (lifted by a
    cloud height) through the shared iso transform — clouds and their shadows slide over the tilted
    ground in world space, never in flat screen space. Zero RNG (pure coordinate hash)."""
    out = []
    span = size + 2 * _MARGIN_CELLS
    for k in range(3):
        rad = 2.4 + 2.0 * terrain_noise(k, 1, 81)          # world-cell radius (the sky PUFF)
        speed = 0.045 + 0.022 * k                          # world cells/frame drift (+x wind) — visibly moving
        wx = (frame * speed + span * terrain_noise(k, 2, 82)) % (span + 2 * rad) - _MARGIN_CELLS - rad
        wy = -_MARGIN_CELLS + terrain_noise(k, 3, 83) * span
        out.append((wx, wy, rad))
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


def world_to_screen_iso(pos: tuple[float, ...], cam: tuple[float, float, float],
                        view: tuple[int, int]) -> tuple[float, float]:
    """V4.6: THE one shared ISOMETRIC transform — a WORLD point (grid coords; optional z height
    above the ground) to screen pixels under camera `cam` = (centre_x, centre_y, cell_px).

    Standard 2:1 diamond projection: a world unit is `cell` half-width and `cell/2` half-height,
    so +x runs down-right and +y down-left. `z` lifts the point by z*cell*_ISO_ZH px (0 for
    everything ground-level this slice). Every map draw goes through here — no draw does its own
    projection maths — and it is pure (unit-tested against its ground-plane inverse below).
    """
    x, y = pos[0], pos[1]
    z = pos[2] if len(pos) > 2 else 0.0
    cx, cy, cell = cam
    hw, hh = cell, cell * 0.5
    sx = (x - y - (cx - cy)) * hw + view[0] * 0.5
    sy = (x + y - (cx + cy)) * hh - z * cell * _ISO_ZH + view[1] * 0.5
    return sx, sy


def screen_to_world_iso(px: tuple[float, float], cam: tuple[float, float, float],
                        view: tuple[int, int]) -> tuple[float, float]:
    """The inverse of world_to_screen_iso on the GROUND plane (z = 0), pure: a screen pixel back
    to the world (x, y) it sits over. Used for cursor-anchored zoom and the minimap viewport."""
    cx, cy, cell = cam
    hw, hh = cell, cell * 0.5
    u = (px[0] - view[0] * 0.5) / hw + (cx - cy)      # = x - y
    v = (px[1] - view[1] * 0.5) / hh + (cx + cy)      # = x + y
    return ((u + v) * 0.5, (v - u) * 0.5)


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

    Controls: SPACE pauses/resumes, ESC / Q / closing the window ends the run (raised as
    KeyboardInterrupt, which the launcher suppresses for a clean exit). Slice 11 camera:
    arrows/WASD (or left-mouse drag) pan, the wheel zooms on the cursor (+/- on the view
    centre), HOME refits the whole world; every move glides and none of it can touch the
    sim — a panned/zoomed seeded run logs byte-identical events.
    """

    def __init__(self, *, sink: Any | None = None, turn_delay: float = 0.4,
                 showcase: bool = False, showcase_motion: bool = False,
                 window: "tuple[int, int] | str | None" = None,
                 pace: str = "normal", total_turns: int = 0) -> None:
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
        # V4.17 (5.3): the crown as a physical object. `_crown_seats` is last turn's sid -> monarch,
        # diffed to notice a crown leaving a head; `_fallen_crowns` are the ones now lying on the
        # grass. Both renderer-local — the sim has no idea either exists.
        self._crown_seats: dict[str, str] = {}
        self._fallen_crowns: list[dict[str, Any]] = []
        self._territory_lerp: dict[str, tuple] = {}        # sid -> mid-lerp realm tint (aftermath)
        self._big_font: Any = None                         # the aftermath banner face
        self._feed_bold: Any = None                        # V4.2: bold face for MAJOR feed rows
        # V4.2: the STORY BANNER queue — plain-words announcements of MAJOR events, held a few
        # seconds each and QUEUED if several fire in one turn. Renderer-local + wall-clock timed,
        # so it never touches world_state or the sim RNG (and is inert in the zero-delay tests).
        self._banner_queue: collections.deque = collections.deque()
        self._banner_text: str | None = None
        self._banner_started: float = 0.0
        self._banner_turn_seen: int = -1
        self._trails: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=6))
        self._current_season = "summer"                    # tracks season for terrain rebakes
        # Slice 9: full-bleed geometry + light/ambient caches (all renderer-local).
        self._margin_px = 0                                # the wilderness ring, in pixels
        self._win_cell = _MAX_CELL                         # V4.6: base-space cell (world-derived; NOT the window)
        self._base_map = 0                                 # V4.11: world-derived SQUARE (base-space reads)
        self._map_px = 0                                   # V4.11: the map zone WIDTH (kept named _map_px)
        self._map_h = 0                                    # V4.11: the map zone HEIGHT (rectangular now)
        self._view = (0, 0)                                # (map_w, map_h) — the iso viewport
        self._paint = (0, 0)                               # V4.14: the full map zone every layer PAINTS
        self._feed_col = 0                                 # V4.14: the showcase feed column (0 otherwise)
        self._cull = 0                                     # max(map_w, map_h) — safe cull bound (never under-cull)
        self._panel_w = _PANEL_W                           # V4.11: proportional side-panel width (clamped)
        self._hud_h = _HUD_H
        self._win_w = self._win_h = 0                      # the DRAWABLE surface size (HiDPI: physical px)
        # V4.13 HiDPI: on a Retina display SDL hands back a BACKING surface larger than the size we
        # asked for (points vs physical pixels). _req_* is what we asked the OS for (the point size
        # resize events and mouse coords speak in); _win_* is what we actually draw into, and every
        # layout/cache/camera figure derives from THAT. _px_scale converts point -> pixel.
        self._req_w = self._req_h = 0
        self._px_scale = 1.0
        self._font_scale = 1.0                             # the scale the UI faces were built at
        self._legacy_split = False                         # window=None keeps the FIXED panel/HUD split
        # V4.11: WINDOW target — None = legacy world-derived square (the tests' path); a (w,h) tuple or
        # "fullscreen" from --window/--fullscreen; _pending_size carries a live VIDEORESIZE.
        # V4.12: "fullscreen" is a STARTING MODE, not a permanent preference — it is normalised to
        # the fullscreen flag plus a windowed fallback, so toggling fullscreen OFF can't be
        # immediately re-forced by the original request (which stranded the user borderless).
        self._window = "desktop" if window == "fullscreen" else window
        self._fullscreen = window == "fullscreen"
        self._pending_size: "tuple[int, int] | None" = None
        self._win_dirty = False                            # a resize/fullscreen toggle asks _ensure_screen to rebuild
        self._win_flags: "int | None" = None               # the flags the display was last created with
        self._mode_sets = 0                                # set_mode call count (must stay flat while steady)
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
        self._light_layer: Any = None                      # V4-fix: offscreen additive-pool accumulator
        # V4.9 JUICE — all renderer-local, all neutral until an effect fires (byte-identical otherwise).
        self._shake = (0, 0)                               # this frame's screen-shake offset (px)
        self._shake_amp = 0.0                              # decaying shake amplitude
        self._punch = 1.0                                  # this frame's zoom-punch scale (1.0 = none)
        self._punch_t: float | None = None                 # wall-clock start of the banner zoom-punch
        self._flash_amp = 0.0                              # decaying decisive-blow impact flash
        self._puffs: list[tuple] = []                      # (wx, wy, birth_frame, kind) rise/collapse dust
        self._prev_counts: dict[str, int] = {}             # sid -> last living count (rise/fall detection)
        self._emitters: list[tuple] = []                   # (sx, sy, kind) coins/embers to draw (close zoom)
        # V4.10 SHOWCASE — hands-off trailer mode (all gated on _showcase; default off = byte-identical).
        self._showcase = bool(showcase)
        # V4.13: showcase is a RECORDING mode — a jittering frame ruins the take. ALL camera motion
        # effects (ambient drift, zoom breath, banner zoom-punch, clash screen-shake) are OFF in
        # showcase by default; the camera then moves ONLY on the deliberate slow eases to events.
        # --showcase-motion (showcase_motion=True) puts them back. Outside showcase: unchanged.
        self._cam_motion = (not showcase) or bool(showcase_motion)
        self._minimal = bool(showcase)                     # start with the stripped UI; 'U' toggles it
        self._start_time: float | None = None              # wall clock of live() open (title card / day pace)
        self._glide = _SHOWCASE_GLIDE if showcase else _CAM_GLIDE   # slow cinematic ease in showcase
        self._title_font: Any = None                       # the big title-card face (built in live())
        self._focus_queue: collections.deque = collections.deque()  # per-banner event focus points
        self._focus_pt: tuple | None = None                # the world point the camera is framing
        self._focus_until = 0.0                            # wall clock the current event-focus holds until
        # V4.14: the FLOATING FEED (showcase only) — (text, born, colour, big) per MAJOR event, and
        # the pacing state: how many majors the turn being drawn carries, and whether the opening
        # scene has already been held.
        self._overlay_feed: collections.deque = collections.deque(maxlen=_OVERLAY_MAX * 2)
        self._turn_majors = 0
        self._opened = False
        # V4.15 DIRECTOR — the per-turn cut plan. `_beats` is this turn's queue of (severity,
        # title, subtitle, focus); the camera cuts between them when a turn carries several.
        # `_turn_sev` paces the turn, `_quiet_run` counts the consecutive quiet turns behind us
        # (which is what compresses a long uneventful stretch), and `_seen_firsts` is the run's
        # memory of which world-firsts have already been spent. All showcase-only.
        self._pace_mode = pace if pace in _PACE_MODES else "normal"
        self._beats: collections.deque = collections.deque()
        self._caption: tuple[str, str | None] | None = None
        self._caption_started = 0.0
        self._caption_hold = 0.0
        self._caption_sev = _director.MINOR
        self._turn_sev = _director.NOISE
        self._quiet_run = 0
        self._seen_firsts: set[str] = set()
        self._turns_total = max(0, int(total_turns))       # the run length, for --showcase-pace tight
        self._turns_left = 0                               # drives --showcase-pace tight
        self._legend_t: float | None = None                # wall clock a legendary wash began
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
        self._zoom_lo = float(_CELL_FLOOR)  # V4.9: interactive zoom-OUT floor (the fit cell)
        self._zoom_hi = float(_CELL_CEIL)   # V4.9: interactive zoom-IN ceiling (close village)
        self._void_bg: Any = None           # V4.9: cached out-of-bounds ocean gradient
        self._vignette: Any = None          # V4.9: cached soft edge vignette overlay
        self._cam_draw = (0.0, 0.0, _MAX_CELL)     # this frame's (centre_x, centre_y, cell)
        self._zoom_buckets: tuple[int, ...] = ()   # the quantized integer-cell zoom ladder
        self._terrain_zoom: dict[int, Any] = {}    # bucket cell -> baked landscape (LRU-capped)
        self._lod = "mid"                  # current detail tier (far/mid/close, hysteresis)
        self._drag: tuple | None = None    # mouse-drag pan anchor (screen px, camera centre)
        self._framed = False               # V4.10: has the camera snapped onto the inhabited region yet?

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
        with contextlib.suppress(Exception):
            self._feed_bold = pygame.font.SysFont("menlo,monospace", 14, bold=True)
        if self._showcase:                        # V4.10: the title-card face + the run clock
            with contextlib.suppress(Exception):
                self._title_font = pygame.font.SysFont("georgia,timesnewroman,serif", 44, bold=True)
            pygame.display.set_caption("AI Civilization — showcase")
            self._start_time = time.monotonic()
        try:
            yield self
        finally:
            # V4.12 SAFETY: however the run ends — ESC/Q, window close, or a Ctrl+C (SIGINT) raising
            # KeyboardInterrupt straight through this context — drop OUT of the borderless fullscreen
            # surface before tearing pygame down, so an interrupted run can never strand the display.
            with contextlib.suppress(Exception):
                if self._fullscreen and pygame.display.get_init():
                    pygame.display.set_mode((960, 640))
            self._fullscreen = False
            pygame.quit()
            if self._owns_sink:
                self.sink.close()

    def _ensure_screen(self, size: int) -> None:
        """Create (or resize) the window: a square MAP zone on the left + a PANEL on the right.

        Slice 3 widens the window by _PANEL_W for the event-feed sidebar; the HUD strip sits
        under the map and the panel spans the full height on the right. V4.6: the map zone is
        the same square VIEWPORT, but the world inside it is now ISOMETRIC — the fit-whole-world
        cell is HALVED (the diamond is twice as wide as tall, so half the cell makes its width
        fill the same window), and terrain bakes into a diamond surface per zoom bucket.
        """
        if self._screen is not None and size == self._size and not self._win_dirty:
            return
        self._win_dirty = False
        self._size = size
        # BASE SPACE (world-derived, independent of the window): the pixels-per-cell + margin the
        # pond/coast/star geometry is stored in, so those reads agree at any window size or zoom.
        win_cell = _cell_size(size)
        self._win_cell = win_cell
        grid_px = win_cell * max(1, size)
        self._margin_px = _MARGIN_CELLS * win_cell
        self._base_map = grid_px + 2 * self._margin_px
        # WINDOW: resolve the target size + flags + panel/HUD split (proportional panel, clamped),
        # then the MAP ZONE is the rectangle that remains. window=None keeps the legacy square.
        req_w, req_h, flags = self._resolve_window()
        # V4.12 FLICKER FIX: if NOTHING actually changed, return before touching the display. A
        # set_mode emits its own VIDEORESIZE, which previously re-entered here and called set_mode
        # again — an endless re-creation loop that read as rapid blinking in borderless fullscreen.
        # The display (and every cached layer) is now rebuilt ONLY on a genuine size/mode change.
        # (The comparison is against the REQUESTED size — what we can ask for again — not the
        # drawable, which HiDPI may return at a different scale.)
        if (self._screen is not None and size == self._size
                and (req_w, req_h, flags) == (self._req_w, self._req_h, self._win_flags)):
            return
        self._screen = pygame.display.set_mode((req_w, req_h), flags)
        self._win_flags = flags
        self._req_w, self._req_h = req_w, req_h
        self._mode_sets += 1                               # instrumentation: must stay flat when steady
        # V4.13 HiDPI: the surface we got may be LARGER than the size we asked for (macOS Retina
        # hands back a physical-pixel backing store for a point-sized window). Layout, caches and
        # the camera fit all derive from the TRUE DRAWABLE SIZE — otherwise the world is drawn into
        # the top-left corner of the backing store and the rest of the screen stays black.
        win_w, win_h = self._drawable_size(req_w, req_h)
        self._px_scale = win_w / max(1, req_w)             # point -> pixel (mouse/resize events)
        self._build_fonts()                                # chrome text follows the backing store
        panel_w, hud_h = self._splits(win_w)               # re-derived from the REAL width
        self._panel_w, self._hud_h = panel_w, hud_h
        self._map_px = max(64, win_w - self._panel_w)      # map zone WIDTH (kept named _map_px)
        self._map_h = max(64, win_h - self._hud_h)         # map zone HEIGHT
        # V4.14: the map still PAINTS full-bleed, but in showcase the PROJECTION viewport is
        # narrowed by the floating feed's column. world_to_screen_iso centres on view[0]/2, so a
        # narrower view both shifts the world left of the text and fits the frame to the space that
        # is actually clear — the camera never puts the action under the feed.
        self._feed_col = int(self._map_px * _OVERLAY_W_FRAC) if self._showcase else 0
        self._view = (max(64, self._map_px - self._feed_col), self._map_h)
        self._paint = (self._map_px, self._map_h)          # layers still cover the WHOLE zone
        self._cull = max(self._map_px, self._map_h)         # safe cull bound (never under-cull the tall axis)
        self._win_w, self._win_h = win_w, win_h
        # V4.9/V4.11: the FIT cell frames the PLAYABLE world across the (now RECTANGULAR) viewport.
        self._cell = self._cell0 = _fit_cell_rect(size, self._view[0], self._map_h)
        # Slice 11: the camera opens on the FIT-WHOLE-WORLD view and owns this grid size —
        # the base-space pond geometry is fixed here so shimmer/stars agree at every zoom.
        self._pond = _pond_geom(grid_px, win_cell, self._margin_px)
        self._zoom_buckets = zoom_buckets(self._cell0)
        # V4.9: interactive zoom bounds — OUT stops at the fit-whole-world cell (never lose the
        # world into the void), IN goes to a close village (the top baked bucket, _CELL_CEIL-capped).
        self._zoom_lo = float(self._cell0)
        self._zoom_hi = float(self._zoom_buckets[-1])
        self._cam_x = self._cam_tx = size / 2.0
        self._cam_y = self._cam_ty = size / 2.0
        self._cam_cell = self._cam_tcell = float(self._cell0)
        self._cam_draw = (self._cam_x, self._cam_y, self._cell)
        self._lod = lod_tier(self._cell0, "mid")
        self._drag = None
        self._framed = False               # V4.10: re-frame onto the inhabited region on the next drawn frame
        self._terrain_zoom = {}
        # Slice 5/9: bake the procedural landscape ONCE for this grid size (cached, blitted each
        # frame). Pure-hash texture/features — no RNG, so it never desyncs a seeded sim.
        self._terrain_bg = self._build_terrain(self._cell0)
        self._grade = self._build_grade()
        # V4.9: the out-of-bounds ocean + soft edge vignette (cached per window size, two blits/frame).
        self._void_bg = self._build_void()
        self._vignette = self._build_vignette_overlay()
        # Slice 10: the sunrise wash and the water-borne starfield are geometry-dependent,
        # so they are (re)built here with the terrain — pure hash, cached, never per frame.
        self._dawn_wash = self._build_dawn_wash()
        self._stars = self._build_stars()
        self._light_layer = None            # V4.11: re-sized to the new viewport on the next night frame
        self._stamps = {}
        # Slice 6: town plans hold pixel offsets, so a resize (new cell size) invalidates them.
        self._town_plans = {}

    def _resolve_window(self) -> tuple[int, int, int]:
        """The target (win_w, win_h, flags) to ASK the OS for this frame (V4.11): a live VIDEORESIZE,
        else fullscreen at the desktop size, else an explicit --window (resizable), else the LEGACY
        world-derived SQUARE with the FIXED panel (window=None — the tests' byte-identical path).

        The panel/HUD split is NOT decided here: it comes from _splits(), off the size we actually
        get back, because HiDPI may hand us a bigger surface than we asked for (V4.13)."""
        self._legacy_split = False
        if self._pending_size is not None:
            w, h = self._pending_size
            self._pending_size = None
            flags = (pygame.FULLSCREEN | pygame.NOFRAME) if self._fullscreen else pygame.RESIZABLE
        elif self._fullscreen:                        # (normalised in __init__; never re-forced here)
            w, h = self._desktop_size()
            flags = pygame.FULLSCREEN | pygame.NOFRAME
        elif self._window == "desktop":                # the DEFAULT for a real run: fill the display
            w, h = self._desktop_size()
            flags = pygame.RESIZABLE
        elif self._window is None:
            self._legacy_split = True                 # legacy fixed layout (non-resizable)
            pw = 0 if self._showcase else _PANEL_W
            hh = 0 if self._showcase else _HUD_H
            return self._base_map + pw, self._base_map + hh, 0
        else:
            w, h = self._window
            flags = pygame.RESIZABLE
        return max(320, int(w)), max(240, int(h)), flags

    def _splits(self, win_w: int) -> tuple[int, int]:
        """The (panel_w, hud_h) split for a window this wide. Showcase strips both to zero; the
        legacy window=None layout keeps the FIXED panel; otherwise the panel is the clamped
        proportion. V4.13: the chrome scales with the HiDPI backing store so it stays the same
        PHYSICAL size on a Retina display instead of shrinking to half."""
        if self._showcase:
            return 0, 0
        if self._legacy_split:
            return _PANEL_W, _HUD_H
        return _panel_width(win_w), int(round(_HUD_H * self._ui_scale()))

    def _ui_scale(self) -> float:
        """The HiDPI chrome/text scale (1.0 on a normal display, ~2.0 on macOS Retina)."""
        return max(1.0, min(3.0, self._px_scale))

    def _drawable_size(self, req_w: int, req_h: int) -> tuple[int, int]:
        """The TRUE size of the surface set_mode gave us — the physical backing store, which on a
        HiDPI display is larger than the point size we requested. Everything downstream (layout,
        caches, camera fit, culling) is derived from this, never from the request (V4.13)."""
        with contextlib.suppress(Exception):
            w, h = self._screen.get_size()
            if w > 0 and h > 0:
                return int(w), int(h)
        return req_w, req_h

    def _build_fonts(self) -> None:
        """(Re)build the UI faces at the current HiDPI scale. No-op at scale 1.0 — where the faces
        live() already built are correct — so the default/test path is untouched (V4.13)."""
        s = self._ui_scale()
        if s == self._font_scale or self._font is None:
            return
        self._font_scale = s
        with contextlib.suppress(Exception):
            self._font = pygame.font.SysFont("menlo,monospace", int(round(14 * s)))
            self._big_font = pygame.font.SysFont("menlo,monospace", int(round(22 * s)), bold=True)
            self._feed_bold = pygame.font.SysFont("menlo,monospace", int(round(14 * s)), bold=True)
            if self._showcase:
                self._title_font = pygame.font.SysFont("georgia,timesnewroman,serif",
                                                       int(round(44 * s)), bold=True)

    @staticmethod
    def _desktop_size() -> tuple[int, int]:
        """The primary display resolution (falls back to a sane default when unavailable)."""
        with contextlib.suppress(Exception):
            sizes = pygame.display.get_desktop_sizes()
            if sizes and sizes[0][0] > 0:
                return sizes[0]
        with contextlib.suppress(Exception):
            info = pygame.display.Info()
            if info.current_w > 0:
                return info.current_w, info.current_h
        return 1280, 800

    def _apply_resize(self, w: int, h: int) -> None:
        """Handle a VIDEORESIZE: recompute the whole layout + caches for the new size (V4.11).

        V4.12: resize events are IGNORED while fullscreen (the mode owns its size) and when the size
        has not actually changed — set_mode emits its own resize event, and acting on that echo is
        what re-created the display every frame."""
        # (compared against the REQUESTED size — resize events speak in points, and on HiDPI the
        # drawable is bigger, so comparing to _win_* would treat every echo as a real change.)
        if self._fullscreen or (w, h) in ((self._req_w, self._req_h), (self._win_w, self._win_h)):
            return
        self._pending_size = (w, h)
        self._win_dirty = True
        self._ensure_screen(self._size)

    def _toggle_fullscreen(self) -> None:
        """F11/F: toggle borderless fullscreen at runtime, rebuilding the layout + caches (V4.11)."""
        self._fullscreen = not self._fullscreen
        if not self._fullscreen:
            # V4.12: leaving fullscreen must land on a REAL windowed size, else the next resolve
            # would put us straight back and the user would be stranded borderless.
            if isinstance(self._window, tuple):
                self._pending_size = self._window
            else:
                dw, dh = self._desktop_size()
                self._pending_size = (max(320, int(dw * 0.8)), max(240, int(dh * 0.8)))
        self._win_dirty = True
        self._ensure_screen(self._size)

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
        self._enqueue_banners(state)      # V4.2: queue this turn's MAJOR-event announcements
        self._track_crowns(state)         # V4.17 (5.3): a dead king's crown falls where he stood
        self._pump_events()
        if self._prev_snapshot is not None and self.turn_delay > 0:
            lines = turn_events(state.get("events") or [], int(state.get("turn", 0)))
            for scene in battle_scenes(lines, self._prev_snapshot, state):
                self._play_cinematic(scene, state)
        self._animate_turn(state)
        self._prev_snapshot = take_snapshot(state)

    # -- input -------------------------------------------------------------
    def _window_event(self, event: Any) -> bool:
        """V4.11: handle a window event (resize / F11-F fullscreen). True when consumed. Quit keys are
        handled by the CALLERS, ahead of this, so nothing can swallow them."""
        if event.type == pygame.VIDEORESIZE:
            self._apply_resize(event.w, event.h)
            return True
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_F11, pygame.K_f):
            self._toggle_fullscreen()
            return True
        return False

    def _pump_events(self) -> None:
        """Drain the OS event queue; window/camera events first, pause on SPACE.

        V4.12: ESC and Q ALWAYS quit immediately, from any mode — fullscreen included. (ESC used to
        merely leave fullscreen, which stranded the viewer in a borderless window with no chrome.)
        QUIT (window close, and macOS Cmd+Q) quits too; F11/F is the fullscreen toggle."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN and event.key in _QUIT_KEYS:
                raise KeyboardInterrupt                  # checked BEFORE any other handler
            if self._window_event(event):
                continue
            if self._handle_camera_event(event):
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_u:            # V4.10: toggle the full UI back on mid-run
                    self._minimal = not self._minimal

    def _pump_cinema_events(self) -> bool:
        """Input during a cinematic: window resize/fullscreen still apply; ESC/Q/QUIT still END THE
        RUN immediately (V4.12 — never trap the viewer mid-battle); any NON-CAMERA key SKIPS the scene."""
        skip = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN and event.key in _QUIT_KEYS:
                raise KeyboardInterrupt
            if self._window_event(event):
                continue
            if self._handle_camera_event(event):
                continue
            if event.type == pygame.KEYDOWN:
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
        centre = (self._map_px // 2, self._map_h // 2)
        pos = self._mouse_px(getattr(event, "pos", None))   # V4.13: points -> drawable pixels
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self._zoom_step(1, centre)
                return True
            if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self._zoom_step(-1, centre)
                return True
            if event.key == pygame.K_HOME:                 # ease back to the inhabited-region view
                hx, hy, hcell = self._home_view(self._last_state or {})
                self._cam_tx, self._cam_ty, self._cam_tcell = hx, hy, hcell
                return True
            return event.key in _CAM_HOLD_KEYS
        if event.type == pygame.MOUSEWHEEL and event.y:
            anchor = centre
            with contextlib.suppress(Exception):
                mx, my = self._mouse_px(pygame.mouse.get_pos())
                if mx < self._map_px and my < self._map_px:
                    anchor = (mx, my)
            self._zoom_step(1 if event.y > 0 else -1, anchor)
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 \
                and pos[0] < self._map_px and pos[1] < self._map_px:
            # V4.6: remember the WORLD point grabbed, so the drag keeps it under the cursor.
            view = self._view
            self._drag = screen_to_world_iso(pos,
                                             (self._cam_tx, self._cam_ty, self._cam_tcell), view)
            return True
        if event.type == pygame.MOUSEMOTION and self._drag is not None:
            self._cam_tx, self._cam_ty = self._cam_for_anchor(
                pos, self._drag[0], self._drag[1], self._cam_tcell)
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            had_drag, self._drag = self._drag is not None, None
            return had_drag
        return False

    def _mouse_px(self, pos: "tuple[int, int] | None") -> tuple[int, int]:
        """A mouse position (which SDL reports in window POINTS) in DRAWABLE pixels — the space the
        whole layout lives in. Identity on a non-HiDPI display (V4.13)."""
        if pos is None:
            return (0, 0)
        s = self._px_scale
        return (int(pos[0] * s), int(pos[1] * s)) if s != 1.0 else (int(pos[0]), int(pos[1]))

    def _zoom_step(self, direction: int, anchor: tuple[int, int]) -> None:
        """V4.9: nudge the zoom TARGET by ONE small MULTIPLICATIVE notch (x_ZOOM_STEP in, its
        reciprocal out), keeping the world point under `anchor` fixed on screen, and clamp to
        [_zoom_lo, _zoom_hi] (fit-whole-world .. close village). The target is a free float; the
        per-frame glide eases the live cell toward it, so zooming feels continuous at every level
        rather than jumping between buckets. Terrain still bakes per bucket — the blit scales the
        nearest bucket to the live cell (one cheap transform), so no bake happens per frame."""
        t = self._cam_tcell
        factor = _ZOOM_STEP if direction > 0 else (1.0 / _ZOOM_STEP)
        new = max(self._zoom_lo, min(self._zoom_hi, t * factor))
        if abs(new - t) < 1e-6:
            return
        view = self._view
        # V4.6: keep the world point under `anchor` fixed across the zoom (iso re-anchor).
        wx, wy = screen_to_world_iso(anchor, (self._cam_tx, self._cam_ty, t), view)
        self._cam_tcell = new
        self._cam_tx, self._cam_ty = self._cam_for_anchor(anchor, wx, wy, new)
        self._cam_tx, self._cam_ty = clamp_camera(self._cam_tx, self._cam_ty, new,
                                                  self._size, view)

    def _cam_for_anchor(self, anchor: tuple[int, int], wx: float, wy: float,
                        cell: float) -> tuple[float, float]:
        """The camera centre that puts world point (wx, wy) under screen `anchor` at `cell`
        zoom (pure iso re-anchor — the inverse used by cursor zoom and drag-pan)."""
        vw, vh = self._view
        hw, hh = cell, cell * 0.5
        u = (wx - wy) - (anchor[0] - vw * 0.5) / hw       # = cx - cy
        v = (wx + wy) - (anchor[1] - vh * 0.5) / hh       # = cx + cy
        return (u + v) * 0.5, (v - u) * 0.5

    def _fit_region_cell(self, side_cells: float) -> float:
        """The cell that frames a square world box of side `side_cells` in the RECTANGULAR map viewport
        (V4.11). Its 2:1 diamond is width 2*side*cell, height side*cell; CONTAIN by whichever dimension
        binds, boosted (_FRAME_FILL) so land fills the frame. Clamped, with zoom-in headroom left."""
        s = max(1.0, side_cells)
        uw = self._view[0] * (1.0 - 2.0 * _FIT_MARGIN)     # V4.14: the CLEAR width (minus the feed column)
        uh = self._map_h * (1.0 - 2.0 * _FIT_MARGIN)
        fill = _FRAME_FILL_SHOWCASE if self._showcase else _FRAME_FILL
        cell = min(uw / (2.0 * s), uh / s) * fill           # width-fit vs height-fit -> contain, boosted
        return float(max(_CELL_FLOOR, min(_CELL_CEIL, self._zoom_hi * 0.9, cell)))

    def _home_view(self, state: dict[str, Any]) -> tuple[float, float, float]:
        """V4.10: the launch/HOME view + zoom-out floor — (centre_x, centre_y, cell) framing the
        INHABITED region: the camera centres on the member-WEIGHTED centroid of settlements and the
        zoom fits a square box that holds every settlement (padded), so the towns fill the frame
        instead of floating in empty wilderness. Falls back to the grid centre + whole-world fit when
        no settlement exists. A pure READ of settlement centres/members."""
        sett = (state or {}).get("settlements") or {}
        wx = wy = wsum = 0.0
        x0 = y0 = float("inf")
        x1 = y1 = float("-inf")
        for rec in sett.values():
            c = rec.get("center") if isinstance(rec, dict) else None
            if c is None:
                continue
            n = float(max(1, len(rec.get("members") or ())))
            cx, cy = float(c[0]), float(c[1])
            wx += cx * n
            wy += cy * n
            wsum += n
            x0, y0 = min(x0, cx), min(y0, cy)
            x1, y1 = max(x1, cx), max(y1, cy)
        if wsum <= 0:                                       # no towns yet -> the old whole-world fit
            return (self._size / 2.0, self._size / 2.0, float(self._cell0))
        cx, cy = wx / wsum, wy / wsum
        # a square box around the CENTROID that holds every settlement, padded and floored so a lone
        # town is not over-zoomed; the zoom fits that box (NOT the whole empty map).
        half = max(_INHABITED_MIN_HALF,
                   _INHABITED_PAD + max(cx - x0, x1 - cx, cy - y0, y1 - cy))
        return (cx, cy, self._fit_region_cell(2.0 * half))

    def _update_camera(self) -> None:
        """Advance the camera one drawn frame: poll held pan keys, glide toward the targets,
        clamp to the world, and freeze this frame's shared transform + LOD tier.

        Runs at the top of every _draw — turn walks, pauses and cinematics all glide. It
        reads input and writes ONLY renderer-local camera fields; panning/zooming during a
        seeded run can never change the event log.
        """
        size, view = self._size, self._view
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
        # V4.9: keep the zoom TARGET inside its interactive bounds (fit-whole-world .. close village)
        # however it was set — held keys, a stale target, or a resize that moved the floor.
        self._cam_tcell = max(self._zoom_lo, min(self._zoom_hi, self._cam_tcell))
        self._cam_tx, self._cam_ty = clamp_camera(self._cam_tx, self._cam_ty,
                                                  self._cam_tcell, size, view)
        # Glide: ease a fraction of the remaining distance per frame; SNAP when close, so a
        # settled zoom sits exactly on its integer bucket (terrain then blits 1:1).
        for attr, target in (("_cam_x", self._cam_tx), ("_cam_y", self._cam_ty),
                             ("_cam_cell", self._cam_tcell)):
            cur = getattr(self, attr)
            nxt = cur + (target - cur) * self._glide
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

        V4.14: the turn's WALK always takes `turn_delay`; the turn's HOLD may be longer (a showcase
        dramatic pause), in which case the movers arrive on time and the settled scene is held.
        Outside showcase hold == walk, exactly as before.
        """
        prev_pos = (self._prev_snapshot or {}).get("positions") or {}
        if self.turn_delay <= 0:
            self._draw(state)
            return
        walk, hold = self.turn_delay, self._pace()
        start = time.monotonic()
        while True:
            self._pump_events()
            if self.paused:
                self._draw(state, paused=True)
                start = time.monotonic()      # resume restarts the walk (the old deadline reset)
                time.sleep(0.01)
                continue
            el = time.monotonic() - start
            self._draw(state, motion=(prev_pos, min(1.0, el / walk)))
            if el >= hold:
                self._opened = True           # the opening scene has had its hold
                return
            time.sleep(1 / 60)

    def _pace(self) -> float:
        """How long THIS turn stays on screen (V4.14). Outside showcase: exactly `turn_delay`.

        In showcase the base pace is brisk so the history moves, and the run slows only where the
        drama is: the staged OPENING scene holds through the title card, and a turn carrying MAJOR
        events holds long enough for each banner to be read (capped, so one busy turn never stalls).
        """
        if not self._showcase:
            return self.turn_delay
        if not self._opened:
            return max(self.turn_delay, _SHOWCASE_OPENING)
        n = len(self._beats)
        if n == 0:
            return self._quiet_pace()
        # One beat gets the full pan-in and hold; several on one turn are CUT between (turn 26 of
        # a seed-7 run fires two uprisings at once — both are shown, neither is dropped), which is
        # brisker per beat but never drops one. The fly-in is paid once, before the first hold.
        per = _HOLD_LEGENDARY if self._turn_sev == _director.LEGENDARY else _HOLD_MAJOR
        if n > 1:
            per = max(_HOLD_QUEUED, per * 0.72)
        return max(self.turn_delay, _CAM_EASE_SECS + per * n)

    def _quiet_pace(self) -> float:
        """The fast-forward rate for a turn with no beat in it.

        A quiet turn is a blink; a RUN of them is flown over harder still (and carries the
        '…years pass…' ticker). Under `--showcase-pace tight` the rate is scaled to the quiet
        turns REMAINING so the whole run lands near _TIGHT_TARGET_SECS — the beats keep their
        full holds either way, because a trailer should be short in its dull parts, not its
        dramatic ones.
        """
        base = _PACE_BLUR if self._quiet_run >= _RUN_BLUR else _PACE_MINOR
        if self._pace_mode == "tight" and self._turns_left > 0:
            # Budget: whatever is left of the target after the beats have taken their share.
            spent = time.monotonic() - (self._start_time or time.monotonic())
            budget = max(0.0, _TIGHT_TARGET_SECS - spent)
            base = min(base, max(0.01, budget / max(1, self._turns_left)))
        return min(base, self.turn_delay) if self.turn_delay > 0 else base

    # -- drawing (pure reads of `state`) -----------------------------------
    def _fx(self, sx: float, sy: float) -> tuple[float, float]:
        """V4.9: apply this frame's JUICE screen affine — the zoom-PUNCH (scale about the map centre)
        and the clash SHAKE (offset) — to a raw projected screen point. A fast identity path keeps
        every frame BYTE-IDENTICAL when no effect is active (the tests' path)."""
        if self._punch == 1.0 and self._shake == (0, 0):
            return sx, sy
        cx, cy = self._map_px * 0.5, self._map_h * 0.5
        s = self._punch
        return cx + (sx - cx) * s + self._shake[0], cy + (sy - cy) * s + self._shake[1]

    def _to_px(self, x: float, y: float, z: float = 0.0) -> tuple[int, int]:
        """V4.6: the GROUND point of world cell (x, y) (its centre) in SCREEN pixels, through the
        ONE shared ISOMETRIC transform — no draw call does its own projection maths. `z` lifts a
        point above the ground plane (0 for everything ground-level this slice). V4.9 rides the
        frame's juice affine (punch/shake) so sprites shake/scale in lockstep with the terrain."""
        sx, sy = world_to_screen_iso((x + 0.5, y + 0.5, z), self._cam_draw,
                                     self._view)
        sx, sy = self._fx(sx, sy)
        return int(round(sx)), int(round(sy))

    def _base_to_screen(self, bx: float, by: float) -> tuple[int, int]:
        """A legacy SQUARE base-space pixel (the fit-view space the star field is stored in) ->
        world cell -> the shared ISO transform, so those baked reads still track the camera.
        (Water shimmer / coast waves are re-seated in V4.8; only the star field uses this now.)"""
        c0 = max(1, self._win_cell)
        return self._to_px(bx / c0 - _MARGIN_CELLS - 0.5, by / c0 - _MARGIN_CELLS - 0.5)

    # -- V4.9: JUICE — per-frame effect bookkeeping + drawers (all frame-clock/hash driven) ----
    def _update_juice(self) -> None:
        """Advance the decaying/eased juice effects for THIS frame. Runs first in _draw, so the
        shake/punch are set before any _to_px. Neutral (byte-identical) whenever nothing fires."""
        f = self._frame
        if not self._cam_motion:
            # V4.13: showcase is a RECORDING mode — the camera-moving juice (clash shake, banner
            # zoom-punch) is what read as a jittering frame, so it is OFF unless --showcase-motion
            # asks for it. The non-camera juice below (flash, puffs, emitters) still runs.
            self._shake_amp, self._shake, self._punch, self._punch_t = 0.0, (0, 0), 1.0, None
        elif self._shake_amp > 0.4:                        # a short, sharp clash shake (decays fast)
            a = self._shake_amp
            self._shake = (int(round((terrain_noise(f, 1, 301) - 0.5) * 2 * a)),
                           int(round((terrain_noise(f, 2, 302) - 0.5) * 2 * a)))
            self._shake_amp *= _SHAKE_DECAY
        else:
            self._shake_amp, self._shake = 0.0, (0, 0)
        if self._punch_t is not None:                      # a gentle banner zoom-punch (ease in/back)
            el = time.monotonic() - self._punch_t
            if el >= _PUNCH_DUR:
                self._punch, self._punch_t = 1.0, None
            else:
                self._punch = 1.0 + _PUNCH_MAX * math.sin(math.pi * (el / _PUNCH_DUR))
        else:
            self._punch = 1.0
        self._flash_amp = self._flash_amp * _FLASH_DECAY if self._flash_amp > 0.02 else 0.0
        if self._puffs:                                    # age out finished rise/collapse puffs
            self._puffs = [p for p in self._puffs if f - p[2] < _PUFF_LIFE]
        self._emitters = []                                # refilled by _emit_town at close zoom

    def _note_population(self, sid: str, living: int, center: tuple, rad_cells: float) -> None:
        """Detect a settlement's building count rising/falling between turns and spawn a DUST burst
        (rise) or a COLLAPSE puff (fall/decay) — a town that grows or dies now announces it."""
        prev = self._prev_counts.get(sid)
        self._prev_counts[sid] = living
        if prev is None or prev == living or self._lod == "far":
            return
        kind = "rise" if living > prev else "fall"
        f, salt = self._frame, sum(ord(c) for c in sid) & 255      # stable per-settlement salt
        for j in range(3):
            ang = terrain_noise(f, j * 7 + salt, 310) * 2 * math.pi
            rr = rad_cells * (0.35 + 0.55 * terrain_noise(f, j, 311))
            self._puffs.append((center[0] + rr * math.cos(ang),
                                center[1] + rr * math.sin(ang), f, kind))

    def _draw_puffs(self) -> None:
        """Dust wisps rising from a new building / a collapse cloud from a fallen one — projected
        onto the ground, drifting up and fading over their short life."""
        if not self._puffs or self._lod == "far":
            return
        cell, view = self._cell, self._cull
        for wx, wy, birth, kind in self._puffs:
            p = (self._frame - birth) / _PUFF_LIFE
            gx, gy = self._to_px(wx, wy)
            if not visible_on_screen(gx, gy, cell * 2, view, view):
                continue
            rise = int(p * cell * 1.1)
            rad = max(1, int(cell * (0.14 + 0.34 * p)))
            alpha = int(115 * (1.0 - p))
            if alpha <= 0:
                continue
            col = _DUST if kind == "rise" else (86, 80, 72)
            stamp = self._soft_stamp(rad, col, alpha)
            self._screen.blit(stamp, (gx - stamp.get_width() // 2, gy - rise - stamp.get_height() // 2))

    def _draw_emitters(self) -> None:
        """CLOSE zoom only: coins drifting up over a market (plaza) and embers over a forge — a few
        per emitter, rising and fading on the frame clock (stable per emitter, so no pan flicker)."""
        if self._lod != "close" or not self._emitters:
            return
        s, f, cell = self._screen, self._frame, self._cell
        for e, (sx, sy, kind) in enumerate(self._emitters):
            for j in range(3):
                ph = (f * 0.035 + terrain_noise(e, j, 320)) % 1.0
                dx = int((terrain_noise(e * 4 + j, 5, 321) - 0.5) * cell * 0.7
                         + math.sin(f * 0.09 + e + j) * 2)
                px, py = sx + dx, sy - int(ph * cell * 1.7)
                a = 1.0 - ph
                if a <= 0.05:
                    continue
                if kind == "coin":
                    pygame.draw.circle(s, PALETTE["coin"], (px, py), max(1, cell // 15))
                    pygame.draw.circle(s, _shade(PALETTE["coin"], -40), (px, py), max(1, cell // 15), 1)
                else:                                      # ember: a warm mote, brighter at night
                    col = lerp_color(PALETTE["forge_glow"], PALETTE["forge_core"], 0.4 + 0.4 * self._nf)
                    pygame.draw.circle(s, col, (px, py), max(1, cell // 18))

    def _draw_flash(self) -> None:
        """A brief warm impact FLASH over the map on a war cinematic's decisive blow (decays fast)."""
        if self._flash_amp <= 0.02:
            return
        ov = pygame.Surface(self._paint)
        ov.fill((255, 248, 232))
        ov.set_alpha(int(150 * min(1.0, self._flash_amp)))
        self._screen.blit(ov, (0, 0))

    # -- V4.10: SHOWCASE — camera auto-direction, title card, minimal readout ------
    def _showcase_direct(self, state: dict[str, Any]) -> None:
        """Drive the camera hands-off: EASE to a firing major event and hold it through its banner,
        else a slow ambient ORBIT of the realm overview so the frame is never static. Sets only the
        glide TARGETS (the slow _SHOWCASE_GLIDE eases the rest); pure read of the settlement layout."""
        self._update_caption()            # V4.15: pop the next beat when the current hold expires
        now = time.monotonic()
        ox, oy, ocell = self._home_view(state)             # the realm overview (inhabited region)
        if self._focus_pt is not None and now < self._focus_until:
            self._cam_tx, self._cam_ty = self._focus_pt    # frame the event, closer in
            # V4.15: a LEGENDARY beat is framed TIGHTER than a major one — the tier is legible in
            # the composition before a word of the caption has been read.
            zoom = (_ZOOM_LEGENDARY if self._caption_sev == _director.LEGENDARY else _ZOOM_MAJOR)
            self._cam_tcell = min(self._zoom_hi * 0.95, ocell * zoom)
        elif not self._cam_motion:
            # V4.13 DEFAULT: no ambient drift, no zoom breath — the camera rests dead still on the
            # realm overview between events, so a recorded take is rock steady. The frame is kept
            # alive by the world itself (agents, weather, light), not by moving the lens.
            self._focus_pt = None
            self._cam_tx, self._cam_ty, self._cam_tcell = ox, oy, ocell
        else:
            self._focus_pt = None
            th = (now - (self._start_time or now)) * _SHOWCASE_ORBIT_SPEED
            # V4.11: the drift amplitude is capped by the pan ROOM each axis actually has before
            # clamp_camera pins the view — so on a wide 16:9 (which crops vertically) the orbit
            # rides the y axis, on a tall window the x axis, and it is never silently clamped flat.
            span = (self._size + 2 * _MARGIN_CELLS) / 2.0
            room_x = max(0.0, span - self._view[0] / (2.0 * max(1e-6, ocell)))
            room_y = max(0.0, span - self._map_h / (2.0 * max(1e-6, ocell)))
            ax = min(_SHOWCASE_ORBIT_R, room_x * 0.8)
            ay = min(_SHOWCASE_ORBIT_R * 0.7, room_y * 0.8)
            self._cam_tx = ox + math.cos(th) * ax                          # a slow idle drift/orbit
            self._cam_ty = oy + math.sin(th) * ay
            # the zoom BREATH is clamp-immune, so the frame still lives even when both axes are pinned
            self._cam_tcell = ocell * (1.0 + 0.055 * math.sin(th * 0.45))

    def _update_caption(self) -> None:
        """Advance the caption/camera cut plan on the wall clock (showcase only).

        Pops the next beat when the current one's hold expires, points the camera at it, and — for
        a LEGENDARY beat — starts the wash that drains the colour out of the rest of the world.
        """
        if not self._showcase:
            return
        now = time.monotonic()
        if self._caption is not None and now - self._caption_started < self._caption_hold:
            return
        if not self._beats:
            if self._caption is not None:
                self._caption = None
                self._legend_t = None
            return
        sev, title, sub, foc = self._beats.popleft()
        legendary = sev == _director.LEGENDARY
        per = _HOLD_LEGENDARY if legendary else _HOLD_MAJOR
        if self._turn_majors > 1:
            per = max(_HOLD_QUEUED, per * 0.72)
        self._caption = (title, sub)
        self._caption_sev = sev
        self._caption_started = now
        self._caption_hold = per
        self._legend_t = now if legendary else None
        if foc is not None:
            self._focus_pt = foc
            self._focus_until = now + per

    def _caption_alpha(self) -> float:
        """The active caption's fade envelope: in over _CAPTION_FADE, out over the same (pure)."""
        if self._caption is None:
            return 0.0
        el = time.monotonic() - self._caption_started
        if el < _CAPTION_FADE:
            return max(0.0, el / _CAPTION_FADE)
        left = self._caption_hold - el
        if left < _CAPTION_FADE:
            return max(0.0, left / _CAPTION_FADE)
        return 1.0

    def _draw_legendary_wash(self) -> None:
        """Drain the world's colour for a LEGENDARY hold — the punctuation mark of the run.

        A dark, desaturating wash over the whole map with a soft hole punched around the focal
        settlement, so the eye is left with the one place that matters. Eased in and out over
        _LEGEND_WASH_EASE, so it arrives as a mood rather than a cut to black. UI overlay: drawn
        under the caption card, over the world.
        """
        if self._legend_t is None:
            return
        el = time.monotonic() - self._legend_t
        env = min(1.0, el / _LEGEND_WASH_EASE)
        left = self._caption_hold - el
        if left < _LEGEND_WASH_EASE:
            env = min(env, max(0.0, left / _LEGEND_WASH_EASE))
        if env <= 0.01:
            return
        ov = pygame.Surface(self._paint, pygame.SRCALPHA)
        ov.fill((*_LEGEND_WASH, int(_LEGEND_WASH_A * env)))
        if self._focus_pt is not None:
            # Punch a soft hole over the focal settlement: concentric transparent discs, so the
            # centre of the frame keeps its colour and the edges of the world fall away.
            fx, fy = self._to_px(*self._focus_pt)
            r0 = int(max(self._map_px, self._map_h) * 0.22)
            for i in range(10):
                r = int(r0 * (1.0 + i * 0.09))
                a = int(_LEGEND_WASH_A * env * (i / 10.0))
                pygame.draw.circle(ov, (*_LEGEND_WASH, a), (fx, fy), r)
        self._screen.blit(ov, (0, 0))

    def _caption_layout(self) -> "tuple[Any, Any, Any, tuple[int, int, int, int], int] | None":
        """Where the caption card sits and what it is made of, or None if there is nothing to draw.

        Split out from the drawing so the geometry is inspectable: (title surface, subtitle surface
        or None, the title face, the plate rect, the pad). Pure apart from font rendering.
        """
        if self._caption is None:
            return None
        title, sub = self._caption
        legendary = self._caption_sev == _director.LEGENDARY
        big = self._title_font if legendary else self._big_font
        big = big or self._big_font or self._font
        small = self._font
        if big is None:
            return None
        # Centre on the FULL painted zone, not `_view`. The viewport is narrowed to keep the feed
        # column clear of the action, but the feed is floating text with no plate — the card has
        # nothing to dodge, and centring on the narrowed view visibly parked it left of centre.
        w = self._paint[0] or self._map_px
        t_surf = big.render(title, True, _CAPTION_FG)
        s_surf = small.render(sub, True, _CAPTION_SUB) if (sub and small) else None
        pad = max(10, int(self._map_h * 0.018))
        tw = max(t_surf.get_width(), s_surf.get_width() if s_surf else 0)
        th = t_surf.get_height() + ((s_surf.get_height() + pad // 2) if s_surf else 0)
        bw, bh = tw + pad * 3, th + pad * 2
        bx = max(0, (w - bw) // 2)
        by = int(self._map_h * _CAPTION_BAND)
        by = min(by, max(0, self._map_h - bh - pad))
        return t_surf, s_surf, big, (bx, by, bw, bh), pad

    def _draw_caption_card(self) -> None:
        """The caption card: a title line and at most one subtitle, across the bottom third.

        Not the raw log line — the director's dramatised rendering of it ('THE RISING OF S0B2' /
        'Lord B falls. 2 risers fell. The hoard is theirs.'). Fades in and out with the hold, sits
        on a translucent plate so it stays legible over bright grass and over night alike, and a
        legendary card is larger with a warm rule above it. UI overlay: never transformed.
        """
        alpha = self._caption_alpha()
        if alpha <= 0.01:
            return
        layout = self._caption_layout()
        if layout is None:
            return
        t_surf, s_surf, _big, (bx, by, bw, bh), pad = layout
        legendary = self._caption_sev == _director.LEGENDARY
        plate = pygame.Surface((bw, bh), pygame.SRCALPHA)
        plate.fill((*_CAPTION_BG, int((205 if legendary else 175) * alpha)))
        self._screen.blit(plate, (bx, by))
        rule_w = int(bw * (0.34 if legendary else 0.2))
        rule = pygame.Surface((rule_w, 2), pygame.SRCALPHA)
        rule.fill((*_CAPTION_RULE, int(235 * alpha)))
        self._screen.blit(rule, (bx + (bw - rule_w) // 2, by + pad // 2))
        t_surf.set_alpha(int(255 * alpha))
        self._screen.blit(t_surf, (bx + (bw - t_surf.get_width()) // 2, by + pad))
        if s_surf is not None:
            s_surf.set_alpha(int(235 * alpha))
            self._screen.blit(s_surf, (bx + (bw - s_surf.get_width()) // 2,
                                       by + pad + t_surf.get_height() + pad // 2))

    def _draw_quiet_ticker(self) -> None:
        """'…years pass…' — an unobtrusive marker that the run is flying over a quiet stretch.

        Only once a RUN of quiet turns has built up (a single dull turn needs no explanation), and
        deliberately small and grey: it tells the viewer the fast-forward is intentional without
        competing with the world for attention.
        """
        if not self._showcase or self._quiet_run < _RUN_BLUR or self._font is None:
            return
        turn = int((self._last_state or {}).get("turn", 0))
        surf = self._font.render(f"…years pass…   turn {turn}", True, _TICKER_FG)
        surf.set_alpha(150)
        w = self._paint[0] or self._map_px          # centred on the window, like the caption card
        self._screen.blit(surf, ((w - surf.get_width()) // 2,
                                 int(self._map_h * _CAPTION_BAND)))

    def _feed_lines(self) -> list:
        """The feed entries actually shown this frame, oldest first (pure read).

        The feed is what has ALREADY happened, so the beat currently under the caption card is
        held back — it was on screen twice in the same words — and rejoins the column the moment
        its hold ends. Filtered BEFORE the _OVERLAY_MAX slice, so holding one line back promotes an
        older beat into the column rather than leaving a gap where the captioned one would have sat.
        """
        live = self._caption[0] if self._caption is not None else None
        return [e for e in self._overlay_feed if e[0] != live][-_OVERLAY_MAX:]

    def _draw_feed_overlay(self) -> None:
        """V4.14: the showcase event feed — text floating straight over the world, no panel.

        A column down the OUTER RIGHT margin (the map viewport is narrowed by the same column, so
        the action never sits under it): MAJOR beats only, newest at the BOTTOM, each line fading
        out as it ages so the column reads as a living feed rather than a log. Every glyph is drawn
        with a dark outline, which is what keeps it legible over bright grass AND over night.

        The feed is what has ALREADY happened, so the beat currently under the caption card is
        held back — it was appearing twice on screen in the same words — and joins the column the
        moment its hold ends.
        """
        now = time.monotonic()
        while self._overlay_feed and now - self._overlay_feed[0][1] > _OVERLAY_LIFE:
            self._overlay_feed.popleft()      # age out first: the feed empties even with no face
        font = self._feed_bold or self._font
        if font is None or not self._overlay_feed:
            return
        pad = max(12, int(self._map_px * _OVERLAY_PAD_FRAC))
        right = self._map_px - pad
        col_w = max(120, self._feed_col - pad)
        cols = max(12, int(col_w / max(1, font.size("n")[0])))
        big = self._big_font or font
        y = self._map_h - pad                              # stack UPWARD from the bottom edge
        shown = self._feed_lines()
        for text, born, color, war in reversed(shown):
            age = (now - born) / _OVERLAY_LIFE
            alpha = 1.0 if age <= _OVERLAY_FULL else max(0.0, (1.0 - age) / (1.0 - _OVERLAY_FULL))
            if alpha <= 0.02:
                continue
            face = big if war else font                    # a war/conquest beat reads LARGER
            fcols = max(10, int(cols * font.size("n")[0] / max(1, face.size("n")[0])))
            rows = textwrap.wrap(text, width=fcols) or [text]
            for row in reversed(rows):
                y -= face.get_height() + 3
                if y < pad:
                    return
                self._blit_outlined(face, row, right, y, color, alpha)
            y -= 7                                          # a breath between entries

    def _blit_outlined(self, font: Any, text: str, right: int, y: int,
                       color: tuple[int, int, int], alpha: float) -> None:
        """Draw `text` RIGHT-aligned at (right, y) with a dark outline and a whole-line alpha —
        readable over bright terrain and over night, with no background box (V4.14)."""
        a = max(0, min(255, int(255 * alpha)))
        body = font.render(text, True, color)
        body.set_alpha(a)
        shadow = font.render(text, True, _OVERLAY_SHADOW)
        shadow.set_alpha(int(a * 0.75))
        x = right - body.get_width()
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (2, 2)):
            self._screen.blit(shadow, (x + dx, y + dy))
        self._screen.blit(body, (x, y))

    def _draw_title_card(self) -> None:
        """A clean opening title card — project name + subtitle — fading in and back over the first
        few seconds, then gone for good. Showcase only."""
        if not self._showcase or self._start_time is None:
            return
        el = time.monotonic() - self._start_time
        if el > _TITLE_DUR:
            return
        fade = max(0.0, min(1.0, el / 1.0, (_TITLE_DUR - el) / 1.4))
        # V4.12: the card is COMPOSED ONCE (cached per viewport size) and only its surface alpha
        # moves per frame — it used to re-render the text and re-fill a full-screen surface every
        # frame, which is exactly the kind of per-frame rebuild that made the open look unstable.
        card = self._stamps.get(("titlecard", self._paint))
        if card is None:
            w, h = self._paint
            mx, my = w // 2, h // 2
            card = pygame.Surface(self._paint, pygame.SRCALPHA)
            card.fill((7, 9, 14, 165))
            tf, sf = (self._title_font or self._big_font or self._font), self._font
            if tf is not None:
                name = tf.render(_TITLE_NAME, True, _STORY_FG)
                card.blit(name, (mx - name.get_width() // 2, my - name.get_height()))
                pygame.draw.line(card, (*_STORY_ACCENT, 220), (mx - 90, my + 6), (mx + 90, my + 6), 2)
            if sf is not None:
                sub = sf.render(_TITLE_SUB, True, _STAT_VALUE)
                card.blit(sub, (mx - sub.get_width() // 2, my + 16))
            self._stamps[("titlecard", self._paint)] = card
        card.set_alpha(int(255 * fade))
        self._screen.blit(card, (0, 0))

    def _draw_showcase_readout(self, state: dict[str, Any]) -> None:
        """The one unobtrusive HUD element left in showcase: a small turn · phase chip, low-corner."""
        if self._font is None:
            return
        txt = f"turn {int(state.get('turn', 0))}  ·  {phase_name(self._phase)}"
        if not self._minimal:
            txt += "   ·   [U] hide UI"
        lab = self._font.render(txt, True, _STAT_VALUE)
        pad = 6
        x, y = self._map_px - lab.get_width() - 12, self._map_h - lab.get_height() - 10
        chip = pygame.Surface((lab.get_width() + 2 * pad, lab.get_height() + pad), pygame.SRCALPHA)
        chip.fill((*_HUD_BG, 120))
        self._screen.blit(chip, (x - pad, y - pad // 2))
        self._screen.blit(lab, (x, y))

    # -- V4.2: the story banner (queue + wall-clock playback + drawing) -----
    def _enqueue_banners(self, state: dict[str, Any]) -> None:
        """Queue plain-words announcements for THIS turn's MAJOR events (once per turn).

        Called from update() as each resolved turn arrives; a pure READ of the event tail.
        Idempotent per turn (guarded by the last-seen turn) and capped, so a busy turn can
        never flood the banner. Zero-delay/test paths never reach update(), so this is inert
        there — the banner is a watch-time-only overlay.
        """
        turn = int(state.get("turn", 0))
        if turn == self._banner_turn_seen:
            return
        self._banner_turn_seen = turn
        if self._showcase:
            self._direct_turn(state, turn)     # V4.15: severity, camera cuts and caption cards
            return
        notable = notable_names(state)
        now = time.monotonic()
        majors = [(banner_text(line), self._event_focus(line, state), event_color(line))
                  for line in turn_events(state.get("events") or [], turn)
                  if event_tier(line, notable) == "major"
                  and not (self._showcase and any(m in line for m in _SHOWCASE_MUTE))]
        if self._showcase:
            # V4.14: a turn where six towns all cross into the Neolithic is ONE beat, not six —
            # fold the repeats so the recording holds on the story rather than on a roll-call.
            majors = collapse_majors(majors)
        self._turn_majors = len(majors)                    # V4.14: drives this turn's dramatic pause
        for text, foc, color in majors:
            self._banner_queue.append(text)
            self._focus_queue.append(foc)                  # V4.10: where to point the camera
            if self._showcase:
                # V4.14: the floating feed carries MAJOR beats ONLY — the minor churn (trust
                # deltas, routine trades/talks) is suppressed outright in a recording.
                self._overlay_feed.append((text, now, color, color == _FEED_WAR))
        while len(self._banner_queue) > _BANNER_MAX_QUEUE:
            self._banner_queue.popleft()
            if self._focus_queue:
                self._focus_queue.popleft()

    # -- V4.15: the DIRECTOR — one cut plan per turn -------------------------
    def _direct_turn(self, state: dict[str, Any], turn: int) -> None:
        """Classify THIS turn and build its cut plan: severity, camera beats, caption cards.

        Showcase only, and a pure READ of the event tail. The director decides which events are
        worth stopping for; everything here just turns that decision into a queue the frame loop
        consumes. A turn with no beat leaves the queue empty, which is what makes it fast-forward.
        """
        lines = turn_events(state.get("events") or [], turn)
        events = _director.classify_turn(lines, self._seen_firsts, _director.crowned_names(state))
        self._turn_sev = _director.turn_severity(events)
        beats = _director.beats(events)
        self._turn_majors = len(beats)
        # A quiet RUN is what earns the harder compression: one dull turn is a beat of rhythm,
        # five in a row is a stretch of history the viewer should be flown over.
        self._quiet_run = 0 if beats else self._quiet_run + 1
        now = time.monotonic()
        self._beats.clear()
        for e in beats:
            title, sub = _director.caption(e)
            self._beats.append((e.severity, title, sub, self._beat_focus(e, state)))
            # The floating feed carries the same beats, in the director's words.
            self._overlay_feed.append((title, now, event_color(e.raw), e.severity == _director.LEGENDARY))
        if self._turns_total:
            self._turns_left = max(0, self._turns_total - turn)

    def _beat_focus(self, e: Any, state: dict[str, Any]) -> tuple | None:
        """The WORLD point a classified beat happened at — its settlement centre, else the first
        actor's cell, else None (the camera then stays wide). Pure read."""
        sets = state.get("settlements") or {}
        rec = sets.get(e.focus) if e.focus else None
        if isinstance(rec, dict) and rec.get("center"):
            return (float(rec["center"][0]), float(rec["center"][1]))
        for name in e.actors:
            for a in state.get("agents", []):
                if getattr(a, "alive", True) and getattr(a, "position", None) and a.name == name:
                    return (float(a.position[0]), float(a.position[1]))
        return None

    def _event_focus(self, line: str, state: dict[str, Any]) -> tuple | None:
        """V4.10: the WORLD point a major event happened at, for the showcase camera to frame — the
        first named settlement's centre, else the first named living agent's cell, else None. READ."""
        for sid, rec in (state.get("settlements") or {}).items():
            if sid in line and isinstance(rec, dict) and rec.get("center"):
                return (float(rec["center"][0]), float(rec["center"][1]))
        for a in state.get("agents", []):
            if getattr(a, "alive", True) and getattr(a, "position", None) and a.name in line:
                return (float(a.position[0]), float(a.position[1]))
        return None

    def _update_banner(self) -> None:
        """Advance the story banner on the wall clock: expire the active one, pop the next."""
        now = time.monotonic()
        if self._banner_text is None:
            if self._banner_queue:
                self._banner_text = self._banner_queue.popleft()
                self._banner_started = now
                self._punch_t = now       # V4.9: a gentle zoom-punch as the banner fires
                foc = self._focus_queue.popleft() if self._focus_queue else None
                if self._showcase and foc is not None:    # V4.10: the camera eases to & holds the event
                    hold = _BANNER_SECS_FAST if len(self._banner_queue) >= 2 else _BANNER_SECS
                    self._focus_pt = foc
                    self._focus_until = now + hold + _SHOWCASE_FOCUS_LEAD
        else:
            hold = _BANNER_SECS_FAST if len(self._banner_queue) >= 2 else _BANNER_SECS
            if now - self._banner_started >= hold:
                self._banner_text = None      # next frame pops the next queued banner

    def _draw_story_banner(self) -> None:
        """Draw the active story banner across the TOP of the map — the feature that makes the
        map self-explanatory without the side panel. Fades in/out; a warm accent bar on the
        left, big bold plain-words text centred in the band. UI overlay: never transformed."""
        if self._banner_text is None:
            return
        font = self._big_font or self._font
        if font is None:
            return
        screen, map_px = self._screen, self._map_px
        # A short fade in at the start and out at the end so banners don't pop.
        el = time.monotonic() - self._banner_started
        hold = _BANNER_SECS_FAST if len(self._banner_queue) >= 2 else _BANNER_SECS
        fade = min(1.0, el / 0.25, max(0.0, (hold - el) / 0.35))
        fade = max(0.0, min(1.0, fade))
        # Shrink to the regular face if the big text would overflow the map width.
        text = self._banner_text
        label = font.render(text, True, _STORY_FG)
        if label.get_width() > map_px - 60 and self._font is not None:
            font = self._font
            label = font.render(text, True, _STORY_FG)
        h = _STORY_H
        band = pygame.Surface((map_px, h), pygame.SRCALPHA)
        band.fill((*_STORY_BG, int(214 * fade)))
        pygame.draw.rect(band, (*_STORY_ACCENT, int(235 * fade)), (0, 0, 5, h))
        pygame.draw.line(band, (*_STORY_ACCENT, int(120 * fade)), (0, h - 1), (map_px, h - 1), 1)
        label.set_alpha(int(255 * fade))
        band.blit(label, ((map_px - label.get_width()) // 2, (h - label.get_height()) // 2))
        screen.blit(band, (0, 0))

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
        self._last_state = state
        self._update_juice()              # V4.9: set this frame's shake/punch BEFORE any projection
        # V4.10: FRAMING — the zoom-OUT floor tracks the inhabited region every frame (min zoom =
        # fit the towns, never the whole empty map), and the FIRST inhabited frame SNAPS the camera
        # onto the settlement centroid so the world opens on the action. HOME re-frames the same way.
        hx, hy, hcell = self._home_view(state)
        self._zoom_lo = hcell
        if not self._framed and state.get("settlements"):
            self._cam_x = self._cam_tx = hx
            self._cam_y = self._cam_ty = hy
            self._cam_cell = self._cam_tcell = hcell
            self._framed = True
        if self._showcase and state.get("settlements"):   # V4.10: auto-direct the camera hands-off
            self._showcase_direct(state)
        # Slice 11: advance the CAMERA first — glide toward its targets, clamp to the world,
        # and freeze this frame's shared transform, effective cell and LOD tier. Everything
        # below draws through them; nothing below does its own camera maths.
        self._update_camera()
        self._update_banner()             # V4.2: advance the story-banner queue (wall-clock)
        cell = self._cell
        # Slice 10: the DAY/NIGHT clock — the phase derives PURELY from the sim turn (made
        # fractional mid-walk so the light glides through the inter-turn animation rather
        # than stepping). Everything below reads these three derived values; nothing writes.
        turn_f = float(state.get("turn", 0))
        if motion is not None:
            turn_f += -1.0 + max(0.0, min(1.0, motion[1]))
        if self._showcase and self._start_time is not None:
            # V4.10: pace the day/night on the WALL CLOCK so a full dusk always arrives inside a
            # couple of minutes of footage, regardless of how fast the staged turns tick.
            self._phase = (0.08 + (time.monotonic() - self._start_time) / _SHOWCASE_DAY_SECS) % 1.0
        else:
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
        screen.fill(_FRAME_OUTER)  # base for the HUD/panel gutters
        # V4.9: the OUT-OF-BOUNDS ocean fills the map zone FIRST — the transparent-void terrain
        # bake then blits on top, so beyond the diamond the world reads as continuing into water.
        if self._void_bg is not None:
            screen.blit(self._void_bg, (0, 0))
        if self._terrain_bg is not None:
            self._blit_terrain()
        elif self._void_bg is None:
            grass_color = PALETTE.get(f"{self._current_season}_grass", _GRASS_BASE)
            screen.fill(grass_color, (0, 0, self._map_px, self._map_h))
        # V4.8: water shimmer & coast waves RE-SEATED onto the projected water — glints on the
        # tilted pond/sea tiles and wave crests along the projected coastline (both cull off-screen).
        self._draw_water_shimmer()
        self._draw_coast_waves()

        # Slice 5: settled land looks CULTIVATED — a translucent tilled-dirt field (now a
        # projected 2:1 ellipse), drawn under the territory + sprites. No-op if no settlements.
        self._draw_settlement_ground(state)
        # V4.6: cloud shadows (and clouds) drift in flat SCREEN space — re-seated onto the iso
        # ground plane in V4.8; drawing them now reads as grey ovals floating over the tilt.

        # V4.6: TERRITORY as PROJECTED GROUND OVERLAYS — realm-coloured 2:1 ellipses (distinct
        # outlined regions from V4.3), drawn on the ground UNDER every sprite. Returns the town
        # render infos for the painter pass below.
        towns = self._draw_territory(state)

        # V4.8: cloud shadows are GROUND DECALS projected onto the diamond, drawn UNDER the sprite
        # pass so buildings correctly occlude the shadow passing behind them; they slide in world space.
        self._draw_cloud_shadows()

        talkers = talkers_this_turn(state.get("events", []) or [], state.get("turn", 0))
        self._update_trails(state, motion)
        self._draw_trails()

        # V4.6: PAINTER'S ALGORITHM. Settlements, food and agents share ONE list sorted
        # back-to-front by projected ground depth (screen y), so nearer things correctly
        # overlap farther ones. Ground=terrain/farmland/territory (already drawn) -> sprites
        # here -> effects (grade/lights/banner) after. V4.7 hangs taller forms off this order.
        sprites: list[tuple[float, int, str, Any]] = []
        label_jobs: list[tuple[str, int, int, int]] = []
        # V4.7: a built-up town (MID/CLOSE) is EXPANDED into its individual iso VOLUMES — each
        # hut/hall/keep/tower/granary/well/palisade-post becomes its own sprite keyed by its
        # projected ground depth, so agents interleave with buildings and are correctly occluded
        # by the ones IN FRONT of them. Its flat ground (plaza/paths/shadows) is painted now, under
        # everything. FAR/tiny towns stay a single block-and-banner sprite (the strategy read).
        for info in towns:
            if self._lod != "far" and self._cell >= _TOWN_MIN_CELL:
                structs, top_y = self._emit_town(info, state)
                sprites.extend(structs)
                label_jobs.append((info["sid"], info["cx"], top_y, info["count"]))
            else:
                sprites.append((info["cy"], 0, "town", info))
        for fx, fy in state.get("food", []):
            px, py = self._to_px(fx, fy)
            if visible_on_screen(px, py, cell, self._cull, self._cull):
                sprites.append((py, 1, "food", (px, py)))
        for agent in state.get("agents", []):
            if not getattr(agent, "alive", True) or not getattr(agent, "position", None):
                continue
            gx, gy = self._agent_px(agent, motion)   # the agent's GROUND point (its feet)
            r = agent_radius(_wealth(agent), cell)
            if not visible_on_screen(gx, gy, r * 4 + cell, self._cull, self._cull):
                continue
            sprites.append((gy, 2, "agent", (agent, gx, gy, r, self._step_squash(agent, motion))))
        sprites.sort(key=lambda s: (s[0], s[1]))
        for _depth, _tie, kind, obj in sprites:
            if kind == "town":
                top_y = self._draw_town_sprite(obj, state)
                label_jobs.append((obj["sid"], obj["cx"], top_y, obj["count"]))
            elif kind == "struct":
                self._draw_structure(obj)
            elif kind == "food":
                px, py = obj
                if self._lod == "far":
                    pygame.draw.circle(screen, _FOOD, (px, py), max(1, cell // 5))
                else:
                    self._draw_wheat(px, py, cell)
            else:
                self._draw_agent_sprite(obj, talkers, state)
        # V4.17 (5.3): a fallen crown lies on the SEAT it belonged to — which means it lies in the
        # middle of a built-up town, where as a ground decal under the sprite pass it was hidden by
        # that town's own halls and label. Drawn OVER the sprites instead: legibility wins over
        # strict depth here, because the whole job of this object is to be noticed.
        self._draw_fallen_crowns()
        self._draw_settlement_labels(label_jobs)
        self._draw_puffs()                # V4.9: rise dust / collapse puffs, over the buildings

        # V4.9: the soft edge VIGNETTE over the map zone — depth, and no hard canvas edge where the
        # world meets the void. Under the day/night grade so night lights still pierce cleanly.
        if self._vignette is not None:
            screen.blit(self._vignette, (0, 0))

        # Slice 9/10: occasional birds (a daytime ambience — they roost as dusk falls), then
        # the full-scene grade. Slice 10 turns the static daylight tint into the day/night
        # cycle: the cached grade surface is REFILLED (never rebuilt) whenever the phase
        # tint moves, and blitted over the whole map zone (the HUD/panel stay ungraded).
        if self._dl > 0.35 and self._lod != "far":   # slice 11: no ambience on the strategy map
            self._draw_birds()                        # birds fly in the SKY — fine unprojected
            # V4.8: cloud PUFFS ride the sky above their ground shadows (lifted by _CLOUD_Z), so a
            # cloud and its shadow slide together across the tilted world.
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
        self._draw_emitters()             # V4.9: coins over a market / embers over a forge (close zoom)

        # Slice 8: the battle cinematic overlay (soldiers/dust/clash/fallen/banner) — drawn
        # ABOVE the grade since slice 10, so a night battle stays vivid and readable (the
        # clash flashes and the outcome banner never dim with the scene).
        if battle is not None:
            self._draw_battle_overlay(*battle)
        self._draw_flash()                # V4.9: the decisive-blow impact flash, over the clash

        self._draw_weather()

        # V4.2: the STORY BANNER rides on top of the map (above weather/cinematics) so a viewer
        # watching ONLY the map follows each major beat in plain words.
        self._draw_story_banner()

        if self._showcase:
            # V4.10: MINIMAL UI — just the banner + a small turn/phase chip (the minimap returns only
            # when the full UI is toggled back on with 'U'); the title card opens the run.
            if not self._minimal:
                self._draw_minimap(state)
            # V4.15: the legendary WASH goes under the text (it dims the world, not the caption),
            # then the caption card / quiet ticker ride on top of everything.
            self._draw_legendary_wash()
            self._draw_feed_overlay()         # V4.14: the event text floats over the world
            self._draw_showcase_readout(state)
            self._draw_caption_card()
            self._draw_quiet_ticker()
            self._draw_title_card()
        else:
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
        if t > 0.82:                          # V4.9: a small HOP as the agent lands on its new cell
            bob = max(bob, math.sin((t - 0.82) / 0.18 * math.pi) * self._cell * 0.14)
        return int(round(px + (cx - px) * e)), int(round(py + (cy - py) * e - bob))

    def _step_squash(self, agent: Any, motion: tuple[dict[str, tuple], float] | None) -> float:
        """V4.9: a subtle vertical squash-&-stretch factor over the walk cycle (>1 stretched at
        mid-stride, <1 squashed at the plant); 1.0 for a still agent (so a settled frame is neutral)."""
        if motion is None:
            return 1.0
        prev_pos, t = motion
        pp = prev_pos.get(agent.name)
        pos = agent.position
        if pp is None or (pp[0], pp[1]) == (pos[0], pos[1]) or t >= 1.0:
            return 1.0
        return 1.0 + 0.08 * math.sin(t * math.pi * 3.0)

    def _agent_world(self, agent: Any,
                     motion: tuple[dict[str, tuple], float] | None) -> tuple[float, float]:
        """The agent's WORLD-cell position, lerped from last turn's cell mid-walk (V4.8). The
        world-space twin of _agent_px — trails store this and project it each frame, so a path
        follows the ground under a pan instead of freezing at stale screen pixels."""
        pos = agent.position
        if motion is None:
            return float(pos[0]), float(pos[1])
        prev_pos, t = motion
        pp = prev_pos.get(agent.name)
        if pp is None or (pp[0], pp[1]) == (pos[0], pos[1]) or t >= 1.0:
            return float(pos[0]), float(pos[1])
        e = ease(t)
        return (pp[0] + (pos[0] - pp[0]) * e, pp[1] + (pos[1] - pp[1]) * e)

    def _visible_world_rect(self) -> tuple[float, float, float, float]:
        """The world-cell AABB currently on screen (V4.8): the four viewport corners un-projected
        to the ground plane. Weather/fog seed their particles inside this rect so they always fall
        WHERE the camera looks, at any pan/zoom, and stay tied to world positions as it moves."""
        view = self._view
        paint = self._paint                          # V4.14: seed over the whole PAINTED zone
        mp = self._map_px
        xs, ys = [], []
        for c in ((0, 0), (paint[0], 0), (0, paint[1]), paint):
            wx, wy = screen_to_world_iso(c, self._cam_draw, view)
            xs.append(wx)
            ys.append(wy)
        return (min(xs), min(ys), max(xs), max(ys))

    # -- V4.6: iso terrain geometry (base-diamond offsets shared by bake + blit) ------
    def _iso_base_offsets(self, cq: int) -> tuple[int, int, int, int]:
        """(OX, OY, W, H) for the base-diamond bake at bucket cell `cq` (pure geometry).

        Tile (cx, cy) bakes centred at base pixel (OX + (cx-cy)*cq, OY + (cx+cy)*cq/2). OX/OY
        shift the whole diamond into a positive canvas of size (W, H), with _ISO_ZPAD_CELLS
        rows of head-room above it for the building height V4.7 will hang here. All quantities
        scale linearly with `cq`, so the per-frame blit derives the screen origin from these.
        """
        lo, hi = -_MARGIN_CELLS, self._size + _MARGIN_CELLS
        hw, hh = cq, cq / 2.0
        umin, umax = (lo - (hi - 1)), ((hi - 1) - lo)      # range of (cx - cy)
        vmin, vmax = 2 * lo, 2 * (hi - 1)                  # range of (cx + cy)
        zpad = int(cq * _ISO_ZPAD_CELLS)
        ox = int(-umin * hw + hw)
        oy = int(-vmin * hh + hh + zpad)
        w = int((umax - umin) * hw + 2 * hw)
        h = int((vmax - vmin) * hh + 2 * hh + zpad)
        return ox, oy, w, h

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
        """Blit the VISIBLE slice of the cached ISO landscape through the camera (V4.6).

        The stated cached-surface strategy carries over: diamond bakes exist only at quantized
        integer-cell zoom buckets. The base-diamond's screen origin is derived analytically from
        _iso_base_offsets (the whole bake maps to screen by a uniform scale r=c/cq + a translate),
        so a settled camera (c == its bucket) blits the visible sub-rect 1:1, and only a mid-glide
        residual ratio needs one cheap pygame.transform.scale. The bake is never touched per frame.
        """
        screen, c = self._screen, self._cell
        vw, vh = self._view                          # V4.11: the map zone is a RECTANGLE (w, h)
        buckets = self._zoom_buckets or (self._cell0,)
        cq = min(buckets, key=lambda b: abs(b - c))
        surf = self._terrain_surface(cq)
        if surf is None:
            return
        ox, oy, W, H = self._iso_base_offsets(cq)
        r = c / cq
        ccx = (self._cam_x - self._cam_y) * c
        ccy = (self._cam_x + self._cam_y) * (c * 0.5)
        x0 = -ox * r - ccx + vw * 0.5                 # screen pos of base pixel (0, 0)
        y0 = -oy * r - ccy + vh * 0.5
        x0, y0 = self._fx(x0, y0)                      # V4.9: the terrain rides the same punch/shake
        r *= self._punch                              # ...as the sprites, so nothing desyncs
        pw, ph = self._paint                           # V4.14: clip to the FULL zone, not the viewport
        vx0, vy0 = max(0, int(x0)), max(0, int(y0))
        vx1, vy1 = min(pw, int(x0 + W * r)), min(ph, int(y0 + H * r))
        if vx1 <= vx0 or vy1 <= vy0:
            return
        if c == cq:                                    # resting ON the bucket: plain blit
            screen.blit(surf, (vx0, vy0), (int(vx0 - x0), int(vy0 - y0),
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
        map_px = map_px if map_px is not None else self._base_map   # V4.11: coast is BASE-space
        if m <= 0:
            return map_px
        row, t = divmod(max(0, y), max(1, cell))
        a = terrain_noise(row, 9, 91)
        b = terrain_noise(row + 1, 9, 91)
        wiggle = (a + (b - a) * (t / max(1, cell))) - 0.5
        return int(map_px - m * (0.55 + 0.35 * wiggle))

    def _tile_kind(self, cx: int, cy: int) -> str:
        """Classify world tile (cx, cy) for the iso bake: 'sea' / 'shallow' / 'sand' / 'land'
        (pure coordinate-hash read). The wilderness on the +x side is the SEA (a meandering
        diagonal coast), plus one deterministic off-centre POND inside the playable land."""
        size = self._size
        coast = size + (1 if terrain_noise(cy, 9, 91) > 0.62 else 0)   # meander the shoreline
        if cx > coast:
            return "sea"
        if cx == coast:
            return "shallow"
        # the pond: a small ellipse of tiles around a fixed off-centre spot in the land
        px, py = int(size * 0.28), int(size * 0.7)
        if size > 6 and ((cx - px) / max(1.0, size * 0.10)) ** 2 + \
                        ((cy - py) / max(1.0, size * 0.08)) ** 2 < 1.0:
            return "sea"
        if cx == coast - 1:
            return "sand"
        return "land"

    def _iso_elev(self, cx: float, cy: float) -> float:
        """A broad rolling ground HEIGHT in [0,1] (coordinate-hash; V4.6 elevation shading)."""
        s = _ISO_ELEV_CELLS
        a = terrain_noise(int(math.floor(cx / s)), int(math.floor(cy / s)), 6)
        b = terrain_noise(int(math.floor(cx / s + 0.5)), int(math.floor(cy / s + 0.5)), 7)
        return 0.5 * a + 0.5 * b

    def _build_terrain(self, cell: int) -> Any:
        """V4.6: bake the FULL-BLEED landscape for one zoom bucket as ISOMETRIC DIAMOND TILES.

        Each world tile (cx, cy) in [-margin, size+margin) is a 2:1 diamond; its colour comes
        from grass value-noise PLUS ELEVATION SHADING — a broad coordinate-hash height field
        (`_iso_elev`), higher ground lighter and the sun-facing (top/NW) slopes lifted — so the
        ground reads as rolling form, not a flat plane. The +x wilderness is SEA (sand -> shallow
        -> water) with a meandering coast, one pond sits in the land, the wilderness ring darkens,
        and trees/rocks are planted back-to-front on land tiles. 100% terrain_noise (never the sim
        RNG); cached per bucket and blitted each frame (never reprojected). z stays 0 this slice.
        """
        size = self._size
        if size <= 0 or cell <= 0:
            return None
        ox, oy, W, H = self._iso_base_offsets(cell)
        if W <= 0 or H <= 0:
            return None
        # V4.9: TRANSPARENT beyond the diamond so the out-of-bounds ocean (_draw_void) shows
        # through the bbox corners and the world reads as continuing into open water.
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        grass = PALETTE.get(f"{self._current_season}_grass", _GRASS_BASE)
        hw, hh = cell, cell * 0.5
        lo, hi = -_MARGIN_CELLS, size + _MARGIN_CELLS

        # 1) TILE FILL: one 2:1 diamond per world cell, coloured by noise + elevation shading.
        for cy in range(lo, hi):
            for cx in range(lo, hi):
                bx = ox + (cx - cy) * hw
                by = oy + (cx + cy) * hh
                kind = self._tile_kind(cx, cy)
                if kind == "sea":
                    col = _WATER
                elif kind == "shallow":
                    col = _WATER_SHALLOW
                elif kind == "sand":
                    col = _SAND
                else:
                    fine = terrain_noise(cx * 2, cy * 2, 1) - 0.5
                    patch = terrain_noise(cx // 2, cy // 2, 2) - 0.5
                    e = self._iso_elev(cx, cy)
                    shade = int(fine * 2 * _GRASS_VAR + patch * 2 * _GRASS_PATCH
                                + (e - 0.5) * _ISO_ELEV * 40)
                    if not (0 <= cx < size and 0 <= cy < size):  # wilderness fringe: darker
                        shade += int(-12 + fine * 14)
                    col = _shade(grass, shade)
                # oversize the diamond by 1px so tessellation leaves no seams
                pts = [(bx, by - hh - 1), (bx + hw + 1, by), (bx, by + hh + 1), (bx - hw - 1, by)]
                pygame.draw.polygon(surf, col, pts)

        # 2) FEATURES: trees + rocks on land tiles, planted BACK-TO-FRONT (increasing cx+cy) so a
        #    nearer tree correctly overlaps one behind it. Circles are rotation-free — they read
        #    fine in iso; their baked shadows fall to the bottom-right like the rest of the scene.
        for depth in range(2 * lo, 2 * (hi - 1) + 1):
            for cx in range(lo, hi):
                cy = depth - cx
                if not (lo <= cy < hi) or self._tile_kind(cx, cy) != "land":
                    continue
                bx = ox + (cx - cy) * hw
                by = oy + (cx + cy) * hh
                fringe = not (0 <= cx < size and 0 <= cy < size)
                if terrain_noise(cx, cy, 4) > (_TREE_THRESHOLD_WILD if fringe else _TREE_THRESHOLD):
                    self._build_tree(surf, int(bx), int(by), cell)
                elif terrain_noise(cx, cy, 5) > _ROCK_THRESHOLD:
                    self._build_rock(surf, int(bx), int(by), cell)
        return surf

    def _build_grade(self) -> Any:
        """The cached full-scene GRADE surface (slice 10: refilled — never rebuilt — whenever
        the interpolated phase tint changes; it starts on the current phase's tint)."""
        if self._map_px <= 0:
            return None
        grade = pygame.Surface(self._paint, pygame.SRCALPHA)
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
        wash = pygame.Surface(self._paint, pygame.SRCALPHA)
        strips = 28
        sw = max(1, self._map_px // strips + 1)
        gold = PALETTE["dawn_gold"]
        for i in range(strips):
            a = int(_DAWN_WASH_MAX_A * (1.0 - i / strips) ** 1.6)
            if a > 0:
                wash.fill((*gold, a), (i * sw, 0, sw, self._map_h))
        return wash

    def _build_stars(self) -> list[tuple[int, int, int]]:
        """Keep only the star_field candidates that land on WATER (the sea past the coast, or
        inside the pond): the top-down night sky appears as REFLECTIONS, so the land stays
        readable. Deterministic per map size — the renderer twinkles them per frame."""
        out: list[tuple[int, int, int]] = []
        for x, y, s in star_field(self._base_map):
            if self._margin_px > 0 and x > self._coast_x(y) + 5 and x < self._base_map - 2:
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
        screen, f, view = self._screen, self._frame, self._cull
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
        # V4.5: every emitter casts an ADDITIVE radial POOL so a town reads as a cluster of warm
        # pools in the blue dark; a small bright core sits at each source. V4-fix: the POOLS
        # accumulate on ONE offscreen layer that is CLAMPED to a warm cap (_LIGHT_ACC_CAP) before it
        # composites, so overlapping pools brighten a wall face toward AMBER and never blow to white.
        if not self._frame_lights:
            return
        # The pools accumulate at HALF resolution (they are smooth gradients, so the 2x upscale is
        # invisible) and only over the lights' UNION rect — so the clamp/composite cost tracks the lit
        # region at a quarter of the pixels, keeping a close-up night town inside the frame budget.
        ds = _LIGHT_DS
        hvw, hvh = (self._map_px + ds - 1) // ds, (self._map_h + ds - 1) // ds
        layer = self._light_layer
        if layer is None or layer.get_size() != (hvw, hvh):
            layer = self._light_layer = pygame.Surface((hvw, hvh))
        _pool_r = {"window": _LIGHT_WINDOW, "hearth": _LIGHT_HEARTH, "forge": _LIGHT_FORGE,
                   "gleam": _LIGHT_GLEAM}
        dirty = None
        for kind, x, y, s in self._frame_lights:
            rr = int(max(6, s * _pool_r.get(kind, _LIGHT_TORCH))) + 4
            rect = pygame.Rect((x - rr) // ds, (y - rr) // ds, 2 * rr // ds + 2, 2 * rr // ds + 2)
            dirty = rect if dirty is None else dirty.union(rect)
        dirty = dirty.clip(pygame.Rect(0, 0, hvw, hvh))
        if dirty.width <= 0 or dirty.height <= 0:
            return
        layer.fill((0, 0, 0), dirty)
        for kind, x, y, s in self._frame_lights:   # pools -> half-res layer; bright cores -> screen
            hx, hy = x // ds, y // ds
            if kind == "window":                           # lit windows — the towns twinkle
                fl = 0.82 + 0.18 * terrain_noise(f // 3, x * 7 + y * 3, 67)
                self._blit_light(hx, hy, max(4, s * _LIGHT_WINDOW / ds), PALETTE["window_glow"],
                                 0.55 * nf * fl, target=layer)
                core = self._soft_stamp(max(1, (s + 1) // 2), PALETTE["window_lit"],
                                        _q8(170 * nf * fl))
                screen.blit(core, (x - core.get_width() // 2, y - core.get_height() // 2))
            elif kind == "hearth":                         # a town hearth over the plaza
                fl = 0.86 + 0.14 * terrain_noise(f // 4, x * 3 + y, 70)
                self._blit_light(hx, hy, max(5, int(s * _LIGHT_HEARTH / ds)), PALETTE["hearth_glow"],
                                 0.5 * nf * fl, target=layer)
            elif kind == "forge":                          # a metallurgy town's forge — hot, pulsing
                fl = 0.62 + 0.38 * terrain_noise(f // 2, x * 5 + y * 2, 78)
                self._blit_light(hx, hy, max(4, int(s * _LIGHT_FORGE / ds)), PALETTE["forge_glow"],
                                 0.7 * nf * fl, target=layer)
                pygame.draw.circle(screen, PALETTE["forge_core"], (x, y), max(1, s // 4))
            elif kind == "gleam":                          # a fallen crown on the night grass
                # Halo ONLY — no core, no ring. The shape must win over the glow, so this does the
                # single job of lifting the gold off dark ground and nothing else.
                self._blit_light(hx, hy, max(3, int(s * _LIGHT_GLEAM / ds)), PALETTE["torch_flame"],
                                 _LIGHT_GLEAM_STRENGTH * nf, target=layer)
            else:                                          # torchlight at the seats of power
                fl = 0.70 + 0.30 * terrain_noise(f // 2, x * 5 + y, 68)
                wob = int(round(terrain_noise(f // 2, x, 69) * 2 - 1))
                self._blit_light(hx, (y + wob) // ds, max(4, int(s * _LIGHT_TORCH / ds)),
                                 PALETTE["torch_flame"], 0.75 * nf * fl, target=layer)
                pygame.draw.circle(screen, PALETTE["torch_core"], (x, y + wob), max(1, s // 4))
                pygame.draw.circle(screen, PALETTE["torch_flame"], (x, y + wob),
                                   max(2, s // 3), 1)
        # CLAMP the accumulated pools to a warm cap (per-channel MIN), then upscale the lit rect back
        # and composite additively — the halo below the cap is untouched, over-bright overlap becomes
        # amber, and the 2x scale softens the pools (a bonus). One scale + one add over the lit region.
        layer.fill(_LIGHT_ACC_CAP, dirty, special_flags=pygame.BLEND_RGB_MIN)
        up = pygame.transform.scale(layer.subsurface(dirty),
                                    (dirty.width * ds, dirty.height * ds))
        screen.blit(up, (dirty.x * ds, dirty.y * ds), special_flags=pygame.BLEND_RGB_ADD)

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

    def _light_stamp(self, radius: float, color: tuple, intensity: float) -> Any:
        """V4.5: a cached RADIAL-GRADIENT light pool (bright warm centre -> transparent edge),
        meant to be blitted with pygame.BLEND_RGB_ADD so pools bloom and overlap additively.

        Cost stays flat: the surface is keyed by (snapped radius, colour, quantized intensity),
        so the handful of distinct lights on screen reuse a small bounded set of baked pools.
        The gradient is baked as RGB (alpha is ignored under additive blend): concentric filled
        circles from the rim inward, brightness rising as (1 - t)**falloff toward the centre.
        """
        r = max(2, int(round(radius / _LIGHT_RADIUS_Q)) * _LIGHT_RADIUS_Q)
        iq = max(0, min(_LIGHT_INTENSITY_STEPS,
                        int(round(intensity * _LIGHT_INTENSITY_STEPS))))
        if iq == 0:
            return None
        key = ("light", r, color, iq)
        stamp = self._stamps.get(key)
        if stamp is None:
            scale = (iq / _LIGHT_INTENSITY_STEPS) * (_LIGHT_MAX_A / 255.0)
            d = 2 * r + 2
            stamp = pygame.Surface((d, d), pygame.SRCALPHA)
            c = r + 1
            steps = max(6, r)
            for i in range(steps, 0, -1):
                t = i / steps                        # 1 at the rim, -> 0 at the centre
                rr = max(1, int(round(r * t)))
                b = ((1.0 - t) ** _LIGHT_FALLOFF) * scale
                col = (min(255, int(color[0] * b)),
                       min(255, int(color[1] * b)),
                       min(255, int(color[2] * b)))
                pygame.draw.circle(stamp, (*col, 255), (c, c), rr)
            self._stamps[key] = stamp
        return stamp

    def _blit_light(self, x: int, y: int, radius: float, color: tuple, intensity: float,
                    target: Any = None) -> None:
        """Blit one additive light pool centred at (x, y). No-op below the intensity floor. Town
        pools accumulate onto a clamped light LAYER (`target`); a lone clash flash may add direct."""
        stamp = self._light_stamp(radius, color, intensity)
        if stamp is not None:
            (target if target is not None else self._screen).blit(
                stamp, (x - stamp.get_width() // 2, y - stamp.get_height() // 2),
                special_flags=pygame.BLEND_RGB_ADD)

    def _draw_water_shimmer(self) -> None:
        """V4.8: ripple glints re-seated onto the PROJECTED water tiles — the off-centre pond inside
        the land and the +x sea margin (the same world tiles _tile_kind classifies as water). Each
        glint is a world point projected through the shared iso transform (so it lands on the tilted
        water at any zoom), drawn as a short horizontal light dash re-hashed every ~1/3s (zero RNG)."""
        if self._lod == "far":
            return
        f, view, cell, size = self._frame, self._cull, self._cell, self._size
        pcx, pcy = size * 0.28, size * 0.7                # the pond's world centre (see _tile_kind)
        for k in range(3):                                # glints on the pond
            u = terrain_noise(f // 18, k, 71) - 0.5
            v = terrain_noise(f // 18, k, 72) - 0.5
            if (u * 1.2) ** 2 + (v * 1.1) ** 2 < 0.20:
                x, y = self._to_px(pcx + u * size * 0.20, pcy + v * size * 0.16)
                if visible_on_screen(x, y, 12, view, view):
                    w = max(2, int(cell * (0.25 + 0.25 * terrain_noise(f // 18, k, 73))))
                    pygame.draw.line(self._screen, _WATER_HI, (x - w, y), (x + w, y), 1)
        if self._margin_px > 0:                           # glints on the open +x sea
            for k in range(5):
                wy = terrain_noise(f // 22, k, 74) * size
                wx = size + 0.5 + terrain_noise(f // 22, k, 75) * max(0.5, _MARGIN_CELLS - 0.8)
                x, y = self._to_px(wx, wy)
                if not visible_on_screen(x, y, 12, view, view):
                    continue
                w = max(2, int(cell * (0.25 + 0.25 * terrain_noise(f // 22, k, 76))))
                pygame.draw.line(self._screen, _WATER_HI, (x - w, y), (x + w, y), 1)

    def _draw_birds(self) -> None:
        """The occasional bird crossing the sky: tiny v-shapes, flapping — pure frame+hash."""
        for x, y, spread in ambient_birds(self._frame, self._map_px):
            xi, yi, s = int(x), int(y), max(1.5, spread)
            pygame.draw.line(self._screen, _BIRD, (int(xi - s), int(yi - s * 0.6)), (xi, yi), 1)
            pygame.draw.line(self._screen, _BIRD, (xi, yi), (int(xi + s), int(yi - s * 0.6)), 1)

    def _build_void(self) -> Any:
        """V4.9: the cached OUT-OF-BOUNDS ocean for the map zone — a radial gradient, deep open
        sea (_VOID_OCEAN) at the centre fading to near-black (_VOID_EDGE) at the frame corners.

        Built once per window size and blitted behind the transparent-void terrain bake, so the
        diamond world sits in open water that darkens outward — no diamond island on flat ground.
        Concentric filled circles (large/dark first, small/light last) give the smooth falloff.
        """
        w, h = self._paint                             # V4.11: rectangular map zone (V4.14: full-bleed)
        if w <= 0 or h <= 0:
            return None
        surf = pygame.Surface((w, h))
        surf.fill(_VOID_EDGE)                          # the far dark past the outermost circle
        cx, cy = w // 2, h // 2
        maxr = int(math.hypot(w, h) * 0.5 + 2)         # reach to the corners
        steps = 48
        for i in range(steps, 0, -1):
            t = i / steps                              # 1 at the rim, ->0 at the centre
            pygame.draw.circle(surf, lerp_color(_VOID_OCEAN, _VOID_EDGE, t), (cx, cy), int(maxr * t))
        self._dress_void(surf, w, h)                   # V4.10: distant sea texture + corner haze
        return surf

    def _dress_void(self, surf: Any, w: int, h: int) -> None:
        """V4.10: make the open water around the coast read INTENTIONAL, not empty background.

        Baked ONCE into the cached void (deterministic terrain_noise, zero RNG): distant WAVE crests
        (short lighter dashes on a jittered grid), a sparse scatter of FOAM flecks, and a faint far-
        SHORELINE haze hugging the extreme corners (a hint of distant land in the mist). Only the ring
        the diamond does not cover is dressed — the inner disc (where the world sits) is skipped, so
        the cost is a handful of thin lines over the corners and nothing shows through the land."""
        cxc, cyc, rref = w / 2.0, h / 2.0, min(w, h)
        inner = (rref * 0.30) ** 2                      # the central disc the world diamond covers
        step = max(16, min(w, h) // 24)
        for gy in range(step // 2, h, step):
            for gx in range(step // 2, w, step):
                dx, dy = gx - cxc, gy - cyc
                if dx * dx + dy * dy < inner:
                    continue                            # skip where the land diamond will sit
                if terrain_noise(gx, gy, 401) < 0.40:
                    continue                            # sparse — open water, not a striped pool
                jx = int((terrain_noise(gx, gy, 402) - 0.5) * step)
                jy = int((terrain_noise(gx, gy, 403) - 0.5) * step)
                x = max(2, min(w - 3, gx + jx))
                y = max(2, min(h - 3, gy + jy))
                base = surf.get_at((x, y))[:3]          # shade RELATIVE to the local sea depth
                ln = max(3, int(step * (0.30 + 0.40 * terrain_noise(gx, gy, 404))))
                pygame.draw.line(surf, _shade(base, 15), (x - ln, y + 1), (x + ln, y), 1)
                pygame.draw.line(surf, _shade(base, 7), (x - ln + 2, y + 3), (x + ln - 2, y + 2), 1)
                if terrain_noise(gx, gy, 405) > 0.88:   # an occasional foam fleck on a crest
                    pygame.draw.circle(surf, _shade(PALETTE["foam"], -70), (x + ln // 2, y), 1)
        # far-shoreline haze: a faint warm-grey bloom in each extreme corner (distant land in mist).
        haze = pygame.Surface((w, h), pygame.SRCALPHA)
        hr = int(min(w, h) * 0.22)
        for cxh, cyh in ((0, 0), (w, 0), (0, h), (w, h)):
            for i in range(hr, 0, -max(1, hr // 12)):
                a = int(22 * (1 - i / hr))              # faint at radius hr, strongest at the corner
                pygame.draw.circle(haze, (*PALETTE["fog"], a), (cxh, cyh), i)
        surf.blit(haze, (0, 0))

    def _build_vignette_overlay(self) -> Any:
        """V4.9: the cached soft edge VIGNETTE for the map zone (SRCALPHA) — transparent in the
        middle, darkening toward the frame edges so the viewport has depth and no hard canvas edge.

        Nested rectangular rings (each drawn once at its own alpha, so nothing double-darkens)
        keep the whole border soft, not just the corners; blitted over the map each frame."""
        w, h = self._paint
        if w <= 0 or h <= 0:
            return None
        vign = pygame.Surface((w, h), pygame.SRCALPHA)
        rings = 30
        band = max(1, min(w, h) // (rings * 2))
        for i in range(rings):
            a = int(_VIGNETTE_MAX * (1 - i / rings) ** 2)   # strongest at the outer edge, ->0 inward
            inset = i * band
            if a > 0 and w - 2 * inset > 0 and h - 2 * inset > 0:
                pygame.draw.rect(vign, (0, 0, 0, a),
                                 (inset, inset, w - 2 * inset, h - 2 * inset), max(1, band))
        return vign

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
        overlay = pygame.Surface(self._paint, pygame.SRCALPHA)
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
            # V4.6: a ground circle projects to an axis-aligned 2:1 ellipse (rx horizontal).
            rx = max(cell, int(rad * _ISO_RX))
            ry = max(2, int(rx * 0.5))
            if not visible_on_screen(cx, cy, rx + cell, self._cull, self._cull):
                continue                                         # slice 11: cull off-screen fields
            pygame.draw.ellipse(overlay, (*_FARMLAND, _FARMLAND_ALPHA),
                                (cx - rx, cy - ry, 2 * rx, 2 * ry))
            # Furrows clipped to the ELLIPSE; spacing scales with the field so the tuft count
            # per frame stays bounded (the ploughed-field read survives at any radius/zoom).
            step = max(2, cell // 2, ry // 9)
            crop_dx = max(4, cell // 2, rx // 18)
            for fy in range(cy - ry + step, cy + ry, step):      # furrows, clipped to the ellipse
                half = int(rx * (max(0.0, 1.0 - ((fy - cy) / ry) ** 2)) ** 0.5)
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
    def _draw_agent_figure(self, cx: int, cy: int, r: int, color: tuple[int, int, int],
                           squash: float = 1.0, rank: str | None = None) -> int:
        """Draw a little person (head circle + trapezoid body) centred on (cx, cy).

        Colour is the personality colour and the whole figure scales with `r` (wealth), so
        slice-1's two encodings survive the upgrade. Returns the y of the figure's TOP (where
        a crown/star/bubble is stacked). A tiny cell (r below _FIGURE_MIN_R) falls back to the
        slice-1 dot so it never collapses into noise. V4.9: `squash` (>1 stretch, <1 squash) scales
        the figure's HEIGHT about its feet with an inverse width — a 1px step spring. squash==1.0 is
        byte-identical to the pre-juice figure.

        V4.17 RANK SILHOUETTE: `rank` (see `agent_role`) changes the figure's SHAPE, not just what
        is stamped over its head. A ruler stands TALLER, carries BROADER shoulders, and wears a
        CLOAK that flares past its feet; the higher the rank the more of each. The point is that
        rank survives being small, dim, or half-behind a roof — you can pick the king out of a
        crowd by outline alone, with the insignia only confirming what the shape already said.
        `rank=None` (a commoner) is byte-identical to the pre-V4.17 figure.
        """
        if r < _FIGURE_MIN_R:
            pygame.draw.circle(self._screen, _OUTLINE, (cx, cy), r + 1)
            pygame.draw.circle(self._screen, color, (cx, cy), r)
            return cy - r
        screen = self._screen
        hs, ws, robe = _RANK_SILHOUETTE.get(rank or "", (1.0, 1.0, None))
        base = cy + r                                # the feet — the squash pivots here
        # Rank lifts the whole figure about its feet, exactly as the juice spring does, so the two
        # compose: a king mid-hop is a tall figure hopping, not a commoner-shaped one.
        stretch = squash * hs
        vy = lambda y: int(round(base - (base - y) * stretch))
        head_r = max(2, round(r * 0.6))
        hx, hy = cx, vy(cy - head_r)                 # head sits in the upper half of the cell
        # Inverse width keeps the SPRING's volume ~constant, then rank widens the shoulders on top
        # of it — a ruler is broader, and must not be thinned by standing taller.
        bw = max(2, round(r * 0.95 * ws / max(0.6, squash)))
        top, bot = vy(cy - 1), base                  # body spans from under the head to the base
        if robe is not None:
            self._draw_robe(cx, top, bot, bw, color, robe)
        body = [(cx - round(bw * 0.5), top), (cx + round(bw * 0.5), top),
                (cx + bw, bot), (cx - bw, bot)]
        pygame.draw.polygon(screen, color, body)
        pygame.draw.polygon(screen, _OUTLINE, body, 1)
        pygame.draw.circle(screen, color, (hx, hy), head_r)
        pygame.draw.circle(screen, _OUTLINE, (hx, hy), head_r, 1)
        return hy - head_r

    def _draw_robe(self, cx: int, top: int, bot: int, bw: int,
                   color: tuple[int, int, int], kind: str) -> None:
        """The cloak (a king/emperor) or short mantle (a monarch/lord) BEHIND a ruler's figure.

        Drawn first, so the body sits on top of it and the flare reads as cloth hanging behind the
        shoulders. A deeper shade of the wearer's own colour, so the robe never fights the
        allegiance/personality read the body carries — it only broadens the silhouette.
        """
        # Deep enough that the body still reads as a body ON the cloth rather than merging with it
        # into one solid cone — the first pass flared to 2x the body width in a near-body shade,
        # and the emperor came out as a red triangle with a head on top.
        deep = _shade(color, -62)
        if kind == "cloak":                       # full length: shoulders to past the feet
            flare, hem = bw * 1.5, bot + max(1, (bot - top) // 8)
            pts = [(cx - round(bw * 0.62), top), (cx + round(bw * 0.62), top),
                   (cx + round(flare), hem), (cx - round(flare), hem)]
        else:                                     # mantle: a short cape over the shoulders only
            flare, hem = bw * 1.24, top + max(2, (bot - top) // 2)
            pts = [(cx - round(bw * 0.6), top), (cx + round(bw * 0.6), top),
                   (cx + round(flare), hem), (cx - round(flare), hem)]
        pygame.draw.polygon(self._screen, deep, pts)
        pygame.draw.polygon(self._screen, _OUTLINE, pts, 1)

    def _draw_role_marker(self, cx: int, top_y: int, r: int, role: str | None) -> None:
        """Stamp a ruler's insignia just above a figure: leader STAR, crowns for the crowned.

        V4.17: the insignia now only CONFIRMS what the silhouette already said (see
        `_RANK_SILHOUETTE`), so it grades with the ladder — an emperor's double crown, a king's
        wide one, a local monarch's plain one. A LORD wears no crown: he holds his seat from a
        king, and his mantle is the whole of his claim.
        """
        if role is None:
            return
        gap = max(2, r // 3)
        base = top_y - gap
        if role == "leader":
            self._draw_star(cx, base - max(3, r // 2), max(3, r * 0.7))
        elif role == "monarch":
            self._draw_crown(cx, base, max(4, r), double=False)
        elif role == "king":
            self._draw_crown(cx, base, max(5, int(r * 1.1)), double=False)
        elif role == "emperor":
            self._draw_crown(cx, base, max(5, int(r * 1.2)), double=True)

    def _track_crowns(self, state: dict[str, Any]) -> None:
        """V4.17 (5.3): notice a crown LEAVING a head, and drop it on the ground (pure read).

        Seats are settlement monarchies (`monarchs[sid]["monarch"]`), which is the one crown
        identity that survives its holder — a kingdom is keyed by the KING'S OWN NAME, so when the
        king changes the key changes with him and there is nothing left to diff against. A king or
        emperor holds his home seat too (kingdoms.py `_king_home`), so every crown in the world is
        covered by this one diff.

        A crown only FALLS when its holder DIED. A monarch who was deposed and lived had his crown
        taken from him — there is nothing lying on the grass — and the drama of that is the coup,
        which the director already frames. A seat that refills claims whatever crown was lying for
        it, which is what makes a succession read as somebody picking the thing up.

        It falls on the SEAT'S settlement centre, not where the body fell. The crown is the
        institution, not the man: a crown lying in open country where a king happened to die is
        unreadable, while a crown lying on the throne it belongs to says "this town has no ruler"
        instantly. Where he actually died stays truthful in the event log — only the symbol moves.
        """
        seats = {sid: rec["monarch"] for sid, rec in (state.get("monarchs") or {}).items()
                 if isinstance(rec, dict) and rec.get("monarch")}
        now = time.monotonic()
        sets = state.get("settlements") or {}
        alive = {a.name for a in state.get("agents", []) if getattr(a, "alive", True)}
        for sid, holder in self._crown_seats.items():
            if seats.get(sid) == holder or holder in alive:
                continue                      # still reigning, or deposed alive: no crown falls
            rec = sets.get(sid)
            centre = rec.get("center") if isinstance(rec, dict) else None
            if centre is not None:            # a seat whose town is gone has no throne to lie on
                self._fallen_crowns.append({"sid": sid, "who": holder, "born": now,
                                            "pos": (float(centre[0]), float(centre[1])),
                                            "taken": None})
        for c in self._fallen_crowns:         # a refilled seat CLAIMS the crown lying for it
            if c["taken"] is None and seats.get(c["sid"]):
                c["taken"] = now
        self._crown_seats = seats

    def _crown_alpha(self, c: dict[str, Any], now: float) -> float:
        """A fallen crown's opacity: solid while it lies, fading once claimed or forgotten (pure)."""
        if c["taken"] is not None:
            return 1.0 - (now - c["taken"]) / _CROWN_FADE
        return 1.0 - max(0.0, (now - c["born"] - _CROWN_LIE_SECS) / _CROWN_FADE)

    def _draw_fallen_crowns(self) -> None:
        """Draw the crowns lying on the grass — a ground decal, under every sprite.

        Flattened to `_CROWN_FLAT` so it reads as lying on its side rather than standing up with
        nobody underneath, with a shadow at its foot and a slow glint so the eye finds it in a
        wide frame. Drawn with the ground decals, so a villager standing in front of a vacant
        crown correctly hides it — which is the right story: the throne is empty, life goes on.
        """
        if not self._fallen_crowns:
            return
        now = time.monotonic()
        cell = self._cell
        live = []
        for c in self._fallen_crowns:
            alpha = self._crown_alpha(c, now)
            if alpha <= 0.0:
                continue                      # forgotten or claimed: gone for good
            live.append(c)
            sx, sy = self._to_px(*c["pos"])
            # The seat's centre is the ground point the town's HALL rises from, so a crown drawn
            # there — over the sprites, where it has to be to stay visible — reads as an ornament
            # sitting on a roof. Nudged forward (down-screen) onto the plaza in front of the hall,
            # where it rests on open ground and reads as an object lying at the foot of the throne.
            sy += int(cell * 0.95)
            w = max(4, int(cell * 0.44))
            h = max(3, int(w * _CROWN_FLAT))
            if not visible_on_screen(sx, sy, w * 3, self._cull, self._cull):
                continue
            # A vacant crown GLEAMS in the dark. Registered with the point-light system (which the
            # night pass consumes later this frame), because night-muting the gold like ordinary
            # terrain sank it into the grass — and the one object on the field that must never be
            # missed is the one saying the throne is empty. A 'gleam' is a HALO ONLY (see
            # _LIGHT_GLEAM): a torch's hot core out-shouted the crown it was meant to reveal.
            self._frame_lights.append(("gleam", sx, sy, max(4, w)))
            pad = w * 2
            surf = pygame.Surface((pad * 2, pad * 2), pygame.SRCALPHA)
            ox = oy = pad
            # A soft shadow where it rests, then the crown itself squashed onto the ground plane.
            pygame.draw.ellipse(surf, (*_OUTLINE, 90),
                                (ox - w, oy - h // 2, w * 2, max(2, h)))
            pts = [(ox - w, oy), (ox - w, oy - h),
                   (ox - w // 2, oy - h // 3), (ox, oy - h),
                   (ox + w // 2, oy - h // 3), (ox + w, oy - h), (ox + w, oy)]
            # Only half-muted at night: the crown is lit by its own gleam, not by the sun.
            gold = lerp_color(_CROWN, night_mute(_CROWN, self._nf), 0.45)
            pygame.draw.polygon(surf, gold, pts)
            pygame.draw.polygon(surf, _OUTLINE, pts, 1)
            # The glint: the gold catches the light on a slow cycle, so a vacant crown draws the
            # eye in a wide frame without ever flashing.
            g = 0.5 + 0.5 * math.sin((self._frame + hash(c["who"]) % 97) * (2 * math.pi / _CROWN_GLINT_PERIOD))
            pygame.draw.line(surf, _shade(gold, int(40 + 55 * g)),
                             (ox - w, oy - h), (ox - w // 2, oy - h // 3), 1)
            surf.set_alpha(int(255 * max(0.0, min(1.0, alpha))))
            self._screen.blit(surf, (sx - pad, sy - pad))
        self._fallen_crowns = live

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

    def _draw_territory(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """V4.6: draw each settlement's TERRITORY as a projected GROUND OVERLAY and return the
        town render infos for the caller's painter pass (READ only).

        A world circle of members' spread projects to an axis-aligned 2:1 ELLIPSE; realm colour
        (emperor > king > lone monarch; the aftermath lerp mid-conquest) fills it very faintly
        with a strong outlined edge (the V4.3 distinct-region read, now on the tilted ground).
        Fills are drawn first, then all edges, so overlapping realms never blur into a blob.
        """
        settlements = state.get("settlements")
        if not settlements:
            return []
        cell, map_px = self._cell, self._map_px
        pos_by_name = {
            a.name: a.position
            for a in state.get("agents", [])
            if getattr(a, "alive", True) and getattr(a, "position", None) is not None
        }
        overlay = pygame.Surface(self._paint, pygame.SRCALPHA)
        towns: list[dict[str, Any]] = []
        regions: list[tuple] = []
        for sid in sorted(settlements):
            rec = settlements[sid]
            center = rec.get("center")
            if center is None:
                continue
            members = rec.get("members") or ()
            member_positions = [pos_by_name[n] for n in members if n in pos_by_name]
            living = len(member_positions)               # V4-fix: only LIVING members size the town
            rad_cells = settlement_radius_cells(center, member_positions)
            self._note_population(sid, living, center, rad_cells)   # V4.9: rise dust / collapse puff
            rx = max(cell, int(round(rad_cells * cell * _ISO_RX)))   # projected ellipse radii
            ry = max(2, rx // 2)
            cx, cy = self._to_px(int(center[0]), int(center[1]))
            if not visible_on_screen(cx, cy, rx + cell * 3, self._cull, self._cull):
                continue
            owner = settlement_realm(sid, state)
            tint = self._territory_lerp.get(sid) or (realm_color(owner) if owner is not None else None)
            if tint is not None:
                tint = night_mute(tint, self._nf)
                fill, fill_a = tint, int(round(_REALM_FILL_ALPHA * 0.5 * (1.0 - 0.45 * self._nf)))
                edge = _shade(tint, 45)
                edge_a = min(235, int(round(_REALM_EDGE_ALPHA * 1.15 * (1.0 - 0.30 * self._nf))))
                edge_w = 3
                if self._lod == "far":
                    fill_a = min(120, int(fill_a * 2.2))
                    edge_a = min(235, int(edge_a * 1.3))
            else:
                fill = _SETTLEMENT_FILL
                fill_a = int(round(_SETTLEMENT_FILL_ALPHA * 0.7))
                edge = _SETTLEMENT_EDGE
                edge_a = min(205, int(round(_SETTLEMENT_EDGE_ALPHA * 1.2)))
                edge_w = 2
            regions.append((cx, cy, rx, ry, fill, fill_a, edge, edge_a, edge_w))
            towns.append({"sid": sid, "center": center, "cx": cx, "cy": cy,
                          "count": living, "rx": rx, "owner": owner})
        for cx, cy, rx, ry, fill, fill_a, _e, _ea, _ew in regions:        # V4.3: fills first
            pygame.draw.ellipse(overlay, (*fill, fill_a), (cx - rx, cy - ry, 2 * rx, 2 * ry))
        for cx, cy, rx, ry, _f, _fa, edge, edge_a, edge_w in regions:     # ...then all edges
            pygame.draw.ellipse(overlay, (*edge, edge_a), (cx - rx, cy - ry, 2 * rx, 2 * ry), edge_w)
        self._screen.blit(overlay, (0, 0))
        # Prune plans for settlements that no longer exist (keeps the cache bounded).
        self._town_plans = {s: v for s, v in self._town_plans.items() if s in settlements}
        return towns

    def _draw_town_sprite(self, info: dict[str, Any], state: dict[str, Any]) -> int:
        """Draw a settlement in the SINGLE-SPRITE modes (painter pass); return the label top-y.

        Only the FAR strategy view (a block cluster + banner) and the tiny-cell view (a ring of
        slice-4 houses) reach here — a built-up MID/CLOSE town is expanded into its individual iso
        VOLUMES by `_emit_town` instead, so its buildings interleave with the agents in the sort.
        """
        sid, center, cx, cy = info["sid"], info["center"], info["cx"], info["cy"]
        count, owner, cell = info["count"], info["owner"], self._cell
        if self._lod == "far":
            self._draw_far_settlement(sid, cx, cy, count, owner, state)
            return cy - cell * 3
        rx = max(cell, int(round(settlement_radius_cells(center, []) * cell * _ISO_RX)))
        self._draw_settlement_houses(cx, cy, rx, count, cell)
        return cy

    def _resolve_plan(self, info: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """The cached town PLAN for this settlement, rebuilt only when its membership / ruler /
        era / zoom-bucket cell changes (the slice-6 cache key). A pure read of the institutions."""
        sid, center, count, cell = info["sid"], info["center"], info["count"], self._cell
        monarchs, leaders = state.get("monarchs", {}), state.get("leaders", {})
        empires = state.get("empires", {})
        mon = monarchs.get(sid, {}).get("monarch")
        led = leaders.get(sid, {}).get("leader")
        if mon is not None:
            kind, ruler, is_emp = "castle", mon, (mon in empires)
        elif led is not None:
            kind, ruler, is_emp = "hall", led, False
        else:
            kind, ruler, is_emp = None, None, False
        pers = ""
        if ruler is not None:
            for a in state.get("agents", []):
                if getattr(a, "name", None) == ruler:
                    pers = getattr(a, "personality", "")
                    break
        color = agent_color(pers, vivid=True) if ruler else _DEFAULT_RULER
        era_style = _ERA_STYLE.get(state.get("eras", {}).get(sid), "neolithic")
        key = (count, kind, ruler, is_emp, cell, era_style)
        cached = self._town_plans.get(sid)
        if cached is None or cached[0] != key:
            cached = (key, build_town_plan(center, count, kind, color, is_emp, cell, era_style))
            self._town_plans[sid] = cached
        return cached[1]

    def _emit_town(self, info: dict[str, Any],
                   state: dict[str, Any]) -> tuple[list[tuple], int]:
        """V4.7: paint a built-up town's flat GROUND now and hand its VOLUMES back as depth-sorted
        sprites for the shared painter pass. Returns (structure_sprites, label_top_y).

        Ground (drawn immediately, under every sprite): the packed-earth plaza, dirt paths out to
        each dwelling, and one soft shadow per structure. Each building/granary/well/seat and every
        palisade post is then emitted as a `('struct', ...)` sprite keyed by its projected ground-y,
        so agents standing IN FRONT of a hut occlude it and those behind are occluded by it. Night
        hearth/forge/torch pools are registered here (plan-derived, stable per town).
        """
        plan, cx, cy, cell = self._resolve_plan(info, state), info["cx"], info["cy"], self._cell
        screen = self._screen
        hw, hh, zh = cell, cell * 0.5, cell * _ISO_ZH

        def off(wdx: float, wdy: float) -> tuple[int, int]:      # world offset -> screen (from centre)
            return int(round(cx + (wdx - wdy) * hw)), int(round(cy + (wdx + wdy) * hh))

        # -- GROUND: plaza disc (2:1 ellipse), paths, and every structure's shadow ---------
        pr = plan["plaza_r"]
        pygame.draw.ellipse(screen, _PLAZA, (cx - pr, cy - pr // 2, 2 * pr, pr))
        for b in plan["buildings"]:
            bx, by = off(b["wdx"], b["wdy"])
            pygame.draw.line(screen, _PATH, (cx, cy), (bx, by), max(1, cell // 7))
        # shadows (back-to-front doesn't matter for the flat ground layer)
        for b in plan["buildings"]:
            bx, by = off(b["wdx"], b["wdy"])
            self._blit_shadow(bx, by + 1, b["fw"] * cell * 1.5, max(3, int(b["fd"] * cell * 0.6)))
        if plan["granary"]:
            gx, gy = off(plan["granary"]["wdx"], plan["granary"]["wdy"])
            self._blit_shadow(gx, gy + 1, _GRANARY_FOOT * cell * 1.5, max(3, int(_GRANARY_FOOT * cell * 0.6)))
        if plan["central"]["kind"]:
            foot = plan["central"]["foot"]
            self._blit_shadow(cx, cy + 2, foot * cell * 2.0, max(4, int(foot * cell * 0.8)))
        wl = plan["well"]
        wx, wy = off(wl["wdx"], wl["wdy"])
        self._blit_shadow(wx, wy + 1, cell * 0.9, max(2, cell // 3))

        # -- VOLUMES: one depth-keyed sprite per structure --------------------------------
        structs: list[tuple] = []
        tallest_top = cy
        for b in plan["buildings"]:
            bx, by = off(b["wdx"], b["wdy"])
            structs.append((by, 0, "struct", ("hut", bx, by, b)))
            tallest_top = min(tallest_top, by - int((b["z"] + 0.7) * zh))
        if plan["granary"]:
            gx, gy = off(plan["granary"]["wdx"], plan["granary"]["wdy"])
            structs.append((gy, 0, "struct", ("granary", gx, gy, plan["granary"])))
        # the palisade — a defensive WALL: a RING of thicker posts joined by rails, era-toned
        # (timber brown / dressed stone, never near-white). Each segment is one depth-sorted sprite.
        if plan["fence_rc"]:
            rc = plan["fence_rc"]
            pal = _PAL_STONE if plan["stone_wall"] else _PAL_WOOD
            n = max(10, min(18, int(round(2 * math.pi * rc / 1.15))))
            ring = [off(rc * math.cos(i / n * 2 * math.pi), rc * math.sin(i / n * 2 * math.pi))
                    for i in range(n)]
            for i in range(n):
                p0, p1 = ring[i], ring[(i + 1) % n]
                mmy = (p0[1] + p1[1]) // 2
                lit = ((p0[0] + p1[0]) // 2 - cx) < 0          # west-facing panels catch the sun
                panel = _shade(pal, _FACE_LIT if lit else _FACE_DARK)
                structs.append((mmy, 0, "struct", ("wall", p0, p1, pal, panel)))
        structs.append((wy, 0, "struct", ("well", wx, wy, wl)))
        if plan["central"]["kind"]:
            structs.append((cy, 1, "struct", ("seat", cx, cy, plan["central"])))
            tallest_top = min(tallest_top, cy - int((plan["central"]["z"] + 1.4) * zh))

        # -- NIGHT POOLS: hearth over the plaza, a forge for a metal-working town, seat torches ---
        if self._nf > _NIGHT_EPS:
            self._frame_lights.append(("hearth", cx, cy, max(5, pr // 2)))
            if plan.get("forge"):
                fx, fy = off(-plan["cluster_rc"] * 0.5, plan["cluster_rc"] * 0.4)
                self._frame_lights.append(("forge", fx, fy, max(4, cell // 2)))
            if plan["central"]["kind"] == "castle":
                th = int(plan["central"]["z"] * 0.5 * zh)
                self._frame_lights.append(("torch", cx - int(cell * 1.1), cy - th, cell))
                self._frame_lights.append(("torch", cx + int(cell * 1.1), cy - th, cell))
            elif plan["fence_rc"]:
                self._frame_lights.append(("torch", wx, wy - int(_WELL_Z * zh) - 2, max(4, cell // 2)))

        # V4.9: CLOSE-zoom life — coins drift up over the market (the plaza's front edge, clear of
        # the seat), embers over a forge. Both in front of the seat so they read over the buildings.
        if self._lod == "close":
            self._emitters.append(off(plan["cluster_rc"] * 0.28, plan["cluster_rc"] * 0.28) + ("coin",))
            if plan.get("forge"):
                self._emitters.append(off(-plan["cluster_rc"] * 0.5, plan["cluster_rc"] * 0.4) + ("ember",))

        return structs, tallest_top - 2

    def _draw_structure(self, obj: tuple) -> None:
        """Draw ONE town volume (painter pass): dispatch by kind to its iso-volume drawer."""
        kind = obj[0]
        if kind == "hut":
            _k, bx, by, b = obj
            self._draw_iso_box(bx, by, b["fw"], b["fd"], b["z"], b["wall"], b["roof"],
                               roof_h=(0.85 if b["hip"] else 0.5), lit=b["lit"],
                               door=True, windows=2)
        elif kind == "granary":
            _k, gx, gy, g = obj
            self._draw_granary(gx, gy, g)
        elif kind == "wall":
            _k, p0, p1, pal, panel = obj
            self._draw_wall_segment(p0, p1, pal, panel)
        elif kind == "well":
            _k, wx, wy, wl = obj
            self._draw_well(wx, wy, wl)
        else:                                          # the ruler's seat
            _k, cx, cy, c = obj
            if c["kind"] == "castle":
                self._draw_castle(cx, cy, c["z"], c["foot"], c["color"], c["emperor"])
            else:
                self._draw_hall(cx, cy, c["z"], c["foot"], c["color"])

    def _draw_settlement_labels(self, jobs: list[tuple[str, int, int, int]]) -> None:
        """Draw the settlement name·size labels LAST, on a translucent chip above each cluster,
        clamped on-map and nudged upward when two collide (so a label never sits on buildings)."""
        cell = self._cell
        if self._font is None or not (cell >= _SETTLEMENT_LABEL_MIN_CELL or self._lod == "far"):
            return
        screen, placed = self._screen, []
        for sid, cx, top_y, count in jobs:
            label = self._font.render(f"{sid}·{count}", True, _SETTLEMENT_LABEL)
            rect = label.get_rect(midbottom=(cx, top_y - 4))
            rect.left = max(3, min(rect.left, self._map_px - rect.width - 3))
            rect.top = max(3, rect.top)
            while any(rect.colliderect(p) for p in placed) and rect.top > 3:
                rect.move_ip(0, -(rect.height + 2))
            chip = pygame.Surface((rect.width + 6, rect.height + 2), pygame.SRCALPHA)
            chip.fill((*_HUD_BG, 110))
            screen.blit(chip, (rect.left - 3, rect.top - 1))
            screen.blit(label, rect)
            placed.append(rect)

    def _draw_agent_sprite(self, obj: tuple, talkers: "set[str]", state: dict[str, Any]) -> None:
        """Draw ONE agent as an UPRIGHT FIGURE standing on the ground plane (painter pass, V4.6).

        `obj` = (agent, gx, gy, r): (gx, gy) is the projected GROUND point (the agent's feet),
        so the figure rises above it and its shadow anchors at the feet. A ruler wears the vivid
        robe + crown; talkers get an emotion icon + speech bubble; CLOSE zoom adds a name tag.
        """
        agent, gx, gy, r, squash = obj
        role = agent_role(agent.name, state)
        # V4.17 (5.2): personality first, then pulled toward the realm this agent answers to — so a
        # secession or a conquest recolours the PEOPLE, on the same fade as the ground beneath them.
        tinted = allegiance_color(agent_color(getattr(agent, "personality", ""),
                                              vivid=role in ("monarch", "king", "emperor")),
                                  getattr(agent, "settlement", None), state, self._territory_lerp,
                                  _ALLEGIANCE_MIX_RULER if role else _ALLEGIANCE_MIX)
        color = night_mute(tinted, self._nf)
        if self._lod == "far":
            dot = max(2, r)
            pygame.draw.circle(self._screen, _OUTLINE, (gx, gy), dot + 1)
            pygame.draw.circle(self._screen, color, (gx, gy), dot)
            return
        self._blit_shadow(gx, gy, r * 2.1, max(2, int(r * 0.8)))     # shadow at the feet
        # feet on the ground point; `role` shapes the silhouette as well as the insignia (V4.17)
        figure_top = self._draw_agent_figure(gx, gy - r, r, color, squash, role)
        self._draw_role_marker(gx, figure_top, r, role)
        if agent.name in talkers:
            pers = getattr(agent, "personality", "")
            self._draw_emotion_icon(gx, figure_top, r,
                                    "heart" if pers == "friendliness"
                                    else "sword" if pers == "independence" else "coin")
            self._draw_speech_bubble(gx + r + 1, figure_top, r)
        if self._lod == "close" and self._font is not None:
            self._draw_name_tag(agent.name, gx, gy + 2)

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
    # -- V4.7: iso BUILDING VOLUMES (3D forms on the tilted ground; pure drawing) --
    def _iso_box_corners(self, bx: float, by: float, fw: float, fd: float, z: float):
        """The four GROUND corners and four TOP corners (lifted by z) of a world-footprint box
        centred on screen point (bx, by), plus the height rise in px. All in screen space through
        the same 2:1 iso geometry the shared transform uses (hw=cell, hh=cell/2)."""
        cell = self._cell
        hw, hh = cell, cell * 0.5
        a, b = fw / 2.0, fd / 2.0
        def g(ox, oy):
            return (bx + (ox - oy) * hw, by + (ox + oy) * hh)
        gN, gE, gS, gW = g(-a, -b), g(a, -b), g(a, b), g(-a, b)
        zpx = z * cell * _ISO_ZH
        up = lambda p: (p[0], p[1] - zpx)
        return (gN, gE, gS, gW), (up(gN), up(gE), up(gS), up(gW)), zpx

    def _ipoly(self, pts, color, width: int = 0) -> None:
        pygame.draw.polygon(self._screen, color,
                            [(int(round(x)), int(round(y))) for x, y in pts], width)

    @staticmethod
    def _face_pt(face, u: float, v: float):
        """A point on a wall FACE [bottomLeft, bottomRight, topRight, topLeft] at fractional
        (u across, v up) — used to seat doors/windows onto the parallelogram face (pure)."""
        g0, g1, t1, t0 = face
        bx = g0[0] + (g1[0] - g0[0]) * u
        by = g0[1] + (g1[1] - g0[1]) * u
        tx = t0[0] + (t1[0] - t0[0]) * u
        ty = t0[1] + (t1[1] - t0[1]) * u
        return (bx + (tx - bx) * v, by + (ty - by) * v)

    def _iso_hip_roof(self, top, apex, roof) -> None:
        """A hip roof: four triangles from the top diamond's edges up to a single apex, the
        west-facing planes catching the sun and the east-facing planes in shade, painted back
        (north) to front so the near slopes overlap the far ones."""
        tN, tE, tS, tW = top
        lit_r, dark_r = _shade(roof, _ROOF_LIT), _shade(roof, _ROOF_DARK)
        edges = ((tN, tE, dark_r), (tE, tS, dark_r), (tS, tW, lit_r), (tW, tN, lit_r))
        for p, q, col in sorted(edges, key=lambda e: e[0][1] + e[1][1]):
            self._ipoly([p, q, apex], col)
            self._ipoly([p, q, apex], _OUTLINE, 1)

    def _draw_iso_box(self, bx: float, by: float, fw: float, fd: float, z: float,
                      wall, roof, *, roof_h: float = 0.0, lit: bool = False,
                      door: bool = False, windows: int = 0, top=None) -> float:
        """The core VOLUME: a box of world footprint (fw, fd) and height z on the ground point
        (bx, by), with a sun-lit south-west wall, a shaded south-east wall and either a hip roof
        (roof_h>0) or a flat top. Optional door/windows seat onto the front faces; a lit window at
        night registers a point-light that spills onto the face it sits in. Returns the height px."""
        cell = self._cell
        (gN, gE, gS, gW), (tN, tE, tS, tW), zpx = self._iso_box_corners(bx, by, fw, fd, z)
        left = [gW, gS, tS, tW]      # south-west face — sun-lit
        right = [gS, gE, tE, tS]     # south-east face — shaded
        self._ipoly(left, _shade(wall, _FACE_LIT))
        self._ipoly(right, _shade(wall, _FACE_DARK))
        if door and cell >= 9:
            self._ipoly([self._face_pt(left, 0.34, 0.0), self._face_pt(left, 0.66, 0.0),
                         self._face_pt(left, 0.66, min(0.55, 1.4 / max(0.6, z))),
                         self._face_pt(left, 0.34, min(0.55, 1.4 / max(0.6, z)))], _DOOR)
        if windows and cell >= 12 and z > 0.5:
            night = lit and self._nf > _NIGHT_EPS
            wc = (lerp_color(_WINDOW_LIT, (255, 238, 168), 0.8 * self._nf) if night
                  else _WINDOW_LIT if lit else _WINDOW_DARK)
            for u in ([0.32, 0.68][:windows]):
                for face, col in ((left, wc), (right, _shade(wc, -22))):
                    c = self._face_pt(face, u, 0.62)
                    hu = 0.10 * cell
                    self._ipoly([(c[0] - hu, c[1] - hu * 0.9), (c[0] + hu, c[1] - hu * 0.9),
                                 (c[0] + hu, c[1] + hu * 0.9), (c[0] - hu, c[1] + hu * 0.9)], col)
                if night:
                    wx, wy = self._face_pt(left, u, 0.62)
                    self._frame_lights.append(("window", int(wx), int(wy), max(3, cell // 3)))
        self._ipoly(left, _OUTLINE, 1)
        self._ipoly(right, _OUTLINE, 1)
        if roof_h > 0:
            apex = (bx, by - zpx - roof_h * cell * _ISO_ZH)
            self._iso_hip_roof((tN, tE, tS, tW), apex, roof)
        else:
            self._ipoly([tN, tE, tS, tW], top if top is not None else roof)
            self._ipoly([tN, tE, tS, tW], _OUTLINE, 1)
        return zpx

    def _iso_cone(self, bx: float, by: float, foot: float, z: float, roof_h: float, color) -> None:
        """A conical/pyramidal cap (a tower roof) in a flat colour, sat on a box's top diamond."""
        _g, top, zpx = self._iso_box_corners(bx, by, foot, foot, z)
        apex = (bx, by - zpx - roof_h * self._cell * _ISO_ZH)
        for p, q in ((top[0], top[1]), (top[1], top[2]), (top[2], top[3]), (top[3], top[0])):
            self._ipoly([p, q, apex], color)
            self._ipoly([p, q, apex], _OUTLINE, 1)

    def _iso_crenel(self, bx: float, by: float, foot: float, z: float) -> None:
        """A row of merlons along a keep/tower's two FRONT top edges (battlement silhouette)."""
        _g, (tN, tE, tS, tW), _z = self._iso_box_corners(bx, by, foot, foot, z)
        m = max(2, self._cell // 5)
        for a, b in ((tW, tS), (tS, tE)):
            for i in range(4):
                u = i / 3.0
                x = a[0] + (b[0] - a[0]) * u
                y = a[1] + (b[1] - a[1]) * u
                pygame.draw.rect(self._screen, _CASTLE_STONE_DK, (int(x) - 1, int(y) - m, 2, m))

    def _pennant(self, cx: int, cy: int, ztop: float, color, *, double: bool = False) -> None:
        """A pennant on a pole rising to world-height `ztop` above the seat (royalty/leader colour).
        V4.9: the cloth RIPPLES in the wind — a wavy trailing edge on the frame clock; an emperor
        flies a second flag below."""
        s, cell = self._screen, self._cell
        base = cy - int(ztop * cell * _ISO_ZH)
        top = base - max(5, int(cell * 0.9))
        pygame.draw.line(s, _OUTLINE, (cx, base), (cx, top), 1)
        self._draw_flag(cx, top, cell, color)
        if double:
            self._draw_flag(cx, top + max(5, int(cell * 0.5)), int(cell * 0.78), _shade(color, 30))

    def _draw_flag(self, cx: int, top: int, cell: int, color) -> None:
        """A rippling pennant: a triangle whose windward TOP edge waves on the frame clock (the
        ripple deepens toward the free-flying fly end), with a lit top edge catching the wind."""
        s, f = self._screen, self._frame
        L = max(6, int(cell * 0.9))
        H = max(3, int(cell * 0.32))
        seg = 4
        top_edge = [(cx + int(L * (i / seg)),
                     top + int(math.sin(f * 0.28 - (i / seg) * 3.2 + cx * 0.25) * (1.0 + (i / seg) * 2.0)))
                    for i in range(seg + 1)]
        poly = top_edge + [(cx + int(L * 0.82), top + H), (cx, top + H)]
        self._ipoly(poly, color)
        self._ipoly(poly, _shade(color, -34), 1)
        pygame.draw.lines(s, _shade(color, 55), False, top_edge, 1)   # the lit windward edge

    def _draw_hall(self, cx: int, cy: int, z: float, foot: float, color) -> None:
        """A trust-leader's HALL: a longhouse volume well above the huts, a big hip roof and a
        pennant in the leader's colour — between a common hut and a monarch's keep."""
        color = night_mute(color, self._nf)
        self._draw_iso_box(cx, cy, foot, foot * 0.8, z, _WALL_TONES[1], _ROOF_TONES[3],
                           roof_h=1.05, lit=True, door=True, windows=2)
        self._pennant(cx, cy, z + 0.7, color)

    def _draw_castle(self, cx: int, cy: int, z: float, foot: float, color, emperor: bool) -> None:
        """A monarch's CASTLE: a tall crenellated stone KEEP that genuinely TOWERS, flanked by two
        towers with conical roofs in the ruler's colour, a gate on the lit face and a banner — a
        capital readable as a capital from across the map by silhouette alone."""
        color = night_mute(color, self._nf)
        cell = self._cell
        # flanking towers seated just OUTSIDE the keep's footprint (at its west/east corners) and a
        # touch behind it, so the keep sits between them and each tower reads distinctly at close zoom.
        tx_off = int(round((foot + _TOWER_FOOT) * cell))
        ty = cy - int(round(cell * 0.15))
        for sgn, tone in ((-1, _CASTLE_STONE), (1, _CASTLE_STONE_DK)):
            tx = cx + sgn * tx_off
            self._draw_iso_box(tx, ty, _TOWER_FOOT, _TOWER_FOOT, _TOWER_Z, tone, _CASTLE_STONE_DK)
            self._iso_cone(tx, ty, _TOWER_FOOT, _TOWER_Z, 0.9, color)
        self._draw_iso_box(cx, cy, foot, foot, z, _CASTLE_STONE, _CASTLE_STONE)   # the keep
        # a dark gate arch on the lit (south-west) face
        (gN, gE, gS, gW), (tN, tE, tS, tW), _z = self._iso_box_corners(cx, cy, foot, foot, z)
        lf = [gW, gS, tS, tW]
        self._ipoly([self._face_pt(lf, 0.40, 0.0), self._face_pt(lf, 0.60, 0.0),
                     self._face_pt(lf, 0.60, 0.34), self._face_pt(lf, 0.40, 0.34)], _GATE)
        self._iso_crenel(cx, cy, foot, z)
        self._pennant(cx, cy, z, color, double=emperor)

    def _draw_granary(self, gx: int, gy: int, g: dict) -> None:
        """A granary: a stout light-walled store as its own tall narrow volume with a high conical
        roof, distinct from the dwellings around it."""
        self._draw_iso_box(gx, gy, g["fw"], g["fd"], g["z"], _GRANARY_WALL, _ROOF_TONES[2],
                           roof_h=1.35, door=True)

    def _draw_well(self, wx: int, wy: int, wl: dict) -> None:
        """A stone well: a low ring volume with dark water on top and a little gabled roof on two
        posts — a distinct civic mark at the village centre."""
        s, cell = self._screen, self._cell
        z = wl["z"]
        self._draw_iso_box(wx, wy, 0.5, 0.5, z, _WELL_STONE, _WELL_WATER, top=_WELL_WATER)
        topy = wy - int(z * cell * _ISO_ZH)
        ph = max(4, int(cell * 0.85))
        for dxp in (-int(cell * 0.32), int(cell * 0.32)):
            pygame.draw.line(s, _TREE_TRUNK, (wx + dxp, topy), (wx + dxp, topy - ph), 1)
        pygame.draw.polygon(s, _ROOF_TONES[0],
                            [(wx - int(cell * 0.45), topy - ph), (wx + int(cell * 0.45), topy - ph),
                             (wx, topy - ph - int(cell * 0.4))])

    def _draw_wall_segment(self, p0: tuple, p1: tuple, pal: tuple, panel: tuple) -> None:
        """One span of PALISADE: a rail/wall panel rising between two ring points, plus a thicker
        POST standing at p0 — so the perimeter reads as a continuous defensive WALL, not a scatter
        of pale shards. `panel` is the era wall tone pre-shaded for this span's sun facing."""
        s, zh = self._screen, self._cell * _ISO_ZH
        wz = _PAL_WALL_Z * zh
        top0, top1 = (p0[0], p0[1] - wz), (p1[0], p1[1] - wz)
        self._ipoly([p0, p1, top1, top0], panel)                 # the rail panel body
        pygame.draw.line(s, _shade(pal, 24), top0, top1, 1)      # a lit cap rail along the top
        self._ipoly([p0, p1, top1, top0], _OUTLINE, 1)           # a defined edge
        self._draw_iso_box(p0[0], p0[1], _PAL_POST_FOOT, _PAL_POST_FOOT, _PAL_POST_Z,
                           pal, _shade(pal, -14), roof_h=0.16)    # a stout post at the junction

    def _draw_hud(self, state: dict[str, Any], map_px: int, paused: bool,
                  in_battle: bool = False) -> None:
        """A one-line status strip under the map zone (turn / living / food / pause / battle).

        V4.1 BUG FIX: the strip is now laid out with MEASURED spacing. A RIGHT cluster (the
        phase dial + name, and the battle chip) is placed first and its left edge becomes a
        hard boundary; the LEFT status segments are placed left-to-right and any segment that
        would cross that boundary (least-important last: the key hints) is dropped instead of
        overprinting. So nothing ever overlaps at any window size or zoom.
        """
        screen = self._screen
        pygame.draw.rect(screen, _HUD_BG, (0, self._map_h, map_px, self._hud_h))   # strip under the map
        font = self._font
        if font is None:
            return
        cy = self._map_h + self._hud_h // 2

        def place_left(surf: Any, x: int) -> int:
            screen.blit(surf, (x, cy - surf.get_height() // 2))
            return x + surf.get_width()

        # RIGHT cluster first — the sun/moon dial at the far right, the phase name to its left,
        # and (while a cinematic plays) the BATTLE chip further left. Each measured; the
        # leftmost of them becomes the boundary the status text must never cross.
        dial_cx = map_px - 14
        self._draw_phase_dial(dial_cx, cy)
        tag = font.render(phase_name(self._phase), True, _STAT_LABEL)
        tag_x = dial_cx - 13 - tag.get_width()
        screen.blit(tag, (tag_x, cy - tag.get_height() // 2))
        right_edge = tag_x
        if in_battle:                       # slice 8: a small indicator while a battle plays
            chip = font.render("BATTLE [any key skips]", True, _BATTLE_CHIP)
            chip_x = right_edge - 14 - chip.get_width()
            screen.blit(chip, (chip_x, cy - chip.get_height() // 2))
            right_edge = chip_x

        # LEFT status segments — mandatory counts first, the droppable key-hint last.
        turn = state.get("turn", 0)
        living = sum(1 for a in state.get("agents", []) if getattr(a, "alive", True))
        food = len(state.get("food", []))
        towns = len(state.get("settlements", {}))
        segs: list[tuple[str, tuple[int, int, int]]] = []
        if paused:
            segs.append(("PAUSED [space] resume", _FEED_GOD))
        segs.append((f"turn {turn}", _HUD_FG))
        segs.append((f"living {living}", _HUD_FG))
        segs.append((f"food {food}", _HUD_FG))
        if towns:                           # only once settlements exist (slice-1 HUD unchanged)
            segs.append((f"towns {towns}", _HUD_FG))
        segs.append((f"zoom {self._cell / max(1, self._cell0):.1f}x", _HUD_FG))
        segs.append(("[spc]pause [wasd]pan [whl]zoom [home]fit", _STAT_LABEL))  # droppable last
        gap = max(6, font.size("  ")[0])
        limit = right_edge - 10             # never draw past the right cluster
        x = 8
        for text, col in segs:
            surf = font.render(text, True, col)
            if x + surf.get_width() > limit and x > 8:
                break                       # this segment (and the rest) won't fit — stop cleanly
            x = place_left(surf, x) + gap

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
                   max_rows: int) -> list[tuple[str, tuple[int, int, int], bool]]:
        """V4.2 STORY rows sized to the panel: MAJOR events in full (flagged bold) + at most
        one aggregated MINOR line per turn. A read of state["events"] (+ institutions)."""
        char_w = max(1, self._font.size("M")[0])
        cols = max(8, inner_w // char_w)
        return story_feed_rows(state.get("events") or [], notable_names(state), cols, max_rows)

    def _draw_panel(self, state: dict[str, Any], map_px: int) -> None:
        """Draw the right sidebar: a STATE summary above a scrolling EVENT feed (READ only).

        Top block = current-state counts (turn/living/food + settlements/kingdoms/empires
        where present); below a divider, the EVENTS feed shows the most recent log lines
        that fit, wrapped to the panel and lightly colour-coded by type, newest at the
        bottom. Pure read — it never touches world_state.
        """
        screen = self._screen
        font = self._font
        win_h, panel_w = self._win_h, self._panel_w       # V4.11: proportional panel, full window height
        x0, pad = map_px, _PANEL_PAD
        inner_w = panel_w - 2 * pad
        pygame.draw.rect(screen, _PANEL_BG, (x0, 0, panel_w, win_h))
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
            screen.blit(val, (x0 + panel_w - pad - val.get_width(), y))
            y += line_h

        # REALMS Scoreboard
        y += 5
        pygame.draw.line(screen, _PANEL_DIV, (x0 + pad, y), (x0 + panel_w - pad, y))
        y += 7
        screen.blit(font.render("REALMS", True, _PANEL_TITLE), (x0 + pad, y))
        y += line_h + 2
        
        # V4.1 BUG FIX: read kingdoms/empires from world_state (via realm_scoreboard) instead of
        # the never-populated state["realms"]. Each realm shows a colour SWATCH matching its map
        # territory, its name (empires bolded + tagged), and its settlement count.
        realms = realm_scoreboard(state)
        if not realms:
            screen.blit(font.render("(none)", True, _FEED_DEFAULT), (x0 + pad, y))
            y += line_h
        else:
            sw = 9
            for name, count, is_emp in realms[:6]:
                color = realm_color(name)
                pygame.draw.rect(screen, color, (x0 + pad, y + 3, sw, sw))
                pygame.draw.rect(screen, _shade(color, 40), (x0 + pad, y + 3, sw, sw), 1)
                nm_font = self._feed_bold if (is_emp and self._feed_bold) else font
                disp = f"{name} (empire)" if is_emp else name
                screen.blit(nm_font.render(disp[:18], True, color), (x0 + pad + sw + 6, y))
                val = font.render(str(count), True, _STAT_VALUE)
                screen.blit(val, (x0 + panel_w - pad - val.get_width(), y))
                y += line_h

        # Divider + EVENTS header.
        y += 5
        pygame.draw.line(screen, _PANEL_DIV, (x0 + pad, y), (x0 + panel_w - pad, y))
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
        # V4.2: MAJOR rows read ABOVE the aggregated churn — bold, in their event colour, on a
        # subtle colour-tinted background bar; the minor summary stays plain muted grey.
        for text, color, is_major in rows:
            if is_major:
                hl = pygame.Surface((panel_w - 2 * pad + 4, line_h), pygame.SRCALPHA)
                hl.fill((*color, _FEED_MAJOR_BG_A))
                screen.blit(hl, (x0 + pad - 2, feed_top - 1))
                row_font = self._feed_bold or font
                screen.blit(row_font.render(text, True, color), (x0 + pad, feed_top))
            else:
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
                                                  self._view)
        # V4.9: the beat time is an ACCUMULATED clock (not wall time), so a brief SLOW-MOTION dip
        # around the decisive blow really slows the picture. A short sharp SHAKE hits when the clash
        # joins (scaled to casualties), and the decisive blow adds a stronger shake + an impact FLASH.
        n_cas = len(scene.get("att_dead") or ()) + len(scene.get("def_dead") or ())
        t2 = _CIN_MUSTER + _CIN_MARCH                      # clash joins here
        blow = t2 + _CIN_CLASH - 0.16                      # the decisive blow, late in the clash
        after_start = _CIN_TOTAL - _CIN_AFTER
        clock, last = 0.0, time.monotonic()
        clash_shook = blow_fired = False
        try:
            while True:
                if self._pump_cinema_events():
                    return                        # skipped -> next frame is the settled end-state
                now = time.monotonic()
                dt, last = now - last, now
                ts = 0.35 if blow - 0.10 <= clock <= blow + 0.22 else 1.0   # slow-mo around the blow
                clock += dt * ts
                if clock >= _CIN_TOTAL:
                    return
                if not clash_shook and clock >= t2:                         # the clash joins
                    clash_shook = True
                    self._shake_amp = min(_SHAKE_MAX, 2.5 + 1.2 * n_cas)
                if not blow_fired and clock >= blow:                        # the decisive blow lands
                    blow_fired = True
                    self._shake_amp = min(_SHAKE_MAX, 4.0 + 1.4 * n_cas)
                    self._flash_amp = 0.42
                frac = 0.0 if clock < after_start else (clock - after_start) / _CIN_AFTER
                self._territory_lerp = self._territory_colors(scene, frac, state)
                self._draw(state, battle=(scene, clock))
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

    def _formation(self, n: int, uwx: float, uwy: float, salt: int) -> list[tuple[float, float]]:
        """Rank-and-file WORLD-cell offsets for `n` soldiers facing world direction (uwx, uwy) —
        deterministic ranks of four with a small per-soldier hash jitter (V4.8). Offsets are in
        world cells and projected per soldier, so a host's ranks recede ALONG the tilted ground
        toward the enemy, not down flat screen space."""
        pwx, pwy = -uwy, uwx                      # the across-the-line direction, in world cells
        out = []
        for i in range(n):
            row, col = divmod(i, 4)
            across = (col - 1.5) * 0.62
            back = row * 0.55
            jx = (terrain_noise(i, salt, 11) - 0.5) * 0.25
            jy = (terrain_noise(i, salt, 12) - 0.5) * 0.25
            out.append((across * pwx - back * uwx + jx, across * pwy - back * uwy + jy))
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
        y0 = (self._map_h - band_h) // 2
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
        # V4.8: the whole scene is laid out in WORLD cells and projected per element, so musters,
        # marches, formations and clashes follow the tilted ground plane (soldiers stand ON it with
        # foot shadows; clash flashes register as point lights at their PROJECTED position).
        awx, awy = float(scene["att_pos"][0]), float(scene["att_pos"][1])
        dwx, dwy = float(scene["def_pos"][0]), float(scene["def_pos"][1])
        ddx, ddy = dwx - awx, dwy - awy
        dist = math.hypot(ddx, ddy) or 1.0
        uwx, uwy = ddx / dist, ddy / dist                 # world march direction, attacker -> defender
        pull = min(dist * 0.5, 2.4)                        # the clash line at the defender's edge (world)
        m_wx, m_wy = dwx - uwx * pull, dwy - uwy * pull
        meet_a = (m_wx - uwx * 0.55, m_wy - uwy * 0.55)
        meet_d = (m_wx + uwx * 0.55, m_wy + uwy * 0.55)
        facing_a = (uwx - uwy) >= 0                        # sign of the direction's SCREEN-x (iso)
        n_a = max(1, min(_MAX_SOLDIER_GLYPHS, int(scene.get("n_att") or 1)))
        n_d = max(0, min(_MAX_SOLDIER_GLYPHS, int(scene.get("n_def") or 0)))
        form_a = self._formation(n_a, uwx, uwy, salt=1)
        form_d = self._formation(n_d, -uwx, -uwy, salt=2)
        frame = int(el * 30)
        t1, t2, t3 = _CIN_MUSTER, _CIN_MUSTER + _CIN_MARCH, _CIN_MUSTER + _CIN_MARCH + _CIN_CLASH
        t4 = t3 + _CIN_FALL

        # Beat state: each host's WORLD centre, how many have mustered, and the melee jitter (cells).
        if el < t1:                                       # MUSTER at the attacker's capital
            a_c, d_c = (awx, awy), meet_d
            vis_a = max(1, int(math.ceil(n_a * (el / t1))))
            vis_d = int(math.ceil(n_d * (el / t1)))
            jit = 0.0
        elif el < t2:                                     # MARCH on the defender's settlement
            p = ease((el - t1) / _CIN_MARCH)
            a_c = (awx + (meet_a[0] - awx) * p, awy + (meet_a[1] - awy) * p)
            d_c, vis_a, vis_d, jit = meet_d, n_a, n_d, 0.0
            for k in range(5):                            # dust puffs behind the moving host (world)
                if terrain_noise(frame, k, 31) > 0.35:
                    back = 0.8 + terrain_noise(frame, k, 32) * 1.6
                    side = (terrain_noise(frame, k, 33) - 0.5) * 1.5
                    dsx, dsy = self._to_px(a_c[0] - uwx * back - uwy * side,
                                           a_c[1] - uwy * back + uwx * side)
                    pygame.draw.circle(screen, _DUST, (dsx, dsy),
                                       max(1, int(terrain_noise(frame, k, 34) * cell * 0.22)))
        else:                                             # CLASH / FALL / AFTERMATH at the line
            a_c, d_c, vis_a, vis_d = meet_a, meet_d, n_a, n_d
            jit = 0.30 if el < t3 else 0.0                # melee shake in world cells

        # The named dead hold the FIRST slots of their side; each falls at its own moment.
        dead = [(nm, True) for nm, _p in scene.get("att_dead") or ()] + \
               [(nm, False) for nm, _p in scene.get("def_dead") or ()]
        fall_at = {}
        for j, (nm, att_side) in enumerate(dead):
            fall_at[(att_side, nm)] = t3 + _CIN_FALL * (j + 0.4) / max(1, len(dead))

        for side_dead, center, form, vis, color, facing in (
                ([nm for nm, s in dead if s], a_c, form_a, vis_a, scene["att_color"], facing_a),
                ([nm for nm, s in dead if not s], d_c, form_d, vis_d, scene["def_color"], not facing_a)):
            att_side = form is form_a
            for i in range(vis):
                ox, oy = form[i]
                wx, wy = center[0] + ox, center[1] + oy
                if jit > 0:                       # melee: world-cell position shake + brief lunges
                    wx += (terrain_noise(frame, i, 41 if att_side else 43) - 0.5) * jit * 2
                    wy += (terrain_noise(frame, i, 42 if att_side else 44) - 0.5) * jit * 2
                x, y = self._to_px(wx, wy)        # project onto the ground plane (feet stand here)
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
                    fx, fy = self._to_px(m_wx + (terrain_noise(frame, k, 22) - 0.5) * 1.8,
                                         m_wy + (terrain_noise(frame, k, 23) - 0.5) * 1.2)
                    fr = 2 + int(terrain_noise(frame, k, 24) * cell * 0.4)
                    # V4.5: each clash casts a bright additive POOL at its PROJECTED position — so a
                    # NIGHT battle is lit by its own fighting (stronger in the dark, subtle by day).
                    self._blit_light(fx, fy, max(10, fr * 4), PALETTE["clash_light"],
                                     0.45 + 0.55 * self._nf)
                    pygame.draw.circle(screen, _FLASH, (fx, fy), fr)
                    pygame.draw.circle(screen, _shade(_FLASH, -70), (fx, fy), fr, 1)

        if el >= t4:                                      # AFTERMATH: the verdict, prominently (UI)
            self._draw_banner(scene["banner"], fade=(el - t4) / 0.25)

    # -- Slice 12: Visual Enhancements --------------------------------------
    def _draw_coast_waves(self) -> None:
        """V4.8: wave crests along the PROJECTED coastline. The shoreline runs down the world column
        cx~=size (the +x sea margin); each world row projects through the iso transform, so the crest
        is the tilted diagonal shore, not a vertical screen line. A slow sine sways the surf; foam
        flecks ride the same swell."""
        if self._margin_px <= 0 or self._lod == "far":
            return
        cell, view, size, f = self._cell, self._cull, self._size, self._frame
        pts = []
        for cy in range(0, size + 1):
            wave = math.sin(f * 0.05 + cy * 0.5) * 0.35
            pts.append(self._to_px(size + 0.15 + wave, cy))
        if len(pts) > 1 and any(visible_on_screen(x, y, cell * 2, view, view) for x, y in pts):
            pygame.draw.lines(self._screen, PALETTE["water_hi"], False, pts, max(1, cell // 10))
        for k in range(8):                                # foam flecks along the surf line
            if terrain_noise(f // 25, k, 92) > 0.62:
                cy = terrain_noise(0, k, 93) * size
                wave = math.sin(f * 0.05 + cy * 0.5) * 0.35
                x, y = self._to_px(size + 0.15 + wave + terrain_noise(f // 12, k, 94) * 0.5, cy)
                if visible_on_screen(x, y, 12, view, view):
                    pygame.draw.circle(self._screen, PALETTE["foam"], (x, y), max(1, int(cell * 0.15)))

    def _draw_cloud_shadows(self) -> None:
        """V4.8: cloud shadows as GROUND DECALS on the diamond. Each drifting cloud's WORLD point
        projects to the ground (z=0) as an axis-aligned 2:1 ellipse (a projected circle) and slides
        across the tilt in world space. Drawn UNDER the sprite pass, so buildings correctly occlude
        the shadow that passes behind them. V4-fix: SMALLER and FAINTER with a DEFINED edge (so it
        reads as a moving shadow, not a fog blob), and it slides faster (see ambient_clouds)."""
        if self._lod == "far" or self._dl <= 0.05:
            return
        cell, view = self._cell, self._cull
        a = int(22 * self._dl)                             # fainter than before (was 34)
        if a <= 0:
            return
        for wx, wy, rad in ambient_clouds(self._frame, self._size):
            gx, gy = self._to_px(wx, wy)
            rx = max(cell, int(rad * cell * _ISO_RX * 0.48))   # a tight shadow (was the full puff)
            ry = max(2, rx // 2)
            if not visible_on_screen(gx, gy, rx + cell, view, view):
                continue
            stamp = pygame.Surface((2 * rx + 2, 2 * ry + 2), pygame.SRCALPHA)
            pygame.draw.ellipse(stamp, (*PALETTE["cloud_shadow"], a), (0, 0, 2 * rx, 2 * ry))
            pygame.draw.ellipse(stamp, (*PALETTE["cloud_shadow"], a + 34), (0, 0, 2 * rx, 2 * ry), 2)
            self._screen.blit(stamp, (gx - rx, gy - ry))

    def _draw_clouds(self) -> None:
        """V4.8: the cloud PUFFS themselves, lifted high above the ground (z=_CLOUD_Z) so each rides
        the sky directly ABOVE its ground shadow and both slide together in world space. Drawn over
        the sprites (they are overhead); faded out at night."""
        if self._lod == "far":
            return
        cell, view = self._cell, self._cull
        a = int(85 * self._dl)
        if a <= 0:
            return
        for wx, wy, rad in ambient_clouds(self._frame, self._size):
            sx, sy = self._to_px(wx, wy, _CLOUD_Z)
            rx = max(cell, int(rad * cell * _ISO_RX))
            ry = max(3, int(rx * 0.55))
            if not visible_on_screen(sx, sy, rx + cell, view, view):
                continue
            stamp = pygame.Surface((2 * rx + 4, 2 * ry + 4), pygame.SRCALPHA)
            pygame.draw.ellipse(stamp, (*PALETTE["cloud"], a), (0, ry // 2, 2 * rx, ry + ry // 2))
            pygame.draw.ellipse(stamp, (*PALETTE["cloud"], a), (rx // 2, 0, rx, ry + 3))
            self._screen.blit(stamp, (sx - rx, sy - ry))

    def _draw_weather(self) -> None:
        """V4.8: rain/snow FALL toward the ground plane and LAND on it; fog HUGS the terrain.

        Each drop is seeded at a WORLD (x, y) inside the visible ground rect and given a HEIGHT z
        that descends each frame; it projects through the iso transform so it is tied to the world
        (tracks a pan) and its ground-contact point is the true tile beneath it. The streak leans
        with the iso wind (down-right), and the drop FADES as z->0 with a small splash on the tile.
        Fog is low translucent banks seated just above the ground (z small) that drift with the wind.
        """
        weather = weather_type(self._phase)
        if weather == "clear" or self._lod == "far":
            return
        screen, view, cell, f = self._screen, self._map_px, self._cell, self._frame
        zh = cell * _ISO_ZH
        wx0, wy0, wx1, wy1 = self._visible_world_rect()
        dw, dh = max(1e-3, wx1 - wx0), max(1e-3, wy1 - wy0)
        layer = pygame.Surface(self._paint, pygame.SRCALPHA)

        if weather == "rain":
            ztop = 6.0
            sx_lean, sy_len = int(cell * 0.28), max(4, int(cell * 0.7))   # iso wind lean + streak
            for k in range(46):
                wx = wx0 + terrain_noise(k, 0, 101) * dw
                wy = wy0 + terrain_noise(k, 1, 102) * dh
                fall = 0.30 + 0.16 * terrain_noise(k, 2, 106)
                z = ztop - ((f * fall + terrain_noise(k, 3, 107) * ztop) % ztop)   # descends ztop->0
                gx, gy = self._to_px(wx, wy)
                if not visible_on_screen(gx, gy, cell * 2, view, view):
                    continue
                py = int(gy - z * zh)
                if z < 0.5:                              # ground contact: streak fades, splash rings out
                    pygame.draw.line(layer, (*PALETTE["rain"], int(70 * z / 0.5)),
                                     (gx - sx_lean, py - sy_len), (gx, py), 1)
                    sa = int(55 * (1 - z / 0.5))
                    pygame.draw.ellipse(layer, (*PALETTE["rain"], sa), (gx - 3, gy - 1, 6, 3), 1)
                else:
                    pygame.draw.line(layer, (*PALETTE["rain"], 72),
                                     (gx - sx_lean, py - sy_len), (gx, py), 1)

        elif weather == "snow":
            ztop = 5.0
            for k in range(30):
                wx = wx0 + terrain_noise(k, 0, 103) * dw
                wy = wy0 + terrain_noise(k, 1, 104) * dh
                fall = 0.06 + 0.04 * terrain_noise(k, 2, 108)
                z = ztop - ((f * fall + terrain_noise(k, 3, 109) * ztop) % ztop)
                gx, gy = self._to_px(wx, wy)
                if not visible_on_screen(gx, gy, cell * 2, view, view):
                    continue
                wob = int(math.sin(f * 0.05 + k) * cell * 0.15)
                px, py = gx + wob, int(gy - z * zh)
                r = 1 if terrain_noise(k, 4, 105) > 0.6 else 2
                pygame.draw.circle(layer, (*PALETTE["snow"], int(95 * min(1.0, z / 0.4))), (px, py), r)
                if z < 0.35:                             # a flake settling on the tile
                    pygame.draw.circle(layer, (*PALETTE["snow"], 60), (gx, gy), 1)

        elif weather == "fog":
            for k in range(5):
                drift = ((f * 0.012 + terrain_noise(k, 0, 110)) % 1.0)
                wx = wx0 + drift * dw
                wy = wy0 + terrain_noise(k, 1, 111) * dh
                gx, gy = self._to_px(wx, wy, 0.3)        # a bank hugging the ground
                rx = max(cell, int(cell * (2.4 + 2.0 * terrain_noise(k, 2, 112)) * _ISO_RX))
                ry = max(3, int(rx * 0.5))
                if not visible_on_screen(gx, gy, rx + cell, view, view):
                    continue
                a = int(18 + 9 * math.sin(f * 0.03 + k * 2))
                stamp = pygame.Surface((2 * rx + 2, 2 * ry + 2), pygame.SRCALPHA)
                pygame.draw.ellipse(stamp, (*PALETTE["fog"], max(0, a)), (0, 0, 2 * rx, 2 * ry))
                layer.blit(stamp, (gx - rx, gy - ry))

        screen.blit(layer, (0, 0))

    def _update_trails(self, state: dict[str, Any], motion: tuple[dict[str, tuple], float] | None) -> None:
        """Record moving agents' WORLD positions to form a footprint trail (V4.8).

        The trail stores world cells (not screen pixels), so it follows the ground under a pan/zoom
        and never freezes at stale screen coordinates — a new point is banked only once the agent has
        moved ~half a cell in WORLD space."""
        if motion is None or self._lod == "far":
            return
        for agent in state.get("agents", []):
            if not getattr(agent, "alive", True) or not getattr(agent, "position", None):
                continue
            wx, wy = self._agent_world(agent, motion)
            dq = self._trails[agent.name]
            if not dq or math.hypot(dq[-1][0] - wx, dq[-1][1] - wy) > 0.4:
                dq.append((wx, wy))

    def _draw_trails(self) -> None:
        """Draw fading footprint paths along the PROJECTED ground (V4.8): each stored world point is
        transformed through the shared iso transform this frame, so the path lies on the diamond."""
        if self._lod == "far":
            return
        view = self._cull
        for name, path in self._trails.items():
            n = len(path)
            for i, (wx, wy) in enumerate(path):
                alpha = int(120 * (i + 1) / n)
                if alpha <= 0:
                    continue
                tx, ty = self._to_px(wx, wy)
                if visible_on_screen(tx, ty, self._cell, view, view):
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
        mm_y = self._map_h - mm_size - margin
        
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
            
        # 4. Viewport rectangle. V4.6: the visible region is an iso DIAMOND on the top-down
        # minimap — take the world bbox of the four screen corners (deliberately keeping the
        # minimap a plain plan read; the rect approximates what the tilted view covers).
        w, h = self._view
        corners = [screen_to_world_iso(c, self._cam_draw, self._view)
                   for c in ((0, 0), (w, 0), (0, h), (w, h))]
        wxs = [c[0] for c in corners]
        wys = [c[1] for c in corners]
        vx = int((min(wxs) + _MARGIN_CELLS) * scale)
        vy = int((min(wys) + _MARGIN_CELLS) * scale)
        vw = int((max(wxs) - min(wxs)) * scale)
        vh = int((max(wys) - min(wys)) * scale)

        pygame.draw.rect(mm, (255, 255, 255), (vx, vy, vw, vh), 1)
        pygame.draw.rect(mm, _OUTLINE, (0, 0, mm_size, mm_size), 2)
        
        self._screen.blit(mm, (mm_x, mm_y))
