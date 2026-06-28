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

A tiny HUD line shows turn / living / food. Later slices add settlements, leaders,
monarchs, kingdoms and war — NOT here.

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

    # -- lifecycle ---------------------------------------------------------
    @contextlib.contextmanager
    def live(self):
        """Open the window for the duration of a run; quit Pygame cleanly on exit."""
        pygame.init()
        pygame.display.set_caption("AI Civilization — live")
        with contextlib.suppress(Exception):
            self._font = pygame.font.SysFont("menlo,monospace", 14)
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

    # -- the per-turn hook the sim calls -----------------------------------
    def update(self, state: dict[str, Any]) -> None:
        """Draw one frame for the just-resolved turn, then pace/handle input (READ only)."""
        self._last_state = state
        self._ensure_screen(int(state.get("size", 0)) or 1)
        self._pump_events()
        self._draw(state)
        self._pace(state)

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

    def _pace(self, state: dict[str, Any]) -> None:
        """Wait out the per-turn delay, staying responsive; block here while paused."""
        deadline = time.monotonic() + self.turn_delay
        while True:
            self._pump_events()
            if self.paused:
                self._draw(state, paused=True)
                deadline = time.monotonic() + self.turn_delay  # don't fast-forward on resume
            elif time.monotonic() >= deadline:
                return
            time.sleep(0.01)

    # -- drawing (pure reads of `state`) -----------------------------------
    def _to_px(self, x: int, y: int) -> tuple[int, int]:
        """Centre of grid cell (x, y) in pixels."""
        c = self._cell
        return (x * c + c // 2, y * c + c // 2)

    def _draw(self, state: dict[str, Any], *, paused: bool = False) -> None:
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
            cx, cy = self._to_px(pos[0], pos[1])
            r = agent_radius(_wealth(agent), cell)
            color = agent_color(getattr(agent, "personality", ""))
            figure_top = self._draw_agent_figure(cx, cy, r, color)
            self._draw_role_marker(cx, figure_top, r, agent_role(agent.name, state))
            if agent.name in talkers:
                self._draw_speech_bubble(cx + r + 1, figure_top, r)

        self._draw_hud(state, grid_px, paused)
        # Slice 3: the right sidebar — a state summary above a scrolling event feed. Drawn
        # last so it owns the right zone cleanly; a pure read of state, like everything else.
        self._draw_panel(state, grid_px)
        pygame.display.flip()

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
            for fy in range(cy - rad + step, cy + rad, step):    # furrows, clipped to the disc
                half = int((rad * rad - (fy - cy) ** 2) ** 0.5)
                if half > 1:
                    pygame.draw.line(overlay, (*_FARMLAND_FURROW, _FARMLAND_ALPHA),
                                     (cx - half, fy), (cx + half, fy), 1)
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
        # One translucent overlay per frame; circles drawn here blend over the terrain
        # WITHOUT darkening the agents (which are drawn afterwards, straight on the screen).
        overlay = pygame.Surface((grid_px, grid_px), pygame.SRCALPHA)
        markers: list[tuple[tuple[int, int], int, str, int]] = []
        for sid in sorted(settlements):
            rec = settlements[sid]
            center = rec.get("center")
            if center is None:
                continue
            members = rec.get("members") or ()
            member_positions = [pos_by_name[n] for n in members if n in pos_by_name]
            radius_px = int(round(settlement_radius_cells(center, member_positions) * cell))
            cx, cy = self._to_px(int(center[0]), int(center[1]))
            pygame.draw.circle(overlay, (*_SETTLEMENT_FILL, _SETTLEMENT_FILL_ALPHA), (cx, cy), radius_px)
            pygame.draw.circle(overlay, (*_SETTLEMENT_EDGE, _SETTLEMENT_EDGE_ALPHA), (cx, cy), radius_px, width=2)
            markers.append(((cx, cy), len(members), sid, radius_px))
        screen.blit(overlay, (0, 0))
        # Slice 4: HOUSES make the region read as a BUILT place. A handful of square+roof
        # glyphs (count ~ membership, capped) ring the centre within the region; the slice-2
        # diamond centre marker + label sit on top. All under food/agents (drawn afterwards).
        m = max(2, cell // 4)
        for (cx, cy), count, sid, radius_px in markers:
            self._draw_settlement_houses(cx, cy, radius_px, count, cell)
            pygame.draw.polygon(screen, _SETTLEMENT_EDGE,
                                [(cx, cy - m), (cx + m, cy), (cx, cy + m), (cx - m, cy)])
            if self._font is not None and cell >= _SETTLEMENT_LABEL_MIN_CELL:
                label = self._font.render(f"{sid}·{count}", True, _SETTLEMENT_LABEL)
                screen.blit(label, (cx + m + 2, cy - label.get_height() // 2))

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

    def _draw_hud(self, state: dict[str, Any], grid_px: int, paused: bool) -> None:
        """A one-line status strip under the grid (turn / living / food / pause)."""
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
