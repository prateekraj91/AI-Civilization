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
_HUD_H = 26                    # status strip height
_WEALTH_CEIL = 60.0            # wealth mapped to the largest radius (sqrt ramp below it)


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
        """Create (or resize) the window to fit a `size`x`size` world."""
        if self._screen is not None and size == self._size:
            return
        self._size = size
        self._cell = _cell_size(size)
        grid_px = self._cell * max(1, size)
        self._screen = pygame.display.set_mode((grid_px, grid_px + _HUD_H))

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

        # Terrain: a flat muted ground, with faint cell lines only when cells are big
        # enough that lines read as texture rather than noise.
        screen.fill(_TERRAIN)
        if cell >= 12:
            for i in range(size + 1):
                p = i * cell
                pygame.draw.line(screen, _GRID_LINE, (p, 0), (p, grid_px))
                pygame.draw.line(screen, _GRID_LINE, (0, p), (grid_px, p))

        # Slice 2: SETTLEMENTS as soft translucent regions UNDER everything else, so a
        # settlement reads as a background "place" with food and agents sitting on top.
        # No-op (slice-1 behaviour) when there are no settlements in world_state.
        self._draw_settlements(state)

        # Food: small green dots.
        food_r = max(1, cell // 6)
        for fx, fy in state.get("food", []):
            pygame.draw.circle(screen, _FOOD, self._to_px(fx, fy), food_r)

        # Agents: a colour-by-personality, size-by-wealth circle each (living only).
        for agent in state.get("agents", []):
            if not getattr(agent, "alive", True):
                continue
            pos = getattr(agent, "position", None)
            if not pos:
                continue
            cx, cy = self._to_px(pos[0], pos[1])
            r = agent_radius(_wealth(agent), cell)
            pygame.draw.circle(screen, _OUTLINE, (cx, cy), r + 1)
            pygame.draw.circle(screen, agent_color(getattr(agent, "personality", "")), (cx, cy), r)

        self._draw_hud(state, grid_px, paused)
        pygame.display.flip()

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
        markers: list[tuple[tuple[int, int], int, str]] = []
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
            markers.append(((cx, cy), len(members), sid))
        screen.blit(overlay, (0, 0))
        # Centre markers + optional labels, drawn opaquely on top of the tint (but still
        # under food/agents, which the caller draws after this method returns).
        m = max(2, cell // 4)
        for (cx, cy), count, sid in markers:
            pygame.draw.polygon(screen, _SETTLEMENT_EDGE,
                                [(cx, cy - m), (cx + m, cy), (cx, cy + m), (cx - m, cy)])
            if self._font is not None and cell >= _SETTLEMENT_LABEL_MIN_CELL:
                label = self._font.render(f"{sid}·{count}", True, _SETTLEMENT_LABEL)
                screen.blit(label, (cx + m + 2, cy - label.get_height() // 2))

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
