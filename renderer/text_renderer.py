"""
renderer/text_renderer.py
=========================

A rich terminal dashboard for AI Civilization (Day 18).

WHAT IT DOES
------------
Turns a `world_state` snapshot into a live, in-place-updating terminal view:

    +-----------------------------+  +--------------------------------+
    |  World — turn 12  food 3    |  | AGENTS                         |
    |  . . . . . . . . . .        |  | Alex   ████░░░░░░  ALIVE       |
    |  . . . * . . . . . .        |  |   allies: Bob   trust Bob:+3   |
    |  . . A . $ . . . . .        |  | Bob    ██████░░░░  ALIVE sick  |
    |  . . . . B . . . . .        |  | Kira   ██████████  DEAD        |
    |  ...                        |  +--------------------------------+
    |                             |  | EVENTS                         |
    |                             |  | turn 10: [GOD] drought ...     |
    |                             |  | turn 11: Kira died (starved)   |
    +-----------------------------+  +--------------------------------+

ARCHITECTURE RULE (do not violate)
----------------------------------
This module ONLY READS world_state. It never mutates the world and never imports
decision logic (strategy / trust / conversation / alliance / personality /
agents / llm / god_mode). The single project import is `world`, used only for
state-reading constants/helpers (grid symbols, HUNGER_MAX, is_sick). A snapshot
goes in; a rich renderable comes out — `render_frame()` is a pure function and is
asserted not to mutate the world by the test suite.

The boundary mirrors god_mode's: god_mode is the only thing that WRITES the world
outside the engine; the renderer is the only thing that DRAWS it, and it only
reads. Everything still funnels through the single source of truth.
"""

import contextlib
import os
import sys
from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# The ONLY project import: world is a state-reading layer (constants + pure reads).
from sim import world

# --- Stable per-agent colours --------------------------------------------
# A fixed palette; each agent maps to one entry deterministically by name, so an
# agent keeps the same colour for the whole run (and across runs/seeds). Chosen to
# stay legible on a dark terminal and distinct from food-green and treasure-yellow.
_PALETTE = (
    "bright_cyan",
    "bright_magenta",
    "bright_blue",
    "orange3",
    "spring_green2",
    "deep_pink2",
    "gold1",
    "turquoise2",
)

# Grid symbols (presentation only; the engine's own world.render() uses ASCII).
_FOOD_SYMBOL = "*"
_TREASURE_SYMBOL = "$"
_SETTLEMENT_SYMBOL = "#"   # M2.1: a settlement centre (read-only overlay)
_EMPTY_SYMBOL = "·"

# M0.3: above this many agents (or this grid edge) the per-letter grid stops being
# legible / fitting a terminal, so render_frame switches to the density heatmap. The
# small-cast per-letter dashboard is unchanged below the threshold.
_HEATMAP_AGENT_THRESHOLD = 40
_HEATMAP_GRID_THRESHOLD = 24
# The heatmap is downsampled to at most this many cells per side, so a 50x50 world
# still fits a terminal; each heat cell aggregates a block of the real grid.
_HEATMAP_MAX_CELLS = 24
# Ramp from empty -> dense; more agents in a block picks a later (hotter) glyph+colour.
_HEAT_RAMP = ((0, "·", "grey23"), (1, "░", "cyan"), (2, "▒", "green"),
              (4, "▓", "yellow"), (7, "█", "orange3"), (12, "█", "bright_red"))


def agent_color(name: str) -> str:
    """A stable colour for `name`, deterministic across processes and seeds.

    Uses a content hash (sum of code points) rather than Python's randomised
    builtin hash, so the same agent is always the same colour — which is what lets
    a viewer track an agent by colour as it moves.
    """
    return _PALETTE[sum(ord(c) for c in name) % len(_PALETTE)]


def _fullness_bar(hunger: int, width: int = 10) -> Text:
    """A draining 'fullness' bar: full+green = fed, empty+red = starving.

    This reads the way a viewer expects a survival meter to: the bar shows how much
    food reserve the agent has LEFT (HUNGER_MAX - hunger), so it drains toward empty
    as the agent starves and refills when it eats. Colour tracks the danger: green
    while comfortably fed, yellow as reserves run low, red on the brink of death.
    Pure read of the agent's hunger against world.HUNGER_MAX — the underlying stat
    is unchanged, only its presentation is flipped to fullness.
    """
    hmax = world.HUNGER_MAX
    fullness = max(0, hmax - hunger)  # reserve left; 0 == starving
    filled = 0 if hmax <= 0 else round(width * min(fullness, hmax) / hmax)
    filled = max(0, min(width, filled))
    ratio = fullness / hmax if hmax else 0
    color = "green" if ratio > 0.5 else ("yellow" if ratio > 0.2 else "red")
    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * (width - filled), style="grey37")
    return bar


def _trust_summary(agent: Any, limit: int = 2) -> str:
    """A short 'Bob:+3 Kira:-2' digest of the agent's strongest opinions.

    Reads agent.relationships ({other: {"trust": int, ...}}); shows the few with
    the largest magnitude so the panel stays compact. Empty string when the agent
    has no recorded opinions.
    """
    rels = getattr(agent, "relationships", {}) or {}
    scored = [(name, rel.get("trust", 0)) for name, rel in rels.items()]
    scored = [s for s in scored if s[1] != 0]
    scored.sort(key=lambda s: abs(s[1]), reverse=True)
    return " ".join(f"{name}:{t:+d}" for name, t in scored[:limit])


def _build_grid(state: dict[str, Any]) -> Table:
    """Render the size x size world as a rich table (pure READ of `state`).

    Agents show their coloured initial; food is green '*', treasure yellow '$',
    empty cells a muted dot. Built from the food/treasure lists and live agent
    positions — never the grid array — so the picture always matches the
    authoritative state (same rule as world.render()).
    """
    size = state["size"]
    food = set(state["food"])
    treasures = {t["pos"] for t in state.get("treasures", [])}
    # M2.1: settlement centres, a read-only overlay drawn UNDER food/treasure/agents
    # (a place persists even when momentarily empty of food or people). Pure read.
    settlements = {s["center"] for s in state.get("settlements", {}).values()}
    occupants = {
        a.position: a
        for a in state["agents"]
        if getattr(a, "alive", True)
    }

    grid = Table.grid(padding=(0, 1))
    for _ in range(size):
        grid.add_column(justify="center")

    for y in range(size):
        row: list[Text] = []
        for x in range(size):
            agent = occupants.get((x, y))
            if agent is not None:
                cell = Text(agent.name[0], style=f"bold {agent_color(agent.name)}")
            elif (x, y) in treasures:
                cell = Text(_TREASURE_SYMBOL, style="bold yellow")
            elif (x, y) in food:
                cell = Text(_FOOD_SYMBOL, style="green")
            elif (x, y) in settlements:
                cell = Text(_SETTLEMENT_SYMBOL, style="bold magenta")
            else:
                cell = Text(_EMPTY_SYMBOL, style="grey30")
            row.append(cell)
        grid.add_row(*row)
    return grid


def _heat_glyph(count: int) -> tuple[str, str]:
    """Pick the (glyph, colour) for a heat cell holding `count` agents (READ only)."""
    glyph, color = _HEAT_RAMP[0][1], _HEAT_RAMP[0][2]
    for threshold, g, c in _HEAT_RAMP:
        if count >= threshold:
            glyph, color = g, c
    return glyph, color


def _build_heatmap(state: dict[str, Any]) -> Table:
    """A downsampled agent-density heatmap for a large cast (pure READ of `state`).

    The per-letter grid is illegible past a few dozen agents, so the world is binned
    into at most _HEATMAP_MAX_CELLS^2 blocks; each block's glyph/colour ramps with how
    many living agents fall in it (food shows as green '*' only where a block is empty
    of agents). Built from live agent positions + the food list — never the grid array
    — so it matches the authoritative state, exactly like _build_grid().
    """
    size = state["size"]
    step = max(1, -(-size // _HEATMAP_MAX_CELLS))  # integer ceil, no math import
    blocks = -(-size // step)

    counts = [[0] * blocks for _ in range(blocks)]
    for a in state["agents"]:
        if getattr(a, "alive", True):
            ax, ay = a.position
            counts[ay // step][ax // step] += 1
    food_blocks: set[tuple[int, int]] = set()
    for fx, fy in state["food"]:
        food_blocks.add((fx // step, fy // step))

    grid = Table.grid(padding=(0, 0))
    for _ in range(blocks):
        grid.add_column(justify="center")
    for by in range(blocks):
        row: list[Text] = []
        for bx in range(blocks):
            n = counts[by][bx]
            if n:
                glyph, color = _heat_glyph(n)
                row.append(Text(glyph, style=f"bold {color}"))
            elif (bx, by) in food_blocks:
                row.append(Text(_FOOD_SYMBOL, style="green"))
            else:
                row.append(Text(_EMPTY_SYMBOL, style="grey23"))
        grid.add_row(*row)
    return grid


def _focal_summary(state: dict[str, Any], limit: int = 8) -> tuple[int, list[str]]:
    """(count, sample names) of agents currently on the focal/LLM tier (READ only).

    Reads each agent's `cognition` field — a plain state read, not decision logic.
    """
    focal = [a.name for a in state["agents"]
             if getattr(a, "alive", True) and getattr(a, "cognition", "llm") == "llm"]
    return len(focal), sorted(focal)[:limit]


def _build_scale_side_panel(state: dict[str, Any]) -> Panel:
    """Population / focal-set / food digest for the heatmap view (pure READ).

    Replaces the per-agent panel (which can't list 200 rows) with aggregate counts:
    living vs total, the focal (LLM) tier size + a sample, sick count, and food.
    """
    agents = state["agents"]
    total = len(agents)
    living = sum(1 for a in agents if getattr(a, "alive", True))
    sick = sum(1 for a in agents if getattr(a, "alive", True) and world.is_sick(a, state))
    focal_n, focal_names = _focal_summary(state)

    table = Table.grid(padding=(0, 1))
    table.add_column(justify="left", style="grey62", no_wrap=True)
    table.add_column(justify="left", no_wrap=True)
    table.add_row("population", Text(f"{living} alive / {total} total", style="bold cyan"))
    if sick:
        table.add_row("sick", Text(str(sick), style="bold magenta"))
    table.add_row("focal (LLM)", Text(f"{focal_n}", style="bold spring_green2"))
    if focal_names:
        sample = ", ".join(focal_names)
        if focal_n > len(focal_names):
            sample += f", +{focal_n - len(focal_names)} more"
        table.add_row("", Text(sample, style="grey70"))
    table.add_row("food on map", Text(str(len(state.get("food", []))), style="green"))
    return Panel(table, title="POPULATION", border_style="blue")


def _build_agents_panel(state: dict[str, Any]) -> Panel:
    """Per-agent status: name, fullness bar, alive/dead, allies, trust digest."""
    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(justify="left", no_wrap=True)   # name
    table.add_column(justify="left", no_wrap=True)   # "fed" label + fullness bar
    table.add_column(justify="left", no_wrap=True)   # status

    for agent in state["agents"]:
        color = agent_color(agent.name)
        alive = getattr(agent, "alive", True)
        name = Text(agent.name, style=f"bold {color}" if alive else "strike grey50")

        if not alive:
            status = Text("DEAD", style="bold red")
        elif world.is_sick(agent, state):
            status = Text("SICK", style="bold magenta")
        else:
            status = Text("alive", style="green")

        # Labelled so the meter is unmistakable: "fed ████░░░░░░" drains as the
        # agent starves (full+green = fed, empty+red = on the brink).
        fed = Text("fed ", style="grey62")
        fed.append_text(_fullness_bar(agent.hunger))
        table.add_row(name, fed, status)

        # Second line per agent: allies + trust digest, only when there's something.
        allies = ", ".join(sorted(getattr(agent, "allies", set())))
        trust = _trust_summary(agent)
        detail_parts = []
        if allies:
            detail_parts.append(f"allies: {allies}")
        if trust:
            detail_parts.append(f"trust {trust}")
        if detail_parts:
            table.add_row(Text(""), Text("  " + "   ".join(detail_parts),
                                         style="grey62"), Text(""))

    return Panel(table, title="AGENTS", border_style="blue")


def event_style(line: str) -> tuple[str, bool]:
    """Classify an event line into (rich style, is_major) for the EVENTS panel.

    The big moments are made unmissable at a glance when recording; routine chatter
    (talk, moves, trust ticks) stays muted so it never competes. Order matters: a
    BETRAYAL line also contains the word "alliance", so betrayal is matched first.

        DEATH     -> bold red          THEFT     -> bold orange3
        ALLIANCE  -> bold green        BETRAYAL  -> bold bright_red
        [GOD] ... -> bold yellow

    Pure string classification — it never reads or changes events[] content.
    """
    low = line.lower()
    if "[god]" in low or "god-script" in low:
        return "bold yellow", True
    if "betrayed" in low:                       # success: "*** X BETRAYED ... ***"
        return "bold bright_red", True
    if "died" in low:
        return "bold red", True
    if "stole" in low:                          # success: "X stole food from Y"
        return "bold orange3", True
    if "formed an alliance" in low:             # success: "X and Y formed an ALLIANCE"
        return "bold green", True
    # Softer, non-major cues for the lead-ups (proposals, newcomers).
    if "proposed an alliance" in low or "appeared" in low:
        return "green", False
    return "grey70", False


def _style_event(line: str) -> Text:
    """Render one event line, emphasising major moments with a colour + marker.

    Major events get a leading '●' bullet in their colour so they pop in a scan of
    the panel; routine lines are indented to align under it without a marker. Styling
    only — the text content is the verbatim events[] string.
    """
    style, major = event_style(line)
    text = Text()
    text.append("● " if major else "  ", style=style)
    text.append(line, style=style)
    return text


def _build_events_panel(state: dict[str, Any], limit: int = 12) -> Panel:
    """The most recent events (deaths, [GOD] interventions, alliances, betrayals)."""
    events = state.get("events", [])
    recent = events[-limit:]
    if recent:
        body: Any = Group(*[_style_event(e) for e in recent])
    else:
        body = Text("(no events yet)", style="grey50")
    return Panel(body, title="EVENTS", border_style="magenta")


def _use_heatmap(state: dict[str, Any]) -> bool:
    """Whether the large-cast heatmap view fits better than the per-letter grid.

    Triggered by EITHER a big population (the per-letter initials become an
    unreadable smear) OR a large grid (won't fit a terminal). Small casts keep the
    original per-agent dashboard, so nothing about the v1/M0.1/M0.2 view changes.
    """
    living = sum(1 for a in state["agents"] if getattr(a, "alive", True))
    total = len(state["agents"])
    return (max(living, total) > _HEATMAP_AGENT_THRESHOLD
            or state.get("size", 0) > _HEATMAP_GRID_THRESHOLD)


def render_frame(state: dict[str, Any]) -> Layout:
    """Build the full dashboard renderable from a `state` snapshot (PURE READ).

    Auto-selects the view by scale (M0.3): a small cast gets the per-letter grid +
    per-agent panel; a large cast (or large grid) gets a density heatmap + aggregate
    population/focal panel. Either way it returns a rich renderable and mutates
    NOTHING — the boundary test drives this to assert world_state is unchanged after
    a render, at small AND large scale.
    """
    turn = state.get("turn", 0)
    food_n = len(state.get("food", []))
    living = sum(1 for a in state["agents"] if getattr(a, "alive", True))
    heatmap = _use_heatmap(state)

    grid_panel = Panel(
        _build_heatmap(state) if heatmap else _build_grid(state),
        title=f"World — turn {turn}" + ("  (density)" if heatmap else ""),
        subtitle=f"food {food_n}   living {living}",
        border_style="green",
    )
    side_top = _build_scale_side_panel(state) if heatmap else _build_agents_panel(state)

    layout = Layout()
    layout.split_row(
        Layout(grid_panel, name="grid", ratio=1),
        Layout(name="side", ratio=1),
    )
    layout["side"].split_column(
        Layout(side_top, name="agents", ratio=1),
        Layout(_build_events_panel(state), name="events", ratio=1),
    )
    return layout


class RichRenderer:
    """Drives a `rich.live.Live` dashboard fed from world_state each turn.

    The renderer DRAWS to the real terminal (`sys.__stdout__`) via its own
    Console, so it is unaffected by any stdout redirection the caller sets up to
    keep the plain per-turn text out of the dashboard. `sink` is where that plain
    text is redirected to: the run's log file under --log (so the log still
    captures the byte-for-byte plain run), else os.devnull.

    Usage::

        r = RichRenderer()
        with r.live():
            for turn in ...:
                ...advance the world...
                r.update(world_state)
    """

    def __init__(self, sink: Any | None = None) -> None:
        # Bind to the TRUE terminal so the live view never lands in the log file
        # and never follows a redirect_stdout the engine may install.
        self.console = Console(file=sys.__stdout__)
        self._owns_sink = sink is None
        self.sink = sink if sink is not None else open(os.devnull, "w")
        self._live: Live | None = None

    @contextlib.contextmanager
    def live(self):
        """Context manager owning the Live display for the duration of a run.

        `screen=False` so the dashboard redraws in place on a TTY rather than
        scrolling, while still degrading to sequential frames when piped/captured.
        """
        with Live(console=self.console, screen=False, auto_refresh=False,
                  transient=False) as live:
            self._live = live
            try:
                yield live
            finally:
                self._live = None
                if self._owns_sink:
                    self.sink.close()

    def update(self, state: dict[str, Any]) -> None:
        """Redraw the dashboard from the latest `state` (READ only)."""
        frame = render_frame(state)
        if self._live is not None:
            self._live.update(frame, refresh=True)
        else:
            # No live context (e.g. a one-shot render) — just print one frame.
            self.console.print(frame)
