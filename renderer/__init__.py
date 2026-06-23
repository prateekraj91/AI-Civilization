"""
renderer
========

Presentation layer for AI Civilization (Day 18).

ARCHITECTURE RULE (same boundary as god_mode, but stricter)
-----------------------------------------------------------
The renderer ONLY READS `world_state`. It must NEVER mutate world_state and must
never touch decision logic (strategy / trust / conversation / alliance /
personality / agents / llm / god_mode). The only project module it may import is
`world`, and only for the state-reading constants/helpers it needs to turn a
snapshot into pixels (grid symbols, HUNGER_MAX, is_sick).

A given snapshot in -> a rich renderable out. No side effects. This is enforced
by an AST boundary test mirroring the god_mode one.
"""

from renderer.text_renderer import RichRenderer, render_frame

__all__ = ["RichRenderer", "render_frame"]
