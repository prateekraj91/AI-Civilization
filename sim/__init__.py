"""
sim
===

THE WORLD AND ITS INSTITUTIONS — the simulation proper.

What lives here
---------------
Everything that IS the world or acts on it: the grid and its state (`world`), the people
(`agents`, `personality`, `population`, `lineage`), what they know and make (`knowledge`,
`writing`, `metallurgy`, `eras`), what they hold (`economy`, `storage`, `labor`, `taxation`),
who they answer to (`leadership`, `monarchy`, `kingdoms`, `empire`, `coalitions`, `diplomacy`,
`intertrade`, `alliance`), what they believe (`beliefs`, `religion`, `culture`), and what they
do about their rulers (`discontent`, `uprising`). `god_mode` is the operator's write channel;
`scenario` stages a world mid-history so a late system can be watched from turn one.

Boundary
--------
No module here imports the renderer, and none reaches for an LLM directly — character enters
through `llm.mind` at the pivots, called from the institution that is deciding. `world` is the
one state module everything else reads; the dependency arrows point at it, never out of it.
"""
