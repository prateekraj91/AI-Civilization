"""
narrative
=========

THE WORLD'S ACCOUNT OF ITSELF — the run turned into something a person reads.

What lives here
---------------
`chronicle` is the structured record: the event stream typed into figures, houses, wars and
risings, with fidelity gated on whether the world had WRITING at the time. `chronicle_book`
turns that record into a HISTORY BOOK — prose chapters, towns and figures under their own
deterministic names (the same names the renderer labels the map with, via `place_name_map`).
`narrator` is the optional LLM retelling, walled off by design: it reads and returns prose, and
can never write back into the record.

Boundary
--------
Presentation over the run, never part of it: pure over world_state, deterministic from the seed,
and never consulted by a decision.
"""
