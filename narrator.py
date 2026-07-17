"""
narrator.py
===========

THE OPTIONAL LLM NARRATOR (V2 milestone M4.16 — a clearly-separated, OFF-by-default GARNISH layer).

This is the ONE place an LLM touches OUTPUT — and it touches ONLY output. It takes the DETERMINISTIC
structured chronicle entries that `chronicle.py` produced and renders them as PROSE ("In the eighth
generation, Rex the Conqueror rose from a common farmer..."). It is:

  * OFF BY DEFAULT (the `--narrate` flag). When off, the chronicle is the deterministic structured text
    and no model is ever called.
  * PURELY PRESENTATIONAL. It reads the structured chronicle and returns strings; it NEVER mutates
    world_state, the sim, or the structured `world_state["chronicle"]` record. Turning it on cannot
    change the simulation, its determinism, or the verifiable structured saga — only the prose wrapper.
  * CACHED per entry (a module-local cache), so re-narrating an entry costs nothing.

Because it is walled off entirely from the verifiable core, ALL verification runs with the narrator OFF
and the structured chronicle is identical whether it is on or off (verified in the tests).
"""

from __future__ import annotations

from typing import Any

import chronicle

_CACHE: dict[str, str] = {}


def _prose_for(entry: dict[str, Any]) -> str:
    """Render one structured saga entry as a single prose sentence via the local model. CACHED. This is
    the only LLM call in the whole chronicle system; it is reached only when --narrate is on."""
    key = f"{entry['turn']}|{entry['name']}|{entry['detail']}"
    if key in _CACHE:
        return _CACHE[key]
    import llm
    prompt = ("Rewrite this single historical record as one vivid sentence of a chronicle, in the past "
              "tense, adding no facts not present:\n"
              f"Turn {entry['turn']}: {entry['name']} — {entry['detail']}.")
    try:
        text = llm.complete(prompt).strip() if hasattr(llm, "complete") else ""
    except Exception:
        text = ""
    if not text:  # any failure / offline provider falls back to the structured text (never blocks)
        text = f"In turn {entry['turn']}, {entry['detail']}."
    _CACHE[key] = text
    return text


def narrate_saga(state: dict[str, Any]) -> str:
    """The saga rendered as LLM prose (one sentence per entry). Read-only; used ONLY under --narrate.

    The structured `world_state['chronicle']` is untouched — this only wraps the entries `chronicle.saga`
    exposes. If the LLM is unavailable the prose falls back to the structured detail, so it never fails."""
    lines = ["# The Chronicle of the Age (narrated)", ""]
    for e in chronicle.saga(state):
        tag = "" if e["fidelity"] == "history" else " (a legend, its names lost)"
        lines.append(f"- {_prose_for(e)}{tag}")
    return "\n".join(lines)
