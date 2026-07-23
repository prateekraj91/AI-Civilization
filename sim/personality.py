"""
personality.py
==============

Turns an agent's free-text personality (e.g. "cautious and territorial") into a
small set of TYPED TRAITS that the simulation can act on (Phase 1).

Why a separate layer?
---------------------
The personality string is human-friendly but useless to code. This module is the
one place that maps words → numbers, so the rest of the system reasons about
traits (`friendliness`, `caution`, ...) instead of parsing prose. Add a keyword
or a trait here and every behaviour picks it up automatically.

Four archetypes are modelled, matching the milestone's examples:
  - curiosity     → explores more, rarely rests
  - caution       → hugs known food, eats early, rests when safe
  - friendliness  → moves toward other agents
  - independence  → keeps away from other agents, explores alone

Traits are independent 0.0–1.0 scores (an agent can be both curious AND
friendly). The single `dominant` trait is used for quick behavioural switching;
the raw scores tune thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

# Trait name -> words that imply it. Matching is substring/word based and
# case-insensitive, so "territorial" counts toward caution, "competitive"
# toward independence, etc.
TRAIT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "curiosity": ("curious", "adventurous", "explorer", "exploring", "inquisitive", "bold"),
    "caution": ("cautious", "careful", "timid", "territorial", "defensive", "wary"),
    "friendliness": ("friendly", "outgoing", "social", "sociable", "kind", "cooperative"),
    "independence": ("independent", "competitive", "aloof", "solitary", "loner", "lone"),
}

TRAIT_NAMES: tuple[str, ...] = tuple(TRAIT_KEYWORDS.keys())

# Tie-break order when two traits score equally (and the default for an
# unrecognised personality, which scores all traits 0.5). Earlier = wins.
# Curiosity leads so a personality with no recognised keywords behaves as a
# neutral curious wanderer rather than silently latching onto another trait.
DOMINANCE_ORDER: tuple[str, ...] = ("curiosity", "caution", "friendliness", "independence")


@dataclass(frozen=True)
class Personality:
    """Typed personality traits derived from an agent's description.

    Each field is a 0.0–1.0 strength. The class is frozen (immutable) because a
    personality never changes mid-run; it is cached per agent in strategy.py.
    """

    curiosity: float = 0.5
    caution: float = 0.5
    friendliness: float = 0.5
    independence: float = 0.5

    @classmethod
    def from_text(cls, text: str) -> "Personality":
        """Parse a personality description into normalised trait scores.

        Counts keyword hits per trait, then normalises by the strongest trait so
        the dominant trait sits at 1.0. A description with no recognised keywords
        yields a balanced personality (all traits 0.5, curiosity dominant via the
        tie-break order) so behaviour is still well-defined.
        """
        lowered = text.lower()
        counts = {
            trait: sum(lowered.count(word) for word in words)
            for trait, words in TRAIT_KEYWORDS.items()
        }
        peak = max(counts.values())
        if peak == 0:
            return cls()  # all defaults (0.5)
        scores = {trait: counts[trait] / peak for trait in TRAIT_NAMES}
        return cls(**scores)

    @property
    def dominant(self) -> str:
        """The single strongest trait (ties broken by DOMINANCE_ORDER)."""
        scores = {
            "curiosity": self.curiosity,
            "caution": self.caution,
            "friendliness": self.friendliness,
            "independence": self.independence,
        }
        best = max(scores.values())
        for trait in DOMINANCE_ORDER:
            if scores[trait] == best:
                return trait
        return "curiosity"  # unreachable, but keeps the type a plain str

    @property
    def eat_threshold(self) -> int:
        """Minimum hunger at which the agent bothers to eat food it's standing on.

        Cautious agents eat early (low threshold) to stay safe; bolder agents let
        hunger build before spending a turn eating. Always >= 1 so an agent never
        wastes an eat at hunger 0. Day 9 rebalance lowered the base (3 -> 2) so
        agents opportunistically top up when already on food, staying fuller and
        freeing turns for social interaction.
        """
        return max(1, round(2 - 2 * self.caution))

    @property
    def comfort(self) -> int:
        """Hunger level a cautious agent will rest below when near food.

        Higher caution → larger comfort band → more resting near a food cache.
        """
        return 3 + round(3 * self.caution)

    def describe(self) -> str:
        """One-line human summary, e.g. 'caution 1.0, curiosity 0.5 (dominant: caution)'."""
        ordered = sorted(
            (("curiosity", self.curiosity), ("caution", self.caution),
             ("friendliness", self.friendliness), ("independence", self.independence)),
            key=lambda kv: -kv[1],
        )
        parts = ", ".join(f"{name} {score:.1f}" for name, score in ordered)
        return f"{parts} (dominant: {self.dominant})"
