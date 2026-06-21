# AI Civilization — Findings

## Day 10 (first emergent moment)

In a 40-turn run with three agents, Bob and Alex talked repeatedly and built mutual trust up to +5, while Alex's trust in Kira fell to -6 after their interactions went badly.

I never wrote any rule telling Alex to dislike Kira or to bond with Bob — those relationships emerged on their own from the talk → reaction → trust loop.

Watching a grudge form between agents I didn't script is the first time the simulation felt like a society rather than three separate programs.

## Day 11 (food scarcity)

I reversed the Day 9 abundance on purpose — INITIAL_FOOD 14 -> 3, and respawn went from "refill to 12 every turn" to a drip of ~1 food every 5 turns. Three agents each need a meal roughly every 7 turns, so demand (~0.43 food/turn) now outruns supply (~0.20 food/turn).

My first cut was scarce but TOO scarce in the wrong way: agents dispersed and starved alone in separate corners — 0 talks, 0 trust changes. Scarcity without a shared resource isn't competition, it's just three lonely deaths. Two small changes fixed it without making the world generous: I raised INITIAL_FOOD 3 -> 5 (they live long enough to meet) and CLUSTERED the scarce food into a central arena where they spawn. Now they converge on the same tiles and compete over them.

In a 50-turn Qwen run (qwen3:8b, think-off), a social split emerged on its own: Bob and Alex traded friendly messages and built mutual trust while Alex's exchanges with Kira turned hostile and his trust in her fell to -6. Food stayed scarce enough that agents still starved under the pressure. I scripted none of it — who allies, who feuds, who hoards, who starves — clustering a scarce resource was enough to turn dispersal into a contested commons.

## Day 12 (stealing + grudges)

I added a steal action — rob a neighbour's food when starving and you distrust them — riding the same strategy call as talk, so it cost zero extra inference. A theft drops the victim's trust by 5 and latches a permanent grudge flag, so no later friendliness can repair it.

The mechanic is verified by regression tests and a deterministic offline run: Kira steals from Alex twice, his trust slides -3 -> -8 -> -13, a later friendly message is REFUSED (you can't be liked back into good standing after stealing), and both memories are logged. The grudge then rides into Alex's every later prompt as "Kira: -13 (low, grudge)" — a betrayal is the first thing in this world that can't be undone.

The interesting part is who steals. Kira (independent, competitive) is the only personality that self-selects into theft — friendly Alex and cautious Bob only rob someone they already actively distrust, but Kira's survival-over-relationship wiring makes her a thief. I never named a thief; her personality made her one.

But under Qwen, competent play kept agents fed enough that the theft window rarely opened in the seeds I tried — which is itself the finding: when the model plays well, betrayal stays a genuine last resort, not a habit. In a clean 50-turn Qwen run no theft occurred at all — instead Kira simply out-competed Bob and Alex for the scarce central food and was the lone survivor at turn 50, while Bob died heading toward her food source.