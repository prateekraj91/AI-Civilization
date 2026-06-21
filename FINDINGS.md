In a 40-turn run with three agents, Bob and Alex talked repeatedly and built mutual trust up to +5, while Alex's trust in Kira fell to -6 after their interactions went badly.

I never wrote any rule telling Alex to dislike Kira or to bond with Bob — those relationships emerged on their own from the talk → reaction → trust loop.

Watching a grudge form between agents I didn't script is the first time the simulation felt like a society rather than three separate programs.

---

Day 11 (food scarcity): I reversed the Day 9 abundance on purpose — INITIAL_FOOD 14 -> 3, and respawn went from "refill to 12 every turn" to a drip of ~1 food every 5 turns. Three agents each need a meal roughly every 7 turns, so demand (~0.43 food/turn) now outruns supply (~0.20 food/turn).

My first cut was scarce but TOO scarce in the wrong way: agents dispersed and starved alone in separate corners — 0 talks, 0 trust changes. Scarcity without a shared resource isn't competition, it's just three lonely deaths. Two small changes fixed it without making the world generous: I raised INITIAL_FOOD 3 -> 5 (they live long enough to meet) and CLUSTERED the scarce food into a 5x5 arena at the centre, right where they spawn. Now they converge on the same tiles and fight over them.

In a 50-turn Qwen run (qwen3:8b, think-off), the agents stood adjacent on 4 turns, Alex talked to Bob twice, and Bob's trust in Alex rose 0 -> 1 -> 2 — all while food was still scarce enough that Bob starved on turn 17, Alex on turn 20, and Kira (who ate 7 times by camping the central food) on turn 39, each dying mid-movement, not stuck. I scripted none of it: who befriends whom, who hoards, who starves — clustering a scarce resource was enough to turn dispersal into a contested commons.

