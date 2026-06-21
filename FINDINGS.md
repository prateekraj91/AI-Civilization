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

## Day 13 (alliances + betrayal)

I added a mutual, two-sided alliance: one agent proposes via `ally_with_<name>`, and the bond only forms when the other answers with its own `ally_with`. Forming grants +3 trust **both ways**, logs an ALLIANCE event, and writes a memory on both. The benefit is mechanically real, not cosmetic: allies **share food sightings** — each contributes the food in its own perception window, so a pair sees more of a scarce map than either alone. The sharing is folded straight into the partner's strategy prompt ("Food your allies can see (shared with you): Alex sees food at (4, 3)"), and it costs zero new inference — ally and betray ride the same cached strategy call as talk and steal. Betrayal (`betray_alliance_<name>`) dissolves the bond, drops the betrayed agent's trust by 8 (bigger than theft's 5), latches a **permanent grudge** (reusing the Day 12 flag), and records a major memory on both. A grudge on *either* side blocks the pair from ever allying again, and a betrayed ally stops receiving shared sightings the instant the alliance ends.

**Who allies and who betrays is personality, not script.** Friendly Alex and cautious Bob ally readily — once talk has built trust to "high", one proposes and the other accepts. Independent/competitive Kira allies only reluctantly (she joins forces solely with someone she already actively trusts), and she is the only personality that will betray: under real survival pressure, starving beside an ally hoarding food, she renounces the alliance rather than keep paying into it. Friendly and cautious agents never betray the alliances they form.

**What I actually saw, by provider (being precise about provenance):**

- **Offline `random` provider, seed 48 — alliance + sharing emerged on its own.** Alex and Bob talked, trust climbed to "high", Bob proposed on turn 4, Alex accepted on turn 7, and the alliance formed (+3 both ways, event + memories logged). That same turn Bob — who could see no food himself — received Alex's private sighting of food at (4,3) through the shared-perception channel, verbatim in his next strategy prompt. The alliance is produced by the deterministic strategy *executor* (the `random` backend never itself emits "ally"); talk built the trust, and the executor turned it into a mutual bond. This is genuinely emergent — I never named who would ally.

- **Qwen (qwen3:8b, think-off) — no alliance formed in the seeds I ran.** Across the full verify run plus two probe seeds (48, 1), the agents talked only ~3 times each before scarcity scattered them, never reaching the trust an alliance needs; one agent (Kira on seed 48, Bob on seed 1) simply out-competed the others for the central food and survived alone. So under Qwen the *competitive collapse* keeps recurring — the same pattern as Day 12's no-theft runs: when one agent dominates the commons, nobody lives long enough beside a trusted neighbour to form a society. I did **not** observe a Qwen alliance, and I'm not claiming one.

- **Betrayal — verified only as a constructed deterministic scenario, not an organic Qwen event.** Because Kira so rarely enters an alliance at all (offline or under Qwen), I built the allied state through the real handlers, then let the real executor decide: starving Kira beside ally Alex-on-food emitted `betray_alliance_Alex` on her own. The alliance dissolved both sides, Alex's trust fell +7 → -1 with a permanent grudge, both memories recorded it, and the pair could no longer re-ally from either direction. The *mechanic* is real production code and is covered by regression tests; the *setup* is hand-built, and I say so in the harness output rather than dressing it up as something Qwen did.

The honest throughline across Days 11–13: scarcity reliably produces competition under Qwen, but cooperation (talk → trust → alliance) needs the agents to survive near each other long enough to build a relationship — and in these scarce-commons seeds, the strongest competitor usually ends that possibility before it starts. Alliances form readily once trust exists; getting to trust under Qwen scarcity is the hard part.

Day 13 (alliances + betrayal): Added mutual alliances (shared food sightings as the benefit) and betrayal (-8 trust, permanent grudge, no re-allying). Both verified by tests and deterministic runs. Under Qwen, no alliance formed on any seed I tried — same as theft in Day 12. The pattern is now clear: when the model plays competently, the dramatic social actions (steal, ally, betray) almost never fire on their own. Agents compete efficiently and one out-survives the rest; cooperation needs them alive near each other long enough to build trust, and scarcity usually ends that window first. Caveat: the alliance benefit only shapes LLM-driven turns, since the Python executor already navigates to all food globally.

## Day 14 (death event + respawn)

Death already existed since Day 6 (starvation flips `alive` and frees the cell). Day 14 turns it from a silent removal into something the *society* registers, and then refills the population. When an agent starves now: a clear `turn 47: Kira died (starved)` line goes to `events[]`, and EVERY agent that was alive at that moment records a bounded memory ("Kira died on turn 47 — they were starving."). Crucially, the survivors' *relationships* toward the deceased are left untouched — a grudge or an alliance with the dead persists as a remembered fact. The dead are remembered. (Pronoun is gender-neutral "they"; we can't infer gender from a name.)

Respawn is a true social cold-start. A death queues exactly one respawn for `death_turn + RESPAWN_DELAY` (10). When it comes due — and only while the living population is below `TARGET_POPULATION` (3) — a brand-new agent enters from a roster distinct from the starting cast (Mira/Otto/Zane/Nova), placed on a valid empty cell near the contested centre. It has blank memory, empty relationships, empty allies, hunger reset to 0, and a name no living *or dead* agent uses (reusing a dead name would silently graft the deceased's reputation onto the newcomer, since names key every relationship map). No one trusts it; it knows no one. Living agents record "A new agent, Mira, appeared on turn 57." The population rule keeps things bounded: each death is a deficit of one, each respawn fills one, surplus respawns (target already met) are dropped — N → N-1 → N, never unbounded.

Like every social mechanic since Day 8, this adds **zero** new per-turn inference. Death and respawn are pure Python over `world_state`; the newcomer's first turn triggers the same ordinary periodic strategy refresh every agent makes. The benchmark confirms it: ~0.21 strategy calls per agent-turn (offline and Qwen), unchanged from before — newcomers included.

**What I actually saw, by provider (being precise about provenance):**

- **Offline `random` provider, seed 7 — death + respawn ACTUALLY OCCURRED end-to-end.** Kira starved on turn 26; Alex and Bob both recorded the death, and Alex's prior distrust of Kira (trust -3) persisted afterward. On turn 36 (= 26 + 10, exactly) blank-slate Mira entered at the centre with empty memory/relationships and hunger 0; the survivors recorded her arrival. The full `main.py` run chains it naturally — Bob and Kira die, Mira enters turn 27, later Alex dies and Otto enters turn 36 — population repeatedly pulled back toward 3.

- **Qwen (qwen3:8b, think-off), seed 7 — same mechanic, regression-confirmed.** Bob starved on turn 17, both survivors recorded it, and Mira entered on turn 27 (= 17 + 10) as a blank slate. I ran Qwen ONCE as a regression check only — death and respawn are deterministic mechanics, not emergent events, so there is nothing to seed-search for. Per-turn inference was unchanged (0.221 calls/agent-turn).

- **Immediate participation — verified as a deterministic structural check.** "The newcomer can be observed and talked to" is a structural property (respawn registers a normal `Agent` on the grid), not an emergent one, so I prove it with a constructed fixture using the real respawn + talk code: a neighbour's `observe()` detects the fresh newcomer and `handle_talk()` delivers a message to its inbox. I deliberately do NOT rely on whether a competitive Qwen agent happens to talk to the newcomer before it scatters — in the Qwen seed-7 run, Mira died later without being talked to, which is exactly the kind of emergent outcome I'm not claiming as a mechanic test.

The honest throughline: Day 14 is a *mechanic*, fully deterministic, and verified offline and confirmed once under Qwen. The only emergent layer on top — who happens to befriend or rob the newcomer — remains rare under competent Qwen play, consistent with Days 12–13.

Day 14 (death + respawn): When an agent starves, survivors record the death in memory and a death event logs; after a delay, blank-slate newcomers enter knowing no one. This fired live under Qwen — Alex and Kira died at turn 17, Mira and Otto entered at 27 with empty memory and relationships, and Mira later died too, her death recorded by the survivors just like the originals. The world now has a continuous history: agents cycle in and out, but the living carry memory of the dead. Bob's end-of-run memory reads like a short biography of everyone he met and outlived.