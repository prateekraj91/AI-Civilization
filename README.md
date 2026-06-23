# AI Civilization

A multi-agent simulation where every citizen is a **real LLM-driven agent**. Each
agent has a personality, weighted goals (survive / wealth / friendship), bounded
memory, and a voice — and they all share one 10×10 world and one scarce food
supply. From nothing but survival pressure and the ability to perceive and talk to
each other, **social behaviour emerges**: agents build trust, form alliances, share
what they see, steal when desperate, and occasionally betray the very allies that
kept them alive. And you are not a spectator — **you are God**, and can reach into
the running world to start a drought, unleash a plague, drop treasure, or introduce
a stranger, then watch the society react.

No event is scripted. The drama (and, just as often, the *lack* of drama) is what
the agents actually did.

---

## Quickstart

### 1. Install

```bash
# clone, then from the repo root:
python3 -m venv .Jarvis
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Local LLM via Ollama + Qwen

The default provider is a **local** model through [Ollama](https://ollama.com).
The simulation never talks to a model directly — it goes through one
provider-agnostic layer (`llm.py`), so this step is only needed if you want real
LLM reasoning rather than the offline backend.

```bash
# install Ollama (see ollama.com), then pull the model the sim defaults to:
ollama pull qwen3:8b
ollama serve            # serves at http://127.0.0.1:11434 (the sim's default URL)
```

No GPU / don't want to run a model? Skip this and use `AICIV_PROVIDER=random` — a
fully offline backend that returns plausible, valid actions. Every command below
works offline by prefixing `AICIV_PROVIDER=random`.

### 3. Run it

```bash
# A) plain text simulation (turn-by-turn log to the terminal)
python main.py --turns 30
#    offline, no model server needed:
AICIV_PROVIDER=random python main.py --turns 30

# B) live rich dashboard — grid + per-agent fullness bars + highlighted events,
#    paced so a human can watch it unfold
python main.py --render rich --speed normal --turns 30

# C) a reproducible, god-scripted run (the demo invocation):
python main.py --seed 7 --turns 40 --render rich --speed slow \
    --god-script "10:trigger_plague Kira;10:trigger_drought;25:drop_treasure 5 5" \
    --log logs/my_demo.txt
```

Run the tests (offline, deterministic — does not contact Qwen):

```bash
AICIV_PROVIDER=random python test_simulation.py
```

---

## Key flags & environment

| Flag / env | What it does |
|---|---|
| `--turns N` | Number of turns to simulate (default 50, or until everyone has died with no respawn pending). |
| `--seed N` | Seed Python's RNG **before** world setup. Fixes agent/food placement *and* the offline `random` provider, so a seeded offline run replays identically. (Qwen sampling is not fully deterministic, so a seed fixes the *world*, not Qwen's word choices.) |
| `--render rich` | Live in-place dashboard via the `rich` library instead of plain text. **Read-only** — it never touches the simulation. Default is plain text. |
| `--speed slow\|normal\|fast\|<secs>` | Pacing for a **rendered** run: slow ≈ 2.0s/turn, normal ≈ 0.5s/turn (default), fast ≈ 0.1s/turn, or a raw number (`--speed 0.3`). Presentation-only — never affects tests, plain/logged runs, or the RNG. |
| `--god-script SPEC` | Run God interventions non-interactively. Inline `"10:trigger_plague Kira;25:drop_treasure 5 5"` or a path to a file of `<turn>:<command>` lines. Each fires at the end of its turn — so a scripted run reproduces a hand-played one exactly. |
| `--log PATH` | Mirror the full plain run (turn log + summary + events) to a file. Coexists with `--render rich`: the dashboard owns the terminal while the plain transcript is captured byte-for-byte to the log. |
| `AICIV_PROVIDER` | `ollama` (default, local Qwen), `gemini` (cloud, needs `GEMINI_API_KEY`), or `random` (offline). |
| `AICIV_GOD_EVERY` | Drop into the **interactive** God menu every N turns (default 0 = off). Ignored when `--god-script` is given so automated runs never block on input. |

**God commands** (for `--god-script` or the interactive menu): `trigger_drought
[turns]`, `trigger_plague [name]`, `drop_treasure <x> <y> [value]`, `spawn_food <x>
<y>`, `spawn_agent <name> <personality...>`, `introduce_stranger <name>
[personality...]`, `status`, `help`.

---

## Architecture

Everything funnels through one authoritative dict, **`world_state`** — the single
source of truth. Three subsystems touch it, each across a hard, test-enforced
boundary:

```
                       ┌───────────────────────────────────┐
                       │            world_state             │
                       │   (the single source of truth)     │
                       │  turn · grid · agents · food ·      │
                       │  treasures · events · respawns ...  │
                       └───────────────────────────────────┘
                          ▲              │              │
              READ state  │              │ READ state   │ READ state
              return an   │              │ (perceive)   │ (snapshot)
              action      │              ▼              ▼
        ┌─────────────────┴──┐   ┌──────────────┐  ┌──────────────────┐
        │   AGENTS — DECIDE  │   │ god_mode.py  │  │   renderer/      │
        │  agents + strategy │   │   MUTATES    │  │   DISPLAYS       │
        │  + conversation +  │   │ (write-only) │  │  (read-only)     │
        │  trust + alliance  │   │  drought,    │  │  rich dashboard: │
        │                    │   │  plague,     │  │  grid, hunger,   │
        │  decisions flow    │   │  treasure,   │  │  events panel    │
        │  BACK through the  │   │  newcomers   │  │                  │
        │  world layer ──────┼──▶│──────────────┼─▶│  never writes    │
        │  (never poke state)│   │  the ONLY    │  │  state; a stale  │
        └────────────────────┘   │  writer from │  │  mark can't leak │
                                  │  outside the │  └──────────────────┘
                                  │  engine      │
                                  └──────────────┘
```

- **Agents DECIDE.** Agents read `world_state` to perceive, then return one of a
  closed set of actions; all world changes flow back through the world layer, never
  by an agent writing globals. (`agents.py`, `strategy.py`, `conversation.py`,
  `trust.py`, `alliance.py`, `personality.py`)
- **God MUTATES (write-only).** `god_mode.py` is the only thing outside the engine
  that *writes* the world. It imports no decision logic — an AST test enforces it.
- **The renderer DISPLAYS (read-only).** `renderer/text_renderer.py` turns a
  snapshot into a `rich` dashboard and mutates nothing. It imports only `rich` +
  `world` — another AST boundary test enforces it.

Two more design points worth knowing:

- **Strategy caching = zero-inference turns.** The LLM is asked for a *high-level
  strategy* only once every few turns (`STRATEGY_INTERVAL`, default 5); in between,
  that cached plan is executed in pure Python. So ~80% of agent-turns make **no LLM
  call at all** — cheaper, faster, and still reproducible. Death, respawn, trust,
  alliances and God interventions are all pure Python and add **zero** inference.
- **Provider abstraction.** The simulation has no idea which model is behind it.
  `llm.py` dispatches to Ollama (local Qwen, default), Gemini (cloud), or a `random`
  offline backend, and always degrades to a safe fallback so a model hiccup can
  never crash a run.

---

## What emerged

The interesting results are written up in two companion docs:

- **[DEMO_STORY.md](DEMO_STORY.md)** — a narrative walkthrough of a single run: a
  drought-driven collapse and an alliance that formed, unprompted, late in the game.
- **[FINDINGS.md](FINDINGS.md)** — the running day-by-day log of what each mechanic
  actually produced, including the dead ends.

One honest finding deserves top billing: **the dramatic social acts — betrayal,
alliance — are RARE under competent play.** When agents play survival well, they
mostly just eat, wander, and exchange the occasional message; full-blown betrayals
emerge only when scarcity squeezes hard enough that cooperating and defecting
genuinely diverge. That rarity isn't a gap in the write-up — it *is* the result.
A world where betrayal is common would be a world whose incentives were mis-tuned.
The mechanics make dramatic acts *possible*; the agents reserve them for when they
actually pay off, which is exactly what makes the rare ones feel earned.

---

## Roadmap / V2

V1 is a survival sandbox with emergent *social* behaviour. V2 points at **emergent
civilization** — letting technology, governance, and economics arise from agent
decisions rather than from new hard-coded rules: agents that trade and accumulate,
specialise, agree on norms, and build institutions that outlive any single citizen.
The architecture (one source of truth, write-only God, read-only display, a
provider-agnostic mind) is built to grow in that direction.

---

## Repo layout

```
world.py            single source of truth + perception/movement/hunger
agents.py           the Agent data model (pure data, no logic)
personality.py      typed-trait instincts per agent
strategy.py         LLM strategy prompt + cached Python execution
conversation.py     talk / steal + message delivery
trust.py            per-relationship trust bookkeeping
alliance.py         alliances, shared sightings, betrayal
population.py       death events + blank-slate respawn
god_mode.py         write-only world interventions (the "you are God" layer)
llm.py              provider-agnostic model layer (Ollama / Gemini / random)
renderer/           read-only rich terminal dashboard
main.py             setup + the shared survival loop + CLI
test_simulation.py  60 deterministic tests (run with AICIV_PROVIDER=random)
logs/               captured demo runs
FINDINGS.md         day-by-day findings
DEMO_STORY.md       narrative of a standout run
```
