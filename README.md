# AI Civilization

A sandbox simulation where AI-driven agents inhabit a shared world, perceive their
surroundings, and act according to their personalities and goals — powered by a local
LLM via Ollama (default), with Google Gemini and an offline backend also supported
behind one provider-agnostic interface.

## What is AI Civilization?

AI Civilization is an experiment in emergent, agent-based simulation. Each inhabitant is
an autonomous **Agent** with a personality, a weighted set of goals (e.g. survival,
wealth, friendship), and a memory of what it has experienced. Agents read from a single
authoritative **world state** to decide how to behave, and a language model (Gemini)
gives them their voice and reasoning.

The long-term vision is a living world you can observe and intervene in — watching
agents form relationships, pursue goals, and react to events over many simulation turns.

## Current Status — Personality & Goal-Driven Behaviour ✅

The simulation now runs a **multi-agent world** end to end, and the agents
**behave differently** according to who they are:

- ✅ World layer (`world_state`) as the single source of truth
- ✅ `Agent` data model (personality, goals, hunger, position, memory, **alive**)
- ✅ 10×10 grid, movement, food, and a per-turn **hunger / starvation** system
- ✅ Provider-agnostic LLM layer: **Ollama** (default, local), **Gemini** (cloud),
  **random** (offline), with a graceful `rest` fallback that never crashes
- ✅ Bounded per-agent **memory** (last 20 events, oldest discarded)
- ✅ **Day 6 — Multiple agents:** Alex, Bob, Kira share one world, take turns
  sequentially, and compete for the same food (eaten food vanishes for everyone)
- ✅ **Day 7 — Agent detection:** observation reports adjacent agents by name
  (e.g. `North: Bob`)
- ✅ **Day 8 — Social memory:** sightings become memories like
  `Observed Bob north of me` / `Observed Kira near food`
- ✅ **Personality impact:** a typed-trait layer (`personality.py`) turns each
  agent's description into instincts — curious agents explore and rarely rest,
  cautious agents hug food and conserve, friendly agents close on others,
  independent agents drift away.
- ✅ **Goal weights in context:** each agent's weighted goals are sent to the
  model so it knows what the agent values.
- ✅ **Memory influence:** the most recent memories are included in the decision
  context (compact) and shape future plans.
- ✅ **Strategy caching (cost control):** the LLM is asked for a high-level
  *strategy* only every N turns; in between, the strategy is executed in pure
  Python. Typical run: **~80% fewer LLM calls** than deciding every turn.

**Intentionally out of scope:** economies, villages, governments, factions,
religion, crafting, combat, trading, conversations, trust / reputation, "God
Mode," and professions.

### How behaviour works (the decision loop)

```
every N turns   →  llm.get_strategy(prompt)   # personality + goals + memory + senses
                   → "seek_food" / "explore north" / "approach Bob" / ...
every turn      →  strategy.choose_action(...) # pure Python, no inference
                   → personality + hunger + surroundings → one concrete action
```

## Architecture Principles

1. **Single source of truth.** `world_state` (in `world.py`) is the one authoritative
   description of the entire simulation. Everything else reads from it.
2. **Agents read, the world layer writes.** Agents never mutate global state directly.
   All changes flow through helpers like `add_agent()` so there is one place to add
   validation, indexing, and future hooks.
3. **Plain data over cleverness.** The world is a plain `dict` and agents are simple
   `@dataclass` containers — easy to serialize (JSON), diff, inspect, and extend with
   new fields/keys without schema migrations.
4. **Secrets stay out of source.** The Gemini API key is read from the environment via
   `.env` and is never hard-coded or committed.
5. **Designed for expansion.** New top-level world keys (grid, weather, economy) and new
   agent fields can be added without breaking existing readers.

## Tech Stack

| Layer            | Choice                                             |
| ---------------- | -------------------------------------------------- |
| Language         | Python 3.14                                         |
| AI model         | Google Gemini (`gemini-2.5-flash`)                  |
| AI SDK           | `google-genai` (the new unified Google Gen AI SDK)  |
| Config / secrets | `python-dotenv`                                     |
| Environment      | `venv` (named **Jarvis**)                           |

## Roadmap (V1 Milestones)

- **Day 1 — Foundation (done):** world state, agent model, LLM handshake.
- **Day 2-3 — Grid, perception & decisions (done):** 2D world, observation, action loop.
- **Day 4-5 — Survival & memory (done):** hunger/starvation, bounded per-agent memory.
- **Day 5.5 — Provider abstraction (done):** Ollama/Gemini behind one interface.
- **Day 6 — Multiple agents (done):** Alex, Bob, Kira share a world and compete for food.
- **Day 7 — Agent detection (done):** agents perceive neighbours by name.
- **Day 8 — Social memory (done):** sightings recorded as bounded memories.
- **Personality & goal-driven behaviour (done):** typed traits, goals in context,
  memory influence, and a strategy-caching layer that cuts inference cost ~80%.
- **Relationships & reputation (next):** turn sightings into per-agent opinions
  (familiarity, trust, rivalry over contested food) that bias strategy choice.
- **Conversation (later):** let adjacent agents exchange short messages that
  influence memory and relationships.
- **Later:** memory summarization, events & economy, God Mode, rendering.

## How to Run Locally

### Prerequisites
- Python 3.14+
- A Gemini API key — get one at https://aistudio.google.com/apikey

### 1. Clone and enter the project
```bash
git clone <your-repo-url>
cd "AI Civilisation"
```

### 2. Create and activate the virtual environment
```bash
python3 -m venv Jarvis
source Jarvis/bin/activate        # macOS / Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your API key
Create a `.env` file in the project root (it is gitignored and must never be committed):
```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

### 5. Run the simulation

Default (local Ollama — requires `ollama serve` running with the configured model):
```bash
python main.py
```

No model server handy? Run fully offline with the built-in `random` provider:
```bash
AICIV_PROVIDER=random python main.py
```

Use the cloud Gemini provider instead:
```bash
AICIV_PROVIDER=gemini python main.py
```

You should see Alex, Bob, and Kira take turns on the shared grid: a per-turn map,
each agent's observation (including neighbours by name), its decision, and the
result. A final summary lists survivors, casualties, and every agent's memory.

### Run the tests
```bash
python test_simulation.py
```
Deterministic checks (no LLM needed) for detection, social memory, the memory
bound, food competition, movement collision, and starvation.

### Deactivating the environment
```bash
deactivate
```

## Project Structure

```
.
├── main.py              # Entry point: builds the world, runs the multi-agent loop
├── world.py             # world_state (single source of truth): grid, food, hunger,
│                        #   movement, perception (scan/observe), detection, memory
├── agents.py            # Agent dataclass (personality, goals, hunger, memory, alive)
├── personality.py       # Free-text personality -> typed traits (curiosity, caution, …)
├── strategy.py          # Strategy model + pure-Python action executor + strategy prompt
├── llm.py               # Provider-agnostic layer: get_strategy/get_decision
│                        #   (ollama / gemini / random) + call counters
├── test_simulation.py   # Deterministic tests (mechanics, personality, strategy)
├── requirements.txt     # Python dependencies
├── .env                 # API key (gitignored — not committed)
└── .gitignore
```
