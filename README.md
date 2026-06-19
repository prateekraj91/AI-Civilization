# AI Civilization

A sandbox simulation where AI-driven agents inhabit a shared world, perceive their
surroundings, and act according to their personalities and goals — powered by Google's
Gemini models.

## What is AI Civilization?

AI Civilization is an experiment in emergent, agent-based simulation. Each inhabitant is
an autonomous **Agent** with a personality, a weighted set of goals (e.g. survival,
wealth, friendship), and a memory of what it has experienced. Agents read from a single
authoritative **world state** to decide how to behave, and a language model (Gemini)
gives them their voice and reasoning.

The long-term vision is a living world you can observe and intervene in — watching
agents form relationships, pursue goals, and react to events over many simulation turns.

## Current Status — Day 1 ✅

Day 1 establishes the foundation and proves the end-to-end pipeline works:

- ✅ World layer (`world_state`) as the single source of truth
- ✅ `Agent` data model (personality, goals, hunger, position, inventory, memory)
- ✅ First agent, **Alex**, registered through the world layer
- ✅ Verified Gemini integration: the agent introduces itself via a live API call

**Intentionally out of scope for Day 1:** world grid, movement, memory summarization,
multiple interacting agents, rendering, and "God Mode."

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

- **Day 1 — Foundation (done):** world state, agent model, Gemini handshake.
- **World grid & movement:** give the world a 2D space and let agents move through it.
- **Perception & decision loop:** agents observe nearby state and choose actions each turn.
- **Multiple agents & interaction:** several agents coexisting and influencing each other.
- **Memory summarization:** compress agent memory so context stays manageable over time.
- **Events & economy:** a chronological event log driving emergent dynamics.
- **God Mode:** controlled mutation of `world_state` to intervene in the simulation.
- **Rendering:** a read-only view to visualize the world as it evolves.

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
```bash
python main.py
```

You should see Alex registered in the world and a short self-introduction generated live
by Gemini.

### Deactivating the environment
```bash
deactivate
```

## Project Structure

```
.
├── main.py            # Day 1 entry point: builds the world, creates Alex, tests Gemini
├── world.py           # world_state (single source of truth) + add_agent()
├── agents.py          # Agent dataclass (personality, goals, memory, ...)
├── requirements.txt   # Python dependencies
├── .env               # API key (gitignored — not committed)
└── .gitignore
```
