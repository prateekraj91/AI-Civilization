"""
llm
===

THE THINKING LAYER — everything that decides what an agent does, model-backed or not.

What lives here
---------------
`llm` itself is the transport (provider selection, prompting, call stats — `from llm import llm`);
`strategy` builds the prompts and owns the action vocabulary; `heuristic` is the zero-call
offline twin of the same choice; `cognition` decides WHO is worth a model call this turn (the
focal budget); `conversation` is agent-to-agent speech; `mind` is character at the pivots (M5.1),
consulted only when the arithmetic is too close to call.

Deliberately kept EMPTY of side effects: importing the package must not load provider config, so
`from llm import heuristic` in an offline run reaches for no key and no network. Import the
transport explicitly (`from llm import llm`) when you actually mean the model.
"""
