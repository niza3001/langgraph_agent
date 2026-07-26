# LangGraph Agent

Minimal chat agent built with LangGraph: one LLM node, one tool-execution
node, looping until the model finishes calling tools.

Uses a free OpenRouter model by default (`openrouter/free`).

## Setup

```bash
# Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
uv sync
cp .env.example .env
```

1. Create an API key at [openrouter.ai/keys](https://openrouter.ai/keys).
2. Set it in `.env` as `OPENROUTER_API_KEY=...`.

Optionally pin a specific free model:

```bash
OPENROUTER_MODEL=openai/gpt-oss-20b:free
```

Browse free models at [openrouter.ai/models](https://openrouter.ai/models?q=free)
(prefer models with `:free` that support tool calling).

## Run

```bash
uv run python main.py
# or
uv run langgraph-agent
```

Type `bye` or `goodbye` to quit and save the conversation to
`logs/session-*.txt`. `exit` and `quit` also work.

## Project layout

| Path | Role |
|------|------|
| `src/state.py` | Graph state (message history) |
| `src/tools.py` | Tools available to the agent |
| `src/graph.py` | Graph nodes, edges, and compile |
| `main.py` | CLI chat loop |

## Adding a tool

Define a function, decorate it with `@tool` from `langchain_core.tools`,
and add it to `TOOLS` in `src/tools.py`. The docstring is what the model
uses to decide when to call the tool.
