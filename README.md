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

## Test

```bash
uv run pytest
```

Tests use a scripted fake LLM and a patched search client, so they make
no network calls and need no API key. `build_graph(llm=...)` accepts any
runnable for this purpose.

To print the graph topology (uses the `grandalf` dev dependency):

```bash
uv run python -c "from dotenv import load_dotenv; load_dotenv(); \
from src.graph import build_graph; print(build_graph().get_graph().draw_ascii())"
```

## Project layout

| Path | Role |
|------|------|
| `src/config.py` | Model defaults and CLI settings |
| `src/state.py` | Graph state (message history) |
| `src/tools.py` | Tools available to the agent |
| `src/prompts/` | System and other prompt templates |
| `src/graph.py` | Graph nodes, edges, and compile |
| `src/formatting.py` | Terminal colors and CLI output |
| `main.py` | CLI chat loop |

## Adding a tool

Define a function, decorate it with `@tool` from `langchain_core.tools`,
and add it to `TOOLS` in `src/tools.py`. The docstring is what the model
uses to decide when to call the tool.

Built-in tools: `get_current_time`, `calculator`, `web_search` (DuckDuckGo via `ddgs`, no API key).
