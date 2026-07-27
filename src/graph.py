"""LangGraph agent definition.

Topology::

    START -> agent -> (tools -> agent)* -> END

The ``agent`` node calls the LLM. If the model requests tools,
``tools_condition`` routes to the ``tools`` node; otherwise the run ends.
After tools execute, control returns to ``agent`` so the model can use
the results.
"""

import os

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.state import AgentState
from src.tools import TOOLS

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. Rules:\n"
    "1. ALWAYS call get_current_time before answering any question about the "
    "date, the day, or a relative period like 'this week', 'today', or "
    "'the coming week'. Never state the date from memory.\n"
    "2. ALWAYS call web_search for events, news, prices, or any fact that may "
    "have changed recently. Never invent event listings, dates, or URLs — if "
    "you did not get it from web_search, do not report it.\n"
    "3. Use calculator for arithmetic instead of computing in your head.\n"
    "After using a tool, base your answer only on the tool's output."
)

# The openrouter/free router always resolves to a currently-available free
# model AND filters for tool calling when tools are bound. Individual :free
# slugs rotate constantly (a pinned one can start returning 404 "paid now"),
# so the router is the resilient default. The hardened SYSTEM_PROMPT above
# forces tool use, mitigating weaker models the router may pick.
# Override with OPENROUTER_MODEL to pin a specific free slug.
DEFAULT_MODEL = "openrouter/free"


def build_graph():
    """Compile and return the chat agent graph."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your-key-here":
        raise ValueError(
            "Set OPENROUTER_API_KEY in .env "
            "(create a key at https://openrouter.ai/keys)."
        )

    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        default_headers={
            "HTTP-Referer": "https://github.com/langgraph-agent",
            "X-Title": "langgraph-agent",
        },
    ).bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        """Invoke the LLM and append its reply to ``messages``."""
        messages = state["messages"]
        if not messages or messages[0].type != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    # MemorySaver keeps per-thread state for the lifetime of this process.
    return graph.compile(checkpointer=MemorySaver())
