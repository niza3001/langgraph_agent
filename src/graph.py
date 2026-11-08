"""LangGraph agent definition.

Topology::

    START -> agent -> (tools -> agent)* -> END

The ``agent`` node calls the LLM. If the model requests tools,
``tools_condition`` routes to the ``tools`` node; otherwise the run ends.
After tools execute, control returns to ``agent`` so the model can use
the results.
"""

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.config import (
    DEFAULT_TEMPERATURE,
    OPENROUTER_APP_TITLE,
    OPENROUTER_BASE_URL,
    OPENROUTER_HTTP_REFERER,
    get_model,
    get_openrouter_api_key,
)
from src.prompts import SYSTEM_PROMPT
from src.state import AgentState
from src.tools import TOOLS


def build_graph():
    """Compile and return the chat agent graph."""
    llm = ChatOpenAI(
        model=get_model(),
        api_key=get_openrouter_api_key(),
        base_url=OPENROUTER_BASE_URL,
        temperature=DEFAULT_TEMPERATURE,
        default_headers={
            "HTTP-Referer": OPENROUTER_HTTP_REFERER,
            "X-Title": OPENROUTER_APP_TITLE,
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
