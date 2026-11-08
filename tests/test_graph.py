"""Graph tests using a scripted fake LLM (no network access)."""

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from src.graph import build_graph
from src.prompts import SYSTEM_PROMPT


class ScriptedLLM:
    """Returns pre-set replies in order and records every input."""

    def __init__(self, replies: list[AIMessage]):
        self.replies = list(replies)
        self.calls: list[list] = []

    def runnable(self) -> RunnableLambda:
        def _invoke(messages):
            self.calls.append(list(messages))
            return self.replies.pop(0)

        return RunnableLambda(_invoke)


def _config(thread_id: str = "test") -> dict:
    return {"configurable": {"thread_id": thread_id}}


def test_default_llm_compiles_without_network(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    app = build_graph()
    assert set(app.get_graph().nodes) == {"__start__", "agent", "tools", "__end__"}


def test_graph_draws_ascii(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    drawing = build_graph().get_graph().draw_ascii()
    assert "agent" in drawing and "tools" in drawing


def test_direct_answer_ends_without_tools():
    llm = ScriptedLLM([AIMessage(content="hello!")])
    app = build_graph(llm.runnable())

    result = app.invoke({"messages": [("user", "hi")]}, _config())

    assert result["messages"][-1].content == "hello!"
    assert len(llm.calls) == 1
    first = llm.calls[0][0]
    assert first["role"] == "system" and first["content"] == SYSTEM_PROMPT


def test_tool_call_loop_runs_tool_and_returns_to_agent():
    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "6 * 7"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="The answer is 42."),
        ]
    )
    app = build_graph(llm.runnable())

    result = app.invoke({"messages": [("user", "what is 6*7?")]}, _config())

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "42"
    assert tool_msgs[0].tool_call_id == "call_1"
    assert result["messages"][-1].content == "The answer is 42."
    # Second LLM call must see the tool result.
    assert isinstance(llm.calls[1][-1], ToolMessage)


def test_memory_persists_per_thread():
    llm = ScriptedLLM([AIMessage(content="one"), AIMessage(content="two")])
    app = build_graph(llm.runnable())

    app.invoke({"messages": [("user", "first")]}, _config("a"))
    app.invoke({"messages": [("user", "second")]}, _config("a"))

    # system + user1 + ai1 + user2 on the second call.
    contents = [getattr(m, "content", None) or m.get("content") for m in llm.calls[1]]
    assert contents[1:] == ["first", "one", "second"]
    # System prompt is injected once per call, never stored in state.
    state = app.get_state(_config("a")).values["messages"]
    assert all(m.type != "system" for m in state)


def test_threads_are_isolated():
    llm = ScriptedLLM([AIMessage(content="one"), AIMessage(content="two")])
    app = build_graph(llm.runnable())

    app.invoke({"messages": [("user", "first")]}, _config("a"))
    app.invoke({"messages": [("user", "other")]}, _config("b"))

    assert len(llm.calls[1]) == 2  # system + this thread's single message
