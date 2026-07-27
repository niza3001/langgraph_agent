"""CLI entry point for the LangGraph chat agent."""

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.formatting import (  # noqa: E402
    ask_user,
    print_agent_reply,
    print_banner,
    print_error,
    print_session_saved,
    print_tool_call,
    print_tool_result,
    thinking,
)
from src.graph import build_graph  # noqa: E402

QUIT_WORDS = {"exit", "quit", "bye", "goodbye"}
LOG_DIR = Path("logs")

# Set DEBUG_TOOLS=0 in the environment to silence tool-call tracing.
DEBUG_TOOLS = os.getenv("DEBUG_TOOLS", "1") != "0"


def _message_text(content) -> str:
    """Convert model message content to a plain string.

    Providers may return either a string or a list of content blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return str(content)


def _reply_text(message) -> str:
    """Extract a displayable reply, making empty/tool-only messages visible.

    A weak model may return an assistant message with no text (everything in
    a reasoning field, or only unparsed tool calls). Returning "" for those
    makes failures look like blank lines, so surface them explicitly instead.
    """
    text = _message_text(getattr(message, "content", message)).strip()
    if text:
        return text
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        names = ", ".join(tc.get("name", "?") for tc in tool_calls)
        return f"[no text — model returned only tool call(s): {names}]"
    return "[no text — model returned empty content]"


def _trace_tools(chunk: dict) -> None:
    """Print tool calls and tool results as the graph streams through nodes."""
    for messages in (v.get("messages", []) for v in chunk.values() if isinstance(v, dict)):
        for msg in messages:
            for tc in getattr(msg, "tool_calls", None) or []:
                print_tool_call(tc.get("name", "?"), tc.get("args"))
            if type(msg).__name__ == "ToolMessage":
                print_tool_result(getattr(msg, "name", "?"), str(msg.content))


def save_log(transcript: list[tuple[str, str]]) -> Path:
    """Write the conversation transcript under ``logs/`` and return its path."""
    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = LOG_DIR / f"session-{stamp}.txt"
    lines = [f"LangGraph agent session — {stamp} UTC", ""]
    for role, text in transcript:
        lines.append(f"{role}> {text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    """Run an interactive chat loop until the user quits."""
    app = build_graph()
    config = {"configurable": {"thread_id": "cli-session"}}
    transcript: list[tuple[str, str]] = []

    print_banner()

    while True:
        user_input = ask_user()
        if user_input is None or user_input.lower() in QUIT_WORDS:
            break
        if not user_input:
            continue

        try:
            with thinking():
                final_message = None
                for chunk in app.stream(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config,
                    stream_mode="updates",
                ):
                    if DEBUG_TOOLS:
                        _trace_tools(chunk)
                    for node_update in chunk.values():
                        msgs = node_update.get("messages") if isinstance(node_update, dict) else None
                        if msgs:
                            final_message = msgs[-1]
            reply = _reply_text(final_message) if final_message is not None else (
                "[no output — graph produced no messages]"
            )
        except Exception as exc:
            print_error(exc)
            continue

        print_agent_reply(reply)
        transcript.append(("you", user_input))
        transcript.append(("agent", reply))

    path = save_log(transcript)
    print_session_saved(path)


if __name__ == "__main__":
    main()
