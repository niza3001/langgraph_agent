"""CLI entry point for the LangGraph chat agent."""

from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.graph import build_graph  # noqa: E402

QUIT_WORDS = {"exit", "quit", "bye", "goodbye"}
LOG_DIR = Path("logs")


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

    print("LangGraph agent ready. Type 'bye' or 'goodbye' to quit and save the log.\n")
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in QUIT_WORDS:
            break
        if not user_input:
            continue

        result = app.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
        )
        reply = _message_text(result["messages"][-1].content)
        print(f"agent> {reply}\n")
        transcript.append(("you", user_input))
        transcript.append(("agent", reply))

    path = save_log(transcript)
    print(f"Session saved to {path}")


if __name__ == "__main__":
    main()
