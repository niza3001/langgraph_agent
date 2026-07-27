"""Terminal formatting helpers for the CLI chat loop."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.text import Text
from rich.theme import Theme

console = Console(
    theme=Theme(
        {
            "you": "bold cyan",
            "agent": "bold green",
            "meta": "dim",
            "ok": "bold green",
            "err": "bold red",
        }
    )
)


def print_banner() -> None:
    """Print the startup panel with quit instructions."""
    console.print(
        Panel.fit(
            "[bold]LangGraph agent[/bold]\n"
            "[meta]Type a message to chat. "
            "Quit with [bold]bye[/bold] / [bold]goodbye[/bold] "
            "(also [bold]exit[/bold] / [bold]quit[/bold]).[/meta]",
            border_style="cyan",
        )
    )
    console.print()


def ask_user() -> str | None:
    """Prompt for user input.

    Only the ``you:`` label is colored; typed text uses the default terminal color.
    Returns ``None`` when the user sends EOF or interrupts (Ctrl+C).
    """
    try:
        console.print(Text("you: ", style="you"), end="")
        return console.input(markup=False).strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None


def thinking() -> AbstractContextManager[Status]:
    """Show a spinner while the agent is working."""
    return console.status("[meta]Thinking…[/meta]", spinner="dots")


def print_agent_reply(reply: str) -> None:
    """Print the agent reply; only the ``agent:`` label is colored."""
    label = Text("agent: ", style="agent")
    text = reply.strip()
    if not text:
        console.print(label + Text("(empty response)", style="meta"))
    elif "\n" in text or "```" in text:
        console.print(label)
        console.print(Markdown(text), style="white")
    else:
        console.print(label + Text(text, style="white"))
    console.print()


def print_tool_call(name: str, args) -> None:
    """Print a tool invocation in dim meta style."""
    console.print(f"[meta]  · tool call: {name}({args})[/meta]")


def print_tool_result(name: str, content: str) -> None:
    """Print a truncated tool result in dim meta style."""
    preview = content.replace("\n", " ")[:200]
    console.print(f"[meta]  · tool result [{name}]: {preview}[/meta]")


def print_error(exc: BaseException) -> None:
    """Print an error message."""
    console.print(f"[err]Error:[/err] {exc}\n")


def print_session_saved(path: Path) -> None:
    """Print the path of the saved session log."""
    console.print(f"[ok]Session saved[/ok] [meta]→ {path}[/meta]")
