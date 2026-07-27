"""Tools registered with the agent.

To add a tool, define a function, decorate it with ``@tool``, and append
it to ``TOOLS``. The function docstring is sent to the model as the tool
description, so keep it accurate and concise.
"""

from datetime import datetime, timezone

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Return the current UTC date and time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (for example ``2 + 2 * 3``).

    Supports ``+``, ``-``, ``*``, ``/``, ``**``, and parentheses.
    Does not evaluate arbitrary Python.
    """
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {expression}")

    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information about a topic.

    Use this for news, facts you are unsure about, or anything that may
    have changed recently. Returns titles, URLs, and short snippets.
    """
    from ddgs import DDGS

    limit = max(1, min(max_results, 10))
    try:
        results = DDGS().text(query, max_results=limit)
    except Exception as exc:
        return f"Error searching for '{query}': {exc}"

    if not results:
        return f"No results found for '{query}'."

    lines = []
    for i, item in enumerate(results, start=1):
        title = item.get("title") or "(no title)"
        href = item.get("href") or item.get("link") or ""
        body = item.get("body") or item.get("snippet") or ""
        lines.append(f"{i}. {title}\n   {href}\n   {body}")
    return "\n\n".join(lines)


TOOLS = [get_current_time, calculator, web_search]
