"""Tools registered with the agent.

To add a tool, define a function, decorate it with ``@tool``, and append
it to ``TOOLS``. The function docstring is sent to the model as the tool
description, so keep it accurate and concise.
"""

import ast
import math
import operator
from datetime import datetime, timezone

from langchain_core.tools import tool

# Calculator safety limits. Without them, inputs like ``9**9**9`` hang the
# process computing a multi-million-digit integer.
MAX_EXPRESSION_LENGTH = 200
MAX_RESULT_DIGITS = 1000

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _check_pow(base: float, exponent: float) -> None:
    """Reject powers whose result would exceed ``MAX_RESULT_DIGITS``.

    Estimates the digit count as ``exponent * log10(|base|)`` so the
    expensive computation is never attempted.
    """
    if exponent <= 0 or abs(base) <= 1:
        return
    if exponent * math.log10(abs(base)) > MAX_RESULT_DIGITS:
        raise ValueError(f"result too large (over {MAX_RESULT_DIGITS} digits)")


def _check_size(value: float) -> float:
    """Reject integers with more than ``MAX_RESULT_DIGITS`` digits."""
    # bit_length is cheap; 3.33 bits per decimal digit.
    if isinstance(value, int) and value.bit_length() > MAX_RESULT_DIGITS * 3.33:
        raise ValueError(f"result too large (over {MAX_RESULT_DIGITS} digits)")
    return value


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate an arithmetic AST node."""
    if isinstance(node, ast.Constant):
        # bool is a subclass of int; strings and None are rejected too.
        if isinstance(node.value, (int, float)) and not isinstance(
            node.value, bool
        ):
            return node.value
        raise ValueError(f"unsupported constant {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            _check_pow(left, right)
        return _check_size(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


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
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return (
            f"Error: expression longer than {MAX_EXPRESSION_LENGTH} characters."
        )
    try:
        return str(_eval_node(ast.parse(expression, mode="eval").body))
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
