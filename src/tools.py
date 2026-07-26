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


TOOLS = [get_current_time, calculator]
