"""Unit tests for the agent's tools (no network access)."""

from datetime import datetime, timezone

import pytest

from src.tools import (
    MAX_EXPRESSION_LENGTH,
    TOOLS,
    calculator,
    get_current_time,
    web_search,
)


def calc(expression: str) -> str:
    return calculator.invoke({"expression": expression})


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", "4"),
        ("2 + 3 * (4 - 1) ** 2", "29"),
        ("10 / 4", "2.5"),
        ("-5 + +3", "-2"),
        ("2 ** -1", "0.5"),
        ("1.5 * 2", "3.0"),
        ("2 ** 100", str(2**100)),
    ],
)
def test_calculator_valid(expression, expected):
    assert calc(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "abs(-1)",
        "x + 1",
        "'a' * 10",  # string constants could allocate huge memory
        "True + 1",
        "[1, 2]",
        "2 % 3",  # unsupported operator
    ],
)
def test_calculator_rejects_non_arithmetic(expression):
    assert calc(expression).startswith("Error")


def test_calculator_division_by_zero():
    assert "division by zero" in calc("1 / 0")


def test_calculator_syntax_error():
    assert calc("2 +").startswith("Error")


@pytest.mark.parametrize(
    "expression",
    [
        "9 ** 9 ** 9",  # would hang computing ~370M digits
        "10 ** 1001",
        "(10 ** 600) * (10 ** 600)",  # large via multiplication
        "2.0 ** 10000",  # float overflow
    ],
)
def test_calculator_rejects_huge_results(expression):
    assert calc(expression).startswith("Error")


def test_calculator_allows_results_at_limit():
    assert calc("10 ** 999") == str(10**999)


def test_calculator_rejects_long_expressions():
    expression = "1+" * MAX_EXPRESSION_LENGTH + "1"
    assert "longer than" in calc(expression)


# ---------------------------------------------------------------------------
# get_current_time
# ---------------------------------------------------------------------------


def test_get_current_time_is_recent_utc_iso():
    result = datetime.fromisoformat(get_current_time.invoke({}))
    assert result.tzinfo is not None
    assert abs((datetime.now(timezone.utc) - result).total_seconds()) < 5


# ---------------------------------------------------------------------------
# web_search (DDGS is patched so no network calls are made)
# ---------------------------------------------------------------------------


class _FakeDDGS:
    results: list[dict] = []
    error: Exception | None = None
    last_max_results: int | None = None

    def text(self, query, max_results):
        type(self).last_max_results = max_results
        if self.error:
            raise self.error
        return self.results


@pytest.fixture
def fake_ddgs(monkeypatch):
    import ddgs

    _FakeDDGS.results = []
    _FakeDDGS.error = None
    _FakeDDGS.last_max_results = None
    monkeypatch.setattr(ddgs, "DDGS", _FakeDDGS)
    return _FakeDDGS


def test_web_search_formats_results(fake_ddgs):
    fake_ddgs.results = [
        {"title": "First", "href": "https://a.example", "body": "alpha"},
        {"title": None, "link": "https://b.example", "snippet": "beta"},
    ]
    out = web_search.invoke({"query": "test"})
    assert "1. First\n   https://a.example\n   alpha" in out
    assert "2. (no title)\n   https://b.example\n   beta" in out


def test_web_search_no_results(fake_ddgs):
    assert web_search.invoke({"query": "nothing"}) == "No results found for 'nothing'."


def test_web_search_error_is_returned_not_raised(fake_ddgs):
    fake_ddgs.error = RuntimeError("rate limited")
    out = web_search.invoke({"query": "q"})
    assert out.startswith("Error searching for 'q'")
    assert "rate limited" in out


@pytest.mark.parametrize(("requested", "used"), [(0, 1), (3, 3), (50, 10)])
def test_web_search_clamps_max_results(fake_ddgs, requested, used):
    web_search.invoke({"query": "q", "max_results": requested})
    assert fake_ddgs.last_max_results == used


def test_tools_registry():
    assert {t.name for t in TOOLS} == {"get_current_time", "calculator", "web_search"}
