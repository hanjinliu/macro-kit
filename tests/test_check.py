from __future__ import annotations
from typing import Any
from macrokit import parse
from macrokit.utils import check_attributes, check_call_args
import pytest

class A:
    x = 1

class B:
    a = A()

@pytest.mark.parametrize(
    "s, ns",
    [
        ("a.sort()", {"a": [1, 2]}),
        ("b.a.x", {"b": B()}),
        ("a.sort()\nb.count()", {"a": [1, 2], "b": (1, 2)}),
        ("for i in range(3):\n    a.append(i)", {"a": []}),
        ("l = []\nl.append(3)", {})
    ]
)
def test_check_attributes_passes(s: str, ns: dict[str, Any]):
    expr = parse(s)
    results = check_attributes(expr, ns)
    assert len(results) == 0

@pytest.mark.parametrize(
    "s, ns, nfail",
    [
        ("a.sort()", {"a": "str"}, 1),
        ("b.y", {"b": B()}, 1),
        ("b.a.y", {"b": B()}, 1),
        ("for i in range(3):\n    a.append(i)", {"a": 1}, 1),
        ("for i in b.a.y:\n   pass", {"b": B()}, 1),
    ]
)
def test_check_attributes_fails(s: str, ns: dict[str, Any], nfail: int):
    expr = parse(s)
    results = check_attributes(expr, ns)
    assert len(results) == nfail

class C:
    def f(self, y: int) -> int:
        pass

    def g(self, x: int, *, y: int = 0) -> int:
        pass

@pytest.mark.parametrize(
    "s, ns",
    [
        ("f(0)", {"f": lambda x: x}),
        ("f(x=0)", {"f": lambda x: x}),
        ("f(0)", {"f": lambda x, y=0: x + y}),
        ("f(0, 1)", {"f": lambda x, y=0: x + y}),
        ("f(0, 1)\n", {"f": lambda x, y=0: x + y}),
        ("c.f(0)", {"c": C()}),
        ("c.f(0)\nc.g(0, y=2)", {"c": C()}),
        ("while c.a > 0:\n    c.f(0)\n    c.g(0, y=2)", {"c": C()}),
    ]
)
def test_check_call_args_passes(s: str, ns: dict[str, Any]):
    expr = parse(s)
    results = check_call_args(expr, ns)
    assert len(results) == 0

def test_check_call_args_with_unknown():
    ns = {"c": C()}
    s = (
        "arr = np.arange(4)\n"
        "c.f(arr.size)"
    )
    expr = parse(s)
    results = check_call_args(expr, ns)
    assert len(results) == 0

@pytest.mark.parametrize(
    "s, ns, nfail",
    [
        ("f()", {"f": lambda x: x}, 1),
        ("f(0, 1)", {"f": lambda x: x}, 1),
        ("c.f()", {"c": C()}, 1),
        ("c.f(0, 1)", {"c": C()}, 1),
        ("c.f(0, x=1)\nc.g(0, 0)", {"c": C()}, 2),
    ]
)
def test_check_call_args_fails(s: str, ns: dict[str, Any], nfail: int):
    expr = parse(s)
    results = check_call_args(expr, ns)
    assert len(results) == nfail
