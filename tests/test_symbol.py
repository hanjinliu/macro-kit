import pytest
from macrokit import symbol, Symbol


@pytest.mark.parametrize(
    "val, expected",
    [
        (1, "1"),
        (2.3, "2.3"),
        ("", "''"),
        (None, "None"),
        ((), "()"),
        ((1,), "(1,)"),
        ([(), (1,), set()], "[(), (1,), set()]"),
        (True, "True"),
        (1 - 3j, "(1-3j)"),
        (bytes("a", encoding="utf-8"), "b'a'"),
        ({"a": [1, 2], "b": [0.1, 0.2]}, "{'a': [1, 2], 'b': [0.1, 0.2]}"),
        (set(), "set()"),
        ({1, 2, 3}, "{1, 2, 3}"),
        (frozenset(), "frozenset()"),
        (frozenset([1, 2]), "frozenset({1, 2})"),
    ]
)
def test_symbol_string_well_defined(val, expected: str):
    assert str(symbol(val)) == expected


@pytest.mark.parametrize(
    "val",
    [1, 1.3, 2 - 3j, "ab", "a.bc", None, (1, "t"), [1, 3, 4],
     {"a": [1, 2], "b": [0.1, 0.2]}, {1, 3, 6}]
)
def test_symbol_eval_well_defined(val):
    assert symbol(val).eval() == val


def test_symbol_eval_name():
    val = object()
    sym = symbol(val)
    with pytest.raises(NameError):
        sym.eval()
    assert sym.eval({sym: val}) is val


def test_symbol_var():
    sym_x = Symbol("x")
    sym_y = Symbol("y")
    var_y0 = Symbol.var("y")
    var_y1 = Symbol.var("y")
    var_y2 = Symbol.var("y")
    var_y3 = Symbol.var("y")

    assert sym_x == sym_x
    assert sym_x != sym_y
    assert sym_y != var_y0
    assert var_y0 == var_y1 and var_y1 == var_y2 and var_y2 == var_y3
