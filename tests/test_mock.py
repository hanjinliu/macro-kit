from macrokit import Mock, Expr


def test_basics():
    mock = Mock("m")
    mock1 = mock.attr
    mock2 = mock1["key"]
    mock3 = mock(1, p="r")
    assert str(mock) == "m"
    assert str(mock1) == "m.attr"
    assert str(mock2) == "m.attr['key']"
    assert str(mock3) == "m(1, p='r')"
    assert str(mock + 1) == "(m + 1)"
    assert str(mock - 1) == "(m - 1)"
    assert str(mock % 1) == "(m % 1)"
    assert str(-mock) == "(-m)"
    mock += 4
    assert str(mock) == "(m + 4)"


def test_mock_to_expr():
    mock = Mock("x")
    expr = Expr("assign", [mock, 10])
    assert str(expr) == "x = 10"

    mock2 = Mock("y")
    assert str(mock + mock2) == "(x + y)"
