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


def test_mock_binary_op():
    m0 = Mock("m0")
    m1 = Mock("m1")
    assert str(m0.and_(m1)) == "(m0 and m1)"
    assert str(m0.or_(m1)) == "(m0 or m1)"
    assert str(m0.in_(m1)) == "(m0 in m1)"
    assert str(m0.not_in_(m1)) == "(m0 not in m1)"
    assert str(m0.is_(m1)) == "(m0 is m1)"
    assert str(m0.is_not_(m1)) == "(m0 is not m1)"


def test_mock_unary_op():
    m = Mock("m")
    assert str(Mock.not_(m)) == "(not m)"
    assert str(-m) == "(-m)"
    assert str(+m) == "(+m)"
    assert str(~m) == "(~m)"
