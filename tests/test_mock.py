from macrokit import Mock


def test_basics():
    mock = Mock("m")
    mock1 = mock.attr
    mock2 = mock1["key"]
    mock3 = mock(1, p="r")
    assert str(mock) == "m"
    assert str(mock1) == "m.attr"
    assert str(mock2) == "m.attr['key']"
    assert str(mock3) == "m(1, p='r')"
