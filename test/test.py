import pytest
from ..macrokit import Macro

def test_function():
    macro = Macro()

    @macro.record
    def add(a, b):
        return a + b
    
    @macro.record
    def add_one(a):
        return add(a, 1)
    
    out1 = add(1.5, 4.2)
    out2 = add_one(100)
    assert out1 == 5.7
    assert out2 == 101
    assert str(macro) == \
        "add(1.5, 4.2)\n" \
        "add_one(100)"

def test_module():
    import pandas as pd
    macro = Macro()
    pd = macro.record(pd)
    df0 = pd.DataFrame({"a": [2,3,5], "b":[True, False, False]})
    df1 = pd.DataFrame({"a": [8,4,-1], "b":[True, True, False]})
    assert str(macro) == \
    "pandas.DataFrame({'a': [2, 3, 5], 'b': [True, False, False]})\n" \
    "pandas.DataFrame({'a': [8, 4, -1], 'b': [True, True, False]})"