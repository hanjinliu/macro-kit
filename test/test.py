import pytest
from macrokit import Macro, Expr, Symbol

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
    assert str(macro.format({out1: Symbol("out1"), out2: Symbol("out2")})) == \
        "out1 = add(1.5, 4.2)\n" \
        "out2 = add_one(100)"

def test_module():
    import pandas as pd
    macro = Macro()
    pd = macro.record(pd)
    df0 = pd.DataFrame({"a": [2,3,5], "b":[True, False, False]})
    df1 = pd.DataFrame({"a": [8,4,-1], "b":[True, True, False]})
    assert str(macro.format([(df0, Symbol("df0")), (df1, Symbol("df1"))])) == \
        "df0 = pandas.DataFrame({'a': [2, 3, 5], 'b': [True, False, False]})\n" \
        "df1 = pandas.DataFrame({'a': [8, 4, -1], 'b': [True, True, False]})"

    import skimage
    import numpy as np
    macro = Macro()
    skimage = macro.record(skimage)
    np = macro.record(np)
    img = np.random.normal(size=(128,128))
    out = skimage.filters.gaussian(img, sigma=2)
    thr = skimage.filters.threshold_otsu(out)
    assert str(macro.format([(img, Symbol("img")), (out, Symbol("out")), (thr, Symbol("thr"))])) == \
        "img = numpy.random.normal(size=(128, 128))\n" \
        "out = skimage.filters.gaussian(img, sigma=2)\n" \
        "thr = skimage.filters.threshold_otsu(out)"