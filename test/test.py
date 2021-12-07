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
    img = np.random.normal(size=(128, 128))
    out = skimage.filters.gaussian(img, sigma=2)
    thr = skimage.filters.threshold_otsu(out)
    assert str(macro.format([(img, Symbol("img")), (out, Symbol("out")), (thr, Symbol("thr"))])) == \
        "img = numpy.random.normal(size=(128, 128))\n" \
        "out = skimage.filters.gaussian(img, sigma=2)\n" \
        "thr = skimage.filters.threshold_otsu(out)"

def test_format():
    macro = Macro()

    @macro.record
    def str_add(a, b):
        return str(a) + str(b)

    val0 = str_add(1, 2)
    val1 = str_add(val0, "xyz")
    macro_str = str(macro.format([(val0, Symbol("X")), (val1, Symbol("Y"))]))
    assert macro_str == \
        "X = str_add(1, 2)\n" \
        "Y = str_add(X, 'xyz')"

def test_class():
    macro = Macro(flags={"Return": False})
    @macro.record
    class A:
        clsvar = 0
        def __init__(self, value: int):
            self.value = value
        
        @property
        def value_str(self):
            return str(self.value)
        
        @value_str.setter
        def value_str(self, value: str):
            self.value = int(value)
        
        def getval(self):
            return self.value
        
        @classmethod
        def set_class_var(cls, v):
            cls.clsvar = v
        
        def __getitem__(self, k):
            return k
        
        def __setitem__(self, k, v):
            return
    
    a = A(4)
    assert a.value_str == "4"
    a.value_str = 4 # This line should not be recorded.
    a.value_str = 5
    assert a.getval() == 5
    A.set_class_var(10)
    assert a["key"] == "key"
    a["a"] = False # This line should not be recorded.
    a["a"] = True
    
    macro_str = str(macro.format([(a, Symbol("a"))]))
    assert macro_str == \
        "a = A(4)\n" \
        "a.value_str\n" \
        "a.value_str = 5\n" \
        "a.getval()\n" \
        "A.set_class_var(10)\n"\
        "a['key']\n" \
        "a['a'] = True"

def test_register_type():
    from macrokit import register_type, symbol
    import numpy as np
    
    macro = Macro()
    
    register_type(np.ndarray, lambda arr: str(arr.tolist()))
    
    @macro.record
    def double(arr: np.ndarray):
        return arr*2
    
    out = double(np.arange(3))
    macro_str = str(macro.format([(out, Symbol("out"))]))
    assert macro_str == "out = double([0, 1, 2])"
    
    @register_type(np.ndarray)
    def _(arr: np.ndarray):
        return f"array{arr.shape}"
    assert str(symbol(np.zeros((4,6)))) == "array(4, 6)"
    
    @register_type(lambda e: e.name)
    class T:
        def __init__(self):
            self.name = "t"
    assert str(symbol(T())) == "t"

def test_symbol_var():
    sym_x = Symbol("x")
    sym_y = Symbol("y")
    var_y0 = Symbol.var("y", str)
    var_y1 = Symbol.var("y", str)
    var_y2 = Symbol.var("y", str)
    var_y3 = Symbol.var("y", str)
    var_y_int = Symbol.var("y", int)
    
    assert sym_x == sym_x
    assert sym_x != sym_y
    assert sym_y != var_y0
    assert var_y0 == var_y1 and var_y1 == var_y2 and var_y2 == var_y3
    assert var_y_int != var_y0

code1 = """
a = np.arange(12)
for i in a:
    print(a)
    if i % 3 == 0:
        print("3n")
"""

code2 = """
def g(a: int = 4):
    return a - 1, a + 1
"""

    
def test_parsing():
    from macrokit import parse
    parse(code1)
    parse(code2)

def test_special_methods():
    macro = Macro(flags={"Return": False})
    @macro.record
    class A:
        def __len__(self):
            return 0
        def __bool__(self):
            return True
        def __int__(self):
            return 0
        def __float__(self):
            return 0.0
        def __str__(self):
            return ""
    
    a = A()
    len(a)
    bool(a)
    int(a)
    float(a)
    str(a)
    
    macro_str = str(macro.format([(a, Symbol("a"))]))
    assert macro_str == \
        "a = A()\n" \
        "len(a)\n" \
        "bool(a)\n" \
        "int(a)\n" \
        "float(a)\n" \
        "str(a)"