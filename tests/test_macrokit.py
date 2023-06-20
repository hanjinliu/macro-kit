import pytest
from macrokit import (
    Expr, Macro, Symbol, parse, register_type, unregister_type, symbol,
    store, store_sequence
)


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
    assert (
        str(macro.format({out1: Symbol("out1"), out2: Symbol("out2")}))
        == "out1 = add(1.5, 4.2)\n"
        "out2 = add_one(100)"
    )


def test_module():
    import pandas as pd

    macro = Macro()
    pd_ = macro.record(pd)
    df0 = pd_.DataFrame({"a": [2, 3, 5], "b": [True, False, False]})
    df1 = pd_.DataFrame({"a": [8, 4, -1], "b": [True, True, False]})
    macro_str = str(macro.format([(df0, Symbol("df0")), (df1, Symbol("df1"))]))
    assert (
        macro_str
        == "df0 = pandas.DataFrame({'a': [2, 3, 5], 'b': [True, False, False]})\n"
        "df1 = pandas.DataFrame({'a': [8, 4, -1], 'b': [True, True, False]})"
    )

    macro.eval()
    macro_str = str(macro.format([(df0, Symbol("df0")), (df1, Symbol("df1"))]))
    assert (
        macro_str
        == "df0 = pandas.DataFrame({'a': [2, 3, 5], 'b': [True, False, False]})\n"
        "df1 = pandas.DataFrame({'a': [8, 4, -1], 'b': [True, True, False]})"
    )

    import numpy as np

    macro = Macro()
    np_ = macro.record(np)
    img = np_.random.normal(size=(128, 128))
    macro_str = str(macro.format([(img, Symbol("img"))]))
    assert macro_str == "img = numpy.random.normal(size=(128, 128))"

    macro.eval()

    macro_str = str(macro.format([(img, Symbol("img"))]))
    assert macro_str == "img = numpy.random.normal(size=(128, 128))"

    macro = Macro()
    df_ = Expr("getattr", [pd, pd.DataFrame])
    ds_ = Expr("getattr", [pd, pd.Series])
    expr1 = Expr.parse_call(df_, ({"a": [1, 2, 4], "b": [True, False, False]},), {})
    expr2 = Expr.parse_call(ds_, ((1, 2, 3),), {})
    macro.append(expr1)
    macro.append(expr2)
    macro_str = str(macro)
    assert (
        macro_str == "pandas.DataFrame({'a': [1, 2, 4], 'b': [True, False, False]})\n"
        "pandas.Series((1, 2, 3))"
    )

    macro.eval()  # pandas should be registered in Symbol
    df_ = Expr("getattr", [pd, pd.DataFrame])
    ds_ = Expr("getattr", [pd, pd.Series])
    assert (
        macro_str == "pandas.DataFrame({'a': [1, 2, 4], 'b': [True, False, False]})\n"
        "pandas.Series((1, 2, 3))"
    )


def test_format():
    macro = Macro()

    @macro.record
    def str_add(a, b):
        return str(a) + str(b)

    val0 = str_add(1, 2)
    val1 = str_add(val0, "xyz")
    macro_str = str(macro.format([(val0, Symbol("X")), (val1, Symbol("Y"))]))
    assert macro_str == "X = str_add(1, 2)\n" "Y = str_add(X, 'xyz')"


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
    a.value_str = 4  # This line should not be recorded.
    a.value_str = 5
    assert a.getval() == 5
    A.set_class_var(10)
    assert a["key"] == "key"
    a["a"] = False  # This line should not be recorded.
    a["a"] = True

    macro = macro.format([(a, Symbol("a"))])
    assert str(macro[0]) == "a = A(4)"
    assert str(macro[1]) == "a.value_str"
    assert str(macro[2]) == "a.value_str = 5"
    assert str(macro[3]) == "a.getval()"
    assert str(macro[4]) == "A.set_class_var(10)"
    assert str(macro[5]) == "a['key']"
    assert str(macro[6]) == "a['a'] = True"


def test_register_type():
    import numpy as np

    macro = Macro()

    register_type(np.ndarray, lambda arr: str(arr.tolist()))

    @macro.record
    def double(arr: np.ndarray):
        return arr * 2

    out = double(np.arange(3))
    macro_str = str(macro.format([(out, Symbol("out"))]))
    assert macro_str == "out = double([0, 1, 2])"

    @register_type(np.ndarray)
    def _(arr: np.ndarray):
        return f"array{arr.shape}"

    assert str(symbol(np.zeros((4, 6)))) == "array(4, 6)"

    @register_type(lambda e: e.name)
    class T:
        def __init__(self):
            self.name = "t"

    assert str(symbol(T())) == "t"

    # cleanup
    unregister_type(np.ndarray)


@pytest.mark.parametrize(
    "line",
    [
        "a.b",
        "a.b",
        "a['b']",
        "del a",
        "del a, b",
        "f(x, y=1)",
        "x = 0",
        "x: int = 0",
        "assert x == 0",
        "assert x == 0, 'x must be 0.'",
        "+a",
        "-a",
        "~a",
        "not a",
        "return",
        "return 0",
        "return 0, 1",
        "yield",
        "yield a",
        "yield a, b",
        "yield from gen()",
        "lambda x, y: x + y",
        "raise Exception()",
        "raise Exception() from e",
        "[x for x in range(10)]",
        "[x for x in range(10) if x % 2 == 0]",
        "import numpy",
        "import numpy as np",
        "import numpy.linalg as la",
        "from numpy import linalg as la",
        "from numpy.linalg import norm",
        "from numpy.linalg import norm as nrm",
        "if a: pass",
        "if a := f(): pass",
    ]
)
def test_parsing_single_line(line: str):
    expr = parse(line)
    str(expr)

code1 = """
a = np.arange(12)
for i in a:
    print(a)
    if i % 3 == 0:
        print("3n")
t = 0
while t < 3:
    t = t + 1
del t
"""

code2 = """
def g(a: int = 4):
    if not isinstance(a, int):
        raise TypeError("a must be an integer, got {type(a)}.")
    b: str = "a"
    return a

def gen(x):
    yield 0
    yield x

def gen2():
    yield from gen(1)
"""

def test_parsing():
    str(parse(code1))

def test_functions():
    str(parse(code2))

@pytest.mark.parametrize(
    "code",
    [
        "@dec\ndef f():\n    return 0",
        "@dec(x, y=0)\ndef f(t):\n    return t",
        "@dec\n@dec2\n@dec3\n@dec4\ndef f():\n    return 0"
        "@dec\nclass A:\n    pass",
        "@dec(x, y=0)\nclass A:\n    pass",
        "@dec\n@dec2\n@dec3\n@dec4\nclass A:\n    pass"
    ]
)
def test_decorator(code: str):
    str(parse(code))

@pytest.mark.parametrize(
    "op",
    # fmt: off
    ["+", "-", "*", "/", "%", "==", "!=", ">", ">=", "<", "<=", "is", "is not",
     "in", "not in", "**", "@", "//", "&", "|", "^", "and", "or"]
    # fmt: on
)
def test_binary_op(op: str):
    str(parse(f"a {op} b"))

@pytest.mark.parametrize(
    "op",
    ["+", "-", "*", "/", "%", "**", "//", "@", "&", "|", "^"],
)
def test_increment_op(op: str):
    str(parse(f"a {op}= b"))

@pytest.mark.parametrize(
    "s",
    [
        "a{b}c",
        "a{b!r}c",
        "a{b!s}c",
        "a{b!a}c",
        "a{x:<2}",
        "a{x:<{y}}",
        "a{x:{t}^{u}}",
        "a{b}{bb}",
        "a{b.attr}{c.attr}",
        "a{b[key]}{c[key]}",
    ],
)
def test_fstring(s: str):
    s0 = "f'" + s + "'"
    parsed = str(parse(s0))
    assert parsed == s0

@pytest.mark.parametrize(
    "s",
    [
        "try:\n    f()\nexcept:\n    g()",
        "try:\n    f()\nfinally:\n    g()",
        "try:\n    f()\nexcept Exception:\n    g()",
        "try:\n    f()\nexcept Exception as e:\n    g(e)",
        "try:\n    f()\nexcept Exception:\n    g()\nfinally:\n    h()",
        "try:\n    f()\nexcept Exception:\n    g()\nelse:\n    h()",
        "try:\n    f()\nexcept Exception:\n    g()\nelse:\n    h()\nfinally:\n    i()",
    ]
)
def test_try_except(s: str):
    str(parse(s))


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
    assert macro_str == "a = A()\nlen(a)\nbool(a)\nint(a)\nfloat(a)\nstr(a)"

@pytest.mark.parametrize(
    "s",
    [
        "(a + 1 for a in range(4))",
        "[a + 1 for a in range(4)]",
        "{a + 1 for a in range(4)}",
        "{a: b for a, b in zip(range(4), range(4))}",
        "{a: b for a, b in zip(range(4), range(4)) if a % 2 == 0}",
        "[a + b for a in range(3) for b in range(3)]",
    ]
)
def test_generators(s: str):
    parse(s)

def test_field():
    class A:
        m = Macro(flags={"Return": False})

        def __init__(self):
            self.m  # activate field

        @m.record
        def f(self, a, b):
            pass

        @m.property
        def value(self):
            pass

        @value.setter
        def value(self, v):
            pass

    a = A()
    a.f(0, 1)
    a.value
    a.value = 5
    a.value = 6

    macro_str = str(a.m.format([(a, "a")]))
    assert macro_str == "a.f(0, 1)\n" "a.value\n" "a.value = 6"


def test_at():
    expr = parse("mod.func(ins.attr.name)")
    assert expr.at(1, 0, 0) == expr.args[1].args[0].args[0]
    assert expr.at(1, 1) == expr.args[1].args[1]


def test_split_getattr():
    expr = parse("a.b.c.d")
    seqs = [x.name for x in expr.split_getattr()]
    assert seqs == list("abcd")


@pytest.mark.parametrize(
    "string",
    ["a.b.c()", "a.b.c = 2", "(a*b).c", "x = a.b.c", "a['b'].c", "a['b']['c']"]
)
def test_split_getattr_errors(string: str):
    expr = parse(string)
    with pytest.raises(ValueError):
        expr.split_getattr()


def test_split_getitem():
    expr = parse("a['b']['c']['d']")
    seqs = [x.name for x in expr.split_getitem()]
    assert seqs == ["a", "'b'", "'c'", "'d'"]


def test_module_update():
    import time as tm

    time_ = symbol(tm)
    macro = Macro()
    macro.append(Expr("call", [Expr("getattr", [time_, "time"])]))
    macro.eval()
    macro_str = str(macro)
    assert macro_str == "time.time()"
    re_compiled = parse(macro_str)
    re_compiled.eval()
    assert re_compiled == macro[0]


def test_eq():
    assert parse("a = 1") == parse("a = 1")
    assert parse("a = 1") != parse("b.a = 1")
    assert parse("func(3)") == parse("func(3)")
    # assert parse("a = 'a'") == parse("a = 'a'")  # this is False, but should it be?
    # assert parse("t['xy'] = func(0, 2)") == parse("t['xy'] = func(0, 2)")


def test_slicing():
    macro = Macro()
    macro.append("a = 1")
    macro.append("b = 1")
    macro.append("c = 1")
    assert isinstance(macro[0], Expr)
    assert isinstance(macro[1:], Macro)
    assert macro.flags == macro[1:].flags


def test_eval_call_args():
    expr = parse("f(1, 2, x=3)")
    args, kwargs = expr.eval_call_args()
    assert args == (1, 2)
    assert kwargs == {"x": 3}


def test_eval_call_args_with_namespace():
    expr = parse("f(a, b, x=c)")
    args, kwargs = expr.eval_call_args(ns=dict(a=1, b=2, c=3))
    assert args == (1, 2)
    assert kwargs == {"x": 3}


def test_store():
    # A array like object
    class X:
        def __init__(self, val):
            self.value = list(val)

        @classmethod
        def arange(cls, n: int):
            return cls(range(n))

        def __add__(self, val: int):
            return X(x + val for x in self.value)

        def __eq__(self, other: "X"):
            return self.value == other.value

    x = X.arange(5)

    def fn(x):
        return x + 1

    store(x)
    store(fn)

    _x = symbol(x)
    _fn = symbol(fn)

    assert _x.eval() == x
    assert _fn.eval() is fn

    _fn_x_expr = Expr.parse_call(_fn, (x,))
    out = _fn_x_expr.eval()
    assert out == x + 1


def test_store_sequence():
    class X:
        pass

    tup = (X(), X(), X())
    store_sequence(tup)
    sym = symbol(tup)
    assert str(symbol(tup[0])) == f"{sym}[0]"
    assert str(symbol(tup[1])) == f"{sym}[1]"
    assert str(symbol(tup[2])) == f"{sym}[2]"
