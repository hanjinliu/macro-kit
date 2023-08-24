[![License BSD-3](https://img.shields.io/pypi/l/macro-kit.svg?color=green)](https://github.com/hanjinliu/macro-kit/blob/main/LICENSE)
[![Python package index download statistics](https://img.shields.io/pypi/dm/macro-kit.svg)](https://pypistats.org/packages/macro-kit)
[![PyPI version](https://badge.fury.io/py/macro-kit.svg)](https://badge.fury.io/py/macro-kit)
[![Conda version](https://anaconda.org/conda-forge/macro-kit/badges/version.svg)](https://anaconda.org/conda-forge/macro-kit/badges/version.svg)

# macro-kit

`macro-kit` is a package for efficient macro recording and metaprogramming in Python using abstract syntax tree (AST).

The design of AST in this package is strongly inspired by [Julia metaprogramming](https://docs.julialang.org/en/v1/manual/metaprogramming/). Similar methods are also implemented in builtin `ast` module but `macro-kit` (Julia-style metaprogramming) is more convenient in code operation and also focused on the macro generation and customization.


## Installation

- use pip

```
pip install macro-kit -U
```

- from source

```
pip install git+https://github.com/hanjinliu/macro-kit
```

## Examples

1. Define a macro-recordable function

```python
from macrokit import Macro, Expr, Symbol
macro = Macro()

@macro.record
def str_add(a, b):
    return str(a) + str(b)

val0 = str_add(1, 2)
val1 = str_add(val0, "xyz")
macro
```
```
[Out]
var0x24fdc2d1530 = str_add(1, 2)
var0x24fdc211df0 = str_add(var0x24fdc2d1530, 'xyz')
```

Use `format` method to rename variable names.

```python
# substitute identifiers of variables
# var0x24fdc2d1530 -> x
macro.format([(val0, "x")])
```
```
[Out]
x = str_add(1, 2)
var0x24fdc211df0 = str_add(x, 'xyz')
```

`format` also support substitution with more complicated expressions.

```python
# substitute to _dict["key"]
expr = Expr(head="getitem", args=[Symbol("_dict"), "key"])
macro.format([(val0, expr)])
```
```
[Out]
_dict['key'] = str_add(1, 2)
var0x24fdc211df0 = str_add(_dict['key'], 'xyz')
```

2. Record class

```python
macro = Macro()

@macro.record
class C:
    def __init__(self, val: int):
        self.value = val

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value: int):
        if not isinstance(new_value, int):
            raise TypeError("new_value must be an integer.")
        self._value = new_value

    def show(self):
        print(self._value)

c = C(1)
c.value = 5
c.value = -10
c.show()
```
```
[Out]
-10
```

Note that value assignments are not recorded in duplicate.

```python
macro.format([(c, "ins")])
```
```
[Out]
ins = C(1)
ins.value = -10
var0x7ffed09d2cd8 = ins.show()
```

`eval` can evaluate macro.

```python
macro.eval({"C": C})
```
```
[Out]
-10
```

3. Record module

```python
import numpy as np
macro = Macro()
np = macro.record(np)  # macro-recordable numpy

arr = np.random.random(30)
mean = np.mean(arr)

macro
```
```
[Out]
var0x2a0a2864090 = numpy.random.random(30)
var0x2a0a40daef0 = numpy.mean(var0x2a0a2864090)
```

Recorded module is stored in `Symbol` so you can safely `eval` the macro without passing the module object as the global variables.

```python
macro.eval()  # this works
```

4. String parsing

`parse` calls `ast.parse` inside so that you can safely make `Expr` from string.

```python
from macrokit import parse

expr = parse("result = f(0, l[2:8])")
expr
```
```
[Out]
:(result = f(0, l[slice(2, 8, None)]))
```
```python
print(expr.dump())
```
```
[Out]
head: assign
args:
 0: result
 1: head: call
    args:
     0: f
     1: 0
     2: head: getitem
        args:
         0: l
         1: slice(2, 8, None)
```
