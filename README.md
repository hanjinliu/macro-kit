# macro-kit

`macro-kit` is a package for efficient macro recording and metaprogramming in Python.

This package is strongly inspired by [Julia metaprogramming](https://docs.julialang.org/en/v1/manual/metaprogramming/).


# Installation

```
pip install git+https://github.com/hanjinliu/macro-kit
```

# Usage

1. Define a macro-recordable function

```python
from macrokit import Macro
macro = Macro()

@macro.record
def add(a, b):
    return a + b
```

2. Just call!

```python
add(1, 2)
add("abc", "xyz")
macro
```
```
add(1, 2)
add('abc', 'xyz')
```
