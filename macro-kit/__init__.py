from .symbol import Symbol, symbol
from .expression import Expr, Head
from .macro import Macro
from functools import wraps

_MACRO = Macro()

@wraps(_MACRO.blocked)
def blocked():
    return _MACRO.blocked()

@wraps(_MACRO.format)
def format(self, mapping, inplace = False) -> Macro:
    return _MACRO.format(mapping, inplace)

@wraps(_MACRO.record)
def record(obj = None, *, returned_callback = None):
    return _MACRO.record(obj, returned_callback=returned_callback)

@wraps(_MACRO.property)
def property(prop):
    return _MACRO.property(prop)

del wraps