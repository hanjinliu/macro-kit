from .symbol import Symbol, symbol, register_type
from .expression import Expr, Head
from .macro import Macro
from functools import wraps

__all__ = ["Symbol", "Head", "Expr", "Macro",
           "symbol", "register_type",
           "blocked", "record", "property", "get_macro"
           ]

_MACRO = Macro()

@wraps(_MACRO.blocked)
def blocked():
    return _MACRO.blocked()

@wraps(_MACRO.record)
def record(obj = None, *, returned_callback = None):
    return _MACRO.record(obj, returned_callback=returned_callback)

@wraps(_MACRO.property)
def property(prop):
    return _MACRO.property(prop)

def get_macro():
    return str(_MACRO)

del wraps