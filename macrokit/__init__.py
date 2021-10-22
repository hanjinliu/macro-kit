__version__ = "0.0.1dev0"

from .symbol import Symbol, register_type
from .expression import Expr, Head, symbol
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

def dump() -> str:
    return _MACRO.dump()

def get_macro() -> str:
    return str(_MACRO)

del wraps