"""macro-kit is a package for macro recording and metaprogramming in Python."""

__version__ = "0.3.8"
__author__ = "Hanjin Liu"
__email__ = "liuhanjin-sc@g.ecc.u-tokyo.ac.jp"

from ._symbol import Symbol, register_type, COMPILED
from .expression import Expr, Head, symbol
from .macro import Macro, MacroFlags
from .mock import Mock
from .ast import parse
from functools import wraps


__all__ = [
    "Symbol",
    "Head",
    "Expr",
    "Macro",
    "Mock",
    "symbol",
    "register_type",
    "parse",
    "blocked",
    "record",
    "property",
    "dump",
    "get_macro",
    "set_flags",
    "COMPILED",
]

# global macro instance and its functions

_MACRO = Macro()


@wraps(_MACRO.blocked)
def blocked():  # noqa: D103
    return _MACRO.blocked()


@wraps(_MACRO.record)
def record(obj=None, *, returned_callback=None):  # noqa: D103
    return _MACRO.record(obj, returned_callback=returned_callback)


@wraps(_MACRO.property)
def property(prop):  # noqa: D103
    return _MACRO.property(prop)


@wraps(_MACRO.dump)
def dump() -> str:  # noqa: D103
    return _MACRO.dump()


def get_macro(mapping=None) -> str:
    """Get macro as a string."""
    if mapping:
        macro = _MACRO.format(mapping)
    else:
        macro = _MACRO
    return str(macro)


def set_flags(
    Get: bool = True, Set: bool = True, Delete: bool = True, Return: bool = True
):
    """Set macro flags."""
    _MACRO._flags = MacroFlags(Get, Set, Delete, Return)
    return None


del wraps
