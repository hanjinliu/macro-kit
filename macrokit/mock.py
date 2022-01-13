from __future__ import annotations
from typing import Any
from .symbol import Symbol
from .expression import Expr
from .macro import BINOP_MAP, UNOP_MAP
from .head import Head


class Mock:
    """
    Helper class for easier Expr object handling.

    Instead of writing like ``Expr("getattr", [a, b])``, Mock object can do it.

    >>> mock = Mock("a")
    >>> mock.b  # Mock<a.b>
    >>> mock.method(arg=10)  # Mock<a.methodarg=10>

    """
    def __init__(self, sym: Symbol | Expr | str):
        if isinstance(sym, str):
            _sym = Symbol.var(sym)
        else:
            _sym = sym
        self._sym: Symbol | Expr = _sym

    @property
    def expr(self) -> Symbol | Expr:
        """Convert Mock into an Expr-style object."""
        return self._sym

    def __getattr__(self, attr: str) -> Mock:
        expr = Expr(Head.getattr, [self._sym, attr])
        return self.__class__(expr)

    def __getitem__(self, key: Any) -> Mock:
        expr = Expr(Head.getitem, [self._sym, key])
        return self.__class__(expr)

    def __call__(self, *args, **kwargs):
        expr = Expr.parse_call(self._sym, args, kwargs)
        return self.__class__(expr)

    def __str__(self) -> str:
        return str(self._sym)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self._sym}>"

    def _binop(self, s: str, other: Any):
        expr = Expr(Head.binop, [BINOP_MAP[s], self._sym, other])
        return self.__class__(expr)

    def __add__(self, other) -> Mock:
        return self._binop("__add__", other)

    def __sub__(self, other) -> Mock:
        return self._binop("__sub__", other)

    def __mul__(self, other) -> Mock:
        return self._binop("__mul__", other)

    def __div__(self, other) -> Mock:
        return self._binop("__div__", other)

    def __mod__(self, other) -> Mock:
        return self._binop("__mod__", other)

    def __eq__(self, other) -> Mock:
        return self._binop("__eq__", other)

    def __neq__(self, other) -> Mock:
        return self._binop("__neq__", other)

    def __gt__(self, other) -> Mock:
        return self._binop("__gt__", other)

    def __ge__(self, other) -> Mock:
        return self._binop("__ge__", other)

    def __lt__(self, other) -> Mock:
        return self._binop("__lt__", other)

    def __le__(self, other) -> Mock:
        return self._binop("__le__", other)

    def __pow__(self, other) -> Mock:
        return self._binop("__pow__", other)

    def __matmul__(self, other) -> Mock:
        return self._binop("__matmul__", other)

    def __floordiv__(self, other) -> Mock:
        return self._binop("__floordiv__", other)

    def __and__(self, other) -> Mock:
        return self._binop("__and__", other)

    def __or__(self, other) -> Mock:
        return self._binop("__or__", other)

    def __xor__(self, other) -> Mock:
        return self._binop("__xor__", other)

    def _aug(self, s: str, other: Any):
        expr = Expr(Head.binop, [BINOP_MAP[s], self._sym, other])
        self._sym = expr
        return self

    def __iadd__(self, other) -> Mock:
        return self._aug("__add__", other)

    def __isub__(self, other) -> Mock:
        return self._aug("__sub__", other)

    def __imul__(self, other) -> Mock:
        return self._aug("__mul__", other)

    def __idiv__(self, other) -> Mock:
        return self._aug("__div__", other)

    def _unop(self, s: str):
        expr = Expr(Head.unop, [UNOP_MAP[s], self._sym])
        return self.__class__(expr)

    def __pos__(self) -> Mock:
        return self._unop("__pos__")

    def __neg__(self) -> Mock:
        return self._unop("__neg__")

    def __invert__(self) -> Mock:
        return self._unop("__invert__")
