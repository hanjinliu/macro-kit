from __future__ import annotations

import builtins
from typing import Any

from macrokit._symbol import Symbol
from macrokit.expression import Expr
from macrokit.head import Head
from macrokit.macro import BINOP_MAP, UNOP_MAP
from macrokit.type_map import register_type


def _mock_to_expr(mock: Mock):
    return mock.expr


@register_type(_mock_to_expr)
class Mock:
    """Helper class for easier Expr object handling.

    Instead of writing like ``Expr("getattr", [a, b])``, Mock object can do it.

    >>> mock = Mock("a")
    >>> mock.b  # Mock<a.b>
    >>> mock.method(arg=10)  # Mock<a.method(arg=10)>
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

    def eval(self, _globals: builtins.dict = {}, _locals: builtins.dict = {}):
        """Evaluate the Mock object."""
        return self._sym.eval(_globals, _locals)

    def __getattr__(self, attr: str) -> Mock:
        """Return a Mock with expression 'mock.attr'."""
        expr = Expr(Head.getattr, [self._sym, attr])
        return self.__class__(expr)

    def __getitem__(self, key: Any) -> Mock:
        """Return a Mock with expression 'mock[key]'."""
        expr = Expr(Head.getitem, [self._sym, key])
        return self.__class__(expr)

    def __call__(self, *args, **kwargs) -> Mock:
        """Return a Mock with expression 'mock(*args, **kwargs)'."""
        expr = Expr.parse_call(self._sym, args, kwargs)
        return self.__class__(expr)

    def __str__(self) -> str:
        """Return symbol string."""
        return str(self._sym)

    def __repr__(self) -> str:
        """Return symbol string in repr."""
        return f"{self.__class__.__name__}<{self._sym}>"

    def _binop(self, s: str, other: Any):
        expr = Expr(Head.binop, [BINOP_MAP[s], self._sym, other])
        return self.__class__(expr)

    def __add__(self, other) -> Mock:
        """Return a Mock with expression 'mock + other'."""
        return self._binop("__add__", other)

    def __sub__(self, other) -> Mock:
        """Return a Mock with expression 'mock - other'."""
        return self._binop("__sub__", other)

    def __mul__(self, other) -> Mock:
        """Return a Mock with expression 'mock * other'."""
        return self._binop("__mul__", other)

    def __div__(self, other) -> Mock:
        """Return a Mock with expression 'mock / other'."""
        return self._binop("__div__", other)

    def __mod__(self, other) -> Mock:
        """Return a Mock with expression 'mock % other'."""
        return self._binop("__mod__", other)

    def __eq__(self, other) -> Mock:  # type: ignore
        """Return a Mock with expression 'mock == other'."""
        return self._binop("__eq__", other)

    def __neq__(self, other) -> Mock:
        """Return a Mock with expression 'mock != other'."""
        return self._binop("__neq__", other)

    def __gt__(self, other) -> Mock:
        """Return a Mock with expression 'mock > other'."""
        return self._binop("__gt__", other)

    def __ge__(self, other) -> Mock:
        """Return a Mock with expression 'mock >= other'."""
        return self._binop("__ge__", other)

    def __lt__(self, other) -> Mock:
        """Return a Mock with expression 'mock < other'."""
        return self._binop("__lt__", other)

    def __le__(self, other) -> Mock:
        """Return a Mock with expression 'mock <= other'."""
        return self._binop("__le__", other)

    def __pow__(self, other) -> Mock:
        """Return a Mock with expression 'mock ** other'."""
        return self._binop("__pow__", other)

    def __matmul__(self, other) -> Mock:
        """Return a Mock with expression 'mock @ other'."""
        return self._binop("__matmul__", other)

    def __floordiv__(self, other) -> Mock:
        """Return a Mock with expression 'mock // other'."""
        return self._binop("__floordiv__", other)

    def __and__(self, other) -> Mock:
        """Return a Mock with expression 'mock & other'."""
        return self._binop("__and__", other)

    def __or__(self, other) -> Mock:
        """Return a Mock with expression 'mock | other'."""
        return self._binop("__or__", other)

    def __xor__(self, other) -> Mock:
        """Return a Mock with expression 'mock ^ other'."""
        return self._binop("__xor__", other)

    def _aug(self, s: str, other: Any):
        expr = Expr(Head.binop, [BINOP_MAP[s], self._sym, other])
        self._sym = expr
        return self

    def __iadd__(self, other) -> Mock:
        """Return a Mock with expression 'mock += other'."""
        return self._aug("__add__", other)

    def __isub__(self, other) -> Mock:
        """Return a Mock with expression 'mock -= other'."""
        return self._aug("__sub__", other)

    def __imul__(self, other) -> Mock:
        """Return a Mock with expression 'mock *= other'."""
        return self._aug("__mul__", other)

    def __idiv__(self, other) -> Mock:
        """Return a Mock with expression 'mock /= other'."""
        return self._aug("__div__", other)

    def _unop(self, s: str):
        expr = Expr(Head.unop, [UNOP_MAP[s], self._sym])
        return self.__class__(expr)

    def __pos__(self) -> Mock:
        """Return a Mock with expression '+mock'."""
        return self._unop("__pos__")

    def __neg__(self) -> Mock:
        """Return a Mock with expression '-mock'."""
        return self._unop("__neg__")

    def __invert__(self) -> Mock:
        """Return a Mock with expression '~mock'."""
        return self._unop("__invert__")

    def and_(self, other) -> Mock:
        """Return a Mock with expression 'mock and other'."""
        expr = Expr(Head.binop, [Symbol._reserved("and"), self._sym, other])
        return self.__class__(expr)

    def or_(self, other) -> Mock:
        """Return a Mock with expression 'mock or other'."""
        expr = Expr(Head.binop, [Symbol._reserved("or"), self._sym, other])
        return self.__class__(expr)

    def len_(self) -> Mock:
        """Return a Mock with expression 'len(mock)'."""
        expr = Expr.parse_call(builtins.len, (self._sym,))
        return self.__class__(expr)

    def in_(self, other) -> Mock:
        """Return a Mock with expression 'mock in other'."""
        expr = Expr(Head.binop, [Symbol._reserved("in"), self._sym, other])
        return self.__class__(expr)

    def not_in_(self, other) -> Mock:
        """Return a Mock with expression 'mock not in other'."""
        expr = Expr(Head.binop, [Symbol._reserved("not in"), self._sym, other])
        return self.__class__(expr)

    def is_(self, other) -> Mock:
        """Return a Mock with expression 'mock is other'."""
        expr = Expr(Head.binop, [Symbol._reserved("is"), self._sym, other])
        return self.__class__(expr)

    def is_not_(self, other) -> Mock:
        """Return a Mock with expression 'mock is not other'."""
        expr = Expr(Head.binop, [Symbol._reserved("is not"), self._sym, other])
        return self.__class__(expr)

    @classmethod
    def len(cls, mock: Mock) -> Mock:
        """Return a Mock with expression 'len(mock)'."""
        expr = Expr.parse_call(builtins.len, (mock._sym,))
        return cls(expr)

    @classmethod
    def abs(cls, mock: Mock) -> Mock:
        """Return a Mock with expression 'abs(mock)'."""
        expr = Expr.parse_call(builtins.abs, (mock._sym,))
        return cls(expr)

    @classmethod
    def round(cls, mock: Mock) -> Mock:
        """Return a Mock with expression 'round(mock)'."""
        expr = Expr.parse_call(builtins.round, (mock._sym,))
        return cls(expr)

    @classmethod
    def min(cls, mock: Mock) -> Mock:
        """Return a Mock with expression 'min(mock)'."""
        expr = Expr.parse_call(builtins.min, (mock._sym,))
        return cls(expr)

    @classmethod
    def max(cls, mock: Mock) -> Mock:
        """Return a Mock with expression 'max(mock)'."""
        expr = Expr.parse_call(builtins.max, (mock._sym,))
        return cls(expr)

    @classmethod
    def sum(cls, mock: Mock) -> Mock:
        """Return a Mock with expression 'sum(mock)'."""
        expr = Expr.parse_call(builtins.sum, (mock._sym,))
        return cls(expr)

    @classmethod
    def any(cls, mock: Mock) -> Mock:
        """Return a Mock with expression 'any(mock)'."""
        expr = Expr.parse_call(builtins.any, (mock._sym,))
        return cls(expr)

    @classmethod
    def all(cls, mock: Mock) -> Mock:
        """Return a Mock with expression 'all(mock)'."""
        expr = Expr.parse_call(builtins.all, (mock._sym,))
        return cls(expr)

    @classmethod
    def not_(cls, mock: Mock) -> Mock:
        """Return a Mock with expression 'not mock'."""
        expr = Expr(Head.unop, [Symbol._reserved("not "), mock._sym])
        return cls(expr)


def tuple(iterable, /):
    """Construct tuple-expression."""
    return Expr.parse_call(builtins.tuple, (builtins.list(iterable),))


def dict(*args, **kwargs):
    """Construct dict-expression."""
    kwargs = builtins.dict(*args, **kwargs)
    return Expr.parse_call(builtins.dict, kwargs=kwargs)


def set(iterable, /):
    """Construct set-expression."""
    return Expr.parse_call(builtins.set, (builtins.list(iterable),))


def list(iterable, /):
    """Construct list-expression."""
    return Expr.parse_call(builtins.list, (builtins.list(iterable),))


def frozenset(iterable, /):
    """Construct frozenset-expression."""
    return Expr.parse_call(builtins.frozenset, (builtins.list(iterable),))


def slice(*args):
    """Construct slice-expression."""
    return Expr.parse_call(builtins.slice, *args)


def range(*args):
    """Construct range-expression."""
    return Expr.parse_call(builtins.range, *args)


def getattr(obj, name: str):
    """Construct getattr-expression."""
    return Expr(Head.getattr, [obj, name])


def setattr(obj, name: str, value):
    """Construct setattr-expression."""
    return Expr.parse_setattr(obj, name, value)


def delattr(obj, name: str):
    """Construct delattr-expression."""
    return Expr.parse_delattr(obj, name)
