from __future__ import annotations
from typing import Any
from .symbol import Symbol
from .expression import Expr
from .head import Head


class Mock:
    def __init__(self, sym: Symbol | Expr | str):
        if isinstance(sym, str):
            sym = Symbol.var(sym)
        self._sym: Symbol | Expr = sym

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
        return f"{self.__class__.__name__} object of {self._sym}"

    @property
    def expr(self) -> Symbol | Expr:
        return self._sym
