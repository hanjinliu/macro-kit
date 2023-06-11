from __future__ import annotations

from typing import Any, Iterator
import inspect
from macrokit._symbol import Symbol
from macrokit.head import Head
from macrokit.expression import Expr


class ExprCheckError(Exception):
    """Exception raised when expression check failed."""

    def __init__(self, msg: str, expr: Symbol | Expr, line: Symbol | Expr):
        super().__init__(msg)
        self._expr = expr
        self._line = line

    def __str__(self) -> str:
        """Error description with the causes."""
        return f"{super().__str__()} (Caused by: `{self._expr}` in `{self._line}`)"


_BUILTIN_FUNCTION_OR_METHOD = type(print)


def _split_left(expr: Symbol | Expr) -> Iterator[Symbol]:
    if isinstance(expr, Symbol):
        yield expr
    elif expr.head in (Head.tuple, Head.list):
        for e in expr.args:
            yield from _split_left(e)
    elif expr.head is Head.star:
        if isinstance(expr.args[0], Symbol):
            yield expr.args[0]


def _split_names(expr: Symbol | Expr) -> Iterator[Symbol]:
    if isinstance(expr, Symbol):
        if expr.name.isidentifier():
            yield expr
    elif expr.head is Head.getattr:
        yield expr.split_getattr()[0]
    elif expr.head is Head.kw:
        yield from _split_names(expr.args[1])
    else:
        for e in expr.args:
            yield from _split_names(e)


def check_names(code: Expr, vars: set[str]) -> list[ExprCheckError]:
    """Check if the given code contains undefined names."""
    vars = {str(v) for v in vars}
    errors = list[ExprCheckError]()
    for line in code.iter_lines():
        if isinstance(line, Symbol):
            if line.name.isidentifier() and line.name not in vars:
                errors.append(
                    ExprCheckError(f"Name {line.name!r} is not defined", line, line)
                )
        else:
            if line.head is Head.assign:
                for left in _split_left(line.args[0]):
                    vars.add(left.name)
            elif line.head is Head.del_:
                for arg in line.args:
                    if isinstance(arg, Symbol):
                        vars.discard(arg.name)
            else:
                for sym in _split_names(line):
                    if sym.name not in vars:
                        errors.append(
                            ExprCheckError(
                                f"Name {sym.name!r} is not defined", sym, line
                            )
                        )
    return errors


def check_attributes(code: Expr, ns: dict[str, Any]) -> list[ExprCheckError]:
    """Check if the given code contains undefined attributes."""
    errors = list[ExprCheckError]()
    for line in code.iter_lines():
        for expr in code.iter_getattr():
            try:
                expr.eval(ns)
            except AttributeError as e:
                errors.append(ExprCheckError(str(e), expr, line))
    return errors


def check_call_args(code: Expr, ns: dict[str, Any]) -> list[ExprCheckError]:
    """Check if all the function calls in the given code are valid."""
    errors = list[ExprCheckError]()
    for line in code.iter_lines():
        if isinstance(line, Symbol):
            continue
        for expr in line.iter_call():
            fexpr, args, kwargs = expr.split_call()
            _method = fexpr.eval(ns)
            if isinstance(_method, _BUILTIN_FUNCTION_OR_METHOD):
                continue
            try:
                inspect.signature(_method).bind(*args, **kwargs)
            except TypeError as e:
                errors.append(ExprCheckError(str(e), expr, line))
    return errors
