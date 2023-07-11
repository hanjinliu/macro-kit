from __future__ import annotations

from typing import Any
import inspect
import warnings
from macrokit._symbol import Symbol
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


# TODO: import
def check_attributes(code: Expr, ns: dict[str, Any] = {}) -> list[ExprCheckError]:
    """Check if the given code contains undefined attributes."""
    errors: list[ExprCheckError] = []
    for line in code.iter_lines():
        if isinstance(line, Symbol):
            continue
        for expr in line.iter_getattr():
            try:
                expr.eval(ns)
            except AttributeError as e:
                errors.append(ExprCheckError(str(e), expr, line))
            except NameError:
                # variables defined during the execution of the code.
                pass
            except Exception as e:
                warnings.warn(f"Unexpected exception in line {line}: {e}")
    return errors


def check_call_args(code: Expr, ns: dict[str, Any] = {}) -> list[ExprCheckError]:
    """Check if all the function calls in the given code are valid."""
    errors: list[ExprCheckError] = []
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
            except (ValueError, NameError):
                # builtin classes such as `range`.
                pass
            except Exception as e:
                warnings.warn(f"Unexpected exception in line {line}: {e}")
    return errors
