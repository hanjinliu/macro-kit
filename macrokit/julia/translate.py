from __future__ import annotations
from typing import Any, Iterable, Callable
from ..expression import Expr, Head
from ..ast import parse

# TODO: mapping of builtin

def as_str(expr: Any, indent: int = 0):
    if isinstance(expr, Expr):
        return _STR_MAP[expr.head](expr, indent)
    else:
        return " "*indent + str(expr)

def rm_par(s: str):
    if s[0] == "(" and s[-1] == ")":
        s = s[1:-1]
    return s

def sjoin(sep: str, iterable: Iterable[Any], indent: int = 0):
    return sep.join(as_str(expr, indent) for expr in iterable)

_STR_MAP: dict[Head, Callable[[Expr, int], str]] = {
    Head.getattr  : lambda e, i: f"{as_str(e.args[0], i)}.{as_str(e.args[1])}",
    Head.getitem  : lambda e, i: f"{as_str(e.args[0], i)}[{as_str(e.args[1])}]",
    Head.call     : lambda e, i: f"{as_str(e.args[0], i)}({sjoin(', ', e.args[1:])})",
    Head.assign   : lambda e, i: f"{as_str(e.args[0], i)} = {e.args[1]}",
    Head.kw       : lambda e, i: f"{as_str(e.args[0])}={as_str(e.args[1])}",
    Head.assert_  : lambda e, i: " "*i + f"@assert {as_str(e.args[0])}, {as_str(e.args[1])}".rstrip(", "),
    Head.comment  : lambda e, i: " "*i + f"# {e.args[0]}",
    Head.binop    : lambda e, i: " "*i + f"({as_str(e.args[1])} {as_str(e.args[0])} {as_str(e.args[2])})",

    Head.block    : lambda e, i: sjoin("\n", e.args, i),
    Head.function : lambda e, i: " "*i + f"function {as_str(e.args[0])}\n{as_str(e.args[1], i+4)}\n" + \
                                 " "*i + "end",
    Head.return_  : lambda e, i: " "*i + f"return {sjoin(', ', e.args)}",
    Head.if_      : lambda e, i: " "*i + f"if {rm_par(as_str(e.args[0]))}\n{as_str(e.args[1], i+4)}\n" + \
                                 " "*i + f"else\n{as_str(e.args[2], i+4)}\n" + \
                                 " "*i + "end",
    Head.for_     : lambda e, i: " "*i + f"for {rm_par(as_str(e.args[0]))}\n{as_str(e.args[1], i+4)}\n" + \
                                 " "*i + "end",
    Head.annotate : lambda e, i: f"{as_str(e.args[0], i)}::{as_str(e.args[1])}"
}

def to_julia_expr(s: str|Expr):
    if isinstance(s, str):
        s = parse(s)
    julia_code = as_str(s)
    return julia_code
