from __future__ import annotations
from functools import wraps
from typing import Callable

from IPython.core.magic import register_line_cell_magic, needs_local_scope

from ..expression import Expr
from ..ast import parse

def register_magic(func: Callable[[Expr], Expr]):
    @wraps(func)
    def _ipy_magic(line: str, cell: str = None, local_ns = None):
        if cell is None:
            cell = line
        block = parse(cell)
        block_out = func(block)
        return block_out.eval(local_ns, local_ns)
    register_line_cell_magic(needs_local_scope(_ipy_magic))
    return func