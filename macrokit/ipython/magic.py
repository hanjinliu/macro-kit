from __future__ import annotations
from functools import wraps
from typing import Callable

from IPython.core.magic import register_line_cell_magic, needs_local_scope

from ..expression import Expr
from ..ast import parse

def register_magic(func: Callable[[Expr], Expr]):
    """
    Make a magic command more like Julia's macro system.
    
    Instead of using string, you can register a magic that uses Expr as the 
    input and return a modified Expr. It is usually easier and safer to
    execute metaprogramming this way.

    Parameters
    ----------
    func : Callable[[Expr], Expr]
        Function that will used as a magic command.

    Returns
    -------
    Callable
        Registered function itself.
    
    Examples
    --------
    
    .. code-block:: python

        @register_magic
        def print_code(expr):
            print(expr)
            return expr
    
    The ``print_code`` magic is registered as an ipython magic.
    
    .. code-block:: python
    
        %print_code a = 1
    
    .. code-block:: python
    
        %%print_code
        def func(a):
            return a + 1
        
    """    
    @register_line_cell_magic
    @needs_local_scope
    @wraps(func)
    def _ipy_magic(line: str, cell: str = None, local_ns = None):
        if cell is None:
            cell = line
        block = parse(cell)
        block_out = func(block)
        return block_out.eval(local_ns, local_ns)
    return func