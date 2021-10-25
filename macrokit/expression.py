from __future__ import annotations
from copy import deepcopy
from typing import Callable, Iterable, Iterator, Any
from enum import Enum
from numbers import Number
from .symbol import Symbol
        
class Head(Enum):
    getattr  = "getattr"
    getitem  = "getitem"
    del_     = "del"
    call     = "call"
    assign   = "assign"
    kw       = "kw"
    comment  = "comment"
    assert_  = "assert"
    binop    = "binop"
    block    = "block"
    function = "function"
    return_  = "return"
    if_      = "if"
    elif_    = "elif"
    for_     = "for"
    annotate = "annotate"

EXEC = (Head.assign, Head.assert_, Head.comment, Head.function, Head.return_, Head.if_, Head.elif_, Head.for_)

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
    Head.del_     : lambda e, i: " "*i + f"del {as_str(e.args[0])}",
    Head.call     : lambda e, i: f"{as_str(e.args[0], i)}({sjoin(', ', e.args[1:])})",
    Head.assign   : lambda e, i: f"{as_str(e.args[0], i)} = {e.args[1]}",
    Head.kw       : lambda e, i: f"{as_str(e.args[0])}={as_str(e.args[1])}",
    Head.assert_  : lambda e, i: " "*i + f"assert {as_str(e.args[0])}, {as_str(e.args[1])}".rstrip(", "),
    Head.comment  : lambda e, i: " "*i + f"# {e.args[0]}",
    Head.binop    : lambda e, i: " "*i + f"({as_str(e.args[1])} {as_str(e.args[0])} {as_str(e.args[2])})",
    
    Head.block    : lambda e, i: sjoin("\n", e.args, i),
    Head.function : lambda e, i: " "*i + f"def {as_str(e.args[0])}:\n{as_str(e.args[1], i+4)}",
    Head.return_  : lambda e, i: " "*i + f"return {sjoin(', ', e.args)}",
    Head.if_      : lambda e, i: " "*i + f"if {rm_par(as_str(e.args[0]))}:\n{as_str(e.args[1], i+4)}\n" + \
                                 " "*i + f"else:\n{as_str(e.args[2], i+4)}",
    Head.elif_    : lambda e, i: " "*i + f"if {rm_par(as_str(e.args[0]))}:\n{as_str(e.args[1], i+4)}\n" + \
                                 " "*i + f"else:\n{as_str(e.args[2], i+4)}",
    Head.for_     : lambda e, i: " "*i + f"for {rm_par(as_str(e.args[0]))}:\n{as_str(e.args[1], i+4)}",
    Head.annotate : lambda e, i: f"{as_str(e.args[0], i)}: {as_str(e.args[1])}"
}

class Expr:
    n: int = 0
        
    def __init__(self, head: Head, args: Iterable[Any]):
        self.head = Head(head)
        self.args = list(map(self.__class__.parse_object, args))
            
        self.number = self.__class__.n
        self.__class__.n += 1
    
    def __repr__(self) -> str:
        s = str(self)
        s = rm_par(s)
        s = s.replace("\n", "\n  ")
        return f":({s})"
    
    def __str__(self) -> str:
        return as_str(self)
    
    def __eq__(self, expr: Expr|Symbol) -> bool:
        if isinstance(expr, self.__class__):
            if self.head == expr.head:
                return self.args == expr.args
            else:
                return False
        else:
            return False
    
    def _dump(self, ind: int = 0) -> str:
        """
        Recursively expand expressions until it reaches value/kw expression.
        """
        out = [f"head: {self.head.name}\n{' '*ind}args:\n"]
        for i, arg in enumerate(self.args):
            if isinstance(arg, Symbol):
                value = arg
            else:
                value = arg._dump(ind+4)
            out.append(f"{i:>{ind+2}}: {value}\n")
        return "".join(out)
    
    def dump(self):
        s = self._dump()
        return s.rstrip("\n") + "\n"
        
    def copy(self):
        return deepcopy(self)
    
    def eval(self, _globals: dict[Symbol|str, Any] = {}, _locals: dict = {}):
        _globals = {(sym.name if isinstance(sym, Symbol) else sym): v 
                    for sym, v in _globals.items()}
        if self.head in EXEC:
            return exec(str(self), _globals, _locals)
        else:
            return eval(str(self), _globals, _locals)
        
    @classmethod
    def parse_method(cls, obj: Any, func: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Expr:
        """
        Make a method call expression.
        Expression: obj.func(*args, **kwargs)
        """
        method = cls(head=Head.getattr, args=[symbol(obj), func])
        inputs = [method] + cls.convert_args(args, kwargs)
        return cls(head=Head.call, args=inputs)

    @classmethod
    def parse_init(cls, 
                   obj: Any,
                   init_cls: type, 
                   args: tuple[Any, ...] = None, 
                   kwargs: dict[str, Any] = None) -> Expr:
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        sym = symbol(obj)
        return cls(Head.assign, [sym, 
                                 cls(Head.call, [init_cls] + cls.convert_args(args, kwargs))
                                 ])
    
    @classmethod
    def parse_call(cls, 
                   func: Callable, 
                   args: tuple[Any, ...] = None, 
                   kwargs: dict[str, Any] = None) -> Expr:
        """
        Make a function call expression.
        Expression: func(*args, **kwargs)
        """
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        inputs = [func] + cls.convert_args(args, kwargs)
        return cls(head=Head.call, args=inputs)
        
    @classmethod
    def convert_args(cls, args: tuple[Any, ...], kwargs: dict[str|Symbol, Any]) -> list:
        inputs = []
        for a in args:
            inputs.append(a)
                
        for k, v in kwargs.items():
            inputs.append(cls(Head.kw, [Symbol(k), v]))
        return inputs
    
    @classmethod
    def parse_object(cls, a: Any) -> Symbol | Expr:
        return a if isinstance(a, cls) else symbol(a)
    
    def iter_args(self) -> Iterator[Symbol]:
        """
        Recursively iterate along all the arguments.
        """
        for arg in self.args:
            if isinstance(arg, self.__class__):
                yield from arg.iter_args()
            elif isinstance(arg, Symbol):
                yield arg
            else:
                raise RuntimeError(arg)
        
    def iter_expr(self) -> Iterator[Expr]:
        """
        Recursively iterate over all the nested Expr, until it reached to non-nested Expr.
        This method is useful in macro generation.
        """        
        yielded = False
        for arg in self.args:
            if isinstance(arg, self.__class__):
                yield from arg.iter_expr()
                yielded = True
        
        if not yielded:
            yield self
    
    def format(self, mapping: dict[Symbol, Symbol|Expr], inplace: bool = False) -> Expr:
        mapping = check_format_mapping(mapping.items())
            
        if not inplace:
            self = self.copy()
            
        return self._unsafe_format(mapping)
    
    def _unsafe_format(self, mapping: dict[Symbol, Symbol|Expr]) -> Expr:
        for i, arg in enumerate(self.args):
            if isinstance(arg, Expr):
                arg._unsafe_format(mapping)
            else:
                try:
                    new = mapping[arg]
                except KeyError:
                    pass
                else:
                    self.args[i] = new
        return self

def check_format_mapping(mapping_list: Iterable[tuple[Any, Any]]) -> dict[Symbol, Symbol|Expr]:
    _dict: dict[Symbol, Symbol] = {}
    for k, v in mapping_list:
        if isinstance(v, Expr) and v.head in EXEC:
            raise ValueError("Cannot replace a symbol to a non-evaluable expression.")
        _dict[symbol(k)] = symbol(v)
    return _dict


def make_symbol_str(obj: Any):
    # hexadecimals are easier to distinguish
    _id = id(obj)
    if obj is not None:
        Symbol._variables.add(_id)
    return f"var{hex(_id)}"

def symbol(obj: Any, constant: bool = True) -> Symbol:
    """
    Make a proper Symbol instance from any objects.
    
    Unlike Symbol(...) constructor, which directly make a Symbol from a string, this function
    checks input type and determine the optimal string to represent the object. Especially,
    Symbol("xyz") will return ``:xyz`` while symbol("xyz") will return ``:'xyz'``.

    Parameters
    ----------
    obj : Any
        Any object from which a Symbol will be created.
    constant : bool, default is True
        If true, object is interpreted as a constant like 1 or "a". Otherwise object is converted
        to a variable that named with its ID.

    Returns
    -------
    Symbol
    """    
    if isinstance(obj, (Symbol, Expr)):
        return obj
        
    objtype = type(obj)
    if not constant or id(obj) in Symbol._variables:
        seq = make_symbol_str(obj)
        constant = False
    elif isinstance(obj, str):
        seq = repr(obj)
    elif isinstance(obj, Number): # int, float, bool, ...
        seq = obj
    elif isinstance(obj, tuple):
        seq = "(" + ", ".join(symbol(a)._name for a in obj) + ")"
        if objtype is not tuple:
            seq = objtype.__name__ + seq
    elif isinstance(obj, list):
        seq = "[" + ", ".join(symbol(a)._name for a in obj) + "]"
        if objtype is not list:
            seq = f"{objtype.__name__}({seq})"
    elif isinstance(obj, dict):
        seq = "{" + ", ".join(f"{symbol(k)}: {symbol(v)}" for k, v in obj.items()) + "}"
        if objtype is not dict:
            seq = f"{objtype.__name__}({seq})"
    elif isinstance(obj, set):
        seq = "{" + ", ".join(symbol(a)._name for a in obj) + "}"
        if objtype is not set:
            seq = f"{objtype.__name__}({seq})"
    elif isinstance(obj, slice):
        seq = f"{objtype.__name__}({obj.start}, {obj.stop}, {obj.step})"
    elif objtype in Symbol._type_map:
        seq = Symbol._type_map[objtype](obj)
    else:
        for k, func in Symbol._type_map.items():
            if isinstance(obj, k):
                seq = func(obj)
                break
        else:
            seq = make_symbol_str(obj)
            constant = False
    
    if isinstance(seq, Symbol):
        # The output of register_type can be a Symbol
        sym = seq
    else:
        sym = Symbol(seq, id(obj), type(obj))
        sym.constant = constant
    return sym