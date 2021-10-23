from __future__ import annotations
from copy import deepcopy
from typing import Callable, Iterable, Iterator, Any
from enum import Enum
from numbers import Number
from .symbol import Symbol
        
class Head(Enum):
    init    = "init"
    getattr = "getattr"
    setattr = "setattr"
    delattr = "delattr"
    getitem = "getitem"
    setitem = "setitem"
    delitem = "delitem"
    call    = "call"
    len     = "len"
    assign  = "assign"
    kw      = "kw"
    comment = "comment"
    assert_ = "assert_"

EXEC = (Head.init, Head.assign, Head.setitem, Head.setattr, Head.assert_, Head.delattr, 
        Head.delitem, Head.comment)

class Expr:
    n: int = 0
    
    # a map of how to conver expression into string.
    _map: dict[Head, Callable[[Expr], str]] = {
        Head.init   : lambda e: f"{e.args[0]} = {e.args[1]}({', '.join(map(str, e.args[2:]))})",
        Head.getattr: lambda e: f"{e.args[0]}.{e.args[1]}",
        Head.setattr: lambda e: f"{e.args[0]}.{e.args[1]} = {e.args[2]}",
        Head.delattr: lambda e: f"del {e.args[0]}.{e.args[1]}",
        Head.getitem: lambda e: f"{e.args[0]}[{e.args[1]}]",
        Head.setitem: lambda e: f"{e.args[0]}[{e.args[1]}] = {e.args[2]}",
        Head.delitem: lambda e: f"del {e.args[0]}[{e.args[1]}]",
        Head.call   : lambda e: f"{e.args[0]}({', '.join(map(str, e.args[1:]))})",
        Head.len    : lambda e: f"len({e.args[0]})",
        Head.assign : lambda e: f"{e.args[0]} = {e.args[1]}",
        Head.kw     : lambda e: f"{e.args[0]}={e.args[1]}",
        Head.assert_: lambda e: f"assert {e.args[0]}, {e.args[1]}".rstrip(", "),
        Head.comment: lambda e: f"# {e.args[0]}",
    }
    
    def __init__(self, head: Head, args: Iterable[Any]):
        self.head = Head(head)
        self.args = list(map(self.__class__.parse, args))
            
        self.number = self.__class__.n
        self.__class__.n += 1
    
    def __repr__(self) -> str:
        return self.__class__._map[self.head](self)
    
    def __eq__(self, expr: Expr|Symbol) -> bool:
        if isinstance(expr, self.__class__):
            if self.head == expr.head:
                return self.args == expr.args
            else:
                return False
        else:
            raise ValueError(f"'==' is not supported between Expr and {type(expr)}")
    
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
        return self._dump()
        
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
        inputs = [sym, init_cls] + cls.convert_args(args, kwargs)
        return cls(head=Head.init, args=inputs)
    
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
    def parse(cls, a: Any) -> Expr:
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
    _dict = {}
    for k, v in mapping_list:
        if isinstance(v, Expr) and v.head in EXEC:
            raise ValueError("Cannot replace a symbol to a non-evaluable expression.")
        _dict[symbol(k)] = v
    return _dict


def make_symbol_str(obj: Any):
    # hexadecimals are easier to distinguish
    _id = id(obj)
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