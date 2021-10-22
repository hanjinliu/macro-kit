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
    assign  = "assign"
    value   = "value"
    comment = "comment"
    assert_ = "assert_"

_EXEC = (Head.init, Head.assign, Head.setitem, Head.setattr, Head.assert_, Head.delattr, Head.delitem)

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
        Head.assign : lambda e: f"{e.args[0]} = {e.args[1]}",
        Head.value  : lambda e: str(e.args[0]),
        Head.assert_: lambda e: f"assert {e.args[0]}, {e.args[1]}".rstrip(", "),
        Head.comment: lambda e: f"# {e.args[0]}",
    }
    
    def __init__(self, head: Head, args: Iterable[Any]):
        self.head = Head(head)
        if head == Head.value:
            self.args = [args[0]]
        else:
            self.args = list(map(self.__class__.parse, args))
            
        self.number = self.__class__.n
        self.__class__.n += 1
    
    def __repr__(self) -> str:
        return self.__class__._map[self.head](self)
    
    def _dump(self, ind: int = 0) -> str:
        """
        Recursively expand expressions until it reaches value/assign expression.
        """
        if self.head in (Head.value, Head.assign):
            return str(self)
        out = [f"head: {self.head.name}\n{' '*ind}args:\n"]
        for i, arg in enumerate(self.args):
            out.append(f"{i:>{ind+2}}: {arg._dump(ind+4)}\n")
        return "".join(out)
    
    def __eq__(self, expr: Expr|Symbol) -> bool:
        if self.head != Head.value:
            if isinstance(expr, self.__class__) and self.head == expr.head:
                return self.args == expr.args
            else:
                return False
        elif isinstance(expr, Symbol):
            return self.args[0] == expr
        elif isinstance(expr, self.__class__):
            return self.args[0] == expr.args[0]
        else:
            raise ValueError(f"'==' is not supported between Expr and {type(expr)}")
    
    def dump(self):
        return self._dump()
        
    def copy(self):
        return deepcopy(self)
    
    def eval(self, _globals: dict[Symbol, Any] = {}, _locals: dict[Symbol, Any] = {}):
        _globals = {sym.name: v for sym, v in _globals.items()}
        _locals = {sym.name: v for sym, v in _locals.items()}
        if self.head in _EXEC:
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
            inputs.append(cls(Head.assign, [Symbol(k), v]))
        return inputs
    
    @classmethod
    def parse(cls, a: Any) -> Expr:
        return a if isinstance(a, cls) else cls(Head.value, [symbol(a)])
    
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
    
    def iter_values(self) -> Iterator[Expr]:
        """
        Recursively iterate along all the values.
        """
        for arg in self.args:
            if isinstance(arg, self.__class__):
                if arg.head == Head.value:
                    yield arg
                else:
                    yield from arg.iter_values()
    
    
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
        if not inplace:
            self = self.copy()
            
        for arg in self.iter_values():
            try:
                new = mapping[arg.args[0]]
            except KeyError:
                pass
            else:
                arg.args[0] = new
        return self


def make_symbol_str(obj: Any):
    return f"var{hex(id(obj))}"

def symbol(obj: Any) -> Symbol:
    if isinstance(obj, (Symbol, Expr)):
        return obj
    
    valid = True
    objtype = type(obj)
    if isinstance(obj, str):
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
            seq = make_symbol_str(obj) # hexadecimals are easier to distinguish
            valid = False
            
    sym = Symbol(seq, id(obj), type(obj))
    sym.valid = valid
    return sym