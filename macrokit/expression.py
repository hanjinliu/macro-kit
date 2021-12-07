from __future__ import annotations
from copy import deepcopy
from typing import Callable, Iterable, Iterator, Any, Hashable, overload
from numbers import Number
from .symbol import Symbol
from .head import Head, EXEC
from ._validator import validator

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
    Head.empty    : lambda e, i: "",
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
        self._head = Head(head)
        self._args = list(map(self.__class__.parse_object, args))
        validator(self._head, self._args)
        self.number = self.__class__.n
        self.__class__.n += 1
    
    @property
    def head(self) -> Head:
        return self._head
    
    @property
    def args(self) -> list[Expr|Symbol]:
        return self._args
        
    def __repr__(self) -> str:
        s = str(self)
        if len(s) > 1:
            s = rm_par(s)
        s = s.replace("\n", "\n  ")
        return f":({s})"
    
    def __str__(self) -> str:
        return as_str(self)
    
    def __eq__(self, expr: Expr | Symbol) -> bool:
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
    
    def dump(self) -> str:
        """
        Dump expression into a tree.
        """        
        s = self._dump()
        return s.rstrip("\n") + "\n"
        
    def copy(self) -> Expr:
        # Always copy object deeply.
        return deepcopy(self)
    
    def eval(self, _globals: dict[Symbol|str, Any] = {}, _locals: dict = {}):
        """
        Evaluate or execute macro as an Python script.
        
        Either ``eval`` or ``exec`` is called for every expressions, as determined by its
        header. Calling this function is much safer than those not-recommended usage of 
        ``eval`` or ``exec``.
        
        Parameters
        ----------
        _globals : dict[Symbol, Any], optional
            Mapping from global variable symbols to their values.
        _locals : dict, optional
            Updated variable namespace. Will be a mapping from symbols to values.
            
        """        
        _globals = {(sym.name if isinstance(sym, Symbol) else sym): v 
                    for sym, v in _globals.items()}
        # TODO: Here should be some better ways to assign proper scope.
        if self.head in EXEC:
            return exec(str(self), _globals, _locals)
        else:
            return eval(str(self), _globals, _locals)
        
    @classmethod
    def parse_method(cls, obj: Any, func: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Expr:
        """
        Parse ``obj.func(*args, **kwargs)``.
        """
        method = cls(head=Head.getattr, args=[symbol(obj), func])
        inputs = [method] + cls.convert_args(args, kwargs)
        return cls(head=Head.call, args=inputs)

    @classmethod
    def parse_init(cls, 
                   obj: Any,
                   init_cls: type = None, 
                   args: tuple[Any, ...] = None, 
                   kwargs: dict[str, Any] = None) -> Expr:
        """
        Parse ``obj = init_cls(*args, **kwargs)``.
        """
        if init_cls is None:
            init_cls = type(obj)
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
        Parse ``func(*args, **kwargs)``.
        """
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        inputs = [func] + cls.convert_args(args, kwargs)
        return cls(head=Head.call, args=inputs)
    
    @classmethod
    def parse_setitem(cls, obj, key, value) -> Expr:
        """
        Parse ``obj[key] = value)``.
        """
        target = cls(Head.getitem, [symbol(obj), Symbol(key)])
        return cls(Head.assign, [target, symbol(value)])
    
    @classmethod
    def parse_setattr(cls, obj, key, value) -> Expr:
        """
        Parse ``obj.key = value``.
        """
        target = cls(Head.getattr, [symbol(obj), Symbol(key)])
        return cls(Head.assign, [target, symbol(value)])
    
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
    
    def issetattr(self) -> bool:
        if self.head == Head.assign:
            target = self.args[0]
            if isinstance(target, Expr) and target.head == Head.getattr:
                return True
        return False
    
    def issetitem(self) -> bool:
        if self.head == Head.assign:
            target = self.args[0]
            if isinstance(target, Expr) and target.head == Head.getitem:
                return True
        return False
    
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
    
    @overload
    def format(self, mapping: dict[Hashable, Symbol|Expr], inplace: bool = False) -> Expr:...
        
    @overload
    def format(self, mapping: Iterable[tuple[Any, Symbol|Expr]], inplace: bool = False) -> Expr:...
    
    def format(self, mapping: dict[Symbol, Symbol|Expr], inplace: bool = False) -> Expr:
        """
        Format expressions in the macro.
        
        Just like formatting method of string, this function can replace certain symbols to
        others. 
        
        Parameters
        ----------
        mapping : dict or iterable of tuples
            Mapping from objects to symbols or expressions. Keys will be converted to symbol.
            For instance, if you used ``arr``, a numpy.ndarray as an input of an macro-recordable
            function, that input will appear like 'var0x1...'. By calling ``format([(arr, "X")])``
            then 'var0x1...' will be substituted to 'X'.
        inplace : bool, default is False
            Expression will be overwritten if true.
            
        Returns
        -------
        Expression
            Formatted expression.
        """        
        if isinstance(mapping, dict):
            mapping = mapping.items()
        mapping = check_format_mapping(mapping)
            
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
    for comp in mapping_list:
        if len(comp) != 2:
            raise ValueError("Wrong style of mapping list.")
        k, v = comp
        if isinstance(v, Expr) and v.head in EXEC:
            raise ValueError("Cannot replace a symbol to a non-evaluable expression.")
        
        if isinstance(v, str) and not isinstance(k, Symbol):
            _dict[symbol(k)] = Symbol(v)
        else:
            _dict[symbol(k)] = symbol(v)
    return _dict


def make_symbol_str(obj: Any):
    # hexadecimals are easier to distinguish
    _id = id(obj)
    if obj is not None:
        Symbol._variables.add(_id)
    return f"var{hex(_id)}"

def symbol(obj: Any, constant: bool = True) -> Symbol | Expr:
    """
    Make a proper Symbol or Expr instance from any objects.
    
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
    Symbol or Expr
    """    
    if isinstance(obj, (Symbol, Expr)):
        return obj
        
    obj_type = type(obj)
    obj_id = id(obj)
    if not constant or obj_id in Symbol._variables:
        seq = make_symbol_str(obj)
        constant = False
    elif obj_type in Symbol._type_map:
        seq = Symbol._type_map[obj_type](obj)
    elif obj_type in Symbol._subclass_map:
        parent_type = Symbol._subclass_map[obj_type]
        seq = Symbol._type_map[parent_type](obj)
    elif isinstance(obj, tuple):
        seq = "(" + ", ".join(str(symbol(a)) for a in obj) + ")"
        if obj_type is not tuple:
            seq = obj_type.__name__ + seq
    elif isinstance(obj, list):
        seq = "[" + ", ".join(str(symbol(a)) for a in obj) + "]"
        if obj_type is not list:
            seq = f"{obj_type.__name__}({seq})"
    elif isinstance(obj, dict):
        seq = "{" + ", ".join(f"{symbol(k)}: {symbol(v)}" for k, v in obj.items()) + "}"
        if obj_type is not dict:
            seq = f"{obj_type.__name__}({seq})"
    elif isinstance(obj, set):
        seq = "{" + ", ".join(str(symbol(a)) for a in obj) + "}"
        if obj_type is not set:
            seq = f"{obj_type.__name__}({seq})"
    elif isinstance(obj, Number): # int, float, bool, ...
        seq = obj
    elif hasattr(obj, "__name__"):
        seq = obj.__name__
    else:
        for k, func in Symbol._type_map.items():
            if isinstance(obj, k):
                seq = func(obj)
                Symbol._subclass_map[obj_type] = k
                break
        else:
            seq = make_symbol_str(obj)
            constant = False
    
    if isinstance(seq, (Symbol, Expr)):
        # The output of register_type can be a Symbol or Expr
        sym = seq
    else:
        sym = Symbol(seq, obj_id)
        sym.constant = constant
    return sym