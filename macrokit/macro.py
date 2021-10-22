from __future__ import annotations
from contextlib import contextmanager
from importlib import import_module
from functools import partial, wraps
import inspect
from collections import UserList
from typing import Callable, Iterable, Iterator, Any, Union, overload, TypeVar
from types import ModuleType

from .expression import Head, Expr, symbol, make_symbol_str
from .symbol import Symbol

# types
MetaCallable = Union[Callable[[Expr], Expr], Callable[[Expr, Any], Expr]]
Recordable = Union[property, Callable, type, ModuleType]
_O = TypeVar("_O")

_NON_RECORDABLE = ("__new__", "__class__", "__repr__", "__getattribute__", "__dir__", 
                   "__init_subclass__", "__subclasshook__")

class Macro(UserList):
    """
    List with pretty output customized for macro.
    """    
    def __init__(self, iterable: Iterable = (), *, active: bool = True):
        super().__init__(iterable)
        self.active = active
        
    def append(self, expr: Expr):
        if not isinstance(expr, Expr):
            raise TypeError("Cannot append objects to Macro except for Expr objecs.")
        return super().append(expr)
    
    def insert(self, key: int, expr: Expr):
        if not isinstance(expr, Expr):
            raise TypeError("Cannot insert objects to Macro except for Expr objecs.")
        return super().insert(key, expr)
    
    def __str__(self) -> str:
        return "\n".join(map(str, self))
    
    @overload
    def __getitem__(self, key: int | str) -> Expr: ...

    @overload
    def __getitem__(self, key: slice) -> Macro[Expr]: ...
        
    def __getitem__(self, key):
        return super().__getitem__(key)
    
    def __iter__(self) -> Iterator[Expr]:
        return super().__iter__()
    
    def __repr__(self) -> str:
        return "\n".join(map(repr, self))
    
    def dump(self) -> str:
        return ",\n".join(expr.dump() for expr in self)
    
    def eval(self, _globals: dict[Symbol, Any] = {}, _locals: dict[Symbol, Any] = {}):
        for expr in self:
            out = expr.eval(_globals, _locals)
        return out
    
    @contextmanager
    def blocked(self):
        """
        Block macro recording within this context.
        """        
        was_active = self.active
        self.active = False
        yield
        self.active = was_active
    
    def format(self, mapping: dict[Symbol|Any, Symbol|Expr], inplace: bool = False) -> Macro:
        if isinstance(mapping, dict) and len(mapping) > 0:
            m = {}
            for k, v in mapping.items():
                if not isinstance(k, Symbol):
                    k = Symbol(k)
                m[k] = v
        else:
            raise TypeError("No format method matched the input.")
        
        return self.__class__(expr.format(m, inplace=inplace) for expr in self)
        
    def record(self, 
               obj: Recordable = None, 
               *, 
               returned_callback: MetaCallable = None
               ):
        def wrapper(_obj):
            if isinstance(_obj, property):
                return MProperty(_obj, macro=self, returned_callback=returned_callback)
            elif inspect.ismodule(_obj):
                return MModule(_obj, macro=self, returned_callback=returned_callback)
            elif inspect.isclass(_obj) and _obj is not type:
                return self.record_methods(_obj, returned_callback=returned_callback)
            elif callable(_obj) and not isinstance(_obj, MObject):
                return MFunction(_obj, macro=self, returned_callback=returned_callback)
            else:
                raise TypeError(f"Type {type(_obj)} is not macro recordable.")
        
        return wrapper if obj is None else wrapper(obj)

    
    def record_methods(self, cls: type, returned_callback: MetaCallable = None):
        _dict = {}
        for name, attr in inspect.getmembers(cls):
            if name in _NON_RECORDABLE:
                continue
            if callable(attr) or isinstance(attr, property):
                _dict[name] = self.record(attr, returned_callback=returned_callback)
            elif isinstance(attr, MacroMixin):
                update_namespace(attr)
        
        newcls = type(cls.__name__, (MacroMixin,) + cls.__bases__, _dict)
        
        return newcls
            
    def property(self, prop: Callable[[_O], Any]):
        """
        Make a macro-recordable property similar to ``@property``.

        Parameters
        ----------
        prop : Callable[[_O], Any]
            Property getter function.

        Returns
        -------
        MObject
        """
        return self.record(property(prop))

class MacroMixin:
    pass

def update_namespace(obj: MacroMixin, namespace: Symbol | Expr) -> None:
    new = Expr(Head.getattr, [namespace, symbol(obj)])
    for name, attr in inspect.getmembers(obj):
        if isinstance(attr, MObject):
            attr.namespace = namespace
        elif isinstance(attr, MacroMixin):
            update_namespace(attr, new)
    
class MObject:
    obj: Any
    def __init__(self, obj: Any, macro: Macro, returned_callback: MetaCallable = None, 
                 namespace: Symbol|Expr = None) -> None:
        self.obj = obj
        self.returned_callback = returned_callback or (lambda expr, out: expr)
        self.namespace = namespace
        _callback_nargs = len(inspect.signature(self.returned_callback).parameters)
        if _callback_nargs == 1:
            self.returned_callback = lambda expr, out: self.returned_callback(expr)
        
        elif _callback_nargs != 2:
            raise TypeError("returned_callback cannot take arguments more than two.")
        
        self.macro: Macro = macro
        for name in ["__name__", "__doc__"]:
            if hasattr(self.obj, name):
                setattr(self, name, getattr(self.obj, name))
        
        self._last_expr: Expr = None
    
    def to_namespace(self, obj) -> Symbol | Expr:
        if self.namespace is None:
            return symbol(obj)
        else:
            return Expr(Head.getattr, [self.namespace, symbol(obj)])
            
class MFunction(MObject):
    obj: Callable
    def __init__(self, function: Callable, macro: Macro, returned_callback: MetaCallable = None,
                 namespace: Symbol|Expr = None):
        super().__init__(function, macro, returned_callback, namespace)
        self._method_type = self._make_method_type()
            
    @property
    def __signature__(self):
        if hasattr(self.obj, "__signature__"):
            return self.obj.__signature__
        else:
            return inspect.signature(self.obj)
            
    def __call__(self, *args, **kwargs):
        with self.macro.blocked():
            out = self.obj(*args, **kwargs)
        if self.macro.active:
            expr = Expr.parse_call(self.to_namespace(self.obj), args, kwargs)
            line = self.returned_callback(expr, out)
            self.macro.append(line)
            self._last_expr = expr
        return out
    
    def _make_method_type(self):
        fname = self.obj.__name__
        if fname == "__init__":
            def make_expr(obj: _O, *args, **kwargs):
                return Expr.parse_init(self.to_namespace(obj), obj.__class__, args, kwargs)
        elif fname == "__call__":
            def make_expr(obj: _O, *args, **kwargs):
                return Expr.parse_call(self.to_namespace(obj), args, kwargs)
        elif fname == "__getitem__":
            def make_expr(obj: _O, *args):
                return Expr(Head.getitem, [self.to_namespace(obj), args[0]])
        elif fname == "__getattr__":
            def make_expr(obj: _O, *args):
                return Expr(Head.getattr, [self.to_namespace(obj), Symbol(args[0])])
        elif fname == "__setitem__":
            def make_expr(obj: _O, *args):
                expr = Expr(Head.setitem, [self.to_namespace(obj), args[0], args[1]])
                if (self._last_expr is not None
                    and self._last_expr.head == Head.setitem
                    and self._last_expr.args[1].args[0] == expr.args[1].args[0]
                    ):
                    self.macro.pop(-1)
                return expr
        elif fname == "__setattr__":
            def make_expr(obj: _O, *args):
                expr = Expr(Head.setattr, [self.to_namespace(obj), Symbol(args[0]), args[1]])
                if (self._last_expr is not None
                    and self._last_expr.head == Head.setattr
                    and self._last_expr.args[1].args[0] == expr.args[1].args[0]
                    ):
                    self.macro.pop(-1)
                return expr
        elif fname == "__delitem__":
            def make_expr(obj: _O, *args):
                return Expr(Head.delitem, [self.to_namespace(obj), args[0]])
        elif fname == "__delattr__":
            def make_expr(obj: _O, *args):
                return Expr(Head.delattr, [self.to_namespace(obj), Symbol(args[0])])
        else:
            def make_expr(obj: _O, *args, **kwargs):
                return Expr.parse_method(self.to_namespace(obj), self.obj, args, kwargs)
        
        @wraps(self.obj)
        def method(obj: _O, *args, **kwargs):
            # TODO: need namespace?
            with self.macro.blocked():
                out = self.obj(obj, *args, **kwargs)
            if self.macro.active:
                expr = make_expr(obj, *args, **kwargs)
                if expr is not None:
                    line = self.returned_callback(expr, out)
                self.macro.append(line)
                self._last_expr = expr
            return out

        return method
        
    def __get__(self, obj: _O, objtype=None):
        """
        Return a method type function that ``obj`` is bound. This subscriptor enables
        creating macro recordable instance methods.
        """
        if obj is None:
            return self.obj
        else:
            return partial(self._method_type, obj)
        
class MProperty(MObject):
    obj: property
    def __init__(self, property: property, macro: Macro, returned_callback: MetaCallable = None,
                 namespace: Symbol|Expr = None):
        super().__init__(property, macro, returned_callback, namespace)
        
    def setter(self, fset: Callable[[_O, Any], None]):
        key = Symbol(fset.__name__)
        @wraps(fset)
        def setter(obj, value):
            with self.macro.blocked():
                out = fset(obj, value)
            if self.macro.active:
                expr = Expr(Head.setattr, [self.to_namespace(obj), key, value])
                if (self._last_expr is not None
                    and self._last_expr.head == Head.setattr
                    and self._last_expr.args[1].args[0] == expr.args[1].args[0]
                    ):
                    self.macro.pop(-1)
                
                line = self.returned_callback(expr, out)
                self.macro.append(line)
                self._last_expr = expr
            return out
        
        return self.obj.setter(setter)
    
    def deleter(self, fdel: Callable[[_O], None]):
        key = Symbol(fdel.__name__)
        @wraps(fdel)
        def deleter(obj):
            with self.macro.blocked():
                out = fdel(obj)
            if self.macro.active:
                expr = Expr(Head.delattr, [self.to_namespace(obj), key])
                line = self.returned_callback(expr, out)
                self.macro.append(line)
                self._last_expr = expr
            return out
        
        return self.obj.deleter(deleter)
    
    def __get__(self, obj: _O, objtype=None):
        if obj is None:
            return self.obj
        
        else:
            return self.obj.__get__(obj)

class MModule(MObject):
    obj: ModuleType
    def __getattr__(self, key: str):
        try:
            func = getattr(self.obj, key)
        except AttributeError:
            try:
                submod = import_module("." + key, self.obj.__name__)
                mmod = MModule(submod, self.macro, self.returned_callback, self.to_namespace(self.obj))
            except ModuleNotFoundError:
                raise ValueError(f"No function or submodule named '{key}'.")
            else:
                setattr(self, key, mmod)
            return mmod
        
        @wraps(func)
        def mfunc(*args, **kwargs):
            with self.macro.blocked():
                out = func(*args, **kwargs)
            if self.macro.active:
                if isinstance(func, type):
                    cls = Expr(Head.getattr, [self.to_namespace(self.obj), Symbol(out.__class__.__name__)])
                    expr = Expr.parse_init(out, cls, args, kwargs)    
                else:
                    expr = Expr.parse_method(self.to_namespace(self.obj), func, args, kwargs)
                line = self.returned_callback(expr, out)
                self.macro.append(line)
                self._last_expr = expr
            return out
        return mfunc
    

def assign_callback(expr: Expr, out: Any):
    out_sym = symbol(out)
    if out_sym.valid:
        expr_assign = Expr("assign", [Symbol(make_symbol_str(out)), expr])
    else:
        expr_assign = Expr("assign", [out_sym, expr])
    return expr_assign