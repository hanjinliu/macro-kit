from __future__ import annotations
from contextlib import contextmanager
from importlib import import_module
from functools import partial, wraps
import inspect
from collections import UserList
from typing import Callable, Iterable, Iterator, Any, overload, TypeVar
from types import ModuleType

from .expression import Head, Expr
from .symbol import Symbol


MetaCallable = Callable[[Expr], Expr]
_O = TypeVar("_O")
_id = lambda e: e
    
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
        return ",\n".join(repr(expr) for expr in self)
    
    @contextmanager
    def blocked(self):
        was_active = self.active
        self.active = False
        yield
        self.active = was_active
    
    def format(self, mapping: dict[Symbol, Symbol|Expr], inplace: bool = False) -> Macro:
        return self.__class__(expr.format(mapping, inplace=inplace) for expr in self)
    
    def record(self, obj: Callable = None, *, returned_callback: MetaCallable = None):
        def wrapper(_obj):
            if isinstance(_obj, property):
                return MProperty(_obj, macro=self, returned_callback=returned_callback)
            elif inspect.ismodule(_obj):
                return MModule(_obj, macro=self, returned_callback=returned_callback)
            else:
                return MFunction(_obj, macro=self, returned_callback=returned_callback)
        
        return wrapper if obj is None else wrapper(obj)
    
    def property(self, prop: Callable[[_O], Any]):
        return self.record(property(prop))

class MObject:
    obj: Any
    def __init__(self, obj: Any, macro: Macro, returned_callback: MetaCallable = None) -> None:
        self.obj = obj
        self.returned_callback = returned_callback or _id
        self.macro: Macro = macro
        self.__name__ = self.obj.__name__
        self.__doc__ = self.obj.__doc__
        self._last_expr: Expr = None
        
class MFunction(MObject):
    obj: Callable
    def __init__(self, function: Callable, macro: Macro, returned_callback: MetaCallable = None):
        super().__init__(function, macro, returned_callback)
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
            expr = Expr.parse_call(self.obj, args, kwargs)
            line = self.returned_callback(expr)
            self.macro.append(line)
            self._last_expr = expr
        return out
    
    def _make_method_type(self):
        fname = self.obj.__name__
        if fname == "__init__":
            def make_expr(obj: _O, *args, **kwargs):
                return Expr.parse_init(obj, obj.__class__, args, kwargs)
        elif fname == "__call__":
            def make_expr(obj: _O, *args, **kwargs):
                return Expr.parse_call(obj, args, kwargs)
        elif fname == "__getitem__":
            def make_expr(obj: _O, *args):
                return Expr(Head.getitem, [obj, args[0]])
        elif fname == "__getattr__":
            def make_expr(obj: _O, *args):
                return Expr(Head.getattr, [obj, Symbol(args[0])])
        elif fname == "__setitem__":
            def make_expr(obj: _O, *args):
                expr = Expr(Head.setitem, [obj, args[0], args[1]])
                if (self._last_expr is not None
                    and self._last_expr.head == Head.setattr
                    and self._last_expr.args[1].args[0] == expr.args[1].args[0]
                    ):
                    self.macro.pop(-1)
                return expr
        elif fname == "__setattr__":
            def make_expr(obj: _O, *args):
                expr = Expr(Head.setattr, [obj, args[0], Symbol(args[1])])
                if (self._last_expr is not None
                    and self._last_expr.head == Head.setattr
                    and self._last_expr.args[1].args[0] == expr.args[1].args[0]
                    ):
                    self.macro.pop(-1)
                return expr
        elif fname == "__getitem__":
            def make_expr(obj: _O, *args):
                return Expr(Head.delitem, [obj, args[0]])
        elif fname == "__getattr__":
            def make_expr(obj: _O, *args):
                return Expr(Head.delattr, [obj, Symbol(args[0])])
        else:
            def make_expr(obj: _O, *args, **kwargs):
                return Expr.parse_method(obj, self.obj, args, kwargs)
        
        @wraps(self.obj)
        def method(obj: _O, *args, **kwargs):
            with self.macro.blocked():
                out = self.obj(obj, *args, **kwargs)
            if self.macro.active:
                expr = make_expr(obj, *args, **kwargs)
                if expr is not None:
                    line = self.returned_callback(expr)
                self.macro.append(line)
                self._last_expr = expr
            return out

        return method
        
    def __get__(self, obj: _O, objtype=None):
        if obj is None:
            return self.obj
        else:
            return partial(self._method_type, obj)
        
class MProperty(MObject):
    obj: property
    def __init__(self, property: property, macro: Macro, returned_callback: MetaCallable = None):
        super().__init__(property, macro, returned_callback)
        
    def setter(self, fset: Callable[[_O, Any], None]):
        key = Symbol(fset.__name__)
        @wraps(fset)
        def setter(obj, value):
            with self.macro.blocked():
                out = fset(obj, value)
            if self.macro.active:
                expr = Expr(Head.setattr, [obj, key, value])
                if (self._last_expr is not None
                    and self._last_expr.head == Head.setattr
                    and self._last_expr.args[1].args[0] == expr.args[1].args[0]
                    ):
                    self.macro.pop(-1)
                
                line = self.returned_callback(expr)
                self.macro.append(line)
                self._last_expr = expr
            return out
        
        return self.obj.setter(setter)
    
    def __get__(self, obj: _O, objtype=None):
        if obj is None:
            return self.obj
        
        else:
            return self.obj.__get__(obj)

class MClass(MObject):
    obj: type
    ...
    
class MModule(MObject):
    obj: ModuleType
    def __getattr__(self, key: str):
        try:
            func = getattr(self.obj, key)
        except AttributeError:
            try:
                submod = import_module("." + key, self.obj.__name__)
                mmod = MModule(submod, self.macro, self.returned_callback)
            except ModuleNotFoundError:
                raise ValueError(f"Could not find any function or submodule named '{key}'.")
            else:
                setattr(self, key, mmod)
            return mmod
        
        @wraps(func)
        def mfunc(*args, **kwargs):
            with self.macro.blocked():
                out = func(*args, **kwargs)
            if self.macro.active:
                if isinstance(func, type):
                    cls = Expr(Head.getattr, [self.obj, Symbol(out.__class__.__name__)])
                    expr = Expr.parse_init(out, cls, args, kwargs)    
                else:
                    expr = Expr.parse_method(self.obj, func, args, kwargs)
                line = self.returned_callback(expr)
                self.macro.append(line)
                self._last_expr = expr
            return out
        return mfunc