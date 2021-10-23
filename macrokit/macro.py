from __future__ import annotations
from contextlib import contextmanager
from importlib import import_module
from functools import partial, wraps
import inspect
from collections import UserList
from typing import Callable, Iterable, Iterator, Any, Union, overload, TypeVar, Hashable
from types import ModuleType

from .expression import Head, Expr, symbol, EXEC, check_format_mapping
from .symbol import Symbol

# types
MetaCallable = Union[Callable[[Expr], Expr], Callable[[Expr, Any], Expr]]
Recordable = Union[property, Callable, type, ModuleType]
_property = property
_O = TypeVar("_O")

_NON_RECORDABLE = ("__new__", "__class__", "__repr__", "__getattribute__", "__dir__", 
                   "__init_subclass__", "__subclasshook__")

_INHERITABLE = ("__module__", "__name__", "__qualname__", "__doc__", "__annotations__")

class Macro(UserList):
    """
    List of expressions. A Macro object corresponds to a Python script.
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
        """
        Make a meta-code in Julian way.

        Returns
        -------
        str
            Meta-code.
        """        
        return ",\n".join(expr.dump() for expr in self)
    
    def eval(self, _globals: dict[Symbol, Any] = {}, _locals: dict[Symbol, Any] = {}) -> Any:
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

        Returns
        -------
        Any
            The last returned value.
        """        
        namespace = _globals.copy()
        names = dict()
        with self.blocked():
            for expr in self:
                namespace.update(names)
                names = {}
                out = expr.eval(namespace, names)
            namespace.update(names)
        _locals.update({Symbol(k): v for k, v in namespace.items()})
        return out
    
    @contextmanager
    def blocked(self):
        """
        Block macro recording within this context.
        """        
        was_active = self.active
        self.active = False
        try:
            yield
        finally:
            self.active = was_active
    
    @overload
    def format(self, mapping: dict[Hashable, Symbol|Expr], inplace: bool = False) -> Macro:...
        
    @overload
    def format(self, mapping: Iterable[tuple[Any, Symbol|Expr]], inplace: bool = False) -> Macro:...
    
    def format(self, mapping, inplace: bool = False) -> Macro:
        """
        Format expressions in the macro.
        
        Just like format method of string, this function can replace certain symbols to
        others. 

        Parameters
        ----------
        mapping : dict or iterable of tuples
            Mapping from objects to symbols or expressions. Keys will be converted to symbol.
            For instance, if you used ``arr``, a numpy.ndarray as an input of an macro-recordable
            function, that input will appear like 'var0x1...'. By calling ``format([(arr, "X")])``
            then 'var0x1...' will be substituted to 'X'.
        inplace : bool, default is False
            Macro will be overwritten if true.

        Returns
        -------
        Macro
            Formatted macro.
        """        
        if isinstance(mapping, dict):
            mapping = mapping.items()
            
        m = check_format_mapping(mapping)
        cls = self.__class__
        
        if inplace:
            return cls(expr._unsafe_format(m) for expr in self)
        else:
            return cls(expr.copy()._unsafe_format(m) for expr in self)
    
    @overload
    def record(self, obj: _property, *, returned_callback: MetaCallable = None) -> mProperty: ...
    
    @overload
    def record(self, obj: ModuleType, *, returned_callback: MetaCallable = None) -> mModule: ...
    
    @overload
    def record(self, obj: type, *, returned_callback: MetaCallable = None) -> type[MacroMixin]: ...
    
    @overload
    def record(self, obj: Callable, *, returned_callback: MetaCallable = None) -> mFunction: ...
    
    @overload
    def record(self, *, returned_callback: MetaCallable = None) -> Callable[[Recordable], mObject|MacroMixin]: ...
    
    def record(self, obj = None, *, returned_callback = None):
        """
        A wrapper that convert an object to a macro-recordable one.

        Parameters
        ----------
        obj : property, module, type or callable, optional
            Base object.
        returned_callback : callable, optional
            A function that will called after new expression is appended. Must take an expression
            or an expression with the last returned value as inputs.
        """        
        def wrapper(_obj):
            if isinstance(_obj, property):
                return mProperty(_obj, macro=self, returned_callback=returned_callback)
            elif inspect.ismodule(_obj):
                return mModule(_obj, macro=self, returned_callback=returned_callback)
            elif inspect.isclass(_obj) and _obj is not type:
                return self.record_methods(_obj, returned_callback=returned_callback)
            elif callable(_obj) and not isinstance(_obj, mObject):
                return mFunction(_obj, macro=self, returned_callback=returned_callback)
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
                _dict[name] = attr
        
        newcls = type(cls.__name__, (MacroMixin, cls), _dict)
        
        return newcls
            
    def property(self, prop: Callable[[_O], Any]) -> mProperty:
        """
        Make a macro-recordable property similar to ``@property``.

        Parameters
        ----------
        prop : callable
            Property getter function.

        Returns
        -------
        mProperty
            Macro-recordable property object.
        """
        return self.record(property(prop))
    
    def call_builtin(self, func: Callable, *args, **kwargs):
        """
        Call Python builtin function in macro recording mode.

        Parameters
        ----------
        func : Callable
            Builtin function.
        """        
        with self.blocked():
            out = func(*args, **kwargs)
        if self.active:
            expr = Expr.parse_call(func, args, kwargs)
            line = _assign_value_callback(expr, out)
            self.append(line)
            self._last_setval = None
        return out

class MacroMixin:
    pass

def update_namespace(obj: MacroMixin, namespace: Symbol | Expr) -> None:
    new = Expr(Head.getattr, [namespace, symbol(obj)])
    for name, attr in inspect.getmembers(obj):
        if isinstance(attr, mObject):
            attr.namespace = namespace
        elif isinstance(attr, MacroMixin):
            update_namespace(attr, new)
    
class mObject:
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
        for name in _INHERITABLE:
            if hasattr(self.obj, name):
                setattr(self, name, getattr(self.obj, name))
        
        self._last_setval: tuple[Head, Any] = None
    
    def to_namespace(self, obj) -> Symbol | Expr:
        if self.namespace is None:
            return symbol(obj)
        else:
            return Expr(Head.getattr, [self.namespace, symbol(obj)])

Symbol.register_type(mObject, lambda o: symbol(o.obj))

class mFunction(mObject):
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
            expr = _assign_value_callback(expr, out)
            line = self.returned_callback(expr, out)
            self.macro.append(line)
            self._last_setval = None
        return out
    
    def _make_method_type(self):
        fname = getattr(self.obj, "__name__", None)
        if fname == "__init__":
            def make_expr(obj: _O, out, *args, **kwargs):
                expr = Expr.parse_init(self.to_namespace(obj), obj.__class__, args, kwargs)
                self._last_setval = None
                return expr
        elif fname == "__call__":
            def make_expr(obj: _O, out, *args, **kwargs):
                expr = Expr.parse_call(self.to_namespace(obj), args, kwargs)
                expr = _assign_value_callback(expr, out)
                self._last_setval = None
                return expr
        elif fname == "__getitem__":
            def make_expr(obj: _O, out, *args):
                expr = Expr(Head.getitem, [self.to_namespace(obj), args[0]])
                expr = _assign_value_callback(expr, out)
                self._last_setval = None
                return expr
        elif fname == "__getattr__":
            def make_expr(obj: _O, out, *args):
                expr = Expr(Head.getattr, [self.to_namespace(obj), Symbol(args[0])])
                expr = _assign_value_callback(expr, out)
                self._last_setval = None
                return expr
        elif fname == "__setitem__":
            def make_expr(obj: _O, out, *args):
                expr = Expr(Head.setitem, [self.to_namespace(obj), args[0], args[1]])
                if self._last_setval == (Head.setitem, args[0]):
                    self.macro.pop(-1)
                else:
                    self._last_setval = (Head.setitem, args[0])
                return expr
        elif fname == "__setattr__":
            def make_expr(obj: _O, out, *args):
                expr = Expr(Head.setattr, [self.to_namespace(obj), Symbol(args[0]), args[1]])
                if self._last_setval == (Head.setattr, args[0]):
                    self.macro.pop(-1)
                else:
                    self._last_setval = (Head.setattr, args[0])
                return expr
        elif fname == "__delitem__":
            def make_expr(obj: _O, out, *args):
                expr = Expr(Head.delitem, [self.to_namespace(obj), args[0]])
                self._last_setval = None
                return expr
        elif fname == "__delattr__":
            def make_expr(obj: _O, out, *args):
                expr = Expr(Head.delattr, [self.to_namespace(obj), Symbol(args[0])])
                self._last_setval = None
                return expr
        else:
            def make_expr(obj: _O, out, *args, **kwargs):
                expr = Expr.parse_method(self.to_namespace(obj), self.obj, args, kwargs)
                expr = _assign_value_callback(expr, out)
                self._last_setval = None
                return expr
        
        @wraps(self.obj)
        def method(obj: _O, *args, **kwargs):
            with self.macro.blocked():
                out = self.obj(obj, *args, **kwargs)
            if self.macro.active:
                expr = make_expr(obj, out, *args, **kwargs)
                if expr is not None:
                    line = self.returned_callback(expr, out)
                self.macro.append(line)
            return out

        return method
        
    def __get__(self, obj: _O, objtype=None):
        """
        Return a method type function that ``obj`` is bound. This discriptor enables
        creating macro recordable instance methods.
        """
        if obj is None:
            return self.obj
        else:
            return partial(self._method_type, obj)
        
class mProperty(mObject, property):
    obj: property
    def __init__(self, prop: property, macro: Macro, returned_callback: MetaCallable = None,
                 namespace: Symbol|Expr = None):
        super().__init__(prop, macro, returned_callback, namespace)
        self.obj = self.getter(prop.fget)
    
    def getter(self, fget: Callable[[_O], None]):
        key = Symbol(fget.__name__)
        @wraps(fget)
        def getter(obj):
            with self.macro.blocked():
                out = fget(obj)
            if self.macro.active:
                expr = Expr(Head.getattr, [self.to_namespace(obj), key])
                expr = _assign_value_callback(expr, out)
                line = self.returned_callback(expr, out)
                self.macro.append(line)
                self._last_setval = None
            return out
        return self.obj.getter(getter)
        
    def setter(self, fset: Callable[[_O, Any], None]):
        key = Symbol(fset.__name__)
        @wraps(fset)
        def setter(obj, value):
            with self.macro.blocked():
                out = fset(obj, value)
            if self.macro.active:
                expr = Expr(Head.setattr, [self.to_namespace(obj), key, value])
                if self._last_setval == (Head.setattr, key.name):
                    self.macro.pop(-1)
                else:
                    self._last_setval = (Head.setattr, key.name)
                
                line = self.returned_callback(expr, out)
                self.macro.append(line)
                    
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
                self._last_setval = None
            return out
        
        return self.obj.deleter(deleter)
    
    def __get__(self, obj: _O, type=None):
        if obj is None:
            return self.obj
        else:
            return self.obj.__get__(obj)
        
    def __set__(self, obj, value) -> None:
        return self.obj.__set__(obj, value)
    
    def __delete__(self, obj) -> None:
        return self.obj.__delete__(obj)
    
class mModule(mObject):
    obj: ModuleType
    def __getattr__(self, key: str):
        try:
            attr = getattr(self.obj, key)
            _type = "module" if inspect.ismodule(attr) else "function"
        except AttributeError:
            try:
                attr = import_module("." + key, self.obj.__name__)
                _type = "module"
            except ModuleNotFoundError:
                raise ValueError(f"No function or submodule named '{key}'.")
        
        if _type == "module":
            mmod = mModule(attr, self.macro, self.returned_callback, self.to_namespace(self.obj))
            setattr(self, key, mmod)
            return mmod
            
        @wraps(attr)
        def mfunc(*args, **kwargs):
            with self.macro.blocked():
                out = attr(*args, **kwargs)
            if self.macro.active:
                if isinstance(attr, type):
                    cls = Expr(Head.getattr, [self.to_namespace(self.obj), Symbol(out.__class__.__name__)])
                    expr = Expr.parse_init(out, cls, args, kwargs)    
                else:
                    expr = Expr.parse_method(self.to_namespace(self.obj), attr, args, kwargs)
                    expr = _assign_value_callback(expr, out)
                line = self.returned_callback(expr, out)
                self.macro.append(line)
                self._last_setval = None
            return out
        return mfunc
    

def _assign_value_callback(expr: Expr, out: Any):
    out_sym = symbol(out, constant=False)
    if expr.head not in EXEC:
        expr_assign = Expr(Head.assign, [out_sym, expr])
    else:
        expr_assign = expr        
    return expr_assign