from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from collections.abc import MutableSequence
from importlib import import_module
from functools import partial, wraps
import inspect
from typing import Callable, Iterable, Iterator, Any, TypedDict, Union, overload, TypeVar, NamedTuple
from types import ModuleType

from .ast import parse
from .expression import Head, Expr, symbol, EXEC
from .symbol import Symbol

_NON_RECORDABLE = ("__new__", "__class__", "__repr__", "__getattribute__", "__dir__", 
                   "__init_subclass__", "__subclasshook__", "__class_getitem__")

_INHERITABLE = ("__module__", "__name__", "__qualname__", "__doc__", "__annotations__")

BINOP_MAP = {
    "__add__": Symbol("+"),
    "__sub__": Symbol("-"),
    "__mul__": Symbol("*"),
    "__div__": Symbol("/"),
    "__eq__": Symbol("=="),
    "__neq__": Symbol("!="),
    "__gt__": Symbol(">"),
    "__ge__": Symbol(">="),
    "__lt__": Symbol("<"),
    "__le__": Symbol("<="),
    "__pow__": Symbol("**"),
    "__matmul__": Symbol("@"),
    "__floordiv__": Symbol("//"),
    "__and__": Symbol("&"),
    "__or__": Symbol("|"),
    "__xor__": Symbol("^")
}

class MacroFlags(NamedTuple):
    """
    This immutable struct gives the infomation of what kind of expressions will be
    recorded in a macro instance
    """    
    Get: bool = True
    Set: bool = True
    Delete: bool = True
    Return: bool = True

# types

MetaCallable = Union[Callable[[Expr], Expr], Callable[[Expr, Any], Expr]]
Recordable = Union[property, Callable, type, ModuleType]
_property = property
_O = TypeVar("_O")
_Class = TypeVar("_Class")

class MacroFlagOptions(TypedDict):
    Get: bool
    Set: bool
    Delete: bool
    Return: bool

class Macro(Expr, MutableSequence[Expr]):
    """
    A special form of an expression with header "block".
    This class behaves in a list-like way and specifically useful for macro recording.
    """    
    _FLAG_MAP = {
        "__getitem__": "Get",
        "__getattr__": "Get",
        "__setitem__": "Set",
        "__setattr__": "Set",
        "__delitem__": "Delete",
        "__delattr__": "Delete"
    }
    
    def __init__(self, args: Iterable[Expr] = (), *, flags: MacroFlagOptions = {}):
        super().__init__(head=Head.block, args=args)
        self.active = True
        self._callbacks = []
        self._flags = MacroFlags(**flags)
    
    @property
    def callbacks(self):
        return self._callbacks
    
    @property
    def flags(self):
        return self._flags
        
    def insert(self, key: int, expr: Expr):
        if isinstance(expr, str):
            expr = parse(expr)
        elif not isinstance(expr, Expr):
            raise TypeError("Cannot insert objects to Macro except for Expr objects.")
        
        self.args.insert(key, expr)
        for callback in self._callbacks:
            callback(expr)
    
    @overload
    def __getitem__(self, key: int | str) -> Expr: ...

    @overload
    def __getitem__(self, key: slice) -> Macro[Expr]: ...
        
    def __getitem__(self, key):
        return self._args[key]
    
    def __setitem__(self, key: int, value: Expr):
        self._args[key] = value
    
    def __delitem__(self, key: int):
        del self._args[key]
    
    def __len__(self) -> int:
        return len(self._args)
    
    def __iter__(self) -> Iterator[Expr]:
        return iter(self._args)
    
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
    def record(self, obj: _property, *, returned_callback: MetaCallable = None) -> mProperty: ...
    
    @overload
    def record(self, obj: ModuleType, *, returned_callback: MetaCallable = None) -> mModule: ...
    
    @overload
    def record(self, obj: _Class, *, returned_callback: MetaCallable = None) -> _Class: ...
    
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
        kwargs = dict(macro=self, 
                      returned_callback=returned_callback, 
                      record_returned=self._flags.Return
                      )
        def wrapper(_obj):
            if isinstance(_obj, property):
                return mProperty(_obj, macro=self, returned_callback=returned_callback, 
                                 flags=self.flags._asdict())
            elif isinstance(_obj, ModuleType):
                return mModule(_obj, **kwargs)
            elif isinstance(_obj, type) and _obj is not type:
                return self._record_methods(_obj, returned_callback=returned_callback)
            elif isinstance(_obj, Callable) and not isinstance(_obj, mObject):
                return mFunction(_obj, **kwargs)
            elif isinstance(_obj, mObject):
                return type(_obj)(_obj.obj, **kwargs)
            else:
                raise TypeError(f"Type {type(_obj)} is not macro recordable.")
        
        return wrapper if obj is None else wrapper(obj)

    
    def _record_methods(self, cls: type, returned_callback: MetaCallable = None):
        _dict: dict[str, mObject] = {}
        for name, attr in inspect.getmembers(cls):
            if name in _NON_RECORDABLE:
                continue
            if isinstance(attr, Callable):
                key = self._FLAG_MAP.get(name, "None")
                if not getattr(self._flags, key, True):
                    continue
                _dict[name] = self.record(attr, returned_callback=returned_callback)
            if isinstance(attr, property):
                _dict[name] = self.record(attr, returned_callback=returned_callback)
            elif isinstance(attr, MacroMixin):
                update_namespace(attr)
                _dict[name] = attr
            else:
                try:
                    _dict[name] = self.record(attr, returned_callback=returned_callback)
                except TypeError:
                    pass
        
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
    
    def optimize(self, inplace=False) -> Macro:
        """
        Optimize macro readability by deleting unused variables.

        Parameters
        ----------
        inplace : bool, default is False
            If true, macro will be updated by the optimized one.

        Returns
        -------
        Macro
            Optimized macro.
        """        
        if not inplace:
            self = deepcopy(self)
        expr_map: list[tuple[Symbol, int]] = []
        need = set()
        for i, expr in enumerate(self):
            expr: Expr|Symbol
            if expr.head == Head.assign:
                # TODO: a, b = func(...) don't work
                expr_map.append((expr.args[0], i))
                args = expr.args[1:]
            else:
                args = expr.args
                
            for arg in args:
                if isinstance(arg, Expr):
                    need |= set(a for a in arg.iter_args() if (not a.constant))
                elif not arg.constant:
                    need.add(arg)
                    
        for sym, i in expr_map:
            if sym not in need:
                self[i] = self[i].args[1]
                
        return self
        
        
    def call_function(self, func: Callable, *args, **kwargs):
        """
        Call function in macro recording mode.

        Parameters
        ----------
        func : Callable
            Function you want to call.
        """        
        with self.blocked():
            out = func(*args, **kwargs)
        if self.active:
            expr = Expr.parse_call(func, args, kwargs)
            if self._flags.Return:
                expr = _assign_value(expr, out)
            self.append(expr)
            self._last_setval = None
        return out

class MacroMixin:
    """
    Any class objects that are wrapped with ``Macro.record`` will inherit this class.
    """

def update_namespace(obj: MacroMixin, namespace: Symbol | Expr) -> None:
    """
    Update the namespace of a ``MacroMixin`` object.

    Parameters
    ----------
    obj : MacroMixin
        Input object.
    namespace : Symbol or Expr
        Namespace.
    """
    new = Expr(Head.getattr, [namespace, symbol(obj)])
    for name, attr in inspect.getmembers(obj):
        if isinstance(attr, mObject):
            attr.namespace = namespace
        elif isinstance(attr, MacroMixin):
            update_namespace(attr, new)
    
class mObject:
    obj: Any
    def __init__(self, obj: Any, macro: Macro, returned_callback: MetaCallable = None, 
                 namespace: Symbol|Expr = None, record_returned: bool = True) -> None:
        self.obj = obj
        
        if returned_callback is not None:
            _callback_nargs = len(inspect.signature(returned_callback).parameters)
            if _callback_nargs == 1:
                returned_callback = lambda expr, out: returned_callback(expr)
            
            elif _callback_nargs != 2:
                raise TypeError("returned_callback cannot take arguments more than two.")
            
            if record_returned:
                self.returned_callback = lambda expr, out: \
                    returned_callback(_assign_value(expr, out), out)
            else:
                self.returned_callback = returned_callback
        
        else:
            if record_returned:
                self.returned_callback = _assign_value
            else:
                self.returned_callback = lambda expr, out: expr
            
        self.namespace = namespace
        
        self.macro = macro
        for name in _INHERITABLE:
            if hasattr(self.obj, name):
                setattr(self, name, getattr(self.obj, name))
        
        self._last_setval: Expr = None
    
    def to_namespace(self, obj: Any) -> Symbol | Expr:
        """
        Return the expression of ``obj`` in the correct name space.
        """        
        sym = symbol(obj)
        if self.namespace is None:
            return sym
        else:
            return Expr(Head.getattr, [self.namespace, sym])

Symbol.register_type(mObject, lambda o: symbol(o.obj))

class mFunction(mObject):
    obj: Callable
    def __init__(self, function: Callable, macro: Macro, returned_callback: MetaCallable = None,
                 namespace: Symbol|Expr = None, record_returned: bool = True):
        self.isclassmethod = isclassmethod(function)
        super().__init__(function, macro, returned_callback, namespace, record_returned)
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
            self._last_setval = None
        return out
    
    def _make_method_type(self):
        # TODO: wrapper_descriptor is not recorded correctly
        fname = getattr(self.obj, "__name__", None)
        if fname == "__init__":
            def make_expr(obj: _O, out, *args, **kwargs):
                expr = Expr.parse_init(self.to_namespace(obj), obj.__class__, args, kwargs)
                self._last_setval = None
                return expr
        elif fname == "__call__":
            def make_expr(obj: _O, out, *args, **kwargs):
                expr = Expr.parse_call(self.to_namespace(obj), args, kwargs)
                self._last_setval = None
                return expr
        elif fname == "__getitem__":
            def make_expr(obj: _O, out, *args):
                expr = Expr(Head.getitem, [self.to_namespace(obj), args[0]])
                self._last_setval = None
                return expr
        elif fname == "__getattr__":
            def make_expr(obj: _O, out, *args):
                expr = Expr(Head.getattr, [self.to_namespace(obj), Symbol(args[0])])
                self._last_setval = None
                return expr
        elif fname == "__setitem__":
            def make_expr(obj: _O, out, *args):
                target = Expr(Head.getitem, [self.to_namespace(obj), args[0]])
                expr = Expr(Head.assign, [target, args[1]])
                if self._last_setval == target:
                    self.macro.pop(-1)
                else:
                    self._last_setval = target
                return expr
        elif fname == "__setattr__":
            def make_expr(obj: _O, out, *args):
                target = Expr(Head.getattr, [self.to_namespace(obj), Symbol(args[0])])
                expr = Expr(Head.assign, [target, args[1]])
                if self._last_setval == target:
                    self.macro.pop(-1)
                else:
                    self._last_setval = target
                return expr
        elif fname == "__delitem__":
            def make_expr(obj: _O, out, *args):
                target = Expr(Head.getitem, [self.to_namespace(obj), args[0]])
                expr = Expr(Head.del_, [target])
                self._last_setval = None
                return expr
        elif fname == "__delattr__":
            def make_expr(obj: _O, out, *args):
                target = Expr(Head.getattr, [self.to_namespace(obj), args[0]])
                expr = Expr(Head.del_, [target])
                self._last_setval = None
                return expr
        elif fname in BINOP_MAP.keys():
            op = BINOP_MAP[fname]
            def make_expr(obj: _O, out, *args):
                expr = Expr(Head.binop, [op, self.to_namespace(obj), args[0]])
                self._last_setval = None
                return expr
        else:
            def make_expr(obj: _O, out, *args, **kwargs):
                expr = Expr.parse_method(self.to_namespace(obj), self.obj, args, kwargs)
                self._last_setval = None
                return expr
        
        @wraps(self.obj)
        def method(obj: _O, *args, **kwargs):
            with self.macro.blocked():
                if self.isclassmethod:
                    out = self.obj(*args, **kwargs)
                else:
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
                 namespace: Symbol|Expr = None, flags: MacroFlagOptions = {}):
        self._flags = MacroFlags(**flags)
        super().__init__(prop, macro, returned_callback, namespace, record_returned=self._flags.Return)
        self.obj = self.getter(prop.fget)
    
    def getter(self, fget: Callable[[_O], None]) -> property:
        if not self._flags.Get:
            return self.obj.getter(fget)
        key = Symbol(fget.__name__)
        @wraps(fget)
        def getter(obj):
            with self.macro.blocked():
                out = fget(obj)
            if self.macro.active:
                expr = Expr(Head.getattr, [self.to_namespace(obj), key])
                expr = self.returned_callback(expr, out)
                self.macro.append(expr)
                self._last_setval = None
            return out
        return self.obj.getter(getter)
        
    def setter(self, fset: Callable[[_O, Any], None]) -> property:
        if not self._flags.Set:
            return self.obj.setter(fset)
        key = Symbol(fset.__name__)
        @wraps(fset)
        def setter(obj, value):
            with self.macro.blocked():
                out = fset(obj, value)
            if self.macro.active:
                target = Expr(Head.getattr, [self.to_namespace(obj), key])
                expr = Expr(Head.assign, [target, value])
                if self._last_setval == target:
                    self.macro.pop(-1)
                else:
                    self._last_setval = target
                
                line = self.returned_callback(expr, out)
                self.macro.append(line)
                    
            return out
        
        return self.obj.setter(setter)
    
    def deleter(self, fdel: Callable[[_O], None]) -> property:
        if not self._flags.Delete:
            return self.obj.deleter(fdel)
        key = Symbol(fdel.__name__)
        @wraps(fdel)
        def deleter(obj):
            with self.macro.blocked():
                out = fdel(obj)
            if self.macro.active:
                target = Expr(Head.getattr, [self.to_namespace(obj), key])
                expr = Expr(Head.del_, [target])
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
                expr = self.returned_callback(expr, out)
                self.macro.append(expr)
                self._last_setval = None
            return out
        return mfunc
    

def _assign_value(expr: Expr, out: Any):
    out_sym = symbol(out, constant=False)
    if expr.head not in EXEC:
        expr_assign = Expr(Head.assign, [out_sym, expr])
    else:
        expr_assign = expr        
    return expr_assign

def isclassmethod(method: Any):
    """
    Check if method is a class method.
    
    https://stackoverflow.com/questions/19227724/check-if-a-function-uses-classmethod
    """    
    bound_to = getattr(method, "__self__", None)
    if not isinstance(bound_to, type):
        # must be bound to a class
        return False
    name = method.__name__
    for cls in bound_to.__mro__:
        descriptor = vars(cls).get(name)
        if descriptor is not None:
            return isinstance(descriptor, classmethod)
    return False