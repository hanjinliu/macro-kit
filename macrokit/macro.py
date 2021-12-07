from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from collections.abc import MutableSequence
from importlib import import_module
from functools import partial, wraps
import inspect
from typing import Callable, Iterable, Iterator, Any, TypedDict, Union, overload, TypeVar, NamedTuple
from types import ModuleType

from .ast import parse, Operator
from .expression import Head, Expr, symbol, EXEC
from .symbol import Symbol

_NON_RECORDABLE = ("__new__", "__class__", "__repr__", "__getattribute__", "__dir__", 
                   "__init_subclass__", "__subclasshook__", "__class_getitem__")

_INHERITABLE = ("__module__", "__name__", "__qualname__", "__doc__", "__annotations__")


BINOP_MAP = {
    "__add__": Symbol("+", type=Operator),
    "__sub__": Symbol("-", type=Operator),
    "__mul__": Symbol("*", type=Operator),
    "__div__": Symbol("/", type=Operator),
    "__mod__": Symbol("%", type=Operator),
    "__eq__": Symbol("==", type=Operator),
    "__neq__": Symbol("!=", type=Operator),
    "__gt__": Symbol(">", type=Operator),
    "__ge__": Symbol(">=", type=Operator),
    "__lt__": Symbol("<", type=Operator),
    "__le__": Symbol("<=", type=Operator),
    "__pow__": Symbol("**", type=Operator),
    "__matmul__": Symbol("@", type=Operator),
    "__floordiv__": Symbol("//", type=Operator),
    "__and__": Symbol("&", type=Operator),
    "__or__": Symbol("|", type=Operator),
    "__xor__": Symbol("^", type=Operator)
}

BUILTIN_MAP = {
    "__hash__": Symbol("hash", type=Callable),
    "__len__": Symbol("len", type=Callable),
    "__str__": Symbol("str", type=Callable),
    "__repr__": Symbol("repr", type=Callable),
    "__bool__": Symbol("bool", type=Callable),
    "__float__": Symbol("float", type=Callable),
    "__int__": Symbol("int", type=Callable),
    "__format__": Symbol("format", type=Callable),
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
_O = TypeVar("_O")
_Class = TypeVar("_Class")

class MacroFlagOptions(TypedDict):
    Get: bool
    Set: bool
    Delete: bool
    Return: bool

# classes

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
        self._last_setval: Expr = None
    
    @property
    def callbacks(self):
        """
        Callback functions when a new macro is recorded.
        """        
        return self._callbacks
    
    @property
    def flags(self):
        """
        Flags that determine what kind of expression will be recorded.
        """        
        return self._flags
        
    def insert(self, key: int, expr: Expr | str):
        """
        Insert expressiong to macro.

        Parameters
        ----------
        key : int
            Position to insert.
        expr : Expr
            Expression. If a string is given, it will be parsed to Expr object.
        """        
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
    def record(self, obj: property, *, returned_callback: MetaCallable = None) -> mProperty: ...
    
    @overload
    def record(self, obj: ModuleType, *, returned_callback: MetaCallable = None) -> mModule: ...
    
    @overload
    def record(self, obj: _Class, *, returned_callback: MetaCallable = None) -> _Class: ...
    
    @overload
    def record(self, obj: Callable, *, returned_callback: MetaCallable = None) -> mFunction: ...
    
    @overload
    def record(self, *, returned_callback: MetaCallable = None) -> Callable[[Recordable], mObject | MacroMixin]: ...
    
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
        # TODO: record called twice
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
                if not isclassmethod(_obj):
                    return mFunction(_obj, **kwargs)
                else:
                    return mClassMethod(_obj, **kwargs)
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
                update_namespace(attr, Symbol(cls))
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
    
    def classmethod(self, method: Callable):
        """
        Make a macro-recordable property similar to ``@classmethod``.

        Parameters
        ----------
        method : callable
            Function that will be a classmethod.

        Returns
        -------
        mClassMethod
            Macro-recordable classmethod object.
        """
        return self.record(classmethod(method))
    
    def staticmethod(self, method: Callable):
        return self.record(staticmethod(method))
    
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

@Symbol.register_type(lambda o: symbol(o.obj))
class mObject:
    """
    Abstract class for macro recorder equipped objects.
    """    
    def __init__(self, obj, macro: Macro, returned_callback: MetaCallable = None, 
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
        
        self._macro = macro
        for name in _INHERITABLE:
            if hasattr(self.obj, name):
                setattr(self, name, getattr(self.obj, name))
            
    def to_namespace(self, obj: Any) -> Symbol | Expr:
        """
        Return the expression of ``obj`` in the correct name space.
        """        
        sym = symbol(obj)
        if self.namespace is None:
            return sym
        else:
            return Expr(Head.getattr, [self.namespace, sym])
    
    @property
    def macro(self) -> Macro:
        return self._macro

# Symbol.register_type(mObject, lambda o: symbol(o.obj))

class mCallable(mObject):
    obj: Callable
    def __init__(self, function: Callable, macro: Macro, returned_callback: MetaCallable = None,
                 namespace: Symbol|Expr = None, record_returned: bool = True):
        super().__init__(function, macro, returned_callback, namespace, record_returned)
    
    @property
    def __signature__(self):
        if hasattr(self.obj, "__signature__"):
            return self.obj.__signature__
        else:
            return inspect.signature(self.obj)
    
    def __call__(self, *args, **kwargs):
        # TODO: This function will be called if __get__ method is the origin, like A.__get__(...).
        with self._macro.blocked():
            out = self.obj(*args, **kwargs)
        if self._macro.active:
            expr = Expr.parse_call(self.to_namespace(self.obj), args, kwargs)
            line = self.returned_callback(expr, out)
            self._macro.append(line)
            self._macro._last_setval = None
        return out
    
class mFunction(mCallable):
    """
    Macro recorder equipped functions. Generated functions are also compatible with methods.
    """
    def __init__(self, function: Callable, macro: Macro, returned_callback: MetaCallable = None,
                 namespace: Symbol|Expr = None, record_returned: bool = True):
        super().__init__(function, macro, returned_callback, namespace, record_returned)
        self._method_type = None
            
    def _make_method_type(self):
        fname = getattr(self.obj, "__name__", None)
        if fname == "__init__":
            def make_expr(obj: _O, *args, **kwargs):
                expr = Expr.parse_init(self.to_namespace(obj), obj.__class__, args, kwargs)
                self._macro._last_setval = None
                return expr
        elif fname == "__call__":
            def make_expr(obj: _O, *args, **kwargs):
                expr = Expr.parse_call(self.to_namespace(obj), args, kwargs)
                self._macro._last_setval = None
                return expr
        elif fname == "__getitem__":
            def make_expr(obj: _O, *args):
                expr = Expr(Head.getitem, [self.to_namespace(obj), args[0]])
                self._macro._last_setval = None
                return expr
        elif fname == "__getattr__":
            def make_expr(obj: _O, *args):
                expr = Expr(Head.getattr, [self.to_namespace(obj), Symbol(args[0])])
                self._macro._last_setval = None
                return expr
        elif fname == "__setitem__":
            def make_expr(obj: _O, *args):
                target = Expr(Head.getitem, [self.to_namespace(obj), args[0]])
                expr = Expr(Head.assign, [target, args[1]])
                if self._macro._last_setval == target:
                    self._macro.pop(-1)
                else:
                    self._macro._last_setval = target
                return expr
        elif fname == "__setattr__":
            def make_expr(obj: _O, *args):
                target = Expr(Head.getattr, [self.to_namespace(obj), Symbol(args[0])])
                expr = Expr(Head.assign, [target, args[1]])
                if self._macro._last_setval == target:
                    self._macro.pop(-1)
                else:
                    self._macro._last_setval = target
                return expr
        elif fname == "__delitem__":
            def make_expr(obj: _O, *args):
                target = Expr(Head.getitem, [self.to_namespace(obj), args[0]])
                expr = Expr(Head.del_, [target])
                self._macro._last_setval = None
                return expr
        elif fname == "__delattr__":
            def make_expr(obj: _O, *args):
                target = Expr(Head.getattr, [self.to_namespace(obj), args[0]])
                expr = Expr(Head.del_, [target])
                self._macro._last_setval = None
                return expr
        elif fname in BINOP_MAP.keys():
            op = BINOP_MAP[fname]
            def make_expr(obj: _O, *args):
                expr = Expr(Head.binop, [op, self.to_namespace(obj), args[0]])
                self._macro._last_setval = None
                return expr
        elif fname in BUILTIN_MAP.keys():
            f = BUILTIN_MAP[fname]
            def make_expr(obj: _O, *args):
                expr = Expr(Head.call, [f, self.to_namespace(obj)] + list(args))
                self._macro._last_setval = None
                return expr
        else:
            def make_expr(obj: _O, *args, **kwargs):
                expr = Expr.parse_method(self.to_namespace(obj), self.obj, args, kwargs)
                self._macro._last_setval = None
                return expr
        
        @wraps(self.obj)
        def method(obj: _O, *args, **kwargs):
            with self._macro.blocked():
                out = self.obj(obj, *args, **kwargs)
            if self._macro.active:
                expr = make_expr(obj, *args, **kwargs)
                if expr is not None:
                    line = self.returned_callback(expr, out)
                self._macro.append(line)
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
            if self._method_type is None:
                self._method_type = self._make_method_type()
            return mMethod(self._method_type, obj)

class mMethod(partial, mCallable):
    """
    Macro recorder equipped method class. Instances are always built by the __get__ method
    in mFunction instance.
    """
    pass

class mClassMethod(mCallable): # TODO: inherit classmethod class if possible
    """
    Macro recorder equipped classmethod class.
    """
    def __init__(self, function: classmethod, macro: Macro, returned_callback: MetaCallable = None, 
                 namespace: Symbol | Expr = None, record_returned: bool = True):
        super().__init__(function, macro, returned_callback, namespace, record_returned)
        self.__self__ = function.__self__
        clsname = Symbol(self.__self__.__name__)
        # TODO: check namespace in MacroMixin
        if self.namespace is None:
            self.namespace = clsname
        else:
            self.namespace = Expr(Head.getattr, [self.namespace, clsname])

class mStaticMethod(mCallable):
    """
    Macro recorder equipped staticmethod class.
    """
    # TODO: How to determine a method is a staticmethod??
    def __init__(self, function: staticmethod, macro: Macro, returned_callback: MetaCallable = None, 
                 namespace: Symbol | Expr = None, record_returned: bool = True):
        super().__init__(function, macro, returned_callback, namespace, record_returned)
        
        clsname = function.__qualname__.split(".")[-2]
        if self.namespace is None:
            self.namespace = Symbol(clsname)
        else:
            self.namespace = Expr(Head.getattr, [self.namespace, clsname])
            
class mProperty(mObject, property):
    """
    Macro recorder equipped property class.
    """
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
            with self._macro.blocked():
                out = fget(obj)
            if self._macro.active:
                expr = Expr(Head.getattr, [self.to_namespace(obj), key])
                expr = self.returned_callback(expr, out)
                self._macro.append(expr)
                self._macro._last_setval = None
            return out
        return self.obj.getter(getter)
        
    def setter(self, fset: Callable[[_O, Any], None]) -> property:
        if not self._flags.Set:
            return self.obj.setter(fset)
        key = Symbol(fset.__name__)
        @wraps(fset)
        def setter(obj, value):
            with self._macro.blocked():
                out = fset(obj, value)
            if self._macro.active:
                target = Expr(Head.getattr, [self.to_namespace(obj), key])
                expr = Expr(Head.assign, [target, value])
                if self._macro._last_setval == target:
                    self._macro.pop(-1)
                else:
                    self._macro._last_setval = target
                
                line = self.returned_callback(expr, out)
                self._macro.append(line)
                    
            return out
        
        return self.obj.setter(setter)
    
    def deleter(self, fdel: Callable[[_O], None]) -> property:
        if not self._flags.Delete:
            return self.obj.deleter(fdel)
        key = Symbol(fdel.__name__)
        @wraps(fdel)
        def deleter(obj):
            with self._macro.blocked():
                out = fdel(obj)
            if self._macro.active:
                target = Expr(Head.getattr, [self.to_namespace(obj), key])
                expr = Expr(Head.del_, [target])
                line = self.returned_callback(expr, out)
                self._macro.append(line)
                self._macro._last_setval = None
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
    """
    Macro recorder equipped module class.
    """
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
            mmod = mModule(attr, self._macro, self.returned_callback, self.to_namespace(self.obj))
            setattr(self, key, mmod)
            return mmod
            
        @wraps(attr)
        def mfunc(*args, **kwargs):
            with self._macro.blocked():
                out = attr(*args, **kwargs)
            if self._macro.active:
                if isinstance(attr, type):
                    cls = Expr(Head.getattr, [self.to_namespace(self.obj), Symbol(out.__class__.__name__)])
                    expr = Expr.parse_init(out, cls, args, kwargs)    
                else:
                    expr = Expr.parse_method(self.to_namespace(self.obj), attr, args, kwargs)
                expr = self.returned_callback(expr, out)
                self._macro.append(expr)
                self._macro._last_setval = None
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