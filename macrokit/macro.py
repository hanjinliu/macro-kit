from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from importlib import import_module
from functools import partial, wraps
import inspect
from typing import (
    Callable,
    Iterable,
    Iterator,
    Any,
    Union,
    overload,
    TypeVar,
    NamedTuple,
    MutableSequence
)
from typing_extensions import TypedDict
from types import ModuleType

from .ast import parse
from .expression import Head, Expr, symbol, EXEC
from .symbol import Symbol

_NON_RECORDABLE = (
    "__new__",
    "__class__",
    "__repr__",
    "__getattribute__",
    "__dir__",
    "__init_subclass__",
    "__subclasshook__",
    "__class_getitem__",
)

_INHERITABLE = ("__module__", "__name__", "__qualname__", "__doc__", "__annotations__")


BINOP_MAP = {
    "__add__": Symbol._reserved("+"),
    "__sub__": Symbol._reserved("-"),
    "__mul__": Symbol._reserved("*"),
    "__div__": Symbol._reserved("/"),
    "__mod__": Symbol._reserved("%"),
    "__eq__": Symbol._reserved("=="),
    "__neq__": Symbol._reserved("!="),
    "__gt__": Symbol._reserved(">"),
    "__ge__": Symbol._reserved(">="),
    "__lt__": Symbol._reserved("<"),
    "__le__": Symbol._reserved("<="),
    "__pow__": Symbol._reserved("**"),
    "__matmul__": Symbol._reserved("@"),
    "__floordiv__": Symbol._reserved("//"),
    "__and__": Symbol._reserved("&"),
    "__or__": Symbol._reserved("|"),
    "__xor__": Symbol._reserved("^"),
}

UNOP_MAP = {
    "__pos__": Symbol._reserved("+"),
    "__neg__": Symbol._reserved("-"),
    "__invert__": Symbol._reserved("~"),
}

BUILTIN_MAP = {
    "__hash__": Symbol("hash"),
    "__len__": Symbol("len"),
    "__str__": Symbol("str"),
    "__repr__": Symbol("repr"),
    "__bool__": Symbol("bool"),
    "__float__": Symbol("float"),
    "__int__": Symbol("int"),
    "__format__": Symbol("format"),
    "__list__": Symbol("list"),
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
_Class = TypeVar("_Class", bound=type)


class MacroFlagOptions(TypedDict, total=False):
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
        "__delattr__": "Delete",
    }

    def __init__(self, args: Iterable[Expr] = (), *, flags: MacroFlagOptions = {}):
        super().__init__(head=Head.block, args=args)
        self.active = True
        self._callbacks: list[Callable[[Symbol | Expr], Any]] = []
        if isinstance(flags, MacroFlags):
            flags = flags._asdict()
        self._flags = MacroFlags(**flags)
        self._last_setval: Expr | None = None

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
            expr_ = parse(expr)
        elif isinstance(expr, Expr):
            expr_ = expr
        else:
            raise TypeError("Cannot insert objects to Macro except for Expr objects.")

        self.args.insert(key, expr_)
        for callback in self._callbacks:
            callback(expr_)

    @overload
    def __getitem__(self, key: int | str) -> Expr:
        ...

    @overload
    def __getitem__(self, key: slice) -> Macro:
        ...

    def __getitem__(self, key):
        return self._args[key]

    def __setitem__(self, key: int, value: Symbol | Expr) -> None:
        self._args[key] = value

    def __delitem__(self, key: int | slice) -> None:
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
    def record(
        self, obj: property, *, returned_callback: MetaCallable = None
    ) -> mProperty:
        ...

    @overload
    def record(
        self, obj: ModuleType, *, returned_callback: MetaCallable = None
    ) -> mModule:
        ...

    @overload
    def record(
        self, obj: Callable, *, returned_callback: MetaCallable = None
    ) -> mFunction:
        ...

    @overload
    def record(self, obj: _Class, *, returned_callback: MetaCallable = None) -> _Class:
        ...

    @overload
    def record(
        self, obj: classmethod, *, returned_callback: MetaCallable = None
    ) -> mClassMethod:
        ...

    @overload
    def record(
        self, obj: staticmethod, *, returned_callback: MetaCallable = None
    ) -> mStaticMethod:
        ...

    @overload
    def record(
        self, *, returned_callback: MetaCallable = None
    ) -> Callable[[Recordable], mObject | MacroMixin]:
        ...

    def record(self, obj=None, *, returned_callback=None):
        """
        A wrapper that convert an object to a macro-recordable one.

        Parameters
        ----------
        obj : property, module, type or callable, optional
            Base object.
        returned_callback : callable, optional
            A function that will called after new expression is appended. Must take an
            expression or an expression with the last returned value as inputs.
        """
        kwargs = dict(
            macro=self,
            returned_callback=returned_callback,
            record_returned=self._flags.Return,
        )

        def wrapper(_obj):
            if isinstance(_obj, property):
                return mProperty(
                    _obj,
                    macro=self,
                    returned_callback=returned_callback,
                    flags=self.flags._asdict(),
                )
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
        _dict: dict[str, mObject | MacroMixin] = {}
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
        expr_map: list[tuple[Symbol | Expr, int]] = []
        need = set()
        for i, expr in enumerate(self):
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

    bound: dict[int, Macro] = {}

    def __get__(self, obj: Any, type=None):
        if obj is None:
            return self
        obj_id = id(obj)
        cls = self.__class__
        try:
            macro = cls.bound[obj_id]
        except KeyError:
            macro = cls(flags=self.flags)
            cls.bound[obj_id] = macro
        return macro


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


@Symbol.register_type(lambda o: symbol(o.obj))  # type: ignore
class mObject:
    """
    Abstract class for macro recorder equipped objects.
    """

    def __init__(
        self,
        obj,
        macro: Macro,
        returned_callback: MetaCallable = None,
        namespace: Symbol | Expr | None = None,
        record_returned: bool = True,
    ) -> None:
        self.obj = obj

        if returned_callback is not None:
            _callback_nargs = len(inspect.signature(returned_callback).parameters)
            if _callback_nargs == 1:
                def _returned_callback(expr: Expr, out):
                    return returned_callback(expr)
            elif _callback_nargs == 2:
                _returned_callback = returned_callback
            else:
                raise TypeError(
                    "returned_callback cannot take arguments more than two."
                )

            if record_returned:
                self.returned_callback = lambda expr, out: _returned_callback(
                    _assign_value(expr, out), out
                )
            else:
                self.returned_callback = _returned_callback

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


_GET = Symbol.var("__get__")
_SET = Symbol.var("__set__")
_DELETE = Symbol.var("__delete__")


class mCallable(mObject):
    obj: Callable
    __name__: str

    def __init__(
        self,
        function: Callable,
        macro: Macro,
        returned_callback: MetaCallable = None,
        namespace: Symbol | Expr | None = None,
        record_returned: bool = True,
    ):
        super().__init__(function, macro, returned_callback, namespace, record_returned)
        if self.__name__ == "__get__":

            def make_expr(*args, **kwargs):
                return Expr.parse_method(args[0], _GET, (args[1], args[2]), {})

        elif self.__name__ == "__set__":

            def make_expr(*args, **kwargs):
                return Expr.parse_method(args[0], _SET, (args[1], args[2]), {})

        elif self.__name__ == "__delete__":

            def make_expr(*args, **kwargs):
                return Expr.parse_method(args[0], _DELETE, (args[1]), {})

        else:

            def make_expr(*args, **kwargs):
                return Expr.parse_call(self.to_namespace(self.obj), args, kwargs)

        self._make_expr = make_expr

    @property
    def __signature__(self):
        if hasattr(self.obj, "__signature__"):
            return self.obj.__signature__
        else:
            return inspect.signature(self.obj)

    def __call__(self, *args, **kwargs):
        # TODO: This function will be called if __get__ method is the origin,
        # like A.__get__(...).
        with self._macro.blocked():
            out = self.obj(*args, **kwargs)
        if self._macro.active:
            expr = self._make_expr(*args, **kwargs)
            line = self.returned_callback(expr, out)
            self._macro.append(line)
            self._macro._last_setval = None
        return out


class mFunction(mCallable):
    """
    Macro recorder equipped functions. Generated functions are also compatible with
    methods.
    """

    def __init__(
        self,
        function: Callable,
        macro: Macro,
        returned_callback: MetaCallable = None,
        namespace: Symbol | Expr | None = None,
        record_returned: bool = True,
    ):
        super().__init__(function, macro, returned_callback, namespace, record_returned)
        self._method_type = None

    def _make_method_type(self):
        fname = getattr(self.obj, "__name__", None)
        if fname == "__init__":

            def make_expr(obj: _O, *args, **kwargs):
                expr = Expr.parse_init(
                    self.to_namespace(obj), obj.__class__, args, kwargs
                )
                self._macro._last_setval = None
                return expr

        elif fname == "__call__":

            def make_expr(obj: _O, *args, **kwargs):
                expr = Expr.parse_call(self.to_namespace(obj), args, kwargs)
                self._macro._last_setval = None
                return expr

        elif fname == "__getitem__":

            def make_expr(obj: _O, *args, **kwargs):
                expr = Expr(Head.getitem, [self.to_namespace(obj), args[0]])
                self._macro._last_setval = None
                return expr

        elif fname == "__getattr__":

            def make_expr(obj: _O, *args, **kwargs):
                expr = Expr(Head.getattr, [self.to_namespace(obj), Symbol(args[0])])
                self._macro._last_setval = None
                return expr

        elif fname == "__setitem__":

            def make_expr(obj: _O, *args, **kwargs):
                target = Expr(Head.getitem, [self.to_namespace(obj), args[0]])
                expr = Expr(Head.assign, [target, args[1]])
                if self._macro._last_setval == target:
                    self._macro.pop(-1)
                else:
                    self._macro._last_setval = target
                return expr

        elif fname == "__setattr__":

            def make_expr(obj: _O, *args, **kwargs):
                target = Expr(Head.getattr, [self.to_namespace(obj), Symbol(args[0])])
                expr = Expr(Head.assign, [target, args[1]])
                if self._macro._last_setval == target:
                    self._macro.pop(-1)
                else:
                    self._macro._last_setval = target
                return expr

        elif fname == "__delitem__":

            def make_expr(obj: _O, *args, **kwargs):
                target = Expr(Head.getitem, [self.to_namespace(obj), args[0]])
                expr = Expr(Head.del_, [target])
                self._macro._last_setval = None
                return expr

        elif fname == "__delattr__":

            def make_expr(obj: _O, *args, **kwargs):
                target = Expr(Head.getattr, [self.to_namespace(obj), args[0]])
                expr = Expr(Head.del_, [target])
                self._macro._last_setval = None
                return expr

        elif fname in UNOP_MAP.keys():
            op = UNOP_MAP[fname]

            def make_expr(obj: _O, *args, **kwargs):
                expr = Expr(Head.unop, [op, self.to_namespace(obj)])
                self._macro._last_setval = None
                return expr

        elif fname in BINOP_MAP.keys():
            op = BINOP_MAP[fname]

            def make_expr(obj: _O, *args, **kwargs):
                expr = Expr(Head.binop, [op, self.to_namespace(obj), args[0]])
                self._macro._last_setval = None
                return expr

        elif fname in BUILTIN_MAP.keys():
            f = BUILTIN_MAP[fname]

            def make_expr(obj: _O, *args, **kwargs):
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
                obj_id = id(obj)
                if obj_id in self._macro.__class__.bound.keys():
                    macro = self._macro.__class__.bound[obj_id]
                    macro.append(line)
                    # TODO: macro._last_setval should also be considered

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
    Macro recorder equipped method class. Instances are always built by the __get__
    method in mFunction instance.
    """


class mClassMethod(mCallable):  # TODO: inherit classmethod class if possible
    """
    Macro recorder equipped classmethod class.
    """

    def __init__(
        self,
        function: classmethod,
        macro: Macro,
        returned_callback: MetaCallable = None,
        namespace: Symbol | Expr | None = None,
        record_returned: bool = True,
    ):
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
    def __init__(
        self,
        function: staticmethod,
        macro: Macro,
        returned_callback: MetaCallable = None,
        namespace: Symbol | Expr | None = None,
        record_returned: bool = True,
    ):
        super().__init__(function, macro, returned_callback, namespace, record_returned)

        clsname = function.__func__.__qualname__.split(".")[-2]
        if self.namespace is None:
            self.namespace = Symbol(clsname)
        else:
            self.namespace = Expr(Head.getattr, [self.namespace, clsname])


class mProperty(mObject, property):
    """
    Macro recorder equipped property class.
    """

    obj: property

    def __init__(
        self,
        prop: property,
        macro: Macro,
        returned_callback: MetaCallable = None,
        namespace: Symbol | Expr = None,
        flags: MacroFlagOptions = {},
    ):
        if isinstance(flags, MacroFlags):
            flags = flags._asdict()
        self._flags = MacroFlags(**flags)
        super().__init__(
            prop,
            macro,
            returned_callback,
            namespace,
            record_returned=self._flags.Return,
        )
        self.obj = self.getter(prop.fget)

    def getter(self, fget: Callable[[_O], Any]) -> property:
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
                self._record_instance(obj, expr)
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

                obj_id = id(obj)
                if obj_id in self._macro.__class__.bound.keys():
                    macro = self._macro.__class__.bound[obj_id]
                    if macro._last_setval == target:
                        macro.pop(-1)
                    else:
                        macro._last_setval = target
                    macro.append(line)

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
                self._record_instance(obj, line)
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

    def _record_instance(self, obj, line):
        obj_id = id(obj)
        if obj_id in self._macro.__class__.bound.keys():
            macro = self._macro.__class__.bound[obj_id]
            macro.append(line)


class mModule(mObject):
    """
    Macro recorder equipped module class.
    """

    obj: ModuleType

    def __init__(
        self,
        obj,
        macro: Macro,
        returned_callback: MetaCallable = None,
        namespace: Symbol | Expr | None = None,
        record_returned: bool = True,
    ) -> None:
        super().__init__(obj, macro, returned_callback, namespace, record_returned)
        self.__all__ = getattr(obj, "__all__", [])

    def __getattr__(self, key: str):
        try:
            attr = getattr(self.obj, key)
            is_module = inspect.ismodule(attr)
        except AttributeError:
            try:
                attr = import_module("." + key, self.obj.__name__)
                is_module = True
            except ModuleNotFoundError:
                raise ValueError(f"No function or submodule named '{key}'.")

        if is_module:
            mmod = mModule(
                attr, self._macro, self.returned_callback, self.to_namespace(self.obj)
            )
            setattr(self, key, mmod)  # cache
            return mmod

        elif not callable(attr):
            # constants, such as "__version__"
            # TODO: should we record this as getattr?
            return attr

        @wraps(attr)
        def mfunc(*args, **kwargs):
            with self._macro.blocked():
                out = attr(*args, **kwargs)
            if self._macro.active:
                if isinstance(attr, type):
                    cls = Expr(
                        Head.getattr,
                        [self.to_namespace(self.obj), Symbol(out.__class__.__name__)],
                    )
                    expr = Expr.parse_init(out, cls, args, kwargs)
                else:
                    expr = Expr.parse_method(
                        self.to_namespace(self.obj), attr, args, kwargs
                    )
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
