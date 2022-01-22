from __future__ import annotations

from copy import deepcopy
from numbers import Number
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator, overload

from ._validator import validator
from .head import EXEC, Head
from .symbol import Symbol


def str_(expr: Any, indent: int = 0):
    """Convert expr into a proper string."""
    if isinstance(expr, Expr):
        return _STR_MAP[expr.head](expr, indent)
    else:
        return " " * indent + str(expr)


def rm_par(s: str):
    """Remove parenthesis."""
    if s[0] == "(" and s[-1] == ")":
        s = s[1:-1]
    return s


def sjoin(sep: str, iterable: Iterable[Any], indent: int = 0):
    """Join expresions into a single string."""
    return sep.join(str_(expr, indent) for expr in iterable)


def _s_(n: int) -> str:
    """Return spaces."""
    return " " * n


def _comma(a, b):
    return f"{a}, {b}".rstrip(", ")


_STR_MAP: dict[Head, Callable[[Expr, int], str]] = {
    Head.empty: lambda e, i: "",
    Head.getattr: lambda e, i: f"{str_(e.args[0], i)}.{str_(e.args[1])}",
    Head.getitem: lambda e, i: f"{str_(e.args[0], i)}[{str_(e.args[1])}]",
    Head.del_: lambda e, i: f"{_s_(i)}del {str_(e.args[0])}",
    Head.call: lambda e, i: f"{str_(e.args[0], i)}({sjoin(', ', e.args[1:])})",
    Head.assign: lambda e, i: f"{str_(e.args[0], i)} = {e.args[1]}",
    Head.kw: lambda e, i: f"{str_(e.args[0])}={str_(e.args[1])}",
    Head.assert_: lambda e, i: f"{_s_(i)}assert {_comma(str_(e.args[0]), str_(e.args[1]))}",  # noqa
    Head.comment: lambda e, i: f"{_s_(i)}# {e.args[0]}",
    Head.unop: lambda e, i: f"{_s_(i)}({str_(e.args[0])}{str_(e.args[1])})",
    Head.binop: lambda e, i: f"{_s_(i)}({str_(e.args[1])} {str_(e.args[0])} {str_(e.args[2])})",  # noqa
    Head.aug: lambda e, i: f"{_s_(i)}{str_(e.args[1])} {str_(e.args[0])}= {str_(e.args[2])}",  # noqa
    Head.block: lambda e, i: sjoin("\n", e.args, i),
    Head.function: lambda e, i: f"{_s_(i)}def {str_(e.args[0])}:\n{str_(e.args[1], i+4)}",  # noqa
    Head.return_: lambda e, i: f"{_s_(i)}return {sjoin(', ', e.args)}",
    Head.raise_: lambda e, i: f"{_s_(i)}raise {str_(e.args[0])}",
    Head.if_: lambda e, i: f"{_s_(i)}if {rm_par(str_(e.args[0]))}:\n{str_(e.args[1], i+4)}\n{_s_(i)}else:\n{str_(e.args[2], i+4)}",  # noqa
    Head.elif_: lambda e, i: f"{_s_(i)}if {rm_par(str_(e.args[0]))}:\n{str_(e.args[1], i+4)}\n{_s_(i)}else:\n{str_(e.args[2], i+4)}",  # noqa
    Head.for_: lambda e, i: f"{_s_(i)}for {rm_par(str_(e.args[0]))}:\n{str_(e.args[1], i+4)}",  # noqa
    Head.while_: lambda e, i: f"{_s_(i)}while {rm_par(str_(e.args[0]))}:\n{str_(e.args[1], i+4)}",  # noqa
    Head.annotate: lambda e, i: f"{str_(e.args[0], i)}: {str_(e.args[1])}",
}


class Expr:
    """An expression object for metaprogramming."""

    n: int = 0

    def __init__(self, head: Head, args: Iterable[Any]):
        self._head = Head(head)
        self._args = list(map(self.__class__.parse_object, args))
        validator(self._head, self._args)
        self.number = self.__class__.n
        self.__class__.n += 1

    @property
    def head(self) -> Head:
        """Return the head of Expr."""
        return self._head

    @property
    def args(self) -> list[Expr | Symbol]:
        """Return args of Expr."""
        return self._args

    def __repr__(self) -> str:
        """Return Julia-like repr."""
        s = str(self)
        if len(s) > 1:
            s = rm_par(s)
        s = s.replace("\n", "\n  ")
        return f":({s})"

    def __str__(self) -> str:
        """Return a string style of the expression."""
        return str_(self)

    def __eq__(self, expr) -> bool:
        """Equals only if indentical Expr is given."""
        if isinstance(expr, Expr):
            if self.head == expr.head:
                return self.args == expr.args
            else:
                return False
        else:
            return False

    def _dump(self, ind: int = 0) -> str:
        """Recursively expand expressions until it reaches value/kw expression."""
        out = [f"head: {self.head.name}\n{' '*ind}args:\n"]
        for i, arg in enumerate(self.args):
            if isinstance(arg, Symbol):
                value = arg.name
            else:
                value = arg._dump(ind + 4)
            out.append(f"{i:>{ind+2}}: {value}\n")
        return "".join(out)

    def dump(self) -> str:
        """Dump expression into a tree."""
        s = self._dump()
        return s.rstrip("\n") + "\n"

    def copy(self) -> Expr:
        """Copy Expr object."""
        # Always copy object deeply.
        return deepcopy(self)

    def at(self, *indices: int) -> Symbol | Expr:
        """
        Easier way of tandem get-item.

        Helper function to avoid ``expr.args[0].args[0] ...``. Also, exception
        descriptions during this function call. ``expr.at(i, j, k)`` is equivalent
        to ``expr.args[i].args[j].args[k]``.
        """
        now = self
        for i in indices:
            try:
                now = now._args[i]  # type: ignore
            except IndexError as e:
                raise type(e)(f"list index out of range at position {i}.")
            except AttributeError as e:
                raise type(e)(f"Indexing encounted Symbol at position {i}.")
        return now

    def eval(self, _globals: dict = {}, _locals: dict = {}):
        """
        Evaluate or execute macro as an Python script.

        Either ``eval`` or ``exec`` will get called, which determined by its header.
        Calling this function is much safer than those not-recommended usage of
        ``eval`` or ``exec``.

        Parameters
        ----------
        _globals : dict[Symbol, Any], optional
            Mapping from global variable symbols to their values.
        _locals : dict, optional
            Updated variable namespace. Will be a mapping from symbols to values.

        """
        _glb: dict[str, Any] = {
            (sym.name if isinstance(sym, Symbol) else sym): v
            for sym, v in _globals.items()
        }

        # use registered modules
        if Symbol._module_symbols:
            format_dict: dict[Symbol, Symbol | Expr] = {}
            for id_, sym in Symbol._module_symbols.items():
                mod = Symbol._module_map[sym.name]
                vstr = f"var{hex(id_)}"
                format_dict[sym] = Symbol(vstr)
                _glb[vstr] = mod
            # Modules will not be registered as alias ("np" will be "numpy" in macro).
            # To avoid name collision, it is safer to rename them to "var0x...".
            self = self.format(format_dict)

        if self.head in EXEC:
            return exec(str(self), _glb, _locals)
        else:
            return eval(str(self), _glb, _locals)

    @classmethod
    def parse_method(
        cls,
        obj: Any,
        func: Callable,
        args: tuple[Any, ...] = None,
        kwargs: dict = None,
    ) -> Expr:
        """Parse ``obj.func(*args, **kwargs)``."""
        method = cls(head=Head.getattr, args=[symbol(obj), func])
        return cls.parse_call(method, args, kwargs)

    @classmethod
    def parse_init(
        cls,
        obj: Any,
        init_cls: type = None,
        args: tuple[Any, ...] = None,
        kwargs: dict = None,
    ) -> Expr:
        """Parse ``obj = init_cls(*args, **kwargs)``."""
        if init_cls is None:
            init_cls = type(obj)
        sym = symbol(obj)
        return cls(Head.assign, [sym, cls.parse_call(init_cls, args, kwargs)])

    @classmethod
    def parse_call(
        cls,
        func: Callable | Symbol | Expr,
        args: tuple[Any, ...] = None,
        kwargs: dict = None,
    ) -> Expr:
        """Parse ``func(*args, **kwargs)``."""
        if args is None:
            args = ()
        elif not isinstance(args, tuple):
            raise TypeError("args must be a tuple")
        if kwargs is None:
            kwargs = {}
        elif not isinstance(kwargs, dict):
            raise TypeError("kwargs must be a dict")
        inputs = [func] + cls._convert_args(args, kwargs)
        return cls(head=Head.call, args=inputs)

    @classmethod
    def parse_setitem(cls, obj: Any, key: Any, value: Any) -> Expr:
        """Parse ``obj[key] = value)``."""
        target = cls(Head.getitem, [symbol(obj), symbol(key)])
        return cls(Head.assign, [target, symbol(value)])

    @classmethod
    def parse_setattr(cls, obj: Any, key: str, value: Any) -> Expr:
        """Parse ``obj.key = value``."""
        target = cls(Head.getattr, [symbol(obj), Symbol(key)])
        return cls(Head.assign, [target, symbol(value)])

    @classmethod
    def _convert_args(cls, args: tuple[Any, ...], kwargs: dict) -> list:
        inputs = []
        for a in args:
            inputs.append(a)

        for k, v in kwargs.items():
            inputs.append(cls(Head.kw, [Symbol(k), symbol(v)]))
        return inputs

    @classmethod
    def parse_object(cls, a: Any) -> Symbol | Expr:
        """Convert an object into a macro-type."""
        return a if isinstance(a, cls) else symbol(a)

    def issetattr(self) -> bool:
        """Determine if an expression is in the form of ``setattr(obj, key value)``."""
        if self.head == Head.assign:
            target = self.args[0]
            if isinstance(target, Expr) and target.head == Head.getattr:
                return True
        return False

    def issetitem(self) -> bool:
        """Determine if an expression is in the form of ``setitem(obj, key value)``."""
        if self.head == Head.assign:
            target = self.args[0]
            if isinstance(target, Expr) and target.head == Head.getitem:
                return True
        return False

    def iter_args(self) -> Iterator[Symbol]:
        """Recursively iterate along all the arguments."""
        for arg in self.args:
            if isinstance(arg, Expr):
                yield from arg.iter_args()
            elif isinstance(arg, Symbol):
                yield arg
            else:
                raise RuntimeError(f"{arg} (type {type(arg)})")

    def iter_expr(self) -> Iterator[Expr]:
        """
        Recursively iterate over all the nested Expr, until reaching to non-nested Expr.

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
    def format(self, mapping: dict, inplace: bool = False) -> Expr:
        ...

    @overload
    def format(
        self, mapping: Iterable[tuple[Any, Symbol | Expr]], inplace: bool = False
    ) -> Expr:
        ...

    def format(self, mapping, inplace=False) -> Expr:
        """
        Format expressions in the macro.

        Just like formatting method of string, this function can replace certain symbols
        to others.

        Parameters
        ----------
        mapping : dict or iterable of tuples
            Mapping from objects to symbols or expressions. Keys will be converted to
            symbol. For instance, if you used ``arr``, a numpy.ndarray as an input of an
            macro-recordable function, that input will appear like 'var0x1...'. By
            calling ``format([(arr, "X")])`` then 'var0x1...' will be substituted to
            'X'.
        inplace : bool, default is False
            Expression will be overwritten if true.

        Returns
        -------
        Expression
            Formatted expression.
        """
        if isinstance(mapping, dict):
            mapping = mapping.items()
        mapping = _check_format_mapping(mapping)

        if not inplace:
            self = self.copy()

        return self._unsafe_format(mapping)

    def _unsafe_format(self, mapping: dict) -> Expr:
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


def _check_format_mapping(mapping_list: Iterable) -> dict[Symbol, Symbol | Expr]:
    _dict: dict[Symbol, Symbol | Expr] = {}
    for comp in mapping_list:
        if len(comp) != 2:
            raise ValueError("Wrong style of mapping list.")
        k, v = comp
        if isinstance(v, Expr) and v.head in EXEC:
            raise ValueError("Cannot replace a symbol to a non-evaluable expression.")

        key = symbol(k)
        if isinstance(key, Expr):
            raise TypeError(
                f"Object of type {type(k).__name__} returns Expr type, thus cannot"
                "be used as a format template."
            )
        if isinstance(v, str) and not isinstance(k, Symbol):
            _dict[key] = Symbol(v)
        else:
            _dict[key] = symbol(v)

    return _dict


def make_symbol_str(obj: Any):
    """Make a string for symbol."""
    # hexadecimals are easier to distinguish
    _id = id(obj)
    if obj is not None:
        Symbol._variables.add(_id)
    return f"var{hex(_id)}"


def symbol(obj: Any, constant: bool = True) -> Symbol | Expr:
    """
    Make a proper Symbol or Expr instance from any objects.

    Unlike Symbol(...) constructor, which directly make a Symbol from a string, this
    function checks input type and determine the optimal string to represent the
    object. Especially, Symbol("xyz") will return ``:xyz`` while symbol("xyz") will
    return ``:'xyz'``.

    Parameters
    ----------
    obj : Any
        Any object from which a Symbol will be created.
    constant : bool, default is True
        If true, object is interpreted as a constant like 1 or "a". Otherwise object is
        converted to a variable that named with its ID.

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
    elif isinstance(obj, Number):  # int, float, bool, ...
        seq = obj
    elif isinstance(obj, ModuleType):
        # Register module to the default namespace of Symbol class. This function is
        # called every time a module type object is converted to a Symbol because users
        # always have to pass the module object to the global variables when calling
        # eval function.
        if obj_id in Symbol._module_symbols.keys():
            sym = Symbol._module_symbols[obj_id]
        else:
            *main, seq = obj.__name__.split(".")
            sym = Symbol(seq, obj_id)
            sym.constant = True
            if len(main) == 0:
                # submodules should not be registered
                Symbol._module_symbols[obj_id] = sym
                Symbol._module_map[seq] = obj
        return sym
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
        return seq
    else:
        sym = Symbol(seq, obj_id)
        sym.constant = constant
        return sym
