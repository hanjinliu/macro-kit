import builtins
import inspect
from copy import deepcopy
from numbers import Number
from types import FunctionType, ModuleType
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Sequence,
    Union,
    overload,
)
from weakref import WeakValueDictionary

from macrokit._symbol import Symbol
from macrokit._validator import validator
from macrokit.head import EXEC, HAS_BLOCK, Head
from macrokit.type_map import _TYPE_MAP, register_type


def str_(expr: Any, indent: int = 0):
    """Convert expr into a proper string."""
    if isinstance(expr, Expr):
        return _STR_MAP[expr.head](expr, indent)
    else:
        return " " * indent + str(expr)


def str_lmd(expr: Any, indent: int = 0):
    """Convert str into a proper lambda function definition."""
    s = str(expr)
    if s.startswith("<lambda>(") and s.endswith(")"):
        call = s[7:-1]
    else:
        call = s
    return " " * indent + f"lambda {call}"


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


def _tuple_str(x: "Expr", i: int):
    args = x.args
    if len(args) == 1:
        return f"{_s_(i)}({str_(args[0])},)"
    else:
        tup = ", ".join(str_(arg) for arg in args)
        return f"{_s_(i)}({tup})"


def _filt_str(x: "Symbol | Expr", i: int):
    if isinstance(x, Symbol):
        return f"{_s_(i)}if {x}"
    return f"{_s_(i)}if {x.args[0]}"


def _assert_str(x: "Expr", i: int):
    args = x.args
    if len(args) == 1:
        return f"{_s_(i)}assert {str_(args[0])}"
    else:
        return f"{_s_(i)}assert {_comma(str_(args[0]), str_(args[1]))}"


def _generator_str(x: "Expr", i: int):
    args = x.args
    if len(args) == 3:
        return f"{_s_(i)}({str_(args[0])} for {str_(args[1])} in {str_(args[2])})"
    else:
        filt = " ".join(_filt_str(f, i) for f in args[3:])
        return (
            f"{_s_(i)}({str_(args[0])} for {str_(args[1])} in {str_(args[2])} {filt})"
        )


def _yield_str(x: "Expr", i: int):
    args = x.args
    if isinstance(args[0], Expr) and args[0].head is Head.from_:
        return f"{_s_(i)}yield from {str_(args[0].args[0])}"
    return f"{_s_(i)}yield {sjoin(', ', args)}"


def _if_str(x: "Expr", i: int):
    args = x.args
    if len(args) == 2:
        return f"{_s_(i)}if {str_(args[0])}:\n{str_(args[1], i+4)}"
    else:
        return f"{_s_(i)}if {str_(args[0])}:\n{str_(args[1], i+4)}\n{_s_(i)}else:\n{str_(args[2], i+4)}"  # noqa


def _try_str(x: "Expr", i: int):
    _try = x.args[0]
    _else = x.args[-2]
    _finally = x.args[-1]
    _excepts = list(x.args[1:-2])
    strs = [f"{_s_(i)}try:\n{str_(_try, indent=i + 4)}"]

    while _excepts:
        _exc_as = _excepts.pop(0)
        _exc_block = _excepts.pop(0)
        assert isinstance(_exc_block, Expr)
        if isinstance(_exc_as, Expr) and _exc_as.head is Head.empty:
            strs.append(f"{_s_(i)}except:\n{str_(_exc_block, indent=i + 4)}")
        else:
            strs.append(
                f"{_s_(i)}except {str_(_exc_as)}:\n{str_(_exc_block, indent=i + 4)}"
            )

    assert isinstance(_else, Expr)
    assert isinstance(_finally, Expr)
    if _else.head is not Head.empty:
        strs.append(f"{_s_(i)}else:\n{str_(_else, indent=i + 4)}")
    if _finally.head is not Head.empty:
        strs.append(f"{_s_(i)}finally:\n{str_(_finally, indent=i + 4)}")
    return "\n".join(strs)


def _import_str(x: "Expr", i: int):
    arg0 = x.args[-1]
    if isinstance(arg0, Expr) and arg0.head == Head.from_:
        args = x.args[:-1]
        prefix = f"from {arg0.args[0]} "
    else:
        args = x.args
        prefix = ""

    return f"{_s_(i)}{prefix}import {sjoin(', ', args, i)}"


def _case_str(x: "Expr", i: int):
    args = x.args
    if len(args) == 2:
        return f"{_s_(i)}case {str_(args[0])}:\n{str_(args[1], i+4)}"
    else:
        return f"{_s_(i)}case {str_(args[0])} {str_(args[2])}:\n{str_(args[1], i+4)}"


def _only_generator(args: "list[Expr | Symbol]") -> bool:
    return (
        len(args) == 1 and isinstance(args[0], Expr) and args[0].head is Head.generator
    )


def _list_str(x: "Expr", i: int) -> str:
    args = x.args
    if _only_generator(args):
        return f"{_s_(i)}[{rm_par(str_(args[0]))}]"
    else:
        return f"{_s_(i)}[{', '.join(str_(arg) for arg in args)}]"


def _braces_str(x: "Expr", i: int) -> str:
    args = x.args
    if _only_generator(args):
        return f"{_s_(i)}{{{rm_par(str_(args[0]))}}}"
    else:
        return f"{_s_(i)}{{{', '.join(str_(arg) for arg in args)}}}"


_NL = "\n"

_STR_MAP: "dict[Head, Callable[[Expr, int], str]]" = {
    Head.empty: lambda e, i: "",
    Head.getattr: lambda e, i: f"{str_(e.args[0], i)}.{str_(e.args[1])}",
    Head.getitem: lambda e, i: f"{str_(e.args[0], i)}[{rm_par(str_(e.args[1]))}]",
    Head.del_: lambda e, i: f"{_s_(i)}del {str_(e.args[0])}",
    Head.tuple: _tuple_str,
    Head.list: _list_str,
    Head.braces: _braces_str,
    Head.call: lambda e, i: f"{str_(e.args[0], i)}({sjoin(', ', e.args[1:])})",
    Head.assign: lambda e, i: f"{str_(e.args[0], i)} = {e.args[1]}",
    Head.kw: lambda e, i: f"{str_(e.args[0])}={str_(e.args[1])}",
    Head.assert_: _assert_str,
    Head.comment: lambda e, i: f"{_s_(i)}# {e.args[0]}",
    Head.unop: lambda e, i: f"{_s_(i)}({str_(e.args[0])}{str_(e.args[1])})",
    Head.binop: lambda e, i: f"{_s_(i)}({(' ' + str_(e.args[0]) + ' ').join(str_(a) for a in e.args[1:])})",  # noqa
    Head.aug: lambda e, i: f"{_s_(i)}{str_(e.args[1])} {str_(e.args[0])}= {str_(e.args[2])}",  # noqa
    Head.block: lambda e, i: sjoin("\n", e.args, i),
    Head.function: lambda e, i: f"{_s_(i)}def {str_(e.args[0])}:\n{str_(e.args[1], i+4)}",  # noqa
    Head.lambda_: lambda e, i: f"{str_lmd(e.args[0], i)}: {str_(e.args[1])}",  # noqa
    Head.return_: lambda e, i: f"{_s_(i)}return {sjoin(', ', e.args)}",
    Head.yield_: _yield_str,
    Head.raise_: lambda e, i: f"{_s_(i)}raise {str_(e.args[0])}",
    Head.if_: _if_str,
    Head.try_: _try_str,
    Head.for_: lambda e, i: f"{_s_(i)}for {rm_par(str_(e.args[0]))}:\n{str_(e.args[1], i+4)}",  # noqa
    Head.while_: lambda e, i: f"{_s_(i)}while {rm_par(str_(e.args[0]))}:\n{str_(e.args[1], i+4)}",  # noqa
    Head.generator: _generator_str,
    Head.filter: _filt_str,
    Head.annotate: lambda e, i: f"{str_(e.args[0], i)}: {str_(e.args[1])}",
    Head.class_: lambda e, i: f"{_s_(i)}class {str_(e.args[0])}:\n{str_(e.args[1], i+4)}",  # noqa
    Head.from_: lambda e, i: f"{_s_(i)}from {str_(e.args[0])}",  # noqa
    Head.with_: lambda e, i: f"{_s_(i)}with {str_(e.args[0])}:\n{str_(e.args[1], i+4)}",  # noqa
    Head.as_: lambda e, i: f"{str_(e.args[0])} as {e.args[1]}",
    Head.import_: _import_str,
    Head.star: lambda e, i: f"{_s_(i)}*{str_(e.args[0])}",
    Head.starstar: lambda e, i: f"{_s_(i)}**{str_(e.args[0])}",
    Head.decorator: lambda e, i: f"{_s_(i)}@{str_(e.args[0])}\n{str_(e.args[1], i)}",
    Head.walrus: lambda e, i: f"({str_(e.args[0])} := {str_(e.args[1])})",
    Head.match: lambda e, i: f"{_s_(i)}match {str_(e.args[0])}:\n{_NL.join(str_(a, i+4) for a in e.args[1:])}",  # noqa
    Head.case: _case_str,  # noqa
}

_Expr = Union[Symbol, "Expr"]


class Expr:
    """An expression object for metaprogramming."""

    def __init__(self, head: Head, args: Iterable[Any]):
        self._head = Head(head)
        self._args = list(map(self.__class__.parse_object, args))
        validator(self._head, self._args)

    @property
    def head(self) -> Head:
        """Return the head of Expr."""
        return self._head

    @property
    def args(self) -> "list[_Expr]":
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

    def dump(self) -> "PrettyString":
        """Dump expression into a tree."""
        s = self._dump()
        return PrettyString(s.rstrip("\n") + "\n")

    def copy(self) -> "Expr":
        """Copy Expr object."""
        # Always copy object deeply.
        return deepcopy(self)

    def at(self, *indices: int) -> _Expr:
        """
        Easier way of tandem get-item.

        Helper function to avoid ``expr.args[0].args[0] ...``. Also, exception
        descriptions during this function call. ``expr.at(i, j, k)`` is equivalent
        to ``expr.args[i].args[j].args[k]``.
        """
        now: _Expr = self
        for i in indices:
            if isinstance(now, Symbol):
                raise TypeError(f"Indexing encounted Symbol at position {i}.")
            try:
                now = now._args[i]
            except IndexError as e:
                raise type(e)(f"list index out of range at position {i}.")

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
        if _STORED_VALUES:
            format_dict: dict[Symbol, _Expr] = {}
            for id_, (sym, obj) in _STORED_VALUES.items():
                if isinstance(sym, Expr):
                    continue
                vstr = Symbol.symbol_str_for_id(id_)
                format_dict[sym] = Symbol(vstr)
                _glb[vstr] = obj
            # Modules will not be registered as alias ("np" will be "numpy" in macro).
            # To avoid name collision, it is safer to rename them to "var...".
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
        args: "tuple[Any, ...] | None" = None,
        kwargs: "dict[str, Any] | None" = None,
    ) -> "Expr":
        """Parse ``obj.func(*args, **kwargs)``."""
        method = cls(head=Head.getattr, args=[symbol(obj), func])
        return cls.parse_call(method, args, kwargs)

    @classmethod
    def parse_init(
        cls,
        obj: Any,
        init_cls: "type | Expr | None" = None,
        args: "tuple[Any, ...] | None" = None,
        kwargs: "dict[str, Any] | None" = None,
    ) -> "Expr":
        """Parse ``obj = init_cls(*args, **kwargs)``."""
        if init_cls is None:
            init_cls = type(obj)
        sym = symbol(obj)
        return cls(Head.assign, [sym, cls.parse_call(init_cls, args, kwargs)])

    @classmethod
    def parse_call(
        cls,
        func: Callable | _Expr,
        args: "tuple[Any, ...] | None" = None,
        kwargs: "dict[str, Any] | None" = None,
    ) -> "Expr":
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
    def empty(cls) -> "Expr":
        """Create an empty expression."""
        return cls(head=Head.empty, args=[])

    def split_call(self) -> "tuple[_Expr, tuple[_Expr, ...], dict[str, _Expr]]":
        """Split ``func(*args, **kwargs)`` to (func, args, kwargs)."""
        if self.head is not Head.call:
            raise ValueError(f"Expected {Head.call}, got {self.head}.")
        args = []
        kwargs = {}
        for arg in self.args[1:]:
            if isinstance(arg, Expr) and arg.head is Head.kw:
                sym = arg.args[0]
                if not isinstance(sym, Symbol):
                    raise RuntimeError(f"Expected Symbol, got {type(sym)}.")
                kwargs[sym.name] = arg.args[1]
            else:
                args.append(arg)
        return self.args[0], tuple(args), kwargs

    @classmethod
    def unsplit_call(
        cls, fn: "_Expr", args: "tuple[_Expr, ...]", kwargs: "dict[str, _Expr]"
    ) -> "Expr":
        """Unsplit ``func(*args, **kwargs)`` to (func, args, kwargs)."""
        inputs = [fn] + cls._convert_args(args, kwargs)
        return cls(head=Head.call, args=inputs)

    def split_method(
        self,
    ) -> "tuple[_Expr, Symbol, tuple[_Expr, ...], dict[str, _Expr]]":
        """Split ``obj.func(*args, **kwargs)`` to (obj, func, args, kwargs)."""
        fn, args, kwargs = self.split_call()
        if not isinstance(fn, Expr) or fn.head is not Head.getattr:
            raise ValueError("Not a method call.")
        obj, attr = fn.args
        if not isinstance(attr, Symbol):
            raise RuntimeError("Unreachable in setitem expression.")
        return obj, attr, args, kwargs

    @classmethod
    def unsplit_method(
        cls,
        obj: "_Expr",
        func: str,
        args: "tuple[_Expr, ...]",
        kwargs: "dict[str, _Expr]",
    ) -> "Expr":
        """Unsplit ``obj.func(*args, **kwargs)`` to (obj, func, args, kwargs)."""
        fn = cls(Head.getattr, [obj, Symbol(func)])
        return cls.unsplit_call(fn, args, kwargs)

    def eval_call_args(
        self,
        ns: dict = {},
    ) -> "tuple[tuple[Any, ...], dict[str, Any]]":
        """
        Evaluate arguments in call expression.

        >>> expr = parse("f(1, 2, x=3)")
        >>> expr.eval_call_args()  # (1, 2), {"x": 3}

        Parameters
        ----------
        ns : Dict[str, Any], optional
            Namespace used for evaluation.
        """
        if self.head is not Head.call:
            raise ValueError(f"Expected {Head.call}, got {self.head}.")
        # search for the index where keyword argument starts
        arguments = self.args[1:]
        i = 0
        for i, arg in enumerate(arguments):
            if isinstance(arg, Expr) and arg.head is Head.kw:
                break
        else:
            i += 1

        _args = arguments[:i]
        _kwargs = arguments[i:]

        # prepare namespaces
        args_ns: dict[str | _Expr, Any] = ns.copy()
        args_ns[symbol(_tuple)] = _tuple
        kwargs_ns = ns.copy()

        # evaluate
        args = Expr(Head.call, [symbol(_tuple)] + _args).eval(args_ns)
        kwargs = Expr(Head.call, [symbol(dict)] + _kwargs).eval(kwargs_ns)
        return args, kwargs

    @classmethod
    def parse_setitem(cls, obj: Any, key: Any, value: Any) -> "Expr":
        """Parse ``obj[key] = value``."""
        target = cls(Head.getitem, [symbol(obj), symbol(key)])
        return cls(Head.assign, [target, symbol(value)])

    def split_setitem(self) -> "tuple[_Expr, Symbol, _Expr]":
        """Return ``obj, key, value`` if ``self`` is ``obj[key] = value``."""
        if self.head is not Head.assign:
            raise ValueError("Not a setitem expression.")
        target, value = self.args
        if not isinstance(target, Expr) or target.head is not Head.getitem:
            raise ValueError("Not a setitem expression.")
        obj, attr = target.args
        if not isinstance(attr, Symbol):
            raise RuntimeError("Unreachable in setitem expression.")
        return obj, attr, value

    @classmethod
    def parse_delitem(cls, obj: Any, key: Any) -> "Expr":
        """Parse ``del obj[key]``."""
        target = cls(Head.getitem, [symbol(obj), symbol(key)])
        return cls(Head.del_, [target])

    def split_delitem(self) -> "tuple[_Expr, Symbol]":
        """Return ``obj, key`` if ``self`` is ``del obj[key]``."""
        if self.head is not Head.del_:
            raise ValueError("Not a delitem expression.")
        arg = self.args[0]
        if not isinstance(arg, Expr) or arg.head is not Head.getitem:
            raise ValueError("Not a delitem expression.")
        obj, attr = arg.args
        if not isinstance(attr, Symbol):
            raise RuntimeError("Unreachable in setattr expression.")
        return obj, attr

    @classmethod
    def parse_setattr(cls, obj: Any, key: str, value: Any) -> "Expr":
        """Parse ``obj.key = value``."""
        target = cls(Head.getattr, [symbol(obj), Symbol(key)])
        return cls(Head.assign, [target, symbol(value)])

    def split_setattr(self) -> "tuple[_Expr, Symbol, _Expr]":
        """Return ``obj, key, value`` if ``self`` is ``obj.key = value``."""
        if self.head is not Head.assign:
            raise ValueError("Not a setattr expression.")
        target, value = self.args
        if not isinstance(target, Expr) or target.head is not Head.getattr:
            raise ValueError("Not a setattr expression.")
        obj, attr = target.args
        if not isinstance(attr, Symbol):
            raise RuntimeError("Unreachable in setattr expression.")
        return obj, attr, value

    @classmethod
    def parse_delattr(cls, obj: Any, key: str) -> "Expr":
        """Parse ``del obj.key``."""
        target = cls(Head.getattr, [symbol(obj), Symbol(key)])
        return cls(Head.del_, [target])

    def split_delattr(self) -> "tuple[_Expr, Symbol]":
        """Return ``obj, key`` if ``self`` is ``del obj.key``."""
        if self.head is not Head.del_:
            raise ValueError("Not a delattr expression.")
        arg = self.args[0]
        if not isinstance(arg, Expr) or arg.head is not Head.getattr:
            raise ValueError("Not a delattr expression.")
        obj, attr = arg.args
        if not isinstance(attr, Symbol):
            raise RuntimeError("Unreachable in setattr expression.")
        return obj, attr

    @classmethod
    def _convert_args(cls, args: "tuple[Any, ...]", kwargs: "dict[str, Any]") -> list:
        inputs = []
        for a in args:
            inputs.append(a)

        for k, v in kwargs.items():
            inputs.append(cls(Head.kw, [Symbol(k), symbol(v)]))
        return inputs

    @classmethod
    def parse_object(cls, a: Any) -> _Expr:
        """Convert an object into a macro-type."""
        return a if isinstance(a, cls) else symbol(a)

    def issetattr(self) -> bool:
        """Determine if an expression is in the form of ``setattr(obj, key value)``."""
        if self.head is Head.assign:
            target = self.args[0]
            if isinstance(target, Expr) and target.head is Head.getattr:
                return True
        return False

    def issetitem(self) -> bool:
        """Determine if an expression is in the form of ``setitem(obj, key value)``."""
        if self.head is Head.assign:
            target = self.args[0]
            if isinstance(target, Expr) and target.head is Head.getitem:
                return True
        return False

    def iter_args(self) -> "Iterator[Symbol]":
        """Recursively iterate along all the arguments."""
        for arg in self.args:
            if isinstance(arg, Expr):
                yield from arg.iter_args()
            elif isinstance(arg, Symbol):
                yield arg
            else:
                raise RuntimeError(f"{arg} (type {type(arg)})")

    def iter_expr(self) -> "Iterator[Expr]":
        """
        Recursively iterate over all the nested Expr, until reaching to non-nested Expr.

        This method is useful in macro generation.
        """
        yielded = False
        for arg in self.args:
            if isinstance(arg, Expr):
                yield from arg.iter_expr()
                yielded = True

        if not yielded:
            yield self

    def _split(self, head: Head) -> "list[Any]":
        if self.head is not head:
            raise ValueError(f"Expected {head}, got {self.head}.")
        left = self.args[0]
        right = self.args[1]
        if isinstance(left, Symbol):
            return [left, right]
        else:
            return left._split(head) + [right]

    def split_getattr(self) -> "list[Symbol]":
        """
        Split an expression into a list of get-attribute symbols.

        >>> expr = parse("a.b.c.d")
        >>> expr.split_getattr()  # [:a, :b, :c, :d]
        """
        return self._split(Head.getattr)

    def split_getitem(self) -> "list[Symbol]":
        """
        Split an expression into a list of get-item symbols/strings.

        >>> expr = parse("a['b']['c']['d']")
        >>> expr.split_getitem()  # [:a, :'b', :'c', :'d']
        """
        return self._split(Head.getitem)

    @classmethod
    def from_callble(cls, f: Callable):
        """Create function expression from the function itself."""
        from .ast import parse

        return parse(inspect.getsource(f))

    @overload
    def format(self, mapping: dict, inplace: bool = False) -> "Expr": ...

    @overload
    def format(
        self,
        mapping: "Iterable[tuple[Any, _Expr]]",
        inplace: bool = False,
    ) -> "Expr": ...

    def format(self, mapping, inplace=False) -> "Expr":
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

    def _unsafe_format(self, mapping: dict) -> "Expr":
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

    def iter_lines(self, allow_one_line: bool = True) -> "Iterator[Symbol | Expr]":
        """Iterate over all the lines in the expression."""
        if self.head not in HAS_BLOCK:
            if allow_one_line:
                yield self
                return
            else:
                raise ValueError(f"Expr of {self.head} does not have code block.")
        yield from _iter_lines(self)

    def iter_getattr(self) -> "Iterator[Expr]":
        """Iterate over all the get-attribute expression."""
        if self.head is Head.getattr:
            yield self
            return
        for arg in self.args:
            if isinstance(arg, Symbol):
                continue
            if arg.head is Head.getattr:
                yield arg
            else:
                yield from arg.iter_getattr()

    def iter_call(self) -> "Iterator[Expr]":
        """Iterate over all the function call."""
        if self.head is Head.call:
            yield self
        for arg in self.args:
            if isinstance(arg, Symbol):
                continue
            if arg.head is Head.call:
                yield arg
            else:
                yield from arg.iter_call()

    def iter_call_args(
        self,
    ) -> "Iterator[tuple[_Expr, tuple[_Expr, ...], dict[str, _Expr]]]":
        """Iterate over all the function call and its split."""
        for expr in self.iter_call():
            yield expr.split_call()


def _iter_lines(expr: Expr) -> "Iterator[Symbol | Expr]":
    for arg in expr.args:
        if isinstance(arg, Symbol):
            continue
        if arg.head is Head.block:
            yield from _iter_lines(arg)
        elif arg.head is Head.if_:  # = cond, if, else
            yield arg.args[0]
            assert isinstance(arg.args[1], Expr)
            yield from _iter_lines(arg.args[1])
            if len(arg.args) > 2:
                a2 = arg.args[2]
                if isinstance(a2, Expr):
                    yield from _iter_lines(a2)
                else:
                    yield a2
        elif arg.head in (Head.for_, Head.while_):  # = binop, block
            yield arg.args[0]
            assert isinstance(arg.args[1], Expr)
            yield from _iter_lines(arg.args[1])
        elif arg.head in (Head.function, Head.class_, Head.with_):
            assert isinstance(arg.args[1], Expr)
            yield from _iter_lines(arg.args[1])
        elif arg.head is Head.try_:
            assert isinstance(arg.args[0], Expr)
            yield from _iter_lines(arg.args[0])
            for a in arg.args[1:-2:2]:
                assert isinstance(a, Expr) and isinstance(a.args[0], Expr)
                yield from _iter_lines(a.args[0])
            _else, _finally = arg.args[-2:]
            if isinstance(_else, Expr) and _else.head is Head.block:
                assert isinstance(_else.args[0], Expr)
                yield from _iter_lines(_else.args[0])
            if isinstance(_finally, Expr) and _finally.head is Head.block:
                assert isinstance(_finally.args[0], Expr)
                yield from _iter_lines(_finally.args[0])
        elif arg.head is Head.decorator:
            assert isinstance(arg.args[1], Expr)
            yield from _iter_lines(arg.args[1])
        else:
            if arg.head is not Head.empty:
                yield arg


def _check_format_mapping(mapping_list: Iterable) -> "dict[Symbol, _Expr]":
    _dict: dict[Symbol, _Expr] = {}
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


def _tuple(*args) -> tuple:
    return args


# Stored symbols and the actual values. These will be used even after parse()
# method, to make parse(str(macro)) executable.
_STORED_VALUES: "dict[int, tuple[_Expr, Any]]" = {}
_STORED_SYMBOLS: "WeakValueDictionary[str, Any]" = WeakValueDictionary()

# Map to speed up type check
_SUBCLASS_MAP: "dict[type, type]" = {}


def symbol(obj: Any, constant: bool = True) -> _Expr:
    """Make a proper Symbol or Expr instance from any objects.

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
    _stored = False
    if constant and obj_id in _STORED_VALUES:
        seq: str | _Expr = _STORED_VALUES[obj_id][0]
        _stored = True
    elif not constant or obj_id in Symbol._variables:
        seq = Symbol.make_symbol_str(obj)
        constant = False
    elif obj_type in _TYPE_MAP:
        seq = _TYPE_MAP[obj_type](obj)
        _stored = True
    elif obj_type in _SUBCLASS_MAP:
        parent_type = _SUBCLASS_MAP[obj_type]
        seq = _TYPE_MAP[parent_type](obj)
        _stored = True
    elif isinstance(obj, Number):  # numpy scalars
        _TYPE_MAP[obj_type] = str
        seq = str(obj)
    elif callable(obj) and hasattr(obj, "__name__"):  # most custom callables
        seq = obj.__name__
    else:
        # search for subclasses
        for k, func in _TYPE_MAP.items():
            if isinstance(obj, k):
                seq = func(obj)
                _SUBCLASS_MAP[obj_type] = k
                break
        else:
            seq = Symbol.make_symbol_str(obj)
            constant = False

    if isinstance(obj, (ModuleType, FunctionType)) and not _stored:
        # Register modules/functions to the default namespace because users
        # always have to pass the object to the global variables when
        # calling the eval() function.
        *main, seq = obj.__name__.split(".")
        sym = Symbol(seq, obj_id)
        sym.constant = True
        if len(main) == 0:
            # submodules should not be registered
            _STORED_VALUES[obj_id] = (sym, obj)
            _STORED_SYMBOLS[sym.name] = obj
        return sym

    if isinstance(seq, (Symbol, Expr)):
        # The output of register_type can be a Symbol or Expr
        return seq
    else:
        sym = Symbol(seq, obj_id)
        sym.constant = constant
        return sym


def store(obj: Any) -> _Expr:
    """Store a variable in a global Symbol namespace."""
    sym = symbol(obj)
    if not isinstance(sym, Symbol):
        raise ValueError(f"Object {obj!r} was not converted into a Symbol.")
    obj_id = id(obj)
    name = sym.name
    if not name.isidentifier():
        raise ValueError(f"{name} is not an identifier.")
    _STORED_VALUES[obj_id] = (sym, obj)
    _STORED_SYMBOLS[sym.name] = obj
    return sym


def store_sequence(obj: Sequence[Any]) -> Symbol:
    """Store a sequence object and make its contents expressed as varXX[i]."""
    if not isinstance(obj, Sequence):
        raise TypeError(f"Expected tuple, got {type(obj)}.")
    obj_sym = Symbol.asvar(obj)
    for idx, each in enumerate(obj):
        expr = Expr(Head.getitem, [obj_sym, idx])
        _id = id(each)
        _STORED_VALUES[_id] = (expr, each)
    return obj_sym


def object_stored_at(id: int) -> Any:
    """Return the object stored at the given ID."""
    return _STORED_VALUES[id][1]


def symbol_stored_at(id: int) -> _Expr:
    """Return the Symbol object stored at the given ID."""
    return _STORED_VALUES[id][0]


class PrettyString(str):
    """Just for showing the dumped expression."""

    def _repr_pretty_(self, p, cycle):
        p.text(self)


register_type(tuple, lambda e: Expr(Head.tuple, [symbol(a) for a in e]))
register_type(list, lambda e: Expr(Head.list, [symbol(a) for a in e]))


@register_type(frozenset)
def _fronset_expr(e: frozenset):
    if len(e) == 0:
        return Expr(Head.call, [frozenset])
    else:
        return Expr(Head.call, [frozenset, set(e)])


@register_type(set)
def _set_expr(e: set):
    if len(e) == 0:
        return Expr(Head.call, [set])
    else:
        return Expr(Head.braces, [symbol(a) for a in e])


@register_type(dict)
def _dict_expr(e: dict):
    return Expr(
        Head.braces,
        [Expr(Head.annotate, [symbol(k), symbol(v)]) for k, v in e.items()],
    )


register_type(memoryview, lambda e: Expr(Head.call, [memoryview, e.tobytes()]))
register_type(bytearray, lambda e: Expr(Head.call, [bytearray, bytes(e)]))
register_type(slice, lambda e: Expr(Head.call, [slice, e.start, e.stop, e.step]))
register_type(range, lambda e: Expr(Head.call, [range, e.start, e.stop, e.step]))

# install builtin variables
for name, obj in builtins.__dict__.items():
    if name.startswith("_") or isinstance(obj, ModuleType):
        continue
    _STORED_VALUES[id(obj)] = (symbol(obj), obj)
