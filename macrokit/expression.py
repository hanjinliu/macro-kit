from copy import deepcopy
from numbers import Number
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator, overload, Union, List, Tuple, Dict
import inspect

from macrokit.head import EXEC, Head
from macrokit._validator import validator
from macrokit._symbol import Symbol


def str_(expr: Any, indent: int = 0):
    """Convert expr into a proper string."""
    if isinstance(expr, Expr):
        return _STR_MAP[expr.head](expr, indent)
    else:
        return " " * indent + str(expr)


def str_lmd(expr: Any, indent: int = 0):
    """Convert str into a proper lambda function definition."""
    s = str(expr)
    call = s.lstrip("<lambda>(").rstrip(")")
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


_STR_MAP: Dict[Head, Callable[["Expr", int], str]] = {
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
    Head.lambda_: lambda e, i: f"{str_lmd(e.args[0], i)}: {str_(e.args[1])}",  # noqa
    Head.return_: lambda e, i: f"{_s_(i)}return {sjoin(', ', e.args)}",
    Head.raise_: lambda e, i: f"{_s_(i)}raise {str_(e.args[0])}",
    Head.if_: lambda e, i: f"{_s_(i)}if {rm_par(str_(e.args[0]))}:\n{str_(e.args[1], i+4)}\n{_s_(i)}else:\n{str_(e.args[2], i+4)}",  # noqa
    Head.elif_: lambda e, i: f"{_s_(i)}if {rm_par(str_(e.args[0]))}:\n{str_(e.args[1], i+4)}\n{_s_(i)}else:\n{str_(e.args[2], i+4)}",  # noqa
    Head.for_: lambda e, i: f"{_s_(i)}for {rm_par(str_(e.args[0]))}:\n{str_(e.args[1], i+4)}",  # noqa
    Head.while_: lambda e, i: f"{_s_(i)}while {rm_par(str_(e.args[0]))}:\n{str_(e.args[1], i+4)}",  # noqa
    Head.annotate: lambda e, i: f"{str_(e.args[0], i)}: {str_(e.args[1])}",
}

_Expr = Union[Symbol, "Expr"]


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
    def args(self) -> List[_Expr]:
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
        _glb: Dict[str, Any] = {
            (sym.name if isinstance(sym, Symbol) else sym): v
            for sym, v in _globals.items()
        }

        # use registered modules
        if Symbol._stored_symbols:
            format_dict: Dict[Symbol, _Expr] = {}
            for id_, sym in Symbol._stored_symbols.items():
                mod = Symbol._stored_variable_map[sym.name]
                vstr = Symbol.symbol_str_for_id(id_)
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
        args: Tuple[Any, ...] = None,
        kwargs: dict = None,
    ) -> "Expr":
        """Parse ``obj.func(*args, **kwargs)``."""
        method = cls(head=Head.getattr, args=[symbol(obj), func])
        return cls.parse_call(method, args, kwargs)

    @classmethod
    def parse_init(
        cls,
        obj: Any,
        init_cls: Union[type, "Expr"] = None,
        args: Tuple[Any, ...] = None,
        kwargs: dict = None,
    ) -> "Expr":
        """Parse ``obj = init_cls(*args, **kwargs)``."""
        if init_cls is None:
            init_cls = type(obj)
        sym = symbol(obj)
        return cls(Head.assign, [sym, cls.parse_call(init_cls, args, kwargs)])

    @classmethod
    def parse_call(
        cls,
        func: Union[Callable, _Expr],
        args: Tuple[Any, ...] = None,
        kwargs: dict = None,
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

    def split_call(self) -> Tuple[_Expr, Tuple[_Expr, ...], Dict[str, _Expr]]:
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

    def split_method(
        self,
    ) -> Tuple[_Expr, Symbol, Tuple[_Expr, ...], Dict[str, _Expr]]:
        """Split ``obj.func(*args, **kwargs)`` to (obj, func, args, kwargs)."""
        fn, args, kwargs = self.split_call()
        if not isinstance(fn, Expr) or fn.head is not Head.getattr:
            raise ValueError("Not a method call.")
        obj, attr = fn.args
        if not isinstance(attr, Symbol):
            raise RuntimeError("Unreachable in setitem expression.")
        return obj, attr, args, kwargs

    def eval_call_args(
        self,
        ns: dict = {},
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
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
        for i, arg in enumerate(arguments):
            if isinstance(arg, Expr) and arg.head is Head.kw:
                break

        _args = arguments[:i]
        _kwargs = arguments[i:]

        # prepare namespaces
        args_ns: dict[Union[str, _Expr], Any] = ns.copy()
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

    def split_setitem(self) -> Tuple[_Expr, Symbol, _Expr]:
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

    def split_delitem(self) -> Tuple[_Expr, Symbol]:
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

    def split_setattr(self) -> Tuple[_Expr, Symbol, _Expr]:
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

    def split_delattr(self) -> Tuple[_Expr, Symbol]:
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
    def _convert_args(cls, args: Tuple[Any, ...], kwargs: dict) -> list:
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

    def iter_args(self) -> Iterator[Symbol]:
        """Recursively iterate along all the arguments."""
        for arg in self.args:
            if isinstance(arg, Expr):
                yield from arg.iter_args()
            elif isinstance(arg, Symbol):
                yield arg
            else:
                raise RuntimeError(f"{arg} (type {type(arg)})")

    def iter_expr(self) -> Iterator["Expr"]:
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

    def _split(self, head: Head) -> List[Any]:
        if self.head is not head:
            raise ValueError(f"Expected {head}, got {self.head}.")
        left = self.args[0]
        right = self.args[1]
        if isinstance(left, Symbol):
            return [left, right]
        else:
            return left._split(head) + [right]

    def split_getattr(self) -> List[Symbol]:
        """
        Split an expression into a list of get-attribute symbols.

        >>> expr = parse("a.b.c.d")
        >>> expr.split_getattr()  # [:a, :b, :c, :d]
        """
        return self._split(Head.getattr)

    def split_getitem(self) -> List[Symbol]:
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
    def format(self, mapping: dict, inplace: bool = False) -> "Expr":
        ...

    @overload
    def format(
        self,
        mapping: Iterable[Tuple[Any, _Expr]],
        inplace: bool = False,
    ) -> "Expr":
        ...

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


def _check_format_mapping(mapping_list: Iterable) -> Dict[Symbol, _Expr]:
    _dict: Dict[Symbol, _Expr] = {}
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


def symbol(obj: Any, constant: bool = True) -> _Expr:
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
        seq: Union[_Expr, str] = Symbol.make_symbol_str(obj)
        constant = False
    elif obj_id in Symbol._stored_symbols.keys():
        seq = Symbol._stored_symbols[obj_id]
    elif obj_type in Symbol._type_map:
        seq = Symbol._type_map[obj_type](obj)
    elif obj_type in Symbol._subclass_map:
        parent_type = Symbol._subclass_map[obj_type]
        seq = Symbol._type_map[parent_type](obj)
    elif isinstance(obj, tuple):
        if len(obj) == 1:
            # length 1 tuple have to be written as (a,) instead of (a).
            seq = f"({symbol(obj[0])},)"
        else:
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
        if len(obj) == 0:
            if obj_type is set:
                seq = "set()"
            else:
                seq = f"{obj_type.__name__}()"
        else:
            seq = "{" + ", ".join(str(symbol(a)) for a in obj) + "}"
            if obj_type is not set:
                seq = f"{obj_type.__name__}({seq})"
    elif isinstance(obj, frozenset):
        seq = ", ".join(str(symbol(a)) for a in obj)
        if obj_type is frozenset:
            seq = f"frozenset({{{seq}}})"
        else:
            seq = f"{obj_type.__name__}({{{seq}}})"
    elif isinstance(obj, Number):  # int, float, bool, ...
        seq = str(obj)
    elif isinstance(obj, ModuleType):
        # Register module to the default namespace of Symbol class. This function is
        # called every time a module type object is converted to a Symbol because users
        # always have to pass the module object to the global variables when calling
        # eval function.
        if obj_id in Symbol._stored_symbols.keys():
            sym = Symbol._stored_symbols[obj_id]
        else:
            *main, seq = obj.__name__.split(".")
            sym = Symbol(seq, obj_id)
            sym.constant = True
            if len(main) == 0:
                # submodules should not be registered
                Symbol._stored_symbols[obj_id] = sym
                Symbol._stored_variable_map[seq] = obj
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
            seq = Symbol.make_symbol_str(obj)
            constant = False

    if isinstance(seq, (Symbol, Expr)):
        # The output of register_type can be a Symbol or Expr
        return seq
    else:
        sym = Symbol(seq, obj_id)
        sym.constant = constant
        return sym


def store(obj: Any) -> None:
    """Store a variable in a global Symbol namespace."""
    sym = symbol(obj)
    if not isinstance(sym, Symbol):
        raise ValueError(f"Object {obj!r} was not converted into a Symbol.")
    obj_id = id(obj)
    name = sym.name
    if not name.isidentifier():
        raise ValueError(f"{name} is not an identifier.")
    if name in Symbol._stored_variable_map:
        _var = Symbol._stored_variable_map[name]
        raise ValueError(f"Variable identifier {name} collides with {_var!r}")
    Symbol._stored_symbols[obj_id] = sym
    Symbol._stored_variable_map[name] = obj
    return None
