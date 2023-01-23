import inspect
from typing import Any, Dict, Set

_OFFSET = None


class Symbol:
    """A class that represents Python symbol in the context of metaprogramming."""

    # ID of global variables
    _variables: Set[int] = set()

    def __init__(self, seq: Any, object_id: int = None):
        self._name = str(seq)
        self.object_id = object_id or id(seq)
        self.constant = True

    def asdict(self) -> Dict[str, Any]:
        """Convert Symbol object into a dict."""
        return {
            "name": self.name,
            "object_id": self.object_id,
            "constant": self.constant,
        }

    @property
    def name(self) -> str:
        """Symbol name as a string."""
        return self._name

    @name.setter
    def name(self, newname: str):
        if not isinstance(newname, str):
            raise TypeError(f"Cannot set non-string to name: {newname!r}.")
        self._name = newname

    def eval(self, _globals: dict = {}, _locals: dict = {}) -> Any:
        """
        Evaluate symbol.

        Parameters
        ----------
        _globals, _locals : dict, optional
            Just for consistency with ``Expr.eval``.
        """
        _globals = {str(k): v for k, v in _globals.items()}
        _locals = _locals.copy()
        if not _globals:
            from .expression import _STORED_VALUES

            out = _STORED_VALUES.get(self.object_id, None)
            if out is not None:
                _globals[self.name] = out[1]
        out = eval(self.name, _globals, _locals)
        return out

    def __repr__(self) -> str:
        """Return a Julia-like repr."""
        if self.constant:
            return self._name
        else:
            return ":" + self._name

    def __str__(self) -> str:
        """Symbol name as a string."""
        return self._name

    def __hash__(self) -> int:
        """Hashed by object ID and whether object is a constant."""
        # To ensure Symbol to be hash safe, we have to consider both object ID and
        # whether the Symbol is a constant because object ID of a variable is defined
        # by the hash value of the identifier.
        return self.object_id * 2 + int(self.constant)

    def __eq__(self, other) -> bool:
        """Return true only if Symbol with same object is given."""
        if not isinstance(other, Symbol):
            return False
        return self.object_id == other.object_id and self.constant == other.constant

    @classmethod
    def var(cls, identifier: str):
        """
        Make a variable symbol.

        Same indentifier with same type always returns identical symbol.
        """
        if not isinstance(identifier, str):
            raise TypeError("'identifier' must be str")
        elif not identifier.isidentifier():
            raise ValueError(f"'{identifier}' is not a valid identifier.")
        self = cls(identifier, hash(identifier))
        self.constant = False
        return self

    @classmethod
    def _reserved(cls, identifier: str):
        # Users should never use this!!
        self = cls(identifier, hash(identifier))
        self.constant = False
        return self

    def as_parameter(self, default=inspect.Parameter.empty) -> inspect.Parameter:
        """Convert symbol as an ``inspect.Parameter`` object."""
        return inspect.Parameter(
            self._name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default
        )

    @classmethod
    def make_symbol_str(cls, obj: Any) -> str:
        """Make a string for symbol."""
        # hexadecimals are easier to distinguish
        _id = id(obj)
        if obj is not None:
            cls._variables.add(_id)
        return cls.symbol_str_for_id(_id)

    @classmethod
    def symbol_str_for_id(cls, _id: int) -> str:
        """Create a symbol string based on an ID."""
        global _OFFSET
        if _OFFSET is None:
            _OFFSET = _id
        a, rem = divmod(_id - _OFFSET, 16)
        if rem == 0:
            if a >= 0:
                return f"var{hex(a)[2:]}"
            else:
                return f"var0{hex(a)[3:]}"
        else:
            if a >= 0:
                return f"var{hex(a)[2:]}_{rem}"
            else:
                return f"var0{hex(a)[3:]}_{rem}"

    @classmethod
    def asvar(cls, obj: Any) -> "Symbol":
        """Convert input object as a variable."""
        out = Symbol(Symbol.make_symbol_str(obj), id(obj))
        out.constant = False
        return out


try:
    import cython
except ImportError:  # pragma: no cover
    COMPILED: bool = False
else:  # pragma: no cover
    try:
        COMPILED = cython.compiled
    except AttributeError:
        COMPILED = False
