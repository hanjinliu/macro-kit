from __future__ import annotations

import inspect
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType
from typing import Any, Callable, TypeVar, overload

from typing_extensions import TypedDict

T = TypeVar("T")


class SymbolDict(TypedDict):
    """Dictionary representin a Symbol object."""

    name: str
    object_id: int
    constant: bool


class Symbol:
    """A class that represents Python symbol in the context of metaprogramming."""

    # Map of how to convert object into a symbol.
    _type_map: dict[type, Callable] = {
        type: lambda e: e.__name__,
        FunctionType: lambda e: e.__name__,
        BuiltinFunctionType: lambda e: e.__name__,
        MethodType: lambda e: e.__name__,
        type(None): lambda e: "None",
        str: repr,
        bytes: repr,
        slice: lambda e: f"slice({e.start}, {e.stop}, {e.step})",
        int: str,
        float: str,
        complex: str,
        bool: str,
    }

    # Map to speed up type check
    _subclass_map: dict[type, type] = {}

    # ID of global variables
    _variables: set[int] = set()

    # Module symbols
    _module_symbols: dict[int, Symbol] = {}
    _module_map: dict[str, ModuleType] = {}

    def __init__(self, seq: Any, object_id: int = None):
        self._name = str(seq)
        self.object_id = object_id or id(seq)
        self.constant = True

    def asdict(self) -> SymbolDict:
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

    @overload
    @classmethod
    def register_type(cls, type: type[T], function: Callable[[T], Any] | None) -> None:
        ...

    @overload
    @classmethod
    def register_type(
        cls, type: type[T]
    ) -> Callable[[Callable[[T], Any]], Callable[[T], Any]]:
        ...

    @overload
    @classmethod
    def register_type(cls, func: Callable[[T], Any]) -> Callable[[type[T]], type[T]]:
        ...

    @classmethod
    def register_type(cls, type_or_function, function=None):
        """
        Define a dispatcher for macro recording.

        .. code-block:: python

            register_type(np.ndarray,
                          lambda arr: str(arr.tolist())
                          )

        or

        .. code-block:: python

            @register_type(np.ndarray)
            def _(arr):
                return str(arr.tolist())

        or if you defined a new type

        .. code-block:: python

            @register_type(lambda t: t.name)
            class T:
                ...

        """
        if isinstance(type_or_function, type):

            def _register_function(func: Callable[[T], Any]):
                if not callable(func):
                    raise TypeError("The second argument must be callable.")
                cls._type_map[type_or_function] = func
                return func

            return (
                _register_function if function is None else _register_function(function)
            )

        else:
            if function is not None or not callable(type_or_function):
                raise TypeError(
                    "'register_type' must take type or function as arguments."
                )

            def _register_type(type_: type[T]) -> type[T]:
                if not isinstance(type_, type):
                    raise TypeError(f"Type expected, got {type(type_)}")
                cls._type_map[type_] = type_or_function
                return type_

            return _register_type


register_type = Symbol.register_type
