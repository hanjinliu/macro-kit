from contextlib import contextmanager
from types import BuiltinFunctionType, FunctionType, MethodType
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Type,
    TypeVar,
    overload,
    Optional,
)

# Map of how to convert object into a symbol.
_TYPE_MAP: Dict[type, Callable[[Any], str]] = {
    type: lambda e: e.__name__,
    FunctionType: lambda e: e.__name__,
    BuiltinFunctionType: lambda e: e.__name__,
    MethodType: lambda e: e.__name__,
    type(None): lambda e: "None",
    type(...): lambda e: "...",
    type(NotImplemented): lambda e: "NotImplemented",
    str: repr,
    bytes: repr,
    int: str,
    float: str,
    complex: str,
    bool: str,
}

T = TypeVar("T")


@overload
def register_type(type: Type[T], function: Optional[Callable[[T], Any]]) -> None:
    ...


@overload
def register_type(type: Type[T]) -> Callable[[Callable[[T], Any]], Callable[[T], Any]]:
    ...


@overload
def register_type(func: Callable[[T], Any]) -> Callable[[Type[T]], Type[T]]:
    ...


def register_type(type_or_function, function=None):
    """
    Define a dispatcher for macro recording.

    >>> register_type(np.ndarray, lambda arr: str(arr.tolist()))

    or

    >>> @register_type(np.ndarray)
    >>> def _(arr):
    ...     return str(arr.tolist())

    or if you defined a new type

    >>> @register_type(lambda t: t.name)
    >>> class T:
    ...     ...

    """
    if isinstance(type_or_function, type):

        def _register_function(func: Callable[[T], Any]):
            if not callable(func):
                raise TypeError("The second argument must be callable.")
            _TYPE_MAP[type_or_function] = func
            return func

        return _register_function if function is None else _register_function(function)

    else:
        if function is not None or not callable(type_or_function):
            raise TypeError("'register_type' must take type or function as arguments.")

        def _register_type(type_: Type[T]) -> Type[T]:
            if not isinstance(type_, type):
                raise TypeError(f"Type expected, got {type(type_)}")
            _TYPE_MAP[type_] = type_or_function
            return type_

        return _register_type


def unregister_type(type_: Type[T], raises: bool = True) -> None:
    """Unregister a type."""
    if _TYPE_MAP.pop(type_, None) is None and raises:
        raise KeyError(f"{type_} is not registered.")


@contextmanager
def type_registered(typemap: Mapping[Type[T], Callable[[T], Any]]):
    """Register types in a dictionary."""
    _old_type_map = _TYPE_MAP.copy()
    _TYPE_MAP.update(typemap)
    try:
        yield
    finally:
        _TYPE_MAP.clear()
        _TYPE_MAP.update(_old_type_map)
