from __future__ import annotations
from functools import wraps
import inspect
from typing import Callable, Any, TypeVar, overload
from types import FunctionType, BuiltinFunctionType, ModuleType, MethodType

T = TypeVar("T")

class Symbol:
    # Map of how to convert object into a symbol.
    _type_map: dict[type[T], Callable[[T], Any]] = {
        type: lambda e: e.__name__,
        FunctionType: lambda e: e.__name__,
        BuiltinFunctionType: lambda e: e.__name__,
        MethodType: lambda e: e.__name__,
        ModuleType: lambda e: e.__name__.split(".")[-1],
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
    
    def __init__(self, seq: str, object_id: int = None):
        self.name = str(seq)
        self.object_id = object_id or id(seq)
        self.constant = True
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, newname: str):
        if not isinstance(newname, str):
            raise TypeError(f"Cannot set non-string to name: {newname!r}.")
        self._name = newname
    
    def __repr__(self) -> str:
        if self.constant:
            return self._name
        else:
            return ":" + self._name
    
    def __str__(self) -> str:
        return self._name
    
    def __hash__(self) -> int:
        return self.object_id
    
    def __eq__(self, other: Symbol) -> bool:
        if not isinstance(other, Symbol):
            raise TypeError(f"'==' is not supported between Symbol and {type(other)}")
        return (self.object_id == other.object_id and 
                self.constant == other.constant)
    
    @classmethod
    def var(cls, identifier: str, type: type = object):
        """
        Make a variable symbol. Same indentifier with same type always returns identical symbol.
        """        
        if not isinstance(identifier, str):
            raise TypeError("'identifier' must be str")
        elif not identifier.isidentifier():
            raise ValueError(f"'{identifier}' is not a valid identifier.")
        self = cls(identifier, 0)
        self.object_id = hash(identifier)
        self.constant = False
        return self
    
    def as_parameter(self, default=inspect._empty):
        return inspect.Parameter(self._name, 
                                 inspect.Parameter.POSITIONAL_OR_KEYWORD, 
                                 default=default)
    
    def replace(self, other: Symbol) -> None:
        self.name = other.name
        self.object_id = other.object_id
        self.constant = other.constant
        return None
    
    @overload
    @classmethod
    def register_type(cls, type: type[T]) -> Callable[[Callable[[T], Any]], Callable[[T], Any]]: ...
    
    @overload
    @classmethod
    def register_type(cls, function: Callable[[T], Any])-> Callable[[type[T]], type[T]]: ...
    
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
            def _register(func):
                if not callable(func):
                    raise TypeError("The second argument must be callable.")
                cls._type_map[type_or_function] = func
                return func
            return _register if function is None else _register(function)
        
        elif isinstance(type_or_function, Callable):
            if function is not None:
                raise TypeError("")
            def _register(type_):
                cls._type_map[type_] = type_or_function
                return type_
            return _register
        
        else:
            raise TypeError()
        
@wraps(Symbol.register_type)
def register_type(type_or_function, function = None):
    return Symbol.register_type(type_or_function, function)

del wraps