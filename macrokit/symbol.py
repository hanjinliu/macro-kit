from __future__ import annotations
import inspect
from typing import Callable, Any, TypeVar, overload, TypedDict
from types import FunctionType, BuiltinFunctionType, ModuleType, MethodType

T = TypeVar("T")

class SymbolDict(TypedDict):
    name: str
    object_id: int
    constant: bool

class Symbol:
    # Map of how to convert object into a symbol.
    _type_map: dict[type[T], Callable[[T], Any]] = {
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
    _modules: dict[int, ModuleType] = {}
    
    def __init__(self, seq: str, object_id: int = None):
        self._name = str(seq)
        self.object_id = object_id or id(seq)
        self.constant = True
    
    def asdict(self) -> SymbolDict:
        return {"name", self.name, 
                "object_id", self.object_id, 
                "constant", self.constant
                }
    
    @property
    def name(self) -> str:
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
        # To ensure Symbol to be hash safe, we have to consider both object ID and whether
        # the Symbol is a constant because object ID of a variable is defined by the hash
        # value of the identifier. 
        return self.object_id * 2 + int(self.constant)
    
    def __eq__(self, other: Symbol) -> bool:
        if not isinstance(other, Symbol):
            raise TypeError(f"'==' is not supported between Symbol and {type(other)}")
        return (self.object_id == other.object_id and 
                self.constant == other.constant)
    
    @classmethod
    def var(cls, identifier: str):
        """
        Make a variable symbol. Same indentifier with same type always returns identical symbol.
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
    
    def as_parameter(self, default=inspect._empty) -> inspect.Parameter:
        return inspect.Parameter(self._name, 
                                 inspect.Parameter.POSITIONAL_OR_KEYWORD, 
                                 default=default)
        
    @overload
    @classmethod
    def register_type(cls, type: type[T]) -> Callable[[Callable[[T], Any]], Callable[[T], Any]]: ...
    
    @overload
    @classmethod
    def register_type(cls, function: Callable[[T], Any])-> Callable[[type[T]], type[T]]: ...
    
    @overload
    @classmethod
    def register_type(cls, type: type[T], function: Callable[[T], Any]): ...
    
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
            raise TypeError("Arguments of 'register_type' must be type and/or function.")
        
register_type = Symbol.register_type
