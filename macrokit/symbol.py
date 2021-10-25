from __future__ import annotations
import inspect
from enum import Enum
from pathlib import Path
from typing import Callable, Any, TypeVar
from types import FunctionType, BuiltinFunctionType, ModuleType

T = TypeVar("T")

# TODO: method of making identical Symbols from string. Maybe using hash(str)?

class Symbol:
    # Map of how to convert object into a symbol.
    _type_map: dict[type, Callable[[Any], str]] = {
        type: lambda e: e.__name__,
        FunctionType: lambda e: e.__name__,
        BuiltinFunctionType: lambda e: e.__name__,
        ModuleType: lambda e: e.__name__.split(".")[-1],
        Enum: lambda e: repr(str(e.name)),
        Path: lambda e: f"r'{e}'",
        type(None): lambda e: "None",
    }
    
    # ID of global variables
    _variables: set[int] = set()
    
    def __init__(self, seq: str, object_id: int = None, type: type = Any):
        self.name = str(seq)
        self.object_id = object_id or id(seq)
        self.type = type
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
        return self.object_id == other.object_id
    
    def as_parameter(self, default=inspect._empty):
        return inspect.Parameter(self._name, 
                                 inspect.Parameter.POSITIONAL_OR_KEYWORD, 
                                 default=default,
                                 annotation=self.type)
    
    def replace(self, other: Symbol) -> None:
        self.name = other.name
        self.object_id = other.object_id
        self.type = other.type
        self.constant = other.constant
        return None
    
    @classmethod
    def register_type(cls, type: type[T], function: Callable[[T], str|Symbol]):
        if not callable(function):
            raise TypeError("The second argument must be callable.")
        cls._type_map[type] = function
        

def register_type(type: type[T], function: Callable[[T], str|Symbol]):
    return Symbol.register_type(type, function)
        