from __future__ import annotations
import inspect
from enum import Enum
from pathlib import Path
from typing import Callable, Any, TypeVar
from types import FunctionType, BuiltinFunctionType, ModuleType
import numpy as np

T = TypeVar("T")

class Symbol:
    
    # Map of how to convert object into a symbol.
    _type_map: dict[type, Callable[[Any], str]] = {
        type: lambda e: e.__name__,
        FunctionType: lambda e: e.__name__,
        BuiltinFunctionType: lambda e: e.__name__,
        ModuleType: lambda e: e.__name__,
        Enum: lambda e: repr(str(e.name)),
        Path: lambda e: f"r'{e}'",
        type(None): lambda e: "None",
    }
    
    def __init__(self, seq: str, object_id: int = None, type: type = Any):
        self.data = str(seq)
        self.object_id = object_id or id(seq)
        self.type = type
        self.valid = True
    
    def __repr__(self) -> str:
        return self.data
    
    def __str__(self) -> str:
        return self.data
    
    def __hash__(self) -> int:
        return self.object_id
    
    def __eq__(self, other: Symbol) -> bool:
        if not isinstance(other, Symbol):
            raise TypeError(f"'==' is not supported between Symbol and {type(other)}")
        return self.object_id == other.object_id
    
    def as_parameter(self, default=inspect._empty):
        return inspect.Parameter(self.data, 
                                 inspect.Parameter.POSITIONAL_OR_KEYWORD, 
                                 default=default,
                                 annotation=self.type)
    
    @classmethod
    def register_type(cls, type: type[T], function: Callable[[T], str]):
        if not callable(function):
            raise TypeError("The second argument must be callable.")
        cls._type_map[type] = function
        
def symbol(obj: Any) -> Symbol:
    if isinstance(obj, Symbol):
        return obj
    
    valid = True
    objtype = type(obj)
    if isinstance(obj, str):
        seq = repr(obj)
    elif np.isscalar(obj): # int, float, bool, ...
        seq = obj
    elif isinstance(obj, tuple):
        seq = "(" + ", ".join(symbol(a).data for a in obj) + ")"
        if objtype is not tuple:
            seq = objtype.__name__ + seq
    elif isinstance(obj, list):
        seq = "[" + ", ".join(symbol(a).data for a in obj) + "]"
        if objtype is not list:
            seq = f"{objtype.__name__}({seq})"
    elif isinstance(obj, dict):
        seq = "{" + ", ".join(f"{symbol(k)}: {symbol(v)}" for k, v in obj.items()) + "}"
        if objtype is not dict:
            seq = f"{objtype.__name__}({seq})"
    elif isinstance(obj, set):
        seq = "{" + ", ".join(symbol(a).data for a in obj) + "}"
        if objtype is not set:
            seq = f"{objtype.__name__}({seq})"
    elif isinstance(obj, slice):
        seq = f"{objtype.__name__}({obj.start}, {obj.stop}, {obj.step})"
    elif objtype in Symbol._type_map:
        seq = Symbol._type_map[objtype](obj)
    else:
        for k, func in Symbol._type_map.items():
            if isinstance(obj, k):
                seq = func(obj)
                break
        else:
            seq = f"var{hex(id(obj))}" # hexadecimals are easier to distinguish
            valid = False
            
    sym = Symbol(seq, id(obj), type(obj))
    sym.valid = valid
    return sym

def register_type(type: type[T], function: Callable[[T], str]):
    return Symbol.register_type(type, function)
        