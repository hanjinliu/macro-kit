from __future__ import annotations
from typing import Callable, Any, Hashable, TypeVar, Generic
from .symbol import Symbol
from .head import Head

_T = TypeVar("_T", bound=Hashable)
_A = TypeVar("_A")

class Validator(Generic[_T, _A]):
    """
    A validator class that will be used for Expr argument validation.
    """
    def __init__(self):
        self._map: dict[_T, Callable[[_A], _A]] = {}
    
    def register(self, value: _T):
        def wrapper(func: Callable[[_A], _A]):
            self._map[value] = func
            return func
        return wrapper
            
    def __call__(self, arg: _T, *args: _A) -> _A:
        try:
            func = self._map[arg]
        except KeyError:
            return arg
        try:
            out = func(*args)
        except ValidationError as e:
            e.args = (f"{args} is incompatible with {arg}",)
            raise e
        return out

class ValidationError(ValueError):
    pass
    
validator = Validator[Head, list[Any]]()

@validator.register(Head.empty)
def _(args):
    if len(args) != 0:
        raise ValidationError()
    return args

@validator.register(Head.del_)
@validator.register(Head.comment)
def _single_arg(args):
    if len(args) != 1:
        raise ValidationError()
    return args

@validator.register(Head.assert_)
@validator.register(Head.getitem)
def _two_args(args):
    if len(args) != 2:
        raise ValidationError()
    return args

@validator.register(Head.getattr)
def _getattr(args):
    if len(args) != 2:
        raise ValidationError()
    k = args[1]
    if isinstance(k, Symbol):
        k.name = k.name.strip("'")
    return args

@validator.register(Head.assign)
@validator.register(Head.kw)
@validator.register(Head.annotate)
def _symbol_and_any(args):
    if len(args) != 2:
        raise ValidationError()
    k, v = args
    if isinstance(k, str):
        k = Symbol.var(k)
    elif isinstance(k, Symbol) and k.constant:
        k = Symbol.var(k.name)
    return [k, v]

@validator.register(Head.binop)
def _three_args(args):
    if len(args) != 3:
        raise ValidationError()
    return args

@validator.register(Head.function)
@validator.register(Head.for_)
def _an_arg_and_a_block(args):
    if len(args) != 2:
        raise ValidationError()
    b = args[1]
    if getattr(b, "head", None) != Head.block:
        raise ValidationError()
    return args

@validator.register(Head.if_)
@validator.register(Head.elif_)
def _two_args_and_a_block(args):
    if len(args) != 3:
        raise ValidationError()
    b = args[2]
    if getattr(b, "head", None) != Head.block:
        raise ValidationError()
    return args
