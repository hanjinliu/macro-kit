from __future__ import annotations
import ast
from functools import singledispatch
import inspect
from typing import Callable

from .symbol import Symbol
from .expression import Expr, Head, symbol


AST_BINOP_MAP = {
    ast.Add: Symbol("+"),
    ast.Sub: Symbol("-"),
    ast.Mult: Symbol("*"),
    ast.Div: Symbol("/"),
    ast.Eq: Symbol("=="),
    ast.NotEq: Symbol("!="),
    ast.Gt: Symbol(">"),
    ast.GtE: Symbol(">="),
    ast.Lt: Symbol("<"),
    ast.LtE: Symbol("<="),
    ast.Is: Symbol("is"),
    ast.IsNot: Symbol("is not"),
    ast.In: Symbol("in"),
    ast.NotIn: Symbol("not in"),
    ast.Pow: Symbol("**"),
    ast.MatMult: Symbol("@"),
    ast.FloorDiv: Symbol("//"),
    ast.BitAnd: Symbol("&"),
    ast.BitOr: Symbol("|"),
    ast.BitXor: Symbol("^"),
    ast.And: Symbol("and"),
    ast.Or: Symbol("or"),
}

NoneType = type(None)

def parse(source: str | Callable) -> Expr | Symbol:
    """
    Convert Python code string into Expr/Symbol objects.
    """
    if callable(source):
        source = inspect.getsource(source)
    body = ast.parse(source).body
    if len(body) == 1:
        ast_object = ast.parse(source).body[0]
    else:
        ast_object = ast.parse(source).body
    
    return from_ast(ast_object)

@singledispatch
def from_ast(ast_object: ast.AST | list | NoneType):
    """
    Convert AST object to macro-kit object.
    """    
    raise NotImplementedError(f"AST type {type(ast_object)} cannot be converted now.")

@from_ast.register
def _(ast_object: NoneType):
    return None

@from_ast.register
def _(ast_object: ast.Expr):
    return from_ast(ast_object.value)

@from_ast.register
def _(ast_object: ast.Constant):
    return symbol(ast_object.value)

@from_ast.register
def _(ast_object: ast.Name):
    return Symbol(ast_object.id)

@from_ast.register
def _(ast_object: ast.Expr):
    return from_ast(ast_object.value)

@from_ast.register
def _(ast_object: ast.Call):
    head = Head.call
    args = [from_ast(ast_object.func)] + [from_ast(k) for k in ast_object.args] + \
        [Expr(Head.kw, [Symbol(k.arg), from_ast(k.value)]) for k in ast_object.keywords]
    return Expr(head, args)

@from_ast.register
def _(ast_object: ast.Assign):
    head = Head.assign
    if len(ast_object.targets) != 1:
        target = tuple(from_ast(x) for x in ast_object.targets)
    else:
        target = from_ast(ast_object.targets[0])
    args = [target, from_ast(ast_object.value)]
    return Expr(head, args)

@from_ast.register
def _(ast_object: ast.Attribute):
    head = Head.getattr
    args = [from_ast(ast_object.value), Symbol(ast_object.attr)]
    return Expr(head, args)

@from_ast.register
def _(ast_object: ast.Subscript):
    head = Head.getitem
    args = [from_ast(ast_object.value), from_ast(ast_object.slice)]
    return Expr(head, args)

@from_ast.register
def _(ast_object: ast.List):
    return symbol([from_ast(k) for k in ast_object.elts])

@from_ast.register
def _(ast_object: ast.Tuple):
    return symbol(tuple(from_ast(k) for k in ast_object.elts))

@from_ast.register
def _(ast_object: ast.Set):
    return symbol(set(from_ast(k) for k in ast_object.elts))

@from_ast.register
def _(ast_object: ast.Slice):
    return symbol(slice(from_ast(ast_object.lower), 
                        from_ast(ast_object.upper),
                        from_ast(ast_object.step))
                  )

@from_ast.register
def _(ast_object: ast.Dict):
    return symbol({from_ast(k): from_ast(v) for k, v in zip(ast_object.keys, ast_object.values)})

@from_ast.register
def _(ast_object: ast.BinOp):
    head = Head.binop
    args = [AST_BINOP_MAP[type(ast_object.op)], from_ast(ast_object.left), from_ast(ast_object.right)]
    return Expr(head, args)

@from_ast.register
def _(ast_object: ast.BoolOp):
    head = Head.binop
    op = AST_BINOP_MAP[type(ast_object.op)]
    args = nest_binop(op, ast_object.values)
    return Expr(head, args)

@from_ast.register
def _(ast_object: ast.Compare):
    head = Head.binop
    args = nest_compare(ast_object.ops, [ast_object.left] + ast_object.comparators)
    return Expr(head, args)

@from_ast.register
def _(ast_object: ast.If):
    head = Head.if_
    args = [from_ast(ast_object.test),
            from_ast(ast_object.body),
            from_ast(ast_object.orelse)]
    return Expr(head, args)

@from_ast.register
def _(ast_object: ast.For):
    head = Head.for_
    top = Expr(Head.binop, [Symbol("in"), from_ast(ast_object.target), from_ast(ast_object.iter)])
    block = from_ast(ast_object.body)
    return Expr(head, [top, block])

@from_ast.register
def _(ast_object: list):
    head = Head.block
    args = [from_ast(k) for k in ast_object]
    return Expr(head, args)

@from_ast.register
def _(ast_object: ast.FunctionDef):
    # TODO: positional only etc. are not supported
    head = Head.function
    fname = Symbol(ast_object.name)
    fargs = ast_object.args
    nargs = len(fargs.args) - len(fargs.defaults)
    args = [from_ast(k) for k in fargs.args[:nargs]]
    kwargs = [Expr(Head.kw, [from_ast(k), from_ast(v)]) 
              for k, v in zip(fargs.args[nargs:], fargs.defaults)]
    return Expr(head, [Expr(Head.call, [fname] + args + kwargs), from_ast(ast_object.body)])

@from_ast.register
def _(ast_object: ast.arg):
    ann = ast_object.annotation
    if ast_object.annotation is not None:
        return Expr(Head.annotate, [Symbol(ast_object.arg), from_ast(ann)])
    else:
        return Symbol(ast_object.arg)

@from_ast.register
def _(ast_object: ast.AnnAssign):
    head = Head.assign
    target = from_ast(ast_object.target)
    args = [Expr(Head.annotate, [target, from_ast(ast_object.annotation)]), 
            from_ast(ast_object.value)]
    return Expr(head, args)

@from_ast.register
def _(ast_object: ast.Pass):
    return Symbol("pass") # Should it be Expr(Head.pass_, [Symbol("pass")]) ??
    
@from_ast.register
def _(ast_object: ast.Return):
    head = Head.return_
    args = [from_ast(ast_object.value)]
    return Expr(head, args)

def nest_binop(op, values: list[ast.AST]):
    if len(values) == 2:
        return [from_ast(op), 
                from_ast(values[0]), 
                from_ast(values[1])]
    else:
        return [from_ast(op), 
                from_ast(values[0]), 
                Expr(Head.binop, nest_binop(op, values[1:]))]

def nest_compare(ops: list[ast.AST], values: list[ast.AST]):
    if len(ops) == 1:
        return [AST_BINOP_MAP[type(ops[0])],
                from_ast(values[0]), 
                from_ast(values[1])]
    else:
        return [AST_BINOP_MAP[type(ops[0])], 
                from_ast(values[0]), 
                Expr(Head.binop, nest_compare(ops[1:], values[1:]))]