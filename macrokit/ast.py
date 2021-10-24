from __future__ import annotations
import ast
from functools import singledispatch
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
    ast.Pow: Symbol("**"),
    ast.MatMult: Symbol("@"),
    ast.FloorDiv: Symbol("//"),
    ast.BitAnd: Symbol("&"),
    ast.BitOr: Symbol("|"),
    ast.BitXor: Symbol("^")
}

NoneType = type(None)

def parse(source: str) -> Expr | Symbol:
    """
    Convert Python code string into Expr/Symbol objects.
    """    
    ast_object = ast.parse(source).body[0]
    return from_ast(ast_object)

@singledispatch
def from_ast(ast_object: ast.AST | NoneType):
    """
    Convert AST object to macro-kit object.
    """    
    raise NotImplementedError(f"AST type {type(ast_object)} cannot be converted now.")

@from_ast.register
def _(ast_object: NoneType):
    return None

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
    