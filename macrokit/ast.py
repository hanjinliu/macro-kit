from __future__ import annotations
import sys
import ast
from functools import singledispatch
import inspect
from typing import Callable

from .symbol import Symbol
from .expression import Expr, Head, symbol

NoneType = type(None)

AST_BINOP_MAP = {
    ast.Add: Symbol._reserved("+"),
    ast.Sub: Symbol._reserved("-"),
    ast.Mult: Symbol._reserved("*"),
    ast.Div: Symbol._reserved("/"),
    ast.Mod: Symbol._reserved("%"),
    ast.Eq: Symbol._reserved("=="),
    ast.NotEq: Symbol._reserved("!="),
    ast.Gt: Symbol._reserved(">"),
    ast.GtE: Symbol._reserved(">="),
    ast.Lt: Symbol._reserved("<"),
    ast.LtE: Symbol._reserved("<="),
    ast.Is: Symbol._reserved("is"),
    ast.IsNot: Symbol._reserved("is not"),
    ast.In: Symbol._reserved("in"),
    ast.NotIn: Symbol._reserved("not in"),
    ast.Pow: Symbol._reserved("**"),
    ast.MatMult: Symbol._reserved("@"),
    ast.FloorDiv: Symbol._reserved("//"),
    ast.BitAnd: Symbol._reserved("&"),
    ast.BitOr: Symbol._reserved("|"),
    ast.BitXor: Symbol._reserved("^"),
    ast.And: Symbol._reserved("and"),
    ast.Or: Symbol._reserved("or"),
}

AST_UNOP_MAP = {
    ast.UAdd: Symbol._reserved("+"),
    ast.USub: Symbol._reserved("-"),
    ast.Not: Symbol._reserved("not "),
    ast.Invert: Symbol._reserved("~"),
}


def parse(source: str | Callable, squeeze: bool = True) -> Expr | Symbol:
    """
    Convert Python code string into Expr/Symbol objects.
    """
    if callable(source):
        source = inspect.getsource(source)
    body = ast.parse(source).body
    ast_object: list[ast.stmt] | ast.stmt
    if len(body) == 1:
        ast_object = body[0]
    else:
        ast_object = body

    out = from_ast(ast_object)
    if not squeeze and len(body) == 1:
        out = Expr(Head.block, [out])
    return out


@singledispatch
def from_ast(ast_object: ast.AST | list | NoneType):
    """
    Convert AST object to macro-kit object.
    """
    raise NotImplementedError(f"AST type {type(ast_object)} cannot be converted now.")


@from_ast.register
def _none(ast_object: NoneType):
    return None


@from_ast.register
def _expr(ast_object: ast.Expr):
    return from_ast(ast_object.value)


if sys.version_info < (3, 8):
    @from_ast.register
    def _(ast_object: ast.Num):
        return symbol(ast_object.n)

    @from_ast.register
    def _str(ast_object: ast.Str):
        return symbol(ast_object.s)

    @from_ast.register
    def _name_constant(ast_object: ast.NameConstant):
        return symbol(ast_object.n)

    @from_ast.register
    def _bytes(ast_object: ast.Bytes):
        return symbol(ast_object.s)

    @from_ast.register
    def _ellipsis(ast_object: ast.Ellipsis):
        return symbol(ast_object.s)

else:
    @from_ast.register
    def _constant(ast_object: ast.Constant):
        return symbol(ast_object.value)


@from_ast.register
def _name(ast_object: ast.Name):
    name = ast_object.id
    if name in Symbol._module_map.keys():
        return Symbol._module_map[name]
    return Symbol(ast_object.id)


@from_ast.register
def _unaryop(ast_object: ast.UnaryOp):
    return Expr(Head.unop, [AST_UNOP_MAP[type(ast_object.op)],
                            from_ast(ast_object.operand)])


@from_ast.register
def _augassign(ast_object: ast.AugAssign):
    target = from_ast(ast_object.target)
    op = AST_BINOP_MAP[type(ast_object.op)]
    value = from_ast(ast_object.value)
    return Expr(Head.aug, [op, target, value])


@from_ast.register
def _call(ast_object: ast.Call):
    head = Head.call
    args = [from_ast(ast_object.func)] + [from_ast(k) for k in ast_object.args] + \
        [Expr(Head.kw, [Symbol(k.arg), from_ast(k.value)]) for k in ast_object.keywords]
    return Expr(head, args)


@from_ast.register
def _assign(ast_object: ast.Assign):
    head = Head.assign
    if len(ast_object.targets) != 1:
        target = tuple(from_ast(x) for x in ast_object.targets)
    else:
        target = from_ast(ast_object.targets[0])
    args = [target, from_ast(ast_object.value)]
    return Expr(head, args)


@from_ast.register
def _attribute(ast_object: ast.Attribute):
    head = Head.getattr
    args = [from_ast(ast_object.value), Symbol(ast_object.attr)]
    return Expr(head, args)


@from_ast.register
def _subscript(ast_object: ast.Subscript):
    head = Head.getitem
    args = [from_ast(ast_object.value), from_ast(ast_object.slice)]
    return Expr(head, args)


@from_ast.register
def _list(ast_object: ast.List):
    return symbol([from_ast(k) for k in ast_object.elts])


@from_ast.register
def _tuple(ast_object: ast.Tuple):
    return symbol(tuple(from_ast(k) for k in ast_object.elts))


@from_ast.register
def _set(ast_object: ast.Set):
    return symbol(set(from_ast(k) for k in ast_object.elts))


@from_ast.register
def _slice(ast_object: ast.Slice):
    return symbol(slice(from_ast(ast_object.lower),
                        from_ast(ast_object.upper),
                        from_ast(ast_object.step))
                  )


@from_ast.register
def _dict(ast_object: ast.Dict):
    return symbol({from_ast(k): from_ast(v)
                   for k, v in zip(ast_object.keys, ast_object.values)})


@from_ast.register
def _binop(ast_object: ast.BinOp):
    head = Head.binop
    args = [AST_BINOP_MAP[type(ast_object.op)], from_ast(
        ast_object.left), from_ast(ast_object.right)]
    return Expr(head, args)


@from_ast.register
def _boolop(ast_object: ast.BoolOp):
    head = Head.binop
    op = AST_BINOP_MAP[type(ast_object.op)]
    args = nest_binop(op, ast_object.values)
    return Expr(head, args)


@from_ast.register
def _compare(ast_object: ast.Compare):
    head = Head.binop
    args = nest_compare(ast_object.ops, [ast_object.left] + ast_object.comparators)
    return Expr(head, args)


@from_ast.register
def _if(ast_object: ast.If):
    head = Head.if_
    args = [from_ast(ast_object.test),
            from_ast(ast_object.body),
            from_ast(ast_object.orelse)]
    return Expr(head, args)


@from_ast.register
def _for(ast_object: ast.For):
    head = Head.for_
    top = Expr(Head.binop, [Symbol.var("in"), from_ast(
        ast_object.target), from_ast(ast_object.iter)])
    block = from_ast(ast_object.body)
    if ast_object.orelse:
        raise ValueError("'else' block is not supported yet")
    return Expr(head, [top, block])


@from_ast.register
def _while(ast_object: ast.While):
    head = Head.while_
    test = from_ast(ast_object.test)
    block = from_ast(ast_object.body)
    if ast_object.orelse:
        raise ValueError("'else' block is not supported yet")
    return Expr(head, [test, block])


@from_ast.register
def _list_of_ast(ast_object: list):
    head = Head.block
    args = [from_ast(k) for k in ast_object]
    return Expr(head, args)


@from_ast.register
def _function_def(ast_object: ast.FunctionDef):
    # TODO: positional only etc. are not supported
    head = Head.function
    fname = Symbol(ast_object.name)
    fargs = ast_object.args
    nargs = len(fargs.args) - len(fargs.defaults)
    args = [from_ast(k) for k in fargs.args[:nargs]]
    kwargs = [Expr(Head.kw, [from_ast(k), from_ast(v)])
              for k, v in zip(fargs.args[nargs:], fargs.defaults)]
    return Expr(head, [Expr(Head.call, [fname] + args + kwargs),
                       from_ast(ast_object.body)])


@from_ast.register
def _arg(ast_object: ast.arg):
    ann = ast_object.annotation
    if ast_object.annotation is not None:
        return Expr(Head.annotate, [Symbol(ast_object.arg), from_ast(ann)])
    else:
        return Symbol(ast_object.arg)


@from_ast.register
def _annotated_assign(ast_object: ast.AnnAssign):
    head = Head.assign
    target = from_ast(ast_object.target)
    args = [Expr(Head.annotate, [target, from_ast(ast_object.annotation)]),
            from_ast(ast_object.value)]
    return Expr(head, args)


@from_ast.register
def _pass(ast_object: ast.Pass):
    return Symbol.var("pass")


@from_ast.register
def _break(ast_object: ast.Break):
    return Symbol.var("break")


@from_ast.register
def _continue(ast_object: ast.Continue):
    return Symbol.var("continue")


@from_ast.register
def _raise(ast_object: ast.Raise):
    exc = ast_object.exc
    if ast_object.cause:
        raise ValueError("'raise XX from YY' is not supported yet")
    return Expr(Head.raise_, [from_ast(exc)])


@from_ast.register
def _return(ast_object: ast.Return):
    head = Head.return_
    args = [from_ast(ast_object.value)]
    return Expr(head, args)


def nest_binop(op, values: list[ast.expr]):
    if len(values) == 2:
        return [op,
                from_ast(values[0]),
                from_ast(values[1])]
    else:
        return [op,
                from_ast(values[0]),
                Expr(Head.binop, nest_binop(op, values[1:]))]


def nest_compare(ops: list[ast.cmpop], values: list[ast.expr]):
    if len(ops) == 1:
        return [AST_BINOP_MAP[type(ops[0])],
                from_ast(values[0]),
                from_ast(values[1])]
    else:
        return [AST_BINOP_MAP[type(ops[0])],
                from_ast(values[0]),
                Expr(Head.binop, nest_compare(ops[1:], values[1:]))]
