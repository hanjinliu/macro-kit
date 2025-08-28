import ast
import inspect
import sys
from typing import Any, Callable, get_type_hints

from macrokit._symbol import Symbol
from macrokit.expression import _STORED_SYMBOLS, Expr, Head, symbol

LAMBDA = Symbol._reserved("<lambda>")

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
    """Convert Python code string into Expr/Symbol objects."""
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


_ParseFunc = Callable[[Any], Symbol | Expr]


class singledispatch:
    """Simplified version of functools.singledispatch."""

    def __init__(self, func: _ParseFunc):
        self.func = func
        self._registry: dict[type, _ParseFunc] = {}

    def __call__(self, *args, **kwargs):
        """Dispatch to the registered function for the type of the first argument."""
        try:
            f = self._registry[args[0].__class__]
        except KeyError:
            f = self.func
        return f(*args, **kwargs)

    def register(self, func: _ParseFunc) -> _ParseFunc:
        """Register function as a handler for a specific type."""
        cls = next(iter(get_type_hints(func).values()))
        self._registry[cls] = func
        return func


@singledispatch
def from_ast(ast_object: ast.AST | list | None):
    """Convert AST object to macro-kit object."""
    raise NotImplementedError(f"AST type {type(ast_object)} cannot be converted now.")


@from_ast.register
def _none(ast_object: None):
    return None


@from_ast.register
def _expr(ast_object: ast.Expr):
    return from_ast(ast_object.value)


@from_ast.register
def _constant(ast_object: ast.Constant):
    return symbol(ast_object.value)


@from_ast.register
def _name(ast_object: ast.Name):
    name = ast_object.id
    if name in _STORED_SYMBOLS:
        return symbol(_STORED_SYMBOLS[name])
    return Symbol(name)


@from_ast.register
def _unaryop(ast_object: ast.UnaryOp):
    return Expr(
        Head.unop, [AST_UNOP_MAP[type(ast_object.op)], from_ast(ast_object.operand)]
    )


@from_ast.register
def _augassign(ast_object: ast.AugAssign):
    target = from_ast(ast_object.target)
    op = AST_BINOP_MAP[type(ast_object.op)]
    value = from_ast(ast_object.value)
    return Expr(Head.aug, [op, target, value])


@from_ast.register
def _call(ast_object: ast.Call):
    head = Head.call
    args = (
        [from_ast(ast_object.func)]
        + [from_ast(k) for k in ast_object.args]
        + [
            Expr(Head.kw, [Symbol(k.arg), from_ast(k.value)])
            for k in ast_object.keywords
        ]
    )
    return Expr(head, args)


@from_ast.register
def _joinedstr(ast_object: ast.JoinedStr):
    seq = "f'" + _nest_joinedstr(ast_object) + "'"
    return Symbol(seq, id(seq))


@from_ast.register
def _assign(ast_object: ast.Assign):
    head = Head.assign
    target0 = ast_object.targets[0]
    target = from_ast(target0)
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
    return symbol({from_ast(k) for k in ast_object.elts})


@from_ast.register
def _slice(ast_object: ast.Slice):
    return symbol(
        slice(
            from_ast(ast_object.lower),
            from_ast(ast_object.upper),
            from_ast(ast_object.step),
        )
    )


@from_ast.register
def _dict(ast_object: ast.Dict):
    return symbol(
        {
            from_ast(k): from_ast(v)
            for k, v in zip(ast_object.keys, ast_object.values, strict=False)
        }
    )


@from_ast.register
def _binop(ast_object: ast.BinOp):
    head = Head.binop
    args = [
        AST_BINOP_MAP[type(ast_object.op)],
        from_ast(ast_object.left),
        from_ast(ast_object.right),
    ]
    return Expr(head, args)


@from_ast.register
def _boolop(ast_object: ast.BoolOp):
    head = Head.binop
    op = AST_BINOP_MAP[type(ast_object.op)]
    args = [from_ast(a) for a in ast_object.values]
    return Expr(head, [op] + args)


@from_ast.register
def _compare(ast_object: ast.Compare):
    head = Head.binop
    args = _nest_compare(ast_object.ops, [ast_object.left] + ast_object.comparators)
    return Expr(head, args)


@from_ast.register
def _if(ast_object: ast.If):
    head = Head.if_
    args = [
        from_ast(ast_object.test),
        from_ast(ast_object.body),
    ]
    if ast_object.orelse:
        args.append(from_ast(ast_object.orelse))
    return Expr(head, args)


@from_ast.register
def _lambda(ast_object: ast.Lambda):
    head = Head.lambda_
    fargs = ast_object.args
    nargs = len(fargs.args) - len(fargs.defaults)
    args = [from_ast(k) for k in fargs.args[:nargs]]
    kwargs = [
        Expr(Head.kw, [from_ast(k), from_ast(v)])
        for k, v in zip(fargs.args[nargs:], fargs.defaults, strict=False)
    ]
    return Expr(
        head,
        [
            Expr(Head.call, [LAMBDA] + args + kwargs),
            from_ast(ast_object.body),
        ],
    )


@from_ast.register
def _for(ast_object: ast.For):
    head = Head.for_
    top = Expr(
        Head.binop,
        [Symbol.var("in"), from_ast(ast_object.target), from_ast(ast_object.iter)],
    )
    block = from_ast(ast_object.body)
    if ast_object.orelse:
        raise ValueError("'else' block is not supported yet")
    return Expr(head, [top, block])


@from_ast.register
def _try(ast_object: ast.Try):
    head = Head.try_
    args = [from_ast(ast_object.body)]
    for k in ast_object.handlers:
        args.extend(_except_to_list(k))
    if ast_object.orelse:
        args.append(from_ast(ast_object.orelse))
    else:
        args.append(Expr.empty())
    if ast_object.finalbody:
        args.append(from_ast(ast_object.finalbody))
    else:
        args.append(Expr.empty())

    return Expr(head, args)


if sys.version_info >= (3, 10):

    @from_ast.register
    def _match(ast_object: ast.Match):
        head = Head.match
        args = [from_ast(ast_object.subject)]
        for k in ast_object.cases:
            args.append(from_ast(k))
        return Expr(head, args)

    @from_ast.register
    def _case(ast_object: ast.match_case):
        head = Head.case
        args = [from_ast(ast_object.pattern)]
        args.append(from_ast(ast_object.body))
        if ast_object.guard is not None:
            args.append(Expr(Head.filter, [from_ast(ast_object.guard)]))
        return Expr(head, args)

    @from_ast.register
    def _match_value(ast_object: ast.MatchValue):
        return from_ast(ast_object.value)

    @from_ast.register
    def _match_as(ast_object: ast.MatchAs):
        if ast_object.pattern is not None:
            return Expr(
                Head.as_, [from_ast(ast_object.pattern), Symbol(ast_object.name)]
            )
        return Symbol(ast_object.name)

    @from_ast.register
    def _match_class(ast_object: ast.MatchClass):
        kwargs = []
        for a, p in zip(ast_object.kwd_attrs, ast_object.kwd_patterns, strict=False):
            kwargs.append(Expr(Head.kw, [Symbol(a), from_ast(p)]))

        expr = Expr(
            Head.call,
            [from_ast(ast_object.cls)]
            + [from_ast(p) for p in ast_object.patterns]
            + kwargs,
        )
        return expr

    @from_ast.register
    def _match_or(ast_object: ast.MatchOr):
        pats = [from_ast(a) for a in ast_object.patterns]
        return Expr(Head.binop, [Symbol._reserved("|")] + pats)

    @from_ast.register
    def _match_singleton(ast_object: ast.MatchSingleton):
        return symbol(ast_object.value)

    @from_ast.register
    def _match_sequence(ast_object: ast.MatchSequence):
        args = [from_ast(a) for a in ast_object.patterns]
        return Expr(Head.list, args)

    @from_ast.register
    def _match_starred(ast_object: ast.MatchStar):
        return Expr(Head.star, [Symbol(ast_object.name)])

    @from_ast.register
    def _match_mapping(ast_object: ast.MatchMapping):
        args = []
        for k, v in zip(ast_object.keys, ast_object.patterns, strict=False):
            args.append(Expr(Head.annotate, [from_ast(k), from_ast(v)]))
        if ast_object.rest:
            args.append(Expr(Head.starstar, [Symbol(ast_object.rest)]))
        return Expr(Head.braces, args)


def _except_to_list(ast_object: ast.ExceptHandler) -> "list[Symbol | Expr]":
    if ast_object.type is not None:
        if isinstance(ast_object.type, ast.Name):
            type_: Symbol | Expr = Symbol(ast_object.type.id)
        else:
            type_ = symbol(ast_object.type)
        if ast_object.name:
            arg0: Symbol | Expr = Expr(Head.as_, [type_, Symbol(ast_object.name)])
        else:
            arg0 = type_
    else:
        arg0 = Expr.empty()

    block = from_ast(ast_object.body)
    return [arg0, block]


@from_ast.register
def _while(ast_object: ast.While):
    head = Head.while_
    test = from_ast(ast_object.test)
    block = from_ast(ast_object.body)
    if ast_object.orelse:
        raise ValueError("'else' block is not supported yet")
    return Expr(head, [test, block])


@from_ast.register
def _listcomp(ast_object: ast.ListComp):
    gen = _generator_to_args(from_ast(ast_object.elt), ast_object.generators)
    return Expr(Head.list, [gen])


@from_ast.register
def _dictcomp(ast_object: ast.DictComp):
    elt = Expr(Head.annotate, [from_ast(ast_object.key), from_ast(ast_object.value)])
    gen = _generator_to_args(elt, ast_object.generators)
    return Expr(Head.braces, [gen])


@from_ast.register
def _setcomp(ast_object: ast.SetComp):
    gen = _generator_to_args(from_ast(ast_object.elt), ast_object.generators)
    return Expr(Head.braces, [gen])


@from_ast.register
def _generator(ast_object: ast.GeneratorExp):
    return _generator_to_args(from_ast(ast_object.elt), ast_object.generators)


def _generator_to_args(elt: "Symbol | Expr", comps: "list[ast.comprehension]"):
    out = _gen(elt, comps[0])
    if len(comps) > 1:
        for comp in comps[1:]:
            out = _gen(out, comp)
    return out


def _gen(elt: "Symbol | Expr", comp: ast.comprehension):
    args = [elt, from_ast(comp.target), from_ast(comp.iter)]
    for _if in comp.ifs:
        args.append(Expr(Head.filter, [from_ast(_if)]))
    return Expr(Head.generator, args)


@from_ast.register
def _list_of_ast(ast_object: list):
    head = Head.block
    args = [from_ast(k) for k in ast_object]
    return Expr(head, args)


@from_ast.register
def _function_def(ast_object: ast.FunctionDef):
    head = Head.function
    _call_args: list[Symbol | Expr] = []
    fname = Symbol(ast_object.name)
    _call_args.append(fname)
    fargs = ast_object.args
    vararg = fargs.vararg
    nargs = len(fargs.args) - len(fargs.defaults)
    _call_args.extend(from_ast(k) for k in fargs.args[:nargs])  # args
    if fargs.posonlyargs:
        _call_args.insert(len(fargs.posonlyargs), Symbol._reserved("/"))
    _call_args.extend(
        Expr(Head.kw, [from_ast(k), from_ast(v)])
        for k, v in zip(fargs.args[nargs:], fargs.defaults, strict=False)
    )  # kwargs
    if vararg is not None:  # *args
        _input_expr = Expr(Head.star, [Symbol(vararg.arg)])
        if vararg.annotation is not None:
            _input_expr = Expr(
                Head.annotate, [_input_expr, from_ast(vararg.annotation)]
            )
        _call_args.append(_input_expr)
    if fargs.kwonlyargs:
        _call_args.append(Symbol._reserved("*"))
        _call_args.extend(
            Expr(Head.kw, [from_ast(k), from_ast(v)])
            for k, v in zip(fargs.kwonlyargs, fargs.kw_defaults, strict=False)
        )
    if fargs.kwarg:  # **kwargs
        _input_expr = Expr(Head.starstar, [Symbol(fargs.kwarg.arg)])
        if fargs.kwarg.annotation is not None:
            _input_expr = Expr(
                Head.annotate, [_input_expr, from_ast(fargs.kwarg.annotation)]
            )
        _call_args.append(_input_expr)

    out = Expr(
        head,
        [Expr(Head.call, _call_args), from_ast(ast_object.body)],
    )
    for dec in ast_object.decorator_list:
        out = Expr(Head.decorator, [from_ast(dec), out])

    return out


@from_ast.register
def _starred(ast_object: ast.Starred):
    return Expr(Head.star, [from_ast(ast_object.value)])


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
    args = [
        Expr(Head.annotate, [target, from_ast(ast_object.annotation)]),
        from_ast(ast_object.value),
    ]
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
    args = [from_ast(ast_object.exc)]
    if ast_object.cause:
        args.append(Expr(Head.from_, [from_ast(ast_object.cause)]))
    return Expr(Head.raise_, args)


@from_ast.register
def _del(ast_object: ast.Delete):
    return Expr(Head.del_, [from_ast(t) for t in ast_object.targets])


@from_ast.register
def _return(ast_object: ast.Return):
    head = Head.return_
    args = [from_ast(ast_object.value)]
    return Expr(head, args)


@from_ast.register
def _yield(ast_object: ast.Yield):
    return Expr(Head.yield_, [from_ast(ast_object.value)])


@from_ast.register
def _yield_from(ast_object: ast.YieldFrom):
    from_ = Expr(Head.from_, [from_ast(ast_object.value)])
    return Expr(Head.yield_, [from_])


@from_ast.register
def _import(ast_object: ast.Import):
    head = Head.import_
    args = [from_ast(k) for k in ast_object.names]
    return Expr(head, args)


@from_ast.register
def _import_from(ast_object: ast.ImportFrom):
    head = Head.import_
    _from = Expr(Head.from_, [Symbol(ast_object.module)])
    _names = [from_ast(name) for name in ast_object.names]
    return Expr(head, _names + [_from])


@from_ast.register
def _alias(ast_object: ast.alias):
    sym = Symbol(ast_object.name)
    if ast_object.asname:
        return Expr(Head.as_, [sym, Symbol(ast_object.asname)])
    else:
        return sym


@from_ast.register
def _with(ast_object: ast.With):
    head = Head.with_
    args = [from_ast(k) for k in ast_object.items]
    args.append(from_ast(ast_object.body))
    return Expr(head, args)


@from_ast.register
def _withitem(ast_object: ast.withitem):
    left = from_ast(ast_object.context_expr)
    if ast_object.optional_vars:
        right = from_ast(ast_object.optional_vars)
        return Expr(Head.as_, [left, right])
    return left


@from_ast.register
def _class(ast_object: ast.ClassDef):
    head = Head.class_
    args = [from_ast(k) for k in ast_object.bases]
    kwargs = [from_ast(kw) for kw in ast_object.keywords]
    name = Symbol(ast_object.name)
    _cls: Symbol | Expr
    if args or kwargs:
        _cls = Expr(Head.call, [name] + args + kwargs)
    else:
        _cls = name
    body = Expr(Head.block, [from_ast(k) for k in ast_object.body])
    out = Expr(head, [_cls, body])
    for dec in ast_object.decorator_list:
        out = Expr(Head.decorator, [from_ast(dec), out])
    return out


@from_ast.register
def _keyword(ast_object: ast.keyword):
    val = from_ast(ast_object.value)
    if ast_object is not None:
        return Expr(Head.kw, [from_ast(ast_object.arg), val])
    return Expr(Head.starstar, [val])


@from_ast.register
def _assert(ast_object: ast.Assert):
    head = Head.assert_
    args = [from_ast(ast_object.test)]
    if ast_object.msg:
        args.append(from_ast(ast_object.msg))
    return Expr(head, args)


@from_ast.register
def _named_expr(ast_object: ast.NamedExpr):
    head = Head.walrus
    target = from_ast(ast_object.target)
    args = [target, from_ast(ast_object.value)]
    return Expr(head, args)


def _nest_compare(ops: "list[ast.cmpop]", values: "list[ast.expr]"):
    if len(ops) == 1:
        return [AST_BINOP_MAP[type(ops[0])], from_ast(values[0]), from_ast(values[1])]
    else:
        return [
            AST_BINOP_MAP[type(ops[0])],
            from_ast(values[0]),
            Expr(Head.binop, _nest_compare(ops[1:], values[1:])),
        ]


def _nest_joinedstr(ast_object: ast.JoinedStr):
    strs: list[str] = []
    for k in ast_object.values:
        if isinstance(k, ast.FormattedValue):
            _id = ast.unparse(k.value)
            if k.format_spec is None:
                if k.conversion == -1:
                    strs.append("{" + f"{_id}" + "}")
                elif k.conversion == 115:
                    strs.append("{" + f"{_id}!s" + "}")
                elif k.conversion == 114:
                    strs.append("{" + f"{_id}!r" + "}")
                elif k.conversion == 97:
                    strs.append("{" + f"{_id}!a" + "}")
                else:
                    raise RuntimeError(f"Unknown conversion: {k.conversion}")
            elif isinstance(k.format_spec, ast.JoinedStr):
                fspec = _nest_joinedstr(k.format_spec)
                strs.append("{" + f"{_id}:{fspec}" + "}")
            else:
                raise RuntimeError(f"Unknown format_spec type: {type(k.format_spec)}")
        elif isinstance(k, ast.Constant):
            strs.append(str(k.value))
        else:
            raise RuntimeError(f"Unknown JoinedStr value type: {type(k)}")
    return "".join(strs)
