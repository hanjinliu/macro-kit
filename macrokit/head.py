from enum import Enum


class Head(Enum):
    """Head of Expr."""

    empty = "empty"
    getattr = "getattr"
    getitem = "getitem"
    del_ = "del"
    call = "call"
    assign = "assign"
    kw = "kw"
    tuple = "tuple"
    list = "list"
    braces = "braces"
    comment = "comment"
    assert_ = "assert"
    unop = "unop"
    binop = "binop"
    aug = "aug"
    block = "block"
    function = "function"
    lambda_ = "lambda"
    return_ = "return"
    yield_ = "yield"
    yield_from = "yield_from"
    raise_ = "raise"
    if_ = "if"
    elif_ = "elif"
    for_ = "for"
    while_ = "while"
    annotate = "annotate"


EXEC = (
    Head.assign,
    Head.assert_,
    Head.comment,
    Head.block,
    Head.function,
    Head.return_,
    Head.if_,
    Head.elif_,
    Head.for_,
)
