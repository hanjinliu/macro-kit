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
    raise_ = "raise"
    if_ = "if"
    elif_ = "elif"
    for_ = "for"
    while_ = "while"
    generator = "generator"
    filter = "filter"
    annotate = "annotate"
    import_ = "import"
    from_ = "from"
    as_ = "as"
    with_ = "with"
    class_ = "class"


EXEC = (
    Head.assign,
    Head.assert_,
    Head.comment,
    Head.del_,
    Head.block,
    Head.function,
    Head.return_,
    Head.raise_,
    Head.if_,
    Head.elif_,
    Head.for_,
    Head.while_,
    Head.annotate,
    Head.import_,
    Head.from_,
    Head.as_,
    Head.with_,
    Head.class_,
)
