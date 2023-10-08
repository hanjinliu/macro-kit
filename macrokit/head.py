from enum import Enum


class Head(Enum):
    """Head of Expr."""

    empty = "empty"
    getattr = "getattr"
    getitem = "getitem"
    del_ = "del"
    call = "call"
    assign = "assign"
    walrus = "walrus"
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
    try_ = "try"
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
    star = "star"
    starstar = "starstar"
    decorator = "decorator"
    match = "match"
    case = "case"


EXEC = {
    Head.assign,
    Head.assert_,
    Head.comment,
    Head.del_,
    Head.block,
    Head.function,
    Head.return_,
    Head.raise_,
    Head.if_,
    Head.for_,
    Head.while_,
    Head.annotate,
    Head.import_,
    Head.from_,
    Head.as_,
    Head.with_,
    Head.class_,
    Head.try_,
    Head.match,
}

HAS_BLOCK = {
    Head.block,
    Head.if_,
    Head.for_,
    Head.while_,
    Head.function,
    Head.class_,
    Head.with_,
    Head.try_,
    Head.case,
}
