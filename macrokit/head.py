from enum import Enum

class Head(Enum):
    empty    = "empty"
    getattr  = "getattr"
    getitem  = "getitem"
    del_     = "del"
    call     = "call"
    assign   = "assign"
    kw       = "kw"
    comment  = "comment"
    assert_  = "assert"
    unop     = "unop"
    binop    = "binop"
    aug      = "aug"
    block    = "block"
    function = "function"
    return_  = "return"
    if_      = "if"
    elif_    = "elif"
    for_     = "for"
    while_   = "while"
    annotate = "annotate"

EXEC = (Head.assign, Head.assert_, Head.comment, Head.block, Head.function, Head.return_,
        Head.if_, Head.elif_, Head.for_)