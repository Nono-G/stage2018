import splearn as sp
import parse5 as parse
import numpy as np
import spextractor_common as co
import time


def suff(hu, code):
    if code == 0:
        return 0
    le = hu.len_code(code)
    ba = hu.pows[le-1]
    diff = (((code-hu.nl[le-1]) // ba)+1) * ba
    return code - diff


def suffs(hu, c):
    ret = [c]
    code = c
    le = hu.len_code(code)
    while le > 0:
        ba = hu.pows[le-1]
        diff = (((code-hu.nl[le-1]) // ba)+1) * ba
        code -= diff
        ret.append(code)
        le -= 1
    return ret


h = co.Hush(20, 4)
print(h.pows)
print(h.nl)
# for i in range(0,1000):
#     assert h.len_code(i) == len(h.decode(i))
#     assert h.suffix_code(i) == h.encode(h.decode(i)[1:])
#     assert [h.decode(s) for s in h.suffixes_codes(i)] == [h.decode(i)[x:] for x in range(len(h.decode(i))+1)]
#     w = h.decode(i)
#     x = h.encode(w[1:])
#     d = i-x
#     s = h.decode(x)
#     # sbis = h.decode()
#     print(i, w, x, d, s, [h.decode(s) for s in h.suffixes_codes(i)])




