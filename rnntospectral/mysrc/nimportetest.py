from hank3 import RangeUnion
from spextractor_common import Hush


def closer_inf(s, x):
    i = 0
    while i<len(s) and s[i]<x:
        i += 1
    return s[i-1]


# for i in range(100):
#     a = i
#     b = h.decode(i)
#     c = closer_inf(h.nl, i)
#     d = (i - c) // h.base
#     if i > 10:
#         e = h.decode(h.nl[len(h.decode(i))-2] + d)
#         print(a, b, c, d, e)
#     else:
#         print(a, b, c, d)

# for i in range(300):
#     a = i
#     b = h.decode(i)
#     print(a, b, [h.decode(kkk) for kkk in h.prefixes_codes(i)])
