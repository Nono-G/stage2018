class Hush:
    """
    (Sequence of integers / integers) codec.
    Warning : this is not a x-based integers to 10-based integers codec, as for instance [0] and [0,0,0] sequences have
    different encodings.
    """
    def __init__(self, maxlen, base):
        self.maxlen = max(2, maxlen)
        self.base = base
        self.nl = []
        self.pows = []
        self.ready()
        self.maxval = self.encode([base - 1] * maxlen)

    def ready(self):
        self.pows = [1] * (self.maxlen+1)
        for i in range(1, self.maxlen+1):
            self.pows[i] = self.pows[i-1] * self.base
        #
        self.nl = [1] * (self.maxlen+1)
        for i in range(1, self.maxlen+1):
            self.nl[i] = self.nl[i-1] + (self.pows[i])
        #

    def get_copy(self):
        h = Hush(self.maxlen, self.maxval)
        return h

    def words_of_len(self, le):
        if le == 0:
            return range(1)
        else:
            r = range(self.nl[le - 1], self.nl[le])
            return r
            # return [i for i in r]

    def encode(self, w):
        if len(w) > self.maxlen:
            print(w)
            raise ValueError
        if len(w) == 0:
            return 0
        else:
            x = self.nl[len(w)-1]
            for i in range(len(w)):
                x += w[i]*self.pows[len(w)-i-1]
            return x

    def decode(self, s):
        if isinstance(s, tuple):
            r = []
            for x in s:
                r += self.decode(x)
            return r
        else:
            return self._decode(s)

    def _decode(self, n):
        if n > self.maxval:
            raise ValueError
        le = 0
        while self.nl[le] <= n:
            le += 1
        x = [0]*le
        reste = n - self.nl[le-1]
        for i in range(le):
            x[le-i-1] = reste % self.base
            reste //= self.base
        return x

    def prefix_code(self, code):
        if code >= self.nl[1]:
            i = 0
            while i < self.maxlen and self.nl[i] <= code:
                i += 1
            i -= 1
            d = (code - self.nl[i]) // self.base
            return self.nl[i-1]+d
        else:
            return 0

    def prefixes_codes(self, code):
        """Return a set of prefixes codes, c itself is excluded"""
        r = set()
        c = code
        i = 0
        while i < self.maxlen and self.nl[i] <= c:
            i += 1
        i -= 1
        while i > 0:
            d = (c - self.nl[i]) // self.base
            c = self.nl[i-1]+d
            r.add(c)
            i -= 1
        r.add(0)
        return r

    def suffix_code(self, code):
        if code == 0:
            return 0
        le = self.len_code(code)
        base = self.pows[le - 1]
        diff = (((code - self.nl[le - 1]) // base) + 1) * base
        return code - diff

    def suffixes_codes(self, c):
        """Return a list of suffixes codes, from shorter to longer, c itself is included"""
        ret = [c]
        code = c
        le = self.len_code(code)
        while le > 0:
            base = self.pows[le - 1]
            diff = (((code - self.nl[le - 1]) // base) + 1) * base
            code -= diff
            ret.append(code)
            le -= 1
        return ret

    def concat_code(self, tup):
        return self.encode(self.decode(tup))

    def len_code(self, code):
        le = 0
        while code >= self.nl[le]:
            le += 1
        return le
