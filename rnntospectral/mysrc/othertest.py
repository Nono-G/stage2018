class Hush:
    def __init__(self, maxlen, base):
        self.maxlen = maxlen
        self.base = base
        self.nl = []
        self.pows = []
        self.mise_en_place()
        self.maxval = self.encode([base - 1] * maxlen)

    def mise_en_place(self):
        self.pows = [1] * (self.maxlen+1)
        for i in range(1, self.maxlen+1):
            self.pows[i] = self.pows[i-1] * self.base
        #
        self.nl = [1] * (self.maxlen+1)
        for i in range(1, self.maxlen+1):
            self.nl[i] = self.nl[i-1] + (self.pows[i])
        #

    def encode(self, w):
        if len(w) == 0:
            return 0
        else:
            x = self.nl[len(w)-1]
            for i in range(len(w)):
                x += w[i]*self.pows[len(w)-i-1]
            return x

    def decode(self, n):
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


h = Hush(11, 6)
print(h.nl)
print(h.pows)
print(h.decode(23))
for j in range(20, 100):
    print(h.encode(h.decode(j)))
