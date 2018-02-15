import numpy as np


def parse_train(filename, wids, padsize):
    x = []
    y = []
    with open(filename, "r") as file:
        nsample, nalpha = [int(x) for x in file.readline().split()]
        for i in range(0, nsample):
            line = file.readline()
            for wid in wids:
                xl, yl = parseline(line, nalpha, wid)
                x += xl
                y += yl
    p = onehot(nalpha+2, nalpha+3)
    x = [pad(a, padsize, p) for a in x]
    return nalpha, np.array(x), np.array(y)


def parse_single_test(filename, nalpha, wid):
    with open(filename, "r") as file:
        x, y = parseline(file.readline(), nalpha, wid)
    return np.array(x), np.array(y)


def parseline(line, nalpha, window):
    # def f(x): return [x]  # représentation normale : singletons
    def f(x): return onehot(x, nalpha+3)  # représentation 'one hot'
    s = [f(nalpha)] + [f(int(x)) for x in line.split()[1:]] + [f(nalpha+1)]
    return windows(s, window, f(nalpha+2))


def pad(seq, length, padsymb):
    return seq + [padsymb]*(length-len(seq))


def windows(seq, width, padsymb):
    if len(seq) <= width:
        x = [pad(seq[:-1], width, padsymb)]
        y = [seq[-1]]
    else:
        x = [seq[i:(i+width)] for i in range(0, len(seq)-width)]
        y = [seq[i+width] for i in range(0, len(seq)-width)]
    return x, y


def onehot(i, size):
    r = [0]*size
    r[i] = 1
    return r


def argmax(x):
    if len(x) < 1:
        return -1
    arg = 0
    for i in range(1, len(x)):
        if x[arg] < x[i]:
            arg = i
    return arg


if __name__ == "__main__":
    print("coucou")
    # (ns, na, a, b) = parsetrain("../00.spice", 4)
    # (a, b) = parse_single_test("../0.spice.public.test", [1], 1)
    # x = [0, 1, 2, 3, 4]
    # xx, yy = windows(x, 10)
    # print(xx)
    # print(yy)
    # print(a, '\n\n', b)
