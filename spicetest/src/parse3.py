import numpy as np


def parse_train(filename, wid=-1):
    xs = []
    ys = []
    maxlen = 0
    with open(filename, 'r') as file:
        nsamples, nalpha = [int(a) for a in file.readline().split()]
        for line in file.readlines():
            ll, (xl, yl) = parse_line(line, nalpha, wid)
            xs += xl
            ys += yl
            if ll > maxlen:
                maxlen = ll
    # On enlève 1 pour que les y restent dans l'espace 0-nalpha+1
    ys = [onehot(a-1, nalpha+2) for a in ys]
    # On finit le padding jusqu'a maxlen :
    if wid == -1:
        xs = [pad_0(a, maxlen+1) for a in xs]
    return nalpha, np.array(xs), np.array(ys)


def parse_test_prefixes(filename, wid):
    xs = []
    maxlen = 0
    with open(filename, 'r') as file:
        nsamples, nalpha = [int(a) for a in file.readline().split()]
        for line in file.readlines():
            x = [nalpha+1]+[int(a)+1 for a in line.split()[1:]]
            if wid > 0:
                if len(x) > wid:
                    x = x[-wid:]
                else:
                    x = pad_0(x, wid)
            else:
                if len(x) > maxlen:
                    maxlen = len(x)
            xs += [x]
    if wid < 1:
        # On finit le padding jusqu'a maxlen :
        xs = [pad_0(a, maxlen) for a in xs]
    return nalpha, np.array(xs)


def parse_targets(targets_file, nalpha):
    targ = []
    with open(targets_file, 'r') as tarf:
        for line in tarf.readlines():
            targ += [onehot(int(line.split()[1]), nalpha+2)]
    return np.array(targ)

def parse_line(line, nalpha, widarg):
    # on ajoute 1 partout pour que le zero soit libre pour servir de bourrage :
    sp = line.split()
    linelen = int(sp[0])
    if widarg == -1:
        wid = linelen+1
    else:
        wid = widarg
    seq = [nalpha+1]+[int(a)+1 for a in sp[1:]]+[nalpha+2]
    return linelen, windows(seq, wid)


def pad_0(seq, size):
    return [0]*(size-len(seq)) + seq


def windows(seq, wid):
    xs = []
    ys = seq[1:]
    for i in range(1, min(wid, len(seq))):
        xs += [pad_0(seq[:i], wid)]
        # ys += [seq[i]]
    for i in range(0, len(seq)-wid):
        xs += [seq[i:i+wid]]
        # ys += [seq[i+wid]]
    return xs, ys


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


def best_n_args(seq, n):
    return np.argsort(seq)[::-1][:n]


if __name__ == "__main__":
    print("coucou")
    n_zz, x_zz = parse_test_prefixes("../data/prefixes/8.spice.prefix.public", 12)
    y_zz = parse_targets("../data/targets/8.spice.target.public", n_zz)
    # print(x)
    print(y_zz)
