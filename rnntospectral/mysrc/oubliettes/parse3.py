import numpy as np


def parse_train(filename, wid=-1, padbefore=True):
    xs = []
    ys = []
    maxlen = 0
    with open(filename, 'r') as file:
        nsamples, nalpha = [int(a) for a in file.readline().split()]
        for line in file.readlines():
            ll, (xl, yl) = parse_line(line, nalpha, wid, padbefore)
            xs += xl
            ys += yl
            if ll > maxlen:
                maxlen = ll
    # On enlève 1 pour que les y restent dans l'espace [0, nalpha+1]
    ys = [onehot(a-1, nalpha+2) for a in ys]
    # On finit le padding jusqu'a maxlen :
    if wid == -1:
        xs = [pad_0(a, maxlen+1, padbefore) for a in xs]
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


def parse_line(line, nalpha, widarg, padbefore=True):
    # on ajoute 1 partout pour que le zero soit libre pour servir de bourrage :
    sp = line.split()
    linelen = int(sp[0])
    if widarg == -1:
        wid = linelen+1
    else:
        wid = widarg
    seq = [nalpha+1]+[int(a)+1 for a in sp[1:]]+[nalpha+2]
    return linelen, windows(seq, wid, padbefore)


def pad_0(seq, size, before=True):
    return pad(seq, 0, size, before)


def pad(seq, elt, size, before=True):
    if len(seq) < size:
        if before:
            return [elt]*(size - len(seq)) + seq
        else:
            return seq + [elt]*(size-len(seq))
    else:
        return seq[-size:]


def windows(seq, wid, padbefore=True):
    xs = []
    ys = seq[1:]
    for i in range(1, min(wid, len(seq))):
        xs += [pad_0(seq[:i], wid, padbefore)]
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


def parse_pautomac_results(filename, epsilon=0.000001):
    with open(filename, "r") as file:
        n = int(file.readline())
        y = np.empty(n)
        for i in range(n):
            y[i] = float(file.readline())
            if y[i] == 0:
                y[i] = epsilon
    return y


def parse_fullwords(filename):
    with open(filename, "r") as file:
        spl = file.readline().split()
        nbx = int(spl[0])
        # nalpha = int(spl[1])
        x = []
        for i in range(nbx):
            x.append([int(s) for s in file.readline().split()[1:]])
    return x


def parse_fullwords_encoded(filename, padsize):
    with open(filename, "r") as file:
        spl = file.readline().split()
        nbx = int(spl[0])
        nalpha = int(spl[1])
        x = []
        for i in range(nbx):
            x.append(pad_0([nalpha+1]+[int(s)+1 for s in file.readline().split()[1:]]+[nalpha+2], padsize))
    return np.array(x)


def to_binary_classf(y, one):
    biny = np.empty((y.shape[0], 2))
    for i in range(len(y)):
        if argmax(y[i]) == one:
            biny[i] = np.array([0, 1])
        else:
            biny[i] = np.array([1, 0])
    return biny


def random_sample(x, y, nb):
    shufflek = np.random.choice(x.shape[0], nb, replace=False)
    x_ret = x[shufflek, :]
    y_ret = y[shufflek, :]
    return x_ret, y_ret


if __name__ == "__main__":
    print("coucou")
    n_zz, x_zz = parse_test_prefixes("../data/prefixes/8.spice.prefix.public", 12)
    y_zz = parse_targets("../data/targets/8.spice.target.public", n_zz)
    # print(x)
    print(y_zz)
