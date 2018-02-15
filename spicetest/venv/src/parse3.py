import numpy as np


def parse_train(filename, wid):
    xs = []
    ys = []
    with open(filename, 'r') as file:
        nsamples, nalpha = [int(a) for a in file.readline().split()]
        for line in file.readlines():
            xl, yl = parse_line(line, nalpha, wid)
            xs += xl
            ys += yl
    # On enlève 1 pour que les y restent dans l'espace 0-nalpha+1
    ys = [onehot(a-1, nalpha+2) for a in ys]
    return nalpha, np.array(xs), np.array(ys)


def parse_line(line, nalpha, wid):
    # on ajoute 1 partout pour que le zero soit libre pour servir de bourrage :
    seq = [nalpha+1]+[int(a)+1 for a in line.split()[1:]]+[nalpha+2]
    return windows(seq, wid)


def pad_0(seq, size):
    return seq + [0]*(size-len(seq))


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


if __name__ == "__main__":
    print("coucou")
    n, x, y = parse_train("../data/00.spice", 3)
    print(x)
    print(y)
