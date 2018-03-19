import numpy as np


def _parsetrain(filename):
    with open(filename, "r") as file:
        (nsamples, nalpha) = [int(x) for x in file.readline().split()]
        data = []
        maxlen = 0
        for i in range(0, nsamples):
            line = file.readline().split()
            l = [_one_hot(int(x), nalpha + 2) for x in line[1:]] + [_one_hot(nalpha, nalpha + 2)]
            if int(line[0]) > maxlen:
                maxlen = int(line[0])
            data.append(l)
        return maxlen, nalpha, data


def _expand(seq, size, nalpha):
    return seq+([_one_hot(nalpha + 1, nalpha + 2)] * (size - len(seq) + 1))


def _sub(seq, nalpha):
    x = []
    y = []
    for i in range(1, len(seq)):
        x.append(seq[:i])  # +[_oneHot(nalpha+1, nalpha+2)])
        y += [seq[i][:(len(seq[i])-1)]]
    return x, y


def _allsub(seq, nalpha):
    x = []
    y = []
    for i in range(0, len(seq)):
        x2, y2 = _sub(seq[i], nalpha)
        x += x2
        y += y2
    return x, y


def parseTrainNpa(filename):
    maxlen, nalpha, data = _parsetrain(filename)
    x, y = _allsub(data, nalpha)
    x = [_expand(a, maxlen, nalpha) for a in x]
    # y = [_oneHot(a, nalpha+1) for a in y]
    with open(filename+".parseSummary", "w") as file:
        try:
            file.write(str(nalpha)+"\n")
            file.write(str(len(x[0])-1)+"\n")
        except :
            print("no summary produced !")
    return nalpha, np.array(x), np.array(y)


def parseTestNpa(filename, trainfilename):
    l = -1
    nalpha = 0
    maxlen = 0
    with open(trainfilename+".parseSummary", "r") as summary:
        nalpha = int(summary.readline())
        maxlen = int(summary.readline())
    with open(filename, "r") as file:
        l = [_one_hot(int(x), nalpha + 2) for x in file.readline().split()[1:]] + [_one_hot(nalpha, nalpha + 2)]
        x, y = _sub(l, nalpha)
        x = [_expand(a, maxlen, nalpha) for a in x]
        # y = [_oneHot(a, nalpha+1) for a in y]
    return np.array(x), np.array(y)


def _one_hot(i, size):
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


def softmax(seq):
    return _one_hot(argmax(seq), len(seq))


if __name__ == "__main__":
    pass
    # x,y = parsenpa('00.spice')
    # print(x)
    # print(y)
    # data = np.array(data)
    # print(l)
    # print(data)
    # x, y = parseTestNpa('0.spice.public.test', 4, 15)
    # print(x)
    # print(y)
