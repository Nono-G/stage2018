import matplotlib.pyplot as mpl
import sys
import numpy as np
from sklearn.decomposition import PCA

mmm = ["perp-test-target", "perp-test-rnn", "perp-test-extr",  # 0 1 2
       "perp-rand-target", "perp-rand-rnn", "perp-rand-extr",  # 3 4 5
       "kld-test-target-rnn", "kld-test-rnn-extr", "kld-test-target-extr",  # 6 7 8
       "kld-rand-target-rnn", "kld-rand-rnn-extr", "kld-rand-extr-rnn", "kld-rand-target-extr",  # 9 10 11 12
       "(1-wer)-test-target", "(1-wer)-test-rnn", "(1-wer)-test-extr",  # 13 14 15
       "(1-wer)-rnnw-rnn", "(1-wer)-rnnw-extr",  # 16 17
       "ndcg1-test-target-rnn", "ndcg1-test-rnn-extr", "ndcg1-test-target-extr", "ndcg1-rnnw-rnn-extr",  # 18 19 20 21
       "ndcg5-test-target-rnn", "ndcg5-test-rnn-extr", "ndcg5-test-target-extr", "ndcg5-rnnw-rnn-extr",  # 22 23 24 25
       "l2dis-target-extr"]  # 26


def parse_one(filename, idn, dico, dico_context):
    with open(filename, 'r') as file:
        line = file.readline()
        context = line.split(sep=" ")[2][:-1]
        while line != "" and not line.startswith("+---"):
            line = file.readline()
        if line == "":
            print(idn, " : parse error in",filename)
            return -1
        line = file.readline()
        ranks = [int(w) for w in line.split(sep="|")[2:-1]]
        for r in ranks:
            dico[idn + "--" + str(r)] = []
        line = file.readline()
        line = file.readline()
        while not line.startswith("+---"):
            spl = line.split(sep="|")[2:-1]
            spl = [w.split(sep="(")[0] if len(w.split(sep="(")) > 1 else w for w in spl]
            for i, r in enumerate(ranks):
                dico[idn + "--" + str(r)].append(float(spl[i]))
            line = file.readline()
        print(idn, "=", context)
        try:
            dico_context[idn] = (int(context.split(sep="-")[0].split(sep="+")[1][1:]),)
        except Exception:
            print(idn, "Error when parsing context")
        return 0


def plot_some(dico, files, caracs, ranks=None):
    leg = []
    for file in files:
        keys = dico.keys()
        keys = [k for k in keys if k.split(sep="--")[0] == file]
        for c in caracs:
            abss = [k.split(sep="--")[1] for k in keys if (ranks is None or int(k.split(sep="--")[1]) in ranks)]
            ords = [dico[k][c] for k in keys if (ranks is None or int(k.split(sep="--")[1]) in ranks)]
            mpl.plot(abss, ords,"+-", label="carac"+str(c))
        leg += [file + ":" + mmm[c] for c in caracs]
    mpl.legend(leg)
    mpl.show()


def plot_pca(dico, files, ranks=None, caracs=None):
    if caracs is None:
        caracs = range(0, len(list(dico.values())[0]))
    pca = PCA(n_components=2)
    pca.fit(np.array([[v[c] for c in caracs] for v in dico.values()]))
    for file in files:
        data = []
        for k in dico.keys():
            key_split = k.split(sep="--")
            if key_split[0] == file:
                if ranks is None or int(key_split[1]) in ranks:
                    data.append([dico[k][c] for c in caracs])
        data = pca.transform(data)
        mpl.scatter(data[:, [0]], data[:, [1]])
    mpl.legend(files)
    mpl.show()


def plot_overall(dico):
    group = 9
    files = range(0,3)
    problems = [2,3,4]
    caracs = [0,1,2]
    discr = 2
    data = np.zeros((len(files), len(caracs)+1))
    for k in dico.keys():
        spl = k.split(sep="--")
        g = int(spl[0][1:]) // group
        if g in files:
            if data[g][0] == 0 or data[g][0] > dico[k][discr]:
                data[g][0] = dico[k][discr]
                for j in range(len(caracs)):
                    data[g][j+1] = dico[k][caracs[j]]
    print(data)
    leg = []
    for k in range(len(problems)):
        mpl.scatter(problems, data[:, [k]])
        leg.append(mmm[caracs[k]])
    mpl.legend(leg)
    mpl.show()


def plot_overall_perp(dico, dico_context, arg_problems=None, ranks=None):
    group = 9
    # problems = range(6,39)
    #
    if arg_problems is None:
        problems = list(range(1, 49))
    else:
        problems = list(arg_problems)
    data = dict()
    for pr in problems:
        data[pr] = [0]*(4+2)
    for k in dico.keys():
        spl = k.split(sep="--")
        params = int(spl[0][1:]) % group
        pr = dico_context[spl[0]][0]
        rank = int(spl[1])
        if pr in problems:
            if ranks is None or rank in ranks:
                if data[pr][2] == 0 or data[pr][2] > dico[k][2]:
                    data[pr][4] = rank
                    data[pr][5] = params
                    for j,c in enumerate([0, 1, 2, 25]):
                        data[pr][j] = dico[k][c]
    print(data)
    rel_data = dict()
    todel = []
    for j, pr in enumerate(problems):
        try:
            rel_data[pr] = [data[pr][0] / data[pr][1], data[pr][1] / data[pr][2]]
        except ZeroDivisionError:
            todel.append(j-len(todel))
    for j in todel:
        del problems[j]
    mpl.scatter(problems, [rel_data[pr][0] for pr in problems])
    mpl.scatter(problems, [rel_data[pr][1] for pr in problems])
    # mpl.scatter(problems, data[:, [3]])
    mpl.legend(["r/t", "e/r", "ndcg5"])
    print([data[pr][4] for pr in problems])
    print([data[pr][5]+1 for pr in problems])
    mpl.grid()
    mpl.show()


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("prefix n suffix nb step start-rank end-rank")
        exit(-666)
    pref = sys.argv[1]
    n = int(sys.argv[2])
    suff = sys.argv[3]
    nb = int(sys.argv[4])
    step = int(sys.argv[5])
    start_r = int(sys.argv[6])
    end_r = int(sys.argv[7])
    filezz = []
    for i in range(nb):
        filezz.append(pref + str(n + i * step) + suff)
    d = dict()
    d2 = dict()
    file_number = 0
    parsed = []
    for fi in filezz:
        name = "f"+str(file_number)
        ret = parse_one(fi, name, d, d2)
        if ret == 0:
            parsed.append(name)
        file_number += 1
    # plot_overall(d)
    # plot_overall_perp(d, d2, problems=[i for i in range(1,30) if i not in [5, 20,21,22,23]])
    plot_overall_perp(d, d2)
    plot_some(d, parsed, [2, 25])
    plot_some(d, parsed, [2, 25])
    plot_some(d, parsed, [2, 11, 12])
    # plot_some(d, parsed, [0, 1, 2], ranks=range(start_r, end_r))
    # plot_some(d, parsed, [1, 2], ranks=range(start_r, end_r))
    # plot_some(d, parsed, [25], ranks=range(start_r, end_r))
    # plot_pca(d, parsed)
    # plot_pca(d, parsed, caracs=[i for i in range(0,27) if i not in [0, 3, 4, 5, 9, 10, 11, 12, 26]])
    # plot_some(d, parsed, [measures.index("kld-rand-extr-rnn")])

    # mpl.legend(["Wailers"])
    # mpl.xlabel("XXX")
    # mpl.ylabel("YYY")
    # mpl.show()
