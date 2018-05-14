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
        try:
            dico_context[idn] = parse_context_string(context)
        except Exception:
            print(idn, "Error when parsing context")
        print(idn, "=", context)
        return 0


def parse_context_string(context):
    pb = int(context.split(sep="-")[0].split(sep="+")[1][1:])
    prefs = int(context.split(sep="(")[1].split(sep=")")[0])
    suffs = int(context.split(sep="(")[2].split(sep=")")[0])
    l_prefs = context.split(sep="(")[0].split(sep="l")[-1]
    l_suffs = context.split(sep="(")[1].split(sep="c")[-1]
    return pb, prefs, suffs, l_prefs, l_suffs


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


def plot_some_appart(dico, context_dico, pb, caracs_chunks, ranks):
    mpl.figure(1)
    data = []
    done = set()
    for key in dico.keys():
        idn,rk = key.split(sep="--")
        rk = int(rk)
        if idn not in done and context_dico[idn][0] == pb:
            done.add(idn)
            data.append([context_dico[idn],[]])
            for carch in caracs_chunks:
                data[-1][1].append([])
                for car in carch:
                    data[-1][1][-1].append([])
                    for r in ranks:
                        data[-1][1][-1][-1].append(dico[idn+"--"+str(r)][car])
    for i in range(len(caracs_chunks)):
        mpl.subplot((100*len(caracs_chunks)+11)+i)
        for c in range(len(caracs_chunks[i])):
            for j in range(len(data)):
                mpl.plot(ranks, [data[j][1][i][c][rk] for rk in range(len(ranks))], "x-")
        mpl.gca().set_title([mmm[car] for car in caracs_chunks[i]])
        # mpl.gca().set_xlabel("Rank")
        mpl.legend([str(data[j][0][1:])+":"+str(mmm[car]) for car in caracs_chunks[i] for j in range(len(data))],
                   bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., prop={'size':6})
    # mpl.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    mpl.tight_layout()
    mpl.subplots_adjust(left=0.03, right=0.85, hspace=0.15, bottom=0.05)
    mpl.gcf().suptitle("Problem {0}".format(pb))
    mpl.show()


def parse_some(input_line, files):
    if input_line == "Q":
        return 0
    elif input_line == "1-*":
        xxx = "extr1/OAR. 1709031 .stdout 288 1 0 1"
        print(xxx)
        return parse_some(xxx, files)
    elif input_line == "2-*":
        xxx = "extr2/OAR. 1710850 .stdout 432 1 0 1"
        print(xxx)
        return parse_some(xxx, files)
    else:
        splt = input_line.split()
        if len(splt) != 7:
            print("ERROR, expected : prefix n suffix total batch phase period")
            return -1
        pref = splt[0]
        n = int(splt[1])
        suff = splt[2]
        total = int(splt[3])
        batch = int(splt[4])
        phase = int(splt[5])
        period = int(splt[6])
        # start_r = int(splt[7])
        # end_r = int(splt[8])
        for i__ in range(total):
            for k__ in range(batch):
                files.append(pref + str(n + phase + k__ + i__ * period) + suff)


if __name__ == "__main__":
    # if len(sys.argv) != 10:
    #     print("prefix n suffix total batch phase period start-rank end-rank")
    #     exit(-666)
    # pref = sys.argv[1]
    # n = int(sys.argv[2])
    # suff = sys.argv[3]
    # total = int(sys.argv[4])
    # batch = int(sys.argv[5])
    # phase = int(sys.argv[6])
    # period = int(sys.argv[7])
    # start_r = int(sys.argv[8])
    # end_r = int(sys.argv[9])
    start_r = int(sys.argv[1])
    end_r = int(sys.argv[2])
    rank_range = range(start_r, end_r+1)
    filezz = []
    while parse_some(input(), filezz) != 0:
        pass  # Effets de bords =)
    # for i__ in range(total):
    #     for k__ in range(batch):
    #         filezz.append(pref + str(n + phase + k__ + i__ * period) + suff)
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
    # plot_some_appart(d, d2, 3, [[2], [7], [25]], rank_range)
    plot_overall_perp(d, d2)
    # plot_some(d, parsed, [2, 25])
    # plot_some(d, parsed, [2, 25])
    # plot_some(d, parsed, [2, 11, 12])

    problems = set([x[0] for x in d2.values()])
    for pb in problems:
        plot_some_appart(d, d2, pb, [[2],[7], [21], [25]], rank_range)
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
