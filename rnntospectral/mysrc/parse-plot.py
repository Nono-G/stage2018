import matplotlib.pyplot as mpl
import sys

measures = ["perp-test-target", "perp-test-rnn", "perp-test-extr",
            "perp-rand-target", "perp-rand-rnn", "perp-rand-extr",
            "kld-test-target-rnn", "kld-test-rnn-extr", "kld-test-target-extr",
            "kld-rand-target-rnn", "kld-rand-rnn-extr", "kld-rand-extr-rnn", "kld-rand-target-extr",
            "(1-wer)-test-target", "(1-wer)-test-rnn", "(1-wer)-test-extr",
            "(1-wer)-rnnw-rnn", "(1-wer)-rnnw-extr",
            "ndcg1-test-target-rnn", "ndcg1-test-rnn-extr", "ndcg1-test-target-extr", "ndcg1-rnnw-rnn-extr",
            "ndcg5-test-target-rnn", "ndcg5-test-rnn-extr", "ndcg5-test-target-extr","ndcg5-rnnw-rnn-extr",
            "l2dis-target-extr"]


def parse_one(filename, dico, n):
    with open(filename, 'r') as file:
        line = file.readline()
        while not line.startswith("+---"):
            line = file.readline()
        line = file.readline()
        ranks = [int(w) for w in line.split(sep="|")[2:-1]]
        for r in ranks:
            dico[n + "--" + str(r)] = []
        line = file.readline()
        line = file.readline()
        while not line.startswith("+---"):
            spl = line.split(sep="|")[2:-1]
            spl = [w.split(sep="(")[0] if len(w.split(sep="("))>1 else w for w in spl]
            for i, r in enumerate(ranks):
                dico[n + "--" + str(r)].append(float(spl[i]))
            line = file.readline()


def plot_some(d, files, caracs):
    leg = []
    for file in files:
        keys = d.keys()
        keys = [k for k in keys if k.split(sep="--")[0] == file]
        for c in caracs:
            abss = [k.split(sep="--")[1] for k in keys]
            ords = [d[k][c] for k in keys]
            mpl.plot(abss, ords,"+-", label="carac"+str(c))
        leg += [file+":"+measures[c] for c in caracs]
    mpl.legend(leg)
    mpl.show()


if __name__ == "__main__":
    pref = sys.argv[1]
    n = int(sys.argv[2])
    suff = sys.argv[3]
    nb = int(sys.argv[4])
    step = int(sys.argv[5])
    filezz = []
    for i in range(nb):
        filezz.append(pref+str(n+i*step)+suff)
    d = dict()
    i = 0
    parsed = []
    for fi in filezz:
        name = "f"+str(i)
        parse_one(fi, d, name)
        parsed.append(name)
        i += 1
    plot_some(d, parsed, [23, 25])
    plot_some(d, parsed, [measures.index("kld-rand-extr-rnn")])

    # tata = [4.1, 5.6, 0.2, 7.6, 8.9, 12.3]
    # abstata = [6,8,10,12,14,16]
    # mpl.plot(abstata, tata, '+-')
    # mpl.legend(["Wailers"])
    # mpl.xlabel("XXX")
    # mpl.ylabel("YYY")
    # mpl.show()
