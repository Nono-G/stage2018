import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as host
import matplotlib as mpl
import sys
import numpy as np
from math import log
from sklearn.decomposition import PCA

mmm = ["perp-test-target", "perp-test-rnn", "perp-test-extr",  # 0 1 2
       "perp-rand-target", "perp-rand-rnn", "perp-rand-extr",  # 3 4 5
       "kld-test-target-rnn", "kld-test-rnn-extr", "kld-test-target-extr",  # 6 7 8
       "kld-rand-target-rnn", "kld-rand-rnn-extr", "kld-rand-extr-rnn", "kld-rand-target-extr",  # 9 10 11 12
       "(1-wer)-test-target", "(1-wer)-test-rnn", "(1-wer)-test-extr",  # 13 14 15
       "(1-wer)-rnnw-rnn", "(1-wer)-rnnw-extr",  # 16 17
       "ndcg1-test-target-rnn", "ndcg1-test-rnn-extr", "ndcg1-test-target-extr", "ndcg1-rnnw-rnn-extr",  # 18 19 20 21
       "ndcg5-test-target-rnn", "ndcg5-test-rnn-extr", "ndcg5-test-target-extr", "ndcg5-rnnw-rnn-extr",  # 22 23 24 25
       "perprnn-test-rnn", "perprnn-test-extr", "perprnn-rnnw-rnn", "perprnn-rnnw-extr"  # 26 27 28 29
       # "l2dis-target-extr"
       ]  # 26

smm = ["kld-test-rnn-extr",  # 0
       "(1-wer)-test-rnn", "(1-wer)-test-extr",  # 1 2
       "(1-wer)-rnnw-rnn", "(1-wer)-rnnw-extr",  # 3 4
       "ndcg1-test-rnn-extr", "ndcg1-rnnw-rnn-extr",  # 5 6
       "ndcg5-test-rnn-extr",  # 7
       "ndcg5-rnnw-rnn-extr",  # 8
       "perprnn-test-rnn", "perprnn-test-extr", "perprnn-rnnw-rnn", "perprnn-rnnw-extr"  # 9 10 11 12
       ]


def parse_one(filename, idn, dico, dico_context, dic_eps):
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
            dico[(idn,r)] = []
        line = file.readline()
        line = file.readline()
        while not line.startswith("+---"):
            spl = line.split(sep="|")[2:-1]
            spl = [(w.split(sep="(")[0],w.split(sep="(")[1]) if len(w.split(sep="(")) > 1 else (w,) for w in spl]
            for i, r in enumerate(ranks):
                dico[(idn,r)].append(float(spl[i][0]))
                if len(spl[i]) > 1:
                    # EPS !
                    e = float(spl[i][1].split(sep="%")[0])
                    dic_eps[(idn,r,len(dico[(idn,r)])-1)] = e
            line = file.readline()
        try:
            dico_context[idn] = parse_context_string(context)
        except Exception:
            print(idn, "Error when parsing context, parser where made tight for some experiments,"
                       " you probably need to re-implement parse_context_string function")
        print(idn, "=", context)
        return 0


def parse_context_string(context):
    if context.startswith("H5"):
        # aut-H5-models+pautartif2-d_models+pautartif2-wl8c8-r-3
        return None
    else:
        # aut-models+m48-d_models+m48-wl0_10(1400)c0_10(1400)-r-1
        # models+m1-d_models+m1-wl0_4(800)c0_4(800)
        if context.startswith("aut-"):
            context = context[4:]
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
            plt.plot(abss, ords, "+-", label="carac" + str(c))
        leg += [file + ":" + mmm[c] for c in caracs]
    plt.legend(leg)
    plt.show()


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
        plt.scatter(data[:, [0]], data[:, [1]])
    plt.legend(files)
    plt.show()


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
        plt.scatter(problems, data[:, [k]])
        leg.append(mmm[caracs[k]])
    plt.legend(leg)
    plt.show()


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
    plt.scatter(problems, [rel_data[pr][0] for pr in problems])
    plt.scatter(problems, [rel_data[pr][1] for pr in problems])
    # mpl.scatter(problems, data[:, [3]])
    plt.legend(["r/t", "e/r", "ndcg5"])
    print([data[pr][4] for pr in problems])
    print([data[pr][5]+1 for pr in problems])
    plt.grid()
    plt.show()


def plot_some_appart(dico, context_dico, pb, caracs_chunks, ranks):
    plt.figure(1)
    data = []
    done = set()
    for (idn,rk) in dico.keys():
        # idn,rk = key.split(sep="--")
        # rk = int(rk)
        if idn not in done and context_dico[idn][0] == pb:
            done.add(idn)
            data.append([context_dico[idn],[]])
            for carch in caracs_chunks:
                data[-1][1].append([])
                for car in carch:
                    data[-1][1][-1].append([])
                    for r in ranks:
                        data[-1][1][-1][-1].append(dico[(idn,r)][car])
    for i in range(len(caracs_chunks)):
        plt.subplot((100 * len(caracs_chunks) + 11) + i)
        for c in range(len(caracs_chunks[i])):
            for j in range(len(data)):
                plt.plot(ranks, [data[j][1][i][c][rk] for rk in range(len(ranks))], "x-")
        plt.gca().set_title([mmm[car] for car in caracs_chunks[i]])
        # mpl.gca().set_xlabel("Rank")
        plt.legend([str(data[j][0][1:]) + ":" + str(mmm[car]) for car in caracs_chunks[i] for j in range(len(data))],
                   bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., prop={'size':6})
    # mpl.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.tight_layout()
    plt.subplots_adjust(left=0.03, right=0.85, hspace=0.15, bottom=0.05)
    plt.gcf().suptitle("Problem {0}".format(pb))
    plt.show()


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


def best(dico, context_dico, discr, pb):
    best_key = None
    for (id,r) in dico.keys():
        if context_dico[id][0] == pb:
            if best_key is None or discr(dico[best_key], dico[(id, r)]) < 0:
                best_key = (id,r)
    return best_key


def latex_best_rnn(dic, cdic, edic, problems):
    print("\\begin{table}")
    print("\\centering")
    print("\\begin{tabular}{|c||c|c|c|c||c|c|c|}")
    print("\\hline")
    print("Pb. & \\multicolumn{4}{c||}{Perplexity Ratio} & \\multicolumn{3}{c|}{NDCG-5} \\\\ \\hline")
    print("\\# & Rank & (P,S) & Value & Zeros & Rank & (P,S) & Value \\\\ \\hline")
    for p in problems:
        (id_p,r_p) = best(dic, cdic, (lambda x, y: y[29]-x[29]), p)
        (id_n,r_n) = best(dic, cdic, (lambda x, y: x[25]-y[25]), p)
        print("{0} & {1} & ({2},{3}) & {4:7.5f} & {5} \\% & {6} & ({7},{8}) & {9:7.5f}\\\\ \\hline"
              .format(p,
                      r_p, cdic[id_p][1], cdic[id_p][2], dic[(id_p,r_p)][28]/dic[(id_p,r_p)][29], edic[(id_p,r_p,2)],
                      r_n, cdic[id_n][1], cdic[id_n][2], dic[(id_n,r_n)][23],)
              .replace("_", "\\_"))
    print("\\end{tabular}")
    print("\\end{table}")
    print("")


def latex_best_test(dic, cdic, edic, problems):
    print("\\begin{table}")
    print("\\centering")
    print("\\begin{tabular}{|c||c|c|c|c||c|c|c|}")
    print("\\hline")
    print("Pb. & \\multicolumn{4}{c||}{Perplexity Ratio} & \\multicolumn{3}{c|}{NDCG-5} \\\\ \\hline")
    print("\\# & Rank & (P,S) & Value & Zeros & Rank & (P,S) & Value \\\\ \\hline")
    for p in problems:
        (id_p,r_p) = best(dic, cdic, (lambda x, y: y[2]-x[2]), p)
        (id_n,r_n) = best(dic, cdic, (lambda x, y: x[23]-y[23]), p)
        print("{0} & {1} & ({2},{3}) & {4:7.5f} & {5} \\% & {6} & ({7},{8}) & {9:7.5f}\\\\ \\hline"
              .format(p,
                      r_p, cdic[id_p][1], cdic[id_p][2], dic[(id_p,r_p)][1]/dic[(id_p,r_p)][2], edic[(id_p,r_p,2)],
                      r_n, cdic[id_n][1], cdic[id_n][2], dic[(id_n,r_n)][23],)
              .replace("_", "\\_"))
    print("\\end{tabular}")
    print("\\end{table}")
    print("")


def fabulous_four(dic, cdic, problems, opt=0):
    bests_perp_t = []
    bests_ndcg_t = []
    bests_perp_r = []
    bests_ndcg_r = []
    titles=["Overall best results on Pautomac",
            "Results of best parameters for perplexity ratio on $S_{test}$",
            "Results of best parameters for NDCG-5 on $S_{test}$",
            "Results of best parameters for perplexity ratio on $S_{RNN}$",
            "Results of best parameters for NDCG-5 on $S_{RNN}$"
            ]
    for pb in problems:
        bests_perp_t.append(best(dic, cdic, (lambda o, n: n[2]-o[2]), pb))
        bests_ndcg_t.append(best(dic, cdic, (lambda o, n: o[24]-n[24]), pb))
        bests_perp_r.append(best(dic, cdic, (lambda o, n: n[29]-o[29]), pb))
        bests_ndcg_r.append(best(dic, cdic, (lambda o, n: o[25]-n[25]), pb))
    if opt == 1:
        # bests_perp_t = bests_perp_t
        bests_ndcg_t = bests_perp_t
        bests_perp_r = bests_perp_t
        bests_ndcg_r = bests_perp_t
    if opt == 2:
        bests_perp_t = bests_ndcg_t
        # bests_ndcg_t = bests_ndcg_t
        bests_perp_r = bests_ndcg_t
        bests_ndcg_r = bests_ndcg_t
    if opt == 3:
        bests_perp_t = bests_perp_r
        bests_ndcg_t = bests_perp_r
        # bests_perp_r = bests_perp_r
        bests_ndcg_r = bests_perp_r
    if opt == 4:
        bests_perp_t = bests_ndcg_r
        bests_ndcg_t = bests_ndcg_r
        bests_perp_r = bests_ndcg_r
        # bests_ndcg_r = bests_ndcg_r
    plt.figure(1).suptitle(titles[opt])
    if opt == 1:
        mpl.rc('axes', linewidth=3)
    plt.subplot(221).set_title("Perplexity Ratio")
    plt.gca().xaxis.grid(True)
    plt.scatter([problems[i] for i in range(len(problems))], [dic[bests_perp_t[i]][1] / dic[bests_perp_t[i]][2] for i in range(len(problems))])
    plt.scatter([problems[i] for i in range(len(problems))], [dic[bests_perp_t[i]][0] / dic[bests_perp_t[i]][1] for i in range(len(problems))], marker="+")
    plt.ylabel("on $S_{test}$")
    plt.legend(["RNN/WA", "Target/RNN"])
    mpl.rc('axes', linewidth=1)
    if opt == 2:
        mpl.rc('axes', linewidth=3)
    plt.subplot(222).set_title("NDCG-5")
    plt.gca().xaxis.grid(True)
    plt.scatter(problems, [dic[bests_ndcg_t[i]][24] for i in range(len(problems))])
    # mpl.legend(["ndcg5-rnn-extr"])
    mpl.rc('axes', linewidth=1)
    if opt == 3:
        mpl.rc('axes', linewidth=3)
    plt.subplot(223)
    plt.gca().xaxis.grid(True)
    plt.scatter([problems[i] for i in range(len(problems))], [dic[bests_perp_r[i]][28] / dic[bests_perp_r[i]][29] for i in range(len(problems))])
    plt.ylabel('on $S_{RNN}$')
    mpl.rc('axes', linewidth=1)
    if opt == 4:
        mpl.rc('axes', linewidth=3)
    plt.subplot(224)
    plt.gca().xaxis.grid(True)
    plt.scatter(problems, [dic[bests_ndcg_r[i]][25] for i in range(len(problems))])
    mpl.rc('axes', linewidth=1)
    # mpl.legend(["ndcg5-rnn-extr"])
    plt.show()


def fabulous_four_hist(dic, cdic, problems, opt=0):
    steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    bests_perp_t = []
    bests_ndcg_t = []
    bests_perp_r = []
    bests_ndcg_r = []
    # titles=["Overall best results on Pautomac",
    #         "Results of best parameters for perplexity ratio on $S_{test}$",
    #         "Results of best parameters for NDCG-5 on $S_{test}$",
    #         "Results of best parameters for perplexity ratio on $S_{RNN}$",
    #         "Results of best parameters for NDCG-5 on $S_{RNN}$"
    #         ]
    for pb in problems:
        bests_perp_t.append(best(dic, cdic, (lambda o, n: n[2]-o[2]), pb))
        bests_ndcg_t.append(best(dic, cdic, (lambda o, n: o[24]-n[24]), pb))
        bests_perp_r.append(best(dic, cdic, (lambda o, n: n[29]-o[29]), pb))
        bests_ndcg_r.append(best(dic, cdic, (lambda o, n: o[25]-n[25]), pb))
    if opt == 1:
        # bests_perp_t = bests_perp_t
        bests_ndcg_t = bests_perp_t
        bests_perp_r = bests_perp_t
        bests_ndcg_r = bests_perp_t
    if opt == 2:
        bests_perp_t = bests_ndcg_t
        # bests_ndcg_t = bests_ndcg_t
        bests_perp_r = bests_ndcg_t
        bests_ndcg_r = bests_ndcg_t
    if opt == 3:
        bests_perp_t = bests_perp_r
        bests_ndcg_t = bests_perp_r
        # bests_perp_r = bests_perp_r
        bests_ndcg_r = bests_perp_r
    if opt == 4:
        bests_perp_t = bests_ndcg_r
        bests_ndcg_t = bests_ndcg_r
        bests_perp_r = bests_ndcg_r
        # bests_ndcg_r = bests_ndcg_r
    plt.figure(1)  # .suptitle(titles[opt])
    plt.subplots_adjust(left=0.07, bottom=0.09, right=0.98, top=0.95, wspace=0.16, hspace=0.46)
    plt.gca().grid(True)
    if opt == 1:
        mpl.rc('axes', linewidth=3)
    sb1 = plt.subplot(221)
    sb1.set_title("on $S_{test}$")
    plt.gca().xaxis.grid(True)
    plt.xlabel("PAutomaC problem")
    plt.ylabel("Perplexity Ratio")
    sb1.set_ylim(0, 1.1)
    plt.scatter([problems[i] for i in range(len(problems))], [dic[bests_perp_t[i]][1] / dic[bests_perp_t[i]][2] for i in range(len(problems))])
    plt.scatter([problems[i] for i in range(len(problems))], [dic[bests_perp_t[i]][0] / dic[bests_perp_t[i]][1] for i in range(len(problems))], marker="s")
    plt.legend(["RNN/WA", "Target/RNN"])
    mpl.rc('axes', linewidth=1)
    if opt == 2:
        mpl.rc('axes', linewidth=3)
    sb2 = plt.subplot(222)
    sb2.set_title("on $S_{test}$")
    plt.xlabel("NDCG$_5$")
    plt.ylabel("# of problems")
    plt.xticks(np.arange(0, 1.1, step=0.1))
    # mpl.gca().xaxis.grid(True)
    # mpl.scatter(problems, [dic[bests_ndcg_t[i]][24] for i in range(len(problems))])
    plt.hist([dic[bests_ndcg_t[i]][24] for i in range(len(problems))], steps, rwidth=0.8)
    # mpl.legend(["ndcg5-rnn-extr"])
    mpl.rc('axes', linewidth=1)
    if opt == 3:
        mpl.rc('axes', linewidth=3)
    sb3 = plt.subplot(223)
    sb3.set_title("on $S_{RNN}$")
    sb3.set_ylim(0, 1.1)
    plt.xlabel("PAutomaC problem")
    plt.gca().xaxis.grid(True)
    plt.scatter([problems[i] for i in range(len(problems))], [dic[bests_perp_r[i]][28] / dic[bests_perp_r[i]][29] for i in range(len(problems))])
    plt.ylabel("Perplexity Ratio")
    mpl.rc('axes', linewidth=1)
    if opt == 4:
        mpl.rc('axes', linewidth=3)
    sb4 = plt.subplot(224)
    sb4.set_title("on $S_{RNN}$")
    plt.xlabel("NDCG$_5$")
    plt.ylabel("# of problems")
    plt.xticks(np.arange(0, 1.1, step=0.1))
    # mpl.gca().xaxis.grid(True)
    # mpl.scatter(problems, [dic[bests_ndcg_r[i]][25] for i in range(len(problems))])
    plt.hist([dic[bests_ndcg_r[i]][25] for i in range(len(problems))], steps, rwidth=0.8)
    mpl.rc('axes', linewidth=1)
    # mpl.legend(["ndcg5-rnn-extr"])
    plt.show()


def fabulous_four_hist_WER(dic, cdic, problems, opt=0):
    steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    bests_wer_t = []
    bests_ndcg_t = []
    bests_wer_r = []
    bests_ndcg_r = []
    # titles=["Overall best results on Pautomac",
    #         "Results of best parameters for perplexity ratio on $S_{test}$",
    #         "Results of best parameters for NDCG-5 on $S_{test}$",
    #         "Results of best parameters for perplexity ratio on $S_{RNN}$",
    #         "Results of best parameters for NDCG-5 on $S_{RNN}$"
    #         ]
    for pb in problems:
        bests_wer_t.append(best(dic, cdic, (lambda o, n: o[15]-n[15]), pb))
        bests_ndcg_t.append(best(dic, cdic, (lambda o, n: o[20]-n[20]), pb))
        bests_wer_r.append(best(dic, cdic, (lambda o, n: o[17]-n[17]), pb))
        bests_ndcg_r.append(best(dic, cdic, (lambda o, n: o[21]-n[21]), pb))
    if opt == 1:
        # bests_wer_t = bests_wer_t
        bests_ndcg_t = bests_wer_t
        bests_wer_r = bests_wer_t
        bests_ndcg_r = bests_wer_t
    if opt == 2:
        bests_wer_t = bests_ndcg_t
        # bests_ndcg_t = bests_ndcg_t
        bests_wer_r = bests_ndcg_t
        bests_ndcg_r = bests_ndcg_t
    if opt == 3:
        bests_wer_t = bests_wer_r
        bests_ndcg_t = bests_wer_r
        # bests_wer_r = bests_wer_r
        bests_ndcg_r = bests_wer_r
    if opt == 4:
        bests_wer_t = bests_ndcg_r
        bests_ndcg_t = bests_ndcg_r
        bests_wer_r = bests_ndcg_r
        # bests_ndcg_r = bests_ndcg_r
    plt.figure(1)  # .suptitle(titles[opt])
    plt.subplots_adjust(left=0.07, bottom=0.09, right=0.98, top=0.95, wspace=0.16, hspace=0.46)
    plt.gca().grid(True)
    if opt == 1:
        mpl.rc('axes', linewidth=3)
    sb1 = plt.subplot(221)
    sb1.set_title("on $S_{test}$")
    plt.gca().xaxis.grid(True)
    plt.xlabel("PAutomaC problem")
    plt.ylabel("Word Error Rate")
    sb1.set_ylim(0, 1.1)
    plt.scatter([problems[i] for i in range(len(problems))], [1-dic[bests_wer_t[i]][15] for i in range(len(problems))])
    plt.scatter([problems[i] for i in range(len(problems))], [1-dic[bests_wer_t[i]][14] for i in range(len(problems))], marker="s")
    plt.legend(["WA", "RNN"])
    mpl.rc('axes', linewidth=1)
    if opt == 2:
        mpl.rc('axes', linewidth=3)
    sb2 = plt.subplot(222)
    sb2.set_title("on $S_{test}$")
    plt.xlabel("NDCG$_1$")
    plt.ylabel("# of problems")
    plt.xticks(np.arange(0, 1.1, step=0.1))
    # mpl.gca().xaxis.grid(True)
    # mpl.scatter(problems, [dic[bests_ndcg_t[i]][24] for i in range(len(problems))])
    plt.hist([dic[bests_ndcg_t[i]][24] for i in range(len(problems))], steps, rwidth=0.8)
    # mpl.legend(["ndcg5-rnn-extr"])
    mpl.rc('axes', linewidth=1)
    if opt == 3:
        mpl.rc('axes', linewidth=3)
    sb3 = plt.subplot(223)
    sb3.set_title("on $S_{RNN}$")
    sb3.set_ylim(0, 1.1)
    plt.xlabel("PAutomaC problem")
    plt.gca().xaxis.grid(True)
    plt.scatter([problems[i] for i in range(len(problems))], [1-dic[bests_wer_r[i]][17] for i in range(len(problems))])
    plt.scatter([problems[i] for i in range(len(problems))], [1-dic[bests_wer_r[i]][16] for i in range(len(problems))], marker="s")
    plt.legend(["WA", "RNN"])
    plt.ylabel("Word Error Rate")
    mpl.rc('axes', linewidth=1)
    if opt == 4:
        mpl.rc('axes', linewidth=3)
    sb4 = plt.subplot(224)
    sb4.set_title("on $S_{RNN}$")
    plt.xlabel("NDCG$_1$")
    plt.ylabel("# of problems")
    plt.xticks(np.arange(0, 1.1, step=0.1))
    # mpl.gca().xaxis.grid(True)
    # mpl.scatter(problems, [dic[bests_ndcg_r[i]][25] for i in range(len(problems))])
    plt.hist([dic[bests_ndcg_r[i]][25] for i in range(len(problems))], steps, rwidth=0.8)
    mpl.rc('axes', linewidth=1)
    # mpl.legend(["ndcg5-rnn-extr"])
    plt.show()


def details(fs):
    dic = dict()
    cdic = dict()
    edic = dict()
    i=0
    for f in fs:
        parse_one(f,i,dic,cdic, edic)
        i += 1
    plt.figure().suptitle("Natural Problem : Spice 4")
    plt.subplot(311)
    for k in range(0,i):
        ranks=set()
        for (idn, rk) in dic.keys():
            if idn == k:
                ranks.add(rk)
        plt.plot(sorted(list(ranks)), [dic[(k, r)][11] / dic[(k, r)][12] for r in sorted(list(ranks))], "+-")
        plt.legend(["Perplexity ratio"])
    plt.subplot(312)
    # for k in range(0,i):
    #     ranks = set()
    #     for (idn, rk) in dic.keys():
    #         if idn == k:
    #             ranks.add(rk)
    #     mpl.plot(sorted(list(ranks)), [dic[(k,r)][12] for r in sorted(list(ranks))], "+-")
    plt.subplot(313)
    for k in range(0,i):
        ranks=set()
        for (idn, rk) in dic.keys():
            if idn == k:
                ranks.add(rk)
        plt.plot(sorted(list(ranks)), [dic[(k, r)][8] for r in sorted(list(ranks))], "+-")
        plt.legend(["NDCG-5 on $S_{RNN}$"])
    plt.show()
    print("")


def details2(fs):
    dic = dict()
    cdic = dict()
    edic = dict()
    i = 0
    for f in fs:
        parse_one(f, i, dic, cdic, edic)
        i += 1
    plt.figure().suptitle("Artificial Problem : Pautomac 37")
    plt.subplot(311)
    for k in range(0, i):
        ranks = set()
        for (idn, rk) in dic.keys():
            if idn == k:
                ranks.add(rk)
        plt.plot(sorted(list(ranks)), [dic[(k, r)][1] / dic[(k, r)][2] for r in sorted(list(ranks))], "+-")
        plt.legend(["Perplexity ratio"])
    plt.subplot(312)
    # for k in range(0,i):
    #     ranks = set()
    #     for (idn, rk) in dic.keys():
    #         if idn == k:
    #             ranks.add(rk)
    #     mpl.plot(sorted(list(ranks)), [dic[(k,r)][12] for r in sorted(list(ranks))], "+-")
    plt.subplot(313)
    for k in range(0, i):
        ranks = set()
        for (idn, rk) in dic.keys():
            if idn == k:
                ranks.add(rk)
        plt.plot(sorted(list(ranks)), [dic[(k, r)][25] for r in sorted(list(ranks))], "+-")
        plt.legend(["NDCG-5 on $S_{RNN}$"])
    plt.show()
    print("")


def details3eps(dic, cdic, edic, nat, p1, p2):
    plt.figure()
    plt.subplots_adjust(wspace=0.40)
    #
    dicnat = dict()
    cdicnat = dict()
    edicnat = dict()
    parse_one(nat, 0, dicnat, cdicnat, edicnat)
    # mpl.subplot(131).set_title("RNN perp PautNat2")
    h1 = host.host_subplot(131)
    h1.set_title("PAutomaC Nat. 2\n|(P,S)| = (800,800)")
    plt.xlabel("WA rank")
    h1.set_ylabel("Perplexity")
    epax1 = h1.twinx()
    epax1.set_ylim(0,100)
    epax1.set_ylabel("Zeros %")
    rrange = range(5, 150)
    h1.plot([r for r in rrange], [(dicnat[(0,r)][12]) for r in rrange], "x-")
    h1.plot([r for r in rrange], [(dicnat[(0,r)][11]) for r in rrange], "+-")
    epax1.plot([r for r in rrange], [edicnat[(0,r,12)] for r in rrange], ".--")
    plt.legend(["RNN-WA", " RNN-RNN", "Zeros (%)"])
    plt.draw()
    #
    (idn,br) = best(dic, cdic, (lambda x, y: y[29]-x[29]), p1)
    h2 = host.host_subplot(132)
    h2.set_title("PAutomaC {0}\n|(P,S)| = ({1},{2})".format(p1, cdic[idn][1], cdic[idn][2]))
    plt.xlabel("WA rank")
    h2.set_ylabel("Perplexity")
    epax2 = h2.twinx()
    epax2.set_ylim(0, 100)
    epax2.set_ylabel("Zeros %")
    rrange = range(4,100)
    h2.plot([r for r in rrange], [(dic[(idn,r)][29]) for r in rrange], "x-")
    h2.plot([r for r in rrange], [(dic[(idn,r)][28]) for r in rrange], "+-")
    epax2.plot([r for r in rrange], [edic[(idn, r, 29)] for r in rrange], ".--")
    plt.legend(["RNN-WA", " RNN-RNN", "Zeros (%)"])
    #
    (idn, br) = best(dic, cdic, (lambda x, y: y[29] - x[29]), p2)
    h3 = host.host_subplot(133)
    h3.set_title("PAutomaC {0}\n|(P,S)| = ({1},{2})".format(p2, cdic[idn][1], cdic[idn][2]))
    plt.xlabel("WA rank")
    h3.set_ylabel("Perplexity")
    epax3 = h3.twinx()
    epax3.set_ylim(0, 100)
    epax3.set_ylabel("Zeros %")
    rrange = range(4, 100)
    h3.plot([r for r in rrange], [(dic[(idn, r)][29]) for r in rrange], "x-")
    h3.plot([r for r in rrange], [(dic[(idn, r)][28]) for r in rrange], "+-")
    epax3.plot([r for r in rrange], [edic[(idn, r, 29)] for r in rrange], ".--")
    plt.legend(["RNN-WA", " RNN-RNN", "Zeros (%)"])
    #
    plt.show()


def details3epsKL(dic, cdic, edic, nat, p1, p2):
    plt.figure()
    plt.subplots_adjust(wspace=0.40)
    #
    dicnat = dict()
    cdicnat = dict()
    edicnat = dict()
    parse_one(nat, 0, dicnat, cdicnat, edicnat)
    # mpl.subplot(131).set_title("RNN perp PautNat2")
    h1 = host.host_subplot(131)
    h1.set_title("PAutomaC Nat. 2\n|(P,S)| = (800,800)")
    plt.xlabel("WA rank")
    h1.set_ylabel("KL-divergence")
    epax1 = h1.twinx()
    epax1.set_ylim(0,100)
    epax1.set_ylabel("Zeros %")
    rrange = range(5, 150)
    h1.plot([r for r in rrange], [(dicnat[(0,r)][0]) for r in rrange], "x-")
    # h1.plot([r for r in rrange], [log(dicnat[(0,r)][11]) for r in rrange], "+-")
    h1.plot([],[])
    epax1.plot([r for r in rrange], [edicnat[(0,r,0)] for r in rrange], ".--")
    plt.legend(["RNN-WA", "Zeros (%)"])
    plt.draw()
    #
    (idn,br) = best(dic, cdic, (lambda x, y: x[7]-y[7]), p1)
    h2 = host.host_subplot(132)
    h2.set_title("PAutomaC {0}\n|(P,S)| = ({1},{2})".format(p1, cdic[idn][1], cdic[idn][2]))
    plt.xlabel("WA rank")
    h2.set_ylabel("KL-divergence")
    epax2 = h2.twinx()
    epax2.set_ylim(0, 100)
    epax2.set_ylabel("Zeros %")
    rrange = range(4,100)
    h2.plot([r for r in rrange], [(dic[(idn,r)][7]) for r in rrange], "x-")
    # h2.plot([r for r in rrange], [log(dic[(idn,r)][28]) for r in rrange], "+-")
    h2.plot([],[])
    epax2.plot([r for r in rrange], [edic[(idn, r, 7)] for r in rrange], ".--")
    plt.legend(["RNN-WA", "Zeros (%)"])
    #
    (idn, br) = best(dic, cdic, (lambda x, y: x[7] - y[7]), p2)
    h3 = host.host_subplot(133)
    h3.set_title("PAutomaC {0}\n|(P,S)| = ({1},{2})".format(p2, cdic[idn][1], cdic[idn][2]))
    plt.xlabel("WA rank")
    h3.set_ylabel("KL-divergence")
    epax3 = h3.twinx()
    epax3.set_ylim(0, 100)
    epax3.set_ylabel("Zeros %")
    rrange = range(4, 100)
    h3.plot([r for r in rrange], [(dic[(idn, r)][7]) for r in rrange], "x-")
    # h3.plot([r for r in rrange], [log(dic[(idn, r)][28]) for r in rrange], "+-")
    h3.plot([])
    epax3.plot([r for r in rrange], [edic[(idn, r, 7)] for r in rrange], ".--")
    plt.legend(["RNN-WA", "Zeros (%)"])
    #
    plt.show()


def details3(dic, cdic, edic, nat, p1, p2):
    plt.figure()
    plt.subplots_adjust(wspace=0.40)
    #
    dicnat = dict()
    cdicnat = dict()
    edicnat = dict()
    parse_one(nat, 0, dicnat, cdicnat, edicnat)
    # mpl.subplot(131).set_title("RNN perp PautNat2")
    h1 = host.host_subplot(131)
    plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    h1.set_title("PAutomaC Nat. 2\n|(P,S)| = (800,800)")
    plt.xlabel("WA rank")
    h1.set_ylabel("Perplexity")
    # epax1 = h1.twinx()
    # epax1.set_ylim(0,100)
    # epax1.set_ylabel("Zeros %")
    rrange = range(5, 150)
    h1.plot([r for r in rrange], [(dicnat[(0,r)][12]) for r in rrange], "x-")
    h1.plot([r for r in rrange], [(dicnat[(0,r)][11]) for r in rrange], "+-")
    # epax1.plot([r for r in rrange], [edicnat[(0,r,12)] for r in rrange], ".--")
    plt.legend(["RNN-WA", " RNN-RNN"])
    #
    (idn,br) = best(dic, cdic, (lambda x, y: y[29]-x[29]), p1)
    h2 = host.host_subplot(132)
    h2.set_title("PAutomaC {0}\n|(P,S)| = ({1},{2})".format(p1, cdic[idn][1], cdic[idn][2]))
    plt.xlabel("WA rank")
    h2.set_ylabel("Perplexity")
    # epax2 = h2.twinx()
    # epax2.set_ylim(0, 100)
    # epax2.set_ylabel("Zeros %")
    rrange = range(4,100)
    h2.plot([r for r in rrange], [(dic[(idn,r)][29]) for r in rrange], "x-")
    h2.plot([r for r in rrange], [(dic[(idn,r)][28]) for r in rrange], "+-")
    # epax2.plot([r for r in rrange], [edic[(idn, r, 29)] for r in rrange], ".--")
    plt.legend(["RNN-WA", " RNN-RNN"])
    #
    (idn, br) = best(dic, cdic, (lambda x, y: y[29] - x[29]), p2)
    h3 = host.host_subplot(133)
    h3.set_title("PAutomaC {0}\n|(P,S)| = ({1},{2})".format(p2, cdic[idn][1], cdic[idn][2]))
    plt.xlabel("WA rank")
    h3.set_ylabel("Perplexity")
    # epax3 = h3.twinx()
    # epax3.set_ylim(0, 100)
    # epax3.set_ylabel("Zeros %")
    rrange = range(4, 100)
    h3.plot([r for r in rrange], [(dic[(idn, r)][29]) for r in rrange], "x-")
    h3.plot([r for r in rrange], [(dic[(idn, r)][28]) for r in rrange], "+-")
    # epax3.plot([r for r in rrange], [edic[(idn, r, 29)] for r in rrange], ".--")
    plt.legend(["RNN-WA", " RNN-RNN"])
    #
    plt.show()


def details3_old(dic, cdic, nat, p1, p2):
    plt.figure().suptitle("Title")
    #
    dicnat = dict()
    cdicnat = dict()
    edicnat = dict()
    parse_one(nat, 0, dicnat, cdicnat, edicnat)
    plt.subplot(131).set_title("RNN perp PautNat2")
    rrange = range(1, 150)
    plt.plot([r for r in rrange], [dicnat[(0, r)][12] for r in rrange], "x-")
    plt.plot([r for r in rrange], [dicnat[(0, r)][11] for r in rrange], "+-")
    #
    (idn,br) = best(dic, cdic, (lambda x, y: y[29]-x[29]), p1)
    plt.subplot(132).set_title("RNN Perp pb {2}, ({0},{1}) block".format(cdic[idn][2], cdic[idn][3], p1))
    rrange = range(1,100)
    plt.plot([r for r in rrange], [dic[(idn, r)][29] for r in rrange], "x-")
    plt.plot([r for r in rrange], [dic[(idn, r)][28] for r in rrange], "+-")
    #
    (idn,br) = best(dic, cdic, (lambda x, y: y[29]-x[29]), p2)
    plt.subplot(133).set_title("RNN Perp pb {2}, ({0},{1}) block".format(cdic[idn][2], cdic[idn][3], p2))
    rrange = range(1,100)
    plt.plot([r for r in rrange], [dic[(idn, r)][29] for r in rrange], "x-")
    plt.plot([r for r in rrange], [dic[(idn, r)][28] for r in rrange], "+-")
    #
    plt.show()


def spice4wave(files):
    labels = ["(300,300)", "(400,400)", "(800,800)"]
    ma = ["+-","x-","<-"]
    rrange = range(1,101)
    dic = dict()
    cdic = dict()
    edic = dict()
    i = 0
    for f in files :
        parse_one(f, i, dic, cdic, edic)
        i+=1
    plt.figure()
    # mpl.rcParams.update({'font.size': 22})
    for k in range(i):
        ranks=[]
        for (id,r) in dic.keys():
            if id == k:
                ranks.append(r)
        ranks = sorted(ranks)
        plt.plot([r for r in ranks if r in rrange], [dic[(k, r)][8] for r in ranks if r in rrange], ma[k])
    plt.legend(labels)
    plt.xlabel("WA rank")
    plt.ylabel("NDCG$_5$")
    plt.show()


def ptitresume(dic, cdic, edic, carac, rk):
    idn = None
    for k in cdic.keys():
        if cdic[k] == carac:
            idn=k
    print("\\begin{figure}")
    print("\\begin{center}")
    print("\\includegraphics[width=\\textwidth]{Plots/}")
    print("\\end{center}")
    print("\\begin{table}")
    print("\\centering")
    print("\\begin{tabular}{|c||c|c||c|c|c|}")
    print("\\hline")
    print("Pb# & Rank & |P,S| & Perplexity Ratio on $S_{Test}$ & Perplexity Ratio on $S_{RNN}$ \\\\ \\hline")
    print("{0} & {1} & ({2},{3}) & {4:7.5f} & {5:7.5f} \\\\ \\hline"
          .format(cdic[idn][0], rk, cdic[idn][1], cdic[idn][2],
                  dic[(idn,rk)][1]/dic[(idn,rk)][2], dic[(idn,rk)][28]/dic[(idn,rk)][29]))
    print("\\end{tabular}")
    print("\\end{table}")
    print("\\end{figure}")


if __name__ == "__main__":
    start_r = int(sys.argv[1])
    end_r = int(sys.argv[2])
    rank_range = range(start_r, end_r+1)
    d1 = dict()
    d2 = dict()
    d3 = dict()
    file_number = 0
    parsed = []
    for fi in sys.argv[3:]:
        name = "f"+str(file_number)
        ret = parse_one(fi, name, d1, d2, d3)
        if ret == 0:
            parsed.append(name)
        file_number += 1
    # plot_overall_perp(d, d2)
    problems = list(set([x[0] for x in d2.values()]))
    #
    mpl.rcParams.update({'font.size': 22})
    #
    #
    # latex_best_rnn(d1, d2, d3, problems)
    #
    # latex_best_test(d1, d2, d3, problems)
    # fabulous_four_hist(d1, d2, problems)
    # for i in [1,2,3,4]:
    #     fabulous_four_hist(d1, d2, problems, opt=i)



    # details3(d1,d2,d3, "../pautnat2/metrics800", 37, 3)
    # details3eps(d1,d2,d3, "../pautnat2/metrics800", 37, 3)
    details3epsKL(d1,d2,d3, "../pautnat2/metrics800", 37, 3)

    print(d1["f282", 6])

    # fabulous_four_hist_WER(d1, d2, problems)
    # spice4wave(sys.argv[3:])

