import math
import sys
import threading
import numpy as np
import keras
# Maison :
import parse3 as parse
import scores
# Sp-learn :
import splearn as sp
import splearn.datasets.base as spparse


# Classes pour la parrallélisation :
class ParaWords(threading.Thread):
    def __init__(self, nalpha, d, name="paraword"):
        threading.Thread.__init__(self)
        self.words = []
        self.d = d
        self.nalpha = nalpha
        self.name = name+str(d)

    def run(self):
        self.words = combinaisons_para(self.nalpha, self.d).tolist()


class ParaCombinaisons(threading.Thread):
    def __init__(self, a, r, p, nalpha):
        threading.Thread.__init__(self)
        self.a = a
        self.r = r
        self.p = p
        self.nalpha = nalpha

    def run(self):
        comb4(self.a, self.r, self.p, self.nalpha)


class ParaBatch(threading.Thread):
    def __init__(self, s, nalpha, pad):
        threading.Thread.__init__(self)
        self.words = s
        self.nalpha = nalpha
        self.pad = pad
        self.batch_words = []
        self.batch_prefixes = []

    def run(self):
        # Il nous faut ajouter le symbole de début (nalpha) au début et le simbole de fin (nalpha+1) à la fin
        # et ajouter 1 a chaque élément pour pouvoir utiliser le zéro comme padding,
        encoded_words = [([self.nalpha + 1] + [1 + elt2 for elt2 in elt] + [self.nalpha + 2]) for elt in self.words]
        batch = []
        nbw = len(encoded_words)
        for i in range(nbw):
            word = encoded_words[i]
            # prefixes :
            batch += [word[:j] for j in range(1, len(word))]
        # padding :
        batch = [parse.pad_0(elt, self.pad) for elt in batch]
        # tuplisation, setisation
        batch = set([tuple(elt) for elt in batch])
        self.batch_prefixes = batch
        self.batch_words = encoded_words


def gen_words(nalpha, lrows, lcols):
    if not type(lrows) is list:
        if not type(lrows) is int:
            raise TypeError()
        else:
            lrows = [i for i in range(lrows+1)]
    if not type(lcols) is list:
        if not type(lcols) is int:
            raise TypeError()
        else:
            lcols = [i for i in range(lcols + 1)]
    lig, col, wl = gen_words_indexes_as_lists_para(nalpha, lrows, lcols)
    return lig, col, wl


def pr(quiet=False, m=""):
    if not quiet:
        print(m)
        sys.stdout.flush


def gen_words_indexes_as_lists_para(nalpha, lrows, lcols, quiet=False):
    lig = []
    col = []
    lig_ths = []
    col_ths = []
    # CREER LES THREADS ET LES LANCER
    pr(quiet, "\tThreads lignes et colonnes...")
    for d in lrows:
        th = ParaWords(nalpha, d, "row")
        lig_ths.append(th)
        th.start()
        # lig += combinaisons(nalpha, d).tolist()
    for d in lcols:
        th = ParaWords(nalpha, d, "lig")
        col_ths.append(th)
        th.start()
        # col += combinaisons(nalpha, d).tolist()
    # ATTENDRE LES THREADS ET RECUPERER LES RESULTATS
    pr(quiet, "\tRécupérations lignes et colonnes...")
    for th in lig_ths:
        th.join()
        lig += th.words
    for th in col_ths:
        th.join()
        col += th.words
    # On trie pour faire comme dans hankel.py, trick pour donner plus d'importance aux mots courts dans la SVD
    pr(quiet, "\tTri lignes et colonnes...")
    col = sorted(col, key=lambda x: (len(x), x))
    lig = sorted(lig, key=lambda x: (len(x), x))
    # ###
    pr(quiet, "\tConstruction des mots...")
    nlig = len(lig)
    ncol = len(col)
    wl = set()
    letters = [[]]+[[i] for i in range(nalpha)]
    for letter in letters:
        for l in range(0, nlig):
            # w.append([])
            for c in range(0, ncol):
                w = lig[l] + letter + col[c]
                # if w not in wl:
                wl.add(tuple(w))
    wl = [list(elt) for elt in wl]
    return lig, col, wl


def combinaisons(nalpha, dim):
    s = math.pow(nalpha, dim)
    a = [[0]*dim]*int(s)
    a = np.array(a)
    p = s
    for i in range(0, dim):
        p /= nalpha
        comb4(a, i, p, nalpha)
    return a


def combinaisons_para(nalpha, dim):
    s = math.pow(nalpha, dim)
    a = [[0]*dim]*int(s)
    a = np.array(a)
    p = s
    ths = []
    for i in range(0, dim):
        p /= nalpha
        th = ParaCombinaisons(a, i, p, nalpha)
        th.start()
        ths.append(th)
        # comb4(a, i, p, nalpha)
    for th in ths:
        th.join()
    return a


# Très légèrement plus rapide que comb3, c'est toujours mieux que rien.
def comb4(a, r, p, nalpha):
    c = 0
    acc = 0
    for i in range(0, len(a)):
        a[i][r] = c
        acc += 1
        if acc == p:
            acc = 0
            c += 1
            if c == nalpha:
                c = 0


def proba_words_para(model, words, nalpha, asdict=True, quiet=False):
    # #### PARAMS : ####
    bsize = 512
    nthreads = 16
    pad = int(model.input.shape[1])  # On déduit de la taille de la couche d'entrée le pad nécéssaire
    # ##################
    if not quiet:
        print("\tProcessing words ...")
        sys.stdout.flush()
    chunkw = int(len(words) / nthreads)+1
    words_chunks = [words[(chunkw*i):(chunkw*(i+1))] for i in range(nthreads)]
    # Lancer les threads
    threads = []
    for i in range(nthreads):
        th = ParaBatch(words_chunks[i], nalpha, pad)
        th.start()
        threads.append(th)
    # Attendre les threads et collecter les résultats
    batch_prefixes = set()
    batch_words = []
    for i in range(nthreads):
        threads[i].join()
        batch_words += threads[i].batch_words
        # batch_prefixes += threads[i].batch_prefixes
        batch_prefixes = batch_prefixes.union(threads[i].batch_prefixes)
    # On restructure tout en numpy
    batch_prefixes = [elt for elt in batch_prefixes]  # set de tuples vers liste de tuples
    batch_prefixes_lists = np.array([list(elt) for elt in batch_prefixes])  # liste de tuples vers liste de listes
    # Prédiction :
    # print(len(batch_prefixes_lists)-len(set([tuple(elt) for elt in batch_prefixes])))
    wpreds = model.predict(batch_prefixes_lists, bsize, verbose=(0 if quiet else 1))
    prefixes_dict = dict()
    for i in range(len(batch_prefixes_lists)):
        key = batch_prefixes_lists[i]
        #  Decodage :
        key = tuple([elt-1 for elt in key if elt > 0][1:])
        prefixes_dict[key] = wpreds[i]
    # Calcul de la probabilité des mots :
    preds = np.empty(len(words))
    if not quiet:
        print("\tCalculating fullwords probas...")
        sys.stdout.flush()
    for i in range(len(words)):
        word = tuple([x for x in words[i]])+(nalpha+1,)
        acc = 1.0
        for k in range(len(word)):
            try:
                pref = word[:k][-pad:]
                proba = prefixes_dict[pref][word[k]]
                acc *= proba
            except :
                print("gabuzomeu", pref)
        preds[i] = acc
        # if not quiet:
        #     print("\r\tCalculating fullwords probas : {0} / {1}".format(i+1, len(batch_words)), end="")
    # if not quiet:
    #     print("\r\tCalculating fullwords probas OK                         ")
    #     sys.stdout.flush()
    if asdict:  # On retourne un dictionnaire
        probas = dict()
        for i in range(len(words)):
            probas[tuple(words[i])] = preds[i]
        return probas
    else:  # On retourne une liste
        return preds


def proba_words(model, x_words, nalpha, asdict=True, quiet=False):
    bsize = 512
    pad = int(model.input.shape[1])  # On déduit de la taille de la couche d'entrée le pad nécéssaire
    preds = np.empty(len(x_words))
    # Il nous faut ajouter le symbole de début (nalpha) au début et le simbole de fin (nalpha+1) à la fin
    # et ajouter 1 a chaque élément pour pouvoir utiliser le zéro comme padding,
    if not quiet:
        print("\tEncoding words...", end="")
    batch_words = [([nalpha+1]+[1+elt2 for elt2 in elt]+[nalpha+2])for elt in x_words]
    if not quiet:
        print("\r\tEncoding OK                            ",
              "\n\tPreparing batch :", end="")
    nbw = len(x_words)
    batch = []
    for i in range(nbw):
        word = batch_words[i]
        # prefixes :
        batch += [word[:j] for j in range(1, len(word))]
        if not quiet:
            print("\r\tPreparing batch : {0} / {1}".format(i+1, nbw), end="")
    if not quiet:
        print("\r\tBatch OK                                 ",
              "\n\tPadding batch...", end="")
    # padding :
    batch = [parse.pad_0(elt, pad) for elt in batch]
    if not quiet:
        print("\r\tPadding OK                           ",
              "\n\tPredicting batch ({0} elts)...".format(len(batch)))
    # On restructure tout en numpy
    batch = np.array(batch)
    # Prédiction :
    wpreds = model.predict(batch, bsize, verbose=(0 if quiet else 1))
    if not quiet:
        print("\tPredicting OK\n\tCalculating fullwords probas:", end="")
    offset = 0
    for i in range(nbw):
        word = batch_words[i]
        acc = 1.0
        for k in range(len(word)-1):
            acc *= wpreds[offset][word[k+1]-1]
            offset += 1
        preds[i] = acc
        if not quiet:
            print("\r\tCalculating fullwords probas : {0} / {1}".format(i+1, nbw), end="")
    if not quiet:
        print("\r\tCalculating fullwords probas OK                         ")
    if asdict:  # On retourne un dictionnaire
        probas = dict()
        for i in range(len(x_words)):
            probas[tuple(x_words[i])] = preds[i]
        return probas
    else:  # On retourne une liste
        return preds


def hankels(ligs, cols, probas, nalpha):
    nligs = len(ligs)
    ncols = len(cols)
    lhankels = [np.zeros((nligs, ncols)) for _ in range(nalpha+1)]
    # EPSILON :
    for l in range(nligs):
        for c in range(ncols):
            lhankels[0][l][c] = probas[tuple(ligs[l]+cols[c])]
    # LETTER MATRICES :
    letters = [i for i in range(nalpha)]
    for letter in letters:
        for l in range(nligs):
            for c in range(ncols):
                lhankels[letter][l][c] = probas[tuple(ligs[l] + [letter] + cols[c])]
    return lhankels


def custom_fit(rank, lrows, lcols, modelfile, perplexity=False, train_file="", target_file=""):
    model = keras.models.load_model(modelfile)
    nalpha = int(model.output.shape[1])-2
    # nalpha = 4
    # train_file = "/home/nono/stage2018/rnntospectral/data/pautomac/4.pautomac.test"
    # target_file = "/home/nono/stage2018/rnntospectral/data/pautomac/4.pautomac_solution.txt"
    ###
    # Params :
    quiet = False
    partial = False
    # Préparations :
    if not quiet:
        print("Construction of set of words...")
        sys.stdout.flush()
    ligs, cols, lw = gen_words(nalpha, lrows, lcols)
    if not quiet:
        print("Prediction of probabilities of words...")
        sys.stdout.flush()
    probas = proba_words_para(model, lw, nalpha)
    if not quiet:
        print("Building of hankel matrices...")
        sys.stdout.flush()
    lhankels = hankels(ligs, cols, probas, nalpha)
    spectral_estimator = sp.Spectral(rank=rank, lrows=lrows, lcolumns=lcols,
                                     version='classic', partial=partial, sparse=False,
                                     smooth_method='none', mode_quiet=quiet)
    # Les doigts dans la prise !
    if not quiet:
        print("Custom fit ...")
        sys.stdout.flush()
    spectral_estimator._hankel = sp.Hankel(sample_instance=None, lrows=lrows, lcolumns=lcols,
                                           version='classic', partial=partial, sparse=False,
                                           mode_quiet=quiet, lhankel=lhankels)
    spectral_estimator._automaton = spectral_estimator._hankel.to_automaton(rank, quiet)
    # OK on a du a peu près rattraper l'état après fit.
    if not quiet:
        print("... Done !")
        sys.stdout.flush()
    # Perplexity :
    if perplexity:
        print("Perplexity :")
        epsilon = 0.0001

        x_test = parse.parse_fullwords(train_file)
        x_test_sp = spparse.load_data_sample(train_file)
        y_test = parse.parse_pautomac_results(target_file)

        perp_proba_rnn = fix_probas(proba_words_para(model, x_test, nalpha, asdict=False, quiet=False), f=epsilon)
        perp_proba_spec = fix_probas(spectral_estimator.predict(x_test_sp.data), f=epsilon)
        print(countlen(x_test, lrows+lcols+1))
        test_perp = scores.pautomac_perplexity(y_test, y_test)
        rnn_perp = scores.pautomac_perplexity(y_test, perp_proba_rnn)
        extract_perp = scores.pautomac_perplexity(y_test, perp_proba_spec)

        test_rnn_kl = scores.kullback_leibler(y_test, perp_proba_rnn)
        rnn_extr_kl = scores.kullback_leibler(perp_proba_rnn, perp_proba_spec)
        test_extr_kl = scores.kullback_leibler(y_test, perp_proba_spec)

        print("\tTest :\t{0}\n\tRNN :\t{1}\n\tExtr :\t{2}"
              .format(test_perp, rnn_perp, extract_perp))
        print("KL Divergence :")
        print("\tTest-RNN :\t{0}\n\tRNN-Extr :\t{1}\n\tTest-Extr :\t{2}"
              .format(test_rnn_kl, rnn_extr_kl, test_extr_kl))
    #
    return spectral_estimator


def fix_probas(seq, p=0.0, f=0.0001, quiet=False):
    z = 0
    n = 0
    for i in range(len(seq)):
        if seq[i] < p:
            seq[i] = f
            n += 1
        elif seq[i] == p:
            seq[i] = f
            z += 1
    if not quiet:
        print("(Epsilon value used {0} / {1} times ({2} neg and {3} zeros))".format(n+z, len(seq), n, z))
    return seq


def countlen(seq, le):
    k = 0
    for i in range(len(seq)):
        if len(seq[i]) > le:
            k += 1
    return k


if __name__ == "__main__":
    if len(sys.argv) < 5 or len(sys.argv) > 7:
        print("Usage :: {0} modelfile rank lrows lcols [testfile testtargetsfile]".format(sys.argv[0]))
        sys.exit(-666)
    # XXXXXX :
    context = (sys.argv[1]+"!"+sys.argv[2]+"!"+sys.argv[3]+"!"+sys.argv[4]).replace(" ", "_").replace("/", "+")
    print("Context :", context)
    arg_model = sys.argv[1]
    arg_rank = int(sys.argv[2])
    try:
        arg_lrows = int(sys.argv[3])
    except ValueError:
        arg_lrows = [int(x) for x in sys.argv[3].split()]
    try:
        arg_lcols = int(sys.argv[4])
    except ValueError:
        arg_lcols = [int(x) for x in sys.argv[4].split()]
    if len(sys.argv) >= 7:
        arg_testfile = sys.argv[5]
        arg_testtargetsfile = sys.argv[6]
        arg_perp = True
    else:
        arg_testfile = ""
        arg_testtargetsfile = ""
        arg_perp = False

    est = custom_fit(arg_rank, arg_lrows, arg_lcols, arg_model, arg_perp, arg_testfile, arg_testtargetsfile)
    sp.Automaton.write(est.automaton, filename=("aut-"+context))


# VIELLES CHOSES ::


# Listes des mots, sans parrallélisation
# def words_indexes_as_lists(nalpha, lrows, lcols):
#     lig = []
#     col = []
#     for d in lrows:
#         lig += combinaisons(nalpha, d).tolist()
#     for d in lcols:
#         col += combinaisons(nalpha, d).tolist()
#     # On trie pour faire comme dans hankel.py mais je comprends pas trop pourquoi
#     col = sorted(col, key=lambda x: (len(x), x))
#     lig = sorted(lig, key=lambda x: (len(x), x))
#     # ###
#     nlig = len(lig)
#     ncol = len(col)
#     wl = []
#     letters = [[]]+[[i] for i in range(nalpha)]
#     for letter in letters:
#         for l in range(0, nlig):
#             # w.append([])
#             for c in range(0, ncol):
#                 w = lig[l] + letter + col[c]
#                 if w not in wl:
#                     wl.append(w)
#     return lig, col, wl


# def comb3(a, r, p, nalpha):
#     for i in range(0, len(a)):
#         a[i][r] = (i // p) % nalpha

    # # Calcul de la probabilité des mots :
    # preds = np.empty(len(words))
    # offset = 0
    # if not quiet:
    #     print("\tCalculating fullwords probas OK")
    #     sys.stdout.flush()
    # for i in range(len(words)):
    #     word = batch_words[i]
    #     acc = 1.0
    #     for k in range(len(word)-1):
    #         acc *= wpreds[offset][word[k+1]-1]
    #         offset += 1
    #     preds[i] = acc
    #     # if not quiet:
    #     #     print("\r\tCalculating fullwords probas : {0} / {1}".format(i+1, len(batch_words)), end="")
    # # if not quiet:
    # #     print("\r\tCalculating fullwords probas OK                         ")
    # #     sys.stdout.flush()

# def proba_words_para2(model, x_words, nalpha, asdict=True, quiet=False):
#     pad = int(model.input.shape[1])  # On déduit de la taille de la couche d'entrée le pad nécéssaire
#     preds = np.empty(len(x_words))
#     # Il nous faut ajouter le symbole de début (nalpha) au début et le simbole de fin (nalpha+1) à la fin
#     # et ajouter 1 a chaque élément pour pouvoir utiliser le zéro comme padding,
#     batch_words = [([nalpha+1]+[1+elt2 for elt2 in elt]+[nalpha+2])for elt in x_words]
#     nbw = len(x_words)
#     for i in range(nbw):
#         word = batch_words[i]
#         # prefixes :
#         batch = [word[:j] for j in range(1, len(word))]
#         # padding :
#         batch = [parse.pad_0(elt, pad) for elt in batch]
#         # On restructure tout en numpy
#         batch = np.array(batch)
#         # Prédiction :
#         wpreds = model.predict(batch, len(batch))
#         acc = 1.0
#         for k in range(len(word)-1):
#             acc *= wpreds[k][word[k+1]-1]
#         preds[i] = acc
#         if not quiet:
#             print("\r{0} / {1}".format(i, nbw), end="")
#     if not quiet:
#         print("")
#     if asdict:  # On retourne un dictionnaire
#         probas = dict()
#         for i in range(len(x_words)):
#             probas[tuple(x_words[i])] = preds[i]
#         return probas
#     else:  # On retourne une liste
#         return preds
