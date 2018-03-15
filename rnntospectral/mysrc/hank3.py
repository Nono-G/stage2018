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
class ParaRowsCols(threading.Thread):
    def __init__(self, nalpha, d,):
        threading.Thread.__init__(self)
        self.words = []
        self.d = d
        self.nalpha = nalpha

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


class ParaWords(threading.Thread):
    def __init__(self, prefixes, suffixes, letter, hush, subthreads=1, print_id=-1, quiet=True):
        threading.Thread.__init__(self)
        self.prefixes = prefixes
        self.suffixes = suffixes
        self.letter = letter
        self.words = set()
        self.subthreads = subthreads
        self.quiet = quiet
        self.id = print_id
        self.h = hush

    def run(self):
        ws = set()
        # Auto-parallelization :
        if self.subthreads > 1:
            pr(self.quiet, "\t\tThread letter {0} begin".format(self.letter))
            thrs = []
            prefparts = parts_of_list(self.prefixes, self.subthreads)
            for i in range(self.subthreads):
                th = ParaWords(prefparts[i], self.suffixes, self.letter, self.h, 1, i)
                thrs.append(th)
                th.start()
            for th in thrs:
                th.join()
                ws = ws.union(th.words)
            pr(self.quiet, "\t\tThread letter {0} end".format(self.letter))
        # Default and bottom case : no parallelization
        else:
            pr(self.quiet, "\t\t\tThread letter {0}, part {1} begin".format(self.letter, self.id))
            for p in self.prefixes:
                for s in self.suffixes:
                    w = self.h.encode(p + self.letter + s)
                    ws.add(w)
            pr(self.quiet, "\t\t\tThread letter {0}, part {1} end".format(self.letter, self.id))
        # Anyway :
        self.words = ws


class Hush:
    def __init__(self, maxlen, base):
        self.maxlen = maxlen
        self.base = base
        self.nl = []
        self.pows = []
        self.mise_en_place()
        self.maxval = self.encode([base - 1] * maxlen)

    def mise_en_place(self):
        self.pows = [1] * (self.maxlen+1)
        for i in range(1, self.maxlen+1):
            self.pows[i] = self.pows[i-1] * self.base
        #
        self.nl = [1] * (self.maxlen+1)
        for i in range(1, self.maxlen+1):
            self.nl[i] = self.nl[i-1] + (self.pows[i])
        #

    def encode(self, w):
        if len(w) > self.maxlen:
            raise ValueError
        if len(w) == 0:
            return 0
        else:
            x = self.nl[len(w)-1]
            for i in range(len(w)):
                x += w[i]*self.pows[len(w)-i-1]
            return x

    def decode(self, n):
        if n > self.maxval:
            raise ValueError
        le = 0
        while self.nl[le] <= n:
            le += 1
        x = [0]*le
        reste = n - self.nl[le-1]
        for i in range(le):
            x[le-i-1] = reste % self.base
            reste //= self.base
        return x


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
    hush = Hush(1+max(lrows)+max(lcols), nalpha)
    lig, col, wl = gen_words_indexes_as_lists_para(nalpha, lrows, lcols, hush)
    return hush, lig, col, wl


def pr(quiet=False, m=""):
    if not quiet:
        print(m)
        sys.stdout.flush()


def gen_words_indexes_as_lists_para(nalpha, lrows, lcols, hush, quiet=False):
    lig = []
    col = []
    thrs = []
    # CREER LES THREADS ET LES LANCER
    dims = [x for x in (set(lrows).union(set(lcols)))]
    dimdict = {}
    pr(quiet, "\tThreads lignes et colonnes...")
    for d in dims:
        th = ParaRowsCols(nalpha, d)
        thrs.append(th)
        th.start()
    # ATTENDRE LES THREADS ET RECUPERER LES RESULTATS
    pr(quiet, "\tRecuperations lignes et colonnes...")
    for th in thrs:
        th.join()
        dimdict[th.d] = th.words
    # FORMER lig et col:
    for d in lrows:
        lig += dimdict[d]
    for d in lcols:
        col += dimdict[d]
    # DEL:
    del thrs
    del dims
    del dimdict
    pr(quiet, "\tTri lignes et colonnes...")
    # Y'a pas la place pour tout le monde :
    # coeff = hush.nl[3] / len(col)
    coeff = 1.0
    shufflek = np.random.choice(len(col), int(len(col)*coeff), replace=False)
    col = [col[i] for i in shufflek]
    # coeff = hush.nl[3] / len(lig)
    shufflek = np.random.choice(len(lig), int(len(lig)*coeff), replace=False)
    lig = [lig[i] for i in shufflek]
    # On trie pour faire comme dans hankel.py, trick pour donner plus d'importance aux mots courts dans la SVD
    col = sorted(col, key=lambda x: (len(x), x))
    lig = sorted(lig, key=lambda x: (len(x), x))
    # ###
    pr(quiet, "\tConstruction des mots...")
    encoded_words_set = set()
    letters = [[]]+[[i] for i in range(nalpha)]
    thrs = []
    for letter in letters:
        th = ParaWords(lig, col, letter, hush, 2, quiet=quiet)
        thrs.append(th)
        th.start()
    for th in thrs:
        th.join()
        encoded_words_set = encoded_words_set.union(th.words)
    return lig, col, list(encoded_words_set)


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


def parts_of_list(li, nparts):
    part_size = int(len(li) / nparts)+1
    parts = [li[(part_size*i):(part_size*(i+1))] for i in range(nparts)]
    return parts


def proba_words_para(model, words, nalpha, asdict=True, quiet=False):
    # #### PARAMS : ####
    bsize = 512
    nthreads = 16
    pad = int(model.input.shape[1])  # On déduit de la taille de la couche d'entrée le pad nécéssaire
    # ##################
    if not quiet:
        print("\tProcessing words ...")
        sys.stdout.flush()
    words_chunks = parts_of_list(words, nthreads)
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
            pref = word[:k][-pad:]
            try:
                proba = prefixes_dict[pref][word[k]]
                acc *= proba
            except KeyError:
                print("gabuzomeu !", pref)
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


def gen_batch(word_list, nalpha, pad, h, batch_vol=1):
    current_v = 0
    batch = []
    for wcode in word_list:
        w = h.decode(wcode)
        w = [nalpha+1]+[elt+1 for elt in w]+[nalpha+2]
        w = parse.pad_0(w, (len(w)+pad-1))
        for i in range(len(w) - pad):
            batch.append(w[i:i + pad])
        current_v += 1
        if current_v == batch_vol:
            current_v = 0
            ret = np.array(batch)
            batch = []
            yield ret
    yield np.array(batch)


def gen_batch_simple(word_list, nalpha, pad, h, batch_vol=1):
    current_v = 0
    batch = []
    for wcode in word_list:
        w = h.decode(wcode)
        w = [nalpha + 1] + [elt + 1 for elt in w]
        w = parse.pad_0(w, pad)
        batch.append(w)
        current_v += 1
        if current_v == batch_vol:
            current_v = 0
            ret = np.array(batch)
            batch = []
            yield ret
    yield np.array(batch)


def proba_words_g(model, x_words, nalpha, hush, asdict=True, quiet=False):
    pad = int(model.input.shape[1])  # On déduit de la taille de la couche d'entrée le pad nécéssaire
    suffs_batch = set()
    pr(quiet, "\tMise en batch des prefixes...")
    for wcode in x_words:
        w = hush.decode(wcode)
        for i in range(len(w)+1):
            suffs_batch.add(hush.encode(w[:i]))
    suffs_batch = list(suffs_batch)
    # ################
    pr(quiet, "\tUtilisation du RNN...")
    batch_vol = 512
    steps = math.ceil(len(suffs_batch)/batch_vol)
    g = gen_batch_simple(suffs_batch, nalpha, pad, hush, batch_vol)
    suffs_preds = model.predict_generator(g, steps, verbose=(0 if quiet else 1))
    suffs_dict = {}
    for k in range(len(suffs_batch)):
        suffs_dict[suffs_batch[k]] = suffs_preds[k]
    del suffs_preds
    del suffs_batch
    pr(quiet, "\tCalcul de la probabilite des mots entiers...")
    preds = np.empty(len(x_words))
    for i in range(len(x_words)):
        word = hush.decode(x_words[i])+[nalpha+1]
        acc = 1.0
        for k in range(len(word)):
            acc *= suffs_dict[hush.encode(word[:k])][word[k]]
        preds[i] = acc
    del suffs_dict
    if asdict:  # On retourne un dictionnaire
        probas = dict()
        for i in range(len(x_words)):
            probas[x_words[i]] = preds[i]
        return probas
    else:  # On retourne une liste
        return preds


def hankels(ligs, cols, probas, nalpha, hush):
    lhankels = [np.zeros((len(ligs), len(cols))) for _ in range(nalpha+1)]
    # EPSILON AND LETTER MATRICES :
    letters = [[]] + [[i] for i in range(nalpha)]
    for letter in range(len(letters)):
        for l in range(len(ligs)):
            for c in range(len(cols)):
                # lhankels[letter][l][c] = probas[tuple(ligs[l] + [letter] + cols[c])]
                lhankels[letter][l][c] = probas[hush.encode(ligs[l] + letters[letter] + cols[c])]
    return lhankels


def custom_fit(ranks, lrows, lcols, modelfile, perplexity=False, train_file="", target_file="", context=""):
    model = keras.models.load_model(modelfile)
    nalpha = int(model.output.shape[1])-2
    ###
    # Params :
    quiet = False
    epsilon = 0.0001
    # Préparations :
    pr(quiet, "Construction of set of words...")
    hush, ligs, cols, lw = gen_words(nalpha, lrows, lcols)
    pr(quiet, "Prediction of probabilities of words...")
    probas = proba_words_g(model, lw, nalpha, hush)
    pr(quiet, "Building of hankel matrices...")
    lhankels = hankels(ligs, cols, probas, nalpha, hush)
    if perplexity:
        pr(quiet, "Perplexity preparations...")
        x_test = parse.parse_fullwords(train_file)
        x_test_sp = spparse.load_data_sample(train_file)
        y_test = parse.parse_pautomac_results(target_file)

        perp_proba_rnn = fix_probas(proba_words_para(model, x_test, nalpha, asdict=False, quiet=False), f=epsilon)
        test_perp = scores.pautomac_perplexity(y_test, y_test)
        rnn_perp = scores.pautomac_perplexity(y_test, perp_proba_rnn)
        test_rnn_kl = scores.kullback_leibler(y_test, perp_proba_rnn)
    else:
        x_test_sp = None
        y_test = None
        rnn_perp = None
        perp_proba_rnn = None
        test_perp = None
        test_rnn_kl = None
    spectral_estimator = None
    for rank in ranks:
        if len(ranks) > 1:
            pr(quiet, "Rank {0} among {1} :".format(rank, ranks))
        # noinspection PyTypeChecker
        spectral_estimator = sp.Spectral(rank=rank, lrows=ligs, lcolumns=cols,
                                         version='classic', partial=True, sparse=False,
                                         smooth_method='none', mode_quiet=quiet)
        # Les doigts dans la prise !
        pr(quiet, "Custom fit ...")
        spectral_estimator._hankel = sp.Hankel(sample_instance=None, lrows=ligs, lcolumns=cols,
                                               version='classic', partial=True, sparse=False,
                                               mode_quiet=quiet, lhankel=lhankels)
        # noinspection PyProtectedMember
        spectral_estimator._automaton = spectral_estimator._hankel.to_automaton(rank, quiet)
        # OK on a du a peu près rattraper l'état après fit.
        pr(quiet, "... Done !")
        # Perplexity :
        sp.Automaton.write(spectral_estimator.automaton, filename=("aut-{0}-r-{1}".format(context, rank)))
        if perplexity:
            print("Perplexity :")
            perp_proba_spec = fix_probas(spectral_estimator.predict(x_test_sp.data), f=epsilon)
            extract_perp = scores.pautomac_perplexity(y_test, perp_proba_spec)
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
        print("Usage :: {0} modelfile ranks lrows lcols [testfile testtargetsfile]".format(sys.argv[0]))
        sys.exit(-666)
    # XXXXXX :
    context_a = ((sys.argv[1] + "!" + sys.argv[2] + "!" + sys.argv[3] + "!" + sys.argv[4])
                 .replace(" ", "_")
                 .replace("/", "+"))
    print("Context :", context_a)
    arg_model = sys.argv[1]
    arg_ranks = [int(e) for e in sys.argv[2].split(sep="_")]
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

    est = custom_fit(arg_ranks, arg_lrows, arg_lcols,
                     arg_model, arg_perp, arg_testfile,
                     arg_testtargetsfile, context_a)
    # sp.Automaton.write(est.automaton, filename=("aut-"+context))
