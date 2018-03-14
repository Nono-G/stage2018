import math
import sys
import threading
import numpy as np
import keras
# Maison :
import parse3 as parse
import scores as scores
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
    def __init__(self, prefixes, suffixes, letter, subthreads=1, print_id=-1):
        threading.Thread.__init__(self)
        self.prefixes = prefixes
        self.suffixes = suffixes
        self.letter = letter
        self.words = set()
        self.subthreads = subthreads
        self.quiet = False
        self.id = print_id

    def run(self):
        ws = set()
        # Auto-parallelization :
        if self.subthreads > 1:
            pr(self.quiet, "\t\tThread letter {0} begin".format(self.letter))
            thrs = []
            prefparts = parts_of_list(self.prefixes, self.subthreads)
            for i in range(self.subthreads):
                th = ParaWords(prefparts[i], self.suffixes, self.letter, 1, i)
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
                    w = tuple(p + self.letter + s)
                    ws.add(w)
            pr(self.quiet, "\t\t\tThread letter {0}, part {1} end".format(self.letter, self.id))
        # Anyway :
        self.words = ws


class BatchGenerator(keras.utils.Sequence):
    def __init__(self, prefs, suffs, letter, nalpha, pad):
        self.prefs = prefs
        self.suffs = suffs
        self.letter = letter
        self.nalpha = nalpha
        self.pad = pad
        self.len = len(prefs)*len(suffs)
        self.sidefx = self.len*[-1]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        p, s = self._indexdecode(item)
        w = [self.nalpha + 1] + [elt + 1 for elt in (self.prefs[p] + self.letter + self.suffs[s])] + [self.nalpha + 2]
        self.sidefx[item] = (len(w))
        w = parse.pad_0(w, (len(w) + self.pad - 1))
        batch = []
        for i in range(len(w) - self.pad):
            batch.append(w[i:i + self.pad])
        return np.array(batch)

    def _indexcode(self, p, s):
        return len(self.suffs)*p + s

    def _indexdecode(self, i):
        return i // len(self.suffs), i % len(self.suffs)


# Utilitaires :
def pr(quiet=False, m=""):
    if not quiet:
        print(m)
        sys.stdout.flush()


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


def parts_of_list(li, nparts):
    part_size = int(len(li) / nparts)+1
    parts = [li[(part_size*i):(part_size*(i+1))] for i in range(nparts)]
    return parts


def countlen(seq, le):
    k = 0
    for i in range(len(seq)):
        if len(seq[i]) > le:
            k += 1
    return k


# Trucs faits pour
def enumerate_words(nalpha, lrows, lcols):
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
    lig, col = _enumerate_words_indexes_as_lists_para(nalpha, lrows, lcols)
    return lig, col


def _enumerate_words_indexes_as_lists_para(nalpha, lrows, lcols, quiet=False):
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
    # On trie pour faire comme dans hankel.py, trick pour donner plus d'importance aux mots courts dans la SVD
    pr(quiet, "\tTri lignes et colonnes...")
    col = sorted(col, key=lambda x: (len(x), x))
    lig = sorted(lig, key=lambda x: (len(x), x))
    return lig, col


def concatenations(nalpha, pref, suff, quiet=False):
    pr(quiet, "\tConstruction des mots...")
    wl = set()
    letters = [[]]+[[i] for i in range(nalpha)]
    thrs = []
    for letter in letters:
        th = ParaWords(pref, suff, letter, 4)
        thrs.append(th)
        th.start()
    for th in thrs:
        th.join()
        wl = wl.union(th.words)
    wl = [list(elt) for elt in wl]
    return wl


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


def generator_batches(pref, suff, letter, nalpha, pad, sidefx, batch_vol=10):
    i = 0
    current_v = 0
    batch = []
    for p in pref:
        for s in suff:
            w = [nalpha+1]+[elt+1 for elt in (p+letter+s)]+[nalpha+2]
            sidefx.append(len(w))
            w = parse.pad_0(w, (len(w)+pad-1))
            i += 1
            for i in range(len(w)-pad):
                batch.append(w[i:i+pad])
            current_v += 1
            if current_v == batch_vol :
                current_v = 0
                ret = np.array(batch)
                batch = []
                yield ret
    yield np.array(batch)


def generator_words(pref, suff, letter):
    for p in pref:
        for s in suff:
            yield p+letter+s


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


def hankels(model, prefs, suffs):
    nalpha = int(model.output.shape[1])-2
    pad = int(model.input.shape[1])
    letters = [[]]+[[i] for i in range(nalpha)]
    lhankels = []
    for l in letters:
        batch_vol = 50
        steps = int(len(prefs)*len(suffs)/batch_vol)+1
        sidefx = []
        gb = generator_batches(prefs, suffs, l, nalpha, pad, sidefx, batch_vol)
        probas_prefs = model.predict_generator(gb, max_queue_size=20, steps=steps, verbose=1)
        # gb = WordGenerator(prefs, suffs, l, nalpha, pad)
        # probas_prefs = model.predict_generator(gb, verbose=1, workers=1, use_multiprocessing=True)
        # sidefx = gb.sidefx
        probas_words = []
        gw = generator_words(prefs, suffs, l)
        offset = 0
        for s in sidefx:
            acc = 1.0
            w = gw.__next__()+[nalpha+1]
            for i in range(s-1):
                acc *= probas_prefs[offset][w[i]]
                offset += 1
            probas_words.append(acc)
        hankel = np.array(probas_words).reshape((len(prefs), len(suffs)))
        lhankels.append(hankel)
    return lhankels


def hankels_para(model, prefs, suffs):
    class ParaHankel(threading.Thread):
        def __init__(self, model_arg, prefs_arg, suffs_arg, letter_arg):
            threading.Thread.__init__(self)
            # self.model = keras.models.load_model(modelfile_arg)
            # self.model._make_predict_function()
            self.model = model_arg
            self.nalpha = int(self.model.output.shape[1])-2
            self.pad = int(self.model.input.shape[1])
            self.prefs = prefs_arg
            self.suffs = suffs_arg
            self.letter = letter_arg
            self.hankel = None

        def run(self):
            sidefx = []
            gb = generator_batches(self.prefs, self.suffs, self.letter, self.nalpha, self.pad, sidefx)
            probas_prefs = self.model.predict_generator(gb, steps=(len(self.prefs)*len(self.suffs)), verbose=1)
            # gb = WordGenerator(self.prefs, self.suffs, self.letter, self.nalpha, self.pad)
            # probas_prefs = model.predict_generator(gb, verbose=1, workers=1, use_multiprocessing=True)
            # sidefx = gb.sidefx
            probas_words = []
            gw = generator_words(prefs, suffs, l)
            offset = 0
            for s in sidefx:
                acc = 1.0
                w = gw.__next__() + [self.nalpha + 1]
                for i in range(s - 1):
                    acc *= probas_prefs[offset][w[i]]
                    offset += 1
                probas_words.append(acc)
            self.hankel = np.array(probas_words).reshape((len(prefs), len(suffs)))

    nalpha = int(model.output.shape[1]) - 2
    letters = [[]]+[[i] for i in range(nalpha)]
    lhankels = []
    thrs = []
    for l in letters:
        # th = ParaHankel(modelfile, prefs, suffs, l)
        th = ParaHankel(model, prefs, suffs, l)
        thrs.append(th)
        th.start()
    for th in thrs:
        th.join()
        lhankels.append(th.hankel)
    return lhankels


def custom_fit(rank, lrows, lcols, modelfile, perplexity=False, train_file="", target_file=""):
    # Params :
    model = keras.models.load_model(modelfile)
    nalpha = int(model.output.shape[1])-2
    quiet = False
    # Préparations :
    pr(quiet, "Enumeration of prefixes and suffixes...")
    ligs, cols = enumerate_words(nalpha, lrows, lcols)
    pr(quiet, "Building of hankel matrices...")
    lhankels = hankels(model, ligs, cols)
    # lhankels = hankels_para(model, ligs, cols)
    spectral_estimator = sp.Spectral(rank=rank, lrows=lrows, lcolumns=lcols,
                                     version='classic', partial=True, sparse=False,
                                     smooth_method='none', mode_quiet=quiet)
    # Les doigts dans la prise !
    pr(quiet, "Custom fit ...")
    spectral_estimator._hankel = sp.Hankel(sample_instance=None, lrows=lrows, lcolumns=lcols,
                                           version='classic', partial=True, sparse=False,
                                           mode_quiet=quiet, lhankel=lhankels)
    # noinspection PyProtectedMember
    spectral_estimator._automaton = spectral_estimator._hankel.to_automaton(rank, quiet)
    # OK on a du a peu près rattraper l'état après fit.
    pr(quiet, "... Done !")
    # Perplexity :
    if perplexity:
        print("Perplexity :")
        epsilon = 0.0001

        x_test = parse.parse_fullwords(train_file)
        x_test_sp = spparse.load_data_sample(train_file)
        y_test = parse.parse_pautomac_results(target_file)

        perp_proba_rnn = fix_probas(proba_words_para(model, x_test, nalpha, asdict=False, quiet=False), f=epsilon)
        perp_proba_spec = fix_probas(spectral_estimator.predict(x_test_sp.data), f=epsilon)
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


# Main :
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
