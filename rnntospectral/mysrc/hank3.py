import math
import sys
import threading
import numpy as np
import keras
# Maison :
import parse3 as parse
import scores
from spextractor_common import ParaRowsCols, Hush, proba_words_para, ParaWords, fix_probas, pr, SpexHush
# Sp-learn :
import splearn as sp
import splearn.datasets.base as spparse


class SpexRandDrop(SpexHush):
    def __init__(self, modelfile, lrows, lcols, pref_drop, suff_drop, perp_train="", perp_targ="", context=""):
        SpexHush.__init__(self, modelfile, lrows, lcols, perp_train, perp_targ, context)
        self.pref_drop = pref_drop
        self.suff_drop = suff_drop

    def gen_words(self):
        pass


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
