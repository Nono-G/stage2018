import sys
import numpy as np
import keras
# Maison :
import parse3 as parse
import scores
from spextractor_common import pr, ParaWords, ParaBatch, ParaRowsCols, parts_of_list
# Sp-learn :
import splearn as sp
import splearn.datasets.base as spparse


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


def gen_words_indexes_as_lists_para(nalpha, lrows, lcols, quiet=False):
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
    # On trie pour faire comme dans hankel.py, trick pour donner plus d'importance aux mots courts dans la SVD
    pr(quiet, "\tTri lignes et colonnes...")
    col = sorted(col, key=lambda x: (len(x), x))
    lig = sorted(lig, key=lambda x: (len(x), x))
    # ###
    pr(quiet, "\tConstruction des mots...")
    wl = set()
    letters = [[]]+[[i] for i in range(nalpha)]
    thrs = []
    for letter in letters:
        th = ParaWords(lig, col, letter, 4)
        thrs.append(th)
        th.start()
    for th in thrs:
        th.join()
        wl = wl.union(th.words)
    wl = [list(elt) for elt in wl]
    return lig, col, wl


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
        batch_words += threads[i].prefixes
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
        key = tuple([elt-1 for elt in key if elt > 0])
        if key[0] == nalpha:
            key = key[1:]
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
            proba = prefixes_dict[pref][word[k]]
            acc *= proba
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
                lhankels[letter+1][l][c] = probas[tuple(ligs[l] + [letter] + cols[c])]
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
    # noinspection PyProtectedMember
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
