import math
import sys
import numpy as np
# Maison :
from spextractor_common import ParaRowsCols, pr, SpexHush


class SpexExpress(SpexHush):
    """
    Spex "Express" : lrows=lcols = un entier, et tout est "plein" dans l'ensemble des mots
    """

    def __init__(self, modelfile, lrows_lcols, perp_train="", perp_targ="", context=""):
        SpexHush.__init__(self, modelfile, lrows_lcols, lrows_lcols, perp_train, perp_targ, context)

    def gen_words(self):
        lig_col = []
        thrs = []
        # CREER LES THREADS ET LES LANCER
        pr(self.quiet, "\tThreads lignes et colonnes...")
        for d in range(self.lrows + 1):
            th = ParaRowsCols(self.nalpha, d)
            thrs.append(th)
            th.start()
        # ATTENDRE LES THREADS ET RECUPERER LES RESULTATS
        pr(self.quiet, "\tRecuperations lignes et colonnes...")
        for th in thrs:
            th.join()
            lig_col += th.words
        # DEL:
        del thrs
        # On trie pour faire comme dans hankel.py, trick pour donner plus d'importance aux mots courts dans la SVD
        pr(self.quiet, "\tTri lignes et colonnes...")
        lig_col = sorted(lig_col, key=lambda x: (len(x), x))
        # ###
        pr(self.quiet, "\tConstruction des mots...")
        return lig_col, lig_col, range(self.hush.nl[2 * self.lrows + 1])

    def proba_words_special(self, words, asdict=True):
        suffs_batch = words  # Seulement car lrows et lcols sont "pleins"
        # ################
        pr(self.quiet, "\tUtilisation du RNN...")
        batch_vol = 2048
        steps = math.ceil(len(suffs_batch) / batch_vol)
        g = self.gen_batch_decoded(suffs_batch, batch_vol)
        suffs_preds = self.model.predict_generator(g, steps, verbose=(0 if self.quiet else 1))
        pr(self.quiet, "\tCalcul de la probabilite des mots entiers...")
        preds = np.empty(len(words))
        for i in range(len(words)):
            word = self.hush.decode(words[i]) + [self.nalpha + 1]
            acc = 1.0
            for k in range(len(word)):
                x = self.hush.encode(word[:k])
                acc *= suffs_preds[x][word[k]]
            preds[i] = acc
        if asdict:  # On retourne un dictionnaire
            probas = dict()
            for i in range(len(words)):
                probas[i] = preds[i]
            return probas
        else:  # On retourne une liste
            return preds


if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 6:
        print("Usage   : {0} modelfile ranks lrows [testfile testtargetsfile]".format(sys.argv[0]))
        print("Exemple : {0} modelRNN1.h5py 6_10_20 3 5.pautomac.test 5.pautomac_solutions.txt".format(sys.argv[0]))
        sys.exit(-666)
    # XXXXXX :
    context_a = (sys.argv[1] + "r" + sys.argv[2] + "l" + sys.argv[3]).replace(" ", "_").replace("/", "+")
    print("Context :", context_a)
    arg_model = sys.argv[1]
    arg_ranks = [int(e) for e in sys.argv[2].split(sep="_")]
    arg_lrows_lcols = int(sys.argv[3])
    if len(sys.argv) >= 6:
        arg_testfile = sys.argv[4]
        arg_testtargetsfile = sys.argv[5]
        arg_perp = True
    else:
        arg_testfile = ""
        arg_testtargetsfile = ""
        arg_perp = False

    spex = SpexExpress(arg_model, arg_lrows_lcols, arg_testfile, arg_testtargetsfile, context_a)
    spex.ready()
    for rank in arg_ranks:
        est = spex.extr(rank)

    # est = custom_fit(arg_ranks, arg_lrows_lcols, arg_model, arg_perp, arg_testfile, arg_testtargetsfile, context_a)
    # sp.Automaton.write(est.automaton, filename=("aut-"+context))


# def gen_words_express(nalpha, lrows_lcols):
#     hush = Hush(2*lrows_lcols + 1, nalpha)
#     lig, col, wl = gen_words_indexes_as_int_para(nalpha, lrows_lcols, hush)
#     return hush, lig, col, wl
#
#
# def gen_words_indexes_as_int_para(nalpha, lrows_lcols, hush, quiet=False):
#     lig_col = []
#     thrs = []
#     # CREER LES THREADS ET LES LANCER
#     pr(quiet, "\tThreads lignes et colonnes...")
#     for d in range(lrows_lcols+1):
#         th = ParaRowsCols(nalpha, d)
#         thrs.append(th)
#         th.start()
#     # ATTENDRE LES THREADS ET RECUPERER LES RESULTATS
#     pr(quiet, "\tRecuperations lignes et colonnes...")
#     for th in thrs:
#         th.join()
#         lig_col += th.words
#     # DEL:
#     del thrs
#     # On trie pour faire comme dans hankel.py, trick pour donner plus d'importance aux mots courts dans la SVD
#     pr(quiet, "\tTri lignes et colonnes...")
#     lig_col = sorted(lig_col, key=lambda x: (len(x), x))
#     # ###
#     pr(quiet, "\tConstruction des mots...")
#     return lig_col, lig_col, range(hush.nl[2*lrows_lcols+1])
#
#
# def gen_batch_simple(word_list, nalpha, pad, h, batch_vol=1):
#     current_v = 0
#     batch = []
#     for wcode in word_list:
#         w = h.decode(wcode)
#         w = [nalpha + 1] + [elt + 1 for elt in w]
#         w = parse.pad_0(w, pad)
#         batch.append(w)
#         current_v += 1
#         if current_v == batch_vol:
#             current_v = 0
#             ret = np.array(batch)
#             batch = []
#             yield ret
#     if len(batch) > 0:
#         yield np.array(batch)


# def proba_words_g(model, x_words, nalpha, hush, asdict=True, quiet=False):
#     pad = int(model.input.shape[1])  # On déduit de la taille de la couche d'entrée le pad nécéssaire
#     suffs_batch = x_words  # Seulement car lrows et lcols sont "pleins"
#     # ################
#     pr(quiet, "\tUtilisation du RNN...")
#     batch_vol = 2048
#     steps = math.ceil(len(suffs_batch)/batch_vol)
#     g = gen_batch_simple(suffs_batch, nalpha, pad, hush, batch_vol)
#     suffs_preds = model.predict_generator(g, steps, verbose=(0 if quiet else 1))
#     pr(quiet, "\tCalcul de la probabilite des mots entiers...")
#     preds = np.empty(len(x_words))
#     for i in range(len(x_words)):
#         word = hush.decode(x_words[i])+[nalpha+1]
#         acc = 1.0
#         for k in range(len(word)):
#             x = hush.encode(word[:k])
#             acc *= suffs_preds[x][word[k]]
#         preds[i] = acc
#     if asdict:  # On retourne un dictionnaire
#         probas = dict()
#         for i in range(len(x_words)):
#             probas[i] = preds[i]
#         return probas
#     else:  # On retourne une liste
#         return preds


# def hankels(ligs, cols, probas, nalpha, hush):
#     lhankels = [np.zeros((len(ligs), len(cols))) for _ in range(nalpha+1)]
#     # EPSILON AND LETTER MATRICES :
#     letters = [[]] + [[i] for i in range(nalpha)]
#     for letter in range(len(letters)):
#         for l in range(len(ligs)):
#             for c in range(len(cols)):
#                 # lhankels[letter][l][c] = probas[tuple(ligs[l] + [letter] + cols[c])]
#                 lhankels[letter][l][c] = probas[hush.encode(ligs[l] + letters[letter] + cols[c])]
#     return lhankels


# def custom_fit(ranks, lrows_lcols, modelfile, perplexity=False, train_file="", target_file="", context=""):
#     model = keras.models.load_model(modelfile)
#     nalpha = int(model.output.shape[1])-2
#     ###
#     # Params :
#     quiet = False
#     epsilon = 0.0001
#     # Préparations :
#     pr(quiet, "Construction of set of words...")
#     hush, ligs, cols, lw = gen_words_express(nalpha, lrows_lcols)
#     pr(quiet, "Prediction of probabilities of words...")
#     probas = proba_words_g(model, lw, nalpha, hush)
#     pr(quiet, "Building of hankel matrices...")
#     lhankels = hankels(ligs, cols, probas, nalpha, hush)
#     if perplexity:
#         pr(quiet, "Perplexity preparations...")
#         x_test = parse.parse_fullwords(train_file)
#         x_test_sp = spparse.load_data_sample(train_file)
#         y_test = parse.parse_pautomac_results(target_file)
#
#         perp_proba_rnn = fix_probas(proba_words_para(model, x_test, nalpha, asdict=False, quiet=False), f=epsilon)
#         test_perp = scores.pautomac_perplexity(y_test, y_test)
#         rnn_perp = scores.pautomac_perplexity(y_test, perp_proba_rnn)
#         test_rnn_kl = scores.kullback_leibler(y_test, perp_proba_rnn)
#     else:
#         x_test_sp = None
#         y_test = None
#         rnn_perp = None
#         perp_proba_rnn = None
#         test_perp = None
#         test_rnn_kl = None
#     spectral_estimator = None
#     for rank in ranks:
#         if len(ranks) > 1:
#             pr(quiet, "Rank {0} among {1} :".format(rank, ranks))
#         spectral_estimator = sp.Spectral(rank=rank, lrows=lrows_lcols, lcolumns=lrows_lcols,
#                                          version='classic', partial=True, sparse=False,
#                                          smooth_method='none', mode_quiet=quiet)
#         # Les doigts dans la prise !
#         pr(quiet, "Custom fit ...")
#         spectral_estimator._hankel = sp.Hankel(sample_instance=None, lrows=lrows_lcols, lcolumns=lrows_lcols,
#                                                version='classic', partial=True, sparse=False,
#                                                mode_quiet=quiet, lhankel=lhankels)
#         # noinspection PyProtectedMember
#         spectral_estimator._automaton = spectral_estimator._hankel.to_automaton(rank, quiet)
#         # OK on a du a peu près rattraper l'état après fit.
#         pr(quiet, "... Done !")
#         # Perplexity :
#         sp.Automaton.write(spectral_estimator.automaton, filename=("aut-{0}-r-{1}".format(context, rank)))
#         if perplexity:
#             print("Perplexity :")
#             perp_proba_spec = fix_probas(spectral_estimator.predict(x_test_sp.data), f=epsilon)
#             extract_perp = scores.pautomac_perplexity(y_test, perp_proba_spec)
#             rnn_extr_kl = scores.kullback_leibler(perp_proba_rnn, perp_proba_spec)
#             test_extr_kl = scores.kullback_leibler(y_test, perp_proba_spec)
#
#             print("\tTest :\t{0}\n\tRNN :\t{1}\n\tExtr :\t{2}"
#                   .format(test_perp, rnn_perp, extract_perp))
#             print("KL Divergence :")
#             print("\tTest-RNN :\t{0}\n\tRNN-Extr :\t{1}\n\tTest-Extr :\t{2}"
#                   .format(test_rnn_kl, rnn_extr_kl, test_extr_kl))
#         #
#     return spectral_estimator
