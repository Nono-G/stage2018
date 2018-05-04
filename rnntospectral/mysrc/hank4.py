import math
import sys
import numpy as np
# Maison :
from spextractor_common import ParaRowsCols, pr, SpexHush


class SpexExpress(SpexHush):
    """
    Spex "Express" : lrows=lcols = un entier, et tout est "plein" dans l'ensemble des mots
    """

    def __init__(self, modelfilestring, lrows_lcols, m_test_set="", perp_mod="", context=""):
        SpexHush.__init__(self, modelfilestring, lrows_lcols, lrows_lcols, m_test_set, perp_mod, context)

    # TODO : Replace threading by multiprocessing here
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
        steps = math.ceil(len(suffs_batch) / self.batch_vol)
        g = self.gen_batch_decoded(suffs_batch, self.batch_vol)
        suffs_preds = self.rnn_model.predict_generator(g, steps, verbose=(0 if self.quiet else 1))
        if suffs_preds.shape[1] > self.nalpha + 2:
            suffs_preds = np.delete(suffs_preds, 0, axis=1)
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
        print("Usage   : {0} modelfile ranks lrows [testfile [testmodel]]".format(sys.argv[0]))
        print("Example : {0} modelRNN1.h5py 6_10_20 3 5.pautomac.test 5.pautomac_model.txt"
              .format(sys.argv[0]))
        sys.exit(-666)
    # XXXXXX :
    context_a = ("H4-"+sys.argv[1] + "r" + sys.argv[2] + "l" + sys.argv[3]).replace(" ", "_").replace("/", "+")
    print("Context :", context_a)
    arg_model = sys.argv[1]
    arg_ranks = [int(e) for e in sys.argv[2].split(sep="_")]
    arg_lrows_lcols = int(sys.argv[3])
    if len(sys.argv) >= 6:
        arg_testfile = sys.argv[4]
        arg_aut_model = sys.argv[5]
    else:
        arg_testfile = ""
        arg_aut_model = ""

    spex = SpexExpress(arg_model, arg_lrows_lcols, arg_testfile, arg_aut_model, context_a)
    for rank in arg_ranks:
        est = spex.extr(rank)
        # sp.Automaton.write(est.automaton, filename=("aut-{0}-{1}".format(context_a, rank)))
    spex.print_metrics_chart()
