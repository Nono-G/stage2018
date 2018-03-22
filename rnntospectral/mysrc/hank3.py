import math
import sys
import numpy as np
from spextractor_common import ParaRowsCols, ParaWords, ParaBatchHush, pr, SpexHush, parts_of_list, ParaProbas
import time


class SpexRandDrop(SpexHush):
    def __init__(self, modelfile, lrows, lcols, pref_drop, suff_drop, perp_train="", perp_targ="", context=""):
        SpexHush.__init__(self, modelfile, lrows, lcols, perp_train, perp_targ, context)
        self.pref_drop = pref_drop
        self.suff_drop = suff_drop
        self.batch_nb_threads = 4

    def gen_words(self):
        if not type(self.lrows) is list:
            if not type(self.lrows) is int:
                raise TypeError()
            else:
                self.lrows = [i for i in range(self.lrows + 1)]
        if not type(self.lcols) is list:
            if not type(self.lcols) is int:
                raise TypeError()
            else:
                self.lcols = [i for i in range(self.lcols + 1)]
        lig, col, wl = self.gen_words_indexes_as_lists_para()
        return lig, col, wl

    def rand_p_closed(self, words, targ_coeff):
        prefs_set = set()
        shufflek = np.random.choice(len(words), len(words), replace=False)
        i = 0
        while len(prefs_set) < len(words)*targ_coeff:
            word = words[shufflek[i]]
            for k in range(len(word)+1):
                prefs_set.add(tuple(word[:k]))
            i += 1
        coeff = len(prefs_set)/len(words)
        pr(self.quiet, "\t\tP-closure checked, real coeff = {0}".format(coeff))
        return [list(elt) for elt in prefs_set]

    def gen_words_indexes_as_lists_para(self):
        lig = []
        col = []
        thrs = []
        # CREER LES THREADS ET LES LANCER
        dims = [x for x in (set(self.lrows).union(set(self.lcols)))]
        dimdict = {}
        pr(self.quiet, "\tThreads lignes et colonnes...")
        for d in dims:
            th = ParaRowsCols(self.nalpha, d)
            thrs.append(th)
            th.start()
        # ATTENDRE LES THREADS ET RECUPERER LES RESULTATS
        pr(self.quiet, "\tRecuperations lignes et colonnes...")
        for th in thrs:
            th.join()
            dimdict[th.d] = th.words
        # FORMER lig et col:
        for d in self.lrows:
            lig += dimdict[d]
        for d in self.lcols:
            col += dimdict[d]
        # DEL:
        del thrs
        del dims
        del dimdict
        # Y'a pas la place pour tout le monde :
        pr(self.quiet, "\tP-closures...")
        col = self.rand_p_closed(col, self.pref_drop)
        lig = self.rand_p_closed(lig, self.suff_drop)
        pr(self.quiet, "\tTri lignes et colonnes...")
        # On trie pour faire comme dans hankel.py, trick pour donner plus d'importance aux mots courts dans la SVD
        col = sorted(col, key=lambda x: (len(x), x))
        lig = sorted(lig, key=lambda x: (len(x), x))
        # ###
        pr(self.quiet, "\tConstruction des mots...")
        encoded_words_set = set()
        letters = [[]] + [[i] for i in range(self.nalpha)]
        thrs = []
        for letter in letters:
            th = ParaWords(lig, col, letter, self.hush, 2, quiet=self.quiet)
            thrs.append(th)
            th.start()
        for th in thrs:
            th.join()
            encoded_words_set = encoded_words_set.union(th.words)
        return lig, col, list(encoded_words_set)

    def proba_words_special(self, words, asdict=True):
        suffs_batch = set()
        pr(self.quiet, "\tMise en batch des prefixes...")
        t1 = time.time()
        thrs = []
        words_chunks = parts_of_list(words, self.batch_nb_threads)
        for i in range(self.batch_nb_threads):
            th = ParaBatchHush(words_chunks[i], self.hush)
            th.start()
            thrs.append(th)
        for i in range(self.batch_nb_threads):
            thrs[i].join()
            suffs_batch = suffs_batch.union(thrs[i].preffixes_set)
        del thrs
        # del words_chunks
        # for wcode in words:
        #     w = self.hush.decode(wcode)
        #     for i in range(len(w) + 1):
        #         suffs_batch.add(self.hush.encode(w[:i]))
        t2 = time.time()
        print("dT : {0}".format((t2-t1)))
        suffs_batch = list(suffs_batch)
        # ################
        pr(self.quiet, "\tUtilisation du RNN...")
        steps = math.ceil(len(suffs_batch) / self.batch_vol)
        g = self.gen_batch_decoded(suffs_batch, self.batch_vol)
        suffs_preds = self.model.predict_generator(g, steps, verbose=(0 if self.quiet else 1))
        suffs_dict = {}
        for k in range(len(suffs_batch)):
            suffs_dict[suffs_batch[k]] = suffs_preds[k]
        del suffs_preds
        del suffs_batch
        pr(self.quiet, "\tCalcul de la probabilite des mots entiers...")
        t1 = time.time()
        preds = np.array([])
        # words_chunks = parts_of_list(words, self.batch_nb_threads)
        thrs = []
        for i in range(self.batch_nb_threads):
            th = ParaProbas(words_chunks[i], self.hush, suffs_dict)
            th.start()
            thrs.append(th)
        for i in range(self.batch_nb_threads):
            thrs[i].join()
            preds = np.concatenate((preds, thrs[i].preds))
        del thrs
        del words_chunks
        # preds = np.empty(len(words))
        # for i in range(len(words)):
        #     word = self.hush.decode(words[i]) + [self.nalpha + 1]
        #     acc = 1.0
        #     for k in range(len(word)):
        #         acc *= suffs_dict[self.hush.encode(word[:k])][word[k]]
        #     preds[i] = acc
        t2 = time.time()
        print("dT : {0}".format((t2 - t1)))
        del suffs_dict
        if asdict:  # On retourne un dictionnaire
            pr(self.quiet, "\tConstitution du dictionnaire...")
            probas = dict()
            for i in range(len(words)):
                probas[words[i]] = preds[i]
            return probas
        else:  # On retourne une liste
            return preds


if __name__ == "__main__":
    if len(sys.argv) < 7 or len(sys.argv) > 9:
        print("Usage :: {0} modelfile ranks lrows lcols coeffrows coeffcols[testfile testtargetsfile]"
              .format(sys.argv[0]))
        sys.exit(-666)
    # XXXXXX :
    context_a = ("{0}r{1}l{2}({3})c{4}({5})"
                 .format(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[5], sys.argv[4], sys.argv[6])
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
    arg_coeffrows = float(sys.argv[5])
    arg_coeffcols = float(sys.argv[6])
    if len(sys.argv) >= 9:
        arg_testfile = sys.argv[7]
        arg_testtargetsfile = sys.argv[8]
        arg_perp = True
    else:
        arg_testfile = ""
        arg_testtargetsfile = ""
        arg_perp = False

    spex = SpexRandDrop(arg_model, arg_lrows, arg_lcols, arg_coeffcols, arg_coeffrows, arg_testfile, arg_testtargetsfile, context_a)
    spex.ready()
    for rank in arg_ranks:
        est = spex.extr(rank)
        # est.Automaton.write(est.automaton, filename=("aut-"+context_a))
