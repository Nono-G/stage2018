# IMPORT
# Libraries
import numpy as np
import multiprocessing
import sys
# Project
from spextractor_common import pr
import parse5 as parse
from hank3 import SpexRandDrop, words_task


class SpexRandRNNW(SpexRandDrop):
    def __init__(self, modelfilestring, pref_drop, suff_drop, m_test_set="", met_model="", context=""):
        SpexRandDrop.__init__(self, modelfilestring, 1, 1, pref_drop, suff_drop, m_test_set, met_model, context)

    def gen_words_indexes_as_lists_para(self):
        nb_pref = self.pref_drop
        nb_suff = self.suff_drop

        hushed_words = set()
        hushed_prefixes = set()
        hushed_suffixes = set()
        # random.seed()
        failed_words = 0
        while (len(hushed_prefixes) < nb_pref or len(hushed_suffixes) < nb_suff) and failed_words < self.patience:
            word = list()
            enc_word = parse.pad_0([self.nalpha + 1], self.pad)  # Begin symbol, padded
            while ((len(word) == 0) or (word[-1] != self.nalpha)) and len(word) <= self.randwords_maxlen:
                try:
                    probas = self.prelim_dict[tuple(word)]
                except KeyError:
                    probas = self.rnn_model.predict(np.array([enc_word]))
                    if probas.shape[1] > self.nalpha + 2:
                        # "couic la colonne de padding !"
                        probas = np.delete(probas, 0, axis=1)
                    # We also remove the "begin" symbol, as it is not a valid choice
                    probas = np.delete(probas, 4, axis=1)
                    probas = probas[0]  # Unwrap the predicted singleton
                    # Normalize it, because we made some deletions:
                    s = sum(probas)
                    probas = [p / s for p in probas]
                    self.prelim_dict[tuple(word)] = probas
                i = np.asscalar(np.random.choice(len(probas), 1, p=probas)[0])
                word += [i]
                enc_word = enc_word[1:] + [i + 1]
            if word[-1] == self.nalpha:
                #  We want to pass around non-encoded words, so we remove the ending symbol at the end
                coded_word = self.hush.encode(word[:-1])
                if coded_word not in hushed_words:
                    failed_words = 0  # Reset patience counter
                    hushed_words.add(coded_word)
                    hushed_prefixes.add(coded_word)
                    hushed_prefixes.update(self.hush.prefixes_codes(coded_word))
                    hushed_suffixes.update(self.hush.suffixes_codes(coded_word))
                else:
                    failed_words += 1
            else:
                failed_words += 1
        if failed_words == self.patience:
            pr(self.quiet, "\t\tRun out of patience !")
        pr(self.quiet,
           "\t\tNb of prefixes from rnn-gen words = {0}, expected = {1}".format(len(hushed_prefixes), nb_pref))
        pr(self.quiet,
           "\t\tNb of suffixes from rnn-gen words = {0}, expected = {1}".format(len(hushed_suffixes), nb_suff))
        # On trie pour faire comme dans hankel.py, trick pour donner plus d'importance aux mots courts dans la SVD
        lig = sorted(list(hushed_prefixes))
        col = sorted(list(hushed_suffixes))
        del hushed_prefixes
        del hushed_suffixes
        # ###
        pr(self.quiet, "\tAssembling words from suffixes and prefixes...")
        letters = [[]] + [[i] for i in range(self.nalpha)]
        encoded_words_set = set()
        if self.nb_proc > 1:
            # MULTIPROCESSED
            args = [(self.hush.maxlen, self.hush.base, self.hush.encode(l), lig, col) for l in letters]
            p = multiprocessing.Pool(self.nb_proc)
            for s in p.map(words_task, args):
                encoded_words_set.update(s)
        else:
            # LINEAR
            for letter in [self.hush.encode(l) for l in letters]:
                for prefix in lig:
                    for suffix in col:
                        encoded_words_set.add(self.hush.encode(self.hush.decode((prefix, letter, suffix))))
        # Splearn takes a list of words, it does not understand (yet) word codes.
        lig = [self.hush.decode(x) for x in lig]
        col = [self.hush.decode(x) for x in col]
        return lig, col, list(encoded_words_set)


if __name__ == "__main__":
    if len(sys.argv) < 5 or len(sys.argv) > 7:
        print("Usage :: {0} modelstring ranks nb_prefs nb_suffs [test_file [model_file]]"
              .format(sys.argv[0]))
        sys.exit(-666)
    # XXXXXX :
    context_a = ("H5-{0}ln{1}cn{2}"
                 .format(sys.argv[1], sys.argv[3], sys.argv[4])
                 .replace(" ", "_")
                 .replace("/", "+"))
    print("Context :", context_a)
    print("Ranks :", sys.argv[2])
    arg_model = sys.argv[1]
    arg_ranks = [int(e) for e in sys.argv[2].split(sep="_")]
    arg_nb_pref = float(sys.argv[3])
    arg_nb_suffs = float(sys.argv[4])
    if len(sys.argv) >= 9:
        arg_testfile = sys.argv[5]
        arg_aut_model = sys.argv[6]
    else:
        arg_testfile = ""
        arg_aut_model = ""

    spex = SpexRandRNNW(arg_model, arg_nb_suffs, arg_nb_pref, arg_testfile, arg_aut_model, context_a)
    for rank in arg_ranks:
        est = spex.extr(rank)
        # est.Automaton.write(est.automaton, filename=("aut-"+context_a))
    spex.print_metrics_chart()
    spex.print_metrics_chart_n_max(8)
