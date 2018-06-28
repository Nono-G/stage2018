# Libraries
import multiprocessing
import sys
import torch
# Project
from spex_commons import pr
from hank3 import SpexRandDrop, words_task

"""Executable file, extraction with rnn-generated basis"""

class SpexRandRNNW(SpexRandDrop):
    """Override basis generation to use rnn-generated basis"""
    def __init__(self, digest, weights, pref_drop, suff_drop, m_test_set="", met_model="", context="", device="cpu"):
        SpexRandDrop.__init__(self, (digest+" "+weights), 1, 1, pref_drop, suff_drop, m_test_set, met_model, context, device)
        self.temp_coeff = 1
        # self.truncate = 20
        # self.randwords_maxlen = 10

    def gen_words_indexes_as_lists_para(self):
        nb_pref = self.pref_drop
        nb_suff = self.suff_drop

        hushed_words = set()
        hushed_prefixes = set()
        hushed_suffixes = set()
        # random.seed

        with torch.no_grad():
            self.rnn_model.eval()
            # genw_set = set()
            failures = 0
            while (len(hushed_prefixes) < nb_pref or len(hushed_suffixes) < nb_suff) and failures < self.patience:
                gen_word, selected = self.gen_one_with_rnn(self.temp_coeff)
                gen_word = tuple(gen_word)
                if selected == 0:
                    encoded_word = self.hush.encode(gen_word)
                    if encoded_word not in hushed_words:
                        failed_words = 0  # Reset patience counter
                        hushed_words.add(encoded_word)
                        if len(hushed_prefixes) < nb_pref:
                            hushed_prefixes.add(encoded_word)
                            hushed_prefixes.update(self.hush.prefixes_codes(encoded_word))
                        if len(hushed_suffixes) < nb_suff:
                            hushed_suffixes.update(self.hush.suffixes_codes(encoded_word))
                    else:
                        failed_words += 1
                    failures = 0
                    # genw_set.add(gen_word)
                else:
                    failures += 1
        if failures >= self.patience:
                pr(2, self.quiet, "Stopping words generation : out of patience")
        pr(2, self.quiet,
           "Nb of prefixes from rnn-gen words = {0}, expected = {1}".format(len(hushed_prefixes), nb_pref))
        pr(2, self.quiet,
           "Nb of suffixes from rnn-gen words = {0}, expected = {1}".format(len(hushed_suffixes), nb_suff))
        #
        # On trie pour faire comme dans hankel.py, trick pour donner plus d'importance aux mots courts dans la SVD
        lig = sorted(list(hushed_prefixes))
        col = sorted(list(hushed_suffixes))
        del hushed_prefixes
        del hushed_suffixes
        # ###
        pr(1, self.quiet, "Assembling words from suffixes and prefixes...")
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


def main():
    if len(sys.argv) < 7 or len(sys.argv) > 9:
        print(("Usage :: {0} DEVICE DIGESTFILE WEIGHTSFILE RANKS NB_PREFS NB_SUFFS [TEST_FILE [MODEL_FILE]]\n" +
               "DEVICE : 'cuda' or 'cpu' or a specific cuda device such as 'cuda:2'\n" +
               "DIGESTFILE : path to a digest file, produced by the model trainer\n" +
               "WEIGHTSFILE : path to a weights file, produced by the model trainer\n" +
               "RANKS : list of ranks to perform extractions, separated by '_' (e.g. '5_10_15_20')\n" +
               "NB_PREFS : number of unique prefixes in the sub-block basis\n" +
               "NB_SUFFS : number of unique suffixes in the sub-block basis\n" +
               "TEST_FILE : path to a file containing test strings (OPTIONAL)\n" +
               "MODEL_FILE : path to a file containing an automaton description (OPTIONAL)\n"
               )
              .format(sys.argv[0]))
        sys.exit(-666)
    # XXXXXX :
    context = ("H5-{0}l{1}c{2}"
               .format(sys.argv[3], sys.argv[5], sys.argv[6])
               .replace(" ", "_")
               .replace("/", "+"))
    print("Context :", context)
    print("Ranks :", sys.argv[4])
    device = sys.argv[1]
    model_digest = sys.argv[2]
    model_weights = sys.argv[3]
    ranks = [int(e) for e in sys.argv[4].split(sep="_")]
    nb_pref = float(sys.argv[5])
    nb_suffs = float(sys.argv[6])
    if len(sys.argv) >= 8:
        testfile = sys.argv[7]
    else:
        testfile = ""
    if len(sys.argv) >= 9:
        aut_model = sys.argv[8]
    else:
        aut_model = ""
    spex = SpexRandRNNW(model_digest, model_weights, nb_suffs, nb_pref, testfile, aut_model, context, device)
    for rank in ranks:
        _ = spex.extr(rank)
        # extr_aut = spex.extr(rank)
        # extr_aut.Automaton.write(est.automaton, filename=("aut-"+context))
    spex.print_metrics_chart()
    spex.print_metrics_chart_n_max(8)


if __name__ == "__main__":
    main()
