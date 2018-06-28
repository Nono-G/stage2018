# Libraries :
import math
import sys
import random
import numpy as np
import multiprocessing
# Project :
from spex_commons import pr, SpexHush
from hush import Hush


class SpexRandDrop(SpexHush):
    def __init__(self, modelfilestring, lrows, lcols, pref_drop, suff_drop, m_test_set="", m_model="", context="", device="cpu"):
        SpexHush.__init__(self, modelfilestring, lrows, lcols, m_test_set, m_model, context, device)
        self.pref_drop = pref_drop
        self.suff_drop = suff_drop
        self.prelim_dict = dict()

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

    # noinspection PyPep8
    def rand_p_closed(self, words, targ_coeff):
        prefs_set = set()
        index_set = set()
        miss = 0
        if targ_coeff > 1.0:
            target = targ_coeff
            tarfun = lambda x, y: x
        else:
            target = len(words)*targ_coeff
            tarfun = lambda x, y: y
        while (tarfun(len(prefs_set), len(index_set)) < target) and miss < self.patience:
            key = random.randrange(0, len(words))
            if key not in index_set:
                index_set.add(key)
                miss = 0
                prefs_set.add(words[key])
                prefs_set.update(self.hush.prefixes_codes(words[key]))
            else:
                miss += 1
        coeff = len(index_set)/len(words)
        if miss == self.patience:
            pr(self.quiet, "\t\tRun out of patience !")
        pr(self.quiet, "\t\tP-closure checked, nb of elements = {0}, real coeff = {1}".format(len(prefs_set), coeff))
        return list(prefs_set)

    def rand_s_closed(self, words, targ_coeff):
        suffs_set = set()
        index_set = set()
        miss = 0
        if targ_coeff > 1.0:
            target = targ_coeff
            # noinspection PyPep8
            tarfun = lambda x, y: x
        else:
            target = len(words)*targ_coeff
            # noinspection PyPep8
            tarfun = lambda x, y: y
        while (tarfun(len(suffs_set), len(index_set)) < target) and miss < self.patience:
            key = random.randrange(0, len(words))
            if key not in index_set:
                index_set.add(key)
                miss = 0
                suffs_set.add(words[key])
                suffs_set.update(self.hush.suffixes_codes(words[key]))
            else:
                miss += 1
        coeff = len(index_set)/len(words)
        if miss == self.patience:
            pr(self.quiet, "\t\tRun out of patience !")
        pr(self.quiet, "\t\tP-closure checked, nb of elements = {0}, real coeff = {1}".format(len(suffs_set), coeff))
        return list(suffs_set)

    def gen_words_indexes_as_lists_para(self):
        lig = RangeUnion([self.hush.words_of_len(x) for x in self.lrows])
        col = RangeUnion([self.hush.words_of_len(x) for x in self.lcols])
        # Y'a pas la place pour tout le monde :
        pr(self.quiet, "\tP-closures...")
        lig = self.rand_p_closed(lig, self.suff_drop)
        col = self.rand_p_closed(col, self.pref_drop)
        # On trie pour faire comme dans hankel.py, trick pour donner plus d'importance aux mots courts dans la SVD
        lig = sorted(list(lig))
        col = sorted(list(col))
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

    def proba_words_special(self, words, asdict=True):
        # predictions = self.rnn_model.probas_tables_numpy(words, self.batch_vol, hush=self.hush,
        #                                                  del_start_symb=True, device=self.device)
        # start_factor = (len(words)) / (sum([self.hush.len_code(w) for w in words]))
        # # start_factor = 1
        # preds = np.empty(len(words))
        # for i, wcode in enumerate(words):
        #     word = [x+1 for x in self.hush.decode(wcode)] + [0]
        #     acc = start_factor
        #     for k in range(len(word)):
        #         acc *= predictions[i,k,word[k]]
        #     preds[i] = acc
        preds = self.rnn_model.full_probas(words, self.batch_vol, self.hush, del_start_symb=True, device=self.device)
        if asdict:  # On retourne un dictionnaire
            probas = dict()
            for i in range(len(words)):
                probas[words[i]] = preds[i]
            return probas
        else:  # On retourne une liste
            return preds


class RangeUnion:
    def __init__(self, ranges):
        self.ranges = ranges
        acc = 0
        for r in self.ranges:
            acc += (r.stop - r.start)
        self.le = acc

    def __len__(self):
        return self.le

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError
        skip = 0
        i = 0
        while item > (skip + len(self.ranges[i]) - 1):
            skip += len(self.ranges[i])
            i += 1
        return self.ranges[i][item - skip]

    def __iter__(self):
        for r in self.ranges:
            for k in self.ranges[r]:
                yield k


def words_task(args):
    maxlen, base, letter, rows, cols = args
    h = Hush(maxlen, base)
    enc_set = set()
    for prefix in rows:
        for suffix in cols:
            enc_set.add(h.encode(h.decode((prefix, letter, suffix))))
    return enc_set


def main():
    if len(sys.argv) < 9 or len(sys.argv) > 11:
        print("Usage :: {0} device digestfile weightsfile ranks lrows lcols coeffrows coeffcols [testfile [modelfile]]"
              .format(sys.argv[0]))
        sys.exit(-666)
    # XXXXXX :
    context = ("H3-{0}l{1}x{2}c{3}x{4}"
                 .format(sys.argv[3], sys.argv[5], sys.argv[7], sys.argv[6], sys.argv[8])
                 .replace(" ", "_")
                 .replace("/", "+"))
    print("Context :", context)
    print("Ranks :", sys.argv[4])
    device = sys.argv[1]
    digest = sys.argv[2]
    weights = sys.argv[3]
    ranks = [int(e) for e in sys.argv[4].split(sep="_")]
    lrows = [int(e) for e in sys.argv[5].split(sep="_")]
    lcols = [int(e) for e in sys.argv[6].split(sep="_")]
    coeffrows = float(sys.argv[7])
    coeffcols = float(sys.argv[8])
    if len(sys.argv) >= 10:
        testfile = sys.argv[9]
    else:
        testfile = ""
    if len(sys.argv) >= 11:
        aut_model = sys.argv[10]
    else:
        aut_model = ""
    spex = SpexRandDrop((digest+" "+weights), lrows, lcols, coeffcols, coeffrows, testfile, aut_model, context, device)
    for rank in ranks:
        _ = spex.extr(rank)
        # est.Automaton.write(est.automaton, filename=("aut-"+context))
    spex.print_metrics_chart()
    spex.print_metrics_chart_n_max(8)


if __name__ == "__main__":
    main()
