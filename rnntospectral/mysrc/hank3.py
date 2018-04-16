# Libraries :
import math
import sys
import random
import time
import numpy as np
import multiprocessing
# Project :
from spextractor_common import ParaRowsCols, ParaWords, ParaBatchHush, pr, SpexHush, parts_of_list, ParaProbas, Hush


class SpexRandDrop(SpexHush):
    def __init__(self, modelfilestring, lrows, lcols, pref_drop,
                 suff_drop, perp_train="", perp_targ="", met_model="", context=""):
        SpexHush.__init__(self, modelfilestring, lrows, lcols, perp_train, perp_targ, met_model, context)
        self.pref_drop = pref_drop
        self.suff_drop = suff_drop

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
        pr(self.quiet, "\tExtraction of prefixes...")
        suffs_batch = set(words)  # Words themselves, then their prefixes
        if self.nb_proc > 1:
            # MULTIPROCESSED :
            words_chunks = parts_of_list(words, self.nb_proc)
            p = multiprocessing.Pool(self.nb_proc)
            args = [(self.hush.maxlen, self.hush.base, words_chunk) for words_chunk in words_chunks]
            for s in p.map(batch_task, args):
                suffs_batch.update(s)
        else:
            # LINEAR :
            for wcode in words:
                suffs_batch.update(self.hush.prefixes_codes(wcode))
        suffs_batch = list(suffs_batch)
        # ################
        pr(self.quiet, "\tPredictions...")
        steps = math.ceil(len(suffs_batch) / self.batch_vol)
        g = self.gen_batch_decoded(suffs_batch, self.batch_vol)
        suffs_preds = self.rnn_model.predict_generator(g, steps, verbose=(0 if self.quiet else 1))
        if suffs_preds.shape[1] > self.nalpha + 2:
            suffs_preds = np.delete(suffs_preds, 0, axis=1)
        suffs_dict = {}
        for k in range(len(suffs_batch)):
            suffs_dict[suffs_batch[k]] = suffs_preds[k]
        del suffs_preds
        del suffs_batch
        pr(self.quiet, "\tFullwords probas...")
        # Here the results are inexpected : Linear < Thread < Multiproc
        # So we keep it simple linear...
        # t1 = time.time()
        # # MULTIPROCESSING
        # preds1 = []
        # if self.nb_proc > 1:
        #     words_chunks = parts_of_list(words, self.nb_proc)
        #     p = multiprocessing.Pool(self.nb_proc)
        #     args = [(self.hush.maxlen, self.nalpha, chunk, dict(suffs_dict)) for chunk in words_chunks]
        #     for a in p.map(fullproba_task, args):
        #         preds1 += a
        # preds1 = np.array(preds1)
        # t2 = time.time()
        # print("dT : {0}".format((t2 - t1)))
        # # THREADING
        # t1 = time.time()
        # preds2 = np.array([])
        # words_chunks = parts_of_list(words, self.nb_proc)
        # thrs = []
        # for i in range(self.nb_proc):
        #     th = ParaProbas(words_chunks[i], self.hush, suffs_dict)
        #     th.start()
        #     thrs.append(th)
        # for i in range(self.nb_proc):
        #     thrs[i].join()
        #     preds2 = np.concatenate((preds2, thrs[i].preds))
        # del thrs
        # del words_chunks
        # t2 = time.time()
        # print("dT : {0}".format((t2 - t1)))
        # # LINEAR
        # t1 = time.time()
        preds3 = np.empty(len(words))
        for i in range(len(words)):
            word = self.hush.decode(words[i]) + [self.nalpha + 1]
            pcode = sorted(list(self.hush.prefixes_codes(words[i]))) + [words[i]]
            acc = 1.0
            for k in range(len(word)):
                # acc *= suffs_dict[self.hush.encode(word[:k])][word[k]]
                acc *= suffs_dict[pcode[k]][word[k]]
            preds3[i] = acc
        # t2 = time.time()
        # print("dT : {0}".format((t2 - t1)))
        del suffs_dict
        # assert np.array_equal(preds2, preds3)
        # assert np.array_equal(preds1, preds2)
        preds = preds3
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


def batch_task(args):
    maxlen, base, words = args
    ret_set = set(words)
    h = Hush(maxlen, base)
    for wcode in words:
        ret_set.update(h.prefixes_codes(wcode))
    return ret_set


def fullproba_task(args):
    maxlen, base, words, dic = args
    h = Hush(maxlen, base)
    preds = [0]*(len(words))
    for i in range(len(words)):
        word = h.decode(words[i]) + [base + 1]
        pcode = sorted(list(h.prefixes_codes(words[i]))) + [words[i]]
        acc = 1.0
        for k in range(len(word)):
            acc *= dic[pcode[k]][word[k]]
        preds[i] = acc
    return preds


if __name__ == "__main__":
    if len(sys.argv) < 7 or len(sys.argv) > 10:
        print("Usage :: {0} modelfile ranks lrows lcols coeffrows coeffcols [testfile testtargetsfile testmodel]"
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
    arg_lrows = [int(e) for e in sys.argv[3].split(sep="_")]
    arg_lcols = [int(e) for e in sys.argv[4].split(sep="_")]
    arg_coeffrows = float(sys.argv[5])
    arg_coeffcols = float(sys.argv[6])
    if len(sys.argv) >= 9:
        arg_testfile = sys.argv[7]
        arg_testtargetsfile = sys.argv[8]
        arg_aut_model = sys.argv[9]
    else:
        arg_testfile = ""
        arg_testtargetsfile = ""
        arg_aut_model = ""

    spex = SpexRandDrop(arg_model, arg_lrows, arg_lcols, arg_coeffcols, arg_coeffrows,
                        arg_testfile, arg_testtargetsfile, arg_aut_model, context_a)
    spex.ready()
    for rank in arg_ranks:
        est = spex.extr(rank)
        # est.Automaton.write(est.automaton, filename=("aut-"+context_a))
    spex.print_metrics_chart()