# Libraries
import threading
import math
import random
import numpy as np
import sys
import splearn as sp
import os
# Project :
import parse5 as parse
import scores
import train2f

"""
Import only file, containing abstract classes, and related classes and functions for spectral extraction
"""


# Parallelization classes:
class ParaRowsCols(threading.Thread):
    """
    Thread enumerating all possible sequence of a given length, with a given alphabet
    see 'combinaisons_para()'
    """
    def __init__(self, nalpha, d,):
        threading.Thread.__init__(self)
        self.words = []
        self.d = d
        self.nalpha = nalpha

    def run(self):
        self.words = combinaisons_para(self.nalpha, self.d).tolist()


class ParaCombinaisons(threading.Thread):
    """
    Thread filling an array column with letters
    see 'comb4()'
    """
    def __init__(self, a, r, p, nalpha):
        threading.Thread.__init__(self)
        self.a = a
        self.r = r
        self.p = p
        self.nalpha = nalpha

    def run(self):
        comb4(self.a, self.r, self.p, self.nalpha)


class ParaWords(threading.Thread):
    """
    Thread enumerating all possible words made of prefix|letter|suffix,
    with a given letter and given prefixes and suffixes sets
    """
    def __init__(self, prefixes, suffixes, letter, hush, subthreads=1, print_id=-1, quiet=True):
        threading.Thread.__init__(self)
        self.prefixes = prefixes
        self.suffixes = suffixes
        self.letter = letter
        self.words = set()
        self.subthreads = subthreads
        self.quiet = quiet
        self.id = print_id
        self.h = hush.get_copy()

    def run(self):
        ws = set()
        # recursive parallelization :
        if self.subthreads > 1:
            pr(self.quiet, "\t\tThread letter {0} begin".format(self.letter))
            thrs = []
            prefparts = parts_of_list(self.prefixes, self.subthreads)
            for i in range(self.subthreads):
                th = ParaWords(prefparts[i], self.suffixes, self.letter, self.h, 1, i)
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
                    w = self.h.encode(self.h.decode((p, self.letter, s)))
                    ws.add(w)
            pr(self.quiet, "\t\t\tThread letter {0}, part {1} end".format(self.letter, self.id))
        # Anyway :
        self.words = ws


class ParaBatchHush(threading.Thread):
    def __init__(self, words, hush):
        threading.Thread.__init__(self)
        self.words = words
        self.preffixes_set = set()
        self.hush = hush.get_copy()
        # self.hush = hush

    def run(self):
        for wcode in self.words:
            w = self.hush.decode(wcode)
            for i in range(len(w) + 1):
                self.preffixes_set.add(self.hush.encode(w[:i]))


class ParaProbas(threading.Thread):
    def __init__(self, words, hush, prob_dict):
        threading.Thread.__init__(self)
        self.words = words
        self.hush = hush
        self.dict = prob_dict
        self.preds = np.empty(len(words))
        self.nalpha = self.hush.base

    def run(self):
        for i in range(len(self.words)):
            word = self.hush.decode(self.words[i]) + [self.nalpha + 1]
            pcode = sorted(list(self.hush.prefixes_codes(self.words[i]))) + [self.words[i]]
            acc = 1.0
            for k in range(len(word)):
                # acc *= self.dict[self.hush.encode(word[:k])][word[k]]
                acc *= self.dict[pcode[k]][word[k]]
            self.preds[i] = acc


# Other classes :
class Hush:
    """
    (Sequence of integers / integers) codec.
    Warning : this is not a x-based integers to 10-based integers codec, as for instance [0] and [0,0,0] sequences have
    different encodings.
    """
    def __init__(self, maxlen, base):
        self.maxlen = max(2, maxlen)
        self.base = base
        self.nl = []
        self.pows = []
        self.ready()
        self.maxval = self.encode([base - 1] * maxlen)

    def ready(self):
        self.pows = [1] * (self.maxlen+1)
        for i in range(1, self.maxlen+1):
            self.pows[i] = self.pows[i-1] * self.base
        #
        self.nl = [1] * (self.maxlen+1)
        for i in range(1, self.maxlen+1):
            self.nl[i] = self.nl[i-1] + (self.pows[i])
        #

    def get_copy(self):
        h = Hush(self.maxlen, self.maxval)
        return h

    def words_of_len(self, le):
        if le == 0:
            return range(1)
        else:
            r = range(self.nl[le - 1], self.nl[le])
            return r
            # return [i for i in r]

    def encode(self, w):
        if len(w) > self.maxlen:
            print(w)
            raise ValueError
        if len(w) == 0:
            return 0
        else:
            x = self.nl[len(w)-1]
            for i in range(len(w)):
                x += w[i]*self.pows[len(w)-i-1]
            return x

    def decode(self, s):
        if isinstance(s, tuple):
            r = []
            for x in s:
                r += self.decode(x)
            return r
        else:
            return self._decode(s)

    def _decode(self, n):
        if n > self.maxval:
            raise ValueError
        le = 0
        while self.nl[le] <= n:
            le += 1
        x = [0]*le
        reste = n - self.nl[le-1]
        for i in range(le):
            x[le-i-1] = reste % self.base
            reste //= self.base
        return x

    def prefix_code(self, x):
        if x >= self.nl[1]:
            i = 0
            while i < self.maxlen and self.nl[i] <= x:
                i += 1
            i -= 1
            d = (x - self.nl[i]) // self.base
            return self.nl[i-1]+d
        else:
            return 0

    def prefixes_codes(self, x):
        r = set()
        c = x
        i = 0
        while i < self.maxlen and self.nl[i] <= c:
            i += 1
        i -= 1
        while i > 0:
            d = (c - self.nl[i]) // self.base
            c = self.nl[i-1]+d
            r.add(c)
            i -= 1
        r.add(0)
        return r

    def concat_code(self, tup):
        return self.encode(self.decode(tup))


class Spex:
    """
    SPctral EXtractor base abstract class.
    """
    def __init__(self, modelfilestring, lrows, lcols, perp_train="", perp_targ="", perp_model="", context=""):
        self.nb_proc = len(os.sched_getaffinity(0))
        self.is_ready = False
        # Semi-constants :
        self.quiet = False
        self.epsilon = 1e-30
        self.batch_vol = 2048
        self.randwords_minlen = 0
        self.randwords_maxlen = 70
        self.randwords_nb = 2000  # Attention risque de boucle infinie si trop de mots !
        self.patience = 250
        # Arguments :
        self.rnn_model = train2f.my_load_model(*(modelfilestring.split()))
        self.lrows = lrows
        self.lcols = lcols
        self.perplexity_train = perp_train
        self.perplexity_target = perp_targ
        self.perplexity_model = perp_model
        self.context = context
        # Attributes derived from arguments :
        try:
            # "mode 0"
            self.nalpha = int(self.rnn_model.layers[0].input_dim) - 3
        except AttributeError:
            # "mode 1", with custom embedding
            self.nalpha = int(self.rnn_model.layers[1].input_dim) - 3
        self.pad = int(self.rnn_model.input.shape[1])
        self.metrics_calc = (perp_train != "" and perp_targ != "" and perp_model != "")
        # Computed attributes
        self.prefixes = None
        self.suffixes = None
        self.words = None
        self.words_probas = None
        self.lhankels = None
        # metrics calculations attributes
        self.ranks = []
        self.true_automaton = None
        self.metrics = dict()

        self.x_test = None
        self.x_rand = None
        self.x_rnnw = None
        self.y_test_target = None
        self.y_test_rnn = None
        self.y_rand_target = None
        self.y_rand_rnn = None
        self.y_rand_extr = None
        self.y_test_target_prefixes = None
        self.y_test_rnn_prefixes = None
        self.y_test_extr_prefixes = None
        self.y_rnnw_rnn_prefixes = None
        self.y_rnnw_extr_prefixes = None
        self.perp_test_target = None
        self.perp_test_rnn = None
        self.perp_test_extr = None
        self.perp_rand_target = None
        self.perp_rand_rnn = None
        self.kld_test_target_rnn = None
        self.kld_test_rnn_extr = None
        self.kld_test_target_extr = None
        self.kld_rand_target_rnn = None
        self.kld_rand_rnn_extr = None
        self.kld_rand_target_extr = None
        self.kld_rand_extr_rnn = None
        self.wer_test_target = None
        self.wer_test_rnn = None
        self.wer_rnnw_rnn = None
        self.wer_test_extr = None
        self.wer_rnnw_extr = None
        self.ndcg1_test_target_rnn = None
        self.ndcg1_test_rnn_extr = None
        self.ndcg1_test_target_extr = None
        self.ndcg1_rnnw_rnn_extr = None
        self.ndcg5_test_target_rnn = None
        self.ndcg5_test_rnn_extr = None
        self.ndcg5_test_target_extr = None
        self.ndcg5_rnnw_rnn_extr = None
        self.perp_rand_extr = None
        self.eps_test_zeros_extr = None
        self.l2dis_target_extr = None
        self.eps_rand_zeros_target = None
        self.eps_kl_rand_model_extr = None
        self.eps_rand_zeros_extr = None

    def ready(self):
        if self.metrics_calc:
            pr(self.quiet, "Metrics prelims...")
            pr(self.quiet, "\tParsings...")

            self.true_automaton = sp.Automaton.load_Pautomac_Automaton(self.perplexity_model)
            self.x_test = parse.parse_fullwords(self.perplexity_train)
            self.y_test_target = parse.parse_pautomac_results(self.perplexity_target)
            pr(self.quiet, "\tGenerating random words and random-rnn words...")
            self.x_rand = self.randwords(self.randwords_nb, self.randwords_minlen, self.randwords_maxlen)
            self.x_rnnw = gen_rnn_forever(self.rnn_model, nb_per_seed=self.randwords_nb, maxlen=self.randwords_maxlen)

            pr(self.quiet, "\tEvaluating words and prefixes...")
            self.y_test_rnn_prefixes = proba_all_prefixes_rnn(self.rnn_model, self.x_test, del_start_symb=True)
            self.y_test_target_prefixes = proba_all_prefixes_aut(self.true_automaton, self.x_test)
            self.y_test_rnn, t, e = self.proba_words_normal(self.x_test, asdict=False, wer=True, dic=self.y_test_rnn_prefixes)
            self.y_rand_target = [self.true_automaton.val(w) for w in self.x_rand]
            self.y_rand_rnn = self.proba_words_normal(self.x_rand, asdict=False)

            pr(self.quiet, "\tRank-independent metrics...")
            self.wer_test_rnn = e / t
            t, e = scores.wer_aut(self.true_automaton, self.x_test)
            self.wer_test_target = e / t
            garb, t, e = self.proba_words_normal(self.x_rnnw, asdict=False, wer=True)
            self.wer_rnnw_rnn = e / t

            self.perp_test_target = scores.pautomac_perplexity(self.y_test_target, self.y_test_target)
            self.perp_test_rnn = scores.pautomac_perplexity(self.y_test_target, self.y_test_rnn)
            self.perp_rand_target = scores.pautomac_perplexity(self.y_rand_target, self.fix_probas(self.y_rand_target))
            self.perp_rand_rnn = scores.pautomac_perplexity(self.y_rand_target, self.y_rand_rnn)

            self.kld_test_target_rnn = scores.kullback_leibler(self.y_test_target, self.y_test_rnn)
            self.kld_rand_target_rnn = scores.kullback_leibler(self.y_rand_target, self.y_rand_rnn)

            self.ndcg1_test_target_rnn = scores.ndcg(self.x_test, self.true_automaton, self.rnn_model, ndcg_l=1,
                                                     dic_ref=self.y_test_target_prefixes,
                                                     dic_approx=self.y_test_rnn_prefixes)
            self.ndcg5_test_target_rnn = scores.ndcg(self.x_test, self.true_automaton, self.rnn_model, ndcg_l=5,
                                                     dic_ref=self.y_test_target_prefixes,
                                                     dic_approx=self.y_test_rnn_prefixes)
            self.eps_rand_zeros_target = len([x for x in self.y_rand_target if x <= 0.0]) / len(self.y_rand_target)

            self.metrics[(-1, "perp-test-target")] = self.perp_test_target
            self.metrics[(-1, "perp-test-rnn")] = self.perp_test_rnn
            self.metrics[(-1, "perp-rand-target")] = self.perp_rand_target
            self.metrics[(-1, "perp-rand-target-eps")] = self.eps_rand_zeros_target
            self.metrics[(-1, "perp-rand-rnn")] = self.perp_rand_rnn
            self.metrics[(-1, "kld-test-target-rnn")] = self.kld_test_target_rnn
            self.metrics[(-1, "kld-rand-target-rnn")] = self.kld_rand_target_rnn
            self.metrics[(-1, "(1-wer)-test-target")] = self.wer_test_target
            self.metrics[(-1, "(1-wer)-test-rnn")] = 1 - self.wer_test_rnn
            self.metrics[(-1, "(1-wer)-rnnw-rnn")] = 1 - self.wer_rnnw_rnn
            self.metrics[(-1, "ndcg1-test-target-rnn")] = self.ndcg1_test_target_rnn
            self.metrics[(-1, "ndcg5-test-target-rnn")] = self.ndcg5_test_target_rnn
        # *********
        pr(self.quiet, "Prefixes, suffixes, words, ...")
        self.prefixes, self.suffixes, self.words = self.gen_words()
        pr(self.quiet, "Prediction of probabilities of words...")
        self.words_probas = self.proba_words_special(self.words, True)
        pr(self.quiet, "Building of hankel matrices...")
        self.lhankels = self.hankels()
        self.is_ready = True

    def extr(self, rank):
        if not self.is_ready:
            self.ready()
        self.ranks.append(rank)
        spectral_estimator = sp.Spectral(rank=rank, lrows=self.lrows, lcolumns=self.lrows,
                                         version='classic', partial=True, sparse=False,
                                         smooth_method='none', mode_quiet=self.quiet)
        # Les doigts dans la prise !
        pr(self.quiet, "Custom fit ...")
        try:
            spectral_estimator._hankel = sp.Hankel(sample_instance=None, lrows=self.lrows, lcolumns=self.lrows,
                                                   version='classic', partial=True, sparse=False,
                                                   mode_quiet=self.quiet, lhankel=self.lhankels)
            # noinspection PyProtectedMember
            spectral_estimator._automaton = spectral_estimator._hankel.to_automaton(rank, self.quiet)
            # OK on a du a peu près rattraper l'état après fit.
        except ValueError:
            pr(False, "Error, rank too big compared to the length of words")
            return None
        pr(self.quiet, "... Done !")
        # Metrics :
        # sp.Automaton.write(spectral_estimator.automaton, filename=("aut-{0}-r-{1}".format(self.context, rank)))
        if self.metrics_calc:
            print("Metrics for rank {0} :".format(rank))
            extr_aut = spectral_estimator.automaton

            pr(self.quiet, "\tEvaluating words and prefixes...")
            y_test_extr = [extr_aut.val(w) for w in self.x_test]
            self.y_rand_extr = [extr_aut.val(w) for w in self.x_rand]
            self.y_test_extr_prefixes = proba_all_prefixes_aut(extr_aut, self.x_test)
            self.y_rnnw_extr_prefixes = proba_all_prefixes_aut(extr_aut, self.x_rnnw)
            self.y_rnnw_rnn_prefixes = proba_all_prefixes_rnn(self.rnn_model, self.x_rnnw, del_start_symb=True, quiet=True)

            pr(self.quiet, "\tRank-dependent metrics...")
            self.perp_test_extr = scores.pautomac_perplexity(self.y_test_target, self.fix_probas(y_test_extr))
            self.kld_test_rnn_extr = scores.kullback_leibler(self.y_test_rnn, self.fix_probas(y_test_extr))
            self.kld_test_target_extr = scores.kullback_leibler(self.y_test_target, self.fix_probas(y_test_extr))
            self.ndcg1_test_rnn_extr = scores.ndcg(self.x_test, self.rnn_model, extr_aut, ndcg_l=1,
                                                   dic_ref=self.y_test_rnn_prefixes, dic_approx=self.y_test_extr_prefixes)
            self.ndcg1_test_target_extr = scores.ndcg(self.x_test, self.true_automaton, extr_aut, ndcg_l=1,
                                                      dic_ref=self.y_test_target_prefixes, dic_approx=self.y_test_extr_prefixes)
            self.ndcg1_rnnw_rnn_extr = scores.ndcg(self.x_rnnw, self.rnn_model, extr_aut, ndcg_l=1,
                                                   dic_ref=self.y_rnnw_rnn_prefixes, dic_approx=self.y_rnnw_extr_prefixes)
            self.ndcg5_test_rnn_extr = scores.ndcg(self.x_test, self.rnn_model, extr_aut, ndcg_l=5,
                                                   dic_ref=self.y_test_rnn_prefixes, dic_approx=self.y_test_extr_prefixes)
            self.ndcg5_test_target_extr = scores.ndcg(self.x_test, self.true_automaton, extr_aut, ndcg_l=5,
                                                      dic_ref=self.y_test_target_prefixes, dic_approx=self.y_test_extr_prefixes)
            self.ndcg5_rnnw_rnn_extr = scores.ndcg(self.x_rnnw, self.rnn_model, extr_aut, ndcg_l=5,
                                                   dic_ref=self.y_rnnw_rnn_prefixes, dic_approx=self.y_rnnw_extr_prefixes)
            self.perp_rand_extr = scores.pautomac_perplexity(self.y_rand_target, self.fix_probas(self.y_rand_extr))
            self.kld_rand_rnn_extr = scores.kullback_leibler(self.y_rand_rnn, self.fix_probas(self.y_rand_extr))
            self.kld_rand_extr_rnn = scores.kullback_leibler(self.y_rand_extr, self.y_rand_rnn)
            self.kld_rand_target_extr = scores.kullback_leibler(self.y_rand_target, self.fix_probas(self.y_rand_extr))
            t, e = scores.wer_aut(extr_aut, self.x_test)
            self.wer_test_extr = e/t
            t, e = scores.wer_aut(extr_aut, self.x_rnnw)
            self.wer_rnnw_extr = e/t

            self.eps_test_zeros_extr = len([x for x in y_test_extr if x <= 0.0]) / len(y_test_extr)
            self.eps_kl_rand_model_extr = neg_zero(self.y_rand_extr, self.y_rand_target)
            self.eps_rand_zeros_extr = len([x for x in self.y_rand_extr if x <= 0.0])/len(self.y_rand_extr)

            self.l2dis_target_extr = scores.l2dist(self.true_automaton, extr_aut, l2dist_method="gramian")

            self.metrics[(rank, "perp-test-extr")] = self.perp_test_extr
            self.metrics[(rank, "perp-test-extr-eps")] = self.eps_test_zeros_extr
            self.metrics[(rank, "perp-rand-extr")] = self.perp_rand_extr
            self.metrics[(rank, "perp-rand-extr-eps")] = self.eps_rand_zeros_extr
            self.metrics[(rank, "kld-test-rnn-extr")] = self.kld_test_rnn_extr
            self.metrics[(rank, "kld-test-rnn-extr-eps")] = self.eps_test_zeros_extr
            self.metrics[(rank, "kld-test-target-extr")] = self.kld_test_target_extr
            self.metrics[(rank, "kld-test-target-extr-eps")] = self.eps_test_zeros_extr
            self.metrics[(rank, "kld-rand-rnn-extr")] = self.kld_rand_rnn_extr
            self.metrics[(rank, "kld-rand-rnn-extr-eps")] = self.eps_rand_zeros_extr
            self.metrics[(rank, "kld-rand-extr-rnn")] = self.kld_rand_extr_rnn
            self.metrics[(rank, "kld-rand-target-extr")] = self.kld_rand_target_extr
            self.metrics[(rank, "kld-rand-target-extr-eps")] = self.eps_rand_zeros_extr
            self.metrics[(rank, "(1-wer)-test-extr")] = 1 - self.wer_test_extr
            self.metrics[(rank, "(1-wer)-rnnw-extr")] = 1 - self.wer_rnnw_extr
            self.metrics[(rank, "ndcg1-test-rnn-extr")] = self.ndcg1_test_rnn_extr
            self.metrics[(rank, "ndcg1-test-target-extr")] = self.ndcg1_test_target_extr
            self.metrics[(rank, "ndcg1-rnnw-rnn-extr")] = self.ndcg1_rnnw_rnn_extr
            self.metrics[(rank, "ndcg5-test-rnn-extr")] = self.ndcg5_test_rnn_extr
            self.metrics[(rank, "ndcg5-test-target-extr")] = self.ndcg5_test_target_extr
            self.metrics[(rank, "ndcg5-rnnw-rnn-extr")] = self.ndcg5_rnnw_rnn_extr
            self.metrics[(rank, "l2dis-target-extr")] = self.l2dis_target_extr

            self.print_last_extr_metrics()
        #
        return spectral_estimator

    def print_metrics_chart(self):
        measures = ["perp-test-target", "perp-test-rnn", "perp-test-extr",
                    "perp-rand-target", "perp-rand-rnn", "perp-rand-extr",
                    "kld-test-target-rnn", "kld-test-rnn-extr", "kld-test-target-extr",
                    "kld-rand-target-rnn", "kld-rand-rnn-extr", "kld-rand-extr-rnn", "kld-rand-target-extr",
                    "(1-wer)-test-target", "(1-wer)-test-rnn", "(1-wer)-test-extr",
                    "(1-wer)-rnnw-rnn", "(1-wer)-rnnw-extr",
                    "ndcg1-test-target-rnn", "ndcg1-test-rnn-extr", "ndcg1-test-target-extr", "ndcg1-rnnw-rnn-extr",
                    "ndcg5-test-target-rnn", "ndcg5-test-rnn-extr", "ndcg5-test-target-extr",
                    "ndcg5-rnnw-rnn-extr", "l2dis-target-extr"
                    ]
        mlen = max([len(m) for m in measures])+3
        width = 23
        print("+", "-"*(mlen-1), "+", ("-" * width + "+") * len(self.ranks), sep="")
        print("| RANKS :", " "*(mlen-9), end="", sep="")
        for r in self.ranks:
            print("|{1:{0}}  ".format(width-2, r), end="")
        print("|")
        print("+", "-" * (mlen - 1), "+", ("-" * width + "+") * len(self.ranks), sep="")
        for m in measures:
            print("| ", m, " "*(mlen-len("  "+m)), sep="", end="")
            for r in self.ranks:
                try :
                    v = self.metrics[(r,m)]
                except KeyError:
                    v = self.metrics[(-1, m)]
                try :
                    e = self.metrics[(r,m+"-eps")]*100
                except KeyError:
                    try:
                        e = self.metrics[(-1, m + "-eps")]*100
                    except KeyError:
                        e = -666
                print("|{1:{0}.5g}".format(width-10,v), end="")
                if e >= 0.0:
                    print(" ({0:5.2f}%) ".format(e), end="")
                else:
                    print(" "*10, end="")
            print("|")
        print("+", "-" * (mlen - 1), "+", ("-" * width + "+") * len(self.ranks), sep="")

    def print_last_extr_metrics(self):
        print("\tPerplexity on test file : ")
        print("\t\t********\tTarget :\t{0}\n"
              "\t\t********\tRNN :\t{1}\t{2:5.4f}\n"
              "\t\t({3:5.2f}%)\tExtr :\t{4}\t{5:5.4f}"
              .format(self.perp_test_target,
                      self.perp_test_rnn, (self.perp_test_target / self.perp_test_rnn),
                      100 * self.eps_test_zeros_extr, self.perp_test_extr, (self.perp_test_rnn / self.perp_test_extr)))
        print("\tPerplexity on random words : ")
        print("\t\t({0:5.2f}%)\tTarget :\t{1}\n"
              "\t\t********\tRNN :\t{2}\n"
              "\t\t({3:5.2f}%)\tExtr :\t{4}"
              .format(100 * self.eps_rand_zeros_target, self.perp_rand_target,
                      self.perp_rand_rnn,
                      100 * self.eps_rand_zeros_extr, self.perp_rand_extr))
        print("\tKL Divergence on test file : ")
        print("\t\t******** \tTarget-RNN :\t{0}\n"
              "\t\t({1:5.2f}%)\tRNN-Extr :\t{2}\n"
              "\t\t({3:5.2f}%)\tTest-Extr :\t{4}"
              .format(self.kld_test_target_rnn,
                      100 * self.eps_test_zeros_extr, self.kld_test_rnn_extr,
                      100 * self.eps_test_zeros_extr, self.kld_test_target_extr))
        print("\tKL Divergence on random words : ")
        print("\t\t********\tTarget-RNN :\t{0}\n"
              "\t\t({1:5.2f}%)\tRNN-Extr :\t{2}\n"
              "\t\t********\tExtr-RNN :\t{3}\n"
              "\t\t({4:5.2f}%)\tModel-Extr :\t{5}"
              .format(self.kld_rand_target_rnn,
                      100 * self.eps_rand_zeros_extr, self.kld_rand_rnn_extr,
                      self.kld_rand_extr_rnn,
                      100 * self.eps_kl_rand_model_extr, self.kld_rand_target_extr, ))
        print("\t(1-WER) Accuracy Rate on test file :")
        print("\t\t********\tModel :\t{0}\n"
              "\t\t********\tRNN :\t{1}\n"
              "\t\t********\tExtr :\t{2}"
              .format(1 - self.wer_test_target,
                      1 - self.wer_test_rnn,
                      1 - self.wer_test_extr))
        print("\t(1-WER) Accuracy Rate on RNN-generated words :")
        print("\t\t********\tRNN :\t{0}\n"
              "\t\t********\tExtr :\t{1}"
              .format(1 - self.wer_rnnw_rnn,
                      1 - self.wer_rnnw_extr))
        print("\tNDCG:1 on test file :")
        print("\t\t********\tModel-RNN :\t{0}\n"
              "\t\t********\tRNN-Extr :\t{1}\n"
              "\t\t********\tModel-Extr :\t{2}"
              .format(self.ndcg1_test_target_rnn,
                      self.ndcg1_test_rnn_extr,
                      self.ndcg1_test_target_extr))
        print("\tNDCG:1 on RNN-generated words :")
        print("\t\t********\tRNN-Extr :\t{0}"
              .format(self.ndcg1_rnnw_rnn_extr))
        print("\tNDCG:5 on test file :")
        print("\t\t********\tModel-RNN :\t{0}\n"
              "\t\t********\tRNN-Extr :\t{1}\n"
              "\t\t********\tModel-Extr :\t{2}"
              .format(self.ndcg5_test_target_rnn,
                      self.ndcg5_test_rnn_extr,
                      self.ndcg5_test_target_extr))
        print("\tNDCG:5 on RNN-generated words :")
        print("\t\t********\tRNN-Extr :\t{0}"
              .format(self.ndcg5_rnnw_rnn_extr))
        print("\tl2-dist :")
        print("\t\t********\tModel-Extr :\t{0}"
              .format(self.l2dis_target_extr))

    def hankels(self):
        return []

    def proba_words_normal(self, words, asdict=True, wer=False, dic=None):
        return proba_words_2(self.rnn_model, words, asdict, self.quiet, wer, prefixes_dict=dic)

    def proba_words_special(self, words, asdict=True):
        return self.proba_words_normal(words, asdict)

    def gen_words(self):
        return [], [], []

    def randwords(self, nb, minlen, maxlen):
        words = set()
        while len(words) < nb:
            le = random.randint(minlen, maxlen)
            w = ()
            for i in range(le):
                w += (random.randint(0, self.nalpha-1),)
            words.add(w)
        words = [list(w) for w in words]
        return words

    def fix_probas(self, seq, p=0.0):
        z = 0
        n = 0
        ret = np.empty(len(seq))
        for i in range(len(seq)):
            if seq[i] < p:
                ret[i] = self.epsilon
                n += 1
            elif seq[i] == p:
                ret[i] = self.epsilon
                z += 1
            else:
                ret[i] = seq[i]
        # pr(self.quiet, "(Epsilon value used {0} / {1} times ({2} neg and {3} zeros))".format(n + z, len(seq), n, z))
        return ret


# SpexHush correspond a Hank3 et Hank4, ceux qui utilisent Hush
class SpexHush(Spex):
    def __init__(self, modelfilestring, lrows, lcols, perp_train="", perp_targ="", perp_mod="", context=""):
        Spex.__init__(self, modelfilestring, lrows, lcols, perp_train, perp_targ, perp_mod, context)
        if type(lrows) is int:
            x = lrows
        else:
            x = max(lrows)
        if type(lcols) is int:
            y = lcols
        else:
            y = max(lcols)
        self.hush = Hush(x+y+1, self.nalpha)

    def gen_batch_decoded(self, word_list, batch_vol=1):
        current_v = 0
        batch = []
        for wcode in word_list:
            w = self.hush.decode(wcode)
            w = [self.nalpha + 1] + [elt + 1 for elt in w]
            w = parse.pad_0(w, self.pad)
            batch.append(w)
            current_v += 1
            if current_v == batch_vol:
                current_v = 0
                ret = np.array(batch)
                batch = []
                yield ret
        if len(batch) > 0:
            yield np.array(batch)

    def hankels(self):
        lhankels = [np.zeros((len(self.prefixes), len(self.suffixes))) for _ in range(self.nalpha + 1)]
        # EPSILON AND LETTER MATRICES :
        letters = [[]] + [[i] for i in range(self.nalpha)]
        for letter in range(len(letters)):
            for l in range(len(self.prefixes)):
                for c in range(len(self.suffixes)):
                    p = self.words_probas[self.hush.encode(self.prefixes[l] + letters[letter] + self.suffixes[c])]
                    lhankels[letter][l][c] = p
        return lhankels


# #######
# Fonctions utiles :
# #######
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


def parts_of_list(li, nparts):
    part_size = int(len(li) / nparts)+1
    parts = [li[(part_size*i):(part_size*(i+1))] for i in range(nparts)]
    return parts


def pr(quiet=False, m=""):
    if not quiet:
        print(m)
        sys.stdout.flush()


def proba_all_prefixes_rnn(model, words, quiet=False, del_start_symb=False):
    try:
        nalpha = int(model.layers[0].input_dim) - 3
    except AttributeError:
        nalpha = int(model.layers[1].input_dim) - 3
    pad = int(model.input.shape[1])
    bsize = 2048
    prefixes = set()
    for w in words:
        for i in range(len(w)+1):
            prefixes.add(tuple(w[:i]))
    prefixes = list(prefixes)
    batch = list()
    for p in prefixes:
        enc_prefix = [nalpha+1] + [x+1 for x in list(p)]
        enc_prefix = parse.pad_0(enc_prefix, pad)
        batch.append(enc_prefix)
    batch = np.array(batch)
    predictions = model.predict(batch, bsize, verbose=(0 if quiet else 1))
    if predictions.shape[1] > nalpha + 2:
        predictions = np.delete(predictions, 0, axis=1)
    if del_start_symb:
        predictions = np.delete(predictions, -2, axis=1)
    prefixes_dict = dict()
    for i in range(len(prefixes)):
        prefixes_dict[prefixes[i]] = predictions[i]
    return prefixes_dict


def proba_all_prefixes_aut(model, words):
    nalpha = model.nbL
    big_a = np.zeros((model.nbS, model.nbS))
    for t in model.transitions:
        big_a = np.add(big_a, t)
    alpha_tilda_inf = np.subtract(np.identity(model.nbS), big_a)
    alpha_tilda_inf = np.linalg.inv(alpha_tilda_inf)
    alpha_tilda_inf = np.dot(alpha_tilda_inf, model.final)
    prefixes = set()
    for w in words:
        for i in range(len(w)):
            prefixes.add(tuple(w[:i]))
    prefixes = list(prefixes)
    prefixes_dict = dict()
    for i in range(len(prefixes)):
        u = model.initial
        for l in prefixes[i]:
            u = np.dot(u, model.transitions[l])
        probas = np.empty(nalpha + 1)
        for symb in range(nalpha):
            probas[symb] = np.dot(np.dot(u, model.transitions[symb]), alpha_tilda_inf)
        probas[nalpha] = np.dot(u, model.final)
        prefixes_dict[prefixes[i]] = probas
    return prefixes_dict


def proba_next_aut(aut, prefix):
    nalpha = aut.nbL
    big_a = np.zeros((aut.nbS, aut.nbS))
    for t in aut.transitions:
        big_a = np.add(big_a, t)
    alpha_tilda_inf = np.subtract(np.identity(aut.nbS), big_a)
    alpha_tilda_inf = np.linalg.inv(alpha_tilda_inf)
    alpha_tilda_inf = np.dot(alpha_tilda_inf, aut.final)
    u = aut.initial
    for l in prefix:
        u = np.dot(u, aut.transitions[l])
    probas = np.empty(nalpha + 1)
    for symb in range(nalpha):
        probas[symb] = np.dot(np.dot(u, aut.transitions[symb]), alpha_tilda_inf)
    probas[nalpha] = np.dot(u, aut.final)
    return probas


def proba_words_2(model, words, asdict=True, quiet=False, wer=False, prefixes_dict=None):
    if prefixes_dict is None:
        prefixes_dict = proba_all_prefixes_rnn(model, words, quiet)
    # Calcul de la probabilité des mots :
    preds = np.empty(len(words))
    total = 0
    errors = 0
    pr(quiet, "\tFullwords probas...")
    # for i in range(len(words)):
    end_symb_index = len(prefixes_dict[tuple()])-1
    for i, _word in enumerate(words):
        # word = tuple([x for x in words[i]])+(nalpha+1,)
        word = _word + [end_symb_index]
        acc = 1.0
        for k in range(len(word)):
            pref = tuple(word[:k])
            proba = prefixes_dict[pref][word[k]]
            acc *= proba
            if wer:
                total += 1
                next_symb = np.argmax(prefixes_dict[pref])
                if next_symb != word[k]:
                    errors += 1
        preds[i] = acc
    # RETURN :
    # tuple_ret = tuple()
    # if asdict:
    #     probas = dict()
    #     for i in range(len(words)):
    #         probas[tuple(words[i])] = preds[i]
    #     tuple_ret += (probas,)
    # else:
    #     tuple_ret += (preds,)
    # if wer:
    #     tuple_ret += (total, errors)
    # if gen:
    #     tuple_ret += (genwords,)
    # return tuple_ret
    if asdict:  # On retourne un dictionnaire
        probas = dict()
        for i in range(len(words)):
            probas[tuple(words[i])] = preds[i]
        if wer:
            return probas, total, errors
        else:
            return probas
    else:  # On retourne une liste
        if wer:
            return preds, total, errors
        else:
            return preds


def neg_zero(seq1, seq2):
    neg = 0
    neg_cover = 0
    epsilon_used = 0
    for i in range(len(seq1)):
        if seq1[i] < 0:
            neg += 1
            if seq2[i] == 0:
                neg_cover += 1
            else:
                epsilon_used += 1
    # print("Overlap neg / zero : {0}".format(neg_cover/neg))
    return epsilon_used/len(seq1)


def gen_rnn(model, seeds=list([[]]), nb_per_seed=1, maxlen=50):
    try:
        nalpha = int(model.layers[0].input_dim) - 3
    except AttributeError:
        nalpha = int(model.layers[1].input_dim) - 3
    pad = int(model.input.shape[1])
    words = list()
    dico = dict()
    random.seed()
    for seed in seeds:
        for n in range(nb_per_seed):
            word = list(seed)
            enc_word = parse.pad_0([nalpha+1]+[elt+1 for elt in seed], pad)
            while ((len(word) == 0) or (word[-1] != nalpha+1)) and len(word) <= maxlen:
                try:
                    probas = dico[tuple(enc_word)]
                except KeyError:
                    probas = model.predict(np.array([enc_word]))
                    if probas.shape[1] > nalpha + 2:
                        # print("couic la colonne de padding !")
                        probas = np.delete(probas, 0, axis=1)
                    dico[tuple(enc_word)] = probas
                r = random.random()
                borne = probas[0][0]
                i = 0
                while r > borne:
                    i += 1
                    borne += probas[0][i]
                word += [i]
                enc_word = parse.pad_0([nalpha+1]+[elt+1 for elt in word], pad)
            if len(word) >= maxlen:
                word += [nalpha+1]
            # Il faut enlever le symbole de fin, par convention entre les fonctions les mots circulent toujours
            # sans aucun encodage : pas de symbole de fin, de début, de +1, ...
            # Si le dernier n'est pas un symbole de fin mais un autre, ben on l'enlève quand même, pas grave.
            # C'est pour ça qu'on ajoute 1 a maxlen.
            words.append(word[:-1])
    return words


def gen_rnn_forever(model, seeds=list([[]]), nb_per_seed=1, maxlen=50, patience=50, quiet=False):
    try:
        nalpha = int(model.layers[0].input_dim) - 3
    except AttributeError:
        nalpha = int(model.layers[1].input_dim) - 3
    pad = int(model.input.shape[1])
    words = set()
    dico = dict()
    random.seed()
    for seed in seeds:
        cur = len(words)
        failed_words = 0
        while (len(words)-cur) < nb_per_seed and failed_words < patience:
            word = list(seed)
            enc_word = parse.pad_0([nalpha+1]+[elt+1 for elt in seed], pad)
            while ((len(word) == 0) or (word[-1] != nalpha+1)) and len(word) <= maxlen:
                try:
                    probas = dico[tuple(enc_word)]
                except KeyError:
                    probas = model.predict(np.array([enc_word]))
                    if probas.shape[1] > nalpha + 2:
                        # print("couic la colonne de padding !")
                        probas = np.delete(probas, 0, axis=1)
                    dico[tuple(enc_word)] = probas
                r = random.random()
                borne = probas[0][0]
                i = 0
                while r > borne:
                    i += 1
                    borne += probas[0][i]
                word += [i]
                enc_word = parse.pad_0([nalpha+1]+[elt+1 for elt in word], pad)
            word = tuple(word)
            if word[-1] == nalpha+1 and word not in words:
                failed_words = 0
                #  We want to pass around non-encoded words, so we remove the ending symbol at the end
                words.add(word[:-1])
            else:
                failed_words +=1
    pr(quiet, "\t\t{0} out of {1} words generated with rnn".format(len(words), len(seeds)*nb_per_seed))
    return [list(w) for w in words]


# #######
# Fonctions Conservées pour historique :
# #######
# def combinaisons(nalpha, dim):
#     s = math.pow(nalpha, dim)
#     a = [[0]*dim]*int(s)
#     a = np.array(a)
#     p = s
#     for i in range(0, dim):
#         p /= nalpha
#         comb4(a, i, p, nalpha)
#     return a
#
#
# def proba_words(model, x_words, nalpha, asdict=True, quiet=False):
#     bsize = 512
#     pad = int(model.input.shape[1])  # On déduit de la taille de la couche d'entrée le pad nécéssaire
#     preds = np.empty(len(x_words))
#     # Il nous faut ajouter le symbole de début (nalpha) au début et le simbole de fin (nalpha+1) à la fin
#     # et ajouter 1 a chaque élément pour pouvoir utiliser le zéro comme padding,
#     if not quiet:
#         print("\tEncoding words...", end="")
#     batch_words = [([nalpha+1]+[1+elt2 for elt2 in elt]+[nalpha+2])for elt in x_words]
#     if not quiet:
#         print("\r\tEncoding OK                            ",
#               "\n\tPreparing batch :", end="")
#     nbw = len(x_words)
#     batch = []
#     for i in range(nbw):
#         word = batch_words[i]
#         # prefixes :
#         batch += [word[:j] for j in range(1, len(word))]
#         if not quiet:
#             print("\r\tPreparing batch : {0} / {1}".format(i+1, nbw), end="")
#     if not quiet:
#         print("\r\tBatch OK                                 ",
#               "\n\tPadding batch...", end="")
#     # padding :
#     batch = [parse.pad_0(elt, pad) for elt in batch]
#     if not quiet:
#         print("\r\tPadding OK                           ",
#               "\n\tPredicting batch ({0} elts)...".format(len(batch)))
#     # On restructure tout en numpy
#     batch = np.array(batch)
#     # Prédiction :
#     wpreds = model.predict(batch, bsize, verbose=(0 if quiet else 1))
#     if wpreds.shape[1] > nalpha + 2:
#         print("couic la colonne de padding !")
#         wpreds = np.delete(wpreds, 0, axis=1)
#     if not quiet:
#         print("\tPredicting OK\n\tCalculating fullwords probas:", end="")
#     offset = 0
#     for i in range(nbw):
#         word = batch_words[i]
#         acc = 1.0
#         for k in range(len(word)-1):
#             acc *= wpreds[offset][word[k+1]-1]
#             offset += 1
#         preds[i] = acc
#         if not quiet:
#             print("\r\tCalculating fullwords probas : {0} / {1}".format(i+1, nbw), end="")
#     if not quiet:
#         print("\r\tCalculating fullwords probas OK                         ")
#     if asdict:  # On retourne un dictionnaire
#         probas = dict()
#         for i in range(len(x_words)):
#             probas[tuple(x_words[i])] = preds[i]
#         return probas
#     else:  # On retourne une liste
#         return preds
#
#
# def countlen(seq, le):
#     k = 0
#     for i in range(len(seq)):
#         if len(seq[i]) > le:
#             k += 1
#     return k
#
#
# def fix_probas(seq, p=0.0, f=0.0001, quiet=False):
#     z = 0
#     n = 0
#     for i in range(len(seq)):
#         if seq[i] < p:
#             seq[i] = f
#             n += 1
#         elif seq[i] == p:
#             seq[i] = f
#             z += 1
#     if not quiet:
#         print("(Epsilon value used {0} / {1} times ({2} neg and {3} zeros))".format(n+z, len(seq), n, z))
#     return seq
