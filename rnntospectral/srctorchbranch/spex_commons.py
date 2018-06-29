# Libraries
import random
import numpy as np
import sys
import splearn as sp
import os
import warnings
import torch
import torch.nn.functional as torch_funcs
from torch.autograd import Variable
# Project :
import trainer
from hush import Hush
import scores

"""
Import only file, containing abstract classes, and related classes and functions for spectral extraction
"""


class Spex:
    """
    SPctral EXtractor base abstract class.
    """
    def __init__(self, modelfilestring, lrows, lcols, m_test_set="", m_model="", context="", device="cpu"):
        self.nb_proc = len(os.sched_getaffinity(0))
        self.device = device
        # self.nb_proc = 1
        self.is_ready = False
        # Semi-constants :
        self.quiet = False
        self.epsilon = 1e-30
        self.batch_vol = 1024
        self.randwords_minlen = 0
        self.randwords_maxlen = 100
        self.randwords_nb = 1000
        # Debug Warning !
        if self.randwords_nb < 1000:
            print("DEBUG - DEBUG - DEBUG - DEBUG - DEBUG")
            print("Low random words number for debug purpose ?")
            print("DEBUG - DEBUG - DEBUG - DEBUG - DEBUG")
        self.patience = 250
        self.rand_temperature = 6  # >= 1
        # Arguments :
        self.rnn_model = trainer.load(*(modelfilestring.split()))  # pytorch OK
        self.rnn_model = self.rnn_model.to(self.device)
        self.lrows = lrows
        self.lcols = lcols
        self.metrics_test_set = m_test_set
        self.metrics_model = m_model
        self.context = context
        # Attributes derived from arguments :
        self.nalpha = self.rnn_model.nalpha
        # self.pad = int(self.rnn_model.input.shape[1])
        self.metrics_calc_level = 0
        if m_test_set != "":
            # We have access to a test set, like in SPICE and PAUTOMAC
            self.metrics_calc_level += 1
            if m_model != "":
                # We have access to a target WA, like in PAUTOMAC
                self.metrics_calc_level += 1
        # Computed attributes
        self.prefixes = None
        self.suffixes = None
        self.words = None
        self.words_probas = None
        self.lhankels = None
        self.last_extr_aut = None
        # metrics calculations attributes
        self.ranks = []
        self.true_automaton = None
        self.metrics = dict()
        self.x_test = None
        self.x_rand = None
        self.x_rnnw = None
        self.y_test_target = None
        self.y_test_rnn = None
        self.y_test_extr = None
        self.y_rand_target = None
        self.y_rand_rnn = None
        self.y_rand_extr = None
        self.y_rnnw_rnn = None
        self.y_rnnw_extr = None
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
        self.eps_rand_zeros_rnn = None
        self.eps_rand_zeros_extr = None
        self.eps_kl_rand_target_extr = None
        self.eps_kl_rand_target_rnn = None
        #
        self.perprnn_test_rnn = None
        self.perprnn_test_extr = None
        self.perprnn_rnnw_rnn = None
        self.perprnn_rnnw_extr = None
        self.eps_rnnw_zeros_extr = None

    @classmethod
    def meter(cls, rnnstring, teststring="", modelstring=""):
        """Make a pseudo-extractor, only to compute metrics"""
        return cls(rnnstring, -1, -1, teststring, modelstring)

    def ready(self):
        """Get ready to perform extractions : generate a basis, evaluate the words, fill Hankel matrices"""
        if self.metrics_calc_level > 0:
            self.rank_independent_metrics()
        # *********
        pr(0, self.quiet, "Prefixes, suffixes, words, ...")
        self.prefixes, self.suffixes, self.words = self.gen_words()
        pr(0, self.quiet, "Prediction of probabilities of words...")
        self.words_probas = self.proba_words_special(self.words, asdict=True)
        pr(0, self.quiet, "Building of hankel matrices...")
        self.lhankels = self.hankels()
        self.is_ready = True

    def rank_independent_metrics(self):
        """Metrics between RNN and target are computed only once, as they are rank independent"""
        pr(0, self.quiet, "Rank independent metrics :")
        self.x_test, _ = trainer.parse(self.metrics_test_set)
        self.x_rnnw = self.gen_with_rnn(nb=self.randwords_nb)
        self.y_test_rnn_prefixes = proba_all_prefixes_rnn(self.rnn_model, self.x_test, bsize=self.batch_vol,
                                                          quiet=self.quiet, device=self.device)

        self.y_test_rnn, t, e = self.proba_words_normal(self.x_test, asdict=False, wer=True,
                                                        prefixes_dict=self.y_test_rnn_prefixes)
        self.wer_test_rnn = e / t
        self.y_rnnw_rnn_prefixes = proba_all_prefixes_rnn(self.rnn_model, self.x_rnnw, bsize=self.batch_vol,
                                                          quiet=self.quiet, device=self.device)

        self.y_rnnw_rnn, t, e = self.proba_words_normal(self.x_rnnw, asdict=False, wer=True,
                                                        prefixes_dict=self.y_rnnw_rnn_prefixes)

        self.wer_rnnw_rnn = e / t
        #
        self.perprnn_test_rnn = scores.pautomac_perplexity(self.y_test_rnn, self.y_test_rnn)
        self.perprnn_rnnw_rnn = scores.pautomac_perplexity(self.y_rnnw_rnn, self.y_rnnw_rnn)

        if self.metrics_calc_level > 1:
            self.true_automaton = sp.Automaton.load_Pautomac_Automaton(self.metrics_model)
            self.x_rand = self.aut_rand_words(self.randwords_nb, self.rand_temperature)
            self.y_test_target = [self.true_automaton.val(w) for w in self.x_test]
            self.y_test_target_prefixes = proba_all_prefixes_aut(self.true_automaton, self.x_test)
            # noinspection PyTypeChecker
            self.y_rand_target = [self.true_automaton.val(w) for w in self.x_rand]
            self.y_rand_rnn = self.proba_words_normal(self.x_rand, asdict=False)
            t, e = scores.wer_aut(self.true_automaton, self.x_test)
            self.wer_test_target = e / t
            self.perp_test_target = scores.pautomac_perplexity(self.y_test_target, self.y_test_target)
            self.perp_test_rnn = scores.pautomac_perplexity(self.y_test_target, self.y_test_rnn)
            self.perp_rand_target = scores.pautomac_perplexity(self.y_rand_target, self.fix_probas(self.y_rand_target))
            self.perp_rand_rnn = scores.pautomac_perplexity(self.y_rand_target, self.fix_probas(self.y_rand_rnn))
            self.kld_test_target_rnn = scores.kullback_leibler(self.y_test_target, self.y_test_rnn)
            self.kld_rand_target_rnn = scores.kullback_leibler(self.y_rand_target, self.fix_probas(self.y_rand_rnn))
            self.ndcg1_test_target_rnn = scores.ndcg(self.x_test, self.true_automaton, self.rnn_model, ndcg_l=1,
                                                     dic_ref=self.y_test_target_prefixes,
                                                     dic_approx=self.y_test_rnn_prefixes)
            self.ndcg5_test_target_rnn = scores.ndcg(self.x_test, self.true_automaton, self.rnn_model, ndcg_l=5,
                                                     dic_ref=self.y_test_target_prefixes,
                                                     dic_approx=self.y_test_rnn_prefixes)
            self.eps_rand_zeros_target = len([x for x in self.y_rand_target if x <= 0.0]) / len(self.y_rand_target)
            self.eps_rand_zeros_rnn = len([x for x in self.y_rand_rnn if x <= 0.0]) / len(self.y_rand_rnn)
            self.eps_kl_rand_target_rnn = neg_zero(self.y_rand_rnn, self.y_rand_target)
        self.metrics[(-1, "perp-test-target")] = self.perp_test_target
        self.metrics[(-1, "perp-test-rnn")] = self.perp_test_rnn
        self.metrics[(-1, "perp-rand-target")] = self.perp_rand_target
        self.metrics[(-1, "perp-rand-target-eps")] = self.eps_rand_zeros_target
        self.metrics[(-1, "perp-rand-rnn")] = self.perp_rand_rnn
        self.metrics[(-1, "perp-rand-rnn-eps")] = self.eps_rand_zeros_rnn
        self.metrics[(-1, "kld-test-target-rnn")] = self.kld_test_target_rnn
        self.metrics[(-1, "kld-rand-target-rnn")] = self.kld_rand_target_rnn
        self.metrics[(-1, "kld-rand-target-rnn-eps")] = self.eps_kl_rand_target_rnn
        self.metrics[(-1, "(1-wer)-test-target")] = self.wer_test_target
        self.metrics[(-1, "(1-wer)-test-rnn")] = (1 - self.wer_test_rnn if self.wer_test_rnn is not None else None)
        self.metrics[(-1, "(1-wer)-rnnw-rnn")] = (1 - self.wer_rnnw_rnn if self.wer_rnnw_rnn is not None else None)
        self.metrics[(-1, "ndcg1-test-target-rnn")] = self.ndcg1_test_target_rnn
        self.metrics[(-1, "ndcg5-test-target-rnn")] = self.ndcg5_test_target_rnn
        #
        self.metrics[(-1, "perprnn-test-rnn")] = self.perprnn_test_rnn
        self.metrics[(-1, "perprnn-rnnw-rnn")] = self.perprnn_rnnw_rnn

    def extr(self, rank):
        """Perform an extraction of the given rank"""
        if not self.is_ready:
            self.ready()
        spectral_estimator = sp.Spectral(rank=rank, lrows=self.lrows, lcolumns=self.lrows,
                                         version='classic', partial=True, sparse=False,
                                         smooth_method='none', mode_quiet=self.quiet)
        # Les doigts dans la prise !
        pr(0, self.quiet, "Custom fit ...")
        try:
            spectral_estimator._hankel = sp.Hankel(sample_instance=None, lrows=self.lrows, lcolumns=self.lrows,
                                                   version='classic', partial=True, sparse=False,
                                                   mode_quiet=self.quiet, lhankel=self.lhankels)
            # noinspection PyProtectedMember
            spectral_estimator._automaton = spectral_estimator._hankel.to_automaton(rank, self.quiet)
            # OK on a du a peu près rattraper l'état après fit.
        except ValueError as err:
            pr(0, False, "Error, rank {0} too big compared to the length of words".format(rank))
            # print(err)
            return None
        pr(0, self.quiet, "... Done !")
        self.last_extr_aut = spectral_estimator.automaton
        sp.Automaton.write(self.last_extr_aut, filename=("aut-{0}-r-{1}".format(self.context, rank)))
        # Metrics :
        if self.metrics_calc_level > 0:
            self.rank_dependent_metrics()
            self.print_last_extr_metrics()
        #
        return spectral_estimator

    def rank_dependent_metrics(self):
        """Metrics involving the extracted automaton depend on the rank"""
        rank = self.last_extr_aut.nbS
        self.ranks.append(rank)
        print("Metrics for rank {0} :".format(rank))
        self.y_test_extr = [self.last_extr_aut.val(w) for w in self.x_test]
        self.y_rnnw_extr = [self.last_extr_aut.val(w) for w in self.x_rnnw]
        self.y_test_extr_prefixes = proba_all_prefixes_aut(self.last_extr_aut, self.x_test)
        self.y_rnnw_extr_prefixes = proba_all_prefixes_aut(self.last_extr_aut, self.x_rnnw)
        self.kld_test_rnn_extr = scores.kullback_leibler(self.y_test_rnn, self.fix_probas(self.y_test_extr))
        self.ndcg1_test_rnn_extr = scores.ndcg(self.x_test, self.rnn_model, self.last_extr_aut, ndcg_l=1,
                                               dic_ref=self.y_test_rnn_prefixes, dic_approx=self.y_test_extr_prefixes)
        self.ndcg1_rnnw_rnn_extr = scores.ndcg(self.x_rnnw, self.rnn_model, self.last_extr_aut, ndcg_l=1,
                                               dic_ref=self.y_rnnw_rnn_prefixes, dic_approx=self.y_rnnw_extr_prefixes)
        self.ndcg5_test_rnn_extr = scores.ndcg(self.x_test, self.rnn_model, self.last_extr_aut, ndcg_l=5,
                                               dic_ref=self.y_test_rnn_prefixes, dic_approx=self.y_test_extr_prefixes)
        self.ndcg5_rnnw_rnn_extr = scores.ndcg(self.x_rnnw, self.rnn_model, self.last_extr_aut, ndcg_l=5,
                                               dic_ref=self.y_rnnw_rnn_prefixes, dic_approx=self.y_rnnw_extr_prefixes)
        t, e = scores.wer_aut(self.last_extr_aut, self.x_test)
        self.wer_test_extr = e / t
        t, e = scores.wer_aut(self.last_extr_aut, self.x_rnnw)
        self.wer_rnnw_extr = e / t
        self.eps_test_zeros_extr = len([x for x in self.y_test_extr if x <= 0.0]) / len(self.y_test_extr)
        self.eps_rnnw_zeros_extr = len([x for x in self.y_rnnw_extr if x <= 0.0]) / len(self.y_rnnw_extr)
        self.perprnn_test_extr = scores.pautomac_perplexity(self.y_test_rnn, self.fix_probas(self.y_test_extr))
        self.perprnn_rnnw_extr = scores.pautomac_perplexity(self.y_rnnw_rnn, self.fix_probas(self.y_rnnw_extr))

        if self.metrics_calc_level > 1:
            self.y_rand_extr = [self.last_extr_aut.val(w) for w in self.x_rand]
            self.perp_test_extr = scores.pautomac_perplexity(self.y_test_target, self.fix_probas(self.y_test_extr))
            self.kld_test_target_extr = scores.kullback_leibler(self.y_test_target, self.fix_probas(self.y_test_extr))
            self.ndcg1_test_target_extr = scores.ndcg(self.x_test, self.true_automaton, self.last_extr_aut, ndcg_l=1,
                                                      dic_ref=self.y_test_target_prefixes,
                                                      dic_approx=self.y_test_extr_prefixes)
            self.ndcg5_test_target_extr = scores.ndcg(self.x_test, self.true_automaton, self.last_extr_aut, ndcg_l=5,
                                                      dic_ref=self.y_test_target_prefixes,
                                                      dic_approx=self.y_test_extr_prefixes)
            self.perp_rand_extr = scores.pautomac_perplexity(self.y_rand_target, self.fix_probas(self.y_rand_extr))
            self.kld_rand_rnn_extr = scores.kullback_leibler(self.fix_probas(self.y_rand_rnn),
                                                             self.fix_probas(self.y_rand_extr))
            self.kld_rand_extr_rnn = scores.kullback_leibler(self.y_rand_extr, self.fix_probas(self.y_rand_rnn))
            self.kld_rand_target_extr = scores.kullback_leibler(self.y_rand_target, self.fix_probas(self.y_rand_extr))
            self.eps_kl_rand_target_extr = neg_zero(self.y_rand_extr, self.y_rand_target)
            self.eps_rand_zeros_extr = len([x for x in self.y_rand_extr if x <= 0.0]) / len(self.y_rand_extr)
            # self.l2dis_target_extr = scores.l2dist(self.true_automaton, extr_aut, l2dist_method="gramian")

        # pr(self.quiet, "\tEvaluating words and prefixes...")
        # pr(self.quiet, "\tRank-dependent metrics...")

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
        self.metrics[(rank, "(1-wer)-test-extr")] = (1 - self.wer_test_extr if self.wer_test_extr is not None else None)
        self.metrics[(rank, "(1-wer)-rnnw-extr")] = (1 - self.wer_rnnw_extr if self.wer_rnnw_extr is not None else None)
        self.metrics[(rank, "ndcg1-test-rnn-extr")] = self.ndcg1_test_rnn_extr
        self.metrics[(rank, "ndcg1-test-target-extr")] = self.ndcg1_test_target_extr
        self.metrics[(rank, "ndcg1-rnnw-rnn-extr")] = self.ndcg1_rnnw_rnn_extr
        self.metrics[(rank, "ndcg5-test-rnn-extr")] = self.ndcg5_test_rnn_extr
        self.metrics[(rank, "ndcg5-test-target-extr")] = self.ndcg5_test_target_extr
        self.metrics[(rank, "ndcg5-rnnw-rnn-extr")] = self.ndcg5_rnnw_rnn_extr
        # self.metrics[(rank, "l2dis-target-extr")] = self.l2dis_target_extr
        self.metrics[(rank, "perprnn-test-rnn")] = self.perprnn_test_rnn
        self.metrics[(rank, "perprnn-test-extr-eps")] = self.eps_test_zeros_extr
        self.metrics[(rank, "perprnn-test-extr")] = self.perprnn_test_extr
        self.metrics[(rank, "perprnn-rnnw-rnn")] = self.perprnn_rnnw_rnn
        self.metrics[(rank, "perprnn-rnnw-extr-eps")] = self.eps_rnnw_zeros_extr
        self.metrics[(rank, "perprnn-rnnw-extr")] = self.perprnn_rnnw_extr

    def print_metrics_chart_n_max(self, n):
        """Print pretty metrics chart, with only n columns, for the human reader's convenience"""
        i = 0
        while i+n < len(self.ranks):
            self.print_metrics_chart(ranks=self.ranks[i:i+n])
            i += n
        self.print_metrics_chart(ranks=self.ranks[i:])

    def print_metrics_chart(self, ranks=None):
        """"Print (pretty) metrics chart, with as much columns as ranks, for the parser's convenience"""
        if self.metrics_calc_level > 1:
            measures = ["perp-test-target", "perp-test-rnn", "perp-test-extr",
                        "perp-rand-target", "perp-rand-rnn", "perp-rand-extr",
                        "kld-test-target-rnn", "kld-test-rnn-extr", "kld-test-target-extr",
                        "kld-rand-target-rnn", "kld-rand-rnn-extr", "kld-rand-extr-rnn", "kld-rand-target-extr",
                        "(1-wer)-test-target", "(1-wer)-test-rnn", "(1-wer)-test-extr",
                        "(1-wer)-rnnw-rnn", "(1-wer)-rnnw-extr",
                        "ndcg1-test-target-rnn", "ndcg1-test-rnn-extr", "ndcg1-test-target-extr", "ndcg1-rnnw-rnn-extr",
                        "ndcg5-test-target-rnn", "ndcg5-test-rnn-extr", "ndcg5-test-target-extr",
                        "ndcg5-rnnw-rnn-extr",  # "l2dis-target-extr",
                        "perprnn-test-rnn", "perprnn-test-extr", "perprnn-rnnw-rnn", "perprnn-rnnw-extr"
                        ]
        else:
            measures = ["kld-test-rnn-extr",
                        "(1-wer)-test-rnn", "(1-wer)-test-extr",
                        "(1-wer)-rnnw-rnn", "(1-wer)-rnnw-extr",
                        "ndcg1-test-rnn-extr", "ndcg1-rnnw-rnn-extr",
                        "ndcg5-test-rnn-extr",
                        "ndcg5-rnnw-rnn-extr",
                        "perprnn-test-rnn", "perprnn-test-extr", "perprnn-rnnw-rnn", "perprnn-rnnw-extr"
                        ]
        if ranks is None:
            ranks = self.ranks
        ranks = sorted(ranks)
        mlen = max([len(m) for m in measures])+3
        width = 23
        print(self.context)
        print("+", "-"*(mlen-1), "+", ("-" * width + "+") * len(ranks), sep="")
        print("| RANKS :", " "*(mlen-9), end="", sep="")
        for r in ranks:
            print("|{1:{0}}  ".format(width-2, r), end="")
        print("|")
        print("+", "-" * (mlen - 1), "+", ("-" * width + "+") * len(ranks), sep="")
        for m in measures:
            print("| ", m, " "*(mlen-len("  "+m)), sep="", end="")
            for r in ranks:
                try:
                    v = self.metrics[(r,m)]
                except KeyError:
                    v = self.metrics[(-1, m)]
                try:
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
        print("+", "-" * (mlen - 1), "+", ("-" * width + "+") * len(ranks), sep="")

    def print_last_extr_metrics(self):
        """Print the metrics relative to the last extraction performed"""
        if self.metrics_calc_level > 1:
            print("\tPerplexity on test file : ")
            print("\t\t********\tTarget :\t{0}\n"
                  "\t\t********\tRNN :\t{1}\t{2:5.4f}\n"
                  "\t\t({3:5.2f}%)\tExtr :\t{4}\t{5:5.4f}"
                  .format(self.perp_test_target,
                          self.perp_test_rnn, (self.perp_test_target / self.perp_test_rnn),
                          100*self.eps_test_zeros_extr, self.perp_test_extr, (self.perp_test_rnn/self.perp_test_extr)))
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
                          100 * self.eps_kl_rand_target_extr, self.kld_rand_target_extr, ))
            print("\t(1-WER) Accuracy Rate on test file :")
            print("\t\t********\tTarget :\t{0}\n"
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
            print("\t\t********\tTarget-RNN :\t{0}\n"
                  "\t\t********\tRNN-Extr :\t{1}\n"
                  "\t\t********\tModel-Extr :\t{2}"
                  .format(self.ndcg1_test_target_rnn,
                          self.ndcg1_test_rnn_extr,
                          self.ndcg1_test_target_extr))
            print("\tNDCG:1 on RNN-generated words :")
            print("\t\t********\tRNN-Extr :\t{0}"
                  .format(self.ndcg1_rnnw_rnn_extr))
            print("\tNDCG:5 on test file :")
            print("\t\t********\tTarget-RNN :\t{0}\n"
                  "\t\t********\tRNN-Extr :\t{1}\n"
                  "\t\t********\tModel-Extr :\t{2}"
                  .format(self.ndcg5_test_target_rnn,
                          self.ndcg5_test_rnn_extr,
                          self.ndcg5_test_target_extr))
            print("\tNDCG:5 on RNN-generated words :")
            print("\t\t********\tRNN-Extr :\t{0}"
                  .format(self.ndcg5_rnnw_rnn_extr))
            # print("\tl2-dist :")
            # print("\t\t********\tTarget-Extr :\t{0}"
            #       .format(self.l2dis_target_extr))
        else:
            print("\tKL Divergence on test file : ")
            print("\t\t({0:5.2f}%)\tRNN-Extr :\t{1}\n"
                  .format(100 * self.eps_test_zeros_extr, self.kld_test_rnn_extr))
            print("\t(1-WER) Accuracy Rate on test file :")
            print("\t\t********\tRNN :\t{0}\n"
                  "\t\t********\tExtr :\t{1}"
                  .format(1 - self.wer_test_rnn,
                          1 - self.wer_test_extr))
            print("\t(1-WER) Accuracy Rate on RNN-generated words :")
            print("\t\t********\tRNN :\t{0}\n"
                  "\t\t********\tExtr :\t{1}"
                  .format(1 - self.wer_rnnw_rnn,
                          1 - self.wer_rnnw_extr))
            print("\tNDCG:1 on test file :")
            print("\t\t********\tRNN-Extr :\t{0}\n"
                  .format(self.ndcg1_test_rnn_extr))
            print("\tNDCG:1 on RNN-generated words :")
            print("\t\t********\tRNN-Extr :\t{0}"
                  .format(self.ndcg1_rnnw_rnn_extr))
            print("\tNDCG:5 on test file :")
            print("\t\t********\tRNN-Extr :\t{0}\n"
                  .format(self.ndcg5_test_rnn_extr))
            print("\tNDCG:5 on RNN-generated words :")
            print("\t\t********\tRNN-Extr :\t{0}"
                  .format(self.ndcg5_rnnw_rnn_extr))
            # print("\tl2-dist :")
            # print("\t\t********\tTarget-Extr :\t{0}"
            #       .format(self.l2dis_target_extr))
            print("\tPerplexity with RNN as reference on test file : ")
            print("\t\t********\tRNN :\t{0}\n"
                  "\t\t({1:5.2f}%)\tExtr :\t{2}\t{3:5.4f}"
                  .format(self.perprnn_test_rnn,
                          100 * self.eps_test_zeros_extr, self.perprnn_test_extr,
                          (self.perprnn_test_rnn / self.perprnn_test_extr)))
            print("\tPerplexity with RNN as reference on RNN-generated words : ")
            print("\t\t********\tRNN :\t{0}\n"
                  "\t\t({1:5.2f}%)\tExtr :\t{2}\t{3:5.4f}"
                  .format(self.perprnn_rnnw_rnn,
                          100 * self.eps_rnnw_zeros_extr, self.perprnn_rnnw_extr,
                          (self.perprnn_rnnw_rnn / self.perprnn_rnnw_extr)))

    def hankels(self):
        """Unimplemented placeholder, must be redefined in classes inheriting from this one"""
        return []

    def proba_words_normal(self, words, asdict=True, wer=False, prefixes_dict=None):
        """Computes the probability of each word in words according to the RNN
        This version does not use any optimization and can be used only for samll sets of words (test sets)
        WER can be computed during the process"""
        if prefixes_dict is None:
            prefixes_dict = proba_all_prefixes_rnn(self.rnn_model, words, quiet=self.quiet,
                                                   bsize=self.batch_vol, device=self.device)
        # Calcul de la probabilité des mots :
        preds = np.empty(len(words))
        total = 0
        errors = 0
        pr(1, self.quiet, "Fullwords probas...")
        start_factor = 1
        # start_factor = len(words) / sum([len(w) for w in words])
        for i, word in enumerate(words):
            indices = [x + 1 for x in word]
            indices.append(0)
            acc = start_factor
            for k in range(len(indices)):
                pref = tuple(word[:k])
                proba = prefixes_dict[pref][indices[k]]
                acc *= proba
                if wer:
                    total += 1
                    next_symb = np.argmax(prefixes_dict[pref])
                    if next_symb != indices[k]:
                        errors += 1
            preds[i] = acc
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

    def proba_words_special(self, words, asdict=True):
        """Unimplemented placeholder for optimized version of proba_words. Does not have to be able to compute WER"""
        return self.proba_words_normal(words, asdict)

    def gen_words(self):
        """Unimplemented placeholder, the choice of the basis generation method will be made here"""
        return [], [], []

    def randwords(self, nb, minlen, maxlen):
        """Produces uniformly random words"""
        words = set()
        while len(words) < nb:
            le = random.randint(minlen, maxlen)
            w = ()
            for i in range(le):
                w += (random.randint(0, self.nalpha-1),)
            words.add(w)
        words = [list(w) for w in words]
        return words

    def aut_rand_words(self, nbw, temperature):
        """Produces random words following the distribution defined by the model automaton
        This distribution can be tweaked using the temperature parameter to obtain more or less likely words"""
        aut = self.true_automaton
        words_set = set()
        random.seed()
        while len(words_set) < nbw:
            word = []
            state = temp_dist_rand(aut.initial, temperature)
            while state != -1:
                next_trans = dict()
                proba = list()
                next_trans[0] = (-1, -1)
                proba.append(aut.final[state])
                i = 1
                for l in range(len(aut.transitions)):
                    for s in range(aut.nbS):
                        next_trans[i] = (l, s)
                        proba.append(aut.transitions[l][state][s])
                        i += 1
                n = temp_dist_rand(proba, temperature)
                word += [next_trans[n][0]]
                state = next_trans[n][1]
            words_set.add(tuple(word[:-1]))
        words_list = [list(w) for w in words_set]
        words_vals = [aut.val(w) for w in words_list]
        pr(2, self.quiet, "Target-temp-rand words : average p = {0}".format(sum(words_vals)/nbw))
        return words_list  # , test_vals

    def fix_probas(self, seq, p=0.0):
        """Replace unwanted values by epsilon."""
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

    def gen_with_rnn(self, nb, temperature=1):
        """Use the RNN to generate a set of unique words, following the probability distribution it defines"""
        with torch.no_grad():
            self.rnn_model.eval()
            genw_set = set()
            failures = 0
            while len(genw_set) < nb and failures < self.patience:
                gen_word, selected = self.gen_one_with_rnn(temperature)
                gen_word = tuple(gen_word)
                # noinspection PyUnboundLocalVariable
                if selected == 0 and gen_word not in genw_set:
                    failures = 0
                    genw_set.add(gen_word)
                else:
                    failures += 1
            if failures >= self.patience:
                pr(1, self.quiet, "Stopping words generation : out of patience")
            pr(1, self.quiet, "{0} out of {1} words generated with rnn".format(len(genw_set), self.randwords_nb))
            return [list(w) for w in genw_set]

    def gen_one_with_rnn(self, temperature=1):
        """Use the RNN to generate one word"""
        gen_word = []
        x = torch.tensor([[self.rnn_model.nalpha + 1]], dtype=torch.int64, device=self.device)
        # size for hidden: (num_layers * num_directions, batch, hidden_size)
        hidden = torch.zeros((self.rnn_model.num_layers, 1, self.rnn_model.hidn_size), device=self.device)
        for i in range(self.randwords_maxlen):
            y_scores, hidden = self.rnn_model(x, hidden)
            dist = torch_funcs.softmax((y_scores[0, 0, :-1] / temperature), dim=-1)
            selected = torch.multinomial(dist, 1).item()
            if selected == 0:
                break
            gen_word.append(selected - 1)
            x[0, 0] = selected
        return gen_word, selected


class SpexHush(Spex):
    """Abstract class for methods using Hush encoding (3 and 5, i.e. all supported methods )"""
    def __init__(self, modelfilestring, lrows, lcols, m_test_set="", perp_mod="", context="", device="cpu"):
        Spex.__init__(self, modelfilestring, lrows, lcols, m_test_set, perp_mod, context, device)
        self.hush = Hush(2*self.randwords_maxlen+5, self.nalpha)

    def hankels(self):
        """Redefinition of hankels(), using words_probas dictionaries with hushed keys"""
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

def temp_dist_rand(dist, temp):
    """Randomly pick among a distribution, with 'temperature' modification"""
    # initial author : Benoit Favre; modified
    with warnings.catch_warnings():
        # It's ok to suppress warning since log(0) = -inf and exp(-inf) = 0, works fine
        warnings.simplefilter("ignore")
        #
        ep = 10e-10
        probas = [p+ep for p in dist]
        probas = np.log(probas) / temp
        e = np.exp(probas)
        probas = e / np.sum(e)
        return np.random.choice(len(probas), 1, p=probas)[0]


def pr(indent=0, quiet=False, m="", end="\n"):
    """Conditional print, with variable indentation"""
    if not quiet:
        for i in range(indent):
            print("\t", end="")
        print(m, end=end)
        sys.stdout.flush()


def proba_all_prefixes_rnn(model, words, bsize=512, quiet=False, device="cpu"):
    """Returns a dict containing next symbol probas after every prefix of every word given, using an rnn model"""
    predictions = model.probas_tables_numpy(words, bsize, quiet=quiet, device=device)
    preds_dict = dict()
    for i, w in enumerate(words):
        for k in range(len(w)+1):
            preds_dict[tuple(w[:k])] = predictions[i,k]
    return preds_dict


def proba_all_prefixes_aut(model, words, end_symbol_first=True):
    """Returns a dict containing next symbol probas after every prefix of every word given, using an automaton model"""
    nalpha = model.nbL
    big_a = np.zeros((model.nbS, model.nbS))
    for t in model.transitions:
        big_a = np.add(big_a, t)
    alpha_tilda_inf = np.subtract(np.identity(model.nbS), big_a)
    try:
        alpha_tilda_inf = np.linalg.inv(alpha_tilda_inf)
    except np.linalg.linalg.LinAlgError:
        alpha_tilda_inf = np.linalg.pinv(alpha_tilda_inf)
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
        if end_symbol_first is True:
            probas[0] = np.dot(u, model.final)
            for symb in range(nalpha):
                probas[symb+1] = np.dot(np.dot(u, model.transitions[symb]), alpha_tilda_inf)
        else:
            for symb in range(nalpha):
                probas[symb] = np.dot(np.dot(u, model.transitions[symb]), alpha_tilda_inf)
            probas[nalpha] = np.dot(u, model.final)
        prefixes_dict[prefixes[i]] = probas
    return prefixes_dict


def proba_next_aut(aut, prefix, end_symbol_first=True):
    """For a given automaton and a given prefix, output the probability distribution over symbols for the next symbol"""
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
    if end_symbol_first is True:
        probas[0] = np.dot(u, aut.final)
        for symb in range(nalpha):
            probas[symb + 1] = np.dot(np.dot(u, aut.transitions[symb]), alpha_tilda_inf)
    else:
        for symb in range(nalpha):
            probas[symb] = np.dot(np.dot(u, aut.transitions[symb]), alpha_tilda_inf)
        probas[nalpha] = np.dot(u, aut.final)
    probas[nalpha] = np.dot(u, aut.final)
    return probas


def neg_zero(seq1, seq2):
    """Counts how many times a negative value appears in seq1 and seq2 contains a non-zero value at same index
    Output the epsilon use rate (negatives which are not covered by a zero, divided by total length)"""
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
    return epsilon_used/len(seq1)
