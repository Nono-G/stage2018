import threading
import math
import random
import numpy as np
import parse3 as parse
import sys
import keras
import scores
import splearn.datasets.base as spparse
import splearn as sp


# Classes pour la parrallélisation :
class ParaRowsCols(threading.Thread):
    def __init__(self, nalpha, d,):
        threading.Thread.__init__(self)
        self.words = []
        self.d = d
        self.nalpha = nalpha

    def run(self):
        self.words = combinaisons_para(self.nalpha, self.d).tolist()


class ParaCombinaisons(threading.Thread):
    def __init__(self, a, r, p, nalpha):
        threading.Thread.__init__(self)
        self.a = a
        self.r = r
        self.p = p
        self.nalpha = nalpha

    def run(self):
        comb4(self.a, self.r, self.p, self.nalpha)


class ParaBatch(threading.Thread):
    def __init__(self, s, nalpha, pad):
        threading.Thread.__init__(self)
        self.words = s
        self.nalpha = nalpha
        self.pad = pad
        self.batch_words = []
        self.batch_prefixes = []

    def run(self):
        # Il nous faut ajouter le symbole de début (nalpha) au début et le simbole de fin (nalpha+1) à la fin
        # et ajouter 1 a chaque élément pour pouvoir utiliser le zéro comme padding,
        encoded_words = [([self.nalpha + 1] + [1 + elt2 for elt2 in elt] + [self.nalpha + 2]) for elt in self.words]
        batch = []
        nbw = len(encoded_words)
        for i in range(nbw):
            word = encoded_words[i]
            # prefixes :
            batch += [word[:j] for j in range(1, len(word))]
        # padding :
        batch = [parse.pad_0(elt, self.pad) for elt in batch]
        # tuplisation, setisation
        batch = set([tuple(elt) for elt in batch])
        self.batch_prefixes = batch
        self.batch_words = encoded_words


class ParaWords(threading.Thread):
    def __init__(self, prefixes, suffixes, letter, hush, subthreads=1, print_id=-1, quiet=True):
        threading.Thread.__init__(self)
        self.prefixes = prefixes
        self.suffixes = suffixes
        self.letter = letter
        self.words = set()
        self.subthreads = subthreads
        self.quiet = quiet
        self.id = print_id
        self.h = hush

    def run(self):
        ws = set()
        # Auto-parallelization :
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
                    w = self.h.encode(p + self.letter + s)
                    ws.add(w)
            pr(self.quiet, "\t\t\tThread letter {0}, part {1} end".format(self.letter, self.id))
        # Anyway :
        self.words = ws


class ParaBatchHush(threading.Thread):
    def __init__(self, words, hush):
        threading.Thread.__init__(self)
        self.words = words
        self.preffixes_set = set()
        self.hush = hush

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
            acc = 1.0
            for k in range(len(word)):
                acc *= self.dict[self.hush.encode(word[:k])][word[k]]
            self.preds[i] = acc


# Autres classes :
class Hush:
    def __init__(self, maxlen, base):
        self.maxlen = maxlen
        self.base = base
        self.nl = []
        self.pows = []
        self.mise_en_place()
        self.maxval = self.encode([base - 1] * maxlen)

    def mise_en_place(self):
        self.pows = [1] * (self.maxlen+1)
        for i in range(1, self.maxlen+1):
            self.pows[i] = self.pows[i-1] * self.base
        #
        self.nl = [1] * (self.maxlen+1)
        for i in range(1, self.maxlen+1):
            self.nl[i] = self.nl[i-1] + (self.pows[i])
        #

    def encode(self, w):
        if len(w) > self.maxlen:
            raise ValueError
        if len(w) == 0:
            return 0
        else:
            x = self.nl[len(w)-1]
            for i in range(len(w)):
                x += w[i]*self.pows[len(w)-i-1]
            return x

    def decode(self, n):
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


class Spex:
    def __init__(self, modelfile, lrows, lcols, perp_train="", perp_targ="", perp_model="", context=""):
        self.is_ready = False
        # Semi-constants :
        self.quiet = False
        self.epsilon = 1e-30
        self.batch_vol = 2048
        self.randwords_minlen = 0
        self.randwords_maxlen = 70
        self.randwords_nb = 10000  # Attention risque de boucle infinie si trop de mots !
        # Arguments :
        self.model = keras.models.load_model(modelfile)
        self.lrows = lrows
        self.lcols = lcols
        self.perplexity_train = perp_train
        self.perplexity_target = perp_targ
        self.perplexity_model = perp_model
        self.context = context
        # Attributes derived from arguments :
        self.nalpha = int(self.model.output.shape[1])-2
        self.pad = int(self.model.input.shape[1])
        self.perplexity_calc = (perp_train != "" and perp_targ != "" and perp_model != "")
        # Computed attributes
        self.prefixes = None
        self.suffixes = None
        self.words = None
        self.words_probas = None
        self.lhankels = None
        # Perplexity calculations attributes
        #       Test sample :
        self.x_test_rnnf = None
        self.x_test_spf = None
        self.y_test = None
        self.y_test_rnn = None
        self.test_self_perp = None
        self.test_rnn_perp = None
        self.test_rnn_kl = None
        #       Random generated words :
        self.perp_model = None
        self.x_rand = None
        self.y_rand = None
        self.y_rand_rnn = None
        self.rand_self_perp = None  # rand sample self perplexity
        self.rand_rnn_perp = None  # rnn perplexity on rand
        self.test_rnn_kl_rand = None  # kl div between model and rnn on rand sample

    def ready(self):
        if self.perplexity_calc:
            pr(self.quiet, "Perplexity preparations...")
            #  Test sample :
            self.x_test_rnnf = parse.parse_fullwords(self.perplexity_train)
            self.x_test_spf = spparse.load_data_sample(self.perplexity_train).data
            self.y_test = parse.parse_pautomac_results(self.perplexity_target)
            # self.y_test_rnnproba = self.fix_probas(self.proba_words_normal(self.x_test_rnnf, asdict=False))
            self.y_test_rnn = self.proba_words_normal(self.x_test_rnnf, asdict=False)

            self.test_self_perp = scores.pautomac_perplexity(self.y_test, self.y_test)
            self.test_rnn_perp = scores.pautomac_perplexity(self.y_test, self.y_test_rnn)
            self.test_rnn_kl = scores.kullback_leibler(self.y_test, self.y_test_rnn)
            #  Random generated words :
            self.perp_model = sp.Automaton.load_Pautomac_Automaton(self.perplexity_model)
            # dopage(self.perp_model, 1000)
            self.x_rand = self.randwords(self.randwords_nb, self.randwords_minlen, self.randwords_maxlen)
            self.y_rand = [self.perp_model.val(w) for w in self.x_rand]
            # self.y_rand_rnnproba = self.fix_probas(self.proba_words_normal(self.x_rand, asdict=False))
            self.y_rand_rnn = self.proba_words_normal(self.x_rand, asdict=False)

            self.rand_self_perp = scores.pautomac_perplexity(self.y_rand, self.fix_probas(self.y_rand))
            self.rand_rnn_perp = scores.pautomac_perplexity(self.y_rand, self.y_rand_rnn)
            self.test_rnn_kl_rand = scores.kullback_leibler(self.y_rand, self.y_rand_rnn)
            del self.x_test_rnnf
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
            pr(True, "Erreur rang trop gros pour la longueur des mots")
            return None
        pr(self.quiet, "... Done !")
        # Perplexity :
        # sp.Automaton.write(spectral_estimator.automaton, filename=("aut-{0}-r-{1}".format(self.context, rank)))
        if self.perplexity_calc:
            # Test file
            print("METRICS :")
            y_test_extr = spectral_estimator.predict(self.x_test_spf)
            test_extr_perp = scores.pautomac_perplexity(self.y_test, self.fix_probas(y_test_extr))
            rnn_extr_kl = scores.kullback_leibler(self.y_test_rnn, self.fix_probas(y_test_extr))
            test_extr_kl = scores.kullback_leibler(self.y_test, self.fix_probas(y_test_extr))

            # Random words
            # y_rand_extr = self.fix_probas(spectral_estimator.predict(self.x_rand_spf))
            y_rand_extr = [spectral_estimator.automaton.val(w) for w in self.x_rand]

            # y_rand_extr = self.fix_probas(y_rand_extr)
            rand_extr_perp = scores.pautomac_perplexity(self.y_rand, self.fix_probas(y_rand_extr))  # extr perp
            rnn_extr_kl_rand = scores.kullback_leibler(self.y_rand_rnn, self.fix_probas(y_rand_extr))  # rnn-extr-kl
            extr_rnn_kl_rand = scores.kullback_leibler(y_rand_extr, self.y_rand_rnn)  # extr-rnn-kl
            model_extr_kl_rand = scores.kullback_leibler(self.y_rand, self.fix_probas(y_rand_extr))  # model-extr kl

            eps_kl_test_modelrnn_extr = len([x for x in y_test_extr if x <= 0.0]) / len(y_test_extr)
            eps_pepr_test_target_extr = len([x for x in y_test_extr if x <= 0.0]) / len(y_test_extr)

            eps_perp_rand_selfmodel = len([x for x in self.y_rand if x <= 0.0])/len(self.y_rand)
            eps_perp_rand_model_extr = len([x for x in y_rand_extr if x <= 0.0])/len(y_rand_extr)
            eps_kl_rand_model_extr = neg_zero(y_rand_extr, self.y_rand)
            eps_kl_rand_rnn_extr = len([x for x in y_rand_extr if x <= 0.0])/len(y_rand_extr)

            print("\tPerplexity on test file : ")
            print("\t\t********\tTest :\t{0}\n"
                  "\t\t********\tRNN :\t{1}\n"
                  "\t\t({2:5.2f}%)\tExtr :\t{3}\n"
                  .format(self.test_self_perp,
                          self.test_rnn_perp,
                          100*eps_pepr_test_target_extr, test_extr_perp))
            print("\tPerplexity on random words : ")
            print("\t\t({0:5.2f}%)\tRand :\t{1}\n"
                  "\t\t********\tRNN :\t{2}\n"
                  "\t\t({3:5.2f}%)\tExtr :\t{4}\n"
                  .format(100*eps_perp_rand_selfmodel, self.rand_self_perp,
                          self.rand_rnn_perp,
                          100*eps_perp_rand_model_extr, rand_extr_perp))
            print("\tKL Divergence on test file : ")
            print("\t\t******** \tTest-RNN :\t{0}\n"
                  "\t\t({1:5.2f}%)\tRNN-Extr :\t{2}\n"
                  "\t\t({3:5.2f}%)\tTest-Extr :\t{4}\n"
                  .format(self.test_rnn_kl,
                          100*eps_kl_test_modelrnn_extr, rnn_extr_kl,
                          100*eps_kl_test_modelrnn_extr, test_extr_kl))
            print("\tKL Divergence on random words : ")
            print("\t\t********\tModel-RNN :\t{0}\n"
                  "\t\t({1:5.2f}%)\tRNN-Extr :\t{2}\n"
                  "\t\t********\tExtr-RNN :\t{3}\n"
                  "\t\t({4:5.2f}%)\tModel-Extr :\t{5}\n"
                  .format(self.test_rnn_kl_rand,
                          100*eps_kl_rand_rnn_extr, rnn_extr_kl_rand,
                          extr_rnn_kl_rand,
                          100*eps_kl_rand_model_extr, model_extr_kl_rand,))
        #
        return spectral_estimator

    def hankels(self):
        return []

    def proba_words_normal(self, words, asdict=True):
        return proba_words_para(self.model, words, self.nalpha, asdict, self.quiet)

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

    # @staticmethod
    # def splearn_code(words):
    #     le = max([len(w) for w in words])
    #     wc = [w+([-1]*(le-len(w))) for w in words]
    #     return np.array(wc)


# SpexHush correspond a Hank3 et Hank4, ceux qui utilisent Hush
class SpexHush(Spex):
    def __init__(self, modelfile, lrows, lcols, perp_train="", perp_targ="", perp_mod="", context=""):
        Spex.__init__(self, modelfile, lrows, lcols, perp_train, perp_targ, perp_mod, context)
        if type(lrows) is int:
            x = lrows
        else:
            x = max(lrows)
        if type(lcols) is int:
            y = lcols
        else:
            y = max(lcols)
        self.hush = Hush(x+y+1, self.nalpha)

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
        batch_words += threads[i].batch_words
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
        key = tuple([elt - 1 for elt in key if elt > 0])
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
            try:
                proba = prefixes_dict[pref][word[k]]
                acc *= proba
            except KeyError:
                print("gabuzomeu !", pref)
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


def fix_probas(seq, p=0.0, f=0.0001, quiet=False):
    z = 0
    n = 0
    for i in range(len(seq)):
        if seq[i] < p:
            seq[i] = f
            n += 1
        elif seq[i] == p:
            seq[i] = f
            z += 1
    if not quiet:
        print("(Epsilon value used {0} / {1} times ({2} neg and {3} zeros))".format(n+z, len(seq), n, z))
    return seq


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

def dopage(aut, coeff):
    aut.initial = np.array([coeff*elt for elt in aut.initial])
    aut.final = np.array([coeff*elt for elt in aut.final])
    aut.transitions = [np.array([coeff*elt for elt in symb]) for symb in aut.transitions]


# #######
# Fonctions Conservées pour historique :
# #######
def combinaisons(nalpha, dim):
    s = math.pow(nalpha, dim)
    a = [[0]*dim]*int(s)
    a = np.array(a)
    p = s
    for i in range(0, dim):
        p /= nalpha
        comb4(a, i, p, nalpha)
    return a


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


def countlen(seq, le):
    k = 0
    for i in range(len(seq)):
        if len(seq[i]) > le:
            k += 1
    return k
