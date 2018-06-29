# External
import math
import numpy as np
from splearn import Automaton
# Project
import spex_commons


"""
Import-only file. Contains scores computations functions : WER, NDCG, KL-Div, Perplexity, Automata distance
"""


def spice_score(rankings, targets_file):
    """
    Copied from SPiCe archives. Compute NDCG5 score, using a target file provided in SPiCe data.
    :param rankings: Ranking proposals, sequence of letters
    :param targets_file: name of SPiCe target file
    :return: NDCG5 computed on the rankings, using the reference target file.
    """
    def find_proba(letter, _t):
        for j in range(len(_t)):
            if _t[j] == letter:
                return float(_t[j + 1])
        return 0

    with open(targets_file, "r") as t:
        score = 0
        nb_prefixes = 0
        i = 0
        for ts in t.readlines():
            nb_prefixes += 1
            target = ts.split()
            denominator = float(target[0])
            prefix_score = 0
            k = 1
            seen = []
            for elmnt in rankings[i]:
                if k == 1:
                    seen = [elmnt]
                    p = find_proba(str(elmnt), target)
                    prefix_score += p / math.log(k + 1, 2)
                elif elmnt not in seen:
                    p = find_proba(str(elmnt), target)
                    prefix_score += p / math.log(k + 1, 2)
                    seen = seen + [elmnt]
                k += 1
                if k > 5:
                    break
            score += prefix_score / denominator
            i += 1
        final_score = score / nb_prefixes
    return final_score


def pautomac_perplexity(y, predict_prob):
    """
    Partly copied from splearn.Spectral.score()
    :param y:
    :param predict_prob:
    :return:
    """
    s_a = sum(predict_prob)
    s_c = sum(y)
    s = 0
    for i in range(len(y)):
        try:
            s += y[i]/s_c * math.log(predict_prob[i]/s_a)
        except ValueError:
            msg = "function loss or score use log " + \
                  "function values can't be" + \
                  " negative, use it with smooth_method" + \
                  "to avoid such problem"
            raise ValueError(msg)
    perplexity = math.exp(-s)
    return perplexity


def kullback_leibler(y, predict_prob):
    s_y = 1  # sum(y)
    s_p = 1  # sum(predict_prob)
    kl = 0.0
    for i in range(len(y)):
        if y[i] > 0:
            p = y[i]/s_y
            q = predict_prob[i]/s_p
            kl += p*math.log(p/q)
    return kl


def wer_aut(aut, input_words, expected_words=None):
    if expected_words is None:
        expected_words = input_words
    big_a = np.zeros((aut.nbS, aut.nbS))
    for t in aut.transitions:
        big_a = np.add(big_a, t)
    alpha_tilda_inf = np.subtract(np.identity(aut.nbS), big_a)
    try:
        alpha_tilda_inf = np.linalg.inv(alpha_tilda_inf)
    except np.linalg.linalg.LinAlgError:
        alpha_tilda_inf = np.linalg.pinv(alpha_tilda_inf)
    alpha_tilda_inf = np.dot(alpha_tilda_inf, aut.final)
    total = 0
    errors = 0
    dico = dict()
    for k in range(len(input_words)):
        in_word = input_words[k] + [aut.nbL]
        ex_word = expected_words[k] + [aut.nbL]
        alpi = aut.initial
        for i in range(len(in_word)):
            total += 1
            key = tuple(in_word[:i])
            try:
                next_symb = dico[key]
            except KeyError:
                predsi = np.empty(aut.nbL+1)
                for symb in range(aut.nbL):
                    predsi[symb] = np.dot(np.dot(alpi, aut.transitions[symb]), alpha_tilda_inf)
                predsi[aut.nbL] = np.dot(alpi, aut.final)
                next_symb = np.argmax(predsi)
                dico[key] = next_symb
            if next_symb != ex_word[i]:
                errors += 1
            if i < len(in_word)-1:
                alpi = np.dot(alpi, aut.transitions[in_word[i]])
                # alpi = np.dot(alpi, 1/np.dot(alpi, aut.final))
    return total, errors


def best_n_args(seq, n):
    try:
        return np.argsort(seq)[::-1][:n]
    except ValueError:
        return np.argsort(seq.detach().cpu().numpy())[::-1][:n]


# Attention a ce que ncdcg_l soit <= la longueur de l'alphabet
def ndcg(words, ref, approx, ndcg_l=5, dic_ref=None, dic_approx=None):
    if dic_ref is None:
        try:
            dic_ref = spex_commons.proba_all_prefixes_rnn(ref, words)
        except AttributeError:
            dic_ref = spex_commons.proba_all_prefixes_aut(ref, words)
    if dic_approx is None:
        try:
            dic_approx = spex_commons.proba_all_prefixes_rnn(approx, words)
        except AttributeError:
            dic_approx = spex_commons.proba_all_prefixes_aut(approx, words)
    s = 0
    nbprefs = 0
    for w in words:
        for p in range(len(w)):
            pref = w[:p]
            nbprefs += 1
            a = best_n_args(dic_approx[tuple(pref)], ndcg_l)
            probas = list(dic_ref[tuple(pref)])
            p = best_n_args(probas, ndcg_l)
            top = 0
            bottom = 0
            for k in range(ndcg_l):
                log = math.log((k+2), 2)
                top += (probas[a[k]])/log
                bottom += (probas[p[k]])/log
            s += (top/bottom)
    s = s / nbprefs
    return s


def ndcg5(words, ref, approx):
    return ndcg(words, ref, approx, ndcg_l=5)


def l2dist(wa1, wa2, l2dist_method='naive', sum_method='splearn'):
    """
    L2-dist and sub-functions made by Guillaume Rabusseau, adapted to python3 by Noé.
    :param wa1: Automaton
    :param wa2: Automaton
    :param l2dist_method: can be 'naive' or 'gramian'
    :param sum_method: (only for 'naive') can be 'splearn' or 'fast' :D
    ('fast' uses linear system solver instead of inversion to compute the sum over all strings)
    :return: l2 distance between wa1 and wa2
    """
    if l2dist_method == 'naive':
        wa1.initial *= -1
        diff = wa1 + wa2
        prod = WA_product(diff,diff)
        if sum_method == 'splearn':
            dist = prod.sum()
        elif sum_method == 'fast':
            dist = WA_sum(prod)
        else:
            raise NotImplementedError("this method is not implemented (valid options are 'splearn' and 'fast')")

        wa1.initial *= -1
        return dist
    elif l2dist_method == 'gramian':
        G1 = compute_gramian(wa1)
        G2 = compute_gramian(wa2)
        G12 = compute_gramian(wa1,wa2)
        alpha1 = wa1.initial
        alpha2 = wa2.initial
        return alpha1.T.dot(G1).dot(alpha1) + alpha2.T.dot(G2).dot(alpha2) - 2*alpha1.T.dot(G12).dot(alpha2)
    else:
        raise NotImplementedError("this method is not implemented (valid options are 'naive' and 'gramian')")


def WA_product(wa1,wa2):
    """
    :param wa1: Automaton
    :param wa2: Automaton
    :return: a WA computing the product of the functions computed by wa1 and wa2
    !!! this WA will have wa1.nbS * wa2.nbS states !!!
    """
    assert wa1.nbL == wa2.nbL, "wa1 and wa2 must be on the same alphabet"
    return Automaton(wa1.nbL,
                     wa1.nbS*wa2.nbS,
                     np.kron(wa1.initial,wa2.initial),
                     np.kron(wa1.final,wa2.final),
                     [np.kron(A1,A2) for (A1,A2) in zip(wa1.transitions,wa2.transitions)])


def compute_gramian(wa1,wa2=None):
    if wa2 is None:
        wa2 = wa1
    M = np.sum([np.kron(A1,A2) for (A1,A2) in zip(wa1.transitions,wa2.transitions)],axis=0)
    I = np.eye(M.shape[0])
    g = np.linalg.solve(I-M,np.kron(wa1.final,wa2.final))
    return g.reshape(wa1.nbS,wa2.nbS)


def WA_sum(wa):
    M = np.sum(wa.transitions, axis=0)
    I = np.eye(M.shape[0])
    x = np.linalg.solve(I - M, wa.final)
    return wa.initial.dot(x)