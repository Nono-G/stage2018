import math
import numpy as np
import parse3 as parse
import spextractor_common


def find_proba(letter, target):
    for i in range(len(target)):
        if target[i] == letter:
            return float(target[i + 1])
    return 0


def spice_score(rankings, targets_file):
    # final_score = -1000
    with open(targets_file, "r") as t:
        score = 0
        nb_prefixes = 0
        i = 0
        for ts in t.readlines():
            nb_prefixes += 1
            # rs = r.readline()
            target = ts.split()
            # ranking = string.split(rs)

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
            # print(nb_prefixes, su)
            # print(rankings[i], "\t" + str(prefix_score) + "\t" + ts[:-1])
            score += prefix_score / denominator
            i += 1
        final_score = score / nb_prefixes
    return final_score


def pautomac_perplexity(y, predict_prob):
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
    alpha_tilda_inf = np.linalg.inv(alpha_tilda_inf)
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


# Attention a ce que ncdcg_l soit <= la longueur de l'alphabet
def faster_ndcg(words, ref, approx, ndcg_l=5):
    try:
        dic_ref = spextractor_common.proba_all_prefixes_rnn(ref, words, del_start_symb=True)
    except AttributeError:
        dic_ref = spextractor_common.proba_all_prefixes_aut(ref, words)
    try:
        dic_approx = spextractor_common.proba_all_prefixes_rnn(approx, words, del_start_symb=True)
    except AttributeError:
        dic_approx = spextractor_common.proba_all_prefixes_aut(approx, words)
    s = 0
    nbprefs = 0
    for w in words:
        for p in range(len(w)):
            pref = w[:p]
            nbprefs += 1
            a = parse.best_n_args(dic_approx[tuple(pref)], ndcg_l)
            probas = list(dic_ref[tuple(pref)])
            p = parse.best_n_args(probas, ndcg_l)
            top = 0
            bottom = 0
            for k in range(ndcg_l):  # 0 1 2 3 4
                log = math.log((k+2), 2)
                top += (probas[a[k]])/log
                bottom += (probas[p[k]])/log
            s += (top/bottom)
    s = s / nbprefs
    return s


def ndcg5(words, ref, approx):
    return faster_ndcg(words, ref, approx, ndcg_l=5)

# Remplacé par faster_ndcg depuis
# Attention a ce que ncdcg_l soit <= la longueur de l'alphabet
# def ndcg(words, ref, approx, ndcg_l=5):
#     dic_ref = dict()
#     dic_approx = dict()
#     s = 0
#     nbprefs = 0
#     for w in words:
#         for p in range(len(w)):
#             pref = w[:p]
#             nbprefs += 1
#             try:
#                 a = dic_approx[tuple(pref)]
#             except KeyError:
#                 a = parse.best_n_args(get_proba_m(approx, pref), ndcg_l)
#                 dic_approx[tuple(pref)] = a
#             try:
#                 probas = dic_ref[tuple(pref)]
#             except KeyError:
#                 probas = get_proba_m(ref, pref)
#                 dic_ref[tuple(pref)] = probas
#             p = parse.best_n_args(probas, ndcg_l)
#             top = 0
#             bottom = 0
#             for k in range(ndcg_l):  # 0 1 2 3 4
#                 log = math.log((k+2), 2)
#                 top += (probas[a[k]])/log
#                 bottom += (probas[p[k]])/log
#             s += (top/bottom)
#     s = s / nbprefs
#     return s


# def get_proba_m(model, prefix):
#     try:
#         nalpha = int(model.layers[0].input_dim) - 3
#         pad = int(model.input.shape[1])
#         probas = model.predict(np.array([parse.pad_0([nalpha+1]+[elt+1 for elt in prefix], pad)]))
#         if probas.shape[1] > nalpha + 2:
#             print("couic la colonne de padding !")
#             probas = np.delete(probas, 0, axis=1)
#         probas = list(probas[0])
#         del probas[-2]
#         return probas
#     except AttributeError:
#         try:
#             nalpha = model.nbL
#             big_a = np.zeros((model.nbS, model.nbS))
#             for t in model.transitions:
#                 big_a = np.add(big_a, t)
#             alpha_tilda_inf = np.subtract(np.identity(model.nbS), big_a)
#             alpha_tilda_inf = np.linalg.inv(alpha_tilda_inf)
#             alpha_tilda_inf = np.dot(alpha_tilda_inf, model.final)
#             u = model.initial
#             for l in prefix:
#                 u = np.dot(u, model.transitions[l])
#             probas = np.empty(nalpha + 1)
#             for symb in range(nalpha):
#                 probas[symb] = np.dot(np.dot(u, model.transitions[symb]), alpha_tilda_inf)
#             probas[nalpha] = np.dot(u, model.final)
#             return probas
#         except AttributeError:
#             return None
