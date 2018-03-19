import math


def find_proba(letter, target):
    for i in range(len(target)):
        if target[i] == letter:
            return float(target[i + 1])
    return 0


def spice_score(rankings, targets_file):
    final_score = -1000
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
            s = s + y[i] / s_c * math.log(predict_prob[i] / s_a)
        except ValueError:
            msg = "function loss or score use log " + \
                  "function values can't be" + \
                  " negative, use it with smooth_method" + \
                  "to avoid such problem"
            raise ValueError(msg)
    perplexity = math.exp(-s)
    return perplexity


def kullback_leibler(y, predict_prob):
    kl = 0.0
    for i in range(len(y)):
        kl += (y[i]*math.log((y[i]/predict_prob[i])))
    return kl
