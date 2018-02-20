import math


def find_proba(letter, target):
    for i in range(len(target)):
        if target[i] == letter:
            return float(target[i + 1])
    return 0


def calc_score(rankings, targets_file):
    final_score = -666.0
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
