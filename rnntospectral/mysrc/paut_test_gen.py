import sys
import splearn as sp
import random


def gen(aut_file, nbw, normalized=True):
    aut = sp.Automaton.load_Pautomac_Automaton(aut_file)
    test_set = set()
    random.seed()
    while len(test_set) < nbw:
        word = []
        state = rand_dist(aut.initial)
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
            n = rand_dist(proba)
            word += [next_trans[n][0]]
            state = next_trans[n][1]
        test_set.add(tuple(word[:-1]))
    test_list = [list(w) for w in test_set]
    test_vals = [aut.val(w) for w in test_list]
    if normalized:
        s = sum(test_vals)
        test_vals = [val/s for val in test_vals]
    return aut.nbL, test_list, test_vals


def rand_dist(dist):
    try:
        r = random.random()
        borne = dist[0]
        i = 0
        while r > borne:
            i += 1
            borne += dist[i]
        return i
    except IndexError:
        return len(dist)-1


def write_test_files(name, nbl, words, vals):
    with open(name+".pautomac.devtest", "w") as file:
        file.write(str(len(words))+" "+str(nbl))
        for w in words:
            file.write("\n"+str(len(w)))
            for s in w:
                file.write(" "+str(s))
    #
    with open(name+".pautomac.devtest_sol", "w") as file:
        file.write(str(len(words)))
        for v in vals:
            file.write("\n"+str(v))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE : dir/ect/ory/ #.model.suff")
        sys.exit(-1)
    #
    nb_prob_pauto = 48
    nb_samples = 2000
    for prob in range(nb_prob_pauto+1):
        try:
            name1 = sys.argv[1]
            name2 = sys.argv[2]
            n, t, v = gen(name1+str(prob)+name2, nb_samples, True)
            write_test_files(name1+str(prob), n, t, v)
            print(str(prob) + " done !")
        except FileNotFoundError:
            print(str(prob)+" not found ?")
