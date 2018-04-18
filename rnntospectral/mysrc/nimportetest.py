import splearn as sp
import parse5 as parse
import numpy as np

a = sp.Automaton.load_Pautomac_Automaton("../data/pautomac/3.pautomac_model.txt")
words = parse.parse_fullwords("../data/pautomac/3.pautomac.test")
sols = parse.parse_pautomac_results("../data/pautomac/3.pautomac_solution.txt")

vals = [a.val(w) for w in words]
vals = np.dot(vals, 1/sum(vals))
for i in range(len(words)):
    r = min(vals[i], sols[i])/max(vals[i], sols[i])
    if r < 0.9999:
        print(i, r, vals[i] - sols[i])

print("5")
a = sp.Automaton.load_Pautomac_Automaton("../data/pautomac/5.pautomac_model.txt")
big = np.zeros((a.nbS, a.nbS))
for t in a.transitions:
    big = np.add(big, t)
big = np.subtract(np.identity(a.nbS),big)
i = np.linalg.pinv(big)
words = parse.parse_fullwords("../data/pautomac/5.pautomac.test")
sols = parse.parse_pautomac_results("../data/pautomac/5.pautomac_solution.txt")

vals = [a.val(w) for w in words]
vals = np.dot(vals, 1/sum(vals))
for i in range(len(words)):
    r = min(vals[i], sols[i])/max(vals[i], sols[i])
    if r < 0.9999:
        print(i, r, vals[i] - sols[i])

