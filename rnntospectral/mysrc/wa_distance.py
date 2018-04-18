"""
Fait par Guillaume Rabusseau, je l'ai adapté pour python3
"""

import numpy as np
from time import time
from splearn.automaton import Automaton
import matplotlib.pyplot as pl


def random_WA(nbL,nbS,mu=0,sigma=0.1):
    return Automaton(nbL,nbS,
                     np.random.normal(mu,sigma,nbS),
                     np.random.normal(mu, sigma, nbS),
                     [np.random.normal(mu,sigma,[nbS,nbS]) for _ in range(nbL)])

def random_PA(nbL,nbS):
    """
    :param nbL:
    :param nbS:
    :return: a random probabilistic automaton
    """
    wa = Automaton(nbL,nbS,
                     np.random.rand(nbS),
                     np.random.rand(nbS),
                     [np.random.rand(nbS,nbS) for _ in range(nbL)])
    wa.initial /= wa.initial.sum()
    v = np.sum(wa.transitions,axis=0).sum(axis=1) + wa.final
    for i in range(nbL):
        wa.transitions[i] = (wa.transitions[i].T / v).T
    wa.final /= v
    return wa


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

def l2dist(wa1, wa2, l2dist_method='naive', sum_method='splearn'):
    """
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

def get_mean_and_std_from_list(L,N_trials):
    M = np.array(L).reshape(N_trials, int(len(L)/N_trials))
    return  M.mean(axis=0), M.std(axis=0)

if __name__ == '__main__':
    # chronos helper
    __t = 0
    def tic():
        global  __t
        __t = time()
    def toc():
        return time() - __t


    nbL = 4
    N_trials = 3

    naive_splearn = []
    naive_fast = []
    gramian = []
    nbS_list = [2,4,8,12,16,20,28,36]
    for n in range(N_trials):
        print("RUN",n)
        for nbS in nbS_list:
            print(nbS, "states")
            wa1 = random_PA(nbL,nbS)
            wa2 = random_PA(nbL,nbS)
            tic()
            d = l2dist(wa1,wa2,l2dist_method='naive',sum_method='splearn')
            t = toc()
            naive_splearn.append(t)
            print('naive_splearn:', t, "s")
            tic()
            d = l2dist(wa1,wa2,l2dist_method='naive',sum_method='fast')
            t = toc()
            naive_fast.append(t)
            print('naive_fast:   ', t, "s")
            tic()
            d = l2dist(wa1,wa2,l2dist_method='gramian')
            t = toc()
            gramian.append(t)
            print('gramian:      ', t, "s")
            print()
        print('_________________________________')
        print()

    means,stds = get_mean_and_std_from_list(naive_splearn,N_trials)
    pl.plot(nbS_list, means, '+-')
    pl.fill_between(nbS_list, means - stds, means + stds, alpha=0.1)


    means,stds = get_mean_and_std_from_list(naive_fast,N_trials)
    pl.plot(nbS_list, means, '+-')
    pl.fill_between(nbS_list, means - stds, means + stds, alpha=0.1)


    means,stds = get_mean_and_std_from_list(gramian,N_trials)
    pl.plot(nbS_list, means, '+-')
    pl.fill_between(nbS_list, means - stds, means + stds, alpha=0.1)


    pl.legend(['naive with splearn sum', 'naive with fast sum', 'gramian'])
    pl.xlabel("# states")
    pl.ylabel("running time (sec.)")
    #pl.title("Time to compute l2 distance between two random probabilistic automata")
    pl.show()
