import math
import sys
import parse3 as parse
import numpy as np
import keras
import splearn as sp


def words(nalpha, lrows, lcols):
    if not type(lrows) is list:
        if not type(lrows) is int:
            raise TypeError()
        else:
            lrows = [i for i in range(lrows+1)]
    if not type(lcols) is list:
        if not type(lcols) is int:
            raise TypeError()
        else:
            lcols = [i for i in range(lcols + 1)]
    return words_indexes_as_lists(nalpha, lrows, lcols)


def words_indexes_as_lists(nalpha, lrows, lcols):
    lig = []
    col = []
    for d in lrows:
        lig += combinaisons(nalpha, d).tolist()
    for d in lcols:
        col += combinaisons(nalpha, d).tolist()
    nlig = len(lig)
    ncol = len(col)
    w=[]
    # a = np.array(a)
    for l in range(0, nlig):
        w.append([])
        for c in range(0, ncol):
            # a[l][c] = np.concatenate((lig[l], col[c]))
            w[l].append(lig[l]+col[c])
    return w


def combinaisons(nalpha, dim):
    s = math.pow(nalpha, dim)
    a = [[0]*dim]*int(s)
    a = np.array(a)
    p = s
    for i in range(0, dim):
        p /= nalpha
        comb3(a, i, p, nalpha)
    return a

# def comb2(a, r, s, e, nalpha):
#     w = int((e-s)/nalpha)
#     pos = s
#     for letter in range(0, nalpha):
#         for i in range(pos, pos+w):
#             a[i][r] = letter
#         pos += w


def comb3(a, r, p, nalpha):
    for i in range(0, len(a)):
        a[i][r] = (i // p) % nalpha


def traduct(n):
    return "abcdefghijklmnopqrstuvwxyz"[n]


def evaluate(modelfile, hankelwords, pad, nalpha):
    model = keras.models.load_model(modelfile)
    hshape = (len(hankelwords), len(hankelwords[0]))
    hankel = np.empty(hshape)
    for i in range(0, len(hankelwords)):
        # Soit la i eme ligne de la matrice de hankel contenant les mots.
        batch = hankelwords[i]
        # Il nous faut ajouter le symbole de début (nalpha) au début et ajouter 1 a chaque élément pour pouvoir utiliser
        # le zéro comme padding.
        batch = [([nalpha+1]+[1+elt2 for elt2 in elt])for elt in batch]
        # Il nous faut bourrer la séquence pour la mettre a la longueur attendue par le model
        batch = [parse.pad_0(elt, pad) for elt in batch]
        # On restructure tout en numpy
        batch = np.array(batch)
        # On fait prédire par le modèle :
        preds = model.predict(batch, len(batch))
        # On ne conserve que les probabilités du caractère de fin de séquence : (nalpha+1)
        preds = np.array([elt[nalpha+1] for elt in preds])
        hankel[i] = preds
    return hankel


def nppad0(arr, s):
    x = s-len(arr)
    return np.concatenate((np.array([0]*x), arr))


if len(sys.argv) != 4:
    print("Usage :: {0} modelfile pad nalpha".format(sys.argv[0]))
    sys.exit(-666)

if __name__ == "__main__":
    print("Words...")
    hw = words(2, [1], 2)
    print("OK")
    print("Hankel Matrix...")
    h = evaluate(sys.argv[1], hw, int(sys.argv[2]), int(sys.argv[3]))
    print("OK")
    # spectr = sp.Spectral()
