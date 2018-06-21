# External :
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import time
import math
import random
# Project :
from models import LangMod

# Give seeds to RNG for reproducibility
torch.manual_seed(666)
random.seed(666)


def perf(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = num = 0
    with torch.no_grad():
        for x, y in loader:
            y_scores, _ = model(x)
            loss = criterion(y_scores.view(y.size(0) * y.size(1), -1), (y.view(y.size(0) * y.size(1))))
            total_loss += loss.item()
            num += len(y)
    return total_loss / num, math.exp(total_loss / num)


def fit(model, epochs, train_loader, valid_loader):
    ti = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(epochs):
        model.train()
        total_loss = num = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            y_scores, _ = model(x)
            loss = criterion(y_scores.view(y.size(0) * y.size(1), -1), (y.view(y.size(0) * y.size(1))))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num += len(y)
        print(epoch, total_loss / num, *perf(model, valid_loader))
    tf = time.time()
    t = tf-ti
    print("Total time : {0}, Average : {1}".format(t, t/epochs))


def generate_probable(model, nb=1, temperature=0.5):
    ti = time.time()
    with torch.no_grad():
        model.eval()
        ret = []
        for n in range(nb):
            gw = []
            x = Variable(torch.zeros((1, 1)).long())
            x[0, 0] = model.nalpha+1
            # size for hidden: (num_layers * num_directions, batch, hidden_size)
            hidden = Variable(torch.zeros(2, 1, model.hidn_size))
            for i in range(200):
                y_scores, hidden = model(x, hidden)
                dist = F.softmax(y_scores/temperature, dim=-1)[0][0]
                y_pred = torch.multinomial(dist, 1)
                # y_pred = torch.max(y_scores, 2)[1]
                selected = y_pred.data.item()
                if selected == 0:
                    break
                # print(rev_vocab[selected], end='')
                gw .append(selected)
                x[0, 0] = selected
            ret.append(gw)
    tf = time.time()
    t = tf - ti
    print("Total : {0}, Average : {1}".format(t, t/nb))
    return ret


def parse(filestring):
    with open(filestring, 'r') as fp:
        ws = []
        line = fp.readline()
        nb, nalpha = [int(x) for x in line.split()]
        for line in fp:
            ws.append([int(x) for x in line.split()[1:]])
        return ws, nalpha, nb


def tensorize(ws, nalpha, nb_sample=-1):
    len_cover = 0.9
    # le = max([len(w) for w in ws])
    le = sorted([len(w) for w in ws])[int(len(ws)*len_cover)]
    X = torch.zeros(len(ws), le).long()
    Y = torch.zeros(len(ws), le).long()
    if nb_sample == -1:
        _nb_sample = len(ws)
    else:
        _nb_sample = min(len(ws), nb_sample)
    for i, w in enumerate(random.sample(ws, _nb_sample)):
        enc_w = [nalpha+1]+[x+1 for x in w]+[0]
        ilen = min(len(enc_w)-1, le)
        X[i, :ilen] = torch.tensor(enc_w[:ilen])
        Y[i, :ilen] = torch.tensor(enc_w[1:ilen+1])
    # print(nalpha, nb)
    # print(ws)
    # print(X.numpy())
    # print(Y.numpy())
    return X, Y


def load(filestring, nalpha, hidn_size=64):
    model = LangMod(nalpha, 16, hidn_size)
    model.load_state_dict(torch.load(filestring))
    return model


def train(trainfilestring, testfilestring):
    batch_size = 16
    epochs = 10
    hidden_size = 64
    #
    wst, nalphat, nbt = parse(trainfilestring)
    Xt, Yt = tensorize(wst, nalphat)
    train_loader = DataLoader(TensorDataset(Xt, Yt), batch_size=batch_size, shuffle=True)
    wsv, nalphav, nbv = parse(testfilestring)
    Xv, Yv = tensorize(wsv, nalphav)
    valid_loader = DataLoader(TensorDataset(Xv, Yv), batch_size=batch_size)
    model = LangMod(nalphat, hidden_size)
    fit(model, epochs, train_loader, valid_loader)
    torch.save(model.state_dict(), "../nonomodel")
    return model


def fullprobas(model, ws):
    print(model.probas_words(ws))


def main():
    # dico = movies_to_pautomac(2000)
    # rev_dico = rev_vocab = {y: x for x, y in dico.items()}
    # model = train("../../data/pautomac/3.pautomac.train", "../../data/pautomac/3.pautomac.test")
    # model = train("../pautomac-movies", "../pautomac-movies-test")
    model = load("../nonomodel", 4)
    # model = load("../nonomodel", 82)
    gens = generate_probable(model, 150, 0.7)
    # dec_gens = ["".join([rev_dico[c-1] for c in gen]) for gen in gens]
    dec_gens = [[x-1 for x in gen] for gen in gens]
    print(*dec_gens, sep="\n")
    avlen = sum([len(w) for w in gens])/len(gens)
    print(avlen)
    fullprobas(model, [[3,3,1,2], [3,0,3], [1,0,2,0,1,1,2,1,1]])


def movies_to_pautomac(qtest):
    with open("../movies-sf.txt", "r") as file:
        d = dict()
        next = 0
        nbl = 0
        for line in file:
            nbl += 1
            for char in line[:-1]:
                if not char in d.keys():
                    d[char] = next
                    next += 1
    with open("../movies-sf.txt", "r") as file:
        with open("../pautomac-movies", "w") as outp:
            with open("../pautomac-movies-test", "w") as outptest:
                outptest.write("{0} {1}\n".format(qtest, next))
                i = 0
                for line in file:
                    outptest.write("{0}".format(len(line)))
                    for char in line[:-1]:
                        outptest.write(" {0}".format(d[char]))
                    outptest.write("\n")
                    i += 1
                    if i >= qtest:
                        break
                outp.write("{0} {1}\n".format(nbl-qtest, next))
                for line in file:
                    outp.write("{0}".format(len(line)))
                    for char in line[:-1]:
                        outp.write(" {0}".format(d[char]))
                    outp.write("\n")
    # with open("../movies-dic", "w") as outp:
    #     for k in d.keys():
    #         outp.write(k+" {0}\n".format(d[k]))
    return d


if __name__ == "__main__":
    main()
