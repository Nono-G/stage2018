# External :
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import sys
import random
# Project :
import models

# TODO Unseed when in production
# Give seeds to RNG for reproducibility
torch.manual_seed(666)
random.seed(666)

"""
Runnable program to train models, and related classes and functions
"""


def save_weights(model, basename, epochid):
    torch.save(model.state_dict(), basename + ("-e{0}".format(epochid)))


def save_digest(model, basename):
    with open(basename+"-d", "w") as file:
        file.write(str(model.nalpha))
        file.write(str(model.hidn_size))
        file.write("\n")


def load(digfile, dictfile):
    with open(digfile, "r") as dig:
        model = models.LangMod(*([int(x) for x in dig.readline().split()]))
    model.load_state_dict(dictfile)
    return model


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
    return total_loss / num  # , math.exp(total_loss / num)


def fit(model, epochs, train_loader, valid_loader=None, save_each_epoch=False):
    ti = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    val_losses = []
    mini = None
    for epoch in range(epochs):
        model.train()  # Train mode, activate dropout and such
        total_loss = num = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            y_scores, _ = model(x)
            loss = criterion(y_scores.view(y.size(0) * y.size(1), -1), (y.view(y.size(0) * y.size(1))))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num += len(y)
        if valid_loader is not None:
            val_loss = perf(model, valid_loader)
            val_losses.append(val_loss)
            if mini is None or val_losses[mini] > val_loss:
                mini = epoch
            print("\rEpoch {0} :: test loss = {1} validation loss = {2}".format(epoch+1, total_loss / num, val_loss))
            print("Current best is at epoch {0} :: {1} on validation".format(mini+1, val_losses[mini]), end="")
            sys.stdout.flush()
        else:
            print("Epoch {0} :: test loss = {1}".format(epoch+1, total_loss / num))
        if save_each_epoch:
            save_weights(model, model.modelname, epoch+1)
    tf = time.time()
    t = tf-ti
    print("\nTotal training time : {0}, Average per epoch : {1}".format(t, t/epochs))


def parse(filestring):
    with open(filestring, 'r') as fp:
        ws = []
        line = fp.readline()
        nb, nalpha = [int(x) for x in line.split()]
        for line in fp:
            ws.append([int(x) for x in line.split()[1:]])
        return ws, nalpha  # , nb


# noinspection PyCallingNonCallable,PyUnresolvedReferences
def enc_tensorize(ws, nalpha, nb_sample=-1, shuffle=False):
    len_cover = 0.9
    le = sorted([len(w) for w in ws])[int(len(ws)*len_cover)]
    x = torch.zeros(len(ws), le).long()
    y = torch.zeros(len(ws), le).long()
    if nb_sample == -1:
        _nb_sample = len(ws)
    else:
        _nb_sample = min(len(ws), nb_sample)
    if shuffle:
        for i, w in enumerate(random.sample(ws, _nb_sample)):
            enc_w = [nalpha + 1] + [x + 1 for x in w] + [0]
            ilen = min(len(enc_w) - 1, le)
            x[i, :ilen] = torch.tensor(enc_w[:ilen])
            y[i, :ilen] = torch.tensor(enc_w[1:ilen + 1])
    else:
        for i, w in enumerate(ws[:_nb_sample]):
            enc_w = [nalpha + 1] + [x + 1 for x in w] + [0]
            ilen = min(len(enc_w) - 1, le)
            x[i, :ilen] = torch.tensor(enc_w[:ilen])
            y[i, :ilen] = torch.tensor(enc_w[1:ilen + 1])
    return x, y


def trainf(modelname, train_file, sample, neurons, epochs, batch, test_file=None):
    wst, nalpha = parse(train_file)
    do_test = test_file is not None
    if do_test:
        wsv, nalphav = parse(test_file)
        if nalpha != nalphav:
            raise ValueError("Training and Validation sets have different alphabets.")
        xv,yv = enc_tensorize(wsv, nalpha)
        valid_loader = DataLoader(TensorDataset(xv,yv))
    else:
        valid_loader = None
    xt, yt = enc_tensorize(wst, nalpha, sample, shuffle=True)
    train_loader = DataLoader(TensorDataset(xt, yt), batch_size=batch)

    model = models.LangMod(modelname, nalpha, neurons)
    save_digest(model, modelname)
    fit(model, epochs, train_loader, valid_loader, save_each_epoch=True)
    return model


def main():
    if len(sys.argv) < 5 or len(sys.argv) > 8:
        sys.exit("ARGS : train_file neurons epochs batch [model_name] [sampleNB] [test_file]")

    # ARGS :
    trainfile = sys.argv[1]
    neurons = int(sys.argv[2])
    epochs = int(sys.argv[3])
    batch = int(sys.argv[4])
    name = ""
    if len(sys.argv) > 5:
        name = sys.argv[5]
    if len(sys.argv) > 6:
        samplenb = int(sys.argv[6])
    else:
        samplenb = -1
    if len(sys.argv) > 7:
        testfile = sys.argv[7]
    else:
        testfile = None

    modelname = ("model-{0}-T{1}N{2}E{3}B{4}S{5}"
                 .format(name, trainfile, neurons, epochs, batch, samplenb)
                 .replace(" ", "_")
                 .replace("/", "+"))
    print(modelname)
    _ = trainf(modelname, trainfile, samplenb, neurons, epochs, batch, testfile)


if __name__ == "__main__":
    main()
