import time
import torch
from torch import nn as nn
from torch.nn import functional as tfunc


class LangMod(nn.Module):
    def __init__(self, modelname, nalpha, hidn_size):
        super().__init__()
        self.modelname = modelname
        self.nalpha = nalpha
        self.hidn_size = hidn_size
        self.mbed_size = 3*nalpha
        self.num_layers = 2
        self.hush = None
        #
        self.embed = nn.Embedding(nalpha+2, self.mbed_size, padding_idx=0)
        self.rnn = nn.GRU(self.mbed_size, self.hidn_size, bias=False,
                          num_layers=self.num_layers, dropout=0.3, batch_first=True)
        self.decision = nn.Linear(self.hidn_size, nalpha+2)

    def forward(self, x, h_0=None):
        embed = self.embed(x)
        output, h_n = self.rnn(embed, h_0)
        return self.decision(output), h_n

    def tensor_generator(self, words_list, batch_vol, hushed):
        padlen = max([len(w) for w in words_list]) + 1
        current_v = 0
        batch = []
        for word in words_list:
            t = torch.zeros(1, padlen).long()
            if hushed:
                w = self.hush.decode(word)
            else:
                w = word
            w = [self.nalpha + 1] + [elt + 1 for elt in w]
            t[0, :len(w)] = torch.tensor(w)
            batch.append(t)
            current_v += 1
            if current_v == batch_vol:
                current_v = 0
                ret = torch.cat(batch, 0)
                batch = []
                yield ret
        if len(batch) > 0:
            yield torch.cat(batch, 0)

    def eval_forward_batch(self, ws, batch=1, hushed=False):
        ti = time.time()
        nb_batch = int(len(ws)/batch)+1
        i = 0
        out = []
        with torch.no_grad():
            self.eval()
            gen = self.tensor_generator(ws, batch, hushed)
            for tens in gen:
                print("\rBatch : {0} / {1} : ".format(i,nb_batch), end="")
                out.append(self.forward(tens)[0])
                print("Done in {0}".format(666), end="")
                i += 1
        tf = time.time()
        print("\rComplete with total time : {0:.2f} seconds.".format(tf-ti))
        return torch.cat(out, 0)

    def probas_tables(self, ws, batch=1, hushed=False):
        t = self.eval_forward_batch(ws, batch, hushed)
        return tfunc.softmax(t[:, :, :-1], dim=2)

    def probas_words(self, ws, batch=1, hushed=False):
        ptab = self.probas_tables(ws, batch, hushed)
        probas = []
        for i, _w in enumerate(ws):
            if hushed:
                w = self.hush.decode(_w)
            else:
                w = _w
            w.append(-1)
            acc = 1
            for k, char in enumerate(w):
                acc *= ptab[i][k][char+1].item()
            probas.append(acc)
        return probas