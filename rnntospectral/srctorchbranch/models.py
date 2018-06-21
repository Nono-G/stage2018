import time
import torch
import numpy as np
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

    def tensor_generator(self, words_list, batch_vol, hush=None, device="cpu"):
        hushed = (hush is not None)
        if hushed:
            padlen = hush.len_code(max(words_list)) + 1
        else:
            padlen = max([len(w) for w in words_list]) + 1
        current_v = 0
        batch = []
        for word in words_list:
            t = torch.zeros((1, padlen), dtype=torch.int64, device=device)
            if hushed:
                w = hush.decode(word)
            else:
                w = word
            w = [self.nalpha + 1] + [elt + 1 for elt in w]
            t[0, :len(w)] = torch.tensor(w, dtype=torch.int64, device=device)
            batch.append(t)
            current_v += 1
            if current_v == batch_vol:
                current_v = 0
                ret = torch.cat(batch, 0)
                batch = []
                yield ret
        if len(batch) > 0:
            yield torch.cat(batch, 0)

    def probas_tables_numpy(self, ws, batch=1, hush=None, del_start_symb=False, quiet=False, device="cpu"):
        """evaluate batch, compute softmax, then copy to cpu to free VRAM, return as numpy to emphasize that"""
        ti = time.time()
        nb_batch = int(len(ws)/batch)+1
        i = 0
        out = []
        with torch.no_grad():
            self.eval()
            gen = self.tensor_generator(ws, batch, hush, device=device)
            for tens in gen:
                if not quiet:
                    print("\rBatch : {0} / {1} : ".format(i,nb_batch), end="")
                preds, _ = self.forward(tens)
                if del_start_symb:
                    preds = tfunc.softmax(preds[:, :, :-1], dim=2)
                else:
                    preds = tfunc.softmax(preds, dim=2)
                out.append(preds.cpu().numpy())
                i += 1
        tf = time.time()
        if not quiet:
            print("\rCompleted {0} batches in : {1:.2f} seconds.".format(i, tf-ti))
        # return torch.cat(out, 0)
        return np.concatenate(out, 0)

    def eval_forward_batch(self, ws, batch=1, hush=None, quiet=False, device="cpu"):
        ti = time.time()
        nb_batch = int(len(ws)/batch)+1
        i = 0
        out = []
        with torch.no_grad():
            self.eval()
            gen = self.tensor_generator(ws, batch, hush, device=device)
            for tens in gen:
                if not quiet:
                    print("\rBatch : {0} / {1} : ".format(i,nb_batch), end="")
                out.append(self.forward(tens)[0])
                # if not quiet:
                #     print("Done in {0}".format(666), end="")
                i += 1
        tf = time.time()
        if not quiet:
            print("\rCompleted {0} batches in : {1:.2f} seconds.".format(i, tf-ti))
        return torch.cat(out, 0)

    def probas_tables(self, ws, batch=1, hush=None, del_start_symb=False, quiet=False, device="cpu"):
        t = self.eval_forward_batch(ws, batch, hush, quiet=quiet, device=device)
        if del_start_symb:
            return tfunc.softmax(t[:, :, :-1], dim=2)
        else:
            return tfunc.softmax(t, dim=2)

    def probas_words(self, ws, batch=1, hushed=False, device="cpu"):
        ptab = self.probas_tables(ws, batch, hushed, device=device)
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
