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
        self.dense_size = int(hidn_size/2)
        self.num_layers = 2
        self.hush = None
        #
        self.embed = nn.Embedding(nalpha+2, self.mbed_size, padding_idx=0)
        self.rnn = nn.GRU(self.mbed_size, self.hidn_size, bias=False,
                          num_layers=self.num_layers, dropout=0.3, batch_first=True)
        self.dense1 = nn.Linear(self.hidn_size, self.dense_size)
        self.decision = nn.Linear(self.dense_size, nalpha+2)

    def forward(self, x, h_0=None):
        tensor = self.embed(x)
        tensor, hidden = self.rnn(tensor, h_0)
        # In pytorch, GRUs automatically do tanh activation on their outputs.
        tensor = self.dense1(tensor)
        tfunc.relu(tensor, inplace=True)
        output = self.decision(tensor)
        return output, hidden

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

    def full_probas(self, ws, batch_vol=1, hush=None, del_start_symb=False, quiet=False, device="cpu"):
        ti = time.time()
        nb_batch = int(len(ws) / batch_vol) + 1
        i = 0
        # out = np.empty(len(ws))
        out = []
        ix = 0
        if hush is None:
            start_factor = len(ws) / sum([len(w) for w in ws])
        else:
            start_factor = (len(ws)) / (sum([hush.len_code(w) for w in ws]))
        with torch.no_grad():
            self.eval()
            gen = self.tensor_generator(ws, batch_vol, hush, device=device)
            for tens in gen:
                if not quiet:
                    print("\rBatch : {0} / {1} : ".format(i, nb_batch), end="")
                preds, _ = self.forward(tens)
                if del_start_symb:
                    preds = tfunc.softmax(preds[:, :, :-1], dim=2)
                else:
                    preds = tfunc.softmax(preds, dim=2)
                tens = torch.cat([tens[:,1:],torch.zeros((len(tens),1), dtype=torch.int64, device=device)], dim=1)
                tens = tens.view(len(tens), -1, 1)
                g = torch.gather(preds, 2, tens)
                g = torch.prod(g.view(len(g), -1), dim=1)
                g *= start_factor
                g = g.cpu().numpy()
                out.append(g)

                # preds = preds.cpu()
                # for j, table in enumerate(preds):
                #     current_word = ws[i * batch_vol + j]
                #     if hush is None:
                #         indexes = [x + 1 for x in current_word] + [0]
                #     else:
                #         indexes = [x + 1 for x in hush.decode(current_word)] + [0]
                #     acc = start_factor
                #     for k in range(len(indexes)):
                #         acc *= table[k, indexes[k]].item()
                #     out[ix] = acc
                #     ix += 1
                i += 1
        tf = time.time()
        if not quiet:
            print("\rCompleted {0} batches in : {1:.2f} seconds.".format(i, tf - ti))
        return np.concatenate(out)

    # def full_probas(self, ws, batch_vol=1, hush=None, del_start_symb=False, quiet=False, device="cpu"):
    #     ti = time.time()
    #     nb_batch = int(len(ws) / batch_vol) + 1
    #     i = 0
    #     out = np.empty(len(ws))
    #     ix = 0
    #     if hush is None:
    #         start_factor = len(ws) / sum([len(w) for w in ws])
    #     else:
    #         start_factor = (len(ws)) / (sum([hush.len_code(w) for w in ws]))
    #     with torch.no_grad():
    #         self.eval()
    #         gen = self.tensor_generator(ws, batch_vol, hush, device=device)
    #         old_preds = None
    #         for tens in gen:
    #             if not quiet:
    #                 print("\rBatch : {0} / {1} : ".format(i, nb_batch), end="")
    #             preds, _ = self.forward(tens)
    #             if del_start_symb:
    #                 preds = tfunc.softmax(preds[:, :, :-1], dim=2)
    #             else:
    #                 preds = tfunc.softmax(preds, dim=2)
    #             preds = preds.cpu()
    #             if old_preds is not None:
    #                 self._table_to_probas(batch_vol, hush, i-1, ix, out, old_preds, start_factor, ws)
    #             old_preds = preds
    #             i += 1
    #         self._table_to_probas(batch_vol, hush, i - 1, ix, out, old_preds, start_factor, ws)
    #     tf = time.time()
    #     if not quiet:
    #         print("\rCompleted {0} batches in : {1:.2f} seconds.".format(i, tf - ti))
    #     return out
    #
    # def _table_to_probas(self, batch_vol, hush, i, ix, out, preds, start_factor, ws):
    #     for j, table in enumerate(preds):
    #         current_word = ws[i * batch_vol + j]
    #         if hush is None:
    #             indexes = [x + 1 for x in current_word] + [0]
    #         else:
    #             indexes = [x + 1 for x in hush.decode(current_word)] + [0]
    #         acc = start_factor
    #         for k in range(len(indexes)):
    #             acc *= table[k, indexes[k]].item()
    #         out[ix] = acc
    #         ix += 1

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

    def one_hot(self, maxlen, ws):
        t = torch.zeros(ws.shape[0], maxlen, self.nalpha).scatter_(1, ws.view(-1, maxlen, 1), 1.0)
        return t