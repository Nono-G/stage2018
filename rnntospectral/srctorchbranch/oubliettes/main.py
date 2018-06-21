import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import time
import math

vocab = collections.defaultdict(lambda: len(vocab))
vocab['<eos>'] = 0
vocab['<start>'] = 1

int_texts = []
with open('../movies-sf.txt', 'r') as fp:
    for line in fp:
        int_texts.append([vocab['<start>']] + [vocab[char] for char in line.strip()])

rev_vocab = {y: x for x, y in vocab.items()}

print(rev_vocab)
print(len(int_texts))

print(int_texts[42])
print(''.join([rev_vocab[x] for x in int_texts[42]]))

max_len = 40
batch_size = 8
embed_size = 16
hidden_size = 64

X = torch.zeros(len(int_texts), max_len).long()
Y = torch.zeros(len(int_texts), max_len).long()

for i, text in enumerate(int_texts):
    length = min(max_len, len(text) - 1) + 1
    X[i,:length - 1] = torch.LongTensor(text[:length - 1])
    Y[i,:length - 1] = torch.LongTensor(text[1:length])

print(X[42].tolist())
print(Y[42].tolist())
print(int_texts[42])
print([rev_vocab[y] for y in Y[42].tolist()])

X_train = X[:6500]
Y_train = Y[:6500]
X_valid = X[6500:]
Y_valid = Y[6500:]


train_set = TensorDataset(X_train, Y_train)
valid_set = TensorDataset(X_valid, Y_valid)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)


class LangMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(vocab), embed_size, padding_idx=vocab['<eos>'])
        self.rnn = nn.GRU(embed_size, hidden_size, bias=False, num_layers=1, dropout=0.3, batch_first=True)
        self.decision = nn.Linear(hidden_size, len(vocab))

    def forward(self, x, h_0=None):
        embed = self.embed(x)
        output, h_n = self.rnn(embed, h_0)
        return self.decision(output), h_n


model = LangMod()
print(model)
output, hidden = model(Variable(X[:2]))
print(output.size(), hidden.size())


def perf(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = num = 0
    with torch.no_grad():
        for x, y in loader:
            y_scores, _ = model(Variable(x))
            loss = criterion(y_scores.view(y.size(0) * y.size(1), -1), Variable(y.view(y.size(0) * y.size(1))))
            total_loss += loss.item()
            num += len(y)
    return total_loss / num, math.exp(total_loss / num)


print(perf(model, valid_loader))


def fit(model, epochs):
    ti = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(epochs):
        model.train()
        total_loss = num = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            y_scores, _ = model(Variable(x))
            loss = criterion(y_scores.view(y.size(0) * y.size(1), -1), Variable(y.view(y.size(0) * y.size(1))))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num += len(y)
        print(epoch, total_loss / num, *perf(model, valid_loader))
    tf = time.time()
    t = tf-ti
    print("Total time : {0}, Average : {1}".format(t, t/epochs))

fit(model, 5)


def generate_most_probable(model, temperature=0.5):
    with torch.no_grad():
        ret = ""
        x = Variable(torch.zeros((1, 1)).long())
        x[0, 0] = vocab['<start>']
        # size for hidden: (batch, num_layers * num_directions, hidden_size)
        hidden = Variable(torch.zeros(1, 1, hidden_size))
        for i in range(200):
            y_scores, hidden = model(x, hidden)
            dist = F.softmax(y_scores/temperature, dim=-1)[0][0]
            y_pred = torch.multinomial(dist, 1)
            # y_pred = torch.max(y_scores, 2)[1]
            selected = y_pred.data.item()
            if selected == vocab['<eos>']:
                break
            # print(rev_vocab[selected], end='')
            ret += rev_vocab[selected]
            x[0, 0] = selected
    return ret

def generate_many(model, nb, temperature=0.5):
    gens = []
    for i in range(nb):
        gens.append(generate_most_probable(model, temperature))
    return gens

generate_many(model,25, 0.7)