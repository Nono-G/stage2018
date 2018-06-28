import trainer
import torch

w = torch.tensor([[2,3,1,5], [0,0,2,6]])
m = trainer.load("../mo-d", "../mo-w")
print(m.nalpha)
oh = m.one_hot(4, w)
print(oh)