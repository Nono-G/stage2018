from splearn.datasets.base import load_data_sample
from splearn.tests.datasets.get_dataset_path import get_dataset_path
from splearn import Spectral


train_file = '00.spice'  # '4.spice.train'
data = load_data_sample(adr=get_dataset_path(train_file))
print(data)
est = Spectral(partial=False, sparse=False)
est.fit(data.data)
a = est.automaton
print(a.initial)