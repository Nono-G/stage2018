
all_probs = [("data/pautomac/{0}.pautomac.train".format(nb), "data/pautomac/{0}.pautomac.devtest".format(nb))
             for nb in range(1, 11)]
all_wid = [-1, 10]
all_neurons = [30, 50, 120]
all_epochs = [50]
all_batch = [32]
all_sample = [75000]
all_name = ["5Avr"]
all_layer = [3]
all_modes = [0]

for (trainf, testf) in all_probs:
    for wid in all_wid:
        for neurons in all_neurons:
            for epochs in all_epochs:
                for batch in all_batch:
                    for sample in all_sample:
                        for name in all_name:
                            for layer in all_layer:
                                for mode in all_modes:
                                    print("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}"
                                          .format(wid, trainf, neurons, epochs,
                                                  batch, sample, name, testf, layer, mode))
