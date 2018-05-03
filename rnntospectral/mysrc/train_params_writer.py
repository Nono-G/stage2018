
all_probs = [("data/spice/train/{0}.spice.train".format(nb), "OSEF")
             for nb in [4, 7, 8]]
all_wid = [-1, 8, 12]
all_neurons = [30, 50, 120, 200]
all_epochs = [50]
all_batch = [32]
all_sample = [45000]
all_name = ["3MaiSpice"]
all_layer = [3]
all_modes = [1]

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
