import keras
import sys
import numpy as np
# Maison :
import parse3 as parse
import scores
import hank


def trainf(train_file, wid, sample, neurons, epochs, batch,
           pautomac=False, pautomac_test_file="", pautomac_sol_file="", layer=1):
    # None things :
    pautomac_test = None
    pautomac_sol = None
    pautomac_perp = []
    # OK :

    nalpha, x_train, y_train = parse.parse_train(train_file, wid, padbefore=True)
    print(sample)
    if -1 < sample < len(x_train):
        x_train, y_train = parse.random_sample(x_train, y_train, sample)
    print(x_train.shape)
    if pautomac:
        pautomac_test = parse.parse_fullwords(pautomac_test_file)
        pautomac_sol = parse.parse_pautomac_results(pautomac_sol_file)

    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(nalpha + 3, 4*nalpha, input_shape=x_train[0].shape, mask_zero=True))
    model.add(nono_layer(layer, units=neurons, return_sequences=True, dropout=0.15))
    model.add(keras.layers.Activation('tanh'))
    model.add(nono_layer(layer, units=neurons))
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.Dense(int(neurons / 2)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(len(y_train[0])))
    model.add(keras.layers.Activation('softmax'))
    print(model.summary())

    model.compile(optimizer=(keras.optimizers.rmsprop()), loss="categorical_crossentropy",
                  metrics=['categorical_accuracy'])
    if pautomac:
        print("Pautomac base perplexity : {0}"
              .format(scores.pautomac_perplexity(pautomac_sol, pautomac_sol)))
    losses = []
    for i in range(epochs):
        h = model.fit(x_train, y_train, batch, 1)
        model.save(modelname + "-" + str(i))
        losses.append(h.history["loss"][0])
        if pautomac:
            pautomac_perp.append(scores.pautomac_perplexity(pautomac_sol, hank.proba_words_para(model, pautomac_test, nalpha, False, True)))
    for i in range(epochs):
        print("Loss at epoch {0} : {1}".format(i, losses[i]))
        if pautomac:
            print("Perplexity at epoch {0} : {1}"
                  .format(i, pautomac_perp[i]))
        sys.stdout.flush()


def nono_layer(t, *args, **kwargs):
    if t == 0:
        return keras.layers.CuDNNLSTM(*args, **kwargs)
    elif t == 1:
        return keras.layers.LSTM(*args, **kwargs)
    elif t == 2:
        return keras.layers.CuDNNGRU(*args, **kwargs)
    elif t == 3:
        return keras.layers.GRU(*args, **kwargs)


if __name__ == "__main__":
    if len(sys.argv) < 6 or len(sys.argv) > 11:
        sys.exit("ARGS : wid file neurons epochs batch [sampleNB] [modelname] [pauttest pautsol] [layer]")

    # ARGS :
    wid_arg = int(sys.argv[1])
    trainfile_arg = sys.argv[2]
    neurons_arg = int(sys.argv[3])
    epochs_arg = int(sys.argv[4])
    batch_arg = int(sys.argv[5])
    if len(sys.argv) > 6:
        sample_arg = int(sys.argv[6])
    else:
        sample_arg = -1
    name = ""
    if len(sys.argv) > 7:
        name = sys.argv[7]
    if len(sys.argv) > 8:
        paut_arg = True
        pauttest_arg = sys.argv[8]
        pautsol_arg = sys.argv[9]
    else:
        paut_arg=False
        pauttest_arg = None
        pautsol_arg = None
    if len(sys.argv) > 10:
        layer_arg = int(sys.argv[10])
    else:
        layer_arg = 1

    modelname = ("model-{0}-W{1}F{2}N{3}E{4}B{5}S{6}"
                 .format(name, wid_arg, trainfile_arg, neurons_arg, epochs_arg, batch_arg, sample_arg)
                 .replace(" ", "_")
                 .replace("/", "+"))
    print(modelname)
    trainf(trainfile_arg, wid_arg, sample_arg, neurons_arg, epochs_arg, batch_arg,
           paut_arg, pauttest_arg, pautsol_arg, layer_arg)
