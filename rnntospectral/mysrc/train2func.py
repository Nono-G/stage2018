import keras
import sys
import keras.backend as k
import numpy as np
# Maison :
import parse5 as parse


class Mbed(keras.layers.Embedding):
    def __init__(self, input_dim, output_dim, dbedl):
        keras.layers.Embedding.__init__(self, input_dim, output_dim, mask_zero=True)
        self.dbedl = dbedl

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            trainable=False,
            dtype=self.dtype)
        self.built = True

    def call(self, inputs):
        if k.dtype(inputs) != 'int32':
            inputs = k.cast(inputs, 'int32')
        out = k.gather(k.transpose(self.dbedl.kernel), inputs)
        # out2 = K.gather(self.embeddings, inputs)
        return out


def trainf(train_file, wid, sample, neurons, epochs, batch, test_file="", layer=1):
    # None things :
    x_val = None
    y_val = None
    losses = []
    val_losses = []
    do_test = (test_file != "")

    nalpha, x_train, y_train = parse.parse_train(train_file, wid, padbefore=True)
    print(sample)
    if -1 < sample < len(x_train):
        x_train, y_train = parse.random_sample(x_train, y_train, sample)
    print(x_train.shape)
    if do_test:
        _, x_val, y_val = parse.parse_train(test_file, wid, padbefore=True)
        # if -1 < sample < len(x_train):
        #     x_val, y_val = parse.random_sample(x_val, y_val, sample)

    dbedl = (keras.layers.Dense(len(y_train[0]), activation="softmax", use_bias=True))
    dbedl.build((1664, 3*nalpha))

    inp = keras.layers.Input(shape=x_train[0].shape)
    # mbed = (keras.layers.Embedding(nalpha + 3, 3 * nalpha, mask_zero=True))(inp)
    mbed = (Mbed(nalpha + 3, 3 * nalpha, dbedl))(inp)
    r1 = (nono_layer(layer, units=neurons, activation="tanh", return_sequences=True, dropout=0.15))(mbed)
    r2 = (nono_layer(layer, units=neurons, activation="tanh"))(r1)
    d1 = (keras.layers.Dense(int(neurons / 2), activation="relu"))(r2)
    d2 = (keras.layers.Dense(3*nalpha))(d1)
    outp = dbedl(d2)
    model = keras.models.Model(inputs=inp, outputs=outp)

    # mmdi = keras.models.Model(inputs=inp, outputs=mbed)

    print(model.summary())

    model.compile(optimizer=(keras.optimizers.rmsprop()), loss="categorical_crossentropy",
                  metrics=['categorical_accuracy'])
    # if pautomac:
    #     print("Pautomac base perplexity : {0}"
    #           .format(scores.pautomac_perplexity(pautomac_sol, pautomac_sol)))

    for i in range(1, epochs+1):
        h = model.fit(x_train, y_train, batch, 1)
        model.save(modelname + "-" + str(i))
        losses.append(h.history["loss"][0])
        if do_test:
            val_losses.append(model.evaluate(x_val, y_val, 2048))
            # pautomac_perp.append(scores.pautomac_perplexity(pautomac_sol,
            #                                                 spextractor_common.proba_words_para(model, pautomac_test,
            #                                                                                     nalpha, False, True)))

        if do_test:
            for e in range(i):
                print("Loss at epoch {0} : {1} on train, {2} on validation".format(e + 1, losses[e], val_losses[e]))
            mini = np.argmin(val_losses, axis=0)
            print("Current best is {0} with {1}".format(mini[0]+1, val_losses[mini[0]]))
            sys.stdout.flush()

            # for e in range(i):
            #     print("Perplexity at epoch {0} : {1}".format(e+1, pautomac_perp[e]))
            #     sys.stdout.flush()
        else:
            for e in range(i):
                print("Loss at epoch {0} : {1} on train".format(e + 1, losses[e]))
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
    if len(sys.argv) < 6 or len(sys.argv) > 10:
        sys.exit("ARGS : wid train_file neurons epochs batch [sampleNB] [modelname] [test_file] [layer]")

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
        test_file = sys.argv[8]
    else:
        test_file = None
    if len(sys.argv) > 9:
        layer_arg = int(sys.argv[9])
    else:
        layer_arg = 1

    modelname = ("model-{0}-W{1}F{2}N{3}E{4}B{5}S{6}L{7}"
                 .format(name, wid_arg, trainfile_arg, neurons_arg, epochs_arg, batch_arg, sample_arg, layer_arg)
                 .replace(" ", "_")
                 .replace("/", "+"))
    print(modelname)
    trainf(trainfile_arg, wid_arg, sample_arg, neurons_arg, epochs_arg, batch_arg, test_file, layer_arg)
