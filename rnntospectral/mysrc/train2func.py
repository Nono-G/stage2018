import keras
import sys
import keras.backend as k
# Maison :
import parse5 as parse
import scores
import spextractor_common


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
    if pautomac:
        print("Pautomac base perplexity : {0}"
              .format(scores.pautomac_perplexity(pautomac_sol, pautomac_sol)))
    losses = []
    for i in range(1, epochs+1):
        # pp = mmdi.predict(x_train[:5])
        # print(pp)
        h = model.fit(x_train, y_train, batch, 1)
        # pp = mmdi.predict(x_train[:5])
        # print(pp)
        model.save(modelname + "-" + str(i))
        losses.append(h.history["loss"][0])
        if pautomac:
            pautomac_perp.append(scores.pautomac_perplexity(pautomac_sol,
                                                            spextractor_common.proba_words_para(model, pautomac_test,
                                                                                                nalpha, False, True)))
        for e in range(i):
            print("Loss at epoch {0} : {1}".format(e+1, losses[e]))
            if pautomac:
                print("Perplexity at epoch {0} : {1}"
                      .format(e+1, pautomac_perp[e]))
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
        paut_arg = False
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
