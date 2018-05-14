# External :
import keras
import sys
import keras.backend as k
import numpy as np
# Project :
import parse5 as parse

"""
Runnable program to train models, and related classes and functions
"""


class DenseLinkedEmbedding(keras.layers.Embedding):
    """
    Custom Keras layer, behaving like the Embedding layer, but using the weights of a specified dense layer instead of
    having its own weights, nothing trainable here then.
    Warning : the additional reference parameter of the constructor breaks the keras.load_model(filename) function,
    Instead a non trained model and load weights instead !
    """

    # noinspection PyUnusedLocal
    def __init__(self, input_dim, output_dim, dbedl, **kwargs):
        keras.layers.Embedding.__init__(self, input_dim, output_dim, mask_zero=True)
        self.dbedl = dbedl

    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
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


def model_shell_normal(xsize, ysize, neurons, nalpha, layer):
    """
    Returns an untrained, uncompiled model made of : Embedding | 2*RNN | Dense | Dense for output
    :param xsize: the length of the input sequences, sometimes called 'pad'
    :param ysize: the dimension of the outputs, usual usage is nalpha + 3 (all letters, padding, start and end)
    :param neurons: number of neurons in the RNN layers.
    :param nalpha: number of letter in the alphabet
    :param layer: Kind of RNN layer to use, see 'rnn_layer()' function
    :return: an untrained, uncompiled model made of : Embedding | 2*RNN | 2*Dense
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(nalpha + 3, 4 * nalpha, input_shape=(xsize, ), mask_zero=True))
    model.add(rnn_layer(layer, units=neurons, return_sequences=True, dropout=0.15))
    model.add(keras.layers.Activation('tanh'))
    model.add(rnn_layer(layer, units=neurons))
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.Dense(int(neurons / 2)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(ysize))
    model.add(keras.layers.Activation('softmax'))
    return model


def model_shell_dbed(xsize, ysize, neurons, nalpha, layer, save_digest=False):
    """
    Returns an untrained, uncompiled model made of : Custom Embedding | 2*RNN | 2*Dense | Dense for output
    See custom 'DenseLinkedEmbedding' embedding class above.
    :param xsize: the length of the input sequences, sometimes called 'pad'
    :param ysize: the dimension of the outputs, usual usage is nalpha + 3 (all letters, padding, start and end)
    :param neurons: number of neurons in the RNN layers.
    :param nalpha: number of letter in the alphabet
    :param layer: Kind of RNN layer to use, see 'rnn_layer()' function
    :param save_digest: Must we, or not, save a file containing the other parameters, which will be used to reload the
                        model, by instantiating a topologically equivalent model and pouring the saved weights in it ?
    :return: an untrained, uncompiled model made of : Custom Embedding | 2*RNN | 2*Dense | Dense for output
    """
    dbedl = (keras.layers.Dense(ysize, activation="softmax", use_bias=True))
    # Only the last element of the input_shape tuple is used, so value 1664 is just a placeholder here
    dbedl.build((1664, 3 * nalpha))

    inp = keras.layers.Input(shape=(xsize, ))
    mbed = (DenseLinkedEmbedding(nalpha + 3, 3 * nalpha, dbedl))(inp)
    r1 = (rnn_layer(layer, units=neurons, activation="tanh", return_sequences=True, dropout=0.15))(mbed)
    r2 = (rnn_layer(layer, units=neurons, activation="tanh"))(r1)
    d1 = (keras.layers.Dense(int(neurons / 2), activation="relu"))(r2)
    d2 = (keras.layers.Dense(3 * nalpha))(d1)
    outp = dbedl(d2)
    model = keras.models.Model(inputs=inp, outputs=outp)
    if save_digest:
        with open(modelname+"+DIGEST", "w") as f:
            f.write("xsize ysize neurons nalpha layer\n")
            f.write(str(xsize)+"\n")
            f.write(str(ysize) + "\n")
            f.write(str(neurons)+"\n")
            f.write(str(nalpha)+"\n")
            f.write(str(layer)+"\n")
    return model


def model_shell_dbed_from_digest(filename):
    """
    Acquires parameters form a 'model digest' file, and creates a custom-embedding-model
    :param filename: The name of the 'model digest' file
    :return: an untrained, uncompiled model made of : Custom Embedding | 2*RNN | 2*Dense | Dense for output
    """
    with open(filename, "r") as file:
        _ = file.readline()
        xsize = int(file.readline())
        ysize = int(file.readline())
        neurons = int(file.readline())
        nalpha = int(file.readline())
        layer = int(file.readline())
        return model_shell_dbed(xsize, ysize, neurons, nalpha, layer, save_digest=False)


def my_load_model(f1, f2=""):
    """
    Can load models using either custom or stock embedding layer.
    :param f1: File produced by keras.model.save(filename), or 'model digest' file
    :param f2: If f1 is a 'model digest', file containing the weights of the model
    :return: a model, trained and compiled, recreated from saved file(s).
    """
    # noinspection PyBroadException
    try:
        # Stock Embedding
        model = keras.models.load_model(f1)
    except Exception:
        # Custom Embedding
        model = model_shell_dbed_from_digest(f1)
        model.load_weights(f2)
    return model


def trainf(train_file, wid, sample, neurons, epochs, batch, test_file="", layer=1, mode=0):
    """
    Trains a model on the specified train set, with specified parameters.
    :param train_file: name of the file containing
    :param wid: the width of the n-grams, which is the length of the input sequences. -1 means 'the longest ever seen in
                train set'
    :param sample: number of prefixes to randomly pick from the train set.
    :param neurons: number of neurons in RNN layers, see model_shell_*() functions
    :param epochs: number of epochs of training
    :param batch: size of the batches of samples fed to the model during training
    :param test_file: (Optional) neme of the file containing a test or validation set, loss on this set is computed at
                      each epoch.
    :param layer: kind of RNN layer used, see model_shell_*() functions
    :param mode: Do we use custom embedding layer or not ? 1 means yes, 0 means no
    :return: a trained model
    """
    # None things :
    x_val = None
    y_val = None
    losses = []
    val_losses = []
    do_test = (test_file != "")
    nalpha, x_train, y_train = parse.parse_train(train_file, wid, padbefore=True)
    print(sample)
    if -1 < sample < len(x_train):
        x_train, y_train = parse.random_sample(x_train, y_train, sample+3000)
        x_val = np.array(list(x_train[sample:]))
        x_train = np.array(list(x_train[:sample]))
        y_val = np.array(list(y_train[sample:]))
        y_train = np.array(list(y_train[:sample]))
    else:
        print("SAMPLING ABORTED, NOT ENOUGH SAMPLES")
        do_test = False
    print(x_train.shape)
    if do_test:
        # _, x_val, y_val = parse.parse_train(test_file, len(x_train[0]), padbefore=True)
        pass
    if mode == 0:
        model = model_shell_normal(len(x_train[0]), len(y_train[0]), neurons, nalpha, layer)
    elif mode == 1:
        model = model_shell_dbed(len(x_train[0]), len(y_train[0]), neurons, nalpha, layer, save_digest=True)
    else:
        raise ValueError("'mode' param must be 0 or 1")
    print(model.summary())
    model.compile(optimizer=(keras.optimizers.rmsprop()), loss="categorical_crossentropy",
                  metrics=['categorical_accuracy'])
    for i in range(1, epochs+1):
        h = model.fit(x_train, y_train, batch, 1)
        if mode == 0:
            model.save(modelname + "-" + str(i))
        elif mode == 1:
            model.save_weights(modelname + "-" + str(i)+"-WEIGHTS")
        losses.append(h.history["loss"][0])
        if do_test:
            val_losses.append(model.evaluate(x_val, y_val, 2048))
        if do_test:
            for e in range(i):
                print("Loss at epoch {0} : {1} on train, {2} on validation".format(e + 1, losses[e], val_losses[e]))
            mini = np.argmin(val_losses, axis=0)
            print("Current best is {0} with {1}".format(mini[0]+1, val_losses[mini[0]]))
            sys.stdout.flush()
        else:
            for e in range(i):
                print("Loss at epoch {0} : {1} on train".format(e + 1, losses[e]))
                sys.stdout.flush()
    return model


def rnn_layer(t, *args, **kwargs):
    """
    Returns a RNN layer, among the Keras layers.
    :param t: type of layer
    :param args: args to give to the layer constructor
    :param kwargs: kwargs to give to the layer constructor
    :return: a RNN layer.
    """
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
        sys.exit("ARGS : wid train_file neurons epochs batch "
                 "[sampleNB] [modelname] [test_file] [layer] [mode: default=0=classic, 1=dbed]")

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
        test_file_arg = sys.argv[8]
    else:
        test_file_arg = None
    if len(sys.argv) > 9:
        layer_arg = int(sys.argv[9])
    else:
        layer_arg = 1
    if len(sys.argv) > 10:
        mode_arg = int(sys.argv[10])
    else:
        mode_arg = 0

    modelname = ("model-{0}-W{1}F{2}N{3}E{4}B{5}S{6}L{7}M{8}"
                 .format(name, wid_arg, trainfile_arg, neurons_arg, epochs_arg,
                         batch_arg, sample_arg, layer_arg, mode_arg)
                 .replace(" ", "_")
                 .replace("/", "+"))
    print(modelname)
    m = trainf(trainfile_arg, wid_arg, sample_arg, neurons_arg, epochs_arg, batch_arg,
               test_file_arg, layer_arg, mode_arg)
