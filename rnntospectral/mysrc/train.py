import parse3 as parse
import keras
import sys
# extra imports to set GPU options
import tensorflow as tf
from keras import backend as k
import numpy as np


if len(sys.argv) < 6 or len(sys.argv) > 8:
    sys.exit("ARGS : wid file neurons epochs batch [sampleNB] [modelname]")

###################################
# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.33

# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
###################################

wid = int(sys.argv[1])
train_file = sys.argv[2]
# train_file = "../data/00.spice"
neurons = int(sys.argv[3])
epochs = int(sys.argv[4])
batch = int(sys.argv[5])
if len(sys.argv) > 6:
    sampleNB = int(sys.argv[6])
    # sampleNB = -1
else:
    sampleNB = -1
name = ""
if len(sys.argv) > 7:
    name = sys.argv[7]

modelname = ("model-{0}-W{1}F{2}N{3}E{4}B{5}S{6}"
             .format(name, wid, train_file, neurons, epochs, batch, sampleNB)
             .replace(" ", "_")
             .replace("/", "+"))
print(modelname)

nalpha, x_train, y_train = parse.parse_train(train_file, wid, padbefore=True)
if -1 < sampleNB < len(x_train):
    x_train, y_train = parse.random_sample(x_train, y_train, sampleNB)
print(x_train.shape)

model = keras.models.Sequential()
model.add(keras.layers.Embedding(nalpha+3, 40, input_shape=x_train[0].shape, mask_zero=True))
model.add(keras.layers.LSTM(neurons, return_sequences=True, dropout=0.15))
model.add(keras.layers.Activation('tanh'))
model.add(keras.layers.LSTM(neurons))
model.add(keras.layers.Activation('tanh'))
model.add(keras.layers.Dense(int(neurons/2)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(len(y_train[0])))
model.add(keras.layers.Activation('softmax'))
print(model.summary())

model.compile(optimizer=(keras.optimizers.rmsprop()), loss="categorical_crossentropy",
              metrics=['categorical_accuracy'])
for i in range(epochs):
    model.fit(x_train, y_train, batch, 1)
    model.save(modelname+"-"+str(i))
