import parse3 as parse
import keras
import sys
# extra imports to set GPU options
import tensorflow as tf
from keras import backend as k


if len(sys.argv) != 7:
    sys.exit("ARGS : 'w1 w2 w3' pad file neurons epochs batch")

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


# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
windows_widths = [int(s) for s in sys.argv[1].split()]
pad_width = int(sys.argv[2])
train_file = sys.argv[3]
neurons = int(sys.argv[4])
epochs = int(sys.argv[5])
batch = int(sys.argv[6])
modelname = ("modelW["+sys.argv[1]+"]P"+sys.argv[2] +
             "F"+sys.argv[3]+"N"+sys.argv[4] +
             "E"+sys.argv[5]+"B"+sys.argv[6])\
    .replace(" ", "_").replace("/", "+")

nalpha, x_train, y_train = parse.parse_train(train_file, windows_widths, pad_width)
# print(y_train)
# print(nalpha)

print(x_train.shape)
# print(y_train.shape)
# print(x_train)
# print(y_train)

model = keras.models.Sequential()
model.add(keras.layers.Embedding(nalpha+3, 64, input_shape=x_train[0].shape, mask_zero=True))
model.add(keras.layers.LSTM(neurons, return_sequences=True, dropout=0.15))
model.add(keras.layers.Activation('tanh'))
# model.add(keras.layers.Reshape((100, 1)))
model.add(keras.layers.LSTM(neurons))
model.add(keras.layers.Activation('tanh'))
# model.add(keras.layers.Dense(10))
# model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(len(y_train[0])))
model.add(keras.layers.Activation('softmax'))
print(model.summary())


model.compile(optimizer=(keras.optimizers.rmsprop()), loss="categorical_crossentropy", metrics=['categorical_accuracy'])
model.fit(x_train, y_train, batch, epochs)
model.save(modelname)
