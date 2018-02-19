import parse3 as parse
import keras
import sys
# extra imports to set GPU options
import tensorflow as tf
from keras import backend as k


if len(sys.argv) != 6:
    sys.exit("ARGS : wid file neurons epochs batch")

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
# windows_widths = [int(s) for s in sys.argv[1].split()]
wid = int(sys.argv[1])
train_file = sys.argv[2]
neurons = int(sys.argv[3])
epochs = int(sys.argv[4])
batch = int(sys.argv[5])
modelname = ("model-MB-W"+sys.argv[1]+
             "F"+sys.argv[2]+"N"+sys.argv[3] +
             "E"+sys.argv[4]+"B"+sys.argv[5])\
    .replace(" ", "_").replace("/", "+")

nalpha, x_train, y_train = parse.parse_train(train_file, wid)

print(x_train.shape)
# print(y_train.shape)
# print(x_train)
# print(y_train)

model = keras.models.Sequential()
model.add(keras.layers.Embedding(nalpha+3, 2*nalpha, input_shape=x_train[0].shape, mask_zero=True))
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

model.compile(optimizer=(keras.optimizers.rmsprop()), loss="categorical_crossentropy",\
              metrics=['categorical_accuracy'])
model.fit(x_train, y_train, batch, epochs)
model.save(modelname)
