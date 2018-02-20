import keras
import parse3 as parse
import score as sc
import sys
import numpy as np
# extra imports to set GPU options
import tensorflow as tf
from keras import backend as k


def prec_seq(y_test, pred):
    s = len(y_test[0])
    occ_test = [0]*s
    occ_pred = [0]*s
    occ_faux = [0]*s
    faux = 0
    for x in range(0, len(y_test)):
        occ_test[parse.argmax(y_test[x])] += 1
    for x in range(0, len(y_test)):
        if parse.argmax(pred[x]) == parse.argmax(y_test[x]):
            occ_pred[parse.argmax(pred[x])] += 1
        else:
            faux += 1
            occ_faux[parse.argmax(pred[x])] += 1
    print("CORRECT :")
    for x in range(0, s):
        if occ_test[x] != 0:
            print(str(x)+":", str(occ_pred[x])+"/"+str(occ_test[x]), "->", occ_pred[x]/occ_test[x], sep="\t")
        else:
            print(str(x)+":", str(occ_pred[x])+"/"+str(occ_test[x]), sep="\t")
    if faux != 0:
        print("INCORRECT :")
        for x in range(0, s):
            print(str(x)+":", str(occ_faux[x])+"/"+str(faux), "->", occ_faux[x]/faux, sep="\t")


if len(sys.argv) != 5:
    sys.exit("ARGS : wid prefixes model targets")

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

# windows_widths = [int(s) for s in sys.argv[1].split()]
# pad_width = int(sys.argv[2])
wid = int(sys.argv[1])
test_file = sys.argv[2]
model_file = sys.argv[3]
targets_file = sys.argv[4]

# a, x_test, y_test = parse.parse_train(test_file, wid)
a, x_test = parse.parse_test_prefixes(test_file, wid)
y_test = parse.parse_targets(targets_file, a)
# print(x_test)
print(x_test.shape)
model = keras.models.load_model(model_file)
scores = model.evaluate(x_test, y_test)
print(scores)
pred = model.predict(x_test, len(x_test))
pred = np.array([parse.best_n_args(aaa, 5) for aaa in pred])
s = sc.calc_score(pred, targets_file)
print(s)

# with open("ret", "w") as fret:
#     for seq in pred:
#         for i in seq:
#             fret.write(str(i)+" ")
#         fret.write("\n")

# print([parse.argmax(x) for x in pred])
# prec_seq(y_test, pred)
