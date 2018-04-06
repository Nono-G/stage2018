import keras
from train2f import DenseLinkedEmbedding
import train2f

m = train2f.my_load_model("../model-test-W5Fdata+pautomac+9.pautomac.trainN20E4B32S320L3M0-3")

print(m.summary())

m = train2f.my_load_model("../model-test-W5Fdata+pautomac+9.pautomac.trainN20E4B32S320L3M1+DIGEST",
                          "../model-test-W5Fdata+pautomac+9.pautomac.trainN20E4B32S320L3M1-3-WEIGHTS")

print(m.summary())