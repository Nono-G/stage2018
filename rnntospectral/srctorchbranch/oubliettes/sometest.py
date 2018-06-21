import random
import splearn

a = splearn.Automaton

random.seed(666)

a = [1,2,3,4,5,6,7,8]
random.shuffle(a)
print(a)