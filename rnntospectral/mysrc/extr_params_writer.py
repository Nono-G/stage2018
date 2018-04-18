pdic = dict()
npb = 48

# noinspection PyPep8
nletters = [-666, 8,18,4,4,6,6,13,8,4,11,20,13,4,12,14,10,13,20,7,18,23,21,7,5,10,6,17,6,6,10,5,4,15,21,20,9,8,10,14,14,7,9,5,13,19,23,15,23]
# noinspection PyPep8
nstates = [-666, 63,63,25,12,56,19,12,49,71,49,47,12,63,15,26,49,22,25,68,11,56,55,33,6,40,73,19,23,36,9,12,43,13,64,47,54,69,14,6,65,54,6,67,73,14,19,61,16]

mode1rnn = [2, 8, 9]

all_rows_cols = ["4", "0_6","10"]
# all_drops = [(200,200),(500,500),(1000,1000),(2000,2000)]
all_drops = [(2000,2000), (2000,3000)]
rank_around = 15


def rank(d,f,step):
    s = ""
    for ix in range(d,f,step):
        s += (str(ix)+"_")
    return s[:-1]


for i in range(1, 11):
    for rc in all_rows_cols:
        for dr in all_drops:
            if i in mode1rnn:
                print("models/m{0}-m".format(i), end="")
            else:
                print("\"models/m{0}-d models/m{0}-w\"".format(i), end="")
            print(" "+rank(nstates[i]-rank_around, nstates[i]+rank_around, 1), rc, rc, dr[0], dr[1],
                  "data/pautomac/{0}.pautomac.test data/pautomac/{0}.pautomac_solution.txt data/pautomac/{0}.pautomac_model.txt".format(i))


