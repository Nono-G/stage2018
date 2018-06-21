import sys

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("pref core suff nb step")
        exit(-666)
    pref = sys.argv[1]
    core = int(sys.argv[2])
    suff = sys.argv[3]
    nb = int(sys.argv[4])
    step = int(sys.argv[5])
    fs = list()
    for k in range(core, core+(nb*step), step):
        fs.append(pref+str(k)+suff)
    # OUTPUT :
    for f in fs:
        print(f, end=" ")