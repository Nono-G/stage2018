import sys


windows_widths = [int(s) for s in sys.argv[1].split()]
pad_width = int(sys.argv[2])
train_file = sys.argv[3]
modelname = ("["+sys.argv[1]+"]"+sys.argv[2]+sys.argv[3]+sys.argv[4]).replace(" ", "_").replace("/", "+")


print(windows_widths)
print(pad_width)
print(train_file)
print(modelname)
