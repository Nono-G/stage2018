import parse3 as parse

x = parse.parse_fullwords("../data/pautomac/4.pautomac.test")
y = parse.parse_pautomac_results("../data/pautomac/4.pautomac_solution.txt")
print(x)
print(y)


