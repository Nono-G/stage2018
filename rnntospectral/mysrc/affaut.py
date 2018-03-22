import sys
import splearn as sp
import graphviz as gv


if len(sys.argv) < 3 or len(sys.argv) > 5:
    print("Usage : trueaut extraut [tau threshold]")
    quit(-666)

if len(sys.argv) >= 5:
    taumin = float(sys.argv[3])
    thresh = float(sys.argv[4])
else:
    taumin = 0.0
    thresh = 0.0

true_automaton = sp.Automaton.load_Pautomac_Automaton(sys.argv[1]).minimisation(taumin)
extr_automaton = sp.Automaton.read(sys.argv[2]).minimisation(taumin)

true_automaton_graph = gv.Source(true_automaton.get_dot(threshold=thresh, title="True automaton"))
extr_automaton_graph = gv.Source(extr_automaton.get_dot(threshold=thresh, title="Extracted automaton"))

true_automaton_graph.render(filename="Pautomac", view=True, cleanup=False)
extr_automaton_graph.render(filename="Extracted", view=True, cleanup=False)
