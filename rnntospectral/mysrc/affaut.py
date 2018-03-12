import sys
import splearn as sp
import graphviz as gv

taumin = 0.0001
thresh = 0.01

monaut = sp.Automaton.read(sys.argv[1]).minimisation(taumin)
refaut = sp.Automaton.load_Pautomac_Automaton(sys.argv[2]).minimisation(taumin)

mgr = gv.Source(monaut.get_dot(threshold=thresh, title="le mien"))
rgr = gv.Source(refaut.get_dot(threshold=thresh, title="Le vrai"))

mgr.render(filename="MonAutomate", view="True", cleanup="True")
rgr.render(filename="Pautomac", view="True", cleanup="True")