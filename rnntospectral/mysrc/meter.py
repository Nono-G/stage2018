import spextractor_common as spc
import splearn as sp
import sys

if __name__ == "__main__":
    print("Context :", sys.argv[4][3:])
    tup = (sys.argv[1], sys.argv[2])
    if sys.argv[3] != "NO":
        tup += (sys.argv[3],)
    m = spc.Spex.meter(*tup)
    m.randwords_nb = 2000
    m.rank_independent_metrics()
    for autstring in sys.argv[4:]:
        try:
            print(autstring)
            aut = sp.Automaton.read(autstring)
            m.last_extr_aut = aut
            m.rank_dependent_metrics()
        except Exception:
            pass
    #
    m.print_metrics_chart()
    m.print_metrics_chart_n_max(8)
