from basedt import *
from num_instances import *
from mlp import *
from gnb import *
from perceptron import *

'''flag for which model to run'''
RUN = "basemlp"

def main(model):
    if(model == "graph"):
        graph_initial_values()
    if(model == "basetree"):
        run_base_tree()
    if (model == "besttree"):
        run_best_tree()
    if (model == "basemlp"):
        run_base_mlp()
    if (model == "bestmlp"):
        run_best_mlp()
    if (model == "gnb"):
        run_gnb()
    if (model == "per"):
        run_perceptron()

if __name__ == '__main__':
    main(RUN)