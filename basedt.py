import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from constants import *
from helper import *
# # Parameters
# n_classes = 3
# plot_colors = "ryb"
# plot_step = 0.02

# # Load data
# iris = load_iris()

# for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
#                                 [1, 2], [1, 3], [2, 3]]):
#     # We only take the two corresponding features
#     X = iris.data[:, pair]
#     y = iris.target


# plt.figure()
# clf = DecisionTreeClassifier(criterion="entropy").fit(iris.data, iris.target)
# plot_tree(clf)
# plt.show()
def run_base_tree():
    filename = FILE_TRAIN_LATIN
    train_data,train_target = convert_csv_to_data_and_target(filename)

    base_tree = train_base_tree(train_data,train_target)
    test_data, test_target = convert_csv_to_data_and_target(testfilename)
    #percent_right = test_base_tree(base_tree,filename)


def train_base_tree(data,target):
    plt.figure()
    base_tree = DecisionTreeClassifier(criterion="entropy").fit(data, target)
    plot_tree(base_tree)
    plt.show()
    return base_tree


if __name__ == '__main__':
    run_Base_tree()