import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from constants import *
from helper import *

def run_base_tree():
    latin_train_data,latin_train_target = convert_csv_to_data_and_target(FILE_TRAIN_LATIN)
    greek_train_data,greek_train_target = convert_csv_to_data_and_target(FILE_TRAIN_GREEK)

    latin_base_tree = train_base_tree(latin_train_data,latin_train_target)
    greek_base_tree = train_base_tree(greek_train_data,greek_train_target)

    latin_test_data_labeled, latin_test_target_labeled = convert_csv_to_data_and_target(FILE_TEST_LABEL_LATIN)
    greek_test_data_labeled, greek_test_target_labeled = convert_csv_to_data_and_target(FILE_TEST_LABEL_GREEK)

    test_tree(latin_base_tree, latin_test_data_labeled, latin_test_target_labeled)
    test_tree(greek_base_tree, greek_test_data_labeled, greek_test_target_labeled)

    test_unlabled_latin = convert_csv_to_data_unlabeled(FILE_TEST_LATIN)
    test_unlabled_greek = convert_csv_to_data_unlabeled(FILE_TEST_LABEL_GREEK)

    get_predictions_from_tree(latin_base_tree, test_unlabled_latin)
    get_predictions_from_tree(greek_base_tree, test_unlabled_greek)

def train_base_tree(data,target):   
    base_tree = DecisionTreeClassifier(criterion="entropy").fit(data, target)
    return base_tree
def test_tree(tree,data,target):
    prediction = tree.predict(data)
    total_right = 0
    for idx  in range(0,len(prediction)):
        ##print("prediction = ", prediction[idx], " actual = ", target[idx])
        if prediction[idx] == target[idx]:
            total_right += 1 
        
    print("Percent Correct = ", total_right/len(target))

def get_predictions_from_tree(tree, data):
    prediction = tree.predict(data)
    for i in range(0,len(data)):
        print(i, " ", prediction[i])
def display_tree(tree):
    plt.figure()
    plot_tree(base_tree)
    plt.show()

if __name__ == '__main__':
    run_Base_tree()