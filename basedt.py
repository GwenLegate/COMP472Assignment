import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from constants import *
from helper import *

def run_base_tree():
    latin_train_data,latin_train_target = convert_csv_to_data_and_target(FILE_TRAIN_LATIN)
    greek_train_data,greek_train_target = convert_csv_to_data_and_target(FILE_TRAIN_GREEK)

    latin_base_tree = train_base_tree(latin_train_data,latin_train_target)
    greek_base_tree = train_base_tree(greek_train_data,greek_train_target)

    latin_test_data_labeled, latin_test_target_labeled = convert_csv_to_data_and_target(FILE_TEST_LABEL_LATIN)
    greek_test_data_labeled, greek_test_target_labeled = convert_csv_to_data_and_target(FILE_TEST_LABEL_GREEK)

    latin_prediction = test_tree(latin_base_tree, latin_test_data_labeled, latin_test_target_labeled)
    greek_prediction = test_tree(greek_base_tree, greek_test_data_labeled, greek_test_target_labeled)
    
    convert_model_output_to_csv("BASE-DT", "DS1", latin_prediction, latin_test_target_labeled)
    convert_model_output_to_csv("BASE_DT", "DS2", greek_prediction, greek_test_target_labeled)
def run_best_tree():
    latin_train_data,latin_train_target = convert_csv_to_data_and_target(FILE_TRAIN_LATIN)
    greek_train_data,greek_train_target = convert_csv_to_data_and_target(FILE_TRAIN_GREEK)
    print("starting latin training")
    latin_base_tree = train_best_tree(latin_train_data,latin_train_target)
    print("starting greek training")
    greek_base_tree = train_best_tree(greek_train_data,greek_train_target)

    latin_test_data_labeled, latin_test_target_labeled = convert_csv_to_data_and_target(FILE_TEST_LABEL_LATIN)
    greek_test_data_labeled, greek_test_target_labeled = convert_csv_to_data_and_target(FILE_TEST_LABEL_GREEK)
    print("starting latin prediction")
    latin_prediction = test_tree(latin_base_tree, latin_test_data_labeled, latin_test_target_labeled)
    print("starting greek prediction")
    greek_prediction = test_tree(greek_base_tree, greek_test_data_labeled, greek_test_target_labeled)
    
    convert_model_output_to_csv("BEST-DT", "DS1", latin_prediction, latin_test_target_labeled)
    convert_model_output_to_csv("BEST-DT", "DS2", greek_prediction, greek_test_target_labeled)

    # test_unlabled_latin = convert_csv_to_data_unlabeled(FILE_TEST_LATIN)
    # test_unlabled_greek = convert_csv_to_data_unlabeled(FILE_TEST_LABEL_GREEK)

    # get_predictions_from_tree(latin_base_tree, test_unlabled_latin)
    # get_predictions_from_tree(greek_base_tree, test_unlabled_greek)

def train_base_tree(data,target):   
    base_tree = DecisionTreeClassifier(criterion="entropy").fit(data, target)
    return base_tree
def train_best_tree(data,target):
    param_dict = {
        "criterion":["gini", "entropy"],
        "max_depth":[None, 10],
        "min_samples_split":[.25,.5,.75,2,3,5,10,20],
        "min_impurity_decrease":[0,1,2,5,10,20],
        "class_weight":[None, "balanced"]
    }
    base_tree = DecisionTreeClassifier()
    grid = GridSearchCV(base_tree,param_grid=param_dict, n_jobs=-1)
    grid.fit(data,target)
    print("Best Params: ",grid.best_params_)
    return grid
def test_tree(tree,data,target):
    prediction = tree.predict(data)
    total_right = 0
    for idx  in range(0,len(prediction)):
        ##print("prediction = ", prediction[idx], " actual = ", target[idx])
        if prediction[idx] == target[idx]:
            total_right += 1 
        
    print("Percent Correct = ", total_right/len(target))
    return prediction

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