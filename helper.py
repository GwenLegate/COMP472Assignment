import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def convert_csv_to_data_and_target(filename):

    raw_csv_nparray = pd.read_csv(filename, sep=',',header=None).to_numpy()
    target = []
    data=[]
    for raw_data in raw_csv_nparray:
        target.append(raw_data[-1])
        data.append(raw_data[:-1])
    data = np.array(data)
    target = np.array(target)
    return data,target
'''
(b) a plot the confusion matrix
(c) the precision, recall, and f1-measure for each class
(d) the accuracy, macro-average f1 and weighted-average f1 of the model
'''
def plot_confusion_matrix():
    print("One day, this will do something")
