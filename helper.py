import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def convert_csv_to_data_and_target(filename):
    '''convert csv into format 2d numpy array with the last number of each row removed and set as the target label
    returns:
    2dnumpy array: data - the 2d array with 0,1s representing the image
    1d numpy array: target - the 1d array with the values corresponding to what each element in data should represent (labels of the data set)
    '''
    raw_csv_nparray = pd.read_csv(filename, sep=',',header=None).to_numpy()
    target = []
    data=[]
    for raw_data in raw_csv_nparray:
        target.append(raw_data[-1])
        data.append(raw_data[:-1])
    data = np.array(data)
    target = np.array(target)
    return data,target

def convert_csv_to_data_unlabeled(filename):
    data = pd.read_csv(filename, sep=',',header=None).to_numpy()
    return data


def plot_confusion_matrix():
    '''
    (b) a plot the confusion matrix
    (c) the precision, recall, and f1-measure for each class
    (d) the accuracy, macro-average f1 and weighted-average f1 of the model
    '''
    print("One day, this will do something")
