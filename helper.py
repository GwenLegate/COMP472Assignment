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

''' creates the *.csv files required for each model.
INPUTS:
model_name: the name of the ML model
dataset: the dataset used, DS1 for A-Z, DS2 for the greek letters 
output: an (n,2) numpy array containing the test set instance line number and the model prediction
confusion: the confusion matrix output
precision: precision of each class class, an (n,) numpy array, where n is the number of classes in the dataset
recall: recall of each class, an (n,) numpy array, where n is the number of classes in the dataset
f1: the f1 measure of each class,  an (n,) numpy array, where n is the number of classes in the dataset
accuracy: the accuracy of the model
ma_f1: the macro-average f1 of the model
wa_f1: the weighted-average f1 of the model
'''
def convert_model_output_to_csv(model_name, dataset, output, confusion, precision, recall, f1, accuracy, ma_f1, wa_f1):
    pd.DataFrame(output).to_csv("csv_output/" + model_name + "-" + dataset + ".csv")
    pd.Dataframe(confusion).to_csv("csv_output/" + model_name + "-" + dataset + ".csv", mode='a')
    pd.Dataframe(precision).to_csv("csv_output/" + model_name + "-" + dataset + ".csv", mode='a')
    pd.Dataframe(recall).to_csv("csv_output/" + model_name + "-" + dataset + ".csv", mode='a')
    pd.Dataframe(f1).to_csv("csv_output/" + model_name + "-" + dataset + ".csv", mode='a')
    model_eval_params = np.array([[accuracy], [ma_f1], [wa_f1]])
    pd.Dataframe(model_eval_params).to_csv("csv_output/" + model_name + "-" + dataset + ".csv", mode='a')

