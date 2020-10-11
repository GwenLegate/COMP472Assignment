import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

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
out_model: numpy array containing the model prediction
out_real: numpy array containing the real values of the test set
'''

def convert_model_output_to_csv(model_name, dataset, out_model, out_real):
    confusion = confusion_matrix(out_real, out_model)
    precision = precision_score(out_real, out_model, average=None, zero_division=0)
    recall = recall_score(out_real, out_model, average=None)
    f1 = f1_score(out_real, out_model, average=None)
    ma_f1 = f1_score(out_real, out_model, average="macro")
    wa_f1 = f1_score(out_real, out_model, average="weighted")
    accuracy = accuracy_score(out_real, out_model)

    pd.DataFrame(out_model).to_csv("csv_output/" + model_name + "-" + dataset + ".csv", header=["Test Set Predictions"])
    pd.DataFrame(confusion).to_csv("csv_output/" + model_name + "-" + dataset + ".csv", mode='a')
    pd.DataFrame(precision).to_csv("csv_output/" + model_name + "-" + dataset + ".csv", mode='a', header=["Precision"])
    pd.DataFrame(recall).to_csv("csv_output/" + model_name + "-" + dataset + ".csv", mode='a', header=["Recall"])
    pd.DataFrame(f1).to_csv("csv_output/" + model_name + "-" + dataset + ".csv", mode='a', header=["f1"])
    model_eval_params = np.array([accuracy, ma_f1, wa_f1])
    pd.DataFrame(model_eval_params).to_csv("csv_output/" + model_name + "-" + dataset + ".csv", mode='a', header=["0.accuracy, 1.macro avg, 2.weighted avg"])