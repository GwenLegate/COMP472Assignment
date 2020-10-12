import pandas as pd
import helper as help
from sklearn.linear_model import Perceptron
import constants as c

latin_train = pd.read_csv(c.FILE_TRAIN_LATIN, sep=',', header=None).to_numpy()
greek_train = pd.read_csv(c.FILE_TRAIN_GREEK, sep=',', header=None).to_numpy()
latin_test = pd.read_csv(c.FILE_TEST_LABEL_LATIN, sep=',', header=None).to_numpy()
greek_test = pd.read_csv(c.FILE_TEST_LABEL_GREEK, sep=',', header=None).to_numpy()

latin_x, latin_y = latin_train[:, :-1], latin_train[:, -1]
greek_x, greek_y = greek_train[:, :-1], greek_train[:, -1]
test_latin_x, test_latin_y = latin_test[:, :-1], latin_test[:, -1]
test_greek_x, test_greek_y = greek_test[:, :-1], greek_test[:, -1]

# flag to set which dataset to use
DATASET = 2
'''run one model at a time'''
if(DATASET == 1):
    per_model = Perceptron()
    per_model.fit(latin_x, latin_y)
    latin_out = per_model.predict(test_latin_x)
    help.convert_model_output_to_csv("PER", "DS1", latin_out, test_latin_y)
    print("latin alphabet for perceptron model")

if(DATASET == 2):
    per_model = Perceptron()
    per_model.fit(greek_x, greek_y)
    greek_out = per_model.predict(test_greek_x)
    help.convert_model_output_to_csv("PER", "DS2", greek_out, test_greek_y)
    print("greek alphabet for perceptron model")

