import pandas as pd
import helper as help
import sklearn.naive_bayes as nb
import constants as c

def run_gnb():
    latin_train = pd.read_csv(c.FILE_TRAIN_LATIN, sep=',', header=None).to_numpy()
    greek_train = pd.read_csv(c.FILE_TRAIN_GREEK, sep=',', header=None).to_numpy()
    latin_test = pd.read_csv(c.FILE_TEST_LABEL_LATIN, sep=',', header=None).to_numpy()
    greek_test = pd.read_csv(c.FILE_TEST_LABEL_GREEK, sep=',', header=None).to_numpy()

    latin_x, latin_y = latin_train[:, :-1], latin_train[:, -1]
    greek_x, greek_y = greek_train[:, :-1], greek_train[:, -1]
    test_latin_x, test_latin_y = latin_test[:, :-1], latin_test[:, -1]
    test_greek_x, test_greek_y = greek_test[:, :-1], greek_test[:, -1]

    gnb_model = nb.GaussianNB()
    gnb_model.fit(latin_x, latin_y)
    latin_out = gnb_model.predict(test_latin_x)
    help.convert_model_output_to_csv("GNB", "DS1", latin_out, test_latin_y)
    print("latin alphabet for GNB model")

    gnb_model = nb.GaussianNB()
    gnb_model.fit(greek_x, greek_y)
    greek_out = gnb_model.predict(test_greek_x)
    help.convert_model_output_to_csv("GNB", "DS2", greek_out, test_greek_y)
    print("greek alphabet for GNB model")

