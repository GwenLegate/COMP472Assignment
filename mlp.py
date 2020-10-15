from sklearn.neural_network import MLPClassifier
import pandas as pd
import helper as help
import constants as c


def run_base_mlp():
    # training sample
    latin_train = pd.read_csv(c.FILE_TRAIN_LATIN, sep=',', header=None).to_numpy()
    greek_train = pd.read_csv(c.FILE_TRAIN_GREEK, sep=',', header=None).to_numpy()
    latin_x, latin_y = latin_train[:, :-1], latin_train[:, -1]
    greek_x, greek_y = greek_train[:, :-1], greek_train[:, -1]

    # test sample
    latin_test = pd.read_csv(c.FILE_TEST_LABEL_LATIN, sep=',', header=None).to_numpy()
    greek_test = pd.read_csv(c.FILE_TEST_LABEL_GREEK, sep=',', header=None).to_numpy()
    test_latin_x, test_latin_y = latin_test[:, :-1], latin_test[:, -1]
    test_greek_x, test_greek_y = greek_test[:, :-1], greek_test[:, -1]

    # setup parameters
    mlpLatin = MLPClassifier(hidden_layer_sizes=(100, ), activation='logistic', solver='sgd')
    mlpGreek = MLPClassifier(hidden_layer_sizes=(100, ), activation='logistic', solver='sgd')

    # train the model
    mlpLatin.fit(latin_x, latin_y)
    mlpGreek.fit(greek_x, greek_y)

    # test the model
    latinPredictions = mlpLatin.predict(test_latin_x)
    greekPredictions = mlpGreek.predict(test_greek_x)

    # evaluate
    help.convert_model_output_to_csv("BASE-MLP", "DS1", latinPredictions, test_latin_y)
    print("Latin alphabet for MLP model")

    help.convert_model_output_to_csv("BASE-MLP", "DS2", greekPredictions, test_greek_y)
    print("Greek alphabet for MLP model")