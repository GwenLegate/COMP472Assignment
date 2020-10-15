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
    mlp_latin = MLPClassifier(hidden_layer_sizes=(100, ), activation='logistic', solver='sgd')
    mlp_greek = MLPClassifier(hidden_layer_sizes=(100, ), activation='logistic', solver='sgd')

    # train the model
    mlp_latin.fit(latin_x, latin_y)
    mlp_greek.fit(greek_x, greek_y)

    # test the model
    latin_predictions = mlp_latin.predict(test_latin_x)
    greek_predictions = mlp_greek.predict(test_greek_x)

    # evaluate
    help.convert_model_output_to_csv("BASE-MLP", "DS1", latin_predictions, test_latin_y)
    print("Latin alphabet for base MLP model")

    help.convert_model_output_to_csv("BASE-MLP", "DS2", greek_predictions, test_greek_y)
    print("Greek alphabet for base MLP model")


def run_best_mlp():
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
    mlp_latin = MLPClassifier(hidden_layer_sizes=(30,50,), activation='identity', solver='sgd')
    mlp_greek = MLPClassifier(hidden_layer_sizes=(30,50,), activation='identity', solver='sgd')

    # train the model
    mlp_latin.fit(latin_x, latin_y)
    mlp_greek.fit(greek_x, greek_y)

    # test the model
    latin_predictions = mlp_latin.predict(test_latin_x)
    greek_predictions = mlp_greek.predict(test_greek_x)

    # evaluate
    help.convert_model_output_to_csv("BEST-MLP", "DS1", latin_predictions, test_latin_y)
    print("Latin alphabet for best MLP model")

    help.convert_model_output_to_csv("BEST-MLP", "DS2", greek_predictions, test_greek_y)
    print("Greek alphabet for best MLP model")