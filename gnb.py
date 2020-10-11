import pandas as pd
import helper as help
import sklearn.naive_bayes as nb
import constants as c

latin_train = pd.read_csv(c.FILE_TRAIN_LATIN, sep=',', header=None).to_numpy()
greek_train = pd.read_csv(c.FILE_TRAIN_GREEK, sep=',', header=None).to_numpy()
latin_test = pd.read_csv(c.FILE_TEST_LABEL_LATIN, sep=',', header=None).to_numpy()
greek_test = pd.read_csv(c.FILE_TEST_LABEL_GREEK, sep=',', header=None).to_numpy()

latin_x, latin_y = latin_train[:, :-1], latin_train[:, -1]
greek_x, greek_y = greek_train[:, :-1], greek_train[:, -1]
test_latin_x, test_latin_y = latin_test[:, :-1], latin_test[:, -1]
test_greek_x, test_greek_y = greek_test[:, :-1], greek_test[:, -1]

'''GNB model config'''
gnb_model = nb.GaussianNB()

'''train with latin dataset'''
gnb_model.fit(latin_x, latin_y)

'''test latin dataset'''
latin_out = gnb_model.predict(test_latin_x)

'''generate .csv with the necessary evaluation metrics'''
help.convert_model_output_to_csv("GNB", "DS1", latin_out, test_latin_y)

'''train with greek dataset'''
gnb_model.fit(greek_x, greek_y)

'''test latin dataset an compute metrics'''
greek_out = gnb_model.predict(test_greek_x)

'''confusion_greek = confusion_matrix(test_greek_y, greek_out)
precision_greek = precision_score(test_greek_y, greek_out, average=None)
recall_greek = recall_score(test_greek_y, greek_out, average=None)
f1_greek = f1_score(test_greek_y, greek_out, average=None)
f1_ma_greek = f1_score(test_greek_y, greek_out, average="macro")
f1_wa_greek = f1_score(test_greek_y, greek_out, average="weighted")
accuracy_greek = accuracy_score(test_greek_y, greek_out)

instance_no = np.arange(1, len(greek_out))
greek = np.vstack((instance_no, greek_out)).T'''

help.convert_model_output_to_csv("GNB", "DS2", greek_out, test_greek_y)