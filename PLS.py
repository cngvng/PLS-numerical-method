import pandas as pd
import numpy as np
import time

import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.cross_decomposition import PLSRegression

from utils import *

types = "short"
normalized = False
binary_classify = True

data_path_unsw_train = "dataset/UNSW_NB15_training-set.csv"
data_path_unsw_test = "dataset/UNSW_NB15_testing-set.csv"

n_compnents = 16
normalized = True
binary_classify = False
label = False  # label=False for Feature Extraction

""" Processing train data and test data for pca """

# ==>process training data
data_train = preprocessing_data_unsw(data_path=data_path_unsw_train, normalized=normalized,
                                     binary_classify=binary_classify)

time_train_start = time.process_time()
y_train = data_train['label']
data_train = data_train.drop(columns=['label'])

# feature extraction using PLSRegression

pls = PLSRegression(n_components=n_compnents)
pls.fit(data_train, y_train)
X_train = pls.transform(data_train)

"""Training procedure"""
# classifier = DecisionTreeClassifier(random_state=77)
classifier = RandomForestClassifier(max_depth=5, random_state=77)

time_train_start = time.process_time()
classifier.fit(X_train, y_train)

time_train_end = time.process_time()

""" Processing test data """
# ==>process testing data
data_test = preprocessing_data_unsw(data_path=data_path_unsw_test, normalized=normalized,
                                    binary_classify=binary_classify)

y_test = data_test['label']
data_test = data_test.drop(columns=['label'])
data_test = align_test_dataset(data_test, data_train)

time_predict_start = time.process_time()
X_test = pls.transform(data_test)
y_pred = classifier.predict(X_test)
time_predict_end = time.process_time()
time_predict = time_predict_end - time_predict_start/len(y_pred)

display_results(y_test, y_pred, time_predict)

y_pred = pd.DataFrame(y_pred)
file_name = str(classifier)+ str(PLSRegression.__name__)

confusion_matrix(y_test, y_pred, binary_classify=binary_classify, file_name = file_name, types=types)
