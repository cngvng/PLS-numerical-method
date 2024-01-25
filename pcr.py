# -*- coding: utf-8 -*-
"""

@author: cngvng
"""

import pandas as pd
import numpy as np
import time

import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error

from utils import *

types = "short"
normalized = True
binary_classify = True

data_path_unsw_train = "dataset/UNSW_NB15_training-set.csv"
data_path_unsw_test = "dataset/UNSW_NB15_testing-set.csv"

n_compnents = 25
normalized = True
binary_classify = True
label = False  # label=False for Feature Extraction

""" Processing train data and test data for pca """

# ==>process training data
data_train = preprocessing_data_unsw(data_path=data_path_unsw_train, normalized=normalized,
                                     binary_classify=binary_classify)

time_train_start = time.process_time()
y_train = data_train['label']
data_train = data_train.drop(columns=['label'])

# feature extraction using PCA
X = data_train.to_numpy()
X_mean = np.mean(X, axis=0)
X_hat = X - X_mean

pca = PCA(n_components=n_compnents)
pca.fit(data_train)
U = pca.components_.T

X_train = np.dot(U.T, X_hat.T).T


"""Training procedure"""
classifier = LinearRegression()

time_train_start = time.process_time()
classifier.fit(X_train, y_train)
time_train_end = time.process_time()
time_train = time_train_end - time_train_start

print("==>start testing phase")
# ==>process testing data
data_test = preprocessing_data_unsw(data_path=data_path_unsw_test, normalized=normalized,
                                    binary_classify=binary_classify)
y_test = data_test['label']
data_test = data_test.drop(columns=['label'])
data_test = align_test_dataset(data_test, data_train)

time_predict_start = time.process_time()
X_test = np.dot(U.T, (data_test.to_numpy() - X_mean).T).T
y_pred = classifier.predict(X_test)
time_predict_end = time.process_time()
time_predict = (time_predict_end - time_predict_start) / len(y_test)
# end testing phase

"==>predict and print results"
y_pred = classifier.predict(X_test)
time_predict_end = time.process_time()
time_predict = (time_predict_end - time_predict_start) / len(y_test)

# display_results(y_test=y_test, y_pred=y_pred, run_time=time_predict)

y_pred = pd.DataFrame(y_pred)
# file_name = str(classifier) # for save figure

# confusion_matrix(y_test=y_test, y_pred=y_pred, binary_classify=binary_classify, types=types)

print("==>start plotting")
plt.figure(figsize=(10, 10))
plt.title("Linear Regression")
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.plot(y_test, y_pred, 'ro')
plt.savefig("plots/LinearRegression-binary.png")

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("R2 score: %.2f" % r2_score(y_test, y_pred))