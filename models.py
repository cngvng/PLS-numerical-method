# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:03:41 2022

@author: cngvng
"""

import pandas as pd
import numpy as np
import time

import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.decomposition import PCA

from utils import *

types = "short"
normalized = False
binary_classify = True

data_path_unsw_train = "dataset/UNSW/UNSW_NB15_training-set.csv"
data_path_unsw_test = "dataset/UNSW/UNSW_NB15_testing-set.csv"

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

# feature extraction using PCA
X = data_train.to_numpy()
X_mean = np.mean(X, axis=0)
X_hat = X - X_mean

pca = PCA(n_components=n_compnents)
pca.fit(data_train)
U = pca.components_.T

X_train = np.dot(U.T, X_hat.T).T


"""Training procedure"""
# classifier = DecisionTreeClassifier(random_state=77)
classifier = RandomForestClassifier(max_depth=5, random_state=77)

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

display_results(y_test=y_test, y_pred=y_pred, time_train=time_train, time_predict=time_predict)

y_pred = pd.DataFrame(y_pred)
file_name = str(classifier) # for save figure

confusion_matrix(y_test=y_test, y_pred=y_pred, binary_classify=binary_classify, file_name=file_name, types=types)
