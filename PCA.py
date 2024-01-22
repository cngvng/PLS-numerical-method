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

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn. linear_model import LinearRegression
import argparse

from utils import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--types', type=str, default="short",
                    help='types of dataset: short or full')
parser.add_argument('--normalized', type=bool, default=True,
                    help='types of dataset: short or full')
parser.add_argument('--binary_classify', type=bool, default=True,
                    help='types of dataset: short or full')

args = parser.parse_args()
types = args.types
normalized = args.normalized
binary_classify = args.binary_classify

# types = "short"
# normalized = False
# binary_classify = True

data_path_unsw_train = "dataset/UNSW_NB15_training-set.csv"
data_path_unsw_test = "dataset/UNSW_NB15_testing-set.csv"

n_compnents = 16
# normalized = True
# binary_classify = False
label = False  # label=False for Feature Extraction

""" Processing train data and test data for pca """

# ==>process training data
data_train = preprocessing_data_unsw(data_path=data_path_unsw_train, normalized=normalized,
                                     binary_classify=binary_classify)

time_train_start = time.process_time()
y_train = data_train['label']
data_train = data_train.drop(columns=['label'])

pcr = make_pipeline(PCA(n_components=n_compnents), LinearRegression())
pcr.fit(data_train, y_train)
pca = pcr.named_steps['pca']

# # feature extraction using PCA
X = data_train.to_numpy()
X_mean = np.mean(X, axis=0)
X_hat = X - X_mean

pca = PCA(n_components=n_compnents)
pca.fit(data_train)
U = pca.components_.T

X_train = np.dot(U.T, X_hat.T).T

# X_train = pca.transform(data_train)


"""Training procedure"""
regressor = LinearRegression()

time_train_start = time.process_time()
regressor.fit(X_train, y_train)
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
# X_test = np.dot(U.T, (data_test.to_numpy() - X_mean).T).T
X_test = pca.transform(data_test)
# end testing phase

"==>predict and print results"
y_pred = regressor.predict(X_test)
time_predict_end = time.process_time()
time_predict = (time_predict_end - time_predict_start) / len(y_test)

display_results(y_test=y_test, y_pred=y_pred, run_time=time_predict)
print(y_pred.shape)
y_pred = pd.DataFrame(y_pred)
file_name = str(regressor.__class__.__name__) + str(PCA.__name__) # for save figure

confusion_matrix(y_test=y_test, y_pred=y_pred, binary_classify=binary_classify, file_name=file_name, types=types)
