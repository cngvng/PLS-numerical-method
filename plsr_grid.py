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
# import metrics for regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
# import metrics for classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from utils import *
from sklearn.cross_decomposition import PLSRegression


types = "short"

data_path_unsw_train = "dataset/UNSW_NB15_training-set.csv"
data_path_unsw_test = "dataset/UNSW_NB15_testing-set.csv"

n_compnents = 16
normalized = True
binary_classify = True

""" Processing train data and test data for pca """

# ==>process training data
data_raw = pd.read_csv(data_path_unsw_train)
data_train = preprocessing_data_unsw(data_path=data_path_unsw_train, normalized=normalized,
                                     binary_classify=binary_classify)
y_train = data_train['label']
X_train = data_train.drop(columns=['label'])

data_test = pd.read_csv(data_path_unsw_test)
y_test = data_test['label']

data_test = data_test.drop(columns=['label'])
data_test['service'].replace('-', 'other', inplace=True)
data_test = pd.get_dummies(data_test, columns=pd.Index(['proto', 'service', 'state']))
data_test = align_test_dataset(data_test, data_train)

parameters_pls = {'n_components': np.arange(1, 31, 1)}
pls = GridSearchCV(PLSRegression(), parameters_pls, scoring='neg_mean_squared_error')
pls.fit(X_train, y_train)

# Visualize the best number of components
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, 31, 1), -pls.cv_results_['mean_test_score'])
plt.xlabel('Number of components')
plt.ylabel('MSE')
plt.title('PLS with cross validation')
plt.savefig("plots/plsr-grid.png")
plt.show()