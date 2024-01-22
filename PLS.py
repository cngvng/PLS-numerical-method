import pandas as pd
import numpy as np
import time

import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.cross_decomposition import PLSRegression
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

regressor = PLSRegression(n_components=1)
regressor.fit(data_train, y_train)

# ==>process testing data
data_test = preprocessing_data_unsw(data_path=data_path_unsw_test, normalized=normalized,
                                    binary_classify=binary_classify)
y_test = data_test['label']
data_test = data_test.drop(columns=['label'])
data_test = align_test_dataset(data_test, data_train)

""" Predicting """
time_test_start = time.process_time()
y_pred = regressor.predict(data_test)
time_test_end = time.process_time()
time_test = time_test_end - time_test_start

display_results(y_test, y_pred, run_time=time_test)

print(y_pred.shape)
y_pred = pd.DataFrame(y_pred)
file_name = str(regressor.__class__.__name__) + str(PLSRegression.__name__)

confusion_matrix(y_test, y_pred, binary_classify=binary_classify, file_name=file_name, types=types)