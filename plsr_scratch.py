from scratch import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error

from utils import *

types = "short"
normalized = True
binary_classify = True

data_path_unsw_train = "dataset/UNSW_NB15_training-set.csv"
data_path_unsw_test = "dataset/UNSW_NB15_testing-set.csv"

n_components = 16
normalized = True
binary_classify = False
label = False  # label=False for Feature Extraction

""" Processing train data and test data for pls """

# ==>process training data
data_train = preprocessing_data_unsw(data_path=data_path_unsw_train, normalized=normalized,
                                     binary_classify=binary_classify)

y_train = data_train['label']
data_train = data_train.drop(columns=['label'])

# feature extraction using PLS
pls = PLSRegression(n_components=n_components)
X_train = data_train.to_numpy()
pls.fit(X_train, y_train)
X_train = pls.transform(X_train)

"""Training procedure"""
# PLSRegression already includes the training procedure

print("==>start testing phase")
# ==>process testing data
data_test = preprocessing_data_unsw(data_path=data_path_unsw_test, normalized=normalized,
                                    binary_classify=binary_classify)
y_test = data_test['label']
data_test = data_test.drop(columns=['label'])
data_test = align_test_dataset(data_test, data_train)

X_test = data_test.to_numpy()
# X_test = pls.transform(X_test)
print("======= Start Testing ======")
y_pred = pls.predict(X_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('R2 score: %.2f' % r2_score(y_test, y_pred))

print("======= Visualize results ======")
plt.scatter(y_test, y_pred)
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()
plt.savefig("plots/plsr-multi.png")