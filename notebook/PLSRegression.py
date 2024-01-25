import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA

rng = np.random.RandomState(1)
n_samples = 1000
cov = [[3, 3], [3, 4]]
X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
pca = PCA(n_components=2)
pca.fit(X)

y = X.dot(pca.components_[1] + 0.1 * rng.randn(2)) + 0.3 * rng.randn(n_samples)
print(X.shape, y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

import cvxpy as cp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = scaler.fit_transform(y_train.reshape(-1, 1))
y_test = scaler.transform(y_test.reshape(-1, 1))

class PLSRegression():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        import cvxpy as cp
        self.w = cp.Variable((X.shape[1], 1))
        loss = cp.sum_squares(X @ self.w - y)
        reg = cp.norm(self.w, 2)
        constraint = [cp.norm(self.w, 2) <= 1,
                      cp.norm(self.w, 1) <= self.n_components]
        objective = cp.Minimize(loss + reg)
        prob = cp.Problem(objective, constraint)
        prob.solve()
        self.w = self.w.value
        return self

    def predict(self, X):
        return X @ self.w

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    def transform(self, X):
        return X @ self.w

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_linear_model(self):
        return self.w


pls = PLSRegression(n_components=1)
pls.fit(X_train, y_train)
y_pred = pls.predict(X_test)
print('R2 score: ', r2_score(y_test, y_pred))
print('MSE: ', mean_squared_error(y_test, y_pred))

# plot the fit line
plt.scatter(X_test[:,0], y_test, label='Test Data')
plt.scatter(X_test[:,0], y_pred, label='Predicted Data')
plt.show()

from sklearn.cross_decomposition import PLSRegression
pls_sklearn = PLSRegression(n_components=1)
pls_sklearn.fit(X_train, y_train)
y_pred_sklearn = pls_sklearn.predict(X_test)
print('R2 score: ', r2_score(y_test, y_pred_sklearn))
print('MSE: ', mean_squared_error(y_test, y_pred_sklearn))

# plot the fit line
plt.scatter(X_test[:,0], y_test, label='Test Data')
plt.scatter(X_test[:,0], y_pred, label='Predicted Data')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(X_train[:, 0], y_train, label='Train Data')
ax[0].scatter(X_test[:, 0], y_test, label='Test Data')
ax[0].set_title('Original Data')
ax[0].legend()

ax[1].scatter(X_train[:, 0], pls.predict(X_train), label='Train Data')
ax[1].scatter(X_test[:, 0], pls.predict(X_test), label='Test Data')
ax[1].set_title('PLS Regression')
ax[1].legend()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(X_test[:, 0], y_test, label='Test Data')
ax[0].scatter(X_test[:, 0], y_pred, label='Predicted Data')
ax[0].set_title('PLS Regression from scratch')

ax[1].scatter(X_test[:, 0], y_test, label='Test Data')
ax[1].scatter(X_test[:, 0], y_pred_sklearn, label='Predicted Data')
ax[1].set_title('PLS Regression from sklearn')
plt.show()



