## LINEAR REGRESSION

## NORMAL EQUATION
# generate a random test set
import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)  # y = 4 + 3x

X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to reach instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

theta_best  # [4.5, 2.96] not too far from our equation

# make new predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
y_predict

# plot it
from matplotlib import pyplot as plt

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

# use sklearn to do the same
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_  # nice

lin_reg.predict(X_new)


# get the theta from sklearn lin_reg model
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
theta_best_svd

# model is exponentially more complex the more feature we add


## GRADIENT DESCENT (tweak parameter during training)

# a quick implementation
eta = 0.1  # learning rate
n_iteration = 1000
m = 100

theta = np.random.randn(2, 1)
for interation in range(n_iteration):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

theta
