import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.random.rand(100, 1)
y = 2.0 + 5 * x**2 + 00.1 * np.random.randn(100, 1)

# 1) My own code for computing the parametrization of the data set fitting a 2nd order polynomial:
X = np.zeros((100, 3))
X[:, 0] = 1; X[:, 1] = x[:, 0]; X[:, 2] = x[:, 0]**2

beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
ytilde = X @ beta

# Sorting the values:
x_sorted = np.sort(x, axis=0)
ytilde = np.sort(ytilde, axis=0)

plt.scatter(x, y, alpha=0.7, label="Data")
plt.plot(x_sorted, ytilde, alpha=0.7, lw=2, color="m", label="Fit")
plt.legend()
plt.show()

"""
# 2) Code for computing the parametrization of the data set fitting a 2nd order polynomial:
linreg = LinearRegression()

#xnew = np.array([[0], [1]])
#ypredict = linreg.predict(xnew)

poly2 = PolynomialFeatures(degree=2)
X = poly2.fit_transform(x[:, np.newaxis])
linreg.fit(X, y)

Xplot = poly2.fit_transform(x[:, np.newaxis])
poly2_plot = plt.plot(x, linreg.predict(Xplot), label="2nd order fit")

plt.plot(x, y, color="red", label="True 2nd order fit")
plt.scatter(x, y, label="Data", color="orange", s=15)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Simple 2nd order Regression')
plt.legend()
plt.show()
"""