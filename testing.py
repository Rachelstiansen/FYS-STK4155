import numpy as np

# Matrices in Python
# Ex. a 3x3 real matrix
A = np.log(np.array([[4, 7, 8], [3, 10, 11], [4, 5, 7]]))

print(np.shape(A))

# Slicing the matrix and printing the first column:
print(A[:, 0])

# We can extract the eigenvalues of the covariance matrix through the
# np.linalg.eig() function

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.random.rand(100,1)
y = 10 * x + 0.1 * np.random.randn(100,1)
linreg = LinearRegression()
linreg.fit(x,y)
xnew = np.array([[0],[1]])
ypredict = linreg.predict(xnew)

plt.plot(xnew, ypredict, "r-")
plt.plot(x, y ,'ro')
plt.axis([0,1.0,0, 5.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Simple Linear Regression')
plt.show()