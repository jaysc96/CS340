import numpy as np
from numpy.linalg import solve
import findMin
import sys
from scipy.optimize import approx_fprime


# Original Least Squares
class LeastSquares:
    # Class constructor
    def __init__(self):
        pass

    def fit(self,X,y):
        # Solve least squares problem

        a = np.dot(X.T, X)
        b = np.dot(X.T, y)
        self.w = solve(a, b)

    def predict(self, Xhat):

        w = self.w
        yhat = np.dot(Xhat, w)
        return yhat

# Least Squares with a bias added
class LeastSquaresBias:
    def __init__(self):
        pass

    def fit(self,X,y):

        # add a column of one to X
        # Solve least squares problem
        n, d = X.shape
        one = np.ones((n,1))
        z = np.append(X, one, axis=1)
        a = np.dot(z.T, z)
        b = np.dot(z.T, y)
        self.w = solve(a, b)

    def predict(self, Xhat):

        n, d = Xhat.shape
        one = np.ones((n, 1))
        zhat = np.append(Xhat, one, axis=1)
        w = self.w
        yhat = np.dot(zhat, w)
        return yhat

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):

        z = self.__polyBasis(X)
        a = np.dot(z.T, z)
        b = np.dot(z.T, y)
        self.w = solve(a, b)

    def predict(self, Xhat):

        zhat = self.__polyBasis(Xhat)
        w = self.w
        yhat = np.dot(zhat, w)
        return yhat

    # A private helper function to transform any matrix X into 
    # the polynomial basis defined by this class at initialization 
    # Returns the matrix Z that is the polynomial basis of X.   
    def __polyBasis(self, X):

        n = X.shape[0]
        d = self.p + 1
        # Z should have as many rows as X and as many columns as (p+1)
        Z = np.ones((n, d))

        # Fill in Z
        for i in range(1,d):
            Z[:,i] = np.power(X,i)[:,0]

        return Z

# Least Squares with RBF Kernel
class LeastSquaresRBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self,X,y):
        self.X = X
        [n, d] = X.shape

        Z = self.__rbfBasis(X, X, self.sigma)

        # Solve least squares problem
        l = 1e-12

        a = Z.T.dot(Z) + l* np.identity(n)
        b = np.dot(Z.T, y)
        self.w = solve(a,b)

    def predict(self, Xtest):
        Z = self.__rbfBasis(Xtest, self.X, self.sigma)
        yhat = Z.dot(self.w)
        return yhat

    def __rbfBasis(self, X1, X2, sigma):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        d = X1.shape[1]
        den = 1 / np.sqrt(2 * np.pi * (sigma** 2))

        D = (X1**2).dot(np.ones((d, n2))) + \
            (np.ones((n1, d)).dot((X2.T)** 2)) - \
            2 * (X1.dot( X2.T))

        Z = den * np.exp(-1* D / (2 * (sigma**2)))
        return Z

# Least Squares where each sample point X has a weight associated with it.
class WeightedLeastSquares:

    def __init__(self):
        pass

    def fit(self,X,y,z):
        a = X.T.dot(np.sqrt(z.T.dot(z))).dot(X)
        b = X.T.dot(np.sqrt(z.T.dot(z))).dot(y)
        self.w = solve(a,b)

    def predict(self,Xhat):
        yhat = Xhat.dot(self.w)
        return yhat


class LinearModelGradient:

    def __init__(self):
        pass

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient))
        else:
            print('User and numerical derivatives agree.')
        

        self.w, f = findMin.findMin(self.funObj, self.w, 100, X, y)

    def predict(self,Xtest):

        w = self.w
        yhat = Xtest*w
        return yhat

    def funObj(self,w,X,y):

        r = X.dot(w) - y
        # Calculate the function value
        f = np.sum(np.log(np.exp(r) + np.exp(-r)))

        # Calculate the gradient value
        g = X.T.dot(np.divide(np.exp(2 * r) - np.ones(r.shape), np.exp(2 * r) + np.ones(r.shape)))

        return (f,g)