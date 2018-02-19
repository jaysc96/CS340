import numpy as np
from utils import find_min, checkRPCAGrad

class PCA:
    '''
    Solves the PCA problem min_Z,W (Z*W-X)^2 using SVD
    '''

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.mu = np.mean(X,0)
        X = X - self.mu

        U, s, V = np.linalg.svd(X)
        self.W = V[:self.k,:]
        return self

    def compress(self, X):
        X = X - self.mu
        Z = np.dot(X, self.W.transpose())
        return Z

    def expand(self, Z):
        X = np.dot(Z, self.W) + self.mu
        return X

class AlternativePCA:
    '''
    Solves the PCA problem min_Z,W (Z*W-X)^2 using gradient descent
    '''

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        n,d = X.shape
        k = self.k
        self.mu = np.mean(X,0)
        X = X - self.mu

        # Randomly initial Z, W
        z = np.random.randn(n*k)
        w = np.random.randn(k*d)

        f = np.sum((np.dot(z.reshape(n,k),w.reshape(k,d))-X)**2)/2
        for i in range(50):
            f_old = f
            z = find_min(self._fun_obj_z, z, 10, False, w, X, k)
            w = find_min(self._fun_obj_w, w, 10, False, z, X, k)
            f = np.sum((np.dot(z.reshape(n,k),w.reshape(k,d))-X)**2)/2
            print('Iteration {:2d}, loss = {}'.format(i, f))
            if f_old - f < 1e-4:
                break

        self.W = w.reshape(k,d)
        return self

    def compress(self, X):
        n,d = X.shape
        k = self.k
        X = X - self.mu
        # We didn't enforce that W was orthogonal so we need to optimize to find Z
        z = np.zeros(n*k)
        z = find_min(self._fun_obj_z, z, 100, False, self.W.flatten(), X, k)
        Z = z.reshape(n,k)
        return Z

    def expand(self, Z):
        X = np.dot(Z, self.W) + self.mu
        return X

    def _fun_obj_z(self, z, w, X, k):
        n,d = X.shape
        Z = z.reshape(n,k)
        W = w.reshape(k,d)

        R = np.dot(Z,W) - X
        f = np.sum(R**2)/2
        g = np.dot(R, W.transpose())
        return f, g.flatten()

    def _fun_obj_w(self, w, z, X, k):
        n,d = X.shape
        Z = z.reshape(n,k)
        W = w.reshape(k,d)

        R = np.dot(Z,W) - X
        f = np.sum(R**2)/2
        g = np.dot(Z.transpose(), R)
        return f, g.flatten()

class RobustPCA:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        n,d = X.shape
        k = self.k
        self.mu = np.mean(X,0)
        X = X - self.mu

        # Randomly initial Z, W
        z = np.random.randn(n*k)
        w = np.random.randn(k*d)
        checkRPCAGrad(self, z, w, X, k)

        f = np.sum(np.abs(np.dot(z.reshape(n, k), w.reshape(k, d))-X))
        for i in range(50):
            f_old = f
            z = find_min(self._fun_obj_z, z, 10, False, w, X, k)
            w = find_min(self._fun_obj_w, w, 10, False, z, X, k)
            f = np.sum(np.abs(np.dot(z.reshape(n, k), w.reshape(k, d))-X))
            print('Iteration {:2d}, loss = {}'.format(i, f))
            if f_old - f < 1e-4:
                break

        self.W = w.reshape(k, d)
        return self

    def compress(self, X):
        n,d = X.shape
        k = self.k
        X = X - self.mu
        # We didn't enforce that W was orthogonal so we need to optimize to find Z
        z = np.zeros(n*k)
        z = find_min(self._fun_obj_z, z, 100, False, self.W.flatten(), X, k)
        Z = z.reshape(n,k)
        return Z

    def expand(self, Z):
        X = np.dot(Z, self.W) + self.mu
        return X

    def _fun_obj_z(self, z, w, X, k, epsilon=0.0001):
        n,d = X.shape
        Z = z.reshape(n,k)
        W = w.reshape(k,d)

        R = np.dot(Z, W) - X
        D = np.sqrt(R**2 + epsilon)
        f = np.sum(D)
        R[:] = R[:] / D[:]
        g = np.dot(R, W.T)
        return f, g.flatten()

    def _fun_obj_w(self, w, z, X, k, epsilon=0.0001):
        n,d = X.shape
        Z = z.reshape(n, k)
        W = w.reshape(k, d)

        R = np.dot(Z, W) - X
        D = np.sqrt(np.power(R, 2) + epsilon)
        f = np.sum(D)
        R[:] = R[:] / D[:]
        g = np.dot(Z.T, R)
        return f, g.flatten()
