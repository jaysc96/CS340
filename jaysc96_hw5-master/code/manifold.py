import numpy as np
from numpy.linalg import norm
import pylab as plt
import utils
from pca import PCA
from utils import find_min

class MDS:

    def __init__(self, n_components):
        self.k = n_components

    def compress(self, X):
        n = X.shape[0]
        k = self.k

        # Compute Euclidean distances
        D = utils.euclidean_dist_squared(X,X)
        D = np.sqrt(D)

        # Initialize low-dimensional representation with PCA
        Z = PCA(k).fit(X).compress(X)

        # Solve for the minimizer
        z = find_min(self._fun_obj_z, Z.flatten(), 500, False, D)
        Z = z.reshape(n, k)
        return Z

    def _fun_obj_z(self, z, D):
        n = D.shape[0]
        k = self.k
        Z = z.reshape(n,k)

        f = 0.0
        g = np.zeros((n,k))
        for i in range(n):
            for j in range(i+1,n):
                # Objective Function
                Dz = norm(Z[i]-Z[j])
                s = D[i,j] - Dz
                f = f + (0.5)*(s**2)

                # Gradient
                df = s
                dgi = (Z[i]-Z[j])/Dz
                dgj = (Z[j]-Z[i])/Dz
                g[i] = g[i] - df*dgi
                g[j] = g[j] - df*dgj

        return f, g.flatten()

class ISOMAP:
    def __init__(self, n_components, n_neighbours):
        self.k = n_components
        self.K = n_neighbours

    def compress(self, X):
        n = X.shape[0]
        k = self.k
        K = self.K

        # Compute Euclidean distances
        D = utils.euclidean_dist_squared(X, X)
        D = np.sqrt(D)
        nbrs = np.argsort(D, axis=1)[:, 1:K+1]
        G = np.zeros((n, n))

        for i in range(n):
            for j in nbrs[i]:
                G[i, j] = D[i, j]
                G[j, i] = D[j, i]

        D = utils.dijkstra(G)
        D[D == np.inf] = -np.inf
        max = np.max(D)
        D[D == -np.inf] = max

        # Initialize low-dimensional representation with PCA
        Z = PCA(k).fit(X).compress(X)

        # Solve for the minimizer
        z = find_min(self._fun_obj_z, Z.flatten(), 500, False, D)
        Z = z.reshape(n, k)
        return Z

    def _fun_obj_z(self, z, D):
        n = D.shape[0]
        k = self.k
        Z = z.reshape(n, k)

        f = 0.0
        g = np.zeros((n, k))
        for i in range(n):
            for j in range(i + 1, n):
                # Objective Function
                Dz = norm(Z[i] - Z[j])
                s = D[i, j] - Dz
                f = f + (0.5) * (s ** 2)

                # Gradient
                df = s
                dgi = (Z[i] - Z[j]) / Dz
                dgj = (Z[j] - Z[i]) / Dz
                g[i] = g[i] - df * dgi
                g[j] = g[j] - df * dgj

        return f, g.flatten()
