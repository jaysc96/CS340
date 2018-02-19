import numpy as np
import pylab as plt
import utils

def fit(X, k, do_plot=False):
    N, D = X.shape
    y = np.ones(N)

    means = np.zeros((k, D))
    for kk in range(k):
        i = np.random.randint(N)
        means[kk] = X[i]

    while True:
        y_old = y

        # Compute euclidean distance to each mean
        dist2 = utils.euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        y = np.argmin(dist2, axis=1)

        means = np.zeros((k, D))
        # Update means
        for kk in range(k):
            means[kk] = X[y == kk].mean(axis=0)

        changes = np.sum(y != y_old)
        print('Running K-means, changes in cluster assignment = {}'.format(changes))

        # Stop if no point changed cluster
        if changes == 0:
            break

    if do_plot and D == 2:
        utils.plot_2dclustering(X, y)
        print("Displaying figure...")
        plt.show()

    model = dict()
    model['means'] = means
    model['predict'] = predict
    model['error'] = error

    return model

def predict(model, X):
    means = model['means']
    dist2 = utils.euclidean_dist_squared(X, means)
    dist2[np.isnan(dist2)] = np.inf
    return np.argmin(dist2, axis=1)

def error(model, X):
    means = model['means']
    err = 0

    y = model['predict'](model, X)

    err += (X[:, :]-means[y[:], :])**2
    return np.sum(err)
