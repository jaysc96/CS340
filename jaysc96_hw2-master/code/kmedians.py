import numpy as np
import pylab as plt
import utils

def fit(X, k, do_plot=False):
    N, D = X.shape
    y = np.ones(N)

    medians = np.zeros((k, D))
    for kk in range(k):
        i = np.random.randint(N)
        medians[kk] = X[i]

    while True:
        y_old = y

        # Compute distance to each median
        for n in range(N):
            dist1 = np.absolute(np.sum(X[n,:])-np.sum(medians, axis=1))
            y[n] = np.argmin(dist1)

        medians = np.zeros((k, D))
        # Update medians
        for kk in range(k):
            medians[kk] = np.median(X[y == kk], axis=0)

        changes = np.sum(y != y_old)
        print('Running K-medians, changes in cluster assignment = {}'.format(changes))

        # Stop if no point changed cluster
        if changes == 0:
            break

    if do_plot and D == 2:
        utils.plot_2dclustering(X, y)
        print("Displaying figure...")
        plt.show()

    model = dict()
    model['medians'] = medians
    model['predict'] = predict
    model['error'] = error

    return model

def predict(model, X):
    medians = model['medians']
    dist2 = utils.euclidean_dist_squared(X, medians)
    dist2[np.isnan(dist2)] = np.inf
    return np.argmin(dist2, axis=1)

def error(model, X):
    medians = model['medians']
    N, D = X.shape
    err = 0

    y = model['predict'](model, X)

    for n in range(N):
        err += np.absolute(np.sum(X[n, :])-np.sum(medians[y[n], :]))
    return err