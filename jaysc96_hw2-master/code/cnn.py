"""
Implementation of condensed-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

def fit(X, y, k):
    """
    Parameters
    ----------
    X : an N by D numpy array
    y : an N by 1 numpy array of integers in {1,2,3,...,c}
    k : the k in k-NN
    """
    Xsub = np.zeros(X.shape)
    ysub = np.zeros(y.shape)
    Xsub[0] = X[0]
    ysub[0] = y[0]
    size = 1
    model = dict()
    i, j = X.shape
    for n in range(1, i):
        model['X'] = Xsub[0:size]
        model['y'] = ysub[0:size]
        model['k'] = k
        model['predict'] = predict
        yhat = predict(model, X[n])
        if yhat[0] != y[n]:
            Xsub[size] = X[n]
            ysub[size] = y[n]
            size += 1

    print("Size of dataset =", size)
    return model


def predict(model, Xtest):
    X = model['X']
    y = model['y']
    k = model['k']

    D = utils.euclidean_dist_squared(X, Xtest)
    D = np.argsort(D, axis=0)
    D = D[0:k, :]
    Y = y[D]
    yhat = np.amax(Y, axis=0)
    return yhat