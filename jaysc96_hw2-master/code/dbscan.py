from sklearn.cluster import DBSCAN
import numpy as np

def fit(X, radius2, min_pts):
    model = dict()
    model['X'] = X
    model['dbscan'] = DBSCAN(eps=np.sqrt(radius2), min_samples=min_pts)
    model['predict'] = predict
    return model

def predict(model, X):
    X = model['X']
    yhat = model['dbscan'].fit_predict(X)
    return yhat
