import numpy as np
from scipy import stats
import decision_tree
import random_tree

def fit(X, y, max_depth=np.inf, n_bootstrap=50):
    trees = []
    N, D = X.shape
    for m in range(n_bootstrap):
        ind = np.random.randint(N, size=N)
        trees.append(random_tree.fit(X[ind], y[ind], max_depth))

    model = dict()
    model['trees'] = trees
    model['predict'] = predict
    return model

def predict(model, X):
    trees = model['trees']
    M = len(trees)
    t = X.shape[0]
    yhats = np.ones((t,M), dtype=np.uint8)

    # Predict using each model
    for m in range(M):
        yhats[:,m] = trees[m]['predict'](trees[m], X)

    # Take the most common label
    return stats.mode(yhats, axis=1)[0].flatten()
