import numpy as np
import utils

# helper function. leaves zeros as zeros.
def log0(x):
    x = x.copy()
    x[x>0] = np.log(x[x>0])
    return x

def fit(X, y):
    N, D = X.shape

    # Address the trivial case where we do not split
    count = np.bincount(y)

    # Compute total entropy (needed for information gain)
    p = count/float(np.sum(count)) # Convert counts to probabilities
    entropyTotal = -np.sum(p*log0(p))

    maxGain = 0
    splitVariable = None
    splitValue = None
    splitSat = np.argmax(count)
    splitNot = None

    # Check if labels are not all equal
    if np.unique(y).size > 1:
        # Loop over features looking for the best split
        for d in range(D):
            thresholds = np.unique(X[:,d])
            for value in thresholds[:-1]:
                # Count number of class labels where the feature is greater than threshold
                y_vals = y[X[:,d] > value]
                count1 = np.bincount(y_vals)
                count1 = np.pad(count1, (0,len(count)-len(count1)), \
                                mode='constant', constant_values=0)  # pad end with zeros to ensure same length as 'count'
                count0 = count-count1

                # Compute infogain
                p1 = count1/float(np.sum(count1))
                p0 = count0/float(np.sum(count0))
                H1 = -np.sum(p1*log0(p1))
                H0 = -np.sum(p0*log0(p0))
                prob1 = np.sum(X[:,d] > value)/float(N)
                prob0 = 1-prob1
                infoGain = entropyTotal - prob1*H1 - prob0*H0

                # Compare to minimum error so far
                if infoGain > maxGain:
                    # This is the highest information gain, store this value
                    maxGain = infoGain
                    splitVariable = d
                    splitValue = value
                    splitSat = np.argmax(count1)
                    splitNot = np.argmax(count0)

    # Store variables as dict
    model = dict()
    model["splitVariable"] = splitVariable
    model["splitValue"] = splitValue
    model["splitSat"] = splitSat
    model["splitNot"] = splitNot
    model["predict"] = predict
    return model

def predict(model, X):
    splitVariable = model["splitVariable"]
    splitValue = model["splitValue"]
    splitSat = model["splitSat"]
    splitNot = model["splitNot"]

    M, D = X.shape

    if splitVariable is None:
        return splitSat * np.ones(M)

    yhat = np.zeros(M)

    for m in range(M):
        if X[m, splitVariable] > splitValue:
            yhat[m] = splitSat
        else:
            yhat[m] = splitNot

    return yhat
