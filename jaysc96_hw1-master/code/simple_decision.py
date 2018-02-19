import numpy as np
import decision_stump

def predict(model, X):
    M, D = X.shape
    y = np.zeros(M)

    # GET VALUES FROM MODEL
    splitModel = model["splitModel"]

    splitVariable = splitModel["splitVariable"]
    splitValue = splitModel["splitValue"]

    j = splitVariable
    value = splitValue

    subModel1 = model["subModel1"]
    j1 = subModel1['splitVariable']
    value1 = subModel1['splitValue']
    splitSat1 = subModel1['splitSat']
    splitNot1 = subModel1['splitNot']

    subModel0 = model["subModel0"]
    j0 = subModel0['splitVariable']
    value0 = subModel0['splitValue']
    splitSat0 = subModel0['splitSat']
    splitNot0 = subModel0['splitNot']

    for m in range(M):
        if X[m, j] > value:
            if X[m, j1] > value1:
                y[m] = splitSat1
            else:
                y[m] = splitNot1

        else:
            if X[m, j0] > value0:
                y[m] = splitSat0
            else:
                y[m] = splitNot0
    return y