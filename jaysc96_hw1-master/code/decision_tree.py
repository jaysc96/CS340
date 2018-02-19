import numpy as np
import decision_stump

def fit(X, y, maxDepth):
    # Fits a decision tree using greedy recursive splitting
    N, D = X.shape

    # Learn a decision stump
    splitModel = decision_stump.fit(X, y)

    if maxDepth <= 1 or splitModel["splitVariable"] is None:
        # If we have reached the maximum depth or the decision stump does
        # nothing, use the decision stump
        model = splitModel
        model["splitModel"] = splitModel

    else:
        # Fit a decision tree to each split, decreasing maximum depth by 1
        j = splitModel["splitVariable"]
        value = splitModel["splitValue"]

        # Find indices of examples in each split
        splitIndex1 = X[:,j] > value
        splitIndex0 = X[:,j] <= value

        # Fit decision tree to each split
        model = {}

        model["splitModel"] = splitModel
        model["subModel1"] = fit(X[splitIndex1], y[splitIndex1], maxDepth-1)
        model["subModel0"] = fit(X[splitIndex0], y[splitIndex0], maxDepth-1)
        model["predict"] = predict

    return model

def predict(model, X):
    M, D = X.shape
    y = np.zeros(M)

    # GET VALUES FROM MODEL
    splitModel = model["splitModel"]

    splitVariable = splitModel["splitVariable"]
    splitValue = splitModel["splitValue"]
    splitSat = splitModel["splitSat"]

    if splitVariable is None:
        # If no further splitting, return the majority label
        y = splitSat * np.ones(M)

    # the case with depth=1, just a single stump.
    elif 'subModel1' not in model.keys():
        return splitModel['predict'](splitModel, X)

    else:
        # Recurse on both sub-models
        j = splitVariable
        value = splitValue

        splitIndex1 = X[:,j] > value
        splitIndex0 = X[:,j] <= value

        subModel1 = model["subModel1"]
        subModel0 = model["subModel0"]

        y[splitIndex1] = subModel1["predict"](subModel1, X[splitIndex1])
        y[splitIndex0] = subModel0["predict"](subModel0, X[splitIndex0])

    return y
