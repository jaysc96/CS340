import numpy as np

def fit(X, y):
    N, D = X.shape

    # Compute the number of class labels
    C = np.unique(y).size

    # Create a mapping from the labels to 0,1,2,...
    # so that we can store things in numpy arrays
    labels = dict()
    for index, label in enumerate(np.unique(y)):
        labels[index] = label

    # Compute the probability of each class i.e p(y==c)
    counts = np.zeros(C)

    for index, label in labels.items():
        counts[index] = np.sum(y==label)
        p_y = counts / N

    # Compute the conditional probabilities i.e.
    # p(x(i,j)=1 | y(i)==c) as p_xy
    # p(x(i,j)=0 | y(i)==c) as p_xy
    p_xy = np.zeros((D, C, 2))
    count1s = np.zeros((D, C))

    for d in range(D):
        for c in range(C):
            count1s[d, c] = np.sum((y==c+1) & (X[:, d]==1))

    p_xy[:, :, 1] = count1s/counts
    p_xy[:, :, 0] = 1-p_xy[:, :, 1]

    # Save parameters in model as dict
    model = dict()

    model["p_y"] = p_y
    model["p_xy"] = p_xy
    model["n_classes"] = C
    model["labels"] = labels

    return model

def fit_wrong(X, y):
    N, D = X.shape

    # Compute the number of class labels
    C = np.unique(y).size

    # Create a mapping from the labels to 0,1,2,...
    # so that we can store things in numpy arrays
    labels = dict()

    for index, label in enumerate(np.unique(y)):
        labels[index] = label

    # Compute the probability of each class i.e p(y==c)
    counts = np.zeros(C)

    for index, label in labels.items():
        counts[index] = np.sum(y==label)
        p_y = counts / N

    # Compute the conditional probabilities i.e.
    # p(x(i,j)=1 | y(i)==c) as p_xy
    # p(x(i,j)=0 | y(i)==c) as p_xy
    p_xy = 0.5 * np.ones((D, C, 2))

    # Save parameters in model as dict
    model = dict()

    model["p_y"] = p_y
    model["p_xy"] = p_xy
    model["n_classes"] = C
    model["labels"] = labels

    return model

def predict(model, X):
    N, D = X.shape
    C = model["n_classes"]
    p_xy = model["p_xy"]
    p_y = model["p_y"]
    labels = model["labels"]

    y_pred = np.zeros(N)

    for n in range(N):
        # Compute the probability for each class
        # This could be vectorized but we've tried to provide
        # an intuitive version.
        probs = p_y.copy()

        for d in range(D):
            if X[n, d] == 1:
                for c in range(C):
                    probs[c] *= p_xy[d, c, 1]

            elif X[n, d] == 0:
                for c in range(C):
                    probs[c] *= p_xy[d, c, 0]

        y_pred[n] = labels[np.argmax(probs)]

    return y_pred
