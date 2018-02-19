import numpy as np
import pickle
import sys
import os
import pylab as plt
from sklearn import datasets

def load_dataset(dataset_name):
    """Loads the dataset corresponding to the dataset name

    Parameters
    ----------
    dataset_name : name of the dataset

    Returns
    -------
    data :
        Returns the dataset as 'dict'
    """

    if dataset_name == "classification":
        X, y = datasets.make_classification(100, 20)

        return {"X": X, "y": y}

    elif dataset_name == "citiesSmall":
        dataset = load_pkl(os.path.join("..","data","citiesSmall.pkl"))

        return dataset

    elif dataset_name == "newsgroups":
        dataset = load_pkl(os.path.join("..","data","newsgroups.pkl"))
        dataset["X"] = dataset["X"].toarray()
        dataset["Xvalidate"] = dataset["Xvalidate"].toarray()

        return dataset

    elif dataset_name == "fluTrends":
        data = load_pkl(os.path.join("..","data","fluTrends.pkl"))

        return data["X"], data["names"]
        

def plotClassifier(model, X, y):
    """plots the decision boundary of the model and the scatterpoints
       of the target values 'y'.

    Assumptions
    -----------
    y : it should contain two classes: '1' and '2'

    Parameters
    ----------
    model : the trained model which has the predict function

    X : the N by D feature array

    y : the N element vector corresponding to the target values

    """
    x1 = X[:, 0]
    x2 = X[:, 1]

    x1_min, x1_max = int(x1.min()) - 1, int(x1.max()) + 1
    x2_min, x2_max = int(x2.min()) - 1, int(x2.max()) + 1

    x1_line =  np.arange(x1_min, x1_max)
    x2_line =  np.arange(x2_min, x2_max)

    x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)

    mesh_data = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

    y_pred = model["predict"](model, mesh_data)
    y_pred = np.reshape(y_pred, x1_mesh.shape)

    plt.xlim([x1_mesh.min(), x1_mesh.max()])
    plt.ylim([x2_mesh.min(), x2_mesh.max()])

    plt.contourf(x1_mesh, x2_mesh, -y_pred,
                cmap=plt.cm.RdBu, label="decision boundary",
                alpha=0.6)

    plt.scatter(x1[y==1], x2[y==1], color="b", label="class 1")
    plt.scatter(x1[y==2], x2[y==2], color="r", label="class 2")
    plt.legend()
    plt.title("Model outputs '1' for blue region\n"
              "Model outputs '2' for red region")


def mode(y):
    """Computes the element with the maximum count

    Parameters
    ----------
    y : an input numpy array

    Returns
    -------
    y_mode :
        Returns the element with the maximum count
    """
    if y.ndim > 1:
        y = y.ravel()
    N = y.shape[0]

    if N == 0:
        return -1

    keys = np.unique(y)

    counts = {}
    for k in keys:
        counts[k] = 0

    # Compute counts for each element
    for n in range(N):
        counts[y[n]] += 1

    y_mode = keys[0]
    highest = counts[y_mode]

    # Find highest count key
    for k in keys:
        if counts[k] > highest:
            y_mode = k
            highest = counts[k]

    return y_mode

def load_pkl(fname):
    """Reads a pkl file.

    Parameters
    ----------
    fname : the name of the .pkl file

    Returns
    -------
    data :
        Returns the .pkl file as a 'dict'
    """
    if sys.version_info[0] < 3:
        # Python 2
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    else:
        # Python 3
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

    return data
