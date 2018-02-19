import os.path
import numpy as np
from numpy.linalg import norm
import pylab as plt
import pickle
import sys
import scipy.sparse

DATA_DIR = 'data'
FIGS_DIR = 'figs'

def savefig(fname, verbose=True):
    path = os.path.join('..', FIGS_DIR, fname)
    plt.savefig(path)
    if verbose:
        print("\nFigure saved as '{}'".format(path))

def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma

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

    return load_pkl(os.path.join('..',DATA_DIR,'{}.pkl'.format(dataset_name)))

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
    if not os.path.isfile(fname):
        raise ValueError('File {} does not exist.'.format(fname))

    if sys.version_info[0] < 3:
        # Python 2
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    else:
        # Python 3
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

    return data
