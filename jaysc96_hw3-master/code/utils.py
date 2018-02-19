import pickle
import os
import sys


DATA_DIR = "data"

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

def load_dataset(dataset_name):

    if dataset_name == "basisData":
        dataset = load_pkl(os.path.join('..', DATA_DIR, 'basisData.pkl'))
    elif dataset_name == "outliersData":
        dataset = load_pkl(os.path.join('..', DATA_DIR, 'outliersData.pkl'))


    return dataset