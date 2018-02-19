import pickle
import os
import sys
import numpy as np
from scipy.optimize import approx_fprime

def load_dataset(dataset_name):
    # Load and standardize the data and add the bias term
    if dataset_name == "logisticData":
        data = load_pkl(os.path.join('..', "data", 'logisticData.pkl'))
        
        X, y = data['X'], data['y']
        Xvalid, yvalid = data['Xvalidate'], data['yvalidate']
    
        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

        return {"X":X, "y":y, 
                "Xvalid":Xvalid, 
                "yvalid":yvalid}

    elif dataset_name == "multiData":
        data = load_pkl(os.path.join('..', "data", 'multiData.pkl'))
        X, y = data['X'], data['y']
        Xvalid, yvalid = data['Xvalidate'], data['yvalidate']
    
        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

        y -= 1
        yvalid -=1

        return {"X":X, "y":y, 
                "Xvalid":Xvalid, 
                "yvalid":yvalid}

def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma, mu, sigma
    
def check_gradient(model, X, y):
    # This checks that the gradient implementation is correct
    w = np.random.rand(model.w.size)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(w,
                                       lambda w: model.funObj(w,X,y)[0], 
                                       epsilon=1e-6)

    implemented_gradient = model.funObj(w, X, y)[1]
    
    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
        raise Exception('User and numerical derivatives differ:\n%s\n%s' % 
             (estimated_gradient[:5], implemented_gradient[:5]))
    else:
        print('User and numerical derivatives agree.')

#def approx_fprime(x, f_func, epsilon=1e-7):
    # Approximate the gradient using the complex step method
    #n_params = x.size
    #e = np.zeros(n_params)
    #gA = np.zeros(n_params)
    #for n in range(n_params):
    #    e[n] = 1.
    #    val = f_func(x + e * np.complex(0, epsilon))
    #    gA[n] = np.imag(val) / epsilon
    #    e[n] = 0

    #return gA

def classification_error(y, yhat):
    return np.sum(y!=yhat) / float(yhat.size)

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