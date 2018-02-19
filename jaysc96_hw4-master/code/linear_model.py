import numpy as np
import minimizers
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape    

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = minimizers.findMin(self.funObj, self.w, 
                                         self.maxEvals, 
                                         self.verbose,
                                         X, y)
    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)

        return np.sign(yhat)

class logRegL2:
    # Logistic Regression L2
    def __init__(self, verbose=1, lammy=1.0, maxEvals=100):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)
        lammy = self.lammy

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + (w.T.dot(w) * lammy / 2)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + (lammy * w)

        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = minimizers.findMin(self.funObj, self.w,
                                             self.maxEvals,
                                             self.verbose,
                                             X, y)

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)

        return np.sign(yhat)

class logRegL1:
    # Logistic Regression L1
    def __init__(self, verbose=1, lammy=1.0, maxEvals=100):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = minimizers.findMinL1(self.funObj, self.w, self.lammy,
                                             self.maxEvals,
                                             self.verbose,
                                             X, y)

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)

        return np.sign(yhat)

# L0 Regularized Logistic Regression
class logRegL0(logReg): # this is class inheritance:
    # we "inherit" the funObj and predict methods from logReg
    # and we overwrite the __init__ and fit methods below.
    # Doing it this way avoids copy/pasting code. 
    # You can get rid of it and copy/paste
    # the code from logReg if that makes you feel more at ease.
    def __init__(self, L0=1.0, verbose=2, lammy=1.0, maxEvals=400):
        self.verbose = verbose
        self.L0 = L0
        self.maxEvals = maxEvals
        self.lammy = lammy

    def fit(self, X, y):
        n, d = X.shape    
        w0 = np.zeros(d)
        minimize = lambda ind: minimizers.findMin(self.funObj, 
                                                  w0[ind], 
                                                  self.maxEvals, 0, 
                                                  X[:, ind], y)
        selected = set()
        selected.add(0) # always include the bias variable 
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss

            if self.verbose > 1:
                print("Epoch %d " % len(selected))
                print("Selected feature: %d" % (bestFeature))
                print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue
                
                selected_new = selected | {i} # add "i" to the set
                # TODO: Fit the model with 'i' added to the features,
                # then compute the score and update the minScore/minInd
                w = np.zeros(d)
                w[list(selected_new)], loss = minimize(list(selected_new))
                self.L0 = (self.lammy * np.count_nonzero(w))
                loss += self.L0
                if loss < minLoss:
                    minLoss = loss
                    bestFeature = i
                    w0 = w

            selected.add(bestFeature)

        
        # re-train the model one last time using the selected features
        self.w = w0
        self.w[list(selected)], _ = minimize(list(selected))

class leastSquaresClassifier:
    # Q3 - One-vs-all Least Squares for multi-class classification
    def __init__(self):
        pass

    def fit(self, X, y):
        n, d = X.shape    
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((d, self.n_classes))
        
        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            self.W[:, i] = np.linalg.lstsq(np.dot(X.T, X), np.dot(X.T, ytmp))[0]

    def predict(self, X):
        yhat = np.dot(X, self.W)

        return np.argmax(yhat, axis=1)

class logLinearClassifier(logReg):
    # Q3 - multi-classification with Logistic loss

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((d, self.n_classes))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y == i] = 1
            ytmp[y != i] = -1

            self.W[:, i], _ = minimizers.findMin(self.funObj, self.W[:, i],
                                         self.maxEvals,
                                         self.verbose,
                                         X, ytmp)

    def predict(self, X):
        yhat = np.dot(X, self.W)

        return np.argmax(yhat, axis=1)

class softmaxClassifier:
    # Q3 - multi-classification with Softmax loss
    def __init__(self, verbose=1, maxEvals=400):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        n, d = X.shape
        W = np.reshape(w, (d, self.n_classes))

        Xw = X.dot(W)
        Xwy = np.zeros((n,))
        g1 = np.zeros(W.shape)
        g2 = np.zeros(W.shape)
        res = np.ones((n,))

        for i in range(n):
            Xwy[i] = Xw[i, y[i]]

        for i in np.unique(y):
            g1[:, i] = -np.sum(X[y == i], axis=0)
            den = np.sum(np.exp(Xw), axis=1)
            num = np.exp(Xw[:, i])
            res[:] = num[:]/den[:]
            g2[:, i] = res.dot(X)

        f = -np.sum(Xwy)+np.sum(np.log(np.sum(np.exp(Xw), axis=1)))

        g = g1 + g2

        return f, g.ravel()

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.w = np.zeros(d*self.n_classes)
        utils.check_gradient(self, X, y)

        self.w, _ = minimizers.findMin(self.funObj, self.w,
                                    self.maxEvals,
                                    self.verbose,
                                    X, y)

        self.w = np.reshape(self.w, (d, self.n_classes))

    def predict(self, X):
        yhat = np.dot(X, self.w)

        return np.argmax(yhat, axis=1)