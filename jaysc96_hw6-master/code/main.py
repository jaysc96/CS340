import sys
import argparse
import os
import numpy as np
import utils
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor as NeuralNet
from sklearn.neural_network import MLPClassifier as NeuralNetClassifier
import sklearn.datasets as datasets

if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True, choices=['1', '2'])

    io_args = parser.parse_args()
    question = io_args.question
    
    if question == '1':

        data = utils.load_dataset('basisData')
        
        X = data['X']
        y = data['y'].ravel()
        Xtest = data['Xtest']
        ytest = data['ytest'].ravel()
        n, d = X.shape
        t = Xtest.shape[0]
        
        model = NeuralNet(
            activation="logistic",
            solver="lbfgs",
            hidden_layer_sizes=(100, ),
            alpha=0.01
        )

        model.fit(X, y)
        
        # Compute training error
        yhat = model.predict(X)
        trainError = np.mean((yhat - y)**2)
        print("Training error = ", trainError)
        
        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean((yhat - ytest)**2)
        print("Test error     = ", testError)
        
        plt.figure()
        plt.plot(X, y, 'b.', label="training data", markersize=2)
        plt.title('Training Data')
        Xhat = np.linspace(np.min(X), np.max(X),1000)[:, None]
        yhat = model.predict(Xhat)
        plt.plot()
        plt.plot(Xhat, yhat, 'g', label="neural network")
        plt.ylim([-300, 400])
        plt.legend()
        figname = os.path.join("..", "figs", "modifiedBasisData.pdf")
        print("Saving", figname)
        plt.savefig(figname)
        
    elif question == '2':
        X, y = datasets.load_iris(return_X_y='true')
        n, d = X.shape
        i1 = np.random.choice(int(n/2), int(n/2))
        i2 = int(n/2)+np.random.choice(int(n/2), int(n/2))
        x = X[i1]
        Y = y[i1]
        XValid = X[i2]
        yValid = y[i2]
        model = NeuralNetClassifier(
            activation="logistic",
            solver="lbfgs",
            hidden_layer_sizes=(40,)
        )

        model.fit(x, Y)

        # Compute training error
        yhat = model.predict(x)
        trainError = np.mean((yhat - Y) ** 2)
        print("Training error = ", trainError)

        # Compute validation error
        yhat = model.predict(XValid)
        valError = np.mean((yhat - yValid) ** 2)
        print("Validation error = ", valError)

