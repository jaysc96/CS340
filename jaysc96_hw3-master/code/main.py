import sys
import argparse
import linear_model
import matplotlib.pyplot as plt
import numpy as np
import utils
import os

if __name__ == "__main__":
    argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required = True, 
        choices = ["2.1", "2.2","3.1","4.1","4.3"])
    io_args = parser.parse_args()
    question = io_args.question
    
    
    if question == "2.1":
        # Load the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']
        
        # get the number of rows(n) and columns(d)
        n,d = X.shape
        t = Xtest.shape[0]

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)
        
        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2) / n
        print("Training error = ", trainError)
        
        # Compute test error
        
        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2) / t
        print ("Test error = ", testError)
        
        # Plot model
        plt.figure()
        plt.plot(X,y,'b.', label = "Training data")
        plt.title('Training Data')
        # Choose points to evaluate the function
        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g', label = "Least squares fit")
        plt.legend(loc="best")
        figname = os.path.join("..","figs","leastSquares.pdf")
        print("Saving", figname)
        plt.savefig(figname)


        # Fit the least squares model with bias
        model = linear_model.LeastSquaresBias()
        model.fit(X, y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y) ** 2) / n
        print("Training error = ", trainError)

        # Compute test error

        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest) ** 2) / t
        print("Test error = ", testError)

        # Plot model
        plt.figure()
        plt.plot(X, y, 'b.', label="Training data")
        plt.title('Training Data')
        # Choose points to evaluate the function
        Xhat = np.linspace(np.min(X), np.max(X), 1000)[:, None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat, yhat, 'g', label="Least squares bias fit")
        plt.legend(loc="best")
        figname = os.path.join("..", "figs", "leastSquaresBias.pdf")
        print("Saving", figname)
        plt.savefig(figname)
        
    elif question == "2.2":
        
        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']
        
        # get the number of rows(n) and columns(d)
        n,d = X.shape
        t = Xtest.shape[0]
        
        for p in range(11):

            # Fit least-squares model
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X, y)

            # Compute training error
            yhat = model.predict(X)
            trainError = np.sum((yhat - y) ** 2) / n
            print("Training error = ", trainError)

            # Compute test error
            yhat = model.predict(Xtest)
            testError = np.sum((yhat - ytest) ** 2) / t
            print("Test error = ", testError)
            
            # Plot model
            plt.figure()
            plt.plot(X,y,'b.', label = "Training data")
            plt.title('Training Data. p = {}'.format(p))
            # Choose points to evaluate the function
            Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]

            #Predict on Xhat
            Yhat = model.predict(Xhat)
            plt.plot(Xhat, Yhat, 'g', label="Least squares poly fit")
            plt.legend()
            figname = os.path.join("..","figs","PolyBasis%d.pdf"%p)
            print("Saving", figname)
            plt.savefig(figname)
        
        
        
    elif question == "3.1":
        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']
        
        # get the number of rows(n) and columns(d)
        (n,d) = X.shape
        t = Xtest.shape[0]

        # Split training data into a training and a validation set
        # Find best value of RBF kernel parameter,
        # training on the train set and validating on the validation set
        K = 10
        N = int(n/K)

        samp = np.random.choice(n, n, replace=False)
        x = X[samp]
        Y = y[samp]

        minErr = np.inf
        for s in range(-15,16):
            sigma = 2 ** s
            validError = 0
            model = linear_model.LeastSquaresRBF(sigma)

            for k in range(K):
                Xtrain = x[np.r_[0:(k*N),((k+1)*N):n]]
                ytrain = Y[np.r_[0:(k*N),((k+1)*N):n]]
                Xvalid = x[(k*N):((k+1)*N)]
                yvalid = Y[(k*N):((k+1)*N)]

                model.fit(Xtrain, ytrain)

                # Compute the error on the validation set
                yhat = model.predict(Xvalid)
                validError += np.sum((yhat - yvalid)**2)/ (n//2)

            validError /= K
            print("Error with sigma = {:e} = {}".format( sigma ,validError))

            # Keep track of the lowest validation error
            if validError < minErr:
                minErr = validError
                bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error = {:e}".format(bestSigma))

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2)/n
        print("Training error = {}".format(trainError))

        # Finally, report the error on the test set
        t = Xtest.shape[0]
        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2)/t
        print("Test error = {}".format(testError))

        # Plot model
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title('Training Data')

        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g',label = "Least Squares with RBF kernel and $\sigma={}$".format(bestSigma))
        plt.ylim([-300,400])
        plt.legend()
        figname = os.path.join("..","figs","least_squares_rbf.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "4.1":
        # loads the data in the form of dictionary
        data = utils.load_dataset("outliersData")
        X = data['X']
        y = data['y']
        n, d = X.shape

        # Plot data
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title("Training data")

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        # Fit weighted least-squares estimator
        l = int(4*n/5)
        z = np.zeros((n, n))
        for i in range(n):
            if i < l:
                z[i,i]=1
            else:
                z[i,i]=0.1

        model = linear_model.WeightedLeastSquares()
        model.fit(X, y, z)
        print(model.w)

        # Draw model prediction
        Xsample = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xsample)
        plt.plot(Xsample,yhat,'g-', label = "Least squares weighted fit")
        plt.legend()
        figname = os.path.join("..","figs","least_squares_weighted_outliers.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "4.3":
        # loads the data in the form of dictionary
        data = utils.load_dataset("outliersData")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        # Plot data
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title("Training data")

        # Draw model prediction
        Xsample = np.linspace(np.min(X), np.max(X), 1000)[:,None]
        yhat = model.predict(Xsample)
        plt.plot(Xsample, yhat, 'g-', label = "Least squares gradient fit")
        plt.legend()
        figname = os.path.join("..","figs","gradient_descent_model.pdf")
        print("Saving", figname)
        plt.savefig(figname)