import sys
import argparse
import pylab as plt
import numpy as np

import utils
import knn
import cnn
import decision_tree
import random_tree
import decision_forest
import kmeans
import dbscan
import kmedians
import quantize_image

if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=['1.1', '1.2', '2.1', '2.2', '3.1', '3.2', '3.3', '4.1', '4.2', '4.3'])

    io_args = parser.parse_args()
    question = io_args.question

    if question == '1.1':
        dataset = utils.load_dataset('citiesSmall')
        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        # part 1: implement knn.predict
        # part 2: print training and test errors for k=1,3,10 (use utils.classification_error)
        # part 3: plot classification boundaries for k=1 (use utils.plot_2dclassifier)
        model = None
        predict = None
        yhat = None
        Yhat = None
        tr_err = 0
        te_err = 0

        for k in [1, 3, 10]:
            model = knn.fit(X, y, k)
            predict = model['predict']
            yhat = predict(model, X)
            Yhat = predict(model, Xtest)
            tr_err = utils.classification_error(y, yhat)
            te_err = utils.classification_error(ytest, Yhat)
            print("Training error for k =", k, "is =", tr_err)
            print("Testing error for k =", k, "is =", te_err)

        utils.plot_2dclassifier(knn.fit(X, y, 1), Xtest, ytest)
        plt.show()

    if question == '1.2':
        dataset = utils.load_dataset('citiesBig1')
        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        # part 1: implement cnn.py
        # part 2: print training/test errors as well as number of examples for k=1
        # part 3: plot classification boundaries for k=1
        k = 1

        model = cnn.fit(X, y, k)
        predict = model['predict']
        yhat = predict(model, X)
        Yhat = predict(model, Xtest)
        tr_err = utils.classification_error(y, yhat)
        te_err = utils.classification_error(ytest, Yhat)
        print("Training error for k =", k, "is =", tr_err)
        print("Testing error for k =", k, "is =", te_err)

        utils.plot_2dclassifier(model, Xtest, ytest)
        plt.show()

    if question == '2.1':
        dataset = utils.load_dataset('vowel')
        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        # part 1: plot decision_tree as depth varies from 1 to 15
        # part 3: implement random_stump and report performance on random_tree
        # part 4: bootstrap at every depth

        N, D = X.shape
        tr_err = np.zeros(15)
        te_err = np.zeros(15)

        for i in range(15):
            model = random_tree.fit(X, y, i+1)
            predict = model['predict']
            yhat = predict(model, X)
            Yhat = predict(model, Xtest)
            tr_err[i] = utils.classification_error(y, yhat)
            te_err[i] = utils.classification_error(ytest, Yhat)

        plt.plot(range(1, 16), tr_err)
        plt.plot(range(1, 16), te_err)
        plt.legend(["Training error", "Testing error"])
        plt.show()

    if question == '2.2':
        dataset = utils.load_dataset('vowel')
        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        # part 1: decision trees
        # part 2: bootstrap sampling
        # part 3: random trees
        # part 4: random trees + bootstrap sampling

        model = decision_forest.fit(X, y)
        predict = model['predict']
        yhat = predict(model, Xtest)
        te_err = utils.classification_error(ytest, yhat)
        print("Testing error =", te_err)

    if question == '3.1':
        X = utils.load_dataset('clusterData')['X']

        model = kmeans.fit(X, k=4)
        low = model['error'](model, X)

        for i in range(49):
            new_model = kmeans.fit(X, k=4)
            err = new_model['error'](new_model, X)
            if err < low :
                model = new_model
                low = err

        utils.plot_2dclustering(X, model['predict'](model, X))
        print("Displaying figure...")
        plt.title("K-Means on clusterData")
        plt.show()

        # part 1: implement kmeans.error
        # part 2: get clustering with lowest error out of 50 random initialization

    if question == '3.2':
        X = utils.load_dataset('clusterData')['X']

        # part 3: plot min error across 50 random inits, as k is varied from 1 to 10

        low = np.zeros(10)

        for k in range(1,11):
            model = kmeans.fit(X, k)
            low[k-1] = model['error'](model, X)

            for i in range(49):
                new_model = kmeans.fit(X, k)
                err = new_model['error'](new_model, X)
                if err < low[k-1]:
                    model = new_model
                    low[k-1] = err

        plt.plot(range(1,11), low)
        plt.ylabel('Error')
        plt.xlabel('k')
        plt.show()

    if question == '3.3':
        X = utils.load_dataset('clusterData2')['X']

        # part 1: using clusterData2, plot min error across 50 random inits, as k is varied from 1 to 10
        # part 3: implement kmedians.py
        # part 4: plot kmedians.error

        low = np.zeros(10)

        model = kmedians.fit(X, k=4)
        low = model['error'](model, X)

        for i in range(49):
            new_model = kmedians.fit(X, k=4)
            err = new_model['error'](new_model, X)
            if err < low:
                model = new_model
                low = err

        utils.plot_2dclustering(X, model['predict'](model, X))
        print("Displaying figure...")
        plt.title("K-Medians on clusterData2")
        plt.show()

    if question == '4.1':
        img = utils.load_dataset('dog')['I']/255
        for b in [1, 2, 4, 6]:
            new_img = quantize_image.quantizeImage(img, b)
            plt.imshow(new_img)
            print("Displaying figure...")
            plt.show()

        # part 1: implement quantize_image.py
        # part 2: use it on the doge

    if question == '4.2':
        X = utils.load_dataset('clusterData2')['X']
        print (X.shape)
        model = dbscan.fit(X, radius2=256, min_pts=3)
        y = model['predict'](model, X)
        utils.plot_2dclustering(X,y)
        print("Displaying figure...")
        plt.show()

    if question == '4.3':
        dataset = utils.load_dataset('animals')
        X = dataset['X']
        animals = dataset['animals']
        traits = dataset['traits']

        model = dbscan.fit(X, radius2=15, min_pts=3)
        y = model['predict'](model, X)

        for kk in range(max(y)+1):
            print('Cluster {}: {}'.format(kk+1, ' '.join(animals[y==kk])))
