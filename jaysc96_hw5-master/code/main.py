import sys
import argparse
import pylab as plt
import numpy as np
from numpy.linalg import norm

import utils
from pca import PCA, AlternativePCA, RobustPCA
from manifold import MDS, ISOMAP

if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=['1.2', '2.1', '3', '3.1', '3.2'])

    io_args = parser.parse_args()
    question = io_args.question

    if question == '1.2':
        dataset = utils.load_dataset('animals')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n, d = X.shape

        # standardize columns
        X = utils.standardize_cols(X)

        # Plot the matrix
        plt.imshow(X)
        utils.savefig('q1_unsatisfying_visualization_1.png')

        ## Randomly plot two features, and label all points
        model = PCA(k=2)
        model.fit(X)
        Z = model.compress(X)
        
        fig, ax = plt.subplots()
        ax.scatter(Z[:,0], Z[:,1])
        for i in range(n):
            ax.annotate(animals[i], (Z[i,0],Z[i,1]))
        utils.savefig('q1_scatterplot_visualization.png')


    if question == '2.1':
        X = utils.load_dataset('highway')['X'].astype(float)/255
        n,d = X.shape
        h,w = 64,64 # height and width of each image

        # the two variables below are parameters for the foreground/background extraction method
        # you should just leave these two as default.

        k = 5 # number of PCs
        threshold = 0.04 # a threshold for separating foreground from background

        model = RobustPCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat = model.expand(Z)

        # save 10 frames for illustration purposes
        for i in range(10):
            plt.subplot(1,3,1)
            plt.title('Original')
            plt.imshow(X[i].reshape(h,w).T, cmap='gray')
            plt.subplot(1,3,2)
            plt.title('Reconstructed')
            plt.imshow(Xhat[i].reshape(h,w).T, cmap='gray')
            plt.subplot(1,3,3)
            plt.title('Thresholded Difference')
            plt.imshow(1.0*(abs(X[i] - Xhat[i])<threshold).reshape(h,w).T, cmap='gray')
            utils.savefig('q2.1_highway_{:03d}.pdf'.format(i))

    if question == '3':
        dataset = utils.load_dataset('animals')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        model = MDS(n_components=2)
        Z = model.compress(X)

        fig, ax = plt.subplots()
        ax.scatter(Z[:,0], Z[:,1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('MDS Features')
        for i in range(n):
            ax.annotate(animals[i], (Z[i,0], Z[i,1]))
        utils.savefig('q3_MDS_animals.png')

    if question == '3.1':
        dataset = utils.load_dataset('animals')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n, d = X.shape

        model = ISOMAP(n_components=2, n_neighbours=3)
        Z = model.compress(X)

        fig, ax = plt.subplots()
        ax.scatter(Z[:, 0], Z[:, 1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('MDS Features')
        for i in range(n):
            ax.annotate(animals[i], (Z[i, 0], Z[i, 1]))
        utils.savefig('q3.1_MDS_animals.png')

    if question == '3.2':
        dataset = utils.load_dataset('animals')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n, d = X.shape

        model = ISOMAP(n_components=2, n_neighbours=2)
        Z = model.compress(X)

        fig, ax = plt.subplots()
        ax.scatter(Z[:, 0], Z[:, 1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('MDS Features')
        for i in range(n):
            ax.annotate(animals[i], (Z[i, 0], Z[i, 1]))
        utils.savefig('q3.2_MDS_animals.png')
