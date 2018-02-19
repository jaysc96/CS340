import sys
import argparse

import utils
import pylab as plt

import numpy as np
import naive_bayes
import decision_stump
import decision_tree
import mode_predictor

from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True, 
        choices=["1.1", "1.2", "2.1", "2.2", "3.1", "4.3"])

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":
        # Q1.1 - This should print the answers to Q 1.1

        # Load the fluTrends dataset
        X, names = utils.load_dataset("fluTrends")

        # part 1: min, max, mean, median and mode
        print "Min = %.3f" % np.amin(X)
        print "Max = %.3f" % np.amax(X)
        print "Mean = %.3f" % np.mean(X)
        print "Median = %.3f" % np.median(X)
        print "Mode = %.3f" % utils.mode(X)

        # part 2: quantiles
        print "10th quantile = %.3f" % np.percentile(X, 10)
        print "25th quantile = %.3f" % np.percentile(X, 25)
        print "50th quantile = %.3f" % np.percentile(X, 50)
        print "75th quantile = %.3f" % np.percentile(X, 75)
        print "90th quantile = %.3f" % np.percentile(X, 90)

        # part 3: maxMean, minMean, maxVar, minVar
        means = np.mean(X, axis=0)
        vars = np.var(X, axis=0)
        print "Highest Mean at %s" % names[np.argmax(means)]
        print "Lowest Mean at %s" % names[np.argmin(means)]
        print "Highest Variance at %s" % names[np.argmax(vars)]
        print "Minimum Variance at %s" % names[np.argmin(vars)]

        # part 4: correlation between columns
        corr = np.corrcoef(np.transpose(X))
        for i,row in enumerate(corr):
            row[i] = np.NaN
        mxi = np.nanargmax(corr, axis=0)
        mni = np.nanargmin(corr, axis=0)
        mx = np.nanmax(corr, axis=0)
        mn = np.nanmin(corr, axis=0)
        print "Lowest Correlation is between %s and %s" % (names[np.argmin(mn)], names[mni[np.argmin(mn)]])
        print "Highest Correlation is between %s and %s" % (names[np.argmax(mx)], names[mxi[np.argmax(mx)]])

    elif question == "1.2":
        # Q1.2 - This should plot the answers to Q 1.2
        # Load the fluTrends dataset
        X, names = utils.load_dataset("fluTrends")
        # Plot required figures

        # part 1
        i, j = X.shape
        x = [m+1 for m in xrange(i)]
        y = [n+1 for n in xrange(j)]
        plt.plot(x, X)
        plt.legend(names)
        plt.xlim(1,52)
        plt.xlabel("Week")
        plt.ylabel("Influenza percentage")
        plt.title("1. Plot Weeks vs Influenza percentages")
        fname = "../figs/q12_plotWeeksVsInfPer.pdf"
        plt.savefig(fname)
        plt.show()

        # part 2
        Xt = np.transpose(X)
        plt.boxplot(Xt)
        plt.xlabel("Week")
        plt.ylabel("Influenza percentage")
        plt.title("2. Boxplot Weeks vs Influenza percentages")
        fname = "../figs/q12_boxWeeksVsInfPer.pdf"
        plt.savefig(fname)
        plt.show()

        # part 3
        plt.hist(X.flatten(), normed=1)
        plt.xlabel("Influenza percentage")
        plt.ylabel("Frequency")
        plt.title("3. Histogram of Influenza percentages across all regions")
        fname = "../figs/q12_histAllReg.pdf"
        plt.savefig(fname)
        plt.show()

        # part 4
        plt.hist(X, normed=1, histtype='bar')
        plt.legend(names)
        plt.xlabel("Influenza percentage")
        plt.ylabel("Frequency")
        plt.title("4. Histogram of Influenza percentages for each region")
        fname = "../figs/q12_histEachReg.pdf"
        plt.savefig(fname)
        plt.show()

        # part 5 & 6
        corr = np.corrcoef(np.transpose(X))
        for i, row in enumerate(corr):
            row[i] = np.NaN
        mxi = np.nanargmax(corr, axis=0)
        mni = np.nanargmin(corr, axis=0)
        mx = np.nanmax(corr, axis=0)
        mn = np.nanmin(corr, axis=0)

        plt.scatter(X[:, np.argmin(mn)], X[:, mni[np.argmin(mn)]])
        plt.xlabel(names[np.argmin(mn)])
        plt.ylabel(names[mni[np.argmin(mn)]])
        plt.title("5. Scatterplot between regions of lowest correlation")
        fname = "../figs/q12_lowCorrScatter.pdf"
        plt.savefig(fname)
        plt.show()

        plt.scatter(X[:, np.argmax(mx)], X[:, mxi[np.argmax(mx)]])
        plt.xlabel(names[np.argmax(mx)])
        plt.ylabel(names[mxi[np.argmax(mx)]])
        plt.title("6. Scatterplot between regions of highest correlation")
        fname = "../figs/q12_highCorrScatter.pdf"
        plt.savefig(fname)
        plt.show()

    elif question == "2.1":
        # Q2.1 - Decision Stump with the inequality rule Implementation

        # 1. Load citiesSmall dataset
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate majority predictor model
        model = mode_predictor.fit(X, y)
        y_pred = mode_predictor.predict(model, X)

        error = np.sum(y_pred != y) / float(X.shape[0])
        print("Mode predictor error: %.3f" % error)

        # 3. Evaluate decision stump with equality rule
        model = decision_stump.fit_equality(X, y)
        y_pred = decision_stump.predict_equality(model, X)

        error = np.sum(y_pred != y) / float(X.shape[0])
        print("Decision Stump with equality rule error: %.3f" 
              % error)

        # 4. Evaluate decision stump with inequality rule

        model = decision_stump.fit(X, y)
        y_pred = decision_stump.predict(model, X)

        error = np.sum(y_pred != y) / float(X.shape[0])
        print("Decision Stump with inequality rule error: %.3f"
              % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)
        fname = "../figs/q21_decisionBoundary.pdf"
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        

    elif question == "2.2":
        # Q2.2 - Decision Tree with depth 2

        # 1. Load citiesSmall dataset
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision tree 
        model = decision_tree.fit(X, y, maxDepth=2)

        y_pred = decision_tree.predict(model, X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)

        # 3. Evaluate decision tree that uses information gain
        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)

    elif question == "3.1":
        # Q3.1 - Training and Testing Error Curves
        
        # 1. Load dataset
        dataset = utils.load_dataset("citiesSmall")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        model = DecisionTreeClassifier(criterion='entropy', max_depth=2)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)
        
        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)

        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "4.3":
        # Q4.3 - Train Naive Bayes

        # 1. Load dataset
        dataset = utils.load_dataset("newsgroups")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        # 2. Evaluate the decision tree model with depth 20
        model = DecisionTreeClassifier(criterion='entropy', max_depth=20)
        model.fit(X, y)
        y_pred = model.predict(X_valid)

        v_error = np.mean(y_pred != y_valid)
        print("Decision Tree Validation error: %.3f" % v_error)        
  
        # 3. Evaluate the Naive Bayes Model
        model = naive_bayes.fit(X, y)

        y_pred = naive_bayes.predict(model, X_valid)

        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes Validation error: %.3f" % v_error)
