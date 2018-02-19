import sys
import argparse
import utils
import linear_model

# Load Binary and Multi -class data
data = utils.load_dataset("logisticData")
XBin, yBin = data['X'], data['y']
XBinValid, yBinValid = data['Xvalid'], data['yvalid']

data = utils.load_dataset("multiData")
XMulti, yMulti = data['X'], data['y']
XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

if __name__ == "__main__":
    argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required = True, 
        choices = ["1", "1.1", "1.2", "1.3", "3", "3.1", "3.4"])
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        # Run Q1 given example
        model = linear_model.logReg(maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogReg Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logReg Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    if question == "1.1":
        # Fit logRegL2 model
        model = linear_model.logRegL2(lammy=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogRegL2 Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logRegL2 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    if question == "1.2":
        # Fit logRegL1 model
        model = linear_model.logRegL1(lammy=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogRegL1 Training error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("logRegL1 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())
        
    if question == "1.3":
        # Fit logRegL0 model
        model = linear_model.logRegL0(lammy=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nTraining error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    if question == "3":
        # Run Q3 given example - Fit One-vs-all Least Squares
        model = linear_model.leastSquaresClassifier()
        model.fit(XMulti, yMulti)

        print("leastSquaresClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("leastSquaresClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))
        
    if question == "3.1":
        # Fit One-vs-all Logistic Regression
        model = linear_model.logLinearClassifier(maxEvals=500, verbose=0)
        model.fit(XMulti, yMulti)

        print("logLinearClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("logLinearClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

    if question == "3.4":
        # Fit logRegL2 model
        model = linear_model.softmaxClassifier(maxEvals=500)
        model.fit(XMulti, yMulti)

        print("Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))
