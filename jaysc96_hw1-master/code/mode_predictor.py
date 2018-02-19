import utils

def fit(X, y):
    """ 
    Simple classifier that always predicts the most 
    common label of the training set
    """
    y_mode = utils.mode(y)
    model = {"y_mode":y_mode, "predict":predict}

    return model

def predict(model, X):
    return model["y_mode"]