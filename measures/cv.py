import numpy as np
class KFold:
    def __init__(self, n_splits=4):
        self.n_splits = n_splits

    def split(self, X):

        minfoldsize = len(X)//self.n_splits
        rem = len(X)-minfoldsize*self.n_splits #remainder, when the splits are not even
        foldsizes = np.repeat(minfoldsize, self.n_splits)
        foldsizes[:rem] += 1 #remainders are evenly added to first rem folds
        endindices = np.cumsum(foldsizes)

        for i, (size, endindex) in enumerate(zip(foldsizes, endindices)):
            startindex = endindex-size
            testindex = np.arange(startindex, endindex)
            trainindex = np.concatenate((np.arange(0, startindex), np.arange(endindex, len(X))))
            yield trainindex, testindex


class SLOO():
    def __init__(self, distances, delta):
        self.dist = distances
        self.delta = delta

    def split(self, X):
        for i in range(len(X)):
            trainindex = np.where(self.dist[i] > self.delta)
            yield trainindex, [i]


def accuracy_score(y, predy):
    '''
    Returns the accuracy score for predicted y and true y
    :param predy: the predicted y
    :param y: the true y
    :return: the accuracy score
    '''
    return np.mean(np.where(np.isclose(predy, y), 1, 0))

def mean_squared_error(y, predy):
    '''
    Returns the sum of squared differences between predicted y and true y
    :param predy: predicted y
    :param y:
    :return:
    '''
    return np.mean((predy - y) ** 2)

def cv_score(model, X, y, cv, score_func, return_predictions=False):
    '''
    Returns the cross-validated
    :param model: the model to use
    :param X: the design matrix X
    :param y: the true values y
    :param cv: the cross-validation object to use, which must implement split
    :param score_func: the scoring function to use
    :return: the mean score of scores returned by score_func for each cross-validation split
    '''
    scores = list()
    if return_predictions:
        preds = list()
    for trainindex, testindex in cv.split(X):
        trainX = X[trainindex]

        trainY = y[trainindex]
        testX = X[testindex]
        testY = y[testindex]
        model.fit(trainX, trainY)
        predy = model.predict(testX)
        if return_predictions:
            preds.append(predy)
        score = score_func(testY, predy)
        scores.append(score)
    result = np.array(scores)
    if return_predictions:
        preds = np.array(preds)
        result = (scores, preds)
    return result

def aggregate_cv(model, X, y, cv, score_func, return_predictions=False):
    preds = []
    for trainindex, testindex in cv.split(X):
        trainX = X[trainindex]

        trainY = y[trainindex]
        testX = X[testindex]
        model.fit(trainX, trainY)
        predy = model.predict(testX)
        preds.append(predy)
    result = score_func(y, np.array(preds))
    if return_predictions:
        result = (result, preds)
    return result

def cv_accuracy_score(model, X, y, cv=KFold(n_splits=4)):
    '''
    Returns the cv-score using accuracy scoring.
    :see cv_score, accuracy_score
    '''
    return cv_score(model, X, y, cv=cv, score_func=accuracy_score)

def cv_squares_score(model, X, y, cv=KFold(n_splits=4)):
    '''
    Returns the cv-score using squared difference scoring
    :see cv_score, squared_diff_score
    '''
    return cv_score(model, X, y, cv=cv, score_func=mean_squared_error)