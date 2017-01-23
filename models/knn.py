import numpy as np
from measures.cv import accuracy_score, squared_diff_score

class KNN:
    def __init__(self, n_neighbors=5, metric=None, regression=False):
        '''
        Initializes the KNN algorithm.
        :param n_neighbors: The number of neighbors to use.
        :param metric: The metric to use. Any distance measure is accepted, even non-metrics. The function must accept a
        1D vector as its first parameter, and a 2D matrix as the second parameter. It should calculate the rowwise distance
        between the first vector and all row vectors in the matrix.
        :param regression: Set to true for regression calculation. Defaults to False, which is classification.
        '''
        self.n_neighbors = n_neighbors
        if metric == None:
            metric = KNN.__euclidean
        self.metric = metric
        self.regression = regression
        self.classification = not regression
        if not regression:
            self.prediction_func = KNN.__classification_prediction
        else:
            self.prediction_func = KNN.__regression_prediction


    def fit(self, X, y):
        # Metrics typically can't handle ints.
        self.X = X.astype(np.float64)
        self.y = y


    def predict(self, X):
        '''
        Returns the KNN prediction for design matrix X.
        :param X: the design matrix X
        :return: the predicted values for rows of X
        '''
        dist = self.__get_distance_matrix(X)

        ranks = np.apply_along_axis(KNN.__rank, 1, dist)
        # Using np.nan to denote values that are not nearest neighbors
        indicators = np.where(ranks < self.n_neighbors, 1, np.nan)

        y = np.multiply(indicators, self.y)

        # Construct new matrix without nan values
        newshape = (np.shape(X)[0], self.n_neighbors)
        y = (np.reshape(y[~np.isnan(y)], newshape))

        prediction = np.apply_along_axis(self.prediction_func, 1, y)
        return prediction

    def score(self, X, y):
        '''
        Returns a score for design matrix X and true labels y. Predicted values for X are scored against true values y.
        :param X: The design matrix to predict
        :param y: The true values of y
        :return: If this is a classification task, returns prediction accuracy
                 If this is a regression task, returns the sum of the squared errors
        '''
        predy = self.predict(X)
        if self.classification:
            return accuracy_score(predy, y)
        else:
            return squared_diff_score(predy, y)

    def __apply_metric_old(self, y):
        '''
        Applies the metric to each row in self.X paired with y
        :param y: the y to give as parameter to metric
        :return: 1-dimensional distance vector, where each value is the pairwise distance of corresponding row in X and y
        '''
        metric = lambda x: self.metric(x, y)
        dist = np.apply_along_axis(metric, 1, self.X)
        return dist

    def __get_distance_matrix(self, X):
        X = X.astype(np.float64)
        # Bind self to apply_metric, needed to bind self.X
        apply_metric = lambda y : self.metric(y, self.X)
        dist = np.apply_along_axis(apply_metric, 1, X)
        return dist

    @staticmethod
    def __rank(x):
        '''
        Returns the ranks of elements of x. Ranks start from 0.
        :param x: 1D-vector to rank
        :return: The ranks of the elements of x.
        '''
        order = x.argsort()
        ranks = np.empty(len(x), int)
        ranks[order] = np.arange(len(x))
        return ranks


    @staticmethod
    def __euclidean(y, X):
        '''
        The euclidean distance metric
        :param y:
        :param X:
        :return:
        '''
        axis = np.ndim(X)-1
        return np.sqrt(np.sum((y-X)**2, axis=axis))

    @staticmethod
    def __classification_prediction(y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    @staticmethod
    def __regression_prediction(y):
        return np.mean(y)

