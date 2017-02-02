import numpy as np
from measures.cv import accuracy_score, mean_squared_error

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
            metric = self.__l2_distance
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
        self.distances_ = self.metric(X)

        ranks = np.apply_along_axis(KNN.__rank, 1, self.distances_)
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
            return mean_squared_error(predy, y)

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


    def __l2_distance(self, X):
        num_test = X.shape[0]
        num_train = self.X.shape[0]
        dists = np.zeros((num_test, num_train))
        dists += np.sum(X * X, axis=1).reshape(-1, 1)  # add extra dimension to dots
        dists += np.sum(self.X * self.X, axis=1)
        dists -= 2 * np.matmul(X, self.X.transpose())
        dists = np.sqrt(dists)
        return dists


    @staticmethod
    def __classification_prediction(y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    @staticmethod
    def __regression_prediction(y):
        return np.mean(y)

