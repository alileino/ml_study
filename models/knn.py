import numpy as np
from measures.cv import accuracy_score, mean_squared_error
from measures.distance import l2_distance_squared
class KNN:
    WEIGHTS = ["none", "ranks", "ranks_squared", "inv_distance", "distance"]

    def __init__(self, n_neighbors=5, metric=None, regression=False, weights=None):
        '''
        Initializes the KNN algorithm.
        :param n_neighbors: The number of neighbors to use.
        :param metric: The metric to use. Any distance measure is accepted, even non-metrics. The function must accept a
        1D vector as its first parameter, and a 2D matrix as the second parameter. It should calculate the rowwise distance
        between the first vector and all row vectors in the matrix.
        :param regression: Set to true for regression calculation. Defaults to False, which is classification.
        :param weights: Available weighting schemes: rank and rank_squared
        '''
        self.n_neighbors = n_neighbors
        if weights == "none":
            weights = None
        if metric is None:
            metric = l2_distance_squared
        self.metric = metric
        self.regression = regression
        self.classification = not regression
        if not regression:
            self.prediction_func = KNN.__classification_prediction
        else:
            self.prediction_func = KNN.__regression_prediction
        self.weights = weights


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
        if X.ndim == 1:
            X = X.reshape(1,-1)

        self.distances_ = self.metric(X, self.X)

        ranks = np.apply_along_axis(KNN.__rank, 1, self.distances_)

        # Using np.nan to denote values that are not nearest neighbors
        indicators = np.where(ranks < self.n_neighbors, 1, np.nan)

        y = np.multiply(indicators, self.y)
        low_distances = np.multiply(indicators, self.distances_)

        # Construct new matrix without nan values
        newshape = (np.shape(X)[0], self.n_neighbors)
        low_distances = low_distances[ranks < self.n_neighbors].reshape(-1, self.n_neighbors)

        y = (np.reshape(y[~np.isnan(y)], newshape))
        low_ranks = ranks[ranks < self.n_neighbors].reshape(-1,self.n_neighbors)

        low_ranks = self.__weight_transform(low_ranks, low_distances)
        prediction = self.prediction_func(self, y, low_ranks )
        return prediction

    def __weight_transform(self, ranks, distances):
        '''
        Transforms ranks and distances to weights for the predictions
        :param ranks: a rank matrix
        :param distances: a distance matrix
        :return: a weight matrix
        '''
        result = len(ranks[0]) - ranks

        distances = np.sqrt(distances)
        if self.weights is None:
            result[:,:] = 1
        elif self.weights == "ranks":
            pass
        elif self.weights == "ranks_squared":
            result = ranks ** 2
        elif self.weights == "inv_distance":
            result = 1 / (1. + distances)
        elif self.weights == "distance":
            result = np.max(distances, axis=1).reshape(-1,1) -distances + 1 # add 1 to prevent zero weights

        return result


    def __weighted_counts(self, y, ranks):
        counts = dict()
        for y, r in zip(y, ranks):
            if y not in counts:
                counts[y] = 0
            counts[y] += r
        return zip(*counts.items())

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
            return accuracy_score(y, predy)
        else:
            return mean_squared_error(y, predy)

    def __get_distance_matrix(self, X):
        X = X.astype(np.float64)
        # Bind self to apply_metric, needed to bind self.X
        apply_metric = lambda y : self.metric(y, self.X)
        dist = np.apply_along_axis(apply_metric, 1, X)
        # dist = np.sqrt(dist)
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


    def __classification_prediction(self, y, weights):
        prediction = np.zeros(len(y))

        for i, y_row in enumerate(y):
            w = weights[i]
            y_row = y_row.astype(np.int64)
            values, counts = self.__weighted_counts(y_row, w)
            valmax = values[np.argmax(counts)]
            prediction[i] = valmax
        return prediction


    def __regression_prediction(self, y, ranks):
        ranksum = float(np.sum(ranks[0]))
        y = np.sum(y*ranks, axis=1)

        y /= ranksum

        return y
