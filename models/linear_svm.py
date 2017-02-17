import numpy as np

class LinearClassifier:
    def __init__(self, C=1e-5, max_iter=100, learning_rate=1e-3, verbose=False, auto_stop=None):
        self.C = C
        self.max_iter = max_iter
        self.lr = learning_rate
        self.verbose = verbose
        self.auto_stop = auto_stop
        self.reset()

    def reset(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False, auto_stop=None):
        """
        Train this linear classifier using stochastic (batch) gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            idx = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[idx]
            y_batch = y[idx]
            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * grad


            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            if auto_stop is not None and len(loss_history) > 1:
                if np.abs(loss-loss_history[-2]) < auto_stop:
                    break

        return loss_history

    def predict(self, X):
        scores = np.dot(X, self.W)
        y_pred = np.argmax(scores, axis=1)

        return y_pred

    def fit(self, X, y):
        self.train(X, y, reg=self.C,  learning_rate=self.lr, num_iters=self.max_iter, verbose=self.verbose, auto_stop=self.auto_stop)

    def loss(self, X_batch, y_batch, reg):
        '''
        Implementers should override this function
        :param X_batch:
        :param y_batch:
        :param reg:
        :return:
        '''
        pass


class LinearSVC(LinearClassifier):

    def loss(self, X, y, reg):
        return LinearSVC.svm_loss(self.W, X, y, reg)

    @staticmethod
    def svm_loss(W, X, y, reg):
        '''
        Inputs have dimension D, there are C classes, and there are N examples

        :param W: numpy array of shape (D, C) containing weights
        :param X: numpy array of shape (N, D) containing training examples
        :param y: numpy array of shape (N,) containing training labels
        :param reg: regularization strength
        :return: tuple (loss, gradient) where loss is a single float, and gradient is the
         analytical gradient with respect to weights W of shape (D, C)
        '''
        num_classes = W.shape[1]
        num_train = X.shape[0]
        loss = 0.0

        S = np.dot(X, W) # shape = (N, C)

        delta = 1.

        # transform S to margin matrix
        S = (S - np.choose(y, S.T)[..., np.newaxis])+ 1

        # Compute loss
        loss += np.sum(S[S > 0]) # shape = (1,)

        loss /= num_train
        # equivalent to subtracting num_train before the division
        # is one because the margins are floated by delta for each N, because the correct class margin = delta.
        loss -= delta

        loss += 0.5 * reg * np.sum(W ** 2)

        # Compute gradient
        S = np.float64(S>0) # Turn margins into indicator matrix
        weights = (np.sum(S, axis=1)+1.)  # sum over classes, shape = (N,)

        S[np.arange(num_train), y] -= weights

        dW = np.dot(S.T, X)

        dW /= num_train
        dW = dW.T
        dW += reg*W
        return loss, dW


    # margins = S - S[range(0,num_train), y].reshape(-1,1) +1 # Alternatives
    # margins = (S - S[range(num_train),y,np.newaxis]) + 1
    # S = (S - np.choose(y, S.T)[..., np.newaxis]) + delta