import numpy as np

class ForwardPerceptron:
    def __init__(self, nOut, eta=0.25, maxiter=10, add_bias=True, debug=False):
        self.nOut = nOut
        self.eta = eta
        self.maxiter = maxiter
        self.add_bias = add_bias
        self.debug = debug

    def activations(self, X, w):
        activations = np.dot(X, w)
        activations = np.where(activations > 0, 1, 0)
        return activations

    def _add_bias(self, X):
        '''
        If add_bias=False, does nothing. Otherwise adds a bias term to design matrix X, where the first column will
        be set to -1.
        :param X: design matrix X
        :return: X with the bias term
        '''
        if self.add_bias:
            return np.concatenate((-np.ones((len(X), 1)), X), axis=1)
        return X

    def fit(self, X, y):
        X = self._add_bias(X)
        y = np.reshape(y, (len(X), self.nOut))
        w = np.random.rand(len(X[0]), self.nOut)*0.1-0.05

        for i in range(self.maxiter):
            activations = self.activations(X, w)
            diff = np.sum(y-activations)
            if diff == 0:
                if self.debug:
                    print("Weights converged, stopped.")
                break
            w += self.eta*np.dot(np.transpose(X),y-activations)
            if self.debug:
                print("%i. iteration, weights:\n%s" %(i, str(w)))
                if i+1 == self.maxiter:
                    diff = np.sum(y-self.activations(X,w))
                    if(np.sum(diff!=0)):
                        print("The network didn't converge. Last diff:%f" % diff)


        self.w = w

    def predict(self, X):
        X = self._add_bias(X)
        return self.activations(X, self.w)
